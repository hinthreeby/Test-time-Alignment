from __future__ import annotations

import tempfile
import unittest
import json
import os
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Method.RAD.adaptive_router_decoding import (
    AdaptiveRADConfig,
    adaptive_rad_generate,
    build_candidate_inputs,
    load_router_checkpoint_strict,
    rad_transform_reward,
    sha256_file,
    write_adaptive_generation_json,
)
from Method.RAD.utils.logits_processor import RewardAugmentedLogitsProcessor
from router.checkpoint import save_router_checkpoint
from router.model import RADTokenRouter
from scripts.validate_models import configure_gpt2_padding, load_reward_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LM_PATH = PROJECT_ROOT / "models" / "gpt2-large"
DEFAULT_RM_BASE_PATH = PROJECT_ROOT / "models" / "gpt2-small"
DEFAULT_RM_TOKENIZER_PATH = PROJECT_ROOT / "models" / "rad_rm_sentiment"
DEFAULT_RM_CHECKPOINT_PATH = DEFAULT_RM_TOKENIZER_PATH / "pytorch_model.bin"
DEFAULT_LEARNED_ROUTER_PATH = Path("/tmp/router_train_smoke/best.pt")


class FakeTokenizer:
    def __init__(self, vocab_size: int = 32, eos_token_id: int = 1, max_text_chars: int = 4) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = eos_token_id
        self.bos_token_id = 2
        self.max_text_chars = max_text_chars

    def get_vocab(self) -> dict[str, int]:
        return {f"tok_{index}": index for index in range(self.vocab_size)}

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True):
        rows = []
        for text in texts:
            ids = [2] + [3 + (ord(char) % 10) for char in str(text)[: self.max_text_chars]]
            rows.append(ids)
        max_len = max(len(row) for row in rows)
        input_ids = []
        attention = []
        for row in rows:
            pad = [self.pad_token_id] * (max_len - len(row))
            input_ids.append(pad + row)
            attention.append([0] * len(pad) + [1] * len(row))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        values = [int(value) for value in ids]
        if skip_special_tokens:
            values = [value for value in values if value not in {self.pad_token_id, self.eos_token_id, self.bos_token_id}]
        return " ".join(f"tok_{value}" for value in values)


class FakeLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch_size, seq_len = input_ids.shape
        base = torch.arange(self.vocab_size, dtype=torch.float, device=input_ids.device)
        logits = base.view(1, 1, -1).repeat(batch_size, seq_len, 1)
        logits = logits + self.anchor * 0.0
        step_bias = (input_ids[:, -1].float() % 3).view(batch_size, 1)
        logits[:, -1, :] = logits[:, -1, :] - step_bias
        return SimpleNamespace(logits=logits)


class FakeRewardModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_shapes: list[tuple[int, int]] = []

    def forward(self, input_ids, attention_mask=None, labels=None, use_cache=False):
        self.seen_shapes.append(tuple(input_ids.shape))
        last_token = input_ids[:, -1].float()
        reward = ((last_token % 7) / 6.0).unsqueeze(-1) + self.anchor * 0.0
        return None, reward


class ConstantRouter(torch.nn.Module):
    def __init__(self, beta: float, beta_max: float) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.beta = float(beta)
        self.beta_max = float(beta_max)

    def forward(self, base_logits: torch.Tensor, reward_scores: torch.Tensor):
        beta = torch.full((base_logits.shape[0], 1), self.beta, dtype=base_logits.dtype, device=base_logits.device)
        gate = torch.full((base_logits.shape[0], 1), self.beta / self.beta_max, dtype=base_logits.dtype, device=base_logits.device)
        return beta, gate


class AdaptiveRADDecodingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(123)
        self.tokenizer = FakeTokenizer()
        self.base_model = FakeLM()
        self.reward_model = FakeRewardModel()
        self.config = AdaptiveRADConfig(top_k=20, beta_max=10.0, max_new_tokens=3, do_sample=False, seed=7)

    def original_rad_reference_generate(self, prompts: list[str], beta: float, config: AdaptiveRADConfig):
        tokenizer = FakeTokenizer()
        base_model = FakeLM()
        reward_model = FakeRewardModel()
        return self.original_rad_reference_generate_with_models(
            prompts,
            tokenizer,
            base_model,
            reward_model,
            beta,
            config,
        )

    def original_rad_reference_generate_with_models(
        self,
        prompts,
        tokenizer,
        base_model,
        reward_model,
        beta: float,
        config: AdaptiveRADConfig,
    ):
        processor = RewardAugmentedLogitsProcessor.__new__(RewardAugmentedLogitsProcessor)
        processor._inverse = config.inverse
        processor._method = "linear"
        processor._beta = beta
        device = next(base_model.parameters()).device
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        selected_token_ids = [[] for _ in prompts]
        candidate_ids_history = [[] for _ in prompts]
        reward_history = [[] for _ in prompts]
        guided_history = [[] for _ in prompts]
        with torch.inference_mode():
            for _step in range(config.max_new_tokens):
                logits = base_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[:, -1, :]
                topk_scores, topk_ids = torch.topk(logits, k=config.top_k, dim=-1)
                candidate_input_ids, candidate_attention_mask = build_candidate_inputs(
                    input_ids,
                    attention_mask,
                    topk_ids,
                    max_reward_length=config.max_reward_length,
                )
                raw_rewards = reward_model(
                    input_ids=candidate_input_ids,
                    attention_mask=candidate_attention_mask,
                    labels=None,
                    use_cache=False,
                )[1]
                rewards = rad_transform_reward(raw_rewards[:, 0].reshape(len(prompts), config.top_k), inverse=config.inverse)
                guided = torch.stack([
                    processor.apply_function(topk_scores[row], rewards[row], beta)
                    for row in range(len(prompts))
                ])
                selected_indices = guided.argmax(dim=-1)
                next_token_ids = topk_ids.gather(1, selected_indices.unsqueeze(-1)).squeeze(-1)
                input_ids = torch.cat([input_ids, next_token_ids.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token_ids.unsqueeze(-1))], dim=-1)
                for row in range(len(prompts)):
                    selected_token_ids[row].append(int(next_token_ids[row].detach().cpu().item()))
                    candidate_ids_history[row].append([int(value) for value in topk_ids[row].detach().cpu().tolist()])
                    reward_history[row].append([float(value) for value in rewards[row].detach().cpu().tolist()])
                    guided_history[row].append([float(value) for value in guided[row].detach().cpu().tolist()])
        texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in selected_token_ids]
        return {
            "selected_token_ids": selected_token_ids,
            "candidate_ids_history": candidate_ids_history,
            "rad_reward_scores_history": reward_history,
            "guided_scores_history": guided_history,
            "texts": texts,
        }

    def test_original_rad_equivalence_with_constant_router(self) -> None:
        prompts = ["good", "bad"]
        beta = 4.0
        adaptive = adaptive_rad_generate(
            prompts,
            self.base_model,
            self.tokenizer,
            self.reward_model,
            self.tokenizer,
            ConstantRouter(beta=beta, beta_max=10.0),
            self.config,
        )
        original = self.original_rad_reference_generate(prompts, beta=beta, config=self.config)
        self.assertEqual([sample.candidate_ids_history for sample in adaptive], original["candidate_ids_history"])
        self.assertEqual([sample.rad_reward_scores_history for sample in adaptive], original["rad_reward_scores_history"])
        self.assertEqual([sample.guided_scores_history for sample in adaptive], original["guided_scores_history"])
        self.assertEqual([sample.selected_token_ids for sample in adaptive], original["selected_token_ids"])
        self.assertEqual([sample.text for sample in adaptive], original["texts"])

    def test_original_fixed_beta_apply_function_regression(self) -> None:
        original = RewardAugmentedLogitsProcessor.__new__(RewardAugmentedLogitsProcessor)
        original._inverse = False
        original._method = "linear"
        original._beta = 4.0
        base_logits = torch.tensor([1.0, 2.0, 3.0])
        raw_reward = torch.tensor([-1.0, 0.25, 2.0])
        expected = base_logits + 4.0 * rad_transform_reward(raw_reward, inverse=False)
        self.assertTrue(torch.equal(original.apply_function(base_logits, raw_reward), expected))

    def test_beta_history_length_matches_generated_tokens(self) -> None:
        samples = adaptive_rad_generate(
            ["abc"],
            self.base_model,
            self.tokenizer,
            self.reward_model,
            self.tokenizer,
            ConstantRouter(beta=2.5, beta_max=10.0),
            self.config,
        )
        sample = samples[0]
        self.assertEqual(len(sample.beta_history), len(sample.selected_token_ids))
        self.assertEqual(len(sample.beta_history), self.config.max_new_tokens)
        self.assertEqual(len(sample.gate_history), self.config.max_new_tokens)

    def test_router_checkpoint_loads_strictly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "router.pt"
            save_router_checkpoint(RADTokenRouter(beta_max=10.0, beta_init=2.0), path)
            loaded = load_router_checkpoint_strict(path, self.config)
            self.assertIsInstance(loaded, RADTokenRouter)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            bad_path = Path(tmp) / "bad.pt"
            bad_state = dict(payload["state_dict"])
            bad_state.pop(next(iter(bad_state)))
            payload["state_dict"] = bad_state
            torch.save(payload, bad_path)
            with self.assertRaises(RuntimeError):
                load_router_checkpoint_strict(bad_path, self.config)

    def test_training_checkpoint_validates_feature_manifest_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            manifest = {
                "top_k": 20,
                "max_reward_length": 256,
                "reward_transform_name": "rad_clamp_0_1",
                "tokenizer_model_identifiers": {
                    "base_model": "base",
                    "reward_checkpoint": "reward.pt",
                },
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            router = RADTokenRouter(beta_max=10.0, beta_init=2.0)
            checkpoint_path = root / "best.pt"
            torch.save(
                {
                    "router_state_dict": router.state_dict(),
                    "config": {
                        "beta_max": 10.0,
                        "beta_init": 2.0,
                        "train_manifest": str(manifest_path),
                        "validation_manifest": str(manifest_path),
                    },
                    "dataset_manifest_hashes": {str(manifest_path): sha256_file(manifest_path)},
                },
                checkpoint_path,
            )
            config = AdaptiveRADConfig(
                top_k=20,
                beta_max=10.0,
                max_new_tokens=1,
                max_reward_length=256,
                base_model_id="base",
                reward_model_id="reward.pt",
            )
            self.assertIsInstance(load_router_checkpoint_strict(checkpoint_path, config), RADTokenRouter)
            manifest["reward_transform_name"] = "raw_reward_scores"
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_router_checkpoint_strict(checkpoint_path, config)

    def test_latency_fields_are_present_and_non_negative(self) -> None:
        sample = adaptive_rad_generate(
            ["abc"],
            self.base_model,
            self.tokenizer,
            self.reward_model,
            self.tokenizer,
            ConstantRouter(beta=2.5, beta_max=10.0),
            self.config,
        )[0]
        required = {
            "latency_scope",
            "total_generation_time",
            "per_token_total_latency",
            "base_lm_latency",
            "reward_model_latency",
            "router_latency",
            "sampling_latency",
            "average_latency_per_token",
        }
        self.assertEqual(set(sample.latency), required)
        self.assertEqual(len(sample.latency["per_token_total_latency"]), self.config.max_new_tokens)
        for key, value in sample.latency.items():
            if isinstance(value, list):
                self.assertTrue(all(item >= 0.0 for item in value))
            elif isinstance(value, str):
                self.assertEqual(value, "batch")
            else:
                self.assertGreaterEqual(value, 0.0)

    def test_base_lm_and_reward_model_parameters_remain_unchanged_and_frozen(self) -> None:
        base_before = [parameter.detach().clone() for parameter in self.base_model.parameters()]
        reward_before = [parameter.detach().clone() for parameter in self.reward_model.parameters()]
        adaptive_rad_generate(
            ["abc"],
            self.base_model,
            self.tokenizer,
            self.reward_model,
            self.tokenizer,
            ConstantRouter(beta=2.5, beta_max=10.0),
            self.config,
        )
        self.assertTrue(all(not parameter.requires_grad for parameter in self.base_model.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in self.reward_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.base_model.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in self.reward_model.parameters()))
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(base_before, self.base_model.parameters())))
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(reward_before, self.reward_model.parameters())))

    def test_strict_compatibility_checks_top_k_and_tokenizers(self) -> None:
        with self.assertRaises(ValueError):
            adaptive_rad_generate(
                ["abc"],
                self.base_model,
                self.tokenizer,
                self.reward_model,
                self.tokenizer,
                ConstantRouter(beta=2.5, beta_max=10.0),
                AdaptiveRADConfig(top_k=10, beta_max=10.0, max_new_tokens=1),
            )
        incompatible_tokenizer = FakeTokenizer(vocab_size=33)
        with self.assertRaises(ValueError):
            adaptive_rad_generate(
                ["abc"],
                self.base_model,
                self.tokenizer,
                self.reward_model,
                incompatible_tokenizer,
                ConstantRouter(beta=2.5, beta_max=10.0),
                self.config,
            )

    def test_reward_input_truncation_uses_max_reward_length(self) -> None:
        tokenizer = FakeTokenizer(max_text_chars=320)
        reward_model = FakeRewardModel()
        config = AdaptiveRADConfig(top_k=20, beta_max=10.0, max_new_tokens=1, max_reward_length=256, do_sample=False)
        adaptive_rad_generate(
            ["x" * 320],
            FakeLM(),
            tokenizer,
            reward_model,
            tokenizer,
            ConstantRouter(beta=2.0, beta_max=10.0),
            config,
        )
        self.assertTrue(reward_model.seen_shapes)
        self.assertEqual(reward_model.seen_shapes[0][1], 256)

    def test_build_candidate_inputs_truncates_last_tokens(self) -> None:
        input_ids = torch.arange(300).view(1, 300)
        attention_mask = torch.ones_like(input_ids)
        candidate_ids = torch.arange(20).view(1, 20)
        candidate_input_ids, candidate_attention_mask = build_candidate_inputs(
            input_ids,
            attention_mask,
            candidate_ids,
            max_reward_length=256,
        )
        self.assertEqual(tuple(candidate_input_ids.shape), (20, 256))
        self.assertEqual(tuple(candidate_attention_mask.shape), (20, 256))
        self.assertEqual(int(candidate_input_ids[0, 0].item()), 45)

    def test_eos_stops_generation_and_history_matches_actual_tokens(self) -> None:
        tokenizer = FakeTokenizer(eos_token_id=31)
        config = AdaptiveRADConfig(top_k=20, beta_max=10.0, max_new_tokens=5, do_sample=False)
        sample = adaptive_rad_generate(
            ["abc"],
            FakeLM(),
            tokenizer,
            FakeRewardModel(),
            tokenizer,
            ConstantRouter(beta=1.0, beta_max=10.0),
            config,
        )[0]
        self.assertEqual(sample.selected_token_ids, [31])
        self.assertEqual(sample.token_ids, [31])
        self.assertEqual(len(sample.beta_history), 1)
        self.assertEqual(len(sample.candidate_ids_history), 1)

    def test_parameter_checks_do_not_concatenate_whole_model_copy(self) -> None:
        import Method.RAD.adaptive_router_decoding as adaptive_module

        self.assertFalse(hasattr(adaptive_module, "tensor_parameter_checksum"))
        self.assertTrue(hasattr(adaptive_module, "assert_frozen_without_grad"))

    def test_metadata_output_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            router_checkpoint = root / "router.pt"
            reward_checkpoint = root / "reward.pt"
            router_checkpoint.write_bytes(b"router")
            reward_checkpoint.write_bytes(b"reward")
            samples = adaptive_rad_generate(
                ["abc"],
                self.base_model,
                self.tokenizer,
                self.reward_model,
                self.tokenizer,
                ConstantRouter(beta=2.0, beta_max=10.0),
                AdaptiveRADConfig(
                    top_k=20,
                    beta_max=10.0,
                    max_new_tokens=1,
                    max_reward_length=256,
                    do_sample=False,
                    base_model_id="base",
                    reward_model_id="reward.pt",
                ),
            )
            output_path = root / "out.json"
            write_adaptive_generation_json(
                output_path,
                samples,
                AdaptiveRADConfig(
                    top_k=20,
                    beta_max=10.0,
                    max_new_tokens=1,
                    max_reward_length=256,
                    do_sample=False,
                    base_model_id="base",
                    reward_model_id="reward.pt",
                ),
                router_checkpoint_path=router_checkpoint,
                reward_checkpoint_path=reward_checkpoint,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["decoding_mode"], "learned_router")
            self.assertEqual(payload["router_checkpoint_sha256"], sha256_file(router_checkpoint))
            self.assertEqual(payload["base_model_id"], "base")
            self.assertEqual(payload["reward_checkpoint_sha256"], sha256_file(reward_checkpoint))
            self.assertEqual(payload["max_reward_length"], 256)
            self.assertEqual(payload["reward_transform_name"], "rad_clamp_0_1")

    @unittest.skipUnless(os.environ.get("RUN_REAL_RAD_INTEGRATION") == "1", "real-model RAD integration smoke is opt-in")
    def test_real_original_rad_matches_constant_router(self) -> None:
        device = torch.device("cpu")
        prompt = "It made my hair feel flat and uncooperative"
        beta = 10.0
        config = AdaptiveRADConfig(
            top_k=20,
            beta_max=30.0,
            max_new_tokens=2,
            max_reward_length=256,
            do_sample=False,
            seed=123,
            inverse=False,
            base_model_id=str(DEFAULT_LM_PATH),
            reward_model_id=str(DEFAULT_RM_CHECKPOINT_PATH),
        )
        tokenizer = AutoTokenizer.from_pretrained(str(DEFAULT_LM_PATH), local_files_only=True)
        configure_gpt2_padding(tokenizer, padding_side="left")
        base_model = AutoModelForCausalLM.from_pretrained(str(DEFAULT_LM_PATH), local_files_only=True, torch_dtype=torch.float32)
        base_model.eval().to(device)
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        reward_model, reward_tokenizer, _metadata = load_reward_model(
            DEFAULT_RM_BASE_PATH,
            DEFAULT_RM_TOKENIZER_PATH,
            DEFAULT_RM_CHECKPOINT_PATH,
            device,
        )
        configure_gpt2_padding(reward_tokenizer, padding_side="left")
        reward_model.eval().to(device)
        for parameter in reward_model.parameters():
            parameter.requires_grad = False

        original = self.original_rad_reference_generate_with_models(
            [prompt],
            tokenizer,
            base_model,
            reward_model,
            beta,
            config,
        )
        constant = adaptive_rad_generate(
            [prompt],
            base_model,
            tokenizer,
            reward_model,
            reward_tokenizer,
            ConstantRouter(beta=beta, beta_max=config.beta_max),
            config,
        )[0]
        learned = adaptive_rad_generate(
            [prompt],
            base_model,
            tokenizer,
            reward_model,
            reward_tokenizer,
            load_router_checkpoint_strict(DEFAULT_LEARNED_ROUTER_PATH, config),
            config,
        )[0]

        original_rewards = torch.tensor(original["rad_reward_scores_history"][0])
        constant_rewards = torch.tensor(constant.rad_reward_scores_history)
        original_guided = torch.tensor(original["guided_scores_history"][0])
        constant_guided = torch.tensor(constant.guided_scores_history)
        max_reward_diff = float((original_rewards - constant_rewards).abs().max().item())
        max_guided_diff = float((original_guided - constant_guided).abs().max().item())

        self.assertEqual(original["selected_token_ids"][0], constant.selected_token_ids)
        self.assertEqual(original["texts"][0], constant.text)
        self.assertEqual(original["candidate_ids_history"][0], constant.candidate_ids_history)
        self.assertTrue(torch.allclose(original_rewards, constant_rewards, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(original_guided, constant_guided, atol=1e-5, rtol=1e-6))
        self.assertEqual(len(original["selected_token_ids"][0]), len(constant.selected_token_ids))

        report = {
            "dependency_versions": {
                "evaluate": version("evaluate"),
                "datasets": version("datasets"),
                "aiohttp": version("aiohttp"),
            },
            "prompt": prompt,
            "config": {
                "top_k": config.top_k,
                "beta": beta,
                "max_new_tokens": config.max_new_tokens,
                "seed": config.seed,
                "greedy": not config.do_sample,
                "inverse": config.inverse,
                "max_reward_length": config.max_reward_length,
            },
            "original_output": {
                "selected_token_ids": original["selected_token_ids"][0],
                "generated_text": original["texts"][0],
            },
            "constant_router_output": {
                "selected_token_ids": constant.selected_token_ids,
                "generated_text": constant.text,
                "beta_history": constant.beta_history,
            },
            "learned_router_output": {
                "selected_token_ids": learned.selected_token_ids,
                "generated_text": learned.text,
                "beta_history": learned.beta_history,
            },
            "equivalence_result": {
                "selected_tokens_match": original["selected_token_ids"][0] == constant.selected_token_ids,
                "generated_text_match": original["texts"][0] == constant.text,
                "candidate_ids_match": original["candidate_ids_history"][0] == constant.candidate_ids_history,
                "max_guided_score_difference": max_guided_diff,
                "max_reward_score_difference": max_reward_diff,
            },
            "test_results": {
                "test_real_original_rad_matches_constant_router": "passed",
            },
        }
        reports_dir = PROJECT_ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        (reports_dir / "stage6_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
