from pathlib import Path
from types import SimpleNamespace
import types
from typing import List
import sys
import torch
from torch.nn import functional as F
from tqdm import tqdm

# import the huggingface transformers libraries
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, LlamaForCausalLM, LlamaForSequenceClassification

#### auto size stuff
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SAFE_RLHF_SOURCE = PROJECT_ROOT / "models" / "safe-rlhf-source"


def load_safe_rlhf_score_model_class(source):
    """Load only Safe-RLHF's score runtime, without its DeepSpeed trainers."""
    package_root = source / "safe_rlhf"
    models_root = package_root / "models"
    safe_package = types.ModuleType("safe_rlhf")
    safe_package.__file__ = str(package_root / "__init__.py")
    safe_package.__path__ = [str(package_root)]
    safe_package.__package__ = "safe_rlhf"
    models_package = types.ModuleType("safe_rlhf.models")
    models_package.__file__ = str(models_root / "__init__.py")
    models_package.__path__ = [str(models_root)]
    models_package.__package__ = "safe_rlhf.models"
    sys.modules["safe_rlhf"] = safe_package
    sys.modules["safe_rlhf.models"] = models_package
    try:
        from safe_rlhf.models.score_model import AutoModelForScore
    except Exception:
        sys.modules.pop("safe_rlhf.models", None)
        sys.modules.pop("safe_rlhf", None)
        raise
    return AutoModelForScore

def factors(x):
    return [i for i in range(1,x+1) if x%i==0]

def auto_size(seq_len, topk):
    estimated = (28672/(seq_len*1.5)) -11.52605
    # hack
    possible_facs = factors(topk)
    if np.all(~(np.array(possible_facs[::-1]) < estimated)): return 1
    return possible_facs[::-1][np.argmax(np.array(possible_facs[::-1]) < estimated)]
###

def create_attention_mask(seq_len, bsz=1):
    return torch.ones((bsz, seq_len))

# From huggingface
def rcache(past_key_values, beam_idx):
    reordered_past = ()
    for layer_past in past_key_values:
        reordered_past += (
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
        )
    return reordered_past

def even_chunk(data, chunk_size=10):
    assert data.shape[0] % chunk_size == 0, "chunk_size must evenly divide the topk"
    for i in range(0, data.shape[0], chunk_size):
        yield data[i:(i+chunk_size)]

# reward based search
class ARGS:
    def __init__(self, llm_path, rm_path, llm_dev="cuda:0", rm_dev="cuda:1", torch_dtype=torch.float16, rm_base_path=None):
        self.llm_dev = llm_dev
        self.rm_dev = rm_dev
        print("Loading LLM...")
        self.LLM = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch_dtype).to(self.llm_dev)
        self.LLM.eval()
        
        print(f"Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.LLM.config.pad_token_id = self.tokenizer.pad_token_id
        self.rm_supports_prefix_cache = True
        self.rm_uses_text_tokenizer = False
        self.rm_tokenizer = None
        self.rm_positive_label_id = None

        print("Loading RM...")
        rm_path_obj = Path(rm_path)
        rad_checkpoint = rm_path_obj / "pytorch_model.bin"
        if rm_base_path is not None and rad_checkpoint.exists() and not (rm_path_obj / "config.json").exists():
            # The local RAD checkpoint is a GPT-2 LM whose vocabulary head was
            # replaced by a scalar reward head. It uses the same GPT-2 tokenizer
            # as ARGS, so it can serve as ARGS' token-prefix reward model.
            from Method.RAD.reward_modeling.reward_model import GPT2RewardModel

            self.RM = GPT2RewardModel(
                reward_model_name=str(rm_base_path),
                out_features=1,
                loss_fn="cumulative_mse",
            )
            state_dict = torch.load(rad_checkpoint, map_location="cpu", weights_only=True)
            load_result = self.RM.load_state_dict(state_dict, strict=False)
            allowed_suffixes = (".attn.bias", ".attn.masked_bias")
            unexpected = [
                key for key in load_result.unexpected_keys
                if not key.endswith(allowed_suffixes)
            ]
            if load_result.missing_keys or unexpected:
                raise RuntimeError(
                    "Incompatible RAD reward checkpoint: "
                    f"missing={load_result.missing_keys}, unexpected={unexpected}"
                )
            self.RM.config = self.RM.model.config
            self.RM.to(device=self.rm_dev, dtype=torch_dtype)
        elif (rm_path_obj / "config.json").exists():
            rm_config = transformers.AutoConfig.from_pretrained(rm_path)
            architectures = set(getattr(rm_config, "architectures", []) or [])
            if "LlamaForScore" in architectures:
                AutoModelForScore = load_safe_rlhf_score_model_class(SAFE_RLHF_SOURCE)
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
                self.RM = AutoModelForScore.from_pretrained(
                    rm_path,
                    torch_dtype=torch_dtype,
                    quantization_config=quantization_config,
                    device_map={"": self.rm_dev},
                )
                self.rm_supports_prefix_cache = False
            else:
                self.RM = AutoModelForSequenceClassification.from_pretrained(
                    rm_path, torch_dtype=torch_dtype,
                ).to(self.rm_dev)
                self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)
                self.rm_positive_label_id = next(
                    (
                        int(index) for index, label in self.RM.config.id2label.items()
                        if "POS" in str(label).upper()
                    ),
                    1 if self.RM.config.num_labels == 2 else None,
                )
                self.rm_supports_prefix_cache = False
                self.rm_uses_text_tokenizer = True
        else:
            self.RM = AutoModelForSequenceClassification.from_pretrained(
                rm_path, torch_dtype=torch_dtype,
            ).to(self.rm_dev)
            self.rm_tokenizer = AutoTokenizer.from_pretrained(rm_path)
            self.rm_positive_label_id = next(
                (
                    int(index) for index, label in self.RM.config.id2label.items()
                    if "POS" in str(label).upper()
                ),
                1 if self.RM.config.num_labels == 2 else None,
            )
            self.rm_supports_prefix_cache = False
            self.rm_uses_text_tokenizer = True
        if not self.rm_uses_text_tokenizer:
            self.RM.config.pad_token_id = self.tokenizer.pad_token_id
        self.RM.eval()

    def _reward_forward(self, **kwargs):
        output = self.RM(**kwargs)
        if hasattr(output, "end_scores"):
            return SimpleNamespace(logits=output.end_scores, past_key_values=None)
        if isinstance(output, tuple):
            _, logits, past_key_values = output
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)
        return output

    def _score_reward_candidates(self, input_ids, rm_cached=None):
        if self.rm_uses_text_tokenizer:
            texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            rm_inputs = self.rm_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.RM.config.max_position_embeddings,
                return_tensors="pt",
            ).to(self.rm_dev)
            logits = self.RM(**rm_inputs).logits
            if logits.shape[-1] == 1:
                rewards = logits.squeeze(-1)
            elif self.rm_positive_label_id is not None:
                rewards = F.log_softmax(logits, dim=-1)[:, self.rm_positive_label_id]
            else:
                raise ValueError("Không xác định được positive label của reward model")
            return SimpleNamespace(logits=rewards, past_key_values=None)
        input_ids = input_ids.to(self.rm_dev)
        attention_mask = create_attention_mask(
            input_ids.shape[1], input_ids.shape[0]
        ).to(self.rm_dev)
        if not self.rm_supports_prefix_cache:
            return self._reward_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        rm_inputs = self.LLM.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=rm_cached,
            use_cache=True,
        )
        return self._reward_forward(**rm_inputs)
        
    def get_input_ids(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.llm_dev)
        return tokens
    
    def tokens_to_text(self, tokens: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
    
    def generate_greedy_step_large(self, mout, input_ids, pre_screen_beam_width=40, weight=0., rm_cached=None, chunk_size=10, debug=True, _use_cache=True):
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")

        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")
        
        new_rm_cached = None
        current_best_score = None
        current_best_tokens = None
        if debug: print(f"{prescreen_logits.flatten().shape=}")
        for chunk, chunk_logits in zip(even_chunk(flat_trme.to(self.rm_dev), chunk_size), even_chunk(prescreen_logits.flatten(), chunk_size)):
            pkv = None if not _use_cache else rm_cached

            rm_out = self._score_reward_candidates(chunk, pkv)
            current_rm_cached = rm_out.past_key_values
            rewards = rm_out.logits.flatten().to(self.llm_dev)
            del rm_out
            if debug: print(f"{rewards=}")
            if debug: print(f"{rewards.shape=}")
            new_scores = rewards * weight + chunk_logits
            if debug: print(f"{new_scores=}")
            
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
            current_score = new_scores[top_k_ids[0]].item()
            if debug: print(f"{current_score=} {current_best_score=} ")
            if (current_best_score is None) or (current_score > current_best_score):
                if debug: print(f"Updated!!")
                
                current_best_score = current_score
                current_best_tokens = chunk.to(self.llm_dev)[top_k_ids]
                if current_rm_cached is not None:
                    new_rm_cached = self.LLM._reorder_cache(current_rm_cached, top_k_ids.repeat(chunk_size,))
            
        if debug: print(f"{new_scores.shape=}")
        
        return current_best_tokens, new_rm_cached
        
    def generate_step(self, mout, input_ids, pre_screen_beam_width=40, weight=0., method="greedy", temperature=0.7, rm_cached=None, debug=True):
        out_logits = mout.logits[:, -1]

        prescreen_logits, prescreen_tokens = torch.topk(out_logits, dim=-1, k=pre_screen_beam_width)

        expanded_tis = torch.unsqueeze(input_ids, 1).repeat(1, pre_screen_beam_width, 1)
        if debug: print(f"{expanded_tis.shape=}")

        to_rm_eval = torch.dstack((expanded_tis, prescreen_tokens))
        if debug: print(f"{to_rm_eval.shape=}")

        if debug: print(f"{out_logits.shape[0] * pre_screen_beam_width=}")
        flat_trme = to_rm_eval.view(out_logits.shape[0] * pre_screen_beam_width, -1)
        if debug: print(f"{flat_trme.shape=}")

        rm_out = self._score_reward_candidates(flat_trme, rm_cached)
        rm_cached = rm_out.past_key_values

        if debug: print(f"{rm_out.logits.flatten()=}")

        rewards = rm_out.logits.flatten().to(self.llm_dev)
        del rm_out
        if debug: print(f"{rewards.shape=}")

        new_scores = rewards * weight + prescreen_logits.flatten()
        if debug: print(f"{new_scores.shape=}")

        if method == "greedy":
            _, top_k_ids = torch.topk(new_scores, dim=-1, k=1)
        elif method == "topk":
            # assume B=1
            assert input_ids.shape[0] == 1
            new_scores = new_scores / temperature
            scores = F.softmax(new_scores, dim=-1)
            top_k_ids = torch.multinomial(scores, num_samples=1)
        else:
            raise ValueError(f"Invalid method '{method}'")
            
        if debug: print(f"{top_k_ids.shape=}")
        if rm_cached is not None:
            rm_cached = self.LLM._reorder_cache(rm_cached, top_k_ids.repeat(pre_screen_beam_width,))
        if debug: print(f"{rewards[top_k_ids]=}")

        return flat_trme[top_k_ids], rm_cached
    
    def generate(self, prompt, weight=0., topk=1, max_new_token=128, method="greedy", temperature=0.7, chunk_size=5, debug=False):
        tokens = self.get_input_ids(prompt)
        initial_len = tokens.shape[-1]
        if chunk_size == "auto":
            chunk_size = auto_size(initial_len + max_new_token, topk)
            print(f"auto {chunk_size=}, {topk=}, {initial_len=}!")
        
        if tokens.shape[-1] > self.LLM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
        
        if tokens.shape[-1] > self.RM.config.to_dict().get("max_sequence_length", 2048):
            print("The sequence of tokens is too long!!! Returning none!")
            return None
          
        rm_cached = None
        cached = None
        
        iterator_obj = range(max_new_token)
        if debug: iterator_obj = tqdm(iterator_obj)
        for _ in iterator_obj:
            if debug: print(f"{type(cached)=}")
            if debug: print(f"{type(rm_cached)=}")
            with torch.no_grad():
                if cached is None:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=None, use_cache=True))
                    cached = mout.past_key_values
                else:
                    mout = self.LLM(**self.LLM.prepare_inputs_for_generation(input_ids=tokens, attention_mask=create_attention_mask(tokens.shape[1], tokens.shape[0]).to(self.llm_dev), past_key_values=cached, use_cache=True))
                    cached = mout.past_key_values
                
                if method == "greedy_large":
                    if debug: print("large")
                    tokens, rm_cached = self.generate_greedy_step_large(mout, tokens, topk, weight, rm_cached, chunk_size, debug)   
                else:
                    tokens, rm_cached = self.generate_step(mout, tokens, topk, weight, method, temperature, rm_cached, debug)
                del mout

        return tokens
