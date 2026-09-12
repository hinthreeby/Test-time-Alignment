"""Production frozen-model loading and Stage 9 prerequisite audits."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from PARM_TARO.adapters.parm_adapter import (
    activate_vendored_parm_runtime,
    freeze_for_inference,
    inspect_parm_adapter_config,
)
from PARM_TARO.adapters.token_alignment import PARMTokenAlignment
from PARM_TARO.data.multi_objective import validate_processed_dataset
from PARM_TARO.training.config import ParmRouterTrainingConfig
from router_v2.device import resolve_device
from router_v2.guide_model.provenance import tokenizer_descriptor
from router_v2.training.audit import assert_frozen_models, hash_tree


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "parm_taro"


@dataclass(frozen=True)
class ParmTrainingRuntime:
    base_model: Any
    guide_model: Any
    tokenizer: Any
    alignment: PARMTokenAlignment
    device: torch.device
    tokenizer_descriptor: dict[str, Any]
    adapter_config: dict[str, Any]
    paths: dict[str, Path]
    model_load_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdapterLayerState:
    """Observable state of one real PEFT adapter layer."""

    name: str
    module_type: str
    disabled: bool
    active_adapters: tuple[str, ...]


@dataclass(frozen=True)
class AdapterStateTransition:
    before: tuple[AdapterLayerState, ...]
    inside: tuple[AdapterLayerState, ...]
    after: tuple[AdapterLayerState, ...]
    requires_grad_restored: bool


def _normalized_active_adapters(module: Any) -> tuple[str, ...]:
    value = getattr(module, "active_adapters", None)
    if callable(value) or value is None:
        value = getattr(module, "active_adapter", None)
    if callable(value) or value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def capture_adapter_layer_state(model: Any) -> tuple[AdapterLayerState, ...]:
    """Capture PEFT layer state without confusing methods with state flags.

    Transformers models expose a callable ``disable_adapters`` method.  Only
    modules carrying PEFT's private boolean ``_disable_adapters`` field are
    adapter layers whose enabled/disabled state can be inspected.
    """

    states = []
    for name, module in model.named_modules():
        disabled = vars(module).get("_disable_adapters")
        if not isinstance(disabled, bool):
            continue
        states.append(
            AdapterLayerState(
                name=name,
                module_type=(
                    f"{type(module).__module__}.{type(module).__qualname__}"
                ),
                disabled=disabled,
                active_adapters=_normalized_active_adapters(module),
            )
        )
    return tuple(states)


def _restore_adapter_layer_state(
    model: Any,
    expected: tuple[AdapterLayerState, ...],
) -> None:
    modules = dict(model.named_modules())
    for state in expected:
        module = modules.get(state.name)
        if module is None:
            raise RuntimeError(f"Adapter layer disappeared: {state.name}")
        current_active = _normalized_active_adapters(module)
        if current_active != state.active_adapters:
            setter = getattr(module, "set_adapter", None)
            if not callable(setter) or not state.active_adapters:
                raise RuntimeError(
                    f"Cannot restore active adapter for layer {state.name}"
                )
            adapter_value: str | list[str]
            if len(state.active_adapters) == 1:
                adapter_value = state.active_adapters[0]
            else:
                adapter_value = list(state.active_adapters)
            setter(adapter_value)
        disabled = vars(module).get("_disable_adapters")
        if disabled != state.disabled:
            toggler = getattr(module, "enable_adapters", None)
            if not callable(toggler):
                raise RuntimeError(
                    f"Cannot restore enabled state for adapter layer {state.name}"
                )
            toggler(not state.disabled)


class FrozenBaseModelView:
    """Run a frozen PEFT model as its base LM with adapters disabled."""

    def __init__(self, peft_model: Any) -> None:
        if not callable(getattr(peft_model, "disable_adapter", None)):
            raise TypeError("Shared Stage 9 backbone must support disable_adapter()")
        self.peft_model = peft_model
        self.config = peft_model.config
        self.last_adapter_transition: AdapterStateTransition | None = None

    def eval(self) -> "FrozenBaseModelView":
        self.peft_model.eval()
        return self

    def parameters(self):
        return self.peft_model.parameters()

    def named_parameters(self):
        return self.peft_model.named_parameters()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        before = capture_adapter_layer_state(self.peft_model)
        if not before:
            raise RuntimeError("Shared Stage 9 backbone has no stateful adapter layers")
        parameters = tuple(self.peft_model.named_parameters())
        requires_grad_before = {
            name: parameter.requires_grad for name, parameter in parameters
        }
        inside: tuple[AdapterLayerState, ...] = ()
        try:
            with self.peft_model.disable_adapter():
                inside = capture_adapter_layer_state(self.peft_model)
                if tuple(state.name for state in inside) != tuple(
                    state.name for state in before
                ):
                    raise RuntimeError("Adapter layer set changed during base forward")
                if not all(state.disabled for state in inside):
                    enabled = [state.name for state in inside if not state.disabled]
                    raise RuntimeError(
                        "PBLORA remained enabled during base forward: "
                        f"{enabled[:5]}"
                    )
                output = self.peft_model(*args, **kwargs)
        finally:
            # Vendored PEFT's context re-enables adapters by calling set_adapter,
            # which can mark frozen adapter weights trainable. Restore both the
            # activation state and each parameter's original frozen flag.
            _restore_adapter_layer_state(self.peft_model, before)
            current_parameters = dict(self.peft_model.named_parameters())
            if set(current_parameters) != set(requires_grad_before):
                raise RuntimeError("Model parameters changed during base forward")
            for name, parameter in current_parameters.items():
                parameter.requires_grad_(requires_grad_before[name])
            after = capture_adapter_layer_state(self.peft_model)
            requires_grad_restored = all(
                current_parameters[name].requires_grad == expected
                for name, expected in requires_grad_before.items()
            )
            self.last_adapter_transition = AdapterStateTransition(
                before=before,
                inside=inside,
                after=after,
                requires_grad_restored=requires_grad_restored,
            )
            if after != before or not requires_grad_restored:
                raise RuntimeError("PBLORA state was not restored after base forward")
        return output


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolved_paths(config: ParmRouterTrainingConfig) -> dict[str, Path]:
    paths = {
        "data_root": project_path(config.data_root),
        "base_model": project_path(config.base_model_path),
        "tokenizer": project_path(config.tokenizer_path),
        "parm_adapter": project_path(config.parm_adapter_path),
        "router_config": project_path(config.router_config_path),
        "output_dir": project_path(config.output_dir),
    }
    if not _within(paths["output_dir"], RESULTS_ROOT):
        raise ValueError("Stage 9 output_dir must stay under results/parm_taro")
    return paths


def audit_training_prerequisites(
    config: ParmRouterTrainingConfig,
) -> dict[str, Any]:
    paths = resolved_paths(config)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "paths": {name: str(path) for name, path in paths.items()},
    }
    checks["base_model_present"] = (paths["base_model"] / "config.json").is_file()
    checks["tokenizer_present"] = paths["tokenizer"].exists()
    checks["router_config_present"] = paths["router_config"].is_file()
    try:
        data_manifest = json.loads(
            (paths["data_root"] / "manifest.json").read_text(encoding="utf-8")
        )
        validate_processed_dataset(
            paths["data_root"],
            source_path=Path(data_manifest["source"]["path"]),
        )
        checks["stage8_data_valid"] = True
        details["data"] = {
            "records": data_manifest["total_records"],
            "split_counts": {
                split: value["records"]
                for split, value in data_manifest["splits"].items()
            },
            "manifest": str(paths["data_root"] / "manifest.json"),
        }
    except Exception as error:
        checks["stage8_data_valid"] = False
        details["data_error"] = f"{type(error).__name__}: {error}"
    try:
        adapter = inspect_parm_adapter_config(
            paths["parm_adapter"], expected_preference_dim=2
        )
        guide_base = project_path(str(adapter["base_model_name_or_path"]))
        if not (guide_base / "config.json").is_file():
            raise FileNotFoundError(f"Missing local PBLORA base model: {guide_base}")
        checks["frozen_two_objective_pblora_present"] = True
        details["parm_adapter"] = {
            "peft_type": adapter["peft_type"],
            "obj_num": adapter["obj_num"],
            "guide_base_model": str(guide_base),
        }
        checks["shared_backbone_compatible"] = (
            not config.share_base_backbone
            or guide_base.resolve() == paths["base_model"].resolve()
        )
    except Exception as error:
        checks["frozen_two_objective_pblora_present"] = False
        checks["shared_backbone_compatible"] = False
        details["parm_adapter_error"] = f"{type(error).__name__}: {error}"
    details["parm_tree"] = hash_tree(PROJECT_ROOT / "PARM")
    checks["parm_tree_matches_baseline"] = (
        details["parm_tree"]["tree_sha256"]
        == "dcbaae05b0c3467e2d9ff3255ca6fc9f2296bc4e97f9a22b650e38f666bac23a"
    )
    ready = all(checks.values())
    return {
        "schema_version": 1,
        "stage": 9,
        "checks": checks,
        "details": details,
        "ready": ready,
        "status": "READY" if ready else "PREREQUISITES_REQUIRED",
    }


def _dtype(name: str, device: torch.device) -> torch.dtype:
    value = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]
    return torch.float32 if device.type == "cpu" else value


def _model_kwargs(
    config: ParmRouterTrainingConfig,
    device: torch.device,
    output_dir: Path,
    component: str,
    *,
    bitsandbytes_config_class: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": _dtype(config.model_dtype, device),
    }
    if device.type == "cuda" and config.load_in_4bit:
        if bitsandbytes_config_class is None:
            raise RuntimeError("CUDA 4-bit loading requires BitsAndBytesConfig")
        kwargs.update(
            {
                "quantization_config": bitsandbytes_config_class(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=_dtype(config.model_dtype, device),
                    bnb_4bit_quant_type=config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=(
                        config.bnb_4bit_use_double_quant
                    ),
                ),
                # A quantized Tulu-2-7B plus the PBLORA and Router fit on the
                # RTX 3080. Pinning the one model avoids auto-dispatch trying
                # to fill all VRAM before the adapter is attached.
                "device_map": {"": device.index if device.index is not None else 0},
            }
        )
    elif device.type == "cuda" and config.model_placement == "accelerate_auto":
        kwargs.update(
            {
                "device_map": "auto",
                "max_memory": {
                    0: f"{config.cuda_memory_per_model_gib}GiB",
                    "cpu": "64GiB",
                },
                "offload_folder": str(output_dir / "model_offload" / component),
            }
        )
    return kwargs


def _load_frozen_model_pair(
    config: ParmRouterTrainingConfig,
    paths: dict[str, Path],
    guide_base: Path,
    *,
    device: torch.device,
    auto_model_class: Any,
    peft_model_class: Any,
    bitsandbytes_config_class: Any | None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load exactly one physical backbone when shared mode is enabled."""

    if config.share_base_backbone and guide_base.resolve() != paths[
        "base_model"
    ].resolve():
        raise ValueError(
            "Shared backbone requires base_model_path to match the PBLORA base"
        )

    def load_backbone(path: Path, component: str) -> Any:
        kwargs = _model_kwargs(
            config,
            device,
            paths["output_dir"],
            component,
            bitsandbytes_config_class=bitsandbytes_config_class,
        )
        model = auto_model_class.from_pretrained(path, **kwargs)
        if config.model_placement == "single_device" and not (
            device.type == "cuda" and config.load_in_4bit
        ):
            model = model.to(device)
        if device.type == "cuda" and config.load_in_4bit and not bool(
            getattr(model, "is_loaded_in_4bit", False)
        ):
            raise RuntimeError("Backbone did not load in required CUDA 4-bit mode")
        return model

    if config.share_base_backbone:
        backbone = load_backbone(paths["base_model"], "shared_backbone")
        guide_model = peft_model_class.from_pretrained(
            backbone,
            paths["parm_adapter"],
            is_trainable=False,
        )
        return (
            FrozenBaseModelView(guide_model),
            guide_model,
            {
                "physical_backbone_count": 1,
                "shared_backbone": True,
                "load_in_4bit": bool(device.type == "cuda" and config.load_in_4bit),
                "base_pass": "PeftModel.disable_adapter",
                "guide_pass": "PBLORA_enabled",
            },
        )

    base_model = load_backbone(paths["base_model"], "base")
    guide_backbone = load_backbone(guide_base, "guide")
    guide_model = peft_model_class.from_pretrained(
        guide_backbone,
        paths["parm_adapter"],
        is_trainable=False,
    )
    return (
        base_model,
        guide_model,
        {
            "physical_backbone_count": 2,
            "shared_backbone": False,
            "load_in_4bit": bool(device.type == "cuda" and config.load_in_4bit),
            "base_pass": "standalone_base",
            "guide_pass": "PBLORA_enabled",
        },
    )


def load_training_runtime(
    config: ParmRouterTrainingConfig,
) -> ParmTrainingRuntime:
    audit = audit_training_prerequisites(config)
    if not audit["ready"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError(f"Stage 9 prerequisites are not ready: {failed}")
    paths = resolved_paths(config)
    adapter_config = inspect_parm_adapter_config(
        paths["parm_adapter"], expected_preference_dim=2
    )
    guide_base = project_path(str(adapter_config["base_model_name_or_path"]))
    vendored = activate_vendored_parm_runtime()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    device = resolve_device(
        config.device, allow_cpu_fallback=config.allow_cpu_fallback
    )
    tokenizer = AutoTokenizer.from_pretrained(
        paths["tokenizer"], local_files_only=True, use_fast=True
    )
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Stage 9 requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model, guide_model, model_load_info = _load_frozen_model_pair(
        config,
        paths,
        guide_base,
        device=device,
        auto_model_class=AutoModelForCausalLM,
        peft_model_class=vendored.peft.PeftModel,
        bitsandbytes_config_class=BitsAndBytesConfig,
    )
    freeze_for_inference(base_model, guide_model)
    assert_frozen_models(base_model, guide_model)
    alignment = PARMTokenAlignment.validate(
        tokenizer,
        base_vocab_size=int(base_model.config.vocab_size),
        guide_vocab_size=int(guide_model.config.vocab_size),
    )
    descriptor = tokenizer_descriptor(tokenizer, paths["tokenizer"])
    return ParmTrainingRuntime(
        base_model=base_model,
        guide_model=guide_model,
        tokenizer=tokenizer,
        alignment=alignment,
        device=device,
        tokenizer_descriptor=descriptor,
        adapter_config=adapter_config,
        paths={**paths, "guide_base_model": guide_base},
        model_load_info=model_load_info,
    )
