"""Read-only access to PARM's vendored runtime and preference-aware adapter."""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARM_ROOT = PROJECT_ROOT / "PARM"
PARM_PEFT_SRC = PARM_ROOT / "peft" / "src"
PARM_ARITHMETIC_SRC = PARM_ROOT / "language-model-arithmetic" / "src"
PARM_GENERATION_SOURCE = PARM_ROOT / "code" / "evaluation" / "generate_outputs.py"
NAMED_PREFERENCE_ORDER = ("helpfulness", "harmlessness")
PARM_PBLORA_PREFERENCE_ORDER = ("harmlessness", "helpfulness")


@dataclass(frozen=True)
class VendoredRuntime:
    peft: ModuleType
    model_arithmetic: ModuleType
    origins: dict[str, str]


def _within(path: str | Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def activate_vendored_parm_runtime() -> VendoredRuntime:
    """Import PARM dependencies from their local read-only source trees."""

    # PARM is an immutable author-code boundary. Keep Python from materializing
    # import caches anywhere under its vendored source trees.
    sys.dont_write_bytecode = True
    for path in (PARM_ARITHMETIC_SRC, PARM_PEFT_SRC):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    loaded_peft = sys.modules.get("peft")
    if loaded_peft is not None and not _within(
        getattr(loaded_peft, "__file__", ""), PARM_PEFT_SRC
    ):
        raise RuntimeError(
            "A non-PARM peft module is already loaded; launch PARM-TARO in a "
            "fresh process so the vendored dependency can be selected"
        )
    peft = importlib.import_module("peft")
    model_arithmetic = importlib.import_module("model_arithmetic")
    origins = {
        "peft": str(Path(peft.__file__).resolve()),
        "model_arithmetic": str(Path(model_arithmetic.__file__).resolve()),
    }
    if not _within(origins["peft"], PARM_PEFT_SRC):
        raise RuntimeError("peft did not resolve to PARM/peft")
    if not _within(origins["model_arithmetic"], PARM_ARITHMETIC_SRC):
        raise RuntimeError(
            "model_arithmetic did not resolve to PARM/language-model-arithmetic"
        )
    return VendoredRuntime(peft, model_arithmetic, origins)


def load_original_parm_generation_module() -> ModuleType:
    """Load PARM's original generation module without writing into PARM/."""

    activate_vendored_parm_runtime()
    module_name = "parm_taro_readonly_original_generate_outputs"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    specification = importlib.util.spec_from_file_location(
        module_name, PARM_GENERATION_SOURCE
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load original PARM source: {PARM_GENERATION_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        specification.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    if not callable(getattr(module, "get_model_arithmetic", None)):
        raise ImportError("Original PARM generation API is unavailable")
    sys.modules[module_name] = module
    return module


def inspect_parm_adapter_config(
    adapter_path: str | Path,
    *,
    expected_preference_dim: int,
) -> dict[str, Any]:
    path = Path(adapter_path)
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing PARM adapter config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("PARM adapter config must be a JSON object")
    if str(config.get("peft_type", "")).upper() != "PBLORA":
        raise ValueError(
            "PARM-TARO requires a preference-aware PBLORA checkpoint; "
            f"got {config.get('peft_type')!r}"
        )
    if int(config.get("obj_num", -1)) != expected_preference_dim:
        raise ValueError("PBLORA obj_num does not match preference_dim")
    if not (path / "adapter_model.safetensors").is_file() and not (
        path / "adapter_model.bin"
    ).is_file():
        raise FileNotFoundError("Missing PBLORA adapter weights")
    return config


def _preference_tensors(model: Any) -> Iterable[tuple[str, torch.Tensor]]:
    seen: set[int] = set()
    for collection in (model.named_parameters(), model.named_buffers()):
        for name, value in collection:
            if "pref_vec" not in name or id(value) in seen:
                continue
            seen.add(id(value))
            yield name, value


def set_parm_preference(model: Any, preference: torch.Tensor) -> list[str]:
    """Set every in-memory PBLORA preference vector without touching files."""

    values = preference.detach().flatten()
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("preference must be a finite non-empty vector")
    updated: list[str] = []
    with torch.no_grad():
        for name, target in _preference_tensors(model):
            if target.numel() != values.numel():
                raise ValueError(
                    f"Preference dimension mismatch for {name}: "
                    f"expected {target.numel()}, got {values.numel()}"
                )
            target.copy_(values.to(device=target.device, dtype=target.dtype))
            updated.append(name)
    if not updated:
        raise ValueError("No PBLORA pref_vec tensors were found in the guide")
    return updated


def named_preference_to_parm(preference: torch.Tensor) -> torch.Tensor:
    """Convert [helpfulness, harmlessness] to PARM's [safe, help] order."""

    if preference.shape[-1:] != (2,):
        raise ValueError("PARM preference conversion requires final dimension 2")
    if not bool(torch.isfinite(preference).all()):
        raise ValueError("preference must be finite")
    return preference.index_select(
        -1,
        torch.tensor([1, 0], dtype=torch.long, device=preference.device),
    )


def freeze_for_inference(*models: Any) -> None:
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
