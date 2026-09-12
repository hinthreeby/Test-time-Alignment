"""Alpha-conditioned residual extension for the isolated Stage 9 Pilot V3."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from router_v2.smart_config import SmartRouterConfig
from router_v2.smart_model import SmartTokenRouter


ALPHA_RESIDUAL_CHECKPOINT_FORMAT = "parm_taro.alpha_residual_router_checkpoint"
ALPHA_RESIDUAL_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AlphaResidualConfig:
    schema_version: int = 1
    method_label: str = "PARM_TARO_ALPHA_RESIDUAL_ROUTER"
    preference_embedding_dim: int = 16
    state_bottleneck_dim: int = 64
    residual_hidden_dim: int = 32
    activation: str = "tanh"
    residual_output_weight_std: float = 0.01

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported alpha residual schema")
        if self.method_label != "PARM_TARO_ALPHA_RESIDUAL_ROUTER":
            raise ValueError("Invalid alpha residual method_label")
        if self.activation != "tanh":
            raise ValueError("Pilot V3 alpha residual requires tanh")
        dimensions = (
            self.preference_embedding_dim,
            self.state_bottleneck_dim,
            self.residual_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Alpha residual dimensions must be positive")
        if self.residual_output_weight_std <= 0.0:
            raise ValueError("Residual output weight std must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "AlphaResidualConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown alpha residual fields: {unknown}")
        return cls(**dict(values))


class AlphaResidualSmartRouter(SmartTokenRouter):
    """Keep the V2 state router and add normalized alpha-state logit residual."""

    def __init__(
        self,
        config: SmartRouterConfig,
        residual_config: AlphaResidualConfig,
    ) -> None:
        if not config.use_preference:
            raise ValueError("Alpha residual Router requires preference features")
        if config.preference_embedding_dim != residual_config.preference_embedding_dim:
            raise ValueError("Preference embedding dimension mismatch")
        if config.fusion_bottleneck_dim != residual_config.state_bottleneck_dim:
            raise ValueError("State bottleneck dimension mismatch")
        super().__init__(config)
        self.residual_config = residual_config
        self.preference_residual_norm = nn.LayerNorm(
            residual_config.preference_embedding_dim
        )
        self.state_residual_norm = nn.LayerNorm(
            residual_config.state_bottleneck_dim
        )
        self.alpha_state_residual = nn.Sequential(
            nn.Linear(
                residual_config.preference_embedding_dim
                + residual_config.state_bottleneck_dim,
                residual_config.residual_hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(residual_config.residual_hidden_dim, 1),
        )
        final = self.alpha_state_residual[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Alpha residual must end in a linear layer")
        nn.init.normal_(
            final.weight,
            mean=0.0,
            std=residual_config.residual_output_weight_std,
        )
        nn.init.zeros_(final.bias)

    def _fuse(
        self,
        groups: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected = tuple(self.feature_slices)
        if set(groups) != set(expected):
            raise ValueError(
                f"Fusion feature groups mismatch: expected={expected}, "
                f"got={tuple(groups)}"
            )
        if "preference" not in groups:
            raise ValueError("Alpha residual requires preference representation")
        features = torch.cat([groups[name] for name in expected], dim=-1)
        if features.shape[-1] != self.feature_dim:
            raise RuntimeError("Alpha residual received invalid fusion dimension")
        normalized = self.fusion_mlp[0](features)
        hidden = self.fusion_mlp[2](self.fusion_mlp[1](normalized))
        state = self.fusion_mlp[4](self.fusion_mlp[3](hidden))
        base_logit = self.fusion_mlp[5](state)
        residual_input = torch.cat(
            (
                self.preference_residual_norm(groups["preference"]),
                self.state_residual_norm(state),
            ),
            dim=-1,
        )
        residual = self.alpha_state_residual(residual_input)
        gate = torch.sigmoid(base_logit + residual).clamp(
            self.config.lambda_eps,
            1.0 - self.config.lambda_eps,
        )
        return gate, self.config.lambda_max * gate

    def alpha_residual_parameters(self) -> tuple[nn.Parameter, ...]:
        modules = (
            self.preference_encoder,
            self.preference_residual_norm,
            self.state_residual_norm,
            self.alpha_state_residual,
        )
        return tuple(
            parameter
            for module in modules
            if module is not None
            for parameter in module.parameters()
        )


def build_from_v2_router(
    source: SmartTokenRouter,
    residual_config: AlphaResidualConfig,
) -> AlphaResidualSmartRouter:
    if not source.config.use_preference:
        raise ValueError("Pilot V3 source must be alpha-aware")
    model = AlphaResidualSmartRouter(source.config, residual_config).to(
        next(source.parameters()).device
    )
    result = model.load_state_dict(source.state_dict(), strict=False)
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith(
            (
                "preference_residual_norm.",
                "state_residual_norm.",
                "alpha_state_residual.",
            )
        )
    }
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise ValueError(
            "Unexpected V2-to-V3 state transplant mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    return model


def alpha_path_parameter_groups(
    model: AlphaResidualSmartRouter,
) -> dict[str, tuple[nn.Parameter, ...]]:
    groups: dict[str, list[nn.Parameter]] = {
        "preference_encoder": [],
        "alpha_residual": [],
        "fusion_layers": [],
        "final_lambda_head": [],
        "state_router_rest": [],
    }
    for name, parameter in model.named_parameters():
        if name.startswith("preference_encoder."):
            group = "preference_encoder"
        elif name.startswith(
            (
                "preference_residual_norm.",
                "state_residual_norm.",
                "alpha_state_residual.",
            )
        ):
            group = "alpha_residual"
        elif name.startswith("fusion_mlp.5."):
            group = "final_lambda_head"
        elif name.startswith("fusion_mlp."):
            group = "fusion_layers"
        else:
            group = "state_router_rest"
        groups[group].append(parameter)
    return {name: tuple(parameters) for name, parameters in groups.items()}


def save_alpha_residual_checkpoint(
    model: AlphaResidualSmartRouter,
    path: str | Path,
    *,
    training_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
    overwrite: bool = False,
) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite Pilot V3 checkpoint: {output}")
    payload = {
        "schema_version": ALPHA_RESIDUAL_CHECKPOINT_SCHEMA_VERSION,
        "format": ALPHA_RESIDUAL_CHECKPOINT_FORMAT,
        "method_label": model.residual_config.method_label,
        "model_class": "AlphaResidualSmartRouter",
        "smart_router_config": model.config.to_dict(),
        "alpha_residual_config": model.residual_config.to_dict(),
        "state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "training_state": dict(training_state),
        "metadata": dict(metadata),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        if output.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite Pilot V3 checkpoint: {output}"
            )
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_alpha_residual_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[AlphaResidualSmartRouter, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    required = {
        "schema_version",
        "format",
        "method_label",
        "model_class",
        "smart_router_config",
        "alpha_residual_config",
        "state_dict",
        "training_state",
        "metadata",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Invalid AlphaResidualSmartRouter checkpoint keys")
    if payload["schema_version"] != ALPHA_RESIDUAL_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Incompatible alpha residual checkpoint schema")
    if payload["format"] != ALPHA_RESIDUAL_CHECKPOINT_FORMAT:
        raise ValueError("Invalid alpha residual checkpoint format")
    if payload["model_class"] != "AlphaResidualSmartRouter":
        raise ValueError("Checkpoint model class mismatch")
    residual_config = AlphaResidualConfig.from_dict(payload["alpha_residual_config"])
    if payload["method_label"] != residual_config.method_label:
        raise ValueError("Checkpoint method label mismatch")
    model = AlphaResidualSmartRouter(
        SmartRouterConfig.from_dict(payload["smart_router_config"]),
        residual_config,
    ).to(torch.device(map_location))
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload
