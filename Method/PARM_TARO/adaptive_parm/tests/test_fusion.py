from __future__ import annotations

import pytest
import torch

from PARM_TARO.adaptive_parm import (
    adaptive_parm_distribution,
    adaptive_parm_logits,
    lambda_to_weight,
)


def policies(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    base = torch.log_softmax(torch.randn(3, 4, 17, generator=generator), dim=-1).to(dtype)
    guide = torch.log_softmax(torch.randn(3, 4, 17, generator=generator), dim=-1).to(dtype)
    return base, guide


def test_weight_zero_is_exact_base_distribution() -> None:
    base, guide = policies()
    actual = adaptive_parm_distribution(base, guide, 0.0)
    torch.testing.assert_close(actual, base.exp(), atol=1e-6, rtol=1e-6)


def test_weight_one_is_exact_parm_distribution() -> None:
    base, guide = policies()
    actual = adaptive_parm_distribution(base, guide, 1.0)
    torch.testing.assert_close(actual, guide.exp(), atol=1e-6, rtol=1e-6)


def test_midpoint_logits_are_exact_arithmetic_midpoint() -> None:
    base, guide = policies()
    actual = adaptive_parm_logits(base, guide, 0.5)
    torch.testing.assert_close(actual, (base + guide) / 2.0, atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("lam", [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0])
def test_lambda_to_weight_parity(lam: float) -> None:
    base, guide = policies()
    weight = lambda_to_weight(lam).to(base.dtype)
    legacy_normalized = (base + lam * guide) / (1.0 + lam)
    canonical = adaptive_parm_logits(base, guide, weight)
    torch.testing.assert_close(canonical, legacy_normalized, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_supported_dtypes_are_finite_and_inputs_are_not_mutated(dtype: torch.dtype) -> None:
    base, guide = policies(dtype)
    base_before, guide_before = base.clone(), guide.clone()
    output = adaptive_parm_logits(base, guide, torch.tensor([0.0, 0.5, 1.0]))
    assert output.dtype == dtype
    assert bool(torch.isfinite(output).all())
    assert torch.equal(base, base_before)
    assert torch.equal(guide, guide_before)


def test_scalar_batch_and_token_weights_broadcast() -> None:
    base, guide = policies()
    scalar = adaptive_parm_logits(base, guide, 0.25)
    batch = adaptive_parm_logits(base, guide, torch.full((3,), 0.25))
    token = adaptive_parm_logits(base, guide, torch.full((3, 4), 0.25))
    torch.testing.assert_close(batch, scalar)
    torch.testing.assert_close(token, scalar)


def test_weight_gradient_is_finite() -> None:
    base, guide = policies()
    weight = torch.tensor(0.37, requires_grad=True)
    loss = -torch.log_softmax(adaptive_parm_logits(base, guide, weight), dim=-1)[..., 0].mean()
    loss.backward()
    assert weight.grad is not None
    assert bool(torch.isfinite(weight.grad))


def test_logit_magnitude_does_not_change_weight_semantics() -> None:
    base, guide = policies()
    weight = 0.3
    fused = adaptive_parm_logits(base, guide, weight)
    for scale in (0.25, 2.0, 16.0):
        scaled = adaptive_parm_logits(scale * base, scale * guide, weight)
        torch.testing.assert_close(scaled, scale * fused, atol=1e-6, rtol=1e-6)
        recovered = (scaled - scale * base) / (scale * (guide - base))
        mask = (guide - base).abs() > 1e-5
        torch.testing.assert_close(
            recovered[mask], torch.full_like(recovered[mask], weight), atol=1e-5, rtol=1e-5
        )


def test_vocabulary_dependent_weight_is_rejected() -> None:
    base, guide = policies()
    with pytest.raises(ValueError, match="vocabulary-invariant"):
        adaptive_parm_logits(base, guide, torch.full_like(base, 0.5))
