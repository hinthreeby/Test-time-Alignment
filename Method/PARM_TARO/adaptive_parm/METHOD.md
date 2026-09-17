# Adaptive PARM Fusion

## Frozen formulation

For vocabulary item `v` at token position `t`, define

```text
a_t(v) = log p_base(v | prefix)
b_t(v, alpha) = log p_PARM(v | prefix, alpha)
z_t(v) = (1 - w_t) a_t(v) + w_t b_t(v, alpha),  0 <= w_t <= 1
p_adaptive(v) = softmax(z_t(v))
```

This is a geometric interpolation of the two policies before normalization:

```text
p_adaptive(v) proportional to
    p_base(v | prefix)^(1-w_t) * p_PARM(v | prefix, alpha)^w_t.
```

The controller predicts **trust in PARM**:

- `w=0`: exact Base distribution.
- `w=0.5`: equal Base/PARM log-policy midpoint, matching the audited
  author-style PARM fusion.
- `w=1`: exact preference-conditioned PARM guide distribution.

## Relation to the previous lambda equation

The former diagnostic path used `a + lambda*b`. Its normalized decomposition is

```text
a + lambda*b = (1 + lambda) * ((1-w)*a + w*b)
w = lambda / (1 + lambda).
```

The leading factor `(1+lambda)` changes effective temperature while `lambda`
also changes relative policy weighting. Adaptive PARM removes that factor and
therefore varies trust without an implicit temperature rescaling. The helper
`lambda_to_weight` exists only for parity audits; new controllers predict `w`
directly.

## Relation to PARM

PARM remains the preference-conditioned guide policy. Adaptive PARM does not
alter PARM parameters or redefine its preference vector `alpha`; it decides how
much to trust the Base versus PARM distribution at each prefix.

## Why this is not TARO

TARO and router V2 were trained around a different lambda-controlled additive
guidance equation and historical objectives. Adaptive PARM is a separate method
with a frozen convex log-policy geometry, a bounded trust variable, independent
code, and independent outputs. Historical TARO checkpoints and results are not
inputs to this formulation.
