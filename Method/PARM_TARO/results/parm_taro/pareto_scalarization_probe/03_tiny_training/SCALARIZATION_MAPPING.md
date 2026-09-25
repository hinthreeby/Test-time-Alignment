# Alpha-to-Tchebycheff mapping

For objective-loss vector `L=(L_help,L_safe)` and the user's preference
`alpha=(alpha_help,alpha_safe)`, the probe uses the identity mapping `w=alpha`:

`smoothmax_i [w_i (L_i-z_i)/s_i] + rho sum_i w_i (L_i-z_i)/s_i`.

The reference `z` and positive scales `s` must be frozen from training-only
pilot batches before either arm starts. Identity is chosen because it preserves
the semantics and endpoints of the original PARM preference, maps 0.5 to equal
importance, is permutation-equivariant, and adds no tuned preference warp.
The smooth maximum uses log-sum-exp temperature 20 and augmentation rho=0.05.
These are probe constants shared across all steps, not validation-tuned values.
