# Rich-state advantage probe

Verdict: **ADAPTIVE_ROUTING_NO_GO**

All results use leave-one-unique-prompt-out evaluation. PCA, Ridge alpha, and intervention delta are fitted/selected only from each outer fold's training prompts.

| Family | Pairwise | Spearman | Policy MIP | Gain | Gap capture | Harmful | Intervention |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0.4291 | -0.0939 | 0.607255 | 0.000371 | 2.58% | 59.09% | 8.80% |
| logit_k16 | 0.4493 | -0.0741 | 0.606589 | -0.000295 | -2.05% | 27.27% | 8.80% |
| logit_k32 | 0.4358 | -0.0835 | 0.606549 | -0.000334 | -2.33% | 50.00% | 8.00% |
| logit_k64 | 0.4527 | -0.0660 | 0.611256 | 0.004373 | 30.44% | 34.29% | 14.00% |
| hidden_pca8 | 0.4527 | -0.0654 | 0.607707 | 0.000823 | 5.73% | 48.15% | 10.80% |
| hidden_pca16 | 0.4358 | -0.0930 | 0.608412 | 0.001528 | 10.64% | 51.43% | 14.00% |
| hidden_pca32 | 0.4088 | -0.1209 | 0.607823 | 0.000939 | 6.54% | 48.39% | 12.40% |
| action_conditioned | 0.4527 | -0.0593 | 0.603983 | -0.002901 | -20.19% | 42.86% | 11.20% |
| combined | 0.4493 | -0.0657 | 0.605818 | -0.001066 | -7.42% | 47.62% | 8.40% |

Best held-out policy: **logit_k64**, MIP `0.611256488`.

Scorer values are labels only and are absent from every inference feature. No backbone or PBLoRA parameter is trained.
