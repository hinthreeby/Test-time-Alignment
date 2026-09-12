# Stage 6 Generation Length Diagnostic

## Diagnostic Status

**PASS - all short outputs are legitimate V1 EOS terminations.**

This diagnostic does not itself mark Stage 6 PASS. The Stage 6 resume
command must recompute the final report with the corrected gate.

## Root Cause

The completed evaluation has `100` records
with realized length below the configured 32-token budget. Every one is
a read-only V1 record and ends in GPT-2 EOS token
`50256`. V1 decoding stops on EOS, but its report builder
hard-coded `eos_generated=false`; the selected token IDs are correct.
Router V2 uses `stop_on_eos=false`, so all V2 records realize exactly 32
tokens even when EOS occurs earlier.

V1 terminal-EOS records: `102` (100 short, 2 at length 32).
V1 source EOS metadata mismatches: `102`.
V2 records containing EOS: `1045` (all length 32).

The GPT-2 Large and GPT-2 Medium tokenizer files are byte-identical, both
use vocabulary size 50,257 and EOS ID 50,256. Tokenizer differences are
therefore not the cause.

## Scientifically Valid Gate

`max_new_tokens` is an upper bound, not a required realized length. An
autoregressive decoder that stops on EOS has completed normally. Padding or
continuing a legacy output after EOS would alter its semantics. Stage 6 now
requires all of the following:

1. `max_new_tokens >= 32`;
2. measured length equals `len(selected_token_ids)`;
3. every realized length is positive and no greater than the budget;
4. every output shorter than the budget terminated on EOS.

Corrected length audit: `PASS`.

## Per-Method Statistics

| Method | Family | N | Short | Min | P05 | Median | Mean | Max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| taro | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| taro_same_average | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v1_base | v1_rad_scalar_reward | 600 | 14 | 6 | 32 | 32 | 31.665 | 32 |
| v1_fixed | v1_rad_scalar_reward | 600 | 41 | 2 | 24.95 | 32 | 31.06 | 32 |
| v1_heuristic | v1_rad_scalar_reward | 600 | 23 | 2 | 32 | 32 | 31.201667 | 32 |
| v1_router | v1_rad_scalar_reward | 600 | 22 | 3 | 32 | 32 | 31.438333 | 32 |
| v2_base | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v2_fixed | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v2_heuristic | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v2_history | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v2_same_average | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |
| v2_state | v2_autoregressive_sentiment_guide | 600 | 0 | 32 | 32 | 32 | 32 | 32 |

## Short Counts By Method And Class

| Method | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| taro | 0 | 0 | 0 | 0 |
| taro_same_average | 0 | 0 | 0 | 0 |
| v1_base | 8 | 2 | 4 | 14 |
| v1_fixed | 22 | 4 | 15 | 41 |
| v1_heuristic | 12 | 1 | 10 | 23 |
| v1_router | 9 | 4 | 9 | 22 |
| v2_base | 0 | 0 | 0 | 0 |
| v2_fixed | 0 | 0 | 0 | 0 |
| v2_heuristic | 0 | 0 | 0 | 0 |
| v2_history | 0 | 0 | 0 | 0 |
| v2_same_average | 0 | 0 | 0 | 0 |
| v2_state | 0 | 0 | 0 | 0 |

## Tokenizer Evidence

| File | V1 GPT-2 Large SHA-256 | V2 GPT-2 Medium SHA-256 | Match |
|---|---|---|---|
| merges.txt | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` | `True` |
| tokenizer.json | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` | `True` |
| vocab.json | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` | `True` |

## Exact Short Records

| Prompt ID | Source ID | Class | Method | Length | Terminal token |
|---|---|---|---|---:|---:|
| rad:test:negative:00000086 | `a070d5c595b789e17c612cde0c318543` | negative | v1_base | 29 | 50256 |
| rad:test:negative:00000099 | `6de461f7c3029c7d7c70e31846d27917` | negative | v1_base | 6 | 50256 |
| rad:test:negative:00000110 | `433c784dbff4cfa6a1f1ba61522cd9a9` | negative | v1_base | 12 | 50256 |
| rad:test:negative:00000121 | `ead98b9eed78c884198fb6e1766e5a3b` | negative | v1_base | 6 | 50256 |
| rad:test:negative:00000129 | `9e340a176640a42516a90e026bf38234` | negative | v1_base | 24 | 50256 |
| rad:test:negative:00000170 | `63796259b581c371e43e7964fee1e99a` | negative | v1_base | 8 | 50256 |
| rad:test:negative:00000189 | `8660866b8504d863ae5e3ad728fb306a` | negative | v1_base | 19 | 50256 |
| rad:test:negative:00000205 | `909b5cbe344bcdc5321f3a6400154c73` | negative | v1_base | 17 | 50256 |
| rad:test:neutral:00000127 | `55b42b776899c937c8f9b8f754c87e99` | neutral | v1_base | 26 | 50256 |
| rad:test:neutral:00000240 | `df32e69d29907f0806a5d0065d957966` | neutral | v1_base | 26 | 50256 |
| rad:test:positive:00000111 | `4f6fb7b00b7d7019322e8497bc7b811c` | positive | v1_base | 14 | 50256 |
| rad:test:positive:00000136 | `d581f49e14d959c7fec5ba18a94e38d6` | positive | v1_base | 23 | 50256 |
| rad:test:positive:00000148 | `a89ead417ef5d99f1e0441bb118b473a` | positive | v1_base | 8 | 50256 |
| rad:test:positive:00000164 | `50c0c560a87f73305bf9f308c1fad2ed` | positive | v1_base | 29 | 50256 |
| rad:test:negative:00000070 | `6aebe1fdede3918f7bad44cb2ceadf06` | negative | v1_fixed | 25 | 50256 |
| rad:test:negative:00000086 | `a070d5c595b789e17c612cde0c318543` | negative | v1_fixed | 28 | 50256 |
| rad:test:negative:00000099 | `6de461f7c3029c7d7c70e31846d27917` | negative | v1_fixed | 6 | 50256 |
| rad:test:negative:00000106 | `4fc37969635cd417af605021ff29a816` | negative | v1_fixed | 31 | 50256 |
| rad:test:negative:00000109 | `f265b0d7fbb2f790cd67503b4681fea3` | negative | v1_fixed | 14 | 50256 |
| rad:test:negative:00000110 | `433c784dbff4cfa6a1f1ba61522cd9a9` | negative | v1_fixed | 2 | 50256 |
| rad:test:negative:00000121 | `ead98b9eed78c884198fb6e1766e5a3b` | negative | v1_fixed | 6 | 50256 |
| rad:test:negative:00000123 | `e2f86d50669a32c6a6ab351baa55acab` | negative | v1_fixed | 24 | 50256 |
| rad:test:negative:00000125 | `d78e070c9374db901da521803882550d` | negative | v1_fixed | 28 | 50256 |
| rad:test:negative:00000138 | `fb46c1afb99925be5f5f1bbc1a11c0d0` | negative | v1_fixed | 20 | 50256 |
| rad:test:negative:00000168 | `9c9c35faf93fa7c6d61be760ac39b3e0` | negative | v1_fixed | 22 | 50256 |
| rad:test:negative:00000170 | `63796259b581c371e43e7964fee1e99a` | negative | v1_fixed | 14 | 50256 |
| rad:test:negative:00000178 | `1c50d0319ec1752c075ddd03d3d4fb6e` | negative | v1_fixed | 31 | 50256 |
| rad:test:negative:00000188 | `4b50f92f8e7f1a22d2a0b193896635cc` | negative | v1_fixed | 4 | 50256 |
| rad:test:negative:00000189 | `8660866b8504d863ae5e3ad728fb306a` | negative | v1_fixed | 19 | 50256 |
| rad:test:negative:00000196 | `a931f679dba3e32a111ac011fd49f549` | negative | v1_fixed | 23 | 50256 |
| rad:test:negative:00000202 | `32a1500a7fa81506ec663dea37d37182` | negative | v1_fixed | 9 | 50256 |
| rad:test:negative:00000205 | `909b5cbe344bcdc5321f3a6400154c73` | negative | v1_fixed | 14 | 50256 |
| rad:test:negative:00000213 | `d5438f16850dedd615e72f5554a265ba` | negative | v1_fixed | 21 | 50256 |
| rad:test:negative:00000228 | `af404f7f248ecc82ab8b50c303fa19f9` | negative | v1_fixed | 10 | 50256 |
| rad:test:negative:00000235 | `c435aee2bddd73f70c599e65598c21d2` | negative | v1_fixed | 21 | 50256 |
| rad:test:negative:00000244 | `0245583dd48096f2ed1eba8597449488` | negative | v1_fixed | 16 | 50256 |
| rad:test:neutral:00000082 | `6bf5484f595febae0add0f914d10a374` | neutral | v1_fixed | 24 | 50256 |
| rad:test:neutral:00000105 | `462545dfc726a1739cd27246aff76d54` | neutral | v1_fixed | 17 | 50256 |
| rad:test:neutral:00000138 | `d1f8978f329b8d039950dcca48fe6bef` | neutral | v1_fixed | 24 | 50256 |
| rad:test:neutral:00000221 | `e2c7d5d4dc2759f8d90444f4efcf47f8` | neutral | v1_fixed | 18 | 50256 |
| rad:test:positive:00000075 | `696df42e5b5288002422ab6b4872b6c6` | positive | v1_fixed | 7 | 50256 |
| rad:test:positive:00000094 | `238adc64b0216fb4d349bf5ff924019f` | positive | v1_fixed | 28 | 50256 |
| rad:test:positive:00000104 | `3e41abcf5efc3ceeb585e0c4f9c54c76` | positive | v1_fixed | 29 | 50256 |
| rad:test:positive:00000111 | `4f6fb7b00b7d7019322e8497bc7b811c` | positive | v1_fixed | 2 | 50256 |
| rad:test:positive:00000136 | `d581f49e14d959c7fec5ba18a94e38d6` | positive | v1_fixed | 23 | 50256 |
| rad:test:positive:00000154 | `f10656b461856162937af959f1e6c3df` | positive | v1_fixed | 10 | 50256 |
| rad:test:positive:00000161 | `9f2ec7f8fd8cceb9643f6d1731d48fc8` | positive | v1_fixed | 6 | 50256 |
| rad:test:positive:00000164 | `50c0c560a87f73305bf9f308c1fad2ed` | positive | v1_fixed | 11 | 50256 |
| rad:test:positive:00000167 | `07934f59c04816c9cb32fba35e54daf4` | positive | v1_fixed | 24 | 50256 |
| rad:test:positive:00000214 | `eebb1e53edf7a164e9afa8f3eecdfefe` | positive | v1_fixed | 6 | 50256 |
| rad:test:positive:00000218 | `5e08c2fbc8841a31fb885689066db076` | positive | v1_fixed | 30 | 50256 |
| rad:test:positive:00000225 | `aef0a56ef5f17f6637466d1747b0f566` | positive | v1_fixed | 27 | 50256 |
| rad:test:positive:00000226 | `9f48e22b77b048d0eff1f747303b1bcc` | positive | v1_fixed | 19 | 50256 |
| rad:test:positive:00000228 | `ffc6c79f16f44145eb1ac4c03609a083` | positive | v1_fixed | 30 | 50256 |
| rad:test:positive:00000246 | `9c8a4a07d536f478a8f909c1baa8a676` | positive | v1_fixed | 25 | 50256 |
| rad:test:negative:00000099 | `6de461f7c3029c7d7c70e31846d27917` | negative | v1_heuristic | 6 | 50256 |
| rad:test:negative:00000106 | `4fc37969635cd417af605021ff29a816` | negative | v1_heuristic | 15 | 50256 |
| rad:test:negative:00000110 | `433c784dbff4cfa6a1f1ba61522cd9a9` | negative | v1_heuristic | 2 | 50256 |
| rad:test:negative:00000121 | `ead98b9eed78c884198fb6e1766e5a3b` | negative | v1_heuristic | 6 | 50256 |
| rad:test:negative:00000129 | `9e340a176640a42516a90e026bf38234` | negative | v1_heuristic | 24 | 50256 |
| rad:test:negative:00000170 | `63796259b581c371e43e7964fee1e99a` | negative | v1_heuristic | 8 | 50256 |
| rad:test:negative:00000188 | `4b50f92f8e7f1a22d2a0b193896635cc` | negative | v1_heuristic | 4 | 50256 |
| rad:test:negative:00000189 | `8660866b8504d863ae5e3ad728fb306a` | negative | v1_heuristic | 19 | 50256 |
| rad:test:negative:00000202 | `32a1500a7fa81506ec663dea37d37182` | negative | v1_heuristic | 13 | 50256 |
| rad:test:negative:00000205 | `909b5cbe344bcdc5321f3a6400154c73` | negative | v1_heuristic | 14 | 50256 |
| rad:test:negative:00000228 | `af404f7f248ecc82ab8b50c303fa19f9` | negative | v1_heuristic | 10 | 50256 |
| rad:test:negative:00000235 | `c435aee2bddd73f70c599e65598c21d2` | negative | v1_heuristic | 8 | 50256 |
| rad:test:neutral:00000221 | `e2c7d5d4dc2759f8d90444f4efcf47f8` | neutral | v1_heuristic | 18 | 50256 |
| rad:test:positive:00000075 | `696df42e5b5288002422ab6b4872b6c6` | positive | v1_heuristic | 7 | 50256 |
| rad:test:positive:00000104 | `3e41abcf5efc3ceeb585e0c4f9c54c76` | positive | v1_heuristic | 14 | 50256 |
| rad:test:positive:00000111 | `4f6fb7b00b7d7019322e8497bc7b811c` | positive | v1_heuristic | 2 | 50256 |
| rad:test:positive:00000136 | `d581f49e14d959c7fec5ba18a94e38d6` | positive | v1_heuristic | 23 | 50256 |
| rad:test:positive:00000148 | `a89ead417ef5d99f1e0441bb118b473a` | positive | v1_heuristic | 4 | 50256 |
| rad:test:positive:00000154 | `f10656b461856162937af959f1e6c3df` | positive | v1_heuristic | 10 | 50256 |
| rad:test:positive:00000161 | `9f2ec7f8fd8cceb9643f6d1731d48fc8` | positive | v1_heuristic | 6 | 50256 |
| rad:test:positive:00000164 | `50c0c560a87f73305bf9f308c1fad2ed` | positive | v1_heuristic | 11 | 50256 |
| rad:test:positive:00000214 | `eebb1e53edf7a164e9afa8f3eecdfefe` | positive | v1_heuristic | 6 | 50256 |
| rad:test:positive:00000228 | `ffc6c79f16f44145eb1ac4c03609a083` | positive | v1_heuristic | 27 | 50256 |
| rad:test:negative:00000086 | `a070d5c595b789e17c612cde0c318543` | negative | v1_router | 29 | 50256 |
| rad:test:negative:00000099 | `6de461f7c3029c7d7c70e31846d27917` | negative | v1_router | 6 | 50256 |
| rad:test:negative:00000121 | `ead98b9eed78c884198fb6e1766e5a3b` | negative | v1_router | 6 | 50256 |
| rad:test:negative:00000129 | `9e340a176640a42516a90e026bf38234` | negative | v1_router | 24 | 50256 |
| rad:test:negative:00000170 | `63796259b581c371e43e7964fee1e99a` | negative | v1_router | 8 | 50256 |
| rad:test:negative:00000189 | `8660866b8504d863ae5e3ad728fb306a` | negative | v1_router | 19 | 50256 |
| rad:test:negative:00000196 | `a931f679dba3e32a111ac011fd49f549` | negative | v1_router | 28 | 50256 |
| rad:test:negative:00000205 | `909b5cbe344bcdc5321f3a6400154c73` | negative | v1_router | 13 | 50256 |
| rad:test:negative:00000228 | `af404f7f248ecc82ab8b50c303fa19f9` | negative | v1_router | 9 | 50256 |
| rad:test:neutral:00000127 | `55b42b776899c937c8f9b8f754c87e99` | neutral | v1_router | 13 | 50256 |
| rad:test:neutral:00000130 | `45631352ad4506dffc59cf15390d13da` | neutral | v1_router | 30 | 50256 |
| rad:test:neutral:00000228 | `64fa9730bc68974d6b492b1f1750de42` | neutral | v1_router | 18 | 50256 |
| rad:test:neutral:00000240 | `df32e69d29907f0806a5d0065d957966` | neutral | v1_router | 10 | 50256 |
| rad:test:positive:00000077 | `3305c3be8d4a4957b2fdf5b9143065c1` | positive | v1_router | 20 | 50256 |
| rad:test:positive:00000104 | `3e41abcf5efc3ceeb585e0c4f9c54c76` | positive | v1_router | 14 | 50256 |
| rad:test:positive:00000111 | `4f6fb7b00b7d7019322e8497bc7b811c` | positive | v1_router | 3 | 50256 |
| rad:test:positive:00000136 | `d581f49e14d959c7fec5ba18a94e38d6` | positive | v1_router | 23 | 50256 |
| rad:test:positive:00000148 | `a89ead417ef5d99f1e0441bb118b473a` | positive | v1_router | 8 | 50256 |
| rad:test:positive:00000154 | `f10656b461856162937af959f1e6c3df` | positive | v1_router | 16 | 50256 |
| rad:test:positive:00000213 | `43b3b85dc02b9ace4f24746434d00d62` | positive | v1_router | 31 | 50256 |
| rad:test:positive:00000225 | `aef0a56ef5f17f6637466d1747b0f566` | positive | v1_router | 12 | 50256 |
| rad:test:positive:00000228 | `ffc6c79f16f44145eb1ac4c03609a083` | positive | v1_router | 27 | 50256 |

## Reuse Decision

All 7,200 completed test records are reusable. No generation output is
changed, padded, dropped, or regenerated. Resume performs zero generation
jobs and only recomputes scoring/report artifacts under the corrected gate.
