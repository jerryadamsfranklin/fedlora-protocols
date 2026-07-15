# Neurocomputing Phase 1 — gate run results

Record `verify_backend_match.py` outcomes here before releasing the remaining 32-run batch.

Tolerance: `atol=0.01`, `rtol=0`.

## Gate 1 — first Non-IID

| Field | Value |
|-------|-------|
| Status | **FAIL** (per-round; final loss within atol) |
| Experiment | `exp_reverse_adaptive_noniid_alpha05` ReverseAdaptive |
| Seed | 42 |
| CUDA tag / path | `results/raw/.../phase1_cuda_rerun/20260715_010658/` |
| MPS path | `results/raw/.../stage2_adaptive/20260502_232453/results.json` |
| final \|Δloss\| | 0.009456 (PASS ≤ 0.01) |
| final \|Δcomm\| | 0.000000 (PASS) |
| Notes | 5 per-round FAILs (max \|Δ\|≈0.022 at R13). Communication exact 1533.125. Do **not** start Gate 2 / remaining 32 until maintainer decides whether per-round or final-only is the gate metric for Non-IID. |

## Gate 2 — first FLoRA (or Two-Phase K=8)

| Field | Value |
|-------|-------|
| Status | pending |
| Experiment | |
| Seed | |
| CUDA tag / path | |
| MPS path | |
| final \|Δloss\| | |
| final \|Δcomm\| | |
| Notes | |

## Interruptions / checkpoint resumes

(none yet)
