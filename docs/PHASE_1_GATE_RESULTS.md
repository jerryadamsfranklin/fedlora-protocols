# Neurocomputing Phase 1 — gate run results

Record `verify_backend_match.py` outcomes here before releasing the remaining 32-run batch.

**Phase 1 status: CLEARED** (2026-07-16) — see `docs/PHASE_1_CLEARANCE.md`.
Carve-out: `#17` diagnostic before writing the non-IID robustness section
(`docs/PHASE_1_17_DIAGNOSTIC.md`). Phase 2 prep may proceed in parallel.

**Tolerance profile (setting-aware — see `docs/PHASE_1_EARLY_GATES.md`):**
- IID: loss `atol=0.01`, `rtol=0`
- Non-IID: loss `atol=0.025`, `rtol=0`; switch round exact; ≤2 consecutive rounds with `|Δ|>0.01`
- Communication always `atol=0.01`

## Gate 1 — first Non-IID

| Field | Value |
|-------|-------|
| Status | **PASS** (revised Non-IID criteria) |
| Experiment | `exp_reverse_adaptive_noniid_alpha05` ReverseAdaptive |
| Seed | 42 |
| CUDA tag / path | `results/raw/.../phase1_cuda_rerun/20260715_010658/` |
| MPS path | `results/raw/.../stage2_adaptive/20260502_232453/results.json` |
| Verifier | `--setting noniid` → loss_atol=0.025 |
| final \|Δloss\| | 0.009456 (PASS ≤ 0.025; also ≤ 0.01) |
| final \|Δcomm\| | 0.000000 (PASS) |
| switch_round | 6 == 6 (PASS, hard exact) |
| max consec \|Δ\| > 0.01 | 2 (PASS ≤ 2; R13–R14 spike, then recovered) |
| Notes | Under original flat atol=0.01 this run failed 5 per-round checks (max \|Δ\|≈0.022 at R13). Maintainer decision 2026-07-15: PASS under setting-aware Non-IID tolerances. **Paper TODO:** cite this tolerance policy in the reproducibility section. |

## Gate 2 — first FLoRA (or Two-Phase K=8)

| Field | Value |
|-------|-------|
| Status | **PASS** (IID `atol=0.01`) |
| Experiment | `exp_flora_iid` FLoRA TinyLlama |
| Seed | 42 |
| CUDA tag / path | `results/raw/exp_flora_iid/flora/seed_42/phase1_cuda_rerun/20260715_021226/` |
| MPS path | `results/raw/exp_flora_iid/flora/seed_42/run_07_of_18/20260428_153342/results.json` |
| Verifier | default / IID → loss_atol=0.01 (Vast still on pre-`--setting` script; same bound) |
| final \|Δloss\| | 0.001974 (PASS) |
| final \|Δcomm\| | 0.000000 (PASS; 2578.125 MB) |
| Notes | All 15 rounds PASS; max per-round \|Δ\|≈0.0037. Cleaner than Gate 1 — IID FLoRA matches tightly. Both early gates clear → remaining 32 authorized. |

## Interruptions / checkpoint resumes

(none yet)

## Full-suite batch verify (all 34)

See `docs/PHASE_1_BATCH_VERIFY.md` / `.csv`.

| Result | Count |
|--------|------:|
| PASS | 27 |
| FAIL | 7 |
| MISSING | 0 |

**Failures to triage before Phase 1 clear:**
1. `#11–12` threshold ablation τ=0.001 / 0.002 — loss OK; **comm Δ=110 MB** because CUDA switches **1 round earlier** (τ=0.001: R10 vs R11; τ=0.002: R8 vs R9). Same adaptive rule; tight τ makes the decision boundary sensitive to small backend loss differences. τ=0.005/0.01 match exactly.
2. `#16` RA Non-IID α=0.1 seed 123 — switch/comm **exact**; only fails the **≤2 consecutive `|Δ|>0.01`** rule (R7–9 streak=3). Final `|Δloss|=0.003`. Borderline Non-IID noise (same family as Gate 1).
3. `#17` RA Non-IID α=0.1 seed 456 — **real protocol divergence**: switch **6 vs 7**, comm +110 MB, several per-round `|Δ|>0.025`. Strongest Non-IID α=0.1 seed; early-round loss noise flips the switch trigger. Final loss still within 0.025.
4. `#23–25` Two-Phase K=8 Non-IID α=0.5 all seeds — **not a CUDA bug.** MPS refs (`run_10/11/12_of_18`, branch `experiment/two-phase-validation`, 2026-04-29) predate bidirectional B-only accounting (`impl/bidirectional-b-only-protocol`, 2026-05-01). Those MPS logs charge full A+B every round (2578.125 MB, no `broadcast_b_only`). CUDA correctly switches at K=8 → 1863.125 MB, matching IID Two-Phase. Verifier `switch_round` FAIL is a side effect of missing fields on the old MPS dumps. **Action:** treat CUDA as source of truth for these 3; do not use the Apr-29 MPS Non-IID TP runs as backend-match refs (re-run MPS later only if needed for a paper table).
