# Phase 1 #17 diagnostic — RA Non-IID α=0.1 data_seed=456

**Purpose:** Distinguish Explanation A (stable CUDA switch at 7 vs MPS at 6) from
Explanation B (switch round stochastic under run-level RNG with fixed data split).

**Status:** complete — **Explanation A**

## Reference (original Phase 1 CUDA)

| Field | Value |
|-------|-------|
| Path | `results/raw/exp_reverse_adaptive_noniid_alpha01/reverse_adaptive/seed_456/phase1_cuda_rerun/20260715_191752/` |
| data_seed / run_seed | 456 / 456 (legacy single `--seed`) |
| switch_round | **7** |
| final_comm_mb | 1643.125 |
| MPS switch_round | **6** (comm 1533.125) |

## Diagnostic run

```bash
python3 scripts/run_experiment.py \
  --config config/exp_reverse_adaptive_noniid_alpha01.yaml \
  --data-seed 456 \
  --run-seed 999 \
  --device cuda \
  --tag phase1_17_diagnostic \
  --save-every 10
```

| Field | Value |
|-------|-------|
| Path | `results/raw/exp_reverse_adaptive_noniid_alpha01/reverse_adaptive/seed_456_run999/phase1_17_diagnostic/20260716_192659/` |
| data_seed | 456 |
| run_seed | 999 |
| switch_round | **7** |
| final_comm_mb | 1643.125 |
| final_avg_loss | 1.314532 |
| git_commit | `6316fb1` (`neurocomputing/phase-1-cuda-reruns`) |

## Interpretation

- [x] **Explanation A** — diagnostic also switches at 7 → stable within-CUDA; report backend-sensitive switch at α=0.1 (1 of 3 seeds crossed the boundary; 2/3 show backend-dependent behavior counting #16).
- [ ] **Explanation B** — diagnostic switches at 6 (or ≠7) → switch timing not fully reproducible even within CUDA at α=0.1; report as distribution / ±1 round.

**Verdict:** With the same Dirichlet partition (`data_seed=456`) and a different training RNG (`run_seed=999`), CUDA still switches at **7** and ends at **1643.125 MB**, matching the original CUDA #17 run. Loss trajectories diverge as expected under a new run seed (max \|Δ\|≈0.052 vs original CUDA), but the adaptive threshold decision does not flip. The MPS switch at **6** is therefore a **backend-sensitive** effect on this seed, not within-CUDA stochasticity of the switch.

**Paper framing:** At α=0.1, seed 456 is the carve-out where CUDA/MPS disagree by one switch round. Prefer reporting CUDA as the primary backend for this cell; note ±1-round backend sensitivity at the aggressive Non-IID setting rather than claiming exact cross-backend switch match.

## Notes

| Run | switch | final_comm | final_loss |
|-----|--------|------------|------------|
| MPS seed 456 | 6 | 1533.125 | 1.348083 |
| CUDA seed 456 (run_seed=456) | 7 | 1643.125 | 1.335038 |
| CUDA diagnostic (run_seed=999) | 7 | 1643.125 | 1.314532 |

B-only starts at round 7 on both CUDA runs; MPS starts B-only at round 6.
