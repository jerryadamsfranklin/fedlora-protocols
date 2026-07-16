# Phase 1 #17 diagnostic — RA Non-IID α=0.1 data_seed=456

**Purpose:** Distinguish Explanation A (stable CUDA switch at 7 vs MPS at 6) from
Explanation B (switch round stochastic under run-level RNG with fixed data split).

**Status:** pending run

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
# after pulling seed-split changes
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
| Path | _(fill after run)_ |
| data_seed | 456 |
| run_seed | 999 |
| switch_round | |
| final_comm_mb | |
| final_avg_loss | |

## Interpretation

- [ ] **Explanation A** — diagnostic also switches at 7 → stable within-CUDA; report backend-sensitive switch at α=0.1 (1 of 3 seeds crossed the boundary; 2/3 show backend-dependent behavior counting #16).
- [ ] **Explanation B** — diagnostic switches at 6 (or ≠7) → switch timing not fully reproducible even within CUDA at α=0.1; report as distribution / ±1 round.

## Notes

_(paste verifier / round table if useful)_
