# Neurocomputing Phase 1 — official clearance

**Status: CLEARED** (maintainer decision 2026-07-16), with one carve-out.

## Clearance summary

33/34 runs are clean or explained-and-acceptable. No systematic CUDA protocol failure.
LLaMA-3.2-3B tier (all 9) is spotless. Phase 2 (8B) prep may start in parallel with the
`#17` diagnostic; the diagnostic gates the **non-IID robustness paper section**, not 8B setup.

### Failure disposition (maintainer)

| IDs | Disposition |
|-----|-------------|
| `#23–25` Two-Phase Non-IID α=0.5 | Invalid pre-B-only MPS refs → **CUDA = source of truth**. No valid MPS baseline for Non-IID Two-Phase; do **not** claim cross-backend robustness for that cell (IID Two-Phase only). |
| `#11–12` τ=0.001 / 0.002 | Accept. Put in **ablation section** as characterized limit: switch timing stable for **τ ≥ 0.005**; below that, decision boundary approaches backend noise (±1 round). Not a buried footnote. |
| `#16` RA α=0.1 seed 123 | Accept (switch/comm exact). Note α=0.1 is noisier than α=0.5 (2nd boundary-touch after Gate 1). |
| `#17` RA α=0.1 seed 456 | **Carve-out.** Do not accept-and-disclose from the single run. Pattern is **2/3 seeds at α=0.1** backend-sensitive (#16 streak, #17 switch). Run **Part 0 diagnostic** (Phase 2 guide) before writing non-IID robustness text. |

### Paper actions (approved)

1. Reproducibility: state tolerances explicitly — IID `atol=0.01`; Non-IID `atol=0.025` + exact switch + ≤2 consecutive `|Δ|>0.01`.
2. Non-IID Two-Phase numbers from CUDA; no MPS↔CUDA agreement claim for that cell.
3. Frame τ and α=0.1 as **characterized operating limits** of the method (not errata). `#17` diagnostic chooses between “differs by 1 round across backends” vs “switch timing not fully reproducible at α=0.1 even within a backend.”

### Carve-out — `#17` diagnostic (required before non-IID section)

See `docs/PHASE_2_IMPLEMENTATION_GUIDE.md` Part 0 and `docs/PHASE_1_17_DIAGNOSTIC.md`.
Hold data partition seed **456**; vary run-level RNG (e.g. **999**). Interpret:

- 2nd CUDA also switches at **7** → Explanation A (stable backend difference)
- 2nd CUDA switches at **6** (or other) → Explanation B (stochastic near boundary)

## Early gates + batch verify

See sections below / `docs/PHASE_1_BATCH_VERIFY.md`. Batch: **27 PASS / 7 FAIL / 0 missing** before triage; after triage, only `#17` remains open for paper wording.
