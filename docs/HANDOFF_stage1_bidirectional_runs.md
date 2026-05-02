# Handoff: Stage 1 bidirectional B-only verification runs (6 IID Two-Phase experiments)

**Audience:** Claude / reviewer follow-up / paper pipeline  
**Repo:** `federated-lora-experiments`  
**Branch used on disk:** `impl/bidirectional-b-only-protocol`  
**Git commit recorded in `run_meta.json`:** `c1ed2a156de222bc0c6e65af5e98c67dcb5fc44a` (runs tagged `stage1_bidirectional`)  
**Note:** `run_meta.json` reports `git_dirty: true` — if strict reproducibility matters, re-record after a clean checkout.

---

## 1. What was run

Six federated training jobs per reviewer Stage 1 plan:

| Config | Seeds | Tag |
|--------|-------|-----|
| `config/exp_two_phase_k8.yaml` (phase boundary **K=8**, IID) | 42, 123, 456 | `stage1_bidirectional` |
| `config/exp_two_phase_k10.yaml` (phase boundary **K=10**, IID) | 42, 123, 456 | `stage1_bidirectional` |

**Command pattern (for reference):**

```bash
for K in 8 10; do
  for seed in 42 123 456; do
    python3 scripts/run_experiment.py \
      --config "config/exp_two_phase_k${K}.yaml" \
      --seed "${seed}" \
      --tag stage1_bidirectional
  done
done
```

**Hardware / stack (from one representative `run_meta.json`):** Python 3.10.18, macOS, device **`mps`**.

---

## 2. Result artifact paths (local workspace)

| Experiment | Seed | `results.json` path |
|------------|------|---------------------|
| k8 | 42 | `results/raw/exp_two_phase_k8/two_phase/seed_42/stage1_bidirectional/20260501_150340/results.json` |
| k8 | 123 | `results/raw/exp_two_phase_k8/two_phase/seed_123/stage1_bidirectional/20260501_180732/results.json` |
| k8 | 456 | `results/raw/exp_two_phase_k8/two_phase/seed_456/stage1_bidirectional/20260501_211149/results.json` |
| k10 | 42 | `results/raw/exp_two_phase_k10/two_phase/seed_42/stage1_bidirectional/20260502_001645/results.json` |
| k10 | 123 | `results/raw/exp_two_phase_k10/two_phase/seed_123/stage1_bidirectional/20260502_032419/results.json` |
| k10 | 456 | `results/raw/exp_two_phase_k10/two_phase/seed_456/stage1_bidirectional/20260502_063204/results.json` |

Each directory also contains `run_meta.json` and `config_merged.yaml` (if your pipeline uses them).

---

## 3. Per-round metrics (how to read the JSON)

Each list entry in `results.json` is **cumulative** for the run so far:

- `upload_mb`, `download_mb`: cumulative totals since round 1.
- Per-round **deltas** (what actually moved that round):  
  \(\Delta u_r = u_r - u_{r-1}\), \(\Delta d_r = d_r - d_{r-1}\) (with \(u_0=d_0=0\)).

New Stage 1 fields:

- `broadcast_b_only`: whether the **metric row** corresponds to a round where the **next** client broadcast is B-only (look-ahead accounting).
- `broadcast_bytes_per_client`: payload size used for download accounting for that round’s broadcast staging.

---

## 4. Detailed analysis — Two-Phase **K=8** (all three seeds)

All three seeds produced **identical communication numbers** (deterministic comm accounting given fixed architecture / rank).

### 4.1 Key per-round deltas (same for seeds 42, 123, 456)

| Round | \(\Delta\) upload (MB) | \(\Delta\) download (MB) | `broadcast_b_only` |
|------:|------------------------|---------------------------|--------------------|
| 8 | 85.9375 | 85.9375 | false |
| 9 | 85.9375 | **30.9375** | **true** |
| 10 | 30.9375 | 30.9375 | true |

**Interpretation:** Matches the reviewer’s “one-round offset” story: on round 9 the client still **uploads full** LoRA (first FFA round), but the **download delta** is already **B-only** because download accounting at end of round 9 stages the broadcast used at the **start** of round 10.

### 4.2 End of round 15 (final row in `results.json`)

| Metric | Value (all k8 seeds) |
|--------|-------------------------|
| Cumulative `upload_mb` | **959.0625** |
| Cumulative `download_mb` | **904.0625** |
| Cumulative `communication_mb` (= upload + download) | **1863.1250** |

**Vs symmetric FLoRA-style baseline** for 15 rounds (full bidirectional each round):  
\(15 \times 85.9375 \times 2 = 2578.125\) MB total movement.

**Reported round-trip savings** vs that baseline:

\[
1 - \frac{1863.125}{2578.125} \approx 27.74\%
\]

This aligns with **Path A** accounting in `docs/stage1_pathB.md` (slightly higher headline savings than strict “no phantom final broadcast” accounting).

### 4.3 Final loss (seed variation)

| Seed | Final `avg_loss` (round 15) |
|------|-----------------------------|
| 42 | 1.269228 |
| 123 | 1.270852 |
| 456 | 1.270770 |

**Mean:** ~**1.2703** — consistent with reviewer expectation **~1.270 ± 0.001** for K=8 IID.

---

## 5. Detailed analysis — Two-Phase **K=10** (all three seeds)

Again, **identical communication** across seeds.

### 5.1 Transition rounds

First round where `broadcast_b_only` is **true** in metrics: **round 11** (expected: first FFA round after 10 FLoRA rounds).

Per-round deltas (representative: seed 42; identical for 123, 456):

| Round | \(\Delta\) upload (MB) | \(\Delta\) download (MB) | Notes |
|------:|------------------------|---------------------------|-------|
| 8–10 | 85.9375 each | 85.9375 each | Still full broadcast |
| **11** | **85.9375** | **30.9375** | First asymmetric row (`broadcast_b_only` true) |
| 12–15 | 30.9375 each | 30.9375 each | B-only both directions |

### 5.2 End of round 15

| Metric | Value (all k10 seeds) |
|--------|-------------------------|
| Cumulative `upload_mb` | **1069.0625** |
| Cumulative `download_mb` | **1014.0625** |
| Cumulative `communication_mb` | **2083.1250** |

**Vs symmetric baseline** \(2578.125\) MB:

\[
1 - \frac{2083.125}{2578.125} \approx 19.20\%
\]

### 5.3 Final loss

| Seed | Final `avg_loss` |
|------|------------------|
| 42 | 1.265454 |
| 123 | 1.267048 |
| 456 | 1.266976 |

**Mean:** ~**1.2665** — matches reviewer ballpark **~1.266 ± 0.001** for K=10 IID.

---

## 6. Acceptance checks vs reviewer Stage 1 table

Reviewer tolerances: upload/download deltas **exact** where stated; cumulative download ±1 MB; loss within ±0.002 of prior post-upload-fix runs.

### K=8 (all seeds — PASS)

| Check | Expected | Observed |
|-------|----------|----------|
| Round 9 upload \(\Delta\) | 85.94 MB | 85.9375 |
| Round 10 upload \(\Delta\) | 30.94 MB | 30.9375 |
| Round 9 download \(\Delta\) | 30.94 MB (asymmetric) | 30.9375 |
| Round 8 download \(\Delta\) | 85.94 MB | 85.9375 |
| Cumulative upload @15 | 959.06 MB | 959.0625 |
| Cumulative download @15 | 904.06 MB | 904.0625 |
| Final loss | ~1.270 ± 0.001 | 1.269–1.271 |

### K=10 (all seeds — PASS)

| Check | Expected | Observed |
|-------|----------|----------|
| Cumulative upload @15 | 1069.06 MB | 1069.0625 |
| Cumulative download @15 | 1014.06 MB | 1014.0625 |
| Final loss | ~1.266 ± 0.001 | 1.265–1.267 |
| B-only broadcast activates | After phase boundary (first metrics row with flag at round **11**) | Round **11** |

**Sanity log:** Confirm in stdout/logs that `[B-only broadcast active starting next round]` appeared once per run at the expected boundary (end of round **9** for k8, end of round **11** for k10). This line is not stored in `results.json`; grep training logs if needed.

---

## 7. Verdict

- **Implementation behaves as simulated:** communication trajectories match the pre-run predictions; upload/download asymmetry appears exactly on the predicted rounds.
- **No seed-dependent drift in comm metrics** (loss varies slightly by seed as expected).
- **Loss magnitudes** sit on the expected IID baselines for both K settings.
- **Stage 1 IID verification goal:** treat as **complete** for proceeding to **Stage 2** work per the TMLR Path B roadmap (subject to maintainer sign-off).

---

## 8. Optional follow-up (Path B accounting)

If you later want “bytes-on-wire strict” savings instead of Path A headline numbers, see `docs/stage1_pathB.md` — deferred accounting tweak, not required for cross-method **fair** comparison (all methods share the same tracker).

---

## 9. Suggested next actions for Claude

1. Archive or copy these six `results.json` paths into whatever artifact store / figure pipeline you use; preserve `run_meta.json` for provenance.
2. If writing methods text: cite **Path A** cumulative comm from `communication_mb` / upload+download; optionally footnote Path B if tightening claims.
3. Begin Stage 2 scope from the implementation guide (non-IID / next milestones as specified there).

---

*Generated from workspace analysis of the six `results.json` files listed in §2.*
