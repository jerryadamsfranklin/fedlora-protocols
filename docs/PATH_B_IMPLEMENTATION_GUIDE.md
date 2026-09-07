# Path B Implementation Brief for Cursor AI

**Audience:** Cursor AI
**Repository:** `fedlora-protocols` (currently on branch `chore/public-release-prep` or main)
**Branch suggestion:** `experiments/path-b-revision`
**Goal:** Run the highest-leverage pre-submission experiments to strengthen the manuscript. Specifically: add two more seeds (123 and 456) at LLaMA-3.2-3B for all three methods (FLoRA, Two-Phase K=8, ReverseAdaptive), produce downstream evaluations on the six new checkpoints, and regenerate analysis artifacts to reflect the expanded data.
**Time budget:** ~7-8 calendar days, mostly overnight compute. Total compute estimate: ~52 hours of training + ~6 hours of evaluation = ~58 hours.

---

## 0. CONTEXT YOU NEED BEFORE STARTING

The maintainer (Jerry) is preparing to submit the manuscript. An external review of the v5 paper draft flagged that single-seed LLaMA-3.2-3B results are the most likely reviewer complaint. This brief addresses that gap directly by adding seeds 123 and 456 at LLaMA-3.2-3B for all three methods evaluated in the paper.

What already exists on disk (do NOT re-run):

- `results/raw/exp_llama3_flora_iid/flora/seed_42/stage5_llama3_full/20260506_172045/results.json` (FLoRA seed 42, training complete)
- `results/raw/exp_llama3_two_phase_k8_iid/two_phase/seed_42/stage5_llama3_full/20260507_005718/results.json` (Two-Phase K=8 seed 42, training complete)
- `results/raw/exp_llama3_reverse_adaptive_iid/reverse_adaptive/seed_42/stage5_llama3_full/20260507_082356/results.json` (ReverseAdaptive seed 42, training complete)
- Downstream evaluations for all three seed-42 LLaMA checkpoints on MMLU, ARC-Easy, BoolQ, HellaSwag

What this brief produces:

1. 6 new LLaMA-3.2-3B training runs (3 methods × 2 seeds)
2. 6 new downstream evaluations on the new checkpoints (4 benchmarks each)
3. Updated `analysis/final_results_table.csv`
4. Updated `analysis/statistical_tests.csv` with new LLaMA-3.2-3B 3-seed paired tests
5. Regenerated `figures/fig6_scale_validation.pdf` with error bars on the LLaMA panel

---

## 1. RULES FOR THIS WORK

1. **No new code unless this brief explicitly specifies it.** Configurations and runners already exist.
2. **Same experimental setup as seed 42 runs.** Use the same configs, same number of rounds (15), same Python environment. Do not "improve" anything between runs.
3. **One commit per numbered task.** Use commit message format: `experiments: <description>`.
4. **Test after each task.** `python3 -m pytest tests/` must pass at the end of every task.
5. **No em-dashes.** Code, comments, configs, commit messages.
6. **Run the test suite at the start.** Confirm 20 tests pass before any work.
7. **Stop and ask if anything seems ambiguous.** Do not improvise.
8. **Save `final_adapter_state.pt` for every run.** Required for downstream evaluation. This is already handled by the existing `_save_results()` in `server.py` so no code change is needed; just verify checkpoints appear after each run.

---

## 2. TASK 1: Pre-flight verification (~10 minutes)

Before launching any runs, verify the environment is ready.

### 2.1 Run the test suite

```bash
python3 -m pytest tests/ -v
```

Expected: 20 passed, 0 failed. If any tests fail, stop and report. Do not proceed.

### 2.2 Verify HuggingFace access for LLaMA-3.2-3B

```bash
huggingface-cli whoami
```

Expected: returns a username (the maintainer's account). If it returns an error, run `huggingface-cli login` and use the token the maintainer provided previously.

### 2.3 Verify the three LLaMA configs exist and parse correctly

```bash
ls config/exp_llama3_*.yaml
```

Expected: three files (`exp_llama3_flora_iid.yaml`, `exp_llama3_two_phase_k8_iid.yaml`, `exp_llama3_reverse_adaptive_iid.yaml`).

### 2.4 Verify existing seed-42 LLaMA results are intact

```bash
find results/raw/exp_llama3_*/*/seed_42/stage5_llama3_full -name "results.json"
```

Expected: three paths, one per method.

### 2.5 Verify disk space for new runs

Each LLaMA-3.2-3B run produces approximately 500 MB of artifacts (results.json + final_adapter_state.pt + run_meta.json). Six new runs need approximately 3 GB free.

```bash
df -h .
```

Confirm at least 10 GB available. If less, the maintainer needs to free space before proceeding.

**Do not commit anything for Task 1.** This is verification only.

---

## 3. TASK 2: LLaMA-3.2-3B seed 123 training runs (~23 hours wall clock)

Three runs, one per aggregation method, all with seed 123. Tag all three as `stage5_llama3_extra_seeds`.

Schedule as one chained overnight command using `caffeinate` to prevent sleep:

```bash
caffeinate -dimsu bash -c '
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_flora_iid.yaml \
      --seed 123 \
      --tag stage5_llama3_extra_seeds && \
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_two_phase_k8_iid.yaml \
      --seed 123 \
      --tag stage5_llama3_extra_seeds && \
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_reverse_adaptive_iid.yaml \
      --seed 123 \
      --tag stage5_llama3_extra_seeds
'
```

Expected wall clock: approximately 7.6 hours per run × 3 = 22.8 hours total. Schedule to start late evening.

### 3.1 Acceptance criteria

After completion, verify each run:

```bash
find results/raw/exp_llama3_*/*/seed_123/stage5_llama3_extra_seeds -name "results.json"
```

Expected: 3 paths.

For each `results.json`, verify the final round (round 15):

- `avg_loss` is finite (not NaN or Inf)
- `avg_loss` is in the range [1.3, 1.5] (consistent with seed-42 LLaMA results)
- `communication_mb` matches the seed-42 result for that method exactly:
  - FLoRA: 2625.0 MB
  - Two-Phase K=8: 1942.5 MB
  - ReverseAdaptive: 1837.5 MB (if switch round is 8; may differ if plateau detection fires differently on seed 123)
- For ReverseAdaptive, log the switch round from `aggregator_stats.switch_round`. If it differs from round 8 (seed 42's switch round), this is a finding to report, not a bug.
- `final_adapter_state.pt` exists in the same directory as `results.json`

If any run produces NaN losses, OOMs, or fails partway through, stop and report. Do not proceed to seed 456.

### 3.2 Commit

```
git add results/raw/exp_llama3_*/*/seed_123/stage5_llama3_extra_seeds/
git commit -m "experiments: add LLaMA-3.2-3B IID seed 123 for all three methods"
```

---

## 4. TASK 3: LLaMA-3.2-3B seed 456 training runs (~23 hours wall clock)

Identical to Task 2 with seed 456 instead of 123. Same chained command pattern.

```bash
caffeinate -dimsu bash -c '
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_flora_iid.yaml \
      --seed 456 \
      --tag stage5_llama3_extra_seeds && \
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_two_phase_k8_iid.yaml \
      --seed 456 \
      --tag stage5_llama3_extra_seeds && \
  python3 scripts/run_experiment.py \
      --config config/exp_llama3_reverse_adaptive_iid.yaml \
      --seed 456 \
      --tag stage5_llama3_extra_seeds
'
```

### 4.1 Acceptance criteria

Same as Task 2 but for seed 456 paths.

### 4.2 Commit

```
git add results/raw/exp_llama3_*/*/seed_456/stage5_llama3_extra_seeds/
git commit -m "experiments: add LLaMA-3.2-3B IID seed 456 for all three methods"
```

---

## 5. TASK 4: Downstream evaluation on the six new checkpoints (~6 hours)

For each of the 6 new checkpoints, run a four-benchmark evaluation (MMLU, ARC-Easy, BoolQ, HellaSwag).

The maintainer ran the seed-42 LLaMA evaluations in two passes (three-benchmark first, then HellaSwag separately). For Task 4, run all four benchmarks in a single eval call per checkpoint, using `--benchmarks mmlu arc_easy boolq hellaswag`.

### 5.1 Script

Use a single bash loop to evaluate all six checkpoints:

```bash
for SEED in 123 456; do
  for METHOD_TUPLE in \
    "exp_llama3_flora_iid:flora" \
    "exp_llama3_two_phase_k8_iid:two_phase" \
    "exp_llama3_reverse_adaptive_iid:reverse_adaptive"; do
    EXP=$(echo $METHOD_TUPLE | cut -d: -f1)
    METHOD=$(echo $METHOD_TUPLE | cut -d: -f2)
    
    CKPT=$(find results/raw/$EXP/$METHOD/seed_$SEED/stage5_llama3_extra_seeds \
      -name "final_adapter_state.pt" | sort | tail -1)
    
    OUTDIR=results/downstream/$EXP/$METHOD/seed_$SEED/stage5_llama3_extra_seeds
    mkdir -p $OUTDIR
    
    echo "Evaluating: $CKPT"
    python3 scripts/evaluate_checkpoint.py \
      --checkpoint "$CKPT" \
      --base-model meta-llama/Llama-3.2-3B \
      --benchmarks mmlu arc_easy boolq hellaswag \
      --num-examples 500 \
      --output "$OUTDIR/downstream_results.json"
  done
done
```

Expected wall clock: approximately 1 hour per checkpoint × 6 = 6 hours total.

### 5.2 Acceptance criteria

After completion:

```bash
find results/downstream/exp_llama3_*/*/seed_{123,456}/stage5_llama3_extra_seeds -name "*.json"
```

Expected: 6 JSON files.

For each downstream JSON, verify it contains accuracy values for all four benchmarks. Spot-check the numbers are sensible:

- MMLU: 0.54-0.58 range (similar to seed-42 LLaMA)
- ARC-Easy: 0.82-0.85
- BoolQ: 0.70-0.75
- HellaSwag: 0.53-0.55

If any benchmark accuracy is at chance (0.25 for 4-way, 0.50 for binary) or wildly outside the expected range, flag it. This indicates an evaluation problem, not necessarily a model problem.

### 5.3 Commit

```
git add results/downstream/exp_llama3_*/*/seed_{123,456}/stage5_llama3_extra_seeds/
git commit -m "experiments: downstream eval for LLaMA-3.2-3B seeds 123 and 456"
```

---

## 6. TASK 5: Regenerate analysis artifacts (~5 minutes)

The master CSV and statistical tests need to incorporate the new runs.

### 6.1 Regenerate the master table

```bash
python3 scripts/build_results_table.py
```

Verify the row count increased by 6 (from 67 to 73). The new rows should have:
- `exp_name` matching the LLaMA-3.2-3B configs
- `setting`: "iid"
- `scale`: "llama3_3b"
- `seed`: "123" or "456"
- `tag`: "stage5_llama3_extra_seeds"
- `final_loss`, `total_mb`, `mmlu_acc`, `arc_easy_acc`, `boolq_acc`, `hellaswag_acc` all populated

### 6.2 Regenerate the statistical tests

The existing `scripts/statistical_analysis.py` only computes TinyLlama IID 3-seed paired tests. The new LLaMA-3.2-3B 3-seed data enables paired tests at 3B scale.

Modify `scripts/statistical_analysis.py` to add LLaMA-3.2-3B 3-seed comparisons. Append to the existing main() function:

```python
llama_flora_pattern = "results/raw/exp_llama3_flora_iid/flora/seed_*/stage5_llama3*/*/results.json"
llama_tp8_pattern = "results/raw/exp_llama3_two_phase_k8_iid/two_phase/seed_*/stage5_llama3*/*/results.json"
llama_ra_pattern = "results/raw/exp_llama3_reverse_adaptive_iid/reverse_adaptive/seed_*/stage5_llama3*/*/results.json"

llama_flora = collect_seeds_iid(llama_flora_pattern)
llama_tp8 = collect_seeds_iid(llama_tp8_pattern)
llama_ra = collect_seeds_iid(llama_ra_pattern)

print(f"\n[LLaMA-3.2-3B IID]")
print(f"FLoRA seeds: {len(llama_flora)} losses {llama_flora}")
print(f"Two-Phase K=8 seeds: {len(llama_tp8)} losses {llama_tp8}")
print(f"ReverseAdaptive seeds: {len(llama_ra)} losses {llama_ra}")

if len(llama_flora) == len(llama_tp8) and len(llama_flora) >= 2:
    rows.append(paired_test(llama_tp8, llama_flora, "[LLaMA-3B] Two-Phase K=8 vs FLoRA (loss)"))
if len(llama_flora) == len(llama_ra) and len(llama_flora) >= 2:
    rows.append(paired_test(llama_ra, llama_flora, "[LLaMA-3B] ReverseAdaptive vs FLoRA (loss)"))
if len(llama_tp8) == len(llama_ra) and len(llama_tp8) >= 2:
    rows.append(paired_test(llama_ra, llama_tp8, "[LLaMA-3B] ReverseAdaptive vs Two-Phase K=8 (loss)"))
```

Then run:

```bash
python3 scripts/statistical_analysis.py
```

Expected: 6 rows in `analysis/statistical_tests.csv` (3 TinyLlama + 3 LLaMA-3.2-3B).

### 6.3 Commit

```
git add analysis/ scripts/statistical_analysis.py
git commit -m "experiments: regenerate analysis artifacts with LLaMA-3.2-3B 3-seed data"
```

---

## 7. TASK 6: Update Figure 6 with error bars (~30 minutes)

`figures/fig6_scale_validation.pdf` currently shows the LLaMA panel as single-seed points. With three seeds now available, the LLaMA panel should show mean ± std error bars, matching the structure of the TinyLlama panel.

### 7.1 Modify the figure generator

In `scripts/generate_paper_figures.py`, locate the `figure6_scale_validation` function. The current LLaMA panel logic likely plots single points without error bars. Update it to:

1. Filter the master CSV for `scale == "llama3_3b"` rows
2. Group by method
3. Compute mean and std of `final_loss` and `total_mb` across seeds
4. Plot as errorbar markers, same style as the TinyLlama panel

Important: keep the TinyLlama panel logic unchanged. The only change is to make the LLaMA panel use the same error-bar treatment.

### 7.2 Regenerate the figures

```bash
python3 scripts/generate_paper_figures.py
```

Verify `figures/fig6_scale_validation.pdf` was updated (check modification time). All other figures should remain unchanged.

### 7.3 Visual check

Manually open `figures/fig6_scale_validation.pdf`. Confirm:

- The left panel (TinyLlama) is unchanged
- The right panel (LLaMA-3.2-3B) now shows error bars on all three method points
- The dotted polyline connecting the three method points is still present
- Axis ranges and labels are unchanged

### 7.4 Commit

```
git add scripts/generate_paper_figures.py figures/fig6_scale_validation.pdf
git commit -m "experiments: add error bars to LLaMA-3.2-3B panel in scale validation figure"
```

---

## 8. TASK 7: Final verification

Run the full validation sequence one more time.

```bash
# Tests
python3 -m pytest tests/ -v

# Regenerate analysis artifacts (idempotent)
python3 scripts/build_results_table.py
python3 scripts/statistical_analysis.py
python3 scripts/generate_paper_figures.py

# Verify expected row counts
wc -l analysis/final_results_table.csv  # expect 74 (header + 73 data rows)
wc -l analysis/statistical_tests.csv     # expect 7 (header + 6 data rows)
ls figures/                              # expect 6 PDFs
```

Expected outcomes:
- All 20 tests pass
- CSV has 73 data rows (was 67)
- Statistical tests CSV has 6 data rows (was 3)
- Six PDFs in figures/, all readable

---

## 9. WHAT TO REPORT BACK

When all tasks complete, send the maintainer:

1. The 6 new training runs' final-round summary as a table:

   | Method | Seed | Final loss | Total MB | Switch round (if RA) |
   |--------|------|-----------|----------|----------------------|
   | FLoRA | 123 | ? | ? | -- |
   | FLoRA | 456 | ? | ? | -- |
   | Two-Phase K=8 | 123 | ? | ? | -- |
   | Two-Phase K=8 | 456 | ? | ? | -- |
   | ReverseAdaptive | 123 | ? | ? | ? |
   | ReverseAdaptive | 456 | ? | ? | ? |

2. The 6 new downstream evaluations' results as a table:

   | Method | Seed | MMLU | ARC-Easy | BoolQ | HellaSwag |
   |--------|------|------|----------|-------|-----------|
   | (one row per checkpoint) | | | | | |

3. The full contents of the new `analysis/statistical_tests.csv` (all 6 rows).

4. Confirmation that:
   - All 20 tests pass
   - `analysis/final_results_table.csv` has 73 data rows
   - All 6 figures in `figures/` exist and are readable
   - `figures/fig6_scale_validation.pdf` shows error bars on the LLaMA panel

5. Any anomalies:
   - ReverseAdaptive switch rounds that differ from seed 42's round 8
   - Loss values outside expected ranges
   - Any FLoRA non-finite warnings during aggregation
   - Any OOM or training-failure events

---

## 10. ANTI-PATTERNS

Do NOT do any of the following:

- Skip the pre-flight verification (Task 1). Compute is expensive; failing fast is cheap.
- Run additional seeds beyond 123 and 456. The brief specifies these two; do not improvise.
- Run additional benchmarks beyond MMLU, ARC-Easy, BoolQ, HellaSwag. These are the four the paper uses.
- Re-run seed-42 LLaMA experiments. They are canonical and bit-exact reproducible.
- Modify any aggregator code, runner code, or evaluation code beyond the one statistical_analysis.py addition and the one generate_paper_figures.py fix specified above.
- Modify the LLaMA configs. They produced bit-exact deterministic communication on seed 42; perturbing them invalidates cross-seed comparison.
- Commit `final_adapter_state.pt` files (they are .gitignored).
- Reformat any existing code with black, isort, or similar.

---

## 11. EXPECTED CALENDAR TIMELINE

Assuming overnight scheduling for training runs and daytime evaluation:

| Day | Activity | Cursor effort | Compute effort |
|-----|----------|---------------|----------------|
| Day 1 | Task 1 verification, launch Task 2 overnight | 30 min | None during day, 23 hr overnight |
| Day 2 | Verify Task 2 results in morning, launch Task 3 overnight | 30 min | 23 hr overnight |
| Day 3 | Verify Task 3 results, run Task 4 (downstream evals) during the day | 1 hr active monitoring | ~6 hr during day |
| Day 4 | Tasks 5 and 6 (analysis regeneration, figure update) | 2 hr active | ~30 min compute |
| Day 5 | Task 7 final verification, report back | 1 hr | None |

Total: ~5 hours of Cursor active work, ~58 hours of compute, ~5 calendar days.

If overnight scheduling is interrupted or runs fail partway, add 1-2 days of buffer.

---

## END OF PATH B BRIEF

After Cursor completes this work, the maintainer will:
1. Update Table 5 in the paper (LLaMA-3.2-3B results table) to show 3-seed mean ± std.
2. Update Table 3b (LLaMA-3.2-3B downstream) similarly.
3. Update Section 4.6 (Scale Validation) prose to reflect 3-seed validation rather than single-seed.
4. Update Section 6 limitations to remove "single seed at LLaMA-3.2-3B" and replace with "single seed for downstream evaluation only" (since training is now 3-seed but downstream is reported per-seed without cross-seed averaging in the main paper).
5. Add the theoretical justification paragraph for the plateau signal to Section 3.3 (the maintainer drafts this; no Cursor work needed).
6. Final LaTeX conversion and submission.

The paper goes from "single-seed LLaMA-3.2-3B" to "three-seed LLaMA-3.2-3B validation" with this brief. This is the single most impactful pre-submission strengthening the maintainer can do.
