# Neurocomputing Phase 1 Implementation Guide: CUDA Reruns of Existing Suite

**For use with Cursor AI as implementer**
**Prerequisite:** Neurocomputing Phase 0 is complete and cleared. Do not start this document if Phase 0's checklist in `PHASE_0_IMPLEMENTATION_GUIDE.md` has any unchecked items.
**Branch:** continue on `neurocomputing/phase-0-validation`, or create `neurocomputing/phase-1-cuda-reruns` off it if the maintainer prefers a fresh branch per phase — confirm with maintainer before branching.
**Gate protocol in effect:** this phase runs under `docs/PHASE_1_EARLY_GATES.md`. Read that file before starting. Summary: the first non-IID run and the first FLoRA (or Two-Phase K=8) run must each pass `verify_backend_match.py` against their MPS counterpart before the remainder of the suite runs unattended.

---

## 0. Goal of this phase

Rerun every one of the existing 34 completed training runs (currently recorded only on Apple M4 Pro MPS) under CUDA, verify each rerun matches its MPS counterpart within the Phase 0-established 0.01 absolute tolerance, and update the paper's hardware description to reflect CUDA as the primary reported backend.

**Do not** add the third model scale (LLaMA-3.1-8B), the second dataset (Dolly-15k), or the two new baselines (FFA-LoRA, FedIT) in this phase. Those are Phases 2, 3, and 4 respectively, and depend on this phase's results being verified first. Scope creep here wastes GPU-hours you'll need later.

---

## 1. Step 1: Build the run manifest

Before renting anything, enumerate the actual existing 34 MPS runs programmatically. Do not assume a count or breakdown from memory or the paper draft — read it directly from the results directory.

```bash
# Run this locally first, no GPU needed
python3 scripts/list_existing_runs.py --results-dir results/raw/ --output docs/phase1_run_manifest.csv
```

If `list_existing_runs.py` doesn't exist yet, write a short script that walks `results/raw/` and extracts, for each completed run: model (TinyLlama-1.1B or LLaMA-3.2-3B), method (FLoRA, Two-Phase K=8, or ReverseAdaptive), data setting (IID or non-IID with Dirichlet alpha value), seed, and the path to its `results.json`. Output this as a CSV with one row per run — this becomes your rerun checklist.

Confirm the manifest totals 34 rows before proceeding. If it doesn't, stop and reconcile the discrepancy with the maintainer before continuing — do not silently proceed with an incomplete or over-complete manifest.

---

## 2. Step 2: Order the manifest according to the gate protocol

Sort the 34-row manifest into this exact execution order:

1. **Gate run 1:** the first non-IID run in the manifest (any method, either model — pick the smallest/cheapest, likely TinyLlama-1.1B, to keep the gate check fast and cheap)
2. **Gate run 2:** the first FLoRA or Two-Phase K=8 run in the manifest (any data setting, either model)
3. **Remaining 32 runs:** all other runs, in any order, batched by hardware tier for rental efficiency (see Step 4)

Do not run anything from group 3 until both gate runs in groups 1 and 2 have independently passed `verify_backend_match.py` against their MPS counterparts.

---

## 3. Step 3: Instance selection by model scale

| Model | Hardware | Estimated hours (full existing suite) | Estimated cost |
|---|---|---|---|
| TinyLlama-1.1B | Vast.ai RTX 4090 | ~15 hrs | ~$4-5 |
| LLaMA-3.2-3B | Vast.ai A100 40GB | ~20 hrs | ~$16-20 |

Rent instances following the same selection criteria used in Phase 0: verified hosts, reliability ≥ 98 percent, PyTorch (Vast) template, single-GPU (not multi-GPU) listings.

Since this phase involves 34 runs rather than one, expect to keep an instance rented for a multi-hour block rather than spinning up per-run. Confirm the instance's **Max Duration** field (shown on the listing) comfortably exceeds your expected session length before renting, to avoid a host-imposed cutoff mid-batch.

---

## 4. Step 4: Run the two gate runs first

For each gate run:

1. Run the experiment under CUDA using the same config that produced the original MPS result, with `--device cuda` and a new tag, e.g. `--tag phase1_cuda_rerun`.
2. Immediately run `scripts/verify_backend_match.py` comparing the new CUDA result against the saved MPS result for that exact config/seed, at `atol=0.01, rtol=0` (the same relaxed tolerance established in Phase 0).
3. Record the result (pass/fail, delta values) in `docs/PHASE_1_GATE_RESULTS.md`.

**If either gate run fails:** stop. Do not proceed to the remaining 32 runs. Report the failure to the maintainer with the specific delta values and which run (model/method/setting/seed) failed. Do not adjust the tolerance further or retry silently — a gate failure on a non-IID or a different-aggregator run means there's a real behavioral difference between backends that Phase 0's single ReverseAdaptive/IID check didn't catch, and it needs investigation, not a workaround.

**If both gate runs pass:** proceed to Step 5.

---

## 5. Step 5: Run the remaining 32 runs

1. Batch the remaining runs by hardware tier: run all remaining TinyLlama-1.1B configs on the RTX 4090 instance, all remaining LLaMA-3.2-3B configs on the A100 40GB instance.
2. Use the same `--device cuda --tag phase1_cuda_rerun` pattern (or an equivalent tag) for every run, so all Phase 1 CUDA results are identifiable as a batch distinct from the original MPS results and from the Phase 0 validation run's `phase0_cuda_validation` tag.
3. Run `verify_backend_match.py` against each run's MPS counterpart as it completes, rather than waiting until all 32 are done. If a run fails partway through the batch, pause and report it before continuing — don't let one bad result sit unnoticed among 30 good ones.
4. If an instance is preempted or interrupted mid-batch (spot pricing risk), use the checkpoint-restart procedure validated in Phase 0 to resume. Log any such interruption and its resolution in `PHASE_1_GATE_RESULTS.md` for the record, even though it's expected to work correctly now.

---

## 6. Step 6: Consolidate results

Once all 34 runs have CUDA counterparts and have passed backend verification:

1. Regenerate the master results CSV to include the CUDA run set. Keep the original MPS results in the repository (do not delete them) but mark the CUDA set as the canonical/reported results going forward.
2. Update the paper's hardware description (Table 1 or equivalent) to state CUDA (specify GPU models used: RTX 4090, A100 40GB) as the primary reported hardware. The MPS results may be retained as a supplementary robustness/reproducibility note if useful, but should not remain the headline hardware claim.
3. Regenerate any of the six existing paper figures that depend on these 34 runs' numeric results, using the CUDA figures.

---

## 7. Step 7: Report back to maintainer

Produce a summary for the maintainer covering:
- Confirmation that the run manifest totaled 34 and all 34 have verified CUDA counterparts
- The two gate run results in detail (deltas, pass confirmation)
- Any interruptions encountered and how they were resolved
- Actual GPU-hours and cost spent, compared against the ~$20-25 estimate in the table above
- Confirmation that both rental instances have been stopped or destroyed

Do not proceed to Phase 2 (LLaMA-3.1-8B) until the maintainer has reviewed this summary and confirmed Phase 1 is cleared, mirroring how Phase 0 required an explicit sign-off before Phase 1 began.

---

## 8. Things not to do in this phase

- Do not add LLaMA-3.1-8B, Dolly-15k, FFA-LoRA, or FedIT — those are later phases.
- Do not modify the aggregator implementations (FLoRA, Two-Phase K=8, ReverseAdaptive) while doing this rerun work — this phase is a hardware migration, not an algorithm change. Any numerical difference between MPS and CUDA results should come from backend/precision differences only, not from code changes made during this phase.
- Do not delete the original MPS results, even after CUDA reruns are verified. They remain useful as a documented cross-backend robustness check.
- Do not skip the gate protocol to save time, even if the Phase 0 single-run check passed cleanly. Non-IID data partitioning and the FLoRA/Two-Phase-K8 aggregators are different code paths than the ReverseAdaptive/IID combination validated in Phase 0, and have not yet been confirmed to match across backends.
