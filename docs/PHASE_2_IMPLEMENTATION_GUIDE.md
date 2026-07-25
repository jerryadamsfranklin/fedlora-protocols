# Neurocomputing Phase 2 Implementation Guide: LLaMA-3.1-8B Third Model Scale

**For use with Cursor AI as implementer**
**Prerequisite:** Neurocomputing Phase 1 is cleared (33/34 runs clean or explained-and-accepted). This guide opens with a Phase 1 closeout task (the #17 diagnostic rerun) that must complete before the paper's non-IID robustness section is written, but does not block starting the 8B setup work below.
**Hard blocker for the 8B runs specifically:** A.1 HuggingFace license access to `meta-llama/Meta-Llama-3.1-8B` must be confirmed before Part 2 of this guide. It was deferred through Phases 0 and 1 and comes due now.
**Branch:** create `neurocomputing/phase-2-llama8b` off the updated main (after Phase 1 is merged). Confirm with maintainer before branching.

---

## Part 0: Phase 1 closeout — the #17 diagnostic rerun (do this first, it is cheap and it gates a paper claim)

### 0.1 Why this exists

Phase 1 batch verification found that run #17 (ReverseAdaptive, Non-IID alpha=0.1, seed 456) switched on a different round across backends (MPS round 6 vs CUDA round 7), with a 110 MB communication difference as a consequence. Run #16 (same setting, seed 123) also misbehaved, breaching the consecutive-spike rule. That means 2 of 3 seeds at alpha=0.1 show backend-dependent behavior, with only seed 42 clean.

The single existing run cannot distinguish two very different explanations:
- **Explanation A (stable backend difference):** MPS reliably switches at 6, CUDA reliably switches at 7. This is a characterizable, reportable cross-backend limitation.
- **Explanation B (stochastic switch near a noisy boundary):** the switch round is itself unstable near alpha=0.1's heterogeneity, and could land on 6 or 7 depending on run-level randomness even within a single backend. This is a stronger and more significant statement — it means switch timing at severe heterogeneity is not fully reproducible even within a backend.

These lead to different sentences in the paper. You cannot write the non-IID robustness section correctly until you know which is true.

### 0.2 The diagnostic

Run the #17 config (`exp_reverse_adaptive_noniid_alpha01.yaml`, seed 456) on CUDA a second time, changing only the run-level random seed (the training/initialization RNG), NOT the data-partitioning seed (keep data seed 456 so the client data split is identical). The question is whether the switch round is stable when the data is held fixed but run-level randomness changes.

```bash
python3 scripts/run_experiment.py \
  --config config/exp_reverse_adaptive_noniid_alpha01.yaml \
  --data-seed 456 \
  --run-seed <a_different_value, e.g. 999> \
  --device cuda \
  --tag phase1_17_diagnostic
```

Adjust flag names to match what the script actually exposes. If the script does not currently separate data-partitioning seed from run-level seed, that separation itself needs to be surfaced (this is important for the paper regardless — a reviewer will want to know whether reported variance is over data splits or over training randomness).

### 0.3 Interpreting the result

- **If the second CUDA run also switches at round 7:** switch round is stable within CUDA for this data split. Combined with MPS switching at 6, this is Explanation A — a stable backend difference. Paper claim: "At alpha=0.1, the adaptive switch fired one round later on CUDA than MPS for one of three seeds; the discrete decision is backend-sensitive at severe heterogeneity."
- **If the second CUDA run switches at round 6 (or anything other than 7):** switch round is NOT stable even within CUDA when only run-level randomness changes. This is Explanation B. Paper claim (stronger, more honest): "At alpha=0.1, adaptive switch timing is sensitive to run-level randomness and may vary by one round; we report switch round as a distribution rather than a fixed value at this heterogeneity level."

### 0.4 Record and report

Write the result to `docs/neurocomputing/PHASE_1_17_DIAGNOSTIC.md` and report back to the maintainer before drafting the non-IID robustness section. This diagnostic does not block starting Part 1 and Part 2 below — run it in parallel with 8B environment setup.

---

## Part 1: 8B B-only fraction pre-check (arithmetic, no GPU)

The A.2 calculation in Phase 0 already computed the LLaMA-3.1-8B B-only fraction as 38.46%, derived from its GQA config (32 query heads / 8 KV heads, hidden 4096, head_dim 128, applied to q/k/v/o projections only, no MLP). Before spending any 8B GPU-hours:

1. Confirm the LoRA target modules in the 8B config match exactly what was assumed in the A.2 calculation: q_proj, k_proj, v_proj, o_proj, and NO MLP modules (gate_proj/up_proj/down_proj). If the 8B config applies LoRA to a different module set than the 1.1B and 3.2B configs did, the 38.46% figure is invalid and the whole cross-scale communication comparison breaks. This is the single most important consistency check in this phase.
2. Confirm LoRA rank r=16 matches the smaller scales (rank cancels out of the fraction, but must be consistent for the parameter-count and absolute-communication numbers to be comparable across scales).

If either differs, stop and reconcile with the maintainer before running anything.

---

## Part 2: HuggingFace access and environment (hard blocker resolution)

### 2.1 Confirm 8B license access

1. Go to `huggingface.co/meta-llama/Meta-Llama-3.1-8B`, confirm "you have been granted access."
2. If still pending, this phase's GPU work cannot start. The diagnostic in Part 0 and the arithmetic in Part 1 can proceed regardless.

### 2.2 Environment pins (these are hard requirements, learned the expensive way in Phase 1)

On the 8B instance, the environment must be pinned to avoid silent breakage:
- `transformers` must be **4.45.x**, NOT 5.x. LLaMA-3.x rope_scaling requires transformers >= 4.43, but transformers 5.x requires torch >= 2.4, which conflicts with the validated torch 2.2 setup. Pin explicitly:
  ```bash
  pip install "transformers==4.45.2" "torch==2.2.*" peft accelerate datasets bitsandbytes huggingface_hub
  ```
- Confirm versions before any run:
  ```bash
  python3 -c "import transformers, torch; print(transformers.__version__, torch.__version__)"
  ```

### 2.3 Disk and instance sizing

- Set disk to **at least 64 GB, ideally 100 GB** at instance creation time. The 8B model weights alone are substantially larger than TinyLlama/3.2B, and the 16 GB default images used in Phase 1 were already tight for the smaller models.
- LLaMA-3.1-8B in float32 will not fit in 24 GB VRAM for training. Rent an **A100 40GB** (as planned), and confirm the memory math before the first full run: verify the model loads and a single training step completes within VRAM before launching multi-round jobs. If 40 GB proves tight with the batch size used at smaller scales, the options in order of preference are: reduce per-client batch size, then enable gradient checkpointing, then (last resort, and only if explicitly agreed with maintainer) drop to bf16 — but note that changing precision from the float32 used at 1.1B/3.2B introduces a confound in the cross-scale comparison and must be disclosed if done.

---

## Part 3: 8B experiment matrix

Run the three existing methods at 8B scale, matching the config pattern used for 1.1B and 3.2B. Keep float32 if VRAM allows (for cross-scale consistency); see 2.3 if it does not.

### 3.1 Core runs (IID)

| Method | Setting | Seeds | Hardware |
|---|---|---|---|
| FLoRA | IID | 42, 123, 456 | A100 40GB |
| ReverseAdaptive | IID | 42, 123, 456 | A100 40GB |
| Two-Phase K=8 | IID | 42, 123, 456 | A100 40GB |

Run downstream evaluation (MMLU, ARC-Easy, BoolQ, HellaSwag) on all resulting checkpoints, matching the eval pipeline used at smaller scales.

### 3.2 Non-IID runs (ReverseAdaptive only, per plan)

| Method | Setting | Seeds | Hardware |
|---|---|---|---|
| ReverseAdaptive | Non-IID alpha=0.5 | 42, 123, 456 | A100 40GB |
| ReverseAdaptive | Non-IID alpha=0.1 | 42, 123, 456 | A100 40GB |

### 3.3 Baseline non-IID coverage gap — DECIDED: option (b)

**Maintainer decision (2026-07-16): option (b).**

Add **6** TinyLlama-1.1B baseline runs — FLoRA and Two-Phase K=8 × Non-IID
**α=0.1** × seeds **42 / 123 / 456**. Scope is strict: TinyLlama only, α=0.1 only,
no other scales. Full downstream eval (MMLU, ARC-Easy, BoolQ, HellaSwag) is
**required** on all 6 — not training loss alone. Run on RTX 4090; fold into the
master CSV as a Phase 1 TinyLlama-tier extension. Does **not** block 8B setup
(parallel OK). Report the 6 results (downstream + final communication MB) before
drafting the non-IID robustness section.

Operational docs: `docs/PHASE_2_PART33B_BASELINES.md`,
`docs/phase2_part33b_manifest.csv`, `scripts/run_part33b_alpha01_baselines.sh`.
Configs: `config/exp_flora_noniid_alpha01.yaml`,
`config/exp_two_phase_k8_noniid_alpha01.yaml`.

---

## Part 4: Backend verification for 8B

The 8B runs are CUDA-only (there is no MPS counterpart — TinyLlama and 3.2B had MPS history, 8B never will, since 8B was never run on the Mac). This means `verify_backend_match.py` cannot be run for 8B in the MPS-vs-CUDA sense.

Instead, apply these internal consistency checks to each 8B run:
1. Confirm the measured B-only communication fraction matches the 38.46% predicted in Part 1 (within a small tolerance for rounding). If the measured fraction deviates meaningfully from 38.46%, the protocol is doing something different at 8B than the arithmetic predicts, and that must be investigated before trusting any 8B result.
2. Confirm the switch round for ReverseAdaptive runs falls in a plausible range consistent with the smaller scales (do not expect an identical round, but a wildly different switch round would signal a problem).
3. For IID 8B runs, confirm loss curves are monotonically decreasing and final losses are sane relative to the 3.2B results (8B should generally reach comparable or lower final loss than 3.2B on the same data; a higher final loss would signal an under-training or config issue).

---

## Part 5: Consolidate and report

1. Add all 8B runs to the master results CSV.
2. Regenerate any paper figures that show scaling across model sizes to now include the 8B point.
3. Update the paper's model-scale description to state three scales (1.1B, 3.2B, 8B).
4. Report to the maintainer: all 8B runs completed, measured vs predicted B-only fraction, downstream eval results, actual GPU-hours and cost vs the ~$36 IID + ~$16 non-IID estimate, and confirmation the A100 instance is stopped or destroyed.

Do not proceed to Phase 3 (Dolly-15k second dataset) until the maintainer reviews and clears Phase 2.

---

## Things not to do in this phase

- Do not start 8B GPU runs before confirming HuggingFace access (Part 2.1) and the LoRA target-module consistency check (Part 1).
- Do not silently switch precision to bf16 to fit VRAM without disclosing it — it introduces a cross-scale confound.
- Do not expand Part 3.3(b) beyond the 6 TinyLlama α=0.1 FLoRA/Two-Phase runs without a new maintainer decision.
- Do not skip the #17 diagnostic (Part 0) before writing the non-IID robustness section — its outcome determines what that section can claim.
- Do not add Dolly-15k, FFA-LoRA, or FedIT here — those are Phases 3 and 4.
