# Neurocomputing Submission: Final Plan

**Replaces:** the previous Phase 2 (LLaMA-3.1-8B), Phase 3 (Dolly), and Phase 4 (baselines) guides. Those are superseded. 8B is dropped from scope entirely.

**Paper:** Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning
**Target:** Neurocomputing (Q1, Elsevier). Submit by early September 2026.
**Branch:** `neurocomputing/final-experiments` off main.

---

## Why this plan looks different from the earlier phases

The earlier plan assumed the paper's problem was scale and hardware credibility. It is not. A competitive-landscape review established that the paper's real weaknesses, in order of how much they matter to a reviewer, are:

1. Single dataset (Alpaca-3k only) — the empirical claim rests on one data distribution
2. No comparison against published baseline methods — only against the author's own variants
3. Quality claims anchored on benchmarks that cannot measure what the fine-tuning changes
4. An overclaim in Section 5.3 about prior work that recent literature contradicts

Adding a third model scale (8B) addresses none of these and costs the most. It is dropped. Everything in this plan targets one of the four items above.

---

## Part 1: Dolly-15k second dataset (highest priority)

**Why:** directly answers the single-dataset objection, which is the paper's biggest empirical weakness.

**Scope, strictly:** TinyLlama-1.1B only. Do not extend to LLaMA-3.2-3B in this round.

| Method | Setting | Seeds |
|---|---|---|
| FLoRA | IID | 42, 123, 456 |
| Two-Phase K=8 | IID | 42, 123, 456 |
| ReverseAdaptive (tau=0.01) | IID | 42, 123, 456 |
| ReverseAdaptive | Non-IID alpha=0.5 | 42, 123, 456 |

12 runs. RTX 4090. Estimated ~$5-8.

**Dataset preparation:** subset Dolly-15k to 3,000 samples to match the Alpaca subset size exactly. In the paper, state explicitly that the subset size was chosen for parity with the Alpaca configuration so that cross-dataset comparison is controlled. Do not leave this unstated.

**Required outputs per run:** final loss, per-round communication MB, switch round, and held-out instruction-following loss/perplexity (see Part 3 for why this metric, not the zero-shot benchmarks).

**What this buys the paper:** the ability to say the communication savings and the switch behavior hold across two distinct instruction datasets, not one. If the savings differ meaningfully between Alpaca and Dolly, that is itself a finding worth reporting honestly, not a problem.

---

## Part 2: FFA-LoRA and FedIT as published baselines

**Why:** the paper currently compares ReverseAdaptive against FLoRA and Two-Phase K=8, both of which are the author's own configurations. A reviewer will observe that the paper positions itself against FFA-LoRA and FedIT in the related work but never compares against them empirically.

**Critical implementation instruction:** use the original authors' released code wherever it exists. Do not reimplement from scratch if an official implementation is available. Where reimplementation is unavoidable, state so explicitly in the paper and describe exactly what was implemented. A reviewer disputing a reimplementation is a fight that cannot be won; transparency is the only defense.

| Method | Dataset | Setting | Seeds |
|---|---|---|---|
| FFA-LoRA | Alpaca-3k | IID | 42, 123, 456 |
| FFA-LoRA | Alpaca-3k | Non-IID alpha=0.5 | 42, 123, 456 |
| FedIT | Alpaca-3k | IID | 42, 123, 456 |
| FedIT | Alpaca-3k | Non-IID alpha=0.5 | 42, 123, 456 |

12 runs, TinyLlama-1.1B, RTX 4090. Estimated ~$5-8.

**Note on FFA-LoRA specifically:** it freezes A at initialization and trains only B from round 1. This is the natural lower bound on communication for the B-only family and is the most important baseline in the paper, because ReverseAdaptive's entire premise is that learning A first and freezing later beats freezing from the start. If FFA-LoRA matches ReverseAdaptive on quality while using less communication, that is a serious result the paper must report honestly rather than bury.

---

## Part 3: Quality metric correction (no new runs, applies to all results)

**The problem:** MMLU, ARC-Easy, BoolQ, and HellaSwag do not move under 3,000-sample instruction tuning of a 1.1B model. Verified directly: base TinyLlama scores at or above every fine-tuned checkpoint on these benchmarks. They cannot support a quality-preservation claim.

**The instrument that works:** held-out instruction-following loss/perplexity on the training distribution. Verified: base loss 1.9352 (ppl 6.9254) vs. fine-tuned 1.3432 (ppl 3.8311). This metric discriminates cleanly.

**Required action:** compute held-out instruction-following loss for every checkpoint in the paper — all existing runs plus everything from Parts 1 and 2. Held-out slice: Alpaca `train[3000:3500]`, 500 examples, and the equivalent held-out slice for Dolly runs. This is evaluation-only against saved adapters; no retraining required.

**How the paper uses this:** the quality-preservation claim is anchored on (a) final training loss with paired t-tests, and (b) held-out instruction-following loss across methods. The four zero-shot benchmarks are retained and reported, but explicitly framed as insensitive at this scale — the paper's existing Section 5.5 already makes this argument correctly and should be strengthened, not removed.

---

## Part 4: Paper revisions

### 4.1 Section 5.3 overclaim (must fix)

The current text claims prior federated LoRA work reports savings only as parameter-count ratios. This is no longer accurate: several recent papers report measured MB with upload/download splits. Leaving this uncorrected gives a reviewer a concrete factual error to attack.

Reframe to what remains true and defensible: existing accounting generally omits the transition-round cost when switching aggregation modes, and does not account for architecture-dependent effects such as GQA on the B-only fraction. That narrower claim is supported by the paper's own evidence.

### 4.2 Hardware framing (must fix)

The paper currently states all experiments ran on Apple M4 Pro MPS with no external GPU, and reports 285 hours of compute. The Phase 1 work reran the full 34-run suite on CUDA.

**Recommended framing:** primary results on Apple M4 Pro MPS, independently reproduced on NVIDIA RTX 4090 and A100 hardware, with cross-backend agreement reported in an appendix. This is more interesting than simply swapping to CUDA, it preserves the resource-constrained research premise, and it offers a reproducibility guarantee most papers in this space do not provide.

Sections requiring edits: Table 1 (Hardware row), Section 4.1, Appendix D.

### 4.3 Cross-backend reproducibility appendix (new)

Add an appendix documenting the MPS-to-CUDA validation: 34 runs compared, setting-aware tolerances (IID atol 0.01; non-IID atol 0.025 with exact switch-round match and a maximum of 2 consecutive rounds exceeding 0.01), and the characterized finding that at alpha=0.1 the discrete switch round is backend-sensitive by at most one round on one of three data partitions, while being exactly reproducible at all other settings.

State the tolerance policy and the alpha=0.1 finding plainly. Framing this as a characterized operating limit is stronger than claiming exact cross-backend identity everywhere.

### 4.4 Abstract (must fix)

Remove "no measurable downstream cost." Replace with the claim the evidence actually supports: statistically indistinguishable final training loss, and no differential degradation across methods on held-out instruction-following.

### 4.5 Reproducibility section reconciliation

Section 5.4 claims the no-switch baseline matches FLoRA to 10 decimal places. This remains true within a single backend and should be kept, but must be stated as a within-backend result so it does not appear to contradict the cross-backend tolerances in the new appendix.

### 4.6 Non-IID seed reconciliation

Table 8 reports five seeds at alpha=0.1 (42, 123, 456, 789, 1000), but only three (42, 123, 456) were rerun during cross-backend validation. Decide explicitly: either rerun 789 and 1000, or report the table with a clear note on which seeds have cross-backend verification. Do not let seeds silently disappear between the earlier submitted version and this one.

---

## Execution order

1. Part 3 held-out evaluation on all existing checkpoints (no GPU rental; establishes the metric before new runs)
2. Part 1 Dolly runs (12 runs, RTX 4090)
3. Part 2 baseline runs (12 runs, RTX 4090)
4. Part 3 held-out evaluation on all new checkpoints
5. Consolidate results, regenerate tables and figures
6. Part 4 paper revisions
7. Final read for consistency between claims and evidence

Total new GPU cost estimate: $10-16. Total new runs: 24, all TinyLlama-1.1B.

---

## Out of scope

- LLaMA-3.1-8B at any point in this submission
- Additional model scales
- Additional datasets beyond Dolly
- Additional baselines beyond FFA-LoRA and FedIT
- Any new mechanism, aggregator variant, or algorithmic contribution

If a reviewer requests any of the above, it becomes revision work with a concrete request attached. Adding it preemptively delays submission past the window.
