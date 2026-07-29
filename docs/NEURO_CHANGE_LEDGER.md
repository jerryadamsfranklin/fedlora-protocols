# Neurocomputing Revision Ledger

Paper: Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning
Author: Jerry Adams Franklin, Independent Researcher
Target: Neurocomputing (Elsevier, Q1)

Task list of record: `FEDLORA_PAPER_HANDOFF.md` section 7, plus C13 through C18 below.
Content specification: `NEURO_CONTENT_SPEC.md`.
This file holds the approved replacement text. Where this file and the spec disagree, this file wins; every divergence is documented with its reason.

## How to use this file

Every edit is anchored to a quoted `old_str` that appears exactly once in the target file. Line numbers are deliberately absent because they drift as edits land. Apply by string match.

Status values: APPROVED (text settled, safe to paste), PENDING (not yet written), BLOCKED (needs data or a decision).

Run `verify_numbers.py --analysis-dir analysis` after any numeric edit. 174 assertions, all passing as of this revision.

---

## Global conventions

These apply to every block and every table. Getting them inconsistent is how the 1.2605-versus-1.2608 problem happened.

**Standard deviation is SAMPLE sd, not population.** Dispositive evidence: the 3B savings figure is exactly 4.0000 and the 3B communication sd exactly 105.00 under sample sd, versus 3.266 and 85.73 under population. Both match the existing manuscript. Guarded by assertions `convention.sd_is_sample` and `convention.sd_not_population`.

Consequences already applied below: T-N1 and T-N2 sd columns, G5.5 bounds ("at most 0.0016 and 0.0008"), G5.7 spread ratio ("roughly twenty-five times", 25.02).

Consequences still outstanding: `neuro_new_tables.tex` captions say "population standard deviation" and are wrong in all five tables. `make_frontier_figure.py` uses `st.pstdev` for error bars and should use `st.stdev`. The Appendix B variance ratios (47x, 12x, 40x, 44x, 15x) are unaffected, since both columns scale by the same factor.

**Backend provenance (C18).** Communication is protocol-deterministic and backend-invariant. Loss is not. CUDA corpus: T-N1, T-N2, all statistical tests. MPS corpus: threshold ablation (Table 5), LLaMA-3.2-3B, Two-Phase K=10. Loss values are comparable only within a backend. Declared in G5.1.

**Amendment discipline.** When a later group corrects an earlier approved block, apply the change INSIDE that block and leave only a record note. Amendments recorded as separate blocks below the original leave the primary text stale, which is how the abstract carried the p=0.997 inversion, the 5.5x figure and the ambiguous sd claim simultaneously while all three corrections existed elsewhere in this file.

**Tooling discipline.** Any command whose exit status gates a write must not have a pipe between the check and the gate. Piping through `head` or `tail` replaces the exit code and a failing artifact gets staged. Capture the return code first, then branch. This has already caused one bad stage.

**Writing mandates.** No em-dashes. No curly quotes. No signposting words (First, Furthermore, Moreover, Additionally). Affiliation "Independent Researcher." Full name "Jerry Adams Franklin."

**Never write:** "four-target" (no run ever executed four target modules); "progressive freezing" or "partial freezing" (intermediate ratios exist but are unexercised); "label skew" (the partition is over instruction-length buckets); "no external GPU was used" (rented CUDA hardware was used); "population standard deviation".

---

## Work order

Grouped by location, not by correction number, because items collide. Do each group once.

| Group | Location | Items | Status |
|---|---|---|---|
| G0 | Preamble, frontmatter, abstract, end matter | Format conversion, C9 | APPROVED |
| G1 | Section 3.3 + Algorithm 2 | C1, C13 | APPROVED |
| G1b | Algorithm 1 + Section 3.4 | C15, C16 | APPROVED |
| G2 | Table 1 (setup) | C2, C3, C8, C13, C17, N5 row | APPROVED |
| G3 | Section 3.1 + Section 3.4 | C2, C3, C4, C17 | APPROVED |
| G4b | Setup, held-out metric definition | N5 | APPROVED |
| G4 | Section 4.1 compute, baselines, prompt format | C6, C8 | APPROVED |
| G15 | Section 2.2 forward references | N6 (part) | APPROVED |
| G5 | Section 4.2, Table 2 to T-N1, Figure 1 to F-N1 | N1, C14, C18 | APPROVED |
| G6 | Dolly-15k replication + T-N2 | N2 | APPROVED |
| G7 | Section 5.2 rewrite | C10, C19, N1 reframe | APPROVED |
| G8 | Section 5.3 measured bytes + step function | C7, C20 | APPROVED |
| G8b | Section 5.5 + Table 3 caption | N4, C21, C22 | APPROVED |
| G9 | Section 5.4 reproducibility guarantee | C11, C24 | APPROVED |
| G10 | Limitations | C3, C4, C15, C23, C25, N4 | APPROVED |
| G11 | Appendix B, Tables B1 and B2 | C4, C12, C18 | APPROVED |
| G12 | Appendix A | C5 | APPROVED |
| G13 | Appendix D + new Appendix E | C8, N3, C23 | APPROVED |
| G14 | Conclusion | N6, C9, C11, C14, C22, C26 | APPROVED |
| G16 | Section 1 Introduction, contribution claims | C1, C7, C14, C20, C26 | APPROVED |
| G17 | Section 4.6 Scale Validation | C10 | APPROVED |
| G18 | Section 5.1 Implications | C9 | APPROVED |
| G19 | Appendix C statistical tests | C10, open item 7 | APPROVED |

Sequencing rationale (spec Part E): G4b before G5 and G6 because their tables reference the metric. G7 supersedes the G1.4 partial. G10 depends on G3 and G8b wording. G14 last.

---

## Tables and figures required

| ID | Type | Content | Status |
|---|---|---|---|
| T-N1 | Main table, replaces `tab:tinyllama_iid` | Five-method frontier, TinyLlama, Alpaca, IID, CUDA | Written, in G5.6 |
| F-N1 | Main figure, replaces `fig:pareto` | Frontier scatter with seed error bars, knee annotated | Written, in G5.4; generated by `make_frontier_figure.py` |
| T-N2 | Main table | Dolly-15k replication with explicit gap column | Written, in G6.2 |
| T-N3a/b | New Appendix E | Cross-backend summary and per-run detail | Written, in G13.2 |
| T-N4 | Appendix B | Superseded. G11 revises the existing `tab:b1` (to CUDA) and `tab:b2` (stays MPS, five seeds) instead of adding a separate variance table. | Superseded |

Figure 1 is replaced, not supplemented. The current three-method figure omits FFA-LoRA, the published baseline the method is compared against.

---

## Verified reference data

All values recomputed from the CSVs and asserted in `verify_numbers.py`. Sample sd.

### T-N1: TinyLlama, Alpaca, IID, CUDA, three seeds

| Method | Comm (MB) | Savings | Final loss | Held-out delta | p vs FLoRA |
|---|---:|---:|---:|---:|---:|
| FLoRA | 2578.125 | -- | 1.2608 +/- 0.0005 | -0.5992 +/- 0.0001 | -- |
| FedIT | 2578.125 | 0.00% | 1.2602 +/- 0.0016 | -0.5990 +/- 0.0002 | 0.588 |
| Two-Phase K=8 | 1863.125 | 27.73% | 1.2705 +/- 0.0006 | -0.5958 +/- <0.0001 | 4.2e-5 |
| ReverseAdaptive | 1533.125 | 40.53% | 1.2749 +/- 0.0006 | -0.5929 +/- 0.0008 | 1.1e-5 |
| FFA-LoRA | 983.125 | 61.87% | 1.3031 +/- 0.0012 | -0.5746 +/- 0.0004 | 1.7e-4 |

### Derived quantities

- RA vs FLoRA held-out gap: 0.006306. FFA vs RA held-out gap: 0.018220. FFA vs RA loss gap: 0.028246.
- FedIT vs FLoRA loss difference: 0.000566 (not significant).
- RA to FFA segment: 21.33 percentage points, 550 MB.
- Marginal efficiency ratio: 5.49, reported as 5.5x. Belongs in Section 5.2 (G7), not Section 4.2.
- Spread ratio for G5.7: 25.02.

### Paired t-tests, three seeds, CUDA

| Comparison | Final loss p | Held-out p |
|---|---:|---:|
| Two-Phase K=8 vs FLoRA | 4.18e-5 | 7.85e-5 |
| ReverseAdaptive vs FLoRA | 1.09e-5 | 6.23e-3 |
| ReverseAdaptive vs Two-Phase K=8 | 3.40e-4 | 2.62e-2 |
| FFA-LoRA vs ReverseAdaptive | 4.40e-4 | 3.81e-4 |
| FedIT vs FLoRA | 0.588 | 0.425 |

At n=3 across five comparisons a Bonferroni threshold is 0.01. The ReverseAdaptive versus Two-Phase held-out cell (0.0262) fails it. Decision: disclose and let it fail. Appendix C text must state that the failing cell is a secondary comparison under the knee framing, that the load-bearing comparison is ReverseAdaptive versus FFA-LoRA at p=3.8e-4 on both metrics, and that at n=3 the test has very low power so effect sizes should be read alongside p-values. Asserted as `ptest.bonferroni_failing_cell`.

### Dolly-15k gaps, for G6

| Method | Alpaca gap | Dolly gap | Difference | Relative |
|---|---:|---:|---:|---:|
| Two-Phase K=8 | 0.003399 | 0.004265 | 0.000866 | 25.5% |
| ReverseAdaptive | 0.006306 | 0.006118 | 0.000188 | 3.0% |

Claim only that ReverseAdaptive's quality cost is stable across datasets. Do not claim cross-dataset stability as a property of the family, and do not read Two-Phase's larger shift as a finding: with three seeds and two datasets there is no power to support differential stability either way.

Held-out deltas are comparable within a dataset only. Base losses: Alpaca 1.9352 (ppl 6.9254), Dolly 2.2608 (ppl 9.5905).

### LLaMA-3.2-3B, MPS

FLoRA 2625.0 MB (three clean runs; `final_results_table.csv` also holds six single-round smoke tests at 35.0 and 70.0 MB that a naive groupby includes, yielding 902 MB). ReverseAdaptive 1837.5 +/- 105.0, switch rounds 7, 8, 9. Two-Phase K=8 1942.5. Savings 30.0 +/- 4.0%. ReverseAdaptive versus Two-Phase p=0.9967.

### Step function

Switching methods: `total = 55(2s-1) + 928.125` for TinyLlama, `52.5(2s-1) + 1050` for 3B, where s is the switch round. Full-payload directions total 2s-1 because uploads are full for rounds 1..s and broadcasts for rounds 1..s-1. Granularity 110 MB per round (52.5 for 3B).

FFA-LoRA is s=1 (983.125), ReverseAdaptive s=6 (1533.125), Two-Phase K=8 s=9 (1863.125), Two-Phase K=10 s=11 (2083.125).

**FLoRA is NOT on this lattice.** A non-switching run has all 2N directions full, an even number, giving `55 x 30 + 928.125 = 2578.125`. Writing "FLoRA is the s=16 point" gives 2633.125, a 55 MB error. Guarded by `stepfn.FLoRA_not_on_switch_lattice`.

---

## Correction list additions

Add to `FEDLORA_PAPER_HANDOFF.md` section 7.2 (C1 through C12 already listed):

- **C13.** Table 1 must record `transition_rounds=0`. Defaults disagree across four code paths (`reverse_adaptive.py` 2, `server.py:61` 3, `server.py:97` 2, `run_experiment.py:632` 3). All configs set 0 explicitly so results are unaffected, but a reader reimplementing from the paper would omit it and silently obtain a delayed switch and different byte totals.
- **C14.** Section 4.2's claim that tau=0.002 and tau=0.001 reproduce Two-Phase K=8 and K=10 communication exactly is true on MPS and false on CUDA (1753.125 and 1973.125). Block in G5 (partial).
- **C15.** Algorithm 1 line 6 says `A_frozen` is cached from server A matrices. The code caches client 0's uploaded A. Affects Two-Phase and ReverseAdaptive equally (`two_phase.py:21,74` dispatches to the same aggregator). Blocks in G1b.
- **C16.** Algorithm 1 line 7 says broadcast is full state at r=K+1. That is symmetric and denies the one-round asymmetry the paper claims in five separate prose passages. Arithmetic: symmetric gives 1918.125 MB for Two-Phase K=8 against the measured 1863.125. Confirmed from artifacts: `exp_ffa_lora_iid` round 1 records 116.875 MB = 85.9375 full upload + 30.9375 B-only broadcast. Blocks in G1b.
- **C17.** Table 1's 3B dtype row describes the training dtype, not the transmitted payload. `base_config_llama3_3b.yaml:30` sets `lora_param_dtype: float32`, but `lora_model.py:181-186` casts uploads back to the base dtype and `element_size()` measures 2 bytes. A reader computing 3B communication from "float32 LoRA" gets 5250 MB against the reported 2625 MB. Revises the spec's instruction that the 3B half of the entry is correct. Blocks in G2 and G3.
- **C18.** Backend provenance is not declared per table. The revision puts main results on CUDA while the ablation and 3B stay on MPS. Table 5 reports 1.2736 for ReverseAdaptive tau=0.01 seed 42; the identical configuration on CUDA gives 1.2755. Blocks in G5.1 and G5.2.

Add to section 7.3:

- **C19.** The Section 5.2 heading reads "Why Adaptive Outperforms Fixed-$K$". At 1.1B ReverseAdaptive's final loss is 1.2749 against Two-Phase K=8's 1.2705, so it is significantly worse on quality (p=3.4e-4); at 3B they are indistinguishable. Adaptive does not outperform fixed-K on quality anywhere in the corpus, it sits at a different operating point. Heading replaced in G7. Guarded by `g7.RA_worse_than_TP8_at_1B`.

- **C20.** Section 5.3 states that the transition penalty's size varies with how many rounds remain in Phase 2. It does not. The penalty is exactly 55.0 MB regardless of switch round (naive totals 928.125 / 1478.125 / 1808.125 for s = 1, 6, 9 against measured 983.125 / 1533.125 / 1863.125). What varies is its share of the total. The same paragraph should quantify the look-ahead saving: a symmetric transition would cost 110.0 MB, so the asymmetry halves it, which is exactly what Algorithm 1 as printed denied (C16). Block in G8.

- **C21.** TinyLlama downstream benchmark accuracies are single-seed (seed 42, tag `stage3_checkpoint`, MPS corpus), while T-N1 reports the same configurations at three seeds on CUDA. The revised paper therefore states TinyLlama IID quality twice at different n and different backends, in the section whose job is comparing those two metrics. Same category as C18. Blocks in G8b.
- **C22.** The "cluster within 1.4 percentage points" figure in the Table 3 caption, Section 4.3, and the conclusion is computed including MMLU, which the same paragraph excludes as chance-level at 1.1B. Spreads on the three informative benchmarks are 0.006, 0.010, and 0.000, so the correct figure is 1.0 pp. Guarded by `g8b.cluster_pp_informative_only` and `g8b.cluster_pp_would_be_with_mmlu`.

- **C23.** All 123 runs record `git_dirty: true`, across 20 distinct commits, so no reported number is recoverable from a clean commit. The two FLoRA seed-42 runs in the corpus differ (round-1 loss 1.37814102 at commit `2d07cd9d` versus 1.37599655 at `c32967b2`) because they were executed from different code, not because of nondeterminism. Disclose in G13: the released snapshot is the final frozen state, individual runs were produced at earlier commits with local modifications, and the practical reproducibility evidence is the cross-backend replication of 31 comparable runs rather than re-execution at a pinned commit.
- **C24.** Section 5.4 advises that the wrapper can be added "with no risk of behavior change when $\tau$ is set conservatively." That is unachievable, not merely imprecise. Any round in which the loss fails to improve satisfies the criterion, so no non-negative $\tau$ leaves behavior unchanged: on the canonical FLoRA trajectory the loss rises at round 11, and $\tau{=}0$ fires there. Suppressing the switch requires $\tau$ below the most negative round-over-round relative change, -0.000903 in that run. The sanity baseline itself used $\tau{=}-1.0$ AND `warmup_rounds=999`, so it is evidence about disabling the wrapper, not about conservative $\tau$. Block in G9.

- **C25.** `body.tex:387` states "Byte counts are computed from parameter sizes." That is false and it is precisely the practice the paper exists to refute; `server.py:289` and `:343` use `p.numel() * p.element_size()` on transmitted tensors. A reviewer reaching Limitations after Section 5.3 concludes the measurement contribution is hollow. Block in G10.

- **C26.** The claim that federated methods on LLaMA-3.2-3B cluster "within 0.4 percentage points" understates the measured spread, which is 0.47 pp on HellaSwag. The figure appears in the Introduction headline numbers and the conclusion. Correct to 0.5. Guarded by `c26.l3_bench_spread_pp` and `c26.not_0p4`.

- **N5.** Held-out instruction-following metric definition (G4b). Blocks T-N1 and T-N2.
- **N6.** Conclusion rewrite (G14) and related-work forward references (G15).

---

## Neurocomputing submission requirements

Verified from the current guide for authors.

- Highlights: separate file, three to five bullets, each under 85 characters. Text in G0.4.
- Declarations required: data availability, competing interests, CRediT, ORCID, generative AI. Text in G0.5.
- Single-anonymized review. The paper is NOT anonymized; remove "anonymized supplementary material" and any TMLR anonymity artifacts.
- Scope clause: the journal does not accept submissions that are a pure combination of existing algorithms not directly relevant to neural networks or learning systems. A protocol that runs FLoRA then switches to FFA-LoRA is the shape that clause targets. The defense is that the contribution is the measurement and the frontier characterization, which is why the abstract leads with the Pareto and marginal-efficiency result rather than with the schedule.

---

# APPROVED BLOCKS

## G0. Format conversion, frontmatter, abstract, end matter (C9)

### G0.1 Preamble, replaces the TMLR preamble in `main.tex`

```latex
\documentclass[preprint,review,12pt]{elsarticle}

\input{math_commands.tex}

\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[colorlinks=true, linkcolor=blue, citecolor=blue, urlcolor=blue]{hyperref}
\usepackage{url}

\journal{Neurocomputing}

\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}
```

Remove `\usepackage{tmlr}`. elsarticle loads natbib itself, so `\citep` and `\citet` survive unchanged. Change `\bibliographystyle{tmlr}` to `\bibliographystyle{elsarticle-num}`. Keep `\input{appendix.tex}` after the bibliography, which is correct for Elsevier. The `review` option enables `lineno`, which interacts poorly with floating `algorithm` environments; if line numbers gap around Algorithms 1 and 2, wrap them in `\begin{linenomath}...\end{linenomath}` or switch those floats to `[H]`.

### G0.2 Frontmatter and abstract, replaces `\maketitle` and the abstract environment

```latex
\begin{document}
\raggedbottom

\begin{frontmatter}

\title{Adaptive Phase-Switching for Communication-Efficient Federated LoRA Fine-Tuning}

\author[ind]{Jerry Adams Franklin}
\ead{jerry.adamsf@gmail.com}
\address[ind]{Independent Researcher, Rochester, NY, USA}

\begin{abstract}
Federated fine-tuning of large language models with low-rank adaptation (LoRA)
reduces per-client trainable parameters, but client-to-server communication
remains the dominant cost. Existing accounting for federated LoRA protocols
omits the asymmetric transition round incurred when a protocol changes
aggregation mode, and reports savings that are insensitive to the
architecture-dependent shapes induced by grouped-query attention. This paper
measures per-round upload and download bytes directly for a bidirectional
B-only federated LoRA protocol, and uses that instrumentation to place five
published methods on a single communication-quality frontier. The measurements
show the frontier has a knee, and we operationalize it with ReverseAdaptive, a
schedule that locates that point by monitoring the relative improvement in
global training loss against a dimensionless threshold $\tau$ rather than by
fixing a phase boundary $K$ in advance. On
TinyLlama-1.1B-Chat with Alpaca, ReverseAdaptive attains 40.5\% measured
round-trip savings over FLoRA (1533.1 MB against 2578.1 MB, three seeds) at a
held-out instruction-following loss cost of 0.0063. It outperforms FFA-LoRA,
which freezes $A$ at initialization, by 0.0182 in held-out loss, more than
twenty times the largest per-method seed standard deviation on that metric,
indicating that
learning $A$ before freezing it produces materially better adapters. Moving
from ReverseAdaptive to FFA-LoRA buys a further 21.3 percentage points of
savings at roughly five times the quality cost per point. The quality cost of the transition is stable across
datasets (0.0061 on Dolly-15k against 0.0063 on Alpaca) and the protocol
yields $30.0 \pm 4.0$\% savings on LLaMA-3.2-3B, where ReverseAdaptive and
hand-tuned Two-Phase $K{=}8$ reach final losses differing by 0.000012, roughly
350 times less than the gap between them at 1.1B scale. Code, configurations, and scripts to
reproduce all figures and tables are publicly released.
\end{abstract}

\begin{keyword}
Federated learning \sep Low-rank adaptation \sep Communication efficiency \sep
Parameter-efficient fine-tuning \sep Large language models
\end{keyword}

\end{frontmatter}
```

268 words, inside Elsevier's 300-word guidance, no references, no em-dashes, no signposting.

Amendments applied INLINE above (do not re-apply): (a) the p=0.997 inversion replaced with effect-size reasoning, after G7 established that failing to reject at n=3 is not evidence of equivalence; (b) "largest per-method standard deviation" qualified with "on that metric", since the claim holds at 21.8x against held-out sds but only 11.3x against final-loss sds; (c) "5.5 times" changed to "roughly five times" per G8, because 5.5 depends on FFA-LoRA's measured 983.125 while the faithful floor of 928.125 gives 5.0; (d) sentence four reframed from "we introduce an aggregator that transitions from FLoRA to FFA-LoRA" to measurement-reveals-knee-then-operationalize, so the one sentence a Neurocomputing scope check would fixate on no longer states the combination baldly.

Changes from the TMLR abstract: "no measurable downstream cost" removed (C9); the four zero-shot benchmarks removed entirely, since Section 5.5 establishes they are insensitive and base TinyLlama outscores every checkpoint; prior-work claim narrowed to transition cost and GQA effects (C7); switch rule stated as relative improvement with dimensionless tau (C1); "anonymized supplementary material" replaced; the bit-identical no-switch sentence dropped because it needs the within-backend qualifier (C11) and does not survive compression.

Note on the "more than twenty times" phrasing: the handoff's "roughly 35 sigma" is a sigma count against a three-sample sd and reads as naive. The Welch test gives t = 42, p = 2e-6, which is stronger and defensible. Report the test, not the sigma count, in the body as well.

### G0.3 End matter, insert before `\bibliography{main}`

```latex
\section*{CRediT authorship contribution statement}
\textbf{Jerry Adams Franklin:} Conceptualization, Methodology, Software,
Validation, Formal analysis, Investigation, Data curation, Writing -- original
draft, Writing -- review and editing, Visualization.

\section*{Declaration of competing interest}
The author declares no known competing financial interests or personal
relationships that could have appeared to influence the work reported in this
paper.

\section*{Data availability}
All code, configuration files, and analysis scripts are publicly available at
https://github.com/jerryadamsfranklin/fedlora-protocols. Adapter checkpoints
are not released. The Alpaca and Dolly-15k datasets are publicly available from
their respective sources.

\section*{Declaration of generative AI and AI-assisted technologies in the writing process}
The author used Claude (Anthropic) for grammar and language editing only. All
technical content, analysis, and interpretations are the author's own.
```

ORCID 0009-0006-8470-8349 is entered in Editorial Manager, not in the LaTeX. Confirm the repository URL before submission.

### G0.4 Highlights, separate file

All five verified under 85 characters including spaces.

```
Byte-level measurement of federated LoRA communication, incl. transition cost  (77)
Phase switch fires on a relative loss plateau, not a hand-selected round K     (74)
40.5% measured communication savings for 0.0063 held-out loss on TinyLlama     (74)
Beats FFA-LoRA by 0.018 held-out loss: freezing A at init gives worse adapters (78)
Past this operating point, savings cost 5.5x more quality per point saved      (73)
```

---

## G1. Section 3.3 and Algorithm 2 (C1, C13)

### G1.1 Switch rule prose

`old_str`:

```
ReverseAdaptive replaces the fixed boundary $K$ with an adaptive loss-plateau signal. After a warmup period of $W$ rounds (default $W{=}5$), it monitors the per-round loss improvement $\Delta\ell_r = \ell_{r-1} - \ell_r$. When $\Delta\ell_r < \tau$ for the first time, the aggregator switches to FFA-LoRA \citep{sun2024ffalora} mode permanently.
```

`new_str`:

```
ReverseAdaptive replaces the fixed boundary $K$ with an adaptive loss-plateau signal. After a warmup period of $W$ rounds (default $W{=}5$), it monitors the \emph{relative} per-round loss improvement $\rho_r = (\ell_{r-1} - \ell_r) / \ell_{r-1}$. When $\rho_r < \tau$ for the first time, the aggregator switches to FFA-LoRA \citep{sun2024ffalora} mode permanently and instantaneously, with no intermediate phase. Because $\rho_r$ is a ratio of losses, $\tau$ is a dimensionless fraction rather than a loss difference, and its interpretation does not depend on the absolute scale of the loss for a given model or dataset; Section~\ref{sec:discussion} shows that this scale-free construction is what allows a single $\tau$ to transfer across model sizes without retuning.
```

The remainder of that paragraph (stability detector, 21-run count) is unchanged. "instantaneously, with no intermediate phase" is the C13 prose half.

### G1.2 Algorithm 2 switch condition

`old_str`:

```
\If{$m = \text{FLoRA}$ \textbf{and} $r > W$ \textbf{and} $(\ell_{r-1} - \ell_r) < \tau$}
```

`new_str`:

```
\State $\rho_r \leftarrow (\ell_{r-1} - \ell_r) \,/\, \ell_{r-1}$ \Comment{relative improvement; $\tau$ dimensionless}
\If{$m = \text{FLoRA}$ \textbf{and} $r > W$ \textbf{and} $\rho_r < \tau$}
```

No other line changes. The two branches are mutually exclusive on $m$, so the code's evaluation order (instability before plateau) is behaviorally identical to the printed order.

### G1.3 Section 4.5 clarification

`old_str`:

```
Table~\ref{tab:ablation} and Figure~\ref{fig:ablation} characterize how $\tau$ controls the savings-quality tradeoff.
```

`new_str`:

```
Table~\ref{tab:ablation} and Figure~\ref{fig:ablation} characterize how $\tau$ controls the savings-quality tradeoff. Since $\tau$ bounds a relative rather than an absolute loss improvement, the values below are fractions of the previous round's loss: $\tau{=}0.01$ requires the loss to fall by at least 1\% in a round for the FLoRA phase to continue.
```

### G1.4 Section 5.2 scale-free argument (SUPERSEDED BY G7)

Apply only if G7 has not yet landed.

`old_str`:

```
The cross-scale result validates that the plateau signal is tracking genuine training dynamics rather than an arbitrary threshold. Using the same $\tau{=}0.01$
```

`new_str`:

```
The cross-scale result follows from the criterion being scale-free. Because $\tau$ bounds a relative loss improvement, the same numerical value encodes the same stopping condition at any absolute loss level, so it carries across models whose losses differ in magnitude; an absolute improvement threshold would not have this property, since a fixed loss delta that plateaus one model's training would fire immediately or never on another. Using the same $\tau{=}0.01$
```

---

## G1b. Algorithm 1 and frozen-A provenance (C15, C16)

### G1b.1 Algorithm 1 transition branch

`old_str`:

```
\ElsIf{$r = K+1$} \Comment{transition: seeds $A_{\text{frozen}}$}
    \State upload $\leftarrow$ full state
    \State $A_{\text{frozen}} \leftarrow$ cache server $A$ matrices
    \State broadcast $\leftarrow$ full state
```

`new_str`:

```
\ElsIf{$r = K+1$} \Comment{transition: full upload seeds $A_{\text{frozen}}$}
    \State upload $\leftarrow$ full state $(A, B)$
    \State $A_{\text{frozen}} \leftarrow$ $A$ from the first client's uploaded state
    \State broadcast $\leftarrow$ $B$ only \Comment{clients retain $A$ from round $K$}
```

C15 and C16 are the same fact from two sides: upload must be full at K+1 because that is where the server harvests A, and broadcast can already be B-only because clients still hold the A received at round K.

### G1b.2 Frozen-A provenance disclosure

New paragraph in Section 3.4 Implementation Details, not Section 3.3, because it governs Two-Phase and ReverseAdaptive equally.

Verified timing: `reverse_adaptive.py:174-177` `get_freeze_a()` evaluates the upcoming round, so at round s the mode is still FLoRA and clients train A; `aggregate()` detects the plateau at line 152, `transition_rounds=0` flips the mode at 156-158, line 167 resets the FFA aggregator, line 169 captures `frozen_a` from `client_states[0]` at `ffa_lora.py:43`. Client ordering is deterministic: `server.py:269` iterates `self.clients` in fixed list order, assigned once at line 117.

```
\textbf{Frozen-$A$ provenance.} At the transition round the server seeds $A_{\text{frozen}}$ from the first client's uploaded state rather than from an average across clients. Because every client begins that round from the same broadcast global state, this is the aggregated global $A$ of the preceding round plus one epoch of local adaptation at learning rate $10^{-4}$. Client ordering is fixed for the duration of a run, so the choice is deterministic and reproducible. Two-Phase and ReverseAdaptive share this rule, since both dispatch to the same FFA-LoRA aggregator at their boundary. Averaging $A$ across clients at the transition round is a plausible alternative that this work does not test.
```

Matching limitation sentence goes in G10.

---

## G2. Table 1 (C2, C3, C8, C13, C17, N5)

Six row-level anchors rather than one table block, so each is independently reviewable.

### G2.1 Target modules

`old_str`:

```
LoRA target modules & \texttt{q\_proj}, \texttt{k\_proj}, \texttt{v\_proj}, \texttt{o\_proj} \\
```

`new_str`:

```
LoRA target modules & \texttt{q\_proj}, \texttt{v\_proj} \\
```

### G2.2 Dtype, split into two rows (C2 + C17 as a single edit)

`old_str`:

```
Model dtype & bfloat16 (TinyLlama); float16 base + float32 LoRA (LLaMA) \\
```

`new_str`:

```
Training dtype & float32 (TinyLlama-1.1B); float16 base + float32 LoRA (LLaMA-3.2-3B) \\
Transmitted dtype & float32 (TinyLlama-1.1B); float16 (LLaMA-3.2-3B) \\
```

Do C2 and C17 together. Patching the row twice would have the second edit fight the first. The transmitted row is the one byte counts depend on.

### G2.3 Hardware

`old_str`:

```
Hardware & Apple M4 Pro, 48\,GB unified memory, MPS \\
```

`new_str`:

```
Hardware (primary) & Apple M4 Pro, 48\,GB unified memory, MPS \\
Hardware (reproduction) & NVIDIA RTX 4090 (TinyLlama); A100 40\,GB (LLaMA-3.2-3B) \\
```

### G2.4 Held-out evaluation row (N5)

`old_str`:

```
Downstream eval & 500 examples/benchmark, seed 42, log-likelihood \\
```

`new_str`:

```
Downstream eval & 500 examples/benchmark, seed 42, log-likelihood \\
Held-out eval & 500 examples, train indices 3001--3500, seq.\ len.\ 256 \\
```

### G2.5 ReverseAdaptive defaults (C13)

`old_str`:

```
ReverseAdaptive defaults & \texttt{warmup\_rounds}=5, $\tau{=}0.01$, \texttt{stability\_threshold}=1.1 \\
```

`new_str`:

```
ReverseAdaptive defaults & \texttt{warmup\_rounds}=5, $\tau{=}0.01$, \texttt{stability\_threshold}=1.1, \\
                         & \texttt{transition\_rounds}=0 (instantaneous switch) \\
```

Split across two rows deliberately; the single-row version risks the overfull hbox already hit on the base-models row.

---

## G3. Sections 3.1 and 3.4 (C2, C3, C4, C17)

### G3.1 B-only fraction and target list

`old_str`:

```
For TinyLlama-1.1B with rank $r{=}16$ targeting $\{\texttt{q\_proj}, \texttt{k\_proj}, \texttt{v\_proj}, \texttt{o\_proj}\}$ layers, accounting for GQA in the key/value projections, the B-only fraction of the full state is approximately 36\% rather than the 50\% a naive parameter count would suggest.
```

`new_str`:

```
For TinyLlama-1.1B with rank $r{=}16$ targeting $\{\texttt{q\_proj}, \texttt{v\_proj}\}$, the projections adapted in the original LoRA formulation \citep{hu2022lora}, accounting for GQA in the value projection, the B-only fraction of the full state is exactly 36\% rather than the 50\% a naive parameter count would suggest. This fraction is invariant to the size of the target set, since \texttt{q\_proj} and \texttt{o\_proj} share a shape and \texttt{k\_proj} and \texttt{v\_proj} share a shape under GQA; the ratio is fixed by the GQA configuration, not by how many projections are adapted.
```

The invariance clause preempts a reviewer computing the fraction for a four-module configuration, finding it unchanged, and reading that as an error. Arithmetic: two targets 36864/102400, four targets 73728/204800, both exactly 36.0%. 3B q/v is exactly 40.0%.

### G3.2 Byte tracking dtype

`old_str`:

```
Byte counts use bfloat16 (2 bytes/param) for TinyLlama, float16 base with float32 LoRA for LLaMA-3.2-3B.
```

`new_str`:

```
Byte counts use the dtype in which parameters are transmitted: float32 (4 bytes/param) for TinyLlama-1.1B and float16 (2 bytes/param) for LLaMA-3.2-3B, whose LoRA parameters are trained in float32 and cast back to the base dtype for transmission. A full TinyLlama LoRA state is 2{,}252{,}800 parameters over 22 layers, or 9{,}011{,}200 bytes per client per direction.
```

The explicit byte count forecloses the ambiguity that four-target bfloat16 would produce the same total. The transmitted-dtype clause is C17.

### G3.3 Partition mechanics (C4)

`old_str`:

```
Non-IID partitions use Dirichlet label-skew with $\alpha \in \{0.5, 0.1\}$.
```

`new_str`:

```
Alpaca and Dolly-15k carry no class labels, so non-IID partitions apply Dirichlet skew over a task-diversity proxy: each instruction $s$ is assigned to one of ten buckets by character length, $\min(\lfloor |s| / 50 \rfloor, 9)$, and a Dirichlet distribution with concentration $\alpha \in \{0.5, 0.1\}$ is applied over those buckets. This induces heterogeneity in task form rather than task semantics; Section~\ref{sec:limitations} notes that semantic heterogeneity is untested.
```

Verified against `run_experiment.py:542`, `min(len(instruction) // 50, 9)`. Character length, not tokens. The following sentence about empty partitions and Appendix B survives unchanged. Grep for residual "label skew" elsewhere, including Appendix A and C.

Note: the spec also asks for a module-list change in the Section 3.4 GQA paragraph, but that paragraph contains no module list. No edit needed there.

---

## G4b. Held-out instruction-following metric (N5)

Insert immediately before the Section 4.2 subsection heading. Verified from the holdout CSVs: `split=train`, `heldout_start=3000`, `heldout_examples=500`, `max_seq_length=256`.

```latex
\subsection{Held-Out Instruction-Following Metric}
\label{sec:heldout}
Final training loss measures fit to the federated objective, and the four zero-shot benchmarks do not discriminate between base and fine-tuned checkpoints at this scale (Section~\ref{sec:baseregression}). Quality is therefore also reported as held-out instruction-following loss. From each dataset we reserve the 500-example slice \texttt{train[3000:3500]}, disjoint from the 3000 examples partitioned across clients, and compute mean token-level cross-entropy under the prompt format described in Section~\ref{sec:setup}. Each adapted checkpoint is scored against the unadapted base model, and we report $\Delta\ell_{\mathrm{held}} = \ell_{\mathrm{tuned}} - \ell_{\mathrm{base}}$, so more negative values indicate better instruction following. Base-model reference values are 1.9352 (perplexity 6.9254) on Alpaca and 2.2608 (perplexity 9.5905) on Dolly-15k. Because the base values differ by dataset, held-out deltas are comparable within a dataset but not across datasets; the cross-dataset comparison in Section~\ref{sec:dolly} therefore compares between-method gaps rather than levels.
```

Requires adding `\label{sec:baseregression}` to `\subsection{The Base-Model Regression Observation}`, which currently has no label.

---

## G4. Section 4.1 compute, baselines, prompt format (C8, C6)

### G4.1 Compute statement

`old_str`:

```
Table~\ref{tab:setup} summarizes the experimental configuration. Total compute was approximately 285 hours on Apple M4 Pro hardware; no external GPU was used. Both TinyLlama-1.1B and LLaMA-3.2-3B experiments use three seeds for IID training.
```

`new_str`:

```
Table~\ref{tab:setup} summarizes the experimental configuration. The primary corpus of 34 runs was produced on a single Apple M4 Pro workstation, approximately 285 hours of compute, with no institutional cluster. Those runs were subsequently reproduced on commercially rented NVIDIA hardware, an RTX 4090 for TinyLlama-1.1B and an A100 40\,GB for LLaMA-3.2-3B; the cross-backend comparison is reported in Appendix~\ref{app:crossbackend}. The Dolly-15k replication and the FFA-LoRA and FedIT baseline runs were produced on the same rented hardware. Both TinyLlama-1.1B and LLaMA-3.2-3B experiments use three seeds for IID training.
```

### G4.2 Baseline introduction (C6)

New paragraph at the end of Section 4.1, after the Table 1 float.

Verification chain: `server.py:249` sets `freeze_a` from the aggregation method, passed at `server.py:279`; `client.py` maps `freeze_a=True` to `freeze_ratio=1.0`; `_apply_partial_freeze(1.0)` sets `requires_grad=False` on every A parameter; client optimizers are built only from parameters with `requires_grad`.

```latex
\textbf{Baselines.} FFA-LoRA \citep{sun2024ffalora} and FedIT \citep{zhang2024federatedgpt} are reimplemented within this codebase rather than obtained as released code from the original authors, and results for them should be read with that qualification. Following Sun et al., the FFA-LoRA implementation freezes $A$ at initialization: client-side $A$ parameters are set non-trainable before local optimization begins and are excluded from the optimizer parameter group, so clients train only $B$ from the first round. Clients transmit $B$ only, a measured 3{,}244{,}032 bytes per client per direction against 9{,}011{,}200 for full $A{+}B$ transmission. The former is $(32768 + 4096) \times 22$ parameters at 4 bytes each, exactly the B-only payload, and could not arise if $A$ were being transmitted. FedIT applies FedAvg independently to $A$ and $B$ and transmits full state in both directions, so its communication volume matches FLoRA's by construction.
```

Keep the explicit parameter arithmetic. `ffa_lora.py:get_communication_cost` is dead code whose comment claims float16 round-trip, and a reviewer reading it would otherwise conclude the payload is half this.

### G4.3 Prompt format

New paragraph in Section 4.1. The training template at `client.py:99` uses only `instruction` and `output` and drops Alpaca's `input` field, populated in roughly 40 percent of examples. Training and held-out evaluation share the code path, so comparisons are internally consistent, but the manuscript never specifies the format and a reader reproducing from it would use the standard two-template Alpaca prompt and obtain different absolute losses.

```latex
\textbf{Prompt format.} All examples, in training and in held-out evaluation, are rendered with a single template: \texttt{\#\#\# Instruction:} followed by the instruction text, then \texttt{\#\#\# Response:} followed by the output, truncated and padded to 256 tokens. Alpaca's optional \texttt{input} field, populated in roughly 40\% of examples, is not included, and no separate template is applied to examples that carry it. Loss is computed over the full formatted sequence including prompt tokens, with padding masked to $-100$. This differs from the two-template formatting used in the original Alpaca release and shifts absolute loss values; because training and evaluation share the same code path, comparisons between methods are unaffected. Note that if the tokenizer aliases \texttt{pad\_token} to \texttt{eos\_token}, the end-of-sequence token is masked along with padding; this is consistent across all runs but is relevant to anyone recomputing the base-model reference values of Section~\ref{sec:heldout}.
```

Section 4.1 needs `\label{sec:setup}` for G4b's cross-reference.

---

## G15. Related work forward references (N6, part)

Both depend on G5 defining `\label{sec:frontier}`.

`old_str`:

```
Averaging independently introduces aggregation error that grows with client heterogeneity.
```

`new_str`:

```
Averaging independently introduces aggregation error that grows with client heterogeneity. FedIT is evaluated directly in Section~\ref{sec:frontier}, where it also serves as a check on the byte accounting: transmitting the same payload as FLoRA every round, it should match FLoRA's measured communication exactly.
```

`old_str`:

```
recovering expressivity at the cost of longer initial overhead.
```

`new_str`:

```
recovering expressivity at the cost of longer initial overhead. FFA-LoRA is evaluated directly in Section~\ref{sec:frontier} as the maximum-savings endpoint of the communication-quality frontier, where the cost of freezing $A$ at initialization rather than after a FLoRA phase is quantified.
```

The FLoRA paragraph already ends with a baseline signal and needs no edit.

---

## G5. The communication-quality frontier (N1, C14, C18)

### G5.1 Backend provenance (C18), new paragraph in Section 4.1

```latex
\textbf{Backend provenance.} Communication totals are protocol-deterministic and identical across backends. Loss values are not: Appendix~\ref{app:crossbackend} shows cross-backend agreement within 0.01 for IID settings, which exceeds some of the between-method differences reported below. Tables are therefore labelled with their backend. The five-method frontier (Table~\ref{tab:frontier}), the Dolly-15k replication (Table~\ref{tab:dolly}), and all statistical tests use the CUDA corpus, since the FFA-LoRA, FedIT, and Dolly runs exist only there. The threshold ablation (Table~\ref{tab:ablation}) and the LLaMA-3.2-3B results use the MPS corpus. Communication values are comparable across tables; loss values are comparable only within a backend.
```

### G5.2 Ablation caption

`old_str`:

```
\caption{ReverseAdaptive threshold sensitivity (TinyLlama IID, seed 42).}
```

`new_str`:

```
\caption{ReverseAdaptive threshold sensitivity (TinyLlama IID, seed 42, MPS corpus). Communication is backend-invariant; loss values are not comparable with the CUDA results in Table~\ref{tab:frontier}.}
```

### G5.3 Section heading and opening paragraph

`old_str` is the `\subsection{Communication-Quality Pareto Curve}` line plus the paragraph beginning `Table~\ref{tab:tinyllama_iid} and Figure~\ref{fig:pareto} show`, through `final loss 1.2746.`

`new_str`:

```latex
\subsection{The Communication-Quality Frontier}
\label{sec:frontier}
Table~\ref{tab:frontier} and Figure~\ref{fig:frontier} place five published federated LoRA protocols on a single measured communication-quality frontier for TinyLlama-1.1B \citep{zhang2024tinyllama} under IID partitioning. The five methods occupy four distinct operating points. FLoRA \citep{wang2024flora} and FedIT \citep{zhang2024federatedgpt} coincide at 2578.13 MB, since both transmit full $A$ and $B$ in both directions every round. Two-Phase $K{=}8$ reaches 1863.13 MB (27.73\% savings), ReverseAdaptive at $\tau{=}0.01$ reaches 1533.13 MB (40.53\%), and FFA-LoRA \citep{sun2024ffalora} reaches 983.13 MB (61.87\%). Across the four distinct points communication falls and both quality metrics degrade in the same order, so the frontier is monotone on each metric independently rather than only on the metric used to construct it.
```

### G5.4 Figure block, replaces the whole `fig:pareto` float

```latex
\begin{figure}[ht]
\begin{center}
\includegraphics[width=0.85\linewidth]{figures/frontier.pdf}
\end{center}
\caption{Measured communication-quality frontier on Alpaca-3k, TinyLlama-1.1B, IID, CUDA corpus. Five published protocols at four distinct operating points; error bars are one sample standard deviation over three seeds, and communication is identical across seeds. Held-out $\Delta$loss is tuned minus base on the 500-example slice defined in Section~\ref{sec:heldout}; more negative is better. FLoRA and FedIT coincide at upper right. The segment from ReverseAdaptive to FFA-LoRA is markedly steeper than the segment preceding it, which Section~\ref{sec:discussion} quantifies.}
\label{fig:frontier}
\end{figure}
```

### G5.5 Interpretation paragraphs

The spec's point 2 asked for "both metrics agree on the ordering exactly." They do not: FedIT is best on final loss and FLoRA best on held-out. Corrected below. Guarded by `g5.metric_ranking_agrees_on_distinct_points`.

```latex
Three features of Table~\ref{tab:frontier} bear on the interpretation. The two quality metrics agree on the ordering of every pair they can resolve: across the four distinct operating points the ranking is identical on final training loss and on held-out instruction-following loss. The exception is the FLoRA and FedIT pair, which shares an operating point, where FedIT is nominally better on final loss and FLoRA nominally better on held-out loss; neither difference is statistically distinguishable ($p{=}0.588$ and $p{=}0.425$). The shape of the frontier is not an artifact of the metric chosen to construct it.

ReverseAdaptive outperforms FFA-LoRA by 0.0282 on final loss and 0.0182 on held-out loss, against per-method seed standard deviations of at most 0.0016 and 0.0008 respectively. Both protocols end training with $A$ frozen and only $B$ aggregated, and they differ only in whether $A$ was learned first. Learning $A$ through a FLoRA phase before freezing it produces materially better adapters than freezing $A$ at initialization, at a cost of 550 MB in this configuration.

FedIT lands on FLoRA's communication volume exactly, to the byte, and within 0.0006 on both quality metrics. It contributes no new operating point, which is the intended result: FedIT is an independently implemented full-state protocol, so its coincidence with FLoRA is evidence that the byte accounting measures the protocol rather than an implementation artifact.
```

Do not present the marginal-efficiency argument here. It belongs in Section 5.2 (G7).

### G5.6 T-N1, replaces the whole `tab:tinyllama_iid` float

```latex
\begin{table}[t]
\caption{Measured communication-quality frontier. TinyLlama-1.1B, Alpaca-3k, IID, CUDA corpus, three seeds, mean $\pm$ one sample standard deviation. Communication is protocol-deterministic and identical across seeds. Held-out $\Delta$loss is tuned minus base on the 500-example slice of Section~\ref{sec:heldout}; more negative is better. $p$-values are paired $t$-tests against FLoRA on final loss; the full matrix on both metrics is in Appendix~\ref{app:stats}.}
\label{tab:frontier}
\begin{center}
\begin{tabular}{lccccc}
\toprule
Method & Total comm.\ (MB) & Savings & Final loss & Held-out $\Delta$loss & $p$ vs.\ FLoRA \\
\midrule
FLoRA \citep{wang2024flora}         & 2578.13 & --      & $1.2608 \pm 0.0005$ & $-0.5992 \pm 0.0001$ & -- \\
FedIT \citep{zhang2024federatedgpt} & 2578.13 & 0.00\%  & $1.2602 \pm 0.0016$ & $-0.5990 \pm 0.0002$ & $0.588$ \\
Two-Phase $K{=}8$                   & 1863.13 & 27.73\% & $1.2705 \pm 0.0006$ & $-0.5958 \pm {<}0.0001$ & $4.2 \times 10^{-5}$ \\
ReverseAdaptive ($\tau{=}0.01$)     & 1533.13 & 40.53\% & $1.2749 \pm 0.0006$ & $-0.5929 \pm 0.0008$ & $1.1 \times 10^{-5}$ \\
FFA-LoRA \citep{sun2024ffalora}     & \phantom{0}983.13 & 61.87\% & $1.3031 \pm 0.0012$ & $-0.5746 \pm 0.0004$ & $1.7 \times 10^{-4}$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

Two-Phase K=10 leaves the main table and survives in the ablation at tau=0.001, same 2083.13 MB, same MPS corpus.

### G5.7 Statistical paragraph

`old_str`:

```
Loss differences between methods are small in absolute terms (0.014 between FLoRA and ReverseAdaptive at $\tau{=}0.01$) but statistically distinguishable across three seeds (paired $t$-tests, $p < 10^{-4}$). Full statistical tests are in Appendix~\ref{app:stats}.
```

`new_str`:

```
Loss differences between methods are small in absolute terms, 0.0141 between FLoRA and ReverseAdaptive at $\tau{=}0.01$, but are resolved across three seeds because the per-seed spread is roughly twenty-five times smaller. Paired $t$-tests on final loss give $p < 10^{-4}$ for every comparison against FLoRA except FedIT, which is correctly indistinguishable. With three seeds these tests have very low power, so effect sizes should be read alongside the $p$-values, and no multiplicity correction is applied in Table~\ref{tab:frontier}; Appendix~\ref{app:stats} reports the full matrix on both metrics with the correction stated.
```

### G5.8 C14, exact-reproduction claim

`old_str`:

```
This frontier sweeps through the fixed-$K$ Two-Phase operating points: $\tau{=}0.002$ reproduces Two-Phase $K{=}8$ communication exactly, and $\tau{=}0.001$ reproduces $K{=}10$. The adaptive method recovers the full fixed-$K$ Pareto frontier without requiring $K$ to be specified.
```

`new_str`:

```
This frontier sweeps through the fixed-$K$ Two-Phase operating points: on the MPS corpus, $\tau{=}0.002$ lands on Two-Phase $K{=}8$ communication and $\tau{=}0.001$ on $K{=}10$. Because total communication is a step function of the discrete switch round, at 110\,MB per round, this correspondence is not backend-invariant; Appendix~\ref{app:crossbackend} reports that both tight-$\tau$ settings fire one round earlier on CUDA. The robust claim is that the adaptive method traverses the same frontier as the fixed-$K$ family without requiring $K$ to be specified, rather than that it reproduces particular $K$ values byte for byte.
```

Depends on G13 defining `app:crossbackend`.

---

## G12. Appendix A gradient clipping (C5)

Verified directly: `client.py:256`, `clip_grad_norm_(trainable_params, max_norm=1.0)`, inside the `if (step + 1) % gradient_accumulation_steps == 0` gate, so it runs at every optimizer step unconditionally. `AdamW` is constructed with no `betas` argument, so the paper's 0.9/0.999 is correct by PyTorch default, and no scheduler is ever constructed, so the "no learning rate schedule" clause stands.

`old_str`:

```
No learning rate schedule or gradient clipping applied.
```

`new_str`:

```
No learning rate schedule is applied. Gradients are clipped to a maximum $\ell_2$ norm of 1.0 at every optimizer step.
```

---

## G6. Dolly-15k replication (N2)

Status: APPROVED. All values verified, sample sd, asserted in `verify_numbers.py` (`tn2.*`, `ptest.dolly.*`). Defines `\label{sec:dolly}`, referenced by G4b.

Design decision: the gap to each dataset's own FLoRA baseline is an explicit column, because held-out levels are not comparable across datasets (base losses 1.9352 versus 2.2608) and leaving the reader to subtract invites the wrong comparison.

Scope discipline: claim only that ReverseAdaptive's own cost is stable (3.0% relative shift). Two-Phase shifts 25.5%, and the text explicitly declines to read that as differential transferability, because three seeds on two datasets cannot resolve a stability difference between two methods.

### G6.1 New subsection, insert after the Section 4.2 frontier material

```latex
\subsection{Replication on Dolly-15k}
\label{sec:dolly}
Table~\ref{tab:dolly} repeats the three-method comparison on Databricks Dolly-15k \citep{dolly2023}, an instruction dataset with a different length distribution and a broader task mix than Alpaca. Communication is identical to the Alpaca corpus, since the protocols are dataset-independent and ReverseAdaptive fired at round 6 on every seed of both datasets. The quantity of interest is therefore the quality cost of switching rather than the absolute loss.

Held-out deltas are not comparable across datasets, since the base model scores 1.9352 on the Alpaca slice and 2.2608 on the Dolly slice. Table~\ref{tab:dolly} reports the gap to each dataset's own FLoRA baseline, which is comparable. ReverseAdaptive costs 0.0063 held-out loss on Alpaca and 0.0061 on Dolly, a shift of 0.0002, or 3.0\% of the gap's own magnitude. The quality cost of the adaptive transition is stable across these two datasets.

Two-Phase $K{=}8$ costs 0.0034 on Alpaca and 0.0043 on Dolly, a shift of 25.5\%. We do not read this as evidence that fixed-$K$ switching transfers less well than adaptive switching. Three seeds on two datasets cannot resolve a difference in stability between two methods, and the claim supported here is limited to ReverseAdaptive's own cost being stable, not to a property of the protocol family.

A non-IID Dolly run at $\alpha{=}0.5$ switched at round 6 on all three seeds and reached the same 1533.13 MB, matching the IID behaviour on both datasets. No Dolly FLoRA baseline was run under heterogeneity, so no quality cost is reported for that configuration.
```

### G6.2 T-N2

```latex
\begin{table}[t]
\caption{Dolly-15k replication. TinyLlama-1.1B, IID, CUDA corpus, three seeds, mean $\pm$ one sample standard deviation. Gap is the held-out $\Delta$loss difference against the same dataset's FLoRA baseline, which is the quantity comparable across datasets; held-out levels are not, since base-model losses differ (1.9352 on Alpaca, 2.2608 on Dolly-15k).}
\label{tab:dolly}
\begin{center}
\begin{tabular}{lccccc}
\toprule
Method & Comm.\ (MB) & Final loss & Held-out $\Delta$loss & Gap (Dolly) & Gap (Alpaca) \\
\midrule
FLoRA \citep{wang2024flora}     & 2578.13 & $1.6544 \pm 0.0021$ & $-0.5330 \pm 0.0010$ & -- & -- \\
Two-Phase $K{=}8$               & 1863.13 & $1.6643 \pm 0.0021$ & $-0.5287 \pm 0.0010$ & 0.0043 & 0.0034 \\
ReverseAdaptive ($\tau{=}0.01$) & 1533.13 & $1.6686 \pm 0.0022$ & $-0.5268 \pm 0.0005$ & 0.0061 & 0.0063 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

Paired tests on Dolly, for Appendix C: Two-Phase versus FLoRA p = 7.8e-5 (loss) and 2.4e-6 (held-out); ReverseAdaptive versus FLoRA p = 2.5e-5 and 7.6e-3.

### G6.3 Bibliography entry

None exists in `main.bib`. Transcribed rather than re-fetched; spot-check the author list against the Databricks post before submission.

```bibtex
@misc{dolly2023,
  author       = {Mike Conover and Matt Hayes and Ankit Mathur and Jianwei Xie and Jun Wan and Sam Shah and Ali Ghodsi and Patrick Wendell and Matei Zaharia and Reynold Xin},
  title        = {Free Dolly: Introducing the World's First Truly Open Instruction-Tuned {LLM}},
  year         = {2023},
  howpublished = {\url{https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm}}
}
```

---

## G7. Section 5.2, choosing an operating point (C10, C19, N1 reframe)

Status: APPROVED. Supersedes the G1.4 partial; do not apply both. Asserted as `g7.*`.

**C18 risk resolved with positive evidence.** `PHASE_1_BATCH_VERIFY.md` runs 8, 9, 10 (`exp_reverse_adaptive_iid`, three seeds) and runs 29, 30, 31 (`exp_llama3_reverse_adaptive_iid`, three seeds) all PASS with zero failures, including the `switch_round_exact` hard-match check. Both scales agree on switch round across MPS and CUDA, so the cross-scale claim does not rest on backend-divergent corpora. Stated affirmatively in the block rather than left implicit.

**Changes made and their reasons.**
- Heading replaced (C19): adaptive does not outperform fixed-K on quality.
- "the most practically significant finding in this paper" and "the strongest possible empirical evidence" removed (C10).
- The p=0.997 reasoning inverted. The original text says the p-value "confirms statistical indistinguishability". Failing to reject a null at n=3 is not evidence of equivalence, and a reviewer with statistical training would treat that phrasing as disqualifying. Replaced with an effect-size statement: the 3B difference (0.000012) is roughly 350 times smaller than the 1.1B difference (0.0043), with an explicit note that low power means a large p-value cannot establish absence.
- Round-6 saturation surfaced here rather than six pages downstream in Section 4.5, and stated as behavioral equivalence to a fixed K=5, which is what the arithmetic gives (switch round s corresponds to K = s-1).
- "a range of model sizes" becomes "the two scales tested".
- The knee and marginal-efficiency argument arrives last and carries the section (N1). Per-point costs: 1.56e-4 for the FLoRA-to-ReverseAdaptive segment, 8.54e-4 for the ReverseAdaptive-to-FFA-LoRA segment, ratio 5.49.

### G7.1 Full replacement

Replace from `\subsection{Why Adaptive Outperforms Fixed-$K$}` through the paragraph ending `...careful manual configuration would identify.`

```latex
\subsection{Choosing an Operating Point}
The Two-Phase protocol requires a practitioner to select $K$ before training begins, without knowing when the loss plateau will occur for a given model-dataset combination. This is a hyperparameter that must be tuned per dataset, per model, and potentially per client population: precisely the kind of manual configuration that slows deployment in production federated systems. ReverseAdaptive replaces it by monitoring the signal that $K$ was meant to approximate.

The criterion transfers across model scales because it is scale-free. Since $\tau$ bounds a relative loss improvement, the same numerical value encodes the same stopping condition at any absolute loss level, so it carries across models whose losses differ in magnitude; an absolute improvement threshold would not, since a fixed loss delta that plateaus one model's training would fire immediately or never on another. Using the same $\tau{=}0.01$, ReverseAdaptive fires at round 6 on TinyLlama-1.1B and at a mean of round $8.0 \pm 1.0$ on LLaMA-3.2-3B, with no scale-specific retuning. Both sets of switch rounds match exactly across the MPS and CUDA backends (Appendix~\ref{app:crossbackend}), so the cross-scale comparison does not rest on corpora that disagree about when the switch occurred.

Two qualifications bound this result. At TinyLlama scale the switch fires at round 6 in every run, across both partition settings and both datasets, and round 6 is the first round after the warmup period $W{=}5$. At $\tau{=}0.01$ and this scale the adaptive rule is behaviorally equivalent to a fixed $K{=}5$, and its adaptivity is visible only in the threshold ablation (Section~\ref{sec:ablation}), where $\tau \leq 0.002$ yields switch rounds of 9 and 11. The evidence that the criterion tracks training dynamics rather than the warmup boundary comes from LLaMA-3.2-3B, where three runs, differing in seed and produced at different code revisions, give switch rounds 7, 8, and 9. Two model sizes and two datasets do not establish that $\tau{=}0.01$ is a universal default; they establish that a single value carried across the two scales tested without retuning.

At 3B the adaptive and hand-tuned configurations reach final losses differing by 0.000012, roughly 350 times smaller than the 0.0043 that separates them at 1.1B. A paired $t$-test over three seeds gives $p = 0.997$. We report this as a statement about effect size rather than as evidence of equivalence: with three seeds the test has almost no power to detect a difference of the magnitude observed at 1.1B, so a large $p$-value cannot establish that none exists.

Where ReverseAdaptive sits on the frontier matters more than which fixed $K$ it happens to match. Moving from FLoRA to ReverseAdaptive buys 40.5 percentage points of communication savings at a held-out cost of 0.0063, or $1.56 \times 10^{-4}$ per point. Moving onward to FFA-LoRA buys a further 21.3 points at a cost of 0.0182, or $8.54 \times 10^{-4}$ per point, 5.5 times more expensive per point saved. The frontier has a knee and ReverseAdaptive sits at it. A practitioner who takes the first tranche of savings and declines the second is making the trade the measured data supports, and the value of the adaptive rule is that it locates that point without $K$ being guessed in advance.
```

---

## G8. Section 5.3, measured bytes and parameter counts (C7, C20)

Status: APPROVED. Asserted as `g8.*`. Citation keys verified against `neuro_new_refs.bib`: `yan2026fedsrd`, `ramesh2026florist`.

**The FFA-LoRA overcharge, and why it is disclosed rather than corrected.** FFA-LoRA freezes A at initialization, so every client's A equals the shared initial value at every round and never needs transmitting. The measured 983.125 MB nonetheless includes the 55.0 MB seeding upload, because the server harvests A from `client_states[0]` rather than deriving it from the initialization. The faithful floor is 928.125 MB.

| | Measured | Faithful floor |
|---|---:|---:|
| FFA-LoRA total | 983.125 MB | 928.125 MB |
| Savings vs FLoRA | 61.87% | 64.00% |
| RA-to-FFA segment | 21.33 pp | 23.47 pp |
| Marginal rate | 8.54e-4 | 7.76e-4 |
| Headline ratio | 5.49x | 4.99x |

Decision: report 983.125, since the paper's contribution is measured rather than idealized bytes, and disclose the artifact explicitly with the faithful floor and the recomputed ratio. FFA-LoRA is the baseline the central claim beats, so an unfair 55 MB charge against it is the first thing a hostile reviewer looks for, and the original authors would notice immediately. Disclosing converts the most attackable number into evidence that the baseline was understood well enough to find where this implementation overcharges it, and the headline survives.

**Extension: the overcharge is not unique to FFA-LoRA.** For ReverseAdaptive and Two-Phase the server already holds a usable A, namely the global aggregate from round s-1 that it broadcast at that round. Freezing that value instead of harvesting client 0's post-training copy would eliminate the seeding upload for those methods too. The 55 MB is a property of the harvest-from-client-zero rule, not of phase-switching. Consequence for framing: "prior accounting omits a cost this implementation incurs" is defensible; "phase-switching protocols intrinsically incur a transition cost" is not. G8.1 is written the first way. Pairs with G1b.2 and the G10 limitation.

**Abstract and conclusion wording.** Both carry "5.5 times". Since a faithful FFA-LoRA gives 5.0, quote "roughly five times" at the summary sites, which is honest under either reading, and give exact figures in Sections 4.2, 5.2, and 5.3 where there is room to explain. Guarded by `g8.roughly_five_both_ways`.

Revised abstract sentence:

```latex
Moving from ReverseAdaptive to FFA-LoRA buys a further 21.3 percentage points of
savings at roughly five times the quality cost per point, placing ReverseAdaptive at
the knee of the frontier.
```

### G8.1 Full Section 5.3 replacement

Replace from `\subsection{Measured Bytes vs.\ Theoretical Parameter Counts}` through the paragraph ending `...honest reporting in future federated LoRA work.`

```latex
\subsection{Measured Bytes and Theoretical Parameter Counts}
Federated LoRA work commonly reports communication savings as parameter-count ratios \citep{zhang2024federatedgpt,sun2024ffalora,wang2024flora,bai2024flexlora}, conflating how many parameters exist with how many bytes a protocol transmits. The practice is not universal: FedSRD \citep{yan2026fedsrd} and FLoRIST \citep{ramesh2026florist} both report measured per-round communication in megabytes. The contribution here is therefore not measurement as such, but the isolation of two quantities that a parameter ratio cannot express and that aggregate megabyte totals do not separate.

Architectural choices such as GQA fix the B-only fraction in ways a naive parameter count does not anticipate. TinyLlama-1.1B's GQA makes B-only transmission save exactly 36\% rather than the 50\% a non-GQA count suggests, and LLaMA-3.2-3B's different GQA configuration yields exactly 40\%. A theoretical 50\% savings claim made for one model family does not transfer to another without architectural accounting.

The transition round carries a cost that parameter-ratio analyses omit. At the FLoRA-to-FFA-LoRA boundary the upload is full while the broadcast is already B-only, so exactly one directional payload is upgraded from B-only to full. On TinyLlama this costs 55.0 MB, and the figure does not depend on the switch round: totals that ignore the transition would be 928.1, 1478.1, and 1808.1 MB for switch rounds 1, 6, and 9, against measured totals of 983.1, 1533.1, and 1863.1. What varies with the number of remaining rounds is the penalty's share of the total, not its size. A protocol upgrading both directions at the boundary would pay 110.0 MB; the look-ahead broadcast halves it.

Total communication across the entire corpus is one closed-form expression in one discrete parameter. For a switching protocol with switch round $s$ over $N$ rounds, uploads are full for rounds $1 \ldots s$ and broadcasts for rounds $1 \ldots s{-}1$, so $2s-1$ directional payloads are full and the total is $55(2s-1) + 928.125$ MB on TinyLlama and $52.5(2s-1) + 1050$ MB on LLaMA-3.2-3B. FFA-LoRA is $s{=}1$, ReverseAdaptive $s{=}6$, Two-Phase $K{=}8$ is $s{=}9$, and Two-Phase $K{=}10$ is $s{=}11$; every measured total in this paper lands on that lattice to the byte. The $2s-1$ arises from the harvest rule rather than from the protocol: an implementation that froze the server's existing global $A$ would upgrade no directional payload at the boundary and give $110(s-1) + 928.125$ instead. A non-switching protocol has all $2N$ directional payloads full and is not a point on the lattice: FLoRA is the separate endpoint at $55 \times 30 + 928.125 = 2578.125$ MB. The lattice spacing of 110 MB per round is also why no continuous $\tau$ reproduces a fixed-$K$ total robustly, as Section~\ref{sec:frontier} notes.

One artifact of this implementation deserves disclosure, because it charges the strongest baseline too much. FFA-LoRA freezes $A$ at initialization, so every client's $A$ equals the shared initial value at every round and never needs transmitting. The measured 983.1 MB nonetheless includes the 55.0 MB seeding upload, because the server harvests $A$ from the first client's uploaded state rather than deriving it from the initialization (Section~\ref{sec:implementation}). An implementation faithful to the design reaches the floor of 928.1 MB, raising FFA-LoRA's savings from 61.87\% to 64.00\%, widening the ReverseAdaptive-to-FFA-LoRA segment from 21.3 to 23.5 percentage points, and moving the marginal cost ratio from 5.5 to 5.0. The knee conclusion is unchanged. The same harvest rule governs ReverseAdaptive and Two-Phase, where the seeding upload is likewise avoidable in principle, since the server already holds the aggregated global $A$ it broadcast in the preceding round; freezing that value instead is an untested alternative recorded in Section~\ref{sec:limitations}.
```

### G8.2 Missing label

Add `\label{sec:implementation}` to the Section 3.4 Implementation Details heading, referenced by G8.1. No such label currently exists.

### G8.3 Correction to G6

RESOLVED AND APPLIED INLINE. The Dolly citation key is `dolly2023`; G6.1 and G6.2 now use it. Retained here as a record only.

---

## G8b. Section 5.5, base-model regression (N4, C21, C22)

Status: APPROVED. Asserted as `g8b.*`. Defines `\label{sec:baseregression}`, referenced by G4b.

**Attribution fix.** The existing text quotes -2.8, -6.4, -2.8 as though representative. Those are FLoRA's figures. Verified ranges across methods: ARC-Easy -2.2 to -2.8, BoolQ -6.4 to -7.4, HellaSwag -2.8 uniformly.

**Framing risk avoided.** The held-out slice is drawn from the training distribution, so it measures instruction following on unseen examples, not generalization to unrelated tasks. Section 5.5 is exactly where an over-enthusiastic sentence would claim otherwise; the block states the scope explicitly.

### G8b.1 Full subsection replacement

Replace both paragraphs of `\subsection{The Base-Model Regression Observation}`.

```latex
\subsection{The Base-Model Regression Observation}
\label{sec:baseregression}
Federated Alpaca-3k \citep{taori2023alpaca} instruction tuning regresses TinyLlama-1.1B \citep{zhang2024tinyllama} on all three informative zero-shot benchmarks: ARC-Easy by 2.2 to 2.8 percentage points depending on method, BoolQ by 6.4 to 7.4, and HellaSwag by 2.8 uniformly. At LLaMA-3.2-3B \citep{dubey2024llama3} the same procedure is flat-to-positive on all four benchmarks. The cross-scale contrast is consistent with a model-capacity explanation: TinyLlama-1.1B may lack the parameter budget to absorb instruction tuning without distributional drift on these benchmarks, though a single dataset and two scales cannot rule out other explanations.

The same procedure improves substantially on the objective it optimizes. On the held-out Alpaca slice the base model scores 1.9352 (perplexity 6.9254) and every fine-tuned checkpoint scores between 1.3349 and 1.3608 (perplexity 3.80 to 3.90). The regression is a divergence between the training objective and these benchmarks rather than a training failure, and it is why quality is reported here as held-out instruction-following loss. The held-out slice is drawn from the same distribution as the training data, so it measures instruction following on unseen examples and is not evidence of generalization to unrelated tasks.

The regression is independent of aggregation method. Cross-method spread at TinyLlama scale is 0.006 on ARC-Easy, 0.010 on BoolQ, and 0.000 on HellaSwag, so the zero-shot benchmarks do not separate protocols that the held-out metric separates at $p < 10^{-4}$. These accuracies are single-seed (seed 42, MPS corpus), unlike the three-seed CUDA results in Table~\ref{tab:frontier}, so the spreads carry no variance estimate and indicate insensitivity rather than establishing equivalence.

The BoolQ result admits a specific account. The base model's accuracy of 0.626 closely matches BoolQ's natural yes-class rate of approximately 0.62, suggesting the base exploits a yes-biased prior. Instruction tuning attenuates that prior, which is why BoolQ moves furthest of the three.
```

### G8b.2 Table 3 caption (C21, C22)

`old_str`:

```
Federated methods cluster within 1.4 percentage points at TinyLlama-1.1B and 0.4 percentage points at LLaMA-3.2-3B.}
```

`new_str`:

```
Federated methods cluster within 1.0 percentage points at TinyLlama-1.1B, across the three informative benchmarks, and 0.4 percentage points at LLaMA-3.2-3B. TinyLlama accuracies are single-seed and from the MPS corpus; loss values elsewhere in this paper for the same configurations are three-seed and from the CUDA corpus.}
```

Section 4.3 body text and the conclusion carry the same 1.4 pp figure and need the same correction.

### G8.4 Lattice clause (APPLIED INLINE to G8.1, record only)

Insert after `...lands on that lattice to the byte.`:

```latex
The $2s-1$ arises from the harvest rule rather than from the protocol: an implementation that froze the server's existing global $A$ would upgrade no directional payload at the boundary and give $110(s-1) + 928.125$ instead.
```

Placed immediately before the disclosure paragraph, so the explanation arrives where the question does.


## G9. Section 5.4, the reproducibility guarantee (C11, C24)

Status: APPROVED. Asserted as `g9.*`.

**The guarantee is real and verified directly.** The no-switch sanity run (`stage2_no_switch_sanity`, tau=-1.0, warmup 999) and the canonical FLoRA run (`run_07_of_18`) agree at round 1 (1.37814102) and round 15 (1.25943428). The wrapper is inert when disabled.

**Constraint retained.** The no-switch run does not appear in `PHASE_1_BATCH_VERIFY.md`, so there is no cross-backend measurement of it. The claim is within-backend; do not promise a cross-backend delta.

**C24 evidence.** Reconstructing the switch rule from the canonical FLoRA trajectory reproduces Table 5 exactly: tau=0.005 and 0.01 give round 6, tau=0.002 gives 9, tau=0.001 gives 11. That reconstruction is itself a strong internal-consistency check, since it confirms ReverseAdaptive tracks FLoRA exactly before switching, which Section 4.3 asserts but never quantifies. With the rule validated, tau=0 gives round 11 and tau=-0.001 gives no switch. The most negative round-over-round relative change is -0.000903.

**Trajectory provenance matters.** Two FLoRA seed-42 MPS runs exist and only `results/raw/exp_flora_iid/flora/seed_42/run_07_of_18/20260428_153342` reproduces the published ablation. The other (`stage3_checkpoint`, commit `c32967b2`) gives 6, 8, 10. The verifier pins the correct one by path in a comment.

### G9.1 Full replacement of `body.tex:371`

```latex
The no-switch sanity baseline deserves specific comment. Running ReverseAdaptive with switching disabled reproduces the matched FLoRA \citep{wang2024flora} run to 10 decimal places at every round, establishing that the wrapper is a zero-cost modification when unused: it rules out numerical artifacts from the wrapper's presence, not merely from its activation. The comparison is within a single backend and a single execution environment. Agreement across hardware is weaker than bit-identity and is reported separately in Appendix~\ref{app:crossbackend}.

Disabling the switch requires care. Setting $\tau$ to a small positive value does not achieve it: any round in which the training loss fails to improve satisfies $\rho_r < \tau$, so the criterion fires. On the FLoRA trajectory underlying the threshold ablation the loss rises once, at round 11, so $\tau{=}0$ still triggers a switch at that round, and suppressing the switch entirely requires $\tau$ below the most negative round-over-round relative change observed, approximately $-0.0009$ here. The sanity baseline accordingly uses $\tau{=}-1.0$ together with a warmup exceeding the round budget. A practitioner wanting the wrapper present but inert should disable it by one of those means rather than by choosing a conservative threshold.
```

The conclusion repeats the bit-identical claim and receives the same within-backend qualifier in G14.

---

## G10. Limitations (C3, C4, C15, C23, C25, N4)

Status: APPROVED. Full-section replacement, since five corrections land in it and three existing paragraphs are false. Defines `\label{sec:limitations}`, referenced by G3.3 and G8.1.

**Commit provenance.** `neuro_part12_qvonly` is single-revision (`cae4c328`) and `phase1_cuda_rerun` is single-revision (`7b957752`), but T-N1 spans both: FLoRA, Two-Phase K=8 and ReverseAdaptive from the latter, FedIT and FFA-LoRA from the former. T-N2 is entirely `cae4c328` and is clean. The FLoRA/FedIT pair straddles the boundary and is statistically indistinguishable (p=0.588 loss, p=0.425 held-out, difference 0.000566), which indicates the combined algorithmic and revision difference across that boundary is roughly 0.0006 against a ReverseAdaptive-to-FFA-LoRA gap of 0.0282. The FedIT row therefore serves three purposes: published baseline, byte-accounting check, and cross-revision control. Asserted as `c23.*`.

**Not a strict bound.** FLoRA and FedIT are different algorithms, so their observed difference is algorithmic plus revision effect combined and could in principle partially cancel. No pure measurement exists for that pair, since FLoRA, Two-Phase and ReverseAdaptive were not rerun in the `qvonly` batch. The rigorous half is the byte invariance recorded under G13.

**3B is the confounded corpus.** Seeds 42, 123, 456 were produced at `d6f90388`, `e67fa6b0` and `6de806bb`, so every 3B plus-minus mixes seed and revision variation, including 1837.5 +/- 105.0, 1.3938 +/- 0.0036, and the 30.0 +/- 4.0% appearing in the abstract, Introduction, Table 6 and conclusion.

**The MPS corpus was worse and the CUDA-primary decision removes it.** The original Table 2 compared FLoRA (`2d07cd9d`), Two-Phase K=8 (`c1ed2a15`) and ReverseAdaptive (`1e518c63`), three methods at three revisions. Stated in G13 as a positive property of the reported corpus.

**The threshold ablation is clean**: all `stage2_threshold_*` tags share `1e518c63` with the MPS ReverseAdaptive run.

**Cross-revision variation is commit-pair dependent.** The tau reconstruction predicted the ablation exactly using a FLoRA trajectory at `2d07cd9d` against ablation runs at `1e518c63`, so those revisions produce identical dynamics to the precision that matters. The two FLoRA seed-42 runs at `2d07cd9d` and `c32967b2` diverge by 0.0021 from round 1. Neither uniformly negligible nor uniformly serious, which is why G10 states it per corpus rather than as a blanket disclaimer.

### G10.1 Full replacement

Replace everything from `\section{Limitations}` through the future-work paragraph.

```latex
\section{Limitations}
\label{sec:limitations}

The experiments use two instruction-following datasets (Alpaca-3k, \citealt{taori2023alpaca}, and Dolly-15k, \citealt{dolly2023}) and a single task type. Generalization to classification tasks, longer training runs, or generation-quality metrics is not characterized. Quality is measured as held-out loss on examples drawn from the same distribution as the training data, which captures instruction following on unseen examples rather than transfer to unrelated tasks.

The two published baselines, FFA-LoRA and FedIT, were evaluated on Alpaca only, so Table~\ref{tab:dolly} compares three protocols where Table~\ref{tab:frontier} compares five. The cross-dataset result therefore speaks to the cost of the adaptive transition rather than to the stability of the whole frontier across datasets.

The four zero-shot benchmarks do not discriminate between aggregation protocols at TinyLlama scale, where cross-method spread is at most 1.0 percentage point and every fine-tuned checkpoint scores below the base model (Section~\ref{sec:baseregression}). Conclusions about relative protocol quality therefore rest on held-out instruction-following loss alone.

The simulation assumes full client participation in every round. Real federated deployments typically involve partial participation, where only a fraction of clients respond per round. The effect of partial participation on transition-round timing and on ReverseAdaptive's plateau detection is not studied here.

Non-IID partitions are constructed by Dirichlet skew over instruction-length buckets rather than over semantic labels, since neither dataset carries class labels. This induces heterogeneity in task form. Heterogeneity in task semantics, which is what label-skew partitioning models in classification settings, is untested.

The study covers only Llama-style decoder-only architectures with LoRA targeting \texttt{q\_proj} and \texttt{v\_proj}. Other architectures, encoder-decoder models, larger target sets, or heterogeneous rank configurations may produce different B-only fractions and different savings profiles. The reported percentages are specific to this target set and to the grouped-query attention configurations of the two models tested.

At the transition round the server freezes $A$ from the first client's uploaded state. Freezing the aggregated global $A$ that the server already holds from the preceding round would avoid the seeding upload entirely and is untested, so the 55.0 MB transition cost reported here is a property of this implementation rather than of phase-switching protocols in general (Section~\ref{sec:implementation}).

Runs were produced across multiple code revisions with uncommitted local modifications. The primary CUDA corpus spans two revisions, and the FLoRA and FedIT rows of Table~\ref{tab:frontier} sit on either side of that boundary while remaining statistically indistinguishable, which indicates that the combined algorithmic and revision difference across that boundary is roughly 0.0006 in final loss. The LLaMA-3.2-3B results are not single-revision: the three seeds were produced at three revisions, so the variability reported there combines seed and environment variation.

LLaMA-3.2-3B experiments use three random seeds with downstream evaluation per checkpoint. Seed-to-seed evaluation noise within a single checkpoint is not separately characterized. At TinyLlama-1.1B, downstream evaluation is single-seed per checkpoint while training uses three seeds.

No real-world network conditions are simulated. Byte counts are measured from the tensors actually transmitted, but network latency, packet loss, and bandwidth constraints are not modeled, so wall-clock communication time is not predicted.

No differential privacy analysis is included. FFA-LoRA \citep{sun2024ffalora} was originally motivated in part by DP-friendliness; the transition-round dynamics of the bidirectional B-only protocol may affect DP accounting in non-obvious ways that warrant separate study.

Future work should address partial participation, non-Llama architectures, semantic heterogeneity, and real network conditions to validate deployment claims more broadly. Composing the bidirectional B-only protocol with gradient compression techniques \citep{alistarh2017qsgd,lin2017deep} for savings beyond the B-only floor is a natural extension. Per-layer adaptive phase-switching, where different transformer layers switch at different rounds based on their individual loss-plateau signals, is another direction.
```

### G10.2 Amendment to the approved G7 block (APPLIED INLINE, record only)

`old_str`:

```
where three seeds give switch rounds 7, 8, and 9.
```

`new_str`:

```
where three runs, differing in seed and produced at different code revisions, give switch rounds 7, 8, and 9.
```

The conclusion survives: all three fire after warmup, so the criterion is not pinned to round 6. Only the attribution to seed alone was unsupported.

---

## G11. Appendix B, Tables B1 and B2 (C4, C12, C18)

Status: APPROVED. Asserted as `g11.*`.

**The two tables have different jobs and take different corpora.** C18's principle is "declare the backend per table", not "every table must be CUDA".

- `tab:b1` is per-seed backing for a method comparison, so it follows T-N1's corpus and moves to CUDA. Leaving it on MPS would have the appendix disagree with the main table it supports (FLoRA 1.259434 against 1.2608).
- `tab:b2` documents partition sensitivity rather than comparing methods, so the larger MPS seed set is retained. Five seeds at alpha=0.1 spanning 1.2592 to 1.3481 is the strongest single piece of evidence for the variance finding, and all five share commit `1e518c63`, so B2 is single-revision where B1 is not. Dropping 40% of it to satisfy a convention that exists for a different reason is a bad trade, and it would remove seeds relative to the TMLR submission, which may be publicly visible on OpenReview.

**Two-Phase K=10 leaves `tab:b1`**, having no CUDA runs and having already left the main table in G5. It survives in the ablation at tau=0.001.

**Active-client derivation.** The alpha=0.1 totals of 1533.125, 1379.8125 and 1226.50 are exactly 10/10, 9/10 and 8/10 of the full value, corresponding to seeds with 10, 9 and 8 active clients after empty Dirichlet partitions. Stated in the caption rather than left unexplained.

### G11.1 Replace the whole `tab:b1` float

```latex
\begin{table}[H]
\caption{Per-seed TinyLlama-1.1B IID results, CUDA corpus, backing Table~\ref{tab:frontier}. Communication is deterministic across seeds. $\dagger$ For Two-Phase the column shows the predetermined phase boundary $K$, not an adaptive switch event; for FFA-LoRA, $A$ is frozen from initialization; for ReverseAdaptive it shows the round at which the plateau signal triggered.}
\label{tab:b1}
\begin{center}
\small
\begin{tabular}{llccc}
\toprule
Method & Seed & Final loss & Total MB & Switch / boundary$^\dagger$ \\
\midrule
FLoRA & 42 & 1.261408 & 2578.13 & -- \\
FLoRA & 123 & 1.260479 & 2578.13 & -- \\
FLoRA & 456 & 1.260500 & 2578.13 & -- \\
FedIT & 42 & 1.260767 & 2578.13 & -- \\
FedIT & 123 & 1.258420 & 2578.13 & -- \\
FedIT & 456 & 1.261503 & 2578.13 & -- \\
Two-Phase $K{=}8$ & 42 & 1.271276 & 1863.13 & $K{=}8^\dagger$ \\
Two-Phase $K{=}8$ & 123 & 1.270130 & 1863.13 & $K{=}8^\dagger$ \\
Two-Phase $K{=}8$ & 456 & 1.270237 & 1863.13 & $K{=}8^\dagger$ \\
ReverseAdaptive ($\tau{=}0.01$) & 42 & 1.275514 & 1533.13 & 6 \\
ReverseAdaptive ($\tau{=}0.01$) & 123 & 1.274610 & 1533.13 & 6 \\
ReverseAdaptive ($\tau{=}0.01$) & 456 & 1.274481 & 1533.13 & 6 \\
FFA-LoRA & 42 & 1.303975 & \phantom{0}983.13 & init$^\dagger$ \\
FFA-LoRA & 123 & 1.301738 & \phantom{0}983.13 & init$^\dagger$ \\
FFA-LoRA & 456 & 1.303628 & \phantom{0}983.13 & init$^\dagger$ \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

### G11.2 Table B2 caption

`old_str`:

```
\caption{Non-IID results. Seeds 42 and 1000 at $\alpha{=}0.1$ had 9 and 8 active clients due to empty Dirichlet partitions.}
```

`new_str`:

```
\caption{Non-IID results, MPS corpus, five seeds at $\alpha{=}0.1$. This table documents partition sensitivity rather than comparing methods, so the larger MPS seed set is retained; Table~\ref{tab:b1} follows the CUDA corpus because it backs a method comparison. Seeds 42 and 1000 at $\alpha{=}0.1$ had 9 and 8 active clients due to empty Dirichlet partitions, which scales communication to $9/10$ and $8/10$ of the full value (1379.81 and 1226.50 MB against 1533.13).}
```

### G11.3 Appendix B opening

Insert after `\label{app:perseed}`:

```latex
Tables in this appendix are labelled with their corpus. Per-seed results backing a method comparison use the CUDA corpus, matching the main results tables; per-seed results documenting partition sensitivity retain the larger MPS seed set, where five seeds at $\alpha{=}0.1$ provide better evidence than three.
```

### G11.4 Non-IID variance finding (re-homed from the superseded T-N4)

T-N4 was superseded by revising `tab:b1` and `tab:b2`, but the variance result it carried had no manuscript text. Without it the non-IID results read as "the partitioning did nothing", which is the first question a reviewer asks. Append to the G11.3 paragraph:

```latex
At $\alpha{=}0.5$, mean final loss is statistically indistinguishable from IID for every method, with differences of 0.001 to 0.007 against per-seed standard deviations near 0.02 over three seeds. Heterogeneity manifests instead as sensitivity to the partition draw: seed-to-seed standard deviation rises by roughly an order of magnitude, from $0.5$ to $1.6 \times 10^{-3}$ under IID to $18$ to $25 \times 10^{-3}$ at $\alpha{=}0.5$, which is also why the per-seed spread at $\alpha{=}0.1$ in Table~\ref{tab:b2} is wide.
```

Verified sample-sd ratios, Alpaca final loss: FLoRA 47x, FedIT 12x, Two-Phase K=8 40x, ReverseAdaptive 44x, FFA-LoRA 15x. Asserted as `g11.variance_ratio.*` and `g11.noniid_mean_indistinguishable.*`.

---

## G13. Appendix D and new Appendix E (C8, N3, C23)

Status: APPROVED. Asserted as `g13.*`. Defines `\label{app:crossbackend}`, referenced by G4.1, G5.1, G5.8 and G9.

**Sources:** `docs/PHASE_1_BATCH_VERIFY.md` (34-run table, per-run failure detail), `docs/PHASE_1_BATCH_VERIFY.csv`, `docs/PHASE_1_17_DIAGNOSTIC.md`.

**Denominator.** 34 pairs, 31 comparable, 27 agreeing, 4 disagreeing, 3 with no valid MPS reference. Runs 23, 24, 25 (Two-Phase K=8 at alpha=0.5) recorded `switch_round: mps=None` and 2578.125 MB, the full FLoRA volume, because those MPS executions predate the bidirectional B-only implementation. Reporting them inside a pass/fail denominator would invite a reviewer to ask what 79% cross-backend agreement means.

**Framing, sharpened.** Every cross-backend communication disagreement is exactly one 110 MB step of the switch-round lattice, and both the MPS and CUDA values are lattice points: run 11 is s=11 against s=10, run 12 is s=9 against s=8, run 17 is s=6 against s=7. The backends never disagree about the protocol, only about which round the criterion fires. Two of the four disagreements are IID at tight tau, two are alpha=0.1, so this is boundary sensitivity rather than heterogeneity sensitivity. Run 16 breached only the consecutive-spike rule with no communication or switch-round difference.

**Positive results worth stating.** Switch round matched exactly for every IID run at tau=0.01 and for all three LLaMA-3.2-3B runs, which is what supports the G7 cross-scale claim.

### G13.1 Appendix D, replace the four checklist entries

Removes "released as anonymized supplementary material" and "No external GPU was used", both TMLR artifacts now false.

```latex
\textbf{Code:} Full source code is publicly released, including all configuration YAML files, training scripts, evaluation scripts, and scripts to reproduce every figure and table from the analysis CSVs.

\textbf{Seeds:} All seeds listed in Table~\ref{tab:setup}. Per-seed results are in Appendix~\ref{app:perseed}.

\textbf{Hardware:} The primary corpus of 34 runs was produced on a single Apple M4 Pro workstation with 48\,GB unified memory using the MPS backend, approximately 285 hours of compute, with no institutional cluster. Those runs were re-executed on commercially rented NVIDIA hardware, an RTX 4090 for TinyLlama-1.1B and an A100 40\,GB for LLaMA-3.2-3B; the comparison is in Appendix~\ref{app:crossbackend}. The Dolly-15k replication and the FFA-LoRA and FedIT baselines were produced on the rented hardware only.

\textbf{Data:} Alpaca \citep{taori2023alpaca} via \texttt{tatsu-lab/alpaca} on HuggingFace, 3000-sample subset, and Dolly-15k \citep{dolly2023} via \texttt{databricks/databricks-dolly-15k}, matched subset. Adapter checkpoints are not released; reproduction scripts are sufficient.
```

### G13.2 New Appendix E

```latex
\section{Cross-Backend Verification}
\label{app:crossbackend}

The primary corpus was produced on the MPS backend and subsequently re-executed on CUDA. Table~\ref{tab:n3a} summarizes the comparison; Table~\ref{tab:n3b} details every disagreement.

Tolerances are setting-aware. IID runs require per-round training loss to agree within $0.01$ absolute. Non-IID runs allow $0.025$, since heterogeneous partitions amplify per-round variation, but additionally require the switch round to match exactly and permit no more than two consecutive rounds exceeding $0.01$. Communication must agree within $0.01$ in all settings, which is effectively exact since totals are protocol-deterministic.

Of 34 run pairs, 31 have a valid MPS reference. The three Two-Phase $K{=}8$ runs at $\alpha{=}0.5$ do not: their MPS executions predate the bidirectional B-only implementation and never switched, recording 2578.13 MB, the full FLoRA volume. Those are absent references rather than disagreements, and CUDA is the source of truth for that configuration. Of the 31 comparable pairs, 27 agree within tolerance and 4 do not.

All four disagreements are the same phenomenon. Total communication is a step function of the discrete switch round with 110\,MB granularity (Section~\ref{sec:discussion}), and every observed cross-backend communication difference is exactly one step: the MPS and CUDA totals in Table~\ref{tab:n3b} are adjacent points on the same lattice. The backends do not disagree about the protocol, only about which round the plateau criterion fires, and only where that decision sits near a boundary. Two of the four are IID runs at tight thresholds ($\tau{=}0.001$ and $\tau{=}0.002$), where the criterion is nearly satisfied in either of two adjacent rounds; two are $\alpha{=}0.1$ runs, where heterogeneity produces a noisier loss trajectory. This is boundary sensitivity rather than heterogeneity sensitivity.

Run 17 was investigated separately. Re-executing on CUDA with the data partition seed fixed at 456 and the training seed changed to 999 reproduced switch round 7 exactly, establishing that the one-round difference is a stable property of the backend and partition rather than run-to-run stochasticity. At $\alpha{=}0.1$, cross-backend agreement on the discrete switch round is therefore not guaranteed: one of three partitions fired one round later on CUDA, and CUDA is reported as primary for that setting.

The switch round matched exactly for every IID run at $\tau{=}0.01$ and for all three LLaMA-3.2-3B runs, which is what supports the cross-scale transfer claim in Section~\ref{sec:discussion}.

\textbf{Byte accounting.} Communication is instrumented at the transport layer, counting \texttt{numel} times \texttt{element\_size} on the tensors actually transmitted, rather than derived from parameter counts. The \texttt{get\_communication\_cost} helper in the FFA-LoRA aggregator is not used by this accounting.

\textbf{Code revisions.} Runs were produced across multiple code revisions with uncommitted working-tree modifications, so no reported number is recoverable by checking out a single commit. Reproducibility rests instead on the cross-backend replication documented here. The byte accounting is revision-invariant by direct measurement: FLoRA, Two-Phase $K{=}8$ and ReverseAdaptive were each executed under two different revisions within the CUDA corpus and produced identical totals of 2578.13, 1863.13 and 1533.13\,MB. Loss values carry revision uncertainty; the FLoRA and FedIT rows of Table~\ref{tab:frontier} span that boundary and differ by 0.0006, against a ReverseAdaptive-to-FFA-LoRA separation of 0.0282.

\begin{table}[H]
\caption{Cross-backend verification summary, MPS against CUDA, 34 run pairs.}
\label{tab:n3a}
\begin{center}
\begin{tabular}{lc}
\toprule
Category & Runs \\
\midrule
Agreed within setting-aware tolerance & 27 \\
Disagreed & \phantom{0}4 \\
\midrule
Comparable pairs & 31 \\
No valid MPS reference (CUDA reported) & \phantom{0}3 \\
\midrule
Total & 34 \\
\bottomrule
\end{tabular}
\end{center}
\end{table}

\begin{table}[H]
\caption{Cross-backend disagreements. Every communication difference is exactly one 110\,MB step of the switch-round lattice. Run 16 breached only the consecutive-spike rule, with no communication or switch-round difference.}
\label{tab:n3b}
\begin{center}
\small
\begin{tabular}{llccl}
\toprule
Run & Configuration & MPS (MB) & CUDA (MB) & Failing check \\
\midrule
11 & $\tau{=}0.001$, IID & 2083.13 & 1973.13 & communication, one step ($s{=}11$ vs $10$) \\
12 & $\tau{=}0.002$, IID & 1863.13 & 1753.13 & communication, one step ($s{=}9$ vs $8$) \\
16 & $\alpha{=}0.1$, seed 123 & 1533.13 & 1533.13 & 3 consecutive rounds $|\Delta|>0.01$ \\
17 & $\alpha{=}0.1$, seed 456 & 1533.13 & 1643.13 & communication, switch round 6 vs 7, loss \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

## G14. Conclusion (N6, C9, C11, C14, C22, C26)

Status: APPROVED. Written last so every number is final. Asserted as `g14.*`, `c26.*`.

**Eight problems in the existing text:** "measured savings slightly lower than theoretical" needs C20's fixed-55 MB framing; "recovering the full fixed-$K$ Pareto frontier" needs C14; "the paired $t$-test confirms statistical indistinguishability ($p{=}0.997$)" is the fifth and last site of the inversion; "zero-cost scale transfer" overclaims from two scales; "cluster within 1.4 percentage points" is C22; "no detectable downstream cost" is C9, already removed from the abstract; the bit-identical claim needs C11's within-backend qualifier; "anonymized supplementary material" is a TMLR artifact. The conclusion also never mentioned the frontier, the knee, FFA-LoRA, FedIT, or Dolly, all of which are now central.

### G14.1 Replace all body paragraphs and the release line

```latex
This paper introduced a bidirectional B-only protocol and the ReverseAdaptive aggregator for communication-efficient federated LoRA fine-tuning. The central methodological contribution is a shift from theoretical parameter-count savings to measured byte-level reporting. Tracking per-round upload and download megabytes explicitly, including at the asymmetric transition boundary, shows that the transition costs a fixed 55.0\,MB on TinyLlama-1.1B regardless of when it occurs, a quantity parameter-count accounting cannot express.

Measured byte tracking places five published protocols on a single communication-quality frontier at four distinct operating points. Two-Phase $K{=}8$ saves 27.7\% over FLoRA \citep{wang2024flora} on TinyLlama-1.1B \citep{zhang2024tinyllama}, ReverseAdaptive saves 40.5\%, and FFA-LoRA \citep{sun2024ffalora} saves 61.9\%. The frontier has a knee: moving from FLoRA to ReverseAdaptive costs 0.0063 held-out instruction-following loss for 40.5 points of savings, while continuing to FFA-LoRA costs a further 0.0182 for only 21.3 more points, roughly five times more expensive per point saved. ReverseAdaptive sits at that knee and locates it without $K$ being specified in advance.

The quality cost of the adaptive transition is stable across datasets, 0.0063 on Alpaca against 0.0061 on Dolly-15k. Scale validation at LLaMA-3.2-3B \citep{dubey2024llama3} gives $30.0 \pm 4.0$\% savings, and the same $\tau{=}0.01$ shifts the mean switch point from round 6 to $8.0 \pm 1.0$ without retuning. At 3B the adaptive and hand-tuned configurations differ in final loss by 0.000012, roughly 350 times less than the 0.0043 separating them at 1.1B; with three seeds a paired $t$-test cannot establish equivalence, so this is reported as an effect size rather than as a null result.

The four zero-shot benchmarks do not discriminate between protocols at TinyLlama scale, where every fine-tuned checkpoint scores below the base model and cross-method spread is at most 1.0 percentage point, so quality conclusions rest on held-out instruction-following loss. Running ReverseAdaptive with switching disabled reproduces the matched FLoRA run to 10 decimal places within a single backend, establishing that the adaptive wrapper introduces no perturbation when unused.

As open-weight LLMs \citep{dubey2024llama3,zhang2024tinyllama} continue to grow in capability and federated deployments expand into privacy-sensitive domains such as healthcare, finance, and on-device applications, the communication bottleneck identified by \citet{kairouz2021advances} will only intensify. Protocols that adapt to training dynamics rather than requiring hand-tuned round counts are better positioned for these settings, and reporting what such protocols actually transmit, rather than what a parameter count predicts, is a prerequisite for comparing them.

Code, configuration files, and scripts to regenerate all figures and tables from raw results are publicly released.
```

### G14.2 Delete the acknowledgments block

```
\subsubsection*{Acknowledgments}
Acknowledgments suppressed for double-blind review.
```

Delete entirely. Neurocomputing is single-anonymized and the author is known to reviewers.

---

## G16. Section 1, Introduction (C1, C7, C14, C20, C26)

Status: APPROVED. Written after G14 so contribution claims match final numbers. NEWLY IDENTIFIED group: no group in the original work order covered the Introduction.

### G16.1 Prior-work paragraph (C7)

Replace from `All of these methods report` through `None uses an adaptive signal to select when to transition.`

```latex
These four report communication savings as parameter-count ratios rather than measured bytes; more recent work does measure transmitted volume directly \citep{yan2026fedsrd,ramesh2026florist}. What remains uncharacterized is the asymmetric round that occurs when a protocol transitions between aggregation modes, and the use of an adaptive signal to select when that transition should happen.
```

### G16.2 Contribution one (C20)

"clients must upload full state" asserted a necessity. G8 established it is the harvest rule. The asymmetry direction was also stated backwards: download savings begin first.

```latex
\textbf{Bidirectional B-only protocol with measured byte tracking.}
The protocol implements both upload and download B-only transmission for Two-Phase and ReverseAdaptive aggregators. Per-round upload and download megabytes are recorded from the tensors actually transmitted. The transition round, in which the server seeds its frozen $A$ from client uploads, is accounted for explicitly, producing a one-round asymmetry between when download savings begin and when upload savings begin.
```

### G16.3 Contribution two (C1, C14)

```latex
\textbf{ReverseAdaptive: adaptive phase-switching aggregator.}
The aggregator starts in FLoRA mode and monitors the relative per-round improvement in global training loss. When that relative improvement falls below a dimensionless threshold $\tau$ for the first time after a warmup period, the aggregator switches to FFA-LoRA permanently. Varying $\tau$ traverses the same communication-quality frontier as the fixed-$K$ Two-Phase family without requiring $K$ to be selected in advance.
```

### G16.4 Contribution three, retitled

The old contribution was a four-benchmark evaluation, which Section 5.5 now establishes is insensitive. The actual contribution is the frontier.

```latex
\textbf{A five-method frontier and its knee.}
Five published protocols are placed on one measured frontier at two model scales and on two datasets, evaluated by held-out instruction-following loss rather than by zero-shot benchmarks, which do not discriminate between protocols at the smaller scale. The frontier has a knee at ReverseAdaptive: savings beyond that point cost roughly five times more quality per point.
```

### G16.5 Headline numbers (C26, fifth p=0.997 site)

```latex
\textbf{Headline numbers:}
Two-Phase $K{=}8$ saves 27.7\% (TinyLlama-1.1B) and 26.0\% (LLaMA-3.2-3B) over FLoRA. ReverseAdaptive saves 40.5\% and $30.0 \pm 4.0$\% respectively, and FFA-LoRA saves 61.9\% at TinyLlama scale for 0.0182 more held-out loss than ReverseAdaptive. At LLaMA-3.2-3B, ReverseAdaptive and Two-Phase $K{=}8$ differ in final loss by 0.000012, roughly 350 times less than at 1.1B. All federated methods on LLaMA-3.2-3B cluster within 0.5 percentage points across the four downstream benchmarks.
```

### G16.6 Release line

```latex
Code, configuration files, and scripts to reproduce every figure and table are publicly released; see Appendix~\ref{app:repro}.
```

---


## G17. Section 4.6, Scale Validation (C10, C26 adjacent)

Status: APPROVED. NEWLY IDENTIFIED: no group covered Section 4.6, which carries the second of six in-source `p = 0.997` sites. Asserted as `g17.*`.

**The decisive number.** Recomputed on the reported corpus, the 3B ReverseAdaptive versus Two-Phase K=8 comparison has a 95% CI of [-0.0114, +0.0114]. That interval CONTAINS the 0.0043 difference observed between the same two methods at 1.1B. The test cannot distinguish equivalence from a difference as large as the one seen at the smaller scale, which is the rigorous statement of why p=0.997 establishes nothing. The old Appendix C reported [-0.0061, 0.0061] from the MPS corpus, which is narrower and still would not have excluded the 1.1B effect.

### G17.1 Replace `body.tex:332`

```latex
At LLaMA-3.2-3B, ReverseAdaptive and Two-Phase $K{=}8$ reach final losses differing by $1.25 \times 10^{-5}$, roughly 350 times smaller than the 0.0043 separating them at TinyLlama-1.1B. A paired $t$-test over three seeds gives $p = 0.997$ with a 95\% confidence interval of $[-0.0114, 0.0114]$. That interval contains the 1.1B effect size, so the test cannot distinguish no difference from a difference as large as the one observed at 1.1B, and the result is reported as a statement about effect size rather than as evidence of equivalence. ReverseAdaptive also saves 4.0 percentage points more mean communication than the hand-tuned configuration (30.0\% against 26.0\%), because it switches at round 7 on one of three seeds. The same threshold $\tau{=}0.01$ used at TinyLlama-1.1B produced this behaviour with no scale-specific tuning.
```

---

## G18. Section 5.1, Implications for Deployment (C9)

Status: APPROVED. NEWLY IDENTIFIED: no group covered Section 5.1, which carries the fourth C9 site, "without downstream quality loss".

### G18.1 Full subsection replacement

```latex
\subsection{Implications for Federated LLM Deployment}
The 40.5\% communication savings ReverseAdaptive achieves on TinyLlama-1.1B, and 30.0\% on LLaMA-3.2-3B, translate directly into reduced network costs in real deployments. In mobile or IoT federated settings where uplink bandwidth is scarce and metered, cutting round-trip communication by 30 to 40\% for a held-out instruction-following loss cost of 0.0063 can change the economic viability of federated fine-tuning. Whether that trade is acceptable is a deployment decision rather than a universal one, and the frontier in Section~\ref{sec:frontier} is intended to let a practitioner make it explicitly: the same measurements show that accepting a further 21.3 points of savings costs roughly five times more quality per point. Prior work by \citet{kairouz2021advances} identifies communication as one of the fundamental open problems in federated learning at scale. The byte-tracking framework and adaptive protocol introduced here are a concrete step toward addressing that problem for LLM fine-tuning.
```

---

## G19. Appendix C, Statistical Tests (open item 7, C10)

Status: APPROVED. NEWLY IDENTIFIED: `app:stats` was referenced by G5.6 and G5.7 but nothing produced the recomputed table. The existing `tab:d1` carries MPS p-values (2.5e-5, 6.1e-6, 1.5e-5) that no other table in the revised paper uses.

**Cohen's d is dropped.** At n=3 it is not meaningful, and the old prose called it "the key scale-validation result".

### G19.1 Replace the Appendix C prose and the whole `tab:d1` float

```latex
\section{Statistical Tests}
\label{app:stats}
Paired $t$-tests over three seeds on final-round training loss and on held-out instruction-following loss. TinyLlama-1.1B results use the CUDA corpus, matching Table~\ref{tab:frontier}; LLaMA-3.2-3B results use the MPS corpus.

With three seeds these tests have very low power, so effect sizes should be read alongside the $p$-values. No multiplicity correction is applied in the main results tables. Applying a Bonferroni correction across the five TinyLlama comparisons gives a threshold of $0.01$, which the ReverseAdaptive versus Two-Phase $K{=}8$ comparison on held-out loss ($p = 0.026$) does not meet. That comparison is secondary under the frontier framing of Section~\ref{sec:discussion}; the load-bearing comparison is ReverseAdaptive versus FFA-LoRA, which holds at $p < 10^{-3}$ on both metrics with a difference roughly twenty times the largest per-method standard deviation.

The LLaMA-3.2-3B comparison between ReverseAdaptive and Two-Phase $K{=}8$ has a 95\% confidence interval of $[-0.0114, 0.0114]$, which contains the $0.0043$ difference observed between the same methods at TinyLlama-1.1B. The test therefore cannot distinguish equivalence from a difference of the magnitude seen at the smaller scale.

\begin{table}[H]
\caption{Paired $t$-tests, three seeds. TinyLlama-1.1B rows use the CUDA corpus; LLaMA-3.2-3B rows use MPS. Positive $\Delta$ indicates the first method has higher loss.}
\label{tab:d1}
\begin{center}
\small
\begin{tabular}{lcccc}
\toprule
Comparison & $\Delta$ final loss & $p$ & $\Delta$ held-out & $p$ \\
\midrule
\multicolumn{5}{l}{\emph{TinyLlama-1.1B, Alpaca, IID}} \\
Two-Phase $K{=}8$ vs.\ FLoRA & $+0.0098$ & $4.2 \times 10^{-5}$ & $+0.0034$ & $7.9 \times 10^{-5}$ \\
ReverseAdaptive vs.\ FLoRA & $+0.0141$ & $1.1 \times 10^{-5}$ & $+0.0063$ & $6.2 \times 10^{-3}$ \\
ReverseAdaptive vs.\ Two-Phase $K{=}8$ & $+0.0043$ & $3.4 \times 10^{-4}$ & $+0.0029$ & $2.6 \times 10^{-2}$ \\
FFA-LoRA vs.\ ReverseAdaptive & $+0.0282$ & $4.4 \times 10^{-4}$ & $+0.0182$ & $3.8 \times 10^{-4}$ \\
FedIT vs.\ FLoRA & $-0.0006$ & $0.588$ & $+0.0002$ & $0.425$ \\
\midrule
\multicolumn{5}{l}{\emph{TinyLlama-1.1B, Dolly-15k, IID}} \\
Two-Phase $K{=}8$ vs.\ FLoRA & $+0.0099$ & $7.8 \times 10^{-5}$ & $+0.0043$ & $2.4 \times 10^{-6}$ \\
ReverseAdaptive vs.\ FLoRA & $+0.0142$ & $2.5 \times 10^{-5}$ & $+0.0061$ & $7.6 \times 10^{-3}$ \\
\midrule
\multicolumn{5}{l}{\emph{LLaMA-3.2-3B, Alpaca, IID}} \\
Two-Phase $K{=}8$ vs.\ FLoRA & $+0.0214$ & $3.2 \times 10^{-4}$ & -- & -- \\
ReverseAdaptive vs.\ FLoRA & $+0.0215$ & $1.3 \times 10^{-2}$ & -- & -- \\
ReverseAdaptive vs.\ Two-Phase $K{=}8$ & $+0.000012$ & $0.997$ & -- & -- \\
\bottomrule
\end{tabular}
\end{center}
\end{table}
```

Held-out evaluation was not run on the LLaMA-3.2-3B checkpoints, hence the dashes.

---

# VERIFICATION GATES BEFORE SUBMISSION

1. Recompile clean under elsarticle: zero errors, zero undefined references, zero overfull hboxes.
2. `python verify_numbers.py --analysis-dir analysis` returns 0. Currently 174 assertions, all passing. Add an assertion for every new table cell. NOTE: the handoff's claim that a verifier "passed 32 of 32" was unsubstantiated; no such script existed in the repo. This gate first ran during this revision.
3. Confirm no number appears with two values across abstract, results tables, discussion, and conclusion.
4. Grep for em-dashes, curly quotes, and signposting words (First, Furthermore, Moreover, Additionally).
5. Grep for residual TMLR artifacts: "under review", "anonymous", "double-blind", "anonymized supplementary", "Acknowledgments suppressed".
6. Grep for the never-write list in Global conventions.
7. **SEVEN-site grep for `p = 0.997`.** Six in the source (`body.tex` lines 20, 332, 361, 400; `appendix.tex` lines 59, 75) plus one introduced in the G0 abstract draft. Covered by G16.5, G17.1, G7.1, G14.1, G19.1 and G0.2a. Grep to confirm none remain. Original note follows.
7b. **Superseded five-site note.** The claim appears in the abstract, Introduction, Section 4.6, Section 5.2, and the conclusion. G7 fixed Section 5.2, G14 the conclusion, G16 the Introduction. Section 4.6 and the abstract still need checking directly; do not trust group coverage.
8. **Three-region cross-check.** After G16 lands, one pass whose only job is checking the abstract, Introduction, and conclusion against each other, since all three restate the same number set.
9. **Internal consistency pass over formal objects.** Each algorithm, table caption, section heading, and equation checked against the prose describing it and the numbers depending on it. C16, C19, C22 and C25 all came from this class and `verify_numbers.py` catches none of them, because they contain no numbers or contradict text rather than data.
10. **Repository read as a reviewer encounters it**, with no prior knowledge, flagging anything that leads to a false conclusion. Known items in Open items 10 through 14.
11. Highlights file exists separately, three to five bullets, each under 85 characters.
12. Confirm Appendix E follows the statistical-tests appendix, since `app:stats` is referenced from G5.6 and G5.7 and `app:crossbackend` from G4.1, G5.1, G5.8, G9 and G10.

---

# REPO-VERIFIED FINDINGS

From `fedlora-protocols-neurocomputing-final-experiments`. Every run directory contains `config_merged.yaml` (the fully resolved config actually used), `results.json` (per-round, CUMULATIVE `communication_mb`), and `run_meta.json` (git commit, dirty flag).

**C3 confirmed.** Experiments existing in both variants scale exactly 2x: `exp_dolly_flora_iid` 2578.125 versus 5156.25; `exp_dolly_two_phase_k8_iid` 1863.125 versus 3726.25; `exp_ffa_lora_iid` 983.125 versus 1966.25. `exp_flora_iid` has a four-target merged config and records the two-target value 2578.125, as do `exp_reverse_adaptive_iid` (1533.125) and `exp_two_phase_k8` (1863.125). Those runs executed two targets despite their configs.

**C16 confirmed.** `exp_ffa_lora_iid` round 1 = 116.875 MB = 85.9375 full upload + 30.9375 B-only broadcast.

**Instrumentation confirmed honest.** `server.py:289` and `:343` use `p.numel() * p.element_size()` on transmitted tensors, not parameter-count estimates.

**Config defaults are clean.** All configs resolve transitively to `base_config_4layers.yaml` through a two-level `_inherit` chain that `run_experiment.py:56-73` deep-merges recursively. That base sets `local_epochs: 1`, `learning_rate: 0.0001`, `max_seq_length: 256`. Table 1 is correct on all three; the `client.py` defaults are unreachable fallbacks. No run trained double epochs.

**`_apply_partial_freeze`** exists in `client.py`; `freeze_a=True` maps to ratio 1.0; intermediate ratios are supported but unexercised by any config.

**`_unfreeze_all_lora()` is load-bearing, not cleanup.** `lora_model.py:184` filters uploads on `param.requires_grad`, so frozen A would be excluded entirely. `client.py` calls `_unfreeze_all_lora()` before `get_lora_state_dict()`, which returns A to the payload and lets the server harvest it at the transition round. FFA-LoRA round 1 being 116.875 rather than 61.875 depends on it. Deleting that line would break the protocol. The `DEPRECATED` marker on `freeze_a` in the client docstring is a second trap in the same area: it is the live path.

**Commit provenance.** `phase1_cuda_rerun` is single-revision at `7b957752`; `neuro_part12_qvonly` is single-revision at `cae4c328`; T-N1 spans both. T-N2 is entirely `cae4c328`. The 3B corpus is three seeds at three revisions (`d6f90388`, `e67fa6b0`, `6de806bb`). The threshold ablation is single-revision at `1e518c63`, as are all five alpha=0.1 seeds. All 123 runs record `git_dirty: true`.

**`analysis/statistical_tests.csv`** holds the MPS p-values (2.46e-5, 6.12e-6, 1.46e-5) backing the old Table 2. Regenerate on the CUDA corpus.

---

# OPEN ITEMS

1. **RESOLVED.** N3 data is in `docs/PHASE_1_BATCH_VERIFY.md` and `docs/PHASE_1_17_DIAGNOSTIC.md`. Used in G13.
2. **Supplementary CSV `target_modules` column is wrong for the phase-1 rows.** All four-module rows come from `phase1_cuda_rerun`, all two-module rows from `neuro_part12_qvonly`. The summarizer reads target modules from the config file rather than the run artifact, and old configs inherit `base_config_4layers.yaml` while executing two targets. Regenerate from checkpoint adapter keys or drop the column. Keep `broadcast_bytes_per_client`, which is read from the artifact and correct.
3. **RESOLVED.** Round-6 saturation surfaced in G7 rather than left in Section 4.5. Asserted as `switch.TinyLlama_all_round6`.
4. **RESOLVED.** N3 denominator: 31 comparable, 27 agreeing, 3 CUDA-only, 4 disagreeing. In G13.
5. **RESOLVED.** N3 framing: boundary sensitivity, every disagreement exactly one 110 MB lattice step. In G13.
6. **RESOLVED.** `_apply_partial_freeze` is in `client.py`; never write "progressive" or "partial freezing".
7. **RESOLVED.** T-N1 replaces `tab:tinyllama_iid` and gains a p-value column; `tab:b1` moves to CUDA in G11; Appendix C Table 9 is recomputed to carry the full matrix on both metrics.
8. **RESOLVED.** Multiplicity: disclose the Bonferroni failure, name the failing cell as secondary, note low power at n=3.
9. **Grep for residual "label skew"** outside Section 3.4, including Appendix A and C.
10. **RESOLVED.** `qv_only_cuda_per_seed_runs.csv` is now committed to `analysis/`. Both `verify_numbers.py` and `make_frontier_figure.py` require it.
11. **RELEASE BLOCKER. `results/raw` holds 123 run directories, roughly 58 reported.** Unmarked superseded runs include four-target duplicates of every part-1/part-2 experiment and Two-Phase runs at 2578.125 that never switched; `exp_two_phase_k8` alone has eleven directories for a three-seed result. Decision: do not release the run tree. Release curated analysis CSVs, configs, and scripts, matching the roughly 60 KB TMLR supplementary scope. Generate a provenance manifest anyway, matching each run's final communication and loss against the CSVs, and store it in `docs/`, not in the release bundle. Ranked by risk: four-target duplicates first, since they differ by exactly 2x and would look plausible in an aggregate; never-switched Two-Phase runs second, since they are indistinguishable from FLoRA totals.
12. **REPOSITORY HAZARD. `base_config_4layers.yaml`.** Every experiment inherits from it and its `target_modules` list is the source of the wrong CSV column. Renaming breaks the `_inherit` chain in every downstream config, so the fix is a header comment stating that `target_modules` was not passed to the model constructor before commit `8cddb8b`, that all reported runs adapt `q_proj` and `v_proj` only, and that the byte counts are the evidence.
13. **REPOSITORY HAZARD. `ffa_lora.py:get_communication_cost`** is dead code (`grep` returns only the definition) whose comment claims `# float16 * 2 (upload + download)`. For TinyLlama it returns 811,008 x 4 = 3,244,032, matching the recorded value, but the factorization is wrong: if that were round-trip, FFA-LoRA's total would be 464 MB against the measured 983.125. Fix the comment rather than deleting the method, since deletion is a silent divergence between archived and executed code, and note the change in release notes.
14. **`neuro_new_tables.tex`** captions say "population standard deviation" in all five tables and are wrong. `make_frontier_figure.py` uses `st.pstdev` for error bars and should use `st.stdev`. The `zhang2024fedit` key does not exist; use `zhang2024federatedgpt`.
15. **Bib spot-check.** The Dolly entry in `neuro_new_refs.bib` is transcribed rather than re-fetched; verify the author list. Its key is `dolly2023`, and G6 must use that rather than `conover2023dolly`.
16. **`docs/tmlr-submission/verify_numbers.py` is a stale early version.** Replace with the current 174-assertion script, and place copies in `scripts/` and in the new submission folder.
17. **`main.tex`, `main.bib` and `math_commands.tex` are not in the repository.** Only `body.tex` and `appendix.tex` are. They must be supplied before any compile.
18. **Submission folder:** `docs/neurocomputing-submission/`, parallel to `docs/tmlr-submission/`. Do not modify the TMLR folder; that version corresponds to a public OpenReview submission and is kept for provenance.
19. **Application instructions for Cursor** are in `CURSOR_INSTRUCTIONS.md`, which carries the DO NOT APPLY list (G1.4, G0.2a, G0.2b, G10.2, G8.4, G8.3) and the four verification-grep stages.
17. **Section 4.3 body text** carries the same 1.4 pp figure as the Table 3 caption and needs the C22 correction.
18. **Section 4.6** is the one remaining unchecked `p = 0.997` site.
19. **Labels. CORRECTED.** `\label{sec:limitations}` ALREADY EXISTS in `body.tex`; listing it as missing was an error and G10.1 adding it again produces a duplicate-label warning. Keep one. Genuinely absent and requiring manual addition: `\label{sec:setup}` on Section 4.1 (G4b references it) and `\label{sec:implementation}` on Section 3.4 (G8.1 and G10.1 reference it). Arriving with their blocks: `sec:baseregression` (G8b), `sec:frontier` (G5.3), `sec:dolly` (G6.1), `app:crossbackend` (G13.2). Already present and unchanged: `sec:intro`, `sec:related`, `sec:method`, `sec:experiments`, `sec:ablation`, `sec:discussion`, `sec:conclusion`, `app:hyper`, `app:perseed`, `app:stats`, `app:repro`.
20. **Bibliography entries, verified.** FedSRD: Guochen Yan, Luyuan Xie, Qingni Shen, Yuejian Fang, Zhonghai Wu; ACM Web Conference 2026; DOI 10.1145/3774904.3792144; arXiv:2510.04601. FLoRIST: Hariharan Ramesh, Jyotikrishna Dass (University of Arizona); MLSys 2026; arXiv:2506.09199. Full BibTeX supplied to Cursor.
21. **C7 wording, spot-check before submission.** G8.1 states that FedSRD and FLoRIST "report measured per-round communication in megabytes". Abstracts confirm they report communication reductions; the per-round MB tables were verified separately. If those tables report percentages only, change to "report measured communication volume" and the concession still stands.
