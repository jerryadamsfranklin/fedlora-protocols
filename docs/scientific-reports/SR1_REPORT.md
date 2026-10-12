# STAGE SR-1 Report

## Step 0
- Branch: `scientificreports/paper-writing` from `clustercomputing/paper-writing` (stage-15)
- HEAD at Step 0: `ae42bef` (sr-1.0)
- `cmp` confirmed `sr_main.tex == main_flat.tex` before edits

## Step 1
- Downloaded Springer Nature Dec 2024 package (zip ~880KB)
- `sn-jnl.cls` ProvidesClass: `2019/11/18 v0.1`
- Copied `sn-jnl.cls` and `sn-nature.bst` into `docs/scientific-reports/`
- `cmp` vs `docs/cluster-computing/sn-jnl.cls`: **IDENTICAL**
- Commit: `6272b9b` sr-1.1

## Step 2
- Class: `\documentclass[pdflatex,sn-nature,lineno]{sn-jnl}`
- Bibliography style: set by class option `sn-nature` (no explicit `\bibliographystyle` line; class injects `sn-nature.bst`)
- All `figure*`/`table*` converted; Fig1/Fig3 at `0.75\textwidth`, Fig2/Fig4 at `\textwidth`
- Removed `\graphicspath`
- Build: 27 pages initially (pre-restructure); bibtex reports 19 `pop empty literal stack` warnings from `sn-nature.bst` on preprint-like entries, but all 26 citations resolve
- Commit: `072961a` sr-1.2

## Step 3
- Inventory at `docs/scientific-reports/INVENTORY.md` (53 structural items, 90 refs)
- Commit: `5fad7fd` sr-1.3

## Step 4
- Restructured to Introduction / Results / Discussion / Methods
- **4c deleted paragraph:** none found (no "rest of this paper is organized" paragraph)
- Related Work paragraphs moved into Introduction before contribution list; section/subsection headings deleted; `sec:related` dropped (no refs)
- Discussion subsection headings removed; Limitations + Conclusion paragraphs appended
- Commit: `43bc2c1` sr-1.4

## Step 5
- Main display counts: **figures 4, tables 3, algorithms 1 (total 8)**
- Commit: `613462a` sr-1.5

## Step 6
### 6d Mapping (old label -> Supplementary Table/Algorithm)
| Old label | SI number |
|-----------|-----------|
| tab:downstream_tiny | Supplementary Table S1 |
| tab:downstream_llama | Supplementary Table S2 |
| tab:ablation | Supplementary Table S3 |
| tab:setup | Supplementary Table S4 |
| tab:b1 | Supplementary Table S5 |
| tab:b2 | Supplementary Table S6 |
| tab:b3 | Supplementary Table S7 |
| tab:b4 | Supplementary Table S8 |
| tab:c1 | Supplementary Table S9 |
| tab:d1 | Supplementary Table S10 |
| tab:d2 | Supplementary Table S11 |
| alg:bonly | Supplementary Algorithm S1 |

Numbering follows first-citation order in main body for S1–S4; S5–S11 are SI-order for tables not directly cited from main body.

### 6e SI citation keys
`clark2018arc`, `clark2019boolq`, `dolly2023`, `dubey2024llama3`, `hendrycks2020measuring`, `loshchilov2019decoupled`, `taori2023alpaca`, `wang2024flora`, `zellers2019hellaswag`, `zhang2024tinyllama`

- SI build: 8 pages
- Commit: `91170d8` sr-1.6

## Step 7
- 59 mechanical replacements logged in `STEP7_REPLACEMENTS.md`
- **SR-REVIEW lines:** none (all refs fit the rules)
- `grep` for `\ref{sec:` / `\ref{app:` in `sr_main.tex`: empty
- Some phrases read a bit bare ("which Discussion quantifies", "slice of Methods") — left as mechanical rule output for prose round
- Commit: `6408e6e` sr-1.7

## Step 8 — old back matter (for review)
```
\backmatter

\bmhead{Acknowledgements}

The Alpaca and Dolly-15k datasets are publicly available from their respective sources.

The author used Claude (Anthropic) for grammar and language editing of the manuscript and as a coding assistant for the released codebase. All technical content, analysis, and interpretations are the author's own.

{\tolerance=2000\emergencystretch=1.5em
\bibliography{main}
}

\section*{Statements and Declarations}

\begin{itemize}
\item \textbf{Funding:} The author received no funding for this work.
\item \textbf{Competing interests:} The author declares no competing interests.
\item \textbf{Ethics approval and consent to participate:} Not applicable.
\item \textbf{Consent for publication:} Not applicable.
\item \textbf{Data availability:} Analysis CSVs, run records, and scripts to regenerate tables and figures are available in the public repository at \url{https://github.com/jerryadamsfranklin/fedlora-protocols} and archived as release v1.0.1 at \doiurl{10.5281/zenodo.23111915}~\cite{franklin2026fedloraprotocols}. Final adapter states for the three seed-42 TinyLlama-1.1B runs evaluated in Supplementary Table S1 are included; other adapter checkpoints are not. A verification script in the repository recomputes the paper's principal quantitative claims from the released records.
\item \textbf{Materials availability:} Not applicable.
\item \textbf{Code availability:} Source code is available in the same repository.
\item \textbf{Author contributions:} The author performed all work associated with this study.
\end{itemize}

\end{document}

```

Notes:
- Old Data availability cited Zenodo via `\cite{franklin2026fedloraprotocols}` — **kept** on first sentence
- "Table 4" / `Table~\ref{tab:downstream_tiny}` referred to **tab:downstream_tiny** → now **Supplementary Table S1**
- Statements and Declarations (Ethics/Consent) removed
- Commit: `5e7af2c` sr-1.8

## Step 9
- `sr_main_submit.tex` / `sr_supplementary_submit.tex` with inline `.bbl`
- Empty-temp-dir builds: main 21 pp, SI 8 pp; no undefined refs/citations
- Commit: `6e4dbbd` sr-1.9

## Step 10

### 10a Display items in submit
figures 4, tables 3, algorithms 1, **total 8**

### 10b Word counts (texcount "Words in text"; excludes headers/captions/floats; math counted separately)
| Section | Words in text |
|---------|---------------|
| Abstract | 150 |
| Introduction | 1087 |
| Results | 1438 |
| Discussion | 2413 |
| **Intro+Results+Discussion** | **4938** |
| Methods (separate) | 1702 |

SR limits: main text ≤4500 (currently **over by ~438**); abstract ≤200 (OK).

### 10c References
- Last number: **[26]**
- Uncited references: **none**

### 10d Text-preservation
After normalizing Step-7 cross-ref rewrites and deleted headings, **9** residual "missing" spans remain; all are float/table-body fragments whose detex boundaries differ after SI moves (not prose deletions). Expected intentional deltas: Step 7 xrefs, old back matter, deleted Related Work / Discussion / Method / Limitations / Conclusion **headings** only.

Smart-missing list:
```
1. [t] [1] round r , phase boundary K , frozen- A cache (initially empty) upload mode, broadcast mode upload full state (A, B) broadcast full state upload full state (A, B) A_ A from the first client's uploaded state broadcast B only upload B only broadcast B only ReverseAdaptive replaces the fixed boundary K with an adaptive loss-plateau signal.

2. [tbp] 6pt 0.95 @ lp 0.72 @ Parameter & Value Base models & TinyLlama-1.1B-Chat-v1.0 ; LLaMA-3.2-3B Datasets & Alpaca ; Dolly-15k ; 3000 samples each Data partitions & IID; Dirichlet 0.5 ; Dirichlet 0.1 Clients / round & 10 / 10 (full participation) Rounds & 15 Local epochs & 1 Batch size & 4 (TinyLlama), 2 (LLaMA-3.2-3B) Grad.

3. accumulation & 4 (TinyLlama), 8 (LLaMA-3.2-3B); effective batch 16 Learning rate & 10^ -4 (AdamW ) Max sequence length & 256 LoRA rank / alpha & 16 / 32 LoRA target modules & , LoRA dropout & 0.1 Hardware (primary) & Apple M4 Pro, 48 ,GB unified memory, MPS Hardware (reproduction) & NVIDIA RTX 4090 (TinyLlama); A100 40 ,GB (LLaMA-3.2-3B) Training dtype & float32 (TinyLlama-1.1B); float16 base + float32 LoRA (LLaMA-3.2-3B) Transmitted dtype & float32 (TinyLlama-1.1B); float16 (LLaMA-3.2-3B) Downstream eval & 500 examples/benchmark, seed 42, log-likelihood Held-out eval & 500 examples, train indices 3001--3500, seq.

4. 256 Benchmarks & MMLU , ARC-Easy , BoolQ , HellaSwag Seeds (TinyLlama IID) & 42, 123, 456 Seeds (LLaMA-3.2-3B) & 42, 123, 456 ReverseAdaptive defaults & =5, 0.01 , =1.1, =0 (instantaneous) FFA-LoRA and FedIT are reimplemented within this codebase rather than taken from the original authors' released code, which places all five protocols on one training loop and one byte-accounting path; their absolute results may differ from those of the original implementations.

5. [tbp] 6pt @ lrrrr@ Method & ARC-Easy & BoolQ & HellaSwag & vs.

6. base (BoolQ) Base (no adapter) & 0.274 & 0.626 & 0.448 & -- FLoRA & 0.246 & 0.562 & 0.420 & -6.4 pp Two-Phase K = 8 & 0.248 & 0.560 & 0.420 & -6.6 pp ReverseAdaptive & 0.252 & 0.552 & 0.420 & -7.4 pp Spread (max--min) & 0.006 & 0.010 & 0.000 & -- At LLaMA-3.2-3B scale, the three federated methods are indistinguishable on downstream benchmarks within seed variation.

7. [bp] 6pt @ lrrrr@ Method & MMLU & ARC-Easy & BoolQ & HellaSwag Base (no adapter) & 0.544 & 0.832 & 0.706 & 0.540 FLoRA & 0.558 0.007 & 0.836 0.007 & 0.736 0.007 & 0.535 0.003 Two-Phase K = 8 & 0.558 0.002 & 0.837 0.008 & 0.738 0.010 & 0.539 0.004 ReverseAdaptive & 0.557 0.003 & 0.839 0.008 & 0.739 0.004 & 0.538 0.004 Supplementary Table S3 and Fig.~ characterize how controls the savings-quality tradeoff.

8. [tbp] 6pt @ lrrrr@ & Switch round & Final loss & Total comm.

9. (MB) & Savings 0.001 & 11 & 1.2638 & 2083.13 & 19.20 % 0.002 & 9 & 1.2675 & 1863.13 & 27.73 % 0.005--0.2 & 6 & 1.2736 & 1533.13 & 40.53 % [tbp] Table~ and Fig.~ confirm that the Pareto structure observed at TinyLlama-1.1B is preserved at LLaMA-3.2-3B.

```

Naive (no Step7 norm) missing count was 96, dominated by Section/Appendix/Table ref rewrites and heading removals.

### 10e SI citation / numbering
- S1, S3, S4, Algorithm S1, Notes 1–4: directly cited in main
- S2: cited via `Supplementary Tables S1--S2`
- S5–S11: SI-only (reachable via Supplementary Notes 2–4); numbering follows SI order after cited tables

### 10f
- No `??` in either PDF
- No unicode em dashes or curly quotes in new tex

### 10g verify_numbers.py
- **Does NOT read any manuscript `.tex` path.** Design note in the script: curated assertions against CSVs in `--analysis-dir` (default `analysis/`), with location strings as metadata only.
- Result: **179 passed, 0 failed, 0 errored, 179 total**
- (Required conda env `fedlora`; system python 3.9 hits a numpy macOS Accelerate segfault.)

### 10h Renders
- 21 pages at 110 dpi in `docs/scientific-reports/render/` (gitignored)
- One Overfull hbox (~11.7pt) in Results table region (lines ~171–186)
- Floats within 0–1 page of first citation; Algorithm 1 colocated in Methods (p16)

## Commit hashes (sr-1.*)
| Step | Hash |
|------|------|
| 0 | ae42bef |
| 1 | 6272b9b |
| 2 | 072961a |
| 3 | 5fad7fd |
| 4 | 43bc2c1 |
| 5 | 613462a |
| 6 | 91170d8 |
| 7 | 6408e6e |
| 8 | 5e7af2c |
| 9 | 6e4dbbd |
| 10 | c6ff72b |

## Artifacts
- `docs/scientific-reports/sr_main.pdf`
- `docs/scientific-reports/sr_supplementary.pdf`
- `docs/scientific-reports/scientific-reports-sr1.zip` (excludes render/)
