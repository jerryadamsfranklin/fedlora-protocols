# CURSOR EXECUTION PLAN: OJ-CS Conversion

**Read this whole file before running any command.**

Repo: `fedlora-protocols`
Current branch: `neurocomputing/paper-writing`
Source dir: `docs/neurocomputing-submission/`

Rules for this task:

- Do not edit `docs/neurocomputing-submission/`. It is the frozen fallback for
  IEEE Access. All work happens in a new directory.
- Do not change any number. Every reported value is verified by
  `scripts/verify_numbers.py` (174 assertions, currently 174/174 passing). If a
  step seems to require changing a number, stop and ask.
- Do not invent replacement text where this file gives verbatim blocks. Use the
  blocks exactly.
- Commit after each numbered STEP. Do not batch commits.
- Writing mandates that apply to every line you write or edit: no em-dashes, no
  curly quotes, no signposting words (First, Furthermore, Moreover,
  Additionally).

---

## STEP 0: Branch and directory

```bash
git checkout neurocomputing/paper-writing
git pull
git checkout -b ojcs/paper-writing
cp -r docs/neurocomputing-submission docs/ojcs-submission
git add docs/ojcs-submission
git commit -m "STEP 0: branch ojcs/paper-writing, seed docs/ojcs-submission from Neurocomputing source"
```

Confirm before proceeding:

```bash
git branch --show-current   # must print: ojcs/paper-writing
ls docs/ojcs-submission     # must contain main.tex body.tex appendix.tex main.bib math_commands.tex figures/
```

**Everything below operates on `docs/ojcs-submission/` unless stated otherwise.**

---

## STEP 1: Delete files that do not belong in an IEEE submission

```bash
cd docs/ojcs-submission
rm -f highlights.txt          # Elsevier-only "Highlights" requirement
rm -f verify_numbers.py       # byte-identical duplicate of scripts/verify_numbers.py
rm -f main.pdf                # stale Neurocomputing build
cd ../..
git add -A && git commit -m "STEP 1: remove Elsevier-only highlights, duplicate verifier, stale PDF"
```

---

## STEP 2: Content corrections (C29-C34)

These are venue-independent bugs found by review. Apply them before any
formatting work.

### C29a. p-value overclaim in Section 4.3

File: `docs/ojcs-submission/body.tex`

FIND (exact):
```
Paired $t$-tests on final loss give $p < 10^{-4}$ for every comparison against FLoRA except FedIT, which is correctly indistinguishable.
```

REPLACE WITH:
```
Paired $t$-tests on final loss give $p < 10^{-3}$ for every comparison against FLoRA except FedIT, which is correctly indistinguishable.
```

Reason: Table 2 reports FFA-LoRA vs FLoRA at $1.7 \times 10^{-4}$, which is
larger than $10^{-4}$. The sentence contradicts the paper's own table. The
number in the table is correct; the sentence is not.

If the FIND string does not match exactly, locate the sentence containing both
`10^{-4}` and `FedIT` in Section 4.3 and make the same substitution. Do not
change any table value.

### C29b. Same overclaim in Section 5.5

File: `docs/ojcs-submission/body.tex`, subsection "The Base-Model Regression Observation"

FIND (exact):
```
so the zero-shot benchmarks do not separate protocols that the held-out metric separates at $p < 10^{-4}$
```

REPLACE WITH:
```
so the zero-shot benchmarks do not separate protocols that the held-out metric separates at $p < 0.05$
```

Reason: on the held-out metric, Table C.10 gives ReverseAdaptive vs FLoRA at
$6.2 \times 10^{-3}$ and ReverseAdaptive vs Two-Phase $K{=}8$ at
$2.6 \times 10^{-2}$. Neither is below $10^{-4}$.

Note: this sentence lives in text that STEP 4 moves to supplemental. Fix it here
anyway so the correction travels with the text.

### C30. Delete the duplicated GQA paragraph

File: `docs/ojcs-submission/body.tex`, Section 3.1 "Background and Notation"

The GQA paragraph appears verbatim in Section 3.1 and again in Section 3.4 under
"GQA B-fraction". Delete the Section 3.1 copy.

FIND (exact, in Section 3.1):
```
The communication cost of a round is the sum of upload bytes and download bytes. For TinyLlama-1.1B with rank $r{=}16$ targeting $\{\texttt{q\_proj}, \texttt{v\_proj}\}$, the projections adapted in the original LoRA formulation \citep{hu2022lora}, accounting for GQA in the value projection, the B-only fraction of the full state is exactly 36\% rather than the 50\% a naive parameter count would suggest. This fraction is invariant to the size of the target set, since \texttt{q\_proj} and \texttt{o\_proj} share a shape and \texttt{k\_proj} and \texttt{v\_proj} share a shape under GQA; the ratio is fixed by the GQA configuration, not by how many projections are adapted. LLaMA-3.2-3B has a B-only fraction of approximately 40\% due to different GQA configuration.
```

REPLACE WITH:
```
The communication cost of a round is the sum of upload bytes and download bytes. The B-only fraction of the full state is not the 50\% a naive parameter count suggests, because grouped-query attention fixes the ratio of $B$ to $A$ parameters; Section~\ref{sec:implementation} derives the exact fractions of 36\% for TinyLlama-1.1B and 40\% for LLaMA-3.2-3B.
```

**Do not touch the Section 3.4 copy.** It is the one that stays.

Verify: `grep -c "invariant to the size of the target set" docs/ojcs-submission/body.tex` must print `1`.

### C31. Reconcile the Two-Phase switch round across figure captions

Figure 2's caption says round 8. Figure 3's caption and the Section 4.5 body say
round 9. Both refer to the same run.

Convention to adopt: `K` is the phase boundary (the last FLoRA round); the
transition round is `K+1`. This matches the lattice formula
`55(2s-1) + 928.125` in `scripts/verify_numbers.py`, where Two-Phase K=8 is
`s=9`. Do not change the formula or any total.

File: `docs/ojcs-submission/body.tex`, Figure 2 caption.

FIND:
```
and Two-Phase $K{=}8$'s phase boundary (round 8)
```

REPLACE WITH:
```
and Two-Phase $K{=}8$'s phase boundary (round 8; the transition round, in which the upload is still full, is round 9)
```

Then check Figure 3's caption reads "Two-Phase $K{=}8$ at round 9" and leave it.
Add nothing else.

### C32. Add the missing row to the statistical-tests table

File: `docs/ojcs-submission/appendix.tex`, Table C.10.

Section 4.3 points readers to this table for "the full matrix on both metrics",
but the table omits FFA-LoRA vs FLoRA, which Table 2 does report.

Add one row to the TinyLlama-1.1B block of Table C.10:

```
FFA-LoRA vs.\ FLoRA & $+0.0423$ & $1.7 \times 10^{-4}$ & $+0.0246$ & --- \\
```

**Before inserting, recompute both deltas and both p-values yourself** from
`analysis/qv_only_cuda_per_seed_runs.csv` and
`analysis/neuro_part12_qvonly_holdout_strict24.csv` using the same paired t-test
code path as `scripts/statistical_analysis.py`. The values above are derived from
Table 2 arithmetic (1.3031 - 1.2608 = 0.0423; -0.5746 - (-0.5992) = 0.0246) and
must be confirmed, not trusted. If your computed values differ, use yours and
flag the discrepancy.

Then add a matching assertion to `scripts/verify_numbers.py` in the same style as
the existing `g14.*` entries, so the new row is covered.

If recomputation is not possible for any reason, do the fallback instead:

File: `docs/ojcs-submission/body.tex`, Section 4.3.
FIND: `the full matrix on both metrics is in`
REPLACE WITH: `additional comparisons on both metrics are in`

### C33. Correct the overstated spread ratio

File: `docs/ojcs-submission/body.tex`, Section 4.3.

FIND: `roughly twenty-five times smaller`
REPLACE WITH: `more than twenty times smaller`

Reason: with sd approximately 0.0006 against a gap of 0.0141, the ratio is about
23.5, not 25.

### C34. Verify the two 2026 references

File: `docs/ojcs-submission/main.bib`

Two entries cite 2026 venues:

- `yan2026fedsrd`: WWW '26, `doi:10.1145/3774904.3792144`, `arXiv:2510.04601`
- `ramesh2026florist`: MLSys 2026, `arXiv:2506.09199`

Resolve both DOIs and both arXiv IDs in a browser. Confirm title, full author
list, venue, and year against the bib entry. Earlier drafts in this project
contained fabricated arXiv IDs and incorrect author lists, so this is not
optional.

Record the outcome in the commit message. Do not proceed to STEP 3 until done.

### Commit STEP 2

```bash
python3 scripts/verify_numbers.py --analysis-dir analysis
# must print: 174 passed, 0 failed, 0 errored, 174 total
#             (175 total if the C32 assertion was added)
git add -A
git commit -m "STEP 2: C29-C34 content corrections; verified both 2026 references resolve"
```

---

## STEP 3: Repository cleanup (R1-R4)

These are outside `docs/ojcs-submission/`.

### R1. `CITATION.cff`

FIND:
```
  journal: "Neurocomputing"
  abstract: "Manuscript in preparation."
```
REPLACE WITH:
```
  journal: "IEEE Open Journal of the Computer Society"
  abstract: "Manuscript under review."
```

### R2. `README.md`, four verified inaccuracies

**R2a.** Citation block.
FIND: `  journal = {Neurocomputing},`
REPLACE WITH: `  journal = {IEEE Open Journal of the Computer Society},`

**R2b.** Test count. The repo contains 28 test functions, not 20.
Verify first: `grep -rho "def test_[a-zA-Z0-9_]*" tests/ | sort -u | wc -l`
FIND: `tests/                   # 20 unit tests covering aggregators and runner`
REPLACE WITH: `tests/                   # 28 unit tests covering aggregators and runner`

**R2c.** Figure count. Seven PDFs exist; six are generated (see R3).
FIND: `ls figures/  # six PDFs corresponding to paper figures`
REPLACE WITH: `ls figures/  # six PDFs corresponding to paper figures (fig1 through fig6)`
This becomes true once R3 deletes the orphan.

**R2d.** Broken quick-start example. The path `results/raw/` does not exist and
adapter checkpoints are not released, so this command cannot run as written.

FIND:
```
# Evaluate a saved checkpoint on downstream benchmarks
python3 scripts/evaluate_checkpoint.py \
    --checkpoint results/raw/.../final_adapter_state.pt \
    --num-examples 500 \
    --output results/downstream/my_eval.json
```
REPLACE WITH:
```
# Evaluate a checkpoint you produced above on downstream benchmarks.
# Adapter checkpoints are not distributed with this repository; run an
# experiment first and point --checkpoint at the state file it writes.
python3 scripts/evaluate_checkpoint.py \
    --checkpoint <path printed by run_experiment.py> \
    --num-examples 500 \
    --output results/downstream/my_eval.json
```

Also fix the repository-layout block, which claims a `raw/` subdirectory that
does not exist:
FIND: `├── results/                # Per-run JSONs (raw/) and downstream eval (downstream/)`
REPLACE WITH: `├── results/                # Run manifest and downstream eval (downstream/)`

### R3. Delete the orphan figure

`scripts/generate_paper_figures.py` emits exactly six files: `fig1_frontier`,
`fig2_convergence`, `fig3_cumulative_comm`, `fig4_downstream_accuracy`,
`fig5_threshold_ablation`, `fig6_scale_validation`. `figures/fig1_pareto_iid.pdf`
has no generator and is a leftover from an earlier naming scheme.

```bash
grep -rn "fig1_pareto_iid" . --exclude-dir=.git   # expect no hits outside figures/
git rm figures/fig1_pareto_iid.pdf
```

Then confirm the directory is fully regenerable:
```bash
rm -f figures/fig*.pdf
python3 scripts/generate_paper_figures.py
ls figures/*.pdf | wc -l   # must print 6
```

### Commit STEP 3

```bash
git add -A
git commit -m "STEP 3: R1-R4 repo cleanup; CITATION.cff venue, README stale counts and broken example, orphan figure removed"
```

---

## STEP 4: Length reduction

`body.tex` is 8,685 words. The IEEE double-column budget after 6 figures, 7
tables, 2 algorithms, and 26 references leaves roughly 5,000 words. Target cut:
approximately 3,600 words.

Do the cuts in this order. Recount after each.

```bash
wc -w docs/ojcs-submission/body.tex
```

### L1. Move the downstream-benchmark block to supplemental (saves ~1,280 words plus ~1.2 pages of floats)

Cut from `body.tex` and paste into a new file
`docs/ojcs-submission/supplemental.tex`:

- The whole of Section 4.6 "Downstream Evaluation" (heading, body text, Figure 4
  block, Table `tab:downstream_tiny`, Table `tab:downstream_llama`)
- The whole of Section 5.5 "The Base-Model Regression Observation" including its
  `\label{sec:baseregression}`

Insert this replacement at the end of Section 4.2 "Held-Out Instruction-Following
Metric", as a new final paragraph:

```
The four zero-shot benchmarks do not discriminate between aggregation protocols at either scale. Cross-method spread is at most 1.0 percentage point at TinyLlama-1.1B, where every fine-tuned checkpoint also scores below the base model, and at most 0.5 percentage points at LLaMA-3.2-3B. Quality conclusions in this paper therefore rest on held-out instruction-following loss. The full downstream results, and an account of the TinyLlama regression, are in the supplemental material.
```

Then repair every dangling reference:

```bash
grep -n "sec:baseregression\|tab:downstream_tiny\|tab:downstream_llama\|fig:downstream" docs/ojcs-submission/body.tex
```

Each hit must become either a reference to the supplemental material or be
deleted. Known sites to check: the Introduction headline-numbers paragraph, the
Section 4.2 opening sentence, and the Limitations paragraph about zero-shot
benchmarks.

This cut is defensible on the merits: the paper's own argument is that these
benchmarks are uninformative. Do not weaken that argument to justify keeping
them.

### L2. Move Section 5.4 to supplemental (saves ~370 words)

Cut the whole of Section 5.4 "The Reproducibility Guarantee" from `body.tex` into
`supplemental.tex`.

Insert this single sentence at the end of Section 4.5 "Convergence and Switching
Behavior", after the existing no-switch paragraph:

```
Disabling the switch requires a negative threshold rather than a small positive one, for reasons set out in the supplemental material.
```

Keep the existing Section 4.5 sentence reporting the 10-decimal-place match. Do
not delete it.

### L3. Compress Related Work (target ~250 words)

File: `body.tex`, Section 2.

**2.1:** delete the QLoRA sentence and the closing "compact and composable"
sentence. Keep the `dettmers2023qlora` citation by folding it into the adapter
sentence.

**2.3:** delete the FedPAQ sentence and the Deep Gradient Compression detail
about 0.1 percent. Keep both citations by compressing to one sentence naming
QSGD, deep gradient compression, and FedPAQ together as quantization and
sparsification approaches.

**2.4:** delete the FedNova sentence detail and the curriculum-scheduling
sentence. Keep the final sentence beginning "To the authors' knowledge", which
is the positioning claim.

**Do not drop any `\cite` key.** Verify:
```bash
grep -o "\\\\citep\?{[^}]*}" docs/ojcs-submission/body.tex | tr ',' '\n' | grep -o "[a-z0-9]*20[0-9][0-9][a-z]*" | sort -u > /tmp/after.txt
```
Compare against the same command run on the Neurocomputing copy. The sets must
be identical.

### L4. Compress Limitations from twelve paragraphs to six (target ~450 words)

File: `body.tex`, Section 6.

**Merge into one paragraph:** the dataset/task-type paragraph and the
Alpaca-only-baselines paragraph.

**Merge into one paragraph:** the code-revisions paragraph and the
LLaMA-multi-revision/seed-noise paragraph.

**Delete outright:** the zero-shot benchmark paragraph (its content moves to
supplemental under L1) and the final "Future work should address..." paragraph,
whose three items are already named elsewhere in the section.

**DO NOT CUT, DO NOT SHORTEN, DO NOT MERGE these three paragraphs:**

1. The partial-participation paragraph
2. The semantic-heterogeneity paragraph ("Non-IID partitions are constructed by
   Dirichlet skew over instruction-length buckets...")
3. The transition-round harvest-rule paragraph ("At the transition round the
   server freezes $A$ from the first client's uploaded state...")

These three are the paper's honest disclosures. Item 3 in particular is the
result of the C20 correction, which found that the paper's own measurement
overcharges the strongest baseline. Removing it to save space would undo that
work. If the page count still does not fit after every other cut, cut prose from
the Discussion instead.

### L5. Compress Section 5.1 to one paragraph (saves ~90 words)

File: `body.tex`, Section 5.1 "Implications for Federated LLM Deployment".

REPLACE the entire subsection body with:

```
In mobile or IoT federated settings where uplink bandwidth is scarce and metered, cutting round-trip communication by 30 to 40\% for a held-out instruction-following loss cost of 0.0063 can change the economic viability of federated fine-tuning. Whether that trade is acceptable is a deployment decision rather than a universal one, and the frontier in Section~\ref{sec:frontier} is intended to let a practitioner make it explicitly: the same measurements show that accepting a further 21.3 points of savings costs roughly five times more quality per point. \citet{kairouz2021advances} identify communication as a fundamental open problem in federated learning at scale, and measured byte-level reporting is a prerequisite for addressing it.
```

### L6. Trim the Introduction (saves ~150 words)

File: `body.tex`, Section 1.

**Delete the roadmap paragraph entirely.** IEEE house style discourages
signposting, and it duplicates the table of contents.

FIND and delete:
```
Section~\ref{sec:related} reviews related work. Section~\ref{sec:method} describes the protocol and ReverseAdaptive. Section~\ref{sec:experiments} presents experiments and results. Section~\ref{sec:discussion} provides discussion. Section~\ref{sec:limitations} lists limitations. Section~\ref{sec:conclusion} concludes.
```

**From the "Headline numbers" paragraph, delete the final sentence**, which
refers to downstream benchmarks now moved to supplemental:
```
All federated methods on LLaMA-3.2-3B cluster within 0.5 percentage points across the four downstream benchmarks.
```

### L7. If still over budget

Compile and measure first. If more is needed, cut in this order and stop as soon
as it fits:

1. Section 5.2 paragraph 1 (restates what Two-Phase requires; the point is made
   again in paragraph 2)
2. Section 4.4 paragraph 4 (the non-IID Dolly run at alpha=0.5, which reports no
   quality cost anyway)
3. Section 5.3 paragraph 4 (the closed-form lattice derivation; move the
   derivation to supplemental and keep the one-sentence result)

**Never cut to fit by reducing font size, shrinking margins, adding negative
`\vspace`, or moving a `\DeclareMathSizes`. IEEE prescreening rejects for this.**

### Commit STEP 4

```bash
wc -w docs/ojcs-submission/body.tex   # target: at or under ~5,100
python3 scripts/verify_numbers.py --analysis-dir analysis
git add -A
git commit -m "STEP 4: L1-L6 length reduction for 12-page IEEE limit; downstream block and reproducibility guarantee moved to supplemental"
```

---

## STEP 5: LaTeX conversion to IEEEtran

### F1. Get the template

Download the IEEE **journal** article template (not conference, not Access) from
the IEEE Template Selector at
`journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/`.

Place `IEEEtran.cls` and `IEEEtran.bst` in `docs/ojcs-submission/`.

### F2. Preamble

File: `docs/ojcs-submission/main.tex`

DELETE these lines:
```
\documentclass[preprint,review,12pt]{elsarticle}
\journal{Neurocomputing}
\usepackage{float}
\newcommand{\fix}{\marginpar{FIX}}
\newcommand{\new}{\marginpar{NEW}}
\raggedbottom
```

REPLACE the documentclass line with:
```
\documentclass[journal]{IEEEtran}
```

KEEP: `graphicx`, `booktabs`, `amsmath`, `amssymb`, `algorithm`,
`algpseudocode`, `multirow`, `siunitx`, `hyperref`, `url`, and
`\input{math_commands.tex}`.

**Then hunt down leftover margin notes:**
```bash
grep -rn "\\\\fix\|\\\\new{" docs/ojcs-submission/*.tex
```
Every hit must be deleted. A submitted PDF with FIX or NEW in the margin looks
careless and is a prescreening risk.

Also remove all `[H]` float specifiers, which require the `float` package:
```bash
grep -n "\[H\]" docs/ojcs-submission/*.tex
```
Replace `[H]` with `[t]` throughout.

### F3. Front matter

DELETE the `\begin{frontmatter}` / `\end{frontmatter}` wrapper.

REPLACE the author block:
```
\author[ind]{Jerry Adams Franklin}
\ead{jerry.adamsf@gmail.com}
\address[ind]{Independent Researcher, Rochester, NY, USA}
```
WITH:
```
\author{Jerry~Adams~Franklin,~\IEEEmembership{Member,~IEEE}% remove membership if not a member at submission
\thanks{J. A. Franklin is an Independent Researcher, Rochester, NY 14450 USA (e-mail: jerry.adamsf@gmail.com). ORCID: 0009-0006-8470-8349.}}
```

If IEEE membership is not active at submission time, delete
`,~\IEEEmembership{Member,~IEEE}` and leave the name alone. Do not claim a
membership grade you do not hold.

REPLACE the keyword block:
```
\begin{keyword}
Federated learning \sep Low-rank adaptation \sep Communication efficiency \sep
Parameter-efficient fine-tuning \sep Large language models
\end{keyword}
```
WITH:
```
\begin{IEEEkeywords}
Federated learning, low-rank adaptation, communication efficiency, parameter-efficient fine-tuning, large language models.
\end{IEEEkeywords}
```

Add after `\begin{document}`:
```
\maketitle
```

### F4. Rewrite the abstract

**Do this by hand. Do not let an LLM generate it. Present the draft to the author
for approval before committing.**

Hard constraints from IEEE Computer Society guidelines:

- 100 to 200 words. Target 185.
- **No mathematical expressions.** The current abstract contains `$\tau$`, `$K$`,
  `$A$`, and `$30.0 \pm 4.0$`. All must go.
- **No bibliographic references.**

Content priority, keep in this order:
1. The byte-measurement contribution and the transition-round omission in prior
   accounting
2. Five methods on one measured frontier, and the knee
3. 40.5 percent savings at 0.0063 held-out cost on TinyLlama-1.1B
4. Beats FFA-LoRA by 0.0182 held-out, more than twenty times the largest seed
   standard deviation

Cut: the Dolly sentence, the LLaMA-3.2-3B sentence, the 0.000012 comparison, the
code-release sentence.

**Item 4 must survive.** It is the strongest empirical claim in the paper and the
one an editor screening for significance will weigh.

Word-count check before committing:
```bash
# paste the abstract into /tmp/abs.txt first
wc -w /tmp/abs.txt   # must be between 100 and 200
grep -c '\$' /tmp/abs.txt   # must be 0
```

### F5. Citations

`body.tex` has 69 `\citep` and 3 `\citet`. `appendix.tex` has 3 `\citep`.
IEEEtran does not load natbib.

```bash
cd docs/ojcs-submission
sed -i 's/\\citep{/\\cite{/g' body.tex appendix.tex supplemental.tex
grep -n "\\\\citet\|\\\\citealt" body.tex appendix.tex supplemental.tex
```

The `\citet` and `\citealt` hits must be rewritten by hand, because they use the
citation as a noun. Pattern:

- `\citet{kairouz2021advances} identify` becomes `Kairouz et al. \cite{kairouz2021advances} identify`
- `(Alpaca-3k, \citealt{taori2023alpaca}, and Dolly-15k, \citealt{dolly2023})` becomes `(Alpaca-3k \cite{taori2023alpaca} and Dolly-15k \cite{dolly2023})`

Then:
```
\bibliographystyle{elsarticle-num-names}  ->  \bibliographystyle{IEEEtran}
```

Verify: `grep -c "citep\|citet\|citealt" docs/ojcs-submission/*.tex` must print 0
for every file.

### F6. Back matter

DELETE from `main.tex`:
```
\section*{CRediT authorship contribution statement}
\textbf{Jerry Adams Franklin:} Conceptualization, Methodology, Software,
Validation, Formal analysis, Investigation, Data curation, Writing -- original
draft, Writing -- review and editing, Visualization.
```
(CRediT is an Elsevier convention. IEEE does not use it.)

DELETE the standalone `\section*{Data availability}` and
`\section*{Declaration of generative AI...}` sections. Replace both with a single
Acknowledgment section placed before the bibliography:

```
\section*{Acknowledgment}
All code, configuration files, and analysis scripts are publicly available at
https://github.com/jerryadamsfranklin/fedlora-protocols. Adapter checkpoints are
not released. The Alpaca and Dolly-15k datasets are publicly available from
their respective sources.

The author used Claude (Anthropic) for grammar and language editing only. All
technical content, analysis, and interpretations are the author's own.
```

Reason: IEEE policy requires AI-use disclosure specifically in the
acknowledgments section, not as a separate section.

KEEP the competing-interest declaration as its own `\section*`.

### F7. Split appendices into a standalone supplemental document

Create `docs/ojcs-submission/supplemental.tex` as a **standalone compilable
document**, not an `\input` of the main file. IEEE requires appendices to be
submitted as separate supplemental files; if they are not designated as such at
submission, the paper is returned.

Contents, in order:
1. Appendix A Hyperparameter Details
2. Appendix B Per-Seed Full Results (Tables B.8, B.9)
3. Appendix C Statistical Tests (Table C.10, with the C32 row)
4. Appendix D Reproducibility Checklist
5. Appendix E Cross-Backend Verification (Tables E.11, E.12)
6. The downstream-benchmark block moved by L1
7. The reproducibility-guarantee text moved by L2

Give it its own preamble, its own `\bibliography{main}`, and a title line reading
"Supplemental Material for: Adaptive Phase-Switching for Communication-Efficient
Federated LoRA Fine-Tuning".

Then delete `appendix.tex` and remove `\input{appendix.tex}` from `main.tex`.

**Fix the doubled-appendix artifact.** elsarticle produced strings like
"Appendix Appendix D" in the current PDF:
```bash
grep -n "Appendix~\\\\ref\|Appendix Appendix" docs/ojcs-submission/*.tex
```
Every body-text reference to an appendix must become a plain-language reference
to the supplemental material, because cross-document `\ref` will not resolve.
Example: `see Appendix~\ref{app:repro}` becomes
`see the supplemental material`.

Verify: the compiled `main.pdf` contains no appendix and no `??` unresolved
references.

### F8. Figures

Keep figures embedded at first callout, sized as intended. Separate figure files
are a final-publication requirement, not a submission one, and incorrectly sized
figures are returned for reformatting.

Figure 4 moves to `supplemental.tex` under L1.

Change `\includegraphics[width=0.85\linewidth]` to `\includegraphics[width=\columnwidth]`
for single-column figures. For the two-panel figures (fig4, fig6), use
`figure*` with `\includegraphics[width=\textwidth]`.

### Commit STEP 5

```bash
cd docs/ojcs-submission
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplemental && bibtex supplemental && pdflatex supplemental && pdflatex supplemental
grep -c "??" main.log   # investigate any unresolved references
cd ../..
git add -A
git commit -m "STEP 5: F1-F8 IEEEtran conversion, appendices split to standalone supplemental"
```

**Report the page count of `main.pdf` before proceeding.** If it exceeds the
limit confirmed in STEP 6, return to L7.

---

## STEP 6: Pre-submission verification

Run all of these. Report each result.

```bash
# 1. Numbers still hold after all editing
python3 scripts/verify_numbers.py --analysis-dir analysis

# 2. Page count
pdfinfo docs/ojcs-submission/main.pdf | grep Pages

# 3. Writing mandate violations
grep -n -- "---" docs/ojcs-submission/*.tex
grep -n "[“”‘’]" docs/ojcs-submission/*.tex
grep -nE "^(First|Furthermore|Moreover|Additionally)," docs/ojcs-submission/*.tex
grep -rn "\\\\fix\|\\\\new{" docs/ojcs-submission/*.tex

# 4. No Elsevier residue
grep -rin "neurocomputing\|elsarticle\|citep\|citet\|CRediT" docs/ojcs-submission/

# 5. Affiliation and name
grep -n "Independent Researcher" docs/ojcs-submission/main.tex
grep -n "Jerry" docs/ojcs-submission/main.tex

# 6. Figures regenerable
rm -f figures/fig*.pdf && python3 scripts/generate_paper_figures.py && ls figures/*.pdf | wc -l
```

Every one of items 3 and 4 must return zero hits.

Then, manually and outside the repo:

- Upload `main.tex` to the **IEEE LaTeX Analyzer** at `latexqc.ieee.org`. Fix
  whatever it reports.
- Run `main.bib` through the **IEEE Reference Preparation Assistant** at
  `refassist.ieee.org`.
- Run the full text through a grammar checker. IEEE prescreening rejects for poor
  grammar before review reaches a reviewer.

### Commit STEP 6

```bash
git add -A
git commit -m "STEP 6: pre-submission verification passed; LaTeX Analyzer and RefAssist clean"
```

---

## STEP 7: Cover letter

Rewrite `docs/ojcs-submission/cover_letter.md`.

**Lead with the measurement contribution.** Paragraph 1 is about byte-level
instrumentation, the transition-round cost that parameter-count accounting
cannot express, and the grouped-query-attention effect on achievable savings.
The adaptive aggregator comes second.

Rationale: OJ-CS screens for high-impact results and has no revision option. An
aggregator variant invites an "incremental scheduling change" reading. The
measurement framing is what makes the five-method frontier non-obvious.

Must include:
- A statement that appendices are submitted as supplemental material
- If the arXiv preprint is already posted, its arXiv ID and one sentence noting
  it is a preprint, not a prior publication. IEEE requires disclosure of any
  portion appearing elsewhere.

**Must not include:** any reference to prior submission history. Standing
instruction from the author.

---

## Open questions for the author, answer before STEP 4

1. **Page limit.** The OJ-CS Author Information page blocks automated retrieval.
   The 12-page figure used throughout this plan comes from the general IEEE
   Computer Society regular-paper policy. Confirm the OJ-CS number manually at
   `computer.org/csdl/journal/oj/write-for-us/75639` and report it. The size of
   the STEP 4 cut depends on the answer.

2. **Editor-in-chief.** Sources disagree on whether the current EIC is Song Guo
   or Vincenzo Piuri. Confirm from the journal masthead before sending any
   correspondence.

3. **IEEE membership status at submission**, which determines whether the
   `\IEEEmembership` tag in F3 stays or goes.

## Stop conditions

Stop and ask the author if any of the following occur:

- `verify_numbers.py` reports any failure
- A FIND string in STEP 2 does not match exactly
- The page count still exceeds the limit after L1 through L7
- A reference in C34 does not resolve, or resolves to different metadata
- Any cut would require removing one of the three protected Limitations
  paragraphs named in L4
