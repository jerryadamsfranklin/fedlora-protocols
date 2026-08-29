# Cursor Instructions: Build the Neurocomputing Submission

You are applying a prepared revision to a LaTeX manuscript. Every replacement text
you need is in `NEURO_CHANGE_LEDGER.md`. Your job is mechanical application, not
authorship.

## Rules, read these first

1. **Apply blocks verbatim.** Do not reword, reformat, shorten, or "improve" any
   text from the ledger. The wording is the result of extensive verification and
   several sentences are deliberately hedged. If a sentence reads awkwardly to
   you, leave it.
2. **Never invent or adjust a number.** If a number in a ledger block appears to
   conflict with the manuscript, stop and report it. Do not reconcile it yourself.
3. **Anchor by string, not by line number.** Every edit in the ledger gives an
   `old_str`. Find that exact string. If it does not appear exactly once, stop and
   report which block and how many matches you found.
4. **Do not apply the four superseded blocks** listed in the DO NOT APPLY table.
5. **House style, enforced on anything you touch:** no em-dashes, no curly quotes,
   no signposting words (First, Furthermore, Moreover, Additionally).
6. **Report, do not guess.** At the end, list every block you applied, every block
   you could not apply, and why.

## File layout

Create `docs/neurocomputing-submission/` alongside the existing
`docs/tmlr-submission/`. Do not modify anything in `docs/tmlr-submission/`; that
version corresponds to a public OpenReview submission and is kept for provenance.

Target contents:

```
docs/neurocomputing-submission/
    main.tex            <- copy, then apply G0 blocks
    body.tex            <- copy from docs/tmlr-submission/, then apply blocks
    appendix.tex        <- copy from docs/tmlr-submission/, then apply blocks
    main.bib            <- copy, then add three entries
    math_commands.tex   <- copy unchanged
    highlights.txt      <- create new, from G0.4
    figures/            <- copy the six existing PDFs, then add frontier.pdf
```

**`main.tex`, `main.bib`, and `math_commands.tex` are not in this repository.**
Jerry must supply them. If they are absent, stop and ask before proceeding.

---

# STAGE 1: mechanical changes, then compile

Do all of Stage 1, compile, and report the log before starting Stage 2. The point
is to surface structural errors before twenty more edits are in the file.

## 1.1 Preamble and frontmatter

Apply from the ledger, in `main.tex`:

- **G0.1** replaces the TMLR preamble. Remove `\usepackage{tmlr}`.
- **G0.2** replaces `\maketitle` and the `abstract` environment with an
  elsarticle `frontmatter` block.
- **G0.3** inserts end matter before `\bibliography{main}`.
- Change `\bibliographystyle{tmlr}` to `\bibliographystyle{elsarticle-num}`.
- Keep `\input{appendix.tex}` after the bibliography. That ordering is correct
  for Elsevier.

Placeholders in G0.2 and G0.3 that Jerry must fill: city, country, and the
repository URL. Leave them as written if he has not supplied them, and list them
in your report.

## 1.2 Add seven labels

These do not exist yet and will otherwise produce undefined-reference errors.
Add each `\label{...}` immediately after the sectioning command it names.

| Label | Add to |
|---|---|
| `\label{sec:setup}` | `\subsection{Setup}` in Section 4 |
| `\label{sec:implementation}` | `\subsection{Implementation Details}` in Section 3 |
| `\label{sec:baseregression}` | `\subsection{The Base-Model Regression Observation}` |
| `\label{sec:limitations}` | `\section{Limitations}` |
| `\label{sec:frontier}` | the Section 4.2 subsection (added by block G5.3) |
| `\label{sec:dolly}` | the Dolly subsection (added by block G6.1) |
| `\label{app:crossbackend}` | the new Appendix E (added by block G13.2) |

The last three arrive with their blocks in Stage 2. Add the first four now.

## 1.3 Bibliography

In `main.bib`, add three entries from `neuro_new_refs.bib`:

- `dolly2023`
- `yan2026fedsrd`
- `ramesh2026florist`

In `neuro_new_tables.tex`, if you use that file at all, change `zhang2024fedit`
to `zhang2024federatedgpt`. That key does not exist in `main.bib`.

Do not use the key `conover2023dolly` anywhere. The correct key is `dolly2023`.

## 1.4 Compile and report

Compile with `pdflatex` twice, then `bibtex`, then `pdflatex` twice more.

Two failures are expected and are not content problems:

- **Overfull hbox on Table 1.** Two rows are deliberately split across two lines
  by blocks G2.3 and G2.5. If the box is still overfull, report the measurement;
  do not fix it by rewording.
- **`lineno` and `algorithm` floats.** The `review` option loads `lineno`, which
  interacts badly with floating algorithm environments. If line numbers break
  around Algorithms 1 or 2, wrap those two floats in
  `\begin{linenomath}...\end{linenomath}`, or change `[t]` to `[H]` on those two
  floats only.

Report the full log. Stop here.

---

# STAGE 2: apply the content blocks

Apply in this order. All are in `NEURO_CHANGE_LEDGER.md` under the matching
heading. Order matters only where noted.

## DO NOT APPLY

| Block | Reason |
|---|---|
| **G1.4** | Superseded by G7.1. Applying both double-edits Section 5.2. |
| **G0.2a**, **G0.2b** | Already folded into G0.2. Record only. |
| **G10.2** | Already folded into G7.1. Record only. |
| **G8.4** | Already folded into G8.1. Record only. |
| **G8.3** | Record only. The Dolly key is already correct in G6. |

## Section 3, Method (`body.tex`)

- **G1.1** switch-rule prose
- **G1.2** Algorithm 2 condition, replaces one line with two
- **G1b.1** Algorithm 1 transition branch, replaces four lines with four
- **G1b.2** new paragraph, frozen-A provenance, in Section 3.4
- **G3.1** B-only fraction and target list, Section 3.1
- **G3.2** byte-tracking dtype, Section 3.4
- **G3.3** partition mechanics, Section 3.4

## Table 1 (`body.tex`)

Six row-level edits, all independent:

- **G2.1** target modules
- **G2.2** dtype, one row becomes two
- **G2.3** hardware, one row becomes two
- **G2.4** held-out eval row added
- **G2.5** ReverseAdaptive defaults, one row becomes two

## Section 4, Experiments (`body.tex`)

Order within this section matters: G4b defines `sec:heldout`, which G5 and G6
reference.

- **G4.1** compute statement
- **G4.2** new paragraph, baselines
- **G4.3** new paragraph, prompt format
- **G5.1** new paragraph, backend provenance
- **G4b** new subsection, held-out metric, inserted immediately before the
  Section 4.2 subsection heading
- **G5.3** Section 4.2 heading and opening paragraph, defines `sec:frontier`
- **G5.4** replaces the whole `fig:pareto` float with `fig:frontier`
- **G5.5** three interpretation paragraphs
- **G5.6** replaces the whole `tab:tinyllama_iid` float with `tab:frontier`
- **G5.7** statistical paragraph
- **G5.8** the exact-reproduction claim
- **G6.1** new subsection, Dolly replication, defines `sec:dolly`
- **G6.2** new table `tab:dolly`
- **G5.2** ablation caption
- **G8b.2** Table 3 caption
- **G17.1** Section 4.6 scale validation

Also in Section 4.3 body text: the phrase "1.4 percentage points" appears
outside the Table 3 caption. Change it to "1.0 percentage points" and add
", across the three informative benchmarks" after it. Report where you found it.

## Section 5, Discussion (`body.tex`)

- **G18.1** Section 5.1, full subsection replacement
- **G7.1** Section 5.2, full replacement including the heading. The heading
  changes from "Why Adaptive Outperforms Fixed-$K$" to "Choosing an Operating
  Point". Do not apply G1.4.
- **G8.1** Section 5.3, full replacement including the heading
- **G9.1** Section 5.4, full replacement
- **G8b.1** Section 5.5, full replacement, defines `sec:baseregression`

## Limitations and Conclusion (`body.tex`)

- **G10.1** full Limitations replacement, defines `sec:limitations`
- **G14.1** conclusion, all body paragraphs and the release line
- **G14.2** delete the `\subsubsection*{Acknowledgments}` block entirely

## Section 1 and 2 (`body.tex`)

Apply last, so contribution claims match the final numbers.

- **G16.1** prior-work paragraph
- **G16.2** contribution one
- **G16.3** contribution two
- **G16.4** contribution three, retitled
- **G16.5** headline numbers
- **G16.6** release line
- **G15** two edits in Section 2.2, appending forward references to the FedIT
  and FFA-LoRA paragraphs

## Appendices (`appendix.tex`)

- **G12** Appendix A, gradient clipping sentence
- **G11.3** Appendix B opening, insert after `\label{app:perseed}`
- **G11.4** append to the G11.3 paragraph
- **G11.1** replaces the whole `tab:b1` float
- **G11.2** `tab:b2` caption
- **G19.1** Appendix C, prose and the whole `tab:d1` float
- **G13.1** Appendix D, four checklist entries
- **G13.2** new Appendix E, defines `app:crossbackend`

**Appendix E must come after Appendix C**, because `app:stats` is referenced from
blocks G5.6 and G5.7 and `app:crossbackend` from G4.1, G5.1, G5.8, G9.1, G10.1
and G13.

---

# STAGE 3: figures, scripts, highlights

## 3.1 Generate the new figure

`make_frontier_figure.py` produces `figures/frontier.pdf`, referenced by block
G5.4. Before running it, make two corrections:

- Change `st.pstdev` to `st.stdev`. The manuscript reports sample standard
  deviation throughout; this is a hard convention.
- If the script or `neuro_new_tables.tex` says "population standard deviation"
  in any caption, change it to "sample standard deviation".

Then run it with `--analysis-dir analysis --out docs/neurocomputing-submission/figures/frontier.pdf`.

## 3.2 Replace the verifier

`docs/tmlr-submission/verify_numbers.py` is an early version. Copy the current
`verify_numbers.py` into `docs/neurocomputing-submission/` and also into
`scripts/`. Run it:

```
python verify_numbers.py --analysis-dir analysis
```

It must report **174 passed, 0 failed, 0 errored**. If anything fails, stop and
report which assertion and what it computed. Do not edit the manuscript to match
the script or the script to match the manuscript.

## 3.3 Highlights

Create `docs/neurocomputing-submission/highlights.txt` containing exactly the
five lines from block G0.4, one per line, with no numbering and no trailing
punctuation. Each must be under 85 characters. Verify the character counts and
report them.

## 3.4 Repository hygiene

Two edits outside the manuscript:

- **`config/base_config_4layers.yaml`**: add a header comment stating that
  `target_modules` was not passed to the model constructor before commit
  `8cddb8b`, that all reported runs adapt `q_proj` and `v_proj` only, and that
  the measured byte counts are the evidence. Do not rename the file; renaming
  breaks the `_inherit` chain in every downstream config.
- **`src/aggregation/ffa_lora.py`**, method `get_communication_cost`: the comment
  `# float16 * 2 (upload + download)` is wrong. The returned value is one client,
  one direction, at float32. Correct the comment and add a docstring line stating
  that this method is not used for reported byte accounting, which is instrumented
  at the transport layer in `server.py`. Do not delete the method.

---

# STAGE 4: verification greps

Run these against the assembled `body.tex`, `appendix.tex` and `main.tex`. Report
every hit with its line and surrounding sentence. Do not fix anything yourself;
these are for review.

## 4.1 The `p = 0.997` grep

```
grep -n "0\.997" body.tex appendix.tex main.tex
```

Expected: zero hits stating or implying that the test establishes equivalence,
indistinguishability, or confirmation. The value may legitimately appear in
Appendix C's table and in Section 4.6, but only alongside the confidence interval
and the effect-size framing. Report every hit and its wording.

## 4.2 TMLR artifacts

```
grep -n -i "under review\|anonymous\|anonymized\|double-blind\|suppressed for" body.tex appendix.tex main.tex
```

Expected: zero hits.

## 4.3 The never-write list

```
grep -n -i "four-target\|progressive freezing\|partial freezing\|label skew\|label-skew\|no external GPU\|population standard deviation" body.tex appendix.tex main.tex
```

Expected: zero hits. Each of these is a factual error about what the code does.

## 4.4 Style

```
grep -n "—\|“\|”\|‘\|’" body.tex appendix.tex main.tex
grep -n "^First,\|^Furthermore,\|^Moreover,\|^Additionally," body.tex appendix.tex main.tex
```

Expected: zero hits.

## 4.5 Stale figures

Confirm that `1.4 percentage points` and `0.4 percentage points` no longer appear
for the TinyLlama and LLaMA-3.2-3B clustering claims respectively. They should
now read 1.0 and 0.5.

```
grep -n "1\.4 percentage\|0\.4 percentage" body.tex appendix.tex main.tex
```

Expected: zero hits.

## 4.6 Undefined references

After the final compile, grep the log:

```
grep -n "undefined\|Undefined\|multiply defined\|Overfull" *.log
```

Report all. Undefined references indicate a missing label from Stage 1.2.

---

# Final report

Produce a summary containing:

1. Every block applied, by ID.
2. Every block not applied, with the reason.
3. Any `old_str` that did not match exactly once, with the match count.
4. The verifier output line.
5. All Stage 4 grep hits.
6. The compile log's error and warning summary.
7. The five highlight character counts.
8. Any placeholder still unfilled: city, country, repository URL.

Do not attempt to resolve items 3, 5, or 8 yourself.
