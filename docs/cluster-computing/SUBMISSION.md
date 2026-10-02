# Cluster Computing submission checklist

Upload these files to the Springer submission system (one manuscript PDF
plus the full LaTeX source set). Build the flat source first:

```bash
cd docs/cluster-computing
./flatten.sh                    # syncs Fig1–Fig4 from ../../figures/ and writes main_flat.tex
latexmk -pdf main_flat.tex      # produces main_flat.pdf (the PDF you upload)
./flatten.sh --check            # fails if split sources are newer than main_flat.tex
latexmk -pdf main.tex           # optional local edit build; main.pdf is gitignored
```

## Files to upload

| File | Role |
|---|---|
| `main_flat.tex` | Single-file manuscript source (do **not** upload the split `main.tex` / `\input` tree) |
| `sn-jnl.cls` | Springer Nature journal class |
| `sn-mathphys-num.bst` | Numbered MathPhys bibliography style used by this build |
| `main.bib` | Bibliography database |
| `main_flat.bbl` | Pre-generated bibliography (submit with the source; the name must match `main_flat.tex`) |
| `figures/Fig1.pdf` … `figures/Fig4.pdf` | Manuscript figures |
| `main_flat.pdf` | Compiled PDF from `main_flat.tex` (19 pages; this is the PDF to upload) |

## Submission-interface notes

- Enter **author contributions** and **competing interests** in the submission interface as well as in the manuscript Declarations section; only the interface copy reaches the published article.
- Choose the **regular track**, not a guest-edited collection or special issue.
- After acceptance, select the **subscription** publishing route so there is no APC.

## Do not upload

- `body.tex`, `appendix_body.tex` (already inlined in `main_flat.tex`)
- `main.pdf` (optional split-source build; gitignored so it cannot drift from `main_flat.pdf` in the repo)
- Intermediate LaTeX auxiliaries (`.aux`, `.log`, `.out`, …)
- Unused figure generators under repo-root `figures/` (`fig3_cumulative_comm`, `fig4_downstream_accuracy`)
