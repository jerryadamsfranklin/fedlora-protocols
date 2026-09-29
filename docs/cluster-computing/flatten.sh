#!/usr/bin/env bash
# Sync Springer figures from the repo figures/ source of truth, then flatten
# main.tex + body.tex + appendix_body.tex into one submission .tex.
# Keep the split sources for editing; upload main_flat.tex / main_flat.pdf.
#
# Usage:
#   ./flatten.sh           # sync figures and rewrite main_flat.tex
#   ./flatten.sh --check   # fail if split sources are newer than main_flat.tex
set -euo pipefail
cd "$(dirname "$0")"

SOURCES=(main.tex body.tex appendix_body.tex)
FLAT=main_flat.tex

if [[ "${1:-}" == "--check" ]]; then
  if [[ ! -f "$FLAT" ]]; then
    echo "error: $FLAT missing; run ./flatten.sh first" >&2
    exit 1
  fi
  flat_mtime=$(stat -f %m "$FLAT" 2>/dev/null || stat -c %Y "$FLAT")
  stale=()
  for src in "${SOURCES[@]}"; do
    src_mtime=$(stat -f %m "$src" 2>/dev/null || stat -c %Y "$src")
    if (( src_mtime > flat_mtime )); then
      stale+=("$src")
    fi
  done
  if ((${#stale[@]} > 0)); then
    echo "error: $FLAT is stale relative to: ${stale[*]}" >&2
    echo "run ./flatten.sh then latexmk -pdf main_flat.tex before uploading" >&2
    exit 1
  fi
  echo "ok: $FLAT is at least as new as ${SOURCES[*]}"
  exit 0
fi

# Figure mapping (checksum-verified against the current manuscript build):
#   figures/fig1_frontier.pdf          -> Fig1.pdf  (Fig. 1)
#   figures/fig2_convergence.pdf       -> Fig2.pdf  (Fig. 2)
#   figures/fig5_threshold_ablation.pdf-> Fig3.pdf  (Fig. 3)
#   figures/fig6_scale_validation.pdf  -> Fig4.pdf  (Fig. 4)
# fig3_cumulative_comm and fig4_downstream_accuracy are generated but unused.
ROOT="$(cd ../.. && pwd)"
FIG_SRC="$ROOT/figures"
FIG_DST="figures"
mkdir -p "$FIG_DST"
cp "$FIG_SRC/fig1_frontier.pdf"          "$FIG_DST/Fig1.pdf"
cp "$FIG_SRC/fig2_convergence.pdf"       "$FIG_DST/Fig2.pdf"
cp "$FIG_SRC/fig5_threshold_ablation.pdf" "$FIG_DST/Fig3.pdf"
cp "$FIG_SRC/fig6_scale_validation.pdf"  "$FIG_DST/Fig4.pdf"
# Drop any leftover lowercase copies so the submission folder stays clean.
rm -f "$FIG_DST"/fig*.pdf

python3 <<'PY'
from pathlib import Path

def strip_trailing_newline(s: str) -> str:
    return s[:-1] if s.endswith("\n") else s

main = Path("main.tex").read_text()
body = strip_trailing_newline(Path("body.tex").read_text())
appendix = strip_trailing_newline(Path("appendix_body.tex").read_text())

if "\\input{body.tex}" not in main:
    raise SystemExit("main.tex: missing \\input{body.tex}")
if "\\input{appendix_body.tex}" not in main:
    raise SystemExit("main.tex: missing \\input{appendix_body.tex}")

flat = main.replace("\\input{body.tex}", body, 1)
flat = flat.replace("\\input{appendix_body.tex}", appendix, 1)
if "\\input{" in flat:
    raise SystemExit("flatten left residual \\input{...}")

out = Path("main_flat.tex")
out.write_text(flat)
print(f"wrote {out} ({out.stat().st_size} bytes)")
print("synced figures: Fig1.pdf Fig2.pdf Fig3.pdf Fig4.pdf")
PY
