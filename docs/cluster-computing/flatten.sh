#!/usr/bin/env bash
# Flatten main.tex + body.tex + appendix_body.tex into one submission .tex.
# Keep the split sources for editing; upload main_flat.tex to Springer.
set -euo pipefail
cd "$(dirname "$0")"
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
PY
