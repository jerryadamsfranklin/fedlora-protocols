# Neurocomputing Phase 1 — early backend-match gates

Phase 0 CUDA validation (IID ReverseAdaptive seed 42) is complete. Before running the
**full** Phase 1 suite unattended, treat the following as **extended validation**
gates. Do **not** continue the remainder of the batch until both comparisons pass
`scripts/verify_backend_match.py` at `atol=0.01`, `rtol=0`.

## Gate 1 — first Non-IID run

- Pick the first Non-IID experiment scheduled in Phase 1 (e.g. ReverseAdaptive
  Non-IID α=0.5 or α=0.1, seed 42 — same config as the existing MPS reference).
- After it completes on CUDA, immediately run:

```bash
python3 scripts/verify_backend_match.py \
  --mps <mps_reference_results.json> \
  --cuda <cuda_results.json>
```

- **PASS:** all reported checks within `atol=0.01`.
- **FAIL:** stop the Phase 1 batch; investigate before more GPU hours.

## Gate 2 — first FLoRA (or Two-Phase K=8) run

- Prefer **FLoRA IID seed 42** if that is the natural baseline; otherwise
  **Two-Phase K=8 IID seed 42**.
- Same verifier + tolerance as Gate 1 against the matching MPS `results.json`.
- **PASS required** before continuing the rest of the ~34-run suite unattended.

## Reporting

Report both verifier outputs (PASS/FAIL table) before authorizing the remainder
of Phase 1. A.1 (HuggingFace LLaMA-3.1-8B access) is **not** a Phase 1 blocker;
it applies starting at Phase 2 (8B scale).
