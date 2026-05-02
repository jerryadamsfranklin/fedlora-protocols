# Stage 1: Path B (TMLR bidirectional communication accounting)

**Purpose:** Track decisions and deferred work that are *not* part of the stable Path B
implementation spec. The local `docs/TMLR_PATH_B_IMPLEMENTATION_GUIDE.md` (gitignored)
should stay a clean checklist; ongoing caveats and “revisit at Stage N” items live
here.

**Source:** Stage 1 validation review (end-to-end server simulation), 2026-05-01.

---

## Path A vs Path B (reviewer recommendation)

**Path A — default for Stage 1 verification runs:** Keep the current download tracker
as implemented. All methods use the same accounting, so comparisons stay fair. Proceed
with the six IID Two-Phase runs; reported metrics will match the reviewer’s simulation.

**Path B — defer until after Stage 1 verification:** Optionally adjust accounting so
cumulative download does not count a “phantom” broadcast prepared at the **end of the
final round** (bytes that would never be consumed because there is no next round). That
change would move headline savings closer to a strict “bytes on wire” interpretation
(~1.2 percentage points lower than Path A for the K=8 IID scenario in the review).

**When to decide:** Revisit at **Stage 5** (methods / claims writing). Do **not** block
Stage 1–2 work on Path B.

---

## Known numerical caveat (do not misinterpret results)

- The upload path and the download path switch to B-only **one round apart** by design:
  download at end-of-round *R* counts what is broadcast to round *R+1* (look-ahead
  tracking).
- Example trajectory for Two-Phase K=8 (IID, per reviewer table): round 9 still has a
  **full** upload delta while download delta for that round can already reflect **B-only**
  (broadcast staged for round 10).

Acceptance checks for the six runs (upload/download deltas, cumulative MB at round 15,
loss tolerances) are defined in the reviewer’s Stage 1 report; use that as the source of
truth when comparing `results.json`.

---

## Path B implementation sketch (when you choose to do it)

- **Goal:** Skip counting download bytes for the broadcast stashed after the **last**
  training round when no clients will train again.
- **Scope:** Small, localized change in `FederatedServer.train` (or equivalent) — *one
  conditional* around the post-aggregation download accounting path; no change to
  aggregator semantics unless required for clarity.

---

## Quick links

- Visual sanity check during runs: log line `[B-only broadcast active starting next round]`
  should appear once at the expected round boundary (e.g. end of round 9 for K=8 IID).
