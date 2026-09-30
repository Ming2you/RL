# FixRound1 Scoped Independent Review

Reviewed 2026-09-30 against `fix1-brief.md`, the original R1 finding verbatim,
the appended FixRound1 report, and `fix1.diff` against the exact previously
reviewed files.

**R1: ADDRESSED. SPEC: PASS. QUALITY: PASS.**

No actionable new breakage was found in this diff. The original R1 review
blocker is cleared for this candidate identity. Actual pilot authorization
remains with the parent; no pilot was run by this review.

## R1 Resolution

- `run_wave.py:113-117`: a known nonzero launcher exit raises while the child
  remains in `active`. The existing `finally` publishes the attempt ABORT at
  line 131 before any launcher wait or actual-worker drain at lines 134-139.
  This also covers failure of the only remaining active child.
- `run_wave.py:53-68` and `116-117`: `worker_exited` performs one nonblocking
  identity check. A successful exited launcher whose actual worker remains
  live stays active, allowing other children to be polled in the same cycle
  and all-slot STOP to be checked at line 108 on the next cycle. Normal
  polling retains the existing five-second cadence at line 127.
- `run_wave.py:59-66`, `71-73`, and `128-144`: the prior actual-worker/launcher
  creation-identity checks and UNKNOWN errors survive the extraction. The
  final drain occurs after shared ABORT, waits before release, and continues
  attempting sibling drains after an OSError before propagating it. No early
  ownership release, forced termination, duplicate dispatch or retry was added.
- `test_workflow.py:222-312`: the three parameterized regressions cover nonzero
  exit, later sibling-slot STOP, and another child's failure while the first
  worker remains live. Assertions require ABORT before actual death/release,
  both launcher drains, matching launcher/worker death before release, no
  completed-output validation on the aborted path, and preserved STOP.

The change is confined to runner polling and its tests. Treatment, scientific
gates, stage widths, completion skipping, and physical interfaces are unchanged.

## Evidence Checked Without Execution

All 17 `before-fix1` snapshot hashes match the originally reviewed `887f768a`
evidence. Only `run_wave.py` and `test_workflow.py` differ in the current
candidate; all 17 current source hashes match the final `bbd349b3` evidence.
The four before/after Git blob IDs match the supplied diff's index entries.

Parsed evidence and JUnit XML, checking their SHA256 linkage:

| Evidence run | Recorded result | R1 observations |
| --- | --- | --- |
| `c204bb75` | 3 expected failures, 0 errors | Old runner; ABORT after death/release at ticks 3 and 4, or absent for sibling failure. |
| `4016313e` | 3 passed, 0 failures/errors/skips | ABORT at ticks 0/1/1 with first worker live and no releases; both launchers drained. |
| `55dccf3c` | 23 targeted passed, 0 failures/errors/skips, 0.871s | Same three passing observations. |
| `bbd349b3` | 72 candidate passed, 0 failures/errors/skips, 50.635s | Same three passing observations. |

The clean reproduction and passing runs use the same workflow-test hash; the
reproduction runner hash is the exact originally reviewed runner. Recorded
preservation manifests still contain the same 202 source/result hashes as the
original evidence. Old code/results were not broadly re-reviewed or rehashed
during this scoped pass.

Verified identities:

- `fix1.diff`: `eb22ffa75e4095e0815e584965c5246d998d5e0d3ce40323ce888e5e8acd2fdf`
- `run_wave.py`: `a9fdf4cc390eba170e79b8439cf518b1959795a800f053bb223a0bff1f18f9af`
- `test_workflow.py`: `bd1b2db2a295a5957a8fa5c50cc6658b4ccf6d25e3a50a987713a089281d1cd5`
- `bbd349b3/evidence.json`: `9d2189bedf735bde76ed48f26e03b861f456a2a8c4653ced1f0a0e3741b76084`
- `bbd349b3/tests.xml`: `0b939500049f497c20b231900ff066c282c1c70608d907505c6ddf9656694c77`

## Limits and Disposition

The regressions use synthetic handles, identities and clock ticks; they do not
establish live Windows timing or a physical pilot outcome. Workers retain their
existing interval/checkpoint ABORT boundaries. Final draining may wait for a
known live worker after ABORT; UNKNOWN ownership remains fail-closed.

No test suite or behavioral probe was rerun. No production command, physical
run, model load, training, source/result edit or commit occurred. The numerical
`nuf_retention_v1` root remains absent. Only this review ledger was written.

Scoped disposition: PASS; no additional R1 fix is requested before the parent's
decision on the single 170 pilot under the existing staged protocol.
