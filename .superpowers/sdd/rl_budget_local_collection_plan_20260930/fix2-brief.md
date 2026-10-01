# Collector fix round 2

Read fix1-review.md first: its F1 and F2 are the full open findings. Implement
only those two repairs in the existing collector modules/helper/tests. The
before-fix2 directory preserves the code reviewed in round1. Preserve approved
exploration and all old/completed sources. No numerical simulation, production
process, model, install, commit or OS changes. Use apply_patch.

F1: reuse the frozen complete-config serialization semantics including runtime
dataclass attributes; preserve the existing config hash. Synthetic nested
dataclass-plus-extra regression must accept intact and reject changed extra.

F2: preserve monotone started/checkpointed/finished cohort identity independently
of moving the output directory and without a later ownership transaction. Do
not block valid checkpoint resume after interrupted completion. Test standalone
completion followed immediately by relocation/retry, checkpoint relocation and
coordinator loss before post-exit release. Keep completion-last and measured
timing scope coherent; no general scheduler or new escape flags.

Reproduce focused failures first, then fix and run covering collector tests plus
the local combined suite. Append evidence to fix1-report.md in a round2 section.
Report exact commands/results and any real limitation. No need to rerun old
immutable phase tests or broaden the review. Parent will generate fix2.diff and
request scoped independent re-review before the first real prefix smoke.
