# SDMPC Shared RL Handoff Data

Start with [the current handoff](../../RL_HANDOFF_20260930.md). This export does
not launch experiments or change the goal/automation state. Old DDQN STOP files
and the separate historical archive remain untouched.

## Included

- 1,627 files, 5,098,378,577 original bytes, from the five SDMPC results families.
- ZIP size652,636,855bytes, split into16parts of at most40MiB each.
- Full ZIP SHA256:
  `0a2fd794882bb36afe5dc18f3c8c714001c6e6504c212e833f35942f2f4a511f`.
- Inventory SHA256:
  `97a8ec70a6cce1e524db737091b8af5abcd577b5d11546b057689b9f93127d96`.
- Actual replay, complete traces, summaries, provenance, final learned models,
  latest/referenced checkpoints, failed diagnostic results and logs are included.
- Source, frozen physical snapshot, plans and project-specific SDD evidence are
  normal Git files outside the ZIP. No personal app settings or credentials,
  Python environment, dependency cache, or paper draft is exported.

The latest model is `results/sdmpc_rl_balanced_goal_20260930/return_mc_v1/model_final.pt`,
SHA256 `820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6`.
Canonical evaluation of this policy has not run. Training losses are not TTT gains.

## Local-Only History

778 redundant historical checkpoint/lock files, 5,915,433,966bytes, remain intact
on the original computer. Their exact paths/sizes/hashes are listed in
`local_only_inventory.json`. This selected handoff is not a byte-complete backup
of every intermediate optimizer or interval checkpoint. No source file or raw
research result was deleted to make it. The separate 2026-09-10 archive is unchanged.

## Verify and Restore

Run from the repository root with Python3.11or later; no Torch needed for export
verification/restoration. Every part is required; no Git LFS client is needed.

```powershell
python -m work.sdmpc_rl_handoff_20260930 verify
python -m work.sdmpc_rl_handoff_20260930 restore
python -m work.sdmpc_rl_handoff_20260930 verify-local
```

The tool verifies part/fullZIP/member hashes and safe relative paths, checks all
existing-file conflicts before restore, and refuses differing files. It does not
unpickle models, install dependencies, clear STOP, or run RL. Allow temporary
space for the ZIP plus the uncompressed payload. The original machine needs no
restore. Existing process records are historical, not live-process evidence.

All1,627 archive members and all split parts/fullZIP passed `verify-local` against
the originals, with0writes and no RL launch. Export-tool synthetic tests cover
roundtrip, STOP preservation, conflict refusal, missing dependency and path escape;
the original snapshot tests also cover tampered parts.

Six combined export-tool tests passed. The latest learner's745bound input files
were checked for publication closure:528match archive inventory hashes and217are
staged in Git with the exact original raw bytes. This does not prove portable
execution on another OS/runtime. A pattern-based staged-text scan found no common
GitHub/OpenAI token or private-key signature; it is not a comprehensive security
audit.

## Resume Boundary

The canonical evaluator passed49focused tests and independent SPEC/QUALITY
review, but no physical admission receipt/run was created for this handoff.
On a restored machine, generate a NEW read-only preflight and exact-hash parent
receipt after verifying source/runtime/contracts and process/STOP/resource state.
Do not reuse the original test preflight blindly: its preservation map includes
historical intermediate files intentionally kept local in this selected export.
Source/model/physical-contract pins must remain exact. Recorded Windows absolute
paths may require an explicitly reviewed portability change, not silent rewrites.
