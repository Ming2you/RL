# Machine-B research archive (2026-10-01)

Raw results of the machine-B continuation of the balanced shared-policy RL budget goal
(`docs/rl_budget_machine_b_20260930.md`, handoff `RL_HANDOFF_20261001_machine_b.md`).

- `research.zip.part001..002`: one ZIP split into 40 MiB parts (77,766,752 bytes in total).
- `manifest.json`: the full-ZIP SHA-256, each part's size and SHA-256, and the inventory hash.
- `inventory.json`: 1,127 archived files with their relative path, size and SHA-256 (734,962,906 bytes).

Contents:

- `results/sdmpc_rl_machine_b_20260930/`: machine-B carry centers (`center_repro_v1`), the canonical
  evaluations (`return_canonical_b2`, `canonical_b1_run1`, `canonical_null_check2` and the failed
  boots `return_canonical_b1`, `canonical_null_check`), all training-profile probes (`probe_p1`..`probe_p8`,
  `probe_b1x`), the canonical registry, receipts and logs.
- `probe_checkpoint_cache/`: carry states at the branch steps (`k01/k14/k16/k18.pt`) and the cached
  carry runs, one folder per training slot. On machine B they live in `D:\RL_data\sdmpc_rl_machine_b_20260930\ckpt`.

The following are left out on purpose because they are large and can be regenerated: per-interval
evaluator checkpoints (`*/checkpoints/*`), runner locks, and the center `checkpoint.pt` files.

Commands (run from the repository root, standard library only, nothing is unpickled):

```powershell
python -m work.sdmpc_machine_b_archive_20261001 verify    # hashes of the parts, the ZIP and every member
python -m work.sdmpc_machine_b_archive_20261001 restore   # writes missing files only, refuses differing ones
```

`restore` puts the checkpoint cache under `results/sdmpc_rl_machine_b_20260930/probe_checkpoint_cache/`.
To use it, pass that path to `probe.py --checkpoint-dir` (or to `queue_runner.py --checkpoint-root`).
