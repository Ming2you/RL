# Critic diagnostic implementation report

Status: implemented; focused synthetic tests pass. No production numerical
diagnostic was run. The coordinator must request the full run separately.

## Owned changes

- `work/sdmpc_rl_recovery_20260929/critic_diagnostic.py`
- `work/sdmpc_rl_recovery_20260929/test_critic_diagnostic.py`
- `.superpowers/sdd/rl_budget_recovery_plan_20260929/critic-report.md`

No prior source, data, checkpoint, runtime, index, schedule, or experiment was
modified. No plant rollout, collection, policy export, or commit was performed.

## Specification self-review

- CLI reads only the specified final model, checks its required SHA-256 before
  deserializing the same in-memory bytes, and checks the input hash again on exit.
  The actual model hash was checked before and after implementation and matches
  `3dd1c06786e8d9ee62da355bdb17ac025290bef8b6d20c863b421a86488d5650`.
- Uses the existing TD3 module by its exact file path. Both clones retain the
  final online critics, critic targets, and Adam history. Both target actors
  are initialized to the final online actor. Actors and actor targets are frozen;
  neither actor optimizer nor TD3's actor-training update method is called.
- Requires five scenarios, 150 transitions each, two true terminals at indices
  74 and 149, and continuity within the two complete 75-step collection episodes.
- Full CLI budget is 3750 additional critic updates. Every shared batch has
  exactly eight samples per scenario, with replacement. Both arms consume the
  same batch and clipped smoothing-noise tensor. Gamma remains 1, noise standard
  deviation 0.2, noise clip 0.5, action clip [-1, 1], and Adam learning rate 3e-4.
- Only the critic-target rate differs: A uses tau 0.005; B uses tau 1. Both
  synchronize after every second diagnostic critic update. Initial targets are
  retained; there is no extra synchronization at update zero.
- Logs updates 0, 375, 750, 1500, 3750. Includes total twin-critic TD MSE,
  last/mean training losses, sample counts, target-update counts, paired-draw
  digest, and per-scenario Q1/Q2/minimum-Q terminal errors and behavior-return
  gaps. Also reports each collection episode and early/middle/late time slices.
- Checkpoint TD loss uses all 750 transitions with a fixed shared smoothing-noise
  draw from a separate RNG. Logging never consumes training random numbers.
- Records source hashes, original TD3 specification, interpreter/package versions
  and paths, parameters, and seeds (default training seed 6529, evaluation seed
  6530). CPU float32 and one intra-op/inter-op thread are explicit.
- Rejects existing output directories; output files use exclusive creation.
  `diagnostics.jsonl` is flushed at each record. STOP is checked between paired
  batches in the output directory and its two ancestors, plus optional
  `--stop-file`. A stop writes current partial metrics and a `stopped` summary
  and returns normally. Output consists only of diagnostics and summary JSON.

## Verification

Final result: **19 passed, 0 failed in 3.80 seconds** using the existing
`.venv-torch` interpreter and `.deps-budget` dependencies. All fixtures are
synthetic; the longest tested diagnostic is four updates. Tests cover true
terminal masking, return boundaries, replay validation, equal initial critics
and optimizer state, clone independence, fixed actors and actor optimizer,
exact sample balance, common samples/noise, smoothing/clipping, target schedule,
CPU float32 defaults, metric/time-slice logging, reproducibility across logging
schedules, STOP partial output, overwrite refusal, and input immutability.

Command (PowerShell, repository root):

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.deps-budget')
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
& '.venv-torch/Scripts/python.exe' -B -m pytest -q -p no:cacheprovider --confcutdir=work/sdmpc_rl_recovery_20260929 work/sdmpc_rl_recovery_20260929/test_critic_diagnostic.py
```

The first sandboxed invocation could not read `.deps-budget/pytest` because of
local ACLs. The brief-authorized elevated execution passed; no runtime or ACL
changes were made. Bytecode writes and pytest caching were disabled.

Final source SHA-256 values:

- Diagnostic: `134d6f95d4c3ea863d15102e03ac62d6d30d4790eb7f5ba97f4478d354367076`
- Tests: `56f7780fe4ba3d42b612783a752f74a89d9056f9f563217351b5c58df4ff76d8`
- Imported original TD3: `e27665133ccb3fac2b1769da1c93df73a60c73572fa4ccede24a1296292e2e8e`

## Quality review and concerns

No implementation blocker found in the focused self-review. This is a small
standalone critic loop with existing networks/loading/optimizers; it introduces
no framework or admission system and is independent of actor-repair code.

Production loading/training and full-budget numerical stability remain untested
by design. No arm-comparison result is claimed. Behavior returns are exploratory
observed reward-to-go, explicitly not current-policy Q truth. Fixed-noise TD MSE
is a reproducible diagnostic estimate, not an expectation over smoothing noise.
Terminal calibration is reported separately; neither loss improvement nor an
arm difference establishes causal policy superiority or traffic improvement.
Stopped runs preserve diagnostic evidence but do not export resumable learners.

For coordinator-requested execution only, use a new output directory:

```powershell
& '.venv-torch/Scripts/python.exe' -B 'work/sdmpc_rl_recovery_20260929/critic_diagnostic.py' --output 'results/sdmpc_rl_recovery_20260929/critic_temporal_v1' --seed 6529
```

This command is documented only and was not executed.
