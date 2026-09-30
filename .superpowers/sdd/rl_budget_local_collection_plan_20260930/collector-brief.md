# Task 2: versioned sequential collection and paired readout

Implement the pending collection stage in docs/rl_budget_local_collection_plan_20260930.md.
Existing physical environment/runtime/controller and completed outputs stay immutable.
New owned files are local_runtime.py, collect.py, validate.py, run_wave.py and
test_collection.py under work/sdmpc_rl_local_20260930. The separate exploration
task owns exploration.py/test_exploration.py, and uses its own reviewed public API.

Requirements:
- Authenticate unchanged existing contract via projection completion hash
  b6fabf1ccaa095fed6b5bcb33dea88cf7f39de4a419ed41b2459a940db6857d3.
  Read only contract/schema metadata, no canonical state/trajectory for training.
  Pin old physical source, runtime, imported validator sources and new scripts.
- Reuse BudgetEnv physical guard, one lower solve, previous-executed action
  anchor, [50,1000] scales, reward -intervalTTT/100, true75step terminal, warmup5.
  No model load/Q/actor updates, extra PFO/response previews or performance guard.
- Five prescribed scenarios, demand seeds6801..6805, exploration seeds6901..6905,
  CPU masks1,4,16,64,256, carry and local waves. Paired training profiles, distinct
  run IDs/behavior keys, 75 sequential transitions per run,750 total,150/scenario.
- Reproducible interval checkpoints include full physical environment, observation,
  trace/experience, behavior state/RNG and immutable settings. Strict resume
  validation including observation chain, simulator boundary and behavior replay.
  Reject completed collection/reused output, settings/source/runtime drift.
  Optional max-new-steps checkpoints only; NEVER promotes a prefix to completion.
  A first carry episode prefix can serve as physical smoke then continue in place.
- Exclusive coordinator and child locks, skip only authenticated completed jobs,
  max5 one-thread collectors, stage barrier between carry/local. Child failures
  drain siblings at interval boundary via per-attempt ABORT, not delete user STOP.
  Honor repo/goal/root/wave/child STOP, no overlapping/orphan-child dispatch.
- Persist PID/command/start, logs, settings, schema, per-interval checkpoint,
  experience, final trace/summary, hashes, completion. Completed marker is last.
  No configuration/source mutation while a runner is active.
- Validate data/provenance, physical execution flags, PFO/one-solve counts, reward,
  observation/budget projection, total75steps/14400s, terminal, full TTT and timings.
  Existing admitted validate_sequence is reused read-only for complete episodes.
  Reconstruct exploration choices from seed/actual anchors to validate audits.
- Pair same-profile training runs only, report interval/total losses, inventories,
  budget drift/request counts, unique projected/actual budgets and physical
  controls (exclude budget labels and diagnostic metadata), fallbacks and timing.
  Fixed design screen: local zeroNUF <=carry+5; localterminalinventory
  <=1.25*max(1,carryinventory), in all5. NOT goal or policy admission.
- Keep original allfive canonical full-run improvement/reproduction goal active.
  No automatic learner export from this collection phase.

Synthetic tests must avoid actual gate/physical data loading and model/simulation
execution. Test resume equivalence, immutable settings, STOP, failures, locks,
completed-output skip/validation, hashes, physical/TTT/timing flags, audit replay,
paired scenario/profile rules, numerical worker count and completion semantics.
Independent review before actual collection, then physical smoke and combined
regression. No commits, network, dependencies or OS changes.
