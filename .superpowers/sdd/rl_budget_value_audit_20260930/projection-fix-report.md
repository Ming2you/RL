# Projection audit fixes

Implemented all four findings from `projection-review.md`; ready for independent scoped review. Only `work/sdmpc_rl_value_audit_20260930/projection_audit.py`, `test_projection_audit.py`, and this report were edited. The parent's two before-fix snapshots remain unchanged.

## Changes

1. Input identity: capture a 73-file SHA-256 manifest before loading the model or collections: both models, pilot completion, and seven consumed files for each of the ten collections. Persist it in settings and completion. Recheck every entry immediately before publishing completion, alongside existing source/model/runtime checks. Missing or changed entries fail closed. Diagnostic source hashes are now captured before model loading/runtime boot, and the round-0 predecessor hash comes from the initial manifest.
2. Provenance: require exactly ten final-model `training_profiles` entries and compare each round's ordered returned collection provenance with its authenticated model slice before transition comparisons or Q calls. All provenance fields participate in exact comparison.
3. Lock guards: recheck output/goal/repository STOP and output emptiness after acquiring the output lock, before writing process metadata or loading inputs. Only the context's `runner.lock` is exempt from the under-lock emptiness check. Retain early guards and recheck STOP immediately before completion.
4. Float32 aliases: generate representable boundary candidates using float32 and the physical transform, advance a rounded boundary with `nextafter` when necessary, and deduplicate/group actual float32 actions by exact projected requests. Critic inputs are deduplicated again after conversion; fewer than two distinct actions cannot form a reported alias group. The post-conversion projection equality check remains.

## Focused verification

Working directory: `C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`.

Exact command used for both the failing regression run and final passing run:

```powershell
.venv-torch/Scripts/python.exe -B -c "import sys; sys.path.insert(0,'.deps-budget'); import pytest; raise SystemExit(pytest.main(['-q','--tb=short','-p','no:cacheprovider','work/sdmpc_rl_value_audit_20260930/test_projection_audit.py']))"
```

- Before implementation fixes: **33 failed, 15 passed in 24.11 s**, reproducing all four findings.
- Final implementation: **48 passed in 22.78 s**, exit code 0.
- An earlier sandbox attempt failed before test collection because `.deps-budget/pytest` was unreadable and imported as a namespace without `pytest.main`. The same command succeeded with elevated filesystem access. No package, dependency, or permission settings were changed.
- Scoped `git diff --no-index --check` inspections produced no whitespace diagnostics. They returned the diff exit status 1 and Git's existing LF-to-CRLF warnings for the saved originals; the original files were not rewritten.

The 48 cases consist of the original 13 plus 35 isolated regressions: one complete orchestration/manifest case, 17 post-load input-mutation cases, ten provenance-field cases spanning both rounds, one repackaged-collection case with unchanged transition values, four lock-acquisition race cases, and two float32 boundary cases. The passing orchestration case asserts manifest capture before model loading, all 750 transition comparisons/Q-dispatches, all five scenarios in both rounds, persisted manifests, and unchanged fixture inputs.

All orchestration tests use temporary files and stubbed model/collection loaders, boot, source-pin verification, runtime versions, lock contexts, and Q dispatch. An autouse fixture redirects production paths and fails on unexpected production loader/runtime entry. File hashing in these fixtures is constrained to the temporary test directory. Real critic evaluation is limited to synthetic linear networks in alias tests. These tests do not execute a production diagnostic, deserialize preserved models, simulate, train, export policy weights, or test actual interprocess Windows locking; the lock-order regressions deterministically inject entries before the lock context yields.

## Source SHA-256

Hashes were collected after the final passing run. Both before-fix hashes matched their respective implementation/test files before edits and were rechecked afterward.

| File | SHA-256 |
| --- | --- |
| `work/sdmpc_rl_value_audit_20260930/projection_audit.py` | `ab3787b94c8bbc4a1e76a9d04b942159a01c6753b292b6d6f5d9c21a256b86db` |
| `work/sdmpc_rl_value_audit_20260930/test_projection_audit.py` | `b0b20654d616b2e5c44c61d3689490254497fa10bbf55350071fe3fa72eb808a` |
| `.superpowers/sdd/rl_budget_value_audit_20260930/projection-before-fix.py` | `cba3344551eb44f6a8825f7c02b5ba601339234b3896a89398d688c9bef6e20e` |
| `.superpowers/sdd/rl_budget_value_audit_20260930/projection-test-before-fix.py` | `bbbd0e4f4aeab047832b43c7a0659df57f77fbc95bdc7d2e10a39237b9d1c893` |

Old sources/data and the other agent's terminal files were not edited. No production diagnostic, simulation, or commit was performed. Independent review remains the next acceptance step.
