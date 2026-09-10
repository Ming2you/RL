# Response-Aware Double DQN 구현 현황

**기준일:** 2026-08-31

## 1. 현재 결정

활성 학습 경로는 dense continuous IQL actor가 아니라 다음 구조다.

```text
P-Stack anchor
-> stable structured nonlinear action catalog
-> actual follower MPC response evaluation
-> physical-response + follower-memory deduplication
-> frozen sequential replay
-> masked parametric Double DQN ensemble
-> anchor-relative LCB and material-margin gate
-> candidate or P-Stack fallback
-> support-aware simulator batch expansion
```

Urban/freeway follower MPC, traffic simulator, P-Stack controller와 물리모형은 수정하지 않았다. 기존 IQL과 과거 DDQN도 baseline으로 그대로 유지한다.

## 2. 기존 기능 위치

| 기능 | 기존 구현 |
|---|---|
| P-Stack anchor | `rl_leader/env.py::prepare_pstack_anchor_context`, `step_prepared_optimizer_anchor` |
| Anchor-relative candidate 실행 | `rl_leader/env.py::step_anchored_candidate` |
| Nonlinear coordination schema | `src/controllers/coordination.py::CoordinationActionSchema` |
| Quadratic/cross potential | `PotentialBlock`, `decode_anchored_residual` |
| Urban/freeway follower response | `src/controllers/rl_stackelberg.py`, `src/controllers/wu_faithful_follower.py` |
| 기존 owner-block candidate | `rl_leader/oracle_candidate_ablation.py` |
| 기존 response dedup | `rl_leader/diagnose_candidate_ablation.py::_dedupe_for_selector` |
| Long-horizon counterfactual | `generate_long_horizon_labels.py`, `run_tail_label_generation.py` |
| Dense offline IQL | `rl_leader/iql.py` |
| 기존 tail selector | `train_tail_selector.py`, `run_tail_selector_pipeline.py` |

현재 schema는 owner마다 `(linear_1, linear_2, l11, l21, l22)`를 이미 제공한다. 따라서 nonlinear potential을 위해 follower MPC를 새로 작성할 필요가 없었다.

## 3. Stable action ID 판정

Fixed discrete action index를 만들 수 있다. ID는 raw price 자체가 아니라 다음 **state-independent operator**를 뜻한다.

```text
(domain, owner, template, normalized magnitude)
```

- Action 0: 현재 state의 exact P-Stack anchor
- Linear: signed axis와 corner
- Quadratic: `l11`, `l22` increase/decrease
- Cross: positive diagonal과 signed `l21`을 결합한 PSD operator
- Magnitude: template direction을 unit L2로 정규화한 trust-region radius

P-Stack 값과 physical effect는 state에 따라 달라지지만 “현재 P-Stack에 이 operator를 적용한다”는 action 의미는 변하지 않는다. Catalog fingerprint가 replay와 checkpoint에 기록되어 ID drift를 차단한다.

## 4. 추가된 구현

### Action과 response

- `rl_leader/response_dqn_catalog.py`
  - deterministic action ID와 catalog fingerprint
  - linear/quadratic/cross family ablation
  - unit-L2 magnitude normalization
- `rl_leader/response_dqn_mask.py`
  - implemented physical control과 post-follower memory 기준 grouping
  - anchor-priority representative
  - invalid/duplicate action mask
  - raw + anchor-relative response feature

### Frozen sequential replay

- `rl_leader/response_dqn_data.py`
  - `(o, action_id, r, o_next, done, option_steps)`
  - current/next valid-action mask
  - current/next response feature matrix
  - episode/event/control-step provenance
  - event-group bootstrap와 contract-safe batch merge
- `rl_leader/merge_response_dqn_replay.py`
  - catalog/schema/response/scenario가 같은 frozen batch만 병합

Training module은 simulator를 import하지 않으며 frozen NPZ만 입력받는다.

### Double DQN과 conservative selection

- `rl_leader/response_dqn.py`
  - `Q(observation, candidate feature, realized response feature)` scorer
  - online/target network 기반 Double DQN
  - next-state executable mask와 behavior-support mask의 교집합으로 target argmax
  - `gamma ** option_steps`
  - event-group bootstrap ensemble
  - anchor-relative mean/std/LCB와 material-margin fallback
  - checkpoint save/load
- `rl_leader/train_response_dqn.py`
  - frozen ensemble training CLI

Anchor action support가 없으면 `Delta_Q`의 기준값을 학습할 수 없으므로 training이 fail-fast한다. 각 bootstrap member에도 globally supported action의 최소 support를 보충한다.

### Simulator-assisted expansion

- `rl_leader/response_dqn_collect.py`
  - 모든 candidate를 common state, anchor, follower seed에서 preview
  - response-equivalent action dedup
  - valid representative 안에서만 exploration
  - 선택된 action만 live environment에 commit
  - random anchored collection 또는 저장된 ensemble LCB policy 사용
- `rl_leader/response_dqn_expansion.py`
  - ensemble disagreement
  - intervention boundary
  - low support
  - novel response
  - 위 항목별 expansion priority와 reason decomposition

이 collector는 모든 candidate에 actual follower solve를 수행하는 `oracle-response` mode다. 방법의 feasibility를 검증하는 단계이며 아직 계산 대체를 주장하지 않는다.

## 5. 중요한 의미 구분

다음 네 quantity는 코드상 별도 객체와 artifact로 유지한다.

1. Nominal coordination operator: catalog action ID와 residual
2. Executable follower response: physical control과 follower-memory outcome
3. Sequential DQN value: learned policy continuation 아래의 cumulative return
4. Paired `Delta_J_H`: 현재 한 번 개입한 뒤 P-Stack continuation을 적용한 진단 label

`Q(s, anchor)`는 현재 step에 anchor를 쓰고 이후 learned policy로 진행하는 값이다. 모든 미래 step에 P-Stack을 쓰는 baseline value가 아니다. 따라서 full P-Stack 우월성은 matched full-run과 paired long-horizon diagnostics로 별도 검증한다.

## 6. 검증 결과

추가된 test는 다음을 확인한다.

- action 0 zero residual과 native anchor identity
- stable catalog roundtrip/fingerprint
- linear/quadratic/cross operator와 equal-L2 radius
- response duplicate/invalid masking과 anchor representative
- selected masked action replay 거부
- next-state masked argmax
- anchor behavior support 없는 학습 거부
- frozen replay save/load와 event-group merge/bootstrap
- Double DQN training/checkpoint roundtrip
- LCB pass/fallback과 valid-only epsilon-greedy

실행 결과:

- 새 response-DQN tests: `12/12` 통과
- 새 tests + coordination/Phase-0 regression: `73/73` 통과
- 기존 DDQN regression: `5/5` 통과
- legacy `rl_leader.iql` import: 통과
- 실제 `sweet_170_incident_w60` 1-step collector: 9 actions, 9 unique responses, NPZ 생성 성공
- 실제 frozen NPZ -> 2-member Double DQN smoke: 성공

Smoke는 plumbing 검증일 뿐 성능 결과가 아니다.

## 7. Pilot 실행

처음부터 모든 owner와 두 magnitude를 열면 candidate 수가 커진다. 먼저 urban `C` 하나에서 action/response headroom을 확인한다.

```powershell
$env:PYTHONPATH='C:\torchlib;.'
python -B -m rl_leader.response_dqn_collect `
  --scenario sweet_170_incident_w60 `
  --t-total 14400 `
  --owners C `
  --families linear,quadratic,cross `
  --magnitudes 0.25 `
  --epsilon 1.0 `
  --anchor-probability 0.25 `
  --seed 0 `
  --out data\response_dqn_170_incident\batch_seed0.npz `
  --log results\response_dqn_170_incident\batch_seed0.jsonl
```

서로 다른 seed의 frozen batch를 병합한다.

```powershell
python -B -m rl_leader.merge_response_dqn_replay `
  --inputs data\response_dqn_170_incident\batch_seed0.npz data\response_dqn_170_incident\batch_seed1.npz `
  --out data\response_dqn_170_incident\batch_merged_v1.npz
```

Frozen batch만 사용해 ensemble을 학습한다.

```powershell
python -B -m rl_leader.train_response_dqn `
  --data data\response_dqn_170_incident\batch_merged_v1.npz `
  --output-dir checkpoints\response_dqn_170_incident_v1 `
  --ensemble-size 5 `
  --gradient-steps 20000 `
  --min-action-support 3
```

동일 catalog 옵션과 학습 ensemble을 사용해 다음 batch를 만든다.

```powershell
python -B -m rl_leader.response_dqn_collect `
  --scenario sweet_170_incident_w60 `
  --t-total 14400 `
  --owners C `
  --families linear,quadratic,cross `
  --magnitudes 0.25 `
  --model-dir checkpoints\response_dqn_170_incident_v1 `
  --z-value 1.96 `
  --material-margin 0.0 `
  --exploration-epsilon 0.05 `
  --seed 2 `
  --out data\response_dqn_170_incident\expansion_seed2.npz `
  --log results\response_dqn_170_incident\expansion_seed2.jsonl
```

## 8. 아직 남은 작업

1. `170 incident`에서 linear/quadratic/cross pilot batch를 실제 수집한다.
2. Action별 최소 support와 traffic-phase coverage를 확인한다.
3. Existing paired drain-out label로 `z`와 material margin을 calibration한다.
4. Untouched event group에서 selected false positive 0을 확인한다.
5. P-Stack, linear, quadratic, cross의 matched full-run 3회를 수행한다.
6. Terminal inventory, queue, density, conservation, follower residual과 runtime을 함께 비교한다.

현재 구현만으로 P-Stack보다 성능이 좋다고 주장할 수는 없다. 다음 의사결정은 24시간 수집이 아니라 작은 `C` owner pilot에서 nonlinear family가 새로운 response class와 long-horizon headroom을 만드는지 확인한 뒤 내린다.

## 9. Recovery-aware DDQN 확장

2026-08-31 추가 구현은 한 스텝 label을 곧바로 good/bad로 확정하지 않는다. 문제 state에서 첫 nonlinear action을 적용한 뒤, 짧은 recovery tree를 허용하고 남은 구간은 P-Stack tail로 평가한다. 이 결과를 terminal Monte-Carlo target으로 저장해서 기존 masked parametric Double DQN trainer에 그대로 넣는다.

추가된 파일:

- `rl_leader/response_ddqn_recovery.py`
  - policy trace prefix replay로 실제 RL trajectory의 pre-action state 복원
  - in-process exact env snapshot capture/restore
  - executable response evaluation 재사용
  - first action + top-K recovery action tree + P-Stack tail rollout
  - `anchor_tail_return`과 `best_recovery_return`을 모두 저장
  - `best_recovery_return`을 DDQN terminal target으로 쓰는 frozen replay 생성
- `src/tests/test_response_ddqn_recovery.py`
  - anchor-tail에서는 손실이지만 recovery action으로 장기 이득이 되는 fake branch graph 검증
  - recovery replay가 `done=1` terminal MC target으로 생성되는지 검증
  - problem-step 선택 로직 검증

권장 실행 순서:

```powershell
$env:PYTHONPATH='C:\torchlib;.'

python -B -m rl_leader.response_ddqn_recovery `
  --scenario sweet_170_incident_w60 `
  --t-total 14400 `
  --policy-trace results\response_dqn_170_incident\actor_parallel_v1_eval.jsonl `
  --pstack-trace results\response_dqn_170_incident\pstack_rl_contract_v2\trace.jsonl `
  --policy-steps 19,21,27 `
  --owners R_F_E `
  --magnitudes 0.25 `
  --families linear,quadratic,cross `
  --domains urban,freeway `
  --recovery-depth 1 `
  --top-k 3 `
  --max-rollout-steps 12 `
  --workers 1 `
  --backend serial `
  --out data\response_dqn_170_incident\recovery_ddqn_step19_21_27_h12_d1.npz `
  --log results\response_dqn_170_incident\recovery_ddqn_step19_21_27_h12_d1.jsonl
```

그 다음 기존 trainer를 그대로 사용한다.

```powershell
python -B -m rl_leader.train_response_dqn `
  --data data\response_dqn_170_incident\recovery_ddqn_step19_21_27_h12_d1.npz `
  --output-dir models\response_dqn_170_incident\recovery_ddqn_step19_21_27_h12_d1 `
  --ensemble-size 5 `
  --gradient-steps 20000 `
  --batch-size 128 `
  --gamma 1.0 `
  --reward-scale 0.01 `
  --min-action-support 1
```

이 데이터의 `reward`는 즉시 reward가 아니라 `best_recovery_return = -TTT(first action + best recovery + P-Stack tail)`이다. 그래서 replay row는 `done=1`로 저장된다. DDQN network와 conservative ensemble selection은 그대로 쓰되, target의 의미가 “한 스텝 bootstrap”에서 “recovery-aware long-horizon value regression”으로 바뀐다.

## 10. 170-incident recovery-DDQN r1 결과

2026-09-01에 `sweet_170_incident_w60`, `t_total=14400`으로 exact policy evaluation을 수행했다. 평가 runner는 `workers=8`, `backend=process`를 사용했고, 실제로 parent 1개와 worker 8개 Python process가 동시에 실행되었다. 다만 병렬 단위가 control-step 내부의 executable follower-response 후보 평가이기 때문에, 전체 wall time은 약 3390초였다.

결과:

- Model: `models\response_dqn_170_incident\recovery_ddqn_r1_steps19_21_27_h4_d1_bw5_v1`
- Eval output: `results\response_dqn_170_incident\recovery_ddqn_r1_eval_v1`
- RL total TTT: `5797.49395681906`
- P-Stack total TTT: `5730.792964723197`
- Gap: `+66.70099209586351` veh-h (`+1.1639051088819998%`)
- Action counts: P-Stack anchor 73회, `action_id=12` 2회
- Non-anchor steps: 20, 22

해석:

- 이번 모델은 P-Stack을 넘지 못했다.
- targeted recovery labels는 step 19/21의 나쁜 nonlinear 선택을 억제했지만, step 20/22 주변 state에서 `quadratic_second_increase`가 작은 양수 LCB로 통과했다.
- 저장된 P-Stack step 27 state에서는 모델이 `action_id=3`을 선택할 정도로 positive label을 학습했다. 하지만 step 20/22에서 먼저 trajectory가 바뀌면서 그 좋은 P-Stack-path state에 도달하지 못했다.
- 다음 루프는 데이터 양을 늘리기보다 step 20/22 misfire state를 negative recovery label로 추가하고, step 27 positive state가 실제 full-run에서 열리는지 검증하는 방향이 맞다.

병렬화 판단:

- 현재 `workers=8`은 시나리오 8개 병렬이 아니라 한 control step 안에서 action 후보들의 follower response preview를 나누는 구조다.
- 속도 개선의 다음 타깃은 worker 수 증가보다 후보 prefiltering, response cache, problem-step targeted evaluation, 또는 policy 실행 시 전체 후보 exact preview를 생략하는 surrogate response model이다.

## 11. r3/r5 장기 label 재검증 결과

2026-09-01 추가 run에서는 `recovery_ddqn_r3_pstack_peak_h4_d1_noboot_v1` 모델을 P-Stack peak window인 control step `20..27`에만 허용하고 full evaluation을 수행했다.

구현상 개선:

- `rl_leader.response_dqn_collect.evaluate_anchor_response()`에 commit-only fast path를 추가했다.
- `rl_leader.evaluate_response_dqn_policy --allow-control-steps`는 허용 window 밖에서 전체 후보 response preview를 생략하고 P-Stack anchor context만 준비한다.
- 테스트는 `src.tests.test_response_dqn`와 `src.tests.test_response_ddqn_recovery` 합산 `26/26` 통과.

Full evaluation:

- Eval output: `results\response_dqn_170_incident\recovery_ddqn_r3_noboot_stepgate20_27_eval_v3`
- Model: `models\response_dqn_170_incident\recovery_ddqn_r3_pstack_peak_h4_d1_noboot_v1`
- RL total TTT: `5837.942498054198`
- P-Stack total TTT: `5730.792964723197`
- Gap: `+107.1495333310013` veh-h (`+1.8697156569880853%`)
- Non-anchor actions: step 26 `action_id=3`, step 27 `action_id=11`

장기 counterfactual:

- r4 output: `data\response_dqn_170_incident\recovery_ddqn_r4_step26_27_h20_d0_bw3_v1.npz`
- r4 log: `results\response_dqn_170_incident\recovery_ddqn_r4_step26_27_h20_d0_bw3_v1.jsonl`
- step 26, h20: action 3 is worse than anchor by about `+5.41` veh-h.
- step 27 after step 26 action 3, h20: action 11 is worse than anchor by about `+104.71` veh-h.
- r5 output: `data\response_dqn_170_incident\recovery_ddqn_r5_step26_all_h20_d0_bw8_v1.npz`
- r5 log: `results\response_dqn_170_incident\recovery_ddqn_r5_step26_all_h20_d0_bw8_v1.jsonl`
- step 26, h20, all 15 catalog actions: anchor is best. Most non-anchor actions are worse by about `+5.42` veh-h; action 4 is worse by about `+5.69` veh-h.

해석:

- r3에서 발견한 step 26 h4 positive는 장기 horizon에서 false positive였다.
- 지금 R_F_E-only catalog 안에서는 step 26 peak state에 P-Stack을 이기는 h20 action이 없다.
- 이 데이터를 그대로 재학습하면 모델은 P-Stack anchor로 돌아가는 안전 모델이 될 가능성이 높다. P-Stack을 넘기려면 새로운 positive source가 먼저 필요하다.

## 12. 병렬화 구조 정리

병렬화는 두 층으로 나눌 수 있다.

- 내부 병렬: 같은 simulator state에서 여러 coordination candidate의 executable follower response를 비교한다. 현재 `response_workers`와 `response_backend=process`가 이 역할을 한다.
- 외부 병렬: 서로 독립적인 first-action branch, action-comparison branch, scenario, seed, 혹은 target state를 병렬 실행한다. 현재 recovery label generator의 `branch_workers`가 first-action branch 병렬이다.

주의할 점:

- 시간축 prefix replay는 앞 step의 결과가 다음 state를 만들기 때문에 일반적으로 병렬화할 수 없다.
- 외부 branch와 내부 response를 둘 다 process로 풀면 worker 수가 곱해진다. 예를 들어 `branch_workers=8`, `response_workers=8`은 Windows에서 최대 64개 Python worker를 만들 수 있어 오히려 느려진다.
- 8-core PC에서는 total worker budget을 정하고 나누는 방식이 낫다. 예: targeted label은 `branch_workers=8, response_workers=1`; full policy evaluation은 `branch_workers=1, response_workers=8`; deep recovery tree는 `branch_workers=2, response_workers=4` 정도가 상한이다.

다음 구현 우선순위:

1. Snapshot cache를 추가해서 같은 policy prefix의 target states를 파일로 저장한다. 2026-09-01 현재 `rl_leader.response_ddqn_recovery --snapshot-cache-dir`로 구현 완료.
2. h1/h4 prefilter 뒤 h20/h40 label만 expensive하게 평가한다.
3. 장기 positive 후보를 centralized/PFO action, owner block, larger magnitude, 또는 learned response surrogate에서 가져온다.
4. h20 이상에서 anchor를 이기는 후보만 DDQN positive target으로 학습한다.
5. Conservative selector는 `horizon_label_level >= h20` 또는 `verified_positive=true`가 없는 action을 production policy에서 막는다.

## 13. r6-r10: h20 verified positive 기반 DDQN 회복 결과

2026-09-01 추가 loop에서는 h4 label의 false positive를 제거하고, P-Stack을 실제로 이기는 장기 action이 있는지 작은 범위에서 다시 찾았다.

Baseline:

- Scenario: `sweet_170_incident_w60`
- P-Stack summary: `results\response_dqn_170_incident\pstack_rl_contract_v2\summary.json`
- P-Stack total TTT: `5730.792964723197`
- P-Stack control TTT: `5621.08961841247`
- Control steps: `75`
- Contract SHA: `95694fc1e5bb06da621e784a7e4d4bad56135b6d75ac360bb3a42b2d18501831`

Positive source search:

- r6 exploratory h20 replay tested step 24 with `R_F_E`, `magnitude=0.25,0.5`, `family=linear`.
- r6 found that step 24 `linear_corner_pp`, `magnitude=0.25` improves h20 TTT by about `-1.10` veh-h versus anchor.
- r7 reproduced the same positive action under the DDQN-compatible catalog `R_F_E`, `magnitude=0.25`, `families=linear,quadratic,cross`.
- This action maps to `action_id=8` in the compatible catalog.

Timing sensitivity check:

- r9 evaluated `action_id=8` at steps 23, 24, and 25 with h20 paired labels.
- Step 23: action 8 is worse than anchor by about `+1.77` veh-h.
- Step 24: action 8 is better than anchor by about `-1.10` veh-h.
- Step 25: action 8 is effectively tied with anchor.
- Interpretation: the useful intervention is a narrow timing effect, not a broadly good action.

Training and selector:

- r10 merged r4, r5, and r9 into `data\response_dqn_170_incident\recovery_ddqn_r10_h20_timing_posneg_v1.npz`.
- The batch has `27` transitions and covers control steps `23,24,25,26,27`.
- `models\response_dqn_170_incident\recovery_ddqn_r10_h20_timing_scale01_support4_noboot_v1` was trained with reward scale `0.1`, no bootstrap target, and `min_action_support=4`.
- The support gate is important: `min_action_support=3` allowed an unseen/under-supported action 11 phantom positive at step 24, while `min_action_support=4` selected action 8 at step 24 and anchor elsewhere.

Full policy evaluation:

- Eval output: `results\response_dqn_170_incident\recovery_ddqn_r10_support4_z196_step24_eval_v1`
- Selector: conservative ensemble LCB with `z=1.96`, `material_margin=0.0`
- Allowed non-anchor step: `24`
- Action counts: anchor `74`, action 8 `1`
- RL total TTT: `5727.850138124504`
- RL control TTT: `5618.146791813777`
- P-Stack total TTT: `5730.792964723197`
- P-Stack control TTT: `5621.0896184124695`
- Gap: `-2.942826598692591` veh-h (`-0.05135112395104178%`)
- `beats_pstack=true`

현재 결론:

- DDQN 구조 자체가 항상 실패하는 것은 아니다. h20 verified label과 strict support gate를 넣으면 `sweet_170_incident_w60` 한 시나리오에서는 P-Stack을 넘겼다.
- 이전 실패의 직접 원인은 데이터 양 부족보다 label horizon mismatch가 더 컸다. h4 positive였던 step 26/27 action은 h20 및 full run에서 손해였다.
- 다만 성능 개선폭은 작고, 아직 one-scenario one-intervention 결과다. 논문용으로는 "P-Stack을 안정적으로 대체했다"가 아니라 "response-aware long-horizon labeling이 P-Stack 대비 개선 가능한 실행가능 action을 찾아냈다" 정도가 안전하다.
- 다음 단계는 5개 시나리오 전체 우세를 목표로 무작정 긴 수집을 하는 것이 아니라, 각 시나리오별로 h20/h40 verified positive 후보를 먼저 찾고 그 후보만 보수적으로 policy에 열어주는 것이다.
- 같은 policy prefix와 target step을 반복 평가할 때는 `--snapshot-cache-dir data\response_dqn_170_incident\snapshot_cache`를 사용한다. 캐시 키에는 scenario, T_total, experiment contract, catalog fingerprint, target step, step 이전 prefix action이 포함된다.

## 14. r11-r12: step24 이후 상태의 반복 action 검증

r10/r12 full-run에서 실제로 방문하는 step 25/26 state는 P-Stack prefix의 step 25/26 state가 아니라, step 24에서 `action_id=8`을 한 번 적용한 뒤의 altered state다. 따라서 r11에서는 r10 eval trace를 policy prefix로 사용해 altered step 25/26을 h20으로 다시 라벨링했다.

r11 label:

- Output: `data\response_dqn_170_incident\recovery_ddqn_r11_after_a8_steps25_26_a8_h20_d0_bw2_v1.npz`
- Log: `results\response_dqn_170_incident\recovery_ddqn_r11_after_a8_steps25_26_a8_h20_d0_bw2_v1.jsonl`
- Policy prefix: `results\response_dqn_170_incident\recovery_ddqn_r10_support4_z196_step24_eval_v1\trace.jsonl`
- Evaluated steps: `25,26`
- Evaluated actions: anchor `0`, `action_id=8`
- Result: action 8 is tied with anchor at both altered step 25 and altered step 26 under h20.

r12 training:

- Merged dataset: `data\response_dqn_170_incident\recovery_ddqn_r12_h20_timing_after_a8_v1.npz`
- Transitions: `31`
- Action support: anchor `8`, action 8 `6`; low-support actions remain blocked.
- Model: `models\response_dqn_170_incident\recovery_ddqn_r12_h20_timing_after_a8_scale01_support6_lr1e4_100k_noboot_v1`
- Training: no-bootstrap, reward scale `0.1`, learning rate `1e-4`, `100000` gradient steps, `min_action_support=6`.

r12 full evaluation:

- Eval output: `results\response_dqn_170_incident\recovery_ddqn_r12_support6_100k_z196_steps24_26_eval_v1`
- Allowed non-anchor steps: `24,25,26`
- Selector: z=1.96 LCB
- Action counts: anchor `74`, action 8 `1`
- Step 24: action 8 selected with LCB `0.10022672472473289`.
- Step 25/26 after action 8: anchor selected because LCB is below margin.
- RL total TTT: `5727.850138124504`
- P-Stack total TTT: `5730.792964723197`
- Gap: `-2.942826598692591` veh-h (`-0.05135112395104178%`)

해석:

- 추가 action8 반복은 이득을 만들지 않았다.
- Longer fitting은 r12의 ensemble variance를 줄여 z=1.96에서도 step24 단발 선택을 안정화했다.
- 다음 개선 후보는 repeated action이 아니라 step24의 action family/magnitude/owner 확장이다.

## 15. r13-r18: exploration과 online action-value update 검토

사용자 질문: "라벨링하지 말고 full-run을 끝까지 돌린 뒤 action value를 업데이트하면 되는 것 아닌가?"

답은 "가능하지만 단독으로는 noisy하다"이다. full-run replay는 실제로 선택한 action의 trajectory return을 제공하므로 Q-learning update에 사용할 수 있다. 다만 선택하지 않은 same-state counterfactual action은 관측되지 않고, delayed reward 때문에 한 번의 bad episode가 근처 state/action 전체를 지나치게 보수화할 수 있다.

r13 magnitude sweep:

- Output: `data\response_dqn_170_incident\recovery_ddqn_r13_step24_rfe_cornerpp_maggrid_h20_d0_bw5_v1.npz`
- Step 24 `linear_corner_pp` magnitudes `0.125,0.25,0.375,0.5` 비교.
- Best: `magnitude=0.25`, h20 gap `-1.1009644395408031` veh-h versus anchor.
- `magnitude=0.125` is tied with anchor, `0.375`/`0.5` are positive but weaker.

r14 all-action h20 sweep:

- Output: `data\response_dqn_170_incident\recovery_ddqn_r14_step24_all15_h20_d0_bw8_process_v1.npz`
- Step 24 all 15 actions under `R_F_E`, `magnitude=0.25`, `families=linear,quadratic,cross`.
- Best-equivalent actions: action 2 `linear_first_positive`, action 7 `linear_corner_pn`, action 8 `linear_corner_pp`; all yield h20 gap about `-1.10` veh-h.
- Most quadratic/cross actions are behaviorally tied with anchor at this state.
- Action 4 `linear_second_positive` is worse than anchor by about `+0.20` veh-h.
- Interpretation: nominally different leader prices can induce the same executable follower response, so response-level deduplication is essential.

r15 support2 training:

- Merged dataset: `data\response_dqn_170_incident\recovery_ddqn_r15_h20_step24_all15_after_a8_v1.npz`
- Model: `models\response_dqn_170_incident\recovery_ddqn_r15_h20_step24_all15_support2_lr1e4_50000_noboot_v1`
- Result: opening all support2 actions caused phantom positives at step 23/25/26. This confirmed that broader exploration without additional long-horizon evidence can make the policy unsafe.

r16 online exploratory full-run:

- Eval output: `results\response_dqn_170_incident\online_ddqn_r16_r15_eps035_steps23_26_eval_v1`
- Policy: r15 model, epsilon `0.35`, non-anchor allowed at steps `23,24,25,26`.
- Selected exploratory actions: step 23 action 11, step 25 action 7, step 26 action 6.
- Total TTT: `5741.286660904063`
- P-Stack total TTT: `5730.792964723197`
- Gap: `+10.493696180866209` veh-h, so this was a deliberately useful bad-experience trajectory.

r17 raw bootstrapped update:

- Dataset: `data\response_dqn_170_incident\online_ddqn_r17_h20_plus_eps035_fullrun_v1.npz`
- Model: `models\response_dqn_170_incident\online_ddqn_r17_h20_plus_eps035_bootstrap_support2_lr1e4_50000_v1`
- Result: bad exploratory actions were suppressed, but the verified step24 positive was also suppressed. Raw 1-step TD mixing was too conservative/noisy for this tiny batch.

r18 MC-window update:

- MC converter: `rl_leader.convert_response_replay_to_mc`
- Converted dataset: `data\response_dqn_170_incident\online_ddqn_r16_eps035_steps23_26_mc_return_v1.npz`
- Merged dataset: `data\response_dqn_170_incident\online_ddqn_r18_h20_plus_eps035_mc_steps23_26_v1.npz`
- Model: `models\response_dqn_170_incident\online_ddqn_r18_h20_plus_eps035_mc_support2_lr1e4_50000_noboot_v1`
- Deterministic eval: `results\response_dqn_170_incident\online_ddqn_r18_mc_z196_steps23_26_eval_v1`
- Action counts: anchor `74`, action 8 `1`
- Total TTT: `5727.850138124504`
- P-Stack total TTT: `5730.792964723197`
- Gap: `-2.942826598692591` veh-h (`-0.05135112395104178%`)

현재 결론:

- Full-run action-value update는 필요하다. 다만 raw TD replay를 그대로 섞는 것보다, selected problem window를 Monte Carlo return-to-go target으로 변환한 뒤 conservative selector로 평가하는 편이 현재 데이터 규모에서는 더 안정적이었다.
- Exploration은 deployment policy가 아니라 data-construction policy에서 수행해야 한다. Deployment는 P-Stack anchor를 포함한 conservative selector로 둔다.
- 다음 반복은 r18을 behavior policy로 사용해 epsilon full-run을 여러 seed로 추가 수집하고, 실제로 성능을 망친 selected state-action만 MC/counterfactual queue에 넣는 방식이 적절하다.

## 16. r19-r20: full-run action-value update의 한계와 advantage target 전환

사용자 질문: "라벨링하지 말고 그냥 끝까지 run해서 action value를 update하면 되는 것 아닌가? Backpropagation으로 해당 action value가 안 좋아질 텐데 왜 안 하는가?"

답은 "해야 한다. 다만 absolute return만으로는 같은 state의 anchor 대비 action advantage를 안정적으로 학습하기 어렵다"이다.

r19 exploratory full-run:

- Behavior model: `models\response_dqn_170_incident\online_ddqn_r18_h20_plus_eps035_mc_support2_lr1e4_50000_noboot_v1`
- Eval output: `results\response_dqn_170_incident\online_ddqn_r19_r18_eps025_steps20_30_eval_v1`
- Exploration: epsilon `0.25`, non-anchor allowed at control steps `20-30`.
- Action counts: anchor `66`, action 8 `4`, action 2 `3`, action 10 `1`, action 13 `1`.
- Total TTT: `6051.680118016705`
- P-Stack total TTT: `5730.792964723197`
- Gap: `+320.88715329350816` veh-h (`+5.599349955037284%`)
- Interpretation: this is a deliberately bad but useful exploratory trajectory. It gives long-horizon evidence for selected state-actions that were not previously explored.

r20 MC-window update:

- r19 replay was converted with `rl_leader.convert_response_replay_to_mc` for control steps `20-30`.
- Converted dataset: `data\response_dqn_170_incident\online_ddqn_r19_eps025_steps20_30_mc_return_v1.npz`
- Merged dataset: `data\response_dqn_170_incident\online_ddqn_r20_h20_plus_two_mc_exploration_v1.npz`
- Model: `models\response_dqn_170_incident\online_ddqn_r20_h20_plus_two_mc_support2_lr1e4_50000_noboot_v1`
- Training: no-bootstrap, reward scale `0.1`, learning rate `1e-4`, `50000` gradient steps, `min_action_support=2`.

r20 diagnostic:

- Diagnostic script: `rl_leader.diagnose_response_dqn_policy`
- Output: `results\response_dqn_170_incident\online_ddqn_r20_h20_plus_two_mc_policy_diagnostic_v1.json`
- On the frozen replay states, selected action counts were anchor `39`, action 8 `22`.
- In the r19 trajectory rows specifically, r20 selected action 8 at steps `20,21,22,23,24` and anchor at steps `25-30`.
- This is risky because the previously verified good behavior was action 8 only at step 24, not a broad step 20-24 repeated intervention.

Direct matched-state check:

- A narrow r21 recovery job was started for r19 altered states at steps `20-24`, comparing only anchor `0` and action 8.
- The job was stopped after step 20 because the full job would be too slow, but step 20 already produced decisive evidence in `results\response_dqn_170_incident\recovery_ddqn_r21_r19_steps20_24_anchor_vs_a8_h20_d0_bw8_v1.jsonl`.
- Same step20 state, h20 tail TTT:
  - Anchor: `2472.2132590405768`
  - Action 8: `2530.5432549948346`
  - Action 8 is worse by `+58.32999595425778` veh-h.
- Yet r20 selected action 8 at this same state. This shows that full-run absolute-return backprop did not provide a reliable same-state comparison against anchor.

Why absolute full-run Q update failed here:

- A full episode observes only the selected action at each visited state. It does not observe the anchor or other alternatives from that exact state.
- The total return is dominated by time/state severity. A bad action at an early congested state can still have a higher or lower absolute target than another action at a different state for reasons unrelated to the action.
- The deployed selector compares `Q(action) - Q(anchor)`, but the current training target is mostly absolute `-TTT`. Learning small action advantages as differences between large negative values is unnecessarily brittle.
- Mixing h20 counterfactual labels and full-episode MC labels creates horizon-scale mismatch unless the target is made anchor-relative.

Implementation update:

- Added `rl_leader.diagnose_response_dqn_policy` for fast frozen-replay Q ranking diagnostics before expensive full evals.
- Added `--reward-mode return|advantage` to `rl_leader.response_ddqn_recovery`.
- Default `return` mode preserves previous behavior: reward is `-tail_TTT`.
- New `advantage` mode stores `anchor_tail_TTT - action_tail_TTT`, so anchor is approximately `0`, good actions are positive, and bad actions are negative.
- This better matches the deployment decision rule, which is an anchor-relative conservative Delta-Q selector.

Next recommended loop:

1. Generate paired recovery labels in `--reward-mode advantage`, starting with the known sensitive steps `20-26`.
2. Train the response DQN on advantage targets instead of absolute return targets.
3. Keep exploration in data collection, but keep deployment conservative: action must beat anchor by LCB margin.
4. Use full-run MC only as a trigger for "which states/actions need paired comparison", not as the only value target.
5. Run deterministic full eval only after frozen-replay diagnostic shows action 8 is selected near step 24 but not prematurely at steps 20-23.
