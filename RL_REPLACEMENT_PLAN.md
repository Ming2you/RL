# RL Leader Replacement Plan

작성일: 2026-08-20

## 0. 목적

이 계획의 목적은 P-Stack의 최적화 leader를 하나의 RL policy로 대체하되, 그 외의 controller 동작은 유지하여 다음 질문에 답할 수 있는 실험 구조를 만드는 것이다. 최종 RL leader는 budget뿐 아니라 green, offset, metering, VSL에 대한 coordination signal도 동시에 출력한다.

> 동일한 plant, follower, 정보, feasibility projection, 상태 커밋, 안전장치 아래에서 MPC leader의 budget 탐색과 finite-difference price 생성을 RL의 budget 및 nonlinear coordination potential 생성으로 바꾸면 성능이 개선되는가?

이전 `code/env.py` 경로는 `StackelbergWuMeteredController.nash_solver.solve()`를 직접 호출했다. 따라서 leader search뿐 아니라 F1 follower 구성, marginal-price refresh, state-dependent link share, PFO fallback, dual-state commit, output closure까지 함께 우회했다. 해당 legacy 결과는 유망한 탐색 결과로 보존하되, "leader만 RL로 대체한 결과"로 해석하지 않는다. 새 production 경로는 `rl_leader/env.py`와 `RLStackelbergController`를 사용한다.

## 1. 성공 기준

### 1.1 구조적 성공 기준

- RL controller와 P-Stack의 차이는 leader가 budget과 coordination signal을 생성하는 방식뿐이다.
- RL actor 하나가 global budget과 green/offset/metering/VSL potential 계수를 동시에 출력한다.
- 같은 state, forecast, previous control, coordination action을 주면 두 경로의 follower control과 다음 plant state가 수치 허용오차 내에서 같다.
- F1 follower, potential 적용 방식, link share, feasible projection, dual commit, output closure, fallback guard가 두 경로에서 같다.
- RL arm에서는 native finite-difference price generator를 모두 비활성화하여 signal 소유권이 겹치지 않는다.
- 요청 action(intent), follower에 전달된 action(projected), 실제 action(realized)을 구분해 기록한다.

### 1.2 데이터 성공 기준

- 모든 transition에 episode 종료 원인이 기록된다: `natural`, `time_limit`, `wall_clock_abort`, `solver_error`, `safety_abort`.
- wall-clock abort가 혼잡 상태와 상관된 검열임을 전제로 처리하며, 단순히 정상 transition과 섞지 않는다.
- 학습 데이터에는 conservation/overflow/rejection/throughput 진단이 포함된다.
- 학습·validation·최종 test scenario가 파일 수준에서 분리되고, 최종 test는 모델과 하이퍼파라미터 확정 전까지 사용하지 않는다.

### 1.3 실험 성공 기준

- full RL leader와 P-Stack leader의 공정한 비교가 가능하다.
- 동일한 full actor에 channel mask를 적용해 budget, linear, quadratic, urban, freeway 기여를 분리할 수 있다.
- `gamma=0.99`와 undiscounted metric의 불일치를 제거하거나 명시적인 ablation으로 정당화한다.
- 최소 3개 학습 seed와 충분한 validation scenario에서 평균, 분산, 최악값을 보고한다.
- 최종 주장은 잠긴 test suite에서 한 번만 평가한 결과를 기준으로 한다.
- TTT 개선은 차량 보존, 처리량, overflow/rejection gate를 모두 통과해야 유효한 개선으로 인정한다.

## 2. 불변조건

아래 항목은 leader 교체 실험에서 controller 간 동일하게 유지한다.

| 구분 | 고정 항목 |
|---|---|
| Plant | topology, demand, METANET/urban dynamics, control interval, buffer, warmup |
| Follower | `F1WuFaithfulFollower`와 모든 follower 옵션 |
| Leader 이후 로직 | potential 적용, link-share update, projection, dual commit, output closure |
| Safety | PFO incumbent/fallback, regret/safety guard, physical action bounds |
| 정보 | state 및 forecast feature의 정의와 시점 |
| 평가 | warmup 제외 구간, TTT 회계, conservation/overflow/throughput gate |

MPC의 finite-difference price 생성은 follower가 아니라 leader의 책임이므로 교체 대상이다. P-Stack arm은 native linear price를 생성하고, RL arm은 native generator를 끈 뒤 학습된 linear/quadratic potential을 생성한다. 그 외의 차이는 별도 ablation 이름으로 분리한다.

## 3. 목표 아키텍처

### 3.1 핵심 원칙

`RLLeaderEnv`가 follower를 직접 소유하지 않게 한다. 대신 production P-Stack controller가 전체 결정 파이프라인을 소유하고, leader coordination action을 만드는 지점만 교체한다.

```text
state + forecast + previous control
                |
                v
   shared state/forecast contract
   + F1 follower configuration
                |
                v
   CoordinationActionProvider
   MPC: budget search + FD linear price
   RL : budget + full nonlinear potential
                |
                v
   shared potential adapter
   + link share + projection
   + follower solve
                |
                v
   shared fallback + dual commit
   + output closure + diagnostics
                |
                v
          plant step
```

### 3.2 최소 구현안

새 controller는 `F1StackelbergWuMeteredController`를 상속한다.

- `CoordinationAction`: budget, potential blocks, references, masks를 담는 공통 schema다.
- `OptimizerCoordinationProvider`: 기존 P-Stack budget search와 finite-difference linear price를 공통 schema로 노출한다.
- `RLCoordinationProvider`: observation을 만들고 actor 출력으로 full `CoordinationAction`을 생성한다.
- `CoordinationPotentialAdapter`: 공통 schema를 기존 follower의 local candidate cost에 적용한다.
- `RLStackelbergController`: RL provider가 낸 action을 기존 projection/follower/commit 경로에 주입한다.
- `RLLeaderEnv`: scenario, simulator, controller, reward/trace를 연결하는 얇은 wrapper만 담당한다.

가능하면 `StackelbergMPCController.decide_with_info()`의 전후 처리 코드를 복제하지 않는다. coordination action을 주입할 수 있는 작은 extension point를 추가하고, optimizer와 RL이 같은 projection, follower, safety, commit 경로를 통과하게 한다.

### 3.3 통합 action 및 potential 계약

RL actor는 처음부터 다음 출력을 모두 가진다.

- global budget: `(N_P_star, N_UF_star)`
- urban blocks: 각 intersection의 green-offset potential
- freeway blocks: 각 merge/link의 metering-VSL potential
- channel masks: budget, green, offset, metering, VSL, linear, quadratic, cross

기존 RL의 고정 `N_P in [0, 2200]`는 제거한다. Budget은 `cfg.leader` 범위, `Leader._candidate_bounds(...)`, 기존 follower-feasible projection을 순서대로 통과한다. Trace에는 raw, bounded, projected, realized budget을 모두 저장한다.

Potential은 follower 소유권에 맞춰 block-local로 제한한다. 모든 lever를 연결하는 dense Hessian은 사용하지 않는다.

Urban intersection `i`:

```text
z_i   = normalized [green_i - green_ref_i, offset_i - offset_ref_i]
Phi_i = g_i^T z_i + 0.5 * ||L_i^T z_i||^2
```

Freeway local block `m`:

```text
z_m   = normalized [local metering deltas, local VSL deltas]
Phi_m = g_m^T z_m + 0.5 * ||L_m^T z_m||^2
```

`H = L L^T` parameterization으로 quadratic curvature를 positive semidefinite로 유지한다. Freeway block이 커지면 full Cholesky 대신 diagonal + low-rank PSD를 사용한다. Cross-term은 같은 intersection 또는 물리적으로 인접한 merge/link 안에서만 허용한다.

모든 delta는 기존 trust radius로 정규화한다. Actor 출력은 bounded transform을 거치며, reference는 직전 intent가 아니라 직전 realized control을 사용한다. Zero potential은 follower의 unpriced local objective와 정확히 같아야 하고, linear-only mask는 현재 marginal-price 인터페이스와 같아야 한다.

RL arm에서는 signal, metering, VSL, offset native price generator를 모두 끈다. Mask로 채널을 끌 때 해당 계수는 0이 되며 MPC price로 fallback하지 않는다. P-Stack arm만 native finite-difference linear price를 소유한다.

### 3.4 observation 계약

관측은 leader가 실제 사용하는 정보와 맞춘다. 최소 feature 집합은 다음과 같다.

- 시간: normalized phase, peak/recovery indicator
- Urban: 방향/권역별 accumulation, movement queue, boundary queue, storage pressure
- Freeway: link별 mean/max density, rho-critical exceedance, speed, origin queue
- Ramp: ramp별 queue, metering realization, receiving headroom
- Forecast: horizon의 first/mean/peak demand, link별 lane loss 및 incident indicator
- Previous action: intent가 아니라 projected 및 realized budget, 실제 metering 합
- Controller state: 필요한 경우 normalized committed dual state와 price reference

관측 feature 이름과 순서를 schema 파일 또는 명시적 dataclass로 고정한다. checkpoint에는 schema version을 저장하고 불일치 시 load를 실패시킨다.

## 4. 단계별 실행 계획

### Phase 0. 재현 가능한 실행 기반 복구

목표: fresh clone에서 학습과 평가 명령이 실제로 실행되게 한다.

작업:

1. `rl_leader.*` import를 현재 `code/` 구조에 맞게 정리하거나 명확한 package로 이동한다.
2. `C:/torchlib` 하드코딩을 제거하고 표준 environment/venv import를 사용한다.
3. `requirements.txt`와 Python/Torch 지원 버전을 고정한다.
4. baseline loader가 `data/holdout/*.csv`를 읽도록 수정한다.
5. 단일 `python -m ...` 실행 방식을 문서화한다.
6. checkpoint 저장 시 config, observation schema, action schema, git commit을 함께 저장한다.

완료 조건:

- clean environment에서 env smoke, IQL `--help`, 10-step 학습, 1-cell short evaluation이 성공한다.
- 절대경로 또는 특정 사용자 디렉터리 없이 실행된다.
- 잘못된 checkpoint/schema 조합은 명확한 오류로 중단된다.

### Phase 1. Leader-only replacement seam 구현

목표: optimizer와 RL이 동일한 controller 파이프라인을 통과하고, leader가 생성한 coordination action만 교체할 수 있게 한다.

작업:

1. production P-Stack 경로의 pre/post-decision 동작 목록을 contract test로 먼저 고정한다.
2. `CoordinationActionProvider` extension point를 최소 범위로 추가한다.
3. `RLStackelbergController`와 얇은 `RLLeaderEnv`를 구현한다.
4. 기존 direct `nash_solver.solve()` 경로는 legacy로 표시하고 새 학습에는 사용하지 않는다.
5. optimizer provider가 기존 P-Stack budget과 native linear price를 그대로 재현하는지 회귀 검증한다.
6. deterministic RL stub provider로 알려진 budget과 zero potential을 주입한다.

필수 contract test:

- 동일한 coordination action을 주면 optimizer-injection 경로와 RL-injection 경로의 follower control이 같다.
- F1 follower type과 주요 옵션이 같다.
- optimizer와 RL의 signal ownership이 겹치지 않는다.
- state-dependent link share가 같다.
- projected/realized action과 output closure가 같다.
- 선택 action의 dual next value가 다음 step standing state에 commit된다.
- fallback 조건에서 두 경로가 같은 PFO control을 선택한다.
- plant next state와 step TTT가 허용오차 내에서 같다.

완료 조건:

- 위 contract test가 모두 통과한다.
- optimizer provider를 사용한 새 controller가 기존 P-Stack short rollout과 동일하다.
- RL provider 사용 시 diagnostics에서 변경점이 provider 종류 하나로 제한된다.

### Phase 2. Full coordination action과 nonlinear potential 구현

목표: green, offset, metering, VSL을 처음부터 하나의 action schema와 actor 출력에 포함한다.

작업:

1. 현재 follower agent와 lever 소유권으로부터 deterministic block map을 만든다.
2. `CoordinationAction` schema에 budget, block coefficients, reference, masks, schema version을 정의한다.
3. Urban intersection별 green-offset `2x2 PSD` potential adapter를 구현한다.
4. Freeway merge/link별 metering-VSL `diagonal + low-rank PSD` potential adapter를 구현한다.
5. linear-only, quadratic-only, cross-off, urban-only, freeway-only, budget-only mask를 구현한다.
6. RL provider 활성 시 모든 native price generator가 비활성화되는 ownership switch를 구현한다.
7. raw actor output의 bounded transform, trust normalization, coefficient scaling을 한 곳에서 수행한다.
8. intent/projected/realized budget과 각 potential term의 local cost 기여를 trace한다.

필수 probe:

- zero potential이 unpriced F1 follower와 같다.
- linear coefficient만 활성화하면 기존 marginal-price 인터페이스와 같다.
- positive quadratic curvature가 reference에서 멀어지는 action의 비용을 증가시킨다.
- green-offset cross는 해당 intersection 밖의 follower cost를 바꾸지 않는다.
- metering-VSL cross는 소유 merge/link 밖의 follower cost를 바꾸지 않는다.
- extreme actor output도 coefficient bound, trust region, physical projection을 벗어나지 않는다.

완료 조건:

- 모든 lever가 full schema에 존재하고 full actor output에서 누락되지 않는다.
- 같은 `CoordinationAction`을 두 provider 경로에 주입했을 때 follower response가 같다.
- RL arm에서 native price refresh count가 0이고 potential 계수만 follower objective에 들어간다.
- mask 조합에 따라 action tensor shape가 변하지 않는다.

### Phase 3. Reward와 회계 안전성 보강

목표: TTT를 낮추기 위해 차량을 projection/rejection으로 제거하는 경로를 차단한다.

작업:

1. 매 step 다음 값을 env info와 trace에 추가한다.
   - movement queue projection vehicles
   - rejected off-ramp vehicles
   - ramp/urban overflow count and duration
   - external arrivals, completed departures, inventory change
   - conservation residual
   - throughput/completion count
2. 학습 reward와 별개로 hard validity gate를 정의한다.
3. conservation 위반 또는 projection/rejection 발생 시 transition에 failure flag를 남긴다.
4. terminal failure cost를 실제 남은 demand/queue의 하한에 맞춰 정의한다.

유효성 gate 초안:

```text
movement_queue_projection_veh == 0
coupling_offramp_arrivals_rejected_veh == 0
conservation_residual <= tolerance
ramp_queue_overflow_duration <= allowed_limit
throughput is not materially below the matched baseline
```

완료 조건:

- 의도적으로 overflow를 유발하는 정책이 낮은 TTT만으로 승자로 판정되지 않는다.
- 모든 결과 표에 TTT와 validity gate 결과가 함께 출력된다.
- 기존 winning checkpoint가 gate를 통과하는지 별도 legacy audit 결과를 남긴다.

### Phase 4. Full-action offline dataset 재구성

목표: 모든 budget/potential 채널의 반응을 학습할 수 있는 데이터를 만들고, wall-clock과 혼잡이 상관된 검열 편향을 제거하거나 측정 가능하게 만든다.

작업:

1. 기존 27,706 transition을 `legacy_v1`로 동결하고 원본을 수정하지 않는다.
2. `legacy_v1`은 budget trunk 초기화나 비교에만 사용하고, full potential head 학습의 근거로 사용하지 않는다.
3. 새 `full_action_v2` transition에 full action, mask, behavior mode, block-local follower response를 저장한다.
4. 다음 structured exploration을 혼합한다.
   - P-Stack/native linear-price trajectory를 공통 schema로 변환한 anchor
   - 한 local block만 perturb하는 trajectory
   - 같은 block 안의 lever를 함께 perturb하는 trajectory
   - 여러 block을 correlated하게 움직이는 full-action trajectory
   - zero/linear/quadratic channel mask trajectory
   - epsilon-mixture behavior: `1-epsilon`은 anchor/current policy, `epsilon`은 위 local perturbation 중 하나
5. epsilon은 전역 47차원 uniform random에 쓰지 않는다. 기본 수집 스케줄은 `0.6 -> 0.1`로 감소시키고, 선택된 behavior mode와 확률을 transition마다 저장한다.
6. 초기 `full_action_v2` 학습 뒤에는 3-seed actor/critic ensemble의 block별 disagreement가 큰 상태와 block을 우선 재수집한다. exploit action은 ensemble mean 또는 Thompson sample, explore action은 해당 block의 trust-region perturbation으로 만든다.
7. episode metadata에 종료 원인과 마지막 state 진단을 저장한다.
8. wall-clock abort는 `done=0`이라는 이유만으로 정상 trajectory와 동일 취급하지 않는다.
9. abort state를 snapshot으로 저장하고 이후 별도 worker에서 resume할 수 있게 한다.
10. resume가 불가능한 경우 pessimistic continuation lower bound 또는 failure terminal cost를 사용한다.
11. complete/aborted dataset의 state, reward, queue, action-block 분포를 매 수집 run마다 비교한다.
12. behavior policy 확률, epsilon, 실제 explore 여부와 mode를 transition 단위로 저장한다.

필수 ablation:

- complete episode only
- legacy nonterminal bootstrap
- pessimistic abort handling
- resumed complete trajectories
- single-block only versus structured full-action exploration

완료 조건:

- abort 비율과 abort/complete 분포 차이가 보고된다.
- main result가 특정 abort 처리 하나에만 의존하지 않는다.
- 각 lever block의 action range와 local response coverage가 보고된다.
- 데이터 manifest에 schema version, scenario seed, episode 수, transition 수, 종료 원인 및 mask 분포가 기록된다.

### Phase 5. 학습 목적과 support 제약 정렬

목표: offline RL objective를 평가 metric인 windowed total TTT와 맞춘다.

작업:

1. finite-horizon 기본값을 `gamma=1.0`으로 두고 `0.99`를 ablation으로 비교한다.
2. time feature와 terminal 처리의 일관성을 검증한다.
3. train/validation loss 외에 validation policy의 constraint-aware metric을 기록한다.
4. actor checkpoint 선택 규칙을 사전에 고정한다.
5. 전체 action뿐 아니라 block별 policy action distribution과 dataset support 거리를 기록한다.
6. support 밖 coefficient에는 behavior regularization 또는 conservative penalty를 적용한다.

완료 조건:

- `gamma=1.0`과 `0.99` 결과가 동일 protocol로 비교된다.
- checkpoint 선택에 최종 test cell이 사용되지 않는다.
- policy action이 dataset support 밖으로 나가는 비율이 전체 및 block별로 보고된다.

### Phase 6. Unified full actor 학습

목표: 하나의 actor가 budget과 모든 block-local potential 계수를 동시에 출력하게 한다.

작업:

1. actor trunk는 global observation을 사용하고, urban/freeway agent family별 parameter-shared head를 둔다.
2. budget, linear, quadratic/cross coefficient를 한 forward pass에서 출력한다.
3. curvature head는 PSD parameterization과 bounded scale을 구조적으로 만족한다.
4. `full_action_v2`를 주 데이터로 사용하고, `legacy_v1`은 호환 가능한 budget trunk 초기화에만 선택적으로 사용한다.
5. 학습 중 channel dropout을 적용하여 동일 checkpoint가 mask ablation을 견디게 한다.
6. 기본 offline 알고리즘은 IQL을 유지하되 high-dimensional action에서 AWR weight와 behavior support를 다시 검증한다.
7. 최소 3개 seed를 학습하고 seed별 coefficient/action distribution을 저장한다.
8. 1차 학습 뒤 ensemble disagreement와 validation 실패 상태를 이용해 `full_action_v2`를 한 차례 이상 증분 수집하고 재학습한다. IQL 개선 정체는 이 active collection 전후로 비교한다.

동일 checkpoint mask ablation:

- `RL-FULL`: 모든 channel 활성
- `RL-LINEAR`: quadratic과 cross를 0
- `RL-BUDGET`: 모든 potential을 0
- `RL-URBAN`: budget + green/offset
- `RL-FREEWAY`: budget + metering/VSL
- `RL-NO-CROSS`: linear + diagonal quadratic만 활성

Mask ablation은 빠른 메커니즘 진단용이다. 핵심 결론은 co-adaptation 영향을 배제하기 위해 주요 arm을 mask와 동일한 action space로 별도 재학습하여 한 번 더 확인한다.

완료 조건:

- full actor가 모든 output head를 실제로 사용하며 dead channel 비율이 보고된다.
- seed별 full policy가 validity gate를 통과한다.
- mask ablation으로 성능 변화의 urban/freeway, linear/quadratic 귀속이 가능하다.
- 주요 효과가 matched retraining에서도 방향이 유지된다.

### Phase 7. 평가 protocol 재설계

목표: validation과 최종 test를 분리하고 작은 개선의 불확실성을 정량화한다.

Scenario 분할:

- Train: 기존 domain randomization 범위와 새 수집 scenario
- Validation: 기존 `190-skew`, `190-incident` 포함. tuning과 mechanism 분석에 사용
- Test: 잠긴 신규 scenario manifest

최종 test manifest에는 최소한 다음 축을 포함한다.

- demand level 여러 개
- skew 방향과 크기
- incident link/segment/start/duration
- incident 없는 고수요
- recovery가 긴 경우
- calibration 범위 경계와 약한 OOD

평가 arm:

| Arm | Leader | Price/Follower/Safety |
|---|---|---|
| PFO | 없음 | production PFO |
| P-Stack | MPC budget + native linear price | production 고정 |
| RL-FULL | RL budget + all nonlinear potential | native price OFF, follower/safety 동일 |
| RL-LINEAR | full actor, quadratic/cross masked | native price OFF, follower/safety 동일 |
| RL-BUDGET | full actor, all potential masked | native price OFF, follower/safety 동일 |
| RL-URBAN | budget + green/offset potential | native price OFF, follower/safety 동일 |
| RL-FREEWAY | budget + metering/VSL potential | native price OFF, follower/safety 동일 |
| P-CENT | centralized | 상한 참조 |

보고 지표:

- windowed total/urban/freeway TTT
- mean, median, standard deviation, worst case
- paired difference versus P-Stack
- throughput and completed vehicles
- overflow/rejection/conservation gate
- runtime and truncation rate
- action intent/projected/realized statistics

완료 조건:

- 최소 3개 학습 seed를 모든 validation scenario에서 평가한다.
- 최종 test 전 모델과 하이퍼파라미터를 freeze한다.
- 최종 결과는 seed와 scenario에 대한 paired 통계와 최악값을 포함한다.

## 5. 테스트 매트릭스

| 계층 | 테스트 | 목적 |
|---|---|---|
| Unit | action scaling/bounds/schema | action 계약 검증 |
| Unit | PSD/low-rank potential parameterization | quadratic 안정성 검증 |
| Unit | channel masks keep fixed shape | 단일 full actor ablation 검증 |
| Unit | lever ownership/locality | 분산 follower 경계 검증 |
| Unit | zero/linear potential parity | 기존 follower/price 회귀 검증 |
| Unit | observation schema/version | checkpoint 호환성 검증 |
| Unit | abort classification | 검열 처리 검증 |
| Unit | conservation accounting | reward loophole 차단 |
| Contract | optimizer vs RL coordination-action injection | leader-only 교체 검증 |
| Contract | native-price ownership switch | 이중 price 적용 방지 |
| Contract | potential/link/dual/closure parity | controller state parity 검증 |
| Integration | short deterministic rollout parity | next state와 TTT 동일성 검증 |
| Integration | safety fallback trigger | PFO guard 유지 검증 |
| Data | block action/response coverage | full-action support 검증 |
| Data | complete vs abort distribution report | censoring 측정 |
| Evaluation | baseline CSV fixture | metric/time-window 회귀 방지 |

## 6. 산출물

- `RL_REPLACEMENT_PLAN.md`: 본 계획서
- versioned observation/action/ownership schema
- `CoordinationAction`, provider, potential adapter
- `RLStackelbergController`와 unified full actor
- active RL env에 대한 unit/contract/integration tests
- full-action dataset manifest, coverage report, abort audit report
- validation scenario manifest
- 잠긴 final test manifest
- seed별 평가 CSV와 paired summary
- legacy checkpoint validity audit

## 7. 우선순위와 중단 기준

### P0: 다음 실험 전에 반드시 완료

1. fresh-clone 실행 복구
2. leader-only replacement seam
3. full coordination action schema와 potential adapter
4. ownership/parity contract tests
5. conservation/overflow/rejection trace와 validity gate

### P1: 새 학습 전에 완료

1. full-action structured exploration dataset
2. dataset censoring 처리
3. observation/action schema 정렬
4. `gamma` objective 정렬
5. train/validation/test manifest 분리

### P2: 구조 검증 후 진행

1. 3-seed unified full-action IQL
2. channel-mask 및 matched-retraining ablation
3. mechanism attribution

다음 조건 중 하나가 발생하면 장시간 학습을 중단하고 구조 단계로 돌아간다.

- contract parity 실패
- projection/rejection으로 인한 invalid TTT
- abort 처리에 따라 승패가 뒤집힘
- 전체 또는 특정 lever block action이 dataset support 밖에서 주로 선택됨
- full actor의 주요 channel이 dead head로 남음
- validation 개선이 seed 또는 단일 scenario 하나에만 집중됨

## 8. 첫 번째 구현 묶음

첫 PR 또는 첫 작업 묶음은 다음 범위로 제한한다.

1. import/path 및 baseline loader 복구
2. versioned `CoordinationAction`과 lever ownership map 정의
3. `CoordinationActionProvider` seam 추가
4. zero/linear potential adapter 구현
5. optimizer provider 회귀 test
6. deterministic RL stub provider와 contract test
7. direct `nash_solver.solve()` 경로를 legacy로 표시

이 묶음에서는 IQL 재학습이나 대규모 데이터 수집을 하지 않는다. 먼저 zero potential과 기존 linear price가 새 adapter에서 정확히 재현되는지 증명한다. 다음 묶음에서 green-offset PSD와 metering-VSL PSD block을 모두 구현한 뒤 `full_action_v2` 수집으로 넘어간다.

## 9. 결과 해석 규칙

- TTT가 낮아도 validity gate를 실패하면 성능 개선으로 보고하지 않는다.
- validation에서 반복 관찰한 190-skew/incident를 "test 전용"이라고 부르지 않는다.
- RL-FULL과 P-Stack의 차이는 budget과 coordination signal을 생성하는 leader 방식으로 한정한다.
- RL-BUDGET, RL-URBAN, RL-FREEWAY, RL-LINEAR는 동일 full actor의 mask 진단이며 단독 최종 결론으로 사용하지 않는다.
- 중요한 channel 효과는 해당 action space로 matched retraining한 결과에서도 확인한다.
- Quadratic 이득은 linear-only 대비 차이로, cross 이득은 no-cross 대비 차이로 분리해 보고한다.
- 메커니즘 주장은 trace와 ablation이 일치할 때만 채택한다.

## 10. 2026-08-20 파일럿 실행 현황

이번 실행은 Phase 0부터 Phase 7까지의 **코드 경로와 실험 protocol을 검증하는 소규모 파일럿**이다. 최종 성능 실험은 아니다.

완료:

1. `code/`를 `rl_leader/` package로 이동하고 하드코딩된 Torch 경로와 holdout loader를 정리했다.
2. 117차원 observation과 47차원 full coordination action schema를 구현했다.
3. budget, green-offset, metering-VSL의 block-local linear/PSD quadratic potential과 channel mask를 구현했다.
4. RL 경로가 production follower, projection, fallback, dual commit, output closure를 사용하며 native price refresh를 0으로 유지하도록 연결했다.
5. conservation, projection, rejection, overflow, throughput validity trace를 추가했다. one-step conservation residual은 약 `1.5e-12 veh`였다.
6. optimizer anchor, budget, linear, quadratic, single-block, correlated-block, full-action 및 epsilon-mixture 수집 모드를 구현했다.
7. epsilon-mixture는 `1-epsilon` budget/anchor exploitation과 `epsilon` local-block exploration을 사용하며 기본 스케줄은 `0.6 -> 0.1`이다. 47차원 전체 uniform random은 사용하지 않는다.
8. 20-transition pilot dataset을 audit한 결과 모든 transition이 validity gate를 통과했고 모든 structured behavior mode가 포함됐다.
9. unified full actor IQL을 3개 seed로 각 40,000 update 학습하고 6개 mask를 실행했다. 18개 short evaluation row가 모두 validity gate와 native-price ownership 검사를 통과했다.
10. 관련 regression/unit test는 `27/27` 통과했다. 전체 기존 suite는 `288`개 중 `13 failure`, `4 error`, `3 skip`이며, RL 변경과 무관한 기존 controller 기대값 및 로컬 Torch import 문제를 별도 부채로 남긴다.

파일럿에서 확인된 한계:

- 학습 데이터가 20 transition뿐이어서 TTT 성능을 판단할 수 없다.
- IQL critic/advantage가 빠르게 거의 0으로 수렴했고 policy action의 support 이탈률이 약 `15-25%`였다. 이는 update 수보다 behavior coverage 부족을 먼저 해결해야 한다는 신호다.
- 3-seed ensemble audit의 다음 탐색 우선순위는 `B`, `R_D_E`, `R_D_W`, `A`, `F` block이다. 현재 seed disagreement 절대값은 작으므로 순위는 초기 heuristic으로만 사용한다.
- 1-step mask 평가는 실행 계약과 회계 검증일 뿐 control 효과의 성능 증거가 아니다.

다음 실행 순서:

1. 위 우선 block에 epsilon-mixture/ensemble-guided perturbation을 집중하여 peak와 recovery를 포함한 full-horizon `full_action_v2`를 증분 수집한다.
2. abort/complete 분포와 block별 action-response coverage를 audit하고 support gap이 남으면 다시 수집한다.
3. 확장 데이터로 IQL 3개 seed를 재학습하고 `gamma=1.0/0.99`, support penalty, channel dropout을 validation에서 선택한다.
4. 모든 seed를 6개 mask와 PFO/P-Stack 기준선에 대해 full-horizon validation한다.
5. validity gate를 통과한 모델만 freeze하고 잠긴 test manifest를 한 번 평가한다.

## 11. 2026-08-20 1-5단계 실행 결과

이번 실행 범위는 runtime profiling, short sanity, active collection, dataset audit, 3-seed IQL 재학습까지다. full-horizon 성능 비교는 아직 수행하지 않았다.

1. RL-FULL 3-step profiling에서 actor 추론은 평균 `0.00078 s`, 환경 step은 평균 `17.913 s`였다. 이 중 follower solve가 `9.783 s`, plant/controller 부대 비용이 `8.130 s`였다. 4-worker 병렬 one-step은 wall time `20.6 s`로 약 `3.5x` 처리량을 보여 이후 수집은 4-worker로 고정했다.
2. `sweet_190_skew15_w60`과 `sweet_190_incident_w60`을 seed 0/1/2, 각 10-step으로 실행했다. 60개 step이 모두 validity gate를 통과했고 평균 누적 TTT는 각각 `834.553`, `854.392 veh-h`였다. incident 관측에서 nested `freeway_lane_loss`를 scalar로 처리하던 오류를 수정하고 regression test를 추가했다.
3. 우선 block `B,R_D_E,R_D_W,A,F`를 70% 확률로 겨냥하는 structured exploration을 추가했다. 4-worker 수집으로 23개 natural episode, `1,725`개 신규 transition을 확보했으며 validity pass rate는 `100%`다. 기존 pilot 20개를 포함한 실제 학습 입력은 8개 파일, `1,745` transition이다. epsilon-mixture 구간의 실제 explore 비율은 `23.56%`였다.
4. 모든 47개 action dimension은 데이터에서 변동해 `dead_action_fraction=0`이었다. 반면 realized follower response에서는 `D/F offset`이 항상 0이고 네 ramp 소유 segment의 VSL이 항상 `100 km/h`였다. 가격 전달은 활성화되어 있으므로 현재 진단은 actor 출력 고장보다 ramp-offset local solver와 이산 VSL 후보/목적함수의 응답성 부족이다. 원인을 확정하려면 후보별 local objective와 선택 guard를 trace해야 한다.
5. 확장 데이터로 IQL을 seed 0/1/2, 각 40,000 update 재학습했다(`gamma=1.0`, channel dropout `0.1`). `support_weight=0.1`의 전체 action seed별 support 이탈률은 `9.60% / 11.53% / 11.11%`, 앙상블 평균 기준은 `10.02%`였다. `support_weight=0.3`은 seed별 `10.86% / 11.74% / 9.09%`, 앙상블 평균 `9.31%`였다. 두 설정 모두 actor output의 dead dimension은 0이지만, 강한 soft penalty의 개선은 작고 seed에 일관되지 않았다.

판정:

- `support_weight=0.3` checkpoint는 비교 후보로 보존하되 robust한 최종 정책으로 채택하지 않는다. 다음 학습 변경은 단순 가중치 증대보다 support-bounded actor parameterization 또는 명시적 action projection으로 검증한다.
- full-action TTT 성능 평가는 `D/F offset`과 VSL response가 실제로 변하도록 follower candidate/ownership/guard를 교정하고 targeted recollection을 마친 뒤 진행한다. 현재 정책으로 Phase 7 결론을 내리면 dead lever를 RL 효과로 잘못 해석할 수 있다.
- native optimizer-anchor episode는 단일 P-Stack step이 30분을 넘겨 중단됐다. 현 collector timeout은 step 반환 뒤에만 검사하므로, optimizer anchor를 다시 사용할 때는 solve 자체를 process-level timeout으로 격리해야 한다.

## 12. 2026-08-20 14,400초 진단 실행 결과와 수정 계획

`support_weight=0.3` RL-FULL checkpoint를 seed 0/1/2에 대해 `sweet_190_skew15_w60`, `sweet_190_incident_w60`에서 실행했다. 각 run은 warmup 900초 뒤 75 policy step을 수행해 simulation time 14,400초에 도달했다. 후보 진단은 opt-in trace로 켰으며, 1-step plain/diagnostic 비교에서 TTT가 일치해 계측이 follower 선택을 바꾸지 않음을 먼저 확인했다.

TTT 결과에서 `개선율`은 기준선보다 낮은 TTT가 양수다.

| Scenario | RL-FULL TTT, mean ± std (veh-h) | NC | PFO | P-Stack | P-CENT | vs PFO | vs P-Stack |
|---|---:|---:|---:|---:|---:|---:|---:|
| skew15 | `5,994.871 ± 56.295` | `6,881.942` | `6,299.302` | `6,378.869` | `5,757.359` | `+4.833%` | `+6.020%` |
| incident | `8,740.002 ± 270.264` | `8,555.873` | `9,229.987` | `8,386.237` | `8,016.293` | `+5.309%` | `-4.218%` |

- skew15 phase 평균은 peak `3,455.250`, recovery `2,539.621 veh-h`다.
- incident phase 평균은 peak `4,241.925`, recovery `4,498.077 veh-h`다. 짧은 10-step sanity가 보지 못한 recovery 혼잡에서 차이가 크게 누적된다.
- 6개 run, 450 policy step이 모두 기존 validity gate를 통과했다. 처리량 평균은 skew15 `35,306.710`, incident `35,308.024 veh`로 기준선 범위 `35,300-35,329 veh` 안이다. 다만 현재 validity 식에는 throughput 하한이 직접 포함되지 않으므로 결과 표에서 별도 지표로 계속 검사한다.
- action support 이탈률 평균은 skew15 `11.735%`, incident `10.638%`다. soft support penalty만으로 최종 정책을 freeze하기에는 여전히 높다.

Full trace에서 확정한 원인:

1. **Key mapping과 adapter 전달은 정상이다.** 6개 run 모두 requested/bounded coordination에서 follower receipt까지 linear 및 quadratic 계수 최대 절대오차가 `0`이었다. VSL 가격 키도 ramp merge segment인 `FW_*__seg3`, `FW_*__seg5`에 정확히 도착했다.
2. **D/F offset은 탐색 부족이 아니라 activation gate에 막힌다.** D와 F 각각 450/450 step에서 가격이 존재했지만 `ramp_offset_enabled=False`로 후보 생성 전에 `offset=0`을 반환했다. 총 900개의 `ramp_offset_disabled` skip이 재현됐다.
3. **VSL 가격은 후보를 구별하지 못한다.** 네 가격 대상 segment의 VSL은 모든 run과 혼잡 step에서 `100 km/h`였다. 대상 segment에서 `rho/rho_crit`는 skew15 최대 `2.051`, incident 최대 `3.028`까지 상승했으므로 혼잡 부재가 원인이 아니다.
4. 총 9,646번의 VSL local solve에서 raw 후보는 평균 12개였지만, 가격 대상 `seg3/seg5`의 후보 간 변화율은 모두 `0%`였고 linear+quadratic+cross 가격 비용 spread도 `0`이었다. 후보 생성기가 가장 이른 병목 상류 segment만 바꾸고 나머지를 최대 VSL로 복구하기 때문이다. 평균 smoothness cost spread는 `11.0`, base cost spread는 약 `0.34`로 현재 선택 차이는 가격보다 기존 smoothness 항이 지배한다.
5. `vsl_set`의 no-VSL anchor는 `115 km/h`지만 action 범위는 `60-106 km/h`, VSL trust radius는 `10 km/h`다. 초기 reference 115에서 첫 활성 후보 100까지 거리가 15라서 약 `1.5%`의 local solve가 trust 후보를 하나도 남기지 못하고 fallback했다. 이것은 주원인인 priced-segment 후보 불변과 별도로 정리해야 할 범위 불일치다.

따라서 현재 결과는 budget, green, metering 등 살아 있는 경로의 효과와 dead offset/VSL 경로가 섞인 진단 기준점이다. skew15에서 PFO/P-Stack보다 낮은 TTT는 유망하지만 incident에서 P-Stack보다 나쁘고 seed 편차가 크므로 최종 RL leader 성능으로 확정하지 않는다.

다음 실행 순서:

1. RL coordination adapter가 offset channel을 활성화했을 때만 ramp-aware D/F offset 후보 경로를 켠다. PFO/P-Stack의 기본 follower 설정은 바꾸지 않는다.
2. VSL 후보 생성기가 가격 소유 `seg3/seg5`를 trust region 안에서 실제로 변동시키도록 고치고, `vsl_set`, action bound, no-VSL anchor, trust radius를 하나의 계약으로 정렬한다.
3. D/F offset과 네 VSL 가격 키 각각에 대해 nonzero candidate variation, nonzero price-cost spread, 선택 반응을 검증하는 unit/integration test를 추가한다.
4. 기존 3개 checkpoint를 그대로 같은 6-run manifest에 재평가해 follower 계약 수정만의 영향을 격리한다. 이 결과는 진단용이며 최종 성능 결론으로 사용하지 않는다.
5. 교정된 follower로 혼잡 peak/recovery를 포함한 targeted `full_action_v2`를 다시 수집하고, support-bounded actor 또는 명시적 projection을 적용해 3-seed IQL을 재학습한다.
6. 재학습 actor를 RL-FULL과 동일 actor mask ablation으로 평가하고 PFO/P-Stack/P-CENT와 throughput, validity, TTT를 함께 비교한다.

## 13. 2026-08-20~21 응답 계약 교정, 12시간 재수집 및 재평가

### 13.1 교정한 follower 응답 계약

RL 경로에만 `rl_offset_vsl_response_v1` 계약을 적용했다. PFO/P-Stack의 기본 follower 설정은 유지한다.

1. RL coordination adapter가 D/F ramp offset 후보를 항상 활성화한다.
2. 가격이 부여된 VSL segment `seg3/seg5`를 local candidate가 실제로 변화시키도록 single-segment 후보를 추가했다.
3. VSL trust radius를 follower의 `max_vsl_step`과 일치시켜 초기 `115 -> 100 km/h` 후보가 잘리는 범위 불일치를 제거했다.
4. dataset, checkpoint, evaluation trace에 response contract를 기록하고 서로 다른 계약의 dataset 혼합 학습을 거부한다.
5. targeted regression test에서 D/F offset 활성화, 가격 대상 VSL 후보 변화, nonzero price-cost spread를 검증한다.

파일럿 300 transition에서 D/F offset은 `0~105 s`, 네 ramp VSL은 `80~115 km/h`로 실제 변화했고 모든 제어 블록의 dead response가 0이었다. 기존 checkpoint one-step 재평가에서도 priced VSL segment variation fraction은 `100%`, trust fallback은 `0%`였다.

### 13.2 12시간 데이터 수집 결과

8개 worker를 2026-08-20 약 20:40 KST부터 2026-08-21 약 08:40 KST까지 wall-clock 기준 12시간 실행했다. transition 목표 개수는 종료 조건으로 사용하지 않았고, 각 worker는 policy step 경계에서 `collection_time_limit`으로 정상 저장 후 종료했다.

- 본 수집: `16,804` transition
- 응답 파일럿 포함 최종 학습 입력: `17,104` transition, 12 files
- observation/action dimension: `117 / 47`
- validity pass: `100%`
- termination: natural `225`, collection time limit `8`
- stressor episode: none `104`, skew `65`, incident `64`
- demand range/mean: `1.536 / 1.979 / 2.410`
- phase transition: peak `5,488`, recovery `11,616`
- behavior mode: budget `2,405`, linear `2,400`, quadratic `2,401`, single-block `2,481`, correlated-block `2,473`, full `2,469`, epsilon-mixture `2,475`

최종 response audit에서 모든 9개 owner block의 `dead_action_fraction=0`, `dead_response_fraction=0`이었다. D/F offset은 각각 8개 이산값 `0~105 s`를 사용했다. 네 ramp VSL은 각각 7개 이산값 `50~115 km/h`를 사용했다. 따라서 이전 dataset의 D/F offset 및 priced VSL dead-response 문제는 재현되지 않는다.

중간/최종 감사 산출물:

- `results/full_action_v3_mid6h_audit.json`
- `results/full_action_v3_final_audit.json`

### 13.3 3-seed support-aware IQL 재학습

새 response contract의 17,104 transition만 사용해 seed 0/1/2를 각각 40,000 update 학습했다. 비교 조건은 `gamma=1.0`, `support_weight=0.3`, `channel_dropout=0.1`이다.

- 마지막 학습 batch support 이탈률: seed 0 `3.3%`, seed 1 `3.6%`, seed 2 `2.4%`
- full rollout support 이탈률: skew15 평균 `1.75%`, incident 평균 `4.00%`
- 이전 1,745-transition 정책: skew15 `11.74%`, incident `10.64%`

데이터 coverage 확장이 IQL의 support 이탈과 seed 불일치를 크게 줄였다. 다만 현재 방식은 soft support penalty이므로 hard support-bounded actor로 해석하지 않는다.

### 13.4 14,400초 RL-FULL 재평가

각 checkpoint를 `sweet_190_skew15_w60`, `sweet_190_incident_w60`에서 75 policy step으로 실행했다. 6개 run, 450개 step 모두 validity gate를 통과했다. 개선율은 기준선보다 TTT가 낮을 때 양수다.

| Scenario | RL-FULL TTT, mean ± std (veh-h) | NC | PFO | P-Stack | P-CENT | vs PFO | vs P-Stack |
|---|---:|---:|---:|---:|---:|---:|---:|
| skew15 | `6,051.381 ± 5.414` | `6,881.942` | `6,299.302` | `6,378.869` | `5,757.359` | `+3.936%` | `+5.135%` |
| incident | `8,601.289 ± 16.209` | `8,555.873` | `9,229.987` | `8,386.237` | `8,016.293` | `+6.811%` | `-2.564%` |

개별 seed TTT:

- skew15: `6,047.814 / 6,059.031 / 6,047.297`
- incident: `8,591.588 / 8,624.127 / 8,588.153`

이전 dead-response 정책과 비교하면 skew15는 `56.510 veh-h`, `0.943%` 악화됐고 incident는 `138.713 veh-h`, `1.587%` 개선됐다. seed 표준편차는 skew15 `56.295 -> 5.414`, incident `270.264 -> 16.209 veh-h`로 크게 감소했다. 즉 평균 성능이 두 scenario에서 모두 좋아진 것은 아니지만 정책 안정성과 incident 일반화는 분명히 개선됐다.

실제 follower response 진단:

- 모든 run에서 D/F offset 후보 상태는 `active`였고 여러 offset 값이 선택됐다.
- 모든 가격 대상 VSL segment의 candidate variation fraction은 `100%`였다.
- VSL price-cost spread 평균은 run별 `0.26~0.50`, trust fallback은 전부 `0%`였다.
- adapter의 linear/quadratic coefficient 전달 최대 절대오차는 전부 `0`이었다.
- 평균 throughput은 skew15 `35,307.096`, incident `35,312.547 veh`였다.

판정:

- skew15에서는 RL leader가 PFO와 P-Stack을 모두 안정적으로 앞선다.
- incident에서는 PFO보다 `6.811%` 낮지만 P-Stack보다 `2.564%` 높아 최종 우월성을 주장할 수 없다.
- offset/VSL 경로는 이제 dead lever가 아니므로 이후 성능 차이는 follower activation 오류가 아니라 학습 objective, incident recovery coverage, action attribution 문제로 다뤄야 한다.

### 13.5 다음 실행 계획

1. 현재 3개 checkpoint를 고정하고 RL-BUDGET, RL-URBAN, RL-FREEWAY, RL-LINEAR 및 block mask ablation을 같은 두 full-horizon scenario에서 실행해 incident 열세의 owner block을 찾는다.
2. incident의 위치, lane-loss 크기와 복구 시점을 분리한 validation manifest를 만들고 recovery TTT 및 throughput을 paired metric으로 비교한다.
3. 추가 수집은 전체 random 12시간 반복보다 incident recovery에서 성능을 악화시키는 block을 겨냥한 ensemble-disagreement/epsilon-mixture 수집으로 제한한다.
4. soft support penalty와 dataset support projection 또는 support-bounded actor를 matched data/seed 조건에서 비교한다.
5. validation에서 반복적으로 개선되는 구성을 freeze한 뒤, 아직 사용하지 않은 demand/stressor manifest에서 PFO/P-Stack/P-CENT와 최종 paired test를 수행한다.

## 14. 2026-08-21 5개 수요·교란 조건 비교

12시간 재수집 데이터로 학습한 `rl_offset_vsl_response_v1` checkpoint 3개를 다음 5개 조건에서 각각 14,400초 실행했다: `sweet_155_w60`, `sweet_170_w60`, `sweet_170_incident_w60`, `sweet_170_skew15_w60`, `sweet_190_w60`. RL은 각 run에서 warm-up 이후 75 policy step을 모두 완료했고, 총 15개 run이 validity gate를 통과했다. 비교 기준선은 현재 코드와 동일한 scenario로 NC, PFO, P-Stack을 새로 실행했으며, 모든 TTT는 RL과 동일하게 최초 900초 warm-up을 제외했다.

개선율은 기준선보다 RL TTT가 낮을 때 양수다.

| Scenario | RL-FULL TTT, mean ± std (veh-h) | NC | PFO | P-Stack | vs NC | vs PFO | vs P-Stack | Support out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 155 | `3,117.754 ± 8.797` | `3,060.664` | `2,987.765` | `2,989.833` | `-1.865%` | `-4.351%` | `-4.279%` | `1.012%` |
| 170 | `3,955.535 ± 17.918` | `4,557.564` | `3,864.465` | `3,809.607` | `+13.209%` | `-2.357%` | `-3.831%` | `1.664%` |
| 170 incident | `5,659.840 ± 103.935` | `5,977.231` | `5,952.928` | `5,546.278` | `+5.310%` | `+4.923%` | `-2.048%` | `4.794%` |
| 170 skew15 | `4,013.774 ± 21.709` | `4,582.271` | `3,938.085` | `3,990.153` | `+12.406%` | `-1.922%` | `-0.592%` | `2.695%` |
| 190 | `5,976.502 ± 41.220` | `6,897.213` | `6,156.688` | `5,734.776` | `+13.349%` | `+2.927%` | `-4.215%` | `3.017%` |

Seed별로 보면 RL은 PFO 대비 170 incident와 190에서 각각 `3/3` 승리했다. P-Stack 대비로는 170 incident와 170 skew15에서 각각 한 seed만 근소하게 이겼고, 5개 조건 모두 평균 승리는 없었다.

Peak는 simulation time `900~5,220초`, recovery는 `5,220~14,400초`로 분리했다.

| Scenario | RL peak | P-Stack peak | RL vs P-Stack | RL recovery | P-Stack recovery | RL vs P-Stack |
|---|---:|---:|---:|---:|---:|---:|
| 155 | `1,899.9` | `1,814.5` | `-4.70%` | `1,217.9` | `1,175.3` | `-3.62%` |
| 170 | `2,457.0` | `2,400.2` | `-2.37%` | `1,498.6` | `1,409.4` | `-6.32%` |
| 170 incident | `3,122.1` | `3,271.9` | `+4.58%` | `2,537.7` | `2,274.4` | `-11.58%` |
| 170 skew15 | `2,501.9` | `2,474.2` | `-1.12%` | `1,511.9` | `1,516.0` | `+0.27%` |
| 190 | `3,464.2` | `3,416.1` | `-1.41%` | `2,512.3` | `2,318.7` | `-8.35%` |

판정:

1. 현재 RL은 NC보다 혼잡 조건 4개에서 `5.31~13.35%` 낮지만, 저수요 155에서는 불필요한 개입으로 NC보다도 `1.87%` 높다.
2. PFO 대비 이득은 170 incident와 190에서 seed 전체에 걸쳐 재현됐다. 반면 P-Stack 대비 평균 이득은 이번 5개 조건에서 재현되지 않았다.
3. 170 incident는 peak에서 P-Stack보다 좋지만 recovery에서 `11.58%` 뒤처져 총 TTT가 역전된다. 가장 큰 seed 편차와 support 이탈률도 이 조건에서 나타나므로 incident recovery의 상태-action coverage가 최우선 보강 대상이다.
4. 170 skew15는 P-Stack과 총 TTT 차이가 `0.59%`로 가장 가깝고 recovery는 근소하게 앞선다. 반대로 190은 PFO를 이기지만 P-Stack의 recovery 관리에는 크게 못 미친다.
5. response contract 교정으로 offset/VSL dead-response와 support 이탈 문제는 크게 줄었지만, 이것만으로 P-Stack의 상태별 제어 품질을 일관되게 대체하지는 못했다.

다음 실행 계획:

1. 현재 checkpoint와 5개 결과를 고정하고, 155·170 incident·190에서 RL-BUDGET/RL-URBAN/RL-FREEWAY 및 owner-block mask를 실행해 저수요 손실과 recovery 손실의 책임 채널을 찾는다.
2. 155에는 no-control/PFO anchor 주변, incident와 190에는 recovery 구간을 우선한 ensemble-disagreement 및 epsilon-mixture transition을 추가 수집한다. 전체 12시간 무차별 반복 수집은 하지 않는다.
3. critic target과 advantage를 peak/recovery 및 baseline-relative return으로 함께 감사해, 장기 recovery 손실이 reward/return 구성에서 과소평가되는지 확인한다.
4. 기존 soft support penalty와 dataset support projection을 동일 데이터·seed로 비교하고, 저수요 상태에서 no-control에 가까운 action을 낼 수 있는지 별도 검증한다.
5. 수정 정책은 이 5개 조건과 기존 190 skew/incident validation을 모두 통과해야 하며, P-Stack 대비 평균뿐 아니라 seed별 승률, recovery TTT, throughput, validity를 함께 freeze 기준으로 사용한다.

집계 산출물은 `results/five_cell_response_v1/comparison.csv`, `comparison.json`, `phase_comparison.csv`에 저장했다.

## 15. 2026-08-21~22 targeted 24시간 수집, 재학습 및 5개 조건 재평가

### 15.1 수집 계약과 데이터 감사

24시간 orchestration window 안에서 8개 worker를 실행했다. worker의 실제 수집 구간은 약 23시간 50분이며, 마지막 transition 경계에서 정상 종료해 강제 종료된 process는 없다.

이번 수집은 전체 상태공간의 무차별 random 확장이 아니라 이전 5개 평가 조건과 recovery 실패를 직접 겨냥했다.

1. target 비중은 `155 20%`, `170 10%`, `170 incident 35%`, `170 skew15 10%`, `190 25%`이며 demand에는 `+-2%` jitter, incident/skew에는 위치와 강도 jitter를 적용했다.
2. `optimizer_local`은 현재 P-Stack b13의 leader coordination action을 같은 상태에서 계산한 뒤, plant를 한 번만 전진시키는 teacher-neighborhood mode다. PFO supervisor는 포함하지 않는다.
3. `loose_anchor`는 최대 budget과 0 price로 표현한 저개입 action이다. 현재 coordination action이 exact NC를 직접 표현하지 못하므로 NC와 동일한 anchor로 해석하지 않는다.
4. `optimizer_local`과 `loose_local`은 simulation time `5,220초`부터 AR(1), `rho=0.95`의 시간 상관 perturbation을 적용해 recovery 주변의 action-response를 넓혔다.

본 수집은 8개 파일, `24,308` transition이며 기존 response-fixed 데이터 `17,104`개와 합친 최종 학습 입력은 20개 파일, `41,412` transition이다.

- validity pass: `100%`
- termination: natural `321`, collection time limit `8`
- target episode: 155 `68`, 170 `38`, 170 incident `103`, 170 skew15 `39`, 190 `81`
- behavior mode: optimizer-local `12,132`, loose-anchor `6,101`, loose-local `6,075`
- phase transition: peak `7,861`, recovery `16,447`
- perturbation active: `12,314`, targeted transition의 `50.66%`
- active perturbation mean normalized L2 distance: `0.2776`
- 9개 owner block 모두 `dead_action_fraction=0`, `dead_response_fraction=0`

targeted 데이터만 보면 ramp VSL response는 `100/115 km/h` 두 값에 집중된다. 이것은 optimizer/loose anchor 주변을 좁게 수집한 설계의 결과다. 기존 broad 데이터와 결합한 최종 dataset에서는 ramp VSL `50~115 km/h`의 7개 값이 유지되므로 dead response는 아니지만, targeted 수집이 VSL amplitude coverage 자체를 넓힌 것은 아니다.

감사 산출물:

- `results/targeted_24h_v1_final_audit.json`
- `results/five_cell_targeted_v1/targeted_dataset_audit.json`
- `results/five_cell_targeted_v1/combined_dataset_audit.json`

### 15.2 3-seed IQL 재학습

결합 dataset으로 seed 0/1/2를 각각 `80,000` update 학습했다. 세 checkpoint의 공통 설정은 `gamma=1.0`, `expectile=0.7`, `beta=3.0`, `reward_scale=0.01`, `support_weight=0.3`, `channel_dropout=0.1`이다.

체크포인트 내부 metadata에서 세 모델 모두 다음 조건을 확인했다.

- training step: `80,000`
- transition count: `41,412`
- observation/action dimension: `117 / 47`
- response contract: `rl_offset_vsl_response_v1`
- checkpoint format: `rl_coordination_checkpoint_v2`

체크포인트:

- `checkpoints/actor_full_iql_targeted24h_s0.pt`
- `checkpoints/actor_full_iql_targeted24h_s1.pt`
- `checkpoints/actor_full_iql_targeted24h_s2.pt`

### 15.3 5개 조건 full-horizon 결과

각 checkpoint를 같은 5개 조건에서 warm-up 뒤 75 policy step, simulation time `14,400초`까지 실행했다. 15개 run이 모두 완주했고 wall-clock truncation 없이 validity gate를 통과했다. NC/PFO/P-Stack은 현재 코드로 생성해 잠가 둔 `results/five_cell_baselines`를 동일한 warm-up 제외 TTT 정의로 재사용했다.

개선율은 기준선보다 RL TTT가 낮을 때 양수다.

| Scenario | Targeted RL TTT, mean +- std (veh-h) | NC | PFO | P-Stack | vs NC | vs PFO | vs P-Stack | Support out |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 155 | `3,068.623 +- 50.165` | `3,060.664` | `2,987.765` | `2,989.833` | `-0.260%` | `-2.706%` | `-2.635%` | `1.787%` |
| 170 | `3,926.046 +- 23.880` | `4,557.564` | `3,864.465` | `3,809.607` | `+13.856%` | `-1.594%` | `-3.056%` | `1.863%` |
| 170 incident | `5,610.356 +- 134.409` | `5,977.231` | `5,952.928` | `5,546.278` | `+6.138%` | `+5.755%` | `-1.155%` | `4.161%` |
| 170 skew15 | `3,989.758 +- 40.138` | `4,582.271` | `3,938.085` | `3,990.153` | `+12.931%` | `-1.312%` | `+0.010%` | `2.005%` |
| 190 | `5,983.253 +- 67.053` | `6,897.213` | `6,156.688` | `5,734.776` | `+13.251%` | `+2.817%` | `-4.333%` | `1.541%` |

이전 17,104-transition 정책 대비 TTT 변화:

| Scenario | Previous RL | Targeted RL | Delta | 판정 |
|---|---:|---:|---:|---|
| 155 | `3,117.754` | `3,068.623` | `-49.130` | 개선 |
| 170 | `3,955.535` | `3,926.046` | `-29.489` | 개선 |
| 170 incident | `5,659.840` | `5,610.356` | `-49.484` | 개선 |
| 170 skew15 | `4,013.774` | `3,989.758` | `-24.016` | 개선 |
| 190 | `5,976.502` | `5,983.253` | `+6.750` | 악화 |

P-Stack과의 phase 비교:

| Scenario | RL peak vs P-Stack | RL recovery vs P-Stack |
|---|---:|---:|
| 155 | `-3.126%` | `-1.877%` |
| 170 | `-1.814%` | `-5.173%` |
| 170 incident | `+4.290%` | `-8.989%` |
| 170 skew15 | `-0.770%` | `+1.283%` |
| 190 | `-1.697%` | `-8.216%` |

seed별 P-Stack 승리는 155 `0/3`, 170 `0/3`, 170 incident `1/3`, 170 skew15 `2/3`, 190 `0/3`이다. `170 skew15`의 평균 우위 `0.010%`는 수치상 승리지만 실질적으로 동률로 취급한다.

### 15.4 판정: 데이터 부족 가설은 일부만 지지된다

1. 데이터를 `17,104 -> 41,412` transition으로 늘리고 실패 상태에 집중하자 5개 중 4개 조건이 `24.0~49.5 veh-h` 개선됐다. 따라서 이전 열세에 데이터 부족과 상태-action coverage 부족이 일부 기여했다는 가설은 지지된다.
2. 그러나 P-Stack 대비 평균은 skew15의 사실상 동률을 제외한 4개 조건에서 여전히 열세다. 190은 데이터 증량 뒤 오히려 `6.75 veh-h` 악화됐다. 데이터 총량만을 주원인으로 보기는 어렵다.
3. incident는 peak에서 P-Stack을 `4.29%` 앞서지만 recovery에서 `8.99%` 뒤처지고, 190도 recovery에서 `8.22%` 뒤처진다. 가장 큰 잔여 병목은 recovery return attribution과 장기 action 품질이다.
4. incident의 seed별 TTT는 `5,484.794 / 5,549.547 / 5,796.727`로 seed 2가 평균을 크게 악화시켰다. support 이탈도 incident가 평균 `4.16%`로 가장 높다. offline policy의 seed 안정성과 support 제어가 아직 freeze 기준에 못 미친다.
5. 이번 다섯 조건은 정책 개발 과정에서 반복 사용한 validation set이다. 결과는 방향 결정에는 사용할 수 있지만 최종 generalization/test 성능으로 주장하지 않는다.

집계 산출물:

- `results/five_cell_targeted_v1/rl_s0.csv`, `rl_s1.csv`, `rl_s2.csv`
- `results/five_cell_targeted_v1/comparison.csv`, `comparison.json`
- `results/five_cell_targeted_v1/phase_comparison.csv`

### 15.5 다음 실행 계획

다음 단계에서는 또 한 번의 무차별 장시간 수집보다 현재 데이터에서 왜 recovery와 seed 2가 실패했는지를 먼저 분리한다.

1. **Checkpoint freeze와 attribution:** 현재 3개 checkpoint와 dataset manifest를 고정한다. 170 incident와 190에서 owner block mask 및 RL-BUDGET/RL-URBAN/RL-FREEWAY counterfactual을 실행해 recovery 손실을 만드는 채널을 찾는다.
2. **Return audit:** peak/recovery별 reward, return, critic value, advantage를 P-Stack teacher-neighborhood와 loose-anchor neighborhood로 나눠 비교한다. recovery의 장기 TTT가 현재 return과 advantage ranking에서 실제로 우수 action으로 표시되는지 확인한다.
3. **정책 표현 개선:** live P-Stack optimizer를 deployment에 남기지 않도록, 수집된 P-Stack/loose anchor를 모사하는 frozen behavior prior를 먼저 학습하고 IQL은 prior 대비 residual을 출력하는 matched ablation을 만든다. 현재 absolute-action IQL과 같은 데이터와 seed로 비교한다.
4. **Support와 seed 안정성:** soft penalty 단독 조건과 action projection 또는 behavior-prior trust region을 비교한다. 3-seed ensemble 평균도 diagnostic으로 평가하되, 같은 5개 validation 결과로 best seed를 고르는 방식은 사용하지 않는다.
5. **선별 재수집:** 1~4에서 확인된 실패 channel에 대해서만 incident/190 recovery의 ensemble-disagreement transition을 추가한다. VSL amplitude가 필요하면 현재 targeted collector의 `100/115 km/h` 집중도 함께 해소한다.
6. **Freeze gate:** validation에서 P-Stack 대비 scenario 열세가 `1%`를 넘지 않고, incident/190 recovery 열세가 `2%` 이내이며, seed별 승률·throughput·validity·support gate를 함께 통과한 구성만 잠근다.
7. **최종 test:** freeze 뒤 아직 학습과 선택에 사용하지 않은 demand, incident 위치/강도, skew 조합에서 NC/PFO/P-Stack/P-CENT와 paired full-horizon test를 한 번 수행한다.

## 16. P-Stack 열세 원인 분리 및 재구축 실행 계획

이번 단계의 목적은 follower 구조를 먼저 교체하는 것이 아니다. P-Stack과 RL이 공유하는 F1 Wu faithful follower의 앞단에서 observation, action 표현, 가격 adapter, teacher data, offline 학습 중 어느 경계가 성능 손실을 만드는지 동일 상태 paired 실험으로 분리한다. 원인이 확인되기 전에는 추가 장시간 학습을 시작하지 않는다.

### 16.1 고정 기준

1. 현재 `41,412` transition dataset, targeted24h checkpoint 3개, 5개 baseline과 RL 결과의 SHA-256 및 코드 계약을 `results/pstack_gap_diagnosis_v1/manifest.json`에 고정한다.
2. 비교 metric은 기존과 동일하게 warm-up `900초`를 제외한 `14,400초` TTT이며, peak는 `900~5,220초`, recovery는 `5,220~14,400초`다.
3. 개발용 5개 조건과 최종 holdout을 분리한다. 개발 중에는 155, 170, 170 incident, 170 skew15, 190만 사용하고 최종 일반화 조건은 policy freeze 뒤 한 번만 연다.

### 16.2 실행 순서와 판정 gate

1. **동일 상태 follower parity:** native P-Stack이 선택한 budget/price를 공통 action schema로 encode한 뒤, 같은 state, forecast, previous control에서 RL adapter follower에 다시 입력한다. budget, green, offset, metering, 모든 segment VSL과 예측 TTT 차이를 기록한다.
2. **표현 가능성 분리:** native P-Stack의 non-merge VSL 가격, PFO supervisor 선택, D/F offset 후보, mask별 제어 기여를 각각 분리한다. encode-decode만으로 사라지는 가격과 follower 재최적화에서 달라지는 제어를 별도 지표로 둔다.
3. **계약 수정:** parity에서 확인된 항목만 수정한다. 우선 검증 대상은 이전 green/offset/VSL 누락에 따른 비-Markov observation, segment별 density/speed/lane-loss 누락, non-merge VSL 가격 action 누락이다. 각 항목은 수정 전 실패하는 regression test를 먼저 둔다.
4. **clean teacher 재수집:** perturbation 없는 P-Stack intent anchor를 실제 RL adapter로 실행해 모든 target과 peak/recovery에 포함한다. 그 주변에서만 bounded AR perturbation과 ensemble-disagreement 탐색을 섞고, behavior mode별 episode 수와 실제 action-response coverage를 감사한다.
5. **matched 3-seed 재학습:** 동일 dataset과 update 수로 absolute IQL, behavior-prior residual IQL, support-bounded variant를 비교한다. scalar advantage가 고차원 전체 action을 평균내는 문제는 owner-block advantage 또는 residual policy ablation으로 검증한다.
6. **5개 조건 full run:** 모든 후보는 같은 3 seed와 75 policy step으로 평가한다. 평균 TTT뿐 아니라 recovery TTT, seed 승률, throughput, validity, state-conditioned joint support를 함께 기록한다.
7. **freeze 및 holdout:** P-Stack 대비 각 개발 scenario 열세 `1%` 이내, incident/190 recovery 열세 `2%` 이내, validity `100%`를 동시에 만족한 구성만 freeze한다. 그 뒤 미사용 demand, incident 위치/강도, skew 조합에서 최종 paired test를 실행한다.

### 16.3 중단 조건

- 동일 상태에서 native P-Stack action을 adapter에 replay했는데도 제어와 예측 TTT가 크게 다르면 학습보다 action/follower 계약을 먼저 고친다.
- parity는 맞지만 RL actor만 열세면 observation coverage, return attribution, support와 policy 표현 문제로 범위를 좁힌다.
- 계약 수정 뒤 clean exact-anchor policy가 P-Stack을 재현하지 못하면 follower 후보공간 또는 supervisor 차이를 별도 controller ablation으로 검증한다.
- 데이터 수만 늘려 개선되는지 확인하려는 장시간 수집은 위 세 경계를 통과하기 전에는 반복하지 않는다.

### 16.4 2026-08-22 parity 진단 결과와 계약 v4

동일 상태 replay로 follower 구조 자체보다 leader 교체 경계의 누락을 먼저 확인했다.

1. 기존 observation은 이전 green/offset/VSL reference, segment별 density/speed/effective lane, segment별 incident 위치를 구분하지 못했다.
2. 기존 action은 전체 16개 VSL segment 중 merge 인접 4개만 가격을 표현했다.
3. RL follower와 native P-Stack follower의 b13 candidate box, segment-agent, primal-dual 설정이 달랐다.
4. P-Stack metering price에는 ramp별 `metering_release_certified`가 동반되지만 RL action은 이를 전달하지 않았다. baseline 5개 조건에서 false certificate도 반복 관측되어 상수로 대체할 수 없다.
5. follower의 `_prev_coupling`과 primal-dual corrector 메모리가 다음 response에 영향을 주지만 observation에 없었다.
6. 가장 중요한 데이터 오류로, 기존 `optimizer_anchor`는 native P-Stack control을 plant에 직접 적용해 action과 transition이 deployable RL 경로에서 불일치했다. `optimizer_local`은 RL adapter를 통과했으므로 이 특정 우회 오류는 없었다.

수정된 계약은 다음과 같다.

- action: `coordination_action_v4`, `75차원`
- observation: `coordination_observation_v4`, `238차원`
- response: `rl_pstack_b13_full_segment_vsl_certificate_v3`
- VSL 가격 coverage: `16/16 segment`
- ramp release certificate: `4개 binary action`
- optimizer anchor transition: `rl_adapter_replay_v1`
- follower observation: warm-start coupling 18개와 primal-dual memory presence/value 포함

coordination/parity 계약 test `21/21`, dataset/policy gate와 기존 follower 관련 test를 합친 관련 suite `33/33`이 통과했다. 94-transition 다중 mode pilot은 validity `100%`, dead action dimension `0`, certificate true fraction `21.3~24.5%`였다. 짧은 pilot에는 recovery와 skew가 없어 dataset gate가 의도대로 실패했다. 2-worker orchestration smoke에서는 두 worker 모두 adapter-consistent dataset을 정상 저장하고 강제 종료 없이 끝났다.

### 16.5 자동 실행 pipeline

1. `work/run_contract_v4_24h.ps1`이 8 worker를 24시간 관리한다. target 5개 조건에서 P-Stack replay anchor, optimizer-local perturbation, loose anchor, structured epsilon exploration을 섞고 recovery 시작 `5,220초`부터 AR perturbation을 활성화한다.
2. `rl_leader/audit_full_action.py`가 schema, adapter transition contract, 75개 action support, phase/target coverage와 certificate 양쪽 결과를 감사한다.
3. `rl_leader/validate_contract_v4_dataset.py` gate가 validity `99.9%` 미만, 10,000 transition 미만, target/recovery 누락, dead action dimension, certificate 한쪽 쏠림이면 학습을 중단한다.
4. gate 통과 후 `work/run_contract_v4_postprocess.ps1`이 clean v4 데이터만으로 IQL seed 0/1/2를 각 80,000 update 학습한다. 구계약 41,412 transition은 schema가 달라 혼합하지 않는다.
5. 세 policy의 support/disagreement audit 뒤 155, 170, 170 incident, 170 skew15, 190을 각각 75 policy step으로 평가하고 고정 P-Stack baseline과 비교한다.
6. 표준 IQL이 freeze gate를 통과하지 못하면 같은 clean dataset에서 behavior-prior residual과 block-wise critic ablation을 수행한다. 데이터 양과 모델 구조 원인을 같은 실험에서 동시에 바꾸지 않는다.

후처리 smoke에서 기존 IQL normalization이 `obs`에만 맞춰진 것도 확인했다. episode 마지막 action으로 처음 변한 reference가 `next_obs`에만 나타난 pilot에서 표준화 절대값이 `130,434.8`, seed 99 첫 critic loss가 `10,621.8`까지 폭발했다. normalization support를 `obs + next_obs`로 합치고 표준편차 하한을 `1e-3`으로 둔 뒤 같은 seed의 첫 critic loss는 `1.268`로 정상화됐다. checkpoint에는 `observation_normalization=obs_and_next_obs_std_floor_1e-3`을 기록한다. 수정 후 2-seed 5-update 학습, ensemble audit, v4 checkpoint 1-step controller load/eval이 모두 통과했다.

첫 장시간 launch는 파일 저장 전 pilot 로그에서 release certificate false가 optimizer action에서는 `-1`, budget/loose action에서는 `0`으로 이중 표현되는 것을 발견해 중단했다. follower decode에서는 둘 다 false지만 MSE actor에는 서로 다른 target이 되어 0 근처 출력의 부호가 불안정해진다. 모든 non-release action을 `-1`, release action을 `+1`로 canonicalize하고 one-step loose dataset에서 `[-1,-1,-1,-1]`과 validity를 확인한 뒤 `data/contract_v4_24h_v2`로 다시 시작했다. 중단된 launch는 worker dataset을 아직 만들지 않았으므로 최종 학습 입력에 포함되지 않는다.

장시간 수집 초반 16개 random episode에서 skew가 우연히 빠져 있었으므로, 최종 target coverage를 운에 맡기지 않고 첫 target이 `sweet_170_skew15_w60`인 seed 974의 adapter-consistent `loose_local` 75-step episode를 추가했다. 이후 `1,350` transition 중간 감사에서 5개 target, peak/recovery `432/918`, dead action dimension `0`, validity `100%`, certificate 양쪽 support를 확인했고 `minimum_transitions=1,000` 중간 dataset gate가 실패 항목 없이 통과했다. 본 수집은 계속 진행해 표본량과 target별 균형을 늘린다.

### 16.6 2026-08-23 종료 checkpoint와 재개 기준

`data/contract_v4_24h_v2` 수집은 2026-08-22 16:13에 시작했고 worker 로그는 2026-08-23 09:30경 중단되어 약 17시간 17분 실행됐다. 완료된 episode만 원자적으로 저장되므로 진행 중이던 8개 episode는 dataset에 포함되지 않았다. 후처리 watcher와 별도 완료 대기 shell은 10:22에 명시적으로 종료했고, 수집·학습·평가 관련 프로세스가 남지 않았음을 확인했다. IQL 재학습과 5개 조건 평가는 아직 시작하지 않았다.

종료 직후 `results/contract_v4_24h_v2/interrupted_collection_audit.json`과 `interrupted_collection_gate.json`을 생성했다.

- transition: `25,125`, 자연 종료 episode: `335`, validity: `100%`
- action/observation dimension: `75/238`, dead action dimension: `0`
- target episode: 155 `60`, 170 `33`, incident `139`, skew `25`, 190 `78`
- phase transition: peak `8,040`, recovery `17,085`
- behavior mode: optimizer anchor `2,550`, optimizer local `6,225`, loose anchor `4,200`, loose local `5,100`, epsilon mixture `2,925`, linear `2,100`, quadratic `2,025`
- release certificate true fraction: ramp별 약 `29.3~29.8%`
- `minimum_transitions=10,000` 최종 dataset gate: **통과**

이 표본은 계약 및 coverage gate를 이미 통과했으므로 우선 그대로 3-seed matched IQL 학습에 사용한다. 정확히 24시간의 wall-clock 수집이 연구 설계상 필수일 때만 기존 디렉터리를 덮어쓰지 않고 별도 top-up dataset을 수집한다. 구계약 데이터, 중단된 `contract_v4_24h_v1`, 현재 v4 데이터는 혼합하지 않는다.

종료 audit에서 출력 디렉터리는 `v2`인데 NPZ 내부 dataset label이 `contract_v4_24h_v1`인 orchestration metadata 오류도 확인했다. schema와 response contract 검증에는 영향이 없지만, 이후 실행은 `work/run_contract_v4_24h.ps1`이 출력 디렉터리 basename을 dataset name으로 기록하도록 수정했다.

재개 순서와 명령은 `RL_NEXT_STEPS.md`를 단일 실행 checklist로 사용한다.
