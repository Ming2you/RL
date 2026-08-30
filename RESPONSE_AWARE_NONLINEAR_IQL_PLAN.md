# Response-Aware Nonlinear IQL 구현 기준서

**상태 기준일:** 2026-08-30
**목적:** 지금까지 구축한 실험 기반을 보존하면서, 논문 초안에서 제안한 비선형 coordination leader를 실제 코드와 실험으로 연결한다.

## 1. 한 줄 결론

현재의 `238-D state -> dense 75-D IQL action` 경로를 그대로 확대하지 않는다. 다음 구현은 **P-Stack을 기준(anchor)으로 구조화된 비선형 coordination 후보를 만들고, 실제 follower가 서로 다른 물리 제어를 실행한 후보만 남긴 뒤, 장기 paired TTT와 불확실성으로 선택하는 response-aware conservative policy improvement**로 정의한다.

첫 배포 단위는 자유롭게 연속 action을 생성하는 actor가 아니라 **유한 후보 집합을 평가하는 anchored option IQL critic**이다. 이 구조가 P-Stack 대비 개선 가능성을 입증한 뒤에만 surrogate top-K와 P-Stack distillation을 통해 진짜 leader replacement로 확장한다.

## 2. 연구 주장과 범위

### 2.1 지금 검증할 주장

1. 선형 marginal price만 사용한 기존 P-Stack보다 quadratic/cross-control potential을 포함한 coordination 후보가 더 좋은 follower response를 만들 수 있다.
2. nominal price vector가 아니라 실제 follower response를 기준으로 support를 정의하면 many-to-one action aliasing을 줄일 수 있다.
3. 동일 상태에서 계산한 paired long-horizon TTT와 conservative uncertainty gate를 사용하면 P-Stack보다 나쁜 개입을 억제할 수 있다.

### 2.2 아직 주장하지 않을 것

1. 매 step P-Stack anchor를 계산하는 oracle selector는 아직 P-Stack leader의 계산 대체가 아니다.
2. simulator에서 반복적으로 batch를 추가하는 전체 과정은 순수한 fixed-dataset offline RL이 아니다. **각 frozen batch 안에서의 학습은 offline**이고, 전체 방법은 `iterative simulator-assisted offline policy improvement`로 부른다.
3. strict anchor 대비 one-shot 개선은 repeated sequential policy의 개선을 자동으로 보장하지 않는다.

논문에서는 성능 주장과 계산 대체 주장을 분리한다.

- **Claim A:** response-aware nonlinear candidate selection이 P-Stack보다 장기 TTT를 개선한다.
- **Claim B:** follower-response surrogate와 distilled anchor가 P-Stack leader solve를 충분히 대체한다.

Claim B는 Claim A가 통과한 뒤의 별도 단계다.

## 3. 지금까지 완료한 것

### 3.1 실행 및 데이터 계약

- 5개 핵심 시나리오(`155`, `170`, `170 incident`, `170 skew`, `190`)와 PFO/P-Stack/P-CENT 비교 경로를 구축했다.
- action/observation schema, anchor envelope, branch, fingerprint를 artifact와 checkpoint에 기록하고 불일치 시 fail-fast하도록 만들었다.
- P-Stack native anchor와 RL residual의 encode/decode, gate-applied action, exact replay 검증을 추가했다.
- candidate branch마다 plant state뿐 아니라 follower optimizer의 hidden memory까지 공통 snapshot에서 복원하는 격리 경로를 만들었다.
- candidate order가 결과를 바꾸는 오염과 action key/scale mapping 문제를 진단하는 도구를 추가했다.

### 3.2 실패 원인 진단

- 초기 dense 75-D IQL은 5개 시나리오에서 P-Stack을 안정적으로 이기지 못했다.
- 더 많은 trajectory와 epoch만으로 해결되지 않는 원인을 확인했다.
  - nominal action은 달라도 follower가 같은 physical control을 실행하는 response aliasing
  - H3에서 좋아 보인 후보가 H12 또는 drain-out에서 나빠지는 horizon mismatch
  - anchor를 계산하기 전 actor가 residual을 출력하면서 생기는 hidden context
  - follower memory와 branch 상태가 빠진 불완전한 state
  - zero/near-anchor sample이 많은 action support와 희소한 positive label
- 따라서 실패를 단순한 데이터 양 부족이나 IQL optimizer 문제로 해석하지 않는다.

### 3.3 장기 counterfactual 및 selector 기반

- frozen state에서 H1 follower response, H3, H12, zero-demand drain-out을 같은 anchor와 dependency hash로 연결하는 label 경로를 구축했다.
- realized physical control과 post-response follower memory가 같은 후보를 동일 outcome으로 묶는 pairwise dataset을 만들었다.
- strict tail gain이 material margin을 넘을 때만 통과하고 그 외에는 P-Stack으로 돌아가는 oracle selector를 만들었다.
- 현재 learned selector는 logistic classifier와 ridge gain/margin regressors의 bootstrap ensemble이다. 이는 **IQL이 아니라 안전 gate와 데이터 계약을 검증하는 supervised baseline**이다.
- completed strict label만 자동 발견하여 pairwise dataset, oracle selector, learned selector를 재생성하는 pipeline과 관련 테스트를 추가했다.

### 3.4 현재 label coverage

마지막 기록된 완료 corpus는 다음과 같다.

| 항목 | 값 |
|---|---:|
| completed drain artifacts | 9 |
| pairwise rows | 16 |
| event groups | 7 |
| positive rows | 5 |
| train/fold false positives | 0 |
| production minimum | event groups 50, positives 20 |

따라서 selector가 fail-closed로 동작한다는 smoke는 통과했지만, deployable model을 주장할 coverage는 아직 없다. Dense frozen-state v3는 시나리오당 20개, 총 100개 event를 목표로 준비되었고 마지막 기록에서는 `155`와 `170`의 40개 event 중 17개가 coordination-eligible이었다. 나머지 시나리오의 freeze와 downstream label 생성은 미완료 상태로 본다.

## 4. 현재 구현과 새 방법의 차이

| 구분 | 현재 코드 | 목표 코드 |
|---|---|---|
| 후보 공간 | owner-block `30-D linear prices`; budget/quadratic/certificate 고정 | linear + quadratic + cross-control + optional budget residual |
| 선택 모델 | logistic/ridge ensemble | anchored option IQL critic + pairwise auxiliary loss + ensemble LCB |
| action 단위 | raw residual과 realized-response pair가 혼재 | response-equivalence class가 기본 단위 |
| 시간 계약 | one-shot + long-horizon label | one option intervention + `K-1` P-Stack cooldown인 semi-MDP |
| 배포 | oracle/smoke selector | oracle-response proof -> surrogate top-K -> distilled anchor |

기존 `rl_leader/iql.py`는 dense actor 실험 baseline으로 유지한다. 새 모델이 그 파일을 덮어쓰지 않도록 별도 format version과 checkpoint family를 사용한다.

## 5. 목표 제어 구조

```text
network state + forecast + follower memory
                  |
             P-Stack anchor
                  |
       structured nonlinear candidates
                  |
       actual follower MPC responses
                  |
  dedupe by physical control + memory outcome
                  |
 anchored option Q / long-horizon LCB ranking
             /                 \
     candidate passes       abstain/OOD
          |                     |
 execute one option          P-Stack
          |
   P-Stack cooldown for K-1 steps
```

성능 입증 단계에서는 모든 후보에 actual follower solve를 허용한다. 계산량 절감은 그 뒤에 response surrogate로 top-K만 actual solve하거나 anchor를 distill하는 방식으로 다룬다.

## 6. 수학적 및 데이터 계약

### 6.1 상태

`o_t`에는 최소한 다음이 포함되어야 한다.

- plant state와 전체 forecast
- 이전에 실제 적용된 green, offset, metering, VSL
- follower optimizer memory와 branch-relevant state
- P-Stack anchor action, anchor response, anchor physical plan
- scenario/event/phase는 split용 provenance로만 사용하고 정책 shortcut feature로 쓰지 않는다.

### 6.2 구조화된 비선형 coordination 후보

각 follower local objective에 주는 potential을 다음과 같이 표현한다.

```text
Phi_i(u_i; s) = p_i(s)^T z_i(u_i)
              + 0.5 z_i(u_i)^T H_i(s) z_i(u_i)
              + c_i(s)^T z_cross(u_i)
```

- `p_i`: 기존 linear marginal price
- `H_i`: quadratic curvature, 기본 구현은 `H_i = L_i L_i^T`로 PSD 보장
- `c_i`: control 간 cross term
- 모든 term은 control별 reference scale로 무차원화한다.
- anchor 주변 trust region과 coefficient clipping을 적용한다.

첫 candidate catalog는 완전한 Cartesian product가 아니라 sparse signed axis/corner로 제한한다.

- Urban: green linear, offset linear, green quadratic, offset quadratic, green-offset cross
- Freeway: metering linear, VSL linear, metering quadratic, VSL quadratic, metering-VSL cross
- 초기 magnitude: `0`, `+/-0.25`, `+/-0.5`, 필요한 2축 corner의 normalized magnitude `0.707`
- budget residual은 nonlinear headroom이 확인된 뒤 독립 ablation으로 추가한다.

### 6.3 response-equivalence

두 leader candidate `a`와 `a'`는 아래가 모두 같을 때 동일 action outcome으로 취급한다.

```text
E(a) = (
  implemented urban physical controls,
  implemented freeway physical controls,
  follower selected discrete candidates,
  post-response follower memory fingerprint
)
```

Raw coefficient 거리가 아니라 `E(a)`를 support와 deduplication의 기준으로 사용한다. 하나의 response class에는 가장 작은 coefficient norm을 가진 대표 candidate와 모든 alias provenance를 보존한다.

### 6.4 anchored option과 reward

한 학습 action은 한 policy interval의 nonlinear intervention이며, 이후 `K-1` step은 P-Stack으로 continuation한다.

```text
y_t = sum_{j=0}^{K-1} gamma^j r_{t+j}
target = y_t + gamma^K V(o_{t+K})
r_t = - incremental_TTT_t
```

종료 시 남은 queue/occupancy가 reward truncation을 일으키지 않도록 zero-demand drain-out 또는 terminal inventory penalty를 사용한다. H3는 physical sanity check, H12는 중간 screening, strict drain/full remainder는 최종 성능 label로 구분한다.

## 7. 구현 단계

### Phase 0. 현재 artifact 동결

1. 기존 dense IQL, tail selector, frozen-state v2/v3 artifact의 format version과 SHA를 보존한다.
2. running 또는 partial artifact는 학습 입력에서 제외하고 terminal 상태만 사용한다.
3. 기존 데이터와 새 nonlinear dataset을 같은 glob으로 읽지 못하게 dataset family를 분리한다.

**통과 조건:** 동일 manifest를 두 번 replay했을 때 anchor, response fingerprint, long-horizon TTT가 tolerance 안에서 동일하다.

### Phase 1. nonlinear candidate contract

새 모듈과 테스트를 추가한다.

- `rl_leader/nonlinear_candidate_catalog.py`
  - typed coefficient blocks와 normalization
  - PSD quadratic parameterization
  - sparse catalog와 deterministic candidate IDs
- `src/controllers/coordination.py`
  - schema version을 올리고 nonlinear terms를 follower objective에 전달
- urban/freeway follower adapter
  - linear, quadratic, cross term을 local candidate objective에 실제 반영
- `src/tests/test_nonlinear_coordination_action.py`
  - encode/decode roundtrip, zero residual parity, PSD, scale, clipping

**통과 조건:** 모든 nonlinear coefficient가 0이면 현재 P-Stack과 exact parity이고, 각 coefficient를 단독으로 바꿨을 때 의도한 local objective term만 변한다.

### Phase 2. nonlinear oracle headroom audit

Dense data를 모으기 전에 170/190의 frozen state 10~20개에서 sparse candidate를 전수 follower solve한다.

1. common state와 follower memory에서 모든 candidate를 실행한다.
2. response-equivalence로 deduplicate한다.
3. unique response만 H3/H12/drain으로 승격한다.
4. linear-only와 nonlinear catalog의 oracle best gain을 비교한다.

**중단 조건:** nonlinear 후보가 새로운 response class를 거의 만들지 않거나 strict tail oracle best가 linear-only/P-Stack보다 개선되지 않으면 RL 학습 전에 follower objective와 candidate field를 다시 설계한다.

### Phase 3. iterative simulator-assisted batch 구축

1. 5개 scenario에서 congestion growth, peak, early recovery, late recovery를 event-group 단위로 freeze한다.
2. scenario별 episode 전체를 무작정 늘리지 않고 coordination-eligible event를 우선 표집한다.
3. 후보 생성 -> actual response -> response dedup -> paired continuation label 순으로 처리한다.
4. event-group 단위로 train/calibration/test split하고 같은 frozen state의 candidate가 split을 넘지 않게 한다.
5. batch를 완전히 freeze한 뒤에만 offline training을 수행한다.

권장 1차 gate는 train 50 event groups, calibration 20, untouched test 20이며 positive event가 각 split에 존재해야 한다. 단순 row 수가 아니라 unique event와 unique response 수를 보고한다.

### Phase 4. anchored option dataset

`rl_leader/build_anchored_option_dataset.py`를 추가한다. 각 row에는 다음을 저장한다.

- observation과 follower-memory fingerprint
- anchor action/response/physical control
- candidate nonlinear coefficients와 catalog ID
- realized response class와 alias IDs
- H1/H3/H12/drain TTT decomposition
- `Delta J = J_anchor - J_candidate`
- option length `K`, continuation policy, termination inventory
- source artifact SHA와 implementation SHA

Invalid, dependency drift, replay mismatch row는 quarantine하고 학습에서 제외한다.

### Phase 5. response-aware anchored option IQL

`rl_leader/train_response_iql.py`를 새로 만든다.

- critic 입력: `(observation, anchor context, candidate coefficients, realized response embedding)`
- candidate catalog 안의 supported finite actions만 평가한다.
- dense unconstrained actor는 두지 않는다.
- expectile value와 advantage-weighted behavior objective는 option transition에 맞춰 `gamma^K`를 사용한다.
- event-group bootstrap ensemble을 학습한다.

학습 loss는 다음처럼 구성한다.

```text
L = L_IQL
  + lambda_rank * L_pairwise(Q_candidate - Q_anchor, sign(Delta J))
  + lambda_gain * L_regression(Q_candidate - Q_anchor, Delta J)
```

IQL target과 paired drain label의 시간 범위가 맞지 않으면 auxiliary loss를 켜지 않는다. 먼저 `K`-step return과 drain/full remainder target 중 논문의 estimand를 하나로 고정한다.

### Phase 6. ensemble LCB와 abstention calibration

`rl_leader/calibrate_response_lcb.py`를 추가한다.

- event-group bootstrap ensemble의 gain distribution을 계산한다.
- held-out calibration event에서 `LCB > material_margin`일 때만 개입한다.
- threshold는 calibration split에서 false positive 0을 우선해 정한다.
- OOD state, unseen response class, low ensemble agreement, replay mismatch에서는 P-Stack fallback한다.

Accuracy나 평균 MSE보다 아래 지표를 우선한다.

- selected candidate false-positive rate
- P-Stack fallback rate
- selected-event mean/median strict tail gain
- positive-event recall
- calibration coverage와 scenario별 worst case

### Phase 7. full-run 평가

1. 170과 190에서 one-shot intervention + P-Stack cooldown을 먼저 평가한다.
2. 통과하면 5개 scenario, 최소 3 seeds로 확장한다.
3. 모든 비교는 matched demand/incident seed와 동일 simulator revision을 쓴다.
4. TTT 전체, freeway/urban TTT, terminal inventory, follower infeasibility, intervention count를 함께 보고한다.

**승격 조건:** 5개 scenario 모두에서 P-Stack 대비 non-inferior이고 aggregate paired TTT가 개선되며, selected intervention의 strict-tail false positive가 0이다. 한 scenario라도 반복 열세면 데이터만 추가하기 전에 실패한 response class와 horizon을 attribution한다.

### Phase 8. 실제 leader replacement

성능 proof가 끝난 뒤 두 단계를 순서대로 수행한다.

1. `rl_leader/response_surrogate.py`: 모든 후보의 follower response와 gain을 예측하고 top-K만 actual follower solve
2. P-Stack anchor distillation: anchor action과 physical response를 모사한 뒤 nonlinear residual selector를 결합

다음 세 모드를 별도로 보고한다.

- `oracle-response`: 모든 actual follower solve, 방법의 성능 상한
- `surrogate-top-k`: 예측 후 top-K actual solve, 계산 절감
- `direct/distilled-anchor`: P-Stack leader solve 없이 실행, 최종 replacement

## 8. 필수 ablation

1. P-Stack
2. current linear owner-block oracle
3. linear-only anchored option IQL
4. linear + quadratic
5. linear + quadratic + cross-control
6. response dedup 제거
7. H3 label만 사용
8. H12/drain label 사용
9. LCB abstention 제거
10. current logistic/ridge selector
11. pairwise ranker
12. IQL, CQL, TD3+BC를 동일 finite candidate support에서 비교

알고리즘 비교에서 candidate set, state, split, label horizon을 바꾸지 않는다. 그래야 성능 차이를 알고리즘 차이로 해석할 수 있다.

## 9. 즉시 실행 순서

1. 현재 dense v3 freeze의 terminal artifact와 coverage를 다시 감사한다. CPU-heavy job은 자동 재개하지 않는다.
2. nonlinear action schema와 zero-residual P-Stack parity test를 구현한다.
3. 170/190 frozen state 소수에서 nonlinear headroom audit을 실행한다.
4. strict tail positive와 unique response 증가가 확인될 때만 5개 scenario batch 수집을 시작한다.
5. anchored option dataset builder와 response-aware IQL critic을 구현한다.
6. event-group calibration으로 LCB gate를 고정한다.
7. one-shot full-run을 통과한 뒤 repeated option과 surrogate/distillation로 넘어간다.

이 순서는 24시간 데이터를 먼저 모으고 방향을 나중에 판단하는 문제를 피한다. 각 단계는 짧은 oracle/headroom gate를 통과해야 다음 계산 예산을 사용한다.

## 10. 완료 정의

다음 조건을 모두 만족해야 논문 방법이 코드로 구현되었다고 본다.

- nonlinear coefficient가 follower objective와 물리 response에 실제 영향을 준다.
- zero nonlinear residual에서 P-Stack parity가 재현된다.
- candidate order와 branch execution이 결과를 오염시키지 않는다.
- 학습 단위가 raw action이 아니라 replayable response-equivalence class다.
- long-horizon label, option reward, deployment continuation이 같은 estimand를 사용한다.
- event-group holdout에서 false-positive 0의 calibrated fallback gate가 있다.
- 5개 scenario matched full-run에서 P-Stack 대비 non-inferiority 및 aggregate improvement를 확인한다.
- oracle-response, surrogate-top-K, direct replacement 결과를 구분해 보고한다.

## 11. 기존 파일의 역할

- `RL_ARCHITECTURE_DEEP_REVIEW.md`: 실패 원인과 구조 변경의 근거
- `RL_NEXT_STEPS.md`: 날짜순 실행 로그와 artifact 상태
- `RL_REPLACEMENT_PLAN.md`: 기존 leader replacement의 상세 이력
- 이 문서: 앞으로의 구현에서 우선하는 짧은 기준 계약

날짜순 로그와 이 문서가 충돌하면, 이미 생성된 artifact의 해석에는 당시 format contract를 사용하고 새 구현에는 이 문서를 적용한다.
