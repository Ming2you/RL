# RL Leader Architecture Deep Review

**작성일:** 2026-08-27  
**범위:** 문제 정의, MDP/POMDP, Stackelberg/follower 계약, 데이터와 counterfactual, reward/credit assignment, IQL 및 대안 알고리즘, deployment gate, 실험 재현성  
**검토 방식:** 6개 독립 감사와 1개 반론 중심 종합 검토. production code는 수정하지 않았다.

## 0. 결론

현재 결과만으로는 **IQL이 나빠서 RL이 P-Stack보다 열세라고 결론낼 수 없다.** 더 근본적인 문제가 알고리즘 바깥에 있다.

1. **성능 비교 계약이 잠겨 있지 않다.** 기존 `3809.607 / 5734.776 veh-h` P-Stack 산출물은 외부 `SUP_PFO` supervisor가 개입한 기록이고, v7 RL 평가는 `pfo_supervisor=false`, `pstack_anchor=true`다. 과거 실행의 전체 config fingerprint도 남아 있지 않다. 따라서 RL의 `3876.872 / 6154.657 veh-h`는 유효한 원시 결과지만, `+1.77% / +7.32%`를 동일 조건 P-Stack 대비 RL의 인과적 손실로 해석하면 안 된다.
2. **P-Stack 후보 평가가 순수 함수가 아닐 가능성이 높다.** 후보들을 같은 mutable follower solver로 순차 평가하며 `_prev_coupling`, numerical predictor/corrector, off-ramp cache 등이 후보 사이에서 바뀐다. 선택된 후보의 complete follower snapshot도 복원하지 않는다. 이 상태에서는 baseline, teacher, counterfactual label 자체가 후보 순서에 의존할 수 있다.
3. **학습 목표와 배포 결정이 다르다.** H12 label은 `candidate 1회 -> P-Stack continuation`의 효과인데, 배포 gate는 H3 동안 각 물리 제어를 고정한 local score로 선택한다. 이후 actor는 다시 매 step 제안한다. 세 계약은 서로 다른 정책을 평가한다.
4. **75차원 가격 residual은 식별 가능성이 낮다.** follower가 discrete green/offset/VSL/meter 후보를 고르므로 넓은 가격 영역이 같은 물리 제어로 매핑된다. v6와 v7의 가격 제안은 달라도 실제 선택 step과 제어가 거의 같았다. 반대로 actor의 평가 action은 학습 positive action보다 훨씬 dense했다.
5. **더 많은 데이터나 IQL update만으로 해결될 증거가 없다.** v7 actor는 positive training action을 거의 정확히 재현했다. 문제는 training fit보다 state/action support, causal label, deployment selection이다.

따라서 권장 방향은 **free-running 75D IQL actor를 잠시 중단**하고 다음으로 바꾸는 것이다.

> 동일 상태에서 P-Stack과 소수의 구조화된 residual 후보를 순수하게 계산하고, 실제 follower response가 다른 후보만 남긴 뒤, 장기 paired TTT advantage와 불확실성으로 선택하는 P-Stack-relative conservative policy improvement.

이 구조에서 첫 모델은 일반적인 offline RL actor보다 **pairwise/quantile ranker + ensemble lower confidence bound(LCB)**가 더 적합하다. IQL, CQL, TD3+BC는 동일한 discrete candidate set 위에서의 ablation으로 남긴다. 장기적으로 진짜 leader replacement가 목적이면 P-Stack distillation과 저차원/hierarchical policy를 별도 단계로 진행한다.

## 1. 무엇이 확인되었는가

### 1.1 현재 데이터와 학습

v7 clean pipeline의 핵심 수치는 다음과 같다.

| 항목 | 값 | 해석 |
|---|---:|---|
| raw transitions | 9,365 | 총량 자체는 작은 smoke 수준이 아님 |
| observation/action | 238 / 75 | 고차원 latent action |
| native zero-anchor rows | 2,443 | actor supervision의 대부분 |
| accepted local rows | 218 | H3를 통과한 residual |
| valid H12 labels | 200 | replay 가능한 accepted local |
| H12 positive | 66 | 실제 nonzero positive actor target |
| H3 false positives vs H12 | 134 | H3 sign test가 H12와 자주 반대 |
| nominal 170 positives | 0 | 핵심 평가 scenario에 positive support 없음 |
| training updates | 40,000 | 단순 update 부족 설명은 약함 |

positive 66개는 scenario별로 `155: 4`, `170 incident: 22`, `170 skew: 5`, `nominal 170: 0`, `190: 35`다. 이는 actor가 nominal 170에서 nonzero action을 선택할 때 학습 분포 내 보간이라고 보기 어렵다는 뜻이다.

### 1.2 actor optimization은 실패하지 않았다

v7 actor의 positive training action 재구성은 다음과 같다.

| 지표 | 값 |
|---|---:|
| target residual L2 mean | 0.246686 |
| predicted residual L2 mean | 0.245780 |
| action MSE | `1.13e-6` |
| cosine similarity mean | 0.999375 |
| zero-anchor predicted L2 mean | 0.004745 |

즉, **actor가 학습 데이터를 못 외운 것이 아니다.** 더 많은 epoch나 optimizer sweep은 training-state fit을 더 좋게 만들 수 있지만, nominal 170 support 부재, dense OOD action, many-to-one follower response, H3/H12 불일치를 해결하지 않는다.

### 1.3 v6와 v7은 behavior가 거의 같았다

v7은 H12 filtering과 terminal/reward 계약을 고쳤지만 full-run 선택은 v6와 같았다.

- nominal 170: step 14에서 1회 선택
- nominal 190: step 28, 35에서 2회 선택
- 190의 total/peak/recovery/throughput은 v6와 v7이 동일
- 170은 total TTT가 `0.206 veh-h`만 차이
- 190의 선택 step에서 일부 raw residual은 달랐지만 realized control은 동일

이는 H12 relabeling이 필요 없었다는 뜻이 아니다. H12 filtering은 잘못된 H3 positive를 actor target에서 제거했다. 다만 **배포 gate와 response topology가 그대로여서 최종 의사결정이 거의 바뀌지 않았다.**

### 1.4 price action은 behaviorally many-to-one다

학습 positive action은 block-sparse했다.

- positive behavior action의 active dimension 중앙값: 13 (`abs(a)>1e-5`)
- 평가에서 accept된 actor action: 50, 52, 54 active dimensions
- 190 step 35는 43 dimensions가 `abs(a)>1e-3`

현재 support guard는 coordinate별 min/max clipping이다. 이는 각 좌표가 범위 안에 있어도 **그 좌표 조합이 학습 데이터에 없었던 joint OOD action**을 탐지하지 못한다. 더 중요한 것은 서로 다른 price vector가 follower의 discrete 후보 선택 후 동일한 physical control이 될 수 있다는 점이다. 이 경우 price coordinate MSE와 Q-gradient는 실제 교통 제어 차이를 잘 나타내지 못한다.

## 2. 기존 성능 해석에서 정정할 부분

### 2.1 확실히 말할 수 있는 것

- v7 RL full run의 TTT는 nominal 170에서 `3876.872`, nominal 190에서 `6154.657 veh-h`다.
- v7 trace는 `pfo_supervisor=false`, `pstack_anchor=true`다.
- 과거 frozen P-Stack baseline의 run log에는 외부 `sup_pick_pfo`가 실제로 선택된 step이 있다.
  - 170: 13개 step
  - 190: 6개 step
- 과거 baseline artifact에는 현재 요구되는 canonical config/environment fingerprint가 없다.

### 2.2 지금은 말하면 안 되는 것

- “동일 환경의 순수 P-Stack보다 RL이 정확히 1.77%/7.32% 나쁘다.”
- “세 번의 RL override가 전체 67/420 veh-h 손실을 만들었다.”
- “24시간 데이터 재학습이 P-Stack 대비 성능을 악화시켰다.”

위 문장들은 동일 initial state, 동일 plant/controller config, 동일 supervisor, 동일 warm-up, 동일 random/exogenous contract로 만든 paired baseline이 있어야 성립한다.

### 2.3 즉시 필요한 세 개 arm

하나의 immutable `ExperimentContract`에서 다음을 동시에 실행해야 한다.

1. **Direct native P-Stack:** 외부 supervisor 없이 native P-Stack만 실행
2. **Forced anchor-only env:** RL env를 사용하되 모든 step에서 zero residual/P-Stack branch 강제
3. **Residual policy:** 같은 env에서 learned residual 허용

1과 2는 75 step 동안 interval TTT, physical control, follower/controller state checksum, final TTT가 tolerance 내에서 같아야 한다. 이 parity가 성립한 뒤에만 2와 3의 차이를 RL override 효과로 해석한다.

## 3. 가장 근본적인 구현 위험: P-Stack candidate purity

### 3.1 현재 구조

`StackelbergMPCController._evaluate_full_candidate()`는 여러 leader candidate를 같은 `nash_solver`에 순차적으로 넣는다. follower solve는 다음 state를 읽고/쓴다.

- `_prev_coupling`
- numerical predictor/corrector memory
- off-ramp flow cache
- primal/dual 관련 runtime state

선택 시 일부 dual 값은 복원되지만, 선택된 candidate의 complete follower/controller snapshot을 commit하지 않는다. 따라서 구현상 mapping은 다음과 같을 수 있다.

```text
(response_i, hidden_i) = follower_solve(hidden_{i-1}, candidate_i)
```

원래 필요한 계약은 다음이다.

```text
(response_i, hidden_i) = follower_solve(common_hidden_0, candidate_i)
```

### 3.2 영향 범위

이 문제는 RL에만 영향을 주지 않는다.

- P-Stack benchmark의 candidate ranking
- optimizer anchor action
- local teacher action
- native price replay
- counterfactual label
- follower-response surrogate 학습
- RL/P-Stack outer branch commit

outer gate에도 같은 위험이 있다. RL branch를 먼저 계산한 뒤 P-Stack을 선택할 때 현재 코드는 정해진 follower field 일부만 복사한다. rejected RL branch가 바꾼 price/beta/regret/controller history 전체를 rollback하거나, 선택된 P-Stack branch의 complete state로 교체한다는 보장이 없다. 따라서 "gate가 P-Stack을 선택했다"와 "P-Stack만 단독 실행했다"가 아직 같은 transition이라고 볼 수 없다.

따라서 **새 RL 알고리즘을 학습하기 전 E0**로 처리해야 한다.

### 3.3 수정 계약

1. decision 시작 시 plant, follower, leader/controller runtime의 complete snapshot을 만든다.
2. 모든 candidate를 동일 snapshot에서 독립 평가한다.
3. candidate order를 random permutation해도 response/objective/rank가 동일해야 한다.
4. 선택된 candidate의 complete post-solve snapshot만 live controller에 commit한다.
5. outer RL/P-Stack branch도 동일 규칙으로 complete branch state를 commit한다.
6. snapshot schema hash와 candidate order를 artifact에 기록한다.

이 테스트가 실패하면 현재 데이터와 checkpoint는 원인 분석 자료로는 쓸 수 있어도 새 architecture의 teacher data로 재사용하면 안 된다.

## 4. 현재 문제는 어떤 의사결정 문제인가

### 4.1 분류

현재 simulator는 policy step이 고정 180초이므로 본질적으로 variable-duration semi-MDP는 아니다. complete state를 가진다면 finite-horizon sampled-data macro-MDP다. 하지만 현재 observation은 transition에 영향을 주는 state를 모두 포함하지 않으므로 배포 관점에서는 **hybrid bilevel POMDP**에 가깝다.

누락 가능성이 큰 state는 다음과 같다.

- freeway upstream/downstream buffer density/speed와 mainline origin queues
- age-indexed urban scheduled-arrival/release buffers
- off-ramp follower cache
- native P-Stack regret/beta/PFO/cache state
- full directional demand, skew, incident remaining duration
- current anchor의 price, physical plan, follower response와 objective decomposition

특히 actor는 현재 step의 P-Stack anchor가 계산되기 전에 residual을 출력하지만, 학습 target은 그 anchor에 상대적으로 정의된다. 즉 target 생성에 필요한 branch/anchor 변수가 actor input에 없는 hidden context다. 새 ranker나 policy는 먼저 anchor를 계산하고 `observation + anchor action + anchor response + candidate`를 입력받아야 한다.

같은 aggregate occupancy라도 예정된 release age가 다르면 다음 state가 다르다. 같은 238D observation과 같은 75D price가 anchor 및 follower hidden state에 따라 다른 physical control을 만들 수도 있다.

### 4.2 현재 reward

현재 environment reward는 정확히 다음과 같다.

```text
r_t = -(urban interval TTT + freeway interval TTT)
```

IQL은 기본 `gamma=1.0`, `reward_scale=0.01`을 사용한다. 자연 종료 trajectory의 reward 합은 scale을 제외하면 `-total TTT`와 정렬된다. 따라서 **reward 정의 자체가 total TTT가 아니라서 실패한 것은 아니다.**

그런데도 해결되지 않은 이유는 다음과 같다.

1. critic은 gate가 실제로 적용한 trajectory를 학습한다. H3에서 reject된 candidate의 장기 결과는 Bellman data에 없다.
2. actor는 Q를 직접 최대화하는 것이 아니라 behavior/deployed action을 advantage-weighted MSE로 모사한다.
3. H12는 critic target이 아니라 actor row filter로만 쓰인다.
4. H12 label은 한 번 intervention 후 P-Stack이고, 실제 actor는 다음 step에 다시 제안한다.
5. H3 gate는 fixed-control 540초를 평가하고, episode objective는 14,400초와 recovery를 평가한다.
6. POMDP aliasing이 있으면 같은 `(obs, action)`에 서로 다른 return이 섞인다.

즉, **좋은 reward는 필요한 조건이지만 올바른 counterfactual action coverage와 policy contract를 자동으로 만들어주지는 않는다.**

### 4.3 bandit인가 RL인가

교통 상태가 다음 의사결정에 영향을 주므로 원래 system은 contextual bandit이 아니다. 다만 권장하는 첫 architecture는 다음 causal estimand를 예측한다.

```text
Delta J(s, a) = J(P-Stack from s)
              - J(a once, then P-Stack from the resulting state)
```

그리고 intervention 뒤 P-Stack cooldown을 강제한다. 이 경우 learned component는 **state-conditioned one-shot treatment selector**로 단순화되며, pairwise ranking/contextual policy improvement가 일반 sequential offline RL보다 더 정확한 모델링이다. 이 단계가 통과한 뒤에만 multi-action option policy로 확장한다.

## 5. 왜 이전 수정들이 먹히지 않았는가

| 수정 | 해결한 실제 문제 | 남은 문제 |
|---|---|---|
| observation/action 117/47 -> 238/75 확장 | 누락된 VSL, offset, previous control, 일부 follower memory 추가 | complete Markov state와 current anchor는 여전히 없음 |
| 1차 price -> quadratic/nonlinear potential | follower에 더 풍부한 incentive 표현 가능 | 75D joint output과 discrete response 때문에 식별성이 더 어려워짐 |
| follower hidden-state teacher sync | teacher가 다른 follower state를 보는 오류 감소 | native P-Stack candidate search 자체의 mutable path는 남음 |
| applied-action critic fix | reject된 proposal을 실제 적용 action으로 기록 | reject된 proposal의 counterfactual quality는 여전히 학습 못함 |
| external truncation/reward fix | wall-clock cutoff를 terminal로 오해하는 오류 수정 | censoring bias와 horizon mismatch는 남음 |
| H12 positive filtering | H3 false positive 134개를 actor target에서 제거 | H3 deployment gate, accepted-only label bias, H12-to-end mismatch는 남음 |
| 24시간 데이터 수집 | row 수와 scenario coverage 확대 | positive 66개, nominal170 0개, action mode 불균형은 해소 안 됨 |
| top-K / H6 / H9 | actor proposal 주변 후보와 긴 local score 시도 | 같은 actor의 scale 변형이라 response 다양성이 낮고 long-run TTT ranker가 아니었음 |
| actor update 증가 | training action fit 개선 | 이미 positive target fit이 거의 완벽해 generalization 문제는 해결 안 됨 |

핵심은 **데이터의 양이 아니라 estimand와 support의 질**이다. 24시간 trajectory를 더 모아도 거의 모두 zero anchor이거나 같은 response cell에 들어가면 유효한 action 비교 정보는 늘지 않는다.

## 6. action architecture 재설계

### 6.1 지금 바로 유지할 것

- follower의 green, offset, meter, VSL response capability
- budget과 nonlinear potential을 표현하는 schema
- exact P-Stack zero residual
- hard physical validity와 slew/box constraints
- total TTT objective

### 6.2 지금 중단할 것

- 한 actor가 75개 coordinate를 동시에 자유 출력
- coordinate min/max만으로 in-support라 판단
- 모든 price 차이를 서로 다른 action으로 간주
- H3 accept 여부만으로 label eligibility 결정
- 같은 actor vector를 단순 scale한 것을 top-K 다양성으로 간주

### 6.3 권장 candidate action

첫 단계의 action은 `owner/template/magnitude`로 구조화한다.

```text
owner:
  urban green block | urban offset block | ramp-meter block | VSL segment/group

template:
  signed linear | convex quadratic | monotone spline | behavior-neighbor shape

magnitude:
  small discrete levels, state-conditional bounded scalar
```

candidate set 예시는 다음과 같다.

1. zero residual, 즉 exact P-Stack
2. single-owner `+/-` perturbation
3. 상호작용이 명확한 two-owner correlated perturbation
4. 현재 state의 nearest successful behavior residual
5. ensemble actor 또는 CEM이 제안한 1개 후보

각 후보를 follower에 넣고 **realized physical control/response가 동일하면 deduplicate**한다. 학습과 support 판단의 기본 단위도 raw price가 아니라 다음 tuple이어야 한다.

```text
(state, anchor intent, residual price template,
 follower response, realized physical control, continuation policy)
```

### 6.4 green/VSL/offset을 왜 동시에 버리지 않는가

모든 제어 family를 최종적으로 고려하는 것은 맞다. 다만 첫 실험부터 전부 dense하게 동시에 출력하는 것과, candidate set에 모든 family를 균형 있게 포함하는 것은 다르다.

- green, offset, meter, VSL을 **모두 candidate library에 포함**한다.
- 한 candidate에서는 우선 1개 block, 이후 검증된 2개 block만 활성화한다.
- owner별 효과와 interaction을 먼저 측정한다.
- 충분한 paired evidence가 있는 block만 joint policy로 승격한다.

이렇게 하면 “전부 본다”는 연구 범위는 유지하면서 어떤 제어 family가 장기 TTT를 실제로 개선했는지 식별할 수 있다.

## 7. 알고리즘 선택

### 7.1 권장 우선순위

| 방법 | 현재 적합성 | 역할 |
|---|---|---|
| Pairwise ranker + quantile/ensemble LCB | **가장 높음** | 명시적 candidate의 P-Stack-relative 장기 gain 선택 |
| SPIBB-style baseline bootstrapping | 높음 | support 부족 state/block에서 P-Stack 강제 |
| Candidate-constrained TD3+BC | 중간 | finite candidate scorer/actor ablation |
| Candidate-constrained CQL | 중간 | pessimistic candidate Q ablation; zero collapse 감시 |
| IQL/AWR | 중간 이하 | 동일 candidate/action redesign 위의 비교 baseline |
| SAC/PPO online simulator RL | 이후 | 저차원 option policy가 검증된 뒤 exploration/fine-tuning |
| CEM/MPPI/model-predictive search | 높음, 비학습 oracle | candidate discovery와 expert label 생성 |
| P-Stack distillation/DAgger | 높음, replacement 단계 | 계산량 절감 및 learned anchor |
| BCQ/BEAR | 낮음 | 66 positive로 75D multimodal behavior model 학습 위험 |
| Decision Transformer | 낮음 | trajectory/positive support 부족; 현재 sequence contract 없음 |
| learned world model offline RL | 낮음 | 이미 실제 simulator가 있어 다음 단계로는 model bias만 추가 |

### 7.2 왜 IQL을 CQL로 바꾸는 것만으로 부족한가

CQL은 OOD action value를 낮출 수 있지만 meaningful 75D action sampling이 필요하다. 현재처럼 positive가 희소하고 zero anchor가 지배하면 CQL이 안전하게 P-Stack parity로 수렴할 가능성은 있어도 **새로운 좋은 response cell을 찾아줄 이유는 없다.**

TD3+BC는 actor를 Q-max 방향으로 더 직접 이동시키지만, 현재 critic support에서는 오히려 extrapolation error를 exploit할 수 있다. 두 방법 모두 action을 구조화하고 paired candidate label을 확보한 후 비교해야 한다.

### 7.3 exploration은 epsilon-greedy가 가능한가

가능하지만 현재 75D continuous space에 무작위 epsilon action을 넣는 방식은 비효율적이다. 권장 exploration은 다음과 같다.

1. discrete owner/template에 epsilon-greedy 또는 Thompson sampling 적용
2. magnitude는 작은 단계 또는 trust region에서 탐색
3. proposal probability를 항상 기록
4. H3 accept/reject와 무관하게 stratified sample을 long-horizon label
5. ensemble uncertainty가 큰 state-action cell을 active learning으로 추가
6. 실제 follower response가 기존 response와 같은 후보는 새 exploration으로 계산하지 않음

온라인 simulator RL 단계에서는 SAC entropy, parameter noise, option-level epsilon-greedy를 쓸 수 있다. 그러나 탐색 단위는 raw 75D price가 아니라 **behaviorally distinct option/response**여야 한다.

## 8. 권장 전체 architecture

### 8.1 Stage A: P-Stack Anchored Conservative Policy Improvement

```text
Full decision snapshot
        |
        +--> Pure native P-Stack candidate (always included)
        |
        +--> Structured residual candidate generator
                    |
                    v
          Pure follower solves from identical snapshot
                    |
                    v
          Dedupe by realized response/control
                    |
                    v
     Long-horizon delta ranker + uncertainty/OOD
                    |
           LCB > material margin?
              /             \
            no               yes
        P-Stack       H3 physics sanity check
                           /       \
                         fail      pass
                       P-Stack   commit complete
                                 branch snapshot
                                      |
                              P-Stack cooldown
```

H3는 performance selector가 아니라 short-horizon physical sanity check로만 남긴다. 장기 selection은 paired continuation으로 학습한 LCB가 담당한다.

### 8.2 Stage B: 저차원 sequential option policy

Stage A의 one-shot intervention이 full-run에서 P-Stack을 이긴 뒤에만 진행한다.

- action: owner/template/magnitude/duration option
- state: plant + full forecast + previous realized control + anchor/response + controller memory
- observation alias가 측정되면 recurrent belief encoder 사용
- label/data: option sequence와 continuation을 실제 deployment policy와 동일하게 수집
- algorithm: simulator-online SAC/PPO 또는 conservative offline-to-online AWAC/TD3+BC
- fallback: state/action OOD 또는 LCB 실패 시 P-Stack

### 8.3 Stage C: 진짜 leader replacement

현재 residual system은 매 step full P-Stack과 RL follower, H3 branches를 모두 계산하므로 leader replacement나 계산량 감소가 아니다. 다음 두 방법 중 연구 목적을 명확히 선택한다.

**Stackelberg 구조 유지:**

1. P-Stack을 DAgger/distillation해 learned anchor 생성
2. learned anchor의 response/TTT parity를 검증
3. structured residual selector 추가
4. P-Stack은 shadow/fallback으로 호출 빈도를 점차 축소

**교통 성능 최우선:**

1. price가 아니라 bounded physical-control residual을 직접 출력
2. follower는 safety/projector 또는 distributed local optimizer 역할
3. P-Stack과 direct-control RL을 별도 연구 질문으로 비교

두 번째는 성능상 더 단순할 수 있지만 “leader가 가격을 내려 follower가 자율 반응하는 Stackelberg mechanism”이라는 논문 주장은 약해진다.

## 9. 새 데이터 수집 설계

### 9.1 24시간 연속 수집보다 먼저 할 것

새로 24시간을 다시 돌리기 전에 `100 snapshots x 4 candidates = 400 paired labels`의 방향성 검증을 권장한다. 각 snapshot의 P-Stack continuation은 한 번 계산해 공유한다.

| Stratum | snapshots | candidate labels |
|---|---:|---:|
| nominal 170 peak/recovery | 24 | 96 |
| nominal 190 peak/recovery | 24 | 96 |
| 170 incident peak/recovery | 20 | 80 |
| 170 skew peak/recovery | 16 | 64 |
| 155 peak/recovery | 16 | 64 |
| **합계** | **100** | **400** |

snapshot당 후보 구성:

- single-owner 두 개: 50%
- two-owner interaction 한 개: 25%
- hard negative/current actor candidate 한 개: 25%

green, offset, meter, VSL owner가 균형 있게 회전하도록 강제한다. H3-rejected 후보도 일부 무작위로 label해 selection bias를 측정한다.

### 9.2 outcome horizon

모든 candidate에 H12를 기록하되 다음을 추가한다.

- stratified subset은 full-to-end 또는 branch reconvergence까지 진행
- H1/H3/H6/H12/full-to-end sign curve 기록
- H12 positive가 full-to-end에서 뒤집히면 H12를 primary label로 사용하지 않음
- terminal inventory total뿐 아니라 위치, speed, age-to-release state를 terminal value에 반영

### 9.3 필수 event schema

```text
experiment_contract_hash
snapshot_schema_hash
episode/scenario/phase/time_to_go
full plant/controller/follower state hash
anchor raw action + anchor response + anchor physical control
proposal source + assignment probability
candidate raw residual + support/OOD scores
candidate follower response + realized physical control
gate scores + selected/applied branch
continuation policy + cooldown contract
H1/H3/H6/H12/end TTT and inventory deltas
replay parity/error and solver diagnostics
```

train/validation/test는 row가 아니라 whole episode/scenario instance 단위로 나눈다. 170/190은 반복 개발에 사용되었으므로 최종 test에는 새로운 locked demand/incident seeds가 필요하다.

## 10. 실행 계획과 stop/go gate

### Phase 0: RL 이전의 deterministic contract

1. canonical `ExperimentContract`와 fingerprint를 baseline/collection/label/eval에 공통 적용
2. P-Stack candidate snapshot purity와 order invariance 구현/검증
3. selected candidate complete state commit 구현/검증
4. outer RL/P-Stack branch complete commit parity 검증
5. direct P-Stack vs forced-anchor env 75-step exact parity

**중단 조건:** order에 따라 candidate response/rank가 달라지거나, direct/anchor-only TTT가 unexplained tolerance를 넘으면 학습하지 않는다.

### Phase 1: 학습 없는 oracle 진단

1. action controllability Jacobian/effective rank 측정
2. raw price 후보를 realized response 기준으로 dedupe
3. H3 accept/reject 전체에서 H12/end label 표본 수집
4. frozen v7 세 accepted event의 paired-to-end attribution
5. small candidate catalog의 oracle headroom 측정

**진행 조건:** locked state의 의미 있는 비율에서 P-Stack 대비 positive full-return candidate가 존재하고, response가 재현 가능해야 한다. 예시 gate는 `>=5%` state에서 remaining TTT의 `>=1%` 개선이다.

### Phase 2: 100 snapshot pilot

1. 400 candidate paired labels 수집
2. H12-to-end sign stability 검증
3. scenario/phase/owner 균형 및 support audit
4. ranker, quantile model, ensemble calibration

**진행 조건:** normal 170/190 holdout에서 positive candidate가 존재하고, LCB-selected candidate의 false-positive가 0이어야 한다. H12 positive가 end에서 한 번이라도 뒤집히면 dynamic-to-recovery label로 전환한다.

### Phase 3: 알고리즘 ablation

동일 candidate set과 split에서 비교한다.

1. pairwise ranker + quantile/ensemble LCB
2. SPIBB-style discrete baseline bootstrapping
3. candidate CQL
4. candidate TD3+BC
5. redesigned IQL/AWR
6. exact simulator oracle upper bound

평가 지표:

- P-Stack-relative paired TTT delta
- false-positive override rate
- positive candidate recall
- peak/recovery TTT
- terminal/recovery inventory
- OOD abstention rate
- realized control diversity
- wall-clock/computation cost

critic loss나 actor MSE로 모델을 선택하지 않는다.

### Phase 4: one-shot full-run

- label과 동일하게 한 번 intervention 후 P-Stack cooldown
- canonical direct P-Stack, forced-anchor, learned selector 세 arm 동시 실행
- 먼저 170/190 seed 0
- 둘 다 non-inferior하고 하나 이상 개선할 때만 3 seeds x 5 scenarios 확장

**중단 조건:** total 또는 recovery TTT가 P-Stack보다 악화되거나, paired holdout에 없던 false-positive override가 나오면 데이터 추가보다 gate/support 원인으로 돌아간다.

### Phase 5: sequential RL 여부 결정

one-shot selector가 반복적으로 유용한 intervention을 찾았을 때만 option sequence 데이터를 새로 수집한다. one-shot label을 그대로 recurrent actor 학습에 재사용하지 않는다.

**진행 조건:** repeated-policy return과 one-shot advantage 합의 sign 및 magnitude가 안정적이고, option interaction을 설명할 수 있어야 한다.

## 11. 가장 먼저 수행할 실험 8개

우선순위는 다음과 같다.

1. **Candidate order-invariance:** candidate 순서를 20개 permutation해 response/objective/rank 비교
2. **Complete snapshot commit:** 선택 candidate와 persisted follower/controller state checksum 일치 검증
3. **75-step fallback parity:** direct native와 forced-anchor-only exact comparison
4. **Frozen v7 event attribution:** 세 accepted state를 P-Stack vs candidate-once-to-end로 분기
5. **Horizon sign curve:** H3 accepted/rejected를 모두 포함해 H1/H3/H6/H12/end 비교
6. **Effective-rank audit:** 각 owner/template perturbation이 realized control과 TTT에 미치는 Jacobian 분석
7. **Oracle headroom:** 100 stratified state에서 small catalog exhaustive paired evaluation
8. **Ranker pilot:** 400 labels로 LCB selector를 만들고 50 unseen paired event에서 false-positive 0 검증

1-3은 RL 실험의 전제다. 4-7은 RL이 실제로 개선 가능한 action field를 갖는지 판정한다. 8이 통과해야 재학습과 full run을 시작한다.

## 12. 연구 주장별 권장 경로

### 목표 A: Stackelberg coordination mechanism을 유지하며 P-Stack 개선

가장 권장한다. follower는 그대로 두고 leader는 structured price/budget candidate를 생성한다. 논문 주장은 “offline RL leader replacement”보다 “uncertainty-aware learned policy improvement over a Stackelberg baseline”이 된다.

### 목표 B: MPC leader 계산량을 학습으로 대체

P-Stack distillation이 먼저다. exact anchor imitation과 physical-response parity를 달성한 뒤 residual improvement를 얹는다. 현재처럼 P-Stack을 매 step 전부 계산하면 replacement 주장을 할 수 없다.

### 목표 C: TTT 성능 자체를 최대화

direct physical-control residual + safety projection을 parallel benchmark로 둔다. 이 방법이 price-space보다 월등하면 price mechanism의 성능 비용을 정직하게 계량할 수 있다.

세 목표를 하나의 architecture로 동시에 달성하려 하면 baseline, action semantics, compute claim이 다시 섞인다. 논문에서는 최소한 A/B/C를 별도 arm으로 분리해야 한다.

## 13. 최종 판단

현재 framework가 전부 잘못된 것은 아니다. 다음은 계속 가져갈 가치가 있다.

- follower를 보존하고 leader incentive를 학습하려는 연구 질문
- P-Stack anchor와 paired simulator counterfactual
- total TTT reward
- nonlinear price capability
- hard validity와 fallback

하지만 현재의 **`238D observation -> dense 75D IQL residual -> follower -> H3 gate`**를 그대로 확장하는 것은 권장하지 않는다. 가장 큰 문제는 IQL이 아니라 다음 네 계약이 하나로 정렬되지 않은 것이다.

1. baseline이 무엇인가
2. 한 action이 실제로 어떤 follower response를 의미하는가
3. label이 어떤 continuation policy와 horizon의 효과인가
4. deployment가 같은 효과를 선택하고 실행하는가

새 방향은 이 네 계약을 먼저 고정한 뒤, raw price actor 대신 **구조화된 후보 + pure follower response + 장기 paired ranker + uncertainty abstention**을 사용하는 것이다. 여기서 개선 신호가 확인되면 sequential RL과 leader distillation로 확장한다. 신호가 없으면 데이터나 알고리즘 부족이 아니라 현재 follower/price action field에서 P-Stack을 이길 oracle headroom이 부족하다는 결론을 낼 수 있다.

## 14. 독립 감사 문서

- `results/rl_architecture_review_20260827/01_mdp_problem_formulation.md`
- `results/rl_architecture_review_20260827/02_offline_rl_algorithms.md`
- `results/rl_architecture_review_20260827/03_stackelberg_control_architecture.md`
- `results/rl_architecture_review_20260827/04_data_counterfactual_causality.md`
- `results/rl_architecture_review_20260827/05_safety_gate_deployment.md`
- `results/rl_architecture_review_20260827/06_code_experiment_forensics.md`
- `results/rl_architecture_review_20260827/07_adversarial_synthesis.md`

주의: 독립 감사는 의도적으로 서로 격리되어 작성되었다. 일부 문서가 과거 `3809/5735`를 equivalent native P-Stack으로 전제하거나 현재-code config materialization을 과거 baseline 실행과 동일시한 부분은 본 종합 검토의 Section 2에서 정정했다. 최종 의사결정에는 본 문서의 교차검증 결론을 우선한다.
