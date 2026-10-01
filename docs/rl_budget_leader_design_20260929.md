# RL leader의 budget 선택: 현재 player별 S-DMPC 적용 설계

작성일: 2026-09-29. 아래는 최초 설계안이며 원본 실행 코드와 과거 결과는 보존했다. **별도 구현의 제한된 pilot v1은 완료했으나 성능 개선은 확인하지 못했다.** 구현·검증·결과 및 이후 논의는 [실행 기록](rl_budget_execution_20260929.md)을 따른다. PFO 없는 이전 실행 budget 기준 방식은 아직 미구현 제안이다.

## 1. 제안과 적용 범위

**한 개의 RL agent가 control interval마다 두 budget 상한을 제안하고, 기존 하위 S-DMPC가 실제 제어를 계산한다.** 매 interval의 PFO 제어와 원 모델 검사를 유지한다. 첫 버전은 현재 inequality + externality ON 조건을 기준으로 한다.

기존: PFO 기준 → budget 후보 최대 3개에 각각 하위 계산 → 예상 TTT로 선택.

제안: PFO 기준 → RL이 budget 1개 제안 → 하위 계산 1회 → 원 모델 검사 및 PFO TTT guard → 실행.

RL은 RM, VSL, green, offset을 직접 출력하지 않는다. 외부효과 민감도와 budget Jacobian도 대체하지 않는다. 하위의 9-player 구조, 4개 freeway player의 그룹 VSL, H3, 최대 6회 반복, 물리 모델, 제어 제한, 순수 TTT 회계를 유지한다. 기존 후보 반복마다 RL을 다시 호출하는 방식은 첫 버전에서 사용하지 않는다.

## 2. 확인한 현재 코드와 계약

기준 실행 진입점은 `work/sdmpc_externality_ablation_20260923/run_scenario.py`의 `--variant upper --externality on`이다. 정본 `src` 전체와 혼동하지 않는다.

| 항목 | 확인한 현재 동작 | RL 설계 |
|---|---|---|
| 시작 budget | 이번 interval의 그룹·격자·변화 제한을 반영한 PFO 제어의 rollout 달성량 | 그대로 사용 |
| leader 후보 | 최대 3개, 중앙 1차 모델에서 회수한 근사 승수 방향 또는 명시적 축 탐색 | actor의 2차원 출력으로 대체 |
| 하위 지역 목적 | own와 externality의 1차항 + budget 가격항 + proximal 항 | 그대로 유지; 원 비선형 own NLP라고 부르지 않음 |
| 민감도 | 기준점이 바뀌면 재계산, 동일 기준점 캐시 | 유지 |
| 가격 | 같은 Jacobi 반복 안에서는 공통 가격, 응답 후 k→k+1에서 갱신 | 유지 |
| 가격 carry-over | 선택된 하위 결과의 가격을 전달; PFO 선택 시 incoming 가격 유지 | 유지 |
| 실행 선택 | budget/물리/제어 유효 + 동일 H3 예상 TTT가 PFO 이하일 때 하위안 선택 | 유지 |
| 최적화 인증 | 실행 가능과 수렴을 구분; `converged=False` 보존 | RL 적용만으로 PASS로 바꾸지 않음 |

**가격 정책 주의:** 과거의 interval 내 고정가격 버전과 달리, 위 기준 코드의 `prox_controller.py`는 하위 반복마다 `update_duals`를 호출한다. RL 비교에서는 이 현재 정책을 고정하고 leader 선택만 변경한다. interval 고정가격을 다시 채택한다면 별도 비교 실험이다.

물리 budget 정의는 다음과 같다.

\[
G_P(X_t,U)=\sum_{h=0}^{H-1}\big(S_{\mathrm{in},h}-S_{\mathrm{out},h}\big),
\qquad G_{UF}(U)=\sum_{r\in\mathcal R}m_r.
\]

\[
G_P\le B_P,\qquad G_{UF}\le B_{UF}.
\]

- `G_P`: 기존 coupling 진단의 inbound service − outbound service를 H3 전체에 합한 **veh**. 단순 도시 stock 자체가 아니다. 부호가 있는 값이므로 음수를 임의로 0으로 자르지 않는다.
- `G_UF`: 현행 시간축 상수 move block의 metering command 합 **veh/h**. 실제 ramp 통과량이나 H3 누적 차량 수가 아니다.
- actor 출력은 budget이다. 하위 가격이나 recovered multiplier와 단위·역할이 다르다.
- upper 모드에는 equality나 추가 band margin을 넣지 않는다.

근거 코드: `anchor_controller.py`, `fixed_policy.py`, `prox_controller.py`, `band_math.py`(동일 작업 디렉터리); `work/sdmpc_selected_dual_20260922/controller.py`; `work/sdmpc_central_kkt_reuse_20260922/central_controller.py`; `work/sdmpc_slide_alignment_20260922/group_block_engine.py`.

## 3. RL이 해결할 문제를 먼저 분명히 한다

동일 상태·예측 구간에서 하위가 전역 최적해를 계산한다고 가정하면,

\[
V_H(X,B)=\min_{U\in\mathcal F(X),\;G(X,U)\le B}J_H(X,U),
\qquad B'\ge B\Rightarrow V_H(X,B')\le V_H(X,B).
\]

상한을 풀면 feasible set이 커진다. 따라서 상·하위가 같은 H3 TTT만 최소화한다면, 이상적인 계산에서는 더 느슨한 상한이 약하게 우월하다. 짧은 예측 TTT를 줄이는 budget 선택만 모방하면 RL이 늘 상한을 크게 출력하거나, 하위 solver의 계산 특성을 학습할 수 있다.

의미 있는 역할은 **하위의 짧은 예측 구간 밖까지 이어지는 실제 폐루프 TTT를 줄이는 budget 선택**이다. 예를 들어 지금 ramp 대기를 허용했을 때 이후 mainline 혼잡을 얼마나 줄이는지, 도시 유입 제한으로 외부 대기열이 얼마나 쌓이는지 학습한다. 이것은 성능 가설이며 개선 보장이 아니다.

현재 하위는 유한 반복의 근사 solver이므로 실제 응답은 최적 반응 \(U^*(X,B)\)가 아니라 \(\widehat U=\mathcal S_6(X,B,w,p)\)이다. RL은 이 solver와 fallback을 포함한 폐루프를 학습한다. 명칭은 **RL-guided budget coordination for hierarchical S-DMPC**가 적절하다. Stackelberg 계층은 유지하지만 정확한 Stackelberg equilibrium을 계산했다고 주장하지 않는다.

## 4. 상태, 행동, 보상

### 상태/관측

actor 관측 \(o_t\)에는 다음을 고정 순서로 정규화해 넣는다.

| 입력 | 목적 |
|---|---|
| 각 freeway segment 밀도·속도 | 평균값에 가려지는 병목 위치 구분 |
| 각 ramp queue, origin queue, offramp storage와 점유율 | 혼잡을 다른 대기열로 옮기는 현상 파악 |
| urban movement별 queue, 링크별 transit storage, 수신 여유 공간 | green·offset·perimeter 상호작용 파악 |
| 지연 도착 buffer와 signal phase/time 정보 | 같은 총 차량 수에서도 미래 전이가 다른 상태 구분 |
| controller가 이용하는 동일 수요 예측, 방향별 비율, 관측된 incident/capacity | skew·incident와 미래 유입에 대응 |
| 이전 실행 제어, incoming 하위 가격, 이전 요청/선택 budget, slack, fallback 상태 | 변화 제한과 solver의 기억 반영 |
| 현재 PFO의 budget·예상 TTT·제어, 남은 episode 시간 | 기준점과 유한시간 목적 반영 |

정규화 상수는 훈련 데이터 또는 물리 용량으로 고정한다. 테스트 데이터로 갱신하지 않는다. 실제 운용에서 알 수 없는 미래 incident나 실현 수요를 추가 입력하지 않는다. simulator의 전체 상태를 기록하되 관측을 축약한다면 POMDP라는 점을 명시하고 history/recurrent actor의 필요성을 검증한다. 도시·고속도로 각 하나의 평균 누적값만으로 시작하지 않는다.

### 행동: PFO 기준의 잔차 budget

\[
B_t^0=G(X_t,U_t^{\mathrm{PFO,mapped}}),\qquad
a_t=\pi_\theta(o_t)\in[-1,1]^2,
\]
\[
B_{P,t}=B_{P,t}^0+50a_{P,t}\;[\mathrm{veh}],
\]
\[
B_{UF,t}=\operatorname{clip}\big(B_{UF,t}^0+1000a_{UF,t},
0,\sum_r m_r^{\max}\big)\;[\mathrm{veh/h}].
\]

50 veh, 1000 veh/h는 확인한 기존 leader의 탐색 반경을 첫 비교의 출발값으로 사용한 것이다. 정책 출력 범위·기존 실제 후보 envelope를 함께 기록한다. 범위 확대는 별도 실험이다. actor 마지막 층을 0으로 초기화하면 처음에는 PFO 달성 budget을 요청한다. 이는 PFO 제어 자체를 실행한다는 뜻은 아니다.

상한은 높이거나 낮출 수 있게 한다. 무조건 낮추는 정책은 ramp 대기열이 찼을 때 필요한 완화를 막는다. clipping은 좌표 범위 검사일 뿐 실현 가능성 보장이 아니다. `B_raw`, `B_requested`, `G_achieved`, `B_executed`를 분리해 기록한다.

### 보상: 실행된 interval의 실제 TTT

\[
r_t=-\frac{\Delta\mathrm{TTT}^{\mathrm{plant}}_t}{C},\qquad
\max_\theta\mathbb E\left[\sum_{t=0}^{T-1}r_t\right],\quad\gamma=1.
\]

`C`는 고정 양의 수치 스케일이다. 기존 차량 보존·소유권 기반 TTT의 plant 누적 차이를 사용한다. origin, boundary queue, ramp queue, transit 차량을 포함하고 합산 중복을 만들지 않는다. horizon 예측 TTT를 매 interval 겹쳐 더해서 보상으로 삼지 않는다.

유한 14,400초 episode와 남은 시간을 상태에 포함한다. 통제되지 않는 동일 warm-up은 상수 비용으로 보존한다. 할인율을 1보다 작게 하거나 reward clipping을 적용하면 원 총 TTT 목적과 달라질 수 있으므로 기본 설계에는 넣지 않는다. queue penalty, throughput reward, 계산시간 penalty도 자동 추가하지 않는다. queue 불균형·용량 근접 시간은 별도 acceptance 지표다. 추가 queue guard는 가능하지만 별도 설계 변경으로 비교한다.

14,400초 이후 잔여 차량 처리 비용은 기존 지표에 없으므로 몰래 보상에 추가하지 않는다. 종료 시 잔여 차량 수와 선택적 clearance TTT를 보조 지표로 보고한다.

## 5. 실행 검사와 fallback

1. 동일 상태·예측으로 PFO warm start를 만들고 현재 그룹 VSL/격자/변화 제한으로 매핑한다.
2. PFO 원 rollout을 평가해 \(B^0,J_H^0\)와 physical/control validity를 기록한다. 기준점이 무효면 현행 fail-closed 규칙을 유지한다.
3. incoming 가격 및 solver 기억을 복사하고 actor로 budget 한 개를 요청한다.
4. 그 budget으로 하위 solver를 최대 6회 수행한다. sensitivity, 가격 갱신, 자원 조정, 선탐색, VSL 양자화는 동일하게 유지한다.
5. 결합한 제어의 원 rollout으로 budget·물리·제어를 직접 검사한다. QP 성공이나 가격 부호를 실행 가능성 대신 쓰지 않는다.
6. 원 검사 통과 및 \(J_H^{RL}\le J_H^0\)이면 하위 제어를 채택한다. 아니면 유효한 PFO 기준 제어를 선택한다.
7. 선택된 경로의 가격 상태만 commit한다. PFO 선택이면 incoming 가격을 보존한다. 실행·관측 후 실제 interval TTT를 학습 transition에 기록한다.

PFO fallback의 달성 budget이 RL 요청 상한을 넘을 수 있다. **RL 요청 실패는 그대로 남기고, PFO 기준 budget으로의 fallback 전환을 별도 기록한다.** RL 요청을 달성량으로 바꿔 성공 처리하지 않는다. 현재 외부 wrapper는 budget 예외안을 그대로 실행하지 않고 PFO guard로 재선택하므로 이 동작을 유지한다.

이 검사는 현재 모델·H3에서의 실행 검사다. 모델 불확실성까지 포함한 recursive feasibility나 장기 TTT 우월성 보장은 아니다.

또한 단기 PFO TTT guard는 ‘지금 TTT를 조금 늘려 나중에 더 줄이는’ RL 행동을 거절할 수 있다. 첫 버전은 이 guard 아래에서 가능한 개선만 측정한다. guard 완화 또는 terminal value 도입은 별도 실험으로 분리한다.

## 6. 학습 방법과 비용

첫 알고리즘은 **TD3 기반의 2차원 연속 행동 정책**을 제안한다. 작은 MLP actor와 twin critic으로 시작한다. 기존 미수렴 solver를 관통하는 KKT 미분은 사용하지 않는다. TD3의 critic 미분은 실제 하위 최적해의 정확한 budget 민감도라는 뜻이 아니다. 최적 알고리즘이라는 주장 없이 연속 행동·replay 재사용이라는 이유로 선택한다.

- 환경 한 step은 ‘PFO + RL budget + 실제 하위 계산 + 원 검사 + 180초 plant 실행’ 전체다.
- replay에는 요청한 actor 행동과 그 행동 때문에 발생한 fallback 포함 실제 transition을 저장한다. fallback budget을 actor가 선택한 행동인 것처럼 바꾸지 않는다.
- 가격·warm start·buffer·시간·수요 generator 상태를 포함한 재현 가능한 checkpoint로 episode를 시작한다. 훈련에서만 exploration noise를 넣고, 평가에서는 policy weight와 전처리를 고정한다.
- 기존 로그는 imitation 초기화와 위험 상태 선별에 쓸 수 있다. 제한된 결정론적 budget 궤적만으로 충분한 action coverage가 있다고 보지 않는다. 서로 다른 A–D 하위를 동일 transition 환경인 것처럼 합치지 않는다.
- shadow 상태에서는 actor 출력을 기록하되 기존 제어로 plant를 진행한다. 이 단계로 action 분포·fallback 가능성을 확인하지만 actor의 폐루프 TTT 성능을 입증하지 않는다.
- 유효한 PFO가 없어서 종료된 episode는 남은 비용 0인 성공으로 학습하지 않는다. failure/truncation을 분리하고 실패 사례는 별도 검증 실패로 보존한다. 물리적으로 불가능한 상태와 단순 solver 실패를 구분한다.
- 짧은 chunk 학습의 끝은 실제 episode 종료가 아니므로 value bootstrap을 유지한다. 최종 평가는 전체 14,400초에서 한다.

학습이 싸지는 것은 아니다. 75개의 제어 interval을 가진 한 episode의 하위 비용은 대략 \(75\tau_{\mathrm{decision}}\)이다. 예를 들어 실제 측정값이 20초라면 한 episode만 약 25분이다. 이는 비용 예시이며 RL 버전 측정 결과가 아니다. 먼저 작은 수의 분기 rollout으로 budget이 실제 제어와 후속 TTT를 바꾸는지 확인한 뒤 학습 규모를 정한다. 한 interval의 후보 평가만 학습하면 contextual bandit/후보 ranking에 가까우며 장기 RL 성능이라고 부르지 않는다.

## 7. 구현 경계

별도 디렉터리 `work/sdmpc_rl_budget_<date>/`에서 다음 책임을 분리한다.

- `BudgetPolicy.propose(observation, reference)`: solver나 공통 가격을 수정하지 않고 요청 budget 반환.
- `evaluate_budget(request, seed, incoming_dual)`: side effect 없는 후보 평가; 후보별 가격 복사.
- `select_and_commit(...)`: 원 검사, PFO guard, 선택 경로 가격 commit.
- `BudgetEnv.step(action)`: 선택 제어의 plant 실행과 실제 보상·다음 상태 반환.

현재 `anchor_controller.decide`는 부모의 전체 후보 탐색을 실행한 뒤 첫 요청=PFO budget을 assert한다. 따라서 actor budget만 앞에 끼워 넣거나 후보마다 `decide`를 재호출해서는 안 된다. PFO 기준 평가, 후보 생성, solve, 선택/commit을 분리한 새 wrapper가 필요하다. 현재 구현은 비교 기준으로 보존한다.

중앙에서 수행하는 것은 PFO, actor 추론, 전체 궤적/민감도, resource QP, 실행 검사다. player 응답 구조는 그대로이며 단일 프로세스에서 시작한다. RL agent가 하나라는 이유로 분산 통신 성능 향상을 주장하지 않는다.

## 8. 검증 순서와 비교군

### 구현 동등성

actor 대신 기존 후보 생성기를 연결했을 때 같은 상태/seed에서 요청, 제어, TTT, 원 잔차, 가격 이력이 기존과 일치해야 한다. 0 residual action의 요청이 현재 mapped PFO witness와 정확히 같아야 한다. 후보 평가 순서·실패가 공통 가격을 오염시키지 않아야 한다.

### 학습 전 행동 영향 검사

각 수요 유형의 저장 상태에서 ±NP, ±NUF와 중심을 동일 조건으로 평가한다. 달성량·slack·active rate·제어 변화·fallback·후속 여러 interval TTT를 그린다. 상한이 전부 비활성이거나 guard가 모든 비중심 행동을 거절한다면, 먼저 행동 범위 또는 guard의 제한을 진단한다. RL을 학습했다고 budget 기여가 자동 생기지 않는다.

### 성능/계산 비교

| 비교군 | 목적 |
|---|---|
| 현행 leader 최대 3후보 + 동일 하위 | 실제 대체 대상 |
| PFO 달성 budget 한 개 + 동일 하위 | 후보를 줄인 효과와 학습 효과 분리 |
| RL budget 한 개 + 동일 하위 | 제안의 online 동작 |
| RL budget을 포함한 총 3후보 + 동일 하위 | 동일 후보 수에서 RL 제안 품질 확인; 별도 실험 |
| budget 없는 S-DMPC, PFO, no control | budget 필요성 및 교통 성능 기준 |

모든 비교에서 역사 물리, 시나리오, seed, 수요, H3, 제어 자유도, 하위 6회, 허용오차, CPU 조건을 맞춘다. 기존 결과는 코드·입력 hash가 일치할 때만 재사용한다. 후보 수가 다른 실험을 순수 RL 개선으로 해석하지 않는다.

기존 155, 170, 170-inc, 170-skew, 190, 190-skew, 190-inc, 220, 220-skew, 220-inc를 유지하고, 훈련/검증/테스트 seed와 수요 실현을 분리한다. incident 발생 시점·강도와 skew 방향의 미관측 조합도 별도로 평가한다. 정책 선택에 테스트 episode를 사용하지 않는다. 220의 기존 열화와 수렴 미인증은 숨기지 않는다.

기록할 결과:

- 전체 및 영역별 실제 TTT, baseline 대비 개선율, 종료 시 잔여 차량.
- ramp/boundary queue 용량 근접 지속시간과 불균형, 도시·고속도로 혼잡.
- 요청/실행 budget, 실제 달성량, 원 잔차, slack와 binding 여부.
- RL 채택률, PFO fallback률과 원인, 물리 실패와 알고리즘 실패.
- 하위 stationarity, 지역 QP 진단, 가격 잔차, 수렴 인증 여부.
- PFO/actor/하위/검사별 wall·CPU time, 전체 p50/p95/max, rollout·미분·지역 QP 횟수. 중첩 타이머 합산 금지.
- 학습 episode 수·총 하위 호출·총 CPU/wall time을 online 추론 비용과 별도 표기.

최종 도표는 학습곡선뿐 아니라 traffic state → budget 선택 → slack/binding → RM/VSL/green/offset → queue·TTT로 연결한다. RL 성능, solver 수렴, controller acceptance는 각각 독립 판정한다.

## 9. 관련 문헌과 채택 범위

1. Yoon et al. (2020), [Design of reinforcement learning for perimeter control using network transmission model based macroscopic traffic simulation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236655). 수요 정보를 RL 관측에 포함하는 교통공학 근거. 해당 논문은 경계 green split 기반 제어이며 우리처럼 S-DMPC budget을 선택하지 않는다. 혼합 throughput/delay reward도 그대로 가져오지 않는다.
2. Gros & Zanon, [Learning for MPC with Stability & Safety Guarantees](https://arxiv.org/abs/2012.07369). RL이 MPC의 파라미터를 조정하고 MPC가 제어 구조를 유지하는 방향의 근거. robust MPC에서 다루는 보장 조건을 현행 근사·이산 S-DMPC가 자동 만족하는 것은 아니다.
3. Wabersich & Zeilinger, [A predictive safety filter for learning-based control of constrained nonlinear dynamical systems](https://arxiv.org/abs/1812.05506). 학습 제안과 모델 기반 실행 검사를 분리하는 참고. 현재 rollout gate에는 그 논문의 안전 backup/불확실성 조건에 대한 증명이 없으므로 같은 이론적 보장을 주장하지 않는다.
4. Fujimoto, van Hoof & Meger (2018), [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477). TD3 원 논문. 연속 행동 actor–critic의 과대추정 완화와 지연 actor 갱신을 참고한다. 교통 문제에서 다른 RL보다 우월하다는 근거는 아니다.

권장 첫 구현 범위는 **PFO 기준 2차원 residual TD3 leader + 현행 inequality 하위 + 현행 PFO guard**다. 학습 시작 전 동일성 검사와 행동 영향 검사를 먼저 수행한다. 이번 문서는 그 설계만 제안하며, 새 RL 성능 수치는 아직 없다.
