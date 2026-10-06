# TTT와 TTD를 이용한 Actor Critic budget leader 설계

2026-10-06. 설계안이며 새 RL 학습·폐루프 성능 검증은 아직 수행하지 않았다. 현재 진행 중인 lower 6회·upper 10후보 실험과 기존 실행 코드는 변경하지 않는다.

**연결할 실제 코드:** 같은 저장소 `main`의 [sdmpc_ttd/](../sdmpc_ttd/README.md).
현재 TTD S-DMPC와 역사 물리 모델을 포함하고, 초기화·실행·하위 fixed-budget 진입점을 설명한다.
기존 `rl_leader/env.py`의 P-Stack/Wu follower와 구분한다. 새 TD3 환경과 학습은 아직 남은 작업이다.

한 개의 TD3 Actor가 180초마다 두 budget 상한을 조정한다. 기존 player별 S-DMPC가 그 budget 아래에서 RM, VSL, green, offset을 계산한다. Critic은 실행된 구간의 `−TTT + α·TTD`와 다음 상태로부터 이후 구간까지의 누적 성과를 학습한다. 학습 후 온라인에서는 Actor를 한 번 호출하고 하위 문제를 한 번 푸는 구조를 기본안으로 한다.

## PPT에서 계승할 부분과 정리할 부분

참고: 사용자가 제공한 `학술.학위논문 요지발표.261008.민건규.pptx` (PPT 원본은 이 저장소에 포함하지 않음). 아래 페이지는 슬라이드에 인쇄된 번호이며 파일의 슬라이드 순서는 각각 1 크다. 텍스트와 RL 도식의 실제 삽입 이미지도 확인했다.

| PPT 위치 | 계승 | 이번 설계의 구체화 |
|---|---|---|
| p.18–19, 38–41 | 학습하는 leader + 유지되는 S-DMPC follower | 기존 후보 탐색을 Actor의 단일 budget 제안으로 교체 |
| p.39 | 이전 budget에 정규화된 2차원 증분 적용 | 초기 PFO budget, 범위 투영, fallback 시 budget 기억 갱신을 명시 |
| p.39–40 | 실행 구간 TTT로 reward 계산 | 실행 구간의 `−TTT + α·TTD`로 변경 |
| p.40 | twin critic, target actor, replay, delayed actor update | TD3 유지. Critic은 양수·음수 모두 출력 |
| p.41 | S-DMPC로 실제 제어 결정 | 최대 6회 유한 반복 응답으로 명시. 수렴 보장이나 정확한 best response로 쓰지 않음 |
| p.36–37 | budget 상한과 dual 방향 정보 | 문구는 band가 남아 있지만 표시 식은 한쪽 상한. 현재 실행과 동일한 두 inequality 상한을 사용 |

PPT p.41의 Actor 식은 `Q(s,a,π(s))`처럼 세 인자가 섞여 있다. `L_actor = −E[Q_1(s,π(s))]`로 통일한다. Critic도 아래 정의대로 현재 `(s_t,a_t)`와 다음 `(s_{t+1},a')`를 구분한다. Actor 범위는 `[-1,1]^2`다. 도식의 plant에는 budget 자체가 물리 제어로 들어가는 것이 아니라 검사를 통과한 첫 구간의 `u_t`가 적용된다.

PPT 전체 framework와 follower 목적 설명도 TTT만 표시된 곳은 `TTT−αTTD`로 맞춰야 한다. 현재 6회 실행은 own와 externality를 1차 근사한 prox-linear 지역 문제다. 이를 원 논문 형태의 완전한 비선형 own NLP로 표기하지 않는다.

## 목적과 보상

실제로 실행한 한 control interval의 TTT와 TTD를 각각 `T_t [veh·h]`, `D_t [veh·km]`라 두면

\[
\ell_t=T_t-\alpha D_t,\qquad r_t=-\ell_t/s_{\rm ref}.
\]

`α [h/km]`는 TTD에 주는 보상의 크기이고, `s_ref > 0 [veh·h]`는 신경망 학습을 위한 수치 스케일이다. budget 가격 λ와 학습률은 별개다. TTT와 TTD를 단위 보정 없이 그대로 빼지 않는다.

기존 비교와 직접 연결하는 첫 모드에서는 이미 동결된 시나리오별 무제어 기준을 사용한다.

\[
\alpha=\frac{T^{NC}_{ref}}{D^{NC}_{ref}},\quad
s_{ref}=\frac{T^{NC}_{ref}}{75},\quad
r_t=75\left(-\frac{T_t}{T^{NC}_{ref}}+\frac{D_t}{D^{NC}_{ref}}\right).
\]

따라서 두 지표의 기준 대비 상대 변화에 같은 비중을 준다. 75는 900초 warm-up 뒤 14,400초까지의 제어 판단 횟수다. 기준은 전체 14,400초 NC 값이며, 이를 75로 나눈 것은 양수 reward scale일 뿐 warm-up 제외 NC 평균이라는 뜻은 아니다.

시나리오별 기존 α 숫자는 로컬 실험 설정에서 읽는다. 이 설계 문서에는 구간 데이터나 시나리오별 실행 결과를 포함하지 않는다.

이는 기존 seed 42 실험의 계약을 재현하는 모드다. 새로운 일반화 실험에서는 **훈련용 NC 집합만으로 공통 α와 scale을 정한 뒤 동결**하는 것을 권한다. 예: `α_train = ΣT_NC_train / ΣD_NC_train`. 이 변경은 별도 설정으로 기록하고, 비교하는 모든 TTD 목적 controller에 동일하게 적용한다. 평가 seed의 미래 실현 NC 결과로 매 episode 정규화 상수를 새로 맞추지 않는다. 여러 α를 학습시키는 모드라면 α와 scale도 관측에 넣는다.

TTT는 기존 차량·queue·transit storage 소유권을 사용한다. TTD는 현재 `completed_link_distance_production_v1`, 즉 실제 제한된 방출 차량수 × 고정 링크 길이의 합이다. command나 예정 도착량이 아니며, 미시 궤적에서 모든 부분 주행거리를 적분한 값과도 다르다. 실제 plant 계측과 예측 계측을 분리하고 각 차량 공간의 시간·거리를 중복 합산하지 않는다.

보상은 **180초 실행 결과**에서 얻는다. H3=540초 예측 목적을 reward에 넣으면 인접 의사결정 간 예측 시간이 겹치고 실행하지 않은 결과까지 학습하게 된다. H3 목적은 후보 검사와 PFO 비교에만 쓴다. 첫 버전에 terminal cost, 가격 수입, 큐 penalty, 계산시간 penalty를 추가하지 않는다.

훈련은 900초 이후 75개 decision의 유한 episode로 정의하고 남은 시간을 관측에 넣는다. 기본 `γ=1`이면

\[
\sum_t r_t=-\frac{\sum_t T_t-\alpha\sum_t D_t}{s_{ref}}.
\]

위 합은 900–14,400초 제어구간이다. 전체구간 목적은 `J_full = J_warmup − s_ref Σr_t`로 복원한다. 같은 초기 상태·수요의 warm-up 비용은 정책에 무관한 상수다. 따라서 현재 finite-horizon 지표와 정확히 연결된다. `γ<1`은 먼 미래의 혼잡 비용을 할인하는 별도 목적이므로 숨겨서 적용하지 않는다. 시간 제한까지의 총량 평가와 무한시간 안정성을 동일시하지 않는다. 최종 잔여 차량·완료 차량·외부 대기열을 함께 보고하고 필요하면 고정 drain 구간 평가를 별도로 추가한다.

## 상태와 행동

Actor 관측은 교통 상태와 controller 기억을 함께 포함한다.

| 관측 묶음 | 구체 항목 |
|---|---|
| Freeway | segment별 밀도·속도, ramp queue, origin queue, buffer, off-ramp storage |
| Urban | movement별 queue, 링크 transit storage, receiving 여유, phase·cycle 내 시간, 지연 도착 buffer |
| 수요와 시간 | 기존 controller와 동일한 예측, 방향별 demand, 현재 관측 가능한 incident, 남은 episode 시간 |
| 이전 결정 | 실제 실행한 RM/VSL/green/offset, commit된 budget, incoming budget 가격, warm-start 기억 |
| 이번 기준 | mapped PFO 제어의 달성 budget·예측 목적, action mapping 범위, α와 reward scale |

전체 simulator checkpoint는 수요 generator와 지연 도착까지 보존한다. 축약된 관측만 사용하면 Markov성이 자동 성립하지 않는다. 첫 구현에서는 고정 순서의 상세 vector를 쓰고, 축약 시에는 부분 관측 문제로 표시해 history 또는 recurrent actor를 별도로 검증한다. 미래 실제 demand·incident를 추가로 알려주지 않는다.

PPT에 맞춘 행동 정의는 다음과 같다.

\[
a_t=\pi_\theta(o_t)\in[-1,1]^2,\quad
B_t^{raw}=B_{t-1}^{mem}+D_a a_t,\quad
D_a=\operatorname{diag}(50\;\mathrm{veh},1000\;\mathrm{veh/h}).
\]

첫 interval의 `B_mem`은 그 상태에서 제한·그룹화를 반영한 PFO rollout의 달성 budget으로 초기화한다. 이후에는 이전에 **선택·commit된 budget**에서 증분을 적용한다. 이는 9/29 설계의 매번 PFO 중심 residual 행동과 다르며 별도 action-mapping 버전으로 기록한다.

첫 비교의 action 범위는 기존 탐색 envelope를 유지한다. 현재 상태의 mapped PFO 달성값을 `B_t^0`라 하면

\[
\mathcal B_t=[B_{P,t}^0-50,B_{P,t}^0+50]
\times\left([B_{UF,t}^0-1000,B_{UF,t}^0+1000]\cap[0,\sum_r m_r^{max}]\right),
\quad B_t^{req}=\operatorname{proj}_{\mathcal B_t}(B_t^{raw}).
\]

현재 총 metering capacity는 6,000 veh/h이며 구현에서는 config에서 계산한다. 이 투영은 **설계에 추가한 anti-drift 범위 검사**다. budget 실현 가능성을 보장하지 않는다. `a=0`도 현재 envelope 밖의 이전 budget을 투영하면 요청값이 움직일 수 있으므로 `B_raw`, `B_req`, clipping flag를 모두 저장한다.

실제 action 도달 집합은 `proj_Bt(B_mem + D_a[-1,1]^2)`이다. 기존 envelope 전체에 도달한다는 뜻은 아니다. PFO 중심이 급변하면 특정 좌표가 포화되어 Actor가 그 좌표를 움직이지 못할 수 있다. clipping·중복 action·PFO 중심 도달 가능성을 pilot에서 필수 검사하고, 심하면 PFO 기준 residual 또는 state-dependent bounded mapping을 별도 실험으로 비교한다. 첫 비교에서 이 효과를 학습 성능으로 혼동하지 않도록 동일 memory/mapping의 `a=0` 대조군을 둔다.

물리 budget은 현재 코드와 동일하다.

\[
G_P(X,U)=\sum_{h=0}^{2}(S_{in,h}-S_{out,h})\le B_P\quad[\mathrm{veh}],
\]
\[
G_{UF}(U)=\sum_r m_r\le B_{UF}\quad[\mathrm{veh/h}].
\]

`G_P`는 540초 horizon net-service다. 도시 stock 자체나 첫 180초 Δstock으로 바꾸지 않는다. `G_UF`는 현재 상수 move block의 metering command 합이다. 실제 ramp 통과 차량수와 구분한다. signed `B_P`를 임의로 0 이상으로 자르지 않는다. 이전 budget을 기억한다는 것은 지난 horizon의 남은 차량 자원을 이월한다는 뜻이 아니다. 매번 새로운 540초 예측창에 같은 물리 정의의 상한을 지정한다.

## S-DMPC 응답과 실행 규칙

\[
\widehat U_t=\mathcal S_6(X_t,B_t^{req},w_t,\lambda_t;\alpha),\qquad
J_H=\sum_{i=1}^{9}\left(T_{i,H}-\alpha D_{i,H}\right).
\]

기존 9-player, freeway 4개 그룹, H3, 하위 최대 6회, 제어 자유도, trust/proximal 항, 원 제약과 검사 허용오차를 유지한다. TTD 하위는 지역 gradient, 외부효과, 전체 목적 및 PFO 목적 guard에 같은 α를 사용한다. 외부효과는 `Σ_{j≠i}∂(TTT_j−αTTD_j)/∂u_i`이며 budget Jacobian과 budget 가격을 대체하지 않는다.

현재 실행 버전은 한 Jacobi 반복 안에서는 같은 기준점과 가격을 사용하고, 응답 후 `k→k+1`에서 하위 가격을 갱신한다. 과거의 interval 내 가격 고정 버전과 혼동하지 않는다. RL 적용 시 이 정책도 고정해 비교한다. Critic gradient `∇_a Q`는 하위 Lagrange multiplier나 정확한 최적 응답 미분이 아니다.

온라인 절차:

1. 현재 state에서 기존 PFO를 계산하고 mapped PFO 제어를 원 모델로 평가한다.
2. Actor가 action 한 개를 내고 budget mapping을 적용한다.
3. 현재 incoming 가격·seed를 복사하여 해당 요청의 하위 S-DMPC를 최대 6회 계산한다.
4. 결합·VSL 실행값 변환·자원 조정 후 원 모델로 budget, physical, control 제약을 검사한다. 최종 응답에서 실행 가능한 후보를 못 얻으면 기존 예외 계층과 같이 **해당 단일 요청의 보존된 이산 반복점**에서 archive recovery를 수행한다. 이 과정의 rollout 비용도 별도 기록한다.
5. 최종 응답 또는 archive 안이 budget/physical/control 유효하며 같은 H3 composite 목적이 PFO 이하이면 선택한다. 아니면 유효한 mapped PFO로 fallback한다. archive의 최소 잔차 점이라도 원 budget 검사를 통과하지 못하면 외부 gate에서 거절한다.
6. 일반 하위안이면 요청 budget과 그 경로의 최종 가격을 commit한다. archive recovery면 요청 budget을 commit하되 **incoming 가격을 보존**하고, 다른 최종 반복점의 승수·stationarity를 회수된 점의 인증으로 복사하지 않는다. PFO면 `B_mem=B_t^0`로 명시적으로 re-anchor하고 incoming 가격을 보존한다.
7. 첫 180초만 plant에 실행하고 실제 `T_t,D_t`, 다음 상태를 기록한다.

fallback이 RL 요청 상한을 만족하지 않을 수 있다. 따라서 **원 요청 실패 + PFO budget으로 운영 전환**으로 기록한다. 요청을 실현값으로 덮어서 feasible로 바꾸지 않는다. 순수 범위 clipping, nonlinear budget feasibility, local solver 종료, 최적화 수렴, controller acceptance는 각각 독립 field다.

PFO도 physical/control 검사를 통과하지 못하면 기존 fail-closed로 plant 실행 전에 중단한다. 첫 학습은 유효한 PFO backup을 확인한 상태에서 시작한다. 양쪽 실패는 정상 episode 종료의 0 tail reward로 처리하지 않고 학습/수집을 중단하여 원인을 진단한다. 이 설계는 실패에 보상을 주는 흡수 상태를 임의로 만들지 않는다. 이런 상태까지 계속 학습하려면 별도 failure cost/제약 RL 설계와 명시적 목표 변경이 필요하다.

현재 H3 PFO guard는 지금 약간 손해 보고 더 먼 미래에 이득을 얻는 행동을 거절할 수 있다. 첫 실험에서는 그 guard 안에서의 학습 효과만 평가한다. Critic value를 guard나 terminal cost에 넣는 실험은 다음 단계로 분리한다.

## TD3 학습

한 환경 step은 위의 PFO·하위 계산·검사·180초 plant 실행 전체다. replay의 action은 Actor가 요청한 정규화 action이며, clipping과 fallback은 환경 전이의 일부다. 선택된 PFO action을 Actor 요청인 것처럼 바꿔 저장하지 않는다.

\[
\mathcal D\ni(o_t,a_t,r_t,o_{t+1},d_t),\qquad
Q_\phi(o,a)\approx\mathbb E\left[\sum_{j=t}^{T-1}\gamma^{j-t}r_j\mid o_t=o,a_t=a\right].
\]

상세 audit에는 requested/mapped/committed budget, actual control, 달성량, 원 잔차, 실행 경로, 이전/다음 가격, seed, source hash, termination/truncation을 추가한다. `o_t`에는 결과를 보기 전 정보만 넣고 이번 step의 결과는 `o_{t+1}`에 넣는다.

\[
a'=\operatorname{clip}\left(\pi_{\bar\theta}(o_{t+1})+
\operatorname{clip}(\epsilon,-c,c),-1,1\right),
\]
\[
y_t=r_t+\gamma(1-d_t)\min_{j=1,2}Q_{\bar\phi_j}(o_{t+1},a'),\quad
L_{Q_j}=\mathbb E[(Q_{\phi_j}(o_t,a_t)-y_t)^2],
\]
\[
L_\pi=-\mathbb E[Q_{\phi_1}(o_t,\pi_\theta(o_t))].
\]

target network를 soft update하고 Actor는 Critic보다 덜 자주 갱신한다. TD3의 twin critic 및 delayed update는 [Fujimoto et al. (2018)](https://proceedings.mlr.press/v80/fujimoto18a.html)의 방법을 따른다. lower solver를 역전파하거나 KKT 미분을 사용하지 않는다. 물리 예측의 기존 직접 민감도 계산과 neural critic의 gradient는 역할이 다르다.

| 첫 구현용 설정안 | 값과 의미 |
|---|---|
| Actor / 각 Critic | 각 128×128 MLP. Actor tanh 출력 2개, Critic 선형 scalar 출력 |
| Discount | γ=1, finite horizon·time-to-go 포함 |
| 학습률 | Actor/Critic 각 3e−4, 검증 집합에서만 조정 |
| Mini-batch / replay | 128 / 최대 50,000 transition |
| Actor 갱신 | Critic 2회마다 1회 |
| Target soft update | τ=0.005 |
| Target action noise | 표준편차 0.2, clip ±0.5, 정규화 action 단위 |
| Exploration | 훈련만 표준편차 0.1부터, [-1,1] clipping. 평가에서는 noise=0 |
| 초기 Actor | 마지막 층을 0으로 두어 초기 증분 0부터 시작 |

이 수치는 시작 설정이며 이 문제에서 최적이라고 검증한 값이 아니다. 충분한 서로 다른 transition이 쌓이기 전 업데이트하지 않고, pilot에서는 transition당 gradient step을 최대 1회로 제한해 작은 replay 과적합을 확인한다. 모델 학습은 평가 run 중 수행하지 않는다.

14,400초 실제 종료는 `d=1`이고 tail value를 0으로 둔다. 훈련 편의를 위한 중간 chunk 끝은 실제 종료가 아니므로 다음 상태가 있을 때 bootstrap을 유지한다. solver failure는 일반 time-limit truncation과 별도로 다룬다.

## 기존 코드와 구현 경계

이 절의 `work/ttd_upper10_20261006` 등 S-DMPC 작업 경로는 원 Numerical Simulation 작업공간의 근거이며 이 RL 저장소에 runtime으로 포함되지 않는다. 이 커밋에는 설계 문서만 포함된다. 아래 신규 모듈 목록도 구현 완료 목록이 아니다.

- [기존 설계](rl_budget_leader_design_20260929.md)는 9/29의 순수 TTT·PFO residual TD3 제안이며 학습 완료 코드가 아니다.
- 기존 `rl_leader/env.py`는 구형 P-Stack/Wu follower 경로다. 새 S-DMPC 환경과 성능을 혼용하지 않는다.
- 해당 `response_dqn.py`의 `interval_negative_ttt`, reward<0 조건과 cost head는 새 보상에 맞지 않는다. `interval_negative_ttt_plus_alpha_ttd_v1` 등 명시적 schema와 부호 제한 없는 Q 출력이 필요하다.
- 현재 `work/ttd_upper10_20261006/sdmpc_objective.py`의 objective accounting·gradient를 별도 동결 snapshot으로 계승한다.
- `anchor_controller.decide()`는 기존 후보 탐색을 내부에서 실행하므로 Actor를 앞에 추가하는 것만으로 탐색이 없어지지 않는다. `PFO reference / propose / solve one request / validate / select and commit`을 분리한 wrapper가 필요하다.

제안 작업 경로는 `work/sdmpc_rl_ttd_<date>/`, 출력은 `outputs/sdmpc_rl_ttd_<date>/attempt_<id>/`다. 아래 모듈은 **구현 예정**이다.

| 모듈 | 책임 |
|---|---|
| `objective_contract.py` | α·scale·time window·TTD metric version 동결 |
| `budget_mapping.py` | 이전 budget 증분, 초기화, clipping, commit/fallback |
| `sdmpc_response.py` | 1요청 평가, 원 제약 및 PFO composite guard |
| `env.py` | state restore, plant 계측, reward, replay transition |
| `td3.py` | Actor/twin critic/target/replay, 부호 제한 없는 값 함수 |
| `train.py`, `evaluate.py` | train/validation/test seed 분리, 체크포인트, 결과 비교 |

중앙에서는 PFO, Actor, 공통 교통 예측·민감도, 공유 자원 조정, 원 모델 검사가 수행된다. player 구조를 유지하더라도 첫 구현은 단일 프로세스 모사다. 통신/병렬 성능을 측정하기 전에는 개선을 주장하지 않는다.

## 왜 budget RL에 의미가 있는가

정확한 하위 최적화와 같은 H3 목적을 가정하면 상한을 높일수록 feasible set이 커지므로 최적값은 악화하지 않는다. 따라서 짧은 H3 점수만 모방하는 Actor는 느슨한 상한만 배우거나 유한 반복 solver 특성에 적응할 수 있다.

설계의 검증 대상은 **180초마다 선택한 budget의 장기 폐루프 성과**다. 예를 들어 이번 interval의 ramp 대기 증가가 이후 freeway 혼잡과 도시 spillback을 얼마나 바꾸는지 Critic이 평가한다. 이는 성능 가설이며, RL이 기존 탐색보다 반드시 좋다는 뜻은 아니다. 특히 budget이 대부분 비활성이거나 PFO fallback이 대부분이면 Actor action의 실제 영향이 작다.

## 검증과 첫 학습 규모

1. **환경 계약 검사:** 같은 저장 상태에서 기존 요청을 wrapper로 평가하면 control·TTT·TTD·원 잔차·가격 이력이 일치해야 한다. clipping/fallback에서 requested와 committed action을 구별한다. 합산 reward와 실제 objective를 검사한다.
2. **작은 행동 영향 pilot:** 5개 시나리오의 혼잡 전·발생·회복 상태에서 중심/±NP/±NUF를 평가한다. 기준을 통과한 일부 상태만 여러 interval로 분기하여 delayed effect, binding, fallback을 확인한다. 먼저 실제 한 transition 비용을 측정하고 학습량을 정한다.
3. **초기 데이터와 학습:** 동일 objective·같은 하위 버전의 기존 실행 transition을 warm-start 자료로 쓴다. 후보 예측 로그는 실행 transition이 아니며 Q 학습용 실현 outcome으로 바꾸지 않는다. action mapping이 다른 로그는 정확한 원 요청 역변환과 상태 복원이 가능한 경우에만 쓰며, 덮어쓴 clipped action은 임의 추정하지 않는다. 적은 고정 정책 데이터만으로 offline TD3를 완성했다고 판단하지 않는다.
4. **짧은 폐루프 pilot:** 유효한 backup을 확보한 저장 상태에서 여러 seed로 학습·평가하고, budget과 실제 control에 충분한 변화가 있는지 확인한다. 이 결과로 전체 학습 시간 상한을 산정한다.
5. **전체 평가:** 155, 170, 170-skew, 170-incident, 190의 14,400초. 기존 seed 42는 직접 비교용이며 이미 반복 확인한 사례라 독립 test로 부르지 않는다. 별도 train/validation/test seed와 미관측 incident·skew 조합을 사용하고, 여러 학습 seed의 분포를 보고한다.

공정한 첫 비교군은 동일 TTD 목적·하위 6회인 (A) 현재 upper 3후보, (B) upper 10후보, (C) 현재 PFO budget 단일 후보, (D) RL 단일 후보, (E) D와 동일 memory/mapping/guard에서 `a=0`인 단일 후보다. **E→D는 학습 정책의 기여, C→E는 budget memory/mapping 효과, A/B→C는 후보 수 감소 효과**를 구분한다. PFO, no control, budget-off를 외부 교통 기준으로 함께 제시한다.

학습 비용은 `환경 transition 수 × (PFO + 하위 + 검사 + plant) 시간`으로 추산한다. Actor 추론이 빨라도 follower 비용은 남는다. 첫 후보가 좋은 로그의 마지막 상태를 복제한 것만으로 전체 episode를 대체하지 않는다. 실제 측정 없이 수만 step 학습이 저렴하다거나 online 계산이 10배 빨라진다고 주장하지 않는다.

## 보고 지표와 수락 기준

- 실제 TTT, TTD, composite J를 각각 보고한다. Urban/Freeway, peak/off-peak, warm-up 포함/제외를 명시한다. PPT p.45의 warm-up 제외 문구와 p.54의 기존 전체구간 수치는 구분이 필요하다.
- α=T_NC/D_NC이면 NC의 전체 J는 0이다. `(J_NC−J)/J_NC` 개선율을 쓰지 않는다. 순수 TTT 개선율, TTD 변화율, `−J/T_NC` 등의 정규화 차이를 사용한다.
- 제공된 수요 총량, 들어온 차량, 완료 차량, 종료 시 잔여 차량, 외부 queue, 보존 잔차를 함께 검사한다. TTD 보상을 위해 차량 경로를 임의 변경하거나 external queue를 회계에서 빼지 않는다.
- requested/committed/achieved budget, slack, binding 비율, projection/fallback 비율, 선택 경로와 가격을 기록한다.
- 지역 solver 상태, stationarity, lever/λ 변화, primal feasibility, 실행 완료, 최적화 수렴, controller acceptance를 분리한다. 작은 iterate 변화와 학습 reward 정체는 수렴 증명이 아니다.
- PFO/Actor/하위/검사/plant별 wall·CPU time, 전체 p50/p95/max, rollout 및 derivative 호출을 기록한다. offline 학습 시간은 online 판단 시간과 별도 표기한다.
- 기존 acceptance의 TTT 개선 기본 8%, boundary queue 균형·control validation·test·review 조건을 RL reward 개선으로 대체하지 않는다. 미통과 결과는 보존한다.

## 검증과 공개 범위

보상 회계는 기존 로컬 실행 기록으로 점검했다. 구간 데이터, 검산 결과 파일, 원본 PPT와 새로운 학습 checkpoint는 이 게시물에 포함하지 않는다. 새 RL 학습이나 simulation 성능 검증을 수행했다는 뜻은 아니다.

구현 시에는 위 reward 합산식과 warm-up 복원식, 부호가 자유로운 보상, 지역·전체 목적 합, 요청/실행 action의 구분을 검증해야 한다. 현재 lower6/upper10 실행과 기존 결과는 별도로 보존한다.

## 문헌과 적용 범위

- [Fujimoto, van Hoof & Meger (2018), Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html): twin critic과 delayed actor update의 출처. 우리 traffic budget 문제에서의 우월성을 보장하지 않는다.
- [Gros & Zanon, Learning for MPC with Stability & Safety Guarantees](https://arxiv.org/abs/2012.07369): 학습이 MPC 파라미터를 바꾸는 연결 방식의 참고. 논문의 robust MPC 보장 조건을 현재 유한 반복·이산 제어 S-DMPC가 충족했다고 주장하지 않는다.
- [Yoon et al. (2020), Design of reinforcement learning for perimeter control using network transmission model based macroscopic traffic simulation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0236655): density 외 demand 정보를 관측에 포함하는 교통 제어 근거. 제어 행동과 보상은 본 설계와 다르다.

논문 표현은 **Actor–critic budget coordination with finite-iteration S-DMPC followers**가 적절하다. 계층 구조를 보존하지만 정확한 Stackelberg equilibrium이나 전역 최적성을 계산했다고 표현하지 않는다.
