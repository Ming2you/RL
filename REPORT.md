# 계층적 Stackelberg 교통제어 → RL Leader: 데이터 및 구현 계획

**작성일**: 2026-07-23
**대상**: 계층적 Stackelberg MPC(P-Stack)의 leader를 강화학습으로 대체하는 연구

---

## 1. 배경 — 제어기 4종

freeway + urban 결합망(w60u, T=14400s, warmup 1h / peak 1h / recovery 2.5h)에서 총통행시간(TTT) 최소화.

| 제어기 | 구조 |
|---|---|
| **NC** | No control (기준선) |
| **PFO** | Followers-only. 리더 없이 각 follower가 국소 혼잡에 자유 반응(per-ramp metering·green) |
| **P-Stack** (본선) | 계층. 리더가 (N_P, N_UF) budget + per-lever 한계가격 → follower가 배분 실행 |
| **P-CENT** | 중앙집중 grid MPC. 전 제어변수를 joint 최적화(상한 참조선) |

RL은 P-Stack 구조를 유지하되 **리더의 예산 탐색을 학습 정책으로 대체**하는 것을 목표로 함.

---

## 2. 최신 데이터 — held-out 일반화 (190 + stressor)

학습분포에 없던 **190-incident, 190-skew**에서 측정. 지표 = windowed TTT(warmup 5스텝 제외).

| 시나리오 | NC | PFO | P-Stack | P-CENT |
|---|---|---|---|---|
| **190-incident** | 8556 | 9230 (**−7.9%**) | 8386 (+2.0%) | 8016 (**+6.3%**) |
| **190-skew** | 6882 | 6299 (+8.5%) | 6379 (+7.3%) | 5757 (**+16.3%**) |

*(괄호 = NC 대비 개선율)*

**관찰**:
- 사고+고부하에서 **PFO는 NC보다 악화**(개입과잉 병리).
- **P-CENT가 모든 셀 최강**. 두 셀 모두 P-Stack이 P-CENT에 뒤짐 = 조정격차.
- **190-skew에서 P-Stack이 PFO한테도 짐**(6379 > 6299) — 리더 레이어가 오히려 손해.

### 2.1 peak/recovery × urban/freeway 분해 (NC 대비 Δ, 음수=개선)

**190-incident**:

| | freeway 완화 | urban 비용 | 순효과 |
|---|---|---|---|
| PFO | −3207 | +3881 | **+674 (악화)** |
| P-Stack | −1740 | +1570 | −170 |
| P-CENT | −2548 | +2008 | −540 |

**190-skew**:

| | freeway 완화 | urban 비용 | 순효과 |
|---|---|---|---|
| PFO | −1802 | +1219 | −583 |
| P-Stack | −720 | +216 | −503 |
| P-CENT | −1673 | +548 | −1125 |

모든 제어기가 **urban을 희생해 freeway를 완화**하는 교환. P-Stack은 셋 중 **가장 적게 개입**(과소).

---

## 3. 진단 — P-Stack이 지는 원인 (코드+데이터 검증)

### 3.1 리더 목적함수의 근시안
리더는 total TTT가 아니라 **~6구간(horizon 3 + value_depth 3) rollout surrogate**를 최소화한다. metering의 즉시비용(램프 대기)은 계산되나, downstream 이득(막은 freeway 붕괴)을 값매기는 far-terminal은 FAR_GATE로 사고/용량저하 시에만 켜진다. hold-back 페널티 가중치는 기본 0. → 짧은 지평 근시안.

### 3.2 skew도 게이트는 열리나 가격이 죽어있음
- skew도 emergent capacity-drop을 일으켜 far-gate가 열림(step 12~39).
- 게이트가 열려도 far는 **예산 목적함수엔** 들어가나(N_UF 5760→5254 소폭 조임) **가격 FD엔 안 붙는다**(`MFD_FAR_PRICE` off). → per-lever 한계가격이 전부 ~0(metering·green·VSL 모두 inert).

### 3.3 정적 budget 천장 (ceiling sweep, skew)
고정 N_UF로 sweep한 windowed TTT:

| N_UF | 6000 | **5254** | 4500 | 3500 | 2500 | 1500 | 800 |
|---|---|---|---|---|---|---|---|
| TTT | 6877 | **6325** | 7800 | 13860 | 23964 | 32683 | 39096 |

- **N_UF≈5254가 스윗스팟(6325)**, 그보다 조이면 **램프 큐 폭발**(비단조).
- **정적 스칼라 budget 천장 ≈ 6325 (≈P-Stack)** — 위상 적응(오라클)도 못 넘음(최선 7061).
- **PFO(6299)·P-CENT(5757)는 정적 스칼라 budget으로 도달 불가.**

### 3.4 가격 = 분배(distribution), budget = 총량(total) — realizability probe
peak 상태에서 per-ramp metering 가격을 주입한 정적 probe:
- 양수 가격 → 해당 램프 metering↑, **같은 merge의 sister 램프가 정확히 그만큼↓** (총량 보존, zero-sum swap).
- merge 간에는 선택적(서쪽 가격이 동쪽 무영향), merge 내에서는 재분배만.
- trust_frac(0.2 vs 0.6) 무관, 반응은 이산적.

**결론**: 가격은 "어느 램프에" (분배), budget은 "얼마나" (총량). 둘은 대체재가 아님.

### 3.5 P-CENT 우위의 정체
P-CENT의 skew peak 총유입은 **~4720 veh/h**(follower 스윗스팟 5254보다 낮음). 정적 budget으로 4720을 강제하면 폭발(3.3의 4500=7800)하는데, P-CENT는 4720에서 5757. **차이는 총량이 아니라 per-ramp 분배** — 같은 낮은 총량을 P-CENT는 큐 안 터지게 분배, follower는 blind하게 분배해 폭발. 즉 **낮은 총량 + 올바른 분배 + 시간 조율**이 P-CENT의 핵심.

---

## 4. RL 구현 계획

### 4.1 아키텍처 (유지)
- **Leader (RL)**: 물리량 압축 관측(13-dim, 시나리오 라벨 없음=scenario-agnostic) → **budget (N_P, N_UF) + per-lever 한계가격**(metering×4, green×5, VSL×16) 출력.
- **Followers (optimizer, 유지)**: budget+가격을 받아 배분·안전 실행(feasibility 보장). RL은 리더 탐색만 대체.
- **reward** = −ΔTTT (dense), γ=0.99.

### 4.2 설계를 조인 핵심 발견
1. **스칼라 budget만으론 천장이 낮다**(≈P-Stack). 헤드룸은 **가격(분배) + 상태-반응형 budget**에 있음.
2. **가격 채널은 살릴 수 있으나 손튜닝으론 inert**(≈0). RL이 **손튜닝보다 나은 가격을 배우는 것**이 핵심 가치.
3. 리더 계산이 무거운 이유는 **매 스텝 ~49 후보 rollout**. 학습된 정책/critic은 이걸 **forward pass로 대체** → 리더 O(후보×rollout) → O(1). (follower 비용은 그대로.)

### 4.3 학습 접근 — online RL은 부적합, teacher가 필요
- **online SAC 파일럿 실패**: BC가 teacher(P-Stack) 대비 −13~−20%(closed-loop 드리프트), SAC 1500스텝은 발산(optimizer의 2~2.7배). 원인 = untrained critic이 BC warm-start 파괴 + env.step ~8.5s라 SAC의 10k~100k step은 며칠. **비싼 env엔 online RL 구조적 부적합.**
- **P-Stack을 teacher로 한 BC는 천장이 P-Stack**(근시안 clone). P-CENT를 직접 clone하는 것도 불가(액션공간·응답함수 불일치 → "이상한 가격").

### 4.4 채택 방향 — inverse optimization으로 teacher 생성
**P-CENT를 목표 거동으로 두고, 그 거동을 재현하는 (budget + 전-lever 가격)을 역산** → 이게 곧 "분산 follower를 중앙최적으로 유도하는 **조정 가격(coordinating prices)**"(price-externality 이론의 정공법).

1. P-CENT 런에서 per-ramp metering·(VSL) 목표 궤적 추출.
2. 각 스텝: follower가 목표를 내도록 **budget(총량) + 가격(분배)을 반복 역산**(폐루프, per-lever 근찾기 + 커플링 반복).
3. 그 (상태→budget·가격) 궤적을 **offline RL(IQL/AWAC) teacher**로 사용.

### 4.5 미해결/위험
- **realizability**: 분산 follower가 P-CENT를 어떤 가격으로도 재현 못 할 수 있음(구조적 한계). 그럼 "가장 가까운 근사"에 멈추고, 그 격차가 **분산화의 대가를 정량화**하는 결과가 됨(실패해도 값짐).
- **P-CENT green 미기록**: run_log에 urban green 목표 없음 → green을 inverse 목표로 쓰려면 P-CENT 재실행 필요(VSL은 skew서 거의 free).
- **env 비용**: follower solve ~8.5s/step(13-player Nash). offline 데이터생성도 무거움 — 병렬/캐싱 필요.

---

## 5. 다음 단계
1. **전-lever inverse solve**(budget+가격 동시, P-CENT 목표) 폐루프 구현 → (a) P-CENT 실현가능성 = 스칼라·가격 천장, (b) teacher 데이터 동시 확보.
2. teacher 확보 시 **offline RL(IQL/AWAC)** 학습 → held-out(190-inc/skew)에서 P-Stack·P-CENT 대비 평가.
3. realizability가 낮으면 → RL 접고 **P-Stack 근시안 직접 수정**(hold-back 가중치/지평/게이트) 또는 **조정격차 정직 보고**로 전환.

---

*본 레포트의 수치는 모두 실측(run_log) 기반. 원 시뮬레이션 코드는 별도 레포(Numerical-Sim).*
