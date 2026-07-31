# HANDOFF — 다른 컴퓨터에서 이어서 작업하기

**작성 2026-07-31.** 이 문서 하나로 맥락 없이 이어받을 수 있게 썼다. 숫자는 전부 실측(run_log/트레이스) 기반이며, 미검증 추측은 "미규명"으로 명시했다.

---

## 0. 30초 요약

계층적 Stackelberg MPC(P-Stack)의 **leader를 RL로 대체**하는 연구. 리더는 저차원 신호(budget + 가격)만 내고, follower(최적화기)가 배분·안전을 실행한다.

**현재 도달점 — offline RL(IQL)이 held-out 2셀 모두에서 P-Stack을 이겼다.** 다만 P-CENT엔 못 미친다.

| windowed TTT (낮을수록 좋음) | NC | PFO | P-Stack | **IQL(RL)** | P-CENT |
|---|---|---|---|---|---|
| **190-skew** | 6882 | 6299 | 6379 | **6290.9** | 5757 |
| **190-incident** | 8556 | 9230 | 8386 | **8260.5** | 8016 |

- vs P-Stack: **−88.0 / −125.8** (둘 다 승, ~1.5%)
- vs PFO: 승(incident −969.5, skew −8.1로 사실상 동률)
- vs P-CENT: **패**(+533.6 / +244.2)
- 이 결과는 **4,194 샘플**로 학습한 `checkpoints/actor_iql.pt`의 것. 이후 데이터가 **27,706개로 늘었으나 재학습·재평가는 미완**(§6).

---

## 1. 문제 설정

- **plant**: freeway(2링크 × 8세그) + urban(5교차로) 결합망. 제어주기 180s, T=14400s.
- **시나리오 타임라인**: warmup [0,900) → **peak [900,5220)** → **recovery [5220,14400]**.
- **지표 = windowed TTT** = `cumulative_total_ttt[마지막] − cumulative_total_ttt[step==4]` (warmup 5스텝 제외). 버퍼 포함, 공간 8세그 아님.
- **held-out(테스트 전용)**: `sweet_190_skew15_w60`, `sweet_190_incident_w60`. 학습 시 stressor 있으면 수요 ≤1.80으로 제한 → 190+stressor는 학습분포에 없음.

### 컨트롤러 4종
| 이름 | 구조 |
|---|---|
| NC | 무제어 |
| PFO | followers-only(리더 없음), 국소 반응 |
| **P-Stack** | 계층. 리더가 (N_P, N_UF) budget + per-lever 가격 → follower 실행 |
| P-CENT | 중앙집중 grid MPC(상한 참조선) |

---

## 2. 확립된 사실 (데이터로 검증됨)

### 2.1 P-Stack 리더는 근시안이다 (핵심 진단)
리더는 total TTT가 아니라 **~6구간(horizon 3 + value_depth 3) rollout surrogate**를 최소화한다.
- metering의 **즉시 비용**(램프 대기 veh·h)은 항상 목적함수에 들어간다.
- metering의 **downstream 이득**(막아낸 freeway 붕괴)을 값매기는 far MFD-tail 항은 `FAR_GATE`로 사고/용량저하 때만 켜진다.
- hold-back 선형 페널티(`w_ramp_queue`, `w_boundary_in`)는 기본 **가중치 0**.
- → 짧은 지평에선 "지금 차 풀면 대기 줄어"만 보여 **헐렁한 N_UF**(≈5254, 램프 총용량 6000의 87%)를 고른다.

### 2.2 skew에서 P-Stack이 PFO한테도 진다
190-skew: P-Stack 6379 > PFO 6299. 리더 레이어가 오히려 손해.
- **정정**: 초기 가설 "skew는 capdrop 없어 far게이트가 안 열린다"는 **틀렸다**. 로그상 `[FAR_GATE m3] step12 ON → step39 OFF`로 실제로 열린다(emergent capdrop, `capacity_drop_active=1.0`).
- 게이트가 열리면 far가 **예산 목적함수엔** 들어가 N_UF를 조금 조인다(5760→5254). 그러나 **가격 FD엔 안 붙는다**(`price_far_enabled`는 `MFD_FAR_PRICE=1`로만 켜지는데 b13 env에 없음) → per-lever 한계가격이 ~0으로 죽어 있다.

### 2.3 가격 = 분배, budget = 총량 (구조적 분리)
정적 probe(peak 상태에서 per-ramp 가격 주입, `code/price_inverse_probe.py`):
- 양수 가격 → 해당 램프 metering↑, **같은 merge의 sister 램프가 정확히 그만큼↓**(step12 ±373, step20 ±186.5). **총량 보존 = zero-sum swap.**
- merge 간엔 선택적(서쪽 가격이 동쪽 무영향), merge 내엔 재분배만.
- `trust_frac` 0.2 vs 0.6 **동일** → trust는 병목 아님. 반응은 이산적(±500이면 이미 포화).
- **결론**: 가격은 "어디에", budget은 "얼마나". 둘은 대체재가 아니다.

### 2.4 정적 스칼라 budget의 천장 = 6325 (≈P-Stack)
`code/ceiling_sweep.py` 실측(190-skew, 고정 N_UF):

| N_UF | 6000 | **5254** | 4500 | 3500 | 2500 | 1500 | 800 |
|---|---|---|---|---|---|---|---|
| TTT | 6877 | **6325** | 7800 | 13860 | 23964 | 32683 | 39096 |

- N_UF≈5254가 스윗스팟, **그보다 조이면 램프 큐 폭발**(비단조).
- 위상별 오라클(peak엔 tight, 그 외 loose)도 최선 7061로 **더 나쁨** → 단순 위상 스케줄링은 답이 아니다.
- **PFO(6299)·P-CENT(5757)는 정적 스칼라 budget으로 도달 불가.**

### 2.5 P-CENT 우위의 정체
P-CENT의 skew peak 총유입 ≈ **4720 veh/h**(follower 스윗스팟 5254보다 **낮다**). 정적으로 4720을 강제하면 폭발(§2.4의 4500=7800)하는데 P-CENT는 5757. → **차이는 총량이 아니라 per-ramp 분배 + 시간 조율.**

### 2.6 ★진짜 병목은 데이터 기근이었다
BC(−13~−20%)·SAC(발산)의 원인을 알고리즘으로만 봤으나, 그 앞단에 **총 375샘플**(5셀×75스텝)이 있었다. 어떤 RL도 375샘플로는 안 된다. 병렬 수집으로 **3시간에 375 → 27,187샘플(72배)** 확보하자 IQL이 바로 P-Stack을 이겼다.

---

## 3. 실패한 접근 (반복 금지)

| 접근 | 결과 | 원인 |
|---|---|---|
| **online SAC** | optimizer의 2~2.7배, NC보다도 나쁨(−80~−177%) | env 8.5s/step(혼잡 시 스텝당 수십 분)이라 10k~100k step 불가 + untrained critic이 BC warm-start 파괴 |
| **BC(teacher=P-Stack)** | teacher 대비 −13~−20% | closed-loop 드리프트 + **천장이 teacher**(근시안 clone) |
| **P-CENT를 직접 clone** | 시도 안 함(설계상 기각) | 액션공간 불일치(P-CENT는 raw control, RL은 budget/가격) + 응답함수 불일치 → "이상한 가격" 학습 |
| **closed-loop P-CENT 추종 distill** | **26시간 CPU에도 단일 롤아웃 미완** | 낮은 총량 추종 시 혼잡 누적 → follower Nash solve가 스텝당 수십 분으로 폭증 |
| **MFD_FAR_PRICE=1** | TTT −26.5(−0.4%)뿐 | green 가격은 살아나나(A 0.006→0.266, 서쪽 차등) metering/예산은 거의 불변 → 레버리지 작음 |

**교훈**: closed-loop env를 루프에 넣는 접근은 전부 실패한다. **env 접촉 0인 offline만 현실적.**

---

## 4. 성공한 파이프라인 (이걸 이어가면 됨)

### 4.1 병렬 데이터 수집 (`code/collect_parallel.py`)
14워커 × 도메인 랜덤화. **3원칙을 반드시 유지할 것**:
1. **에피소드마다 증분 저장** — distill이 26h 돌고 산출물 0이던 사고 방지.
2. **에피소드 wall-clock 가드**(기본 1800s) — congested 폭증 차단. abort돼도 전이는 저장되고, time-limit 절단은 bootstrapping상 올바른 처리.
3. **행동 다양성** sweet 35% / uniform 35% / **reactive 30%** — teacher 궤적만 모으면 advantage가 없어 offline RL이 BC 천장에 갇힌다.

### 4.2 offline RL (`code/iql.py`)
IQL. V=expectile 회귀(τ=0.7), Q=TD with V(s'), 정책=advantage-weighted regression(β=3.0). **환경 접촉 0** → env 비용·발산 문제를 전부 우회. 실측 advantage 스프레드 −1.59~+0.30(비퇴화) 확인.

### 4.3 평가 (`code/eval_guarded.py`)
- **wall-clock 가드**(셀당 3600s). 절단 시 "비교불가"로 명시하고 숫자를 조용히 내지 않는다.
- per-step **트레이스 저장**(`--trace-dir`) → 메커니즘 분석용.
- budget 스케줄 진단(N_UF std, corr(N_UF, rho_max)) 자동 출력.

---

## 5. ★미규명 — 왜 이겼는지 모른다 (다음 작업 1순위)

예측했던 "혼잡하면 조인다"를 **데이터가 반박**했다.

```
190-skew      corr(N_UF, rho_max) = +0.246
190-incident  corr(N_UF, rho_max) = +0.442
```
양의 상관 = 혼잡할수록 오히려 **푸는** 쪽. 단 N_UF·rho가 peak에서 함께 오르므로 **수요 프로파일에 교란된 지표**라 이것만으로 단정도 불가.
참고: N_UF 평균 5142/5229(P-Stack 5254보다 소폭 조임), std 248/262.

→ **`code/analyze_mechanism.py`**가 이걸 규명하도록 준비돼 있다. 구간(peak/recovery) × 서브시스템(urban/freeway)으로 TTT를 분해해 **이득이 어디서 왔는지** 귀속한다. 실행하려면 트레이스가 필요하다(§6-2).

> 프로젝트 규율: **메커니즘은 측정 후에만 주장한다.** 이 세션에서 최소 3번(폭발 추정, far게이트 가설, 조임 가설) 추측이 데이터로 반박됐다.

---

## 6. 즉시 이어서 할 일 (미완, 우선순위 순)

### 6-1. 27k 데이터로 3시드 재학습 + 평가 ← **중단된 상태**
- 스냅샷 `data/rl_dataset/`(27,706 샘플, 423 에피소드)로 seed 0/1/2 학습 40,000 step.
- **이전 시도는 step 1에서 프로세스가 죽어 미완**(`actor_iql_s*.pt`는 쓸모없으므로 레포에 넣지 않았다).
- 목적: 현재 마진(−88/−126, 약 1.5%)이 **시드 노이즈인지 실재하는지** 확인. 단일 시드·단일 롤아웃이라 오차범위가 없다.

```bash
# 학습(환경 접촉 0, 빠름)
python code/iql.py --data "data/rl_dataset/w*.npz" --steps 40000 --seed 0 --out checkpoints/actor_iql_s0.pt
python code/iql.py --data "data/rl_dataset/w*.npz" --steps 40000 --seed 1 --out checkpoints/actor_iql_s1.pt
python code/iql.py --data "data/rl_dataset/w*.npz" --steps 40000 --seed 2 --out checkpoints/actor_iql_s2.pt

# 평가(셀당 ~40분, 반드시 순차 — 병렬은 CPU 과다구독으로 시간가드 오작동)
python code/eval_guarded.py checkpoints/actor_iql_s0.pt --max-sec 3600 --trace-dir traces --tag s0
python code/eval_guarded.py checkpoints/actor_iql_s1.pt --max-sec 3600 --trace-dir traces --tag s1
python code/eval_guarded.py checkpoints/actor_iql_s2.pt --max-sec 3600 --trace-dir traces --tag s2
```

### 6-2. 메커니즘 귀속 (§5)
```bash
python code/analyze_mechanism.py --trace-dir traces --tag s0
```
출력: 구간×서브시스템 Δ(IQL − P-Stack) + 구간별 N_UF·램프큐 비교. **이득이 특정 구간·서브시스템에 몰려 있으면 그것이 메커니즘.**

### 6-3. 가격을 액션에 추가 (P-CENT 격차 공략)
현재 액션은 budget 2차원뿐. P-CENT 격차(+534/+244)는 §2.3·2.5에 따라 **per-ramp 공간 타겟팅** 부재가 원인으로 보인다(미검증). `env.py`의 action_dim을 늘려 per-ramp metering 가격(4차원)을 추가하고, follower의 `metering_marginal_price`에 주입하면 된다(주입 인터페이스는 §7.3).

### 6-4. 데이터 추가 수집
필요하면 `code/collect_parallel.py`를 워커 수만큼 띄운다(§7.4).

---

## 7. 새 컴퓨터 세팅

### 7.1 저장소 구성
```
src/            시뮬레이터 + 최종 컨트롤러(plant·coupling·P-Stack/PFO/P-CENT·config)
work/           러너 run_claude_style_five_controller.py
code/           RL 구현 (env·iql·collect_parallel·eval_guarded·analyze_mechanism …)
data/rl_dataset/  offline RL 학습 데이터 27,706 샘플(14 npz)
data/holdout/     held-out baseline run_log 8개(평가 기준선)
data/pcent_teacher/  P-CENT 궤적(inverse-optimization 목표)
checkpoints/    actor_iql.pt(★우승), actor_bc.pt(참고)
```

### 7.2 의존성
```bash
pip install -r requirements.txt   # numpy 필수, torch(RL), pandas·matplotlib(분석)
```
- **주의(Windows)**: torch를 기본 경로에 설치하면 긴 경로(260자) 에러가 난다. 이 프로젝트는 `pip install --target C:/torchlib torch --index-url https://download.pytorch.org/whl/cpu`로 우회했고, `code/nets.py`·`iql.py`가 `sys.path`에 `C:/torchlib`을 추가한다. **다른 환경이면 그 부트스트랩 줄을 지우면 된다.**
- `code/*.py`는 `ROOT`를 `parents[1]`로 잡아 `src/`를 import한다. 레포 루트에서 실행할 것.

### 7.3 컨트롤러 재현 (baseline 재생성이 필요할 때)
공통 물리 env:
```
WARMUP_NC_STEPS=5;FW_BUFFER=8;TERM_ZG=1;VFREE=115;RHO_CRIT=31.5;TAU_H=0.0056111;NU_BASE=22.5;KAPPA=10;MERGE_DELTA=0.9
```
| 컨트롤러 | ID | 추가 env |
|---|---|---|
| NC | `NO-CONTROL` | — |
| PFO | `WU-FAITHFUL-FOLLOWER` | `BASELINE_BOX=1` |
| **P-Stack(b13)** | `P-STACK-WU-FAITHFUL-ALLPRICE-JOINT` | `BOX_WALK=1;BOX_WALK_VG=1;VSL_BOX=15;METER_BOX=300;NP_PD_ITER=4;NP_BIAS=1;CROSS_OFF=1;FAR_STATE_AWARE=1;SEG13=1;FAR_REAL_V=1;FAR_GATE=3;BASELINE_BOX=1;SUP_PFO=1;SUP_GATE=fargate` |
| P-CENT | `P-CENT` | `CENT_REFRESH_SEC=180;FAR_REAL_V=1` |

follower 가격 주입 인터페이스(§6-3용):
```python
follower.metering_marginal_price = {ramp: price}   # dict, 4 ramps
follower.metering_marginal_price_ref = {ramp: 현재 운영점}
follower.metering_marginal_price_trust_frac = 0.20
```

### 7.4 병렬 수집 실행 예
```bash
for i in $(seq 0 13); do
  python code/collect_parallel.py --seed $((100+i)) --episodes 40 \
    --max-ep-sec 1800 --out data/rl_dataset/w$(printf "%02d" $i)_new.npz &
done
```
수집 속도 실측 ≈ **2,900 샘플/시간**(14워커, 20코어). 에피소드 ≈ 75스텝, 22분.

---

## 8. 함정 (반드시 알고 시작할 것)

1. **env가 비싸다.** follower solve ~8.5s/step(정상), **혼잡하면 스텝당 수십 분**으로 폭증. 롤아웃 하나가 26시간+ 걸린 사례가 있다. 어떤 스크립트든 **wall-clock 가드 필수.**
2. **긴 런을 서브에이전트에 넣지 말 것.** 에이전트가 못 기다리고 종료하며 자식 프로세스까지 죽인다. 메인 세션 background로 돌릴 것.
3. **증분 저장 필수.** 끝에만 저장하면 26시간이 0이 된다.
4. **프로세스 정리는 PID 지정으로.** 이 세션에서 blanket kill(`Stop-Process python*`)로 사용자의 다른 배치를 날린 사고가 있었다.
5. **`wu_b3_meter_fd3_*_base`는 가격이 아니다** — 전 램프 공유 TTT 스텐실 중심값(동일해 보이는 게 정상). 진짜 per-ramp 가격은 `wu_b3_meter_price_{ramp}`.
6. **평가 단위 정합성은 검증됨** — env 롤아웃 N_UF=6000이 6877로 NC 6882와 일치. run_log windowed TTT와 직접 비교 가능.
7. **결과 비교는 항상 windowed TTT**(warmup 5스텝 제외)로. 원시 total과 섞지 말 것.

---

## 9. 논문과의 관계

- 본 논문(TRB) 주제는 **조정격차 정량화**(PFO ↔ P-Stack ↔ P-CENT)와 계산비용(P-Stack O(n) vs P-CENT O(n³)).
- RL은 **후속/확장**. 차별점 후보 3가지:
  1. **RL-for-coordination, not RL-for-control** — RL이 raw control이 아니라 optimizer들에게 내리는 **가격·예산**을 배운다. follower가 optimizer로 남아 feasibility·안전 보장.
  2. **learned coordination의 도달 가능 상한과 물리적 이유를 해부** — §2.3~2.5(가격=분배, 정적 budget 천장 6325, P-CENT 우위의 정체)가 이에 해당. 대부분의 RL 논문에 없는 분석.
  3. **계산**: 리더 탐색(매 스텝 ~49 후보 rollout) → forward pass 1회. 성능이 동급만 나와도 계산 이득은 실측 가능.
- 관련 레포: 시뮬레이션 본체는 `Ming2you/Numerical-Sim`(이 레포의 `src/`는 그 사본).
