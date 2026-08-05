# HANDOFF — 다른 컴퓨터에서 이어서 작업하기

**최초 2026-07-31, 최종 갱신 2026-08-05.** 이 문서 하나로 맥락 없이 이어받을 수 있게 썼다. 숫자는 전부 실측(run_log/트레이스) 기반이며, 미검증 추측은 "미규명"으로 명시했다.

> **이어받는 사람은 §0 → §13(다음 작업) → §8(함정) 순으로 읽으면 된다.**
> §10~§12는 2026-08-01~05에 수행한 실험 기록(대부분 negative result)이고, 반복하지 않기 위한 근거다.

---

## 0. 30초 요약

계층적 Stackelberg MPC(P-Stack)의 **leader를 RL로 대체**하는 연구. 리더는 저차원 신호(budget)만 내고, follower(최적화기)가 배분·안전을 실행한다.

**현재 최고 구성 = budget 2차원 액션 + potential-based 보상 shaping(Φ=hinge 근사, w=600).**
액션 확장은 4구성 모두 기각됐다(§12.9). 유일하게 통한 개입은 **보상 shaping**이다(§12.12–12.13).

| windowed TTT (낮을수록 좋음) | NC | PFO | P-Stack | **현재 최고 3시드** | P-CENT |
|---|---|---|---|---|---|
| **190-skew** | 6882 | 6299 | 6379 | **6234.3 ± 63.6** | 5757 |
| **190-incident** | 8556 | 9230 | 8386 | **8072.2 ± 14.7** | 8016 |

- **vs P-Stack: 양 셀 3/3 시드 승** (skew −144.6, inc −314.0). skew 최악 시드 6277.3 < P-Stack 6379.
- **vs P-CENT: skew +476.9 / inc +55.9.** incident는 사실상 따라잡았고(격차 179.7 → 55.9, **69% 축소**), **skew는 +473에서 정체**.
- 체크포인트: `checkpoints/actor_hg600_s{0,1,2}.pt` (데이터 `data/rl_dataset/w*.npz` 27.7k, `iql.py --shape hinge --shape-w 600`).
- 참고 이전 기준선(shaping 없음, Phase 2): skew 6324.8 ± 116.0 / inc 8196.0 ± 76.8.

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
정적 probe(peak 상태에서 per-ramp 가격 주입, `rl_leader/price_inverse_probe.py`):
- 양수 가격 → 해당 램프 metering↑, **같은 merge의 sister 램프가 정확히 그만큼↓**(step12 ±373, step20 ±186.5). **총량 보존 = zero-sum swap.**
- merge 간엔 선택적(서쪽 가격이 동쪽 무영향), merge 내엔 재분배만.
- `trust_frac` 0.2 vs 0.6 **동일** → trust는 병목 아님. 반응은 이산적(±500이면 이미 포화).
- **결론**: 가격은 "어디에", budget은 "얼마나". 둘은 대체재가 아니다.

### 2.4 정적 스칼라 budget의 천장 = 6325 (≈P-Stack)
`rl_leader/ceiling_sweep.py` 실측(190-skew, 고정 N_UF):

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

### 4.1 병렬 데이터 수집 (`rl_leader/collect_parallel.py`)
14워커 × 도메인 랜덤화. **3원칙을 반드시 유지할 것**:
1. **에피소드마다 증분 저장** — distill이 26h 돌고 산출물 0이던 사고 방지.
2. **에피소드 wall-clock 가드**(기본 1800s) — congested 폭증 차단. abort돼도 전이는 저장되고, time-limit 절단은 bootstrapping상 올바른 처리.
3. **행동 다양성** sweet 35% / uniform 35% / **reactive 30%** — teacher 궤적만 모으면 advantage가 없어 offline RL이 BC 천장에 갇힌다.

### 4.2 offline RL (`rl_leader/iql.py`)
IQL. V=expectile 회귀(τ=0.7), Q=TD with V(s'), 정책=advantage-weighted regression(β=3.0). **환경 접촉 0** → env 비용·발산 문제를 전부 우회. 실측 advantage 스프레드 −1.59~+0.30(비퇴화) 확인.

### 4.3 평가 (`rl_leader/eval_guarded.py`)
- **wall-clock 가드**(셀당 3600s). 절단 시 "비교불가"로 명시하고 숫자를 조용히 내지 않는다.
- per-step **트레이스 저장**(`--trace-dir`) → 메커니즘 분석용.
- budget 스케줄 진단(N_UF std, corr(N_UF, rho_max)) 자동 출력.

---

## 5. ★규명됨(2026-07-31) — 이득의 메커니즘 = 시간축 방류 평탄화

구간(peak/recovery)×서브시스템 분해를 **4정책(구 ckpt + 3시드) × 2셀 = 8롤아웃**에서 교차검증했다. 8/8에서 일관된 서명:

1. **이득은 전부 recovery 구간에서 온다. peak에선 오히려 진다.** (inc: peak +314~+432 / recovery −547~−592, skew: recovery −167~−263)
2. **recovery에서 IQL은 N_UF를 −450~−700 더 조인다**(5065~5272 vs P-Stack 5734~5765). P-Stack leader는 recovery에 큐 방류 모드로 열고, 그 방류가 회복 중인 freeway를 재혼잡시킨다. IQL은 절제를 유지한다.
3. **조였는데 램프큐는 오히려 절반**(recovery 72~90 vs 143~145) — freeway를 흐르게 유지하니 처리량이 높아져 큐가 더 빨리 빠진다. §2.1 leader 근시안("지금 풀면 대기 줄어")의 정확한 반례.
4. **incident peak에선 반대로 +770~+940 덜 조인다**(P-Stack은 사고 때 4669까지 과잉 조임 → 램프큐 438). peak freeway를 내주고 urban recovery에서 −1162~−1263을 회수한다.
5. 요약: **위기 때 과잉 조임, 회복 때 왕창 방류라는 P-Stack의 진폭을 평탄화하는 시간축 스케줄링**이 메커니즘이다. 공간 분배가 아니다.

구 지표 corr(N_UF, rho_max)는 시드마다 부호까지 다르다(skew: −0.38/+0.69/+0.54) → **교란 의심이 맞았고, 지표로 무효.** 구간 분해가 올바른 렌즈다.

skew의 시드 분산(§0)도 여기서 설명된다: recovery 이득(−167~−263)은 시드 불변인데 **peak 행동이 시드마다 달라**(N_UF Δ −445/+96/+234) peak에서 얼마를 되돌려주는지가 갈린다. incident는 peak 행동도 일관돼(전부 덜 조임) 마진이 안정적이다.

> 프로젝트 규율: **메커니즘은 측정 후에만 주장한다.** 이전 세션에서 최소 3번(폭발 추정, far게이트 가설, 조임 가설) 추측이 데이터로 반박됐다. 위 서명은 8/8 롤아웃 측정 기반이다.

---

## 6. 즉시 이어서 할 일 (미완, 우선순위 순)

### 6-1. ✅완료(2026-07-31) — 27k 3시드 재학습 + 평가
- 결과는 §0 표. **incident 승은 실재(3/3, −190±77), skew 승은 기각(범위가 0을 걸침, 동률).**
- 재현 게이트도 통과: 구 체크포인트 6308.3/8270.2 (원본 6290.9/8260.5 대비 +0.28%/+0.12%, torch 빌드 차이의 부동소수점 표류로 추정. P-Stack 우위 유지로 비교가능성 성립).
- 이전 "step 1에서 죽음"의 원인은 알고리즘이 아니라 환경/경로였다(§10).
- 학습: `rl_leader/iql.py`(시드당 ~12.5분, 3시드 병렬 ~13분). 평가: `rl_leader/run_seeds_eval.ps1`(6롤아웃 병렬 ~38분, §11). 트레이스는 `traces/`에 있음(repro/s0/s1/s2 × skew/inc, 8개).

### 6-2. ✅완료(2026-07-31) — 메커니즘 귀속
- 결과는 §5. 재실행: `python rl_leader/analyze_mechanism.py --trace-dir traces --tag s0` (tag: repro/s0/s1/s2)

### 6-3. ❌기각(2026-08-01~04) — 가격을 액션에 추가
당시 가설: P-CENT 격차는 per-ramp 공간 타겟팅 부재 때문. **4구성 전부 실패**했다 — 상세 §12.4~12.9. 요약은 §13-A.

### 6-4. ❌기각(2026-08-04) — 데이터 추가 수집 / on-policy 라운드
`SCALING=SATURATED`(50% 데이터 = 100% 데이터)로 양적 증량이 무의미함이 확인됐고, on-policy 라운드도 null이었다(§12.15).

### 6-5. ✅완료(2026-08-04~05) — 보상 shaping
**유일하게 성공한 개입.** §12.12~12.14. 현재 최고 구성이 여기서 나왔다.

---

## 7. 새 컴퓨터 세팅

### 7.1 저장소 구성
```
src/            시뮬레이터 + 최종 컨트롤러(plant·coupling·P-Stack/PFO/P-CENT·config)
work/           러너 run_claude_style_five_controller.py
rl_leader/      RL 구현 (env·iql·collect_parallel·eval_guarded·analyze_mechanism …)
data/rl_dataset/  offline RL 학습 데이터 27,706 샘플(14 npz)
data/holdout/     held-out baseline run_log 8개(평가 기준선)
data/pcent_teacher/  P-CENT 궤적(inverse-optimization 목표)
checkpoints/    actor_iql.pt(★우승), actor_bc.pt(참고)
```

### 7.2 의존성
```bash
pip install -r requirements.txt   # numpy 필수, torch(RL), pandas·matplotlib(분석)
```
- torch는 CPU 휠로 충분: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- **주의(Windows)**: 환경에 따라 긴 경로(260자) 에러가 날 수 있다. 그때만 `pip install --target C:/torchlib torch ...`로 우회하면 되고, 스크립트들은 **`C:/torchlib`이 존재할 때만** `sys.path`에 추가하도록 조건부로 돼 있어 양쪽 환경에서 그대로 돈다(수정 불필요).
- `rl_leader/*.py`는 `ROOT`를 `parents[1]`로 잡아 `src/`를 import한다. 레포 루트에서 실행할 것.

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
  python rl_leader/collect_parallel.py --seed $((100+i)) --episodes 40 \
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

---

## 10. Phase 0 — 실행 환경 복구 (2026-07-31, 신규 머신)

인수인계 커밋(`176d22e`) 상태의 레포는 **어떤 RL 스크립트도 실행되지 않았다.** 원인 3가지와 조치:

| # | 증상 | 원인 | 조치 |
|---|---|---|---|
| 1 | 모든 스크립트 `ModuleNotFoundError: torch` | torch 미설치(`C:/torchlib` 부재) | CPU 휠 설치(torch 2.13.0+cpu). **긴 경로 에러 없이 기본 경로에 설치됨** |
| 2 | `ModuleNotFoundError: rl_leader` | 커밋 시 `rl_leader/` → `code/`로 옮겼으나 import 22곳 미수정 | **`code/` → `rl_leader/` 되돌림**(디렉터리명이 import와 일치). `code`는 stdlib 모듈명이라 패키지로 쓰면 안 됨 |
| 3 | 베이스라인 0개 / "P-Stack run_log 없음" | 코드가 `outputs/_wang3/{ho_x}/{CTRL}/run_log.csv`를 읽는데 레포는 `data/holdout/ho_x_{tag}.csv`로 배포 | `data/holdout/`·`data/pcent_teacher/`·`data/rl_dataset/`·`checkpoints/`로 경로 교정 |

- §6-1의 "step 1에서 죽었다"는 **알고리즘 문제가 아니라 1·2번이었을 가능성이 높다**(당시 로그가 없어 단정 불가). 교정 후 step 1 통과 확인.
- **baseline 정체성 검증(2026-07-31)**: `ho_pstack_*`는 논문 표(table2)의 Proposed와 **동일 설정**이다 — 둘 다 b13 full sampling, `leader_candidate_count` 프로파일 동일(mean 51.4~51.6, min 49, max 75), 코드도 본체 HEAD와 동일(vendored src는 +PRICE-HINGE 기본 OFF뿐). 커밋 메시지의 "Proposed=CAND25"는 **table3 계산비용의 채택 세팅**("adopted lossless setting", ~26s/step)을 가리키며, 성능 표(table2)는 CAND49(~36s/step) 런이다. 즉 RL이 이긴 P-Stack = 논문 최종본이 맞다. 숫자가 논문 표와 달라 보이는 건 셀이 다르기 때문(논문: 155/170/170-skew/170-inc/190, RL held-out: 190-skew/190-inc는 논문 표에 없음).
- `C:/torchlib` 부트스트랩은 삭제하지 않고 **디렉터리가 존재할 때만 적용**되도록 조건부화 → 원 머신·신규 머신 양쪽에서 무수정 동작.
- **`rl_leader/*.sh` 4개는 원 머신 전용이라 여기서 안 돈다** — `cd /c/Users/alsrj/...`와 codex 런타임 python 경로가 하드코딩돼 있고, 한글 경로(`찐찐막`)에서 Git Bash가 깨진다. Phase 1 이후는 PowerShell로 직접 구동할 것.

### Phase 0 검증 결과 (전부 통과)
```
env.py 5스텝 롤아웃      obs_dim=13 action_dim=2, follower solve 정상
baselines(skew/inc)      6881.9/6299.3/6378.9/5757.4, 8555.9/9230.0/8386.2/8016.3  ← README 표와 일치
iql.load_data            27,706 전이 obs=13 act=2
load_pcent(skew/inc)     80스텝
iql.py --steps 200       step 1 통과, adv −0.593/−0.109/+0.079(비퇴화)
체크포인트 로드           actor_iql.pt / actor_bc.pt 모두 정상
```
추가 확인: P-Stack CSV의 `step_urban_ttt + step_freeway_ttt`를 t≥900에서 합하면 6378.4/8385.7로 cumulative 기반 windowed TTT(6378.9/8386.2)와 일치 → **§6-2 구간×서브시스템 분해의 전제가 성립한다.**

---

## 11. 평가 병렬화 실측 (2026-07-31) — §6-1 "반드시 순차"의 조건 해제

eval 롤아웃 프로세스의 CPU를 6초 간격으로 샘플링한 결과(21샘플):

```
proc_cores_used = 1.00 (모든 샘플 99.0~100.2%), 워킹셋 198MB 고정
IQL 3시드 학습(스레드 1 고정)을 동시에 붙여도 eval은 여전히 1.00 코어 — 간섭 없음
```

- **롤아웃은 완전한 단일 스레드다.** (스레드 20+개가 떠 있지만 전부 대기)
- §6-1의 "반드시 순차 — CPU 과다구독" 경고는 **수집기 14개가 동시에 돌던 당시 상황 한정**이다. 유휴 머신에선 (물리코어 − 여유분)개까지 동시 롤아웃이 안전하다. 이 머신(10물리/20논리/64GB)은 6개 동시(3시드×2셀)가 여유롭게 들어간다 → **~4시간 → ~40분.**
- 단, **수집(collect_parallel) 워커와 겹쳐 돌리지 말 것** — 그 순간 §6-1 경고가 되살아난다(시간가드가 벽시계 기준이라 경합 시 멀쩡한 롤아웃이 절단됨).
- 실행 도구: `rl_leader/run_seeds_eval.ps1`(셀×시드 병렬, `OMP/MKL_NUM_THREADS=1` 고정). `eval_guarded.py`에 `--cells skew|inc` 옵션을 추가해 셀 단위 분할이 가능하다.
- IQL 40k step 학습 실측: 시드당 ~12.5분(CPU, 2000step 37.4s 외삽). 3시드 병렬로 ~13분에 완료. **GPU(RTX A6000 있음)는 이 워크로드에 불필요** — 병목은 학습이 아니라 롤아웃(단일 스레드 follower solve)이다.

---

## 12. Phase 4 — per-ramp 가격 액션 (2026-07-31 구현·발사)

### 12.1 구현 (`price_action=True`일 때만 활성, 기본 OFF는 비트 보존)
| 파일 | 변경 |
|---|---|
| `env.py` | action_dim 2→**6**(budget 2 + per-ramp 가격 4, `[-1,1]×PRICE_MAG=500`). obs 13→**23**(공간 신호 10 추가: per-ramp 큐 4 + per-link ρ_max 2 + per-ramp 도착예보 4). N_P 하한 **−1000** 개방. 주입은 probe 패턴 그대로(`metering_marginal_price`/`_ref`=직전 release/`_trust_frac`=0.20, solve 후 `finally`로 None 원복) |
| `collect_parallel.py` | 가격 행동정책 zero 30%/iid 35%/**contrast 35%**(merge 내 반대부호 쌍 = 가격의 유일 작용축). budget 모드와 독립 추출. 출력 `data/rl_dataset_p6/`. **원자적 증분 저장**(tmp→`os.replace`) |
| `eval_guarded.py` | 체크포인트 `act_dim>=6`이면 price env 자동 추론. 가격·실현 release 트레이스 |

**관측 확장은 §6-3 원안에 없던 추가 판단**: 기존 13차원은 전부 집계량이라 서-동 비대칭이 정책에 보이지 않는다. 공간 타겟팅을 배우려면 공간이 보여야 한다.

### 12.2 발사 전 게이트 (`rl_leader/smoke_price_env.py`, 전부 통과)
```
[1] zero-price 중립성 : 8스텝 max|Δ| step_ttt=0.0  N_UF=0.0  obs[:13]=6e-08
    → 주입이 dynamics를 바꾸지 않는다 = 기존 27.7k와 동일 MDP의 확장
[2] 가격 응답(probe 재현): R_D_W +500 → R_D_W −124.3, sister R_F_W +124.3,
    총합 Δ=0.0, 반대 merge(E) 무영향 → zero-sum swap + 선택성 (§2.3 일치)
[3] 6차원 랜덤 + 음수 N_P(−500) 매핑 정확
backcompat: 기존 체크포인트 5개 전부 obs13/act2 호환·1스텝 정상, iql.py는 23/6 무수정 학습
```
**가격 부호 규약: 양수 가격 = 해당 램프 release 감소(조임).**

### 12.5 ★Phase 4 결론 — 가격은 무용한 게 아니라 **총량 채널이 봉인돼 있었다**

3시드 결과: skew 6401.8±136.6 / inc 8246.5±46.9 → budget-only(6324.8/8196.0) 대비 **악화**, P-CENT 격차도 +644/+230으로 **벌어짐**. 판정 BAD.

**진단 3단계(전부 실측):**
1. **데이터-스케일링**: frac 0.5 = frac 1.0 (skew gain +50.2 < 임계 58, inc −39.6) → `SCALING=SATURATED`. 데이터 문제 아님.
2. **zero-price ablation**: 학습된 가격을 0으로 강제하면 skew 6301.3 / inc 8153.3으로 **개선**(+100.5/+93.2). 그리고 이 값이 budget-only와 std 내 동등 → **관측 확장(13→23)·절단 데이터(완주 0%)·N_P 음수 개방은 전부 무해**했고 열화는 100% 가격 탓.
3. **근본 원인(`probe_price_level.py`)**: `follower.metering_price_split=True`(기본값)이고 RL은 항상 `leader≠None ∧ N_UF>0`이라 `price_split` 조건이 매 스텝 성립 → `priced_metering=False`로 **총량 채널이 봉인**된다.

```
동일 상태, budget N_UF=5254 고정, 균일 가격 g를 전 램프에 인가:
  g(veh/h)   split=True 총량   split=False 총량
    -1000          5254.0            6000.0
        0          5254.0            5100.0
     +1000          5254.0            4200.0
  변동폭             0.0            1800.0
```
`split=True`에선 가격 ±1000에도 총량이 **소수점까지 불변**. Phase 4가 실험한 것은 "총량 고정 후 merge 내부 배분을 정하는 선택자"였고, 그 축의 가치 상한은 오라클·회귀 두 방법이 독립적으로 **15~16 TTT**(skew 격차 567의 2.7%)로 일치했다. 학습도 불가능했다: `dQ/da`가 N_UF는 adv sd의 4.93배인데 가격은 0.07~0.18배, 보상 회귀에 가격 추가 시 ΔR²=0.000000(순열 귀무 p95 미만) → AWR 타깃이 0으로 수렴(학습된 가격 sd가 데이터의 0.27~0.32배, 시드 간 부호 반전).

**`split=False`면 `priced_metering` 분기가 열린다**(`wu_faithful_follower.py:3048-3058`): budget이 hard 제약이 아니라 soft anchor(`w·|Σmeter − ω·N_UF|`)가 되고 가격이 방류 수준을 유도한다. 주석 그대로 "가격이 방류 수준을 유도하고 budget은 anchor"(:3064). 원래 이 스위치를 껐던 이유는 P-Stack **leader 탐색**의 병리(incumbent↔후보 교대 커밋 → Σmeter TV 1.74배, 과소방류 −800, :3028-3032)인데, **RL leader는 탐색을 하지 않으므로 해당하지 않는다**(`probe_price_stability.py`로 검증).

**부수 발견**: 유효 가격 범위는 대략 [−250,+250]이고 그 밖은 포화 → `PRICE_MAG=500`은 대부분이 포화 구간이었다.

### 12.6 그 밖에 확인된 설계 결손 (설계 재검토 워크플로, 36 findings)
- **ω_F(링크 배분)가 0.5로 영구 고정**: RL env가 `leader.solve`를 우회(`env.py:144`)하는데 ω_F를 갱신하는 코드는 그 안에만 있다(`stackelberg_wu_metered.py:463-469`) → `wu_distributed.py:119` 생성자 기본값 유지. **RL 트레이스 8개 전부 `max|W총합−E총합| = 0.0000`**. P-Stack은 이 축을 0.26~0.98로 쓴다 → **RL이 P-Stack보다 자유도가 적었다(불공정 비교)**.
- **행동정책이 매 스텝 i.i.d.**라 "낮고 평평한 clamp"가 데이터에 없다: peak N_UF std<218인 에피소드 **0/560**, [4550,4900] 연속 최장 6스텝 vs **P-CENT 19스텝**. P-CENT 정책이 지지 밖 → behavior-constrained offline RL은 원리적으로 재현 불가. `SATURATED`의 근본 원인.
- **P-CENT는 merge 내부 배분을 거의 안 쓴다**: skew D/(D+F) = W 0.498±0.007, E 0.489±0.014(사실상 50:50). §2.5의 "공간 타겟팅" 전제가 교사 궤적과 어긋난다.
- **P-CENT 우위의 실체는 시간축**: 총유입 변조폭 P-CENT ≈1280 vs RL ≈390 veh/h. 4900 미만 연속 유지 P-CENT 19스텝 vs RL 0~1스텝.
- **urban이 지불 수단**: P-CENT는 freeway에서 이기고(−953/−809) urban에서 진다(+332/+439). VSL은 80스텝 전부 115.0으로 미사용(sd=0). green sd는 P-Stack의 3.5~10배. RL엔 per-signal urban 채널이 없고 λ_P는 89% 스텝에서 0(액션 1차원이 사실상 사망).

### 12.7 ★Phase 5·6H 결과 — 공간 축(가격·ω)은 세 구성 모두에서 무익하거나 유해

| windowed TTT | skew | inc | 구성 |
|---|---|---|---|
| **Phase 2 (budget 2차원)** | **6324.8 ± 116** | **8196.0 ± 77** | 하드 budget, 27.7k i.i.d. |
| Phase 4 | 6401.8 ± 137 | 8246.5 ± 47 | 하드 budget + 가격, ω 고정 |
| Phase 5 | 6880.7 ± 4.5 | 8777.7 ± 199 | **soft** budget + 레벨가격 + ω |
| Phase 6H full | 6725.2 ± 223 | 8579.4 ± 211 | 하드 budget + 가격 + ω |
| Phase 6H zero-price | 6674.2 ± 222 | 8338.4 ± 66 | ω만 |
| Phase 6H fix-omega | 6738.6 ± 110 | 8525.0 ± 171 | 가격만 |
| **Phase 6H 두 축 모두 off** | **6509.2 ± 136** | **8190.1 ± 99** | budget 행동만 |

**축별 분해(inc 셀이 결정적 — 데이터/차원 효과가 0이라 축 효과만 남는다):**
```
budget 행동(새 데이터/7차원)  8190.1   ← Phase 2의 8196.0과 사실상 동일(Δ −5.9)
  + ω                        8338.4   (+148)
  + 가격                      8525.0   (+335)
  + 둘 다                     8579.4   (+389)
```
데이터 축소(27.7k→19.5k)·행동정책 변경(i.i.d.→hold)·액션 차원 증가(2→7)의 **합산 비용이 inc에서 0**인데, 축을 켤 때마다 단조 악화한다. skew에선 데이터/차원 비용 +184가 추가로 있으나 축 비용(+215)이 여전히 지배적이다.

**왜 유해한가(해석)**: follower는 이미 배분을 최적화하는 최적화기다. 리더가 내리는 가격·ω는 그 결정을 **덮어쓴다**. 리더가 follower보다 더 아는 게 없으면, 최적해를 잡음으로 교체하는 것이라 구조적으로 손해다. §12.6 실측대로 **P-CENT는 skew에서 merge 내부 분배를 거의 쓰지 않으므로**(D/(D+F) = 0.498±0.007) 이 축에는 리더가 전달할 정보 자체가 없다.

**Phase 5 특기사항(별도 병리)**: `split=False`는 가격 채널을 추가하는 게 아니라 budget을 hard→soft(w≈0.05)로 강등시킨다. 게다가 가격 기준점이 직전 스텝이면 "지금보다 더 줄여라"가 반복되는 **래칫**이 생겨 미터링이 0까지 붕괴한다(probe_price_anchor.py: [4200,3000,2100,1500,0,…], TTT +76%). 기준점을 budget 함의 수준에 고정하면 래칫은 끊기나(16스텝 std 0.0), soft budget 자체가 통제를 잃어 peak 방류가 cap 6000(=NC 수준)에 고정된다.

**시간축 메커니즘 손실**: §5의 이득 원천인 recovery 조임이 Phase 6H에선 거의 사라졌다.
```
recovery N_UF Δ(vs P-Stack)  Phase 2: −493/−700/−560(skew)  −612/−637/−570(inc)
                             Phase 6H: −73/−67/−339          −150/−240/−100
```

**결론**: freeway 공간 축은 닫혔다. 남은 격차의 실체는 §12.6대로 **시간 조율 + urban 채널**이다 — P-CENT는 freeway에서 이기고(−953/−809) urban에서 지불한다(+332/+439). RL엔 per-signal urban 채널이 없고 유일한 urban 레버 λ_P는 89% 스텝에서 0이다. **다음 액션 후보는 가격·ω가 아니라 per-signal green 가격**(`signal_marginal_price`, 주입 인터페이스 존재)이다.

### 12.8 ★Phase 7 — green 가격도 무익. 리더 가격 채널 4연속 기각

가설(사용자): 램프 가격 단독이 실패한 건 유입만 조이면 큐를 램프·도시로 옮기기 때문이다. P-CENT는 freeway에서 이기고 **urban에서 지불**하므로(§12.6), green 채널을 함께 주면 그 거래가 성립한다.

액션 11차원 `[N_P, N_UF, 램프가격×4, green가격×5]`, 총량 하드(split=True), green은 `signal_marginal_price` 주입(mag 0.20 — probe 대역 [0.05,0.3]에 맞춤). ω는 제외(Phase 6H 기여 ≈0). 데이터 19,500샘플, abort 0%.

| windowed TTT | skew | inc |
|---|---|---|
| **Phase 2 (budget 2차원)** | **6324.8 ± 116** | **8196.0 ± 77** |
| p7 full (램프+green) | 6681.8 ± 295 | 8687.6 ± 190 |
| p7zr (램프 off, green on) | 6654.7 ± 148 | 8377.3 ± 197 |
| p7zg (green off, 램프 on) | 6633.3 ± 137 | 8505.8 ± 88 |
| **p7bo (둘 다 off)** | **6511.7 ± 159** | **8293.3 ± 68** |

**축 분해(inc — 데이터/차원 효과를 both-off가 흡수하므로 축 효과만 남는다):**
```
both off          8293.3
 + green만        8377.3   (+84.0)
 + 램프만         8505.8   (+212.5)
 + 둘 다          8687.6   (+394.3)   교호작용 +97.8 → 상쇄가 아니라 가중
```
**green이 램프 가격을 살리지 못한다.** 오히려 inc에서 함께 켜면 개별 합(+296)보다 더 나쁘다(+394). skew는 교호작용이 −95로 약간 상쇄되나 두 축 모두 개별 유해는 동일.

### 12.9 종합 — 리더 가격 채널에 대한 결론

| Phase | 구성 | 결과 |
|---|---|---|
| 4 | 램프 가격(merge 내), ω 고정 | 유해 (끄면 −100/−93) |
| 5 | 레벨가격 + soft budget + ω | 붕괴 (NC 수준, 래칫) |
| 6H | 램프 가격 + ω, 총량 하드 | 둘 다 유해 (+148/+335) |
| 7 | 램프 가격 + **green 가격** | 둘 다 유해, 가중 (+394) |

**4개 구성 전부에서 리더 가격은 무익하거나 유해했고, 최고 성능은 여전히 Phase 2의 budget 2차원(6324.8/8196.0)이다.**

**기전(측정 기반 해석)**: follower는 이미 배분을 최적화하는 최적화기다. 리더 가격은 그 국소 최적해를 **덮어쓴다**. 리더가 유용한 신호를 낼 수 있으려면 (a) follower가 모르는 것을 알고 (b) 그것을 데이터에서 배울 수 있어야 하는데, 둘 다 실측으로 부정됐다:
- (a) P-CENT조차 merge 내부 분배를 안 쓴다(skew D/(D+F)=0.498±0.007). 가격 축 가치 상한 15~16 TTT(격차의 2.7%).
- (b) `dQ/da`가 N_UF는 adv sd의 4.93배인데 가격 차원은 0.07~0.18배, 보상 회귀 ΔR²=0.000000. Q가 가격에 의존하지 않아 AWR 타깃이 0으로 수렴.
→ **배울 수 없는 채널에 액션 차원을 쓰면 잡음을 실행하게 되고, 잡음이 최적해를 대체하므로 구조적으로 손해다.**

**남은 방향(미검증)**: 격차의 실체는 일관되게 **스칼라 총량의 시간 프로파일**을 가리킨다 — P-CENT는 총유입을 peak-to-trough 1280 veh/h 변조하고 4900 미만을 19스텝 연속 유지하는데, RL은 변조 ~390에 연속 유지 0~1스텝. 데이터에 hold 지지는 확보됐으나(P-CENT급 ≥19스텝이 27%) 정책이 그 영역을 안 고른다. 원인은 액션공간이 아니라 **신용 할당**일 가능성이 높다: 데이터의 한계 통계가 "더 풀수록 좋다"(corr(peak N_UF, TTT) = −0.612)를 가리키는데, 이는 즉시 램프큐 감소만 보이고 지연된 breakdown 비용이 안 잡히기 때문이다.

### 12.10 ★신용할당 가설도 기각 — 시뮬레이터가 "조이면 손해"라고 답한다

§12.9는 "스텝 신호가 즉시 램프큐만 보고 지연된 breakdown 비용을 못 잡아 리더를 오도한다"고 추정했다. 그렇다면 **에피소드 전체 TTT**로 보면 부호가 뒤집혀야 한다. 완주 260 에피소드(hold 지지 포함)로 직접 쟀다(`rl_leader/analyze_clamp_value.py`).

**뒤집히지 않았다.** 수요·stressor 통제 후:
```
peak 평균 N_UF  beta = −3.445 (t = −13.79)   ← 많이 풀수록 TTT 낮음
clamp 지속길이   beta = +83.7  (t =  +6.85)   ← 오래 조일수록 TTT 높음
층화 비교: 7개 층 중 6개에서 긴 clamp가 +965 ~ +5,017 나쁨
```
비선형(구간별 조정 TTT)으로 봐도 단조다: N_UF 3700-4200 **+2893** → 4500-4800 −91 → 5000-5200 −1047 → 5600-6000 −1360.

**수요별로 갈라도 뒤집히지 않는다**(held-out은 고수요 끝단이므로 결정적):

| 수요대 | n | N_UF 기울기 | 3700-4400 구간 |
|---|---|---|---|
| <1.80 | 149 | −3.34 (t −11.3) | +2198 |
| 1.80~2.05 | 71 | −2.72 (t −6.0) | +2066 |
| **≥2.05** | 40 | **−5.00 (t −5.7)** | **+2602** |

고수요에서 오히려 기울기가 더 가파르다. **어느 수요대에서도 조이면 손해다.**

**함의(중요)**:
1. **신용할당 문제가 아니다.** γ·n-step·보상 shaping 어느 것도 이 부호를 바꾸지 못한다 — 에피소드 수준 진실이 스텝 신호와 같은 방향이다.
2. **RL 정책은 오히려 옳게 학습하고 있었다.** 자기 액션공간 안에서 최적을 고르고 있으며, clamp를 안 하는 게 정답이다.
3. **P-CENT 격차는 이 액션공간에서 구조적으로 도달 불가**다. P-CENT가 peak 총유입 ≈4720(스윗스팟 5254보다 낮음)으로도 5757을 내는 건 §12.6대로 **urban에서 지불하고 freeway에서 회수**하기 때문인데(urban +332/+439, freeway −953/−809), budget 리더가 같은 수준으로 조이면 비용만 지불하고 회수를 못 한다. §2.5의 "정적 4720 강제 시 폭발(7800)"과 정확히 일치한다.

**§2.4 스윗스팟(5254)과의 관계**: 그것은 190-skew 단일 셀·고정 N_UF 스윕의 값이고, 위 결과는 도메인 랜덤화 분포 전반의 관측이다. 두 결과가 함께 말하는 건 "스윗스팟 아래로 내려가면 급격히 나빠진다"이며, P-CENT의 운영점은 그 아래에 있다.

### 12.11 ★단, '레벨'과 '반응성'은 별개 축이고 **반응성은 값이 있다** (유일한 양성 결과)

데이터의 clamp는 개루프(무작위 레벨을 8~20스텝 유지)라 "개루프 조임이 나쁘다"가 "상태반응형 조임도 나쁘다"를 뜻하지 않는다. 수요·stressor·**peak 평균 N_UF까지 통제**한 뒤 반응성(에피소드 내 `corr(N_UF, rho_max)`)의 효과를 쟀다:

| corr(N_UF, rho) | n | 조정 TTT |
|---|---|---|
| −1.00 ~ −0.40 (혼잡시 조임) | 82 | **−202** |
| −0.40 ~ −0.15 | 42 | −132 |
| −0.15 ~ +0.15 | 63 | +42 |
| +0.15 ~ +0.40 | 35 | +299 |
| +0.40 ~ +1.00 (혼잡시 풂) | 38 | +237 |

**반응성 계수 +724.4 (se 253.9, t = +2.85)** — 같은 평균 레벨에서 혼잡시 조이는 정책이 유리하다. 모드별로도 `reactive` −65 / `clamp` −290 / `hold_wide` +458.

**즉 축이 둘로 분리된다:**
- **레벨**(평균 N_UF를 낮춤): 유해. 효과 크기 +2600 ~ −1360 — 지배적.
- **반응성**(혼잡시 조이고 여유시 풂): **유익**. 효과 크기 ≈ ±220, 전체 스윙 ~440.

**그런데 학습된 정책이 이 축을 못 잡고 있다.** Phase 2 3시드의 skew `corr(N_UF, rho_max)` = **−0.380 / +0.694 / +0.543** — 2/3 시드가 위 분석상 **불리한 부호**다. 레벨 신호(±2600)가 반응성 신호(±220)를 압도해 AWR이 레벨에만 맞추는 것으로 보인다.

**→ 다음 작업 후보(액션공간 확장 없이)**: 행동정책에 **반응성 대비**를 명시적으로 깔아 이 축의 지지를 키운다. 현재 `reactive` 모드는 25%이고 반응 이득이 고정(−1200·tight)이라 대비가 얇다. 반응 이득 k를 층화 샘플링(예: k ∈ {−1500, −800, 0, +800})하면 정책이 이 축을 분리해 배울 수 있다. 기대 회수는 스윙 ~440으로 inc 격차(+180)를 덮고 skew 격차(+567)의 상당 부분을 설명한다.

### 12.12 ★★보상 shaping — 유일하게 성공한 개입 (신기록, 2026-08-04)

사용자 제안("−TTT만이 아니라 far도 더해서 최대화"). **목적함수 변경이 아니라 potential-based shaping**으로 구현했다:
```
r' = r + γΦ(s') − Φ(s),   Φ = −w·(혼잡 포텐셜),  w=300
Φ = obs[6] = ρ_crit 초과 세그먼트 수(over)     ← 13차원·23차원 관측 모두에 존재
종료 전이는 Φ(s')=0 (흡수상태, 부트스트랩 끊김)
```
**최적 정책이 TTT 기준 불변**임이 보장되면서(Ng et al.) 신호만 조밀해진다. 지금 문제가 "무엇이 최적인지 모른다"가 아니라 "신호가 0이라 못 배운다"였으므로 정확한 처방이었다 — §12.10 실측: **N_UF ±300의 1스텝 TTT 효과가 0.02**(차량이 램프큐↔본선으로 '이전'될 뿐 둘 다 TTT에 계산). 학습 신호가 전부 V(s')로만 흘러 희박했다.

**결과 — budget 2차원(Phase 2 액션공간·27.7k 데이터, 재수집 없이 학습만 재실행):**

| | 190-skew | 190-inc |
|---|---|---|
| Phase 2 (shaping 없음) | 6324.8 ± 116.0  [6192.5, 6373.1, 6408.8] | 8196.0 ± 76.8  [8107.9, 8231.2, 8248.9] |
| **+ shaping(over, w=300)** | **6223.9 ± 51.9**  [6173.6, 6220.7, 6277.3] | **8152.1 ± 77.7**  [8067.8, 8167.7, 8220.8] |
| 평균 개선 | **−100.9** | **−43.9** |
| P-Stack 이긴 시드 | 2/3 → **3/3** | 3/3 → 3/3 |
| P-CENT 격차 | 567.4 → **466.5** (18% 축소) | 179.7 → **135.8** (24% 축소) |

- **처음으로 3시드 전부가 양 셀에서 P-Stack을 이겼다**(skew 최악 시드 6277.3 < P-Stack 6379). Phase 2는 skew에서 1시드가 졌다(§0의 "skew 동률" 판정을 개선).
- **시드 분산이 절반 이하로 줄었다**(skew sd 116.0 → 51.9). 신호가 조밀해져 학습이 안정화.
- 판정 스크립트는 VERDICT=BAD를 냈으나 이는 임계가 "개선 > 기준선 std"로 보수적이기 때문이다(skew −100.9 vs std 116). 시드 짝지음·P-Stack 승률·분산 감소가 모두 같은 방향이므로 실질 개선으로 본다.

**Φ 형태 스크리닝(p7 11차원 데이터, 3시드)**: `over`(w=300)가 inc에서 **3/3 시드 개선(−219.4)**으로 최선. `link`(링크별 초과분, 공간 분해)는 2/3(−103), `rho`·`mix`는 중간. **"skew가 안 되는 건 Φ에 공간 해상도가 없어서"라는 가설은 기각**됐다(link300이 skew 1/3, +91.8).

### 12.13 ★★hinge 형태 Φ가 더 낫다 (2026-08-04, 현 최고기록)

사용자 지적: "green이나 metering은 결국 차량 **위치만** 바꾸는데 TTT는 위치를 구분 안 하니 RL이 반응 못 한다. hinge/far를 켜야 한다." → §12.10 실측(N_UF ±300의 1스텝 TTT 효과 0.02)과 정확히 일치하는 진단.

`over`(초과 세그먼트 **개수**)는 1만큼 넘든 50만큼 넘든 동일하게 센다. 진짜 hinge는 `Σ max(0, ρ_i − ρ_crit)·L·λ`로 **초과량**에 비례한다. 세그먼트별 밀도가 관측에 없어 근사를 썼다:
```
Φ_hinge ≈ obs[6] × max(0, obs[5] − 1)   =  "몇 개가 넘었나" × "얼마나 깊이"
```

**budget 2차원(27.7k, 재수집 없이 학습만) 3시드 결과:**

| Φ / w | skew | inc | inc P-CENT 격차 |
|---|---|---|---|
| 없음 (Phase 2) | 6324.8 ± 116.0 | 8196.0 ± 76.8 | +179.7 |
| over / 150 | 6235.4 ± 3.9 | 8193.8 ± 76.6 | +177.5 |
| over / 300 | **6223.9** ± 51.9 | 8152.1 ± 77.7 | +135.8 |
| over / 600 | 6230.4 ± 58.8 | 8143.7 ± 41.4 | +127.4 |
| **hinge / 600** | 6234.3 ± 63.6 | **8072.2 ± 14.7** | **+55.9** |
| hinge / 1500 | 6234.9 ± 20.1 | 8119.5 ± 59.1 | +103.2 |

- **inc: hinge@600이 over@300 대비 −79.9. P-CENT 격차 179.7 → 55.9 (69% 축소).** 시드 std 14.7로 전 구성 중 최고 안정(범위 8061.6~8089.0).
- **skew: 6개 구성 전부 6224~6235에 몰림**(퍼짐 11.5 = 시드 노이즈 수준). shaping 도입이 −100을 벌어준 뒤 Φ 형태·가중치에 **완전 무감각**. 격차 473에서 정체.
- 해석: inc(국소 사고)는 한 지점이 깊게 임계를 넘어 depth 가중이 정보를 더한다. skew(분산 비대칭)는 깊이가 안 생겨 depth가 무의미 — `link`(공간 분해) Φ가 skew를 못 살린 것과 같은 방향.

**→ 진짜 `leader_hinge_cost`/`_mfd_far_cost_to_go` 로깅이 정당화됐다.** 근사만으로 inc에서 80을 벌었으므로, 세그먼트별 가중이 들어간 진짜 항은 더 낼 수 있다. 수집기에 두 값을 per-step 로깅하면 되고(재수집 필요), Φ 후보를 학습 시점에 자유롭게 바꿀 수 있게 된다.

**현 최고 구성: budget 2차원 + Φ=hinge 근사, w=600 → skew 6234.3 / inc 8072.2.**

### 12.14 far 항은 기각 — 다른 정보이나 이 액션공간엔 방향이 반대 (2026-08-05)

**probe로 재수집 14h를 두 번 회피했다**(`probe_hinge_far.py`, `probe_far_approx.py`):

| 항 | 관측 근사와의 상관 | 판정 |
|---|---|---|
| **진짜 hinge**(`leader_hinge_cost`) | **r = +0.971** | 로깅 불필요 — 현 근사가 이미 담고 있다 |
| **far**(`_mfd_far_cost_to_go`) | r = +0.686 | 다른 정보. 단 **urban 차량수와 r = +0.881**(freeway 지표는 전부 \|r\|<0.16) → obs[0]로 근사 가능, 역시 로깅 불필요 |

호출 비용은 hinge 0.00 ms / far 0.42 ms로 무시 가능(env 스텝 ~15,000 ms). far는 hinge와 성격이 다르다 — hinge는 진동(현재 혼잡), far는 에피소드 내내 단조 증가(1.7→277, 누적).

**far 근사(Φ=uveh)를 3시드 시험한 결과 — 기각:**

| Φ / w | skew | inc |
|---|---|---|
| **hinge / 600 (기록)** | **6234.3 ± 63.6** | **8072.2 ± 14.7** |
| farx / 200 (far 단독) | 6341.7 ± 71.2 | 8182.7 ± 40.2 |
| hingefar / 600 | 6251.7 ± 40.7 | 8113.2 ± 33.9 |
| hingefar / 1200 | 6224.2 ± 58.4 | 8090.3 ± 50.7 |

- **far 단독은 shaping 없는 Phase 2(6324.8)보다도 나쁘다**(skew 6341.7).
- hinge+far 합성도 hinge 단독을 못 넘는다(hf1200이 skew −10.1로 동률권, inc +18.1).

**해석**: Φ = −w·uveh는 "도시 누적이 비싸다"는 신호이므로 잠재적으로 **더 방류하라**는 방향을 가리킨다 — 미터링 리더에겐 정확히 반대다. far는 §12.6대로 P-CENT의 '지불'을 값매기는 항인데, **우리 리더는 urban을 관리할 수단이 없어 그 지불을 회수하지 못한다.** 같은 정보가 P-CENT에겐 유용하고 budget 리더에겐 해롭다.

**최종: Φ = hinge 근사(`obs[6]×max(0,obs[5]−1)`), w=600 이 최선.**

### 12.15 ② on-policy 수집 라운드 — null (2026-08-04)

`SATURATED`(같은 분포에서 더 뽑아도 무익) 대응으로 현 정책(hg600_s0)을 σ∈{0.05,0.15,0.30}로 굴려 50에피소드(3,750샘플, abort 0%) 수집 → 기존 27.7k와 결합(31,456) → 재학습.

| | skew | inc |
|---|---|---|
| hinge@600 (기록) | 6234.3 ± 63.6 | 8072.2 ± 14.7 |
| + on-policy 데이터 | 6238.1 ± 14.6 | 8105.1 ± 8.6 |

**개선 없음**(skew +3.8, inc +32.9). 얻은 건 분산 감소뿐(skew sd 63.6→14.6, inc 14.7→8.6).

**해석**: IQL은 behavior-constrained다. on-policy 데이터를 넣으면 행동정책이 학습 정책에 가까워져 AWR이 그쪽으로 더 강하게 묶인다 — 결과가 정확히 그 모양(변화 없음 + 분산 감소)이다. **모방학습의 DAgger 논리가 behavior-constrained offline RL엔 그대로 적용되지 않는다.**
확인된 사실 하나: on-policy 데이터의 N_UF 하한이 1,890까지 내려갔다(스케줄 수집은 3,500) — 정책이 미탐색 영역을 방문하는 건 사실이나, 그걸 메워도 성능은 안 올랐다.

**남은 것**: skew 격차 466~473은 여전히 크고 shaping 형태·가중치·데이터 분포 어느 것에도 무반응이다.

---

## 13. ★다음 작업 (2026-08-05 인수인계 — 여기부터 읽으면 됨)

### 13-A. 이미 기각된 것 — 반복하지 말 것

| 시도 | 결과 | 근거 |
|---|---|---|
| per-ramp metering 가격 (merge 내 배분) | 유해 | §12.5, 가치 상한 15~16 TTT = 격차의 2.7% |
| 레벨가격 + soft budget (`split=False`) | 붕괴 (NC 수준) | §12.7, 기준점 래칫으로 미터링 0까지 |
| ω_F 링크 배분 | 기여 ≈ 0 | §12.7 (skew +13 / inc −54) |
| per-signal urban green 가격 | 유해, 램프 가격과 가중 | §12.8 (+394) |
| 데이터 증량 | 무의미 | `SCALING=SATURATED` (§12.5) |
| on-policy 수집 라운드 | null | §12.15 (분산만 감소) |
| far 항 (Φ에 urban 누적) | 유해 | §12.14 (방향이 반대) |
| N_P 박스 확장 | 무의미 | §13-C (레버가 물리적으로 slack) |

**공통 기전**: follower는 이미 배분을 최적화하는 최적화기다. 리더 신호가 유용하려면 (a) follower가 모르는 걸 알고 (b) 데이터에서 배울 수 있어야 하는데, 공간 축은 둘 다 부정됐다(`dQ/da`가 N_UF 대비 1/30, 보상 회귀 ΔR²=0.000000). **배울 수 없는 채널에 액션 차원을 쓰면 잡음이 follower 최적해를 덮어써서 구조적으로 손해다.**

### 13-B. 저비용 후속 후보 (재수집 불필요, 회당 ~1시간)

전부 `data/rl_dataset/w*.npz`(27.7k)로 학습만 다시 하면 된다.

1. **Φ 가중치 정밀 스윕** — hinge w ∈ {400, 800, 1000}. 현재 600은 {600, 1500} 2점만 비교한 값이라 최적점이 아닐 수 있다. inc는 w=600이 1500보다 −47 우수했으므로 600 부근을 좁게 볼 것.
2. **Φ 항 조합 재탐색** — `iql.py --shape`에 후보가 이미 구현돼 있다(over/rho/accum/mix/link/linkmix/hinge/farx/hingefar). 아직 안 해본 조합: hinge + 램프큐, hinge + 속도.
3. **5시드로 확정** — 현재 3시드. skew sd 63.6이라 ±100 주장을 굳히려면 5시드가 안전하다.

### 13-C. skew 격차(+473)에 대한 현재 이해 — 미해결

측정이 일관되게 **urban 채널 부재**를 가리킨다. P-CENT는 freeway에서 이기고(−953/−809) **urban에서 지불한다**(+332/+439). 우리 리더에겐 그 지불 수단이 없다:

- **N_P(누적 목표)는 물리적으로 slack이다.** 보호영역 누적이 458 veh인데 임계는 1142 veh(40%). deadband(`np_dual_deadband_frac=0.9`)가 항상 저stock으로 판정해 λ_P가 0으로 감쇠한다 → N_P −3500~+2200을 흔들어도 follower green이 8상태 중 1곳에서만 12초 움직인다. **박스를 넓혀도 무의미.**
- **green(신호 배분)은 학습이 안 된다.** Phase 7에서 직접 줬으나 조건수가 나쁘고(무반응 대역 \|g\|≤0.02, 포화 \|g\|≥0.5, 후보 6초 격자) 기여가 음수였다.

→ skew는 **이 액션공간의 구조적 한계일 가능성이 높다.** 뒤집으려면 리더가 urban을 실제로 조향할 수 있는 채널이 필요한데, 후보(N_P·green)가 각각 위 이유로 막혀 있다. **미해결로 남긴다.**

> ⚠️ 발견된 버그 1건(수정됨, 기본 OFF): `follower.solve()`는 λ_next를 diagnostics로만 내놓고 `_lambda_P`를 갱신하지 않는다. 되쓰기는 컨트롤러(`stackelberg_wu_metered.py:2436-2446`)에만 있는데 RL env가 그 레이어를 우회한다. `env.py`의 `_commit_np_dual()`로 복제했고 `RLLeaderEnv(np_dual=True)`로 켤 수 있다. 현 시나리오에선 위 deadband가 가려 효과가 없지만, **도시 수요가 임계에 근접하는 시나리오에선 차이가 날 수 있다.** ω_F도 같은 유형이었다(§12.6).

### 13-D. 재현 명령 (현재 최고 구성)

```bash
# 학습 3시드 (env 접촉 0, 시드당 ~13분 — 병렬 가능)
python rl_leader/iql.py --data "data/rl_dataset/w*.npz" --steps 40000 --seed 0 \
    --shape hinge --shape-w 600 --out checkpoints/actor_hg600_s0.pt

# 평가 (셀당 ~30분. 6롤아웃 병렬 안전 — §11, 단 수집 워커와 겹치지 말 것)
python rl_leader/eval_guarded.py checkpoints/actor_hg600_s0.pt \
    --max-sec 3600 --cells skew --tag hg6000 --trace-dir traces

# 3시드 판정 (기준선 대비 자동 비교)
python rl_leader/judge_p6.py hg600 0,1,2
```

### 13-E. 진단 도구 (전부 `rl_leader/`, 재수집 없이 도는 것들)

| 스크립트 | 용도 |
|---|---|
| `probe_hinge_far.py` | 진짜 hinge/far vs 관측 근사 상관 — **재수집 가치 판정** |
| `probe_far_approx.py` | far를 obs로 근사 가능한가 |
| `probe_np_range.py` / `probe_np_multistate.py` | N_P가 follower를 움직이나 |
| `probe_np_diag.py` | N_P dual 루프가 어디서 끊기나 |
| `probe_price_level.py` / `probe_price_anchor.py` / `probe_price_stability.py` | 가격 채널 특성 |
| `probe_hard_budget_split.py` | 총량 보존 + ω/가격 도달집합 |
| `analyze_clamp_value.py` | clamp가 에피소드 수준에서 이득인가 |
| `inspect_p6.py` | 임의 데이터셋 요약(샘플/차원/모드/커버리지) |
| `judge_p6.py` / `scaling_verdict.py` | 3시드 판정 / 데이터-스케일링 판정 |

**교훈: 14시간짜리 수집을 걸기 전에 probe로 먼저 재라.** 이 방식으로 재수집 2회(28시간)를 회피했다(§12.14). §12.6대로 P-CENT는 urban에서 지불하고 freeway에서 회수하는데(urban +332/+439, freeway −953/−809), budget 리더에겐 그 지불 수단이 없다. w·Φ 형태 튜닝(w 스윕, Φ에 램프큐·속도 항 추가)이 남은 저비용 후보다.

### 12.3 수집
14워커 × 40에피소드, `rl_leader/launch_collect_p6.ps1`(PID는 `logs/collect_p6_pids.txt`, 진행은 `logs/collect_p6_progress.log`). **증분 저장이라 조기 중단해도 그 시점까지 유효** — PID 지정 kill로 끊어도 된다(blanket kill 금지, §8-4).
수집 중에는 평가를 병행하지 말 것(§11 — 그 순간 CPU 과다구독).

### 12.4 ★수집 실측 — `--max-ep-sec 1800`은 14워커에 너무 빡빡하다 (다음에 반드시 조정)
2026-08-01 실측(420에피소드 시점):
```
abort율 100% (420/420), 에피소드당 58.8스텝(전체 75), 전부 ~1800s에 걸림
비교: budget-only 27.7k는 abort 41%, 65.5스텝/ep, 완주 에피소드 249개
```
- **원인은 가격이 아니라 경합**: Phase 1 실측에서 **단독** 롤아웃도 75스텝에 1610~1875s로 이미 1800s 경계였다. 14워커(10물리코어) 경합이 스텝당 21.5s→30.5s로 늘려 57스텝에서 잘린다.
- **영향**: 에피소드가 t≈11,160s에서 끝나 **recovery 꼬리 18스텝(3,240s)을 한 번도 관측하지 못한다.** §5에서 IQL 이득이 전부 recovery에서 나온다고 규명했으므로, 성과가 드러나는 구간이 학습 데이터에서 체계적으로 결손된 셈이다. 치명적이진 않다(절단은 `done=0`이라 bootstrapping상 올바른 처리, 평가는 3600s 가드로 완주 → 결과 숫자 자체는 정직).
- **다음 수집 권고**: `--max-ep-sec 2700`(경합 하 75스텝 ≈ 2,290s 소요) 또는 워커 10개로 축소. 둘 중 하나만 해도 완주 에피소드를 확보할 수 있다.
- 수집 속도 실측: 14워커에서 **≈1,235 샘플/시간**(§7.4의 2,900/h는 이 머신·이 설정에 맞지 않는다).
