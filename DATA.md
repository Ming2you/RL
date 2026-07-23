# 데이터 및 코드 안내

RL leader 학습에 필요한 데이터·코드. 수치는 모두 실측(run_log). 시각화·설명은 [REPORT.md](REPORT.md).

## `code/` — RL 구현
| 파일 | 역할 |
|---|---|
| `env.py` | Gym형 env. budget(N_P,N_UF) → follower.solve → plant. 13-dim 관측, reward=−ΔTTT. `make_random_scenario`(도메인 랜덤화). |
| `nets.py` | Actor(tanh-squash) + Critic(twin Q). |
| `sac.py` | SAC 학습(BC init, `--continuous` 도메인 랜덤화). **online SAC는 발산 — 참조용.** |
| `collect_bc_data.py`, `bc_train_torch.py`, `bc_eval.py` | BC 수집·학습·평가(teacher=P-Stack optimizer, 천장 P-Stack). |
| `eval_holdout.py`, `eval_all.py` | held-out/전체 셀 windowed TTT 평가. |
| `ceiling_sweep.py` | 고정 N_UF sweep(스칼라 budget 천장). |
| `oracle_measure.py` | 위상별 budget 오라클. |
| `price_inverse_probe.py` | per-ramp 가격 → follower 반응 정적 probe(가격=분배 확인). |
| `pcent_follow_test.py` | P-CENT 총량+분배 추종 테스트. |

> ⚠️ 코드는 시뮬레이터(별도 레포 **Numerical-Sim**)에 의존한다(`from src.controllers...` 등). 이 레포만으론 실행 불가 — sim을 함께 두고 경로 조정 필요.

## `data/holdout/` — held-out baseline (평가 참조)
`ho_{controller}_{cell}.csv` = NC/PFO/P-Stack/P-CENT × {inc, skew}의 run_log(190+stressor).
**windowed TTT** = `cumulative_total_ttt`[마지막] − `cumulative_total_ttt`[step==4] (warmup 5스텝 제외).

## `data/pcent_teacher/` — inverse-optimization 목표 (핵심 학습 데이터)
P-CENT의 제어·상태 궤적. 계획된 접근(§4.4)의 **역산 목표** = "follower가 이걸 재현하게 만드는 coordinating price를 역산".
| 파일 | 내용 |
|---|---|
| `control_timeseries.csv` | P-CENT의 per-step 제어 결정 |
| `state_timeseries.csv` | per-step 상태 |
| `run_log.csv` | 전체 로그. per-ramp metering = `ramp_metering_release_actual_R_{D,F}_{W,E}_veh` (per-step veh, ×20 = veh/h) |

현재 skew·inc 2셀. green은 P-CENT run_log에 **미기록**(VSL은 skew서 거의 free) — green 목표가 필요하면 P-CENT 재실행 요.

## `data/bc/` — BC 데이터 (state→budget)
`{cell}.npz` = optimizer(P-Stack) leader의 (관측 X, budget 액션 Y). BC 학습용(천장 P-Stack, 계획상 offline RL로 대체).

---
## 학습 파이프라인 (계획, §4)
1. `data/pcent_teacher/`에서 P-CENT 제어 목표 추출.
2. 각 스텝 (budget+전-lever 가격) **역산**(follower가 목표 재현) → (상태→가격) teacher 데이터 생성.
3. **offline RL(IQL/AWAC)** 학습 → `data/holdout/`으로 평가.
