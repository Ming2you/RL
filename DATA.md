# 데이터 및 코드 안내

RL leader 학습에 필요한 데이터·코드. 수치는 모두 실측(run_log). 시각화·설명은 [REPORT.md](REPORT.md).

## `rl_leader/` — RL 구현

### 핵심 파이프라인 (현재 최고 구성이 쓰는 것)
| 파일 | 역할 |
|---|---|
| `env.py` | Gym형 env. budget(N_P,N_UF) → follower.solve → plant. 13-dim 관측, reward=−ΔTTT. 옵션 플래그로 가격/ω/green 축 확장 가능(전부 기각됨, 기본 OFF). |
| `nets.py` | Actor(tanh-squash) + Critic(twin Q). |
| `iql.py` | offline IQL. **`--shape` 로 potential-based 보상 shaping**(hinge/over/far 등 9종), `--frac` 데이터-스케일링. |
| `collect_parallel.py` | 병렬 수집. piecewise-constant 행동정책(hold 8~20스텝), `--policy`로 on-policy 롤아웃, `--np-wide`/`--price`/`--omega`/`--green` 축 토글. |
| `eval_guarded.py` | held-out 평가(시간가드+트레이스). ablation: `--zero-price`/`--zero-green`/`--fix-omega`. |
| `analyze_mechanism.py` | 구간(peak/recovery) × 서브시스템(urban/freeway) 이득 귀속. |
| `judge_p6.py` | 3시드 판정(기준선 대비 자동 비교). |
| `inspect_p6.py` | 임의 데이터셋 요약(샘플/차원/모드/커버리지). |
| `run_seeds_eval.ps1` | 시드×셀 병렬 평가 러너(6롤아웃 동시). |
| `chain_*.ps1` | 수집→학습→평가→판정 무인 체인(Phase별). |

### 진단 probe — **14시간 수집 전에 이걸 먼저 돌릴 것**
이 방식으로 재수집 2회(28시간)를 회피했다(HANDOFF §12.14).

| 파일 | 무엇을 재나 |
|---|---|
| `probe_hinge_far.py` | 진짜 `leader_hinge_cost`/`_mfd_far_cost_to_go` vs 관측 근사의 상관 → **재수집 가치 판정** |
| `probe_far_approx.py` | far를 obs 13차원으로 근사 가능한가 |
| `probe_np_range.py`, `probe_np_multistate.py` | N_P가 follower를 실제로 움직이나 |
| `probe_np_diag.py`, `probe_np_dual.py` | N_P dual 루프가 어디서 끊기나 |
| `probe_price_level.py`, `probe_price_anchor.py`, `probe_price_stability.py` | 가격 채널 특성(총량/기준점/안정성) |
| `probe_hard_budget_split.py` | 총량 보존 + ω/가격 도달집합 |
| `probe_green_price.py` | green 가격 응답곡선 |
| `analyze_clamp_value.py` | clamp가 에피소드 수준에서 이득인가 |
| `scaling_verdict.py` | 데이터-스케일링 판정(DATA_LIMITED / SATURATED) |
| `smoke_*.py` | 각 구성의 발사 전 게이트 |

### 레거시 (참조용, 현 파이프라인 미사용)
`sac.py`(online SAC — 발산), `collect_bc_data.py`/`bc_train_torch.py`/`bc_eval.py`(BC — 천장이 teacher),
`eval_holdout.py`/`eval_all.py`, `ceiling_sweep.py`, `oracle_measure.py`, `price_inverse_probe.py`,
`pcent_follow_test.py`, `inverse_distill.py`, `run_*.sh`(원 머신 전용 — 경로 하드코딩, HANDOFF §10)

> 코드는 시뮬레이터(`from src.controllers...` 등)에 의존한다. 시뮬레이터는 **이 레포의 `src/`에 포함돼 있어 별도 레포 없이 실행된다**(원본은 `Ming2you/Numerical-Sim`). 레포 루트에서 실행할 것.

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
`{cell}.npz` = optimizer(P-Stack) leader의 (관측 X, budget 액션 Y). BC 학습용(천장 P-Stack, offline RL로 대체됨).

## offline RL 데이터셋 — 어느 걸 쓸지

| 디렉터리 | 샘플 | obs/act | 용도 |
|---|---|---|---|
| **`data/rl_dataset/`** | **31,456** | **13 / 2** | **★현재 최고 구성이 쓰는 것.** `w*.npz` 27,706(스케줄 행동정책) + `onpol_w*.npz` 3,750(on-policy 라운드) |
| `data/rl_dataset_p6/` | 32,687 | 23 / 6 | Phase 4 — 램프 가격(merge 내 배분). 기각(§12.5) |
| `data/rl_dataset_p5/` | 19,500 | 23 / 7 | Phase 5 — 레벨가격 + ω, soft budget. 기각(§12.7) |
| `data/rl_dataset_p6h/` | 19,500 | 23 / 7 | Phase 6H — 하드 budget + 가격 + ω. 기각(§12.7) |
| `data/rl_dataset_p7/` | 19,500 | 23 / 11 | Phase 7 — 램프 가격 + green 가격. 기각(§12.8) |

기각된 4개는 **negative result 재현·재분석용**으로 남겼다(총 56시간의 수집 결과물). 차원이 서로 달라 **섞어 쓸 수 없다** — `iql.py`의 glob이 여러 디렉터리를 걸치면 `np.concatenate`에서 실패한다.

`w*.npz` 키: `obs, act, budget, prices, omega, green, rew, next_obs, done, ep, meta`.
`meta`는 에피소드별 dict(`mode`, `price_mode`, `green_mode`, `sigma`, `demand`, `stressor`, `steps`, `aborted`, `cum_ttt`, `sec`).

> **수집 품질 주의(HANDOFF §12.4)**: `--max-ep-sec 1800`은 14워커에 너무 빡빡해 **100% 절단**된다(에피소드당 ~58/75스텝). `rl_dataset_p6`가 그 상태다. 이후 데이터셋은 **10워커 + 2700초**로 abort 0%를 달성했다.

---
## 학습·평가 (현재)
```bash
python rl_leader/iql.py --data "data/rl_dataset/w*.npz" --steps 40000 --seed 0 \
    --shape hinge --shape-w 600 --out checkpoints/actor_hg600_s0.pt
python rl_leader/eval_guarded.py checkpoints/actor_hg600_s0.pt --max-sec 3600 \
    --cells skew --tag hg6000 --trace-dir traces
python rl_leader/judge_p6.py hg600 0,1,2
```
