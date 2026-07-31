# RL Leader for Hierarchical Stackelberg Traffic Control

계층적 Stackelberg MPC(P-Stack)의 leader를 강화학습으로 대체하는 연구.

> ## 👉 이어서 작업하려면 **[HANDOFF.md](HANDOFF.md)** 부터 읽으세요.
> 현재 도달점·확립된 사실·실패한 접근·즉시 이어서 할 일·새 컴퓨터 세팅·함정이 전부 정리돼 있습니다.

## 현재 결과 — offline IQL이 held-out 2셀에서 P-Stack 격파

| windowed TTT | NC | PFO | P-Stack | **IQL(RL)** | P-CENT |
|---|---|---|---|---|---|
| **190-skew** | 6882 | 6299 | 6379 | **6290.9** | 5757 |
| **190-incident** | 8556 | 9230 | 8386 | **8260.5** | 8016 |

vs P-Stack **−88.0 / −125.8**(둘 다 승) · vs PFO 승 · **vs P-CENT는 여전히 패**(+534 / +244).
가격 없이 budget만으로, budget+가격+far게이트+감독자를 다 갖춘 손튜닝 P-Stack을 이겼습니다.

## 핵심 발견 3가지

1. **진짜 병목은 데이터 기근이었다** — BC/SAC 실패 원인은 알고리즘보다 **총 375샘플**. 병렬 수집으로 3시간에 27,706샘플(72배) 확보하자 IQL이 바로 P-Stack을 넘었습니다.
2. **online은 불가, offline만 현실적** — env가 8.5s/step(혼잡 시 스텝당 수십 분)이라 online SAC은 발산·불가. IQL은 **env 접촉 0**이라 이 문제를 전부 우회합니다.
3. **가격 = 분배, budget = 총량** — per-ramp 가격은 merge 내 zero-sum 재분배만 하고 총량을 못 바꿉니다. 정적 스칼라 budget 천장은 6325(≈P-Stack)로 닫혔고, P-CENT 격차는 공간 타겟팅 부재에서 옵니다.

⚠️ **이겼다는 측정됐지만 왜 이겼는지는 아직 미규명**입니다(HANDOFF §5).

## 문서
- **[HANDOFF.md](HANDOFF.md)** — 이어받기용 종합 문서(최우선)
- [REPORT.md](REPORT.md) — 배경·데이터·초기 RL 설계
- [DATA.md](DATA.md) — 데이터·코드 안내
- [SIM.md](SIM.md) — 시뮬레이터·최종 컨트롤러 설정·실행법

## 구성
```
src/                시뮬레이터 + 최종 컨트롤러(P-Stack/PFO/P-CENT)
work/               러너
code/               RL 구현(env·iql·collect_parallel·eval_guarded·analyze_mechanism …)
data/rl_dataset/    offline RL 학습 데이터 27,706 샘플
data/holdout/       held-out baseline run_log(평가 기준선)
data/pcent_teacher/ P-CENT 궤적
checkpoints/        actor_iql.pt(★우승), actor_bc.pt
```
