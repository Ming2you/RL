# RL Leader for Hierarchical Stackelberg Traffic Control

계층적 Stackelberg MPC(P-Stack)의 leader를 강화학습으로 대체하는 연구.

> ## 👉 이어서 작업하려면 **[HANDOFF.md](HANDOFF.md)** 부터 읽으세요.
> 현재 도달점·확립된 사실·실패한 접근·즉시 이어서 할 일·새 컴퓨터 세팅·함정이 전부 정리돼 있습니다.

## 현재 결과 — budget 2차원 + 보상 shaping, 양 셀에서 P-Stack 격파

| windowed TTT | NC | PFO | P-Stack | **현재 최고 3시드** | P-CENT |
|---|---|---|---|---|---|
| **190-skew** | 6882 | 6299 | 6379 | **6234.3 ± 63.6** | 5757 |
| **190-incident** | 8556 | 9230 | 8386 | **8072.2 ± 14.7** | 8016 |

- **vs P-Stack: 양 셀 3/3 시드 승** (skew −144.6, inc −314.0)
- **vs P-CENT: incident는 사실상 따라잡음**(+55.9, 격차 69% 축소) · **skew는 +476.9로 정체**

가격·ω·green 어떤 액션 확장도 없이, **budget 2차원 + potential-based 보상 shaping**만으로 얻은 결과입니다.
손튜닝 P-Stack(budget+가격+far게이트+감독자)을 양 셀에서 앞섭니다.

## 핵심 발견 5가지

1. **진짜 병목은 데이터 기근이었다** — BC/SAC 실패 원인은 알고리즘보다 **총 375샘플**. 병렬 수집으로 27,706샘플 확보하자 IQL이 바로 P-Stack을 넘었습니다.
2. **online은 불가, offline만 현실적** — env가 스텝당 14~21초라 online RL은 10⁵~10⁶ 스텝(55시간~23일)이 필요합니다. IQL은 **env 접촉 0**이라 이를 우회합니다.
3. **이득의 메커니즘은 시간축 방류 평탄화** — P-Stack은 위기에 과잉 조임·회복에 왕창 방류하는데, RL은 그 진폭을 눌러 recovery 재혼잡을 막습니다. 8/8 롤아웃에서 일관 확인(HANDOFF §5).
4. **★리더 가격 채널은 전부 기각** — 램프 가격·ω·green 가격 4구성 모두 무익하거나 유해했습니다. follower가 이미 최적화기라, 배울 수 없는 채널에 액션 차원을 쓰면 **잡음이 최적해를 덮어씁니다**(HANDOFF §12.9).
5. **★유일하게 통한 건 보상 shaping** — TTT는 차량 위치를 구분 안 해서 미터링을 흔들어도 1스텝 효과가 0.02입니다. `Φ=hinge` potential shaping이 그 신호를 살려 신기록을 냈습니다(HANDOFF §12.12–12.13).

⚠️ **skew 격차(+477)는 미해결**입니다 — urban 채널 부재가 원인으로 보이나, 후보 레버(N_P·green)가 각각 물리적·학습적 이유로 막혀 있습니다(HANDOFF §13-C).

## 문서
- **[HANDOFF.md](HANDOFF.md)** — 이어받기용 종합 문서. **§0 → §13(다음 작업) → §8(함정)** 순으로 읽으세요
- [REPORT.md](REPORT.md) — 배경·데이터·초기 RL 설계
- [DATA.md](DATA.md) — 데이터·코드 안내
- [SIM.md](SIM.md) — 시뮬레이터·최종 컨트롤러 설정·실행법

## 구성
```
src/                 시뮬레이터 + 최종 컨트롤러(P-Stack/PFO/P-CENT)
work/                러너
rl_leader/           RL 구현 + 진단 probe 12종(HANDOFF §13-E)
data/rl_dataset/     offline RL 학습 데이터 27.7k (+ on-policy 3.75k)
data/rl_dataset_p*/  기각된 구성의 데이터(가격/ω/green) — negative result 재현용
data/holdout/        held-out baseline run_log(평가 기준선)
data/pcent_teacher/  P-CENT 궤적
checkpoints/         actor_hg600_s{0,1,2}.pt(★현재 최고), actor_iql_s*.pt(shaping 없음 기준선)
traces/              per-step 트레이스(메커니즘 귀속 분석용)
```
