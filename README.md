# RL Leader for Hierarchical Stackelberg Traffic Control

계층적 Stackelberg MPC(P-Stack)의 leader를 강화학습으로 대체하는 연구.

- **[REPORT.md](REPORT.md)** — 최신 데이터(P-Stack / PFO / P-CENT held-out 성능) + RL 구현 계획.
- **[DATA.md](DATA.md)** — 코드(`code/`)·데이터(`data/`) 안내.

## 구성
- `code/` — RL 구현(env·sac·inverse probe·eval). *sim(Numerical-Sim) 의존.*
- `data/holdout/` — held-out baseline run_logs(평가 참조).
- `data/pcent_teacher/` — P-CENT 궤적(inverse-optimization 목표).
- `data/bc/` — BC 데이터(state→budget).

## 요약
- **문제**: freeway+urban 결합망 TTT 최소화. 리더(budget+가격) → follower(배분 실행) 계층.
- **핵심 발견**: 스칼라 budget만으론 천장 ≈ P-Stack. 헤드룸은 **per-lever 가격(분배) + 상태-반응형 budget**. 손튜닝 가격은 inert(≈0) — RL이 배울 가치가 여기 있음.
- **학습 접근**: online SAC 부적합(비싼 env·발산). **P-CENT를 목표로 한 inverse optimization으로 coordinating prices를 역산 → offline RL teacher.**

시뮬레이션 코드는 별도 레포(Numerical-Sim).
