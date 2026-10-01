# RL 연구 인수인계: 2026-09-30

## 현재 결론

**다섯 시나리오를 균등 학습하는 공유 정책과 가치함수는 구현·학습했다.
그러나 최신 정책이 다섯 기준 시나리오에서 TTT를 개선했다는 결과는 아직 없다.**
최신 정책의 canonical full-run 평가는 실행 전이다. 학습 오차 감소를 교통 성능
개선으로 해석하면 안 된다.

이 문서는 현재 `codex/sdmpc-rl-budget-20260929` 작업의 진입점이다.
이전 IQL/DDQN 기록과 170-incident/P-Stack 5% 목표는 별도 역사 기록이다.
현재 비교 기준은 아래의 **previous-executed-budget carry center**이며, 이를
이전 P-Stack 결과와 같은 기준이라고 부르지 않는다.

사용자의 새 요청은 현재 결과·데이터 정리 및 GitHub 푸시다. 이를 우선하여
새 수치 실험을 시작하지 않았다. 사용자가 목표 자체의 중지를 요청한 것은
아니므로 앱 목표 및 기존 3시간 heartbeat를 임의로 해제하지 않았다.
복원·푸시 자체는 실험을 시작하지 않는다. STOP와 이후 사용자 지시가 우선이다.

## 구현과 완료 내역

- 하나의 상태 조건부 actor와 공유 twin Q. 시나리오별 별도 모델이 아니다.
- 모든 최적화 미니배치40개 중 각 시나리오8개, 정확히20%씩 사용한다.
- actor는 이전 실행 budget을 기준으로 NP/NUF 증분을 제안한다. follower,
  물리적 feasibility guard, 초기·복구 PFO는 유지한다. 매번 새 PFO budget
  탐색을 수행하는 구조는 아니다. reference/하위 최적화 비용은 여전히 존재한다.
- reward는 `-interval_TTT/100`, gamma1, 실제 terminal, 연속75제어구간이다.
  총 simulation은 warmup5구간을 포함하여14,400초다.
- terminal 노출과 budget projection을 진단했고, fallback에 의한 NUF anchor
  누적 하향이 일부 실험을 크게 악화시킴을 재현했다. 데이터량만의 문제로
  단정하지 않는다.
- 수정 체인에서 local750 + NUF-retention375 + 현재 정책375의 실제 전이를
  보존했다. 초기화 학습에는 carry375와 수정 local375만 사용했고, 후속 MC
  보정에는 현재 정책375만 사용했다. 재사용을 새 수집으로 세지 않는다.

### 최근 학습

1. carry 장기 수익으로 Phi1000회, critic250회, actor10회 업데이트했다.
2. 같은 고정 actor로 새로운 학습용 profile5개를 각각75구간 실행했다.
3. 이 actor가 실제로 만든 remaining return으로 공유 critic만250회 보정했다.
   actor/Phi는 byte-equivalent tensor hash를 유지했다. critic Adam은250에서500으로
   이어졌으며 별도의 새 actor 업데이트는 없었다.

최종 모델: `results/sdmpc_rl_balanced_goal_20260930/return_mc_v1/model_final.pt`.
SHA256: `820f62dd337bb40c6ac634e0e2bdb564d111a6e138a59872dc1d42fa376fbfa6`.
Q1의 현재375개 학습 표본 MSE는221.374979에서164.911100으로 감소했다.
terminal MAE는8.474237에서3.600586으로 감소했지만, 190의 장기 예측 오차는
여전히 크다. 앞 수치는 carry-Q와 새 정책 return의 차이, 뒤 수치는 in-sample
fit이다. 독립 검증 정확도나 다른 action의 가치 순위를 보장하지 않는다.
MC 학습 locked-session44.9066693초는 인증·프로세스 시작 등을 제외한 시간이다.

### 최신 정책의 학습용 Full Run

| 시나리오 | 학습 profile seed | 전체 TTT | 최종 inventory |
| --- | ---: | ---: | ---: |
| 155 | 7301 | 3988.486017 | 420.094125 |
| 170 | 7302 | 4262.399108 | 414.943501 |
| 170 incident | 7303 | 5652.260310 | 414.477843 |
| 170 skew15 | 7304 | 3992.545990 | 414.792208 |
| 190 | 7305 | 6791.023647 | 413.375862 |

모두 full75구간/14,400초와 무결성·건전성 검사를 통과했다. 하지만 이5개
profile의 matched carry 결과는 없으므로 아래 canonical baseline과 직접
빼서 개선율을 계산하면 안 된다. 정책 NUF 요청375개 전부6000 상한이었다.
따라서 이번 wave는 NUF action 변화에 대한 가치 정보를 주지 않으며,
비선형 가격을 잘 학습했다는 증거도 아니다.

## 성공 기준과 다음 순서

| Canonical 시나리오 | 재사용할 carry TTT | 최신 정책 canonical TTT |
| --- | ---: | --- |
| sweet_155_w | 3103.0110715680044 | 미실행 |
| sweet_170_w | 3935.903236508048 | 미실행 |
| sweet_170_incident_w | 5546.224352256691 | 미실행 |
| sweet_170_skew15_w | 4250.876599300032 | 미실행 |
| sweet_190_w | 6604.2970168093225 | 미실행 |

1. 새 평가기 `work/sdmpc_rl_return_eval_20260930`는 독립 검토까지 통과했다.
   구현 테스트49개, 실제375개 관측의 actor bitexact parity, 기존5개 baseline
   인증과 독립 SPEC/QUALITY 검토는 통과했다. 아직 실제 실행 승인은 만들지
   않았으며 fresh preflight와 STOP/프로세스/자원 확인을 별도로 요구한다.
2. 새 `return_canonical_v1`에서 동일 모델로5개 canonical episode를 실행한다.
   기존 center는 재수집하지 않는다. 다른 프로젝트의 수치 작업까지 고려해
   총8worker 안에서 실행한다. 현재 결과가 없으므로 완료된 것처럼 재개하지 않는다.
3. exploration/학습/Q선택/performance gate 없이 정상 reset부터 평가한다.
   physical guard와 초기·복구 PFO는 유지하고 모든 비용을 시간에 포함한다.
4. 각각 baseline 대비 `max(1e-6,1e-8*baseline)`보다 큰 TTT 감소가 있어야 한다.
   평균만 좋아서는 통과가 아니다. 모두 통과하면 checkpoint/config를 동결하고
   별도 위치에서5개 전부 재현한다. 결정론적 재현은 일반화 증명이 아니다.
5. 실패 시 action/실행 반응 다양성, interval 손실, terminal inventory, Q 오차를
   보고 반증 가능한 원인을 정한 뒤 작은 균형 데이터/정책 업데이트로 이어간다.
   같은 설정 반복이나 임의24시간 수집으로 되돌아가지 않는다.

현재 평가기 검토와 작업 경계의 최신 기록:
`.superpowers/sdd/rl_budget_return_initialized_policy_20260930/progress.md`.
진행 중 코드/완료 결과/STOP를 수정하거나 완료된 수집을 다시 돌리지 않는다.
과거 process.json은 기록일 뿐이다. 실제 PID+creation+command를 재확인한다.

## 데이터와 재현

코드·고정 물리 snapshot·연구 메모·검토 기록은 일반 Git 파일로 저장한다.
현재 결과 데이터는 `artifacts/sdmpc_research_handoff_20260930`의 분할 ZIP으로
저장한다. 1,627파일(원본5.098GB)을 약653MB,16parts로 압축했고 전체 원본과
해시 대조를 통과했다. 모든 parts가 필요하며 원래 상대 경로와 SHA256으로
검증한다. 최신 모델·실제 replay·trace·summary·완료 manifest·필수 checkpoint를
포함한다. 반복 저장된 중간 checkpoint와 lock의 원본은 로컬에 그대로 두고
`local_only_inventory.json`에 제외 목록/크기/해시를 남긴다. 전체 원본 백업과
동일하다고 주장하지 않는다. 제외778파일은 약5.915GB이며 삭제하지 않았다.
런타임/개인 설정/자격증명은 포함하지 않는다.

```powershell
python -m work.sdmpc_rl_handoff_20260930 verify
python -m work.sdmpc_rl_handoff_20260930 restore
python -m work.sdmpc_rl_handoff_20260930 verify-local
```

복원 도구는 ZIP이나 checkpoint를 실행·unpickle하지 않고, 기존 파일이 다르면
덮어쓰지 않는다. 원래 컴퓨터에서는 복원할 필요가 없다. Python/Torch/NumPy/
SciPy 등 환경과 raw-byte source pin을 먼저 맞춘다. Windows 절대 경로가 들어간
기록의 이식성은 별도 확인해야 하며 핀을 조용히 고쳐서는 안 된다.
선택적 데이터 복원 후에는 테스트 당시의 전체 중간 checkpoint 보존 목록을
그대로 쓰지 말고 새 read-only preflight를 만든다. 모델·소스·물리 계약은
유지하고 새 머신의 실제 보존 목록과 실행 승인을 다시 묶어야 한다.

주요 상세 문서:
- `docs/rl_budget_balanced_goal_20260930.md`: 현재 목표와 금지/중지 규칙.
- `docs/rl_budget_recovery_results_20260930.md`: 이전 공유정책 실패와 복구 이력.
- `docs/rl_budget_local_collection_results_20260930.md`: NUF 누적 하향 진단.
- `docs/rl_budget_nuf_retention_results_20260930.md`: 수정 수집의 paired 결과.
- `docs/rl_budget_return_init_results_20260930.md`: 초기화와 첫 정책.
- `docs/rl_budget_return_wave_results_20260930.md`: 실제5개 학습 trajectory.
- `docs/rl_budget_return_mc_results_20260930.md`: 최신 공유 Q 보정과 한계.

보고된 테스트는 단계별 scoped 검증이다. 이번 푸시에서 전체 저장소의 모든
테스트를 재실행했다거나 모든 역사 코드를 새로 검토했다는 뜻은 아니다.
