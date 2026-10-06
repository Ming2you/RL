# 현재 TTD S-DMPC 실행 코드 / RL 연결 안내

**저장소: `Ming2you/RL`, 브랜치: `main`, 폴더: `sdmpc_ttd/`.**

2026-10-06 TTD S-DMPC 실행 코드와 의존 모델을 함께 보존한 배포본이다.
저장소 루트의 `src/`, `rl_leader/env.py`는 이전 P-Stack/Wu RL 구현이다.
현재 TTD S-DMPC를 쓰려면 반드시 이 폴더의 `runtime.bootstrap()` 또는
`runtime.make_runtime()`부터 호출해야 한다. 서로 다른 `src`를 같은 process에
먼저 import하면 실행을 중단한다. 기존 실험 출력 폴더나 개인 PC 경로는 필요 없다.

## 설치와 실행

검증 환경은 Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Windows이다.
별도 venv에서 저장소 루트를 작업 폴더로 사용한다. Linux affinity 경로는 제공하지만
이번 배포에서 Linux 수치 동등성은 시험하지 않았다.

```bash
python -m pip install -r sdmpc_ttd/requirements.txt
python -m unittest sdmpc_ttd.test_runtime -v

# 5회 warm-up + 첫 제어 판단: 원 plant와 NC를 함께 실행
python -m sdmpc_ttd.run --scenario sweet_155_w --steps 6 --max-candidates 3 --output outputs/ttd_smoke

# 원 비교 조건: 14,400초, 하위 최대 6회, 상위 최대 3후보
python -m sdmpc_ttd.run --scenario sweet_170_w --steps 80 --max-candidates 3 --output outputs/ttd_170

# 별도 진행 중이던 상위 탐색 확장 조건
python -m sdmpc_ttd.run --scenario sweet_170_w --steps 80 --max-candidates 10 --output outputs/ttd_170_upper10
```

시나리오는 `sweet_155_w`, `sweet_170_w`, `sweet_170_skew15_w`,
`sweet_170_incident_w`, `sweet_190_w`이다. 출력 폴더는 새 경로여야 한다.
기본적으로 사용 가능한 논리 CPU 하나를 선택하며 BLAS thread도 1로 둔다.
Windows에서는 예를 들어 `--cpu-mask 8`로 CPU index 3을 지정할 수 있다.
다른 무거운 작업과 함께 실행한 시간을 독립 성능 측정으로 해석하면 안 된다.

실행 때 원 결정적 수요 생성기로 입력을 재생성하고, 보존한 fingerprint와 대조한다.
전체 14,400초 NC를 먼저 실행해 정규화를 검산한다. 제어 목적에 사용하는 α는
`config/alpha.json`의 **기존 실험 계수**로 고정한다. 재계산 비율과의 허용 차이는
원 실행기의 `1e-15`이며, 동역학/실행 허용오차를 변경한 것이 아니다.
NC 검산·초기화 시간은 `decision_wall_seconds`에 포함하지 않는다.

## 비교 조건

| 항목 | 배포본 기준 |
|---|---|
| 목적 | `TTT − α·TTD`, α 단위 h/km |
| TTT / TTD | veh·h / veh·km |
| TTD 정의 | 실제 모델 방출량 × 완료 링크 길이; 미시 차량 궤적의 연속 주행거리 아님 |
| 물리 모델 | 역사 d6241bb 기반 METANET + 도시 queue/transit 모델, 기존 선언된 호환 수정 포함 |
| 실행 / seed | 14,400초 / 42 |
| 제어 간격 / warm-up | 180초 / 처음 900초 무제어 |
| 예측 | 3구간 = 540초, 구간 전체에 동일한 제어를 쓰는 기존 move block |
| player | freeway 4개(방향별 전반 4·후반 4 segment), urban 5개 |
| 제어 | 공통 그룹 VSL 4개, RM 4개, green 5개, offset 5개 |
| lower / upper | lower 최대 6회; upper 최대 3후보 기본, 10후보 별도 옵션 |
| `N_P` | H3 전체 inbound service − outbound service [veh]의 상한; 음수 가능 |
| `N_UF` | ramp metering **명령** 합 [veh/h]의 상한; 실제 통과량과 다름 |
| 지역 문제 | 현재 prox-linear own + 외부효과 + budget 가격 + proximal QP |
| 민감도 | 공통 기준점의 결합 동역학 직접/AD 미분, 위험 분기 FD fallback 유지 |
| 가격 | Jacobi 응답 결합 후 k→k+1 갱신; 후보마다 동일 수신 가격으로 시작 |
| 실행 선택 | 원 비선형 물리·제어·budget 검사, composite 목적 PFO guard |
| 계산 구조 | 한 process, 직렬 player 응답; 중앙 결합 예측·미분·자원 QP |

현재 구현은 과거의 interval 내부 가격 고정 프로토타입이나 비선형 own NLP와 다르다.
지역 trial에는 교통 rollout이 없고, 공통 기준점과 결합 제어의 원 rollout을 수행한다.
민감도의 비매끄러운 분기 및 stationarity/승수 인증 한계도 그대로 보존했다.

## 학습 코드에서 연결할 위치

```python
# 새 process에서, 저장소의 다른 src를 import하기 전에 실행
from sdmpc_ttd.runtime import make_runtime

rt = make_runtime("sweet_170_w", max_candidates=3)
# rt.cfg, rt.options: 원 설정
# rt.profile.horizon(time_sec, 3): 원 수요/예측
# rt.plant: MixedTrafficSimulator
# rt.solver: TTTTDAnchorSDMPC (현재 수동 budget search 기준선)
# rt.warm_controller: 기존 PFO proposal 생성기
# rt.alpha: 기존 시나리오별 고정 계수
# rt.previous: 초기 무제어 명령
```

기존 기준선의 한 판단은 `rt.solver.decide(state, forecast, previous, pfo_control)`이다.
실제 warm-up, PFO 초기화, 원 제약 검사, plant 진행 순서는 [run.py](run.py)를 따른다.

**Actor budget을 넣을 하위 진입점은 `solver.solve(budget, seed, initial_dual.copy())`다.**
먼저 `solver.begin(state, forecast, previous)`를 호출하고,
`solver.coords.quantize(solver.coords.encode(pfo_control))`로 같은 초기 제어를 만든다.
`budget`은 `[N_P_veh, N_UF_veh_per_h]`의 물리 단위 numpy vector이다.
반환 후보를 `solver.execution_check(result['point'], result['budget'])`로 검사한다.
이 호출 자체는 plant 실행/가격 commit/보상 계산을 대신하지 않는다.

Actor의 budget을 `solver.last_budget`에 넣고 기존 `decide()`를 호출하는 방법은 쓰면 안 된다.
현재 `PFOAnchorSDMPC.decide()`는 매 interval PFO가 달성한 budget에서 다시 탐색하며
`last_budget`을 재설정하기 때문이다. RL 환경에서는 해당 **상위 후보 생성 부분만 교체**하고,
후보 평가의 가격 격리·실제 제약 검사·실행 선택·선택된 가격 commit·PFO guard를 연결해야 한다.
현재 `make_runtime()`은 연결 부품이며 Gym 환경이나 학습 완료 모델이 아니다.

가격은 상한 제약의 정규화 승수이고 외부효과 민감도와 별개다.
기준선은 선택 후보의 가격만 넘긴다. PFO fallback 또는 archive 복원에서는 기존 수신 가격을
유지하는 경로가 있으므로 `anchor_audit`/`execution_audit`를 기록해야 한다.
요청 budget, 실제 달성량, 적용 budget, fallback과 최종 실행 action을 분리해야 critic에
다른 action의 결과를 잘못 라벨링하지 않는다.

보상은 plant 한 구간의 실제 `TTT` 및 독립 `DistanceCapture`의 `TTD`로 계산한다.
예측 H3의 겹치는 비용을 episode reward에 더하지 않는다. 실행 전에 원 gate가 실패하면
plant를 진행하지 않는다. 자세한 TD3 설계는
[설계 문서](../docs/rl_ttt_ttd_actor_critic_design_20261006.md)를 따른다.

## 주요 파일과 출처

- [runtime.py](runtime.py): 초기화, 혼합 import 차단, 입력 재생성, NC 정규화, 연결 API.
- [run.py](run.py): 원 실행기의 폐루프·원 제약·PFO guard·기록 절차를 보존한 TTD CLI.
- [config/](config/): 물리/factory 설정, solver 옵션, 시나리오, 기존 α, 입력 fingerprint.
- [sdmpc_objective.py](vendor/work/ttd_upper10_20261006/sdmpc_objective.py): TTT−αTTD objective adapter와 실제 controller class.
- [upper_search.py](vendor/work/ttd_upper10_20261006/upper_search.py): 교체할 기존 상위 budget 탐색.
- [prox_controller.py](vendor/work/sdmpc_budget_ablation_20260922/prox_controller.py): player별 근사 응답·공동 자원 조정.
- [anchor_controller.py](vendor/work/sdmpc_budget_ablation_20260922/anchor_controller.py): PFO budget 시작점과 실행 guard.
- [group_block_engine.py](vendor/work/sdmpc_slide_alignment_20260922/group_block_engine.py): 결합 동역학 직접/AD 민감도.
- [source_manifest.json](source_manifest.json): 원/배포 SHA-256. vendor 124개 파일 중 123개는 byte 동일,
  `sdmpc_objective.bootstrap()`만 포함된 설정을 읽도록 경로 초기화 변경.

새 실행의 원 후보·잔차·승수·종료 사유는 `decision_*.json`, 실제 구간 TTT/TTD는
`run_log.json`, 상태·제어는 `plant_*.json`, 완료/실패는 `completion.json`/`failure.json`에 남는다.
원 실험 궤적·PPT·개인 체크포인트는 이 코드 배포에 포함하지 않았다.

## 검증 범위와 남은 작업

배포 전 7개 단위 검사(5개 시나리오 입력 fingerprint, 회계/물리 보존, 후보 격리,
3후보 기존 순서, 10후보 범위, 실패 시 가격 보존, source hash)를 통과했다.
5회 warm-up + 1회 제어 판단의 실제 폐루프를 실행했다.
세부 수치 동등성 결과는 [검증 기록](VALIDATION.md)에 정리한다.

이 배포는 재현 가능한 코드 연결 작업이다. 기존 lower/upper 최적성 인증 실패를
PASS로 바꾸지 않는다. `controller_acceptance=False`, `convergence_certified=False`를 유지한다.
전체 5개 시나리오를 새 배포본으로 다시 실행한 결과나 새 RL 학습 성능을 주장하지 않는다.
다음 작업은 이 runtime을 사용하는 RL step/commit 환경, TD3 actor/critic, 동일 조건 기준선 비교다.
