# 다섯 조건 공통 정책 — 2026-09-10

사용자는 170-incident 단기 실험에서 돌아와 **155, 170, 170-incident, 170-skew, 190 전체를 하나의 공통 정책으로 학습**하도록 요청했다. 2026-09-10 대화에서 공통 정책을 명시적으로 선택했다.

기존 `sequential_multistep_v1`의 중지 상태는 별도 역사 실험으로 보존한다. 현재 작업은 [계획](work/five_cell_shared_v1.json)의 새 실험이며, 이 문서가 작업 방향의 최신 기준이다. 이전 pull 및 결과 검증은 [종합 점검](LOCAL_AUDIT_20260910.md)을 참고한다.

## 학습·평가 설계

| 항목 | 설정 |
| --- | --- |
| 학습 조건 | sweet_155_w60 / sweet_170_w60 / sweet_170_incident_w60 / sweet_170_skew15_w60 / sweet_190_w60 |
| 조건 비중 | 각 20%, 같은 수의 완전한 75-step episode |
| 정책 | 모든 조건이 공유하는 3-member greedy DDQN ensemble |
| 알고리즘 | interval -TTT, gamma 1, 1-step DDQN, cost-head, CQL .01 |
| 학습량 | 각 라운드 멤버당 24,000 updates, 2라운드 |
| 초기 데이터 | 조건별 anchor 기준선 1회 + 탐색 2회 = 15 episode / 1,125 전이 |
| 두 번째 데이터 | 같은 공통 정책의 탐색 episode 조건별 1회 추가 = 20 episode / 1,500 전이 |
| 평가 | 라운드마다 동일한 frozen ensemble을 다섯 조건에서 각 75단계 완주 |
| 물리·회계 | warmup 5, 총 14,400초, 전체 TTT에 warmup 포함 |
| 병렬 상한 | actor 2 × preview worker 4 = 최대 8 preview workers |
| 중단·재개 | 매 제어단계 env/RNG/replay checkpoint, STOP, 한 invocation 4시간 guard |

명목 행동은 anchor와 두 freeway owner의 선형·2차·교차·조합 후보 총 17개다. 단일 incident에서 골랐던 수작업 artifact action을 공통 정책 후보로 옮기지 않았다. 동일 실제 반응은 검증된 continuation identity로 구분·묶으며, 실제 선택 다양성은 trace로 따로 확인한다.

첫 라운드는 1-step을 사용한다. 과거 5-step의 완주 성능이 아직 미판정이므로 그 우위를 전제하지 않는다. 학습 loss나 평균 지표만으로 정책을 개선으로 판정하지 않는다.

## 검증과 판정

- 다섯 조건의 정확한 계약 SHA를 명시적으로 허용하는 `post_commit_continuation_five_cell_v1`을 추가했다. 기존 단일 조건의 기본 identity는 유지된다.
- 수요/사고 조건 외 물리 설정·follower 제약·warmup·TTT 회계는 공통이다. 다른 계약을 같은 계약으로 위장해 merge하지 않는다.
- 학습 데이터에는 조건별 계약과 전역 episode ID/원본 출처를 저장한다. 다섯 조건 모두 완주하고 같은 episode 수가 되기 전에는 학습하지 않는다.
- 각 조건의 실제 P-Stack TTT와 정책 TTT를 보고한다. 요약은 **조건별 개선율의 단순 평균**과 최악 조건 개선율이며 원시 TTT 합계만으로 비교하지 않는다.
- 다섯 조건은 모두 학습에 사용한다. 이 평가가 미사용 조건에 대한 일반화를 증명하지는 않는다.
- 새 런타임 코드·계획·계약·모델·데이터 해시와 Python/package 버전을 저장한다. 실행 중 pin을 바꾸지 않는다.
- 신규 단위 테스트와 다섯 조건의 실제 preview/commit·checkpoint 재개 smoke 결과를 확인한 뒤 장시간 파이프라인을 시작한다.

신규 테스트는 데이터 12개, runner 7개, worker 20개, continuation 11개로 **50개 통과**했다. 기존 continuation 관련 42개도 통과했고, 기존 170-incident identity의 전체 JSON이 변경 전과 byte 단위로 같음을 확인했다. 실제 startup 결과는 `results/five_cell_shared_checks/startup_smoke_v1/smoke_result.json`에 저장한다.

실제 startup 검사는 **5/5 조건 통과**했다. 각 조건에서 후보 preview와 첫 commit을 실행하고, 별도 프로세스에서 저장한 env/RNG를 복원해 두 번째 commit까지 검증했다. 실제 전이는 총 10개이며 모두 nonterminal로 보존했다. 검사 벽시간은 1,393초였고, 이 짧은 구간의 TTT를 완주 성능으로 해석하지 않는다.

합성 375행(조건별 75행)을 병합한 뒤 실제 optimizer로 모델 3개를 각각 2회 업데이트하고 저장·복원·다섯 조건 추론까지 연결하는 검사도 통과했다. 저장 전후 Q값의 최대 차이는 0이고 terminal Q=0을 확인했다. 이는 소프트웨어 통합 검사이며 실제 정책 성능 결과가 아니다. 근거는 `outputs/audit_20260910/synthetic_training_integration.json`이다.

각 조건의 데이터 비중과 terminal 비중은 정확히 20%다. 무작위 minibatch의 조건 비중은 기대치가 20%이며 개별 batch마다 강제하는 방식은 아니다. 행동 빈도는 episode별 `summary.json`의 `action_support_counts`와 trace에 저장한다. `process.json`의 시간은 해당 runner 호출의 벽시간으로, 여러 번 재개한 전체 누적 시간으로 해석하지 않는다.

## 실행과 상태

**2026-09-10 15:11:49 KST 정식 백그라운드 파이프라인 시작.** 최초 runner PID는 **16008**이다. 실행 직후 `process.json`의 `state=running`, `status.json`의 `phase=baseline_collection` 및 155/170의 초기 env 체크포인트 생성을 확인했다. 아직 완주 데이터나 실제 학습 모델이 나온 상태는 아니다. 아래 상태 파일이 이 시점 기록보다 우선한다.

stdout: `results/five_cell_shared_v1/runner_20260910_151149_863.stdout.log`
stderr: `results/five_cell_shared_v1/runner_20260910_151149_863.stderr.log`

```powershell
# 구성 검증과 해시 고정, 모델 학습/물리 시뮬레이션 없음
python -m work.run_five_cell_shared --prepare-only

# 실제 두 제어단계/조건, 저장 후 재개 검증 — 정식 데이터와 별도
python -m work.check_five_cell_startup

# 숨김 실행. 완료한 episode/모델은 검증 후 재사용
powershell -File work/start_five_cell_shared.ps1
```

상태 파일은 `results/five_cell_shared_v1/status.json`, 프로세스는 `process.json`과 `launch.json`, 결과는 각 `round_XX/comparison.json`이다. 현재 STOP 경로는 **`results/five_cell_shared_v1/STOP`**이며 옛 170-incident STOP과 독립이다.

중단이 필요하면 이 새 STOP 파일을 만든다. 다음 저장 경계에서 멈춘다. 재개할 때는 실패 원인·source/model/data pin·기존 process를 확인하고 동일 계획으로 실행한다. 이미 완료한 데이터 수집이나 모델을 무조건 다시 만들지 않는다. 운영체제 강제 종료와 파일 훼손은 fail-closed로 처리하며 검증 없이 기존 파일을 덮어쓰지 않는다.

새 실험이 완료되어야 학습·평가 완료라고 보고한다. 진행 중에는 완료된 episode/단계와 현재 phase만 보고한다.

## 라운드별 예약 검토 — 사용자 추가 요청

2026-09-10 사용자가 완료 시간을 추정해 예약하고 학습·평가마다 방향을 조정하도록 요청했다. 이 대화에 **매시간 확인하는 예약** `다섯 조건 공통 정책 학습·평가 점검`을 등록했다(automation ID: `automation`, ACTIVE). 정상 진행은 반복 통지하지 않고 새 라운드 완료·확인된 실패·큰 일정 변경·사용자 판단 필요 시 알린다.

현재 runner 자체는 두 라운드를 이어 실행한다. 예약 점검이 새 `round_XX/comparison.json`의 다섯 완주 평가를 확인하면 이 실험의 STOP을 생성해 저장 경계에서 일시중지한다. 따라서 평가 완료와 점검 사이에는 다음 라운드의 데이터 수집이 일부 진행될 수 있다. 조건별 TTT·평균/최악 개선율·terminal inventory·유효성·행동 비중을 검토하고 사용자와 다음 방향을 결정한 뒤 이어간다. 검토 전에 실행 중인 source/config pin을 바꾸지 않는다.

예약의 확인 이력과 마지막 보고한 사건은 `results/five_cell_shared_v1/review_monitor.json`에 저장한다. 최신 대화의 방향 변경과 이 절이 최초의 무조건 두 라운드 자동 진행 설명보다 우선한다.

**첫 전체 라운드 잠정 ETA (9/10 15:20경 측정): 9/11 13시~9/12 09시 KST.** 시작 시점에서 약 22~42시간이며, 기준선 5회·탐색 10회·평가 5회 완주와 첫 모델 학습을 모두 포함한다. 초기 연속 구간은 155 약 105~111초, 170 약 150~158초/제어단계였다. actor 2개와 단계 사이 대기를 반영하면 약 11개 episode-wave × 75단계가 필요하다. 초기화와 재개 시 중복 preview 때문에 smoke 전체 시간을 단순 비례하지 않는다. 아직 다른 조건과 후반 혼잡 구간을 충분히 측정하지 않아 확정 시각이 아니며, 첫 기준선 완주와 후속 진행률로 계속 갱신한다.

## 기준선 완주 확인 — 9/10 23:30 점검

다섯 기준선은 **9/10 22:43 KST 모두 완료**했고, 현재 `balanced_initial_collection`에서 탐색 데이터를 수집한다. 실제 신경망 학습과 정책 평가는 아직 시작 전이다.

| 조건 | 현재 런타임의 P-Stack 전체 TTT | 수집 벽시간 |
| --- | ---: | ---: |
| 155 | 3083.932458 | 2.291시간 |
| 170 | 4015.829470 | 2.372시간 |
| 170-incident | 5729.052814 | 2.556시간 |
| 170-skew | 4165.113995 | 2.695시간 |
| 190 | 6834.284438 | 2.684시간 |

각 기준선은 75단계 완주·terminal·validity를 통과했다. checkpoint/replay 해시와 warmup+reward 회계를 검증했고, 실제 다섯 replay의 병합도 **375행·5 terminal·370 연결·각 조건 20%**로 통과했다. `results/five_cell_shared_v1/baseline_validation.json`에 근거를 저장했다. 표의 벽시간은 후보 preview를 포함하는 수집 시간이며 native P-Stack 단독 실행 속도 비교값은 아니다.

이후 정책 성능은 위의 새 기준선과 비교한다. 예전 런타임의 170-incident 기준선 5730.793을 섞지 않는다. 최초 탐색 prefix는 155 18단계·170 17단계에서 모두 유효했고, anchor 비중은 각각 38.9%·35.3%로 실제 비-anchor 탐색도 진행 중이다. 짧은 prefix로 성능 개선을 판정하지 않는다.

다섯 완주 시간을 반영한 2-actor FIFO 재산정의 중심 예상은 **9/11 저녁**이다. 수집·평가 실행비용을 기준선의 0.9~1.35배로 바꾼 민감도 범위는 9/11 17시~9/12 02시경이며, 통계적 신뢰구간은 아니다. 초기 안내 범위(9/11 13시~9/12 09시) 안에 있어 일정 변경 알림은 보내지 않았다. 아직 incident/skew/190의 탐색과 학습 정책의 실행비용은 확인 전이다.

## 첫 공통 정책 학습 완료 — 9/11 13:38 점검

초기 수집은 **9/11 13:04:42 KST** 완료했고, 모델 3개 학습은 **13:09:41 KST** 끝났다. 실제 학습 벽시간은 **296.857초**다. `round_00/dataset_audit.json`은 1,125행·15 episode·15 terminal·1,110 successor 연결, 조건별 225행·3 episode·3 terminal(각 20%)을 기록한다. 학습 replay의 실제 SHA가 audit 및 ensemble manifest와 같고, 모델 세 파일의 실제 SHA도 manifest와 일치한다.

별도 읽기 전용 점검에서 실제 replay 배열의 행·완주·terminal·관측/반응/mask 연결 수가 위 audit과 같음을 재계산했다. 모델별 24,000개의 유한한 loss와 유한한 가중치, seed 20260910~20260912, action support·catalog·five-cell mode, 데이터/설정/seed/source로 재계산한 학습 서명도 모두 일치한다. 집계 manifest에 최초 155의 `continuation_contract_scope`가 남아 있지만, 집계 검증과 모델 계약은 전체 다섯 조건 map을 사용하므로 허용 조건을 155로 제한하지 않는다.

현재 `phase=five_cell_evaluation`, `round=0`에서 **155와 170의 완주 평가가 진행 중**이다. 13:38 점검의 trace 검증 범위는 각각 11·10단계이며 동일한 모델 세 개와 해시, `ensemble_greedy`, 탐색 없음, Q 진단값 정합성을 확인했다. 초기 수집 1,125개 전이는 모두 검증됐고, 평가를 포함한 누적 검증 전이는 1,146개다. 아직 완주한 정책 평가나 `round_00/comparison.json`은 없다.

현재 속도와 남은 2-actor FIFO 평가 순서로 계산한 첫 전체 평가의 중심 예상은 **9/11 20:48 KST**이며, 정책의 후반부 실행비용에 따라 달라질 수 있다. 기존 안내 범위 안이므로 정상적인 단계 전환 알림은 보내지 않았다. 다섯 조건의 평가가 모두 끝나면 예약 검토에서 결과를 확인하고 다음 라운드를 저장 경계에서 중지하는 기존 방침을 유지한다. 실행 중 source/config/model/data는 변경하지 않았다.
