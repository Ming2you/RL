# 2026-09-10 로컬 복원·연구 종합 점검

점검 대상은 `7a8f1a88571d5c021fd865b2e54ed77a6746f3ba`이다. 이후 사용자가 5조건 공통 정책 학습을 요청하여 `codex/five-cell-shared-policy`에서 새 구현을 진행했다. 아래 원본 검증 결과와 새 구현 검증을 구분한다.

## 저장소와 원본 보존

- 원격 `main`과 로컬 `main`을 정확히 `7a8f1a8`로 맞췄다. 현재 개발 브랜치는 해당 커밋에서 분기했다.
- 이전 `handoff/2026-08-05`의 `4515a83`은 최신 main의 조상이 아니다. 공통 조상 `176d22e`에서 갈라진 이력을 임의 병합하지 않았다.
- 기존 미커밋 변경은 `stash@{0}`의 `pre-pull 7a8f1a8 preserved local work 2026-09-10`에 보존했다. 원래 브랜치도 남아 있다.
- 추가 백업: `outputs/audit_20260910/pre_pull/local_work.zip`, `manifest.json`, `tracked.patch`. 미커밋 파일과 체크포인트 총 122개를 포함하며 ZIP 무결성을 검사했다.
- 원격 분할 ZIP의 모든 조각, 전체 ZIP, 원본 **3,039개 파일**의 해시를 검증했다. 로컬에 없던 **2,941개 파일**을 복원했고 기존 파일을 덮어쓰지 않았다.
- 기존 170-incident 실험의 STOP 2개와 두 평가의 중단 지점은 그대로 보존했다. 사용자 신규 지시는 별도의 5조건 실험에 적용한다.

## 최신 연구 결과는 원본과 일치

- 실행 코드 **177/177**, 모델 **6/6**, 고정 입력 **10/10**, Git에서 바로 읽는 JSON 요약 **82/82**의 원본 해시가 일치했다.
- 현 최고 ungated full-run: TTT **5694.030775216803**, P-Stack **5730.792964723197** 대비 **0.641485% 개선**.
- 5% 기준은 **5444.2533164870365**로, 최고 결과는 여전히 **249.77746 TTT** 높다. 목표 달성으로 볼 수 없다.
- 최고 결과의 실제 replay reward와 warmup으로 TTT를 재구성한 차이는 약 **0.0000133**이다.
- 기존 1-step/5-step 학습 데이터는 **856개 전이, 실제 terminal 12개, 검증된 연결 844개**다.
- 중지된 두 평가는 각각 **48/75단계**, 마지막 step 47, terminal=false이다. 누적 TTT **5186.874183055881 / 5186.788407053982**는 완주 결과가 아니다.
- 기존 구현은 native P-Stack과 여러 후보의 실제 follower 반응을 계산하는 선택기이다. preview-free RL 추론이나 다른 시나리오 일반화로 표현하면 안 된다.

세부 기계 판독 결과: `outputs/audit_20260910/evidence.json`, `evidence.md`.

## 코드와 테스트

- 최신 계약·DDQN·중단 복원 관련 선별 회귀 **130개 통과**. 처음 fixture 부재로 skip된 25개는 스냅샷 복원 후 모두 재검증했다.
- 다중 전이 target을 독립 scalar Bellman 구현과 대조했다. **800 cases / 26,400 sampled starts**에서 terminal, 비탐욕 행동 cutoff, 데이터 경계, mask/support, tie를 검증했다. 최대 차이는 float32 누적 오차 수준인 **1.53e-5**였다.
- cutoff·true terminal·online 선택/target 평가를 혼동한 오류는 발견하지 못했다. 기존 문서에 남은 '최종 독립 핵심 리뷰 미완'은 이 검토 범위에서 해소됐다.
- 학습 도중 사용한 실제 backup 길이는 저장되지 않는다. `greedy_path_coverage`는 학습 종료 모델 기준의 길이 통계이므로 실제 학습 중 5-step 사용률로 해석하면 안 된다.
- 기존 전체 suite를 240초 범위로 실행했을 때 legacy distributed-coordinator 테스트 3개가 실패했다. 해당 테스트와 구현은 이전 `4515a83`에서 `7a8f1a8`까지 줄바꿈 외 변경이 없고, 최신 코드에서도 개별 재현됐다. 새 pull 때문에 생긴 실패는 아니다.
- 두 실패는 relaxed VSL quantization이 115를 100으로 바꾸는 동작과 테스트 기대의 충돌, 하나는 3-step Euler 혼잡 proxy penalty가 ramp-service 이득을 상쇄하는 상황이다. 현 response-DDQN 경로의 실패와 구분한다. 전체 테스트가 전부 통과했다고 주장하지 않는다.
- 실제 startup에서 반복되는 merge `{4,6}` 대 현재 `{3,5}` 캘리브레이션 경고도 점검했다. 현재 배치는 설정 파일의 의도된 배치이고, 경고 플래그는 진단·출력에만 사용된다. 새 다섯 조건의 계약 불일치는 아니다. 기존 deadband/far 가중치가 현재 배치에서도 최적인지는 별도 연구 문제이며, 새 기준선과 RL 모두 같은 설정으로 비교한다.

재현 자료: `outputs/audit_20260910/test_audit.md`, `legacy_failure_repro.log`, `multistep_independent_review.md`, `independent_multistep_reference.py`.

## 과거 로컬 IQL·residual 작업

최신 DDQN과 다른 조건의 역사 결과이다. 과거 데이터 및 trace 수치 대부분은 재계산과 일치하지만 문서 해석에는 다음 정정이 필요하다.

- hg600 skew 최악 시드는 **6307.6**이다. 옛 문서의 6277.3은 다른 설정의 수치다. P-Stack보다 3/3 시드가 좋다는 판정은 유지된다.
- stressor 수요에 상한을 적용한 뒤 ±2% jitter를 넣어 실제 상한은 **1.836**이다. '항상 ≤1.80'은 틀리지만 1.90+stressor 직접 누출은 발견하지 못했다.
- 표본 표준편차와 모집단 표준편차가 혼용됐다. 서로 다른 수치만으로 결과 훼손을 뜻하지는 않는다.
- common-price residual의 평균 악화 **skew +137.21 / incident +24.06**과 '개선 없음'은 정확하다. 'SAFE'는 평균 기준이고 skew seed 2는 **+193.38**로 개별 시드 +150을 넘는다.
- 옛 evaluator가 rollout 예외를 출력하고 종료코드 0으로 끝나면 과거 trace를 재사용할 수 있었다. 옛 env.reset은 follower 기억을 초기화하지 않았다. 신규 worker는 실패·중단·완료와 저장 해시를 구분한다.
- elastic transfer의 both-off 악화를 OOD budget drift 하나로 단정하면 안 된다. 당시 soft-budget dynamics와 N_P 물리 범위 차이도 혼재했다.
- 두 'held-out' 셀을 반복 튜닝에 사용했으므로 독립 최종 일반화 평가와 구분해야 한다.

## 다섯 조건으로 돌아갈 때 필요한 조치

사용자가 지정한 조건은 `sweet_155_w60`, `sweet_170_w60`, `sweet_170_incident_w60`, `sweet_170_skew15_w60`, `sweet_190_w60`이다. 하나의 공통 정책 학습을 명시적으로 선택했다.

- 보관된 response replay 128개는 모두 170-incident이다. 다른 네 조건에 필요한 후보별 response replay는 없다.
- 옛 25,125행 full-action 데이터에는 현재 response-DDQN의 action ID/catalog/mask/후보별 반응이 없어 직접 합치지 않는다.
- 옛 5조건 P-Stack 170-incident TTT **5655.981**과 최신 계약의 **5730.793**은 다르다. 새 환경에서 조건별 기준선을 수집해 같은 계약으로 비교한다.
- 과거 snapshot의 Python 3.12.14 / NumPy 2.3.5 / Torch 2.14.0+cpu와 로컬 3.12.2 / 2.1.3 / 2.13.0+cpu는 다르다. 새 실험은 현재 환경을 기록·고정하고 기준선과 RL을 같은 환경에서 실행한다. 과거 결과의 비트 재현을 주장하지 않는다.
- 신규 실험 계획·상태·검증은 [RL_FIVE_CELL_HANDOFF_20260910.md](RL_FIVE_CELL_HANDOFF_20260910.md)에 기록한다.
