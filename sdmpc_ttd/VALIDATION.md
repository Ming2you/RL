# TTD S-DMPC runtime 배포 검증 — 2026-10-06

물리식·solver 변경의 성능 검증이 아니라, 현재 코드의 독립 배포 재현 검사다.
원본 실행과 진행 중인 upper-10 실험은 수정·중단하지 않았다.

| 검사 | 결과 | 범위 |
|---|---|---|
| vendor 원본 대조 | 통과 | 124개 중 123개 byte 동일; objective 모듈의 bootstrap만 변경 |
| 단위 검사 | 7/7 통과 | 후보 가격 격리·실패 보존·기존 3후보 순서·10후보 범위·회계·관측기·입력 |
| 5개 시나리오 입력 | 통과 | config, 초기 상태, 87개 예측 시점, 시나리오 fingerprint 일치 |
| 실제 폐루프 | 완료 | 155, seed 42, 0–1080초, warm-up 5회 + 제어 판단 1회 |
| 원본 대비 실행 제어·상태·TTT | 차이 0 | 위 6구간 모두; NC 상태·TTT도 동일 |
| 원본 대비 3개 후보 | 차이 0 | 요청 budget, 최종 제어, 잔차, dual, 예측 TTT·목적값, 실행/수렴 판정 |
| 구간별 TTD | 최대 차이 4.55×10⁻¹³ | veh·km, 누계 연산의 부동소수점 차이 |
| 5개 전체 NC 재실행 | 통과 | 각각 14,400초; TTT 차이 0, TTD 최대 차이 1.17×10⁻¹⁰ veh·km |
| objective 계수 α | 일치 | 기존 계수 고정; 재생성 NC 비율은 1e-15 기준으로 검산 |

원본 실행기의 수치 재현 기준 `1e-8` 및 정규화 계수 검사 `1e-15`를 유지했다.
배포 초안은 α를 새 NC 누계로 직접 재계산했고 155의 α에 약 `8e-18` 차이가 생겨
미세한 제어 차이가 발생했다. α를 원 실험 계수로 고정하고 NC 재계산은 검산에만
쓰도록 수정한 뒤 위 결과를 얻었다. 초안의 로컬 출력도 보존했다.

명령은 [README](README.md)의 unit test와 smoke 명령이다. 원 실험 결과와의 대조는
보존된 2026-09-30 lower-6 / upper-3 TTD 기록을 로컬에서 읽어서 수행했다.
기존 연구 결과 파일은 배포하지 않았으며 새 사용자는 포함된 코드와 시나리오로 실행한다.

소스 무결성 검사는 `source_manifest.json`의 packaged SHA-256을 사용한다.
원본 파일 SHA-256과 package SHA-256을 별도로 남겨 초기화 수정 범위를 구분한다.
기존 연구 코드의 `Band` 등 클래스명은 남아 있지만, 실제 budget 의미는
`fixed_policy.VARIANT='upper'`와 활성 제약 mask로 결정된다.

이번 smoke의 실행안은 원 제약 검사를 통과했지만 lower 최적화 수렴 인증은 통과하지 않았다.
전체 5개 시나리오의 새 S-DMPC 장기 재실행, Linux 동등성, RL 학습, 실시간 성능,
8% TTT 개선, boundary queue balancing, 최종 controller acceptance를 새로 인증하지 않는다.
다른 실험이 동시에 실행 중이므로 smoke 계산시간을 성능 비교에 사용하지 않는다.
`controller_acceptance=False`, `convergence_certified=False`를 보존한다.
