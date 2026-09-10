# RL 연구 인수인계: 2026-09-10

## 먼저 읽을 상태

**사용자 요청으로 일시 중지 중이다. 이 문서의 GitHub 저장은 재개 승인이 아니다.**
학습·시뮬레이션 프로세스는 종료됐고, 3시간 주기 Codex 예약 `ddqn`은 PAUSED다.
사용자가 명시적으로 재개하기 전까지 실행하거나 예약을 켜지 않는다.
이미 대기 중인 예약 메시지도 재개 승인으로 해석하지 않는다.

- 현재 실험: `results/response_dqn_170_incident/sequential_multistep_v1`
- 학습: 1-step / 최대 5-step 모델 모두 완료. 각각 3개 멤버, 멤버당 24,000 updates.
- 평가: 양쪽 모두 **48/75단계 완료**, 마지막 `control_step=47`, 아직 실제 terminal이 아니다.
- 중지 파일: 현재 실험의 `STOP`과 `sequential_td_audit_v1/STOP` 두 개.
- 상세 작업 기억: [RL_LOGIC_AUDIT_20260907.md](RL_LOGIC_AUDIT_20260907.md).
- 과거 [HANDOFF.md](HANDOFF.md), [DATA.md](DATA.md), [RL_NEXT_STEPS.md](RL_NEXT_STEPS.md)는 역사 기록이다.
  서로 다른 시나리오·계약·평가 구간의 결과를 현재 성능으로 합치지 않는다.

## 연구 목적과 현재 구현

기존 Stackelberg-game hierarchical MPC의 follower는 유지하고, leader의 budget·가격 조정에 RL을 도입한다.
현재 구현은 **native P-Stack을 기준 후보로 포함한 executable follower-response 선택기**다.
순수한 preview-free RL leader로 완성된 상태가 아니며, 모든 후보 preview와 P-Stack 계산이 시간 측정에 들어간다.

10개 조정 후보에는 anchor, 선형 조정, 2차항, 교차항, 조합이 있다. 실제 follower MPC가 실행 가능한 입력을 결정한다.
서로 다른 가격 벡터가 같은 물리 제어를 낼 수 있으므로 명목 action 개수와 실제 반응 다양성을 구분한다.
현재 equivalence mode는 `post_commit_continuation_v1`이며 물리 입력뿐 아니라 인과적으로 필요한 후속 상태도 확인한다.

- 관측 238차원, response features 60차원, nominal residual 74차원.
- reward는 실제 한 제어구간의 `-interval TTT`, scale `0.01`, gamma `1`.
- `done`은 실제 환경 terminal. 수집 중단·파일 끝을 terminal로 바꾸지 않는다.
- DDQN은 online network로 다음 행동을 선택하고 target network로 값을 평가한다.
- optional CQL, 정확히 상수인 feature mask, terminal stratum 25%, cost-head를 사용한다.
- cost-head는 `Q_scaled = -remaining_intervals * softplus(logit)`으로 양의 Q와 terminal 이후 비용을 구조적으로 제한한다.
  이것이 정확한 Q ranking이나 성능 향상을 보장하지는 않는다.
- simulator로 데이터를 구축한 뒤 각 frozen batch에서 학습하는 반복 구조다. 고정 역사 데이터만 쓰는 순수 offline 실험으로 과장하지 않는다.

## 판정 기준

| 항목 | 고정값 |
| --- | --- |
| 시나리오 | `sweet_170_incident_w60` |
| 전체 평가 | normal reset, warmup 5개, 제어 75개, 총 14,400초 |
| 계약 SHA256 | `95694fc1e5bb06da621e784a7e4d4bad56135b6d75ac360bb3a42b2d18501831` |
| P-Stack total TTT | `5730.792964723197` |
| 최소 5% 개선 기준 | **`5444.2533164870365` 이하** |
| 평가 정책 | epsilon=0, 강제 첫 행동·개입 시점 gate·LCB fallback 없음 |
| 목표 확인 | checkpoint/config 고정 및 해시 기록 후 독립 full-run 재현 |

수요·incident·warmup·follower 제약·TTT 회계를 변경해 목표를 맞추지 않는다.
같은 결정론적 조건의 재현은 다른 시나리오로의 일반화를 증명하지 않는다.

## 검증된 결과

아래 개선율은 `100 * (1 - RL TTT / P-Stack TTT)`이며 양수가 개선이다.

| 실험 / 모델 | Full TTT | 개선율 | 평가 wall time |
| --- | ---: | ---: | ---: |
| value_head / finite_horizon_cost | 5695.705196 | +0.6123% | 3841.03초 |
| nonlinear_coverage / refit_only | **5694.030775** | **+0.6415%** | 3927.08초 |
| nonlinear_coverage / targeted | 5698.656123 | +0.5608% | 4001.66초 |
| cost_refinement / long_cql_010 | 5729.759969 | +0.0180% | 4246.87초 |
| cost_refinement / long_cql_001 | 5755.017597 | -0.4227% | 4373.29초 |

각 모델은 preview 4워커로 동시 비교한 실측 시간이다. P-Stack 기준 시간은 2155.12초지만 자원 배치 차이를 고려해야 한다.
현재 검증된 ungated 최적은 +0.6415%이며 **5% 목표는 미달**이다.

다음 수치는 이 표와 섞지 않는다:
- 이전 gated best `5682.100819335536`: 개입 제한이 있었던 참고 결과.
- 고정 prefix에서 2차 후보를 강제로 선택한 tail `5662.790913673756`: 탐색 결과(+1.1866%), ungated 정책 성능 아님.
- 현재 중지된 두 평가의 누적 TTT: one_step `5186.874183055881`, greedy_five_step `5186.788407053982`.
  두 값 모두 48단계 중간 값이며 P-Stack full TTT보다 작다고 승리 판정하면 안 된다.

## 지금까지 배운 점

1. 데이터 양만 늘려도 좋아진다는 가정은 성립하지 않았다. 오래 수집한 IQL도 성능이 악화했고,
   계약·키 매핑·가격 scale·지원 범위·후속 상태 분포·학습 목적을 함께 점검해야 했다.
2. 과거 정책을 끝까지 따라간 fixed-tail return은 그 정책의 결과이지 최적 action-value가 아니다.
   장기 라벨을 최종 진리로 고정하면 이후 회복 행동을 배울 여지를 제한할 수 있어 실제 sequential TD로 전환했다.
3. 명목 가격 차이가 행동 차이를 뜻하지 않는다. 반응 중복 제거와 preview/실제 commit의 continuation parity가 필요하다.
4. 유리한 forced quadratic 경로를 수집해도 학습 정책이 그 진입 행동을 선택하지 않았다.
   CQL이 모든 유리한 행동을 직접 억제한다는 단순 설명도 gradient 진단으로 지지되지 않았다.
5. 706개 데이터에서 6천→2만4천 updates로 늘리면 학습 TD 오차는 줄었지만 full TTT는 개선되지 않았다.
   24k CQL .1/.01 모델은 실제 방문한 75개 상태 전부에서 남은 TTT를 과소예측했다.
   평균 과소예측은 각각 약 204.51 / 277.60 TTT였고, 최종 상태의 즉시 보상 bound 위반도 남았다.
6. nonlinear action을 골랐다는 사실만으로 nonlinear-price의 인과적 개선을 주장할 수 없다.
   마지막 .01 모델의 유일한 quadratic 선택은 step6였으며, 유리했던 탐색 step1과 다르다.

원인으로 확정한 항목과 가설을 혼동하지 않는다. 특히 shared neural approximation과 bootstrapping,
분포 이동이 어느 정도 기여했는지는 아직 분리 검증 중이다.

## 진행 중인 비교

[설정](work/response_multistep_v1.json)과 [러너](work/run_response_multistep_ablation.py)는 동결되어 있다.

- 기존 706개 + 완료된 최근 두 평가 75개씩 = **856개 전이**, 12 terminal, 844개 확인된 연결.
- 새로운 simulator 수집은 하지 않았다. 두 모델에 같은 데이터·seed·architecture·CQL .01을 적용했다.
- 대조군: 1-step. 실험군: 최대 5개 실제 전이를 연결하는 greedy-consistent backup.
- 기록된 다음 행동이 현재 멤버의 supported/masked online greedy 행동과 일치할 때만 연장한다.
  불일치 또는 데이터 경계에서는 online argmax를 target network로 평가하고, true terminal에서는 0을 쓴다.
- targets는 매 update 다시 계산한다. full-return 고정 라벨, uncorrected n-step, full Retrace/lambda 구현이 아니다.
- 학습 완료 시간: 1-step ensemble 약 104.14초, 5-step ensemble 약 178.05초.
- 현재 각각 평가 27단계가 남아 있다. 지금 중지된 상태에서는 실행하지 않는다.

주요 해시:

| 대상 | SHA256 |
| --- | --- |
| plan | `e045d687afa3122613772bbf7a431d5d55e15231275c757f76a6ce44eacdf875` |
| 856-row replay | `e3893d1ffd2491660a466b2821cc686d3fad5fb742c46a6e4b5844deb9733d21` |
| one_step evaluation checkpoint | `baaea49b73810ee78734d5ad5e71f57887bacd659cfec79809ff7cc3b0365ba2` |
| greedy_five_step evaluation checkpoint | `b0b5bee6e38d3fd72a0658e744bee5f451db169d7015e55c2e18921c8f44d56a` |

모델 6개의 해시는 해당 실험의 `model_hashes.json`, 실행 코드 177개 해시는 `implementation_hashes.json`에 있다.

## 데이터와 기억의 보관

[artifacts/research_snapshot_20260910](artifacts/research_snapshot_20260910)은
`data/`, `results/`, `models/`, `checkpoints/` 아래 원본 파일 전체를 분할 ZIP으로 보관한다.
원래 NPZ/PT/PKL/JSON/JSONL/로그를 재인코딩하지 않으며, 파일별 경로·크기·SHA256은 `inventory.json`,
분할 조각과 전체 ZIP 해시는 `manifest.json`에 있다. 계열별 파일 수·용량도 manifest에 기록한다.
주요 sequential 실험의 JSON 요약은 `summaries/`에서 GitHub로 바로 읽을 수 있다.

실행 코드·테스트·설정은 일반 Git 파일이다. `.gitattributes`는 코드와 설정의 자동 줄바꿈 변환을 막아
기존 source pin을 보존한다. 이는 동작을 바꾸는 리팩터링이 아니다.
가상환경, 논문 Word 파일, 개인 앱 메모리/설정/인증정보는 업로드하지 않는다.
이 문서와 audit가 프로젝트 기억의 정리본이며 내부 사고 기록이나 개인 앱 상태를 덤프한 것이 아니다.

## 복원과 재개

복원 도구는 Python 3.11 이상 표준 라이브러리만 필요하며 RL을 시작하지 않는다.
저장소 루트에서 다음을 실행할 수 있다:

```powershell
python -m work.research_snapshot verify
python -m work.research_snapshot restore
python -m work.research_snapshot verify-local
```

`restore`는 없는 파일만 복원하며 기존 파일이 다른 해시이면 중단한다. STOP도 복원된다.
기존 PC는 원본을 지우거나 다시 풀 필요가 없다. 필요한 디스크는 원본 약 2.1GB와 임시 압축본 여유 공간이다.
PT/PKL은 신뢰하는 이 연구 저장소의 파일만 사용한다. 복원 도구는 파일을 unpickle하거나 실행하지 않는다.

**명시적 재개 승인 후** 다음 순서로 진행한다:

1. 현재 process/status/STOP/모델 해시와 코드 pin을 다시 확인한다. 다른 러너가 없는지 확인한다.
2. 재개 승인 범위에 해당하는 두 STOP만 해제한다. 지금 문서를 읽었다는 이유로 해제하지 않는다.
3. 같은 환경에서 `work/start_sequential_response_ddqn.ps1 -Config work/response_multistep_v1.json
   -RunnerModule work.run_response_multistep_ablation`로 재개한다. 학습 완료 모델과 평가 checkpoint를 재사용한다.
4. 두 모델 모두 75단계/14,400초를 완료한 뒤 contract·TTT·terminal·greedy action·trace를 다시 대조한다.
5. 1-step 대 5-step의 full TTT, 실제 연결 길이, Q 보정, 상태 커버리지, 구간별 손실, terminal inventory,
   비선형 후보의 실제 반응과 총 계산시간을 비교한다. 중간 TTT나 낮은 학습 loss로 승격하지 않는다.
6. 목표 미달이면 한 가지 반증 가능한 가설을 정해 작은 진단/표적 수집/검증된 수정으로 다음 버전을 만든다.
   동일 실패 설정 무한 반복이나 근거 없는 24시간 수집은 하지 않는다. 총 preview 워커 한도는 8이다.
7. 5% 후보가 생기면 모델·설정을 freeze/hash하고 같은 전체 평가를 새로 재현한다.
   재현에서도 기준을 충족한 뒤에만 목표 달성 선언 및 추가 실행 중단. 예약은 사용자 요청이 있을 때만 다시 켠다.

다른 컴퓨터에서는 `.venv-torch`를 새로 구성해야 한다. requirements는 범위 지정이고 실제 버전은 snapshot의 환경 기록을 참고한다.
일부 과거 manifest·checkpoint에는 Windows 절대 경로와 런타임 정보가 포함되어 있다.
원본을 수정해 기존 pin을 무효화하지 말고, 새 경로/환경용 버전을 별도 만들고 계약·재현을 검증한다.
**다른 OS/경로에서 자동으로 이어진다는 보장은 없다.**

## 검증 기록과 남은 위험

- 2026-09-09 관련 회귀 테스트 **235개 통과**, 149.38초.
- 다중 전이 전용 23개 테스트, orchestration 및 source 조건 검증 포함.
- 실제 기존 모델 6개의 첫 8 update loss가 기존 기록과 정확히 일치, 두 전체 평가의 75개 행동도 재현.
- 실제 856개 데이터에서 5-step 8-update smoke 통과. 이는 장기 학습 성능 보장이 아니다.
- 다중 전이 코드의 마지막 독립 리뷰는 당시 완료되지 못했다. 부모 에이전트 리뷰와 테스트는 통과했으나,
  다음 코드 검토 때 cutoff/terminal/target semantics를 다시 독립적으로 확인하면 좋다.
- 최신 full-run이 아직 미완이므로 5-step의 우열은 미판정이며 5% 목표도 미달이다.
