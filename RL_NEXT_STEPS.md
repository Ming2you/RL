# RL Leader Next Steps

마지막 갱신: 2026-08-23 10:22 KST

## 1. 현재 정지 상태

- `data/contract_v4_24h_v2` 수집은 약 17시간 17분 실행 후 중단됐다.
- 수집 worker, 후처리 watcher, 완료 대기 shell은 모두 종료됐으며 관련 프로세스는 남아 있지 않다.
- 완료된 335개 episode, 25,125 transition은 정상 저장됐다. 진행 중 episode는 저장 대상에서 제외됐다.
- action/observation/response 계약은 각각 `coordination_action_v4` 75차원, `coordination_observation_v4` 238차원, `rl_pstack_b13_full_segment_vsl_certificate_v3`다.
- validity는 100%, dead action dimension은 0이며 5개 target과 peak/recovery coverage가 모두 존재한다.
- `minimum_transitions=10,000` dataset gate는 통과했다.
- 새 IQL 학습과 5개 조건 평가는 아직 시작하지 않았다.

근거 파일:

- `results/contract_v4_24h_v2/interrupted_collection_audit.json`
- `results/contract_v4_24h_v2/interrupted_collection_gate.json`
- `RL_REPLACEMENT_PLAN.md` 16.6절

## 2. 재개 시 권장 결정

현재 25,125 transition을 clean contract-v4 dataset으로 사용해 바로 학습한다. 모든 사전 gate를 통과했으므로 단순히 24시간을 채우기 위한 재수집보다, 동일 데이터에서 정책 구조 차이를 분리하는 편이 우선이다.

정확한 24시간 wall-clock이 실험 조건으로 반드시 필요할 때만 약 6시간 43분의 top-up을 별도 디렉터리에 수집한다. 기존 `data/contract_v4_24h_v2`를 덮어쓰거나 구계약 dataset과 혼합하지 않는다. top-up을 선택하면 multi-dataset 입력 계약부터 명시적으로 추가하고 다시 audit한다.

## 3. 실행 순서

1. **Dataset 고정:** 현재 9개 NPZ의 hash와 final audit를 manifest에 기록한다.
2. **3-seed matched IQL:** seed 0/1/2를 각각 80,000 update 학습한다. dataset, normalization, support weight와 update 수는 모두 동일하게 둔다.
3. **Policy audit:** seed 간 action disagreement, state-conditioned support distance, certificate 비율과 block별 action 분산을 확인한다.
4. **5개 조건 full run:** 155, 170, 170 incident, 170 skew15, 190에서 seed별 75 policy step을 실행한다. 총 15개 RL run이다.
5. **P-Stack 비교:** 전체 TTT, recovery TTT, throughput, validity, support를 같은 baseline과 비교한다.
6. **Freeze gate:** scenario별 P-Stack 대비 열세 1% 이내, incident/190 recovery 열세 2% 이내, validity 100%를 동시에 요구한다.
7. **구조 ablation:** gate 실패 시 같은 dataset에서 behavior-prior residual IQL과 owner-block critic을 비교한다. 추가 데이터 수집과 모델 구조 변경을 한 실험에서 동시에 하지 않는다.
8. **최종 holdout:** freeze된 정책만 미사용 demand, incident 위치/강도, skew 조합에서 한 번 평가한다.

## 4. 재개 명령

저장소 루트에서 다음 pipeline 하나로 audit, 학습, ensemble 검사, 평가와 gate 판정을 수행한다.

```powershell
$env:PYTHONPATH = "C:\torchlib;."
powershell -ExecutionPolicy Bypass -File work/run_contract_v4_postprocess.ps1 `
  -Data "data/contract_v4_24h_v2/worker_*.npz" `
  -ResultDir "results/contract_v4_24h_v2" `
  -TrainingSteps 80000 `
  -MinimumTransitions 10000 `
  -Python "C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
```

주요 산출물은 `checkpoints/actor_contract_v4_24h_v2_s{0,1,2}.pt`, `results/contract_v4_24h_v2/ensemble_audit.json`, `comparison.csv`, `phase_comparison.csv`, `policy_gate.json`이다.

## 5. 결과 해석

- 세 seed가 모두 P-Stack에 비슷하게 지면 데이터 양보다 action attribution 또는 scalar critic 구조를 우선 의심한다.
- seed 편차만 크면 support 경계와 behavior-prior 제약을 먼저 본다.
- incident/190 recovery에서만 지면 recovery transition 가중치와 VSL/metering block credit assignment를 분리한다.
- native P-Stack replay까지 다시 어긋나면 정책 학습을 멈추고 hidden follower state와 candidate-search side effect를 재진단한다.
