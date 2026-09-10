# RL Leader Next Steps

> 최신 상태(2026-09-10): 사용자 요청으로 실행·예약 중지. 아래는 누적 연구 이력이며,
> 현재 우선순위와 재개 절차는 [RL_HANDOFF_20260910.md](RL_HANDOFF_20260910.md)를 따릅니다.

마지막 갱신: 2026-08-25

## 1. 현재 상태

- contract-v4 clean dataset `25,125` transition으로 IQL seed 0/1/2를 각각 `80,000` update 학습했다.
- 5개 scenario x 3 seed full-run은 모두 validity를 통과했지만 P-Stack freeze gate는 실패했다.
- full-data IQL의 seed 평균은 P-Stack 대비 170 `-19.78%`, incident `-7.75%`, skew `-15.33%`, 190 `-18.53%`다. 155만 `-0.06%`로 사실상 동률이다.
- optimizer anchor/local만 남긴 `8,775` transition으로 seed 0 diagnostic IQL을 재학습했다. 170은 `4,532.999 -> 4,421.411`, 190은 `6,859.880 -> 6,522.048 veh-h`로 개선됐지만 P-Stack에는 여전히 크게 뒤졌다.
- PFO-anchor exact top-10 follower-response selector를 구현했다. 기본값 1은 기존 동작이며 `--response-candidates 10`에서만 후보 평가가 활성화된다.
- actor observation H=3을 유지하면서 response ranking만 H=6/H=9로 연장하는 `--response-value-depth`를 구현했다. 후보와 PFO는 같은 장기 objective로 비교되고, H=3 rollout TTT와 장기 TTT/penalty를 별도로 trace한다.
- optimizer-neighborhood seed 0의 190 결과는 single H3 `6,522.048`, top-10 H3 `6,526.624`, H6 `6,575.747`, H9 `6,611.511 veh-h`다. 모두 P-Stack `5,734.776`보다 `13.73~15.29%` 나쁘다.
- H6/H9가 single actor를 `1%` 이상 개선해야 한다는 사전 중단 기준을 모두 실패했다. exact top-K와 horizon 확장은 production 승격 대상이 아니라 진단 도구로 동결한다.

`data/contract_v5_residual_24h_v1` base 수집은 2026-08-23 23:59:43 KST에 시작했고, 방향성 검증을 데이터 양보다 우선하기 위해 2026-08-24 17:39에 완료 episode까지만 보존하고 종료했다. final base는 `6,224` transition, validity `100%`다. trainable dead residual은 없지만 `perturb-start-step=24` 때문에 비영 residual이 recovery에만 있고 peak에는 없다. release certificate 4개는 작은 연속 perturbation이 485회 발생해도 실제 부호 전환이 한 번도 없어 P-Stack native anchor에 고정하고 residual actor target에서 제외했다.

`data/contract_v5_residual_direction_pilot_v1`의 2시간 local-only directional pilot은 2026-08-24 19:35에 forced PID 없이 완료됐다. pilot은 `611` transitions, 11 episodes, 5개 target scenario, peak 비영 residual `239`, recovery `372`, validity `100%`다. 21개 continuous owner 중 20개가 선택됐고 빠진 `vsl.FW_W__seg2`는 base에 support가 있다. base+pilot 합계는 `6,835` transitions, residual actor supervision `5,362`, peak/recovery 비영 residual `239/3,420`, trainable dead `0`, exact leader replay `239`, validity `100%`이며 directional minimum `6,000` gate를 통과했다.

이 데이터로 seed 0을 먼저 학습하고 170/190 full run으로 방향성을 판정한다. `10,000+` production dataset은 residual RL이 실제로 채택되고 P-Stack TTT 개선 신호를 보인 뒤에만 추가 수집한다.

## 2. 확인된 원인

1. **데이터 분포 희석:** 전체 `25,125`개 중 optimizer anchor/local은 `8,775`개, 약 `35%`뿐이다. 나머지 loose/epsilon/linear/quadratic mode가 동일 상태에서 상충하는 action mode를 만든다. 같은 seed의 optimizer-only 모델이 full-data 모델보다 170 `2.46%`, 190 `4.93%` 좋아 이 원인을 직접 지지한다.
2. **고차원 multimodal averaging:** 계약이 `117/47 -> 238/75`로 커졌지만 IQL actor는 전체 75차원 action에 scalar advantage 하나를 적용한 deterministic AWR-MSE다. optimizer의 활성 VSL/certificate와 loose 계열의 비활성 action을 평균내 follower response threshold 아래로 보낸다.
3. **Teacher 불일치:** optimizer anchor/local teacher에는 production P-Stack의 PFO supervisor 선택이 없다. PFO 선택 비중이 큰 high-demand/recovery 상태에서 actor가 실제 P-Stack의 switching rule을 학습할 수 없다.
4. **상태 불균형:** 335 episode 중 170은 33개, skew는 25개이며 optimizer episode는 각각 12개뿐이다. transition 총량이 늘어도 scenario x mode x phase의 유효 표본 수는 오히려 부족하다.
5. **후보 탐색만으로는 부족:** top-K는 full-data actor를 일부 개선했지만 optimizer-only actor는 개선하지 못했다. 후보가 평균화된 동일 actor action의 scale 변형에 머물러 새로운 owner별 joint mode를 만들지 못한다.
6. **장기 score calibration 오류:** H9가 채택한 44개 step에서 평균 score 이득 `29.64` 중 실제 장기 TTT-base 이득은 `6.96`이고 penalty 이득이 `22.68`, 즉 `76.5%`였다. peak 채택 21회 중 15회는 H=3 TTT가 PFO보다 나쁜 후보였다. horizon을 늘리자 장기 교통 개선보다 density penalty 감소를 과대평가해 total TTT가 더 악화됐다.

## 3. 다음 작업 순서

1. **실험 동결과 manifest:** H3/H6/H9 코드, checkpoint, 결과 CSV와 trace hash를 고정한다. 현재 top-K는 재현 및 paired-data 진단에만 사용한다.
2. **Production teacher 계약 완료:** deployable follower hidden state를 teacher query 전에 동기화하고, native leader와 internal/external PFO branch를 분리해 기록한다. exact replay가 불가능한 native PFO anchor는 연속 actor target에서 제외한다.
3. **Directional residual dataset:** native P-Stack action을 같은 상태의 `anchor_action`으로 저장하고 optimizer-local action은 `action - anchor_action` residual로 학습한다. 6,224-transition base에 2시간 peak pilot만 합쳐 먼저 방향성을 검증한다. production용 `10,000+` 수집은 seed-0 개선 신호 뒤로 미룬다.
4. **Residual IQL 재학습:** critic은 gate가 실제 적용한 residual action을 사용한다. P-Stack anchor와 기각 local은 `0`, 채택 local만 `action - anchor_action`이다. actor는 zero anchor와 gate-accepted local residual만 감독한다. learned behavior prior와 block critic은 첫 residual seed-0 결과가 실패 원인을 그쪽으로 지목할 때만 별도 ablation으로 진행한다.
5. **P-Stack anchor 배포 gate:** 매 step RL과 native P-Stack을 같은 follower state에서 평가한다. H=3 cumulative TTT가 `max(0.1, 0.1%)` 이상 개선되고 terminal inventory가 증가하지 않을 때만 RL control을 적용한다.
6. **Seed-0 개발 gate:** residual seed 0을 먼저 학습하고 170/190의 14,400초 full run을 `--pstack-anchor`로 평가한다. 외부 PFO supervisor는 native P-Stack 기준선을 바꾸므로 함께 사용하지 않는다. P-Stack 대비 total/peak/recovery TTT와 RL 채택 step을 함께 비교한다.
7. **원인 검증 루프:** seed 0이 P-Stack을 이기지 못하면 proposal support 부족, RL 채택 부족, 잘못된 RL 채택을 trace로 분리한다. 한 번에 한 변수만 바꾸고 재학습 및 170/190 비교를 반복한다.
8. **최종 gate:** seed-0 gate를 통과한 구조만 3 seed x 5 scenario로 확장한다. 모든 run validity `100%`와 P-Stack 대비 안정적인 TTT 우위를 요구한다.

## 4. 현재 실행 명령

기존 single-candidate 평가:

```powershell
$env:PYTHONPATH = "C:\torchlib;."
python -B -m rl_leader.eval_full_action `
  checkpoints/actor_contract_v4_optimizer_neighborhood_s0.pt `
  --scenarios sweet_190_w60 `
  --masks RL-FULL `
  --max-steps 75 `
  --response-candidates 1 `
  --out results/single_candidate.csv
```

실험용 exact top-10 H3 평가:

```powershell
$env:PYTHONPATH = "C:\torchlib;."
python -B -m rl_leader.eval_full_action `
  checkpoints/actor_contract_v4_optimizer_neighborhood_s0.pt `
  --scenarios sweet_190_w60 `
  --masks RL-FULL `
  --max-steps 75 `
  --response-candidates 10 `
  --out results/rl_response_topk_v1/rl_optimizer_s0_190.csv `
  --trace-dir results/rl_response_topk_v1/traces_optimizer_s0
```

H6/H9 ranking 진단은 actor 입력 계약을 바꾸지 않고 각각 depth `3/6`을 사용한다.

```powershell
$env:PYTHONPATH = "C:\torchlib;."
python -B -m rl_leader.eval_full_action `
  checkpoints/actor_contract_v4_optimizer_neighborhood_s0.pt `
  --scenarios sweet_190_w60 `
  --masks RL-FULL `
  --max-steps 75 `
  --response-candidates 10 `
  --response-value-depth 3 `
  --out results/rl_response_value_depth_v1/optimizer_s0_190_h6.csv `
  --trace-dir results/rl_response_value_depth_v1/traces_optimizer_s0_h6
```

## 5. 주요 산출물

- `results/contract_v4_24h_v2/comparison.csv`
- `results/contract_v4_24h_v2/phase_comparison.csv`
- `results/contract_v4_24h_v2/policy_gate.json`
- `results/contract_v4_optimizer_neighborhood_v1/rl_s0_probe.csv`
- `results/rl_response_topk_v1/rl_s0_170.csv`
- `results/rl_response_topk_v1/rl_s0_190.csv`
- `results/rl_response_topk_v1/rl_optimizer_s0_190.csv`
- `results/rl_response_topk_v1/traces_s0/*.jsonl`
- `results/rl_response_topk_v1/traces_optimizer_s0/*.jsonl`
- `results/rl_response_value_depth_v1/optimizer_s0_190_h6.csv`
- `results/rl_response_value_depth_v1/optimizer_s0_190_h9.csv`
- `results/rl_response_value_depth_v1/traces_optimizer_s0_h6/*.jsonl`
- `results/rl_response_value_depth_v1/traces_optimizer_s0_h9/*.jsonl`

세부 근거와 판정은 `RL_REPLACEMENT_PLAN.md` 17~18절을 기준으로 한다.

## 6. 2026-08-23 contract-v5 교정

추가 parity 진단에서 기존 optimizer anchor에는 두 가지 독립적인 오류가 확인됐다.

1. teacher optimizer follower의 hidden coupling/dual state가 deployable RL follower와 달랐다. teacher query 전에 follower 전체를 deep-copy해 같은 상태에서 묻도록 수정했다.
2. P-Stack의 내부 PFO fallback은 연속 budget/price action으로 항상 재현되지 않는다. 5-step production smoke에서 leader 행은 response 오차 `0`, PFO 행은 exact replay `75%`, 최대 오차 `15`였다. PFO 분기는 연속 actor target이 아니라 별도 baseline branch다.

이에 따라 IQL은 실제 RL-adapter transition 전체를 critic에 사용하되, actor AWR/MSE에는 일반 behavior action과 exact-replay 가능한 optimizer leader 행만 사용한다. PFO 및 non-replayable optimizer anchor는 actor supervision에서 제외한다. 기존 contract-v4 `25,125` transition을 이 규칙으로 보면 critic `25,125`, actor supervision `22,575`다.

또한 RL 배포 경로에 native P-Stack anchor를 추가했다. 매 step 동일 follower state에서 native P-Stack과 RL control을 구하고, 공통 H=3 cumulative TTT가 P-Stack보다 `max(0.1, 0.1%)` 이상 낮으면서 terminal inventory도 증가하지 않을 때만 RL을 채택한다. zero actor 4-step 대조에서 RL은 모두 기각됐고 native P-Stack 대비 control과 TTT 오차가 매 step `0`이었다.

다음 실행 순서는 다음과 같다.

1. 2시간 directional pilot 종료 후 base+pilot audit에서 peak/recovery 비영 residual과 trainable dead `0`을 확인한다.
2. production minimum-transition gate는 잠시 적용하지 않고 seed 0 방향성 모델을 먼저 재학습한다.
3. 170/190에서 외부 PFO 없이 `--pstack-anchor` full run을 수행한다.
4. P-Stack 채택률, RL 채택 step의 score gain, peak/recovery TTT를 비교한다.
5. seed 0이 P-Stack을 이기지 못하면 proposal과 gate 중 실패 지점을 먼저 분리하고 한 변수씩 교정한다.
6. seed 0이 P-Stack 개선 신호를 보일 때만 추가 targeted data를 수집하고 production `10,000+` gate와 3 seed x 5 scenario 최종 gate를 실행한다.

## 7. 2026-08-25 contract-v6 native P-Stack 교정

contract-v5 seed 0은 80,000 update 학습을 완료했지만 유효한 production 비교가 아니다. 170/190 TTT는 각각 `4,021.755/5,893.915 veh-h`로 P-Stack보다 `5.57%/2.78%` 나빴고, 평가에서 외부 PFO supervisor와 P-Stack anchor가 동시에 켜져 기준선 자체가 바뀌었다. 수집도 fixed warm-up, continuous search, 외부 PFO가 섞인 구 계약이었다. 따라서 checkpoint와 `6,835` transition은 진단 산출물로만 보존하고 corrected residual 학습에는 사용하지 않는다.

native baseline을 공통 factory로 통합한 뒤 다음 parity를 확인했다.

- warm-up 종료 상태가 production P-Stack과 정확히 일치
- 첫 native optimizer decision의 TTT, budget, 49-candidate grid 결과가 정확히 일치
- zero-residual 4-step에서 P-Stack fallback, follower memory handoff, step TTT 오차 `0`

추가 smoke에서 native P-Stack의 최종 marginal price를 초기 RL follower에 한 번 replay하는 것만으로는 metering response가 재현되지 않음을 확인했다. 가격은 동일했지만 native response는 약 `1,305~1,320 veh/h`, standalone RL replay는 `1,500 veh/h`였다. P-Stack response가 내부 leader-search follower state에 의존하기 때문이다.

contract-v6 수집은 이 잘못된 exact-price-replay 가정을 버리고 실제 배포 MDP를 사용한다.

1. 외부 PFO supervisor를 항상 끈다.
2. uncontrolled warm-up과 native ALLPRICE-JOINT 49-candidate P-Stack을 사용한다.
3. 각 residual마다 RL response와 native P-Stack을 공통 H=3 TTT/inventory gate로 비교한다.
4. 실제 선택된 control로 plant와 follower memory를 전이한다.
5. zero residual 및 gate가 채택한 개선 RL residual만 actor supervision에 넣고, 기각 residual은 critic transition으로만 사용한다.

현재 `data/contract_v6_native_pstack_direction_pilot_v1`을 8 worker, 2시간 상한으로 수집 중이다. gate가 선택한 nonzero RL residual, peak/recovery coverage, validity를 먼저 확인한다. 이 신호가 없으면 재학습하지 않고 perturbation scale과 block 조합을 교정한다. 신호가 있을 때만 seed 0을 40,000 update 학습하고 170/190 full run으로 넘어간다.

## 8. 2026-08-25 contract-v6 재학습 및 full-run 판정

### 데이터와 학습

directional pilot과 190 targeted top-up을 합쳐 `946` transition을 수집했다. validity는 `100%`이고 5개 target scenario, peak `414`, recovery `532` transition을 포함한다. residual actor supervision은 zero anchor `447`, gate-accepted local `21`, 합계 `468`이다. accepted nonzero residual은 peak `7`, recovery `14`에 불과하다.

이 데이터로 seed 0 IQL을 `40,000` update 재학습해 `checkpoints/actor_contract_v6_native_pstack_direction_s0.pt`를 생성했다. actor는 accepted-local residual을 거의 정확히 복원했다. accepted-local 평균 절대 target은 `0.009632`, 예측은 `0.009639`이며 MSE는 수치상 거의 `0`이다. 따라서 이번 실패는 재학습 누락이나 actor residual scale 축소로 설명되지 않는다.

### 14,400초 결과

평가는 외부 PFO 없이 native P-Stack anchor, uncontrolled warm-up, checkpoint support clipping을 사용했다.

| Scenario | P-Stack TTT | RL TTT | P-Stack 대비 | Peak 대비 | Recovery 대비 | RL 채택 |
|---|---:|---:|---:|---:|---:|---:|
| 170 | `3,809.607` | `3,876.666` | `-1.76%` | `-0.37%` | `-4.13%` | `1/75` |
| 190 | `5,734.776` | `6,154.657` | `-7.32%` | `-4.76%` | `-11.09%` | `2/75` |

모든 validity gate는 통과했다. support 밖 원출력은 차원 기준 `45.05%/46.58%`였지만 clipping 평균 변화량은 `1.13e-5/1.83e-5`로 작아 성능 악화의 주원인이 아니다.

170의 유일한 채택은 `t=3,420초`에서 H3 gain `0.374 veh-h`로 판정됐지만 recovery를 P-Stack보다 `4.13%` 악화시켰다. 190의 두 채택은 `t=5,940/7,200초`, H3 gain `0.978/0.145`였지만 recovery를 `11.09%` 악화시켰다. H3 cumulative TTT와 terminal inventory guard가 장기 재고 비용을 판별하지 못했다.

### 추가 학습 계약 오류와 판정

수집기의 `act`는 gate가 기각한 local proposal도 절대 제안 action으로 저장한다. 기존 residual IQL critic은 이를 그대로 사용해, 실제 P-Stack zero residual transition `478`개를 실행되지 않은 절대 action에 연결했다. critic action을 실제 gate-applied residual로 교정했다.

- P-Stack anchor 및 기각 local: critic action `0`
- 채택 local: critic action `act - anchor_action`
- checkpoint critic contract: `pstack_gate_applied_residual_v1`
- 관련 regression test 포함 총 `66`개 통과

같은 데이터와 seed로 `checkpoints/actor_contract_v6_native_pstack_direction_s0_criticfix.pt`를 다시 `40,000` update 학습했다. 기존 actor와 평균 출력 차이는 `0.001129`이고, 이전 장기 손실 채택 시점에서는 residual이 오히려 커졌다. actor가 여전히 H3-accepted label을 정확히 모방하므로 동일 gate로 full run을 반복하지 않는다.

### 다음 순서

1. 기존 170/190 trace의 RL 채택 시점에서 P-Stack branch와 RL branch를 recovery까지 이어가는 paired counterfactual label을 생성한다.
2. primary label을 H3 gain이 아니라 `multi-step cumulative TTT delta + terminal vehicle inventory delta`로 바꾼다. density/queue penalty는 hard safety guard 또는 auxiliary target으로만 둔다.
3. long-horizon label이 양수인 residual만 actor positive target으로 사용하고, H3-only accepted이지만 장기 손실인 residual은 reject/negative label로 전환한다.
4. gate는 H3 rollout 단독 판정 대신 learned long-horizon value의 lower-confidence bound가 P-Stack보다 좋을 때만 RL을 채택한다. 불확실하면 exact P-Stack으로 fallback한다.
5. 170/190의 짧은 paired holdout에서 ranking 정확도와 false-positive 채택률을 먼저 검증한다. false-positive `0`과 P-Stack 대비 예측 개선 신호가 있을 때만 14,400초 full run을 재개한다.
6. 이 gate가 방향성을 통과한 뒤에만 accepted residual을 targeted 수집한다. 동일 H3 label의 24시간 추가 수집은 중단한다.

산출물:

- `results/contract_v6_native_pstack_direction_s0/dataset_audit.json`
- `results/contract_v6_native_pstack_direction_s0/dataset_gate.json`
- `results/contract_v6_native_pstack_direction_s0/rl_s0_170_190.csv`
- `results/contract_v6_native_pstack_direction_s0/traces/*.jsonl`
- `checkpoints/actor_contract_v6_native_pstack_direction_s0.pt`
- `checkpoints/actor_contract_v6_native_pstack_direction_s0_criticfix.pt`

## 9. 2026-08-25 long-horizon counterfactual 판정

### 현재 reward 계약

환경의 정책 step reward는 다음과 같다.

```text
r_t = -(urban_ttt_t + freeway_ttt_t)
```

control interval은 `180초`, IQL 기본값은 `gamma=1.0`, `reward_scale=0.01`이다. 따라서 자연 종료까지 끊기지 않은 trajectory의 return은 scale을 제외하면 `-total TTT`와 같다. safety abort, wall-clock abort, solver error에는 기본 `5,000` failure cost가 추가되지만, 현재 contract-v6 946개 transition은 natural 또는 collection time limit로만 끝나 이 penalty를 포함하지 않는다.

다만 기존 로더는 환경 종료가 아닌 `collection_time_limit`도 `done=1`로 사용했다. 19개 episode 중 natural은 9개, collection time limit는 10개였으며, 후자의 마지막 transition에서 critic bootstrap이 잘렸다. IQL 로더를 `collection_time_limit_bootstrap_v1`로 교정해 실제 terminal 수는 `19 -> 9`가 됐다.

reward가 total TTT와 정렬돼도 actor가 자동으로 장기 최적 action을 찾는 것은 아니다. 현재 actor는 Q를 직접 최대화하지 않고 critic advantage로 가중된 behavior target MSE를 학습한다. 따라서 H3 gate가 장기 손실 action을 positive target으로 넘기면, sparse offline support와 부정확한 critic 아래에서는 해당 action을 그대로 모방할 수 있다.

### generator와 H12 결과

`generate_long_horizon_labels.py`는 gate가 채택한 nonzero residual에서 두 branch를 만든다.

1. candidate: 해당 residual을 한 번 적용
2. baseline: 같은 state에서 native P-Stack을 강제 적용
3. 이후 양쪽 모두 native P-Stack 폐루프로 재최적화
4. H1/H3/H6/H12 누적 TTT와 terminal inventory를 비교

H12는 `12 x 180 = 2,160초`다. 양성 기준은 P-Stack 대비 gain이 `max(0.1 veh-h, 0.1%)`를 넘고 terminal inventory가 증가하지 않는 것이다.

기존 accepted residual 21개 중 실제 단일-query 배포 경로를 exact replay한 것은 12개였다. 결과는 장기 양성 7개, H3 false-positive 5개다. 나머지 9개는 수집기가 `optimizer_anchor_action()` preview 후 `env.step()`에서 P-Stack을 다시 호출하면서 preview의 내부 상태 변경이 transition에 섞여 exact replay에 실패했다. preview를 deep-copy 기반 side-effect-free query로 수정했고 실제 P-Stack 대조에서 observation, reward, anchor 오차 `0`을 확인했다.

| Scenario | exact labels | H12 positive | H3 false-positive | mean H12 gain |
|---|---:|---:|---:|---:|
| sweet_155_w60 | 6 | 4 | 2 | `+0.527 veh-h` |
| sweet_170_incident_w60 | 3 | 3 | 0 | `+24.773 veh-h` |
| sweet_190_w60 | 3 | 0 | 3 | `-21.763 veh-h` |
| Total | 12 | 7 | 5 | `+1.016 veh-h` |

original H3 gain과 H12 gain의 상관은 Pearson `-0.113`, Spearman `-0.231`이다. 190의 세 action은 모두 H3 양성이었지만 H12 gain이 `-42.925/-0.162/-22.201 veh-h`로 뒤집혔다. H3 gain 크기는 장기 action ranking에 사용할 수 없다.

### 판정과 다음 순서

현재 21개 accepted target은 확정 양성 7개, 확정 false-positive 5개, 배포경로 비재현 9개다. 이 데이터를 그대로 재학습하는 작업은 중단한다. H12 양성 7개도 full-to-end 보증이 아니므로 최종 positive label이 아니라 새 수집 방향을 정하는 seed로만 사용한다.

1. side-effect-free preview와 timeout bootstrap 계약으로 짧은 clean directional pilot을 다시 수집한다.
2. accepted event 발생 시 live pre-step state에서 H12 branch를 즉시 평가하거나, exact branch가 가능한 state snapshot을 저장한다.
3. actor supervision에는 H12 positive만 넣고 H12 false-positive는 reject/negative set으로 분리한다.
4. learned gate는 P-Stack 대비 long-horizon gain의 conservative lower bound가 양수일 때만 RL을 채택한다.
5. 170/190 paired holdout에서 false-positive `0`과 P-Stack 개선 신호를 확인한 뒤에만 14,400초 full run을 재개한다.
6. full run이 P-Stack을 이긴 뒤에만 수집 시간을 늘리고 3 seed x 5 scenario로 확장한다.

산출물:

- `results/contract_v6_native_pstack_direction_s0/counterfactual_h12.json`
- `results/contract_v6_native_pstack_direction_s0/counterfactual_h12.csv`
- `rl_leader/generate_long_horizon_labels.py`
- `src/tests/test_long_horizon_counterfactual.py`

## 10. 2026-08-25 contract-v7 clean 24시간 파이프라인

side-effect-free optimizer preview, collection-time-limit bootstrap, H12 positive actor supervision을 하나의 실행 계약으로 묶었다. `work/run_contract_v7_clean_24h_pipeline.ps1`이 다음 단계를 순차 실행한다.

1. 8 worker contract-v7 residual 수집
2. raw audit 및 minimum-transition gate
3. accepted residual의 H1/H3/H6/H12 exact counterfactual 생성
4. H12 positive만 actor supervision에 남긴 NPZ relabel
5. relabeled audit 및 gate
6. seed 0 IQL 40,000 update
7. 170/190 14,400초 full-run

파이프라인은 2026-08-25 17:15:33 KST에 PID `29080`으로 시작했다. 수집 deadline은 2026-08-26 17:15:33 KST다. worker PID는 `9304, 17124, 29864, 30424, 35036, 964, 33616, 35028`이며 시작 직후 `8/8` alive와 빈 stderr를 확인했다.

재학습은 자동이지만 무조건 수행하지 않는다. 다음 pretrain gate를 모두 통과해야 한다.

- raw transition `6,000+`
- generator replay error `0`
- accepted residual의 H12 label coverage `100%`
- H12 positive `20+`
- `sweet_170_w60`, `sweet_190_w60` 각각 H12 positive 존재

상태와 로그:

- `results/contract_v7_clean_24h_h12_s0/pipeline_state.json`
- `results/contract_v7_clean_24h_h12_s0/pipeline.out.log`
- `results/contract_v7_clean_24h_h12_s0/pipeline.err.log`
- `data/contract_v7_clean_24h_v1/monitor.log`

예정 산출물:

- `data/contract_v7_clean_24h_v1/worker_*.npz`
- `results/contract_v7_clean_24h_h12_s0/counterfactual_h12.json`
- `data/contract_v7_clean_24h_h12_v1/worker_*.npz`
- `checkpoints/actor_contract_v7_clean_24h_h12_s0.pt`
- `results/contract_v7_clean_24h_h12_s0/rl_s0_170_190.csv`

## 11. 2026-08-27 contract-v7 H12 복구 및 재학습

24시간 수집은 `9,365` transition, validity `100%`로 raw gate를 통과했다. H12
generator는 accepted residual `218`개 중 유효 라벨 `200`개를 생성했고, 양성은
`66`, H3 false-positive는 `134`였다. `worker_1000` episode 3의 마지막 transition은
수집기가 `wall_clock_abort` 시 추가한 `5,000` failure cost 때문에 reward replay가
불일치했고, 해당 episode의 accepted residual `18`개는 결과에서 제외됐다.

`wall_clock_abort`는 교통 MDP의 실패가 아니라 외부 계산시간 truncation이다. 따라서
다음 계약으로 교정했다.

- generator parity는 wall-clock failure cost를 제거한 환경 reward를 비교한다.
- IQL critic reward에서도 wall-clock failure cost를 제거한다.
- `collection_time_limit`과 `wall_clock_abort` 모두 bootstrap한다.
- 기존 실패 episode의 미확정 residual `18`개는 양성으로 추정하지 않고 actor
  supervision에서 격리한다.
- relabel completeness는 `200 valid + 18 quarantined = 218 accepted`로 검증한다.

시나리오별 H12 양성은 155 `4`, 170 incident `22`, 170 skew `5`, nominal 170
`0`, 190 `35`다. nominal 170에 양성이 없다는 사실은 숨기지 않으며, 해당 셀의
local residual은 actor target에 들어가지 않는다. pretrain gate는 전체 양성 `20+`,
170/190 label coverage, 190 양성 존재를 요구한다. nominal 170 개선 여부는 full-run
평가에서 별도로 판정한다.

관련 회귀 테스트 `63`개와 relabeled dataset gate가 통과했다. actor supervision은
P-Stack zero residual과 H12-positive residual을 합쳐 `2,509` transition이다. IQL seed
0은 `40,000` update를 완료해
`checkpoints/actor_contract_v7_clean_24h_h12_s0.pt`를 생성했고, 현재 native P-Stack
anchor 조건의 `sweet_170_w60`, `sweet_190_w60` full-run 평가도 완료했다. 최종 판정은
아래 12절에 이어서 기록한다.

## 12. 2026-08-27 Phase 0/1 full-run 및 centralized-oracle 판정

### Matched full-run

공통 warmup `109.703 veh-h`를 제외한 75 controlled step에서 frozen-v7 RL은
P-Stack보다 nominal 170에서 `+52.891 veh-h (+1.354%)`, 190에서
`+39.797 veh-h (+0.592%)` 나빴다. 같은 계약의 P-CENT는 P-Stack보다 각각
`93.675 veh-h (2.398%)`, `998.614 veh-h (14.845%)` 좋았다.

### 현재 flat actor를 더 학습하면 안 되는 이유

accepted event 5개의 actor residual L2는 `0.0031~0.0087`이었고 follower physical
response는 모두 adapter anchor와 exact alias였다. 하지만 adapter anchor 자체는 native
P-Stack을 5개 상태 중 한 번도 exact round-trip하지 못했다. normalized response RMSE는
`0.137~1.017`이었다. 현재 정책은 실질적으로 75-D continuous price policy가 아니라
`{exact-zero native P-Stack, lossy adapter reconstruction}` binary branch selector다.

따라서 데이터 양, IQL exploration, critic 개선을 논하기 전에 reference, trust radius,
selected P-Stack/PFO branch, certificate, follower runtime state를 보존하는 새 action
contract가 필요하다. 기존 75-D checkpoint와 dataset은 새 schema 학습에 섞지 않는다.

### Centralized data를 쓰는 범위

matched P-CENT 170/190 제어를 deterministic replay해 각 75행의 teacher artifact를
생성했다. observation은 238-D, follower/dual-excluded state는 205-D, physical action은
30-D이고 replay parity는 전 행 exact다. 변하는 축은 green 5, offset 5, meter 4뿐이며
VSL 16축은 고정, budget은 항상 0이다.

이 artifact는 `state -> centralized physical action` BC baseline과 trajectory ceiling에는
사용할 수 있다. 그러나 budget/price 정답은 포함하지 않으며, legacy
`data/pcent_teacher`는 post-action state temporal leakage와 provenance 결손 때문에 최종
학습에 사용하지 않는다. 새 P-CENT run은 `pcent_teacher_contract_v1` source hash를
저장한다.

### Same-state executable oracle 결과

P-CENT 방향/반대 방향의 owner/family price proposal 352개를 실행한 결과 state별 고유
follower response는 합계 47개였다. five-event H12 paired test에서 최선의 실행 가능한
price candidate는 170/9 `+15.679`, 170/13 `+69.942`, 190/25 `+11.330 veh-h`로
P-Stack을 이겼다. 170/25는 개선 후보가 없었고, 190/30은 모든 proposal이 하나의
response로 붕괴했다. frozen actor는 `1/5`, structured candidates는 `3/5` 장기
양성이었다.

Direct P-CENT one-step action도 `2/5`만 장기 양성이었다. 170/9에서는 P-CENT에서 더
먼 후보가 좋았고, 190/25에서는 P-CENT 방향 이동량을 키울수록 H12가 악화됐다.
따라서 P-CENT는 후보 방향과 upper-bound 정보이지 각 state의 배포 정답이 아니다.
H3 gate 역시 장기 양성의 false-negative와 false-positive를 모두 만들었다.

### 다음 실행 순서

1. **Exact action contract:** zero special-case 없이 native P-Stack의 response, control,
   follower next-state를 `100% exact` round-trip한다. 실패하면 이후 단계와 데이터 수집을
   중단한다.
2. **No-learning executable oracle:** P-Stack, response-diverse structured perturbation,
   optional inverse-certified price를 same-state에서 실행하고 physical response와 follower
   post-memory로 deduplicate한다. H1/H3/H6/H12 및 일부 remaining-horizon label을 만든다.
3. **Conservative ranker:** raw 75-D actor 대신 deployable candidate의 pairwise H12 ranker
   또는 contextual selector를 학습한다. P-Stack 대비 gain lower-confidence bound가
   양수이고 inventory debt가 없을 때만 개입한다.
4. **Guarded DAgger:** ranker가 방문하는 state에 한해 same-state P-CENT query와
   executable candidate label을 추가한다. 무작위 24시간 수집은 하지 않는다.
5. **Closed-loop gate:** pilot에서 false-positive intervention `0`, validity `100%`를 먼저
   확인하고, seed 0의 170/190 full run이 모두 P-Stack을 이긴 뒤에만 3 seed x 5
   scenario와 장시간 수집으로 확장한다.

상세 수치와 artifact 경로는
`results/rl_phase0_implementation_20260827/PHASE0_PHASE1_REPORT.md`에 정리했다.

## 13. 2026-08-28 exact-native contract와 P-CENT guided oracle pilot

### Exact-native 실행 계약

기존 flat `encode(anchor) + residual` 경로는 native reference, trust radius, physical
quadratic coefficient, raw budget intent, follower candidate seed와 selected branch를 잃었다.
다음 hard migration boundary로 교정했다.

- action schema: `coordination_action_v6_exact_native_potential`
- response contract: `rl_pstack_b13_exact_native_potential_v5`
- anchored residual: `native_coordination_delta_v2`
- anchor context: `pstack_same_state_anchor_context_v3_hidden_follower_fingerprint`
- collection transition: `pstack_anchor_gate_exact_common_follower_seed_v5`
- dataset/checkpoint: `rl_coordination_dataset_v4_exact_native` /
  `rl_coordination_checkpoint_v4_exact_native`

170과 190의 initial native P-Stack 대 coordination-zero 실행은 control, physical state,
observation, reward와 동적 follower memory가 모두 exact였다. 새 collector 1-row smoke도
teacher response replay L-infinity `0`, validity `100%`, anchor fingerprint 검증을 통과했다.
관련 집중 테스트는 `103/103` 통과했다. 이전 dataset/checkpoint는 새 runtime에서
fail-closed 처리하며 변환하지 않는다.

### 이산 branch가 필요한 이유

`sweet_170_w60` policy step 13에서 native P-Stack은 `fallback_pfo`를 선택했다. 이때
native 제어는 `N_UF=6000`, coordination-zero follower는 `N_UF=5810.459`였으며 response와
follower memory도 달랐다. 이는 residual scale 오류가 아니라 PFO가 가격 potential로
표현되는 leader candidate가 아니기 때문이다.

따라서 deployable action은 다음 두 수준으로 정의한다.

```text
branch = native_anchor | coordination
coordination branch에서만 residual delta를 해석
불확실하거나 장기 이득이 없으면 native_anchor
```

`fallback_pfo`에서 coordination-zero는 zero identity가 아니라 명시적 branch-switch
후보다. direct P-CENT physical control은 deployable follower-memory transition이 없으므로
H>1 P-Stack recovery label에서 제외하고 one-step reference로만 기록한다.

Oracle artifact는 `pcent_guided_follower_oracle_v2_branch_complete`로 올렸다. 성공
artifact는 요청한 모든 horizon을 정확히 완료하고, 각 checkpoint의 persistent follower
memory hash, source isolation, reverse-order replay, branch contract를 모두 통과해야 한다.
`--skip-order-check` 결과나 terminal 근처 truncated rollout은 `passed`가 될 수 없다.

v2 H1 integration은 170 step 0의 `coarse`와 step 13의 실제 `fallback_pfo`에서 모두
통과했다. step 0은 coordination-zero가 native identity였고, step 13은 둘이 다르며
`requires_explicit_branch_selector=true`였다. 두 경우 모두 terminal follower memory와
order check가 exact였고 conservative H1 선택은 `native_anchor`였다.

### Same-state H12 pilot 결과

두 pilot 모두 native P-Stack trajectory의 사전 지정 state에서 P-CENT를 새로 질의했다.
과거 accepted RL trace나 v4 artifact는 사용하지 않았다. 모든 후보는 독립 clone에서
follower를 실행했고 source mutation, candidate order invariance, H12 identity replay를
exact 검증했다.

아래 수치는 v1 exact-native performance pilot이다. 이후 v2 변경은 simulation/action
동작이 아니라 horizon completeness, persistent-memory provenance, branch/order fail-closed
검증을 추가한 것이다. 학습 label을 배포하기 전에는 선택된 balanced pilot state를 v2
H12로 새로 생성하며, 기존 JSON을 새 format으로 변환하지 않는다.

| Scenario/step | Native branch | Oracle choice | H1 gain | H3 gain | H6 gain | H12 gain | H12 inventory delta |
|---|---|---|---:|---:|---:|---:|---:|
| 170 / 9 | coarse | `away:signal:C:0.5` | `+0.050` | `+0.067` | `-0.001` | `+15.723` | `-96.992 veh` |
| 170 / 13 | fallback_pfo | `coordination_zero` | `+0.001` | `+0.065` | `+1.382` | `+14.727` | `-40.797 veh` |

step 9 후보는 `urban.C.g_green=-0.5`, `urban.C.g_offset=+0.5`이며 P-CENT normalized
response RMSE를 `0.5105 -> 0.4768`로 줄였다. 그러나 H1/H3는 conservative margin 미달,
H6는 사실상 동률이고 H12에서만 명확한 양성이 됐다. step 13의
`toward:meter:0.5`도 H12 `+10.499 veh-h`였지만 coordination-zero보다 나빴다.

이 결과는 marginal-price instrument가 원천적으로 틀렸다는 가설을 반박하지만,
P-CENT physical action이 price 정답이라는 뜻은 아니다. step 13의 주 이득은 branch
선택에서 나왔고 step 9도 P-CENT-guided 후보와 target-independent 탐색의 대조군이 아직
없다. 확정 가능한 결론은 다음 세 가지다.

1. follower가 실제로 실행할 수 있는 coordination action 중 P-Stack보다 좋은 행동이
   존재한다.
2. one-step/H3/H6 label만으로는 H12 positive를 놓칠 수 있다.
3. 기존 성능 악화는 데이터 양만의 문제가 아니라 lossy action contract, PFO branch
   혼합, short-horizon supervision이 함께 만든 구조적 문제다.

### 다음 실행 순서

1. **Branch-aware label schema:** `native_anchor/coordination` 이산 label과 coordination
   residual을 분리한다. PFO를 연속 price target으로 회귀하지 않는다.
2. **Oracle ablation:** 같은 pre-state에서 P-CENT-guided, target-independent structured,
   local-Jacobian, random/orthogonal 후보를 동일 개수로 평가해 P-CENT 방향의 추가 가치를
   분리한다.
3. **Staged label budget:** H1 follower probe는 전체 후보에 적용하되 response와 follower
   memory로 deduplicate하고, diversity/uncertainty를 통과한 top `2~3`개만 H12로 올린다.
   이번 single-event 비용은 step 9 `1,875초`, step 13 `2,486초`였다.
4. **Small balanced pilot:** 다섯 scenario의 early/peak/recovery에서 먼저 소수 state를
   균형 표집한다. 모든 transition에 H12를 붙이거나 24시간부터 수집하지 않는다.
5. **Conservative selector/ranker:** branch head와 candidate ranker를 학습하고 H12 gain
   lower-confidence bound가 양수이며 inventory debt가 없을 때만 coordination을 선택한다.
6. **Closed-loop gate:** fresh 170/190에서 false-positive intervention `0`, validity `100%`,
   P-Stack 대비 paired TTT 개선을 확인한 뒤에만 3 seed x 5 scenario와 데이터 규모를
   늘린다.

주요 산출물:

- `results/rl_phase0_implementation_20260828/anchor_context_parity_v13_hidden_follower/`
- `results/rl_phase0_implementation_20260828/contract_v4_exact_native_smoke_v2/`
- `results/rl_phase0_implementation_20260828/pcent_guided_oracle_170_step9_h12/`
- `results/rl_phase0_implementation_20260828/pcent_guided_oracle_170_step13_h12_v2/`
- `results/rl_phase0_implementation_20260828/pcent_guided_oracle_v2_h1_smoke_step0/`
- `results/rl_phase0_implementation_20260828/pcent_guided_oracle_v2_fallback_h1/`
- `rl_leader/diagnose_pcent_guided_oracle.py`

## 14. 2026-08-28 branch-aware linear-price ablation과 다음 판정

### 구현된 label 및 후보 계약

기존 v4 IQL transition에 oracle 결과를 덧붙이지 않고 별도
`pcent_guided_oracle_labels_v1_branch_aware` dataset 계약을 만들었다. native P-Stack은
`native_anchor` 이산 branch이며 continuous residual은 `null/invalid`다. coordination-zero는
coordination branch의 finite zero vector라서 둘을 같은 action으로 취급하지 않는다.
P-CENT control/response는 candidate provenance로만 저장하고 deployable ranker feature에는
포함하지 않는다.

첫 ablation은 budget, raw budget, release certificate와 quadratic coefficient를 고정하고
30-D linear price만 바꿨다. 각 생성기에 logical H1 slot 8개와 L2 radius `0.5`를 동일하게
배정했다.

- `pcent_sign`: green/offset/meter/VSL P-CENT 방향과 반대 방향
- `structured`: target-independent family bundle 축
- `pcent_jacobian`: 4개 secant와 4개 regularized inverse
- `orthogonal_random`: state-hash seed의 직교 랜덤 4축과 양방향

동일 residual의 물리 solve는 공유하지만 logical query slot은 각 생성기에 그대로
귀속한다. H12 selector는 P-CENT target을 입력으로 받지 않고 H1 TTT, inventory,
residual hash만 사용했다. 후보 probe와 H12 replay는 response와 persistent follower memory가
모두 exact여야 한다.

리뷰 후 fail-closed 계약도 강화했다.

- selector dedupe key에 post-step physical-state SHA를 추가했다.
- pool runtime/forecast/manifest SHA, source isolation, identity/zero parity를 필수화했다.
- coordination row의 residual SHA, post follower/physical SHA, requested budget/certificate
  검사를 필수화했다.
- H12-selected row는 H1 response와 follower-memory replay 증거 없이는 ranker input이 될 수
  없다.
- legacy reachability attribution은 `coarse/refined` 이외 branch, truncated horizon,
  follower-memory mismatch를 fail-closed한다.
- 관련 집중 테스트는 `126/126` 통과했다. follower-memory와 physical-state SHA를 모두
  포함한 최종 170 step 0 smoke도 33 rows, positive 0으로 validator를 통과했다.

### 170 step 9 H12 결과

동일 state의 native anchor fingerprint는
`35f97d43706b6a5d21d2d24eb92046956ea09698b93575d4dfd13e677681ac7f`다. native H12 TTT는
`1399.115866 veh-h`, terminal inventory는 `2785.204625 veh`였다. P-CENT는 235회 평가 후
`converged=false`였고 objective는 `4143.227393`이다. 32 logical candidate는 22 physical
residual로 줄었고, reverse-order replay와 source isolation은 exact였다.

| Generator | H12 candidate | H1 gain | H3 gain | H6 gain | H12 gain | H12 inventory delta |
|---|---|---:|---:|---:|---:|---:|
| P-CENT sign | `green:toward` | `+0.080755` | `+0.020365` | `-0.036981` | `-0.041487` | `+0.001070 veh` |
| Structured | `green:negative` | `+0.044609` | `-0.006084` | `-0.061314` | `-0.065784` | `+0.001063 veh` |
| Local Jacobian | `green:secant` | `+0.015085` | `-0.066583` | `-0.107220` | `-0.108183` | `+0.000031 veh` |
| Orthogonal random | `q2:negative` | `+0.078918` | `-0.007185` | `-0.141272` | `-0.521742` | `+2.497980 veh` |

모든 후보가 H12에서 P-Stack보다 나빠 `native_anchor`가 winner다. 따라서 이 candidate
pool에는 학습할 positive coordination label이 없다. 실행시간은 전체 `3221.328초`, event
내부 `2844.984초`였다.

이 H12 artifact는 provenance hardening 직전에 생성돼 수치 진단용으로 보존한다. 당시
runner가 H1 response/memory replay를 실행 중 fail-closed로 확인했지만 그 증거를 JSON row에
명시적으로 쓰지 않았으므로, 강화된 validator는 training ingest에서 의도적으로 거부한다.
실패 후보군을 같은 형태로 다시 54분 돌리지 않고 다음 v2 후보군부터 강화 계약으로 새로
생성한다.

### 이 결과가 marginal price 실패를 뜻하지 않는 이유

같은 anchor에서 이전 v1 oracle은 `away:signal:C:0.5`를 찾아 H12 `+15.722691 veh-h`,
inventory `-96.992107 veh`를 기록했다. 두 run은 anchor fingerprint, native H12 TTT,
P-CENT objective와 235회 평가가 모두 exact하게 같다. 차이는 candidate support다.

이전 양성 residual은 다음 owner-local cross block이다.

```text
urban.C.g_green  = -0.5
urban.C.g_offset = +0.5
L2               = sqrt(0.5) ~= 0.7071
```

이번 equal-L2 pool에는 이 action이 없다. P-CENT green 후보는 B/C/D/F green 4축에
`0.25`씩 분산됐고 offset은 별도 후보였다. structured/Jacobian green도 다섯 signal을
동시에 움직였다. 즉 global family bundle이 알려진 유효한 owner-local green-offset
상호작용을 지웠다.

selector도 알려진 양성을 놓치는 구조였다. 이전 C 후보의 gain은 H1 `+0.050`, H3
`+0.067`, H6 `-0.001`로 어느 short-horizon margin도 통과하지 않았지만 H12에서만
`+15.723`으로 커졌다. 이번 `pcent_sign:green:toward`는 H1 `+0.081`이라 H1 최저-TTT
규칙에서 C 후보보다 먼저 선택되지만 H12는 `-0.041`이다. 따라서 H1 top-1은 H12
candidate selector로 사용할 수 없다.

판정은 다음과 같다.

1. 현재 결과는 marginal-price instrument 자체를 기각하지 않는다.
2. P-CENT physical action이나 그 sign도 정답이 아니다. 알려진 양성은 P-CENT의 반대
   방향에서 나왔고 centralized solve도 `converged=false`였다.
3. centralized 정보는 어떤 owner/block을 probe할지 제안하는 privileged guide로만 쓰고,
   deployable 정답은 follower-executable H12 counterfactual로 결정해야 한다.
4. 현재 병목은 데이터 양보다 candidate support와 long-horizon selection이다.

### 다음 실행 순서

1. **Known-positive canary:** 강화 계약에서 170 step 9의 기존 C green-offset residual을
   먼저 H12 재생한다. anchor fingerprint, P-Stack H12와 candidate H12가 기존 artifact와
   exact해야 한다. 실패하면 후보 확장이나 학습을 중단하고 replay 차이부터 조사한다.
2. **Sparse owner-block v2:** global family bundle 대신 owner별 urban
   `(green, offset)` 5개와 ramp별 `(meter, VSL)` 4개를 독립 block으로 만든다. 첫 urban
   ablation은 각 active coordinate magnitude `0.5`, block L2 `sqrt(0.5)`로 맞춰 알려진 C
   후보를 support에 포함한다. P-CENT-guided sign과 target-independent sign을 분리한다.
3. **H1 top-1 폐기:** 모든 response/physical-diverse 후보를 H1에서 검증하되 margin으로
   제거하지 않는다. sparse urban pilot은 H3까지 전 후보를 보내고, H12에는 H3 TTT 최선,
   inventory 최선, response-diversity 최선과 known-positive canary를 보낸다. H6는 C 후보를
   탈락시키므로 pruning gate로 쓰지 않는다.
4. **Balanced small pilot:** canary 재현 후 155, 170, 170 incident, 170 skew, 190의
   frozen early/peak/recovery state에서 소수 pool만 만든다. 24시간 수집은 아직 하지 않는다.
5. **Conservative branch ranker:** 여러 pool에서 positive/negative H12 label이 확보된 뒤에만
   `native_anchor` default의 pairwise ranker를 학습한다. P-CENT provenance는 입력에서
   제외하고, lower-confidence bound가 양수이며 inventory debt가 없을 때만 coordination을
   선택한다.
6. **Quadratic price 보류:** sparse linear owner-block support가 여러 state에서 P-Stack을
   이기는지 먼저 판정한다. 그 뒤 동일 block/radius/selector를 유지한 별도 ablation에서만
   quadratic/cross term을 추가한다.

주요 산출물:

- `results/rl_phase0_implementation_20260828/linear_price_candidate_ablation_v1_step9_h12/`
- `results/rl_phase0_implementation_20260828/linear_price_candidate_ablation_v1_post_physical_step0_smoke/`
- `rl_leader/oracle_label_contract.py`
- `rl_leader/oracle_candidate_ablation.py`
- `rl_leader/diagnose_candidate_ablation.py`
- `src/tests/test_oracle_label_contract.py`
- `src/tests/test_oracle_candidate_ablation.py`
- `src/tests/test_reachable_candidate_attribution.py`

## 15. 2026-08-28 known-positive canary 재현과 owner-block v2 구현

### 강화 계약 canary 결과

`sweet_170_w60`의 policy step 9에서 과거 양성 residual을 현재 branch-aware 계약으로
다시 실행했다.

```text
urban.C.g_green  = -0.5
urban.C.g_offset = +0.5
L2               = 0.7071067691
```

anchor fingerprint는 과거 artifact와 같은
`35f97d43706b6a5d21d2d24eb92046956ea09698b93575d4dfd13e677681ac7f`였고, 네 horizon의
수치도 exact하게 재현됐다.

| Horizon | Candidate TTT | P-Stack TTT | TTT gain | Inventory delta | Positive |
|---:|---:|---:|---:|---:|---|
| H1 | `97.878415` | `97.928712` | `+0.050297` | `0.000000` | no |
| H3 | `303.434537` | `303.501737` | `+0.067200` | `-1.411027` | no |
| H6 | `635.943466` | `635.942879` | `-0.000587` | `-2.127976` | no |
| H12 | `1383.393175` | `1399.115866` | `+15.722691` | `-96.992107` | yes |

H1 probe와 H12 rollout의 response, persistent follower memory SHA, post-step physical-state
SHA가 모두 exact했고 artifact validator를 통과했다. 따라서 이전 v1 실패의 원인은 source
state drift나 replay 구현 차이가 아니라 candidate support와 H1 top-1 selector다.

산출물:

- `results/rl_phase0_implementation_20260828/known_positive_canary_step9_h12_final_contract/`

### Sparse owner-block v2

초기 v2 smoke는 active coordinate magnitude `0.5`, block L2 `sqrt(0.5)`를 사용했다.
독립 리뷰 후 negative 결과의 support false-negative를 줄이기 위해 현재 library는 magnitude
`0.25/0.5`를 모두 포함한다.

- urban owner 5개마다 `(green, offset)` signed single-axis와 four-corner 후보를 만든다.
- full library에서는 ramp owner 4개마다 `(meter, VSL)`에 같은 axis/corner 후보를 추가한다.
- target-independent structured row와 P-CENT-guided toward/away row를 별도 provenance로
  유지한다. 같은 residual의 물리 solve는 공유한다.
- owner-block H12 실행은 selector를 사용하지 않고 cumulative H3 validity를 통과한 모든
  unique realized `(response, follower memory, physical state)` outcome을 H12로 보낸다. H3
  selector는 oracle support 확인 뒤 별도 ablation으로만 평가한다.
- known C canary 강제 선택은 `170/step 9` 회귀검사에만 쓰고 일반 pilot 통계에서는 다른 C
  structured residual과 동일하게 취급한다.
- H6는 label로 기록하되 pruning gate로 사용하지 않는다.

step 0 urban H1 smoke에서 20 structured + 10 guided logical row가 20 unique physical
residual로 deduplicate됐다. 첫 smoke에서 guided `away`의 비활성 좌표 `-0.0`이 `+0.0`과 다른
SHA를 만들어 25개로 잘못 세는 문제를 발견했고, candidate 생성 시 signed zero를
canonicalize한 뒤 다시 검증했다. 최종 smoke는 reverse-order replay, source isolation,
identity parity와 artifact validation을 모두 통과했으며 실행시간은 `109.75초`였다.
이 smoke 이후 candidate support와 H1/H3 replay metric binding 계약이 강화됐으므로, 해당
artifact는 구현 이력일 뿐 현재 코드 freeze의 training input으로 사용하지 않는다.

리뷰 반영 후 확장된 urban library는 owner 5개에 대해 `axis/corner x 0.25/0.5`의
80 structured row와 20 P-CENT-guided logical alias를 만든다. 최신 H1 결과는 다음과 같다.

| State | Logical rows | Exact residuals | Realized outcomes | Elapsed | Contract |
|---|---:|---:|---:|---:|---|
| 170 step 0 | 100 | 80 | 11 | `297.890초` | passed |
| 170 step 9 | 100 | 80 | 7 | `1100.985초` | passed |

step 9 anchor fingerprint는 canary와 같은 `35f97d...681ac7f`다. known C residual은
structured row와 privileged P-CENT-away alias로 모두 보존됐고 H1 gain `+0.050296532`를
exact 재현했다. 해당 realized outcome에는 8개 price alias가 묶였다. 전체 7개 outcome 중
6개는 alias 8개, 하나는 alias 52개로, raw price보다 realized follower response를 action
단위로 써야 한다는 근거가 더 강해졌다.

최신 산출물:

- `results/rl_phase0_implementation_20260828/owner_block_v2_urban_axes_corners_step0_h1_smoke/`
- `results/rl_phase0_implementation_20260828/owner_block_v2_urban_axes_corners_step9_h1/`

### Exhaustive H12와 remaining-horizon 결과

step 9의 7개 unique realized coordination outcome을 모두 H12까지 평가했다. native anchor를
포함하면 8개 branch이며, artifact validation과 모든 H1/H3 replay gate를 통과했다. 총
실행시간은 `7516.281초`였다.

| Realized outcome | H12 TTT gain | H12 inventory delta | Positive |
|---|---:|---:|---|
| C green decrease | `+15.722691` | `-96.992107` | yes |
| C green increase | `0.000000` | `0.000000` | no |
| D green decrease | `0.000000` | `0.000000` | no |
| F green decrease | `0.000000` | `0.000000` | no |
| A green decrease | `-0.026293` | `-0.000005` | no |
| B green decrease | `-0.041487` | `+0.001070` | no |
| A green increase | `-0.108183` | `+0.000031` | no |

유일한 양성 outcome에는 P-CENT C away, C green 단일축, C green-offset 양쪽 corner와
magnitude `0.25/0.5`를 합친 8개 alias가 묶였다. response, persistent follower memory,
post-physical SHA가 모두 같으므로 이전의 "C green-offset 결합효과" 해석은 기각한다. 이
state에서 실효적인 개입은 **C green price 감소**이고 offset 및 magnitude 차이는 follower
response를 추가로 바꾸지 않았다. 장기 실행에는 privileged target을 쓰지 않는 최소 alias
`structured:urban:C:axis:green-negative:m0.25`를 선택했다.

이 최소 개입을 policy step 9에서 한 번 실행하고 P-Stack으로 복귀해 남은 66 policy step을
simulation 종료까지 평가했다. H12의 candidate/native TTT, terminal inventory, follower-memory
SHA, physical-state SHA 여덟 항목이 source artifact와 exact 일치한 뒤에만 종료 결과를
인정했다.

| Horizon | Candidate TTT | P-Stack TTT | TTT gain | Inventory delta | Verdict |
|---|---:|---:|---:|---:|---|
| step 9 to 14,400 s | `3191.884677` | `3310.543242` | `+118.658565` (`+3.584%`) | `-1.271157` | positive |

따라서 이 state에서는 H12 양성이 recovery 종료까지 유지됐고 inventory guard도 통과했다.
이는 raw 75-D price regression보다 realized follower response를 action 단위로 삼는 구조가
타당하다는 첫 장기 양성 증거다. 다만 **한 번의 intervention + P-Stack continuation** 결과이며,
14,400초 동안 RL action을 반복 적용한 policy 평가로 해석해서는 안 된다. remaining-horizon
두 branch의 실행시간은 `5043.421초`였다.

산출물:

- `results/rl_phase0_implementation_20260828/owner_block_v2_urban_axes_corners_step9_exhaustive_h12/`
- `results/rl_phase0_implementation_20260828/owner_block_v2_urban_step9_remaining_horizon/`
- `rl_leader/diagnose_oracle_remaining_horizon.py`
- `src/tests/test_oracle_remaining_horizon.py`

H12는 12 policy step, 즉 `2160초` 동안의 **one-shot price impulse + P-Stack continuation
screening label**이다. 14,400초 반복 RL policy의 우위를 의미하지 않는다. 이후 다른 state에서
얻는 H12 positive도 remaining-horizon/recovery 종료까지 재평가한 뒤에만 최종 positive로
승격한다.

산출물:

- `results/rl_phase0_implementation_20260828/owner_block_v2_urban_step0_h1_smoke_canonical/`
- `rl_leader/oracle_candidate_ablation.py`
- `rl_leader/diagnose_candidate_ablation.py`
- `src/tests/test_oracle_candidate_ablation.py`

### 다음 gate

완료된 gate는 독립 리뷰 반영, 확장 support H1 smoke, 7개 realized outcome exhaustive H12,
H12 양성의 simulation-end 연장 판정이다. 다음 순서는 아래와 같다.

1. 155, 170, 170 incident, 170 skew, 190에서 early/peak/recovery frozen state를 균형
   표집하고, state별 native branch와 anchor fingerprint를 manifest로 고정한다.
2. urban owner-block을 먼저 H1 전수 probe하고 realized response/memory/physical state로
   deduplicate한다. 그 뒤 freeway `(meter, VSL)` owner-block도 같은 계약으로 추가한다.
3. H3는 전 unique outcome에 적용하고 H12는 diversity, uncertainty, short-horizon
   disagreement를 만족한 소수 후보에만 적용한다. H1 top-1은 pruning 기준으로 쓰지 않는다.
4. H12 positive만 remaining-horizon/recovery gate로 승격한다. 이 고비용 gate는 모든
   transition이 아니라 최종 positive 후보에만 사용한다.
5. scenario와 congestion phase를 모두 포함한 positive/negative response label이 확보된 뒤
   deployable branch classifier와 pairwise response ranker를 학습한다.
6. ranker는 `RL candidate vs P-Stack anchor`의 보수적 선택기로 평가하고, PFO fallback은
   별도 실험군으로 유지한다.
7. balanced pilot 전에는 24시간 추가 수집, raw 75-D actor 재학습, quadratic price 확장을
   시작하지 않는다.

## 16. 2026-08-28 balanced frozen-state pilot 시작

### Stratum 교정과 fail-closed manifest

초기 `policy step 0/9/30`의 early/peak/recovery 표기는 실제 observation 및 pulse 정의와
맞지 않았다. step 0의 simulation time은 `900초`이고 pulse fraction은 0이며, recovery
feature는 `5220초`부터 켜진다. 따라서 공통 stratum을 아래와 같이 교정했다.

| Stratum | Policy step | Simulation time | 의미 |
|---|---:|---:|---|
| ramp_up | 1 | `1080초` | pulse fraction `0.5` |
| plateau | 9 | `2520초` | pulse fraction `1.0` |
| recovery_boundary | 24 | `5220초` | recovery feature 시작 |

`sweet_170_incident_w60`에는 incident onset step 5 (`1800초`)와 incident end step 15
(`3600초`)를 별도 stratum으로 추가했다. 총 17개 frozen state다.

manifest v2는 hash만 저장하지 않는다. experiment contract payload, dependency file SHA,
action/observation schema, raw observation, anchor envelope, forecast, current demand, physical
snapshot, pulse/incident predicate, RL/optimizer controller fingerprint를 함께 저장한다. Pilot
runner는 candidate probe 전에 별도 P-Stack replay를 수행하고 이 payload를 canonical exact
비교한다. JSON tuple/list round-trip 때문에 raw dict 비교가 실패한 smoke를 통해 canonical
normalization 경계도 회귀 테스트로 고정했다.

### Native branch 분모

17개 state 중 coordination price candidate를 적용할 수 있는 coarse/refined anchor는 10개,
`fallback_pfo`라 제외되는 state는 7개였다.

| Scenario | ramp-up | plateau | incident onset/end | recovery boundary |
|---|---|---|---|---|
| 155 | excluded | eligible | - | excluded |
| 170 | excluded | eligible | - | eligible |
| 170 incident | eligible | eligible | both excluded | eligible |
| 170 skew | excluded | eligible | - | eligible |
| 190 | excluded | eligible | - | eligible |

따라서 balanced pilot의 coordination 분모는 15개나 17개가 아니라 **10개**다. 제외 state는
삭제하지 않고 branch-classifier negative/fallback 표본으로 manifest에 남긴다. 기존 170
step 9 exhaustive H12 artifact는 새 plateau state의 simulation step/time, branch, anchor,
runtime, forecast와 모두 exact 일치했다.

### Incident ramp-up H1 smoke

처음 보는 eligible state인 170 incident ramp-up에서 urban owner-block v2 H1 smoke를
실행했다. preflight와 사후 frozen-state check가 모두 exact 통과했다.

- logical candidates `100`, physical residuals `80`, realized outcomes `9`
- oracle probe elapsed `616.406초`, preflight 포함 runner elapsed `678.484초`
- 모든 H1 validity gate 통과
- unique outcome H1 gain 범위 `-0.013403 ~ +0.001608 veh-h`
- 최대 H1 gain은 B green 증가 response였지만 장기 margin과 비교하면 사실상 동률

이 artifact는 `screening_horizon=1`, `selection_valid=false`다. native winner나 H12 양성을
정의하지 않으며 response diversity와 validity만 사용한다. 9개 outcome은 H3 전수 평가로
보내되, H12는 H3 TTT/inventory, response diversity, short-horizon disagreement를 대표하는
소수 outcome만 승격한다.

산출물:

- `results/rl_phase0_implementation_20260828/balanced_owner_block_frozen_states_v2/manifest.json`
- `results/rl_phase0_implementation_20260828/balanced_owner_block_h1_smoke_v2_incident_rampup/`
- `rl_leader/balanced_oracle_manifest.py`
- `rl_leader/run_balanced_owner_block_pilot.py`
- `src/tests/test_balanced_oracle_manifest.py`
- `src/tests/test_balanced_owner_block_pilot.py`

### 다음 실행

1. incident ramp-up의 9개 unique outcome을 H3까지 전수 평가한다.
2. H12 승격 규칙을 frozen artifact로 기록하고 representative `2~3`개만 H12로 평가한다.
3. 같은 규칙이 170 plateau known-positive를 반드시 보존하는지 회귀검사한다.
4. 남은 8개 eligible state의 H1 probe를 시나리오별 병렬 실행한다.
5. freeway `(meter, VSL)` block은 urban H12 승격 규칙이 통과한 뒤 같은 state에 추가한다.

### Incident ramp-up H3 및 selective H12 결과

9개 realized outcome을 H3까지 전수 평가하고, H3 TTT 최선, H3 inventory 최선, native
response diversity 최선의 합집합을 H12로 승격했다. inventory와 diversity 기준이 같은 D
outcome을 선택해 실제 H12 실행은 2개였다. H1/H3 replay는 follower memory, physical state,
TTT, inventory까지 exact 통과했다.

| Candidate | H3 gain | H3 inventory delta | H12 gain | H12 positive |
|---|---:|---:|---:|---|
| B green decrease | `+0.062455` | `-0.550389` | `-0.641277` | no |
| D green increase | `+0.061760` | `-1.482756` | `-1.760528` | no |

두 short-horizon 양성은 H12에서 모두 뒤집혔다. 하지만 비영 response 8개의 normalized
native-response RMSE가 모두 `0.182574`로 같아 diversity criterion은 owner/sign을 구분하지
못했다. 따라서 나머지 7개를 negative로 간주할 수 없고 이 v1 selector는 calibration 실패다.
특히 기존 170 plateau의 C-green 양성은 H3 최선이라 보존되지만, 다른 state의 delayed C
effect를 보존한다는 보장은 없다.

다음 실행은 24시간 수집이나 ranker 학습이 아니다.

1. owner/sign coverage를 보장하거나 H6 disagreement를 추가한 selector v2를 만든다.
2. 이 incident ramp-up state에서는 selector recall을 측정하기 위해 9개 outcome H12
   exhaustive 결과를 정답으로 만든다.
3. v2가 incident ramp-up exhaustive와 170 plateau known-positive를 모두 보존할 때만 남은
   8개 eligible state로 확장한다.

산출물:

- `results/rl_phase0_implementation_20260828/balanced_owner_block_h3_selective_h12_v1_incident_rampup/`
- `rl_leader/evaluate_balanced_horizons.py`
- `src/tests/test_balanced_horizon_selection.py`

## 17. 2026-08-29 balanced urban pilot 완료

### Incident ramp-up exhaustive H12와 selector v2

v1 selector의 false-negative 가능성을 제거하기 위해 incident ramp-up의 native 포함 9개
realized outcome을 모두 H12까지 실행했다. 실행시간은 `3149.390초`였고 모든 replay 및
artifact gate를 통과했다. H12 positive는 없었다. 가장 나은 C green decrease도
`+0.916823 veh-h`였으나 required margin을 넘지 못했고, 나머지 outcome의 gain은
`+0.120435 ~ -1.760528 veh-h`였다. 따라서 v1이 놓친 incident positive는 없었지만,
v1이 고른 B green decrease와 D green increase는 각각 `-0.641277`, `-1.760528 veh-h`로
실제 상위 outcome을 대표하지 못했다.

selector v2는 native와 같은 identity outcome을 제외하고 **owner별 H3 TTT 최선 response를
하나씩** H12로 보낸다. 이 규칙은 incident ramp-up에서 urban A/B/C/D 네 owner를 보존하고,
기존 170 plateau의 known-positive C green decrease도 H3-best C outcome의 alias로 exact
보존했다. 이는 H3가 장기 정답이라는 가정이 아니라 owner coverage를 보장하는 bounded-cost
screening rule이다.

산출물:

- `results/rl_phase0_implementation_20260829/balanced_owner_block_exhaustive_h12_v1_incident_rampup/`
- `rl_leader/complete_balanced_h12.py`
- `rl_leader/evaluate_balanced_horizons.py`
- `src/tests/test_complete_balanced_h12.py`
- `src/tests/test_balanced_horizon_selection.py`

### 남은 8개 eligible state 실행 결과

남은 8개 state에서 owner-block v2의 80개 unique physical residual을 모두 H1 probe했다.
각 실행은 별도 P-Stack preflight와 사후 frozen-state exact check를 통과했다. selector v2로
H3 전 unique outcome을 평가하고 owner별 최대 5개를 H12로 승격한 결과는 아래와 같다.

| Scenario / stratum | Realized outcomes | H12 selected | H12 positive | Best H12 gain |
|---|---:|---:|---:|---:|
| 155 plateau | 11 | 5 | 1 | `+1.372296` |
| 170 recovery | 11 | 5 | 0 | `+0.356043` |
| 170 incident plateau | 5 | 2 | 0 | `-0.005819` |
| 170 incident recovery | 9 | 4 | 0 | `+0.480286` |
| 170 skew plateau | 4 | 2 | 0 | `+0.012826` |
| 170 skew recovery | 9 | 5 | 0 | `+0.630161` |
| 190 plateau | 3 | 1 | 0 | `-0.053208` |
| 190 recovery | 11 | 5 | 4 | `+15.785008` |

8개 state에서 H12로 승격된 outcome은 총 29개이고 H12 positive는 5개다. H3 gain이 양수인
17개 중 5개는 H12에서 0 이하로 뒤집혔고, H3가 0 이하인데 H12에서 양수가 된 outcome도
2개였다. 따라서 H3는 owner별 계산량 제한용 selector에는 쓸 수 있지만 training target이나
최종 accept gate로 쓸 수 없다.

산출물:

- `results/rl_phase0_implementation_20260829/balanced_h1_v2_*/`
- `results/rl_phase0_implementation_20260829/balanced_h3_h12_v2_*/`

### H12 positive의 simulation-end 판정

새 balanced sidecar gate는 원본 H1 artifact와 manifest SHA를 확인하고, candidate/native의
H12 TTT, inventory, follower memory SHA, physical-state SHA가 exact 재현된 경우에만 한 번의
price intervention 뒤 P-Stack으로 복귀해 `14,400초`까지 계산한다.

155 plateau의 B green decrease는 H12에서 positive였지만 종료 시점에는 보수적 margin을
넘지 못했다.

| State / owner response | H12 gain | Simulation-end gain | Required | Inventory delta | Final |
|---|---:|---:|---:|---:|---|
| 155 plateau / B green decrease | `+1.372296` | `+2.190239` | `2.447505` | `-0.207658` | margin fail |
| 190 recovery / B green decrease | `+2.728535` | `-7.106419` | `3.118299` | `+2.043514` | negative |
| 190 recovery / C green increase | `+3.869476` | `-8.599249` | `3.118299` | `+2.043514` | negative |
| 190 recovery / D green decrease | `+10.348804` | `+20.598767` | `3.118299` | `-0.000000` | **positive** |
| 190 recovery / F green decrease | `+15.785008` | `+34.841212` | `3.118299` | `+1.922609` | inventory blocked |

기존 170 plateau C green decrease의 simulation-end `+118.658565 veh-h`와 새 190 recovery
D green decrease의 `+20.598767 veh-h`가 현재 strict contract의 두 장기 positive다. 190의
B/C 반전은 H12 exact replay 뒤에 발생했으므로 구현 drift가 아니라 horizon mismatch다.
F는 TTT 개선은 가장 크지만 종료 inventory가 남아 strict contract에서는 negative다. 이
결과는 H12 positive를 그대로 supervised truth로 쓰면 안 되며, recovery tail과 terminal
inventory를 label에 명시적으로 넣어야 함을 보여준다.

산출물:

- `results/rl_phase0_implementation_20260829/balanced_remaining_horizon_v2_155_plateau/`
- `results/rl_phase0_implementation_20260829/balanced_remaining_horizon_v2_190_recovery/`
- `rl_leader/diagnose_balanced_remaining_horizon.py`
- `src/tests/test_balanced_remaining_horizon.py`

### 다음 실행 계획

1. **Tail-aware oracle contract**: H12 label과 별도로 recovery-end, simulation-end, 필요하면
   demand 종료 뒤 drain-out까지 기록한다. terminal inventory는 hard guard와 terminal value
   두 판정을 함께 보존하되, strict 결과를 기본값으로 둔다.
2. **Freeway owner-block 확장**: 동일한 10개 coordination-eligible frozen state에서 ramp별
   `(meter, VSL)` axis/corner `0.25/0.5`를 H1 probe한다. urban과 같은 realized response /
   follower memory / physical-state dedup 및 selector v2 계약을 사용한다.
3. **Joint response 후보**: urban과 freeway 단독 positive가 확보된 뒤에만 상위 단독
   response의 sparse cross-owner 조합을 만든다. raw 75-D quadratic price 전수 탐색은 하지
   않는다.
4. **학습 단위 변경**: raw price 회귀 대신 deployable branch classifier와 realized-response
   pairwise ranker를 학습한다. target은 `candidate vs P-Stack`의 tail-aware conservative
   verdict이고 P-Stack을 항상 fallback으로 둔다.
5. **Frozen holdout 평가**: scenario와 stratum 단위로 train/validation을 분리하고, H12
   positive recall뿐 아니라 simulation-end false-positive rate, inventory violation, P-Stack
   대비 TTT를 함께 gate한다.
6. 위 gate가 통과한 뒤에만 repeated RL policy를 14,400초 full run으로 평가한다. 현재 결과는
   여전히 one-shot intervention oracle이며 반복 policy 성능으로 해석하지 않는다.

## 18. 2026-08-30 zero-demand drain-out 및 joint response 단계

### Strict drain-out 판정

`14,400초` 이후 외부 유입을 0으로 고정하고 candidate/native 두 branch가 모두 안정적으로
비워질 때까지 같은 native P-Stack을 적용했다. main/tail arrival sequence, component
inventory 합, validity, H12 replay가 모두 일치해야만 positive/negative로 판정하며, 완료되지
않은 branch는 negative가 아니라 quarantine한다.

| State / response | Drain-out TTT gain | Required | Final |
|---|---:|---:|---|
| 155 plateau / B green decrease | `+2.153173` | `2.468292` | negative |
| 170 plateau / C green+offset decrease | `+118.714482` | `3.331504` | **positive** |
| 170 skew plateau / R_F_W meter+VSL increase | `-5.321385` | `3.479087` | negative |
| 190 recovery / B green decrease | `-6.972900` | `3.139901` | negative |
| 190 recovery / C green increase | `-8.428013` | `3.139901` | negative |
| 190 recovery / D green decrease | `+20.598767` | `3.139901` | **positive** |
| 190 recovery / F green decrease | `+34.996308` | `3.139901` | **positive** |

190 F는 simulation-end inventory guard에서는 막혔지만 동일한 zero-demand drain-out 뒤에는
native보다 더 빨리 inventory를 처리해 strict positive가 됐다. 따라서 terminal inventory를
고정 시점의 hard reject 하나로만 쓰면 장기 이득을 false negative로 버릴 수 있고, 동일 유입
하의 clearance TTT까지 함께 봐야 한다.

### Tail-pairwise 학습 계약

완료된 drain artifact만 결합해 `tail_pairwise_realized_response_v1`을 만들었다. 현재 corpus는
4개 event group, 7개 unique realized outcome, positive 3개, negative 4개다. scenario,
stratum, candidate ID, H3/H12/tail label은 model input에서 금지하고 observation, anchor
envelope, native branch, execution branch, residual만 허용한다. event group별 weight 합은 1이며
split은 leave-one-scenario-out이다.

이 7개는 계약과 training code 검증에는 충분하지만 deployable ranker calibration에는 부족하다.
특히 incident terminal label이 0개이므로 지금 모델을 학습해 반복 정책에 넣으면 안 된다.

산출물:

- `rl_leader/diagnose_balanced_drain_out.py`
- `rl_leader/evaluate_balanced_joint_horizons.py`
- `rl_leader/build_tail_pairwise_dataset.py`
- `results/rl_phase0_implementation_20260830/tail_pairwise_v1/current.json`
- `src/tests/test_balanced_joint_horizon_selection.py`
- `src/tests/test_tail_pairwise_dataset.py`

### 현재 실행

155, 170 incident plateau/recovery, 170 skew plateau/recovery에서 urban H12 representative와
freeway H12 representative의 sparse cross를 만든다. 동일 residual은 한 번만 실행하되 모든
component pairing provenance를 보존하고, H1 follower-memory/physical-state가 같은 결과는
realized outcome으로 deduplicate한다. 모든 unique joint outcome은 H3 exact replay 뒤 H12까지
실행하며, H12 positive만 strict drain-out으로 승격한다.

joint 양성이 확보되더라도 이는 one-shot 정책의 support다. repeated policy는 첫 intervention
후 ranker가 실제 방문한 post-intervention state에서 새로운 paired continuation label을 수집한
뒤에만 활성화한다.

## 19. 2026-08-30 P-Stack anchored tail selector 시작

IQL actor는 잠시 동결하고, 완료된 strict drain-out label만 사용하는 oracle selector를 먼저
구현했다. 이 selector는 deployable learned model이 아니라 `tail_pairwise_realized_response_v1`
corpus에서 어떤 realized response가 P-Stack 대비 장기적으로 통과 가능한지 정의하는 teacher
단계다.

선택 규칙은 fail-closed다.

1. `target_valid=true`이고 strict tail verdict가 positive인 후보만 통과한다.
2. `gain_veh_h > required_gain_veh_h`와 추가 `min_margin_ratio`를 만족해야 한다.
3. 같은 frozen event에서 여러 positive가 있으면 `gain_veh_h`, `margin_ratio`, 작은 residual
   norm 순으로 하나를 고른다.
4. 통과 후보가 없거나 label이 invalid/quarantine이면 P-Stack fallback을 선택한다.
5. 선택 결과는 scenario/candidate ID를 진단용으로 기록하지만, 이후 learned ranker input에는
   기존 pairwise dataset 계약처럼 observation, anchor envelope, native branch, execution
   branch, residual만 허용한다.

현재 `results/rl_phase0_implementation_20260830/tail_pairwise_v1/current.json` 기준 결과는
4개 event group 중 2개 select, 2개 fallback이다.

| Scenario / stratum | Decision | Selected candidate | Gain | Required |
|---|---|---|---:|---:|
| 155 plateau | P-Stack fallback | - | - | - |
| 170 plateau | select | C green-offset decrease | `+118.714482` | `3.331504` |
| 190 recovery | select | F green decrease | `+34.996308` | `3.139901` |
| 170 skew plateau | P-Stack fallback | - | - | - |

산출물:

- `rl_leader/select_long_horizon_candidates.py`
- `results/rl_phase0_implementation_20260830/tail_selector_v1/current.json`
- `src/tests/test_long_horizon_selector.py`

### Learned selector smoke

oracle selector의 출력 계약을 모사하는 첫 learned selector smoke를 추가했다. 모델은
`observation + anchor_envelope + native branch + execution branch + candidate residual`만
입력으로 쓰며, classifier probability, gain regression, margin regression의 ensemble LCB가
모두 통과한 경우에만 candidate를 선택한다. 현재 단계에서는 model artifact를 만들 수 있는지와
fail-closed gate가 작동하는지만 확인한다.

현재 `current_plus_joint` corpus는 completed strict drain artifact 5개를 묶은 `8` rows,
`4` event groups, positive `3`, negative `5`다. 이 데이터로 smoke 학습은 성공했지만
`min_event_groups=50`, `min_positive_rows=20` production gate를 만족하지 못하므로
`deployable_gate_pass=false`가 정상 판정이다.

| Scope | Event groups | Selected | False positives | Missed positives |
|---|---:|---:|---:|---:|
| train | `4` | `2` | `0` | `0` |
| heldout 155 | `1` | `0` | `0` | `0` |
| heldout 170 skew | `1` | `0` | `0` | `0` |
| heldout 170 | `1` | `0` | `0` | `1` |
| heldout 190 | `1` | `0` | `0` | `1` |

산출물:

- `rl_leader/train_tail_selector.py`
- `src/tests/test_tail_selector_training.py`
- `results/rl_phase0_implementation_20260830/tail_pairwise_v1/current_plus_joint.json`
- `results/rl_phase0_implementation_20260830/tail_selector_v1/current_plus_joint.json`
- `results/rl_phase0_implementation_20260830/tail_selector_model_v1/current_plus_joint_smoke.json`

다음 순서는 이 oracle selector를 학습 모델로 바로 치환하는 것이 아니라, label coverage를 늘리는
것이다.

1. 진행 중인 incident/skew joint drain-out artifact를 완료 또는 failed/quarantine으로 확정한다.
2. 155와 incident에서 H12 positive는 아니었지만 H3/H6 disagreement가 큰 후보 일부를 explicit
   drain-out으로 보내 false-negative support를 확인한다.
3. `P-CENT` physical direction을 proposal source로 추가하되, label은 여전히 P-Stack 대비
   strict tail gain으로 판정한다.
4. event group이 최소 `50~100`개가 된 뒤 pairwise/quantile ranker와 LCB abstention을 학습한다.
5. learned selector는 frozen holdout에서 false-positive `0`을 달성할 때만 one-shot full-run으로
   승격한다.

### Pipeline 자동화 및 현재 재학습 결과

완료된 drain artifact만 자동으로 발견해서 `tail_pairwise -> oracle selector -> learned selector`
를 순서대로 재생성하는 CLI를 추가했다. `running`, `failed`, `passed=false`, outcome이 비어 있는
artifact는 제외한다. 따라서 백그라운드 drain-out이 끝난 뒤 같은 명령을 다시 실행하면 corpus와
model gate가 동일한 기준으로 갱신된다.

2026-08-30 16:28 KST 기준 `current_complete` 재학습 결과는 completed drain artifact `9`개,
pairwise row `16`개, event group `7`개, positive row `5`개다. train false positive와 fold
false positive는 모두 `0`이지만, label coverage가 `min_event_groups=50`,
`min_positive_rows=20`에 미달하므로 `deployable_gate_pass=false`다. 이는 실패가 아니라 현재
안전 gate의 정상 동작이다.

실행 명령:

```powershell
$env:PYTHONPATH='C:\torchlib;.'
& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m rl_leader.run_tail_selector_pipeline `
  --roots results\rl_phase0_implementation_20260829 results\rl_phase0_implementation_20260830 `
  --result-dir results\rl_phase0_implementation_20260830 `
  --name current_complete `
  --ensemble-size 8 `
  --seed 0 `
  --logistic-steps 2000 `
  --logistic-lr 0.1 `
  --min-event-groups 50 `
  --min-positive-rows 20
```

산출물:

- `rl_leader/run_tail_selector_pipeline.py`
- `src/tests/test_tail_selector_pipeline.py`
- `results/rl_phase0_implementation_20260830/tail_pairwise_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_selector_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_selector_model_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_selector_pipeline_v1/current_complete.json`

### Tail label coverage audit

H12 source와 strict drain-out 사이의 coverage를 따로 감사하는 리포터를 추가했다. 이 리포터는
artifact path 기준으로만 보지 않고 같은 scenario/stratum/policy step/residual이 이미 strict
complete로 존재하면 duplicate source를 새 backlog로 세지 않는다. 또한
`candidate_selection=h12_positive_default`로 running 중인 drain-out은 해당 source의 H12
positive 후보 전체를 running coverage로 간주한다.

현재 리포트 기준:

| Metric | Count |
|---|---:|
| H12 candidate rows | `161` |
| H12 positive candidate rows | `14` |
| strict completed candidate rows | `16` |
| strict positive / negative rows | `5 / 11` |
| running candidate rows | `0` |
| duplicate-complete candidate rows | `2` |
| undrained H12-positive candidate rows | `0` |
| event groups with complete strict labels | `7` |

따라서 이미 발견된 H12 positive 후보는 모두 strict complete, running, 또는 duplicate-complete로
분류됐다. 현 시점에서 남은 label coverage 확장은 새 후보를 더 drain에 넣는 문제가 아니라,
더 많은 frozen event를 생성하고 그 eligible shard를 H1/H3/H12/drain으로 보내는 문제다.

중요한 구조적 제약도 확인했다. 현재 frozen-state manifest v2는 총 17개 event, 그중
coordination-eligible event가 10개뿐이다. `min_event_groups=50` gate는 이 manifest만으로는
통과할 수 없으므로, 최종 learned selector/ranker gate를 통과시키려면 다음 단계에서 dense
frozen-state manifest v3가 필요하다. 즉 5개 scenario 전체에서 ramp-up/plateau/recovery를
각 1개 대표 시점만 쓰는 대신, congestion growth/peak/early recovery/late recovery를 여러
policy step으로 표집해 최소 50개 이상의 eligible event group을 만들어야 한다.

다음 실행 단위:

1. 기존 running drain-out batch는 모두 terminal complete/passed로 닫혔으므로, 새 dense v3 frozen
   event에서 나온 H12 positive만 strict drain-out으로 승격한다.
2. complete가 늘면 `run_tail_selector_pipeline`을 재실행하고 `current_complete`를 갱신한다.
3. running batch 이후에도 event group이 50 미만이면, dense frozen-state manifest v3를 새
   계약으로 추가한다. 기존 v2 artifact는 dependency drift 때문에 덮어쓰지 않는다.
4. v3에서는 각 scenario별 여러 policy step을 freeze하고, H1/H3/H12는 same-state exact replay
   계약을 유지한다. strict label은 지금과 동일하게 P-Stack 대비 zero-demand drain-out gain으로
   판정한다.

산출물:

- `rl_leader/summarize_tail_label_coverage.py`
- `src/tests/test_tail_label_coverage.py`
- `results/rl_phase0_implementation_20260830/tail_label_coverage_v1/current.json`

### Dense frozen-state v3 준비

`balanced_oracle_manifest`에 기존 v2와 별도인 `balanced_oracle_frozen_states_v3` 계약을 추가했다.
v2 artifact는 그대로 유지하고, v3는 gate 통과를 위한 label coverage 확장을 목표로 한다.

v3 계획:

| Scenario | Planned frozen events |
|---|---:|
| sweet_155_w60 | `20` |
| sweet_170_w60 | `20` |
| sweet_170_incident_w60 | `20` |
| sweet_170_skew15_w60 | `20` |
| sweet_190_w60 | `20` |
| total | `100` |

minimum coordination-eligible gate는 `50`이다. incident scenario의 policy step `5`와 `15`는 각각
`incident_onset`, `incident_end`로 label override한다. 실제 eligible 수는 freeze 후 native
branch를 보고 확정한다.

또한 `run_balanced_owner_block_pilot`에 `--policy-steps` 필터와 `--strata all` 지원을 추가했다.
중요하게, dense v3는 한 scenario 전체를 한 artifact로 H1 생성하면 downstream H12 승격기가
기대하는 one-pool source 계약과 맞지 않는다. 따라서 v3 실행은 policy step 단위 shard로 잘라야
한다.

예상 실행 순서:

```powershell
$env:PYTHONPATH='C:\torchlib;.'
& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m rl_leader.balanced_oracle_manifest freeze-dense `
  --scenario sweet_155_w60 `
  --output results\rl_phase0_implementation_20260830\balanced_owner_block_frozen_states_v3\sweet_155_w60.json

& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m rl_leader.balanced_oracle_manifest merge-dense `
  --inputs results\rl_phase0_implementation_20260830\balanced_owner_block_frozen_states_v3\sweet_155_w60.json ... `
  --output results\rl_phase0_implementation_20260830\balanced_owner_block_frozen_states_v3\manifest.json

& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m rl_leader.run_balanced_owner_block_pilot `
  --manifest results\rl_phase0_implementation_20260830\balanced_owner_block_frozen_states_v3\manifest.json `
  --output-dir results\rl_phase0_implementation_20260830\balanced_h1_v4_dense_sweet_155_step9 `
  --scenarios sweet_155_w60 `
  --strata all `
  --policy-steps 9 `
  --candidate-mode owner_block_v2_all
```

산출물:

- `rl_leader/balanced_oracle_manifest.py`
- `rl_leader/run_balanced_owner_block_pilot.py`
- `src/tests/test_balanced_oracle_manifest.py`
- `src/tests/test_balanced_owner_block_pilot.py`

### Dense v3 actual freeze smoke

`sweet_155_w60`에 대해 dense v3 freeze를 실제 실행했다. 중간에 프로세스 세션이 한 번 끊겼고,
partial artifact가 9개 event까지 남아 있었다. 이를 버리지 않기 위해 `freeze-dense --resume`
을 추가했고, 기존 event를 actual replay로 exact 재검증한 뒤 이어서 freeze하도록 했다. 이후
resume 중 기존 prefix를 다시 쓰며 partial artifact가 줄어드는 문제가 보여, resumed event
검증 중에는 파일을 쓰지 않고 새 event가 생길 때부터만 write하도록 보완했다.

결과:

| Scenario | Planned events | Complete | Coordination-eligible | Branch counts |
|---|---:|---:|---:|---|
| sweet_155_w60 | `20` | `20` | `9` | coarse `7`, refined `2`, fallback_pfo `11` |
| sweet_170_w60 | `20` | `20` | `8` | coarse `8`, fallback_pfo `12` |

155와 170 모두 ramp-up 초기와 recovery 이후가 대부분 `fallback_pfo`였고, growth/plateau
구간이 주된 eligible source였다. 두 scenario 합계는 40 planned event 중 eligible `17`개다.
따라서 나머지 incident/skew/190에서도 유사 비율이면 merged eligible이 `50` gate에 못 미칠 수
있고, 더 혼잡한 scenario에서 eligible branch가 늘어나는지 실제 freeze 결과로 판단한다.

다음 dense freeze 대상:

1. `sweet_170_incident_w60` - 2026-08-30 16:15 KST에 `freeze-dense --resume` 시작
2. `sweet_170_skew15_w60`
3. `sweet_190_w60`

각 scenario는 `freeze-dense --resume`을 기본으로 실행한다. 5개 scenario가 모두 complete된 뒤
`merge-dense`를 실행하고, `run_balanced_owner_block_pilot --strata all --policy-steps <step>`로
eligible policy step shard만 H1/H3/H12/drain 단계에 넘긴다.

산출물:

- `results/rl_phase0_implementation_20260830/balanced_owner_block_frozen_states_v3/sweet_155_w60.json`
- `results/rl_phase0_implementation_20260830/balanced_owner_block_frozen_states_v3/sweet_170_w60.json`

### 2026-08-30 v4 manifest와 tail selector 현재 상태

목표는 IQL actor를 바로 다시 밀어붙이는 것이 아니라, P-Stack보다 장기 TTT가 좋은 후보만
통과시키는 selector/ranker label을 먼저 만드는 쪽으로 전환했다. 이를 위해 dense v3 base
manifest에 supplement event를 추가해 `balanced_oracle_frozen_states_v4_extended` manifest를
생성했다.

v4 frozen-state coverage:

| Scenario | Base eligible | Supplement eligible | Total eligible |
|---|---:|---:|---:|
| sweet_155_w60 | `9` | `1` | `10` |
| sweet_170_w60 | `8` | `0` | `8` |
| sweet_170_incident_w60 | `9` | `0` | `9` |
| sweet_170_skew15_w60 | `7` | `0` | `7` |
| sweet_190_w60 | `10` | `6` | `16` |
| total | `43` | `7` | `50` |

새로 추가한 실행 코드:

- `balanced_oracle_manifest freeze-dense-extra`
- `balanced_oracle_manifest merge-dense-extended`
- `run_tail_label_generation`
- urban/freeway horizon evaluator의 owner별 `--max-h3-candidates-per-owner`
- `max_h3_candidates_per_owner=1` fast path: H12 rollout 한 번에서 H1/H3/H12 checkpoint를 함께 사용
- runner-local replay cache: 같은 event 안에서 urban/freeway/joint/drain이 frozen replay를 재사용
- persistent `replay_cache.json` / `replay_cache.pkl`: 같은 event를 재실행할 때 frozen replay를 건너뛰고,
  새 H1 생성 시에도 downstream용 replay payload를 바로 남김

검증:

```powershell
$env:PYTHONPATH='C:\torchlib;.'
& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m py_compile `
  rl_leader\balanced_oracle_manifest.py `
  rl_leader\run_tail_label_generation.py `
  rl_leader\evaluate_balanced_horizons.py `
  rl_leader\evaluate_balanced_freeway_horizons.py `
  rl_leader\evaluate_balanced_joint_horizons.py `
  rl_leader\diagnose_balanced_drain_out.py

& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest `
  src.tests.test_balanced_oracle_manifest `
  src.tests.test_tail_label_generation `
  src.tests.test_balanced_horizon_selection `
  src.tests.test_balanced_freeway_horizon_selection `
  src.tests.test_balanced_joint_horizon_selection `
  src.tests.test_balanced_drain_out
```

위 테스트 묶음은 통과했다. Python 시작 시 `C:\torchlib\_distutils_hack\__init__.py` permission
warning이 출력되지만 exit code는 0이었다.

현재 selector 학습 상태:

```text
complete_drain_artifacts = 9
pairwise rows = 16
positive rows = 5
negative rows = 11
event_groups = 7
selected_event_groups = 3
pstack_fallback_event_groups = 4
deployable_gate_pass = false
```

즉 preliminary learned selector/ranker 학습은 실행됐지만, deployable gate는 의도대로 실패했다.
실패 이유는 모델 학습 시간이 아니라 label 수 부족이다. 현재 gate 기준은 `event_groups >= 50`,
`positive_rows >= 20`이고, 실제 strict labels는 event group `7`, positive `5`뿐이다.

중요한 병목:

`sweet_155_w60 step 6` smoke를 `max_h3_candidates_per_owner=1`로 실행했지만, H1 artifact가 이미
있는데도 30분 이상 downstream artifact가 나오지 않았다. 계측 로그상 병목은 후보 수 이전의
frozen event replay와 native/candidate H12 P-Stack rollout이다. runner-local replay reuse와
persistent replay cache를 추가했지만, native/candidate H12 rollout 자체는 여전히 남는다. 따라서
단순히 H1 후보를 줄이는 것만으로는 50개 event label을 실용 시간 안에 만들기 어렵다.

다음 작업 순서:

1. persistent replay cache가 실제 긴 event에서 `replay-cache-hit`로 작동하는지 작은 재실행으로
   검증한다.
2. H12 rollout 수를 더 줄이는 second-stage selector를 추가한다. 예를 들어 H1/H3 top-1을 모든
   owner별로 H12까지 보내는 대신, urban/freeway/joint 통합 후보 중 lower-bound가 좋은 소수만
   strict drain-out으로 보낸다.
3. `run_tail_label_generation --max-h3-candidates-per-owner 1`을 cache-hit 상태에서 다시 smoke한다.
   한 event가 수분 이내로 닫히면 v4 eligible 50개를 shard로 실행한다.
4. complete drain labels가 늘 때마다 `run_tail_selector_pipeline`을 재실행한다.
5. `event_groups >= 50` 및 `positive_rows >= 20`을 통과한 뒤에만 learned selector를 deployable
   gate 후보로 승격한다.
6. learned selector gate가 P-Stack보다 좋은 후보만 넘기는지 170/190 short holdout에서 false-positive
   `0`을 먼저 확인한다.
7. 그 다음 5개 scenario 14,400초 full run으로 P-Stack 대비 TTT를 비교한다.

따라서 RL/selector 학습 자체는 이미 몇 초 만에 preliminary run까지 들어갔다. 하지만 실제로
"P-Stack보다 좋아야 넘기는" deployable RL gate는 label throughput 병목을 해결하고 strict label
coverage를 채운 뒤에야 의미 있게 끝난다.

## 11. 2026-08-30 budgeted H12와 no-positive drain skip

`sweet_155_w60 step 6` smoke에서 추가 병목과 계약 오류를 확인했다. H12 후보 1개는
`positive=false`, H12 TTT gain `-5.181 veh-h`였는데도 runner가 이를 strict drain-out으로 넘겼다.
이 경우 drain-out은 원래 시뮬레이션 끝까지 P-Stack을 계속 돌린 뒤 zero-demand tail까지 확인하므로
early-step smoke 하나가 30분 가까이 소모됐다.

수정한 계약:

- `run_tail_label_generation`의 drain 후보 선택을 H12 positive-only로 바꿨다.
- H12 positive가 없으면 drain-out을 실행하지 않고 `skipped_no_h12_positive` artifact를 기록한다.
- pipeline event status는 `no_h12_positive`로 닫히며, 학습용 pairwise dataset에는 들어가지 않는다.
- `budgeted_h12` completion check는 `selector_version`, `max_h12_candidates_per_event`, `domain_pool_size`
  를 함께 확인한다. smoke용 `max_h12=1` artifact가 이후 `max_h12=2` run을 가로막지 않게 했다.
- downstream label 소비자는 frozen event payload hash 검증은 유지하되, 오래 걸려 만든 manifest/H1 artifact를
  코드 개선 뒤에도 재사용할 수 있도록 implementation/dependency drift를 명시적으로 허용한다. manifest 생성/병합
  쪽 기본 검증은 여전히 strict다.

검증:

```powershell
$env:PYTHONPATH='C:\torchlib;.'
& 'C:\Users\alsrj\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B -m unittest `
  src.tests.test_tail_label_generation `
  src.tests.test_balanced_oracle_manifest `
  src.tests.test_balanced_owner_block_pilot `
  src.tests.test_budgeted_tail_horizons `
  src.tests.test_balanced_horizon_selection `
  src.tests.test_balanced_freeway_horizon_selection `
  src.tests.test_balanced_joint_horizon_selection `
  src.tests.test_balanced_drain_out `
  src.tests.test_tail_pairwise_dataset `
  src.tests.test_tail_selector_pipeline `
  src.tests.test_tail_selector_training
```

위 테스트는 `82`개 통과했다. Python startup의 `C:\torchlib\_distutils_hack\__init__.py` permission
warning은 계속 출력되지만 exit code는 0이다.

최신 coverage:

```text
h12_candidate_rows = 162
h12_positive_candidate_rows = 14
strict_completed_candidate_rows = 16
strict_positive_rows = 5
strict_negative_rows = 11
running_candidate_rows = 0
duplicate_complete_candidate_rows = 2
undrained_h12_positive_candidate_rows = 0
event_groups_with_complete_strict_labels = 7
```

최신 selector pipeline:

```text
complete_drain_artifacts = 9
pairwise rows = 16
positive rows = 5
negative rows = 11
event_groups = 7
selected_event_groups = 3
pstack_fallback_event_groups = 4
train_false_positive_event_groups = 0
fold_false_positive_event_groups = 0
fold_missed_positive_event_groups = 1
deployable_gate_pass = false
```

산출물:

- `results/rl_phase0_implementation_20260830/tail_label_coverage_v1/current.json`
- `results/rl_phase0_implementation_20260830/tail_selector_pipeline_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_pairwise_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_selector_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_selector_model_v1/current_complete.json`
- `results/rl_phase0_implementation_20260830/tail_label_generation_v1/sweet_155_w60_step06/budgeted_h12/sweet_155_w60.json`
- `results/rl_phase0_implementation_20260830/tail_label_generation_v1/sweet_155_w60_step06/drain/sweet_155_w60.json`

현재 ETA 판단:

- 코드/테스트/selector 재학습은 완료됐다.
- deployable learned selector gate는 아직 끝난 것이 아니다. gate false의 직접 원인은 학습 시간이 아니라
  strict label coverage 부족이다.
- selector/ranker 학습 자체는 몇 초 단위지만, 새 strict drain label은 event 위치와 positivity에 따라 수십 분까지
  갈 수 있다. 특히 early policy step drain-out은 smoke 대상으로 부적절하다.

다음 실행 계획:

1. full 50 event를 무작정 순차 실행하지 않는다.
2. `max_h12_candidates_per_event=2`, `domain_pool_size=2` 기준으로 recovery-boundary 이후 또는 가능한 late event부터
   작은 batch를 실행한다.
3. H12 negative-only event는 즉시 `no_h12_positive`로 닫아 drain 예산을 쓰지 않는다.
4. H12 positive event만 strict zero-demand drain-out으로 보낸다.
5. complete strict labels가 늘 때마다 coverage와 `run_tail_selector_pipeline --name current_complete`를 재실행한다.
6. `event_groups >= 50`, `positive_rows >= 20`, fold false-positive `0`을 통과해야 RL gate/actor 쪽으로 넘어간다.
7. 그 전까지는 IQL actor가 아니라 P-Stack anchored selector/ranker label generator를 우선 완성한다.

## 12. 2026-08-31 budgeted selector v2 smoke

`sweet_190_w60 step24` 신규 budgeted v1 smoke는 H1 joint-sum top-2를 선택했지만 둘 다 H12에서
`-11.398/-11.840 veh-h`로 음수였다. 반면 기존 190 recovery urban H12 artifact에서는 같은 step24의
urban single D/F가 strict tail-positive였다. 따라서 v1의 문제는 "joint 후보가 나쁘다"가 아니라,
작은 H12 budget에서 H1 gain 합산 joint가 single-owner 후보를 모두 밀어내는 recall 실패였다.

수정:

- `BUDGETED_SELECTOR_VERSION = h1_budgeted_owner_diverse_singles_first_v2`
- H1 representative를 domain 내부 owner-diverse로 고른다.
- H12 budget ranking에서 single 후보를 joint보다 먼저 보낸다.
- drain 후보는 H12 positive 또는 H12 TTT gain 양수인 promising row만 보낸다.
- runner `pipeline_index.json`은 요청 batch만 남기지 않고 기존 event 기록을 누적 보존한다.

검증:

- 관련 unittest `83`개 통과.
- `sweet_190_w60 step27` 신규 v2 run 완료.
- H12 rows `5`개 중 drain-worthy 후보 `2`개를 strict drain-out으로 평가했다.
- 두 후보 모두 final strict status는 `negative`였다.

최신 coverage와 selector pipeline:

```text
h12_candidate_rows = 169
h12_positive_candidate_rows = 14
strict_completed_candidate_rows = 18
strict_positive_rows = 5
strict_negative_rows = 13
running_candidate_rows = 0
event_groups_with_complete_strict_labels = 8

complete_drain_artifacts = 10
pairwise rows = 18
positive rows = 5
negative rows = 13
event_groups = 8
selected_event_groups = 3
pstack_fallback_event_groups = 5
train_false_positive_event_groups = 0
fold_false_positive_event_groups = 0
fold_missed_positive_event_groups = 1
deployable_gate_pass = false
```

판정:

- selector/ranker 학습은 최신 strict labels로 다시 실행됐다.
- deployable gate는 여전히 false다. 원인은 label coverage 부족이고, 특히 positive rows가 `5/20`에 머문다.
- v2는 v1보다 recall 구조가 낫지만, 190 step27에서는 새 positive를 얻지 못했다.

다음 후보:

1. `sweet_170_w60 step21,24` 또는 `sweet_170_incident_w60 step27,30`처럼 기존 strict-positive가 있던
   plateau/recovery 인접 구간을 우선 실행한다.
2. 190만 계속 파는 것은 negative label만 늘릴 위험이 있으므로, 다음 batch는 170/incident 쪽으로 이동한다.
3. batch 크기는 1-2 event로 유지하고, H12/drain 결과가 positive를 늘리는지 확인한 뒤 확대한다.

## 13. 2026-08-31 sweet_170_w60 step21/24 v2 batch

실행:

```text
python -B -m rl_leader.run_tail_label_generation --manifest results/rl_phase0_implementation_20260830/balanced_owner_block_frozen_states_v4_extended/manifest.json --output-dir results/rl_phase0_implementation_20260830/tail_label_generation_v1 --scenarios sweet_170_w60 --policy-steps 21,24 --horizon-strategy budgeted --max-h12-candidates-per-event 5 --domain-pool-size 5 --drain-candidates-per-event 2 --keep-going
```

관측:

- 총 실행은 약 `04:13`부터 `08:44`까지 걸렸다. 병목은 학습이 아니라 H1/H12/drain label generation이다.
- `step21`은 H12 positive는 없었지만 positive-gain promising 후보 2개를 strict drain까지 보냈고, 둘 다 final negative였다.
- `step24`도 H12 positive는 없었지만 positive-gain promising 후보 2개를 strict drain까지 보냈다.
- `step24`의 `structured:freeway:R_F_E:axis:meter-positive:m0.5:a7e3b882287b`는 strict drain에서 positive였다.
- 따라서 v2의 "H12 positive뿐 아니라 H12 TTT gain 양수 row를 drain으로 보낸다"는 정책은 실제 strict positive recall을 1건 회복했다.

strict drain 결과:

```text
sweet_170_w60 step21
- structured:urban:B:axis:green-negative:m0.25:02eb62a91648
  negative, gain = 0.819560, required = 1.932388
- structured:freeway:R_F_W:axis:meter-positive:m0.5:e7ca8fc02c2f
  negative, gain = 0.086228, required = 1.932388

sweet_170_w60 step24
- structured:freeway:R_F_E:axis:meter-positive:m0.5:a7e3b882287b
  positive, gain = 3.123368, required = 1.504695
- structured:urban:D:axis:green-positive:m0.25:8c9492ee5b90
  negative, gain = 1.395852, required = 1.504695
```

최신 coverage:

```text
h12_candidate_rows = 179
h12_positive_candidate_rows = 14
strict_completed_candidate_rows = 22
strict_positive_rows = 6
strict_negative_rows = 16
running_candidate_rows = 0
duplicate_complete_candidate_rows = 2
undrained_h12_positive_candidate_rows = 0
event_groups_with_complete_strict_labels = 10
```

최신 selector pipeline:

```text
complete_drain_artifacts = 12
pairwise rows = 22
positive rows = 6
negative rows = 16
event_groups = 10
selected_event_groups = 4
pstack_fallback_event_groups = 6
train_selected_event_groups = 2
train_false_positive_event_groups = 0
fold_false_positive_event_groups = 0
fold_missed_positive_event_groups = 3
deployable_gate_pass = false
```

판정:

- learned selector/ranker는 재학습까지 완료됐다.
- deployable gate는 여전히 false다. 원인은 모델 학습 시간이 아니라 strict label coverage 부족이다.
- 이번 batch로 strict positive가 `5 -> 6`, complete strict event group이 `8 -> 10`으로 늘었다.
- H12 strict-positive count는 늘지 않았지만, positive-gain promising drain 정책 덕분에 H12 label만 보면 놓쳤을 후보를 회수했다.

다음 계획:

1. 같은 방식으로 `sweet_170_incident_w60 step27,30`을 우선 실행한다.
2. 그 다음 `sweet_170_w60 step17,19`처럼 이번 positive 주변의 인접 이벤트를 실행한다.
3. 매 batch 후 coverage와 `run_tail_selector_pipeline --name current_complete`를 다시 돌린다.
4. positive label이 충분히 늘기 전에는 IQL actor 학습으로 성능 논쟁을 재개하지 않는다.

## 14. 2026-08-31 sweet_170_incident_w60 step27/30 v2 batch

실행:

```text
python -B -m rl_leader.run_tail_label_generation --manifest results/rl_phase0_implementation_20260830/balanced_owner_block_frozen_states_v4_extended/manifest.json --output-dir results/rl_phase0_implementation_20260830/tail_label_generation_v1 --scenarios sweet_170_incident_w60 --policy-steps 27,30 --horizon-strategy budgeted --max-h12-candidates-per-event 5 --domain-pool-size 5 --drain-candidates-per-event 2 --keep-going
```

관측:

- `step27`과 `step30` H1/H12/drain batch가 정상 종료됐다.
- `step27`은 H12 positive는 없었고 positive-gain promising 후보 2개를 strict drain까지 보냈다.
- `step30`도 H12 positive는 없었고 positive-gain promising 후보 1개를 strict drain까지 보냈다.
- strict drain 결과는 모두 final negative였다.
- `step27`의 첫 후보는 gain `1.787905` vs required `1.828717`로 near-miss였다.

strict drain 결과:

```text
sweet_170_incident_w60 step27
- structured:freeway:R_F_W:corner:meter-positive_vsl-negative:m0.25:71a99d39e806
  negative, gain = 1.787905, required = 1.828717
- structured:freeway:R_D_W:axis:vsl-negative:m0.25:2284579f230c
  negative, gain = 1.641813, required = 1.828717

sweet_170_incident_w60 step30
- structured:urban:B:axis:green-negative:m0.25:02eb62a91648
  negative, gain = 0.496630, required = 1.424349
```

최신 coverage:

```text
h12_candidate_rows = 189
h12_positive_candidate_rows = 14
strict_completed_candidate_rows = 25
strict_positive_rows = 6
strict_negative_rows = 19
running_candidate_rows = 0
duplicate_complete_candidate_rows = 2
undrained_h12_positive_candidate_rows = 0
event_groups_with_complete_strict_labels = 12
```

최신 selector pipeline:

```text
complete_drain_artifacts = 14
pairwise rows = 25
positive rows = 6
negative rows = 19
event_groups = 12
selected_event_groups = 4
pstack_fallback_event_groups = 8
train_selected_event_groups = 2
train_false_positive_event_groups = 0
fold_false_positive_event_groups = 0
fold_missed_positive_event_groups = 4
deployable_gate_pass = false
```

판정:

- learned selector/ranker 재학습은 완료됐지만 deployable gate는 여전히 false다.
- incident 후반은 positive recall에는 실패했고, negative/near-miss label만 늘렸다.
- 다음 batch는 incident를 계속 파기보다 `sweet_170_w60 step17,19`처럼 이미 positive가 나온 `step24` 주변으로 이동한다.
