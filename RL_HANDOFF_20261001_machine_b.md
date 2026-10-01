# RL 연구 인수인계: 2026-10-01 (machine B)

이 문서는 `RL_HANDOFF_20260930.md` 이후 두 번째 PC(machine B)에서 이어 한 작업의 진입점이다.
세부 실험 기록은 [docs/rl_budget_machine_b_20260930.md](docs/rl_budget_machine_b_20260930.md),
결과 표는 [docs/machine_b_results/](docs/machine_b_results/), 원본 결과는
[artifacts/sdmpc_machine_b_20261001/](artifacts/sdmpc_machine_b_20261001/)에 있다.

## 1. 현재 결론

**목표(5개 정식 시나리오 모두 carry 대비 엄격 개선)는 아직 달성하지 못했다.**

| 정식 평가(machine B) | 155 | 170 | incident | skew15 | 190 | 판정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Codex 최신 모델 `return_mc_v1` | -5.53% | -14.33% | -9.06% | -4.12% | -1.61% | 불합격(0/5) |
| B1 perimeter bind (registry #2) | **+1.74%** | -6.28% | -2.47% | **+2.28%** | -12.87% | 불합격(2/5) |

부호는 개선율이다(+가 좋음). 기준은 machine B에서 다시 돌린 zero-action carry center
(직전 실행 budget 유지)다. P-Stack도, budget 없는 S-DMPC도, 무제어도 아니다.

B1은 학습 프로필(정식 수요 x U(0.98,1.02) 배율 3개)에서 중-고혼잡 시나리오를 실제로 개선한다.
아래는 seed별 carry 대비 변화이며, 음수가 개선이다.

| 시나리오 | B1 변화(%) | 평균 | 개선 |
| --- | --- | ---: | ---: |
| 155 | -7.5, +4.2, +3.0, -0.1, -0.4 | -0.2% | 3/5 |
| 170 | -6.2, -9.6, -3.8, +1.8, -10.8 | -5.7% | 4/5 |
| incident | -3.1, -3.5, -0.6, -1.8, +1.3 | -1.5% | 4/5 |
| skew15 | -0.9, +2.5, -3.2, +2.1, +2.2 | +0.6% | 2/5 |
| 190 | +1.5, -1.9, -1.7, -10.3, -1.8, -5.0 | -3.2% | 5/6 |

B1 정의는 다음과 같다. spec은 `work/sdmpc_rl_perimeter_b_20261001/specs/b1_exact_bind50_w16_25_return.json`이다.
- control step 15까지는 carry를 유지한다.
- step 16-25에는 NP를 직전 달성값보다 50 낮게 요청한다(action 한계 때문에 구간당 최대 -50 veh).
- step 26부터는 step 16에 저장한 anchor로 복귀한다.

## 2. 이 PC의 상태와 재현성

- 저장소는 `C:\Users\alsrj\Desktop\RL`, 브랜치는 `codex/sdmpc-rl-budget-20260929`이다.
  원래 PC의 작업 폴더(`C:/Users/alsrj/Desktop/학술/2026_EAST/Codex/RL`)는 이 PC에 없다.
- `.venv-torch`는 Codex runtime Python 3.12.14(원래 PC와 같은 빌드)로 만들었다.
  torch 2.14.0+cpu, numpy 2.3.5, scipy 1.16.3, PyYAML 6.0.3이 들어 있고 pytest, py-spy를 추가했다.
  `runtime_versions()`는 계약과 정확히 같다.
- Codex 결과는 `python -m work.sdmpc_rl_handoff_20260930 restore`로 복원했다(1,627 파일, verify PASS).
- CPU는 Ryzen 5 3500X 6코어(AVX2)다. 원래 PC보다 구간당 약 2배 느리다.
- **PC가 바뀌면 기준선도 바뀐다.** float32 actor 출력이 최대 4 ULP 다르고, carry 궤적이 16-20구간부터
  갈라져 최종 TTT가 최대 0.16% 다르다. 그래서 machine-B 기준선을 새로 만들었다
  (`results/sdmpc_rl_machine_b_20260930/center_repro_v1`).
  155 3101.4778926950567, 170 3934.6182377626324, incident 5546.224358313904,
  skew15 4244.34789709013, 190 6593.685996544476.
- 정식 평가기는 `work/sdmpc_rl_perimeter_b_20261001/canonical_eval.py`다.
  null(0-행동) actor로 정식 skew15 기준선을 비트 단위까지 재현했다(4244.34789709013).
  실행 전에 machine fingerprint, 물리 계약, 소스 핀, 사전 등록(`canonical_registry.json`)을 검사한다.

## 3. 핵심 발견

1. **카오스.** 하위 S-DMPC는 반복 6회(`budget_runtime.boot`에서 강제) 안에 한 번도 수렴하지 않는다
   (정식 기준선 5개 모두 0/75). 사용자 경험상 30회로 늘려도 수렴하지 않고 성능 차이도 작았다.
   하위 solver가 흡수하지 못한 작은 budget 변화나 1e-10 수준의 수치 차이도 최종 TTT를 ±2-5% 움직인다.
   그래서 시나리오당 1회 실행으로는 정책 효과와 운을 구분할 수 없다.
2. **정식 기준선의 sweet spot.** 같은 `forecast.json`에 배율만 곱한 학습 carry와 비교하면 정식 170과
   incident의 carry는 어떤 학습 carry보다도 좋다. 배율이 모두 1보다 작은 seed도 마찬가지다.

   | 시나리오 | 학습 carry | 배율 1.0 회귀 예측 | 정식 carry | 차이 |
   | --- | --- | ---: | ---: | ---: |
   | 155 | 2967-3293 | 3159 | 3101.5 | -1.8% |
   | 170 | 4208-4587 | 4316 | 3934.6 | **-8.8%** |
   | incident | 5772-6082 | 5877 | 5546.2 | **-5.6%** |
   | skew15 | 4069-4418 | 4164 | 4244.3 | +1.9% |
   | 190 | 6444-7171 | 6741 | 6593.7 | -2.2% |

   B1의 정식 170(4181.6)과 incident(5683.4)는 학습 carry 전부보다 좋았지만, 이 특별한 기준선은 넘지 못했다.
3. **결과 수준 관점.** 정책은 seed와 무관하게 비교적 일정한 결과 수준을 만든다. 그래서 carry가 운 좋게
   낮은 seed에서는 지고, 나쁜 seed에서는 이긴다. 정식 평가를 통과하려면 정책의 통상 결과 수준이 정식 carry
   값보다 낮아야 한다. 학습 carry 평균 대비로 필요한 평균 개선은 대략 다음과 같다(카오스 여유분 별도).
   - 155 -1.8%, 170 -8.8%, incident -5.6%, 190 -2.2% 이상
   - skew15는 +1.9% 이내
4. **도시 누적이 TTT를 결정한다.** 좋은 궤적은 차량을 램프와 고속도로 쪽에 담고 도시 대기열을 낮게 유지한다.
   혼잡이 시작될 때의 NP 조임(perimeter gating)은 궤적을 이 방향으로 민다.
5. **수단별 결과**(학습 프로필 기준):
   - NP 조임은 δ≥50이면 action 한계 때문에 최대 속도 조임이 되어 δ와 무관하게 궤적이 같다.
     의미 있는 파라미터는 시작 시점과 기간이다.
   - **B1의 10구간 조임(16-25)이 가장 안전하다.** 학습 slot 26개 중 최악이 +4.2%이고 붕괴가 없다.
   - P8에서 시험한 12구간 이상(16-27, 16-30, 16-35)은 붕괴를 일으킨다(4절 표).
     일부 seed에서 170을 더 크게 개선하지만(-7.7%, -11.2%) 일관되지 않다.
   - 시작을 18로 늦추면(18-32) 190 seed 8705의 붕괴는 피하지만 다른 seed에서 +3.6..+4.6% 손실이 난다.
   - 상태 없는 복귀 근사(V2/V3)는 190에서 +22% 붕괴가 있었다. 복귀는 latched anchor 방식을 쓴다.
   - NUF 조임은 모든 시나리오에서 크게 악화한다(+7..+47%).
   - gate 대기열을 포함한 대기열 피드백은 스스로 latch되어 190에서 +55%, +70%로 붕괴한다.
   - 약한 조임(δ30 + 복귀)은 +6..+13% 손실 사례가 있다.
   - 초반 NP 완화(fallback 회피)는 단독으로는 결과가 엇갈리고, 조임과 섞으면 손해다.
6. **붕괴는 도시 gridlock이다.** 190 seed 8705에서 도시 TTT가 +3189..+3522 veh-h 늘었다.
   step 16 이후 fallback은 8-9회였다(B1은 2회). 도시 대기열 최고치는 3575-4013 veh로 carry의 2943보다 높았다.
   램프 대기열 최고치는 carry와 B1에서도 상한(720 veh)에 도달하므로, 최고치만으로는 붕괴를 구분할 수 없다.
7. **조임 구간 안의 fallback.** 정식 B1 실행에서는 5개 시나리오 모두 조임 구간 안에서
   reference control로 fallback했다(step 16과 22-25 중 일부). 기준선의 fallback은 step 2와 15이고,
   170과 incident에서는 17도 있다. 조임 요청이 구간 후반에 infeasible해진다는 뜻이다.
   개선의 일부가 이 fallback을 통해 나오는지는 아직 모른다.

## 4. P8 결과(긴 조임과 늦은 시작, 학습 프로필 16 slot)

carry 대비 변화(%)이며 음수가 개선이다. 셀은 평균(개선 seed 수/전체), 최악값 순이다.
열마다 실행한 seed가 다르므로 seed별 표는 ledger에 있다.

| 시나리오 | B1 16-25 | 16-27 | 16-30 | 16-35 | 18-32 |
| --- | --- | --- | --- | --- | --- |
| 155 | +0.8 (2/3), +3.0 | +0.1 (1/2), +0.5 | +0.0 (2/3), +1.2 | -0.1 (2/3), +1.2 | +0.1 (1/2), +1.0 |
| 170 | -4.3 (2/3), +1.8 | -3.1 (1/2), +3.5 | -5.9 (2/3), +1.1 | -1.8 (1/2), +2.8 | -3.9 (1/2), +3.9 |
| incident | -0.3 (2/3), +1.3 | +0.3 (0/2), +0.4 | -1.3 (2/3), +1.6 | -1.9 (2/2), -1.7 | -1.5 (1/2), +1.6 |
| skew15 | +0.4 (1/3), +2.2 | +9.5 (0/2), **+15.6** | +6.0 (1/3), **+15.6** | -0.2 (1/2), +3.4 | +1.3 (1/2), +3.6 |
| 190 | -4.7 (4/4), -1.7 | +29.0 (1/2), **+60.2** | +12.2 (2/4), **+48.7** | +15.8 (1/3), **+48.7** | +0.7 (1/2), +4.6 |

결론: B1보다 나은 창은 없다. 창을 늘려 170과 incident의 개선을 키우려면 붕괴를 막는 장치가 먼저 필요하다.

## 5. 목표

성공 기준은 바꾸지 않는다(`docs/rl_budget_balanced_goal_20260930.md`).
- 한 개의 공유 정책으로 5개 정식 시나리오 모두에서 같은 PC의 carry center 대비
  `max(1e-6, 1e-8 x center)`보다 큰 TTT 감소를 낸다.
- 통과하면 정책을 동결하고 새 출력 폴더에서 5개를 다시 재현한다.
- 정식 평가 전에 후보를 `canonical_registry.json`에 등록하고 시도 횟수를 기록한다.
  지금까지 정책 후보 2회(`return_mc_v1`, B1), null 검증 2회를 시도했다.
- 정식 데이터로 학습하거나 정식 결과를 보고 후보를 고르지 않는다. 정식 근방 학습은 사용자 승인 사항이다.

중간 목표는 다음과 같다(학습 프로필 기준, 정식 평가 전에 통과해야 함).
- 붕괴(carry 대비 +10% 이상) 0회.
- 시나리오별 평균 개선이 3절 3번의 필요 수준 이상: 170 -9% 이하, incident -6% 이하, 155 -2% 이하,
  190 -3% 이하, skew15 +1% 이하. B1 대비로는 170과 incident를 3-5%p 더 낮춰야 한다.

## 6. 다음에 할 일(우선순위 순)

1. **붕괴 메커니즘 분석(계산 없이 가능).** 190 seed 8705에서 B1(16-25, -1.8%)과 16-27(+60.2%)의 차이는
   조임 2구간뿐이다. 두 branch JSON의 구간별 row를 비교해 다음이 언제 시작되는지 찾는다.
   - fallback
   - 도시 대기열 급증
   - 램프 대기열 포화 지속 시간
   - 경계 대기열 증가

   skew15 seed 8804(16-27, 16-30 모두 +15.6%)도 같이 본다. 결과물은 붕괴 직전 신호와 guard 규칙이다.
   데이터는 `results/sdmpc_rl_machine_b_20260930/probe_p8/<slot>/branches/*.json`과 `carry.json`에 있다.
2. **조임 구간 fallback의 역할 확인.** 정식 B1 trace(`canonical_b1_run1/<scenario>/trace.json`)와 기준선 trace
   (`center_repro_v1/<scenario>/episode_00_trace.json`, 필드 `selection_source`)에서 fallback 전후의 budget,
   달성값, 도시 누적을 비교한다. 조임 효과가 하위 해의 변화인지, reference fallback 때문인지 구분한다.
3. **guard가 있는 긴 조임 후보.** 예를 들어 16-30 조임에 다음 조건을 붙인다.
   - 조임 중 fallback이 생기면 즉시 latched anchor로 복귀
   - 또는 1번에서 찾은 붕괴 직전 신호(예: 도시 대기열 증가율)가 임계를 넘으면 복귀

   P8의 16 slot과 이전 10 slot에서 B1과 같은 seed로 비교한다. 6코어에서 16 slot × 3 option에 약 3시간 걸린다.
4. **사전 검증 기준을 정식 평가 전에 고정한다.** 시나리오당 학습 seed 5개 이상, 5절의 중간 목표 통과.
   통과한 후보만 `register.py`로 등록하고 정식 평가한다(5개 병렬 약 40분, 순차 약 2.5시간).
5. **사용자 결정 사항.**
   - 학습 분포를 정식 수요 근방(예: ±0.2%)으로 좁힐지
   - 다중 실현 통계 기준을 함께 쓸지

   아직 결정되지 않았다. 사용자는 현재 방식(±2% 학습 프로필)을 유지하되 실제 개선을 원했고,
   하위 반복 횟수는 6회로 유지한다.

## 7. 실행 방법

분기 실험은 같은 carry와 checkpoint에서 여러 option을 분기한다. option 파일은
`work/sdmpc_rl_probe_b_20260930/`에 둔다.

```powershell
.venv-torch\Scripts\python.exe -B -u work\sdmpc_rl_probe_b_20260930\queue_runner.py --name p9 --options options_p9.json --branch-steps 16 --jobs "sweet_190_w:8705,sweet_170_skew15_w:8804" --cpus 0,1,2,3,4,5 --checkpoint-root D:\RL_data\sdmpc_rl_machine_b_20260930\ckpt
```

출력은 `results/sdmpc_rl_machine_b_20260930/probe_<name>/<scenario>_s<seed>/`,
로그는 `results/sdmpc_rl_machine_b_20260930/logs/`에 쌓인다. 진행 상황은 stdout 로그로 본다.
결과 표는 `python work/sdmpc_rl_probe_b_20260930/export_results.py`로 다시 만든다.

정식 평가는 `register.py --spec <spec> --rationale ... --validation ...`로 등록한 다음 실행한다.
`run_canonical_sequential.sh <spec> <run_root> <cpu_mask>`가 readout까지 만든다.

## 8. 데이터와 코드 위치

| 위치 | 내용 |
| --- | --- |
| `docs/rl_budget_machine_b_20260930.md` | 실험 순서대로 쓴 상세 기록(ledger) |
| `docs/machine_b_results/*.csv` | centers, canonical, carries(배율 포함), branches(모든 분기 결과와 원본 hash) |
| `artifacts/sdmpc_machine_b_20261001/` | machine-B 원본 결과의 분할 ZIP(1,127 파일, 78 MB)과 inventory, manifest |
| `work/sdmpc_rl_return_eval_b_20260930/` | Codex 정식 평가기의 machine-B판(`return_mc_v1` 전용, 59 tests) |
| `work/sdmpc_rl_probe_b_20260930/` | 분기 실험 러너(`probe.py`), 큐(`queue_runner.py`), 집계와 내보내기 |
| `work/sdmpc_rl_perimeter_b_20261001/` | 정책 모듈(`perimeter_actor.py`), 정식 평가기, 등록, readout, specs |
| `work/sdmpc_machine_b_archive_20261001.py` | 분할 ZIP build/verify/restore |

복원(다른 PC에서, 또는 결과가 없을 때):

```powershell
python -m work.sdmpc_machine_b_archive_20261001 verify
python -m work.sdmpc_machine_b_archive_20261001 restore
```

분기 상태 캐시는 `results/sdmpc_rl_machine_b_20260930/probe_checkpoint_cache/`로 복원된다.
machine B의 원래 위치는 `D:\RL_data\sdmpc_rl_machine_b_20260930\ckpt`이며, `--checkpoint-dir`나
`--checkpoint-root`로 지정한다.

## 9. 주의사항

- C 드라이브 여유가 약 8 GB다. 큰 출력은 `D:\RL_data`에 둔다.
- 실행 중인 worker가 쓰는 `status.json`을 읽지 않는다. Windows에서 파일 교체가 실패할 수 있다.
- 평가기 모듈 이름이 고정 snapshot 모듈(`policy`)과 겹치면 부팅에서 실패한다. 회귀 테스트가 있다.
- 6코어를 모두 쓰면 시스템이 매우 느려진다. probe와 평가기는 스스로 우선순위를 낮춘다.
- `perimeter_b` 폴더의 .py를 고치면 evaluator source hash가 바뀌므로 정식 평가 전에 다시 등록해야 한다.
- machine-A와 machine-B의 기준선은 서로 바꿔 쓸 수 없다. 다른 PC에서 정식 평가하려면 그 PC의 center부터 다시 만든다.
- 원래 PC의 Codex 3시간 heartbeat(`ddqn`)가 켜져 있으면 두 PC가 같은 일을 하게 된다.
  일시중지 여부는 사용자 확인이 필요하다.
- 8월 31일 로컬 작업은 이 PC의 `git stash@{0}`에 보관돼 있다(사용자 지시로 폐기 대상, push하지 않음).
