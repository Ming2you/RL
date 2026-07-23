# 시뮬레이터 & 최종 컨트롤러

freeway+urban 결합 교통 시뮬레이터와 최종 컨트롤러 구현. RL env(`code/env.py`)가 이 `src/`를 import한다.

## 구성
```
src/
├── models/        # plant/상태/수요 (metanet, local_freeway_plant, demand, state)
├── simulation/    # MixedTrafficSimulator, coupling (urban↔freeway)
├── controllers/   # 컨트롤러 (P-Stack=stackelberg_wu_metered + stackelberg_mpc,
│                  #   PFO=wu_faithful_follower, P-CENT=centralized_mpc, leader.py …)
├── config/        # default.yaml (물리·MPC 파라미터), scenarios.yaml (시나리오)
├── rl/            # 기존 RL 스캐폴드 (ddqn, env, action_space, rewards …)
├── analysis/, evaluation/, experiments/, tests/
work/run_claude_style_five_controller.py   # 러너(컨트롤러 env 플래그 정의)
requirements.txt   # numpy(필수), torch(RL), pandas·matplotlib(분석)
```

## 최종 컨트롤러 설정 (env 플래그 + controller ID)

공통 물리(base):
```
WARMUP_NC_STEPS=5;FW_BUFFER=8;TERM_ZG=1;VFREE=115;RHO_CRIT=31.5;TAU_H=0.0056111;NU_BASE=22.5;KAPPA=10;MERGE_DELTA=0.9
```

| 컨트롤러 | controller ID | 추가 env |
|---|---|---|
| **NC** | `NO-CONTROL` | (base만) |
| **PFO** | `WU-FAITHFUL-FOLLOWER` | `BASELINE_BOX=1` |
| **P-Stack (최종/b13)** | `P-STACK-WU-FAITHFUL-ALLPRICE-JOINT` | `BOX_WALK=1;BOX_WALK_VG=1;VSL_BOX=15;METER_BOX=300;NP_PD_ITER=4;NP_BIAS=1;CROSS_OFF=1;FAR_STATE_AWARE=1;SEG13=1;FAR_REAL_V=1;FAR_GATE=3;BASELINE_BOX=1;SUP_PFO=1;SUP_GATE=fargate` |
| **P-CENT** | `P-CENT` | `CENT_REFRESH_SEC=180;FAR_REAL_V=1` |

> **P-Stack 최종본**은 위 b13 플래그 조합 = far게이트(m3, 하이브리드) + 감독자(SUP_PFO/fargate) + BOX-WALK + SEG13(13-player) + NP_BIAS. 코드 자체는 `stackelberg_wu_metered.py`(follower Nash + per-lever 가격 FD) + `stackelberg_mpc.py`(leader 탐색).

## 실행 예
```bash
# 환경변수(위 base + 컨트롤러별 추가)를 export 후:
python work/run_claude_style_five_controller.py \
  --scenario sweet_190_skew15_w60 --T-total 14400 \
  --controllers P-STACK-WU-FAITHFUL-ALLPRICE-JOINT \
  --output outputs/myrun
# → outputs/myrun/<ID>/run_log.csv (windowed TTT = cumulative_total_ttt[last]−[step==4])
```

RL env는 리더 탐색을 우회하고 budget을 직접 주입:
```python
from code.env import RLLeaderEnv   # code/를 sys.path에 추가
env = RLLeaderEnv(scenario_name="sweet_190_skew15_w60")
```

## 시나리오 (config/scenarios.yaml)
학습 셀 `sweet_{155,170}_w60`, `sweet_170_{skew15,incident}_w60`, `sweet_190_w60`.
held-out(test) `sweet_190_incident_w60`, `sweet_190_skew15_w60`.

> ⚠️ config 파서는 자체 구현(PyYAML 불필요). 관측/보상 등 상세는 `code/env.py`·`src/rl/` 참조.
