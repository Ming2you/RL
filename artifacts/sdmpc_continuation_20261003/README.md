# P9–P17 원본 결과와 모델 (2026-10-03)

P17 v2의 첫 5개 새 학습 seed 평가까지 포함한 스냅샷이다.
원본 1,252개 파일, 90,973,992바이트를 47,236,151바이트 ZIP으로 묶었다.
40 MiB씩 나눈 `research.zip.part001`과 `research.zip.part002`를 함께 보관한다.

- `manifest.json`: 각 조각과 전체 ZIP의 크기·SHA-256, inventory 해시.
- `inventory.json`: 모든 원본의 저장 경로·크기·SHA-256.
- 포함: P9–P17 결과 루트, 모델 가중치, 정책 설정, 학습 궤적, carry와 k16
  캐시, 사전 등록 문서, 검증 보고서, 로그, 중단·복구 및 채택하지 않은 초안 증거.
- 제외: queue lock, 임시 파일, Python bytecode. 기존 P8 아카이브는 그대로 둔다.

저장소 루트에서 표준 Python으로 실행한다. 이 도구는 모델을 로드하거나 실험을
실행하지 않는다. `restore`는 누락된 파일만 복원하고, 기존 파일의 내용이 다르면 중단한다.

```powershell
python -m work.sdmpc_continuation_archive_20261003 verify
python -m work.sdmpc_continuation_archive_20261003 restore
```

복원 위치는 원래의 `results/sdmpc_rl_p*/` 경로다. 기록 속 PID·절대 경로·시각은
원래 PC의 실행 이력이다. 다른 PC에서는 새 출력 폴더와 해당 PC의 Python 경로를
사용해야 하며, 기존 완료 폴더를 덮어쓰지 않는다. 보존된 `wave1/STOP`은 P9의
중단된 첫 wave에만 해당한다.

요약: [검증 결과](../../docs/rl_results_20261003/README.md).
최신 정책과 다음 단계: [P17 v2](../../docs/rl_continuation_p17_v2_20261003.md).
