"""A-D 실험: 지역 목적의 외부효과만 제거한다. own·budget 총미분은 보존한다."""
import os
import numpy as np

MODE = os.environ.get('SDMPC_EXTERNALITY', 'on')
if MODE not in ('on', 'off'):
    raise ValueError('SDMPC_EXTERNALITY must be on or off')
ENABLED = MODE == 'on'
ARMS = {'A': ('upper', 'on'), 'B': ('upper', 'off'),
        'C': ('none', 'on'), 'D': ('none', 'off')}


def local_externality(computed):
    # 입력과 공유 메모리를 갖지 않아 캐시의 전체 미분을 훼손하지 않는다.
    return np.array(computed, copy=True) if ENABLED else np.zeros_like(computed)


def audit_rows(candidates):
    """지역 응답 및 공통 기준점 로그에서 ablation 적용을 검사한다."""
    count = 0
    for candidate in candidates:
        for row in candidate['rows'] + candidate['local_rows']:
            assert row['externality_enabled'] == ENABLED
            expected = local_externality(np.asarray(row['externality_computed']))
            assert np.array_equal(np.asarray(row['externality']), expected)
            count += 1
    assert count > 0, 'No local iteration to validate'
    return count
