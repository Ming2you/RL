"""명세04/05/10/12/15: 기존 탐색을 우선하고 동일 범위의 추가 budget을 평가한다."""
import numpy as np
from multiplier import direction_from_recovery


def next_request(center, requests, result, m, capacity, allow_refinement):
    """기존 후보 우선순위는 유지하고 고갈될 때만 사전 정의한 내부 격자를 쓴다."""
    radius = np.array([50., 1000.])  # NP[veh], NUF command[veh/h]
    hint = direction_from_recovery(result['leader_multiplier']) if result['feasible'] else np.zeros(2)
    origin = 'selected_action_multiplier'
    if not np.any(hint):
        hint = np.array([1.,0.]) if m == 0 else np.array([0.,1.])
        origin = 'explicit_axis_search_no_valid_multiplier_direction'
    trials = [(center+radius*hint, origin), (center-radius*hint, origin+'_opposite'),
              (center+radius*np.array([0.,1.]), 'explicit_NUF_axis'),
              (center-radius*np.array([0.,1.]), 'explicit_NUF_axis_opposite')]
    if allow_refinement:
        # 기존 직사각형 범위 안에서만 보완한다. 중심/목표/허용오차를 이동시키지 않는다.
        directions = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
        for factor in (1., .5, .25, .75):
            for direction in directions:
                trials.append((center+factor*radius*np.array(direction),
                               f'additional_grid_scale_{factor:g}_direction_{direction[0]}_{direction[1]}'))
    for trial, source in trials:
        trial[1] = np.clip(trial[1], 0., capacity)
        if not any(np.array_equal(trial, previous) for previous in requests):
            return trial, source
    return None, 'unique_candidates_exhausted'


def decide(self, state, forecast, previous, warm):
    """목적 필드는 기존 objective adapter에서만 변환한다. 하위 문제는 수정하지 않는다."""
    self.begin(state, forecast, previous)
    seed = self.coords.encode(warm)
    witness = self.evaluate(seed)
    center = witness.budget_vector.copy() if self.last_budget is None else self.last_budget.copy()
    requests, results = [], []
    initial_dual = self.dual.copy()
    request = center.copy()
    origin = 'PFO_witness' if self.last_budget is None else 'previous_selected_budget'
    termination = 'candidate_limit'
    for m in range(self.options.max_candidates):
        requests.append(request.copy())
        # 모든 후보는 같은 초기 제어·수신 가격에서 출발한다. 후보별 가격 상태를 공유하지 않는다.
        result = self.solve(request, seed, initial_dual.copy())
        result['request_origin'] = origin
        results.append(result)
        if m+1 >= self.options.max_candidates:
            break
        request, origin = next_request(center, requests, result, m,
            self.cfg.network.total_ramp_capacity, self.options.max_candidates > 3)
        if request is None:
            termination = 'unique_candidates_exhausted'
            break
    self.upper_search_audit = dict(candidate_limit=self.options.max_candidates,
        evaluated_candidates=len(results), termination=termination, center=center.tolist(),
        radius=[50.,1000.], additional_grid_enabled=self.options.max_candidates>3,
        policy='legacy_first_then_fixed_center_interior_grid', upper_convergence_certified=False)
    eligible = [r for r in results if r['feasible']]
    if not eligible:
        return None, results
    selected = min(eligible, key=lambda r:(r['evaluation'].total_ttt, tuple(r['budget'])))
    self.dual = selected['dual'].copy()
    self.last_budget = np.asarray(selected['budget']).copy()
    return selected, results
