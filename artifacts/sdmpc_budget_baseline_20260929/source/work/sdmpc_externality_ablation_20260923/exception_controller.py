"""원 budget 실패를 보존하면서 물리적으로 유효한 최소 band 초과 반복점을 실행한다."""
import copy
import time
import numpy as np
from central_controller import CentralKKTSDMPC
from fixed_policy import halfwidth,excess as policy_excess,contract


def budget_excess(achieved, budget, scales):
    """NP[veh], NUF[veh/h] 초과량을 기존 solver scale로 무차원화한다."""
    residual = np.asarray(achieved) - np.asarray(budget)
    excess = policy_excess(achieved,budget)
    return residual, excess, excess / np.asarray(scales)


def choose_archive(rows):
    """원 물리/이산 제어 유효성은 필수. 실행 가능점 우선, 위반량 동률은 TTT 우선."""
    eligible = [r for r in rows if r['physical_control_valid'] and
                np.isfinite(r['ttt']) and np.all(np.isfinite(r['scaled_excess']))]
    feasible = [r for r in eligible if r['budget_feasible']]
    if feasible:
        return min(feasible, key=lambda r: (r['ttt'], tuple(r['budget']), tuple(r['point'])))
    return min(eligible, key=lambda r: (max(r['scaled_excess']), r['ttt'],
               tuple(r['budget']), tuple(r['point']))) if eligible else None


class BudgetExceptionSDMPC(CentralKKTSDMPC):
    def begin(self, *args):
        super().begin(*args)
        self.execution_archives = []
        self._discrete_archive = None
        self.execution_audit = dict(policy='minimum_original_band_excess', triggered=False)

    def feasible(self, y, budget, discrete=False):
        # 기존 하위 계산의 실행 격자 검사점만 저장한다. 유한차분 stencil은 후보가 아니다.
        if discrete and self._discrete_archive is not None:
            z = np.asarray(y, dtype=float)
            self._discrete_archive.setdefault(z.tobytes(), (z.copy(), 'existing_discrete_trial'))
        return super().feasible(y, budget, discrete)

    def solve(self, budget, seed, initial_dual):
        self._discrete_archive = {}
        try:
            result = super().solve(budget, seed, initial_dual)
            points = self._discrete_archive
        finally:
            self._discrete_archive = None
        # 수락된 다음 점은 다음 반복 anchor로 나타난다. 마지막 선택점/초기점도 포함한다.
        for z, source in [(seed, 'seed'), (result['point'], 'lower_final')] + [
                (r['anchor'], 'iteration_anchor_' + str(r['k'])) for r in result['rows']]:
            q = self.coords.quantize(np.asarray(z, dtype=float))
            points.setdefault(q.tobytes(), (q.copy(), source + '_quantized'))
        self.execution_archives.append(points)
        return result

    def execution_check(self, point, budget):
        ev = self.evaluate(point)
        control = self.coords.decode(point)
        valid = bool(ev.physical_valid and ev.control_valid and
                     self.coords.validate(control, True)['valid'] and np.isfinite(ev.total_ttt) and
                     np.all(np.isfinite(ev.budget_vector)))
        residual, excess, scaled = budget_excess(ev.budget_vector, budget, self.scales)
        return dict(physical_control_valid=valid, budget_feasible=bool(np.all(excess == 0.)),
                    original_residual=residual.tolist(), band_excess=excess.tolist(),
                    scaled_excess=scaled.tolist(), achieved=ev.budget_vector.tolist(),
                    budget=list(budget), point=np.asarray(point).tolist(), ttt=ev.total_ttt, budget_policy=contract())

    def decide(self, state, forecast, previous, warm):
        previous_dual = self.dual.copy()
        selected, results = super().decide(state, forecast, previous, warm)
        if selected is not None:
            check = self.execution_check(selected['point'], selected['budget'])
            selected.update(execution_exception=False, execution_allowed=True, execution_check=check)
            return selected, results
        # 모든 기존 budget 후보가 실패한 경우에만 보존한 반복점의 실행 격자를 원 모델로 검사한다.
        started = time.perf_counter()
        before = copy.deepcopy(self.counts)
        archive = []
        for index, (result, points) in enumerate(zip(results, self.execution_archives)):
            for point, source in points.values():
                row = self.execution_check(point, result['budget'])
                row.update(candidate_index=index, source=source)
                archive.append(row)
        chosen = choose_archive(archive)
        self.execution_audit = dict(policy='minimum_original_band_excess', triggered=True,
            records=archive, selected_record=chosen, normalization=self.scales.tolist(),
            norm='Linf; tie break actual TTT then budget and control',
            price_policy='hold_previous_interval_prices_for_archive_recovery',
            wall_seconds=time.perf_counter()-started,
            counts_delta={k: self.counts[k]-before[k] for k in before})
        if chosen is None:
            return None, results
        source = results[chosen['candidate_index']]
        point = np.asarray(chosen['point'])
        ev = self.evaluate(point)
        control = self.coords.decode(point)
        control.N_P_star, control.N_UF_star = chosen['budget']
        # 하위 마지막 점의 승수/정지성은 다른 실행점의 인증으로 복사하지 않는다.
        selected = dict(budget=chosen['budget'], point=point, control=control, evaluation=ev,
            feasible=chosen['budget_feasible'], converged=False, execution_allowed=True,
            execution_exception=not chosen['budget_feasible'], execution_check=chosen,
            status='feasible_archive_recovery_uncertified' if chosen['budget_feasible'] else 'budget_exception',
            reason='archive_recovery_after_all_final_candidates_failed',
            original_residual=chosen['original_residual'], stationarity=None,
            stationarity_point=None, local_solvers_converged=False,
            dual=previous_dual.copy(), leader_multiplier=None,
            source_candidate_index=chosen['candidate_index'], source_candidate_status=source['status'],
            source_candidate_final_point=np.asarray(source['point']).tolist(),
            selected_differs_from_source_final=not np.array_equal(point, source['point']))
        self.dual = previous_dual.copy()
        # 달성량으로 목표를 바꾸지 않는다. 선택한 원 요청을 다음 leader 중심으로 전달한다.
        self.last_budget = np.asarray(chosen['budget']).copy()
        return selected, results
