"""원 분기 동역학을 유지하는 희소 순방향 자동미분 scalar.

각 산술 연산의 지역 Jacobian을 비영 입력 tangent에만 적용한다. 시간 전파에
따라 물리적 이웃을 넘어 의존성이 퍼질 수 있다. 활성 경로의 미분이며 원 물리를
평활화하거나 유한차분을 새 이름으로 부르지 않는다.
"""
from __future__ import annotations

from collections import Counter
import builtins
import math
import numbers
import sys
import numpy as np


class Trace:
    def __init__(self, steps):
        self.steps = dict(enumerate(steps))
        self.operations = 0
        self.tangent_entries = 0
        self.max_tangent_entries = 0
        self.exact_tie_axes = set()
        self.stencil_crossing_axes = set()
        self.discrete_axes = set()
        self.counts = Counter()
        self.branch_risks = (0, 0)
        self.ignore_predicate_events = False
        self.predicate_sources = {}

    def scalar(self, value, tangent, parents=(), risks=(0, 0)):
        tangent = {k: v for k, v in tangent.items() if v != 0.0}
        exact, stencil = risks
        for parent in parents:
            if isinstance(parent, Dual):
                exact |= parent.risks[0]
                stencil |= parent.risks[1]
        self.operations += 1
        self.tangent_entries += len(tangent)
        self.max_tangent_entries = max(self.max_tangent_entries, len(tangent))
        return Dual(value, tangent, self, (exact, stencil)) if tangent or exact or stencil else float(value)

    def branch(self, a, b, *, discrete=False, output_connected=False):
        """정규화 좌표 stencil의 보수적 분기 위험을 기록한다.

        진단용 비교도 포함되므로 이벤트만으로 최종 TTT의 미분 부존재를
        증명할 수 없다. 실제 출력에 연결된 위험은 별도 전파한다.
        """
        da, db = derivative(a), derivative(b)
        gap = primal(a) - primal(b)
        exact_mask, stencil_mask = 0, 0
        if abs(gap) <= 1e-11 * max(1.0, abs(primal(a)), abs(primal(b))):
            # 앞선 kink에서 선택된 영 tangent도 동률 비교를 통해 영향을 줄 수 있다.
            # constant가 먼저 선택되더라도 기존 출력 위험을 잃지 않는다.
            for candidate in (a, b):
                if isinstance(candidate, Dual):
                    exact_mask |= candidate.risks[0]
                    stencil_mask |= candidate.risks[1]
        for key in da.keys() | db.keys():
            slope = da.get(key, 0.0) - db.get(key, 0.0)
            if abs(slope) <= 1e-14:
                continue
            if abs(gap) <= 1e-11 * max(1.0, abs(primal(a)), abs(primal(b))):
                self.exact_tie_axes.add(key)
                self.counts['exact_tie_comparisons'] += 1
                exact_mask |= 1 << key
            if abs(gap) <= abs(slope) * self.steps.get(key, 0.0):
                self.stencil_crossing_axes.add(key)
                self.counts['possible_stencil_crossing'] += 1
                stencil_mask |= 1 << key
            if discrete:
                self.discrete_axes.add(key)
                self.counts['discrete_dependent_comparisons'] += 1
        risks = (exact_mask, stencil_mask)
        if not output_connected and not self.ignore_predicate_events:
            self.branch_risks = (self.branch_risks[0] | exact_mask,
                                 self.branch_risks[1] | stencil_mask)
            if exact_mask or stencil_mask:
                frame = sys._getframe(1)
                while frame and frame.f_code.co_filename == __file__:
                    frame = frame.f_back
                source = f'{frame.f_code.co_name}:{frame.f_lineno}' if frame else 'unknown'
                old = self.predicate_sources.get(source, (0, 0))
                self.predicate_sources[source] = (old[0] | exact_mask, old[1] | stencil_mask)
        return risks

    def result(self, outputs=()):
        exact, stencil = self.branch_risks
        for value in outputs:
            if isinstance(value, Dual):
                exact |= value.risks[0]
                stencil |= value.risks[1]
        axes = lambda mask: [i for i in self.steps if mask & (1 << i)]
        return dict(operations=self.operations, tangent_entries=self.tangent_entries,
                    max_tangent_entries=self.max_tangent_entries,
                    exact_tie_axes=sorted(self.exact_tie_axes),
                    stencil_crossing_axes=sorted(self.stencil_crossing_axes),
                    discrete_dependent_axes=sorted(self.discrete_axes),
                    output_connected_exact_tie_axes=axes(exact),
                    output_connected_stencil_crossing_axes=axes(stencil),
                    unresolved_predicate_exact_tie_axes=axes(self.branch_risks[0]),
                    unresolved_predicate_stencil_crossing_axes=axes(self.branch_risks[1]),
                    unresolved_predicate_sources={key: dict(exact_tie_axes=axes(value[0]),
                        stencil_crossing_axes=axes(value[1])) for key, value in self.predicate_sources.items()},
                    event_counts=dict(self.counts),
                    event_scope='min/max/abs risks reaching costs/budgets plus conservative control-flow/ceil predicates')


def primal(value):
    return value.value if isinstance(value, Dual) else float(value)


def derivative(value):
    return value.tangent if isinstance(value, Dual) else {}


def float_keep(value=0.0):
    return value if isinstance(value, Dual) else float(value)


class Dual:
    __slots__ = ('value', 'tangent', 'trace', 'risks')
    __array_priority__ = 1000

    def __init__(self, value, tangent, trace, risks=(0, 0)):
        self.value = float(value)
        self.tangent = tangent
        self.trace = trace
        self.risks = risks

    def __deepcopy__(self, memo):
        # 산술 값은 불변이며 해당 평가의 기록기만 공유한다.
        return self

    def __float__(self):
        raise TypeError('Implicit float conversion would silently discard a traffic tangent')

    def __repr__(self):
        return f'Dual({self.value!r}, {self.tangent!r})'

    def __format__(self, spec):
        return format(self.value, spec)

    def _combine(self, other, value, self_scale=1.0, other_scale=1.0):
        out = {k: self_scale * v for k, v in self.tangent.items()}
        for key, val in derivative(other).items():
            out[key] = out.get(key, 0.0) + other_scale * val
        return self.trace.scalar(value, out, parents=(self, other))

    def __add__(self, other):
        return self._combine(other, self.value + primal(other))

    __radd__ = __add__

    def __sub__(self, other):
        return self._combine(other, self.value - primal(other), 1.0, -1.0)

    def __rsub__(self, other):
        return self._combine(other, primal(other) - self.value, -1.0, 1.0)

    def __mul__(self, other):
        return self._combine(other, self.value * primal(other), primal(other), self.value)

    __rmul__ = __mul__

    def __truediv__(self, other):
        denominator = primal(other)
        return self._combine(other, self.value / denominator, 1.0 / denominator,
                             -self.value / denominator**2)

    def __rtruediv__(self, other):
        return self._combine(other, primal(other) / self.value,
                             -primal(other) / self.value**2, 1.0 / self.value)

    def __pow__(self, other):
        if isinstance(other, Dual):
            value = self.value ** other.value
            return self._combine(other, value, other.value * self.value ** (other.value-1),
                                 value * math.log(self.value))
        value = self.value ** other
        scale = other * self.value ** (other-1) if other != 0 else 0.0
        return self.trace.scalar(value, {k: scale*v for k, v in self.tangent.items()}, parents=(self,))

    def __neg__(self):
        return self.trace.scalar(-self.value, {k: -v for k, v in self.tangent.items()}, parents=(self,))

    def __pos__(self):
        return self

    def __abs__(self):
        risks = self.trace.branch(self, 0.0, output_connected=True)
        selected = self if self.value >= 0.0 else -self
        return self.trace.scalar(selected.value, selected.tangent, parents=(selected,), risks=risks)

    def __mod__(self, other):
        denominator = primal(other)
        quotient = math.floor(self.value / denominator)
        self.trace.branch(self, quotient * denominator, discrete=True)
        self.trace.branch(self, (quotient+1) * denominator, discrete=True)
        return self._combine(other, self.value % denominator, 1.0, -quotient)

    def __bool__(self):
        self.trace.branch(self, 0.0, discrete=True)
        return bool(self.value)

    def _cmp(self, other, op):
        if not isinstance(other, (Dual, numbers.Real)):
            return NotImplemented
        self.trace.branch(self, other)
        return op(self.value, primal(other))

    def __lt__(self, other):
        return self._cmp(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._cmp(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._cmp(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._cmp(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._cmp(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._cmp(other, lambda a, b: a != b)

    def __ceil__(self):
        low = math.floor(self.value)
        self.trace.branch(self, low, discrete=True)
        self.trace.branch(self, low+1, discrete=True)
        return math.ceil(self.value)

    def __floor__(self):
        low = math.floor(self.value)
        self.trace.branch(self, low, discrete=True)
        self.trace.branch(self, low+1, discrete=True)
        return low


class MathProxy:
    def __getattr__(self, name):
        return getattr(math, name)

    @staticmethod
    def exp(value):
        if not isinstance(value, Dual):
            return math.exp(value)
        result = math.exp(value.value)
        return value.trace.scalar(result, {k: result*v for k, v in value.tangent.items()}, parents=(value,))

    @staticmethod
    def isfinite(value):
        return math.isfinite(primal(value))


class NumpyProxy:
    """역사 numpy의 물리 scalar clip과 진단 배열/평균을 지원한다.

    배열에 object scalar를 유지하여 지원하지 않은 ufunc가 미분을 조용히
    삭제하지 않고 예외를 발생시키도록 한다.
    """
    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def clip(value, lower, upper):
        if isinstance(value, Dual) or isinstance(lower, Dual) or isinstance(upper, Dual):
            return minimum(maximum(value, lower), upper)
        return np.clip(value, lower, upper)

    @staticmethod
    def asarray(values, dtype=None):
        values = list(values) if not isinstance(values, np.ndarray) else values
        if any(isinstance(value, Dual) for value in values):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=dtype)

    @staticmethod
    def std(values, *args, **kwargs):
        # 역사 코드의 경계 CV 진단 전용이며 동역학·TTT·budget은 소비하지 않는다.
        return np.std([primal(value) for value in values], *args, **kwargs)

    @staticmethod
    def max(values):
        return maximum(values)

    @staticmethod
    def min(values):
        return minimum(values)


def extremum(is_max, *args, **kwargs):
    """원 min/max 선택값을 유지하면서 출력에 도달하는 분기 위험만 전파한다."""
    if kwargs:
        # key/default는 역사 plant의 활성 수치 경로에 없고, 정적 구성에서 사용된다.
        function = builtins.max if is_max else builtins.min
        return function(*args, **kwargs)
    values = list(args[0]) if len(args) == 1 else list(args)
    if not values:
        raise ValueError('max()/min() arg is an empty sequence')
    selected = values[0]
    trace = next((value.trace for value in values if isinstance(value, Dual)), None)
    if trace is None:
        function = builtins.max if is_max else builtins.min
        return function(values)
    for candidate in values[1:]:
        if (primal(candidate) > primal(selected)) if is_max else (primal(candidate) < primal(selected)):
            selected = candidate
    risks = (0, 0)
    for candidate in values:
        if candidate is selected:
            continue
        addition = trace.branch(candidate, selected, output_connected=True)
        risks = risks[0] | addition[0], risks[1] | addition[1]
    return trace.scalar(primal(selected), derivative(selected), parents=(selected,), risks=risks)


def minimum(*args, **kwargs):
    return extremum(False, *args, **kwargs)


def maximum(*args, **kwargs):
    return extremum(True, *args, **kwargs)
