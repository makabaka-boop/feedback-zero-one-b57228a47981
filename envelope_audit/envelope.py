"""Exact envelope computation.

Given two piecewise-constant rate profiles

* arrival  -- defines the cumulative input       A(t) = integral of arrival rate
* service  -- defines the cumulative capacity    S(u) = integral of service rate

compute, for every t >= 0,

    F(t) = sup_{u >= 0} { A(t + u) - S(u) }.

Mathematical structure used (no symbolic supremum solver, no network-calculus
library, no grid sampling, no float tolerance, no finite time truncation):

For a fixed t, g_t(u) = A(t + u) - S(u) is piecewise linear in u, so its
supremum is attained at one of finitely many kinks (or at +infinity).  Every
such kink is one of:

  (a) u = 0;
  (b) a kink of S, u = beta_j;
  (c) a shifted kink of A, u = alpha_i - t (when u >= 0, i.e. t <= alpha_i);
  (d) u -> +infinity (only finite when the tail arrival rate does not exceed
      the tail service rate).

Expanding A(.) and S(.) at those points turns each candidate into a
piecewise-linear function of t over explicit t-intervals.  The answer F is the
pointwise maximum (upper envelope) of all candidate segments.  Upper
envelopes of rational lines are computed with an exact monotone convex-chain
construction (sorted slopes, exact Fraction intersections).

All values are :class:`fractions.Fraction`; every output fraction is reduced
with a positive denominator (denominator 1 is printed as ``p/1``).
"""

from bisect import bisect_right
from fractions import Fraction

ZERO = Fraction(0, 1)
INF = float("inf")


def cumulative(series):
    """Integrate a rate profile into the cumulative piecewise-linear function.

    Returns a list of kinks ``(x, value, right_slope)`` with strictly
    increasing ``x``, starting at ``x = 0`` with value 0 and right slope equal
    to the first rate.  The last kink represents the tail extending to
    +infinity.
    """
    kinks = []
    value = ZERO
    for index, (time, rate) in enumerate(series):
        if index > 0:
            previous_time, previous_rate = series[index - 1]
            value += previous_rate * (time - previous_time)
        kinks.append((time, value, rate))
    return kinks


def _value_at(kinks, x, times=None):
    """Evaluate a piecewise-linear function (kink format of :func:`cumulative`)."""
    if times is None:
        times = [k[0] for k in kinks]
    index = bisect_right(times, x) - 1
    x0, value0, slope = kinks[index]
    return value0 + slope * (x - x0)


def _candidate_u_zero(arrival_kinks):
    """Candidate (a): u = 0 -> F_a(t) = A(t), all t >= 0."""
    segments = []
    for index, (time, value, slope) in enumerate(arrival_kinks):
        end = arrival_kinks[index + 1][0] if index + 1 < len(arrival_kinks) else INF
        intercept = value - slope * time
        segments.append((time, end, slope, intercept))
    return ("A(t)", segments)


def _candidate_service_kinks(arrival_kinks, service_kinks):
    """Candidates (b): u = beta_j -> F_bj(t) = A(t + beta_j) - S(beta_j).

    For every arrival linear piece on [alpha_i, alpha_{i+1}), substituting
    u = beta_j shifts its window to [alpha_i - beta_j, alpha_{i+1} - beta_j).
    Windows are intersected with the domain t >= 0; a fully negative window is
    discarded.  Each emitted segment stores its true (possibly infinite) end
    so the solver never evaluates a line outside its window.
    """
    candidates = []
    for beta, service_value, _service_slope in service_kinks:
        segments = []
        for index, (alpha, arrival_value, slope) in enumerate(arrival_kinks):
            window_start = alpha - beta
            if index + 1 < len(arrival_kinks):
                window_end = arrival_kinks[index + 1][0] - beta
            else:
                window_end = INF
            # Intersect [window_start, window_end) with t >= 0.
            visible_start = window_start if window_start > ZERO else ZERO
            if window_end != INF and window_end <= ZERO:
                continue
            if window_end != INF and visible_start >= window_end:
                continue
            visible_end = window_end if (window_end == INF or window_end > ZERO) else ZERO
            # Global intercept of the unclipped line: at t = alpha - beta the
            # value is A(alpha) - S(beta), hence
            #   value(t) = A(alpha) - S(beta) + slope * (t - (alpha - beta)).
            intercept = arrival_value - service_value - slope * window_start
            segments.append((visible_start, visible_end, slope, intercept))
        candidates.append((f"A(t+{beta})-S({beta})", segments))
    return candidates


def _candidate_arrival_kinks(arrival_kinks, service_kinks):
    """Candidates (c): u = alpha_i - t -> F_ci(t) = A(alpha_i) - S(alpha_i - t).

    Only defined for t in [0, alpha_i] (u >= 0).
    """
    service_times = [k[0] for k in service_kinks]
    candidates = []
    for alpha, arrival_value, _arrival_slope in arrival_kinks:
        if alpha == ZERO:
            # u = -t with u >= 0 only exists at t = 0; the point is covered
            # by candidate (a), so skip the degenerate candidate.
            continue

        # Kinks of t -> S(alpha - t): t = alpha - beta_j, clipped to [0, alpha].
        breaks = {ZERO, alpha}
        for beta, _value, _slope in service_kinks:
            t_break = alpha - beta
            if ZERO < t_break < alpha:
                breaks.add(t_break)
        breaks = sorted(breaks)

        segments = []
        for left, right in zip(breaks, breaks[1:]):
            # t in [left, right) maps to u = alpha - t in (alpha-right,
            # alpha-left].  Because u decreases as t increases, the containing
            # service piece is found at the u-interval's midpoint (which is
            # strictly interior, never on a kink); the value is then anchored
            # exactly at t = left using the cumulative evaluator.
            u_mid = alpha - (left + right) / 2
            service_index = bisect_right(service_times, u_mid) - 1
            _beta, _service_value, service_slope = service_kinks[service_index]
            # F(t) = arrival_value - S(alpha - t).  With S linear here,
            # S(alpha - t) = const - service_slope * t, so F is linear in t
            # with slope +service_slope.
            slope = service_slope
            f_at_left = arrival_value - _value_at(service_kinks, alpha - left)
            intercept = f_at_left - slope * left
            segments.append((left, right, slope, intercept))
        # Constant tail t >= alpha (u = 0), matching candidate (a) at alpha.
        segments.append((alpha, INF, ZERO, arrival_value))
        candidates.append((f"A({alpha})-S({alpha}-t)", segments))
    return candidates


def _candidate_infinity(tail_arrival_rate, arrival_kinks, service_kinks):
    """Candidate (d): limit u -> +infinity of g_t(u).

    On the tails, A(x) = a_m * x + C_A and S(u) = s_n * u + C_S, hence
    g_t(u) = a_m * t + (a_m - s_n) * u + (C_A - C_S).

    Excluded entirely unless the two tail rates are equal.  If a_m > s_n the
    caller already reports unbounded.  If a_m < s_n then g_t(u) -> -infinity,
    so no infinity-level candidate exists (the tail line's maximum is at its
    finite left endpoint, already covered by candidates (a)-(c)).  Only when
    a_m == s_n does the (a_m - s_n)*u term vanish and the limit exist.
    """
    tail_service_rate = service_kinks[-1][2]
    if tail_arrival_rate != tail_service_rate:
        # If a_m > s_n the caller already reported unbounded.  Here a_m < s_n
        # and g_t(u) -> -infinity on the tails: no infinity-level candidate
        # exists; the tail line's maximum sits at its left endpoint, a finite
        # kink already covered by candidates (a)-(c).
        return None
    # Equal tail rates (including both zero): the (a_m - s_n)*u term vanishes
    # and the limit exists.  Tail constants: A(x) - a_m*x and S(u) - s_n*u.
    alpha_m, arrival_value_m, _ = arrival_kinks[-1]
    c_arrival = arrival_value_m - tail_arrival_rate * alpha_m
    beta_n, service_value_n, _ = service_kinks[-1]
    c_service = service_value_n - tail_service_rate * beta_n
    slope = tail_arrival_rate
    intercept = c_arrival - c_service
    return ("tail-limit", [(ZERO, INF, slope, intercept)])


def _upper_envelope_on_cell(left, right, lines):
    """Exact upper envelope of y = slope*x + intercept on one interval.

    ``lines`` is a list of ``(slope, intercept)`` tuples active on the cell
    ``[left, right)`` (``right`` may be infinity).  Returns
    ``(first_slope, first_intercept, interior_kinks, chain)`` where
    ``interior_kinks`` holds the exact Fraction x-coordinates strictly inside
    the cell at which the winning line changes, and ``chain`` lists
    ``(slope, intercept, win_start)`` triples.

    The winning line at the cell's left endpoint is chosen from the values
    *at ``left``* -- a line whose global crossing lies before the cell starts
    must not be treated as the cell's first winner.
    """
    # Keep the highest line for each slope, then sort by increasing slope.
    best_for_slope = {}
    for slope, intercept in lines:
        current = best_for_slope.get(slope)
        if current is None or intercept > current:
            best_for_slope[slope] = intercept
    ordered = sorted(best_for_slope.items())

    # Monotone chain over lines sorted by increasing slope.  Line i wins from
    # its crossing with the previous winner onward (larger slope eventually
    # dominates).  If that crossing is no later than where the current last
    # line starts winning, the current last line never wins and is popped.
    chain_slopes = []
    chain_intercepts = []
    chain_starts = []
    for slope, intercept in ordered:
        while chain_slopes:
            previous_slope = chain_slopes[-1]
            previous_intercept = chain_intercepts[-1]
            # Solve prev_slope*x + prev_intercept = slope*x + intercept;
            # slopes differ here (deduped above), so the divisor is nonzero.
            crossing = (previous_intercept - intercept) / (slope - previous_slope)
            if crossing <= chain_starts[-1]:
                chain_slopes.pop()
                chain_intercepts.pop()
                chain_starts.pop()
            else:
                break
        start = ZERO if not chain_slopes else crossing
        chain_slopes.append(slope)
        chain_intercepts.append(intercept)
        chain_starts.append(start)

    full_chain = list(zip(chain_slopes, chain_intercepts, chain_starts))

    # Restrict the global chain to this cell: the first winner is the last
    # chain line whose win-start is <= left (all are active on this cell).
    first_index = 0
    for index, start in enumerate(chain_starts):
        if start <= left:
            first_index = index
        else:
            break
    first_slope = chain_slopes[first_index]
    first_intercept = chain_intercepts[first_index]
    cell_chain = full_chain[first_index:]

    interior_kinks = [
        start for start in chain_starts[first_index + 1:]
        if right == INF or start < right
    ]
    return first_slope, first_intercept, interior_kinks, cell_chain


def solve(arrival_series, service_series):
    """Compute F over the full parameter domain t >= 0.

    Returns ``{"status": "unbounded"}`` when the supremum is +infinity for a
    fixed t, otherwise ``{"status": "ok", "curve": [...]}`` with curve entries
    ``[time, F_value, right_slope]`` as reduced ``p/q`` strings, the first at
    time 0 and the last representing the infinite tail.
    """
    arrival_kinks = cumulative(arrival_series)
    service_kinks = cumulative(service_series)

    tail_arrival_rate = arrival_kinks[-1][2]
    tail_service_rate = service_kinks[-1][2]
    if tail_arrival_rate > tail_service_rate:
        # g_t(u) -> +infinity as u -> infinity for every t (the tail rate
        # difference is positive).  F(t) growing with t itself is not this
        # case; only a positive tail rate gap is.
        return {"status": "unbounded"}

    candidates = [_candidate_u_zero(arrival_kinks)]
    candidates.extend(_candidate_service_kinks(arrival_kinks, service_kinks))
    candidates.extend(_candidate_arrival_kinks(arrival_kinks, service_kinks))
    tail_candidate = _candidate_infinity(
        tail_arrival_rate, arrival_kinks, service_kinks
    )
    if tail_candidate is not None:
        candidates.append(tail_candidate)

    # Global event coordinates: every candidate segment boundary.  Between two
    # consecutive events every candidate is a single straight line.
    events = {ZERO}
    for _name, segments in candidates:
        for start, end, _slope, _intercept in segments:
            if start > ZERO:
                events.add(start)
            if end != INF:
                events.add(end)
    events = sorted(events)

    # Pre-index segments per candidate for a binary-search lookup.
    indexed = []
    for name, segments in candidates:
        starts = [seg[0] for seg in segments]
        indexed.append((name, starts, segments))

    raw_points = []  # (x, value, right_slope)
    for cell_index, left in enumerate(events):
        right = events[cell_index + 1] if cell_index + 1 < len(events) else INF
        active_lines = []
        for _name, starts, segments in indexed:
            # Locate via the cell's left endpoint: a candidate segment starts
            # at an event coordinate, so the segment active at `left` stays
            # active throughout the open cell.
            position = bisect_right(starts, left) - 1
            if position < 0:
                continue
            seg_start, seg_end, slope, intercept = segments[position]
            if seg_start <= left and (seg_end == INF or left < seg_end):
                active_lines.append((slope, intercept))

        first_slope, first_intercept, interior_kinks, chain = (
            _upper_envelope_on_cell(left, right, active_lines)
        )
        value_at_left = first_slope * left + first_intercept
        raw_points.append((left, value_at_left, first_slope))

        # Kinks strictly inside the cell: the winning chain line changes at an
        # exact rational intersection; record the right-hand winning slope.
        # Skip an intersection that coincides with a global event coordinate:
        # it is already emitted as the next cell's left endpoint.
        for kink_index, kink_x in enumerate(interior_kinks):
            if right != INF and kink_x >= right:
                continue
            if kink_x == left:
                continue
            old_slope, old_intercept, _start = chain[kink_index]
            new_slope = chain[kink_index + 1][0]
            kink_value = old_slope * kink_x + old_intercept
            raw_points.append((kink_x, kink_value, new_slope))

    # Merge any points sharing an x-coordinate (keep the one carrying the
    # right-hand slope, which is emitted last for interior intersections that
    # coincide with a global event coordinate).
    raw_points.sort(key=lambda point: point[0])
    merged = []
    for point in raw_points:
        if merged and merged[-1][0] == point[0]:
            merged[-1] = point
        else:
            merged.append(point)

    # Minimal continuous piecewise-linear representation.  Each point carries
    # the slope of the segment starting there.  An interior kink is redundant
    # exactly when that slope equals the slope of the preceding retained
    # segment.  Run-length compression therefore keeps the first point, then
    # every point whose right slope differs from the last retained point's
    # right slope.  The infinite-tail point is retained whenever it introduces
    # a slope change; if it does not, the single surviving segment already
    # extends to infinity so no extra point is needed.
    pruned = [merged[0]]
    for point in merged[1:]:
        if point[2] != pruned[-1][2]:
            pruned.append(point)

    curve = [[_rational_text(x), _rational_text(value), _rational_text(slope)]
             for x, value, slope in pruned]
    return {"status": "ok", "curve": curve}


def _rational_text(value):
    """Reduced p/q string with positive denominator; keep denominator 1."""
    fraction = Fraction(value)
    return f"{fraction.numerator}/{fraction.denominator}"
