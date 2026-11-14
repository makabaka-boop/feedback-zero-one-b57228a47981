"""Standalone heavy cross-validation of the solver against an independent,
from-scratch brute reference (no code shared with the production solver).

For hundreds of random rational profiles it checks, at every exact candidate
time plus many random interior times, that:

* the produced curve equals the direct supremum evaluated over the exact set
  of candidate u-kinks;
* the curve is continuous and its segment slopes match finite differences;
* every sampled A(t+u)-S(u) value is bounded above by the curve.

Run:  python3 tests/cross_validate.py [seed] [iterations]
Exits non-zero on the first discrepancy.
"""

import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envelope_audit.envelope import solve  # noqa: E402


def random_series(rng, max_pieces=5):
    count = rng.randint(1, max_pieces)
    times = [Fraction(0)]
    for _ in range(1, count):
        times.append(times[-1] + Fraction(rng.randint(1, 5), rng.randint(1, 4)))
    rates = [Fraction(rng.randint(0, 6), rng.randint(1, 4)) for _ in range(count)]
    return list(zip(times, rates))


def evaluator(series):
    pieces = series

    def value(x):
        total = Fraction(0)
        for index, (start, rate) in enumerate(pieces):
            end = pieces[index + 1][0] if index + 1 < len(pieces) else None
            if end is None or x <= end:
                return total + rate * (x - start)
            total += rate * (end - start)
        return total

    return value


def brute_f(t, arrival, service, a_eval, s_eval):
    candidates = {Fraction(0)}
    for beta, _r in service:
        candidates.add(beta)
    for alpha, _r in arrival:
        if alpha >= t:
            candidates.add(alpha - t)
    values = [a_eval(t + u) - s_eval(u) for u in candidates]
    ta, ts = arrival[-1][1], service[-1][1]
    if ta == ts:
        am = arrival[-1][0]
        bn = service[-1][0]
        c_a = a_eval(am) - ta * am
        c_s = s_eval(bn) - ts * bn
        values.append(ta * t + c_a - c_s)
    return max(values)


def curve_eval(curve, t):
    points = [(Fraction(x), Fraction(v), Fraction(s)) for x, v, s in curve]
    for index, (x, value, slope) in enumerate(points):
        nxt = points[index + 1][0] if index + 1 < len(points) else None
        if nxt is None or t < nxt:
            return value + slope * (t - x)
    raise AssertionError("unreachable")


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 987654321
    iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = random.Random(seed)
    checked = 0

    for iteration in range(iterations):
        arrival = random_series(rng)
        service = random_series(rng)
        ta = arrival[-1][1]
        mode = rng.random()
        if mode < 0.45:
            service[-1] = (service[-1][0], ta + Fraction(rng.randint(1, 3)))
            expect_unbounded = False
        elif mode < 0.9:
            service[-1] = (service[-1][0], ta)
            expect_unbounded = False
        else:
            if ta > 0:
                service[-1] = (service[-1][0], max(Fraction(0), ta - 1))
            expect_unbounded = service[-1][1] < ta

        result = solve(arrival, service)
        if expect_unbounded:
            assert result == {"status": "unbounded"}, (iteration, result)
            continue
        assert result["status"] == "ok", (iteration, result)
        curve = result["curve"]
        a_eval, s_eval = evaluator(arrival), evaluator(service)

        probe_times = {Fraction(0)}
        for series in (arrival, service):
            for x, _r in series:
                probe_times.add(x)
        for x, _v, _s in curve:
            probe_times.add(Fraction(x))
        for _ in range(40):
            probe_times.add(Fraction(rng.randint(0, 60), rng.randint(1, 11)))

        for t in probe_times:
            expected = brute_f(t, arrival, service, a_eval, s_eval)
            got = curve_eval(curve, t)
            assert got == expected, (iteration, t, got, expected)
            checked += 1

        # Upper-bound property on a net of u values.
        u_net = {Fraction(0), Fraction(7, 3), Fraction(50, 7)}
        for series in (arrival, service):
            for x, _r in series:
                u_net.add(x)
        for t in list(probe_times)[:20]:
            for u in u_net:
                assert a_eval(t + u) - s_eval(u) <= curve_eval(curve, t), (
                    iteration, t, u,
                )

        # Continuity + finite-difference slopes.
        points = [(Fraction(x), Fraction(v), Fraction(s)) for x, v, s in curve]
        for index in range(1, len(points)):
            x, value, _s = points[index]
            px, pv, ps = points[index - 1]
            assert pv + ps * (x - px) == value, (iteration, "discontinuity")
        for index in range(len(points) - 1):
            x, _v, slope = points[index]
            nxt = points[index + 1][0]
            t0, t1 = x + (nxt - x) / 3, x + (nxt - x) * 2 / 3
            fd = (curve_eval(curve, t1) - curve_eval(curve, t0)) / (t1 - t0)
            assert fd == slope, (iteration, "slope", slope, fd)
        assert points[-1][2] == ta, (iteration, "tail slope")

    print(f"OK: {iterations} profiles, {checked} exact point checks, seed={seed}")


if __name__ == "__main__":
    main()
