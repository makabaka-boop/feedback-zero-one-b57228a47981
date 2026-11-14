"""Acceptance tests for the envelope audit job.

Three independent layers:

1. Hand-computed examples where the entire output curve is checked exactly,
   covering equal tail slopes, zero rates, several competing local extrema and
   envelope kinks that are not input breakpoints (interior intersections).
2. A brute-force re-implementation of F(t) = sup_u{A(t+u)-S(u)} over the exact
   candidate u-set, run on many random rational profiles, plus an independent
   upper-bound check over sampled u and finite-difference slope checks.
3. Schema validation paths and CLI atomic-replace behaviour.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envelope_audit import process_text, ValidationError  # noqa: E402
from envelope_audit.parsing import parse_input  # noqa: E402
from envelope_audit.envelope import solve, cumulative  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "run_audit.py")


def fr(value):
    return Fraction(value)


# ---------------------------------------------------------------------------
# Independent brute-force reference implementation (written from scratch,
# shares no code with the production solver).
# ---------------------------------------------------------------------------

def make_cumulative(series):
    """Return an exact evaluator for the integral of a step-rate profile."""
    pieces = list(series)

    def evaluate(x):
        total = Fraction(0)
        for index, (start, rate) in enumerate(pieces):
            end = pieces[index + 1][0] if index + 1 < len(pieces) else None
            if end is None or x <= end:
                # A point exactly on a kink belongs to the piece ending there
                # (the next piece is zero-length at that point), so use <=.
                return total + rate * (x - start)
            total += rate * (end - start)
        return total

    return evaluate


def brute_supremum(t, arrival, service):
    """Direct sup of A(t+u)-S(u) over the exact candidate u-kink set."""
    a_func = make_cumulative(arrival)
    s_func = make_cumulative(service)
    candidates_u = {Fraction(0)}
    for beta, _rate in service:
        candidates_u.add(beta)
    for alpha, _rate in arrival:
        # Shifted arrival kink u = alpha - t, only valid when u >= 0.
        if alpha >= t:
            candidates_u.add(alpha - t)

    values = [a_func(t + u) - s_func(u) for u in candidates_u]

    tail_arrival = arrival[-1][1]
    tail_service = service[-1][1]
    if tail_arrival == tail_service:
        alpha_m = arrival[-1][0]
        beta_n = service[-1][0]
        c_a = a_func(alpha_m) - tail_arrival * alpha_m
        c_s = s_func(beta_n) - tail_service * beta_n
        values.append(tail_arrival * t + c_a - c_s)
    return max(values)


def evaluate_curve(curve, t):
    points = [(Fraction(x), Fraction(v), Fraction(s)) for x, v, s in curve]
    for index, (x, value, slope) in enumerate(points):
        next_x = points[index + 1][0] if index + 1 < len(points) else None
        if next_x is None or t < next_x:
            return value + slope * (t - x)
    raise AssertionError("unreachable")


def assert_curve_structure(test_case, result):
    test_case.assertEqual(result["status"], "ok")
    curve = result["curve"]
    test_case.assertGreaterEqual(len(curve), 1)
    reduced_re = __import__("re").compile(r"-?[0-9]+/[0-9]+")
    previous_time = None
    previous_slope = None
    for index, entry in enumerate(curve):
        test_case.assertEqual(len(entry), 3)
        for text in entry:
            test_case.assertRegex(text, reduced_re)
            fraction = Fraction(text)
            # Reduced, positive denominator, denominator kept even when 1.
            test_case.assertEqual(text, f"{fraction.numerator}/{fraction.denominator}")
            test_case.assertGreater(fraction.denominator, 0)
        time, value, slope = (Fraction(x) for x in entry)
        test_case.assertGreaterEqual(slope, 0)
        if index == 0:
            test_case.assertEqual(time, 0)
        else:
            test_case.assertGreater(time, previous_time)
            # No adjacent redundant equal-slope boundaries.
            test_case.assertNotEqual(slope, previous_slope)
        previous_time, previous_slope = time, slope


# ---------------------------------------------------------------------------
# 1. Hand-computed full curves.
# ---------------------------------------------------------------------------

class HandComputedTests(unittest.TestCase):
    def run_case(self, arrival, service, expected):
        result = solve(arrival, service)
        self.assertEqual(result, {"status": "ok", "curve": expected})

    def test_sample_from_spec(self):
        # arrival rate 2 until t=1 then 0; service rate 1 forever.
        # F(t) = 1+t on [0,1], then 2.
        self.run_case(
            [ (fr(0), fr(2)), (fr(1), fr(0)) ],
            [ (fr(0), fr(1)) ],
            [["0/1", "1/1", "1/1"], ["1/1", "2/1", "0/1"]],
        )

    def test_equal_constant_rates(self):
        # Both rates 1: F(t) = t, still a finite curve.
        self.run_case(
            [ (fr(0), fr(1)) ],
            [ (fr(0), fr(1)) ],
            [["0/1", "0/1", "1/1"]],
        )

    def test_zero_arrival(self):
        self.run_case(
            [ (fr(0), fr(0)) ],
            [ (fr(0), fr(1)) ],
            [["0/1", "0/1", "0/1"]],
        )

    def test_unbounded_tail_gap(self):
        # Tail arrival 2 > tail service 1: supremum is +infinity for every t.
        result = solve([(fr(0), fr(2))], [(fr(0), fr(1))])
        self.assertEqual(result, {"status": "unbounded"})

    def test_equal_tail_slopes_with_kink(self):
        # arrival: 3 on [0,2], then 1 ; service: 2 on [0,1], then 1.
        # Verified line-by-line and against the independent brute reference:
        # on [0,1] the tail-limit line t+3 wins; on [1,2] the service-kink
        # candidate 2t+2 wins (they meet at t=1, value 4); for t>=2 the tail
        # line t+4 wins again (meet at t=2, value 6).  Equal tail slopes (1),
        # yet the whole curve stays finite.
        self.run_case(
            [ (fr(0), fr(3)), (fr(2), fr(1)) ],
            [ (fr(0), fr(2)), (fr(1), fr(1)) ],
            [
                ["0/1", "3/1", "1/1"],
                ["1/1", "4/1", "2/1"],
                ["2/1", "6/1", "1/1"],
            ],
        )

    def test_zero_service(self):
        # No service at all: F(t) = sup_{u>=0} A(t+u).  A saturates at value
        # 1 at t=1, so the supremum is the plateau value 1 for every t:
        # a constant curve with a single entry (the redundant kink at t=1
        # must be pruned).
        self.run_case(
            [ (fr(0), fr(1)), (fr(1), fr(0)) ],
            [ (fr(0), fr(0)) ],
            [["0/1", "1/1", "0/1"]],
        )

    def test_zero_service_saturating_is_pruned_to_single_line(self):
        # Cross-check the constant zero-service result against the brute
        # reference at several t values (the curve has no visible kink).
        arrival = [(fr(0), fr(1)), (fr(1), fr(0))]
        service = [(fr(0), fr(0))]
        result = solve(arrival, service)
        self.run_case(arrival, service, [["0/1", "1/1", "0/1"]])
        for t in (fr(0), fr(1) / 2, fr(1), fr(3)):
            self.assertEqual(
                evaluate_curve(result["curve"], t),
                brute_supremum(t, arrival, service),
            )

    def test_both_zero_rates(self):
        self.run_case(
            [ (fr(0), fr(0)) ],
            [ (fr(0), fr(0)) ],
            [["0/1", "0/1", "0/1"]],
        )

    def test_interior_intersection_not_an_input_breakpoint(self):
        # arrival: 4 on [0,2], then 1.
        # service: 2 on [0,1/2], 3 on [1/2,1], then 1.
        # Hand-checked winning lines (cross-checked with the brute reference):
        #   [0,1]   : tail-limit line y = t + 9/2;
        #   [1,3/2] : service-kink line y = 3t + 5/2 (crossing at t = 1);
        #   [3/2,2] : line y = 2t + 4 (crossing 3t+5/2 vs 2t+4 at t = 3/2);
        #   [2,inf) : tail line y = t + 6.
        # The kink at t = 3/2 is neither an arrival breakpoint (0, 2) nor a
        # service breakpoint (0, 1/2, 1): it is a pure interior intersection.
        self.run_case(
            [ (fr(0), fr(4)), (fr(2), fr(1)) ],
            [ (fr(0), fr(2)), (fr(1)/2, fr(3)), (fr(1), fr(1)) ],
            [
                ["0/1", "9/2", "1/1"],
                ["1/1", "11/2", "3/1"],
                ["3/2", "7/1", "2/1"],
                ["2/1", "8/1", "1/1"],
            ],
        )

    def test_fractional_data(self):
        # arrival rate 1/2 until 3/2 then 0; service rate 1/3 forever.
        # Hand check at t=0: sup_u {A(u)-S(u)} = max of u/6 on [0,3/2] = 1/4
        # (attained at the arrival kink u=3/2).  On [0,3/2] the envelope has
        # slope 1/3 (checked via F(1)-F(0) = 7/12 - 1/4), then flattens at
        # the saturated value 3/4.
        self.run_case(
            [ (fr(0), fr(1)/2), (fr(3)/2, fr(0)) ],
            [ (fr(0), fr(1)/3) ],
            [["0/1", "1/4", "1/3"], ["3/2", "3/4", "0/1"]],
        )

    def test_multiple_competing_extrema_spot_checks(self):
        # Richer profile: several local extrema compete; full curve is
        # cross-checked by the brute-force layer, with a few hand values here.
        arrival = [(fr(0), fr(5)), (fr(2), fr(2)), (fr(5), fr(0))]
        service = [(fr(0), fr(3)), (fr(1), fr(4)), (fr(4), fr(1))]
        result = solve(arrival, service)
        assert_curve_structure(self, result)
        self.assertEqual(evaluate_curve(result["curve"], fr(0)), fr(3))
        self.assertEqual(evaluate_curve(result["curve"], fr(5)), fr(16))
        # Tail is constant 16 (tail arrival rate 0).
        last = result["curve"][-1]
        self.assertEqual(last[1], "16/1")
        self.assertEqual(last[2], "0/1")
        # Every hand candidate value must not exceed the envelope.
        for t in (fr(0), fr(1), fr(2), fr(3), fr(4), fr(5), fr(7)):
            self.assertEqual(
                evaluate_curve(result["curve"], t),
                brute_supremum(t, arrival, service),
            )


# ---------------------------------------------------------------------------
# 2. Randomised brute-force verification.
# ---------------------------------------------------------------------------

def random_series(rng, max_pieces=4):
    count = rng.randint(1, max_pieces)
    times = [Fraction(0)]
    for _ in range(1, count):
        times.append(times[-1] + Fraction(rng.randint(1, 4), rng.randint(1, 3)))
    rates = [Fraction(rng.randint(0, 5), rng.randint(1, 3)) for _ in range(count)]
    return list(zip(times, rates))


class RandomisedTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261114)

    def _verify_bounded(self, arrival, service):
        result = solve(arrival, service)
        self.assertEqual(result["status"], "ok")
        assert_curve_structure(self, result)
        curve = result["curve"]

        all_times = {Fraction(0)}
        for series in (arrival, service):
            for time, _rate in series:
                all_times.add(time)
                for delta in (Fraction(1, 1000), Fraction(1, 7), Fraction(2, 3)):
                    if time - delta > 0:
                        all_times.add(time - delta)
                    all_times.add(time + delta)
        for x, _v, _s in curve:
            t = Fraction(x)
            all_times.add(t)
            all_times.add(t + Fraction(1, 1000))
        for _ in range(60):
            all_times.add(Fraction(self.rng.randint(0, 40), self.rng.randint(1, 9)))

        for t in all_times:
            expected = brute_supremum(t, arrival, service)
            got = evaluate_curve(curve, t)
            self.assertEqual(got, expected, f"curve mismatch at t={t}")

        # Independent upper-bound check: F(t) dominates A(t+u)-S(u) for a
        # broad net of u values (midpoints between kinks included).
        a_func = make_cumulative(arrival)
        s_func = make_cumulative(service)
        u_net = {Fraction(0), Fraction(13, 5), Fraction(99, 7)}
        for beta, _r in service:
            u_net.add(beta)
        for alpha, _r in arrival:
            u_net.add(alpha)
        ordered_u = sorted(u_net)
        for left, right in zip(ordered_u, ordered_u[1:]):
            u_net.add((left + right) / 2)
        for t in list(all_times)[:25]:
            for u in u_net:
                self.assertLessEqual(
                    a_func(t + u) - s_func(u),
                    evaluate_curve(curve, t),
                )

        # Finite-difference slope check inside every segment.
        points = [(Fraction(x), Fraction(v), Fraction(s)) for x, v, s in curve]
        for index, (x, value, slope) in enumerate(points[:-1]):
            next_x = points[index + 1][0]
            t0 = x + (next_x - x) * Fraction(1, 3)
            t1 = x + (next_x - x) * Fraction(2, 3)
            delta = (evaluate_curve(curve, t1) - evaluate_curve(curve, t0))
            self.assertEqual(delta / (t1 - t0), slope)

        # Continuity at every interior kink.
        for index in range(1, len(points)):
            x, value, _slope = points[index]
            previous = points[index - 1]
            left_value = previous[1] + previous[2] * (x - previous[0])
            self.assertEqual(left_value, value)

        # Tail slope equals the tail arrival rate for every bounded case.
        self.assertEqual(points[-1][2], arrival[-1][1])

    def test_bounded_random_profiles(self):
        for _ in range(40):
            arrival = random_series(self.rng)
            service = random_series(self.rng)
            tail_arrival = arrival[-1][1]
            # Force a bounded tail: service tail rate strictly above, or equal.
            if self.rng.random() < 0.5:
                service[-1] = (service[-1][0], tail_arrival + Fraction(self.rng.randint(1, 3)))
            else:
                service[-1] = (service[-1][0], tail_arrival)
            self._verify_bounded(arrival, service)

    def test_unbounded_random_profiles(self):
        for _ in range(30):
            arrival = random_series(self.rng)
            service = random_series(self.rng)
            service[-1] = (
                service[-1][0],
                # Strictly smaller tail service rate, possibly zero.
                arrival[-1][1] - Fraction(1, 1) if arrival[-1][1] >= 1 else Fraction(0),
            )
            if arrival[-1][1] <= service[-1][1]:
                continue
            result = solve(arrival, service)
            self.assertEqual(result, {"status": "unbounded"})

    def test_cumulative_continuity_invariants(self):
        for _ in range(20):
            series = random_series(self.rng)
            kinks = cumulative(series)
            for index in range(1, len(kinks)):
                x, value, _slope = kinks[index]
                prev_x, prev_value, prev_slope = kinks[index - 1]
                self.assertEqual(value, prev_value + prev_slope * (x - prev_x))


# ---------------------------------------------------------------------------
# 3. Validation and CLI behaviour.
# ---------------------------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def assert_invalid(self, text, path_prefix):
        with self.assertRaises(ValidationError) as ctx:
            process_text(text)
        self.assertTrue(
            str(ctx.exception).startswith(path_prefix),
            f"{ctx.exception!r} does not start with {path_prefix!r}",
        )

    def test_unknown_member(self):
        self.assert_invalid(
            json.dumps({
                "arrival": [["0/1", "1/1"]],
                "service": [["0/1", "1/1"]],
                "extra": 1,
            }),
            "$.extra: unknown member",
        )

    def test_duplicate_member(self):
        text = (
            '{"arrival":[["0/1","1/1"]],'
            '"arrival":[["0/1","1/1"]],"service":[["0/1","1/1"]]}'
        )
        self.assert_invalid(text, "$.arrival: duplicate member")

    def test_missing_member(self):
        self.assert_invalid(
            json.dumps({"arrival": [["0/1", "1/1"]]}),
            "$: missing required member",
        )

    def test_syntax_error_rooted_at_dollar(self):
        self.assert_invalid('{"arrival": ', "$: invalid JSON syntax")

    def test_root_must_be_object(self):
        self.assert_invalid("[]", "$: root must be a JSON object")

    def test_empty_series(self):
        self.assert_invalid(
            json.dumps({"arrival": [], "service": [["0/1", "1/1"]]}),
            "$.arrival: array must be non-empty",
        )

    def test_bad_pair_length(self):
        self.assert_invalid(
            json.dumps({"arrival": [["0/1", "1/1", "2/1"]],
                        "service": [["0/1", "1/1"]]}),
            "$.arrival[0]: pair must contain exactly 2 elements",
        )

    def test_bad_rational_forms(self):
        for bad in ["1/2 ", " 1/2", "1", "1.5/2", "-1/2", "1/0", "1//2",
                    "0x1/2", "1e3/2", ""]:
            self.assert_invalid(
                json.dumps({"arrival": [["0/1", bad]],
                            "service": [["0/1", "1/1"]]}),
                "$.arrival[0][1]:",
            )

    def test_leaf_wrong_types(self):
        self.assert_invalid(
            json.dumps({"arrival": [["0/1", 1]], "service": [["0/1", "1/1"]]}),
            "$.arrival[0][1]:",
        )
        self.assert_invalid(
            json.dumps({"arrival": [["0/1", True]], "service": [["0/1", "1/1"]]}),
            "$.arrival[0][1]:",
        )
        self.assert_invalid(
            json.dumps({"arrival": [["0/1", None]], "service": [["0/1", "1/1"]]}),
            "$.arrival[0][1]:",
        )

    def test_first_time_must_be_zero(self):
        self.assert_invalid(
            json.dumps({"arrival": [["1/1", "1/1"]],
                        "service": [["0/1", "1/1"]]}),
            "$.arrival[0][0]: first time must be 0/1",
        )

    def test_times_strictly_increasing(self):
        self.assert_invalid(
            json.dumps({
                "arrival": [["0/1", "1/1"], ["1/1", "1/1"], ["1/1", "2/1"]],
                "service": [["0/1", "1/1"]],
            }),
            "$.arrival[2][0]: times must be strictly increasing",
        )

    def test_nested_path_indexes(self):
        self.assert_invalid(
            json.dumps({
                "arrival": [["0/1", "1/1"], ["1/1", "nope"]],
                "service": [["0/1", "1/1"]],
            }),
            "$.arrival[1][1]:",
        )
        self.assert_invalid(
            json.dumps({
                "arrival": [["0/1", "1/1"]],
                "service": [["0/1", "1/1"], ["2/1", 3]],
            }),
            "$.service[1][1]:",
        )

    def test_valid_input_accepts_reduced_and_unreduced(self):
        arrival, service = parse_input(
            json.dumps({"arrival": [["0/1", "2/4"]],
                        "service": [["0/1", "1/1"]]})
        )
        self.assertEqual(arrival[0][1], Fraction(1, 2))


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="audit-cli-")

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _write(self, name, content):
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _run(self, input_path, output_path):
        return subprocess.run(
            [sys.executable, "-I", ENTRY, input_path, output_path],
            capture_output=True, text=True,
        )

    def test_success_writes_expected_file(self):
        input_path = self._write(
            "in.json",
            json.dumps({
                "arrival": [["0/1", "2/1"], ["1/1", "0/1"]],
                "service": [["0/1", "1/1"]],
            }),
        )
        output_path = self._path("out.json")
        proc = self._run(input_path, output_path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(output_path, encoding="utf-8") as handle:
            self.assertEqual(
                json.load(handle),
                {"status": "ok",
                 "curve": [["0/1", "1/1", "1/1"], ["1/1", "2/1", "0/1"]]},
            )

    def test_failure_keeps_old_output_and_reports_dollar_path(self):
        good = self._write(
            "good.json",
            json.dumps({"arrival": [["0/1", "1/1"]],
                        "service": [["0/1", "1/1"]]}),
        )
        bad = self._write("bad.json", '{"arrival": [["0/1", "1/1"]],}')
        output_path = self._path("out.json")
        proc_ok = self._run(good, output_path)
        self.assertEqual(proc_ok.returncode, 0, proc_ok.stderr)
        with open(output_path, "rb") as handle:
            old_bytes = handle.read()

        proc_bad = self._run(bad, output_path)
        self.assertEqual(proc_bad.returncode, 2)
        self.assertTrue(proc_bad.stderr.startswith("$: invalid JSON syntax"))
        with open(output_path, "rb") as handle:
            self.assertEqual(handle.read(), old_bytes)

    def test_schema_failure_keeps_old_output(self):
        good = self._write(
            "good.json",
            json.dumps({"arrival": [["0/1", "1/1"]],
                        "service": [["0/1", "1/1"]]}),
        )
        bad = self._write(
            "bad.json",
            json.dumps({"arrival": [["0/1", "1/1"], ["1/1", "0/0"]],
                        "service": [["0/1", "1/1"]]}),
        )
        output_path = self._path("out.json")
        self.assertEqual(self._run(good, output_path).returncode, 0)
        with open(output_path, "rb") as handle:
            old_bytes = handle.read()
        proc = self._run(bad, output_path)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("$.arrival[1][1]:", proc.stderr)
        with open(output_path, "rb") as handle:
            self.assertEqual(handle.read(), old_bytes)

    def test_unbounded_output(self):
        input_path = self._write(
            "in.json",
            json.dumps({"arrival": [["0/1", "2/1"]],
                        "service": [["0/1", "1/1"]]}),
        )
        output_path = self._path("out.json")
        proc = self._run(input_path, output_path)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(output_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {"status": "unbounded"})

    def test_no_temp_files_left_behind(self):
        input_path = self._write(
            "in.json",
            json.dumps({"arrival": [["0/1", "1/1"]],
                        "service": [["0/1", "1/1"]]}),
        )
        output_path = self._path("out.json")
        self.assertEqual(self._run(input_path, output_path).returncode, 0)
        leftovers = [name for name in os.listdir(self.tmp)
                     if name.startswith(".audit-")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
