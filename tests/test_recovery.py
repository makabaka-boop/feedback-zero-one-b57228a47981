"""Acceptance tests for the bounded flow-summary recovery layer.

Layers, kept independent of the solver internals:

1. Hand-computed summaries spanning all three verdicts, including a
   rank-deficient matrix whose non-negativity constraints still pin a unique
   solution, a rank-deficient matrix with infinitely many solutions, and
   directly contradictory summaries proved infeasible by an explicit
   certificate evaluated by hand-written exact arithmetic.
2. Randomised exact checks: witnesses must be non-negative, satisfy ``Mx = b``,
   be distinct and admit a whole feasible ray; feasibility certificates must
   satisfy ``y^T M >= 0`` and ``y^T b < 0``; full-column-rank matrices must
   yield a unique solution equal to the planted vector; zero rows with a
   nonzero right-hand side must be infeasible.
3. Strict schema rejection with ``$``-rooted paths and CLI behaviour in both
   modes (including the legacy two-argument form and atomic replacement).
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

from envelope_audit import (  # noqa: E402
    ValidationError,
    process_summary_text,
)
from envelope_audit.envelope import solve  # noqa: E402
from envelope_audit.parsing import parse_summary  # noqa: E402
from envelope_audit.recovery import recover, summary_series  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRY = os.path.join(ROOT, "run_audit.py")
RATIONAL_RE = __import__("re").compile(r"-?[0-9]+/[0-9]+")


def fr(value):
    return Fraction(value)


def as_fractions(summary):
    return {
        "t": summary["t"],
        "M": summary["M"],
        "b": summary["b"],
        "tail": summary["tail"],
    }


def check_fraction_text(test_case, text):
    """Outputs are reduced p/q strings with a positive denominator."""
    test_case.assertRegex(text, RATIONAL_RE)
    value = Fraction(text)
    test_case.assertEqual(text, f"{value.numerator}/{value.denominator}")
    test_case.assertGreater(value.denominator, 0)
    return value


def matvec(matrix, vector):
    """Independent exact matrix-vector product."""
    return [sum((entry * value for entry, value in zip(row, vector)), Fraction(0))
            for row in matrix]


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def rank(matrix):
    """Independent exact rank via naive elimination."""
    rows = [list(row) for row in matrix]
    if not rows:
        return 0
    width = len(rows[0])
    pivot_row = 0
    for column in range(width):
        found = None
        for row in range(pivot_row, len(rows)):
            if rows[row][column] != 0:
                found = row
                break
        if found is None:
            continue
        rows[pivot_row], rows[found] = rows[found], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row != pivot_row and rows[row][column] != 0:
                factor = rows[row][column]
                rows[row] = [rows[row][index] - factor * rows[pivot_row][index]
                             for index in range(width)]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row


def build_summary(times, matrix, rhs, tail):
    return {
        "t": [fr(value) for value in times],
        "M": [[fr(value) for value in row] for row in matrix],
        "b": [fr(value) for value in rhs],
        "tail": [fr(value) for value in tail],
    }


# ---------------------------------------------------------------------------
# 1. Hand-computed verdicts.
# ---------------------------------------------------------------------------

class HandComputedRecoveryTests(unittest.TestCase):
    def test_spec_example_unique(self):
        summary = build_summary(["0/1", "1/1"], [["1/1", "1/1"]], ["0/1"],
                                ["0/1", "0/1"])
        result = recover(summary)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["x"], ["0/1", "0/1"])
        # The nested audit is exactly the original envelope result for the
        # reconstructed zero-rate profiles.
        arrival, service = summary_series(summary, [fr(0), fr(0)])
        self.assertEqual(result["audit"], solve(arrival, service))
        self.assertEqual(result["audit"]["status"], "ok")

    def test_rank_deficient_still_unique(self):
        # M has rank 1 (row two is all zeros) yet x >= 0 and x1 + x2 = 0
        # force x = [0, 0]: a rank-deficient but uniquely recoverable summary.
        summary = build_summary(
            ["0/1", "1/1"],
            [["1/1", "1/1"], ["0/1", "0/1"]],
            ["0/1", "0/1"],
            ["0/1", "0/1"],
        )
        self.assertLess(rank(summary["M"]), 2)
        result = recover(summary)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["x"], ["0/1", "0/1"])

    def test_unique_with_nontrivial_amounts_and_audit_curve(self):
        # t = [0,1], one input interval amount 2 (rate 2), one processing
        # amount 3 (rate 3); tails both 1.
        summary = build_summary(
            ["0/1", "1/1"],
            [["1/1", "0/1"], ["0/1", "1/1"]],
            ["2/1", "3/1"],
            ["1/1", "1/1"],
        )
        result = recover(summary)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["x"], ["2/1", "3/1"])
        self.assertEqual(
            result["audit"],
            {"status": "ok",
             "curve": [["0/1", "0/1", "2/1"], ["1/1", "2/1", "1/1"]]},
        )

    def test_unique_but_recovered_audit_is_unbounded(self):
        summary = build_summary(
            ["0/1", "1/1"],
            [["1/1", "0/1"], ["0/1", "1/1"]],
            ["2/1", "3/1"],
            ["2/1", "1/1"],
        )
        result = recover(summary)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["x"], ["2/1", "3/1"])
        self.assertEqual(result["audit"], {"status": "unbounded"})

    def test_ambiguous_with_finite_data(self):
        # M = [[1, 1]] x = 1 over one interval: every convex split of the
        # single unit between input and processing is feasible, so there are
        # infinitely many solutions.
        summary = build_summary(["0/1", "1/1"], [["1/1", "1/1"]], ["1/1"],
                                ["0/1", "0/1"])
        result = recover(summary)
        self.assertEqual(result["status"], "ambiguous")
        first = [check_fraction_text(self, text) for text in result["witnesses"][0]]
        second = [check_fraction_text(self, text) for text in result["witnesses"][1]]
        for vector in (first, second):
            self.assertTrue(all(value >= 0 for value in vector))
            self.assertEqual(matvec(summary["M"], vector), summary["b"])
        self.assertNotEqual(first, second)

    def test_ambiguous_unbounded_family(self):
        # M = [[1, 0]] x = 0: the processing amount is completely unconstrained
        # above; both witnesses must satisfy the summary and differ.
        summary = build_summary(["0/1", "1/1"], [["1/1", "0/1"]], ["0/1"],
                                ["1/1", "1/1"])
        result = recover(summary)
        self.assertEqual(result["status"], "ambiguous")
        first = [check_fraction_text(self, text) for text in result["witnesses"][0]]
        second = [check_fraction_text(self, text) for text in result["witnesses"][1]]
        self.assertEqual(first[0], 0)
        self.assertEqual(second[0], 0)
        self.assertLess(first[1], second[1])
        self.assertEqual(matvec(summary["M"], first), summary["b"])
        self.assertEqual(matvec(summary["M"], second), summary["b"])

    def test_contradictory_summary_infeasible(self):
        # x1 = 1, x2 = 1 and x1 + x2 = 3 cannot all hold with x >= 0.
        summary = build_summary(
            ["0/1", "1/1"],
            [["1/1", "0/1"], ["0/1", "1/1"], ["1/1", "1/1"]],
            ["1/1", "1/1", "3/1"],
            ["1/1", "1/1"],
        )
        result = recover(summary)
        self.assertEqual(result["status"], "infeasible")
        certificate = [check_fraction_text(self, text) for text in result["y"]]
        self.assertEqual(len(certificate), len(summary["b"]))
        self.assertTrue(all(value >= 0 for value in matvec_certificate(
            summary["M"], certificate)))
        self.assertLess(dot(certificate, summary["b"]), 0)

    def test_negative_rhs_infeasible(self):
        summary = build_summary(["0/1", "1/1"], [["1/1", "0/1"]], ["-1/1"],
                                ["0/1", "0/1"])
        result = recover(summary)
        self.assertEqual(result["status"], "infeasible")
        certificate = [check_fraction_text(self, text) for text in result["y"]]
        # For M = [[1, 0]] and b = -1 the natural certificate is y = [1].
        self.assertEqual(certificate, [fr(1)])
        self.assertGreaterEqual(dot(certificate, summary["M"][0]), 0)
        self.assertLess(dot(certificate, summary["b"]), 0)

    def test_example_files_produce_the_documented_verdicts(self):
        # The independent examples under examples/ pin one instance of each
        # verdict and are also exercised by run_tests.sh through the CLI.
        expectations = {
            "recover_unique.json": ("unique", ["2/1", "3/1"]),
            "recover_unique_rank_deficient.json": ("unique", ["0/1", "0/1"]),
            "recover_ambiguous.json": ("ambiguous", None),
            "recover_ambiguous_bounded.json": ("ambiguous", None),
            "recover_infeasible.json": ("infeasible", None),
        }
        for name, (status, x) in expectations.items():
            path = os.path.join(ROOT, "examples", name)
            with open(path, encoding="utf-8") as handle:
                result = process_summary_text(handle.read())
            self.assertEqual(result["status"], status, name)
            if x is not None:
                self.assertEqual(result["x"], x, name)

    def test_rank_deficient_example_proof_of_uniqueness(self):
        # M = [[1,1],[0,0]], b = [0,0], x >= 0: x1 + x2 = 0 forces x = 0 even
        # though rank(M) = 1 < 2 = number of unknowns.
        path = os.path.join(ROOT, "examples",
                            "recover_unique_rank_deficient.json")
        with open(path, encoding="utf-8") as handle:
            summary = parse_summary(handle.read())
        self.assertEqual(rank(summary["M"]), 1)
        result = recover(summary)
        self.assertEqual(result["status"], "unique")
        self.assertEqual(result["x"], ["0/1", "0/1"])


def matvec_certificate(matrix, certificate):
    """y^T M, computed independently of any recovery code."""
    width = len(matrix[0]) if matrix else 0
    return [sum((certificate[row] * matrix[row][column]
                 for row in range(len(matrix))), Fraction(0))
            for column in range(width)]


# ---------------------------------------------------------------------------
# 2. Randomised exact property checks.
# ---------------------------------------------------------------------------

class RandomisedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(20261114)

    def _matrix(self, rows, columns, low=-2, high=3):
        return [[Fraction(self.rng.randint(low, high)) for _ in range(columns)]
                for _ in range(rows)]

    def _verify_feasible_output(self, result, summary, expected=None):
        text = result["x"] if "x" in result else result["witnesses"]
        vectors = [text] if "x" in result else text
        for vector_text in vectors:
            vector = [check_fraction_text(self, entry) for entry in vector_text]
            self.assertEqual(len(vector), len(summary["M"][0]))
            self.assertTrue(all(value >= 0 for value in vector))
            self.assertEqual(matvec(summary["M"], vector), summary["b"])
            if expected is not None:
                self.assertEqual(vector, expected)

    def test_full_column_rank_is_unique_and_recovers_planted_vector(self):
        for _ in range(60):
            n = self.rng.randint(1, 3)
            rows = self.rng.randint(1, 2 * n)
            matrix = self._matrix(rows, 2 * n)
            if rank(matrix) != 2 * n:
                continue  # only test exactly-determined-by-rank systems
            planted = [Fraction(self.rng.randint(0, 4), self.rng.randint(1, 3))
                       for _ in range(2 * n)]
            rhs = matvec(matrix, planted)
            summary = build_summary(
                ["0/1"] + [Fraction(index + 1) for index in range(n)],
                matrix, rhs, [Fraction(self.rng.randint(0, 3))
                              for _ in range(2)],
            )
            result = recover(summary)
            self.assertEqual(result["status"], "unique")
            self._verify_feasible_output(result, summary, expected=planted)
            arrival, service = summary_series(summary, planted)
            self.assertEqual(result["audit"], solve(arrival, service))

    def test_duplicated_columns_are_ambiguous(self):
        for _ in range(60):
            n = self.rng.randint(1, 3)
            width = 2 * n
            matrix = self._matrix(n, width)
            # Duplicate one column into another (keeps rank deficient).
            source = self.rng.randrange(width)
            target = self.rng.randrange(width)
            for row in range(len(matrix)):
                matrix[row][target] = matrix[row][source]
            planted = [Fraction(self.rng.randint(0, 6)) for _ in range(width)]
            if planted[source] == 0:
                planted[source] = Fraction(1, 1)
            rhs = matvec(matrix, planted)
            summary = build_summary(
                ["0/1"] + [Fraction(index + 1) for index in range(n)],
                matrix, rhs, [Fraction(self.rng.randint(0, 3))
                              for _ in range(2)],
            )
            result = recover(summary)
            self.assertEqual(result["status"], "ambiguous")
            first, second = result["witnesses"]
            first_v = [check_fraction_text(self, text) for text in first]
            second_v = [check_fraction_text(self, text) for text in second]
            for vector in (first_v, second_v):
                self.assertTrue(all(value >= 0 for value in vector))
                self.assertEqual(matvec(summary["M"], vector), summary["b"])
            self.assertNotEqual(first_v, second_v)
            # The difference lies in the kernel, and the whole segment between
            # the two witnesses is feasible (P is convex).
            difference = [b - a for a, b in zip(first_v, second_v)]
            self.assertEqual(matvec(summary["M"], difference),
                             [Fraction(0)] * len(summary["b"]))
            for step in (Fraction(0), Fraction(1, 3), Fraction(1, 2),
                         Fraction(1)):
                ray = [a + step * d for a, d in zip(first_v, difference)]
                self.assertTrue(all(value >= 0 for value in ray))
                self.assertEqual(matvec(summary["M"], ray), summary["b"])

    def test_zero_row_with_nonzero_rhs_is_infeasible(self):
        for _ in range(40):
            n = self.rng.randint(1, 3)
            width = 2 * n
            matrix = self._matrix(self.rng.randint(1, 4), width)
            row = self.rng.randrange(len(matrix))
            matrix[row] = [Fraction(0)] * width
            rhs = [Fraction(self.rng.randint(0, 3)) for _ in range(len(matrix))]
            rhs[row] = Fraction(self.rng.choice([-3, -1, 1, 2, 4]), 1)
            summary = build_summary(
                ["0/1"] + [Fraction(index + 1) for index in range(n)],
                matrix, rhs, [Fraction(0), Fraction(0)],
            )
            result = recover(summary)
            self.assertEqual(result["status"], "infeasible")
            certificate = [check_fraction_text(self, text) for text in result["y"]]
            self.assertEqual(len(certificate), len(rhs))
            product = matvec_certificate(matrix, certificate)
            self.assertTrue(all(value >= 0 for value in product))
            self.assertLess(dot(certificate, rhs), 0)

    def test_random_systems_never_contradict_their_certificate(self):
        # For arbitrary small systems, whichever verdict is produced must be
        # verifiable by direct exact arithmetic (no solver code involved).
        for _ in range(80):
            n = self.rng.randint(1, 2)
            width = 2 * n
            matrix = self._matrix(self.rng.randint(1, 3), width)
            rhs = [Fraction(self.rng.randint(-2, 4)) for _ in range(len(matrix))]
            summary = build_summary(
                ["0/1"] + [Fraction(index + 1) for index in range(n)],
                matrix, rhs, [Fraction(0), Fraction(0)],
            )
            result = recover(summary)
            if result["status"] == "infeasible":
                certificate = [check_fraction_text(self, text)
                               for text in result["y"]]
                self.assertTrue(all(value >= 0
                                    for value in matvec_certificate(
                                        matrix, certificate)))
                self.assertLess(dot(certificate, rhs), 0)
            elif result["status"] == "unique":
                self._verify_feasible_output(result, summary)
            else:
                for vector_text in result["witnesses"]:
                    vector = [check_fraction_text(self, text)
                              for text in vector_text]
                    self.assertTrue(all(value >= 0 for value in vector))
                    self.assertEqual(matvec(matrix, vector), rhs)
                self.assertNotEqual(result["witnesses"][0],
                                    result["witnesses"][1])


# ---------------------------------------------------------------------------
# 3. Schema and CLI behaviour.
# ---------------------------------------------------------------------------

class SummaryValidationTests(unittest.TestCase):
    def assert_invalid(self, payload, prefix):
        with self.assertRaises(ValidationError) as ctx:
            process_summary_text(json.dumps(payload))
        self.assertTrue(str(ctx.exception).startswith(prefix),
                        f"{ctx.exception!r} does not start with {prefix!r}")

    def base(self):
        return {
            "t": ["0/1", "1/1"],
            "M": [["1/1", "1/1"]],
            "b": ["0/1"],
            "tail": ["0/1", "0/1"],
        }

    def test_unknown_and_duplicate_members(self):
        payload = self.base()
        payload["extra"] = 1
        self.assert_invalid(payload, "$.extra: unknown member")
        text = ('{"t":["0/1","1/1"],"t":["0/1","1/1"],'
                '"M":[["1/1","1/1"]],"b":["0/1"],"tail":["0/1","0/1"]}')
        with self.assertRaises(ValidationError) as ctx:
            process_summary_text(text)
        self.assertTrue(str(ctx.exception).startswith("$.t: duplicate member"))

    def test_missing_member(self):
        payload = self.base()
        del payload["tail"]
        self.assert_invalid(payload, "$: missing required member")

    def test_syntax_error_rooted_at_dollar(self):
        with self.assertRaises(ValidationError) as ctx:
            process_summary_text('{"t": ["0/1", "1/1"], "M": ')
        self.assertTrue(str(ctx.exception).startswith("$: invalid JSON syntax"))

    def test_t_structure(self):
        self.assert_invalid({"t": ["0/1"], "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]}, "$.t:")
        self.assert_invalid({"t": "0/1", "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]}, "$.t:")
        self.assert_invalid({"t": ["1/1", "2/1"], "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.t[0]: first time must be 0/1")
        self.assert_invalid({"t": ["0/1", "1/1", "1/1"],
                             "M": [["1/1", "1/1", "1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.t[2]: times must be strictly increasing")
        self.assert_invalid({"t": ["0/1", "-1/1"], "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.t[1]:")

    def test_matrix_structure(self):
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [],
                             "b": [], "tail": ["0/1", "0/1"]}, "$.M:")
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [["1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.M[0]: row must have 2n")
        self.assert_invalid({"t": ["0/1", "1/1", "2/1"],
                             "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.M[0]: row must have 2n")
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [[1, 2]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]},
                            "$.M[0][0]:")

    def test_b_and_tail_structure(self):
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
                             "b": [], "tail": ["0/1", "0/1"]}, "$.b:")
        self.assert_invalid({"t": ["0/1", "1/1"],
                             "M": [["1/1", "1/1"], ["0/1", "0/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1"]}, "$.b:")
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1"]}, "$.tail:")
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
                             "b": ["0/1"], "tail": ["0/1", "0/1", "1/1"]},
                            "$.tail:")
        self.assert_invalid({"t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
                             "b": ["-1/2.5"], "tail": ["0/1", "0/1"]},
                            "$.b[0]:")

    def test_bad_numeric_forms(self):
        for bad in [" 1/2", "1/2 ", "1", "1.5/2", "1//2", "1e3/2", ""]:
            self.assert_invalid({"t": ["0/1", "1/1"],
                                 "M": [["1/1", bad]],
                                 "b": ["0/1"], "tail": ["0/1", "0/1"]},
                                "$.M[0][1]:")

    def test_signed_matrix_and_vector_values(self):
        parsed = parse_summary(json.dumps({
            "t": ["0/1", "1/1"],
            "M": [["-2/4", "1/1"]],
            "b": ["-3/2"],
            "tail": ["0/1", "0/1"],
        }))
        self.assertEqual(parsed["M"][0][0], Fraction(-1, 2))
        self.assertEqual(parsed["b"][0], Fraction(-3, 2))


class RecoveryCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="recovery-cli-")

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _write(self, name, content):
        path = self._path(name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return path

    def _run(self, arguments):
        return subprocess.run(
            [sys.executable, "-I", ENTRY] + arguments,
            capture_output=True, text=True,
        )

    def test_recover_mode(self):
        input_path = self._write("in.json", json.dumps({
            "t": ["0/1", "1/1"],
            "M": [["1/1", "0/1"], ["0/1", "1/1"]],
            "b": ["2/1", "3/1"],
            "tail": ["1/1", "1/1"],
        }))
        output_path = self._path("out.json")
        proc = self._run(["recover", input_path, output_path])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(output_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {
                "status": "unique",
                "x": ["2/1", "3/1"],
                "audit": {"status": "ok",
                          "curve": [["0/1", "0/1", "2/1"],
                                    ["1/1", "2/1", "1/1"]]},
            })

    def test_legacy_two_argument_form_is_audit(self):
        input_path = self._write("in.json", json.dumps({
            "arrival": [["0/1", "2/1"], ["1/1", "0/1"]],
            "service": [["0/1", "1/1"]],
        }))
        output_path = self._path("out.json")
        proc = self._run([input_path, output_path])
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(output_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle), {
                "status": "ok",
                "curve": [["0/1", "1/1", "1/1"], ["1/1", "2/1", "0/1"]],
            })

    def test_explicit_audit_mode(self):
        input_path = self._write("in.json", json.dumps({
            "arrival": [["0/1", "1/1"]],
            "service": [["0/1", "1/1"]],
        }))
        output_path = self._path("out.json")
        proc = self._run(["audit", input_path, output_path])
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_recovery_failure_keeps_old_output(self):
        good = self._write("good.json", json.dumps({
            "t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
            "b": ["0/1"], "tail": ["0/1", "0/1"],
        }))
        bad = self._write("bad.json", json.dumps({
            "t": ["0/1", "1/1"], "M": [["1/1", "1/1"]],
            "b": ["0/1"], "tail": ["0/1", "0/1"], "extra": "x",
        }))
        output_path = self._path("out.json")
        self.assertEqual(self._run(["recover", good, output_path]).returncode, 0)
        with open(output_path, "rb") as handle:
            old_bytes = handle.read()
        proc = self._run(["recover", bad, output_path])
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(proc.stderr.startswith("$.extra: unknown member"))
        with open(output_path, "rb") as handle:
            self.assertEqual(handle.read(), old_bytes)

    def test_wrong_usage(self):
        proc = self._run(["not-a-mode", "a", "b"])
        self.assertEqual(proc.returncode, 64)
        self.assertIn("usage:", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
