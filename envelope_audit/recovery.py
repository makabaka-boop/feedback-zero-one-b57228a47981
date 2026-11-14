"""Exact recovery of per-interval flow rates from a linear summary.

A gateway stores, for a horizon split by ``t[0] = 0 < t[1] < ... < t[n]``, a
linear summary of the flows it saw:

* ``x`` holds ``2n`` non-negative interval *amounts*, first the ``n`` input
  amounts, then the ``n`` processing amounts;
* the rate on interval ``i`` is ``x[i] / (t[i+1] - t[i])`` (input) and
  ``x[n+i] / (t[i+1] - t[i])`` (processing);
* after ``t[n]`` the rates are the two entries of ``tail``;
* the stored constraints are ``M x = b`` with ``M`` of shape ``m x 2n``.

Before auditing, the job must decide whether the rates are uniquely
recoverable, i.e. whether ``x`` is the only vector in

    P = { x >= 0 : M x = b }.

Three outcomes are possible:

* ``unique``     -- ``P = {x}``; the audit of the reconstructed rate profiles
                    is run and returned next to ``x``;
* ``ambiguous``  -- ``P`` holds at least two points; two distinct feasible
                    witnesses are returned;
* ``infeasible`` -- ``P`` is empty; a Farkas certificate ``y`` (length ``m``)
                    with ``y^T M >= 0`` and ``y^T b < 0`` is returned.

Uniqueness is decided exactly, never by threshold: take any feasible ``x0``.
Every other feasible point is ``x0 + d`` with ``M d = 0`` and, on the
coordinates where ``x0`` is zero, ``d >= 0``; small positive multiples of such
a ``d`` stay feasible.  Writing ``Z`` for the zero coordinates of ``x0``, the
feasible set is a singleton exactly when

    { d : M d = 0, d_Z >= 0 } = {0}.

That cone is non-trivial iff some coordinate can be fixed to ``+1`` or ``-1``
inside it (cones are closed under positive scaling), which is a plain
non-negativity feasibility problem: free coordinates are split into positive
and negative parts.  Each such test is an exact rational Phase-I LP.
"""

from fractions import Fraction

from .envelope import solve
from .exact_lp import feasibility

ZERO = Fraction(0, 1)
ONE = Fraction(1, 1)


def summary_series(summary, x):
    """Reconstruct the input and processing rate profiles from amounts ``x``.

    Returns ``(arrival, service)`` in the ``(time, rate)`` format consumed by
    :func:`envelope_audit.envelope.solve`: one kink per interval boundary plus
    the final tail kink at ``t[n]``.
    """
    times = summary["t"]
    n = len(times) - 1
    arrival = []
    service = []
    for index in range(n):
        duration = times[index + 1] - times[index]
        arrival.append((times[index], x[index] / duration))
        service.append((times[index], x[n + index] / duration))
    arrival.append((times[n], summary["tail"][0]))
    service.append((times[n], summary["tail"][1]))
    return arrival, service


def _kernel_test_system(matrix, zero_coordinates, coordinate, target):
    """Build the non-negativity system for ``M d = 0, d_Z >= 0, d_j = target``.

    Free coordinates ``j`` are split as ``d_j = u_j - v_j`` with ``u_j, v_j >=
    0``; coordinates in ``Z`` use a single ``d_j = u_j >= 0`` column.  Returns
    ``(rows, rhs, columns)`` where ``columns`` maps a column index to
    ``(coordinate, sign)``.
    """
    columns = []
    for index in range(len(matrix[0])):
        if index in zero_coordinates:
            columns.append((index, 1))
        else:
            columns.append((index, 1))
            columns.append((index, -1))

    rows = []
    for matrix_row in matrix:
        rows.append([sign * matrix_row[index] for index, sign in columns])
    target_row = [ZERO] * len(columns)
    for column_index, (index, sign) in enumerate(columns):
        if index == coordinate:
            target_row[column_index] = Fraction(sign, 1)
    rows.append(target_row)
    rhs = [ZERO] * len(matrix) + [Fraction(target, 1)]
    return rows, rhs, columns


def _nonzero_kernel_direction(matrix, x0):
    """Return ``d`` with ``M d = 0``, ``d_Z >= 0``, ``d != 0``, or ``None``.

    ``Z`` is the set of zero coordinates of ``x0``.  Existence of such a ``d``
    is exactly non-uniqueness of ``{x >= 0 : Mx = b}``.
    """
    zero_coordinates = {index for index, value in enumerate(x0) if value == 0}
    for coordinate in range(len(x0)):
        for target in (1, -1):
            if target == -1 and coordinate in zero_coordinates:
                continue  # d_Z >= 0 forbids a negative value here
            rows, rhs, columns = _kernel_test_system(
                matrix, zero_coordinates, coordinate, target
            )
            feasible, payload = feasibility(rows, rhs)
            if feasible:
                direction = [ZERO] * len(x0)
                for column_index, (index, sign) in enumerate(columns):
                    direction[index] += sign * payload[column_index]
                if any(value != 0 for value in direction):
                    return direction
    return None


def _witness_pair(x0, direction):
    """Two distinct feasible vectors built from ``x0`` and a kernel direction."""
    # A direction in the cone may decrease only coordinates where x0 > 0.
    step = None
    for index, delta in enumerate(direction):
        if delta < 0:
            ratio = x0[index] / (-delta)
            if step is None or ratio < step:
                step = ratio
    if step is None:
        step = ONE
    else:
        step = step / 2
    first = [x0[index] + step * direction[index] for index in range(len(x0))]
    second = [x0[index] + 2 * step * direction[index] for index in range(len(x0))]
    return first, second


def _text_list(values):
    return [f"{value.numerator}/{value.denominator}" for value in values]


def recover(summary):
    """Decide recoverability and, when unique, run the envelope audit.

    ``summary`` is the validated structure returned by
    :func:`envelope_audit.parsing.parse_summary`.
    """
    matrix = summary["M"]
    rhs = summary["b"]
    width = len(matrix[0])

    feasible, payload = feasibility(matrix, rhs)
    if not feasible:
        return {"status": "infeasible", "y": _text_list(payload)}

    x0 = payload
    direction = _nonzero_kernel_direction(matrix, x0)
    if direction is not None:
        first, second = _witness_pair(x0, direction)
        return {
            "status": "ambiguous",
            "witnesses": [_text_list(first), _text_list(second)],
        }

    arrival, service = summary_series(summary, x0)
    return {
        "status": "unique",
        "x": _text_list(x0),
        "audit": solve(arrival, service),
    }


__all__ = ["recover", "summary_series"]
