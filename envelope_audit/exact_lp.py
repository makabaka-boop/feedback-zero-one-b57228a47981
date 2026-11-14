"""Exact rational linear-programming primitives.

Only *feasibility* of systems ``{z >= 0 : A z = b}`` is needed by the
recovery job.  When such a system is infeasible the solver returns a Farkas
certificate ``y`` with

    y^T A >= 0      (componentwise)      and      y^T b < 0,

which proves infeasibility by direct arithmetic: any feasible ``z >= 0``
would give ``0 <= y^T A z = y^T b < 0``.

The implementation is an exact Phase-I simplex over the sign-normalised rows
of ``b``: artificial variables form the initial basis and the objective is
their sum.  Bland's rule (smallest entering column, smallest leaving basic
variable on ratio ties) guarantees finite termination without cycling.

Everything is :class:`fractions.Fraction`: no floating point, no tolerances,
no numerical rank decisions.
"""

from fractions import Fraction

ZERO = Fraction(0, 1)
ONE = Fraction(1, 1)


def _solve_linear(matrix, rhs):
    """Solve ``matrix * x = rhs`` exactly (Gauss-Jordan); ValueError if singular."""
    size = len(matrix)
    work = [list(row) + [rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = None
        for row in range(column, size):
            if work[row][column] != 0:
                pivot = row
                break
        if pivot is None:
            raise ValueError("singular matrix")
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
        scale = work[column][column]
        work[column] = [value / scale for value in work[column]]
        for row in range(size):
            if row == column:
                continue
            factor = work[row][column]
            if factor != 0:
                work[row] = [
                    work[row][index] - factor * work[column][index]
                    for index in range(size + 1)
                ]
    return [work[row][size] for row in range(size)]


def feasibility(matrix, rhs):
    """Decide whether ``{z >= 0 : matrix z = rhs}`` is non-empty.

    ``matrix`` is a rectangular list of rows of :class:`Fraction`; ``rhs`` is
    a list of :class:`Fraction` of the same height.  Returns

    * ``(True, z)``  -- ``z`` is a feasible vector (length = column count);
    * ``(False, y)`` -- ``y`` is a Farkas certificate (length = row count)
      with ``y^T matrix >= 0`` and ``y^T rhs < 0``.
    """
    row_count = len(rhs)
    width = len(matrix[0]) if matrix else 0
    if row_count == 0:
        return True, [ZERO] * width

    # Scale each row by -1 where needed so the right-hand side is >= 0; the
    # artificial identity block then gives a feasible initial basis.
    signs = [ONE if rhs[index] >= 0 else -ONE for index in range(row_count)]
    rows = [[signs[index] * value for value in matrix[index]]
            for index in range(row_count)]
    scaled_rhs = [signs[index] * rhs[index] for index in range(row_count)]

    total = width + row_count
    tableau = [rows[index] + [ONE if column == index else ZERO
                              for column in range(row_count)]
               for index in range(row_count)]
    basic = [width + index for index in range(row_count)]
    values = list(scaled_rhs)

    costs = [ZERO] * width + [ONE] * row_count

    def multipliers():
        """Dual vector y' solving B^T y' = c_B for the current basis."""
        def basis_entry(row, column):
            source = basic[column]
            if source < width:
                return rows[row][source]
            return ONE if row == source - width else ZERO

        basis_matrix = [[basis_entry(row, column) for column in range(row_count)]
                        for row in range(row_count)]
        transposed = [[basis_matrix[column][row] for column in range(row_count)]
                      for row in range(row_count)]
        return _solve_linear(transposed, [costs[source] for source in basic])

    while True:
        basic_costs = [costs[source] for source in basic]
        entering = None
        for column in range(total):
            reduced = costs[column]
            for index in range(row_count):
                if basic_costs[index] != 0:
                    reduced -= basic_costs[index] * tableau[index][column]
            if reduced < 0:
                entering = column  # Bland: smallest qualifying column
                break
        if entering is None:
            break

        direction = [tableau[index][entering] for index in range(row_count)]
        candidates = [index for index in range(row_count) if direction[index] > 0]
        if not candidates:
            # The Phase-I objective is bounded below by zero, so this cannot
            # occur for a consistent tableau.
            raise AssertionError("phase I objective unbounded")
        ratio = min(values[index] / direction[index] for index in candidates)
        leaving = min(
            (index for index in candidates
             if values[index] / direction[index] == ratio),
            key=lambda index: basic[index],  # Bland: smallest basic variable
        )
        pivot = direction[leaving]
        tableau[leaving] = [value / pivot for value in tableau[leaving]]
        values[leaving] = values[leaving] / pivot
        for index in range(row_count):
            if index == leaving:
                continue
            factor = direction[index]
            if factor != 0:
                tableau[index] = [
                    tableau[index][column] - factor * tableau[leaving][column]
                    for column in range(total)
                ]
                values[index] -= factor * values[leaving]
        basic[leaving] = entering

    objective = sum(costs[source] * values[index]
                    for index, source in enumerate(basic))
    if objective > 0:
        dual = multipliers()
        # y' is dual-feasible for the normalised system: y'^T (S A) <= 0 and
        # y'^T (S b) = objective > 0.  Undo the row scaling.
        certificate = [-signs[index] * dual[index] for index in range(row_count)]
        return False, certificate

    solution = [ZERO] * width
    for index, source in enumerate(basic):
        if source < width:
            solution[source] = values[index]
    return True, solution
