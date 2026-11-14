"""Strict UTF-8 JSON validation for the audit and recovery modes.

Two schemas are accepted, each on its own path; anything else is rejected with
a ``$``-rooted field path.

Envelope audit (``parse_input``)::

    {
      "arrival": [[time, rate], ...],
      "service": [[time, rate], ...]
    }

* exactly the two members ``arrival`` / ``service``, no unknown or duplicate
  members;
* both fields are non-empty arrays of 2-element arrays;
* every leaf is a string of the exact form ``"p/q"``: p non-negative integer,
  q positive integer, no whitespace, no other numeric form;
* the first time of each series is zero, all following times are strictly
  increasing.

Flow-summary recovery (``parse_summary``)::

    {
      "t": ["0/1", ..., "1/1"],          # n + 1 strictly increasing times
      "M": [["p/q", ...], ...],          # non-empty m x 2n, signed entries
      "b": ["p/q", ...],                 # length m, signed entries
      "tail": ["p/q", "p/q"]             # input tail, processing tail
    }

``M`` and ``b`` accept an optional leading minus so the stored linear
constraints can be arbitrary; the amounts themselves stay non-negative at the
mathematical layer.

All arithmetic downstream uses :class:`fractions.Fraction`, so values are
exact rationals -- no floats, no tolerances.  Unreduced input fractions are
accepted.
"""

import json
import re
from fractions import Fraction

from .errors import ValidationError

# Exact "p/q" only: digits, a single slash, digits.  No sign, no decimal
# point, no whitespace, no scientific notation.
_RATIONAL_RE = re.compile(r"[0-9]+/[0-9]+")
# Same shape with an optional leading minus, for members that may be negative
# (the recovery summary's matrix ``M`` and vector ``b``).
_SIGNED_RATIONAL_RE = re.compile(r"-?[0-9]+/[0-9]+")
_KNOWN_MEMBERS = ("arrival", "service")


class _Object(list):
    """Marker type: a JSON object, preserved as an ordered list of pairs.

    Using ``object_pairs_hook`` keeps duplicate keys visible (plain ``dict``
    would silently drop them) and lets the validator distinguish a JSON
    object from a JSON array, which both arrive as Python ``list`` otherwise.
    """


def _pairs_hook(pairs):
    return _Object(pairs)


def _type_name(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, _Object):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _parse_fraction(value, path):
    return _parse_rational(value, path, _RATIONAL_RE, "non-negative")


def _parse_signed_fraction(value, path):
    return _parse_rational(value, path, _SIGNED_RATIONAL_RE, "signed")


def _parse_rational(value, path, pattern, kind):
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValidationError(
            f"{path}: expected a p/q string, got {_type_name(value)}"
        )
    if not pattern.fullmatch(value):
        raise ValidationError(
            f"{path}: invalid rational {value!r}; expected {kind} 'p/q' with "
            "integer p, positive integer q and no whitespace"
        )
    p_text, q_text = value.split("/")
    q = int(q_text)
    if q == 0:
        raise ValidationError(f"{path}: denominator must be positive, got {value!r}")
    # Fraction normalizes the sign and reduces the fraction.
    return Fraction(int(p_text), q)


def _validate_series(node, path):
    if isinstance(node, _Object) or not isinstance(node, list):
        raise ValidationError(f"{path}: expected a non-empty array, got {_type_name(node)}")
    if len(node) == 0:
        raise ValidationError(f"{path}: array must be non-empty")

    series = []
    previous_time = None
    for index, item in enumerate(node):
        item_path = f"{path}[{index}]"
        if isinstance(item, _Object) or not isinstance(item, list):
            raise ValidationError(
                f"{item_path}: expected a [time, rate] array, got {_type_name(item)}"
            )
        if len(item) != 2:
            raise ValidationError(
                f"{item_path}: pair must contain exactly 2 elements, got {len(item)}"
            )
        time = _parse_fraction(item[0], f"{item_path}[0]")
        rate = _parse_fraction(item[1], f"{item_path}[1]")
        if index == 0:
            if time != 0:
                raise ValidationError(
                    f"{item_path}[0]: first time must be 0/1 (zero), got {item[0]!r}"
                )
        elif time <= previous_time:
            raise ValidationError(
                f"{item_path}[0]: times must be strictly increasing; "
                f"{time} is not greater than {previous_time}"
            )
        previous_time = time
        series.append((time, rate))
    return series


def _load_object(text, known_members):
    """Parse strict JSON text and validate the top-level member set.

    Returns a ``dict`` of member name -> raw value.  Syntax errors, a non-object
    root, duplicate members and unknown/missing members are all reported at a
    ``$``-rooted path.
    """
    try:
        root = json.loads(text, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        # JSON syntax errors are reported at the root path, per contract.
        raise ValidationError(
            f"$: invalid JSON syntax: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc

    if not isinstance(root, _Object):
        raise ValidationError(f"$: root must be a JSON object, got {_type_name(root)}")

    seen = set()
    for key, _value in root:
        if key in seen:
            raise ValidationError(f"$.{key}: duplicate member")
        seen.add(key)
    for key in seen:
        if key not in known_members:
            raise ValidationError(f"$.{key}: unknown member")
    missing = [key for key in known_members if key not in seen]
    if missing:
        raise ValidationError(f"$: missing required member(s): {', '.join(missing)}")
    return dict(root)


def parse_input(text):
    """Parse and validate the raw JSON text.

    Returns ``(arrival, service)`` where each is a list of
    ``(Fraction time, Fraction rate)`` tuples.
    """
    members = _load_object(text, _KNOWN_MEMBERS)
    arrival = _validate_series(members["arrival"], "$.arrival")
    service = _validate_series(members["service"], "$.service")
    return arrival, service


# ---------------------------------------------------------------------------
# Bounded flow-summary schema (the recovery layer).
# ---------------------------------------------------------------------------

_SUMMARY_MEMBERS = ("t", "M", "b", "tail")


def _validate_array(node, path):
    if isinstance(node, _Object) or not isinstance(node, list):
        raise ValidationError(f"{path}: expected an array, got {_type_name(node)}")
    return node


def _validate_summary(text):
    members = _load_object(text, _SUMMARY_MEMBERS)

    # --- t: length n + 1, starting at zero, strictly increasing, non-negative.
    t_node = _validate_array(members["t"], "$.t")
    if len(t_node) < 2:
        raise ValidationError("$.t: must contain n+1 >= 2 times (n >= 1)")
    times = []
    for index, value in enumerate(t_node):
        time = _parse_fraction(value, f"$.t[{index}]")
        if index == 0:
            if time != 0:
                raise ValidationError(
                    f"$.t[0]: first time must be 0/1 (zero), got {value!r}"
                )
        elif time <= times[-1]:
            raise ValidationError(
                f"$.t[{index}]: times must be strictly increasing; "
                f"{time} is not greater than {times[-1]}"
            )
        times.append(time)
    n = len(times) - 1

    # --- M: non-empty m x 2n matrix of signed rationals.
    m_node = _validate_array(members["M"], "$.M")
    if len(m_node) == 0:
        raise ValidationError("$.M: matrix must be non-empty (m >= 1)")
    matrix = []
    for row_index, row_node in enumerate(m_node):
        row_path = f"$.M[{row_index}]"
        _validate_array(row_node, row_path)
        if len(row_node) != 2 * n:
            raise ValidationError(
                f"{row_path}: row must have 2n = {2 * n} entries, "
                f"got {len(row_node)}"
            )
        matrix.append([
            _parse_signed_fraction(value, f"{row_path}[{column_index}]")
            for column_index, value in enumerate(row_node)
        ])
    m = len(matrix)

    # --- b: length m, signed rationals.
    b_node = _validate_array(members["b"], "$.b")
    if len(b_node) != m:
        raise ValidationError(f"$.b: must have m = {m} entries, got {len(b_node)}")
    rhs = [_parse_signed_fraction(value, f"$.b[{index}]")
           for index, value in enumerate(b_node)]

    # --- tail: exactly two non-negative rates (input tail, service tail).
    tail_node = _validate_array(members["tail"], "$.tail")
    if len(tail_node) != 2:
        raise ValidationError(
            f"$.tail: must contain exactly 2 rates ([input, service]), "
            f"got {len(tail_node)}"
        )
    tail = [_parse_fraction(value, f"$.tail[{index}]")
            for index, value in enumerate(tail_node)]

    return {"t": times, "M": matrix, "b": rhs, "tail": tail}


def parse_summary(text):
    """Parse a bounded flow summary.

    Returns ``{"t": [Fraction], "M": [[Fraction]], "b": [Fraction],
    "tail": [Fraction]}`` with ``len(t) == n + 1``, ``n >= 1``, ``M`` of
    shape ``m x 2n`` and ``len(b) == m``.
    """
    return _validate_summary(text)
