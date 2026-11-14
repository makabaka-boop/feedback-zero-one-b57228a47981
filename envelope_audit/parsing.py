"""Strict UTF-8 JSON validation for the envelope audit job.

Schema (anything else is rejected with a ``$``-rooted field path)::

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

All arithmetic downstream uses :class:`fractions.Fraction`, so values are
exact rationals -- no floats, no tolerances.
"""

import json
import re
from fractions import Fraction

from .errors import ValidationError

# Exact "p/q" only: digits, a single slash, digits.  No sign, no decimal
# point, no whitespace, no scientific notation.
_RATIONAL_RE = re.compile(r"[0-9]+/[0-9]+")
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
    if isinstance(value, bool) or not isinstance(value, str):
        raise ValidationError(
            f"{path}: expected a p/q string, got {_type_name(value)}"
        )
    if not _RATIONAL_RE.fullmatch(value):
        raise ValidationError(
            f"{path}: invalid rational {value!r}; expected 'p/q' with a "
            "non-negative integer p, positive integer q and no whitespace"
        )
    p_text, q_text = value.split("/")
    p, q = int(p_text), int(q_text)
    if q == 0:
        raise ValidationError(f"{path}: denominator must be positive, got {value!r}")
    # Fraction normalizes the sign and reduces the fraction.
    return Fraction(p, q)


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


def parse_input(text):
    """Parse and validate the raw JSON text.

    Returns ``(arrival, service)`` where each is a list of
    ``(Fraction time, Fraction rate)`` tuples.
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
        if key not in _KNOWN_MEMBERS:
            raise ValidationError(f"$.{key}: unknown member")
    missing = [key for key in _KNOWN_MEMBERS if key not in seen]
    if missing:
        raise ValidationError(f"$: missing required member(s): {', '.join(missing)}")

    members = dict(root)
    arrival = _validate_series(members["arrival"], "$.arrival")
    service = _validate_series(members["service"], "$.service")
    return arrival, service
