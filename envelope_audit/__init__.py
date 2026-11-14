"""Pure-backend audit job computing the arrival/service envelope F(t).

Two entry points live side by side:

* the original full-domain envelope audit (``arrival`` / ``service`` input);
* the bounded flow-summary recovery layer (``t`` / ``M`` / ``b`` / ``tail``
  input), which decides whether the interval rates are uniquely recoverable
  and runs the envelope audit on the recovered profiles only when they are.
"""

from .envelope import solve
from .errors import ValidationError
from .parsing import parse_input, parse_summary
from .recovery import recover

__all__ = [
    "parse_input",
    "parse_summary",
    "recover",
    "solve",
    "ValidationError",
    "process_summary_text",
    "process_text",
]


def process_text(text):
    """Validate raw envelope-audit JSON text and return the audit result."""
    arrival, service = parse_input(text)
    return solve(arrival, service)


def process_summary_text(text):
    """Validate raw flow-summary JSON text and return the recovery result."""
    return recover(parse_summary(text))
