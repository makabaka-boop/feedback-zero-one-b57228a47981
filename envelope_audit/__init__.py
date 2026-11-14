"""Pure-backend audit job computing the arrival/service envelope F(t)."""

from .envelope import solve
from .errors import ValidationError
from .parsing import parse_input

__all__ = ["parse_input", "solve", "ValidationError", "process_text"]


def process_text(text):
    """Validate raw JSON text and return the serializable audit result."""
    arrival, service = parse_input(text)
    return solve(arrival, service)
