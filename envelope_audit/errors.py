"""Error type shared by parsing and the CLI."""


class ValidationError(Exception):
    """Raised for any malformed input.

    The message always starts at the JSON root ``$`` and carries the exact
    field path (with zero-based indices), e.g. ``$.arrival[2][0]``.
    """
