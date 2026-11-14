"""Command line entry point.

Usage:
    python -m envelope_audit INPUT.json OUTPUT.json
    run_audit.py INPUT.json OUTPUT.json

Reads UTF-8 JSON with exactly the ``arrival`` and ``service`` members,
computes the full-domain envelope, and atomically replaces the output file.
On any validation failure the previous output file is left untouched and the
error (rooted at ``$``) is printed to stderr with exit code 2.
"""

import json
import os
import sys
import tempfile

from . import process_text
from .errors import ValidationError


def run(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    result = process_text(text)

    # Atomic replace: write to a sibling temporary file, flush to disk, then
    # os.replace (atomic on POSIX).  If anything fails before the replace, the
    # old output (if any) remains in place.
    output_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".audit-", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, output_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m envelope_audit INPUT.json OUTPUT.json", file=sys.stderr)
        return 64
    input_path, output_path = argv
    try:
        return run(input_path, output_path)
    except ValidationError as exc:
        # All validation failures (syntax errors start with '$' too).
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"$: cannot read input: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"$: output failure: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
