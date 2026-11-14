"""Command line entry point.

Usage:
    python -m envelope_audit MODE INPUT.json OUTPUT.json
    run_audit.py MODE INPUT.json OUTPUT.json

``MODE`` is either

* ``audit``   -- the original envelope audit; input has exactly the members
                 ``arrival`` and ``service`` (fully backwards compatible);
* ``recover`` -- the bounded flow-summary recovery layer; input has exactly
                 the members ``t``, ``M``, ``b`` and ``tail``.

Legacy compatibility: the two-argument form ``INPUT.json OUTPUT.json`` is
still accepted and is treated as ``audit``.

Both modes write UTF-8 JSON and atomically replace the output file: a sibling
temporary file is written, fsynced and ``os.replace``d.  On any validation or
computation failure the previous output file is left untouched and the error
(rooted at ``$``) is printed to stderr with exit code 2.
"""

import json
import os
import sys
import tempfile

from . import process_summary_text, process_text
from .errors import ValidationError

_MODES = {"audit": process_text, "recover": process_summary_text}
_LEGACY_MODE = "audit"


def _process(mode, text):
    return _MODES[mode](text)


def run(mode, input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as handle:
        text = handle.read()

    result = _process(mode, text)

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
    if len(argv) == 2:
        mode = _LEGACY_MODE
        input_path, output_path = argv
    elif len(argv) == 3 and argv[0] in _MODES:
        mode, input_path, output_path = argv
    else:
        print(
            "usage: python -m envelope_audit {audit|recover} INPUT.json OUTPUT.json",
            file=sys.stderr,
        )
        return 64
    try:
        return run(mode, input_path, output_path)
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
