#!/usr/bin/env python3
"""Thin entry point so the job can be run without installing the package:

    python run_audit.py {audit|recover} path/to/input.json path/to/output.json
    python run_audit.py path/to/input.json path/to/output.json   # legacy

The two-argument legacy form still means ``audit``.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from envelope_audit.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
