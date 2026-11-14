#!/usr/bin/env bash
# Reproducible acceptance run for the envelope audit job.
# Uses only the Python standard library; -I isolates the interpreter from
# site packages and the current directory's sys.path surprises.
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "== unit tests (strict schema, hand curves, recovery, CLI atomic replace) =="
"$PYTHON" -I -m unittest discover -s tests -v

echo
echo "== heavy independent cross-validation (300 random rational profiles) =="
"$PYTHON" -I tests/cross_validate.py 20261114 300

echo
echo "== CLI smoke tests (legacy two-argument audit form) =="
"$PYTHON" -I run_audit.py examples/spec_example.json /tmp/audit_spec_out.json
cat /tmp/audit_spec_out.json
"$PYTHON" -I run_audit.py audit examples/interior_kink.json /tmp/audit_kink_out.json
cat /tmp/audit_kink_out.json
"$PYTHON" -I run_audit.py audit examples/unbounded.json /tmp/audit_unbounded_out.json
cat /tmp/audit_unbounded_out.json

echo
echo "== CLI smoke tests (recovery mode) =="
for example in unique unique_rank_deficient ambiguous ambiguous_bounded infeasible; do
  "$PYTHON" -I run_audit.py recover "examples/recover_${example}.json" \
    "/tmp/recover_${example}_out.json"
  cat "/tmp/recover_${example}_out.json"
done

echo
echo "ALL ACCEPTANCE CHECKS PASSED"
