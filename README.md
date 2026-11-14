# Envelope Audit Job

A pure-backend, dependency-free Python job that computes the exact full-domain
envelope

```
F(t) = sup_{u >= 0} { A(t + u) - S(u) },      t >= 0
```

where `A` is the cumulative arrival input and `S` is the cumulative service
capacity. Both are defined by piecewise-constant rate profiles. Everything is
exact rational arithmetic (`fractions.Fraction`): no floats, no tolerances, no
grid sampling, no finite-time truncation, no symbolic supremum solver and no
third-party network-calculus engine.

## Input contract (UTF-8 JSON)

```json
{
  "arrival": [["0/1", "2/1"], ["1/1", "0/1"]],
  "service": [["0/1", "1/1"]]
}
```

* Exactly the two members `arrival` and `service`. Unknown or duplicate
  members are rejected.
* Each is a non-empty array of `[time, rate]` pairs.
* Every leaf is a string of the exact form `"p/q"`: `p` a non-negative
  integer, `q` a positive integer. No whitespace, signs, decimals or scientific
  notation. Unreduced fractions are accepted on input; outputs are always
  reduced with a positive denominator and keep the denominator even when it
  is `1` (e.g. `"2/1"`).
* The first time of each series must be `0/1`; later times must be strictly
  increasing. The last rate segment extends to `+infinity` and may be zero.
* The first cumulative value is zero; later values follow by integrating the
  rates, so the cumulative functions are continuous by construction.

Validation errors carry the exact field path with zero-based indices, rooted
at `$` (e.g. `$.arrival[2][0]: ...`). JSON syntax errors are reported at `$`
as well.

## Output contract

If, for some fixed `t`, the supremum is `+infinity` (this happens exactly when
the tail arrival rate strictly exceeds the tail service rate), the output is
exactly:

```json
{"status": "unbounded"}
```

The mere fact that a finite `F(t)` grows with `t` is **not** unbounded.

Otherwise:

```json
{
  "status": "ok",
  "curve": [["0/1", "1/1", "1/1"], ["1/1", "2/1", "0/1"]]
}
```

`curve` is the unique minimal continuous piecewise-linear representation of
`F` over the whole domain:

* entries are `[time, F_value, right_slope]`, sorted by increasing time;
* the first entry is at time `0/1`, the last entry represents the infinite
  tail segment;
* adjacent entries always have different slopes — redundant kinks are removed.

## How the core solver works

For a fixed `t`, `g_t(u) = A(t+u) - S(u)` is piecewise linear in `u`, so its
supremum is attained at a finite kink (or at infinity). Every possible
maximizer is one of:

1. `u = 0`;
2. a service kink `u = beta_j`;
3. a shifted arrival kink `u = alpha_i - t` (only while `u >= 0`);
4. `u -> +infinity`, finite only when the two tail rates are equal.

Substituting each into the piecewise-linear definitions of `A` and `S` turns
every candidate into a set of straight-line segments of `t`, each valid on an
explicit window. `F` is the pointwise maximum of all these segments. The upper
envelope of a set of rational lines on each interval is built with an exact
monotone convex-chain construction (lines sorted by slope, intersections
computed as `Fraction`s), so envelope kinks that are not input breakpoints
(e.g. crossings of two candidate lines) are discovered exactly.

## Layout

```
envelope_audit/
  parsing.py     strict UTF-8 JSON validation, $-rooted error paths
  envelope.py    exact cumulative integration, candidate expansion, envelope
  cli.py         command line entry + atomic output replacement
  errors.py      ValidationError
run_audit.py     self-bootstrapping entry point (works under python -I)
examples/        spec example, an interior-kink example, an unbounded example
tests/
  test_audit.py        unit tests: hand curves, random brute verification,
                       schema paths, CLI atomic-replace acceptance
  cross_validate.py    heavy independent brute cross-validation
run_tests.sh      one-command reproducible acceptance
```

## Running

Process one file to another (the output is replaced atomically: a sibling
temporary file is written, fsynced and `os.replace`d; on any failure the
previous output is left untouched):

```
python3 run_audit.py INPUT.json OUTPUT.json
# or, when the package is importable:
python3 -m envelope_audit INPUT.json OUTPUT.json
```

Exit codes: `0` success, `2` validation/input error (message on stderr,
rooted at `$`), `3` output I/O failure, `64` wrong usage.

## Reproducible acceptance

```
./run_tests.sh
```

This runs the unit suite, 300 random rational profiles through an independent
brute-force reference (11k+ exact point checks, continuity and slope checks,
upper-bound checks), and CLI smoke tests. The brute reference in the tests is
written from scratch and shares no solver code, so agreement is an
independent check rather than a self-consistency tautology.

Hand-computed coverage includes: the spec example, equal constant rates
(`F(t) = t`, still finite), equal tail slopes with an interior kink, zero
arrival, zero service, multiple competing local extrema, an envelope kink at a
non-breakpoint intersection, fractional data, and the unbounded tail gap.
