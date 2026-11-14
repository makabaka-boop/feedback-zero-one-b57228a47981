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

Two modes share the same binary:

* **`audit`** -- the original full-domain envelope audit (`arrival` /
  `service` input, described below);
* **`recover`** -- a bounded flow-summary recovery layer (`t` / `M` / `b` /
  `tail` input): it decides whether the interval rates are *uniquely*
  recoverable before auditing them. See
  [Flow-summary recovery](#flow-summary-recovery) further down.

The historical two-argument invocation (`INPUT.json OUTPUT.json`) still works
and remains an alias for `audit`.

## Input contract for `audit` (UTF-8 JSON)

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

## Flow-summary recovery

The gateway stores only a *linear summary* of the flows, so before auditing,
the job must confirm the rates are uniquely recoverable. `recover` mode does
exactly that.

### Input contract (UTF-8 JSON)

```json
{
  "t": ["0/1", "1/1"],
  "M": [["1/1", "0/1"], ["0/1", "1/1"]],
  "b": ["2/1", "3/1"],
  "tail": ["1/1", "1/1"]
}
```

* Exactly the four members `t`, `M`, `b`, `tail`; unknown or duplicate members
  are rejected.
* `t` has length `n + 1` with `n >= 1`: it starts at `0/1`, is non-negative
  and strictly increasing. Interval `i` is the half-open `[t[i], t[i+1])`.
* `M` is a non-empty `m x 2n` matrix; `b` has length `m`. Their entries are
  *signed* `p/q` strings (the flow amounts themselves stay non-negative).
* `tail` is exactly `[input tail rate, processing tail rate]`, both
  non-negative.
* Every leaf is a string `"p/q"`: integer `p`, positive `q`, no whitespace,
  signs only where noted. Outputs are always reduced with a positive
  denominator and keep `/1`.

### Semantics

The unknown `x` is the concatenation of `n` input amounts and `n` processing
amounts (all `x[j] >= 0`), constrained by `M x = b`. The rate on interval `i`
is the amount divided by its duration; after `t[n]` the rates are `tail`.

The job returns one of:

```json
{"status": "unique",     "x": ["p/q", ...], "audit": {...}}
{"status": "ambiguous",  "witnesses": [["p/q", ...], ["p/q", ...]]}
{"status": "infeasible", "y": ["p/q", ...]}
```

* `unique` -- `x` is the **only** feasible vector. The rates are reconstructed
  and the original envelope audit runs on them; its untouched result is
  nested under `audit` (so it may itself be `ok` or `unbounded`).
* `ambiguous` -- at least two distinct feasible vectors exist; two are
  returned. Witnesses are chosen freely (not canonicalised).
* `infeasible` -- no feasible `x` exists; `y` has length `m` and satisfies
  `y^T M >= 0` componentwise and `y^T b < 0`, which proves infeasibility by
  pure arithmetic.

Since `x >= 0`, `Mx = b` alone is not enough for uniqueness: a rank-deficient
`M` can still pin one solution (e.g. `x1 + x2 = 0`), and a full-rank system can
still have none. The solver therefore never decides by rank or by a float
threshold. It takes one feasible `x0`, and asks whether the cone
`{ d : M d = 0, d_j >= 0 for x0_j = 0 }` is trivial -- exactly the condition
for a singleton feasible set. That test is itself an exact rational
feasibility LP (free coordinates split into positive and negative parts), so
uniqueness, ambiguity and infeasibility are all established by
`fractions.Fraction` arithmetic only.

The linear programs are solved by an exact Phase-I simplex with Bland's rule
(`envelope_audit/exact_lp.py`). Every pivot is a `Fraction` operation; the
Farkas certificate is read off from the final basis by solving `B^T y = c_B`
exactly, so infeasibility is proved by exact arithmetic rather than by
rounding-based rank tests.

### Examples

`examples/recover_*.json` hold one independent instance of each verdict, all
checked by `run_tests.sh`:

| file | verdict |
| --- | --- |
| `recover_unique.json` | `unique`, `x = [2, 3]` |
| `recover_unique_rank_deficient.json` | `unique` despite `rank(M) < 2n` |
| `recover_ambiguous.json` | `ambiguous` via an unconstrained coordinate |
| `recover_ambiguous_bounded.json` | `ambiguous` via a convex split |
| `recover_infeasible.json` | `infeasible` with certificate `y = [1, -1]` |

The specification example `t = [0, 1]`, `M = [[1, 1]]`, `b = [0]`,
`tail = [0, 0]` recovers `x = [0, 0]` uniquely.

## Layout

```
envelope_audit/
  parsing.py     strict UTF-8 JSON validation, $-rooted error paths
  envelope.py    exact cumulative integration, candidate expansion, envelope
  exact_lp.py    exact rational Phase-I simplex + Farkas certificates
  recovery.py    recovery/uniqueness analysis, summary-to-profile reconstruction
  cli.py         command line entry + atomic output replacement
  errors.py      ValidationError
run_audit.py     self-bootstrapping entry point (works under python -I)
examples/        spec example, interior-kink example, unbounded example,
                 and one recovery example per verdict
tests/
  test_audit.py        unit tests: hand curves, random brute verification,
                       schema paths, CLI atomic-replace acceptance
  test_recovery.py     recovery verdicts, certificate checks, schema paths,
                       recovery CLI acceptance
  cross_validate.py    heavy independent brute cross-validation
run_tests.sh      one-command reproducible acceptance
```

## Running

Process one file to another (the output is replaced atomically: a sibling
temporary file is written, fsynced and `os.replace`d; on any failure the
previous output is left untouched):

```
python3 run_audit.py audit   INPUT.json OUTPUT.json
python3 run_audit.py recover INPUT.json OUTPUT.json
python3 run_audit.py         INPUT.json OUTPUT.json   # legacy = audit
# or, when the package is importable:
python3 -m envelope_audit {audit|recover} INPUT.json OUTPUT.json
```

Exit codes: `0` success, `2` validation/input error (message on stderr,
rooted at `$`), `3` output I/O failure, `64` wrong usage.

## Reproducible acceptance

```
./run_tests.sh
```

This runs the unit suite, 300 random rational profiles through an independent
brute-force reference (11k+ exact point checks, continuity and slope checks,
upper-bound checks), and CLI smoke tests for both modes. The brute reference in
the tests is written from scratch and shares no solver code, so agreement is an
independent check rather than a self-consistency tautology.

Hand-computed coverage includes: the spec example, equal constant rates
(`F(t) = t`, still finite), equal tail slopes with an interior kink, zero
arrival, zero service, multiple competing local extrema, an envelope kink at a
non-breakpoint intersection, fractional data, and the unbounded tail gap. The
recovery layer is covered by a rank-deficient-but-unique summary, an
underdetermined summary with infinitely many solutions, a contradictory
summary, and randomised exact checks of every witness and certificate.

