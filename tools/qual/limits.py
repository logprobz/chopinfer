"""The limit rules of the quality gates, implemented once, in Rust
(crates/runtime-eval/src/limits.rs): each function here is one `limit` call
of `chopinfer eval rescore` (kinds.evaluated), and the constants are read
from it once per process.

Spread rule (variant-spread.v2): a check scored against a reference's gate
variant is held to the floors of the numerical contract, raised to the
reference's own spread. Each other variant of the reference, scored against
the gate variant like a server, has a mean |dlogprob| and a decisive top-1
rate; then

    mean limit = max(mean floor, worst variant mean x mean_factor rounded up
                     to mean_step)
    top-1 limit = max(top-1 floor, worst variant decisive top-1 rate)
    near-tie limit = max(near-tie floor, worst variant near-tie rate
                         rounded up to near_tie_step)
    total top-1 limit = max(total floor, worst variant top-1 disagreement)

The mean floor is LARGE_SET_MEAN_FLOOR for a set of at least
LARGE_SET_POSITIONS scored positions and MEAN_FLOOR below. mean_factor is 1
unless a gate declares otherwise: a factor above 1 keeps a proportional
headroom over the spread, so a spread just below a mean_step multiple does
not leave the limit at the spread itself. A gate may declare a cap on the
mean limit; the default rule has none. A teacher category or media case
whose top-1 limit sits within two binomial standard errors of the bf16
variant's own rate there also passes that metric by the paired rule's top-1
clause against the bf16 variant (category power, docs/check-kinds.md); a
pooled set never does.

Near-tie rule (top1-near-tie.v1): a scored position where the engine's top-1
differs from the reference's is a flip. The reference's own logprob gap from
its top-1 to the engine's top-1, read from its stored top-k, makes the flip a
near tie when it is at most NEAR_TIE_NATS, and decisive otherwise or when
the engine's token is outside the stored top-k. The limit is inclusive, as
in the image-probe gate. The top-1 limit applies to the
decisive rate (decisive flips over scored positions), and the near-tie limit
to the near-tie rate, so systematic near-tie flips (the two leading tokens
swapped while the teacher token's logprob holds) fail too. All flips together
are held to the total top-1 limit, so the two allowances cannot add up past
the reference's own disagreement (or the total floor). The paired GSM8K rule
(gsm8k_branches.py) reads the gap between the two arms' branch tokens the
same way.

Paired rule (paired-margin.v1): a test arm and a baseline arm scored on the
same positions. The test arm may sit at most a mean margin farther from the
reference on average, and its extra top-1 disagreements must stay within a
number of standard deviations of the discordant count.

Exact rule: two answers are equal bit for bit (floats by their IEEE 754
bits); the first difference is reported.
"""

from functools import cache

import kinds

CONSTANTS = ("MEAN_FLOOR", "LARGE_SET_MEAN_FLOOR", "LARGE_SET_POSITIONS", "TOP1_FLOOR", "MEAN_STEP", "MEAN_FACTOR",
             "NEAR_TIE_FLOOR", "NEAR_TIE_STEP", "TOP1_TOTAL_FLOOR", "NEAR_TIE_NATS", "POSITION_FIELDS",
             "ANSWER_FIELDS")


@cache
def constants():
    return evaluate([("constants", {})])[0]


def __getattr__(name):
    if name in CONSTANTS:
        value = constants()[name]
        return tuple(value) if name.endswith("_FIELDS") else value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def evaluate(calls, error=ValueError):
    """The values of `limit` calls, (rule, args) each, in order; a rule
    that cannot judge its input raises `error`."""
    results = kinds.evaluated([{"op": "limit", "rule": rule, "args": args} for rule, args in calls])
    for result in results:
        if "error" in result:
            raise error(result["error"]["message"])
    return [result["value"] for result in results]


def one(name, error=ValueError, **args):
    return evaluate([(name, args)], error)[0]


def rounded_up(value, step):
    """`value` rounded up to a multiple of `step`, both read as their shortest
    decimals: 0.15 is a multiple of 0.05, while 0.1500000001 and the binary
    sum 0.1 + 0.05 (0.15000000000000002) round up to 0.2."""
    return one("rounded_up", ArithmeticError, value=value, step=step)


def near_tie(gap, limit=None):
    """Whether two tokens `gap` nats apart are a near tie: the gap is known
    and at most `limit` (NEAR_TIE_NATS) in magnitude."""
    return one("near_tie", gap=gap, limit=limit)


def flip(top1, served, top=None, limit=None):
    """The near-tie rule at one scored position: None when the engine's top-1
    `served` is the reference's `top1`; otherwise the reference's logprob gap
    from its top-1 to `served`, read from `top`, its stored top-k there as
    [token, logprob] pairs, and whether the flip is decisive."""
    return one("flip", top1=top1, served=served, top=top, limit=limit)


def flips(positions, limit=None):
    """`flip` at each of `positions`, (top1, served, top) each, in one call."""
    return one("flips", positions=[{"top1": top1, "served": served, "top": top} for top1, served, top in positions],
               limit=limit)


def top1_summary(flips, scored):
    """Raw, near-tie and decisive top-1 counts and rates over `scored`
    positions, from the flips (`flip` results, None for agreement) there."""
    return one("top1_summary", flips=list(flips), scored=scored)


def spread_limits(variants, positions, **rule):
    """The spread rule's limits of a set of `positions` scored positions.
    `variants` holds each other reference variant's `mean_abs_dlogprob`,
    `top1_decisive`, `top1_near_tie` and `top1_disagreement` rates against
    the gate variant over the same positions; `rule` replaces the floors,
    steps, factor or cap."""
    return one("spread_limits", ArithmeticError, variants=list(variants), positions=positions, rule=rule)


def within(measured, limits):
    """Whether measured scores meet spread limits."""
    return one("within", measured=measured, limits=limits)


def paired(test, baseline, *, mean_margin, sigmas):
    """The paired rule over two arms' (|dlogprob|, disagrees) at the same
    positions."""
    return one("paired", test=[list(pair) for pair in test], baseline=[list(pair) for pair in baseline],
               mean_margin=mean_margin, sigmas=sigmas)


def bit_equal(a, b):
    """Equal values, with any float and the number it is compared with equal
    only in their IEEE 754 bits."""
    return one("bit_equal", left=a, right=b)


def first_difference(left, right, positions=None, answer=None):
    """The exact rule: the first (field, position, left, right) where two
    answers differ, position by position over the `positions` fields and then
    once over the `answer` fields (position None); None when bit-equal."""
    value = one("first_difference", left=left, right=right,
                positions=None if positions is None else list(positions),
                answer=None if answer is None else list(answer))
    return None if value is None else tuple(value)
