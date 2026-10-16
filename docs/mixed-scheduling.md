# Mixed prefill and decode scheduling

While a step prefills, every decoding request in it waits for the whole step.
With `--prefill-budget fixed`, a step beside decodes carries a full prefill
chunk (2,048 tokens by default): on MiMo V2.6 TP4 that is about 200 ms, 14
times a decode-only step. `--prefill-budget tbt` caps the prefill of each step
beside decoding requests so the predicted step time fits a target. It changes
only batch composition and chunk boundaries; for MiMo V2.6 and Qwen3.8, chunk
boundaries do not change prefill logits ([numerics](numerics.md)).

## When to use it

Use `tbt` when streaming smoothness matters and prompts keep arriving while
other requests decode, as in agentic coding. Keep `fixed` for batch and
throughput jobs.

The MiMo `tp4-mtp` and `tp4-dflash` profiles select `tbt` with a 150 ms
target. The other profiles, including every Qwen profile, use `fixed`, since
`tbt` costs Qwen throughput. Explicit
flags override the profile; without a profile the default is `fixed`. For a
manual setup, pass `--prefill-budget tbt --decode-tbt-target-ms 150`.

| Flag | Default | Meaning |
|---|---|---|
| `--decode-tbt-target-ms N` | 100 | Target predicted time of a step that prefills beside decodes |
| `--mixed-min-prefill-tokens N` | 512 | Progress floor, and the cap before the model trains |
| `--mixed-decode-steps N` | 0 (at most 8) | Decode-only steps after each mixed step; not recommended |

These flags require `--prefill-budget tbt`. The experimental
`--prefill-budget decode-aware` instead fits the whole step to
`--prefill-itl-target-ms N` (default 200, at most 60,000), or lets prefill
cost `--prefill-decode-steps N` (1 to 64) decode steps, in 512-token
multiples.

## How it works

- **Cost model:** step time is fitted online as decode time per item bucket
  (verify steps included), plus a fixed prefill cost, plus costs per prefill
  token and per attention pair over the cached context, by nonnegative least
  squares over the latest 128 prefill steps.
- **Warmup:** until the fit has at least 8 prefill observations, a step
  beside decoding requests prefills at most the progress floor. The target is
  an estimate, not a hard bound.
- **Prefill-only steps:** one active, unscored request may use the full step
  token budget (8,192 by default), under every policy, when the bundle
  declares an explicit prefill capacity. Legacy bundles retain the configured
  chunk limit. Prefix capture points
  still end chunks, and non-final chunks respect the bundle granule. Competing
  requests keep the configured chunk limit. Scored requests keep their
  traffic-independent event boundaries. An arrival during a running step
  waits for it; decode neighbors already present keep the mixed-step budget.
- **Fairness:** capped steps alternate between the oldest arrival and the
  longest-waiting prefill, so new prompts cannot starve older ones.
- **Scope:** preemption recompute, prefix restores and offload use the same
  path. A chunk that stops short of the prompt end ends on a multiple of the
  bundle's prefill granule (64 for Qwen3.8, 1 for MiMo).
- **Code:** `crates/runtime-scheduler/src/mixed_budget.rs`.

## Metrics

- `chop_prefill_budget_selected_tokens`: the cap of each step beside decoding
  requests.
- `chop_mixed_step_predicted_seconds` and
  `chop_mixed_step_actual_over_predicted`: prediction quality.
- `chop_mixed_budget_term_seconds{term}`, `chop_mixed_budget_samples` and
  `chop_mixed_budget_trained`: the fit.
- With `CHOPINFER_STEP_TRACE=1`, each step's `prefill_cap` and
  `predicted_us` ([step traces](operations-observability.md)).

`tools/qual/agentic_mixed.py` drives the agentic load used for A/B runs.

## Lone-prompt measurements (2026-10-03)

A before/after scheduler comparison used a configured 2,048-token chunk and
an 8,192-token step budget, with five cold C1 requests per prompt length and
one generated token each. At 8K, MiMo TP2 prefill throughput rose from 7,144
to 8,626 tokens/s and median first-token response time fell from 1,146 to
950 ms. MiMo TP4 rose from 10,908 to 12,675 tokens/s, with response time
falling from 751 to 646 ms. Qwen TP2 rose from 9,039 to 10,069 tokens/s,
with response time falling from 906 to 813 ms; Qwen TP4 rose from 9,748 to
10,568 tokens/s, with response time falling from 841 to 773 ms.

At 32K, MiMo TP2 rose from 6,447 to 8,438 tokens/s and median response time
fell from 5,111 to 3,871 ms. MiMo TP4 rose from 9,945 to 12,078 tokens/s,
with response time falling from 3,292 to 2,712 ms. Qwen TP2 rose from 8,968
to 10,482 tokens/s, with response time falling from 3,654 to 3,127 ms;
Qwen TP4 rose from 9,651 to 10,900 tokens/s, with response time falling
from 3,397 to 3,006 ms. The long-prompt gain was 8.4-20.7% at 8K and
13.0-30.9% at 32K. The five-trial latency ranges did not overlap between
arms at either length. At 2K, throughput changed by about 1% or less.
Separate main-build chunk-size controls showed the same direction of gain.
These are samples from one measurement session, not confidence intervals
from repeated independent windows.

Mixed agentic load used C1, C4 and C8, each for 30 seconds, with the same
profile policy in both arms. It showed no uniform tail-latency improvement.
At C8, p95 time per output token changed from 29.05 to 28.27 ms on MiMo TP2,
16.30 to 17.15 ms on MiMo TP4, 14.82 to 14.71 ms on Qwen TP2, and 10.44 to
11.03 ms on Qwen TP4. Qwen TP4 at C4 had a larger p95 first-token increase,
130 to 466 ms, and within-request gaps over 100 ms rose from 4.60 to 9.02
per 1,000 gaps. The active-neighbor rule preserves mixed-step composition;
it cannot bound the wait for a neighbor arriving during a larger lone
prefill step. Repeated fixed-arrival measurements are needed to attribute
that tail change. MiMo TP2 hit the same agent output ceiling in both arms;
its timing evidence does not establish a passing agent-completion gate.

All four chunk-size and four scheduler before/after determinism comparisons
were exact, including cold repeats and prefix hits. Candidate teacher and
prefix-exact checks passed for both models at TP2 and TP4. The concurrent
split check failed on main and candidate at TP2 for both models, while all
three serial controls per model passed exactly. The split check compares
later decode rows whose arithmetic can depend on concurrent batch shape
([numerics](numerics.md#batch-shape)); its concurrent failure does not prove
a prefill regression. The displayed zero first-row logprob difference does
not describe the later rows. TP4 concurrent splits also failed; a separate
TP4 serial split comparison was not run. Serialize this check in a separate
qualification-tool change, preserving its exact continuation coverage.
