# Adaptive speculation

With `--speculation adaptive` (the default whenever `--speculative-tokens` is
positive), a shared controller chooses each request's draft count from its
acceptance estimate and measured step times, and drafts nothing when AR is
cheaper. `--speculation fixed` drafts `--speculative-tokens` on every step.
Every emitted token is sampled from a target row, and target rows do not
depend on those choices: at C1 every MiMo and Qwen verify row equals the AR
row at its position bit for bit ([exact verify rows](#exact-verify-rows)), so
greedy speculation serves AR's tokens and logprobs whatever the policy picks.

## Step costs

`--draft-cost per-token` (MTP) prices a step as the AR estimate of its row
bucket plus a drafting excess measured per concurrency and draft shape. An
unmeasured shape borrows a conservative excess from a measured one, and a
shape with more than twice the measured drafting sequences must be probed
(warmed and measured) before normal drafting uses it.
`--draft-cost per-block` (DFlash) prices whole draft blocks by row bucket.

Fresh drafters pay a catch-up cost; two probes separate it only when their
concurrency, bucket and draft shape match. Only a probe that beats AR by the
5% resume margin resets probe spacing; marginal gains back off. If AR later
slows by more than that margin, the controller probes again after 16 idle
plans. The controller is independent of model and bundle code; its tests run
with `cargo test -p runtime-engine --locked --offline`.

## State and verification contract

Verification prepares K + 1 rows transactionally. Commit keeps only the
accepted prefix and the correction or bonus token; rejected KV, recurrent
state and drafter state are restored before the next step. Streaming, stop
handling and logprobs describe committed target tokens only. Greedy acceptance
compares target token IDs; stochastic acceptance preserves the target
distribution rather than forcing AR text equality. Graph row capacities are
compile-time bounds; unused rows never become samples or committed state.

The MiMo release preview records both speculation and AR with
`tools/qual/spec_verify.py` (16 copy and 16 natural prompts, 64 greedy output
tokens, top-5 logprobs). Over common prefixes, the copy set must keep mean
|dlogprob| <= 0.15 and top-1 disagreement <= 0.08. Both GSM8K arms must score
at least 193/200, lose at most one AR-correct item and drop at most one total
correct answer. C1 greedy differences are informational. See
[the release gates](../tools/qual/README.md).

### Exact verify rows

MiMo verify steps of up to eight rows (`EXACT_VERIFY_ROWS`, every C1 step) run
each row with the one-row AR step's arithmetic: attention uses the decode
kernels and split plan per row, and the all-reduce, experts, GEMMs, router,
head, RoPE and KV writes give every row the one-row bits. Verify steps of more
than eight rows use the verify kernels, whose rows are row-local but follow
the verify bucket's reduction order, and above C1 the bucket depends on the
other requests, so the guarantee covers requests served one at a time
([batch shape](numerics.md#batch-shape)).
`CHOPINFER_MIMO_VERIFY_EXACT=0` restores the verify kernels for A/B runs, and
`CHOPINFER_MIMO_EXACT_ATTENTION=rows` runs one CTA set per row (same bits).
The diagnostic `CHOPINFER_MIMO_VERIFY_ATTENTION=rows` runs the attention of
wider verify buckets on the same per-row entries at the bucket's token rows,
while every other stage keeps the bucket's plans.
Exact rows cost little up to 32K tokens of context and about 3.5% of an
eight-row verify step at 120K (2026-09-30).

Qwen MTP K3 rows at C1 already use AR's reduction order (the decode attention
entry and split count, cluster-decode GEMM geometry up to four rows, and the
same MoE splits at buckets 1 to 4).

The release preview requires bit-identical C1 receipts between two starts of
the default profile (the second prefilling in 1,024-token chunks), between
speculation and AR, and between prefix-cache hits and cold sends, for both
models (the `tools/qual/determinism.py` request set, recorded by `chopinfer
eval capture --kind determinism`).
