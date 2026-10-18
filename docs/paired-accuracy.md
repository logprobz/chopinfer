# Paired release accuracy

`paired_teacher` compares this run's teacher capture with an accepted capture
of the same configuration and arm, both scored against the same independently
pinned reference, by declared margins. The accepted capture is the one the
[correctness runner](../tools/qual/README.md#correctness-runner) stores: the
check runs once against the last-good store and once against the seed store,
and both rows gate. It supplements the absolute teacher gate, including
reference-spread limits, and the existing C1 exactness checks. It does not
replace either.

The check uses `teacher.score`, `limits.paired` and the existing deterministic
95% text-clustered bootstrap (2,000 resamples). It reports both arms' mean and
maximum absolute logprob error and decisive, near-tie and total top-one counts.
It reports paired error changes, discordant counts and affected text names,
categories and absolute token positions. Pooled results and every category
must pass separately. Near ties use the pinned reference's stored top-k gaps,
including the existing 0.3-nat boundary. Classification never uses the other
serving arm as ground truth.

## Adopted margins and repeat calibration

On 2026-10-03 the maintainers approved capture of the accepted alpha.2
release twice independently per shipped profile, followed by margin
adoption. On 2026-10-04 they clarified that deterministic repeat spread does
not estimate legitimate neutral numerics changes and approved the reviewed
proposal below, supported by accepted-only TP2-versus-TP4 evidence. This approves the calibration
procedure, not missing evidence or an uncalibrated final policy. On 2026-10-04,
the seven complete profiles adopted authenticated alpha.2 capture digests and
the reviewed neutral-change margins. Missing or removed pins fail closed.

The [declaration fragment](../tools/qual/templates/paired-teacher.json) keeps
all acceptance numbers in one `paired-teacher-margins/v1` object: the margins,
the reference pin, the approval and its rationale. These are **approved
margins**:

| Rule | Proposed value | Meaning |
|---|---|---|
| Pooled mean error | 0.005 nats | Upper clustered CI95 of candidate-minus-accepted absolute error |
| Category mean error | 0.01 nats | Point increase, without category CI gating |
| Pooled top-one | 0.01 | Upper clustered CI95 of the per-position net added disagreement rate |
| Category top-one harm | alpha 0.05 | Whole-text one-sided sign test, with Holm over all categories |
| Minimum CI clusters | 8 texts | No bootstrap CI is computed or gated below this count |
| Sparse pooled point rules | false | Pooled evidence below the minimum text count is insufficient unless explicitly allowed |
| Tail growth threshold | 0.5 nats | Strictly greater absolute-error increase/decrease at a position |
| Pooled tail allowance | 2 net positions | Increases beyond the threshold minus decreases beyond it |

The available accepted-only neutral replay reached a 0.002744-nat upper mean
CI. The proposed 0.005-nat allowance covers that observation and is 5% of the
0.1-nat large-set absolute floor, or 20 times tighter than that floor. It is
four times tighter than the existing 0.02-nat informational mean margin.
Accepted-only repeat calibration is recorded separately from the adopted margins.
The 1% pooled
top-one proposal permits small neutral changes rather than rejecting every
added flip. The category mean point margin protects a category from being
hidden by pooled improvements. Category top-one harm compares ordinal
severity at each position: agreement < near tie < decisive flip. Let b be
texts whose sum of candidate-minus-accepted ordinal severity increases and c
those whose sum decreases. Tied texts are excluded. The one-sided sign-test
p-value is P(X >= b | b+c, 1/2), assuming independent texts and equally likely
positive and negative signs under the null. Positions within a text are never
independent trials. Sort category p-values and reject in Holm step-down order
only while p < alpha / remaining categories. A decisive-to-near-tie switch is
an improvement, not an added near-tie harm.
Raw decisive, near-tie and total counts are still reported for both arms.

The eight-text minimum is a proposed coverage floor, not proof that eight
texts establish general model quality. Sparse pooled evidence is `insufficient`
by default, including bit-identical arms. Only a policy approved before capture
with `allow_sparse_pooled: true` permits pooled point mean and top-one rules.
This exception provides weaker evidence than CI gating and needs a documented
coverage rationale. Categories always use the mean point rule and whole-text
sign test. Sparse categories and explicitly allowed sparse pooled scopes report
`point_sparse`, cluster count and null CI fields. Refused pooled scopes report
`insufficient`. Categories with sufficient texts report `category_point`.
Bit-identical scored arms skip bootstrapping and report zero change.

The tail check is pooled and reports both added and removed tail positions,
including category counts for inspection. Two net positions are a small
allowance for occasional rounding outliers, not a calibrated noise
budget. Improvements elsewhere can offset tail counts only as the declared
net rule specifies. Accepted-only TP4/TP2 replay has net tail changes -9 for
MiMo and +1 for Qwen; these are sanity checks, not repeatability calibration.

The accepted-only TP2-versus-TP4 AR replay on 2026-10-03 passes pooled and all
categories for both models under the proposed 0.005-nat mean margin. Qwen's
pooled mean CI95 upper bound is 0.002744 nats; MiMo's is 0.000394 nats.
The proposal uses this accepted-only evidence, not candidate results, and
was adopted after the independent repeats confirmed deterministic serving.
Different topologies remain separate accepted arms in production qualification.

On 2026-10-04, each profile completed two independent accepted-arm captures
and a fresh candidate-role capture of alpha.2 followed by a live paired replay.
All 21 captures have complete paired identities and checkpoint content
verification. All seven paired replays passed both absolute gates and the
paired comparison. MiMo scored 4,108 positions per capture; Qwen scored 4,176.
Each repeat pair had identical served scored rows, despite distinct startup
nonces. Both directions had zero pooled mean change, zero mean CI95 bounds,
zero top-one net change and CI95 bounds, zero category mean increase, and zero
net tail increase. Category Holm harm tests passed.

| Family | Profile | Mean CI95 upper spread (nats) | Category point spread (nats) | Top-one CI95 upper spread | Net tail spread |
| --- | --- | --- | --- | --- | --- |
| MiMo | TP2 AR | 0 | 0 | 0 | 0 |
| MiMo | TP4 AR | 0 | 0 | 0 | 0 |
| MiMo | TP4 dflash | 0 | 0 | 0 | 0 |
| Qwen | TP2 AR | 0 | 0 | 0 | 0 |
| Qwen | TP2 MTP | 0 | 0 | 0 | 0 |
| Qwen | TP4 AR | 0 | 0 | 0 | 0 |
| Qwen | TP4 MTP | 0 | 0 | 0 | 0 |

Zero repeat spread reflects deterministic captures and carries no information
about legitimate neutral numerics changes. It must not narrow the reviewed
proposal to zero. The adopted margins therefore come from the accepted-only
TP2-versus-TP4 neutral evidence: 0.005-nat pooled upper CI95, 0.01-nat category
point mean, 1% pooled top-one upper CI95 and two net tail positions above
0.5 nats. Holm alpha remains 0.05 and sparse pooled evidence remains refused.
The neutral replay reads these margins directly from the activated TP4 AR
policy and uses the pinned complete TP4 and TP2 accepted arms. It passes pooled,
category, harm and tail checks for both models. This topology comparison is a
numerical diagnostic, never a production qualification: arm identities remain
bound to their declared profile, and no historical capture is rebound. The
2026-10-04 replay confirms pooled mean CI95 upper bounds of 0.000394 nats
(MiMo) and 0.002744 nats (Qwen), top-one upper bounds of 0.109% and 0.754%,
and net tail changes of -9 and +1. Both models pass every category and Holm
harm check under the activated margins.

The declarations retain the 2026-10-03 procedure approval and the
2026-10-04 margin clarification as each policy's `approval`. Reference digests
stay fixed. Evaluation receipts and full calibration directions remain outside
the repository. The accepted alpha.2 captures are the MiMo and Qwen seed arms;
historical replays are evidence for calibration, not receipts qualifying a
later run. MiMo TP4 MTP declares no margins: alpha.2 has no accepted capture
of that mode, so no stored arm exists for it.

### GLM 5.3 Flash

On 2026-10-11 GLM's accepted arm and margins were approved. GLM then had no
accepted release; its adopted qualification was the final release gate of
2026-10-11.
That gate's teacher receipts are release-gate receipts without paired
bindings, artifact digests or content verification, so they cannot become
accepted arms. The accepted arm of each policy is a fresh GPU capture of that
gate's install, served from its own package cache, with one
`chopinfer verify` per capture job.

Each GLM configuration qualifies one serving configuration, and its teacher
check admits only that configuration's profile (context-parallel size, KV
element, drafting). An accepted capture must pass the absolute teacher gate
of its own configuration, and each configuration compares with its own
stored arm. `models/glm5next/qual.json` declares the `tp4-mtp` margins (the
FP8 checkpoint at DCP 1); the `dcp2` and `dcp4` configurations apply them
to their own arm. BF16 KV and the five-draft profile declare
no paired check; the `kv-bf16` configuration's `teacher_arms` reports the
BF16 KV teacher difference ([BF16 KV](model-glm5next.md#bf16-kv)).

`tp4-ar` has no margins (decided 2026-10-12; its approval reads
`operator:2026-10-12:glm-tp4-ar-no-accepted-arm`). Every GLM live check, the
teacher's included, requires a drafting server, so no GLM gate run serves
`tp4-ar` and it has neither an accepted nor a candidate teacher arm. Its accuracy link to `tp4-mtp` is the AR-versus-MTP
pairs of the same gate run: determinism requires the MTP record to equal
the AR record bit for bit, and spec_verify and verify_exact compare AR and
MTP rows. In the final release gate (2026-10-11) both measured zero difference (mean
|dlogprob| 0.0 and no top-one flips; verify_exact over 3,715 forced decode
positions), within gates that allow margins. An accepted `tp4-ar` arm needs
a declaration whose teacher check admits an AR server, with independent
accepted captures and calibration.

The margins come from GLM's own same-build neutral evidence: a CPU diagnostic
over the final release gate's teacher receipts (2026-10-11), accepted DCP 1 against each other
configuration, 35 texts and 3,781 positions. Only the multilingual category
(9 texts) reaches the eight-text CI minimum.

| Same-build change | Pooled mean change (upper CI95) | Top-one net (upper CI95) | Net tail | Worst category point |
| --- | --- | --- | --- | --- |
| DCP 2 | +0.0011 (0.0040) | -0.13% (0.06%) | +3 | chat +0.0088 |
| DCP 4 | +0.0029 (0.0098) | -0.08% (0.08%) | +7 | chat +0.0208 |
| BF16 KV | -0.0008 (0.0037) | -0.03% (0.53%) | -3 | chat +0.0107 |
| Five drafts | 0 (bit-identical) | 0 | 0 | 0 |

The pooled mean (0.005 nats) and top-one (0.01) margins are the reviewed ones.
The category point margin is 0.015 nats, 1.4 times the largest increase
observed against DCP 1's own arm (BF16 KV chat), and the tail allowance is four
net positions, DCP 2's +3 plus one. Under the reviewed 0.01 and 2, a
reduction-order change such as DCP 1 to DCP 2 fails on its tail alone. DCP 4
exceeds the pooled margin against DCP 1 (upper CI95 0.0098), so DCP 2 and 4
are held to their own accepted arms, each configuration's stored arm. Holm alpha
0.05, the eight-text minimum, sparse pooled refusal and the 0.5-nat tail
threshold are unchanged.

On 2026-10-12 each policy had two independent accepted captures of the
final release gate's install, from its own package cache, each start with its own
startup nonce, after one `chopinfer verify` per capture job and checkpoint
(no file failed, missing or unrecorded). Every capture scored 3,781
positions and passed its declaration's absolute teacher gate: mean
|dlogprob| 0.06285 (FP8), 0.06395 (DCP 2) and 0.06579 (DCP 4). Each repeat
pair served identical scored rows: zero pooled and category change, zero
top-one and tail change in both directions. Each configuration's first
accepted capture is its seed arm. In the accepted-versus-accepted dry run, a
fresh candidate-role capture of the same install, with its server kept alive,
passed the live paired comparison for the FP8, DCP 2 and DCP 4
configurations.

GLM's layer check (`text_layers`, `taps` configuration) takes its floor from
HF's BF16 run with the served FP8 latent. On TP-split main one row's routing
is a near tie at layer 16 (expert 260 against 92, score gap 3.0e-5), and that
row fails 5 of 329 taps on the aggregate rule. The maintainers accepted that
row on 2026-10-14 without changing a limit; a later change makes `text_layers`
judge rows behind a near-tie routing flip separately.

### Adopted margins by declaration

Each declared set of margins has one row: the declaration, its profile key,
its margins and its approval. A configuration applies its model's margins to
its own stored arm.

| Declaration | Profile key | Margins (mean/category/top-one/tail) | Approval |
| --- | --- | --- | --- |
| `models/mimo26/qual.json` | `tp2-ar` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/mimo26/qual.json` | `tp4-ar` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/mimo26/qual.json` | `tp4-dflash` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/qwen38/qual.json` | `tp2-ar` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/qwen38/qual.json` | `tp2-mtp` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/qwen38/qual.json` | `tp4-ar` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/qwen38/qual.json` | `tp4-mtp` | 0.005 / 0.01 / 0.01 / 2 | `operator:2026-10-03:accepted-repeat-calibrate-then-adopt;operator:2026-10-04:reviewed-neutral-margins` |
| `models/glm5next/qual.json` | `tp4-mtp` | 0.005 / 0.015 / 0.01 / 4 | `operator:2026-10-11:glm-accepted-8c75e90bd;glm-neutral-margins` |

A margin change is a reviewed declaration change with its own approval and
rationale. It does not move a stored arm or change the reference.

## Activation and calibration

The check selects its margins by the live rank count and effective
speculation mode (`tp{N}-{mode}`); a profile running context parallelism
C > 1 selects `tp{N}-{mode}-cp{C}` when the declaration states it, and its
DCP 1 counterpart's margins otherwise. An unknown profile, or one without
margins, is refused. A single-policy external declaration can still use
`POLICY`.

The runner runs `paired_teacher` twice per arm, as `paired_teacher/last-good`
and `paired_teacher/seed`, each with that store's teacher receipt of the same
configuration and arm as `--accepted` (`{stored_teacher}`) and this run's
teacher receipt, which holds every scored text's served rows, as
`--candidate` (`{teacher}`). A store without a capture of the arm reports "no baseline
yet" as information. The comparison reads the live server's profile and
checkpoint binding, recomputes absolute errors from both arms' served rows,
reports a candidate absolute numerical failure as `fail`, and makes a
failing accepted arm `insufficient`. The paired check's constants cannot
loosen the shipped teacher policy. Last good moves only on a passing
full run of every declared configuration; the seed only through
`baseline reseed` with a recorded reason.

```sh
CHOPINFER_QUAL_CONFIG=models/MODEL/qual.json \
  python3 -B tools/qual/paired_teacher.py \
  --accepted "$STORED_CAPTURE" --candidate "$CANDIDATE_CAPTURE" \
  --url "$CANDIDATE_URL"
```

Complete capture files may be up to 64 MiB and use the shared no-follow JSON
reader; duplicate keys and nonfinite numbers are refused. Receipts carry
token-ID digests, score starts, categories and complete served rows.
Rescoring recomputes both arms and refuses missing, duplicate, reordered or
differently scored texts. To score another pinned set, add `--reference-name
NAME` and a separate `--check-id ID` to the check arguments, with the same
independently adopted reference for both arms, never a newly captured
accepted-server output as the reference.

`tools/qual/paired_calibrate.py --first A --second B --out FILE` validates two
accepted repeat captures, their absolute gates, independent startup nonces
and pinned reference. It scores both repeat directions and records the
maximum nonnegative upper CI95 for pooled mean and top-one, the largest
category point increase and net tail count in `observed_spread`, beside the
reviewed proposal in `margins`; deterministic zero spread does not adopt zero
margins. Refuse repeats that fail the reviewed proposal, including category
harm, rather than loosening it. Repeats verify observed repeatability; the
accepted-only neutral topology comparison supports margins for legitimate
numerics changes. Neither is a population noise bound. If a paired
comparison fails, preserve the evidence and resolve the cause; do not widen
the margin or move a baseline to make it pass.

## Existing evidence and held-out coverage

The alpha.2 final matrix has teacher receipts for seven configurations:
MiMo TP4 AR, TP2 AR and TP4 dflash; Qwen TP4 AR, TP2 AR, TP4 MTP and TP2 MTP.
MiMo TP4 MTP has no accepted teacher arm in that matrix and declares no
margins. A TP4 or TP2 arm cannot stand in for another topology.
Its teacher receipts include served rows, but lack the per-text prompt-token
digests, score starts and categories a complete paired arm needs. The accepted
arms are the separate 2026-10-04 alpha.2 captures above. Do not add provenance retrospectively to
convert them into accepted evidence. Recapture missing arms on GPUs using the
exact accepted release.

Proposed additional held-out coverage should include at least eight independent
texts per category: coding (multi-file edits, tests, structured output, Unicode
and boundary cases), tool use (schema-constrained calls, multiple calls,
tool-result continuation, refusal and malformed-result recovery), and long
context (dispersed evidence, code symbol retrieval, conflicting distractors,
and multi-turn tool histories at several supported lengths). Freeze text
content, tokenizer IDs and scored spans before any candidate run. Include
middle and late positions, not just the final token. Store these workloads
outside the repository, derive references through the GPU reference runner,
then independently adopt their locks and workload digests. Candidate and
accepted captures use those identical pinned workloads. Expanding evaluation
coverage must not replace or loosen the current frozen teacher set.
