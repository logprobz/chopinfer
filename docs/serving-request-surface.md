# Serving request surface

`chopinfer serve` exposes an OpenAI-compatible HTTP API modeled on vLLM's.
This page is the reference for its endpoints, request parameters, defaults,
refusals and queueing. Server flags are listed by `chopinfer serve --help` and
described in [operator commands](cli.md) and
[deployment](deploy-common.md#common-http-options); metrics are in
[metrics and traces](operations-observability.md).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /v1/completions` | Text completions; prompts may be strings or token-ID arrays |
| `POST /v1/chat/completions` | Chat, with tools, reasoning, structured outputs and media parts |
| `POST /v1/responses`, `GET /v1/responses/{id}`, `POST /v1/responses/{id}/cancel` | [Responses API](#responses-api) |
| `GET /v1/models` | Served model names, effective `max_model_len` and served `modalities` |
| `POST /tokenize`, `POST /detokenize` | The installed tokenizer, without generation |
| `GET /health`, `GET`/`POST /ping` | Readiness and the [serving identity](deploy-common.md#what-health-reports) |
| `GET /version` | Chopinfer version |
| `GET /metrics` | Prometheus metrics |
| `GET /debug/step-profile` | Sampled step timings, when enabled |

With an API key configured, only paths under `/v1` (and vLLM's `/v2`,
`/inference`, `/cohere` prefixes) need it. Omitting `model` selects the first
served name.

## Request parameters and defaults

**Output length.** Completions without `max_tokens` generate 16 tokens. Chat
and Responses requests without an output limit may use the remaining
context; in chat, `max_completion_tokens` takes precedence over `max_tokens`.
Explicit `max_tokens: 0` is refused.

**Sampling.** Defaults come from the package's frontend descriptor
(`generation_config`); precedence is neutral defaults, model defaults, then
non-null request values, for temperature, top-p, top-k, min-p and repetition
penalty on both endpoints. MiMo uses temperature 1.0, top-p 0.95 and no top-k;
Qwen uses temperature 1.0, top-p 0.95 and top-k 20. Omission and null inherit
the default; `top_k` 0 or -1 disables top-k. Positive temperatures below 0.01
become 0.01, and exactly 0 is greedy. `repetition_penalty` must be from 0.01
through 10. A `seed` (other than -1) makes sampling repeatable within one
server; omission or -1 draws a fresh seed per choice. Random streams differ
from vLLM's.

**Stopping.** `stop` takes at most four strings. `stop_token_ids`,
`ignore_eos`, `skip_special_tokens`, `include_stop_str_in_output` and
`min_tokens` behave as in vLLM: `min_tokens` suppresses stop tokens and stop
strings until the minimum length, and `ignore_eos` disables only model EOS
termination. Explicit stop IDs still stop generation.

**Choices.** `n` may be 1 through 128. Completion prompts may be a list of
strings or token-ID arrays, with at most 128 choices in total; choice indexes
are `prompt_index * n + choice_index`, and each choice uses the request seed
plus its choice index. Usage counts each prompt once. `best_of` is accepted
and ignored. See [batched prompts](#batched-prompts).

**Logprobs.** Completions accept legacy `logprobs` and `echo=true`; chat
accepts `logprobs`/`top_logprobs`. The `prompt_logprobs` extension takes 0
through 20 on both endpoints and returns `prompt_token_ids` and one entry per
prompt token (the first is null). Positive `prompt_logprobs` with
`stream=true` is refused. Prompt scoring needs a bundle that declares
`chop.prompt-rows.v2`. With a grammar that admits fewer tokens than
requested, masked entries serialize with logprob `-9999`.
`return_token_ids` adds generated IDs (and prompt IDs) to responses and
stream chunks.

**Truncation.** `truncate_prompt_tokens` keeps at most N prompt tokens after
tokenization; `-1` uses the context minus the requested output. The default
keeps the last tokens; `truncation_side="right"` keeps the first. Zero, a
value above the input budget, or truncation through a media span returns 400.
Left truncation can drop the chat template's system header, as in vLLM.

**Structured outputs.** `response_format` (`json_object`, `json_schema`) and
`structured_outputs` (`json`, `json_object`, `regex`, `choice`) compile to a
grammar before admission, and every sampled row is masked with it.
Unsupported constraints (`grammar`, `structural_tag`, schema keywords
outside the supported subset) return 400 naming the field. JSON output allows
at most 257 consecutive whitespace characters between tokens; set
`structured_outputs.disable_any_whitespace` to `false` for vLLM's unbounded
whitespace, or `true` for compact separators. In chat, the constraint starts
after an open reasoning section.

The JSON Schema subset refuses unsupported assertions, and assertions beside
`$ref` or combinators, instead of ignoring them. Provably vacuous assertions
(`uniqueItems: false`, zero minimum counts, an empty `required`) are accepted,
and type-specific assertions are ignored when no permitted type is in their
domain (`minimum` on a string). `oneOf` and `allOf` take a single branch; use
`anyOf` for unions. String lengths are at most 4096 and array counts at most
1024; length bounds cannot accompany `pattern` or `format`. Numeric bounds
must be below 2^53 in magnitude, and `number` bounds integral. `multipleOf`
needs an integer schema and either no bounds with a divisor of at most 32, or
bounds spanning at most 4096 multiples.

Grammar resource limits are refusal limits. Exhausting one while compiling
returns 400 with a constraint resource error; exhausting one while matching
or generating fails that request (no token is emitted for the failed step,
earlier streamed tokens stand) and names the budget and its limit, for example
`grammar scanner work budget exceeded (limit: 4096 units per closure)`. A
constraint is never dropped or relaxed, and other requests sharing the
compiled grammar are unaffected. A valid schema can be refused if proving its
continuations exceeds a limit:

| Resource | Limit |
| --- | --- |
| Compile work | 536,870,912 units per schema construction or automaton compilation |
| Compile allocation accounting | 2 GiB (not a process RSS ceiling) |
| Choices or schema-source array elements | 16,384 |
| Determinization growth | 32 DFA states per reachable NFA state in a rule, counted as if every use of a shared fragment were built separately |
| Expression expansion | 64 MiB of expression-copy accounting |
| Reference analysis | 1,000,000 expression and reference visits |
| Literal compilations | 1,024 per schema construction |
| Schema recursion; literal narrowing | 128 active calls; 32 nested sibling compilations |
| Regex source; groups; counted repetition | 65,536 bytes; 64 nested; 4,096 |
| Automaton size | 2,097,152 NFA states; 16,777,216 DFA table entries |
| Matcher closure; depth | 4,096 work units per byte closure; 2,048 live frames |
| Rollback history | 64 accepted tokens |

`CHOPINFER_GRAMMAR_BUDGET_REPORT` names a file (under `$CHOPINFER_DATA`) that
records each successful compilation's costs, limits and smallest margin,
without schema text; it never changes limits or refusals.

**Tools and reasoning.** `tool_choice` `required`, a named function, or `auto`
with a strict tool constrains output to the model's call format. Tool names
can stream before a call closes; argument deltas wait for the close, since a
later parameter can replace an earlier one. A generation that ends inside a
call (length, stop token or client stop) finishes it with the same complete
parameters in streamed and whole responses, with `tool_calls`. `parallel_tool_calls=false` returns only the
first call. A parameter whose declared schema rules out a string is read as
JSON, and a closing marker inside one of its string literals is content, in
both declared tool-call formats: for example `</parameter>` or `</function>`
inside a string of an object parameter. vLLM's parser for the
function-parameter format ends the value at the first marker; this server
diverges so that what the tool-choice grammar writes parses back whole. String
parameters, parameters of a tool with `strict: false`, and parameters of an
undeclared function are raw text and still end at the first marker. `include_reasoning=false` hides parsed reasoning but keeps its
usage. Chat accepts `add_generation_prompt`, `continue_final_message` and
`chat_template_kwargs`; `add_special_tokens` is accepted but does not change
prompt IDs.

**Template keywords and roles.** The server reads the `chat_template_kwargs`
keys it knows (`enable_thinking`, `thinking`, `reasoning_effort`,
`preserve_thinking`, `clear_thinking`, `add_vision_id`) and refuses a value of
the wrong type for the first four. It then passes the whole object to the
model's frontend, every key in the order sent with its value unchanged, so a
model's own keyword needs no server change. A known key sent as null is left
out, as it reads as absent. A key named twice is passed once, where it first
appeared, with its last value, as `serde_json` and Python's `json` read it; a
known key named twice is refused. An array is read by position into the known
keys and passes as the known keys it sets. Message roles are `system`,
`developer`, `user`, `assistant` and `tool`, plus any extra roles the model's
frontend declares (none in the maintained models). A message with an extra
role carries string content only. Any other role is refused with "message role
must be system, developer, user, assistant, or tool", which also names the
declared roles when there are any.

**Streaming.** `stream_options.include_usage` adds final usage, and
`continuous_usage_stats` adds cumulative usage to every chunk. Usage reports
`prompt_tokens_details.cached_tokens` from the prefix cache.

**Errors.** Error envelopes use vLLM's type names; `error.code` is the backend
failure code or the HTTP status. Shutdown and worker-unavailable errors return
503 with `Retry-After: 1` before streaming starts.

## Limits and refusals

| Limit | Default | Flag |
|---|---|---|
| Request body: chat completions and `/tokenize` | 64 MiB plus 4 MiB per accepted image and each served video or audio item's declared size in base64 (320 MiB with 64 images); at most 1 GiB | `--max-request-body-bytes` |
| Request body: other endpoints | 64 MiB | `--max-request-body-bytes` |
| Image parts per request, across all messages | 64 | `--max-images-per-request` (1 to 65535) |
| Any string in the body | 32 MiB | `--max-request-string-bytes` |
| Text to tokenize | 32 MiB | `--max-tokenizer-input-bytes` |
| Token-ID prompt | 1,048,576 tokens | none |
| Choices per request | 128 | none |
| Stop strings | 4 | none |

`--max-request-body-bytes N` (at most 1 GiB) sets every endpoint's body limit.
Startup logs the body limits as `startup request_body_bytes` and the image
limit as `startup images_per_request`.

Unknown request fields are ignored on chat and completions, as in vLLM.
Recognized vLLM controls that this server cannot honor return 400:
`allowed_token_ids`, `bad_words`, `chat_template`, `documents`,
`length_penalty`, `logprob_token_ids`, `media_io_kwargs`,
`mm_processor_kwargs`, `priority`, `prompt_embeds`, `repetition_detection`,
`thinking_token_budget`, `use_beam_search`, `watermarking`, and the transfer,
session and return-option extensions (`kv_transfer_params`,
`ec_transfer_params`, `request_id`, `session_id`, `stream_interval`,
`vllm_xargs`, `return_*`, `routed_experts_prompt_start`). Chat `echo` is
unsupported.

## Queueing and admission

A server runs at most `--max-sequences` live sequences (8 without a profile;
the serving profiles and the image use 32). Further requests wait:

- The waiting queue is unbounded by default, so load makes requests wait
  instead of returning 429. `--queue-depth N` bounds it to N requests beyond
  the live sequences (0 admits only the live ones); a full bounded queue
  returns 429.
- Waiting has no time limit by default. `--queue-wait-ms N` returns 429 for a
  request not admitted to the backend within N ms of arrival.
- A client disconnect cancels a queued request. A streaming client that stops
  reading for 60 s is cancelled.
- Backend admission is first in, first out. `/health` reports `queue_depth`
  and `waiting`, and `chop_requests_waiting_by_stage` counts waiting requests
  by stage (transport, intake, preparation, staging, Responses storage,
  backend admission).

Payload memory stays bounded while requests wait: connections share a
512 MiB transport budget (1 MiB each), bodies wait for an intake reservation,
and media preparation waits for a 4 GiB share of an 8 GiB media budget.
Requests over their own size or media limit return 400; requests waiting for
memory are not refused.

**KV admission.** `--admission optimistic` (the default) admits a request when
its prompt fits, grows its KV as it decodes, and preempts the latest-arrived
request when the pool runs short, as vLLM does. `--admission-watermark F`
(default 0.01) keeps that fraction of the pool free when admitting beside
live requests, and `--max-preemptions N` (default 3) bounds how often one
request is preempted before its worst case is reserved. `--admission reserved`
admits a request only when the pool holds its prompt plus its whole
`max_tokens`. Requests with media or prompt logprobs always reserve their
worst case. With prefix caching on, a preempted request usually resumes from
a saved anchor; otherwise it recomputes its prompt and emitted tokens, which
can change later tokens at near-ties ([numerics](numerics.md)).

**Scheduling.** The batch budget (`--max-batch-tokens`, default 8192) and
prefill chunk (`--prefill-chunk-tokens`, default 2048) bound each step. A
package's `batch_limits` bound them further: startup refuses
`--max-sequences` above `maximum_sequences` or `--max-batch-tokens` above
`maximum_prefill_tokens`, and `--prefill-chunk-tokens` must be a multiple of
`prefill_granule` (64 for Qwen3.8). Decode items run first and a prefill chunk
fills the rest of the step. `--prefill-budget fixed|tbt|decode-aware` caps
prefill beside decoding requests; see [mixed scheduling](mixed-scheduling.md).

### Batched prompts

All choices of a request (prompt lists, or `n > 1`) are submitted together,
so they can share a prefill step when the sequence, token and KV budgets
allow. Choices still queue and may be preempted or span several steps. With
prefix caching on, choices sharing at least one full KV page wait for an
earlier sibling to commit that prefix and then reuse it; if that sibling is
cancelled, they prefill cold. A large request places all its choices ahead of
later requests, which can raise a later small request's time to first token.

The response starts as soon as any choice is admitted. A failed first
admission returns an HTTP error; a later failure returns an error (or a
terminal stream event) and cancels the siblings. A group's `--queue-wait-ms`
limit applies until its first choice is admitted.

**Prompt logprobs with prefix caching.** Scoring requests skip cache lookup
but still capture anchors, so an identical repeat can split its prefill
differently and return slightly different prompt logprobs. For bit-identical
repeats, serve with `--prefix-cache off`. `cache_salt` isolates cache keys but
does not disable capture.

## Media inputs

`/health` and `/v1/models` report the served `modalities`, for example
`["text", "image"]` (see `--modalities` in [operator commands](cli.md)). A
chat media part outside that set returns 400 before any fetch or decode.

- Images: PNG (1 to 8-bit, any color type, interlaced or not; not 16-bit)
  and JPEG (8-bit baseline or progressive Huffman), up to 89,478,485 pixels;
  the model's recipe resizes them.
- Audio: WAV only, up to 600 seconds; convert other formats first.
- Video: MP4 through NVDEC, when the package serves video
  (`--modalities all` or a list naming video).
- Image parts are counted across the whole request (see below). Video and
  audio counts, and each item's bytes and tokens, come from the model's
  recipes.

Content arrays accept `image_url`, `audio_url`, `video_url` and
`input_audio` (`format: "wav"`) with data URLs. Local files need
`--allowed-local-media-path DIR`.

### Many images

A request may carry 64 image parts across all of its messages
(`--max-images-per-request N`). One more returns 400 before any image is
fetched or decoded:

```json
{"error":{"message":"At most 64 image(s) may be provided in one request; this request carried 65.","type":"BadRequestError","param":"image","code":400}}
```

Only this count refusal uses `param: "image"` and the `At most N image(s)`
wording, which clients that trim old images look for. An image that fails to
decode, has an unsupported format or exceeds its byte limit returns 400 with
another `param`. For a model that accepts images, `/v1/models` reports
`architecture.input_modalities` (the served modalities) and
`max_images_per_request`.

Under the count, two limits bind next: the context window, since each image
costs its recipe's tokens (Qwen3.8 up to 16,384, GLM 5.3 Flash up to 8,000),
and the 4 GiB of host memory one request may use to prepare images not
prepared before, which holds only a few images at the largest sizes the
recipes produce. The body limit grows with the image limit. The GPU media
arena is a cache, not a per-request limit: prefill encodes images as its
chunks reach them ([media arena](serving-kv-pool.md#media-arena)).

Sending the same images on every turn is cheap. With prefix caching on, the
next turn resumes from the cached conversation and encodes only the images
after it, and an image sent again (same bytes and MIME type) reuses its
decoded pixels while the media budget has room. A client that drops old
images should drop several at once, for example the oldest half as it nears
the limit, rather than one per turn: removing an image changes the prompt
from that image on, so everything after it is prefilled again.

### Remote media URLs

HTTP(S) media is off by default. Allow exact host names with
`--allowed-media-domains media.example,cdn.example` (commas or a repeated
flag; no ports, wildcards or implicit subdomains). Every redirect must match
the list, and private, loopback and special-use addresses are refused unless
`--media-allow-private-addresses on`. URLs with credentials are refused,
proxies are not used, HTTPS verifies system CAs and cannot redirect to HTTP.
The runtime needs `/usr/bin/curl` and CA certificates. `input_audio.data` also
accepts a URL, unlike vLLM.

| Flag | Default | Range |
|---|---|---|
| `--media-connect-timeout-ms` | 3000 | 1 to 300000 |
| `--media-fetch-timeout-ms` (one item, all redirects) | 10000 | 1 to 300000 |
| `--media-request-timeout-ms` (all items of a request) | 30000 | 1 to 300000 |
| `--media-max-redirects` | 3 | 0 to 10 |

There are no retries. At most eight fetches run at once in the process; others
wait within their deadline. Fetch and policy failures return 400.

## Responses API

`POST /v1/responses` shares chat's preparation, decoder, grammar, sampling and
admission. It accepts:

- `input` as a string or a list of text messages, `function_call`,
  `function_call_output` and unencrypted `reasoning` items. Message content
  accepts strings and `input_text`/`output_text` parts.
- `instructions`, function `tools`, `tool_choice` (`auto`, `none`,
  `required` or a named function), `parallel_tool_calls`, `reasoning.effort`,
  `chat_template_kwargs` and `include_reasoning`.
- `text.format` with `text`, `json_object` or `json_schema` (with `name`,
  `schema` and `strict` inside `format`).
- `max_output_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, penalties,
  `seed`, `logit_bias` and stop controls. Logprobs need
  `include: ["message.output_text.logprobs"]`, with up to 20
  `top_logprobs`. `metadata` takes up to 16 string pairs.

With `stream: true`, named SSE events run from `response.created` to
`response.completed` (also sent for an `incomplete` response), with no
`[DONE]`; errors produce an `error` event.

**Storage** is off by default, and `store: true` is then ignored, as in vLLM.
`--responses-store-bytes N` enables a process-local store of at most 128
records within N serialized bytes. With storage on, finished responses can be
retrieved and continued with `previous_response_id`, and `background: true`
runs a request in the background (cancel with `POST
/v1/responses/{id}/cancel`). Unknown, evicted, active or unstored IDs return
404. A record that cannot fit returns 429; completed records are evicted
oldest first. Storage admission is first in, first out.

Refused with 400: background streaming, stored event replay, automatic
truncation, encrypted reasoning, generated reasoning summaries, hosted tools,
prompt references, multimodal input, extra service tiers, unknown request
fields, and continuation of an active background response.
`prompt_cache_key` is accepted and ignored.

## Differences from vLLM

The main differences, besides the refused fields above:

- The waiting queue is unbounded by default and bounded by `--queue-depth`;
  request bodies share a bounded intake budget, and a stalled streaming
  client is cancelled after 60 s.
- `/health` returns JSON with the serving identity rather than an empty 200.
  `/v1/models` has no permission objects. `/tokenize` does not insert special
  tokens or accept request templates.
- Structured outputs support a subset of JSON Schema and regex; JSON
  whitespace is bounded by default.
- With `echo` and `logprobs=k`, prompt rows list only the top k.
- Tool call IDs derive from the request and index. Some optional members
  (`stop_reason`, `refusal`, `service_tier`, `system_fingerprint`) are
  omitted. Usage has `prompt_tokens_details.cached_tokens` but no
  `completion_tokens_details`.
- Responses echo the requested model alias.
- Speculative decoding is adaptive by default.
- Prefix caching resumes at captured anchors rather than at any cached
  64-token block ([KV pool](serving-kv-pool.md)).
- The Responses store is bounded; vLLM's is not.

## Qwen3.8 API parity

Qwen3.8 differs from a frozen vLLM capture only as follows; finish reasons,
logprobs shape, `n` with a seed and the reasoning/content split match.

| Area | vLLM | Chopinfer |
| --- | --- | --- |
| Video | Refused with `At most 0 video(s)` | Refused when video is not served |
| `reasoning_effort: "high"` | 400 with the template's message, param null | 400, `reasoning_effort must be xhigh, medium, or low`, param `reasoning_effort` |
| Prompt too long | 400, param `input_tokens` | 400, param null, `prompt token count ... exceeds the serving limit 262144` |
| Stream chunks | `prompt_token_ids`/`token_ids: null`; first tool delta has no `arguments` | `usage: null` with `include_usage`; first tool delta has `arguments: ""`; concatenated deltas are identical |
| Identifiers | `chatcmpl-<hex>`, `chatcmpl-tool-<hex>` | `chatcmpl_<request>`, `call_<request>_<choice>_<index>` |
| Model name | the served name chosen at launch | `local-<checkpoint identity>` by default, from a `qwen38-package` package and through `serve --model` alike, the pinned revision included ([compatible checkpoints](compatible-checkpoints.md)); `--served-model-name` chooses another, as the qualification runner does with `Qwen3.8-Flash-Next-FP8` |

To check parity offline, `models/qwen38/compiler/tests/frontend_replay.rs`
(part of `ci/cpu.sh`, skipped without `CHOPINFER_QWEN38_CHECKPOINT`) replays
the capture's render and tokenizer requests through `/tokenize` and requires
identical HF token IDs. Compare recorded server frontend observations with
`python3 tools/qual/frontend_goldens.py CAPTURE`
([qualification tools](../tools/qual/README.md)).
