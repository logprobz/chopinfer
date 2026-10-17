# Agentic benchmark test inputs

`corpus/` contains the pinned default source inputs used by coding and
decode probes and their CPU tests. Its [notice](corpus/NOTICE.md) describes
provenance, licensing, and workload identity.

`v2-sweep-fixed.json` and `v2-sweep-tbt.json` are synthetic inputs for the
legacy v2 comparison tests in `tools/qual/test_agentic_mixed.py`. They are
hand-authored examples of the fields the comparator reads, not captured
benchmark results. Each contains one concurrency level, two measured
sessions, and four shared request/answer identities per session.

Both fixtures use `synthetic-agentic.v2`, no source commit, and SHA256 values
of synthetic identity strings. The profile digest is the SHA256 of its
sorted, compact JSON encoding. Metric snapshots explicitly identify each
budget mode; only the tbt fixture advances the selected-budget counter.
The fixed/tbt decode-step means are 4/3 ms, their client TTFT values are
8/6 ms, and their output rates are 100/125 tokens per second. These values
exercise comparison arithmetic. They carry no host observations and do
not support performance claims.

Keep these small fixtures as test inputs. Write actual qualification
receipts and measurement outputs outside the repository.
