# Captured target verify attention

The diagnostic observer records the served native Flash TP4 attention
boundary at verify widths four and 32. It observes layers zero, two and
twenty, covering window-only, mixed32 and mixed64 attention. It keeps the
current source, producer and ordered merge kernels. Graph replay is
replaced by the same eager launches in a separate diagnostic source export.
The ordinary bundle source and serving admission are unchanged.

All ranks enqueue each launch before rank zero is read back. Each mapped
Engram layer retains its own readiness wait before its consumer. The
observer adds no stream, buffer alias or device write. Inputs are captured
before their source launch; F32 partials are copied after the body, and
BF16 output, sink and RoPE operands after the merge. The observer keeps
every split and selected-key order.

Only referenced compressed pages and window state slots are copied. The
corpus retains original table metadata and physical identities alongside
compact copies. Rebased tables preserve each selected address and its
validity. The CPU validator reconstructs both mappings and checks every
blob digest and extent. A real-operand GPU replay must match every captured
partial and output byte after two scratch poison resets before the capture
job passes. This is an eager boundary check, not served graph or token
qualification for a new kernel.

The job uses the default `tp4-dspark` profile at capacity 32 and its original
memory fraction. Cold request groups cover 8192 and 32768 prompt tokens.
Two different C1 requests contribute one four-row capture each per context.
One atomic eight-prompt request contributes two 32-row captures per context;
a target graph may include both verify and decode rows. Eight recorded
diverse prompt seeds are expanded to the requested lengths. C8 requests
generate 256 tokens so early requests stay live while other prefills finish.
C1 requests generate 32 tokens. Zero cached-token checks, arming after
readiness, real request identities and captured positions establish scope.

All operand data and replay receipts are written beneath a new private
root-volume temporary directory. The frozen install, job, logs and
manifests remain outside the repository. One server starts once. The
whole-machine wrapper reserves 840 seconds, with a 240-second readiness
cap and a 780-second work deadline before cleanup. Previous HC captures
reached readiness in about 125-129 seconds and completed three servers in
about 395 seconds. The longer cold-prefill requests have separate bounded
timeouts; this comparison is a budget basis, not a runtime guarantee.

`test_capture.py` checks complete and incomplete corpora, page and slot
rebasing, real-request origins, cold-response counts, replay hash coverage,
every phase failure, actual request construction, readiness/exit/timeouts
and process cleanup without CUDA. The staged dry run uses the actual
frozen server parser and the actual frozen replay binary's CPU selectors.
Recorded earlier logs additionally exercise shared result and origin
parsers before staging. GPU work is always submitted by the orchestrator.

The new verify-width pair harness compares the existing selected-key cache
with the served native route. Its numerical contract remains eight ordered
splits and unchanged merge arithmetic. Qualification and timing on the
new capture remain pending until the capture and its replay pass.
