# Ordered prefill source scheduling

The `prefill_attention` example compares the current native paired or
fused route with two candidates: the existing fused head kernel at more
chunk offsets, and a separate fused kernel that avoids staging and MMA
for wholly invalid 64-key units. Neither comparison changes serving
admission or a serving lock.

The empty-unit kernel retains the original rescale exponentials and
denominator/output FMAs. An empty contribution can change negative zero
to positive zero, so dropping the entire update is not sufficient for
bitwise equality. Every nonempty unit keeps the original source body,
including the four ordered 128-channel QK chains, score fold, softmax,
BF16 probabilities, PV chain and fused output boundary. A CPU check
reverses only the new control block and requires exact source equality.

The reference route depends on both the original chunk offset and the
cache allocation. A full native chunk at a positive offset currently
uses paired attention even when its rows are consecutive. Decoded-cache
allocation requires serving capacity one; measuring only the cached
later-offset route misses the native fallback used at larger capacity.
Retained native tails and parts currently admit fused heads only at
offsets zero and 24576. The comparison includes native full chunks at all
four offsets, cached full chunks at positive offsets, all twenty retained
counts, and both 4096-row parts.

Build the example with locked, offline Cargo, then run `empty-build OUT`
to add the candidate to an external qualified snapshot. `empty-snapshot
OUT` verifies its complete module inventory, including the harness probe
and the unchanged reference cubins. Snapshot validation also plans every
partition on CPU, including the empty second interval of a 64-row case.
Empty intervals produce no CUDA launch. `empty-check OUT LISTS OFFSET` and
`empty-time OUT LISTS OFFSET` consume full captured selections. The
allocated pair jobs run checks before timing and reject missing,
duplicate, skipped or mislabelled cases. CPU parser controls read an
earlier recorded timing output before exercising the new record schema.

Correctness compares every F32 partial and BF16 output byte, verifies
inactive poison, refreshes queries and live counts, tests fully empty
and internal-hole units, and compares two-part execution. The existing
fused body also has an independent output and partial check. Timed
functions receive a null partial pointer and must leave that buffer
poisoned. The separate `empty-memcheck` command uses small native and
unaligned cached cases, including invalid batches, slots and pages.
Timing reports served-to-fused and fused-to-empty comparisons separately,
with equal cache eviction and cache construction included.

All buffers belong to one rig and one stream. Reference and candidate
outputs and partials are disjoint. Each cache producer finishes before
its consumers; partitioned launches use disjoint output ranges. The new
kernel adds no global scratch, asynchronous producer or cross-stream
dependency. CUDA correctness and performance remain unproven until the
allocated jobs pass. CPU checks alone do not qualify a serving route.
