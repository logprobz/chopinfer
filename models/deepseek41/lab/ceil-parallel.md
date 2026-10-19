# Parallel activation ceil quantization

The 2026-10-19 decode profile attributed 22.771 ms per TP4 C1 step to the
seven ceil quantizers and the shared expert's quantized SwiGLU, across 337
launches at a 250 W power cap. These are exclusive times with PDL overlap
removed. They describe the serial entries.

The serial entries run one warp per row. A 5120-wide row has 160 independent
32-value groups. The warp quantizes all 160 to validate the row, combines
its status, then quantizes all 160 again to write it. Only 16 lanes load a
pair in each group. C1 therefore has one active warp on one SM and long
chains of loads, five shuffle/max steps, correctly rounded divisions,
conversions, and representability checks. The ceil scale itself is an F32
reciprocal product followed by exponent and mantissa bit operations; it
uses neither log2 nor ceil. SwiGLU also repeats its expf and product work.

The parallel entries use up to 32 warps in one CTA per row. Warps own
independent groups in a strided loop. Each group keeps the serial entry's
floating-point expressions and five-step XOR maximum reduction. The
non-finite input filter, F32-to-BF16 rounding, amax floor, UE8M0 code,
saturating payload conversion, clamp and representability checks stay the
same. NaNs contribute a status flag and safe zero before the maximum.
Integer OR is the only reduction whose order changes.

A shared-memory barrier collects every warp's flags before one thread ORs
the row status, including previously set bits. A second barrier publishes
that result before output. Any flagged row still writes all-zero payloads
and scale 127. The CTA reads the device live count before either pass accesses a
row. No workspace, allocation, cross-CTA dependency or changed ABI is
required. All 17 new entries compile with zero local and stack bytes.

The shared implementation is opt-in. DeepSeek registers it at capacity 32,
the maximum decode bucket, alongside the original capacity-8192 entries.
Dense input quantizers, the layer-input quantizer, shared SwiGLU and the
latent ring write prepare a separate small-row operation because their
original operation was prepared once at prefill capacity. Launches of at
most 32 rows select that operation. Larger launches, including staged
prefill window writes, keep the original entries. Other models keep their
original registrations. The original CUDA headers and every existing lock
remain unchanged; `ceil_parallel` is a new bundle lock family.

The six text journal fingerprints change for these bindings only. Each
rank adds 17 entry resolutions and 18 releases (the entries and module),
renumbering handles. The selected small-row symbols and thread blocks
change, with unchanged buffer sizes, arguments, launch order and launch
counts. Normalizing those exact differences reproduces each baseline
journal. The native and BF16 journals have the same record-count change:

| Journal | Added records | New records | New fingerprint |
| --- | ---: | ---: | --- |
| TP1 native | 35 | 12723 | d2034d124f73b8aa |
| TP1 BF16 | 35 | 12721 | b138a9c221ec4898 |
| TP2 native | 70 | 25323 | 5ee9559049b12292 |
| TP2 BF16 | 70 | 25319 | 84b38c26372486eb |
| TP4 native | 140 | 50531 | f7bb59239a1c4cf6 |
| TP4 BF16 | 140 | 50523 | ec7b6e6207a486cb |
| TP2 image lifecycle | 70 | 14147 | 64bd0dda06d7da99 |
| TP4 image lifecycle | 140 | 28251 | 4dd53770b86369f5 |

The image journals change for the same added entries and the text decode
that follows the image. Encoder launches are unchanged. The journal test
also asserts that captured small ceil launches actually use parallel
entries, which catches a registration that remains hidden behind a
prefill-capacity preparation.

The model lab's `ceil_parallel` tests cover all 17 entry pairs at 1, 8 and
32 rows. They compare every payload, scale and status byte on seeded finite
and wide-exponent inputs, signed zero, denormals, maximal magnitudes,
scale boundaries, early and late NaNs and infinities, up-only non-finites,
and sticky prior status. Poisoned eager launches and graph replays vary
the live count through full, zero, partial, over-capacity and full again.
They check inactive rows and shifted status sentinels. Three negative
controls corrupt device payload, scale and status independently and require
the same comparison to reject them. Paired graph timings alternate old/new
order for every width and row count, after bitwise comparisons pass.

CPU checks do not establish GPU exactness or a speedup. The staged pair job
must pass 765 bitwise scenarios, 153 negative controls and 51 paired timings
on each idle GPU. Its frozen inventory includes every served cubin the
suite loads and `probe_sm120`, authenticated against the existing
kernel-lab lock. The CPU dry run verifies module hashes and required
exports without loading the CUDA driver modules. The job builds nothing,
checks a 250 W cap, records kernel-journal Xids, and refuses missing
coverage. GPU results remain pending.

The estimated saving is 18 to 21 ms per C1 step, roughly 13 to 15% of the
143 ms baseline. This is a projection from reducing serial group work by
16 to 32 times, with allowance for launch and barrier overhead. The job's
paired timings and a later serving comparison must establish the actual
saving; no prefill performance claim is made.
