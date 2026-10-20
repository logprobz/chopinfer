# Full-chunk query norm and ceil

This candidate preserves the F32 query norm output and produces its first
E4M3/32 codes and ceil UE8M0 scales in the same CTA. Its sum uses the original
256 logical threads, eight consecutive elements per chain, warp shuffle
tree and CTA tree. It keeps both F32 multiplies, the original rsqrt, and the
quantizer's BF16 operand rounding. Input errors retain precedence over
arithmetic errors; sticky row status zeros the same output planes.

Each CTA owns a shared arena. The task accepts a virtual row index and an
arena; only its entry wrapper performs grid dependency operations. It uses
no global scratch or tickets. The cached variant reads the F32 input once;
the reread variant retains the norm's second global read. Both stage the
normalized row and provisional codes before whole-row admission.

Serving preparation selects the cached variant only for Flash prefill with
8192 rows. Decode, verify and smaller prefill buckets keep their previous
selection. The later indexer quantizer remains because intervening kernels
can change row status. A full-row 40-layer schedule removes 40 launches per
rank. The profiled default cone schedule has 20 eligible full-row boundaries
per chunk. This is a candidate until the pair job passes bitwise and timing gates.

The lab uses authenticated 8192-row query operands and BF16 norm weights
captured at layers 0 and 39. Each GPU executes 28 matrix cases, twelve
chunk/position/concurrency checks, two kernel mutation controls and eight paired
timing comparisons. Memcheck executes four additional full-chunk cases.
No selected GPU, zero tests, missing cases and ignored tests are failures.
Mismatch collection continues through the full matrix.

Set `CHOPINFER_QUERY_PREFILL` to the frozen artifact directory. Run the CPU
artifact test before the two ignored GPU tests. The job parser accepts
libtest prefixes and checks exact case identities and counts. Timing uses
alternating AB/BA CUDA events around the whole old chain and fused launch.
Cold samples evict L2; hot samples add an untimed replay after eviction.
Kernel-event savings require fresh served attribution before a per-chunk
claim, especially when upper-layer skip changes row counts.
