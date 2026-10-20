# Fused candidate publication

This lab candidate replaces block maxima, local candidate selection and the
ordered final merge when the physical width holds at most twice the compiled
candidate budget. It does not change position scoring or serving defaults.

Each thread computes the maximum of the same eight consecutive scores in
the same order. NaN and positive infinity fail the row even in the newest
block. Negative infinity denotes a hole. The newest nonempty block is pinned
after its scores are checked, including when all its scores are holes.
Invalid visible counts, source failure, empty rows, inactive rows, output
padding, row tags and the compiled output stride keep their existing meaning.

The block score/index keys stay in registers. Four radix passes find the
boundary score, then an increasing-block scan retains its lowest-index ties.
This is the same total order as the original eight-pass 64-bit key cutoff.
When fewer eligible blocks exist than the budget, the zero-rank cutoff keeps
all of them. When every visible block fits, no radix pass runs.

The original live producer already skips radix cutoff for sufficiently short
lists. This candidate additionally removes the maxima allocation's traffic,
the 32 padded shard lists and two launches. It still reads and validates all
visible input scores. Skipping position scoring outright would lose the
existing refusal on invalid operands or arithmetic and is not admitted here.

The adapter accepts only a previously validated three-stage valid producer,
preserving its input and output bindings. Its private workspace is not an
output and need not be materialized. Wider inputs keep the original producer.
GPU bitwise, poisoned-input and timing qualification is required before use.
