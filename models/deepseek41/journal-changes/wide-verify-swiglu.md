# Wide verify SwiGLU

At Flash TP4 with more than 64 verify rows, the shared expert's quantized
SwiGLU prepares one op per wide verify bucket the instance reaches
(`verify_bounds::WIDE`), as the ceil quantizers do
([wide-verify-ceil](wide-verify-ceil.md)). The decode activation, registered
at 64 rows, gains a copy at 192 rows with the same symbol, so a verify step
of 65 to 192 rows launches the parallel activation instead of the serial
one.

The [change specification](wide-verify-swiglu.json) is the identity
specification of every fixture except two journals. In the C16 odd-width
allocation fixture (verify steps of 65 rows) and the C32 maximum fixture
(192 rows), a launch rule moves every launch of
`shared_swiglu_w576_v1` at the step's rows to
`shared_swiglu_w576_parallel_v1` with a block of 576 threads, 320 launches
in each. Every grid, argument, allocation, module, resolve, launch count
and operation order is preserved; every other journal, create range and
device history is identical. Both forms compute each group with the same
SwiGLU product and ceil arithmetic (the parallel header's group body
differs only in lane indexing), so codes, scales and row status are
unchanged.

On 2026-10-31 the journal proof of the SwiGLU commit against the ceil
quantizer commit was PROVEN with full history coverage: 21 journals
identical, 2 proven, 23 of 23 fixture histories compared.
