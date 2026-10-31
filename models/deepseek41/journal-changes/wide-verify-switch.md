# Wide verify candidate option

The `wide_verify` model option (`execution_options::WIDE_VERIFY`, default
off) gates the wide verify buckets of the
[ceil quantizers](wide-verify-ceil.md) and the
[shared SwiGLU](wide-verify-swiglu.md). Without it
`Shape::wide_verify_rows` is empty, so the package prepares, allocates and
launches what it did before those changes. The allocation test pins two
more fixtures with the option on: the C16 odd-width fixture (verify steps
of 65 rows) and the C32 maximum fixture (192 rows).

The [change specification](wide-verify-switch.json) is the identity
specification of every fixture, with those two journals declared added,
against the parent of the ceil quantizer change. On 2026-10-31 its journal
proof was PROVEN: 23 journals identical, 2 declared added, 23 of 25 fixture
histories compared (the added fixtures are compared to nothing). The two
added journals are byte-identical to the C16 odd-width and C32 maximum
journals of the SwiGLU change, so with the option on the package launches
what those changes proved.
