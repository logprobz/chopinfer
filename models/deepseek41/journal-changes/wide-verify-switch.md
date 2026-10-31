# Wide verify candidate option

The `wide_verify` model option (`execution_options::WIDE_VERIFY`, default
off) gates the wide verify buckets of the
[ceil quantizers](wide-verify-ceil.md) and the
[shared SwiGLU](wide-verify-swiglu.md). Those two changes served the wide
buckets unconditionally; the option makes them opt-in. Without it
`Shape::wide_verify_rows` is empty, so the package prepares, allocates and
launches what it did before those changes: with the option off, every
fixture keeps the selections of the parent of the ceil quantizer change,
not those of the SwiGLU change. The allocation test pins two more fixtures
with the option on: the C16 odd-width fixture (verify steps of 65 rows) and
the C32 maximum fixture (192 rows).

The [change specification](wide-verify-switch.json) is the identity
specification of every fixture, with those two journals declared added.
Its base is, by design, the parent of the ceil quantizer change rather
than the SwiGLU change: identity against that base is the claim that the
option off restores the pre-change selections. On 2026-10-31 its journal
proof was PROVEN: 23 journals identical, 2 declared added, 23 of 25 fixture
histories compared (the added fixtures are compared to nothing). The two
added journals are byte-identical to the C16 odd-width and C32 maximum
journals of the SwiGLU change, so with the option on the package launches
what those changes proved.
