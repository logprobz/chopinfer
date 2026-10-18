# Trained block quantization and row status

`BlockQuantizeQuery` rounds each input to BF16 before amax and code decisions.
Successful rows retain the E4M3/32 ceil-UE8M0, E2M1/E4M3-per-16 or
E2M1/32 ceil-UE8M0 arithmetic. Output is packed codes/scales or lossless BF16.

Every launch takes a caller-owned U32 status plane and a U64 word offset.
The plane covers `(status_row_offset + planned_rows) * 4` bytes, checked for
overflow before launch. The kernel atomically ORs flags into the corresponding
live words. Callers clear the selected range before an independent sequence,
order launches which share a row, and check status before consuming results.
The kernel never clears a prior bit. Other words, including inactive rows,
stay unchanged. Status cannot alias another operand.

| Bit | Name | Meaning |
| --- | --- | --- |
| 1 | `NONFINITE_INPUT` | An original or BF16-rounded input is NaN or infinity. |
| 2 | `SCALE_OVERFLOW` | Generic caller-reported scale failure. Finite E2M1/E4M3 saturation does not set it. |
| 4 | `UNREPRESENTABLE_OUTPUT` | A quantized result is not finite and exact in BF16, so its packed pair would fail a shared BF16 decoder's range guard. |

A nonzero prior or newly raised status makes the entire row positive zero.
Packed output has zero codes and scale 1: UE8M0 byte 127 or E4M3 byte `0x38`.
These pairs are finite and accepted by every shared decoder. No failure uses
NaN scale byte `0xff`. Validation precedes every row write, so one bad group
cannot race valid groups into leaving a partial row. The block quantizer
retains no traps, including in fake-quant conversion. Width, group divisibility,
input/output types, capacity, buffer extents and aliases are checked on the host.
Zero planned rows produce no launch.

## Quantized clamped SwiGLU status

`SiluMulQuantizeBuffers` uses the same required U32 `status` plane and U64
`status_row_offset` in words. Its entry arguments are `gate, up, codes,
scales, live_rows, status, status_row_offset`. Status must cover
`(status_row_offset + planned_rows) * 4` bytes, with checked arithmetic,
and cannot alias any other role. Clear the selected words once per step,
order launches sharing rows, and inspect the accumulated flags before
accepting output.

Any NaN or infinity in either original F32 operand sets `NONFINITE_INPUT`.
That element skips the product entirely, so a negative infinite gate cannot
produce NaN through division by an infinite exponential. A product that
rounds to BF16 infinity, or a packed result not finite and exact in BF16,
sets `UNREPRESENTABLE_OUTPUT`. The registration refusal for clamps with
`max(gate_max, 1) * up_limit` above the BF16 maximum remains. Both former
device traps are removed.

One 32-thread CTA validates every group of a row before updating status
with atomic OR and writing output. Any prior or new bit produces positive
zero codes and UE8M0 scale 127 throughout that row. Passing groups retain
their original F32 operation order, single BF16 product rounding, warp
amax reduction, ceil scale selection and E4M3 conversion. The lab retains
its boundary-aware conformance cases and compares every finite input set
bitwise with a frozen previous cubin. Status cases cover both operands,
prior bits, ordered launches, offset guards, graph and PDL replay. Device
controls drop a flag, address another status row, zero only part of a row,
or omit the product skip; each must fail the unchanged oracle.

## Finite E2M1/E4M3 scale saturation

TileLang 0.1.12, the pinned vendor dependency, saturates the scale to 448.
A finite BF16 amax above `6 * 448` is valid: first compute `amax / 6` in F32,
convert it with FP8 E4M3 SATFINITE, widen the scale, divide each operand by
that scale and clamp it to `[-6, 6]` before E2M1 conversion. This yields
scale byte `0x7e` and clips values above magnitude 2688, with no status flag.
A non-finite original or BF16-rounded operand is still invalid.

The locally installed sources and the vendor's generated CUDA establish the
following lowering chain. Paths below are relative to the TileLang 0.1.12
package or its vendored CUTLASS tree; no assumption about another cast API is
needed.

- Vendor `inference/kernel.py:163` casts `amax / 6` to FP8 and widens it.
  Lines 171 and 175 clamp the normalized payload in both output paths.
- The compiled `fp4_quant_kernel` CUDA, line 41, uses
  `float(fp8_e4_t(amax_local[0] / 6.0f))` for this scalar expression.
- `src/tl_templates/cuda/cuda_fp8.h:7` aliases `fp8_e4_t` to
  `tl::float_e4m3_t`. `src/tl_templates/cuda/common.h:677-678` inherits
  the constructors of `cute::float_e4m3_t`.
- `3rdparty/cutlass/include/cute/numeric/numeric_types.hpp:91` imports
  `cutlass::float_e4m3_t`. Its float constructor in
  `3rdparty/cutlass/include/cutlass/float8.h:491-492` calls `from_float`.
  Line 421 emits `cvt.rn.satfinite.e4m3x2.f32`; the software fallback
  saturates as well. The vector lowering in
  `src/cuda/codegen/codegen_cuda.cc:1729` independently uses
  `__NV_SATFINITE`, although the observed scale takes the scalar path.

The old finite-overflow refusal was stricter than this vendor behavior.
Valid values within that prior domain keep their exact code and scale bits.

## Fused paged key writers

Index-key and compressed-KV writes use the same quantizer helper after F32
normalization and RoPE. Both take caller-owned `status` and a U64
`status_row_offset`, with the same OR flags and whole-row zero/scale-1 rule.
The index path records decoder representability failures before using the
trap-free conversion. The writers retain no traps.

Negative or out-of-range positions, item IDs, rotary-table rows, logical
pages and physical pages are holes. They read no key data, write no cache
bytes, and leave the corresponding status word unchanged. Live, valid
addresses receive the row's status and its defined quantized output.
Status cannot alias any operand or cache plane. Projection quantizer status
likewise cannot alias any other bound role of its composed operation,
including final output, weights, residual, routing or compute scratch.


The position-key writer also exposes explicit `IndexerKeyFormat` storage.
`Fp4` keeps the native E2M1 and UE8M0 planes. `Bf16` stores each exact native
product, in channel order, in the value plane with no scale allocation.
It applies the same norm, RoPE, trained quantization and finite-product
checks. Changing storage does not admit a previously refused row. Its
256-byte entries and the compressed latent writer's existing 1,024-byte
BF16 entries use byte page strides, allowing both streams to be embedded
in a shared physical page. The BF16 backend requires an empty scales view
and scale stride zero for index keys. A separate entry family adds these
instantiations without changing existing cubins.
