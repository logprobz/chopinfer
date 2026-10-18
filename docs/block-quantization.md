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
