# HF BF16 text RoPE fixture

`hf_rope_cpu.bin` records Torch 2.14.0+cpu operations for theta 10,000 and
10,000,000 with 64 rotary dimensions. It contains no model weights.

The first 256 bytes are 64 little-endian u32 inverse-frequency bit patterns,
32 per theta, from `1 / theta ** (arange(0, 64, 2).float() / 64)`.
Each remaining 18-byte record is a u8 theta index, u8 pair, u32 position,
then six u16 BF16 codes: x, y, cos, sin, first output, second output.
The BF16 inputs use four fixed random seeds. Positions include zero,
previous failures, long-context edges, and trig-rounding boundary probes.

Angle multiplication is F32. Trig uses Torch double evaluation of that F32
argument, then F32 and BF16 casts. Outputs are `(x*cos)-(y*sin)` and
`(y*cos)+(x*sin)` using eager BF16 products and addition. This fixture
checks the HF angle and cos/sin convention and preserves the pure-HF
BF16 rotation for distance checks. Served rotation uses F32 product/FMA
and final storage rounding. The fixture does not reproduce the few
boundary differences in Torch's approximate F32 trig implementation.
