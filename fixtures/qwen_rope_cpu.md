# Qwen text RoPE CPU reference fixture

`qwen_rope_cpu.bin` records the Qwen4Exp default frequency function with
rotary width 64 and theta 10000000, executed with Torch 2.14 CPU.
The source SHA256 is
`77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`.
All 32 F32 inverse-frequency words match the shared HF constants.

The first 128 bytes are those little-endian u32 words. Each following
17-byte record contains a u8 pair index, u32 position, then six BF16 u16
words: x, y, cosine, sine, x*cos-y*sin, y*cos+x*sin.
Four Torch CPU generator seeds (104 through 107) draw x and y uniformly
from [-0.7, 0.7] and round them to BF16. Each seed covers all 32 pairs at
positions 0, 1, 7, 36, 127, 511, 4095, 65535, 131071 and 262143.

The angle is the F32 product of the captured frequency and F32 position.
Trig is evaluated in F64 at that argument, rounded to F32, then BF16.
Products and sums execute separately in Torch BF16. This checks the
adopted accurate-trig convention, not backend-specific approximate F32
Torch transcendental bits. There are 1280 pair records.

The cos/sin words are exact served table expectations. The last two words
are the unserved pure-HF rotation comparison, not served output bits;
served F32 output is checked against the exact sum of BF16 products.
