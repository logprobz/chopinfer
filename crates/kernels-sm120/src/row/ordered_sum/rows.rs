//! Rows of terms for the ordered sum's CPU and GPU tests: random rows
//! across exponent ranges ([`random_row`]) and adversarial rows
//! ([`adversarial_rows`]): ties at every position, binade crossings at
//! every chunk boundary, mispredicted binades, a dominant term, equal terms,
//! subnormals and zeros, sums near overflow, and infinities, NaNs and
//! negative terms.

use super::{Layout, class};

/// xorshift64*.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    pub fn bits(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniform in `0..n` (n > 0), from the high bits.
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.bits() >> 32) * n) >> 32
    }

    /// Uniform in `low..=high`.
    pub fn range(&mut self, low: i32, high: i32) -> i32 {
        low + self.below((high - low + 1) as u64) as i32
    }
}

/// 2^e for e in -149..=127.
pub fn pow2(e: i32) -> f32 {
    assert!((-149..=127).contains(&e), "2^{e}");
    if e >= -126 {
        f32::from_bits(((e + 127) as u32) << 23)
    } else {
        f32::from_bits(1 << (e + 149))
    }
}

/// The ulp of the running sum `s`'s class (s finite and non-negative).
pub fn ulp(s: f32) -> f32 {
    pow2(class(s.to_bits()) as i32 - 150)
}

impl Rng {
    /// A finite non-negative F32 with biased exponent field `exponent` (0 to
    /// 254) and random mantissa bits, keeping the top `bits` (1 to 23) of them.
    fn value(&mut self, exponent: i32, bits: u32) -> f32 {
        let mantissa = (self.bits() >> 41) as u32 & (0x7f_ffff << (23 - bits.min(23))) & 0x7f_ffff;
        f32::from_bits(((exponent.clamp(0, 254) as u32) << 23) | mantissa)
    }

    /// The square of a random BF16 value in binade `exponent` (unbiased): a
    /// collapsed row's square as the norm computes it.
    fn bf16_square(&mut self, exponent: i32) -> f32 {
        let biased = (exponent + 127).clamp(0, 254) as u32;
        let x = f32::from_bits((biased << 23) | (((self.bits() >> 57) as u32) << 16));
        x * x
    }
}

/// Fills `row` with a random row: one of ten kinds over random exponent
/// ranges (squares of BF16 values, few-bit and full-bit F32 values,
/// subnormals, growing and dominant terms, half-ulp ties of the running
/// sum), with occasional zeros and rare infinities, NaNs and negative
/// terms.
pub fn random_row(rng: &mut Rng, row: &mut [f32]) {
    let n = row.len();
    match rng.below(10) {
        0..=2 => {
            let top = rng.range(-70, 56);
            let spread = rng.range(0, 24);
            for t in row.iter_mut() {
                *t = {
                    let e = top - rng.range(0, spread);
                    rng.bf16_square(e)
                };
            }
        }
        3 | 4 => {
            let low = rng.range(0, 240);
            let spread = rng.range(0, 30);
            let bits = rng.range(1, 23) as u32;
            for t in row.iter_mut() {
                *t = {
                    let e = low + rng.range(0, spread);
                    rng.value(e, bits)
                };
            }
        }
        5 => {
            for t in row.iter_mut() {
                *t = f32::from_bits((rng.bits() >> 39) as u32 & 0x00ff_ffff);
            }
        }
        6 => {
            let low = rng.range(0, 200);
            let rise = rng.range(1, 40);
            let bits = rng.range(1, 23) as u32;
            for (i, t) in row.iter_mut().enumerate() {
                *t = rng.value(low + (i as i32 * rise) / n as i32, bits);
            }
        }
        7 => {
            let low = rng.range(1, 180);
            let spread = rng.range(0, 10);
            for t in row.iter_mut() {
                *t = {
                    let e = low + rng.range(0, spread);
                    rng.value(e, 23)
                };
            }
            let at = rng.below(n as u64) as usize;
            row[at] = {
                let e = low + rng.range(10, 60);
                rng.value(e, 23)
            };
        }
        8 => {
            // Odd multiples of half the running sum's ulp: a tie at most
            // positions.
            let mut s = {
                let e = rng.range(10, 220);
                rng.value(e, 23)
            };
            let width = rng.range(0, 12) as u32;
            row[0] = s;
            for t in row.iter_mut().skip(1) {
                let odd = (2 * rng.below(1 << width) + 1) as f32;
                *t = ulp(s) * 0.5 * odd;
                s += *t;
            }
        }
        _ => {
            let low = rng.range(0, 250);
            let spread = rng.range(0, 254 - low);
            for t in row.iter_mut() {
                *t = {
                    let e = low + rng.range(0, spread);
                    rng.value(e, 23)
                };
            }
        }
    }
    if rng.below(16) == 0 {
        for _ in 0..rng.range(1, 64) {
            row[rng.below(n as u64) as usize] = 0.0;
        }
    }
    if rng.below(256) == 0 {
        let special = [
            f32::INFINITY,
            f32::NAN,
            f32::from_bits(0x7f80_0001),
            -1.0,
            -0.0,
        ];
        row[rng.below(n as u64) as usize] = special[rng.below(special.len() as u64) as usize];
    }
}

/// A named adversarial row.
pub struct Adversarial {
    pub name: &'static str,
    pub terms: Vec<f32>,
}

/// Ties at every position in one binade: 2^(23 + e), then odd halves of
/// 2^e (each a tie on the running grid of ulp 2^e), so the running parity
/// decides every add.
fn ties_in_one_binade(rng: &mut Rng, n: usize, e: i32) -> Vec<f32> {
    let mut row = vec![pow2(23 + e)];
    row.extend((1..n).map(|_| (rng.below(4) as f32 + 0.5) * pow2(e)));
    row
}

/// A tie at every position across binades: each term an odd multiple of
/// half the running sum's ulp, from a start just below a power of two.
fn ties_across_binades(rng: &mut Rng, n: usize, start: i32, width: u32) -> Vec<f32> {
    let mut s = pow2(start) * (2.0 - pow2(-rng.range(1, 12)));
    let mut row = vec![s];
    for _ in 1..n {
        let odd = (2 * rng.below(1 << width) + 1) as f32;
        let t = ulp(s) * 0.5 * odd;
        row.push(t);
        s += t;
    }
    row
}

/// A binade crossing at every chunk boundary of `layout`: small terms (odd
/// multiples of the half ulp and random), then a last term that lands the
/// running sum exactly on the next power of two, one ulp below it, or past
/// it with a tie; or a first term that crosses.
fn crossings_at_chunks(rng: &mut Rng, layout: Layout, start: i32) -> Vec<f32> {
    let mut row = Vec::with_capacity(layout.terms);
    let mut s = 0.0_f32;
    for worker in 0..layout.workers {
        let chunk = layout.chunk(worker);
        let first_crosses = rng.below(4) == 0;
        for i in chunk.clone() {
            let next = if s == 0.0 {
                pow2(start + rng.range(0, 3))
            } else {
                // The next power of two above s.
                pow2(class(s.to_bits()) as i32 - 126)
            };
            let u = ulp(s);
            let t = if s == 0.0 {
                next * 0.75
            } else if first_crosses && i == chunk.start {
                (next - s) + u * 0.5 * (2 * rng.below(3) + 1) as f32
            } else if !first_crosses && i + 1 == chunk.end {
                match rng.below(3) {
                    0 => next - s,
                    1 => (next - s) - u,
                    _ => (next - s) + u * 0.5,
                }
            } else {
                u * 0.5 * (2 * rng.below(8) + 1) as f32
            };
            row.push(t);
            s += t;
            if !s.is_finite() {
                s = 0.0;
            }
        }
    }
    row
}

/// A running sum the chunk sums mispredict: it starts `below` ulps under a
/// power of two, and each term is between half an ulp and one ulp, so every
/// add rounds up to a whole ulp while the chunk sums count about three
/// quarters of one: the predictions place the crossing too late, and runs
/// predicted in the lower binade end in the upper one. With `low` the
/// terms lie under half an ulp instead: every add rounds away while the
/// predictions cross the power of two and place later runs in the upper
/// binade.
fn mispredicted(rng: &mut Rng, n: usize, e: i32, below: u32, low: bool) -> Vec<f32> {
    let s = pow2(e) - pow2(e - 24) * below as f32;
    let u = pow2(e - 24);
    let mut row = vec![s];
    for _ in 1..n {
        let quarter = rng.below(1 << 20) as f32 / (1 << 22) as f32;
        row.push(if low {
            u * (0.25 + quarter)
        } else {
            u * (0.5 + quarter + 0.0001)
        });
    }
    row
}

/// Every adversarial row of `layout`'s width, drawn from `seed`.
pub fn adversarial_rows(layout: Layout, seed: u64) -> Vec<Adversarial> {
    let n = layout.terms;
    let mut rng = Rng::new(seed);
    let mut rows = Vec::new();
    let mut push = |name: &'static str, terms: Vec<f32>| {
        assert_eq!(terms.len(), n, "{name}");
        rows.push(Adversarial { name, terms });
    };
    for e in [-120, -24, 0, 50, 100] {
        push("ties in one binade", ties_in_one_binade(&mut rng, n, e));
    }
    for (start, width) in [(-120, 0), (-100, 3), (0, 1), (40, 8), (100, 2)] {
        push(
            "ties across binades",
            ties_across_binades(&mut rng, n, start, width),
        );
    }
    for start in [-149, -146, -140, -131] {
        push(
            "crossings at chunk boundaries",
            crossings_at_chunks(&mut rng, layout, start),
        );
    }
    for (e, below, low) in [
        (0, 30, false),
        (-100, 7, false),
        (60, 200, false),
        (0, 1, true),
        (90, 3, true),
    ] {
        push(
            "mispredicted binades",
            mispredicted(&mut rng, n, e, below, low),
        );
    }
    for (big, small) in [(100, 0), (60, -60), (13, 0)] {
        let mut row: Vec<f32> = (0..n)
            .map(|_| {
                let e = 127 + small + rng.range(-3, 3);
                rng.value(e, 23)
            })
            .collect();
        row[0] = rng.value(127 + big, 23);
        push("dominant first term", row.clone());
        let at = rng.below(n as u64) as usize;
        row.swap(0, at);
        push("dominant term", row);
    }
    for v in [
        1.0,
        3.0,
        0.1,
        1e-40,
        f32::MIN_POSITIVE,
        pow2(-149),
        7.0e30,
        f32::MAX / 8192.0,
    ] {
        push("equal terms", vec![v; n]);
    }
    push("zeros", vec![0.0; n]);
    push(
        "subnormals",
        (0..n)
            .map(|_| f32::from_bits((rng.bits() >> 41) as u32))
            .collect(),
    );
    push(
        "zeros then subnormals then normals",
        (0..n)
            .map(|i| match 3 * i / n {
                0 => 0.0,
                1 => f32::from_bits((rng.bits() >> 42) as u32),
                _ => {
                    let e = rng.range(1, 30);
                    rng.value(e, 23)
                }
            })
            .collect(),
    );
    push(
        "alternating zeros and subnormals",
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    0.0
                } else {
                    pow2(-149) * (i % 7) as f32
                }
            })
            .collect(),
    );
    push(
        "huge terms",
        (0..n)
            .map(|_| {
                let e = rng.range(250, 254);
                rng.value(e, 23)
            })
            .collect(),
    );
    push(
        "terms up to the largest finite sum",
        (0..n)
            .map(|i| {
                if i == 0 {
                    f32::MAX / 2.0
                } else {
                    f32::MAX / (2 * n) as f32
                }
            })
            .collect(),
    );
    // The largest finite sum, then half its ulp: a tie that rounds to even,
    // the overflow.
    push(
        "a tie to overflow",
        (0..n)
            .map(|i| if i == 0 { f32::MAX } else { pow2(103) })
            .collect(),
    );
    let specials = [
        f32::INFINITY,
        f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::from_bits(0xffc0_0000),
        -1.0,
        -0.0,
    ];
    for special in specials {
        for at in [0, n / 2, n - 1, layout.chunk(layout.workers / 2).start] {
            let mut row: Vec<f32> = (0..n)
                .map(|_| {
                    let e = rng.range(-4, 4);
                    rng.bf16_square(e)
                })
                .collect();
            row[at] = special;
            push("special term", row);
        }
    }
    let mut both: Vec<f32> = (0..n)
        .map(|_| {
            let e = rng.range(-4, 4);
            rng.bf16_square(e)
        })
        .collect();
    both[n / 3] = f32::INFINITY;
    both[2 * n / 3] = f32::NAN;
    push("infinity then NaN", both);
    rows
}
