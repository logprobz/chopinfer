//! Shared inputs for exact router projection checks.
use super::{CAPACITY, K, N, bf16, widen};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pattern {
    /// Pseudo-random rows; every 61st row from the 11th all zero.
    Clean,
    /// Clean, plus every 37th row from the 3rd with one 1e15 element (the
    /// rest normalize below the tensor domain), every 53rd from the 7th
    /// with a NaN, and the [`split_row`]s.
    Outliers,
}
pub struct Rng(u64);
impl Rng {
    pub fn step(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Roughly normal, from the sum of four uniforms.
    pub fn normal(&mut self) -> f32 {
        (0..4)
            .map(|_| (self.step() >> 40) as f32 / (1u64 << 24) as f32)
            .sum::<f32>()
            * 1.732
            - 3.464
    }
}

/// Outlier rows whose odd elements are scaled by 2^-40, so they normalize
/// below the tensor domain while the even ones keep many significant
/// terms: every 41st row from the 9th that is not a zero, 1e15 or NaN row.
/// Only these uncertified rows are certain to sum differently on the
/// tensor chain than on the SIMT path; a 1e15 row has one significant
/// term and a NaN row is NaN on both.
pub fn split_row(r: usize) -> bool {
    r % 41 == 9 && r % 61 != 11 && r % 37 != 3 && r % 53 != 7
}

pub fn input_rows(pattern: Pattern) -> Vec<u16> {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut rows = vec![0u16; (CAPACITY * K) as usize];
    for r in 0..CAPACITY as usize {
        let row = &mut rows[r * K as usize..(r + 1) * K as usize];
        if r % 61 == 11 {
            continue;
        }
        for x in row.iter_mut() {
            *x = bf16(rng.normal());
        }
        if pattern == Pattern::Outliers {
            if split_row(r) {
                for x in row.iter_mut().skip(1).step_by(2) {
                    *x = bf16(widen(*x) * (-40f32).exp2());
                }
            }
            if r % 37 == 3 {
                row[r % K as usize] = bf16(1e15);
            }
            if r % 53 == 7 {
                row[(r * 7) % K as usize] = 0x7fc0;
            }
        }
    }
    rows
}
pub fn norm_weight() -> Vec<u16> {
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    (0..K).map(|_| bf16(1.0 + 0.25 * rng.normal())).collect()
}
/// Router weights; experts 80 to 95 are below the tensor domain, so those
/// columns take the SIMT path in both arms.
pub fn router_weight() -> Vec<u16> {
    let mut rng = Rng(0x1405_7b7e_f767_814f);
    let mut w = vec![0u16; (N * K) as usize];
    for f in 0..N as usize {
        for k in 0..K as usize {
            let scale = if (80..96).contains(&f) { 1e-20 } else { 0.02 };
            w[f * K as usize + k] = bf16(scale * rng.normal());
        }
    }
    w
}
