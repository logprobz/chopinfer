//! Exact arithmetic where the Python rules use it: `statistics.mean` sums
//! floats as exact fractions and rounds the mean once.

/// Limbs of 64 bits in units of 2^-1074, the smallest subnormal: enough for
/// any finite double (2^1024 is 2098 bits above the unit) summed 2^64 times.
const LIMBS: usize = 35;

#[derive(Clone)]
struct Magnitude([u64; LIMBS]);

impl Magnitude {
    fn zero() -> Self {
        Self([0; LIMBS])
    }

    /// Adds `mantissa` shifted left by `shift` bits.
    fn add_shifted(&mut self, mantissa: u64, shift: u32) {
        let limb = (shift / 64) as usize;
        let offset = shift % 64;
        let wide = u128::from(mantissa) << offset;
        let mut carry = 0u128;
        for (index, part) in [wide as u64, (wide >> 64) as u64].into_iter().enumerate() {
            let sum = u128::from(self.0[limb + index]) + u128::from(part) + carry;
            self.0[limb + index] = sum as u64;
            carry = sum >> 64;
        }
        let mut index = limb + 2;
        while carry != 0 {
            let sum = u128::from(self.0[index]) + carry;
            self.0[index] = sum as u64;
            carry = sum >> 64;
            index += 1;
        }
    }

    fn less_than(&self, other: &Self) -> bool {
        self.0.iter().rev().cmp(other.0.iter().rev()).is_lt()
    }

    /// `self - other`, with `other <= self`.
    fn minus(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        let mut borrow = 0u64;
        for index in 0..LIMBS {
            let (difference, under) = self.0[index].overflowing_sub(other.0[index]);
            let (difference, under_borrow) = difference.overflowing_sub(borrow);
            result.0[index] = difference;
            borrow = u64::from(under || under_borrow);
        }
        result
    }

    /// The quotient by `divisor` and the remainder.
    fn divide(&self, divisor: u64) -> (Self, u64) {
        let mut quotient = Self::zero();
        let mut remainder = 0u128;
        for index in (0..LIMBS).rev() {
            let current = (remainder << 64) | u128::from(self.0[index]);
            quotient.0[index] = (current / u128::from(divisor)) as u64;
            remainder = current % u128::from(divisor);
        }
        (quotient, remainder as u64)
    }

    fn top_bit(&self) -> Option<u32> {
        (0..LIMBS)
            .rev()
            .find(|index| self.0[*index] != 0)
            .map(|index| index as u32 * 64 + 63 - self.0[index].leading_zeros())
    }

    fn bit(&self, position: u32) -> bool {
        self.0[(position / 64) as usize] >> (position % 64) & 1 == 1
    }

    fn any_below(&self, position: u32) -> bool {
        let limb = (position / 64) as usize;
        self.0[..limb].iter().any(|part| *part != 0)
            || self.0[limb] & ((1u64 << (position % 64)) - 1) != 0
    }

    /// Bits `low` through `low + 52` as an integer.
    fn bits_from(&self, low: u32) -> u64 {
        let mut value = 0u64;
        for offset in 0..53 {
            if self.bit(low + offset) {
                value |= 1 << offset;
            }
        }
        value
    }
}

/// The mean of `values` as Python's `statistics.mean` gives it: the exact
/// sum divided by the count, rounded to the nearest double (ties to even).
/// `None` for no values; a non-finite value gives the naive mean.
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Some(values.iter().sum::<f64>() / values.len() as f64);
    }
    let (mut positive, mut negative) = (Magnitude::zero(), Magnitude::zero());
    for value in values {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as u32;
        let fraction = bits & ((1 << 52) - 1);
        let (mantissa, shift) = if exponent == 0 {
            (fraction, 0)
        } else {
            (fraction | (1 << 52), exponent - 1)
        };
        let target = if value.is_sign_negative() {
            &mut negative
        } else {
            &mut positive
        };
        target.add_shifted(mantissa, shift);
    }
    let (sum, sign) = if positive.less_than(&negative) {
        (negative.minus(&positive), -1.0)
    } else {
        (positive.minus(&negative), 1.0)
    };
    let count = values.len() as u64;
    let (quotient, remainder) = sum.divide(count);
    // The mean is (quotient + remainder / count) units of 2^-1074.
    let magnitude = match quotient.top_bit() {
        Some(top) if top >= 53 => {
            let low = top - 52;
            let mut mantissa = quotient.bits_from(low);
            let half = quotient.bit(low - 1);
            let sticky = quotient.any_below(low - 1) || remainder != 0;
            let mut top = top;
            if half && (sticky || mantissa & 1 == 1) {
                mantissa += 1;
                if mantissa == 1 << 53 {
                    mantissa >>= 1;
                    top += 1;
                }
            }
            // The value is 1.fraction times 2^(top - 1074).
            let biased = u64::from(top) + 1023 - 1074;
            f64::from_bits((biased << 52) | (mantissa & ((1 << 52) - 1)))
        }
        _ => {
            // Below 2^53 units every integer is a double whose bits are the integer.
            let mut units = quotient.0[0];
            let twice = u128::from(remainder) * 2;
            if twice > u128::from(count) || (twice == u128::from(count) && units & 1 == 1) {
                units += 1;
            }
            f64::from_bits(units)
        }
    };
    Some(sign * magnitude)
}

#[cfg(test)]
mod tests {
    use super::mean;

    #[test]
    fn the_mean_is_rounded_once() {
        assert_eq!(mean(&[]), None);
        assert_eq!(mean(&[0.1, 0.2, 0.3]), Some(0.2));
        // The naive mean of these is 0.20000000000000004.
        assert_eq!((0.1 + 0.2 + 0.3) / 3.0, 0.20000000000000004);
        assert_eq!(mean(&[1e308, 1e308, -1e308]), Some(1e308 / 3.0));
        assert_eq!(mean(&[1e16, 1.0, -1e16]), Some(1.0 / 3.0));
        assert_eq!(mean(&[0.05, -0.0, 0.0]), Some(0.016666666666666666));
        assert_eq!(mean(&[5e-324, 0.0]), Some(0.0));
        assert_eq!(mean(&[5e-324, 5e-324, 5e-324, 0.0]), Some(5e-324));
        assert_eq!(mean(&[-0.25, -0.5]), Some(-0.375));
        assert_eq!(mean(&[f64::MAX, f64::MAX]), Some(f64::MAX));
    }
}
