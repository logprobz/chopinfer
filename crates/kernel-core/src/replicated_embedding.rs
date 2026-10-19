//! Replicate a vocabulary-sharded BF16 lookup through immutable peer reads.
//! Each value follows the old local gather, widening, rank-ordered F32 sum
//! from positive zero, final BF16 round and residual-stream broadcast.
//! Replicated callers provide the same IDs and live count on every rank;
//! every table is ready and immutable before any lookup launches.
use crate::contract::{MAX_COMM_RANKS, Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReplicatedEmbedding {
    pub vocab: u32,
    pub width: u32,
    pub ranks: u32,
    pub streams: u32,
}
impl Validate for ReplicatedEmbedding {
    fn validate(&self) -> Result<(), Refusal> {
        if self.ranks == 0
            || self.ranks as usize > MAX_COMM_RANKS
            || self.vocab == 0
            || !self.vocab.is_multiple_of(self.ranks)
            || self.width == 0
            || self.width > i32::MAX as u32 - 1024
            || !self.width.is_multiple_of(4)
            || self.streams == 0
            || self.streams > 8
        {
            return Err(Refusal("invalid replicated embedding geometry"));
        }
        u64::from(self.vocab)
            .checked_mul(u64::from(self.width))
            .and_then(|n| n.checked_mul(2))
            .ok_or(Refusal("embedding table extent overflows"))?;
        Ok(())
    }
}

crate::scratch_record! {
    #[derive(Clone, Debug)]
    pub struct ReplicatedEmbeddingBuffers {
        /// Immutable vocabulary shards in ascending rank order.
        pub tables: values,
        pub ids: value,
        /// Original local gather output, zero on non-owning ranks.
        pub shard: value,
        /// Original F32 widening of the local gather output.
        pub wide: value,
        pub output: value,
        pub expanded: value,
        pub live_rows: value,
    }
}

/// Closed form of the old ordered sum and final device BF16 conversion
/// when exactly one rank owns a BF16 value and every other rank adds +0.
pub fn reduced_bits(bits: u16) -> u16 {
    if bits & 0x7fff > 0x7f80 {
        0x7fff
    } else if bits == 0x8000 {
        0
    } else {
        bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_code_and_owner_matches_the_old_collective_and_catches_raw_copy() {
        let codes: Vec<u16> = (0..=u16::MAX).collect();
        let wide: Vec<f32> = codes
            .iter()
            .map(|&b| f32::from_bits(u32::from(b) << 16))
            .collect();
        let expected: Vec<u16> = codes.iter().copied().map(reduced_bits).collect();
        for ranks in [1, 2, 4, 8] {
            for owner in 0..ranks {
                let inputs: Vec<_> = (0..ranks)
                    .map(|rank| {
                        if rank == owner {
                            wide.clone()
                        } else {
                            vec![0.0; codes.len()]
                        }
                    })
                    .collect();
                assert_eq!(
                    crate::oracle::all_reduce_rounded(&inputs).unwrap(),
                    expected
                );
            }
        }
        assert_eq!(
            codes.iter().zip(&expected).filter(|(a, b)| a != b).count(),
            254
        );
        assert_ne!(reduced_bits(0x8000), 0x8000);
        assert_ne!(reduced_bits(0x7f81), 0x7f81);
    }
}
