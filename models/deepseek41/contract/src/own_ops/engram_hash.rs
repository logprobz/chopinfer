//! Exact integer Engram hashing over device target tokens and committed lookback.
use super::engram::{DEAD, HashTables};
use kernel_core::backend::Need;
use kernel_core::contract::Refusal;

pub const CONTRACT: &str = "deepseek41.engram-hash.u64.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EngramHash {
    pub vocab: u32,
    pub ngram: u32,
    pub heads: u32,
    pub layers: u32,
    pub items: u32,
    /// Fixed output row stride between layers, in target positions.
    pub stride: u32,
}

impl EngramHash {
    pub fn validate(self, need: Need) -> Result<(), Refusal> {
        if self.vocab == 0
            || !(2..=8).contains(&self.ngram)
            || self.heads == 0
            || self.layers == 0
            || self.items == 0
            || need.capacity == 0
            || need.capacity > self.stride
        {
            return Err(Refusal(
                "Engram device hash has invalid geometry or capacity",
            ));
        }
        self.ngram
            .checked_sub(1)
            .and_then(|n| n.checked_mul(self.heads))
            .and_then(|n| n.checked_mul(self.layers))
            .ok_or(Refusal("Engram device hash columns overflow"))?;
        4_u64
            .checked_mul(u64::from(self.layers))
            .and_then(|bytes| bytes.checked_mul(u64::from(self.stride)))
            .and_then(|bytes| bytes.checked_mul(self.columns()))
            .ok_or(Refusal("Engram device hash output size overflows"))?;
        Ok(())
    }
    pub fn columns(self) -> u64 {
        u64::from(self.ngram - 1) * u64::from(self.heads)
    }
    /// U64 words: compressed-vocab size, pad ID, raw-token map, multipliers,
    /// primes, then offsets. All values come from the authenticated HashTables.
    pub fn constant_words(self) -> u64 {
        2 + u64::from(self.vocab)
            + u64::from(self.layers) * u64::from(self.ngram)
            + 2 * u64::from(self.layers) * self.columns()
    }
    pub fn output_bytes(self) -> u64 {
        4 * u64::from(self.layers) * u64::from(self.stride) * self.columns()
    }
}

kernel_core::scratch_record! {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct EngramHashBuffers {
        pub tokens: value,
        pub token_batch: value,
        pub q_start: value,
        /// Last ngram-1 compressed IDs per item, oldest first; left-pad with DEAD.
        pub history: value,
        pub text: value,
        pub constants: value,
        /// Layer-major indices, with a fixed stride between layers.
        pub ids: value,
        /// OR one on malformed device metadata; never trap on device input.
        pub status: value,
    }
}

pub struct HashInput<'a> {
    pub tokens: &'a [u32],
    pub token_batch: &'a [i32],
    pub q_start: &'a [i32],
    pub history: &'a [i64],
    pub text: &'a [u8],
}

pub fn oracle(
    op: EngramHash,
    hash: &HashTables,
    input: HashInput<'_>,
    ids: &mut [u32],
    status: &mut [u32],
) -> Result<(), Refusal> {
    let rows = u32::try_from(input.tokens.len())
        .map_err(|_| Refusal("Engram device hash rows exceed U32"))?;
    op.validate(Need::rows(rows))?;
    let engram = hash.engram();
    if hash.raw_vocab() != op.vocab
        || engram.max_ngram != op.ngram
        || engram.heads != op.heads
        || engram.layers.len() != op.layers as usize
        || input.token_batch.len() < rows as usize
        || input.text.len() < rows as usize
        || input.q_start.len() < op.items as usize
        || input.history.len() < op.items as usize * (op.ngram - 1) as usize
        || ids.len() as u64 * 4 < op.output_bytes()
        || status.len() < rows as usize
    {
        return Err(Refusal(
            "Engram device hash oracle has incompatible tables or spans",
        ));
    }
    let columns = op.columns() as usize;
    let lookback = (op.ngram - 1) as usize;
    for (row, flag) in status[..rows as usize].iter_mut().enumerate() {
        let run = || -> Result<Vec<u32>, Refusal> {
            let item = usize::try_from(input.token_batch[row])
                .ok()
                .filter(|&item| item < op.items as usize)
                .ok_or(Refusal("Engram device hash item is invalid"))?;
            let first = usize::try_from(input.q_start[item])
                .ok()
                .filter(|&first| first <= row)
                .ok_or(Refusal("Engram device hash start is invalid"))?;
            let mut recent = Vec::with_capacity(op.ngram as usize);
            for shift in 0..op.ngram as usize {
                let id = if shift <= row - first {
                    let at = row - shift;
                    if input.token_batch[at] != item as i32 || input.text[at] > 1 {
                        return Err(Refusal("Engram device hash row metadata is invalid"));
                    }
                    hash.compress(input.tokens[at], input.text[at] != 0)?
                } else {
                    let back = shift - (row - first) - 1;
                    input.history[item * lookback + lookback - 1 - back]
                };
                recent.push(id);
                // The host oracle stops at DEAD before considering any older ID.
                if id == DEAD {
                    break;
                }
            }
            let mut result = vec![0; op.layers as usize * columns];
            hash.position(&recent, &mut result)?;
            Ok(result)
        };
        let result = if *flag == 0 {
            run()
        } else {
            Err(Refusal("prior row failure"))
        };
        if result.is_err() {
            *flag |= 1;
        }
        for layer in 0..op.layers as usize {
            let to = (layer * op.stride as usize + row) * columns;
            match &result {
                Ok(result) => {
                    ids[to..to + columns].copy_from_slice(&result[layer * columns..][..columns])
                }
                Err(_) => ids[to..to + columns].fill(0),
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
