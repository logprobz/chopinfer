//! The hashed n-gram embedding over items (`cuda/row/qwen38_ple.cuh` decode
//! items, `cuda/qwen38_speculation.cuh` verify items and
//! `cuda/qwen38_ngram_hash.cuh` in the bundle): each item's rows hash their
//! token and the ids before it into table rows of this rank's heads, and,
//! for a gather form, read those rows' E4M3 codes times the scale.
//!
//! An item's n-gram history is two `u32` ids, the older first, `entry_words`
//! words after the previous item's; the kernels leave in the output history
//! the ids of the item's last two rows.

use kernel_core::codec::SmallFloat;
use kernel_core::contract::{NgramQuery, Refusal, Validate};
use kernel_core::ngram_oracle::{NgramTables, ngram_rows};

use super::{ItemForm, ItemRows, entry, item_rows};
use crate::contract::Shape;

/// What an n-gram launch writes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NgramWrites {
    /// Table row ids, embedding rows and the history.
    Gather,
    /// Table row ids and the history: a separate lookup gathers the rows.
    /// Decode items without a row this step get `table_start` as every
    /// head's id, a valid row for that lookup, and keep their history.
    Hash,
}

/// `NgramQuery::HashGather` over the items of one launch at rank `rank`:
/// global head `rank * local_heads + h` is this rank's head `h`. Heads
/// `0..heads_per_order` hash the token and one id before it, the others
/// the token and two; an id at or before an end-of-document id reads as
/// that id. Embedding rows are `[rows][local_heads][head_width]`, each code
/// decoded from E4M3 times the BF16 scale in F32, rounded to BF16. All math.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ngram {
    pub orders: u32,
    pub heads_per_order: u32,
    pub head_width: u32,
    pub local_heads: u32,
    pub rank: u32,
    pub eos: u32,
    /// Words from one item's history to the next one's.
    pub entry_words: u64,
    pub writes: NgramWrites,
    pub items: ItemForm,
}

impl Ngram {
    /// `shape`'s n-gram of rank `rank` in `items` form, writing `writes`.
    pub fn of(shape: Shape, rank: u32, writes: NgramWrites, items: ItemForm) -> Ngram {
        Ngram {
            orders: shape.ngram_size(),
            heads_per_order: shape.ngram_heads_per_order(),
            head_width: shape.ngram_head_width(),
            local_heads: shape.ngram_local_heads,
            rank,
            eos: shape.ngram_eos(),
            entry_words: shape.ple_entry_bytes() / 4,
            writes,
            items,
        }
    }

    /// Every rank's heads.
    pub fn heads(&self) -> u32 {
        (self.orders - 1) * self.heads_per_order
    }

    /// The query of `capacity` rows.
    pub fn query(&self, capacity: u32) -> NgramQuery {
        NgramQuery::HashGather {
            orders: self.orders,
            heads_per_order: self.heads_per_order,
            head_width: self.head_width,
            local_heads: self.local_heads,
            capacity,
        }
    }
}

impl Validate for Ngram {
    fn validate(&self) -> Result<(), Refusal> {
        if self.orders < 2
            || self.heads_per_order == 0
            || self.head_width == 0
            || self.local_heads == 0
            || !self.heads().is_multiple_of(self.local_heads)
            || self.rank >= self.heads() / self.local_heads
            || (self.items != ItemForm::One && self.entry_words < 2)
            || matches!(self.items, ItemForm::Verify { rows: 0 })
        {
            return Err(Refusal(
                "an n-gram op needs a rank's share of the heads and two history words",
            ));
        }
        Ok(())
    }
}

kernel_core::scratch_record! {
    /// An n-gram launch's buffers: the token ids, the item histories, the
    /// hash multipliers and every head's table size and offset (`u64`), the
    /// rank's E4M3 table shard and BF16 scale (gather forms), the embedding
    /// rows (gather forms), the `u64` row ids, the output histories, each
    /// item's row count, and for verify forms each item's first row and the
    /// device `u32` count of live items. The table shard holds global rows
    /// `[table_start, table_start + table_rows)`. Alignment by role, which
    /// the one-item form's views offset to its item keep: the `u32` tokens,
    /// histories, row counts, first rows and count at 4 bytes; the `u64`
    /// multipliers, sizes, offsets and row ids at 8; the BF16 scale and
    /// embedding rows at 2; the E4M3 table at 1.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub struct NgramBuffers {
        pub tokens: value,
        pub history: value,
        pub multipliers: value,
        pub sizes: value,
        pub offsets: value,
        pub table: option,
        pub scale: option,
        pub output: option,
        pub row_ids: value,
        pub history_out: value,
        pub cu_rows: option,
        pub q_len: value,
        pub live_items: option,
        pub table_start: scalar(u64),
        pub table_rows: scalar(u64),
    }
}

/// The table shard a rank reads: global rows `[start, start + rows)`, and
/// for gather forms their codes and scale.
#[derive(Clone, Copy, Debug)]
pub struct Shard<'a> {
    pub start: u64,
    pub rows: u64,
    pub codes: Option<&'a [u8]>,
    pub scale: u16,
}

/// One item's results.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramItem {
    /// `[rows][local_heads]` global table rows.
    pub row_ids: Vec<u64>,
    /// `[rows][local_heads][head_width]` BF16, gather forms only.
    pub output: Vec<u16>,
    /// The output history, the older id first.
    pub history: [u32; 2],
}

/// The BF16 the row kernels store for F32 `value` (`cvt.rn.bf16.f32`):
/// nearest, ties to even, NaN as the canonical `0x7fff`.
fn stored_bf16(value: f32) -> u16 {
    if value.is_nan() {
        0x7fff
    } else {
        kernel_core::codec::f32_to_bf16(value)
    }
}

/// One item's hashes (and gathered rows) of `tokens` after `history`. A
/// zero table size, a row outside the shard or a missing table is refused:
/// the kernels trap on the first two.
pub fn ngram_item(
    op: &Ngram,
    tables: &NgramTables,
    history: [u32; 2],
    tokens: &[u32],
    shard: Shard<'_>,
) -> Result<NgramItem, Refusal> {
    op.validate()?;
    if tables.sizes.len() != op.heads() as usize || tables.sizes.contains(&0) {
        return Err(Refusal("an n-gram head has no table rows"));
    }
    let heads = op.heads() as usize;
    let local = op.local_heads as usize;
    let first = op.rank as usize * local;
    let all = ngram_rows(
        tables,
        op.heads_per_order as usize,
        &history,
        tokens,
        op.eos,
    )?;
    let row_ids = all
        .rows
        .chunks_exact(heads)
        .flat_map(|row| row[first..first + local].iter().copied())
        .collect::<Vec<_>>();
    if row_ids
        .iter()
        .any(|&id| id < shard.start || id - shard.start >= shard.rows)
    {
        return Err(Refusal("an n-gram row is outside the rank's table shard"));
    }
    let output = match op.writes {
        NgramWrites::Hash => Vec::new(),
        NgramWrites::Gather => {
            let codes = shard
                .codes
                .ok_or(Refusal("an n-gram gather needs its table shard"))?;
            let scale = kernel_core::codec::bf16_to_f32(shard.scale);
            let width = op.head_width as usize;
            let mut output = Vec::with_capacity(row_ids.len() * width);
            for &id in &row_ids {
                let at = (id - shard.start) as usize * width;
                let row = codes
                    .get(at..at + width)
                    .ok_or(Refusal("an n-gram table shard is too small"))?;
                // E4M3 times a BF16 has at most 12 significant bits: the F32
                // product is exact before it rounds to BF16.
                output
                    .extend(row.iter().map(|&code| {
                        stored_bf16(SmallFloat::E4M3.decode_f64(code) as f32 * scale)
                    }));
            }
            output
        }
    };
    Ok(NgramItem {
        row_ids,
        output,
        history: all.history.try_into().expect("two history ids"),
    })
}

/// One launch's writes, item by item, over `items` grid items: each item
/// with rows and its [`NgramItem`]. `tokens` are the launch's, `histories`
/// its items' (`entry_words` apart; the one-item form's are the item's).
/// A hash decode launch's items without a row are in [`NgramLaunch::idle`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NgramLaunch {
    pub written: Vec<(ItemRows, NgramItem)>,
    pub idle: Vec<u32>,
}

#[allow(clippy::too_many_arguments)]
pub fn ngram_launch(
    op: &Ngram,
    tables: &NgramTables,
    tokens: &[u32],
    histories: &[u32],
    q_len: &[i32],
    cu_rows: Option<&[i32]>,
    live_items: Option<u32>,
    items: u32,
    shard: Shard<'_>,
) -> Result<NgramLaunch, Refusal> {
    op.validate()?;
    let written = item_rows(op.items, q_len, cu_rows, live_items, items)?
        .into_iter()
        .map(|rows| {
            let words = if op.items == ItemForm::One {
                0
            } else {
                u64::from(rows.item) * op.entry_words
            };
            let history = [
                entry(histories, words as usize, "an item history is missing")?,
                entry(histories, words as usize + 1, "an item history is missing")?,
            ];
            let first = rows.start as usize;
            let tokens = tokens
                .get(first..first + rows.rows as usize)
                .ok_or(Refusal("an item's tokens are outside the launch's"))?;
            Ok((rows, ngram_item(op, tables, history, tokens, shard)?))
        })
        .collect::<Result<Vec<_>, Refusal>>()?;
    let idle = if op.items == ItemForm::Decode && op.writes == NgramWrites::Hash {
        (0..items)
            .filter(|item| written.iter().all(|(rows, _)| rows.item != *item))
            .collect()
    } else {
        Vec::new()
    };
    Ok(NgramLaunch { written, idle })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables() -> NgramTables {
        kernel_core::ngram_oracle::ngram_tables(248_064, 3, 8, 17, 0, 1, 1).unwrap()
    }

    #[test]
    fn serving_ngrams_split_the_heads_over_the_ranks() {
        for shape in [Shape::TP2, Shape::TP4] {
            for rank in 0..shape.ranks() {
                let op = Ngram::of(shape, rank, NgramWrites::Gather, ItemForm::Decode);
                assert!(op.validate().is_ok());
                assert_eq!(op.heads(), 16);
                assert_eq!(op.local_heads * shape.ranks(), 16);
            }
            let past = Ngram::of(shape, shape.ranks(), NgramWrites::Gather, ItemForm::Decode);
            assert!(past.validate().is_err());
        }
    }

    /// The rank's heads of the kernel-core hashes, its codes times the scale
    /// and the history of the last two ids.
    #[test]
    fn items_hash_this_ranks_heads_and_gather_their_codes() {
        let tables = tables();
        let op = Ngram {
            local_heads: 4,
            rank: 1,
            ..Ngram::of(Shape::TP4, 1, NgramWrites::Gather, ItemForm::One)
        };
        let start = tables.offsets[4];
        let rows = tables.sizes[4..8].iter().sum::<u64>();
        let codes = (0..rows as usize * 160)
            .map(|i| (i % 127) as u8)
            .collect::<Vec<_>>();
        let shard = Shard {
            start,
            rows,
            codes: Some(&codes),
            scale: kernel_core::codec::f32_to_bf16(0.125),
        };
        let item = ngram_item(&op, &tables, [19, 23], &[100, 101], shard).unwrap();
        let all = ngram_rows(&tables, 8, &[19, 23], &[100, 101], op.eos).unwrap();
        assert_eq!(item.row_ids[..4], all.rows[4..8]);
        assert_eq!(item.row_ids[4..], all.rows[20..24]);
        assert_eq!(item.history, [100, 101]);
        let code = codes[(item.row_ids[1] - start) as usize * 160 + 7];
        assert_eq!(
            item.output[160 + 7],
            kernel_core::codec::f32_to_bf16(SmallFloat::E4M3.decode_f64(code) as f32 * 0.125)
        );
        // One row keeps the older id of the history.
        let one = ngram_item(&op, &tables, [19, 23], &[100], shard).unwrap();
        assert_eq!(one.history, [23, 100]);
        // A shard that misses a row is refused.
        assert!(ngram_item(&op, &tables, [19, 23], &[100], Shard { rows: 1, ..shard }).is_err());
    }

    #[test]
    fn hash_decode_launches_mark_items_without_a_row() {
        let tables = tables();
        let op = Ngram::of(Shape::TP4, 0, NgramWrites::Hash, ItemForm::Decode);
        let shard = Shard {
            start: 0,
            rows: tables.rows,
            codes: None,
            scale: 0,
        };
        let entry = op.entry_words as usize;
        let mut histories = vec![0; 3 * entry];
        histories[entry] = 248_044;
        histories[entry + 1] = 248_044;
        let launch = ngram_launch(
            &op,
            &tables,
            &[7, 8, 9],
            &histories,
            &[1, 1, 0],
            None,
            None,
            3,
            shard,
        )
        .unwrap();
        assert_eq!(launch.idle, [2]);
        assert_eq!(launch.written.len(), 2);
        assert_eq!(launch.written[1].1.history, [248_044, 8]);
        assert!(launch.written[0].1.output.is_empty());
    }
}
