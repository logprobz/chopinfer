//! Cases of the context-parallel ops one group rank runs alone: the
//! rank-ordered merge of a group's attention partials and the localization
//! of key lists to a group rank, with the SM120 selection of the lab's
//! context-parallel entries (`families::context_parallel`), at group sizes 2
//! and 4.
//!
//! Key localization is exact and compares as bits. The partial merge
//! compares within tolerance against the contract's merge in F64 (the
//! device's `exp2` is within its error bound of the exact weight, so only
//! ranks of equal maxima would merge bit for bit); the partials hold equal
//! and unequal maxima, ranks with no keys and a pair no rank holds keys of,
//! in the packed segment layout and in a strided one (rank rows past the
//! capacity, unread heads around a nonzero head offset), whose unread pairs
//! dominate any merge that reads them.

use super::super::{OpCase, Required, Spec, bf16_values};
use crate::LabResult;
use crate::conformance::{Expected, Images, Rows};
use kernel_core::backend::{Need, Prepared, View};
use kernel_core::codec::bf16_to_f32;
use kernel_core::context_parallel::{
    ContextParallelQuery, LocalizeKeysBuffers, PartialMergeBuffers,
};
use kernel_core::context_parallel_oracle::localize_keys;
use kernel_core::contract::DType;
use kernel_core::oracle::OracleOutput;
use kernels_sm120::backend::Sm120;

/// The token rows every context-parallel case serves.
pub const CAPACITY: u32 = 16;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The segment layout of a group's partials (`PartialMergeBuffers`): each
/// rank's segment holds `rank_rows` rows of `input_heads` pairs, and row
/// `r`'s head `h` is pair `r * input_heads + head_offset + h`.
#[derive(Clone, Copy, Debug)]
pub struct MergeLayout {
    pub rank_rows: usize,
    pub input_heads: usize,
    pub head_offset: usize,
}

impl MergeLayout {
    /// The packed layout of `rows` rows of `heads` heads: no other heads,
    /// no offset.
    pub fn packed(rows: usize, heads: usize) -> Self {
        Self {
            rank_rows: rows,
            input_heads: heads,
            head_offset: 0,
        }
    }

    /// Row `row`'s head `head`'s pair in a segment.
    fn pair(&self, row: usize, head: usize) -> usize {
        row * self.input_heads + self.head_offset + head
    }

    fn pairs(&self) -> usize {
        self.rank_rows * self.input_heads
    }
}

/// The maximum of every pair no row's head reads: far above every pair it
/// reads, so a merge that reads one is dominated by it.
const UNREAD_MAXIMUM: f32 = 40.0;

/// A group's partials of `rows` rows of `heads` heads of `width` in
/// `layout` (per rank: `m`, then `l`, then `o`). The `n`-th pair a row reads
/// on rank r has no keys when `(n + r) % 5 == 0`, pair 1 has none on any
/// rank, and every third pair has one maximum on every rank that holds keys;
/// every pair no row reads (other heads, rows past `rows`) holds
/// [`UNREAD_MAXIMUM`] and large values.
fn partials(
    ranks: usize,
    layout: MergeLayout,
    rows: usize,
    heads: usize,
    width: usize,
) -> Vec<f32> {
    let pairs = layout.pairs();
    // The read index of each pair, if a row reads it.
    let mut read = vec![None; pairs];
    for row in 0..rows {
        for head in 0..heads {
            read[layout.pair(row, head)] = Some(row * heads + head);
        }
    }
    let mut out = Vec::with_capacity(ranks * pairs * (width + 2));
    for rank in 0..ranks {
        let drawn = bf16_values(61 + rank as u64, pairs * (width + 2))
            .into_iter()
            .map(bf16_to_f32)
            .collect::<Vec<_>>();
        let empty = |n: usize| (n + rank) % 5 == 0 || n == 1;
        let maximum = read.iter().enumerate().map(|(pair, read)| match *read {
            None => UNREAD_MAXIMUM,
            Some(n) if empty(n) => f32::NEG_INFINITY,
            Some(n) if n % 3 == 0 => 0.5,
            Some(_) => drawn[pair],
        });
        let denominator = read.iter().enumerate().map(|(pair, read)| match *read {
            Some(n) if empty(n) => 0.0,
            _ => drawn[pairs + pair].abs() * 2.0,
        });
        out.extend(maximum);
        out.extend(denominator);
        for (pair, read) in read.iter().enumerate() {
            for col in 0..width {
                let value = drawn[2 * pairs + (pair * width + col) % (pairs * width)];
                out.push(match *read {
                    None => value * 1024.0,
                    Some(n) if empty(n) => 0.0,
                    Some(_) => value,
                });
            }
        }
    }
    out
}

/// The merge in F64 of the pairs `rows` rows of `heads` heads read in
/// [`partials`]' layout: per pair `M = max m_r`, `a_r = exp(m_r - M)`,
/// `O / L` with S the weighted numerators' magnitude over `L` and K the
/// ranks' terms (zeros, with no error, for a pair no rank holds keys of).
fn merge(
    partials: &[f32],
    ranks: usize,
    layout: MergeLayout,
    (rows, heads, width): (usize, usize, usize),
) -> LabResult<OracleOutput> {
    let pairs = layout.pairs();
    let segment = pairs * (width + 2);
    let mut values = Vec::with_capacity(rows * heads * width);
    let mut bounds = Vec::with_capacity(rows * heads * width);
    for row in 0..rows {
        for head in 0..heads {
            let pair = layout.pair(row, head);
            let at = |rank: usize| &partials[rank * segment..][..segment];
            let maximum = (0..ranks)
                .map(|rank| f64::from(at(rank)[pair]))
                .fold(f64::NEG_INFINITY, f64::max);
            if maximum == f64::NEG_INFINITY {
                values.extend(std::iter::repeat_n(0.0, width));
                bounds.extend(std::iter::repeat_n(0.0, width));
                continue;
            }
            let weights = (0..ranks)
                .map(|rank| (f64::from(at(rank)[pair]) - maximum).exp())
                .collect::<Vec<_>>();
            let total = (0..ranks)
                .map(|rank| weights[rank] * f64::from(at(rank)[pairs + pair]))
                .sum::<f64>();
            for col in 0..width {
                let numerator = |rank: usize| {
                    f64::from(at(rank)[2 * pairs + pair * width + col]) * weights[rank]
                };
                values.push((0..ranks).map(numerator).sum::<f64>() / total);
                bounds.push((0..ranks).map(|rank| numerator(rank).abs()).sum::<f64>() / total);
            }
        }
    }
    let mut output = OracleOutput::new(values, bounds, 1).map_err(|e| e.0.to_string())?;
    // Each rank's weight, its two FMAs and the division.
    output.reduction_lengths = vec![4 * ranks; output.values.len()];
    Ok(output)
}

/// The case of a partial merge over `capacity` rows in `layout`: the
/// group's F32 partials ([`partials`]) as one buffer, and the merge's BF16
/// rows within tolerance of the F64 merge.
pub fn partial_merge_case<'a>(
    op: ContextParallelQuery,
    layout: MergeLayout,
    capacity: u32,
    prepared: Prepared<PartialMergeBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let ContextParallelQuery::PartialMerge {
        ranks,
        heads,
        width,
        ..
    } = op
    else {
        return Err("a partial merge case needs a partial merge".into());
    };
    let (ranks, heads, width, rows) = (
        ranks as usize,
        heads as usize,
        width as usize,
        capacity as usize,
    );
    if layout.rank_rows < rows || layout.head_offset + heads > layout.input_heads {
        return Err("the merge layout does not hold every row's heads".into());
    }
    let partials = partials(ranks, layout, rows, heads, width);
    let roles = PartialMergeBuffers {
        partials: Spec::input(
            DType::F32,
            (Rows::Fixed(1), partials.len()),
            f32_bytes(&partials),
        ),
        rank_rows: layout.rank_rows as u32,
        input_heads: layout.input_heads as u32,
        head_offset: layout.head_offset as u32,
        live_rows: Spec::live(),
        output: Spec::output(DType::Bf16, (Rows::Live, heads * width)),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let output = merge(
            &images.f32s("partials")?,
            ranks,
            layout,
            (rows, heads, width),
        )?;
        Ok(vec![("output".into(), Expected::Values(output))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// `count` positions below `span`, spread over its pages.
fn key_lists(count: usize, span: usize) -> Vec<i32> {
    (0..count)
        .map(|at| ((at * 37 + at / 7) % span) as i32)
        .collect()
}

/// The case of a key localization to `rank` over `capacity` rows: lists of
/// positions over eight pages per rank, and the contract's local lists as
/// exact bits. The lists hold no -1 holes: the PDL copy producer rewrites
/// the lists, and a hole's bytes are the poison pattern it must tell
/// apart, so a hole's localization is not exercised (each unowned position
/// still localizes to -1).
pub fn localize_keys_case<'a>(
    op: ContextParallelQuery,
    rank: u32,
    capacity: u32,
    prepared: Prepared<LocalizeKeysBuffers<View>>,
) -> LabResult<OpCase<'a>> {
    let ContextParallelQuery::LocalizeKeys {
        ranks, page, width, ..
    } = op
    else {
        return Err("a localization case needs a key localization".into());
    };
    let lists = key_lists(
        capacity as usize * width as usize,
        (ranks * page * 8) as usize,
    );
    let roles = LocalizeKeysBuffers {
        lists: Spec::input(DType::I32, (Rows::Live, width as usize), i32_bytes(&lists)),
        live_rows: Spec::live(),
        // Unowned positions localize to -1, all ones, which the default
        // fill would read as unwritten.
        output: Spec::output(DType::I32, (Rows::Live, width as usize)).filled_with(&[0x80]),
    };
    let oracle = move |images: &Images, _: u32| -> LabResult<Vec<(String, Expected)>> {
        let local = localize_keys(op, rank, &images.i32s("lists")?).map_err(|e| e.0.to_string())?;
        Ok(vec![("output".into(), Expected::Bits(i32_bytes(&local)))])
    };
    Ok(OpCase::new(prepared, capacity, roles)?.oracle(oracle))
}

/// SM120 over the context-parallel `entries`.
pub fn sm120(entries: Vec<kernels_sm120::context_parallel::Entry>) -> Sm120 {
    Sm120::new().with_context_parallel(entries)
}

/// Every context-parallel entry's op at [`CAPACITY`] rows, a localization
/// to the group's last rank, each required.
pub fn required<'a>(entries: &[kernels_sm120::context_parallel::Entry]) -> Vec<Required<'a>> {
    let need = Need::rows(CAPACITY);
    let mut ops = Vec::new();
    for entry in entries {
        let op = entry.query();
        let required = match op {
            ContextParallelQuery::PartialMerge {
                ranks,
                heads,
                width,
                ..
            } => {
                // Packed, and strided: rank rows past the capacity, unread
                // heads on both sides of an owner offset.
                let rows = CAPACITY as usize;
                let strided = MergeLayout {
                    rank_rows: rows + 3,
                    input_heads: heads as usize + 3,
                    head_offset: 2,
                };
                for (form, layout) in [
                    ("packed", MergeLayout::packed(rows, heads as usize)),
                    ("strided", strided),
                ] {
                    ops.push(
                        Required::new(
                            format!(
                                "partial merge {ranks} ranks {heads} heads width {width} {form} \
                                 {layout:?}"
                            ),
                            need,
                            move |backend, need| backend.partial_merge(&op, need),
                            move |prepared| partial_merge_case(op, layout, need.capacity, prepared),
                        )
                        .required(),
                    );
                }
                continue;
            }
            ContextParallelQuery::LocalizeKeys {
                ranks, page, width, ..
            } => {
                let rank = ranks - 1;
                Required::new(
                    format!("localize keys to rank {rank} of {ranks} page {page} width {width}"),
                    need,
                    move |backend, need| backend.localize_keys(&op, rank, need),
                    move |prepared| localize_keys_case(op, rank, need.capacity, prepared),
                )
            }
            _ => continue,
        };
        ops.push(required.required());
    }
    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A merge that ignores the strided layout (rank rows, input heads or
    /// the head offset) reads other pairs, which differ from the case's.
    #[test]
    fn strided_merges_differ_from_any_misread_layout() {
        let (ranks, rows, heads, width) = (2, 4, 3, 8);
        let strided = MergeLayout {
            rank_rows: rows + 3,
            input_heads: heads + 3,
            head_offset: 2,
        };
        let partials = partials(ranks, strided, rows, heads, width);
        let truth = merge(&partials, ranks, strided, (rows, heads, width)).unwrap();
        for wrong in [
            MergeLayout {
                rank_rows: rows,
                ..strided
            },
            MergeLayout {
                input_heads: heads,
                head_offset: 0,
                ..strided
            },
            MergeLayout {
                head_offset: 0,
                ..strided
            },
        ] {
            // A misread segment stride or pair index lands elsewhere.
            let misread = merge(&partials, ranks, wrong, (rows, heads, width)).unwrap();
            assert_ne!(misread.values, truth.values, "{wrong:?}");
        }
    }

    /// The lists localize differently to each group rank, so a localization
    /// that ignores its rank fails the case's bits.
    #[test]
    fn localization_follows_the_rank() {
        let op = ContextParallelQuery::LocalizeKeys {
            ranks: 4,
            page: 64,
            width: 8,
            capacity: 16,
        };
        let lists = key_lists(16 * 8, 4 * 64 * 8);
        let local = |rank| localize_keys(op, rank, &lists).unwrap();
        let mut seen = (0..4).map(local).collect::<Vec<_>>();
        seen.sort();
        seen.dedup();
        assert_eq!(seen.len(), 4);
        // Every rank owns some of the positions.
        assert!((0..4).all(|rank| local(rank).iter().any(|&p| p >= 0)));
    }
}
