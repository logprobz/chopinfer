//! A KDA layer's projections of its normed sublayer input in one launch
//! (`kernels_sm120::gemm::stacked`), which the decode and verify graphs run
//! in place of their six GEMMs and four layout copies: q, k and v write
//! their thirds of the convolution's `[rows, 3 * width]` input and beta its
//! rank's 16 of 64 padded columns, where the copies (`precision::copy`)
//! place them; forget-a and gate-a write their contiguous rows. Each part
//! keeps its own strip-chain arithmetic, so every output element is bitwise
//! the one-split tiled GEMM's, as each projection's own entry's (kernel-lab
//! family `strip_chains_stacked`: every live count of every bucket).
//!
//! Every bucket runs 4 strips at one split with the in-kernel L2 prefetch
//! of each CTA's later weight chunks: the lab's fastest stacked schedule at
//! every bucket it timed behind a collective and mixer window (cold L2, 1
//! to 32 rows: 29.9 to 33.8 us per layer against 74.8 to 112.9 us for
//! today's launches); buckets 2 and 8 take their neighbours' schedule.

use kernel_build::CubinSpec;
use kernel_core::contract::{DType, DenseKernel, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernels_sm120::gemm::stacked::{self, Part, Stacked};
use kernels_sm120::precision::Compute;

use super::precision;
use crate::contract::Shape;
use crate::precision::layout_formats;
use crate::quantization::LAYOUTS;
use crate::sm120::engines::PROFILE;

/// The lock family of [`CUBIN`] (`bundle/kernels.lock.d`).
pub const FAMILY: &str = "projections_stacked";
/// The stacked entries' cubin.
pub const CUBIN: &str = "projections_stacked_glm5next_sm120";

/// A KDA layer's projections of its normed input, in part order: the
/// order of their stages, q first.
pub const PROJECTIONS: [&str; 6] = [
    "linear.q",
    "linear.k",
    "linear.v",
    "linear.forget-a",
    "linear.beta",
    "linear.gate-a",
];

/// Weight strips per CTA, K splits and the L2 prefetch of every entry.
pub const STRIPS: u8 = 4;
pub const SPLITS: u8 = 1;
pub const PREFETCH: bool = true;

/// The most rows a stacked entry takes, as a strip-chain entry.
pub const MAX_ROWS: u32 = 32;

/// One part of a stacked entry: its projection, its weight rows and
/// placement, and the first column it writes in its output's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placed {
    pub id: &'static str,
    pub part: Part,
    pub column_offset: u32,
}

/// Projection `id`'s part at `usage`: its padded weight rows, placed
/// where its layout copy puts its contiguous output, or contiguous where it
/// has none.
fn place(id: &'static str, usage: OperationUse) -> Result<Placed, String> {
    let rows = precision::domain(id, WeightFormat::Bf16, usage)?
        .geometry
        .ok_or_else(|| format!("projection {id} has no geometry"))?
        .rows;
    let rows = u32::try_from(rows).map_err(|e| e.to_string())?;
    let Some(copy) = precision::copy(id, WeightFormat::Bf16, usage)? else {
        return Ok(Placed {
            id,
            part: Part::contiguous(rows),
            column_offset: 0,
        });
    };
    let (width, source, destination) = copy.layout();
    if (
        source.row_stride,
        source.column_stride,
        source.column_offset,
    ) != (rows, 1, 0)
        || destination.column_stride != 1
    {
        return Err(format!(
            "projection {id}'s copy reads or writes columns a stacked part does not"
        ));
    }
    Ok(Placed {
        id,
        part: Part {
            rows,
            width,
            row_stride: destination.row_stride,
        },
        column_offset: destination.column_offset,
    })
}

/// The parts of the stacked entry of `usage`, in [`PROJECTIONS`] order.
pub fn placed(usage: OperationUse) -> Result<Vec<Placed>, String> {
    PROJECTIONS.iter().map(|&id| place(id, usage)).collect()
}

/// The stacked entry serving `usage`, a decode or verify use of at most
/// [`MAX_ROWS`] rows. Refused unless each part computes what its
/// projection's own offer at the use does: the same GEMM over BF16 weights
/// under a kernel whose outputs are the one-split tiled GEMM's.
fn stacked_use(usage: OperationUse) -> Result<Stacked, String> {
    if usage.phase == ExecutionPhase::Prefill || usage.capacity > MAX_ROWS {
        return Err(format!("no stacked KDA input projections at {usage:?}"));
    }
    let placed = placed(usage)?;
    let entry = Stacked {
        symbol: format!(
            "chop_gemm_stacked_{}_kda_inputs_m{}_s{STRIPS}x{SPLITS}{}_v1",
            PROFILE.tag(),
            usage.capacity,
            if PREFETCH { "_pf" } else { "" }
        ),
        capacity: usage.capacity,
        k: 0,
        output: DType::Bf16,
        parts: placed.iter().map(|placed| placed.part).collect(),
        strips: STRIPS,
        splits: SPLITS,
        prefetch: PREFETCH,
    };
    let mut k = None;
    for (index, placed) in placed.iter().enumerate() {
        let generated = precision::offer(placed.id, WeightFormat::Bf16, usage)?;
        let Compute::Dense(spec) = &generated.offer.handle().compute else {
            return Err(format!("{} is not a dense GEMM", placed.id));
        };
        let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
            return Err(format!("{} has no GEMM query", placed.id));
        };
        if !matches!(
            spec.key.dense,
            Some(
                DenseKernel::Tiled { splits: 1 }
                    | DenseKernel::TiledChains { .. }
                    | DenseKernel::StripChains { .. }
            )
        ) {
            return Err(format!(
                "{} at {usage:?} runs {:?}, not the one-split tiled GEMM's bits",
                placed.id, spec.key.dense
            ));
        }
        if *k.get_or_insert(query.k) != query.k {
            return Err("the stacked KDA input projections differ in K".into());
        }
        let part = Stacked {
            k: query.k,
            ..entry.clone()
        }
        .part_key(index)
        .map_err(|r| format!("{}: {r}", placed.id))?;
        if part.query != spec.key.query {
            return Err(format!(
                "{}'s stacked part computes another GEMM than its offer at {usage:?}",
                placed.id
            ));
        }
    }
    Ok(Stacked {
        k: k.unwrap_or(0),
        ..entry
    })
}

/// A projection's part in the stacked entry that runs its use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StackedPart {
    pub entry: Stacked,
    /// The part's index in `entry` ([`PROJECTIONS`] order).
    pub index: usize,
    pub placed: Placed,
}

/// The stacked entry and part that run projection `id` over `weight` at
/// `usage`, if one does: a KDA input projection over BF16 weights at a
/// decode or verify use of at most [`MAX_ROWS`] rows. The precision
/// component binds such a use to the entry, in place of its GEMM and copy.
pub fn part(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Option<StackedPart>, String> {
    use std::collections::BTreeMap;
    use std::sync::{Mutex, OnceLock};
    type Parts = Result<(Stacked, Vec<Placed>), String>;
    static USES: OnceLock<Mutex<BTreeMap<String, Parts>>> = OnceLock::new();
    let Some(index) = PROJECTIONS.iter().position(|&stacked| stacked == id) else {
        return Ok(None);
    };
    if weight != WeightFormat::Bf16
        || usage.phase == ExecutionPhase::Prefill
        || usage.capacity > MAX_ROWS
    {
        return Ok(None);
    }
    let mut uses = USES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|e| e.to_string())?;
    let (entry, placed) = uses
        .entry(format!("{usage:?}"))
        .or_insert_with(|| Ok((stacked_use(usage)?, placed(usage)?)))
        .clone()?;
    Ok(Some(StackedPart {
        entry,
        index,
        placed: placed[index],
    }))
}

/// The stacked entries, one per decode and verify bucket of at most
/// [`MAX_ROWS`] rows (a decode and a verify use of one capacity share
/// it), in capacity order. Refused unless every checkpoint layout stores
/// the parts' weights in BF16.
pub fn entries(shape: Shape) -> Result<Vec<Stacked>, String> {
    for layout in LAYOUTS {
        for (id, weight) in layout_formats(shape, layout)? {
            if PROJECTIONS.contains(&id) && weight != WeightFormat::Bf16 {
                return Err(format!(
                    "the {} layout stores {id} in {weight:?}, not the stacked BF16",
                    layout.id
                ));
            }
        }
    }
    let mut entries: std::collections::BTreeMap<u32, Stacked> = Default::default();
    for (usage, _) in crate::precision::uses(PROJECTIONS[0])? {
        if usage.phase == ExecutionPhase::Prefill || usage.capacity > MAX_ROWS {
            continue;
        }
        let entry = stacked_use(usage)?;
        if entries
            .get(&usage.capacity)
            .is_some_and(|other| *other != entry)
        {
            return Err(format!(
                "the stacked decode and verify entries of {} rows differ",
                usage.capacity
            ));
        }
        entries.insert(usage.capacity, entry);
    }
    Ok(entries.into_values().collect())
}

/// [`CUBIN`]: the stacked entries.
pub fn cubin_specs(shape: Shape) -> Result<Vec<CubinSpec>, String> {
    Ok(vec![
        stacked::cubin_spec(CUBIN, &entries(shape)?).map_err(|r| r.0.to_string())?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Six buckets, each stacking q, k and v into their thirds of the
    /// `[rows, 6144]` convolution input, forget-a and gate-a contiguous and
    /// beta's 16 rank columns of its 64 padded rows, at 4 strips, one split,
    /// prefetching: 101 CTAs.
    #[test]
    fn every_decode_and_verify_bucket_stacks_the_kda_inputs() {
        let entries = entries(Shape::TP4).unwrap();
        let capacities: Vec<u32> = entries.iter().map(|e| e.capacity).collect();
        assert_eq!(capacities, [1, 2, 4, 8, 16, 32]);
        let third = Part {
            rows: 2048,
            width: 2048,
            row_stride: 6144,
        };
        let beta = Part {
            rows: 64,
            width: 16,
            row_stride: 16,
        };
        for entry in &entries {
            assert_eq!(
                entry.parts,
                [
                    third,
                    third,
                    third,
                    Part::contiguous(128),
                    beta,
                    Part::contiguous(128)
                ]
            );
            assert_eq!(
                (entry.k, entry.output, entry.strips, entry.splits),
                (4096, DType::Bf16, 4, 1)
            );
            assert!(entry.prefetch);
            assert_eq!(
                entry.symbol,
                format!(
                    "chop_gemm_stacked_glm5next_kda_inputs_m{}_s4x1_pf_v1",
                    entry.capacity
                )
            );
            let plan = entry.plan(entry.capacity).unwrap();
            assert_eq!(plan.grid, [3 * 32 + 2 + 1 + 2, 1, 1]);
            assert_eq!(entry.semaphore_bytes(), 0);
        }
        // The thirds' columns.
        for (usage, _) in crate::precision::uses("linear.q").unwrap() {
            if usage.phase == ExecutionPhase::Prefill {
                continue;
            }
            let offsets: Vec<u32> = placed(usage)
                .unwrap()
                .iter()
                .map(|p| p.column_offset)
                .collect();
            assert_eq!(offsets, [0, 2048, 4096, 0, 0, 0]);
        }
        let cubins = cubin_specs(Shape::TP4).unwrap();
        assert_eq!(cubins.len(), 1);
        assert_eq!(cubins[0].name, CUBIN);
        assert_eq!(cubins[0].entries.len(), 6);
        assert!(
            cubins[0].entries[0]
                .body
                .contains("run_a16_strips_stacked_ordered<6, 4, ")
        );
        assert!(cubins[0].entries[0].body.contains(", true>(parts, map_b,"));
    }

    /// Prefill and buckets above 32 rows have no stacked entry.
    #[test]
    fn prefill_and_wide_verify_keep_their_launches() {
        for (usage, _) in crate::precision::uses("linear.q").unwrap() {
            let stacked = stacked_use(usage);
            assert_eq!(
                stacked.is_ok(),
                usage.phase != ExecutionPhase::Prefill && usage.capacity <= MAX_ROWS,
                "{usage:?}"
            );
        }
    }
}
