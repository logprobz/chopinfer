//! What an instance runs: a generated geometry
//! (`model_deepseek41_contract::geometry`) at a rank count. Per-rank sizes
//! follow from them: attention at `heads / ranks` heads and
//! `o_groups / ranks` output groups per rank, the routed and shared experts
//! split by intermediate column, the indexer, the window and compressed
//! keys and the router replicated, the embedding and head split by
//! vocabulary row (D3).

use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry, RuntimeValues};
use model_deepseek41_contract::serving::DECODE_BATCH;

/// One instance's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub geometry: &'static Geometry,
    pub ranks: Ranks,
    pub runtime: RuntimeValues,
    /// The most rows a decode or verify launch carries: the verify rows the
    /// host asks for (`--speculative-rows`), and never fewer than the widest
    /// decode bucket. Every op is prepared for this many decode-phase rows.
    pub verify_rows: u32,
    /// Whether verify steps above the 64-row decode route run the wide
    /// verify buckets (the `wide_verify` candidate option).
    pub wide_verify: bool,
    /// Whether Flash TP4 verify steps of 33 to 64 rows run the measured
    /// exact F32 row tiles (the `f32_row_tiles` candidate option).
    pub f32_row_tiles: bool,
    /// Whether decode and verify steps' complete all-reduces take the phased
    /// F32 rounding entries (the `phased_decode_reduce` candidate option).
    pub phased_decode_reduce: bool,
    /// Whether verify steps of at most `DECODE_BATCH` rows run the decode
    /// row fusions, and the drafter's small window writes the parallel
    /// decode ceil entry (the `verify_row_fusions` candidate option).
    pub verify_row_fusions: bool,
}

impl Shape {
    /// `geometry` at `ranks`, or the field that refuses the split.
    pub fn new(geometry: &'static Geometry, ranks: Ranks) -> Result<Shape, String> {
        check_ranks(geometry, ranks.get())?;
        Ok(Shape {
            geometry,
            ranks,
            runtime: geometry.runtime_defaults(),
            verify_rows: DECODE_BATCH,
            wide_verify: false,
            f32_row_tiles: false,
            phased_decode_reduce: false,
            verify_row_fusions: false,
        })
    }

    /// This shape serving `rows` verify rows a step: the instance's host
    /// limit, raised to the widest decode bucket.
    pub fn with_verify_rows(mut self, rows: u32) -> Shape {
        self.verify_rows = rows.max(DECODE_BATCH);
        self
    }

    /// This shape with the wide verify buckets on or off
    /// (`model_deepseek41_contract::execution_options::WIDE_VERIFY`).
    pub fn with_wide_verify(mut self, enabled: bool) -> Shape {
        self.wide_verify = enabled;
        self
    }

    /// This shape with the exact F32 row tiles of 33 to 64 verify rows on
    /// or off (`model_deepseek41_contract::execution_options::F32_ROW_TILES`).
    pub fn with_f32_row_tiles(mut self, enabled: bool) -> Shape {
        self.f32_row_tiles = enabled;
        self
    }

    /// This shape with the phased decode reduction on or off
    /// (`model_deepseek41_contract::execution_options::PHASED_DECODE_REDUCE`).
    pub fn with_phased_decode_reduce(mut self, enabled: bool) -> Shape {
        self.phased_decode_reduce = enabled;
        self
    }

    /// This shape with the decode row fusions at verify steps of at most
    /// `DECODE_BATCH` rows on or off
    /// (`model_deepseek41_contract::execution_options::VERIFY_ROW_FUSIONS`).
    pub fn with_verify_row_fusions(mut self, enabled: bool) -> Shape {
        self.verify_row_fusions = enabled;
        self
    }

    /// Whether `phase` runs the decode row fusions: the fused query norm
    /// and ceil and the fused window write, prepared for `DECODE_BATCH`
    /// rows of every phase. Decode steps run them; verify steps of at
    /// most `DECODE_BATCH` rows with the `verify_row_fusions` option.
    pub(crate) fn row_fusions(&self, phase: crate::program::Phase) -> bool {
        match phase {
            crate::program::Phase::Decode(_) => true,
            crate::program::Phase::Verify { rows, .. } => {
                self.verify_row_fusions && rows <= DECODE_BATCH
            }
            crate::program::Phase::Prefill { .. } => false,
        }
    }

    /// The prepared row bound of a Flash TP4 decode-route family whose
    /// widest verify step is `bound` (`model_deepseek41_contract::verify_bounds`):
    /// the bound capped by this instance's verify rows. Other topologies
    /// keep the widest decode bucket.
    pub(crate) fn verify_rows_within(&self, bound: u32) -> u32 {
        if self.geometry.id == model_deepseek41_contract::geometry::FLASH.id && self.ranks() == 4 {
            self.verify_rows.min(bound)
        } else {
            DECODE_BATCH
        }
    }

    /// The wide verify buckets this instance reaches
    /// (`model_deepseek41_contract::verify_bounds::WIDE`), each capped by
    /// its verify rows: none without the `wide_verify` option, at 64 verify
    /// rows or fewer, or outside Flash TP4.
    pub(crate) fn wide_verify_rows(&self) -> Vec<u32> {
        use model_deepseek41_contract::{serving::VERIFY_DECODE_ROWS, verify_bounds::WIDE};
        let mut buckets = Vec::new();
        let mut covered = self.verify_rows_within(VERIFY_DECODE_ROWS);
        if !self.wide_verify || covered < VERIFY_DECODE_ROWS {
            return buckets;
        }
        for bucket in WIDE {
            if self.verify_rows <= covered {
                break;
            }
            covered = self.verify_rows.min(bucket);
            buckets.push(covered);
        }
        buckets
    }

    /// The geometry's dimensions.
    pub const fn dims(&self) -> &'static model_deepseek41_contract::geometry::Shape {
        &self.geometry.shape
    }

    pub const fn ranks(&self) -> u32 {
        self.ranks.get()
    }

    /// Attention heads per rank.
    pub const fn local_heads(&self) -> u32 {
        self.dims().attention.heads / self.ranks.get()
    }

    /// `wo_a` groups per rank.
    pub const fn local_groups(&self) -> u32 {
        self.dims().attention.o_groups / self.ranks.get()
    }

    /// Routed and shared expert intermediate columns per rank.
    pub const fn local_inter(&self) -> u32 {
        self.dims().moe.intermediate / self.ranks.get()
    }

    /// Vocabulary rows per rank.
    pub const fn vocab_rows(&self) -> u32 {
        self.dims().vocab / self.ranks.get()
    }

    /// The topology ID of this shape: `deepseek41-<geometry>-tp<ranks>`.
    pub fn topology(&self) -> String {
        topology_id(self.geometry, self.ranks)
    }
}

/// The topology ID of `geometry` at `ranks`.
pub fn topology_id(geometry: &Geometry, ranks: Ranks) -> String {
    format!("deepseek41-{}-tp{}", geometry.id, ranks.get())
}

/// Every topology the build serves: each generated geometry at each rank
/// count it splits over.
pub fn topologies() -> Vec<Shape> {
    GEOMETRIES
        .iter()
        .flat_map(|geometry| {
            SUPPORTED_RANKS
                .iter()
                .filter_map(|&ranks| Shape::new(geometry, Ranks::new(ranks)?).ok())
        })
        .collect()
}

/// The shape a topology ID names.
pub fn of_topology(id: &str) -> Option<Shape> {
    topologies()
        .into_iter()
        .find(|shape| shape.topology() == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn flash_splits_evenly_at_four_ranks() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        assert_eq!(
            (
                shape.local_heads(),
                shape.local_groups(),
                shape.local_inter()
            ),
            (16, 2, 576)
        );
        assert_eq!(shape.vocab_rows(), 32_320);
        assert_eq!(
            shape.topology(),
            model_deepseek41_contract::FLASH_TP4_TOPOLOGY
        );
        assert_eq!(of_topology("deepseek41-flash-tp4"), Some(shape));
        assert_eq!(of_topology("deepseek41-flash-tp8"), None);
    }

    /// With the `wide_verify` option, Flash TP4 reaches a wide verify
    /// bucket only above 64 verify rows, each bucket capped by its verify
    /// rows; other topologies reach none, and without it no shape does.
    #[test]
    fn wide_verify_buckets_follow_the_verify_rows() {
        let flash = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for (rows, buckets) in [
            (32, &[][..]),
            (64, &[]),
            (65, &[65]),
            (128, &[128]),
            (129, &[128, 129]),
            (192, &[128, 192]),
        ] {
            let shape = flash.with_verify_rows(rows);
            assert_eq!(shape.with_wide_verify(true).wide_verify_rows(), buckets);
            assert!(shape.wide_verify_rows().is_empty());
        }
        for shape in topologies() {
            if shape != flash {
                let shape = shape.with_verify_rows(192).with_wide_verify(true);
                assert!(shape.wide_verify_rows().is_empty());
            }
        }
    }
}
