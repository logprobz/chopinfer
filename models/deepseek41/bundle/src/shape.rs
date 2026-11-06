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
    /// Whether Flash TP4 prefill projections run the compact prefill entry
    /// at the row counts the candidate table admits (the
    /// `prefill_compact_dense_rows` candidate option).
    pub prefill_compact_dense_rows: bool,
    /// Whether drafter steps of more than twelve drafting items run them in
    /// the fewest forwards the widest wide verify bucket holds, on the wide
    /// verify buckets (the `wide_draft` candidate option).
    pub wide_draft: bool,
    /// Whether decode and verify steps' complete all-reduces take the phased
    /// F32 rounding entries (the `phased_decode_reduce` candidate option).
    pub phased_decode_reduce: bool,
    /// Whether drafter Markov steps of two or more items run the staged
    /// step over each position's base logits in place (the
    /// `dspark_markov_staged` candidate option).
    pub dspark_markov_staged: bool,
    /// Whether a drafter stage's sublayer boundaries, after its attention
    /// and into the next stage, fuse the round, expansion, next mixer and
    /// input norm at admitted rows (the `dspark_hc_fusion` candidate
    /// option).
    pub dspark_hc_fusion: bool,
    /// Whether drafter confidence heads stage their weight chains before
    /// the dependency wait and their row operands in shared memory (the
    /// `dspark_confidence_staged` candidate option).
    pub dspark_confidence_staged: bool,
    /// Whether drafter forwards run the routed experts through the direct
    /// path at the rows the drafter table admits (the `dspark_moe_direct`
    /// candidate option).
    pub dspark_moe_direct: bool,
    /// Whether drafter forwards of one to six items admit the HC finish
    /// route through the drafter table (the `dspark_hc_draft_rows`
    /// candidate option, which takes effect with `dspark_hc_fusion`).
    pub dspark_hc_draft_rows: bool,
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
            prefill_compact_dense_rows: false,
            wide_draft: false,
            phased_decode_reduce: false,
            dspark_hc_draft_rows: false,
            dspark_moe_direct: false,
            dspark_confidence_staged: false,
            dspark_hc_fusion: false,
            dspark_markov_staged: false,
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

    /// This shape with the compact prefill entry's candidate row ranges on
    /// or off
    /// (`model_deepseek41_contract::execution_options::PREFILL_COMPACT_DENSE_ROWS`).
    pub fn with_prefill_compact_dense_rows(mut self, enabled: bool) -> Shape {
        self.prefill_compact_dense_rows = enabled;
        self
    }

    /// This shape with the wide drafter forwards on or off
    /// (`model_deepseek41_contract::execution_options::WIDE_DRAFT`).
    pub fn with_wide_draft(mut self, enabled: bool) -> Shape {
        self.wide_draft = enabled;
        self
    }

    /// The most drafting items one wide drafter forward runs: the widest
    /// wide verify bucket's rows over the five block rows of an item, at
    /// most the decode batch. `None` without the `wide_draft` option or a
    /// wide verify bucket.
    pub(crate) fn wide_draft_items(&self) -> Option<u32> {
        use model_deepseek41_contract::serving::MAX_DRAFT_TOKENS;
        let widest = self.wide_verify_rows().last().copied()?;
        self.wide_draft
            .then(|| (widest / MAX_DRAFT_TOKENS).min(DECODE_BATCH))
    }

    /// This shape with the phased decode reduction on or off
    /// (`model_deepseek41_contract::execution_options::PHASED_DECODE_REDUCE`).
    pub fn with_phased_decode_reduce(mut self, enabled: bool) -> Shape {
        self.phased_decode_reduce = enabled;
        self
    }

    /// This shape with the staged drafter Markov step on or off
    /// (`model_deepseek41_contract::execution_options::DSPARK_MARKOV_STAGED`).
    pub fn with_dspark_markov_staged(mut self, enabled: bool) -> Shape {
        self.dspark_markov_staged = enabled;
        self
    }

    /// This shape with the `dspark_hc_fusion` candidate on or off
    /// (`model_deepseek41_contract::execution_options::DSPARK_HC_FUSION`).
    pub fn with_dspark_hc_fusion(mut self, enabled: bool) -> Shape {
        self.dspark_hc_fusion = enabled;
        self
    }

    /// This shape with the `dspark_confidence_staged` candidate on or off
    /// (`model_deepseek41_contract::execution_options::DSPARK_CONFIDENCE_STAGED`).
    pub fn with_dspark_confidence_staged(mut self, enabled: bool) -> Shape {
        self.dspark_confidence_staged = enabled;
        self
    }

    /// This shape with the `dspark_moe_direct` candidate on or off
    /// (`model_deepseek41_contract::execution_options::DSPARK_MOE_DIRECT`).
    pub fn with_dspark_moe_direct(mut self, enabled: bool) -> Shape {
        self.dspark_moe_direct = enabled;
        self
    }

    /// This shape with the `dspark_hc_draft_rows` candidate on or off
    /// (`model_deepseek41_contract::execution_options::DSPARK_HC_DRAFT_ROWS`).
    pub fn with_dspark_hc_draft_rows(mut self, enabled: bool) -> Shape {
        self.dspark_hc_draft_rows = enabled;
        self
    }

    /// Whether `phase` runs the decode row fusions: the fused query norm
    /// and ceil and the fused window write, prepared for `DECODE_BATCH`
    /// rows of every phase. Decode steps run them, and verify steps of the
    /// rows `latent.row_fusions.verify` admits.
    pub(crate) fn row_fusions(&self, phase: crate::program::Phase) -> bool {
        match phase {
            crate::program::Phase::Decode(_) => true,
            crate::program::Phase::Verify { rows, .. } => self.row_fusion_rows(rows),
            crate::program::Phase::Prefill { .. } => false,
        }
    }

    /// Whether the drafter prepares its small window writes on the
    /// parallel decode ceil entry, for the writes `Self::row_fusion_rows`
    /// admits: with a `latent.row_fusions.verify` row.
    pub(crate) fn fuses_verify_rows(&self) -> bool {
        (1..=DECODE_BATCH).any(|rows| self.row_fusion_rows(rows))
    }

    /// Whether `latent.row_fusions.verify` admits `rows` rows: verify steps
    /// of that many rows run the decode row fusions, and the drafter's
    /// window writes of that many rows quantize with the parallel decode
    /// ceil entry. Its bucket is `DECODE_BATCH`, the rows the fused entries
    /// are prepared for.
    pub(crate) fn row_fusion_rows(&self, rows: u32) -> bool {
        use model_deepseek41_contract::tuning::{Family, select};
        select(
            self.geometry,
            Family::Latent,
            "latent.row_fusions.verify",
            "none",
            &[("ranks", self.ranks())],
            DECODE_BATCH,
            rows,
        )
        .is_some()
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

    /// A wide drafter forward holds the items whose five block rows fit
    /// the widest wide verify bucket, at most the decode batch, and only
    /// with both options.
    #[test]
    fn wide_drafter_forwards_fit_the_widest_bucket() {
        let flash = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for (rows, items) in [
            (64, None),
            (65, Some(13)),
            (80, Some(16)),
            (128, Some(25)),
            (160, Some(32)),
            (192, Some(32)),
        ] {
            let shape = flash.with_verify_rows(rows).with_wide_verify(true);
            assert_eq!(shape.wide_draft_items(), None);
            assert_eq!(
                shape.with_wide_draft(true).wide_draft_items(),
                items,
                "{rows}"
            );
            assert_eq!(
                flash
                    .with_verify_rows(rows)
                    .with_wide_draft(true)
                    .wide_draft_items(),
                None
            );
        }
    }
}
