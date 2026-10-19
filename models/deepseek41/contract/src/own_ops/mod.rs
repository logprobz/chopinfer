//! DeepSeek V4.1's own kernels as contracts: the Engram gate, RoPE cosine
//! and sine rows, embedding row widening and DSpark heads, plus the Engram
//! hash the host computes.
//! No other model runs them, so they live here rather than in kernel-core
//! (an op moves there when a second model needs it).
//!
//! Each device op is a contract type whose fields are the math the model
//! needs, a role record of its buffers and a CPU oracle. A backend prepares
//! each op through [`OwnOps`] or refuses it; SM120 choices (threads, grids,
//! compiled constants) are its refusals, not contract rules.

use kernel_core::backend::{Need, Prepared, Refused, View};

pub mod dspark;
pub mod engram;
pub mod ring_gather;
pub mod rope;
pub mod widen;

use dspark::{ConfidenceHead, ConfidenceHeadBuffers, MarkovStep, MarkovStepBuffers};
use engram::{EngramGate, EngramGateBuffers};
use rope::{RopeCosSin, RopeCosSinBuffers};
use widen::{WidenRows, WidenRowsBuffers};

/// A backend of DeepSeek V4.1's own kernels. Each method prepares one op or
/// refuses it; every method refuses by default, so a backend implements the
/// ops it has kernels for. Each method's doc says which
/// [`kernel_core::backend::Counts`] its launches read.
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

    /// Copy committed ring slots into draft rows. Counts::rows is the item
    /// capacity; the device live_items count selects the items copied.
    fn dspark_ring_gather(
        &self,
        op: &ring_gather::RingGather,
        need: Need,
    ) -> Result<Prepared<ring_gather::RingGatherBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Correct one position of each DSpark block on a vocabulary shard.
    /// Launches read Counts::rows as the item capacity; the device live_rows
    /// count selects the items written. The predecessor embeddings have
    /// already been gathered across vocabulary shards.
    fn dspark_markov_step(
        &self,
        op: &MarkovStep,
        need: Need,
    ) -> Result<Prepared<MarkovStepBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Compute diagnostic confidence for one position of each DSpark block.
    /// Launches read Counts::rows and the device live_rows count as above.
    /// The scores never steer serving before a separate trimming capability.
    fn dspark_confidence(
        &self,
        op: &ConfidenceHead,
        need: Need,
    ) -> Result<Prepared<ConfidenceHeadBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// The Engram gate of one Engram layer.
    ///
    /// Launches read `Counts::rows`, the rows of the step (at most the
    /// prepared capacity), and record them; the device count of
    /// `live_rows` (at most `Counts::rows`) selects the rows written.
    fn engram_gate(
        &self,
        op: &EngramGate,
        need: Need,
    ) -> Result<Prepared<EngramGateBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// Cosine and sine rows of one RoPE table.
    ///
    /// Launches read `Counts::rows` and record them: for `RopeRows::Range`
    /// the rows written, for `RopeRows::Positions` the most rows, with the
    /// device count of `live_rows` selecting the rows written. Every
    /// position is below `rope::MAX_POSITIONS`.
    fn rope_cos_sin(
        &self,
        op: &RopeCosSin,
        need: Need,
    ) -> Result<Prepared<RopeCosSinBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }

    /// BF16 rows widened to F32: the embedding's partials before its
    /// all-reduce.
    ///
    /// Launches read `Counts::rows`, the rows of the step (at most the
    /// prepared capacity); the device count of `live_rows` selects the rows
    /// written.
    fn widen_rows(
        &self,
        op: &WidenRows,
        need: Need,
    ) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
        let _ = (op, need);
        Err(Refused::Unsupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rope::{RopeRows, RopeTable};

    struct Nothing;

    impl OwnOps for Nothing {
        fn name(&self) -> &str {
            "nothing"
        }
    }

    fn unsupported<T>(result: Result<T, Refused>) {
        assert!(matches!(result, Err(Refused::Unsupported)));
    }

    #[test]
    fn every_method_refuses_by_default() {
        let backend = Nothing;
        assert_eq!(backend.name(), "nothing");
        unsupported(backend.dspark_ring_gather(
            &ring_gather::RingGather {
                bytes: 64,
                source_stride: 64,
                destination_stride: 80,
                slots: 4,
            },
            Need::rows(4),
        ));
        unsupported(backend.dspark_markov_step(
            &MarkovStep {
                rank: 256,
                vocab_rows: 32320,
                logits_stride: 5 * 32320,
            },
            Need::rows(32),
        ));
        unsupported(backend.dspark_confidence(
            &ConfidenceHead {
                hidden: 5120,
                markov_rank: 256,
                hidden_stride: 5 * 5120,
                score_stride: 5,
            },
            Need::rows(32),
        ));
        unsupported(backend.engram_gate(
            &EngramGate::of(&crate::geometry::FLASH, true),
            Need::rows(32),
        ));
        unsupported(backend.rope_cos_sin(
            &RopeCosSin {
                table: RopeTable::Compressed,
                rows: RopeRows::Positions,
            },
            Need::rows(32),
        ));
        unsupported(backend.widen_rows(&widen::WidenRows { width: 4096 }, Need::rows(32)));
    }
}
