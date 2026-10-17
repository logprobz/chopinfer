//! DeepSeek V4.1's own kernels as contracts: the Engram gate and the RoPE
//! cosine and sine rows, plus the Engram hash the host computes. No other
//! model runs them, so they live here rather than in kernel-core (an op
//! moves there when a second model needs it).
//!
//! Each device op is a contract type whose fields are the math the model
//! needs, a role record of its buffers and a CPU oracle. A backend prepares
//! each op through [`OwnOps`] or refuses it; SM120 choices (threads, grids,
//! compiled constants) are its refusals, not contract rules.

use kernel_core::backend::{Need, Prepared, Refused, View};

pub mod engram;
pub mod rope;

use engram::{EngramGate, EngramGateBuffers};
use rope::{RopeCosSin, RopeCosSinBuffers};

/// A backend of DeepSeek V4.1's own kernels. Each method prepares one op or
/// refuses it; every method refuses by default, so a backend implements the
/// ops it has kernels for. Each method's doc says which
/// [`kernel_core::backend::Counts`] its launches read.
pub trait OwnOps: Send + Sync {
    /// The backend's name in refusals and reports, such as `sm120`.
    fn name(&self) -> &str;

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
        unsupported(backend.engram_gate(&EngramGate::FLASH, Need::rows(32)));
        unsupported(backend.rope_cos_sin(
            &RopeCosSin {
                table: RopeTable::Compressed,
                rows: RopeRows::Positions,
            },
            Need::rows(32),
        ));
    }
}
