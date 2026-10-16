//! The environment switches an instance reads once, at create. Lowering,
//! acquisition and the step driver read these values, never the
//! environment, so a step's launches depend only on what create saw.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Switches {
    /// Decode graphs above 32 rows may take the wide all-reduce, unless
    /// `launch::WIDE_COMM` is `0`. The residency check at create may still
    /// keep them on the phased pair (`Instance::wide_comm`).
    pub wide_comm: bool,
    /// Row-owned decode and verify logits (`logits::ROW_OWNED_LOGITS`).
    pub row_owned_logits: bool,
    /// The exact HC producer bottleneck where the geometry qualifies it
    /// (`hc::SWITCH`).
    pub hc_precompute: bool,
    /// The padded precomputed mix layout (`CHOPINFER_QWEN38_HC_PADDED_MIX`).
    pub hc_padded_mix: bool,
    /// Every drafter run eager (`CHOPINFER_QWEN38_MTP_EAGER=1`).
    pub mtp_eager: bool,
    /// Diagnostic-taps builds write each prefill's overlap split
    /// (`CHOPINFER_QWEN38_OVERLAP_EVIDENCE=1`).
    pub overlap_evidence: bool,
}

impl Switches {
    /// The switches of an instance of `shape`.
    pub(crate) fn read(shape: Shape) -> Switches {
        Switches {
            wide_comm: launch::wide_comm_enabled(std::env::var(launch::WIDE_COMM).ok().as_deref()),
            row_owned_logits: logits::enabled(),
            hc_precompute: hc_available(shape) && hc::enabled(),
            hc_padded_mix: hc_available(shape) && hc::padded_enabled(),
            mtp_eager: std::env::var_os("CHOPINFER_QWEN38_MTP_EAGER")
                .is_some_and(|value| value == "1"),
            overlap_evidence: cfg!(feature = "diagnostic-taps")
                && std::env::var("CHOPINFER_QWEN38_OVERLAP_EVIDENCE").as_deref() == Ok("1"),
        }
    }
}
