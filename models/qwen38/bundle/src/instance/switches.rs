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
    /// Candidate: every drafter row a prefilling item's step writes runs in
    /// the prefill phase (`CANONICAL_DRAFTER`), through the whole drafter
    /// program or only its persistent stages.
    pub canonical_drafter: Canonical,
    /// Diagnostic-taps builds write each prefill's overlap split
    /// (`CHOPINFER_QWEN38_OVERLAP_EVIDENCE=1`).
    pub overlap_evidence: bool,
    /// Diagnostic-taps builds log every drafting item's drafts
    /// (`CHOPINFER_QWEN38_DRAFT_LOG=1`).
    pub draft_log: bool,
}

/// What the candidate (`CANONICAL_DRAFTER`) runs for the drafter rows a
/// prefilling item's step writes: its chunk-start catch-up rows, as a run
/// of their own, and every row its chunk completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Canonical {
    /// Unset: catch-up rows join the drafting items' first run, and runs of
    /// at most 32 rows replay a verify graph bucket of decode entries.
    Off,
    /// `1`: every such run in the prefill phase at every row count, through
    /// the whole drafter program, queued one stage at a time.
    Full,
    /// `2`: as `Full`, but such a run, which never drafts, runs only the
    /// stages that write persistent drafter state
    /// (`program::drafter_persistent_stages`), queued whole on each rank.
    Persistent,
}

impl Canonical {
    /// Whether rows a prefilling item's step writes run in the prefill
    /// phase.
    pub(crate) fn on(self) -> bool {
        self != Canonical::Off
    }

    /// The name the overlap evidence records.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Canonical::Off => "off",
            Canonical::Full => "full",
            Canonical::Persistent => "persistent",
        }
    }
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
            canonical_drafter: canonical_drafter(),
            overlap_evidence: cfg!(feature = "diagnostic-taps")
                && std::env::var("CHOPINFER_QWEN38_OVERLAP_EVIDENCE").as_deref() == Ok("1"),
            draft_log: cfg!(feature = "diagnostic-taps")
                && std::env::var("CHOPINFER_QWEN38_DRAFT_LOG").as_deref() == Ok("1"),
        }
    }
}

/// The candidate switch that runs every drafter row a prefilling item's
/// step writes (its chunk-start catch-up rows and every row its chunk
/// completes) in the prefill phase at every row count, so a row's bits
/// follow from its own inputs, not from where chunks started or what else
/// the step held: `1` through the whole drafter program, `2` through its
/// persistent stages only ([`Canonical`]). Off by default until adopted.
pub(crate) const CANONICAL_DRAFTER: &str = "CHOPINFER_QWEN38_CANONICAL_DRAFTER";

#[cfg(test)]
thread_local! {
    /// Test-only candidate selection, independent of the process environment.
    pub(crate) static FORCE_CANONICAL_DRAFTER: std::cell::Cell<Option<Canonical>> =
        const { std::cell::Cell::new(None) };
}

fn canonical_drafter() -> Canonical {
    #[cfg(test)]
    if let Some(forced) = FORCE_CANONICAL_DRAFTER.with(std::cell::Cell::get) {
        return forced;
    }
    match std::env::var(CANONICAL_DRAFTER).as_deref() {
        Ok("1") => Canonical::Full,
        Ok("2") => Canonical::Persistent,
        _ => Canonical::Off,
    }
}

/// Runs `create` with the candidate `mode`, whatever the environment.
#[cfg(test)]
pub(crate) fn with_canonical_drafter<T>(mode: Canonical, create: impl FnOnce() -> T) -> T {
    let previous = FORCE_CANONICAL_DRAFTER.with(|forced| forced.replace(Some(mode)));
    let result = create();
    FORCE_CANONICAL_DRAFTER.with(|forced| forced.set(previous));
    result
}
