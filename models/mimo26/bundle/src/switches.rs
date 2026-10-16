//! MiMo's selection switches, read once when an instance is created and
//! passed to every step's lowering, so no step's lowering reads them.
//!
//! Some controls are still read later: `execute_step` reads the tracing
//! variables `CHOPINFER_STEP_TRACE` and `CHOPINFER_MIMO_TRACE_LAUNCHES` on
//! every step, and `verify_eager` reads `CHOPINFER_MIMO_VERIFY_EAGER` on
//! every verify execution (`instance.rs`). The folds behind
//! `CHOPINFER_MIMO_FOLD_DECODE` and `CHOPINFER_MIMO_ATTENTION_FOLD` are read
//! once per process rather than per instance.

/// The dense projections of wide verify and decode buckets run the
/// medium-M entries where the catalog selects one for the bucket's rows
/// (64-token tiles at 33 to 64 rows; larger buckets run the prefill
/// entries). Their outputs are bitwise equal to the prefill entries they
/// replace. On by default since the TP4 A/B of 2026-09-26 (identical
/// records, fixed K7 +1% to +4%); `0` restores the prefill entries.
pub const VERIFY_DENSE_MID: &str = "CHOPINFER_MIMO_VERIFY_DENSE_MID";

pub(crate) fn verify_dense_mid_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(VERIFY_DENSE_MID).as_deref() != Ok("0"))
}

/// `CHOPINFER_MIMO_WIDE_COMM=0`, read once per process, keeps graphs
/// above 32 rows on the reduce-scatter and fused gather stages and the
/// combine row, as before the wide entries (an A/B baseline).
pub const WIDE_COMM: &str = "CHOPINFER_MIMO_WIDE_COMM";

pub(crate) fn wide_comm_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var(WIDE_COMM).as_deref() != Ok("0"))
}

/// The switches an instance's steps follow.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Switches {
    /// Verify buckets of up to `EXACT_VERIFY_ROWS` rows run exact rows
    /// (`selection::VERIFY_EXACT`).
    pub exact_verify: bool,
    /// Wide verify buckets run per-row exact attention
    /// (`selection::VERIFY_ATTENTION`).
    pub verify_attention_rows: bool,
    /// Prefill row kernels grid the prefill capacity instead of the live
    /// rows' bucket (`selection::PREFILL_ROW_GRIDS`).
    pub capacity_row_grids: bool,
    /// Exact rows run paged attention on the row-group entries
    /// (`selection::EXACT_ATTENTION`).
    pub exact_attention_grouped: bool,
    /// Wide verify and decode buckets run the medium-M dense entries
    /// ([`VERIFY_DENSE_MID`]).
    pub dense_mid: bool,
    /// Decode attention folds RoPE and the KV write in
    /// (`attention_plan::FOLD_ROPE`).
    pub fold_rope: bool,
}

impl Switches {
    /// The process environment's switches.
    pub fn configured() -> Self {
        Self {
            exact_verify: crate::selection::verify_exact_enabled(),
            verify_attention_rows: crate::selection::verify_attention_rows(),
            capacity_row_grids: crate::selection::capacity_row_grids(),
            exact_attention_grouped: crate::selection::exact_attention_grouped(),
            dense_mid: verify_dense_mid_enabled(),
            fold_rope: crate::attention_plan::fold_rope_enabled(),
        }
    }
}

/// The controls an instance's steps follow that shape no launch: tracing,
/// eager verify and decode folding, read once at create.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Controls {
    /// `CHOPINFER_STEP_TRACE=1`: each step reports its shape and times.
    pub step_trace: bool,
    /// `CHOPINFER_MIMO_TRACE_LAUNCHES` set: each step records its launches.
    pub trace_launches: bool,
    /// `CHOPINFER_MIMO_VERIFY_EAGER=1`: verify steps run stage by stage
    /// instead of as graphs, to compare the two.
    pub verify_eager: bool,
    /// A step's plain decode rows run in its prefill forward
    /// (`instance::FOLD_DECODE`).
    pub fold_decode: bool,
}

impl Controls {
    /// The process environment's controls.
    pub fn configured() -> Self {
        Self {
            step_trace: std::env::var("CHOPINFER_STEP_TRACE").as_deref() == Ok("1"),
            trace_launches: std::env::var_os("CHOPINFER_MIMO_TRACE_LAUNCHES").is_some(),
            verify_eager: std::env::var_os("CHOPINFER_MIMO_VERIFY_EAGER")
                .is_some_and(|value| value == "1"),
            fold_decode: std::env::var(crate::instance::FOLD_DECODE).as_deref() == Ok("1"),
        }
    }
}
