//! Select only compiled MiMo engine entries for each rank-program stage.

use kernel_core::contract::{AttentionPhase, Refusal};

use crate::contract::{DECODE_ENTRY_ROWS, DECODE_EXPERT_ROWS, Mimo, PREFILL_TOKENS};
use crate::switches::Switches;

/// The shared phase (`bundle_runtime::phase`). MiMo's verify phase
/// runs several rows per item over its token bucket: token-indexed stages
/// run at their decode capacity for the bucket, or above the decode batch
/// the prefill kernels at the bucket's rows; verify attention shares each
/// item's keys across its queries, and the sliding-window ring is written
/// after attention, as prefill writes it.
pub use bundle_runtime::phase::Phase;

/// The row buckets of a target's row kernels (`Target::row_buckets`): the
/// smallest that holds a step's rows, which eager prefill row kernels grid.
/// The default holds no rows, as before an instance prepares its ops.
#[derive(Clone, Copy)]
pub struct RowBuckets(fn(u32) -> Option<u32>);

impl RowBuckets {
    /// The buckets whose smallest holding `rows` is `smallest(rows)`.
    pub const fn new(smallest: fn(u32) -> Option<u32>) -> Self {
        Self(smallest)
    }

    /// The smallest bucket that holds `rows`.
    pub fn smallest(self, rows: u32) -> Option<u32> {
        (self.0)(rows)
    }
}

impl Default for RowBuckets {
    fn default() -> Self {
        Self(|_| None)
    }
}

/// MiMo's policy over the shared phase: its geometry and decode kernels,
/// exact verify rows and attention phases.
pub trait MimoPhase: Copy {
    /// The prepared token capacity of token-indexed stages.
    fn capacity(self) -> Result<u32, Refusal>;
    /// The grid rows of a row kernel over `live_tokens` rows
    /// (`Phase::row_grid_in`): prefill grids the smallest of `buckets` that
    /// holds them. `CHOPINFER_MIMO_PREFILL_ROW_GRIDS=capacity`
    /// (`Switches::capacity_row_grids`) restores the capacity grid for A/B
    /// runs.
    fn row_grid(
        self,
        live_tokens: u32,
        switches: Switches,
        buckets: RowBuckets,
    ) -> Result<u32, Refusal>;
    /// Whether token-indexed stages run the prefill kernels: prefill, and
    /// buckets above the decode kernels' 32 rows. Attention keeps its
    /// decode and verify entries at every bucket.
    fn prefill_kernels(self) -> bool;
    /// The capacity the selected kernels are compiled for.
    fn kernel_capacity(self) -> Result<u32, Refusal>;
    /// The attention entries' phase: exact verify rows run the decode
    /// variant's rows entries (`Variant::plan_rows`).
    fn attention(self, switches: Switches) -> AttentionPhase;
    fn label(self) -> &'static str;
    /// Whether this is a verify bucket whose rows are exact (see
    /// [`EXACT_VERIFY_ROWS`]).
    fn exact_rows(self, switches: Switches) -> bool;
    /// Whether this verify bucket's attention runs the decode variant's
    /// per-row exact rows entries while its other stages keep the bucket's
    /// plans: a bucket wider than [`EXACT_VERIFY_ROWS`] under the diagnostic
    /// [`VERIFY_ATTENTION`].
    fn rows_attention(self, switches: Switches) -> bool;
    fn exact_rows_with(self, enabled: bool) -> bool;
    /// Whether the step routes its experts as prefill does: the router
    /// GEMM, top-8 and tiled permutation, then the prefill expert entries.
    /// Prefill does, and decode and verify buckets above
    /// [`DECODE_EXPERT_ROWS`].
    fn prefill_route(self) -> bool;
}

impl MimoPhase for Phase {
    fn capacity(self) -> Result<u32, Refusal> {
        self.capacity_in::<Mimo>().ok_or(match self {
            Self::Decode(_) => Refusal("invalid MiMo decode batch"),
            _ => Refusal("invalid MiMo verify rows"),
        })
    }

    fn row_grid(
        self,
        live_tokens: u32,
        switches: Switches,
        buckets: RowBuckets,
    ) -> Result<u32, Refusal> {
        match self {
            Self::Prefill if !switches.capacity_row_grids => self
                .row_grid_in::<Mimo>(live_tokens, |rows| buckets.smallest(rows))
                .ok_or(Refusal("MiMo prefill rows exceed the prefill capacity")),
            _ => self.capacity(),
        }
    }

    fn prefill_kernels(self) -> bool {
        self.prefill_kernels_in::<Mimo>(DECODE_ENTRY_ROWS)
    }

    fn kernel_capacity(self) -> Result<u32, Refusal> {
        if self.prefill_kernels() {
            Ok(PREFILL_TOKENS)
        } else {
            self.capacity()
        }
    }

    fn attention(self, switches: Switches) -> AttentionPhase {
        match self {
            Self::Verify(_) if self.exact_rows(switches) || self.rows_attention(switches) => {
                AttentionPhase::Decode
            }
            Self::Decode(_) => AttentionPhase::Decode,
            Self::Verify(_) => AttentionPhase::Verify,
            Self::Prefill => AttentionPhase::Prefill,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Decode(_) => "decode",
            Self::Verify(_) | Self::Prefill => "prefill",
        }
    }

    fn exact_rows(self, switches: Switches) -> bool {
        self.exact_rows_with(switches.exact_verify)
    }

    fn rows_attention(self, switches: Switches) -> bool {
        switches.verify_attention_rows
            && matches!(self, Self::Verify(_))
            && self
                .capacity()
                .is_ok_and(|capacity| capacity > EXACT_VERIFY_ROWS)
    }

    fn prefill_route(self) -> bool {
        match self {
            Self::Prefill => true,
            Self::Decode(_) | Self::Verify(_) => self
                .capacity()
                .is_ok_and(|capacity| capacity > DECODE_EXPERT_ROWS),
        }
    }

    fn exact_rows_with(self, enabled: bool) -> bool {
        enabled
            && matches!(self, Self::Verify(_))
            && self
                .capacity()
                .is_ok_and(|capacity| capacity <= EXACT_VERIFY_ROWS)
    }
}

/// Verify buckets of at most [`EXACT_VERIFY_ROWS`] rows run exact rows.
/// Their attention runs the decode kernels over each row's own keys and
/// split plan, the all-reduce norm sums squares in the one-row order, and
/// the experts take the one-row K splits and schedule; the dense GEMMs,
/// router and head already give every row the one-row bits.
pub use crate::contract::EXACT_VERIFY_ROWS;

/// The exact-rows switch and the expert selection rule are the contract's,
/// so the capability offers name the entries selected here.
pub use crate::contract::{VERIFY_EXACT, verify_exact_enabled};

/// The diagnostic switch that runs wide verify attention as per-row exact
/// rows ([`MimoPhase::rows_attention`]).
pub use crate::contract::{VERIFY_ATTENTION, verify_attention_rows};

/// Exact rows run global attention on the row-group entries
/// (`Variant::plan_rows_shared`: up to four rows of a class per CTA over one
/// read of their keys). Set to `rows` for the per-row entries
/// (`Variant::plan_rows`), bitwise the same rows at more cost from 8K on.
pub const EXACT_ATTENTION: &str = "CHOPINFER_MIMO_EXACT_ATTENTION";

/// Whether exact rows run paged attention on the row-group entries.
pub fn exact_attention_grouped() -> bool {
    static GROUPED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GROUPED.get_or_init(|| std::env::var(EXACT_ATTENTION).as_deref() != Ok("rows"))
}

/// See [`MimoPhase::row_grid`].
pub const PREFILL_ROW_GRIDS: &str = "CHOPINFER_MIMO_PREFILL_ROW_GRIDS";

pub(crate) fn capacity_row_grids() -> bool {
    static CAPACITY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CAPACITY.get_or_init(|| std::env::var(PREFILL_ROW_GRIDS).as_deref() == Ok("capacity"))
}
