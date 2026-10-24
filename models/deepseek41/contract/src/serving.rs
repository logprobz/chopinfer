//! The `deepseek41.v1` serving contract: the page, batch and verify limits
//! the bundle's kernels and buffers are built for, its topologies and its
//! serving KV layouts.

/// Tokens per serving KV page: whole groups of every compress ratio (at
/// most 2) and whole candidate blocks of ratio-1 positions (8).
pub const PAGE_TOKENS: u64 = 64;
/// Policy: prefill launches cap scheduler chunks (the profiles'
/// `max_batch_tokens`).
pub const PREFILL_TOKENS: u32 = 8192;
/// Positions a prefill chunk may end at before the prompt end: whole pages,
/// so a chunk boundary never splits a compression group or a candidate
/// block (the manifest's `batch_limits.prefill_granule`).
pub const PREFILL_GRANULE: u32 = PAGE_TOKENS as u32;
/// The package maximum of served sequences and items per step.
pub const DECODE_BATCH: u32 = 32;
/// Graph row buckets shared by model preparation and target registration.
pub const DECODE_BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Verify rows one item may carry: the bonus token and a DSpark block's
/// five drafts.
pub const MAX_VERIFY_ROWS: u32 = 6;
pub const MAX_DRAFT_TOKENS: u32 = MAX_VERIFY_ROWS - 1;
/// Policy: token buffers align to the runtime allocation granule.
pub const TOKEN_GRANULE: u32 = 256;
/// Most logits rows a prefill phase scores, the package's
/// `maximum_scored_rows`: a multiple of the prefill granule that holds the
/// last rows of every served item.
pub const SCORED_ROWS: u32 = 64;
const _: () =
    assert!(SCORED_ROWS >= PREFILL_GRANULE && SCORED_ROWS.is_multiple_of(PREFILL_GRANULE));
const _: () = assert!(SCORED_ROWS >= DECODE_BATCH);

/// Package-facing identifiers of the serving KV layouts (plan §2.9): the
/// trained mixed-plane formats (the default), or BF16 holding the same
/// values (the comparison profile).
pub const KV_LAYOUT_NATIVE: &[u8] = b"deepseek41.csa2-paged-native.v1";
pub const KV_LAYOUT_BF16: &[u8] = b"deepseek41.csa2-paged-bf16.v1";

/// Served device-result parity is qualified on Flash at TP4.
pub fn default_device_results(geometry: &crate::geometry::Geometry, ranks: u32) -> bool {
    geometry.id == crate::geometry::FLASH.id && ranks == 4
}

/// The per-item state layout (`state_layout` under
/// `chop.state-per-item.v1`): every window ring and compressor tail, and
/// with DSpark its rings and retained rows (`crate::state`).
pub const REQUEST_STATE_ID: &str = "deepseek41.request-state.v1";

/// The planned text image of every rank (the weight plan's image role),
/// and the schema each rank's index is written under.
pub const RANK_IMAGE_ROLE: &str = "deepseek41.rank-images.v1";
pub const INDEX_SCHEMA: &str = "chopinfer.deepseek41.rank-image.v1";

/// The component role of rank `rank`'s index of the text image.
pub fn index_role(rank: u32) -> String {
    bundle_runtime::index::index_role("deepseek41", bundle_runtime::index::ImageKind::Text, rank)
}

/// The package geometry for the shared bundle runtime.
pub struct DeepSeek;

impl bundle_runtime::BundleGeometry for DeepSeek {
    const PAGE_TOKENS: u64 = PAGE_TOKENS;
    const PREFILL_TOKENS: u32 = PREFILL_TOKENS;
    const DECODE_BATCH: u32 = DECODE_BATCH;
    const MAX_VERIFY_ROWS: u32 = MAX_VERIFY_ROWS;
    /// Every item at its deepest block. An instance serves the verify rows
    /// its create record asks for (`chop.speculative-requested-rows.v1`).
    const VERIFY_TOKENS: u32 = DECODE_BATCH * MAX_VERIFY_ROWS;
    const EXACT_VERIFY_GRAPHS: bool = true;
    const TOKEN_GRANULE: u32 = TOKEN_GRANULE;
}

#[cfg(test)]
mod tests {
    use super::DeepSeek;
    use bundle_runtime::geometry::Items;
    use bundle_runtime::graphs::{decode_bucket, verify_bucket_within};

    #[test]
    fn verify_graph_rows_keep_the_exact_width() {
        for items in [1, 3, 8, 32] {
            let items = Items::within::<DeepSeek>(items).unwrap();
            for rows in 1..=192 {
                assert_eq!(verify_bucket_within::<DeepSeek>(rows, items), Some(rows));
            }
            assert_eq!(verify_bucket_within::<DeepSeek>(0, items), None);
            assert_eq!(verify_bucket_within::<DeepSeek>(193, items), None);
        }
        assert_eq!(decode_bucket::<DeepSeek>(3), Some(4));
    }
}
