//! The host's startup warmup schedule. The host drives it before it
//! publishes readiness, and a bundle's CPU conformance tests drive the same
//! wave sizes, so a step the host's warmup sends cannot first fail on a GPU.

mod shapes;

pub use shapes::{VerifyShape, VerifyWarmup, context_ladder, verify_shapes};

/// Wave sizes: the powers of two below `sequences`, then `sequences`.
pub fn waves(sequences: usize) -> Vec<usize> {
    let mut waves = std::iter::successors(Some(1_usize), |wave| wave.checked_mul(2))
        .take_while(|&wave| wave < sequences)
        .collect::<Vec<_>>();
    waves.push(sequences.max(1));
    waves
}

/// Setting this to "1" keeps a narrow-rows package's declared verify rows.
pub const WIDE_ROWS_ENV: &str = "CHOPINFER_SPECULATIVE_WIDE_ROWS";

/// Whether a package declaring the narrow-rows capability
/// (`chop.speculative-narrow-rows.v1`) narrows its served verify rows.
pub fn narrows_rows(capability: bool) -> bool {
    capability && std::env::var(WIDE_ROWS_ENV).as_deref() != Ok("1")
}

/// The verify rows a host serves of a package's `declared` maximum: at most
/// 32 when the package narrows rows and serves at most 32 sequences.
pub fn served_verify_rows(declared: usize, sequences: usize, bounded: bool) -> usize {
    if bounded && sequences <= 32 {
        declared.min(32)
    } else {
        declared
    }
}

/// Rows one verify item may take: the package's declared row buckets, or
/// every depth up to `maximum_draft_tokens` when it declares none.
pub fn item_rows(row_buckets: &[usize], maximum_draft_tokens: usize) -> Vec<usize> {
    if row_buckets.is_empty() {
        (1..=maximum_draft_tokens + 1).collect()
    } else {
        row_buckets.to_vec()
    }
}
