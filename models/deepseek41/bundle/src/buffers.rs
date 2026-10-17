//! One rank's buffers: the activations of the widest phase, the logits
//! rows, the per-item request state the host allocated, the placed weight
//! views and the serving KV pool.
//!
//! Every activation holds `scratch_rows()` rows, the most any phase runs: a
//! prefill chunk, or the widest decode or verify bucket. Each phase binds
//! its first rows. Families add their buffers as they lower stages.

use std::collections::BTreeMap;

use bundle_runtime::state::{Extent, RequestState};
use kernel_core::allocation::Allocation;
use model_deepseek41_contract::serving::{PREFILL_TOKENS, SCORED_ROWS};
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};

use crate::shape::Shape;

const BF16: u64 = 2;
const F32: u64 = 4;

/// Rows of every activation buffer.
pub fn scratch_rows() -> u32 {
    PREFILL_TOKENS.max(head_rows())
}

/// Most rows the head scores in a decode or verify phase: the widest verify
/// bucket, which holds every decode bucket.
pub fn head_rows() -> u32 {
    crate::ops::verify_rows()
}

/// Rows the local logits shard holds: a prefill's scored rows or a decode
/// or verify phase's.
pub fn logits_rows() -> u32 {
    SCORED_ROWS.max(head_rows())
}

kernel_core::scratch_record! {
    /// One rank's activations and logits.
    #[derive(Clone, Copy, Debug)]
    pub struct Scratch {
        /// The rank's embedding rows of its vocabulary shard (BF16, zero for
        /// other shards' ids).
        pub embedding: value,
        /// The embedding the all-reduce assembled (BF16).
        pub embedded: value,
        /// The residual streams, `[rows, streams, hidden]` BF16.
        pub streams: value,
        /// The streams collapsed by a mixer's `pre` (BF16).
        pub collapsed: value,
        /// A sublayer input's RMSNorm (BF16).
        pub normed: value,
        /// The rank's F32 logits of its vocabulary shard.
        pub head: value,
        /// The decode and verify rows' full F32 logits on the rank that owns
        /// their block.
        pub decode_logits: value,
        /// The prefill's scored rows' full F32 logits, on rank 0.
        pub prefill_logits: value,
    }
}

impl Scratch<Allocation> {
    /// Every buffer of one rank of `shape`.
    pub fn of(shape: &Shape) -> Self {
        let rows = u64::from(scratch_rows());
        let hidden = u64::from(shape.dims().hidden);
        let vocab = u64::from(shape.dims().vocab);
        let shard = u64::from(shape.vocab_rows());
        let row = |bytes: u64| Allocation::bytes(rows * bytes);
        Self {
            embedding: row(hidden * BF16),
            embedded: row(hidden * BF16),
            streams: row(u64::from(shape.dims().hc.streams) * hidden * BF16),
            collapsed: row(hidden * BF16),
            normed: row(hidden * BF16),
            head: Allocation::bytes(u64::from(logits_rows()) * shard * F32),
            decode_logits: Allocation::bytes(u64::from(head_rows()) * vocab * F32),
            prefill_logits: Allocation::bytes(u64::from(SCORED_ROWS) * vocab * F32),
        }
    }
}

/// The per-item request state of one rank: each buffer of the declaration
/// at its offset in the host's state view, in declaration order.
#[derive(Clone, Debug)]
pub struct StateSpans {
    pub buffers: Vec<DeviceBufferSpanV1>,
}

impl StateSpans {
    /// `state`'s buffers for `extent`, laid end to end in the host's view
    /// `view` of the per-item state.
    pub fn new(
        state: &RequestState,
        extent: Extent,
        view: DeviceBufferSpanV1,
    ) -> Result<Self, BundleAbiError> {
        let mut offset = 0;
        let mut buffers = Vec::with_capacity(state.buffers.len());
        for buffer in &state.buffers {
            let bytes = buffer.bytes(extent);
            if offset + bytes > view.length {
                return Err(BundleAbiError::InvalidRange(
                    "deepseek41 request state exceeds the host's view",
                ));
            }
            buffers.push(DeviceBufferSpanV1::new(
                view.buffer,
                view.offset + offset,
                bytes,
            ));
            offset += bytes;
        }
        Ok(Self { buffers })
    }
}

/// One rank's buffers.
#[derive(Clone, Debug)]
pub struct RankBuffers {
    pub scratch: Scratch<DeviceBufferSpanV1>,
    pub state: StateSpans,
    /// The rank image's placed views, by name.
    pub views: BTreeMap<String, DeviceBufferSpanV1>,
    pub kv_pool: DeviceBufferSpanV1,
}

impl RankBuffers {
    /// The placed view `name`, or the error naming it.
    pub fn view(&self, name: &str) -> Result<DeviceBufferSpanV1, String> {
        self.views
            .get(name)
            .copied()
            .ok_or_else(|| format!("the rank image has no view {name}"))
    }
}

/// The first `rows` rows of `span`, a buffer of `scratch_rows()` rows.
pub fn rows(span: DeviceBufferSpanV1, rows: u32) -> DeviceBufferSpanV1 {
    let row = span.length / u64::from(scratch_rows());
    DeviceBufferSpanV1::new(span.buffer, span.offset, row * u64::from(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bundle_runtime::geometry::Ranks;
    use kernel_core::allocation::Traversal;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn flash_scratch_fits_beside_the_weights() {
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        let mut bytes = 0;
        Scratch::of(&shape)
            .visit(&mut |allocation: &Allocation| bytes += allocation.allocation_bytes());
        // The streams dominate: 8192 rows of four 5120-wide BF16 copies.
        assert!(bytes < 1 << 30, "{bytes}");
    }

    #[test]
    fn state_spans_follow_the_declaration() {
        let state = model_deepseek41_contract::state::request_state(
            &FLASH,
            model_deepseek41_contract::state::KvStorage::Native,
            false,
        );
        let extent = Extent {
            served: 2,
            bucket: 2,
            verify_rows: 0,
        };
        let total: u64 = state.buffers.iter().map(|b| b.bytes(extent)).sum();
        let view = DeviceBufferSpanV1::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(1, 1),
            256,
            total,
        );
        let spans = StateSpans::new(&state, extent, view).unwrap();
        assert_eq!(spans.buffers.len(), state.buffers.len());
        assert_eq!(
            spans.buffers[1].offset,
            256 + state.buffers[0].bytes(extent)
        );
        let short = DeviceBufferSpanV1::new(view.buffer, 256, total - 1);
        assert!(StateSpans::new(&state, extent, short).is_err());
    }
}
