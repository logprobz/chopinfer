//! GLM's row copies as the shared ops (`kernel_core::row_copy`): the
//! embedding copied into every residual stream, and the router's scores of
//! every expert, the leading columns of the rows its GEMM writes over the
//! weight's zero-padded rows. Both move bytes only; every view starts at
//! 16-byte alignment.

use kernel_core::contract::DType;
use kernel_core::row_copy::{ColumnSlice, StreamBroadcast};

use crate::contract::Shape;

/// The alignment every copy view of the bundle starts at.
pub const ALIGNMENT_BYTES: u32 = 16;

/// Each BF16 hidden row into every residual stream of `shape`: `[rows,
/// hidden]` into `[rows, streams, hidden]`.
pub fn expand_streams(shape: Shape) -> StreamBroadcast {
    StreamBroadcast {
        dtype: DType::Bf16,
        width: shape.profile.hidden,
        streams: shape.profile.hc_streams,
        alignment_bytes: ALIGNMENT_BYTES,
    }
}

/// The router's F32 scores of every expert of `shape` from its GEMM's
/// padded rows (`precision::router_columns`).
pub fn router_scores(shape: Shape) -> ColumnSlice {
    let (experts, padded) = crate::precision::router_columns(shape);
    ColumnSlice {
        dtype: DType::F32,
        width: experts,
        stride: padded,
        offset: 0,
        alignment_bytes: ALIGNMENT_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::Validate;

    #[test]
    fn the_copies_are_the_served_shapes() {
        let shape = Shape::TP4;
        let broadcast = expand_streams(shape);
        assert_eq!((broadcast.width, broadcast.streams), (4096, 4));
        broadcast.validate().unwrap();
        let scores = router_scores(shape);
        assert_eq!((scores.width, scores.stride, scores.offset), (288, 320, 0));
        scores.validate().unwrap();
    }
}
