//! Whole-K exact F32 row tiles of decode and verify steps
//! (`crate::sm120::f32_verify_rows`): one op per projection and site the
//! measured tiles serve, taken only at the rows the instance's options
//! reach. Every output keeps the split route's bits: the same components,
//! the same eight-MMA chains merged in order and added from +0 in K order,
//! and the same SIMT chains outside the tensor domain.
use crate::ops::Lowered;
use kernel_core::backend::{OpInfo, Refused};
use kernel_core::contract::Refusal;
use runtime_bundle_abi::DeviceBufferSpanV1;

/// A step's spans of one row-tile projection. The workspace holds the
/// preparation's row domain and components and, with K slices, every
/// chain's partial.
#[derive(Clone, Copy)]
pub(crate) struct Spans {
    pub activation: DeviceBufferSpanV1,
    pub weight: DeviceBufferSpanV1,
    pub workspace: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub weight_domain: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
}

impl Spans {
    pub(crate) fn validate(self) -> Result<(), Refusal> {
        let spans = [
            (self.activation, 16),
            (self.weight, 16),
            (self.workspace, 256),
            (self.live_rows, 4),
            (self.weight_domain, 4),
            (self.output, 4),
        ];
        for (i, (a, alignment)) in spans.iter().enumerate() {
            if a.length == 0
                || !a.offset.is_multiple_of(*alignment)
                || a.offset.checked_add(a.length).is_none()
            {
                return Err(Refusal("row tile span is empty, unaligned or overflowing"));
            }
            for (b, _) in &spans[..i] {
                if a.buffer == b.buffer
                    && a.offset < b.offset + b.length
                    && b.offset < a.offset + a.length
                {
                    return Err(Refusal("row tile spans overlap"));
                }
            }
        }
        Ok(())
    }
}

pub(crate) trait RowsProjection: Send + Sync {
    fn info(&self) -> &OpInfo;
    /// Whether a measured tile serves `rows` rows.
    fn admits(&self, rows: u32) -> bool;
    fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal>;
    fn lower(&self, rows: u32, spans: Spans) -> Result<Vec<Lowered>, Refusal>;
}

/// The verify rows an instance runs in row tiles, if any: from 33 rows to
/// its narrow verify rows, and on through the wide verify buckets with the
/// `wide_verify` option.
pub(crate) fn admitted(shape: &crate::shape::Shape) -> Option<[u32; 2]> {
    use model_deepseek41_contract::serving::{DECODE_BATCH, VERIFY_DECODE_ROWS};
    let last = shape
        .wide_verify_rows()
        .last()
        .copied()
        .unwrap_or(shape.verify_rows_within(VERIFY_DECODE_ROWS));
    (last > DECODE_BATCH).then_some([DECODE_BATCH + 1, last])
}

/// The decode steps and the verify steps of the decode bucket or fewer
/// rows an instance runs in row tiles: every one on Flash TP4.
pub(crate) fn decode_admitted(shape: &crate::shape::Shape) -> Option<[u32; 2]> {
    use model_deepseek41_contract::{geometry::FLASH, serving::DECODE_BATCH};
    (shape.geometry.id == FLASH.id && shape.ranks() == 4).then_some([1, DECODE_BATCH])
}

/// The steps a projection's row tiles serve: verify steps above the decode
/// bucket, or decode steps and verify steps within it. Each has its own
/// tuning site and module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Steps {
    Verify,
    Decode,
}

/// A projection's row tiles for one kind of step, prepared only for an
/// instance that admits them.
pub(crate) struct Rows {
    rows: [u32; 2],
    steps: Steps,
    op: Box<dyn RowsProjection>,
}

/// The tiles of `steps` for the projection of `n` outputs from `k` inputs
/// at `rows`, or `None` where no measured tile serves them.
fn prepare_tiles(
    steps: Steps,
    rows: Option<[u32; 2]>,
    asks: Option<(&crate::ops::Asks, &str)>,
    n: u32,
    k: u32,
) -> Result<Option<Rows>, Refused> {
    let Some(rows) = rows else {
        return Ok(None);
    };
    let op = match crate::Kernels::rows_projection(steps, n, k, rows) {
        Ok(op) => op,
        Err(Refused::Unsupported) => return Ok(None),
        Err(error) => return Err(error),
    };
    let need = crate::ops::exact::need(rows[1], crate::ops::exact::verify(rows[1]));
    need.check(op.info())?;
    if let Some((asks, name)) = asks {
        asks.record(name, need, op.info());
    }
    Ok(Some(Rows { rows, steps, op }))
}

/// The verify row tiles of the target's projection of `n` outputs from
/// `k` inputs, or `None` where no measured tile serves the admitted rows.
pub(crate) fn prepare(
    shape: &crate::shape::Shape,
    asks: &crate::ops::Asks,
    name: &str,
    n: u32,
    k: u32,
    target: bool,
) -> Result<Option<Rows>, Refused> {
    prepare_tiles(
        Steps::Verify,
        admitted(shape).filter(|_| target),
        Some((asks, name)),
        n,
        k,
    )
}

/// The decode row tiles of the target's projection of `n` outputs from
/// `k` inputs, or `None` where no measured tile serves the decode bucket.
pub(crate) fn prepare_decode(
    shape: &crate::shape::Shape,
    asks: &crate::ops::Asks,
    name: &str,
    n: u32,
    k: u32,
    target: bool,
) -> Result<Option<Rows>, Refused> {
    prepare_tiles(
        Steps::Decode,
        decode_admitted(shape).filter(|_| target),
        Some((asks, name)),
        n,
        k,
    )
}

/// The drafter's row tiles of its projection of `n` outputs from `k`
/// inputs: the decode site's through the decode bucket, and the verify
/// site's above it to the narrow verify rows the instance admits. With the
/// `wide_draft` option its wide forwards take the verify site's tiles from
/// the 65th row to the widest wide verify bucket.
pub(crate) fn prepare_drafter(
    shape: &crate::shape::Shape,
    n: u32,
    k: u32,
) -> Result<Vec<Rows>, Refused> {
    use model_deepseek41_contract::serving::VERIFY_DECODE_ROWS;
    let narrow = admitted(shape).map(|rows| [rows[0], rows[1].min(VERIFY_DECODE_ROWS)]);
    let wide = admitted(shape)
        .filter(|rows| shape.wide_draft && rows[1] > VERIFY_DECODE_ROWS)
        .map(|rows| [VERIFY_DECODE_ROWS + 1, rows[1]]);
    let mut tiles = Vec::new();
    for (steps, rows) in [
        (Steps::Decode, decode_admitted(shape)),
        (Steps::Verify, narrow),
        (Steps::Verify, wide),
    ] {
        tiles.extend(prepare_tiles(steps, rows, None, n, k)?);
    }
    Ok(tiles)
}

impl Rows {
    /// Whether the tiles serve `phase`: a verify step of rows they admit,
    /// and for decode steps' tiles a decode step too.
    pub(crate) fn for_phase(&self, phase: crate::program::Phase) -> bool {
        use crate::program::Phase;
        let steps = match phase {
            Phase::Verify { .. } => true,
            Phase::Decode(_) => self.steps == Steps::Decode,
            Phase::Prefill { .. } => false,
        };
        steps && self.admits(phase.rows())
    }

    /// Whether a measured tile serves `rows` rows within the prepared
    /// range.
    pub(crate) fn admits(&self, rows: u32) -> bool {
        (self.rows[0]..=self.rows[1]).contains(&rows) && self.op.admits(rows)
    }

    pub(crate) fn info(&self) -> &OpInfo {
        self.op.info()
    }

    pub(crate) fn project(
        &self,
        rows: u32,
        activation: DeviceBufferSpanV1,
        weight: DeviceBufferSpanV1,
        output: DeviceBufferSpanV1,
        live_rows: DeviceBufferSpanV1,
        state: super::ProjectionState,
    ) -> Result<Vec<Lowered>, Refusal> {
        if !(self.rows[0]..=self.rows[1]).contains(&rows) {
            return Err(Refusal("row tiles are outside the instance's rows"));
        }
        let bytes = self.op.workspace_bytes(rows)?;
        if state.workspace.length < bytes {
            return Err(Refusal("row tile workspace is too short"));
        }
        self.op.lower(
            rows,
            Spans {
                activation,
                weight,
                workspace: DeviceBufferSpanV1::new(
                    state.workspace.buffer,
                    state.workspace.offset,
                    bytes,
                ),
                live_rows,
                weight_domain: state
                    .weight_domain
                    .ok_or(Refusal("row tile weights are unclassified"))?,
                output,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ops::Asks, program::Phase, shape::Shape};
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::FLASH;

    /// A Flash TP4 target admits verify row tiles from 33 rows to its
    /// narrow verify rows and, with the wide verify buckets, on to the
    /// widest bucket: the head's and the compressor's at every admitted row
    /// and the router's above 64 rows, where its separate spans end. The
    /// index weights and the drafter keep their routes, and the tiles serve
    /// verify steps alone.
    #[test]
    fn verify_row_tiles_follow_the_verify_rows() {
        let asks = Asks::default();
        let tp4 = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let tp2 = Shape::new(&FLASH, Ranks::TP2).unwrap();
        for shape in [
            tp4,
            tp2.with_verify_rows(64),
            tp2.with_verify_rows(192).with_wide_verify(true),
        ] {
            assert!(admitted(&shape).is_none());
            assert!(
                prepare(&shape, &asks, "head", 32320, 5120, true)
                    .unwrap()
                    .is_none()
            );
        }
        for (verify_rows, wide, last) in [
            (33, false, 33),
            (64, false, 64),
            (192, false, 64),
            (64, true, 64),
            (65, true, 65),
            (96, true, 96),
            (128, true, 128),
            (129, true, 129),
            (192, true, 192),
        ] {
            let shape = tp4.with_verify_rows(verify_rows).with_wide_verify(wide);
            assert_eq!(admitted(&shape), Some([33, last]));
            assert!(
                prepare(&shape, &asks, "drafter", 384, 5120, false)
                    .unwrap()
                    .is_none()
            );
            // The index weights keep the split route.
            assert!(
                prepare(&shape, &asks, "index weights", 32, 5120, true)
                    .unwrap()
                    .is_none()
            );
            for (n, name, served) in [
                (32320, "head", 33..=last),
                (512, "compressor", 33..=last),
                (384, "router", 65..=last),
            ] {
                let op = prepare(&shape, &asks, name, n, 5120, true).unwrap();
                assert_eq!(
                    op.is_some(),
                    !served.is_empty(),
                    "{name} {verify_rows} {wide}"
                );
                for rows in 1..=192 {
                    let phase = Phase::Verify {
                        rows,
                        items: rows.div_ceil(6),
                    };
                    assert_eq!(
                        op.as_ref().is_some_and(|op| op.for_phase(phase)),
                        served.contains(&rows),
                        "{name} rows={rows} verify_rows={verify_rows} wide={wide}"
                    );
                    assert!(
                        !op.as_ref()
                            .is_some_and(|op| op.for_phase(Phase::Decode(rows))
                                || op.for_phase(Phase::Prefill { rows }))
                    );
                }
            }
        }
    }

    /// A Flash TP4 target has the head's decode tiles: every decode step
    /// and every verify step of the decode bucket or fewer rows. No other
    /// projection, the drafter's op or another topology has decode tiles.
    #[test]
    fn head_decode_tiles_serve_the_decode_bucket() {
        let asks = Asks::default();
        let tp4 = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let tp2 = Shape::new(&FLASH, Ranks::TP2).unwrap();
        assert!(decode_admitted(&tp2).is_none());
        assert!(
            prepare_decode(&tp2, &asks, "head", 32320, 5120, true)
                .unwrap()
                .is_none()
        );
        for (verify_rows, wide) in [(32, false), (64, false), (192, false), (192, true)] {
            let shape = tp4.with_verify_rows(verify_rows).with_wide_verify(wide);
            assert_eq!(decode_admitted(&shape), Some([1, 32]));
            assert!(
                prepare_decode(&shape, &asks, "drafter", 32320, 5120, false)
                    .unwrap()
                    .is_none()
            );
            for (n, name) in [(384, "router"), (512, "compressor"), (32, "index weights")] {
                assert!(
                    prepare_decode(&shape, &asks, name, n, 5120, true)
                        .unwrap()
                        .is_none()
                );
            }
            let op = prepare_decode(&shape, &asks, "head", 32320, 5120, true)
                .unwrap()
                .unwrap();
            for rows in 1..=192 {
                assert_eq!(op.for_phase(Phase::Decode(rows)), rows <= 32);
                assert_eq!(
                    op.for_phase(Phase::Verify {
                        rows,
                        items: rows.div_ceil(6),
                    }),
                    rows <= 32
                );
                assert!(!op.for_phase(Phase::Prefill { rows }));
            }
        }
    }

    /// The drafter's head takes the decode tiles through the decode bucket
    /// and the verify head tile above it, to the narrow verify rows the
    /// instance admits; its router keeps its routes. Flash TP2 has none.
    #[test]
    fn drafter_row_tiles_follow_the_instance_rows() {
        let tp4 = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let tp2 = Shape::new(&FLASH, Ranks::TP2).unwrap().with_verify_rows(64);
        assert!(prepare_drafter(&tp2, 32320, 5120).unwrap().is_empty());
        for (verify_rows, wide, last) in [
            (32, false, 32),
            (48, false, 48),
            (64, false, 64),
            (192, false, 64),
            (192, true, 64),
        ] {
            let shape = tp4.with_verify_rows(verify_rows).with_wide_verify(wide);
            let tiles = prepare_drafter(&shape, 32320, 5120).unwrap();
            for rows in 1..=160 {
                assert_eq!(
                    tiles.iter().filter(|t| t.admits(rows)).count(),
                    usize::from(rows <= last),
                    "rows={rows} verify_rows={verify_rows} wide={wide}"
                );
            }
            assert!(prepare_drafter(&shape, 384, 5120).unwrap().is_empty());
        }
        // With `wide_draft` the wide forwards take the verify site's tiles
        // from the 65th row to the widest bucket.
        for verify_rows in [128, 192] {
            let shape = tp4
                .with_verify_rows(verify_rows)
                .with_wide_verify(true)
                .with_wide_draft(true);
            let tiles = prepare_drafter(&shape, 32320, 5120).unwrap();
            for rows in 1..=192 {
                assert_eq!(
                    tiles.iter().filter(|t| t.admits(rows)).count(),
                    usize::from(rows <= verify_rows),
                    "rows={rows} verify_rows={verify_rows}"
                );
            }
        }
    }
}
