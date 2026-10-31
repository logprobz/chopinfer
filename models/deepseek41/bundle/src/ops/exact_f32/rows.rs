//! Whole-K exact F32 row tiles of verify steps
//! (`crate::sm120::f32_verify_rows`): one op per projection the measured
//! tiles serve, taken only at the verify rows the instance's options reach.
//! Every output keeps the split route's bits: the same components, the
//! same eight-MMA chains merged in order and added from +0 in K order, and
//! the same SIMT chains outside the tensor domain.
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
    /// Whether a measured tile serves `rows` verify rows.
    fn admits(&self, rows: u32) -> bool;
    fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal>;
    fn lower(&self, rows: u32, spans: Spans) -> Result<Vec<Lowered>, Refusal>;
}

/// The verify rows an instance runs in row tiles, if any: 33 to 64 rows
/// with the `f32_row_tiles` candidate option, and the wide verify buckets
/// above the 64-row decode route with the `wide_verify` option.
pub(crate) fn admitted(shape: &crate::shape::Shape) -> Option<[u32; 2]> {
    use model_deepseek41_contract::serving::{DECODE_BATCH, VERIFY_DECODE_ROWS};
    let narrow = (shape.f32_row_tiles
        && shape.verify_rows_within(VERIFY_DECODE_ROWS) > DECODE_BATCH)
        .then(|| shape.verify_rows_within(VERIFY_DECODE_ROWS));
    let last = shape.wide_verify_rows().last().copied().or(narrow)?;
    let first = if narrow.is_some() {
        DECODE_BATCH + 1
    } else {
        VERIFY_DECODE_ROWS + 1
    };
    Some([first, last])
}

/// A projection's row tiles, prepared only for an instance that admits
/// them.
pub(crate) struct Rows {
    rows: [u32; 2],
    op: Box<dyn RowsProjection>,
}

/// The row tiles of the target's projection of `n` outputs from `k`
/// inputs, or `None` where no measured tile serves the admitted rows.
pub(crate) fn prepare(
    shape: &crate::shape::Shape,
    asks: &crate::ops::Asks,
    name: &str,
    n: u32,
    k: u32,
    target: bool,
) -> Result<Option<Rows>, Refused> {
    let Some(rows) = admitted(shape).filter(|_| target) else {
        return Ok(None);
    };
    let op = match crate::Kernels::rows_verify_projection(n, k, rows) {
        Ok(op) => op,
        Err(Refused::Unsupported) => return Ok(None),
        Err(error) => return Err(error),
    };
    let need = crate::ops::exact::need(rows[1], crate::ops::exact::verify(rows[1]));
    need.check(op.info())?;
    asks.record(name, need, op.info());
    Ok(Some(Rows { rows, op }))
}

impl Rows {
    pub(crate) fn for_phase(&self, phase: crate::program::Phase) -> bool {
        matches!(phase, crate::program::Phase::Verify { .. })
            && (self.rows[0]..=self.rows[1]).contains(&phase.rows())
            && self.op.admits(phase.rows())
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
            return Err(Refusal("row tiles are outside the instance's verify rows"));
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

    /// Without the `f32_row_tiles` option only a Flash TP4 target with the
    /// wide verify buckets prepares row tiles, and they serve only verify
    /// steps above 64 rows.
    #[test]
    fn row_tiles_follow_the_wide_verify_buckets() {
        let asks = Asks::default();
        let tp4 = Shape::new(&FLASH, Ranks::TP4).unwrap();
        for shape in [
            tp4.with_verify_rows(192),
            tp4.with_verify_rows(64).with_wide_verify(true),
            Shape::new(&FLASH, Ranks::TP2)
                .unwrap()
                .with_verify_rows(192)
                .with_wide_verify(true),
        ] {
            assert!(admitted(&shape).is_none());
            assert!(
                prepare(&shape, &asks, "router", 384, 5120, true)
                    .unwrap()
                    .is_none()
            );
        }
        for verify_rows in [65, 128, 129, 192] {
            let shape = tp4.with_verify_rows(verify_rows).with_wide_verify(true);
            assert_eq!(admitted(&shape), Some([65, verify_rows]));
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
            for (n, name) in [(384, "router"), (512, "compressor"), (32320, "head")] {
                let op = prepare(&shape, &asks, name, n, 5120, true)
                    .unwrap()
                    .unwrap();
                for rows in 1..=192 {
                    assert_eq!(
                        op.for_phase(Phase::Verify {
                            rows,
                            items: rows.div_ceil(6),
                        }),
                        (65..=verify_rows).contains(&rows),
                        "{name} rows={rows}"
                    );
                    assert!(!op.for_phase(Phase::Decode(rows)));
                    assert!(!op.for_phase(Phase::Prefill { rows }));
                }
            }
        }
    }

    /// The `f32_row_tiles` option adds the head's and the compressor's
    /// tiles at 33 to 64 rows; the router keeps its separate spans there.
    #[test]
    fn row_tile_option_adds_the_measured_narrow_rows() {
        let asks = Asks::default();
        let tp4 = Shape::new(&FLASH, Ranks::TP4)
            .unwrap()
            .with_f32_row_tiles(true);
        for shape in [
            tp4,
            tp4.with_verify_rows(192)
                .with_wide_verify(false)
                .with_f32_row_tiles(false),
            Shape::new(&FLASH, Ranks::TP2)
                .unwrap()
                .with_verify_rows(64)
                .with_f32_row_tiles(true),
        ] {
            assert!(admitted(&shape).is_none());
        }
        for (verify_rows, wide, last) in [
            (33, false, 33),
            (64, false, 64),
            (192, false, 64),
            (96, true, 96),
            (192, true, 192),
        ] {
            let shape = tp4.with_verify_rows(verify_rows).with_wide_verify(wide);
            assert_eq!(admitted(&shape), Some([33, last]));
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
                }
            }
        }
    }
}
