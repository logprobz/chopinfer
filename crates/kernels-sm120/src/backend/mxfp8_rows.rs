//! Prepare MXFP8 row decoding through registered SM120 entries.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernel_core::mxfp8_rows::{Mxfp8Rows, Mxfp8RowsBuffers};
use kernel_core::properties::Invariance;

use super::select::{Candidate, choose};
use super::{MAX_CAPACITY, check_plan};
use crate::row::mxfp8_decode::{Decode, declared_entries};

/// The decode `op` through a registered decode entry.
pub(super) fn decode_mxfp8_rows(
    symbols: &[String],
    op: &Mxfp8Rows,
    need: Need,
) -> Result<Prepared<Mxfp8RowsBuffers<View>>, Refused> {
    let chosen = choose(
        "MXFP8 row decode",
        symbols
            .iter()
            .map(|symbol| Candidate::new(symbol.clone(), symbol, symbol.clone(), declared(symbol))),
        need,
    )?;
    if need.capacity == 0 || need.capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 MXFP8 row decode serves 1 to {MAX_CAPACITY} rows"
        )));
    }
    let decode = Decode::new(*op, need.capacity, &chosen.entry)?;
    check_plan(&decode.plan(need.capacity)?)?;
    let op = DecodeOp {
        info: OpInfo::new(
            "sm120.row.mxfp8-to-bf16.rne.v1",
            chosen.guarantees,
            vec![(chosen.entry, 0)],
        ),
        decode,
    };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The guarantees the shared decode's declaration claims for `symbol`;
/// none for an undeclared entry.
fn declared(symbol: &str) -> Invariance {
    declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// `chop::row::mxfp8_decode::rows`: one CTA per row of the smallest row
/// bucket holding the launch's grid rows.
struct DecodeOp {
    decode: Decode,
    info: OpInfo,
}

impl PreparedOp<Mxfp8RowsBuffers<View>> for DecodeOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &Mxfp8RowsBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let (rows, grid) = self.decode.rows(counts.rows, counts.padded_rows)?;
        let op = self.decode.op();
        buffers.codes.covers(op.code_bytes(rows)?)?;
        buffers.scales.covers(op.scale_bytes(rows)?)?;
        buffers.live_rows.covers(4)?;
        buffers.output.covers(op.output_bytes(rows)?)?;
        let plan = self.decode.plan(grid)?;
        let arguments = self.decode.arguments(
            buffers.codes.handle,
            buffers.scales.handle,
            buffers.live_rows.handle,
            buffers.output.handle,
        )?;
        Ok(vec![Launch {
            symbol: self.decode.symbol().into(),
            rows: plan.grid[0],
            plan,
            arguments,
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use crate::row::mxfp8_decode::{CUBIN, SYMBOL, cubin_spec};
    use kernel_core::backend::Backend;
    use kernel_core::contract::Argument;

    fn buffers(rows: u64, width: u64) -> Mxfp8RowsBuffers<View> {
        Mxfp8RowsBuffers {
            codes: view(1, rows * width),
            scales: view(2, rows * width / 32),
            live_rows: view(3, 4),
            output: view(4, rows * width * 2),
        }
    }

    #[test]
    fn decode_prepares_through_dyn_backend_and_checks_every_view_and_count() {
        let backend = Sm120::new().with_mxfp8_rows(SYMBOL);
        let backend: &dyn Backend = &backend;
        for width in [32, 128, 256, 6144, u32::MAX - 31] {
            let op = Mxfp8Rows { width };
            let need = Need::rows(32).with_exact(Invariance::any_rows(32));
            let prepared = backend.decode_mxfp8_rows(&op, need).unwrap();
            let buffers = buffers(5, u64::from(width));
            let counts = Counts {
                padded_rows: 32,
                ..Counts::rows(5)
            };
            let launches = prepared.launches(&buffers, View::EMPTY, counts).unwrap();
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].symbol, SYMBOL);
            assert_eq!(launches[0].plan.grid, [32, 1, 1]);
            assert_eq!(launches[0].plan.block, [256, 1, 1]);
            assert!(launches[0].plan.pdl);
            assert_eq!(
                launches[0].arguments.0,
                [
                    Argument::DeviceBuffer(1),
                    Argument::DeviceBuffer(2),
                    Argument::DeviceBuffer(3),
                    Argument::DeviceBuffer(4),
                    Argument::U32(width),
                ]
            );
            for bad in [
                Mxfp8RowsBuffers {
                    codes: view(1, buffers.codes.bytes - 1),
                    ..buffers
                },
                Mxfp8RowsBuffers {
                    scales: view(2, buffers.scales.bytes - 1),
                    ..buffers
                },
                Mxfp8RowsBuffers {
                    live_rows: view(3, 3),
                    ..buffers
                },
                Mxfp8RowsBuffers {
                    output: view(4, buffers.output.bytes - 1),
                    ..buffers
                },
                Mxfp8RowsBuffers {
                    output: view(1, buffers.output.bytes),
                    ..buffers
                },
            ] {
                assert!(prepared.launches(&bad, View::EMPTY, counts).is_err());
            }
            for counts in [
                Counts::rows(0),
                Counts::rows(33),
                Counts {
                    padded_rows: 4,
                    ..Counts::rows(5)
                },
            ] {
                assert!(prepared.launches(&buffers, View::EMPTY, counts).is_err());
            }
        }
        for capacity in [0, MAX_CAPACITY + 1] {
            assert!(
                backend
                    .decode_mxfp8_rows(&Mxfp8Rows { width: 128 }, Need::rows(capacity))
                    .is_err()
            );
        }
        assert!(
            backend
                .decode_mxfp8_rows(&Mxfp8Rows { width: 33 }, Need::rows(32))
                .is_err()
        );
        assert_eq!(cubin_spec().name, CUBIN);
        assert_eq!(cubin_spec().entries.len(), 1);
        assert_eq!(cubin_spec().entries[0].params.len(), 5);
    }

    #[test]
    fn unregistered_and_undeclared_decoders_cannot_claim_exactness() {
        let op = Mxfp8Rows { width: 128 };
        let need = Need::rows(32).with_exact(Invariance::any_rows(32));
        assert!(Sm120::new().decode_mxfp8_rows(&op, need).is_err());
        let other = Sm120::new().with_mxfp8_rows("chop_row_mxfp8_other_v1");
        assert!(matches!(
            other.decode_mxfp8_rows(&op, need),
            Err(Refused::Inexact { .. })
        ));
        assert!(other.decode_mxfp8_rows(&op, Need::rows(32)).is_ok());
    }
}
