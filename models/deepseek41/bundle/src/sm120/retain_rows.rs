use super::retain_rows_entries::{SYMBOL, THREADS};
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::own_ops::retain_rows::{CONTRACT, RetainRows, RetainRowsBuffers};

const TILES: u32 = 8;

pub fn prepare(op: RetainRows, need: Need) -> Result<Prepared<RetainRowsBuffers<View>>, Refused> {
    op.validate(need)?;
    let info = OpInfo::new(
        CONTRACT,
        Invariance::any_rows(need.capacity),
        vec![(SYMBOL.into(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Copy {
        op,
        capacity: need.capacity,
        info,
    }))
}

struct Copy {
    op: RetainRows,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<RetainRowsBuffers<View>> for Copy {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &RetainRowsBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows == 0 || counts.rows > self.capacity {
            return Err(Refusal("retention segment count exceeds capacity"));
        }
        buffers.source.covers(self.op.source_bytes())?;
        buffers.destination.covers(self.op.destination_bytes())?;
        buffers.segments.covers(u64::from(counts.rows) * 12)?;
        if buffers.source.handle == buffers.destination.handle
            || buffers.source.handle == buffers.segments.handle
            || buffers.destination.handle == buffers.segments.handle
        {
            return Err(Refusal("retention buffers alias"));
        }
        Ok(vec![Launch {
            symbol: SYMBOL.into(),
            rows: counts.rows,
            plan: LaunchPlan {
                grid: [counts.rows, TILES, 1],
                block: [THREADS, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1; 3],
                pdl: true,
            },
            arguments: ArgumentList(vec![
                Argument::DeviceBuffer(buffers.source.handle),
                Argument::DeviceBuffer(buffers.segments.handle),
                Argument::DeviceBuffer(buffers.destination.handle),
                Argument::U32(self.op.row_bytes),
                Argument::U32(self.op.source_rows),
                Argument::U32(self.op.destination_rows),
            ]),
            collective: false,
            persistent: false,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retained_rows_bind_one_pdl_launch_and_refuse_invalid_views() {
        let op = RetainRows {
            row_bytes: 30720,
            source_rows: 192,
            destination_rows: 32 * 134,
        };
        let prepared = prepare(op, Need::rows(64)).unwrap();
        let buffers = RetainRowsBuffers {
            source: View {
                handle: 1,
                bytes: op.source_bytes(),
            },
            segments: View {
                handle: 2,
                bytes: 64 * 12,
            },
            destination: View {
                handle: 3,
                bytes: op.destination_bytes(),
            },
        };
        let workspace = View {
            handle: 0,
            bytes: 0,
        };
        let calls = prepared
            .launches(&buffers, workspace, Counts::rows(64))
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].plan.grid, [64, 8, 1]);
        assert!(calls[0].plan.pdl);
        for rows in [0, 65] {
            assert!(
                prepared
                    .launches(&buffers, workspace, Counts::rows(rows))
                    .is_err()
            );
        }
        for bad in [
            RetainRowsBuffers {
                source: View {
                    bytes: op.source_bytes() - 1,
                    ..buffers.source
                },
                ..buffers
            },
            RetainRowsBuffers {
                destination: View {
                    bytes: op.destination_bytes() - 1,
                    ..buffers.destination
                },
                ..buffers
            },
            RetainRowsBuffers {
                segments: View {
                    bytes: 64 * 12 - 1,
                    ..buffers.segments
                },
                ..buffers
            },
            RetainRowsBuffers {
                destination: View {
                    handle: 1,
                    ..buffers.destination
                },
                ..buffers
            },
        ] {
            assert!(
                prepared
                    .launches(&bad, workspace, Counts::rows(64))
                    .is_err()
            );
        }
        let (name, _) =
            crate::sm120::fold_cubin_for_symbol(SYMBOL).expect("embedded retained-row kernel");
        assert_eq!(name, crate::sm120::retain_rows_entries::CUBIN);
    }
}
