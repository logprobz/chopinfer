//! Bind the measured whole-K exact-F32 row tiles of verify steps.
use super::f32_verify_rows::{self, Served};
use crate::ops::{
    Bind, Lowered,
    exact_f32::rows::{RowsProjection, Spans},
};
use kernel_core::backend::{OpInfo, Refused};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use kernels_sm120::f32_projection::rows::Buffers;
use model_deepseek41_contract::verify_bounds::F32_ROWS;

struct Projection {
    tiles: Vec<Served>,
    rows: [u32; 2],
    info: OpInfo,
}
impl Projection {
    fn tile(&self, rows: u32) -> Result<&Served, Refusal> {
        self.tiles
            .iter()
            .find(|t| (t.rows[0]..=t.rows[1]).contains(&rows))
            .filter(|_| (self.rows[0]..=self.rows[1]).contains(&rows))
            .ok_or(Refusal("no measured row tile serves these rows"))
    }
}
impl RowsProjection for Projection {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn admits(&self, rows: u32) -> bool {
        self.tile(rows).is_ok()
    }
    fn workspace_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.tile(rows)?.candidate.workspace_bytes(rows)
    }
    fn lower(&self, rows: u32, spans: Spans) -> Result<Vec<Lowered>, Refusal> {
        spans.validate()?;
        let served = self.tile(rows)?;
        let mut bind = Bind::default();
        let buffers = Buffers {
            activation: bind.view(spans.activation),
            weight: bind.view(spans.weight),
            workspace: bind.view(spans.workspace),
            live_rows: bind.view(spans.live_rows),
            weight_domain: bind.view(spans.weight_domain),
            output: bind.view(spans.output),
        };
        let launches = served.candidate.launches(buffers, rows)?;
        let mut calls = bind.lowered_launches(launches);
        if kernel_core::route::Metadata::enabled() {
            for call in &mut calls {
                let tag = super::wide_routes::tag(&call.launch.symbol, "f32_rows", served.rows[0]);
                call.launch.bindings.set_route(Some(tag));
            }
        }
        Ok(calls)
    }
}
impl crate::Kernels {
    /// The measured row tiles of the projection of `n` outputs from `k`
    /// inputs, admitted at `rows` only, prepared once before lowering.
    /// Selecting them still requires the caller's verify phase and row gate.
    pub(crate) fn rows_verify_projection(
        n: u32,
        k: u32,
        rows: [u32; 2],
    ) -> Result<Box<dyn RowsProjection>, Refused> {
        let tiles: Vec<_> = f32_verify_rows::tiles(n, k)?
            .into_iter()
            .filter(|t| t.rows[0] <= rows[1] && rows[0] <= t.rows[1])
            .collect();
        if tiles.is_empty() {
            return Err(Refused::Unsupported);
        }
        let mut symbols: Vec<(String, u32)> = Vec::new();
        let mut workspace = 0;
        for t in &tiles {
            let shared = t.candidate.tile().shared_bytes();
            for symbol in t.candidate.symbols() {
                if super::fold_cubin_for_symbol(&symbol).is_none() {
                    return Err(Refused::Rule("exact F32 row tile cubin is missing".into()));
                }
                let bytes = if symbol == t.candidate.symbol() {
                    shared
                } else {
                    0
                };
                if !symbols.iter().any(|(s, _)| *s == symbol) {
                    symbols.push((symbol, bytes));
                }
            }
            workspace = workspace.max(t.candidate.workspace_bytes(t.rows[1].min(rows[1]))?);
        }
        let mut info = OpInfo::new(
            "deepseek41.verify.exact-f32.rows",
            Invariance::any_rows(F32_ROWS),
            symbols,
        );
        info.workspace = workspace;
        Ok(Box::new(Projection { tiles, rows, info }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};

    fn span(handle: u64, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(handle), offset, length)
    }
    fn spans(n: u64, rows: u32, workspace: u64) -> Spans {
        Spans {
            activation: span(1, 0, u64::from(rows) * 5120 * 4),
            weight: span(2, 0, n * 5120 * 2),
            workspace: span(3, 0, workspace),
            live_rows: span(4, 0, 4),
            weight_domain: span(5, 0, n.div_ceil(16) * 4),
            output: span(6, 0, u64::from(rows) * n * 4),
        }
    }

    #[test]
    fn default_instances_do_not_load_the_row_tiles() {
        use bundle_runtime::geometry::Ranks;
        use model_deepseek41_contract::geometry::FLASH;
        let tp4 = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        for rows in [32, 64, 65, 128, 192] {
            for wide in [false, true] {
                let shape = tp4.with_verify_rows(rows).with_wide_verify(wide);
                assert_eq!(
                    crate::Kernels::loads(f32_verify_rows::CUBIN, &shape),
                    wide && rows > 64
                );
            }
        }
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP2)
            .unwrap()
            .with_verify_rows(192)
            .with_wide_verify(true);
        assert!(!crate::Kernels::loads(f32_verify_rows::CUBIN, &shape));
    }

    /// Each projection's launches are its tile's preparation, compute and,
    /// with K slices, reducer, each tagged with the rows its tuning row
    /// admits; the compute's shared memory and the widest step's workspace
    /// are declared once.
    #[test]
    fn row_tiles_lower_their_stages_with_declared_resources() {
        let _metadata = kernel_core::route::Metadata::enable();
        for (n, stages, workspace) in [
            (384_u32, 3, 192 * 5120 * 6 + 192 * 384 * 40 * 4 + 48 * 16),
            (512, 3, 192 * 5120 * 6 + 192 * 512 * 40 * 4 + 48 * 16),
            (32320, 2, 192 * 5120 * 6 + 48 * 16),
        ] {
            let op = crate::Kernels::rows_verify_projection(n, 5120, [65, 192]).unwrap();
            assert_eq!(op.info().workspace, workspace);
            let compute: Vec<_> = op
                .info()
                .symbols
                .iter()
                .filter(|(_, shared)| *shared > 48 << 10)
                .collect();
            assert_eq!(compute.len(), 1);
            assert!(!op.admits(64));
            for rows in 65..=192 {
                assert!(op.admits(rows));
                let bytes = op.workspace_bytes(rows).unwrap();
                let calls = op.lower(rows, spans(n.into(), rows, bytes)).unwrap();
                assert_eq!(calls.len(), stages);
                assert!(calls[0].launch.symbol.ends_with("_rows_prepare_v1"));
                assert_eq!(calls[1].launch.symbol, compute[0].0);
                assert_eq!(calls[1].launch.plan.dynamic_shared_bytes, compute[0].1);
                for call in &calls {
                    assert_eq!(call.launch.rows, rows);
                    let tag = call.launch.bindings.route().expect("selected row tile");
                    assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
                    assert_eq!(tag.family, "f32_rows");
                    assert!(tag.admitted.iter().any(|range| range.contains(&rows)));
                }
                let short = spans(n.into(), rows, bytes - 1);
                assert!(op.lower(rows, short).is_err());
                let mut wrong = spans(n.into(), rows, bytes);
                wrong.workspace.offset = 16;
                assert!(op.lower(rows, wrong).is_err());
                let mut wrong = spans(n.into(), rows, bytes);
                wrong.output.buffer = wrong.activation.buffer;
                assert!(op.lower(rows, wrong).is_err());
            }
        }
        // The router keeps its separate spans at 33 to 64 rows.
        assert!(matches!(
            crate::Kernels::rows_verify_projection(384, 5120, [33, 64]),
            Err(Refused::Unsupported)
        ));
        assert!(matches!(
            crate::Kernels::rows_verify_projection(32, 5120, [65, 192]),
            Err(Refused::Unsupported)
        ));
    }
}
