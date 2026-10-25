//! Bind the bounded exact-F32 stages through explicit, checked spans.
use crate::ops::{
    Bind,
    exact_f32::cached::{CachedProjection, CachedProjectionStages, Spans},
};
use kernel_core::backend::{OpInfo, Refused};
use kernel_core::contract::Refusal;
use kernel_core::properties::Invariance;
use kernels_sm120::f32_projection::cached_spans::{Buffers, Candidate, DOMAIN_BYTES};

struct Projection {
    candidate: Candidate,
    info: OpInfo,
}
impl CachedProjection for Projection {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn partial_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.candidate.partial_bytes(rows)
    }
    fn lower(&self, rows: u32, spans: Spans) -> Result<CachedProjectionStages, Refusal> {
        spans.validate()?;
        let mut bind = Bind::default();
        let buffers = Buffers {
            activation: bind.view(spans.activation),
            weight: bind.view(spans.weight),
            row_domain: bind.view(spans.row_domain),
            partials: bind.view(spans.partials),
            live_rows: bind.view(spans.live_rows),
            weight_domain: bind.view(spans.weight_domain),
            output: bind.view(spans.output),
        };
        let stages = self.candidate.stages(buffers, rows)?;
        let mut calls = bind.lowered_launches(stages.into_launches().into());
        if kernel_core::route::Metadata::enabled() {
            for call in &mut calls {
                let tag = super::wide_routes::tag(&call.launch.symbol, "f32_projection", 33);
                call.launch.bindings = call.launch.bindings.with_route(Some(tag));
            }
        }
        CachedProjectionStages::new(calls, self.candidate.symbols(), rows, spans)
    }
}
impl crate::Kernels {
    /// Prepare once, before lowering. Selecting the wide router or standalone
    /// projection still requires the caller's qualified phase and row gate.
    pub(crate) fn cached_verify_projection(
        n: u32,
        k: u32,
    ) -> Result<Box<dyn CachedProjection>, Refused> {
        let candidate = super::f32_verify::entries()?
            .into_iter()
            .find(|c| c.entry().query().n == n && c.entry().query().k == k)
            .ok_or(Refused::Unsupported)?;
        let symbols = candidate.symbols();
        if symbols
            .iter()
            .any(|s| super::fold_cubin_for_symbol(s).is_none())
        {
            return Err(Refused::Rule(
                "wide cached projection cubin is missing".into(),
            ));
        }
        let mut info = OpInfo::new(
            "deepseek41.verify.exact-f32.spans",
            Invariance::any_rows(64),
            symbols.into_iter().map(|s| (s, 0)).collect(),
        );
        info.workspace = DOMAIN_BYTES + candidate.partial_bytes(64)?;
        Ok(Box::new(Projection { candidate, info }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime_bundle_abi::{BundleDeviceBufferHandleV1, DeviceBufferSpanV1};
    fn span(handle: u64, offset: u64, length: u64) -> DeviceBufferSpanV1 {
        DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_raw(handle), offset, length)
    }
    fn spans(rows: u32) -> Spans {
        Spans {
            activation: span(1, 0, u64::from(rows) * 5120 * 4),
            weight: span(2, 0, 384 * 5120 * 2),
            row_domain: span(3, 0, 256),
            partials: span(3, 256, u64::from(rows) * 61440),
            live_rows: span(4, 0, 4),
            weight_domain: span(5, 0, 96),
            output: span(6, 0, u64::from(rows) * 384 * 4),
        }
    }
    #[test]
    fn default_instances_do_not_load_the_wide_family() {
        use bundle_runtime::geometry::Ranks;
        use model_deepseek41_contract::geometry::FLASH;
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP4).unwrap();
        for rows in [32, 33, 48, 64, 128] {
            let shape = shape.with_verify_rows(rows);
            assert_eq!(
                crate::Kernels::loads(super::super::f32_verify::CUBIN, &shape),
                rows > 32
            );
        }
        let shape = crate::shape::Shape::new(&FLASH, Ranks::TP2)
            .unwrap()
            .with_verify_rows(64);
        assert!(!crate::Kernels::loads(
            super::super::f32_verify::CUBIN,
            &shape
        ));
    }

    #[test]
    fn router_keeps_three_named_stages_and_disjoint_domain_span() {
        let _metadata = kernel_core::route::Metadata::enable();
        let op = crate::Kernels::cached_verify_projection(384, 5120).unwrap();
        assert_eq!(op.info().workspace, 256 + 64 * 61440);
        for rows in 33..=64 {
            let s = spans(rows);
            assert_eq!(op.partial_bytes(rows).unwrap(), s.partials.length);
            let calls = op.lower(rows, s).unwrap().into_launches();
            assert_eq!(calls.len(), 3);
            for call in &calls {
                let tag = call
                    .launch
                    .bindings
                    .route()
                    .expect("selected cached projection entry");
                assert_eq!(tag.class, kernel_core::route::RouteClass::Fast);
                assert_eq!(tag.family, "f32_projection");
                assert!(tag.admitted.iter().any(|range| range.contains(&rows)));
            }
            assert!(calls[0].launch.symbol.ends_with("_rows_v1"));
            assert!(calls[1].launch.symbol.ends_with("_compute_v1"));
            assert!(calls[2].launch.symbol.ends_with("_reduce_v1"));
            assert!(
                op.lower(rows, s)
                    .unwrap()
                    .into_producer(rows, s.row_domain, s.partials)
                    .is_ok()
            );
            assert!(
                op.lower(rows, s)
                    .unwrap()
                    .into_producer(rows - 1, s.row_domain, s.partials)
                    .is_err()
            );
            let mut wrong = s;
            wrong.partials.offset = 128;
            assert!(op.lower(rows, wrong).is_err());
            let mut wrong = s;
            wrong.partials.offset = 0;
            assert!(op.lower(rows, wrong).is_err());
            let mut wrong = s;
            wrong.partials.length -= 1;
            assert!(op.lower(rows, wrong).is_err());
            let mut wrong = s;
            wrong.row_domain.length = 128;
            assert!(op.lower(rows, wrong).is_err());
            let mut wrong = s;
            wrong.activation.offset = u64::MAX - 15;
            assert!(op.lower(rows, wrong).is_err());
        }
        for rows in [0, 1, 32, 65] {
            assert!(op.lower(rows, spans(rows)).is_err());
        }
    }
}
