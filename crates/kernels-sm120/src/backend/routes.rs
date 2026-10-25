//! Route metadata derived from the same entries and row admissions as prepare.
use super::Sm120;
use kernel_core::backend::{Need, Prepared};
use kernel_core::route::{RouteClass, RouteTag};

impl Sm120 {
    /// Additional model-owned wrappers, registered beside the shared entries.
    pub fn with_route_tags(mut self, tags: impl IntoIterator<Item = RouteTag>) -> Self {
        self.route_tags.extend(tags);
        self
    }

    pub fn route_tags(&self) -> Vec<RouteTag> {
        let mut tags = self.route_tags.clone();
        // fused_all_reduce::push_symbol selects these shared entries for its
        // continuous decode range, including exact uses at low-latency rows.
        for symbol in [
            crate::comm::DECODE_SYMBOL_V2,
            crate::comm::DECODE_COMBINE_SYMBOL,
        ] {
            tags.push(RouteTag::new(
                symbol,
                RouteClass::Fast,
                "fused_all_reduce",
                [1..=crate::comm::DECODE_MAX_ROWS],
            ));
        }
        if let Some(offers) = &self.f32_projections {
            tags.extend(offers.route_tags());
        }
        self.prefill_route_tags(&mut tags);
        self.sinkhorn_route_tags(&mut tags);
        tags
    }

    fn prefill_route_tags(&self, tags: &mut Vec<RouteTag>) {
        use crate::gemm::mxfp8_fold::Entry;
        use kernel_core::backend::View;
        use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
        let symbols = |op: Prepared<Buffers<View>>| {
            self.fold_weights(op)
                .expect("valid fold layout")
                .info()
                .symbols
                .iter()
                .map(|(s, _)| s.clone())
                .collect::<Vec<_>>()
        };
        let mut register = |base: &Entry, split| {
            let q = &base.query;
            let mut admitted = Vec::new();
            let mut wrappers = Vec::new();
            let mut fast = |rows, wrapper, op| {
                admitted.push(rows..=rows);
                wrappers.push(wrapper);
                tags.extend(symbols(op).into_iter().map(|symbol| {
                    RouteTag::new(symbol, RouteClass::Fast, "mxfp8_prefill", [rows..=rows])
                        .with_wrapper(wrapper)
                }));
            };
            for (e, r) in self
                .mxfp8_prefill_compacts
                .iter()
                .filter(|(e, _)| e.base.query == *q)
            {
                fast(
                    *r,
                    "compact",
                    e.prepare(*r).expect("registered compact route"),
                );
            }
            for (e, r) in self
                .mxfp8_prefill_matrices
                .iter()
                .filter(|(e, _)| e.base.query == *q)
            {
                fast(
                    *r,
                    "matrix",
                    e.prepare(*r).expect("registered matrix route"),
                );
            }
            for (e, r) in self
                .mxfp8_prefill_tma_matrices
                .iter()
                .filter(|(e, _)| e.base.query == *q)
            {
                fast(
                    *r,
                    "TMA-matrix",
                    e.prepare(*r).expect("registered TMA matrix route"),
                );
            }
            for (e, r) in self
                .mxfp8_prefill_tma_scales
                .iter()
                .filter(|(e, _)| e.base.query == *q)
            {
                fast(
                    *r,
                    "TMA-scales",
                    e.prepare(*r).expect("registered TMA scales route"),
                );
            }
            if admitted.is_empty() {
                return;
            }
            let base = super::mxfp8_prefill::prepare_with_tail(
                std::slice::from_ref(base),
                q,
                Need::rows(q.m_capacity),
                split,
            )
            .expect("registered prefill fallback");
            for symbol in symbols(base) {
                let mut tag = RouteTag::new(
                    symbol,
                    RouteClass::Baseline,
                    "mxfp8_prefill",
                    admitted.clone(),
                );
                tag.wrappers = wrappers.clone();
                tags.push(tag);
            }
        };
        for base in &self.mxfp8_prefills {
            register(base, None);
        }
        for (base, policy) in &self.mxfp8_prefill_splits {
            register(base, Some(*policy));
        }
    }

    fn sinkhorn_route_tags(&self, tags: &mut Vec<RouteTag>) {
        // A prefill wrapper replaces the complete lagged mixer at its rows.
        for (entry, rows) in &self.sinkhorn.lagged_prefill {
            let need = Need::rows(
                entry
                    .reference
                    .capacity()
                    .expect("registered mixer capacity"),
            );
            let inner =
                super::sinkhorn_lagged::mix_collapse(&self.sinkhorn, &entry.reference.query, need)
                    .expect("registered prefill mixer fallback");
            for (symbol, _) in &inner.info().symbols {
                tags.push(kernel_core::route::branch(
                    symbol,
                    RouteClass::Baseline,
                    "sinkhorn_mix",
                    "prefill-fallback",
                    rows.iter().map(|&r| r..=r),
                ));
            }
            tags.push(kernel_core::route::branch(
                &entry.reference.symbol,
                RouteClass::Fast,
                "sinkhorn_mix",
                "prefill",
                rows.iter().map(|&r| r..=r),
            ));
        }
        for entry in &self.sinkhorn.lagged_warp {
            tags.push(
                RouteTag::new(
                    entry.symbol(),
                    RouteClass::Fast,
                    "sinkhorn_mix",
                    entry.buckets.iter().map(|&r| r..=r),
                )
                .with_wrapper("warp"),
            );
        }
        for (entry, rows) in &self.finish_norm {
            for symbol in [
                entry.mixer.symbols()[1].clone(),
                entry.norm.symbols()[2].clone(),
            ] {
                tags.push(kernel_core::route::branch(
                    &symbol,
                    RouteClass::Baseline,
                    "sinkhorn_finish_norm",
                    "unfused-tail",
                    rows.iter().map(|&r| r..=r),
                ));
            }
            tags.push(kernel_core::route::branch(
                &entry.symbols()[1],
                RouteClass::Fast,
                "sinkhorn_finish_norm",
                "fused",
                rows.iter().map(|&r| r..=r),
            ));
        }
    }
}

/// Capacity is the row admission; every other contract field must match.
pub(super) fn same_fold_shape(
    mut a: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
    mut b: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery,
) -> bool {
    a.m_capacity = 0;
    b.m_capacity = 0;
    a == b
}
