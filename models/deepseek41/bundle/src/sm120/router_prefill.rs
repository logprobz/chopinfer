//! Prefill-only wrapper; initialization and rows below 320 use the base.
use crate::sm120::f32_zero as family;
use kernel_core::{
    backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{Argument, ArgumentList, LaunchPlan, Refusal},
    precision::buffers::ProjectionBuffers,
};
use kernels_sm120::f32_projection::zero_prefill as zero;
pub(crate) fn prepare(
    shape: &crate::shape::Shape,
    base: Prepared<ProjectionBuffers<View>>,
) -> Result<Prepared<ProjectionBuffers<View>>, Refused> {
    let e = family::entry();
    let mut info = base.info().clone();
    if shape.verify_rows >= family::ROWS[0] {
        return Err(Refusal("router prefill buckets overlap verification rows").into());
    }
    info.implementation.push_str("+certified_bf16_prefill_v1");
    info.workspace = info.workspace.max(zero::workspace_bytes(&e, 8192));
    info.symbols.extend([
        (zero::prepare_symbol(&e), 0),
        (
            zero::symbol(&e, 32),
            zero::plan(&e, 32, 8192)
                .map_err(Refused::from)?
                .dynamic_shared_bytes,
        ),
    ]);
    Ok(Box::new(Router { base, info }))
}
struct Router {
    base: Prepared<ProjectionBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<ProjectionBuffers<View>> for Router {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn initialize(&self, b: &ProjectionBuffers<View>) -> Result<Vec<Launch>, Refusal> {
        self.base.initialize(b)
    }
    fn launches(
        &self,
        b: &ProjectionBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let old = self.base.launches(b, workspace, counts)?;
        if !(family::ROWS[0]..=8192).contains(&counts.rows) {
            return Ok(old);
        }
        let e = family::entry();
        workspace.covers(zero::workspace_bytes(&e, counts.rows))?;
        let domain = b.scratch.weight_domain.ok_or(Refusal(
            "router prefill requires initialized weight admission",
        ))?;
        domain.covers(e.weight_domain_bytes())?;
        if workspace.handle == 0
            || [b.activation, b.weights.codes, b.output, b.live_rows, domain]
                .iter()
                .any(|v| v.handle == workspace.handle)
        {
            return Err(Refusal("router workspace aliases an operand"));
        }
        let ptr = |v: View| Argument::DeviceBuffer(v.handle);
        let make = |symbol, plan, arguments| Launch {
            symbol,
            rows: counts.rows,
            plan,
            arguments: ArgumentList(arguments),
            collective: false,
            persistent: false,
        };
        Ok(vec![
            make(
                zero::prepare_symbol(&e),
                LaunchPlan {
                    grid: [counts.rows, 1, 1],
                    block: [128, 1, 1],
                    dynamic_shared_bytes: 0,
                    workspace_bytes: 0,
                    cluster: [1; 3],
                    pdl: true,
                },
                vec![
                    ptr(b.activation),
                    ptr(b.live_rows),
                    Argument::U32(counts.rows),
                    ptr(workspace),
                ],
            ),
            make(
                zero::symbol(&e, 32),
                zero::plan(&e, 32, counts.rows)?,
                vec![
                    ptr(b.activation),
                    ptr(b.weights.codes),
                    ptr(workspace),
                    ptr(b.output),
                    ptr(b.live_rows),
                    Argument::U32(counts.rows),
                    ptr(domain),
                ],
            ),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        backend::{Backend, Need},
        precision::buffers::{ProjectionScratch, QuantizedWeights},
    };
    use kernels_sm120::backend::Sm120;
    use model_deepseek41_contract::geometry::FLASH;
    fn base() -> Prepared<ProjectionBuffers<View>> {
        let backend = crate::sm120::register_f32_projections(Sm120::new(), &FLASH, 4).unwrap();
        let q = crate::ops::exact_f32::query(384, 5120);
        let op = crate::ops::exact_f32::operation("moe.router", "ffn.gate", q).unwrap();
        backend
            .projection(&op, crate::ops::exact_f32::USE, Need::rows(8192))
            .unwrap()
            .handle
    }
    fn buffers() -> ProjectionBuffers<View> {
        let v = |handle, bytes| View { handle, bytes };
        ProjectionBuffers {
            activation: v(1, 8192 * 5120 * 4),
            live_rows: v(2, 4),
            output: v(3, 8192 * 384 * 4),
            residual: None,
            weights: QuantizedWeights {
                codes: v(4, 384 * 5120 * 2),
                local_scales: None,
                global_scales: None,
            },
            activation_global: None,
            routing: None,
            scratch: ProjectionScratch {
                weight_domain: Some(v(5, 384 / 16 * 4)),
                partial: v(6, 1),
                semaphores: v(7, 1),
                unused: v(8, 1),
                temporary: None,
                gathered: None,
                codes: None,
                scales: None,
                activation_global: None,
                row_amax: None,
                output_codes: None,
                output_scales: None,
                quantizer_status: None,
                quantizer_status_row_offset: 0,
            },
            row_offset: 0,
            shard_rows: 0,
        }
    }
    fn tp4() -> crate::shape::Shape {
        crate::shape::Shape::new(&FLASH, bundle_runtime::geometry::Ranks::TP4).unwrap()
    }
    /// The widest verify rows stay below the first measured prefill count.
    #[test]
    fn verify_rows_below_the_measured_counts_are_admitted() {
        assert!(prepare(&tp4().with_verify_rows(192), base()).is_ok());
        assert!(prepare(&tp4().with_verify_rows(319), base()).is_ok());
        assert!(prepare(&tp4().with_verify_rows(family::ROWS[0]), base()).is_err());
    }
    #[test]
    fn prefill_range_preserves_base_initialization_and_other_launches() {
        let old = base();
        let new = prepare(&tp4(), base()).unwrap();
        let b = buffers();
        let work = View {
            handle: 10,
            bytes: new.info().workspace,
        };
        assert_eq!(old.initialize(&b).unwrap(), new.initialize(&b).unwrap());
        for rows in 1..=8192 {
            let counts = Counts::rows(rows);
            let launches = new.launches(&b, work, counts).unwrap();
            if (320..=8192).contains(&rows) {
                assert_eq!(launches.len(), 2);
                assert_eq!(launches[0].symbol, zero::prepare_symbol(&family::entry()));
                assert_eq!(launches[1].symbol, zero::symbol(&family::entry(), 32));
                assert_eq!(launches[1].plan.dynamic_shared_bytes, 32768);
            } else {
                assert_eq!(launches, old.launches(&b, work, counts).unwrap());
            }
        }
        assert_eq!(family::ROWS.len(), 27);
    }
    #[test]
    fn changed_geometry_bad_buffers_and_aliases_are_refused() {
        assert!(family::selected(&FLASH, 4, 384));
        assert!(!family::selected(&FLASH, 2, 384));
        assert!(!family::selected(&FLASH, 4, 128));
        let op = prepare(&tp4(), base()).unwrap();
        let b = buffers();
        let work = View {
            handle: 10,
            bytes: op.info().workspace,
        };
        for rows in [0, 8193] {
            assert!(op.launches(&b, work, Counts::rows(rows)).is_err());
        }
        for handle in [0, 1, 2, 3, 4, 5] {
            assert!(
                op.launches(&b, View { handle, ..work }, Counts::rows(8192))
                    .is_err()
            );
        }
        assert!(
            op.launches(
                &b,
                View {
                    bytes: zero::workspace_bytes(&family::entry(), 8192) - 1,
                    ..work
                },
                Counts::rows(8192)
            )
            .is_err()
        );
        let mut bad = b;
        bad.scratch.weight_domain = None;
        assert!(op.launches(&bad, work, Counts::rows(8192)).is_err());
    }
}
