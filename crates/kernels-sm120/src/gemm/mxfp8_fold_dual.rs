//! One launch for two independent native folds sharing their quantized input.
use super::mxfp8_fold_stream;
use crate::backend::Sm120;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::{Backend, Counts, Launch, Need, Refused, View};
use kernel_core::contract::{Argument as A, ArgumentList, DType, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldStreamBuffers;

#[derive(Clone, Debug)]
pub struct Entry {
    pub projections: [mxfp8_fold_stream::Entry; 2],
    pub symbol: String,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        for e in &self.projections {
            e.validate()?;
        }
        let [a, b] = &self.projections;
        let x = a.ticket.query;
        let y = b.ticket.query;
        if a.ticket.paired
            || b.ticket.paired
            || a.defer_admission
            || b.defer_admission
            || a.ticket.tile_n() != b.ticket.tile_n()
            || (
                x.k,
                x.groups,
                x.activation_row_stride,
                x.activation_group_stride,
                x.output,
                x.m_capacity,
            ) != (
                y.k,
                y.groups,
                y.activation_row_stride,
                y.activation_group_stride,
                y.output,
                y.m_capacity,
            )
            || self.symbol.is_empty()
        {
            return Err(Refusal(
                "dual folds require matching input, output, capacity and CTA contracts",
            ));
        }
        Ok(())
    }

    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let mut total = 0u64;
        for e in &self.projections {
            let p = e.ticket.plans(rows)?[1];
            total += p.grid.into_iter().map(u64::from).product::<u64>();
        }
        let plan = LaunchPlan {
            grid: [
                u32::try_from(total).map_err(|_| Refusal("dual grid exceeds u32"))?,
                1,
                1,
            ],
            block: [self.projections[0].ticket.tile_n() * 4, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub fn launch(
        &self,
        b: &[Mxfp8FoldStreamBuffers<View>; 2],
        work: [View; 2],
        rows: u32,
        delayed: bool,
    ) -> Result<Launch, Refused> {
        self.validate()?;
        let first = b[0].first;
        let second = b[1].first;
        if (
            first.activations,
            first.activation_scales,
            first.live_rows,
            first.status,
            first.status_row_offset,
        ) != (
            second.activations,
            second.activation_scales,
            second.live_rows,
            second.status,
            second.status_row_offset,
        ) {
            return Err(Refusal("dual projections must share input and row status spans").into());
        }
        let mut mutable = Vec::new();
        let mut immutable = vec![
            first.activations.handle,
            first.activation_scales.handle,
            first.live_rows.handle,
            first.status.handle,
        ];
        for i in 0..2 {
            let e = &self.projections[i];
            let backend = Sm120::new().with_mxfp8_fold_stream(e.clone());
            let op =
                backend.mxfp8_fold_gemm_stream(&e.ticket.query, false, Need::rows(rows.max(1)))?;
            op.launches(&b[i], work[i], Counts::rows(rows))?;
            mutable.extend([b[i].first.output.handle, b[i].state.handle, work[i].handle]);
            immutable.extend([
                b[i].first.weights.handle,
                b[i].first.weight_scales.handle,
                b[i].packed_weights[0].handle,
            ]);
        }
        for (i, h) in mutable.iter().enumerate() {
            if immutable.contains(h) || mutable[..i].contains(h) {
                return Err(
                    Refusal("dual projections require disjoint outputs and scratch").into(),
                );
            }
        }
        let d = |v: View| A::DeviceBuffer(v.handle);
        Ok(Launch {
            symbol: if delayed {
                format!("{}_delayed_v1", self.symbol)
            } else {
                self.symbol.clone()
            },
            plan: self.plan(rows)?,
            rows,
            collective: false,
            persistent: false,
            arguments: ArgumentList(vec![
                d(first.activations),
                d(first.activation_scales),
                d(b[0].packed_weights[0]),
                d(first.weight_scales),
                d(first.output),
                d(work[0]),
                d(b[0].state),
                d(b[1].packed_weights[0]),
                d(second.weight_scales),
                d(second.output),
                d(work[1]),
                d(b[1].state),
                d(first.live_rows),
                d(first.status),
                A::U64(first.status_row_offset),
                A::U32(rows),
                A::U32(self.projections[0].ticket.folds_per_cta(rows)),
                A::U32(self.projections[1].ticket.folds_per_cta(rows)),
            ]),
        })
    }

    fn spec(&self, delayed: bool) -> EntrySpec {
        let q = self.projections[0].ticket.query;
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let u32p = |name: &str| EntryParam::U32 { name: name.into() };
        let parameters = vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("packed0", "const uint8_t"),
            ptr("scales0", "const uint8_t"),
            ptr("output0", "void"),
            ptr("workspace0", "uint8_t"),
            ptr("state0", "uint32_t"),
            ptr("packed1", "const uint8_t"),
            ptr("scales1", "const uint8_t"),
            ptr("output1", "void"),
            ptr("workspace1", "uint8_t"),
            ptr("state1", "uint32_t"),
            ptr("live_rows", "const uint32_t"),
            ptr("status", "uint32_t"),
            EntryParam::U64 {
                name: "status_row_offset".into(),
            },
            u32p("rows"),
            u32p("folds0"),
            u32p("folds1"),
        ];
        EntrySpec::new(
            if delayed {
                format!("{}_delayed_v1", self.symbol)
            } else {
                self.symbol.clone()
            },
            parameters,
            format!(
                "chop::gemm::mxfp8_fold_dual::entry<{},{},{},{},{},{},{},{},{}>(activations,activation_scales,packed0,scales0,output0,workspace0,state0,packed1,scales1,output1,workspace1,state1,live_rows,status,status_row_offset,rows,folds0,folds1);",
                q.n,
                self.projections[1].ticket.query.n,
                q.k,
                q.groups,
                q.activation_row_stride,
                q.activation_group_stride,
                self.projections[0].ticket.tile_n(),
                q.output == DType::F32,
                delayed
            ),
            self.projections[0].ticket.tile_n() * 4,
            1,
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("dual folds require entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for e in entries {
        e.validate()?;
        if !symbols.insert(e.symbol.clone()) {
            return Err(Refusal("duplicate dual fold symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_dual.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .flat_map(|e| [e.spec(false), e.spec(true)])
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::{Mxfp8FoldGemmBuffers, Mxfp8FoldGemmQuery};
    use kernel_core::precision::execution::BlockScaledExecution;
    fn entry() -> Entry {
        let query = |n| Mxfp8FoldGemmQuery {
            m_capacity: 64,
            n,
            k: 5120,
            groups: 1,
            activation_row_stride: 5120,
            activation_group_stride: 5120,
            output: DType::F32,
            execution: BlockScaledExecution::NativeBlockScaledMma,
        };
        Entry {
            projections: [
                mxfp8_fold_stream::Entry::new(query(1280), "dual_first", false).unwrap(),
                mxfp8_fold_stream::Entry::new(query(512), "dual_second", false).unwrap(),
            ],
            symbol: "dual_test".into(),
        }
    }
    fn views(e: &Entry, rows: u32) -> ([Mxfp8FoldStreamBuffers<View>; 2], [View; 2]) {
        let v = |handle, bytes| View { handle, bytes };
        let b = std::array::from_fn(|i| {
            let q = &e.projections[i];
            let bytes = q.ticket.legacy().unwrap().bytes(rows);
            let h = 10 + i as u64 * 10;
            Mxfp8FoldStreamBuffers {
                first: Mxfp8FoldGemmBuffers {
                    activations: v(1, bytes.activations),
                    activation_scales: v(2, bytes.activation_scales),
                    live_rows: v(3, 4),
                    status: v(4, bytes.status + 12),
                    status_row_offset: 3,
                    weights: v(h, bytes.weights),
                    weight_scales: v(h + 1, bytes.weight_scales),
                    output: v(h + 2, bytes.output),
                },
                second: None,
                packed_weights: [v(h + 3, q.packed_bytes()), View::EMPTY],
                state: v(h + 4, q.state_bytes(rows).unwrap()),
            }
        });
        (
            b,
            std::array::from_fn(|i| {
                v(
                    30 + i as u64,
                    e.projections[i].ticket.workspace(rows).unwrap(),
                )
            }),
        )
    }
    #[test]
    fn preserve_both_task_grids_and_reject_cross_projection_aliases() {
        let e = entry();
        for rows in [1, 4, 8, 16, 32, 64] {
            let (b, work) = views(&e, rows);
            let call = e.launch(&b, work, rows, false).unwrap();
            let total = e
                .projections
                .iter()
                .map(|p| {
                    p.ticket.plans(rows).unwrap()[1]
                        .grid
                        .into_iter()
                        .product::<u32>()
                })
                .sum::<u32>();
            assert_eq!(call.plan.grid, [total, 1, 1]);
            assert_eq!(call.arguments.0.len(), 18);
            let mut bad = b;
            bad[1].state = bad[0].state;
            assert!(e.launch(&bad, work, rows, false).is_err());
            let mut bad = b;
            bad[1].first.activations.handle = 999;
            assert!(e.launch(&bad, work, rows, false).is_err());
            assert!(e.launch(&b, [work[0]; 2], rows, false).is_err());
            assert!(
                e.launch(&b, work, rows, true)
                    .unwrap()
                    .symbol
                    .ends_with("_delayed_v1")
            );
        }
        let mut bad = e.clone();
        bad.projections[1].ticket.query.k = 1024;
        assert!(bad.validate().is_err());
        let mut bad = e.clone();
        bad.projections[1].ticket.paired = true;
        assert!(bad.validate().is_err());
        assert!(e.plan(65).is_err());
    }
}
