//! Opt-in native FP8 tiles with exact-size plans and caller-owned row status.
use super::{mxfp8_fold, mxfp8_prefill};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
use kernel_core::properties::Invariance;

#[derive(Clone, Debug)]
pub struct Entry {
    pub base: mxfp8_fold::Entry,
    pub stages: u32,
}
impl Entry {
    pub fn new(base: mxfp8_fold::Entry, stages: u32) -> Result<Self, Refusal> {
        base.validate()?;
        if base.query.k > i32::MAX as u32 - 127 {
            return Err(Refusal("compact prefill rounded K exceeds signed bounds"));
        }
        if ![2, 3].contains(&stages) {
            return Err(Refusal("compact prefill needs two or three stages"));
        }
        let result = Self { base, stages };
        for rows in [16, 32, 64] {
            result.plan(1, rows)?;
        }
        Ok(result)
    }
    pub fn tile(rows: u32) -> u32 {
        if rows <= 16 {
            16
        } else if rows <= 32 {
            32
        } else {
            64
        }
    }
    pub fn symbol(&self, tile: u32) -> String {
        format!(
            "{}_compact_m{tile}_n128_s{}_v1",
            self.base.symbol, self.stages
        )
    }
    pub fn admission_symbol(&self) -> String {
        format!("{}_admit_v1", self.symbol(64))
    }
    pub fn shared_bytes(&self, tile: u32) -> u32 {
        self.stages * ((tile + 128) * 128 + (tile + 128) * 4) + tile * 4
    }
    pub fn plan(&self, rows: u32, tile: u32) -> Result<LaunchPlan, Refusal> {
        if ![16, 32, 64].contains(&tile) {
            return Err(Refusal("compact prefill row tile must be 16, 32 or 64"));
        }
        let mut plan = self.base.plan(rows)?;
        plan.grid = [
            self.base.query.n.div_ceil(128),
            rows.max(1).div_ceil(tile),
            self.base.query.groups,
        ];
        plan.block = [if tile == 64 { 256 } else { 128 }, 1, 1];
        plan.dynamic_shared_bytes = self.shared_bytes(tile);
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        let mut admit = mxfp8_prefill::specs(&self.base)?.remove(0);
        admit.symbol = self.admission_symbol();
        let mut result = vec![admit];
        for tile in [16, 32, 64] {
            for omit in [false, true] {
                let mut spec = self.base.spec();
                spec.symbol = self.symbol(tile);
                if omit {
                    spec.symbol.push_str("_omit_first_v1");
                }
                let q = self.base.query;
                spec.body = format!(
                    "chop::gemm::mxfp8_prefill_compact::run<{},{},{},{},{},{},{tile},{},{}>(activations,activation_scales,weights,weight_scales,output,live_rows,status,status_row_offset,rows);",
                    q.n,
                    q.k,
                    q.groups,
                    q.activation_row_stride,
                    q.activation_group_stride,
                    q.output == DType::F32,
                    self.stages,
                    omit
                );
                spec.max_threads = if tile == 64 { 256 } else { 128 };
                spec.min_blocks = 2;
                result.push(spec);
            }
        }
        Ok(result)
    }
    /// One plan per exact host shape. A replay may shorten only its device
    /// live count; changing the host row count requires another prepared op.
    /// Code views require 16-byte alignment and scale views 4-byte alignment.
    /// Each concurrently executing op owns its output and sticky row status.
    pub fn prepare(&self, rows: u32) -> Result<Prepared<Buffers<View>>, Refusal> {
        if rows == 0 {
            return Err(Refusal("compact prefill preparation needs nonzero rows"));
        }
        let compute = self.plan(rows, Self::tile(rows))?;
        let admit = mxfp8_prefill::validation_plan(&self.base, rows)?;
        let b = self.base.bytes(rows);
        let mut info = OpInfo::new(
            format!(
                "sm120.mxfp8-prefill-compact.s{}:{:?}",
                self.stages, self.base.query
            ),
            Invariance::any_rows(rows),
            vec![
                (self.admission_symbol(), 0),
                (self.symbol(Self::tile(rows)), compute.dynamic_shared_bytes),
            ],
        );
        info.bytes = vec![
            ("activations".into(), b.activations),
            ("activation_scales".into(), b.activation_scales),
            ("weights".into(), b.weights),
            ("weight_scales".into(), b.weight_scales),
            ("output".into(), b.output),
            ("live_rows".into(), 4),
            ("status".into(), b.status),
        ];
        info.weight_layouts = vec![("weights".into(), self.base.weight_layout())];
        Ok(Box::new(Op {
            entry: self.clone(),
            rows,
            admit,
            compute,
            info,
        }))
    }
}
struct Op {
    entry: Entry,
    rows: u32,
    admit: LaunchPlan,
    compute: LaunchPlan,
    info: OpInfo,
}
impl PreparedOp<Buffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, b: &Buffers<View>, _: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        if counts.rows != self.rows {
            return Err(Refusal(
                "compact prefill requires its prepared host row count",
            ));
        }
        let q = self.entry.base.bytes(self.rows);
        for (view, bytes) in [
            (b.activations, q.activations),
            (b.activation_scales, q.activation_scales),
            (b.weights, q.weights),
            (b.weight_scales, q.weight_scales),
            (b.output, q.output),
            (b.live_rows, 4),
        ] {
            view.covers(bytes)?;
        }
        let end = b
            .status_row_offset
            .checked_add(u64::from(self.rows))
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or(Refusal("compact prefill status extent overflows"))?;
        b.status.covers(end)?;
        let reads = [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.live_rows.handle,
        ];
        if reads.contains(&b.output.handle)
            || reads.contains(&b.status.handle)
            || b.output.handle == b.status.handle
        {
            return Err(Refusal(
                "compact prefill output or row status aliases an input",
            ));
        }
        let handles = Buffers {
            activations: b.activations.handle,
            activation_scales: b.activation_scales.handle,
            weights: b.weights.handle,
            weight_scales: b.weight_scales.handle,
            output: b.output.handle,
            live_rows: b.live_rows.handle,
            status: b.status.handle,
            status_row_offset: b.status_row_offset,
        };
        let launch = |symbol, plan, arguments| Launch {
            symbol,
            plan,
            arguments,
            rows: self.rows,
            collective: false,
            persistent: false,
        };
        Ok(vec![
            launch(
                self.entry.admission_symbol(),
                self.admit,
                ArgumentList(vec![
                    Argument::DeviceBuffer(handles.activations),
                    Argument::DeviceBuffer(handles.activation_scales),
                    Argument::DeviceBuffer(handles.live_rows),
                    Argument::DeviceBuffer(handles.status),
                    Argument::U64(handles.status_row_offset),
                    Argument::U32(self.rows),
                ]),
            ),
            launch(
                self.entry.symbol(Entry::tile(self.rows)),
                self.compute,
                self.entry.base.arguments(handles, self.rows),
            ),
        ])
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut bases = entries.iter().map(|e| e.base.clone()).collect::<Vec<_>>();
    bases.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    bases.dedup_by(|a, b| a.symbol == b.symbol && a.query == b.query);
    let mut spec = mxfp8_fold::cubin_spec(name, &bases)?;
    spec.headers = vec!["#include \"gemm/mxfp8_prefill_compact.cuh\"".into()];
    spec.entries = entries
        .iter()
        .map(Entry::specs)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
    use kernel_core::precision::execution::BlockScaledExecution;
    fn entry() -> Entry {
        Entry::new(
            mxfp8_fold::Entry::new(
                Mxfp8FoldGemmQuery {
                    m_capacity: 8192,
                    n: 160,
                    k: 576,
                    groups: 2,
                    activation_row_stride: 1216,
                    activation_group_stride: 608,
                    output: DType::F32,
                    execution: BlockScaledExecution::NativeBlockScaledMma,
                },
                "compact_test",
            )
            .unwrap(),
            2,
        )
        .unwrap()
    }
    #[test]
    fn exact_plans_keep_group_strides_tail_bounds_and_alias_refusals() {
        let e = entry();
        for rows in [1, 16, 17, 32, 33, 65, 257, 2048, 8192] {
            let op = e.prepare(rows).unwrap();
            let sizes = e.base.bytes(rows);
            let b = Buffers {
                activations: View {
                    handle: 1,
                    bytes: sizes.activations,
                },
                activation_scales: View {
                    handle: 2,
                    bytes: sizes.activation_scales,
                },
                weights: View {
                    handle: 3,
                    bytes: sizes.weights,
                },
                weight_scales: View {
                    handle: 4,
                    bytes: sizes.weight_scales,
                },
                output: View {
                    handle: 5,
                    bytes: sizes.output,
                },
                live_rows: View {
                    handle: 6,
                    bytes: 4,
                },
                status: View {
                    handle: 7,
                    bytes: sizes.status + 12,
                },
                status_row_offset: 3,
            };
            let calls = op.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!(calls.len(), 2);
            assert_eq!(calls[1].plan.grid, [2, rows.div_ceil(Entry::tile(rows)), 2]);
            assert_eq!(calls[1].plan.workspace_bytes, 0);
            assert!(
                op.launches(&b, View::EMPTY, Counts::rows(rows - 1))
                    .is_err()
            );
            let mut bad = b;
            bad.status.bytes -= 1;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            for source in [
                b.activations,
                b.activation_scales,
                b.weights,
                b.weight_scales,
                b.live_rows,
                b.status,
            ] {
                let mut bad = b;
                bad.output.handle = source.handle;
                assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
            }
            let mut bad = b;
            bad.status_row_offset = u64::MAX;
            assert!(op.launches(&bad, View::EMPTY, Counts::rows(rows)).is_err());
        }
        assert!(e.prepare(0).is_err());
        assert!(e.prepare(8193).is_err());
        assert!(e.plan(1, 48).is_err());
        assert!(Entry::new(e.base, 1).is_err());
    }
}
