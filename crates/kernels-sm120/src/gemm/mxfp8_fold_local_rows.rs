//! Fold up to 32 rows in one CTA per column tile, without global partials.
use super::{mxfp8_fold, mxfp8_fold_ticket};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{Argument as A, ArgumentList, DType, LaunchPlan, Refusal},
    mxfp8_gemm::Mxfp8FoldGemmBuffers,
};

#[derive(Clone, Debug)]
pub struct Entry {
    pub base: mxfp8_fold::Entry,
    pub tile_n: u32,
    /// Folds in flight: 0 keeps the three-slot pair loop (`run`); 3 or more
    /// selects the refilling ring (`run_deep`) with the same arithmetic.
    pub stages: u32,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.base.validate()?;
        let q = self.base.query;
        if q.output != DType::F32
            || q.m_capacity > 32
            || q.k % 128 != 0
            || q.activation_row_stride % 128 != 0
            || q.activation_group_stride % 128 != 0
            || ![8, 16, 32].contains(&self.tile_n)
        {
            return Err(Refusal(
                "local row folds require aligned FP8 groups, F32 output and at most 32 rows",
            ));
        }
        if self.stages != 0
            && (self.stages < 3
                || self.stages > q.k / 128
                || self.stages * (32 + self.tile_n) * 132 + 128 > 48 * 1024)
        {
            return Err(Refusal(
                "a deep local fold ring holds 3 to K/128 folds in static shared memory",
            ));
        }
        Ok(())
    }
    pub fn prepare(&self, need: Need) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        self.validate()?;
        let legacy = crate::backend::Sm120::new()
            .with_mxfp8_fold(self.base.clone())
            .mxfp8_fold_gemm(&self.base.query, need)?;
        let mut info = legacy.info().clone();
        info.implementation = format!("sm120.mxfp8-fold.local-rows.v1:{self:?}");
        info.workspace = self.workspace();
        info.symbols = self
            .specs()
            .into_iter()
            .take(2)
            .map(|s| (s.symbol, 0))
            .collect();
        need.check(&info)?;
        Ok(Box::new(Op {
            entry: self.clone(),
            info,
            capacity: need.capacity,
        }))
    }
    pub fn symbol(&self) -> String {
        if self.stages == 0 {
            format!("{}_local32_n{}_v1", self.base.symbol, self.tile_n)
        } else {
            format!(
                "{}_local32_n{}_s{}_v1",
                self.base.symbol, self.tile_n, self.stages
            )
        }
    }
    pub fn workspace(&self) -> u64 {
        256
    }
    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        self.validate()?;
        if rows > self.base.query.m_capacity {
            return Err(Refusal("local fold rows exceed capacity"));
        }
        let plan = |grid, threads| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        let result = [
            plan([rows.max(1), 1, 1], 128),
            plan(
                [
                    self.base.query.n.div_ceil(self.tile_n),
                    1,
                    self.base.query.groups,
                ],
                self.tile_n * 4,
            ),
        ];
        for p in &result {
            crate::backend::check_plan(p)?;
        }
        Ok(result)
    }
    pub fn launches(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        work: View,
        rows: u32,
    ) -> Result<Vec<Launch>, Refusal> {
        let plans = self.plans(rows)?;
        let bytes = self.base.bytes(rows);
        b.activations.covers(bytes.activations)?;
        b.activation_scales.covers(bytes.activation_scales)?;
        b.weights.covers(bytes.weights)?;
        b.weight_scales.covers(bytes.weight_scales)?;
        b.output.covers(bytes.output)?;
        b.live_rows.covers(4)?;
        let status = b
            .status_row_offset
            .checked_add(u64::from(rows))
            .and_then(|n| n.checked_mul(4))
            .filter(|&n| n <= kernel_core::precision::MAX_TENSOR_BYTES)
            .ok_or(Refusal("local fold status extent overflows"))?;
        b.status.covers(status)?;
        work.covers(self.workspace())?;
        let handles = [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.output.handle,
            b.live_rows.handle,
            b.status.handle,
            work.handle,
        ];
        if handles
            .iter()
            .enumerate()
            .any(|(i, h)| *h == 0 || handles[..i].contains(h))
        {
            return Err(Refusal("local fold buffers must have distinct handles"));
        }
        let d = |v: View| A::DeviceBuffer(v.handle);
        let args = [
            vec![
                d(b.activations),
                d(b.activation_scales),
                d(work),
                d(b.live_rows),
                d(b.status),
                A::U64(b.status_row_offset),
                A::U32(rows),
            ],
            vec![
                d(b.activations),
                d(b.activation_scales),
                d(b.weights),
                d(b.weight_scales),
                A::NullDevicePointer,
                A::NullDevicePointer,
                d(b.output),
                A::NullDevicePointer,
                d(work),
                d(b.live_rows),
                A::U32(rows),
                A::U32(0),
            ],
        ];
        Ok(args
            .into_iter()
            .zip(plans)
            .enumerate()
            .map(|(i, (args, plan))| Launch {
                symbol: if i == 0 {
                    format!("{}_admit_v1", self.symbol())
                } else {
                    self.symbol()
                },
                plan,
                arguments: ArgumentList(args),
                rows,
                collective: false,
                persistent: false,
            })
            .collect())
    }
    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.base.query;
        let ticket = mxfp8_fold_ticket::Entry::new(q, self.symbol(), false).unwrap();
        let mut specs = ticket.specs();
        let ns = "chop::gemm::mxfp8_fold_local_rows";
        specs[0].symbol = format!("{}_admit_v1", self.symbol());
        specs[0].body = format!(
            "{ns}::admit<{},{},{},{},false>(activations,activation_scales,workspace,live_rows,status,status_row_offset,rows);",
            q.k, q.groups, q.activation_row_stride, q.activation_group_stride
        );
        specs[1].max_threads = self.tile_n * 4;
        specs[1].body = if self.stages == 0 {
            format!(
                "{ns}::run<{},{},{},{},{},{},false>(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);",
                q.n, q.k, q.groups, q.activation_row_stride, q.activation_group_stride, self.tile_n
            )
        } else {
            format!(
                "{ns}::run_deep<{},{},{},{},{},{},{},false>(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);",
                q.n,
                q.k,
                q.groups,
                q.activation_row_stride,
                q.activation_group_stride,
                self.tile_n,
                self.stages
            )
        };
        let mut omit = specs[1].clone();
        omit.symbol.push_str("_omit_v1");
        omit.body = omit.body.replace(",false>(", ",true>(");
        let mut delayed = specs[0].clone();
        delayed.symbol.push_str("_delayed_v1");
        delayed.body = delayed.body.replace(",false>(", ",true>(");
        specs.extend([omit, delayed]);
        specs
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("local row folds require entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(entry.symbol()) {
            return Err(Refusal("duplicate local row fold entry"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: if entries.iter().any(|e| e.stages > 0) {
            vec!["#include \"gemm/mxfp8_fold_local_deep.cuh\"".into()]
        } else {
            vec!["#include \"gemm/mxfp8_fold_local_rows.cuh\"".into()]
        },
        defines: vec![],
        entries: entries.iter().flat_map(Entry::specs).collect(),
    })
}

struct Op {
    entry: Entry,
    info: OpInfo,
    capacity: u32,
}
impl PreparedOp<Mxfp8FoldGemmBuffers<View>> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        work: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal("local fold rows exceed prepared capacity"));
        }
        self.entry.launches(b, work, counts.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
    #[test]
    fn keep_one_weight_pass_and_reject_output_workspace_aliases() {
        let q = Mxfp8FoldGemmQuery {
            n: 8192,
            k: 1280,
            groups: 1,
            m_capacity: 32,
            activation_row_stride: 1280,
            activation_group_stride: 1280,
            output: DType::F32,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        };
        for tile_n in [8, 16, 32] {
            let e = Entry {
                base: mxfp8_fold::Entry::new(q, "local_fold").unwrap(),
                tile_n,
                stages: 0,
            };
            for rows in [0, 8, 32] {
                let v = |handle, bytes| View { handle, bytes };
                let size = e.base.bytes(rows);
                let b = Mxfp8FoldGemmBuffers {
                    activations: v(1, size.activations),
                    activation_scales: v(2, size.activation_scales),
                    weights: v(3, size.weights),
                    weight_scales: v(4, size.weight_scales),
                    output: v(5, size.output),
                    live_rows: v(6, 4),
                    status: v(7, (u64::from(rows) + 3) * 4),
                    status_row_offset: 3,
                };
                let calls = e.launches(&b, v(8, 256), rows).unwrap();
                assert_eq!(calls.len(), 2);
                let op = e.prepare(Need::rows(32)).unwrap();
                assert_eq!(op.info().workspace, 256);
                assert_eq!(
                    op.launches(&b, v(8, 256), Counts::rows(rows)).unwrap(),
                    calls
                );
                assert!(e.prepare(Need::rows(33)).is_err());
                if rows == 32 {
                    assert!(
                        e.prepare(Need::rows(8))
                            .unwrap()
                            .launches(&b, v(8, 256), Counts::rows(rows))
                            .is_err()
                    );
                }
                assert_eq!(calls[1].plan.grid, [8192 / tile_n, 1, 1]);
                assert_eq!(calls[1].arguments.0[11], A::U32(0));
                assert!(e.launches(&b, v(5, 256), rows).is_err());
                assert!(e.launches(&b, v(8, 255), rows).is_err());
                let mut invalid = b;
                invalid.status_row_offset = u64::MAX;
                assert!(e.launches(&invalid, v(8, 256), rows).is_err());
            }
            assert!(e.plans(33).is_err());
            assert_eq!(e.specs().len(), 4);
        }
        let mut e = Entry {
            base: mxfp8_fold::Entry::new(q, "local_fold").unwrap(),
            tile_n: 64,
            stages: 0,
        };
        assert!(e.validate().is_err());
        e.tile_n = 32;
        for (stages, valid) in [(2, false), (3, true), (5, true), (6, false), (11, false)] {
            e.stages = stages;
            assert_eq!(e.validate().is_ok(), valid, "{stages}");
        }
        e.tile_n = 16;
        e.stages = 6;
        assert!(e.validate().is_ok());
        assert!(e.symbol().ends_with("_local32_n16_s6_v1"));
        assert!(
            e.specs()[1]
                .body
                .contains("run_deep<8192,1280,1,1280,1280,16,6,false>")
        );
        let spec = cubin_spec("deep", std::slice::from_ref(&e)).unwrap();
        assert_eq!(
            spec.headers,
            ["#include \"gemm/mxfp8_fold_local_deep.cuh\""]
        );
        e.stages = 0;
        e.tile_n = 32;
        e.base.query.k = 1216;
        assert!(e.validate().is_err());
    }
}
