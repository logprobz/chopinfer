//! Row-local folds (`mxfp8_fold_local_rows`) on column tiles of any multiple
//! of eight up to 64 (`mxfp8_fold_local_wide.cuh`), failing rows by status:
//! the row status is the fold's failed-flag table, so no admission launch
//! runs. That equals the admission kernel's result for operands of an E4M3
//! ceil block quantizer that zeroes the rows it rejects (see
//! `mxfp8_fold_cluster_fused.cuh`).
use super::{mxfp8_fold, mxfp8_fold_packed, mxfp8_fold_ticket};
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
    /// selects the swizzled refilling ring (`run_deep`).
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
            || self.tile_n % 8 != 0
            || !(8..=64).contains(&self.tile_n)
        {
            return Err(Refusal(
                "wide local folds require aligned FP8 groups, F32 output, at most 32 rows and eight-column multiples up to 64",
            ));
        }
        let slots = if self.stages == 0 { 3 } else { self.stages };
        if (self.stages != 0 && (self.stages < 3 || self.stages > q.k / 128))
            || slots * (32 + self.tile_n) * 132 + 128 > 48 * 1024
        {
            return Err(Refusal(
                "a wide local fold holds 3 to K/128 folds in static shared memory",
            ));
        }
        Ok(())
    }
    pub fn symbol(&self) -> String {
        let ring = if self.stages == 0 {
            String::new()
        } else {
            format!("_s{}x", self.stages)
        };
        format!(
            "{}_local32_n{}{ring}_status_v1",
            self.base.symbol, self.tile_n
        )
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let q = self.base.query;
        if rows > q.m_capacity {
            return Err(Refusal("wide local fold rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [q.n.div_ceil(self.tile_n), 1, q.groups],
            block: [self.tile_n * 4, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    /// The row-local fold's entry with this tile; its failed-flag table
    /// parameter (`workspace`) is bound to the row status.
    pub fn spec(&self) -> EntrySpec {
        self.spec_impl(self.symbol(), "")
    }
    /// The entry reading `mxfp8-fold-packed-8x128-v1` weight codes
    /// (`gemm::mxfp8_fold_packed`): the same parameters, plan and
    /// arithmetic, so a prepared op moves to packed codes by renaming its
    /// launch.
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let q = self.base.query;
        mxfp8_fold_packed::check(q.groups, q.n, q.k)?;
        Ok(self.spec_impl(mxfp8_fold_packed::symbol(&self.symbol()), ",true"))
    }
    fn spec_impl(&self, symbol: String, packed: &str) -> EntrySpec {
        let q = self.base.query;
        let ticket = mxfp8_fold_ticket::Entry::new(q, self.symbol(), false).unwrap();
        let mut spec = ticket.specs().swap_remove(1);
        let shape = format!(
            "{},{},{},{},{},{}",
            q.n, q.k, q.groups, q.activation_row_stride, q.activation_group_stride, self.tile_n
        );
        let args = "(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);";
        spec.symbol = symbol;
        spec.max_threads = self.tile_n * 4;
        spec.min_blocks = 1;
        spec.body = if self.stages == 0 {
            format!("chop::gemm::mxfp8_fold_local_wide::run<{shape},false{packed}>{args}")
        } else {
            format!(
                "chop::gemm::mxfp8_fold_local_wide::run_deep<{shape},{},true,false{packed}>{args}",
                self.stages
            )
        };
        spec
    }
    /// The fold reading row `r`'s status at `status[r]`: the status must
    /// start at the batch's first row.
    pub fn launch(&self, b: &Mxfp8FoldGemmBuffers<View>, rows: u32) -> Result<Launch, Refusal> {
        let plan = self.plan(rows)?;
        let bytes = self.base.bytes(rows);
        b.activations.covers(bytes.activations)?;
        b.activation_scales.covers(bytes.activation_scales)?;
        b.weights.covers(bytes.weights)?;
        b.weight_scales.covers(bytes.weight_scales)?;
        b.output.covers(bytes.output)?;
        b.live_rows.covers(4)?;
        if b.status_row_offset != 0 {
            return Err(Refusal(
                "a wide local fold reads the status from the batch's first row",
            ));
        }
        b.status.covers(u64::from(rows) * 4)?;
        let handles = [
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.output.handle,
            b.live_rows.handle,
            b.status.handle,
        ];
        if handles
            .iter()
            .enumerate()
            .any(|(i, h)| *h == 0 || handles[..i].contains(h))
        {
            return Err(Refusal(
                "wide local fold buffers must have distinct handles",
            ));
        }
        let d = |v: View| A::DeviceBuffer(v.handle);
        Ok(Launch {
            symbol: self.symbol(),
            plan,
            arguments: ArgumentList(vec![
                d(b.activations),
                d(b.activation_scales),
                d(b.weights),
                d(b.weight_scales),
                A::NullDevicePointer,
                A::NullDevicePointer,
                d(b.output),
                A::NullDevicePointer,
                d(b.status),
                d(b.live_rows),
                A::U32(rows),
                A::U32(0),
            ]),
            rows,
            collective: false,
            persistent: false,
        })
    }
    pub fn prepare(&self, need: Need) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        self.validate()?;
        if need.capacity > self.base.query.m_capacity {
            return Err(Refusal("wide local fold rows exceed capacity").into());
        }
        let legacy = crate::backend::Sm120::new()
            .with_mxfp8_fold(self.base.clone())
            .mxfp8_fold_gemm(&self.base.query, need)?;
        let mut info = legacy.info().clone();
        info.implementation = format!("sm120.mxfp8-fold.local-wide-status.v1:{self:?}");
        info.workspace = 0;
        info.symbols = vec![(self.symbol(), 0)];
        need.check(&info)?;
        Ok(Box::new(Op {
            entry: self.clone(),
            info,
            capacity: need.capacity,
        }))
    }
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
        _work: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal("wide local fold rows exceed prepared capacity"));
        }
        Ok(vec![self.entry.launch(b, counts.rows)?])
    }
}

/// N, K, groups, rows, column tile and folds in flight (0 for the pair
/// loop) of one lab entry.
type LabShape = (u32, u32, u32, u32, u32, u32);

/// Both rings (the pair loop and the refilling ring), column tiles from 8 to
/// 64 including tiles that overhang N, odd fold counts, groups, and WQ_b's
/// DeepSeek V4.1 Flash TP4 decode fold. The fold needs K divisible by 128,
/// so no shape has a final 64-K fold.
#[rustfmt::skip]
const LAB_SHAPES: &[LabShape] = &[
    (104, 384, 1, 32, 48, 0),
    (64, 512, 2, 16, 32, 0),
    (40, 640, 3, 32, 8, 3),
    (64, 768, 2, 32, 24, 4),
    (136, 1024, 1, 32, 64, 3),
    (8192, 1280, 1, 32, 48, 4),
    (8192, 1280, 1, 32, 48, 0),
];

/// The lab entries whose packed twins the lab compares bit for bit
/// ([`LAB_SHAPES`]).
pub fn packed_lab_entries() -> Vec<Entry> {
    LAB_SHAPES
        .iter()
        .map(|&(n, k, groups, rows, tile_n, stages)| {
            let query = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                m_capacity: rows,
                n,
                k,
                groups,
                activation_row_stride: groups * k,
                activation_group_stride: k,
                output: DType::F32,
                execution:
                    kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
            };
            let entry = Entry {
                base: mxfp8_fold::Entry::new(
                    query,
                    format!("chop_mxfp8_fold_local_wide_packed_lab_n{n}_k{k}_g{groups}_m{rows}"),
                )
                .unwrap(),
                tile_n,
                stages,
            };
            entry.validate().unwrap();
            entry
        })
        .collect()
}

/// [`packed_lab_entries`] and their packed twins in one cubin.
pub fn packed_lab_cubin_spec(name: &str) -> Result<CubinSpec, Refusal> {
    let mut entries = Vec::new();
    for entry in packed_lab_entries() {
        entries.push(entry.spec());
        entries.push(entry.packed_spec()?);
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_local_wide.cuh\"".into()],
        defines: vec![],
        entries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
    fn entry(tile_n: u32, stages: u32) -> Entry {
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
        Entry {
            base: mxfp8_fold::Entry::new(q, "fold").unwrap(),
            tile_n,
            stages,
        }
    }
    #[test]
    fn wide_folds_cover_the_columns_and_read_the_status_as_flags() {
        let e = entry(48, 4);
        e.validate().unwrap();
        assert_eq!(e.symbol(), "fold_local32_n48_s4x_status_v1");
        assert_eq!(entry(48, 0).symbol(), "fold_local32_n48_status_v1");
        let v = |handle, bytes| View { handle, bytes };
        let size = e.base.bytes(32);
        let b = Mxfp8FoldGemmBuffers {
            activations: v(1, size.activations),
            activation_scales: v(2, size.activation_scales),
            weights: v(3, size.weights),
            weight_scales: v(4, size.weight_scales),
            output: v(5, size.output),
            live_rows: v(6, 4),
            status: v(7, 128),
            status_row_offset: 0,
        };
        let call = e.launch(&b, 32).unwrap();
        assert_eq!(call.plan.grid, [171, 1, 1]);
        assert_eq!(call.plan.block, [192, 1, 1]);
        assert_eq!(call.arguments.0[8], A::DeviceBuffer(7));
        let op = e.prepare(Need::rows(32)).unwrap();
        assert_eq!(op.info().workspace, 0);
        assert_eq!(op.info().symbols, [(e.symbol(), 0)]);
        assert_eq!(
            op.launches(&b, View::EMPTY, Counts::rows(16)).unwrap()[0].rows,
            16
        );
        assert!(op.launches(&b, View::EMPTY, Counts::rows(33)).is_err());
        assert!(
            e.launch(
                &Mxfp8FoldGemmBuffers {
                    status_row_offset: 1,
                    ..b
                },
                32
            )
            .is_err()
        );
        assert!(
            e.launch(
                &Mxfp8FoldGemmBuffers {
                    status: v(7, 124),
                    ..b
                },
                32
            )
            .is_err()
        );
        assert!(
            e.launch(
                &Mxfp8FoldGemmBuffers {
                    status: v(6, 128),
                    ..b
                },
                32
            )
            .is_err()
        );
        let spec = e.spec();
        assert_eq!(spec.max_threads, 192);
        assert!(spec.body.starts_with(
            "chop::gemm::mxfp8_fold_local_wide::run_deep<8192,1280,1,1280,1280,48,4,true,false>(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);"
        ));
        assert!(
            entry(48, 0)
                .spec()
                .body
                .contains("::run<8192,1280,1,1280,1280,48,false>(")
        );
        // The served row-local fold's parameters, whose workspace slot the
        // status takes.
        let served = crate::gemm::mxfp8_fold_local_rows::Entry {
            base: e.base.clone(),
            tile_n: 32,
            stages: 0,
            swizzle: false,
        };
        assert_eq!(spec.params, served.specs()[1].params);
    }
    #[test]
    fn packed_twins_change_only_the_weight_address() {
        for stages in [0, 4] {
            let e = entry(48, stages);
            let (spec, packed) = (e.spec(), e.packed_spec().unwrap());
            assert_eq!(packed.symbol, format!("{}_packed_v1", e.symbol()));
            assert_eq!(packed.body, spec.body.replace(">(", ",true>("));
            assert_eq!(packed.params, spec.params);
            assert_eq!(packed.max_threads, spec.max_threads);
        }
        assert!(entry(44, 0).packed_spec().is_err());
        let lab = packed_lab_cubin_spec("lab").unwrap();
        assert_eq!(lab.entries.len(), 2 * packed_lab_entries().len());
    }
    #[test]
    fn wide_folds_refuse_odd_tiles_and_oversized_rings() {
        assert!(entry(44, 0).validate().is_err());
        assert!(entry(72, 0).validate().is_err());
        assert!(
            entry(48, 5).validate().is_err(),
            "five 80-row slots exceed 48 KB"
        );
        assert!(entry(48, 2).validate().is_err());
        assert!(entry(64, 3).validate().is_ok());
    }
}
