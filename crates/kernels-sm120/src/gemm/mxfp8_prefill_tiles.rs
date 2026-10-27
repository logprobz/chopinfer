//! Native FP8 prefill tiles with scalar or matrix fragment fetches.
//! Both fetch modes preserve the entry ABI and ordered K128 arithmetic.
pub mod compact;
pub mod matrix;
use super::{mxfp8_fold, mxfp8_fold_packed, mxfp8_prefill};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, View};
use kernel_core::contract::{Argument, ArgumentList, DType, LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers as Buffers;
use kernel_core::properties::Invariance;

#[derive(Clone, Debug)]
pub struct Entry<const MATRIX_FETCH: bool> {
    pub base: mxfp8_fold::Entry,
    pub stages: u32,
    /// Fixed matrix tile; scalar fetch chooses its tile from the row count.
    pub tile_rows: u32,
}
impl Entry<false> {
    pub fn new(base: mxfp8_fold::Entry, stages: u32) -> Result<Self, Refusal> {
        Self::build(base, stages, 0)
    }
    pub fn packed_spec(&self, tile: u32) -> Result<EntrySpec, Refusal> {
        self.packed_tile_spec(tile)
    }
    pub fn packed_specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.tiles()
            .into_iter()
            .map(|tile| self.packed_tile_spec(tile))
            .collect()
    }
}
impl Entry<true> {
    pub fn new(base: mxfp8_fold::Entry, stages: u32, tile_rows: u32) -> Result<Self, Refusal> {
        Self::build(base, stages, tile_rows)
    }
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.packed_tile_spec(self.tile_rows)
    }
}
impl<const MATRIX_FETCH: bool> Entry<MATRIX_FETCH> {
    fn build(base: mxfp8_fold::Entry, stages: u32, tile_rows: u32) -> Result<Self, Refusal> {
        base.validate()?;
        if base.query.k > i32::MAX as u32 - 127 {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill rounded K exceeds signed bounds"
            } else {
                "compact prefill rounded K exceeds signed bounds"
            }));
        }
        if !(2..=if MATRIX_FETCH { 4 } else { 3 }).contains(&stages) {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill has an unsupported ring depth"
            } else {
                "compact prefill has an unsupported ring depth"
            }));
        }
        let result = Self {
            base,
            stages,
            tile_rows,
        };
        for tile in result.tiles() {
            result.plan(1, tile)?;
        }
        Ok(result)
    }
    pub const fn fetch_name() -> &'static str {
        if MATRIX_FETCH { "matrix" } else { "compact" }
    }
    fn tiles(&self) -> Vec<u32> {
        if MATRIX_FETCH {
            vec![self.tile_rows]
        } else {
            vec![16, 32, 64]
        }
    }
    fn selected_tile(&self, rows: u32) -> u32 {
        if MATRIX_FETCH {
            self.tile_rows
        } else {
            Self::tile(rows)
        }
    }
    pub fn tile(rows: u32) -> u32 {
        if rows <= 16 {
            16
        } else if rows <= 128 {
            32
        } else {
            64
        }
    }
    pub fn symbol(&self, tile: u32) -> String {
        format!(
            "{}_{}_m{tile}_n128_s{}_v1",
            self.base.symbol,
            Self::fetch_name(),
            self.stages
        )
    }
    pub fn admission_symbol(&self) -> String {
        format!(
            "{}_admit_v1",
            self.symbol(if MATRIX_FETCH { self.tile_rows } else { 64 })
        )
    }
    pub fn shared_bytes(&self, tile: u32) -> u32 {
        self.stages * ((tile + 128) * 128 + (tile + 128) * 4) + tile * 4
    }
    pub fn plan(&self, rows: u32, tile: u32) -> Result<LaunchPlan, Refusal> {
        if !(if MATRIX_FETCH {
            [32, 64, 128]
        } else {
            [16, 32, 64]
        })
        .contains(&tile)
        {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill row tile is unsupported"
            } else {
                "compact prefill row tile is unsupported"
            }));
        }
        let mut plan = self.base.plan(rows)?;
        plan.grid = [
            self.base.query.n.div_ceil(128),
            rows.max(1).div_ceil(tile),
            self.base.query.groups,
        ];
        plan.block = [if tile >= 64 { 256 } else { 128 }, 1, 1];
        plan.dynamic_shared_bytes = self.shared_bytes(tile);
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        let mut admit = mxfp8_prefill::specs(&self.base)?.remove(0);
        admit.symbol = self.admission_symbol();
        let mut result = vec![admit];
        for tile in self.tiles() {
            for omit in [false, true] {
                result.push(self.compute_spec(tile, omit, false));
            }
        }
        Ok(result)
    }
    fn packed_tile_spec(&self, tile: u32) -> Result<EntrySpec, Refusal> {
        let q = self.base.query;
        mxfp8_fold_packed::check(q.groups, q.n, q.k)?;
        self.plan(1, tile)?;
        Ok(self.compute_spec(tile, false, true))
    }
    fn compute_spec(&self, tile: u32, omit: bool, packed: bool) -> EntrySpec {
        let mut spec = self.base.spec();
        spec.symbol = self.symbol(tile);
        if omit {
            spec.symbol.push_str("_omit_first_v1");
        }
        if packed {
            spec.symbol = mxfp8_fold_packed::symbol(&spec.symbol);
        }
        let q = self.base.query;
        spec.body = format!(
            "chop::gemm::mxfp8_prefill_compact::run<{},{},{},{},{},{},{tile},{},{MATRIX_FETCH},{}{}>(activations,activation_scales,weights,weight_scales,output,live_rows,status,status_row_offset,rows);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            q.output == DType::F32,
            self.stages,
            omit,
            if packed { ",true" } else { "" }
        );
        spec.max_threads = if tile >= 64 { 256 } else { 128 };
        spec.min_blocks = if MATRIX_FETCH { 1 } else { 2 };
        spec
    }
    /// One plan per exact host shape. A replay may shorten only its device
    /// live count; changing the host row count requires another prepared op.
    /// Code views require 16-byte alignment and scale views 4-byte alignment.
    /// Each concurrently executing op owns its output and sticky row status.
    pub fn prepare(&self, rows: u32) -> Result<Prepared<Buffers<View>>, Refusal> {
        if rows == 0 {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill preparation needs nonzero rows"
            } else {
                "compact prefill preparation needs nonzero rows"
            }));
        }
        let compute = self.plan(rows, self.selected_tile(rows))?;
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
                (
                    self.symbol(self.selected_tile(rows)),
                    compute.dynamic_shared_bytes,
                ),
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
struct Op<const MATRIX_FETCH: bool> {
    entry: Entry<MATRIX_FETCH>,
    rows: u32,
    admit: LaunchPlan,
    compute: LaunchPlan,
    info: OpInfo,
}
impl<const MATRIX_FETCH: bool> PreparedOp<Buffers<View>> for Op<MATRIX_FETCH> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, b: &Buffers<View>, _: View, counts: Counts) -> Result<Vec<Launch>, Refusal> {
        if counts.rows != self.rows {
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill requires its prepared host row count"
            } else {
                "compact prefill requires its prepared host row count"
            }));
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
            .ok_or(Refusal(if MATRIX_FETCH {
                "matrix prefill status extent overflows"
            } else {
                "compact prefill status extent overflows"
            }))?;
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
            return Err(Refusal(if MATRIX_FETCH {
                "matrix prefill output or row status aliases an input"
            } else {
                "compact prefill output or row status aliases an input"
            }));
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
                self.entry.symbol(self.entry.selected_tile(self.rows)),
                self.compute,
                self.entry.base.arguments(handles, self.rows),
            ),
        ])
    }
}
pub fn cubin_spec<const MATRIX_FETCH: bool>(
    name: &str,
    entries: &[Entry<MATRIX_FETCH>],
) -> Result<CubinSpec, Refusal> {
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
