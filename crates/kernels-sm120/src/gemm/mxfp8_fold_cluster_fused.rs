//! Cluster split-K folds (`mxfp8_fold_cluster`) that admit rows by status
//! instead of an admission launch and can quantize their F32 output for the
//! next projection in the merge, with the row-sliced E4M3 quantizer that
//! feeds them (`row/ceil_sliced.cuh`). See `mxfp8_fold_cluster_fused.cuh`
//! for the status contract.
use super::{mxfp8_fold, mxfp8_fold_cluster, mxfp8_fold_packed};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{
    backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{Argument as A, ArgumentList, LaunchPlan, Refusal},
    mxfp8_gemm::Mxfp8FoldGemmBuffers,
    precision::quantizer::BlockQuantization,
};

/// The amax floor of the E4M3 ceil quantizers whose rows the fused folds
/// admit by status.
pub fn floor_bits() -> u32 {
    match BlockQuantization::E4M3_CEIL32 {
        BlockQuantization::E4M3Ceil32 { amax_floor_bits } => amax_floor_bits,
        _ => unreachable!(),
    }
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    /// The fold and its schedule; never distributed.
    pub cluster: mxfp8_fold_cluster::Entry,
    /// Per-row flag bytes a fused producer left (one per 32-column block of
    /// the activation row), or zero.
    pub pending: u32,
    /// Also store the F32 output's E4M3 codes, scale codes and block flags.
    pub quantize: bool,
}

/// The fold's quantized output: codes, scale codes and one flags byte per
/// 32-column block, row-major over every group's columns.
#[derive(Clone, Copy, Debug)]
pub struct Quantized {
    pub codes: View,
    pub scales: View,
    pub flags: View,
}

/// A fused fold's buffers: the projection's, its quantized output (with
/// `Entry::quantize`) and its producer's block flags (with
/// `Entry::pending`).
#[derive(Clone, Copy, Debug)]
pub struct FusedBuffers {
    pub fold: Mxfp8FoldGemmBuffers<View>,
    pub quantized: Option<Quantized>,
    pub pending: Option<View>,
}

impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        let c = &self.cluster;
        c.validate()?;
        let q = c.base.query;
        if c.distributed {
            return Err(Refusal("fused cluster folds merge through the workspace"));
        }
        if self.pending != 0
            && (self.pending * 32 != q.activation_row_stride || self.pending % 16 != 0)
        {
            return Err(Refusal(
                "pending flags cover the activation row in 32-column blocks",
            ));
        }
        if self.quantize && (c.tile_n % 32 != 0 || (c.tile_rows * c.tile_n / c.splits) % 32 != 0) {
            return Err(Refusal(
                "quantizing cluster folds merge whole 32-column blocks per warp",
            ));
        }
        Ok(())
    }
    pub fn symbol(&self) -> String {
        let base = self.cluster.symbol();
        format!(
            "{}_status{}{}_v1",
            base.strip_suffix("_v1").unwrap_or(&base),
            if self.pending != 0 { "p" } else { "" },
            if self.quantize { "q" } else { "" }
        )
    }
    /// The partial slabs; no admission flags.
    pub fn workspace(&self) -> u64 {
        self.cluster.workspace() - 256
    }
    fn info(&self, need: Need) -> Result<OpInfo, Refused> {
        self.validate()?;
        // The cluster fold's guarantees, one launch and no admission.
        let mut info = self.cluster.prepare(need)?.info().clone();
        info.implementation = format!("sm120.mxfp8-fold.cluster-status.v1:{self:?}");
        info.workspace = self.workspace();
        info.symbols = vec![(self.symbol(), self.cluster.shared_bytes())];
        need.check(&info)?;
        Ok(info)
    }
    /// The fold over its own and its fused buffers.
    pub fn prepare(&self, need: Need) -> Result<Prepared<FusedBuffers>, Refused> {
        Ok(Box::new(Op {
            entry: self.clone(),
            info: self.info(need)?,
            capacity: need.capacity,
        }))
    }
    /// A fold with neither a quantized output nor pending flags over the
    /// projection's buffers alone.
    pub fn prepare_plain(
        &self,
        need: Need,
    ) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        if self.quantize || self.pending != 0 {
            return Err(Refusal("a fused fold with fused buffers needs them bound").into());
        }
        Ok(Box::new(Op {
            entry: self.clone(),
            info: self.info(need)?,
            capacity: need.capacity,
        }))
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        let c = &self.cluster;
        let q = c.base.query;
        if rows > q.m_capacity {
            return Err(Refusal("fused cluster fold rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [q.n / c.tile_n, 1, q.groups * c.splits],
            block: [c.tile_n * 4, 1, 1],
            dynamic_shared_bytes: c.shared_bytes(),
            workspace_bytes: 0,
            cluster: [1, 1, c.splits],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    fn body(&self, ignore_status: bool, packed: bool) -> String {
        let c = &self.cluster;
        let q = c.base.query;
        format!(
            "chop::gemm::mxfp8_fold_cluster_fused::run<{},{},{},{},{},{},{},{},{},{},{},{}u,{}{}>(activations,activation_scales,weights,weight_scales,output,codes_out,scales_out,flags_out,pending,workspace,live_rows,status,status_row_offset,rows);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            c.tile_n,
            c.tile_rows,
            c.splits,
            c.stages,
            self.pending,
            self.quantize,
            floor_bits(),
            ignore_status,
            if packed { ",true" } else { "" }
        )
    }
    pub fn spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol(),
            vec![
                ptr("activations", "const uint8_t"),
                ptr("activation_scales", "const uint8_t"),
                ptr("weights", "const uint8_t"),
                ptr("weight_scales", "const uint8_t"),
                ptr("output", "float"),
                ptr("codes_out", "uint8_t"),
                ptr("scales_out", "uint8_t"),
                ptr("flags_out", "uint8_t"),
                ptr("pending", "const uint8_t"),
                ptr("workspace", "uint8_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
                EntryParam::U32 {
                    name: "rows".into(),
                },
            ],
            self.body(false, false),
            self.cluster.tile_n * 4,
            1,
        )
    }
    /// The entry reading `mxfp8-fold-packed-8x128-v1` weight codes
    /// (`gemm::mxfp8_fold_packed`): the same parameters, plan, shared memory
    /// and arithmetic, so both the status and the plain fold move to packed
    /// codes by renaming their launch.
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let q = self.cluster.base.query;
        mxfp8_fold_packed::check(q.groups, q.n, q.k)?;
        let mut spec = self.spec();
        spec.symbol = mxfp8_fold_packed::symbol(&spec.symbol);
        spec.body = self.body(false, true);
        Ok(spec)
    }
    /// The lab control, which no serving cubin holds: every live row
    /// admitted whatever its status.
    pub fn control_spec(&self) -> EntrySpec {
        let mut spec = self.spec();
        spec.symbol.push_str("_ignore_v1");
        spec.body = self.body(true, false);
        spec
    }
    pub fn launch(
        &self,
        b: &Mxfp8FoldGemmBuffers<View>,
        quantized: Option<Quantized>,
        pending: Option<View>,
        work: View,
        rows: u32,
    ) -> Result<Launch, Refusal> {
        let plan = self.plan(rows)?;
        let c = &self.cluster;
        let q = c.base.query;
        let bytes = c.base.bytes(rows);
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
            .ok_or(Refusal("fused cluster fold status extent overflows"))?;
        b.status.covers(status)?;
        work.covers(self.workspace())?;
        if quantized.is_some() != self.quantize || pending.is_some() != (self.pending != 0) {
            return Err(Refusal(
                "fused cluster fold buffers must match its quantized output and pending flags",
            ));
        }
        let columns = u64::from(rows) * u64::from(q.n * q.groups);
        if let Some(out) = quantized {
            out.codes.covers(columns)?;
            out.scales.covers(columns / 32)?;
            out.flags.covers(columns / 32)?;
        }
        if let Some(p) = pending {
            p.covers(u64::from(rows) * u64::from(self.pending))?;
        }
        let mut handles = vec![
            b.activations.handle,
            b.activation_scales.handle,
            b.weights.handle,
            b.weight_scales.handle,
            b.output.handle,
            b.live_rows.handle,
            b.status.handle,
            work.handle,
        ];
        handles.extend(
            quantized
                .iter()
                .flat_map(|o| [o.codes.handle, o.scales.handle, o.flags.handle]),
        );
        handles.extend(pending.iter().map(|p| p.handle));
        if handles
            .iter()
            .enumerate()
            .any(|(i, h)| *h == 0 || handles[..i].contains(h))
        {
            return Err(Refusal(
                "fused cluster fold buffers must have distinct handles",
            ));
        }
        let d = |v: View| A::DeviceBuffer(v.handle);
        let null = |v: Option<View>| v.map_or(A::NullDevicePointer, d);
        Ok(Launch {
            symbol: self.symbol(),
            plan,
            arguments: ArgumentList(vec![
                d(b.activations),
                d(b.activation_scales),
                d(b.weights),
                d(b.weight_scales),
                d(b.output),
                null(quantized.map(|o| o.codes)),
                null(quantized.map(|o| o.scales)),
                null(quantized.map(|o| o.flags)),
                null(pending),
                d(work),
                d(b.live_rows),
                d(b.status),
                A::U64(b.status_row_offset),
                A::U32(rows),
            ]),
            rows,
            collective: false,
            persistent: false,
        })
    }
}

struct Op {
    entry: Entry,
    info: OpInfo,
    capacity: u32,
}
impl PreparedOp<FusedBuffers> for Op {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &FusedBuffers,
        work: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        if counts.rows > self.capacity {
            return Err(Refusal("fused cluster fold rows exceed prepared capacity"));
        }
        Ok(vec![self.entry.launch(
            &b.fold,
            b.quantized,
            b.pending,
            work,
            counts.rows,
        )?])
    }
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
            return Err(Refusal("fused cluster fold rows exceed prepared capacity"));
        }
        Ok(vec![self.entry.launch(b, None, None, work, counts.rows)?])
    }
}

/// E4M3 ceil quantization of decode rows with `slices` CTAs of `warps`
/// warps per row (`row/ceil_sliced.cuh`), for consumers that admit by
/// status.
#[derive(Clone, Debug)]
pub struct Quantizer {
    pub width: u32,
    pub slices: u32,
    pub warps: u32,
    pub f32_input: bool,
    pub capacity: u32,
    pub symbol: String,
}

impl Quantizer {
    pub fn validate(&self) -> Result<(), Refusal> {
        let groups = self.width / 32 / self.slices.max(1);
        if self.width == 0
            || self.slices == 0
            || self.width % (32 * self.slices) != 0
            || !(1..=32).contains(&self.warps)
            || groups % self.warps != 0
            || groups / self.warps > 16
        {
            return Err(Refusal(
                "sliced quantizers give each warp 1 to 16 whole groups of a row slice",
            ));
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if rows > self.capacity {
            return Err(Refusal("sliced quantizer rows exceed capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows.max(1), self.slices, 1],
            block: [self.warps * 32, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }
    pub fn spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol.clone(),
            vec![
                ptr(
                    "input",
                    if self.f32_input {
                        "const float"
                    } else {
                        "const uint16_t"
                    },
                ),
                ptr("codes", "uint8_t"),
                ptr("scales", "uint8_t"),
                ptr("live_rows", "const uint32_t"),
                ptr("status", "uint32_t"),
                EntryParam::U64 {
                    name: "status_row_offset".into(),
                },
            ],
            format!(
                "chop::row::block_quantize_sliced::run<{},{},{},{},{}u>(input,codes,scales,live_rows,status,status_row_offset);",
                self.width,
                self.slices,
                self.warps,
                self.f32_input,
                floor_bits()
            ),
            self.warps * 32,
            1,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        input: View,
        codes: View,
        scales: View,
        live_rows: View,
        status: View,
        status_row_offset: u64,
        rows: u32,
    ) -> Result<Launch, Refusal> {
        let plan = self.plan(rows)?;
        let columns = u64::from(rows) * u64::from(self.width);
        input.covers(columns * if self.f32_input { 4 } else { 2 })?;
        codes.covers(columns)?;
        scales.covers(columns / 32)?;
        live_rows.covers(4)?;
        status.covers((status_row_offset + u64::from(rows)) * 4)?;
        let handles = [
            input.handle,
            codes.handle,
            scales.handle,
            live_rows.handle,
            status.handle,
        ];
        if handles
            .iter()
            .enumerate()
            .any(|(i, h)| *h == 0 || handles[..i].contains(h))
        {
            return Err(Refusal(
                "sliced quantizer buffers must have distinct handles",
            ));
        }
        let d = |v: View| A::DeviceBuffer(v.handle);
        Ok(Launch {
            symbol: self.symbol.clone(),
            plan,
            arguments: ArgumentList(vec![
                d(input),
                d(codes),
                d(scales),
                d(live_rows),
                d(status),
                A::U64(status_row_offset),
            ]),
            rows,
            collective: false,
            persistent: false,
        })
    }
}

/// The cubin of `entries` and `quantizers`, with each entry's lab control
/// after it when `controls` is set.
pub fn cubin_spec(
    name: &str,
    entries: &[Entry],
    quantizers: &[Quantizer],
    controls: bool,
) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() && quantizers.is_empty() {
        return Err(Refusal("fused cluster folds require entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(entry.symbol()) {
            return Err(Refusal("duplicate fused cluster fold entry"));
        }
    }
    for quantizer in quantizers {
        quantizer.validate()?;
        if !symbols.insert(quantizer.symbol.clone()) {
            return Err(Refusal("duplicate sliced quantizer entry"));
        }
    }
    let mut headers = Vec::new();
    if !entries.is_empty() {
        headers.push("#include \"gemm/mxfp8_fold_cluster_fused.cuh\"".into());
    }
    if !quantizers.is_empty() {
        headers.push("#include \"row/ceil_sliced.cuh\"".into());
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers,
        defines: vec![],
        entries: entries
            .iter()
            .flat_map(|e| {
                let mut specs = vec![e.spec()];
                if controls {
                    specs.push(e.control_spec());
                }
                specs
            })
            .chain(quantizers.iter().map(Quantizer::spec))
            .collect(),
    })
}

/// N, K, groups, rows, column tile, splits, stages, pending flags and
/// quantized output of one lab entry.
type LabShape = (u32, u32, u32, u32, u32, u32, u32, bool, bool);

/// Every column tile (16, 32 and 64), both row tiles, groups, splits whose
/// last CTA holds fewer folds, the status fold with a quantized output and
/// with pending flags, and each schedule DeepSeek V4.1 Flash TP4 serves for
/// WQ_a, WKV, WO_a and WO_b decode. The fold needs K divisible by 128, so no
/// shape has a final 64-K fold.
#[rustfmt::skip]
const LAB_SHAPES: &[LabShape] = &[
    (64, 1024, 1, 16, 16, 4, 2, false, false),
    (48, 768, 3, 16, 16, 2, 3, false, false),
    (32, 640, 1, 32, 32, 2, 3, false, false),
    (128, 512, 2, 32, 32, 2, 2, false, true),
    (128, 512, 2, 32, 64, 2, 2, true, false),
    (1280, 5120, 1, 16, 32, 8, 5, false, false),
    (1280, 5120, 1, 32, 64, 8, 5, false, false),
    (512, 5120, 1, 16, 16, 8, 5, false, false),
    (512, 5120, 1, 32, 16, 8, 5, false, false),
    (1024, 4096, 2, 16, 32, 8, 4, false, true),
    (1024, 4096, 2, 32, 64, 8, 3, false, true),
    (5120, 2048, 1, 16, 32, 2, 4, true, false),
    (5120, 2048, 1, 32, 32, 2, 4, true, false),
];

/// The lab entries whose packed twins the lab compares bit for bit
/// ([`LAB_SHAPES`]).
pub fn packed_lab_entries() -> Vec<Entry> {
    LAB_SHAPES
        .iter()
        .copied()
        .map(
            |(n, k, groups, rows, tile_n, splits, stages, pending, quantize)| {
                let query = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
                    m_capacity: rows,
                    n,
                    k,
                    groups,
                    activation_row_stride: groups * k,
                    activation_group_stride: k,
                    output: kernel_core::contract::DType::F32,
                    execution:
                        kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
                };
                let entry = Entry {
                    cluster: mxfp8_fold_cluster::Entry {
                        base: mxfp8_fold::Entry::new(
                            query,
                            format!("chop_mxfp8_fold_cluster_fused_packed_lab_n{n}_k{k}_g{groups}"),
                        )
                        .unwrap(),
                        tile_n,
                        tile_rows: rows,
                        splits,
                        stages,
                        distributed: false,
                    },
                    pending: if pending { groups * k / 32 } else { 0 },
                    quantize,
                };
                entry.validate().unwrap();
                entry
            },
        )
        .collect()
}

/// [`packed_lab_entries`] and their packed twins in one cubin.
pub fn packed_lab_cubin_spec(name: &str) -> Result<CubinSpec, Refusal> {
    let entries = packed_lab_entries();
    let mut spec = cubin_spec(name, &entries, &[], false)?;
    for entry in &entries {
        spec.entries.push(entry.packed_spec()?);
    }
    Ok(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{contract::DType, mxfp8_gemm::Mxfp8FoldGemmQuery};
    fn cluster(
        n: u32,
        k: u32,
        groups: u32,
        rows: u32,
        tile_n: u32,
        splits: u32,
        stages: u32,
    ) -> mxfp8_fold_cluster::Entry {
        let q = Mxfp8FoldGemmQuery {
            n,
            k,
            groups,
            m_capacity: rows,
            activation_row_stride: k * groups,
            activation_group_stride: k,
            output: DType::F32,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        };
        mxfp8_fold_cluster::Entry {
            base: mxfp8_fold::Entry::new(q, "fold").unwrap(),
            tile_n,
            tile_rows: rows,
            splits,
            stages,
            distributed: false,
        }
    }
    #[test]
    fn fused_folds_launch_the_cluster_schedule_with_status_buffers() {
        let a = Entry {
            cluster: cluster(1024, 4096, 2, 32, 64, 8, 3),
            pending: 0,
            quantize: true,
        };
        let b = Entry {
            cluster: cluster(5120, 2048, 1, 32, 32, 2, 4),
            pending: 64,
            quantize: false,
        };
        assert_eq!(a.symbol(), "fold_cluster32_n64_c8s3_statusq_v1");
        assert_eq!(b.symbol(), "fold_cluster32_n32_c2s4_statusp_v1");
        assert_eq!(a.workspace(), a.cluster.workspace() - 256);
        let v = |handle, bytes| View { handle, bytes };
        for rows in [1, 16, 32] {
            let size = a.cluster.base.bytes(rows);
            let fold = Mxfp8FoldGemmBuffers {
                activations: v(1, size.activations),
                activation_scales: v(2, size.activation_scales),
                weights: v(3, size.weights),
                weight_scales: v(4, size.weight_scales),
                output: v(5, size.output),
                live_rows: v(6, 4),
                status: v(7, u64::from(rows) * 4),
                status_row_offset: 0,
            };
            let out = Quantized {
                codes: v(9, u64::from(rows) * 2048),
                scales: v(10, u64::from(rows) * 64),
                flags: v(11, u64::from(rows) * 64),
            };
            let work = v(8, a.workspace());
            let call = a.launch(&fold, Some(out), None, work, rows).unwrap();
            assert_eq!(call.plan.grid, [16, 1, 16]);
            assert_eq!(call.plan.cluster, [1, 1, 8]);
            assert_eq!(call.plan.dynamic_shared_bytes, a.cluster.shared_bytes());
            assert_eq!(call.arguments.0.len(), 14);
            assert_eq!(call.arguments.0[8], A::NullDevicePointer);
            assert!(a.launch(&fold, None, None, work, rows).is_err());
            assert!(
                a.launch(&fold, Some(out), Some(v(12, 2048)), work, rows)
                    .is_err()
            );
            let short = Quantized {
                flags: v(11, u64::from(rows) * 64 - 1),
                ..out
            };
            assert!(a.launch(&fold, Some(short), None, work, rows).is_err());
            let same = Quantized {
                flags: v(9, u64::from(rows) * 64),
                ..out
            };
            assert!(a.launch(&fold, Some(same), None, work, rows).is_err());
        }
        assert!(a.spec().body.contains(&format!(
            "run<1024,4096,2,8192,4096,64,32,8,3,0,true,{}u,false>(",
            floor_bits()
        )));
        assert!(b.control_spec().body.contains(",64,false,"));
        assert!(b.control_spec().body.ends_with(",true>(activations,activation_scales,weights,weight_scales,output,codes_out,scales_out,flags_out,pending,workspace,live_rows,status,status_row_offset,rows);"));
        assert_eq!(floor_bits(), 1e-4_f32.to_bits());
    }
    #[test]
    fn packed_twins_change_only_the_weight_address() {
        let a = Entry {
            cluster: cluster(5120, 2048, 1, 32, 32, 2, 4),
            pending: 64,
            quantize: false,
        };
        let (spec, packed) = (a.spec(), a.packed_spec().unwrap());
        assert_eq!(
            packed.symbol,
            "fold_cluster32_n32_c2s4_statusp_v1_packed_v1"
        );
        assert_eq!(
            packed.body,
            spec.body.replace(">(activations", ",true>(activations")
        );
        assert_eq!(packed.params, spec.params);
        assert_eq!(
            (packed.max_threads, packed.min_blocks),
            (spec.max_threads, spec.min_blocks)
        );
        let mut bad = a.clone();
        bad.cluster.distributed = true;
        assert!(bad.packed_spec().is_err());
        let lab = packed_lab_cubin_spec("lab").unwrap();
        assert_eq!(lab.entries.len(), 2 * packed_lab_entries().len());
    }
    #[test]
    fn prepared_fused_folds_declare_one_launch_and_its_ring() {
        let a = Entry {
            cluster: cluster(1024, 4096, 2, 32, 64, 8, 3),
            pending: 0,
            quantize: true,
        };
        let op = a.prepare(Need::rows(32)).unwrap();
        assert_eq!(op.info().workspace, a.workspace());
        assert_eq!(op.info().symbols, [(a.symbol(), a.cluster.shared_bytes())]);
        assert!(a.prepare(Need::rows(33)).is_err());
        assert!(a.prepare_plain(Need::rows(32)).is_err());
        let v = |handle, bytes| View { handle, bytes };
        let size = a.cluster.base.bytes(32);
        let fold = Mxfp8FoldGemmBuffers {
            activations: v(1, size.activations),
            activation_scales: v(2, size.activation_scales),
            weights: v(3, size.weights),
            weight_scales: v(4, size.weight_scales),
            output: v(5, size.output),
            live_rows: v(6, 4),
            status: v(7, 128),
            status_row_offset: 0,
        };
        let quantized = Some(Quantized {
            codes: v(9, 32 * 2048),
            scales: v(10, 32 * 64),
            flags: v(11, 32 * 64),
        });
        let work = v(8, a.workspace());
        let buffers = FusedBuffers {
            fold,
            quantized,
            pending: None,
        };
        let calls = op.launches(&buffers, work, Counts::rows(16)).unwrap();
        assert_eq!(calls, [a.launch(&fold, quantized, None, work, 16).unwrap()]);
        assert!(op.launches(&buffers, work, Counts::rows(33)).is_err());
        let kv = Entry {
            cluster: cluster(512, 5120, 1, 32, 16, 8, 5),
            pending: 0,
            quantize: false,
        };
        let plain = kv.prepare_plain(Need::rows(32)).unwrap();
        let size = kv.cluster.base.bytes(32);
        let fold = Mxfp8FoldGemmBuffers {
            activations: v(1, size.activations),
            activation_scales: v(2, size.activation_scales),
            weights: v(3, size.weights),
            weight_scales: v(4, size.weight_scales),
            output: v(5, size.output),
            ..fold
        };
        let work = v(8, kv.workspace());
        assert_eq!(
            plain.launches(&fold, work, Counts::rows(32)).unwrap(),
            [kv.launch(&fold, None, None, work, 32).unwrap()]
        );
    }
    #[test]
    fn fused_folds_refuse_partial_blocks_and_distributed_merges() {
        let mut e = Entry {
            cluster: cluster(512, 5120, 1, 32, 16, 8, 5),
            pending: 0,
            quantize: true,
        };
        assert!(e.validate().is_err(), "16-column tiles split blocks");
        e.quantize = false;
        e.validate().unwrap();
        e.pending = 64;
        assert!(e.validate().is_err(), "pending flags of another row width");
        e.pending = 160;
        e.validate().unwrap();
        e.cluster.distributed = true;
        assert!(e.validate().is_err());
        // 16 rows of 32 columns over 8 ranks leave each rank 64 outputs.
        let small = Entry {
            cluster: cluster(1024, 4096, 2, 16, 32, 8, 4),
            pending: 0,
            quantize: true,
        };
        small.validate().unwrap();
    }
    #[test]
    fn sliced_quantizers_cover_rows_with_whole_groups() {
        let q = Quantizer {
            width: 8192,
            slices: 8,
            warps: 8,
            f32_input: false,
            capacity: 32,
            symbol: "sliced_v1".into(),
        };
        let v = |handle, bytes| View { handle, bytes };
        let call = q
            .launch(
                v(1, 32 * 8192 * 2),
                v(2, 32 * 8192),
                v(3, 32 * 256),
                v(4, 4),
                v(5, 128),
                0,
                32,
            )
            .unwrap();
        assert_eq!(call.plan.grid, [32, 8, 1]);
        assert_eq!(call.plan.block, [256, 1, 1]);
        assert!(q.spec().body.contains("run<8192,8,8,false,"));
        assert!(
            q.launch(
                v(1, 32 * 8192 * 2 - 1),
                v(2, 32 * 8192),
                v(3, 32 * 256),
                v(4, 4),
                v(5, 128),
                0,
                32
            )
            .is_err()
        );
        assert!(q.plan(33).is_err());
        let mut bad = q.clone();
        bad.warps = 3;
        assert!(bad.validate().is_err());
        bad.warps = 8;
        bad.slices = 3;
        assert!(bad.validate().is_err());
        let spec = cubin_spec("fused", &[], std::slice::from_ref(&q), true).unwrap();
        assert_eq!(spec.headers, ["#include \"row/ceil_sliced.cuh\""]);
        assert!(cubin_spec("fused", &[], &[q.clone(), q], false).is_err());
    }
}
