//! Split-K folds of up to 32 rows whose K splits form one thread-block
//! cluster: every fold's loads are issued at once, and the ranks sum the
//! tile's fold partials in increasing K order after one cluster barrier.
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
    /// Rows one CTA folds: 16 (one MMA row group) or 32 (two).
    pub tile_rows: u32,
    /// CTAs of one cluster, each owning consecutive 128-K folds.
    pub splits: u32,
    /// Folds of its split a CTA holds in shared memory at once (at least
    /// two, at most all of them); later folds refill freed slots.
    pub stages: u32,
    /// Keep the fold partials in shared memory and merge them through
    /// distributed shared memory instead of the workspace.
    pub distributed: bool,
}
impl Entry {
    pub fn folds(&self) -> u32 {
        self.base.query.k / 128
    }
    /// Folds one CTA owns; the last split may hold fewer.
    pub fn folds_per_cta(&self) -> u32 {
        self.folds().div_ceil(self.splits)
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.base.validate()?;
        let q = self.base.query;
        if q.output != DType::F32
            || q.k % 128 != 0
            || q.activation_row_stride % 128 != 0
            || q.activation_group_stride % 128 != 0
            || ![16, 32, 64].contains(&self.tile_n)
            || q.n % self.tile_n != 0
            || ![16, 32].contains(&self.tile_rows)
            || q.m_capacity > self.tile_rows
        {
            return Err(Refusal(
                "cluster folds require aligned FP8 groups, F32 output, whole 16/32/64-column tiles and at most 32 rows",
            ));
        }
        let per = self.folds_per_cta();
        if !(2..=8).contains(&self.splits)
            || per > 8
            || (self.splits - 1) * per >= self.folds()
            || (self.tile_rows * self.tile_n) % self.splits != 0
            || !(2..=per).contains(&self.stages)
        {
            return Err(Refusal(
                "cluster folds split K over 2 to 8 CTAs of at most 8 folds, each owning folds, with 2 to all of them in flight",
            ));
        }
        if self.distributed
            && per * self.tile_rows * self.tile_n * 4
                > self.stages * (self.tile_rows + self.tile_n) * 128
        {
            return Err(Refusal(
                "distributed cluster folds keep their partials in the operand slots",
            ));
        }
        if self.shared_bytes() > crate::backend::SHARED_MEMORY_BYTES {
            return Err(Refusal(
                "a cluster fold's operand ring exceeds shared memory",
            ));
        }
        Ok(())
    }
    /// Dynamic shared memory (`mxfp8_fold_cluster::Layout`): each slot's
    /// operand rows and scale quads, then one flag per row.
    pub fn shared_bytes(&self) -> u32 {
        let rows = self.tile_rows + self.tile_n;
        self.stages * rows * (128 + 4) + self.tile_rows * 4
    }
    pub fn prepare(&self, need: Need) -> Result<Prepared<Mxfp8FoldGemmBuffers<View>>, Refused> {
        self.validate()?;
        if need.capacity > self.tile_rows {
            return Err(Refusal("cluster fold rows exceed its row tile").into());
        }
        let legacy = crate::backend::Sm120::new()
            .with_mxfp8_fold(self.base.clone())
            .mxfp8_fold_gemm(&self.base.query, need)?;
        let mut info = legacy.info().clone();
        info.implementation = format!("sm120.mxfp8-fold.cluster.v1:{self:?}");
        info.workspace = self.workspace();
        // The admission kernel takes no dynamic shared memory; the fold
        // takes its operand ring.
        info.symbols = self
            .specs()
            .into_iter()
            .take(2)
            .zip([0, self.shared_bytes()])
            .map(|(s, bytes)| (s.symbol, bytes))
            .collect();
        need.check(&info)?;
        Ok(Box::new(Op {
            entry: self.clone(),
            info,
            capacity: need.capacity,
        }))
    }
    pub fn symbol(&self) -> String {
        format!(
            "{}_cluster{}_n{}_c{}s{}{}_v1",
            self.base.symbol,
            self.tile_rows,
            self.tile_n,
            self.splits,
            self.stages,
            if self.distributed { "d" } else { "" }
        )
    }
    fn grid(&self) -> [u32; 3] {
        let q = self.base.query;
        [q.n / self.tile_n, 1, q.groups * self.splits]
    }
    /// Admission flags, then (unless distributed) one partial slab per CTA.
    pub fn workspace(&self) -> u64 {
        if self.distributed {
            return 256;
        }
        let [x, _, z] = self.grid();
        256 + u64::from(x)
            * u64::from(z)
            * u64::from(self.folds_per_cta())
            * u64::from(self.tile_rows)
            * u64::from(self.tile_n)
            * 4
    }
    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        self.validate()?;
        if rows > self.base.query.m_capacity {
            return Err(Refusal("cluster fold rows exceed capacity"));
        }
        let plan = |grid, threads, shared, cluster| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: shared,
            workspace_bytes: 0,
            cluster,
            pdl: true,
        };
        let result = [
            plan([rows.max(1), 1, 1], 128, 0, [1, 1, 1]),
            plan(
                self.grid(),
                self.tile_n * 4,
                self.shared_bytes(),
                [1, 1, self.splits],
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
            .ok_or(Refusal("cluster fold status extent overflows"))?;
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
            return Err(Refusal("cluster fold buffers must have distinct handles"));
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
    /// Admission, then the fold.
    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.base.query;
        let ticket = mxfp8_fold_ticket::Entry::new(q, self.symbol(), false).unwrap();
        let mut specs = ticket.specs();
        specs[0].symbol = format!("{}_admit_v1", self.symbol());
        specs[0].body = format!(
            "chop::gemm::mxfp8_fold_local_rows::admit<{},{},{},{},false>(activations,activation_scales,workspace,live_rows,status,status_row_offset,rows);",
            q.k, q.groups, q.activation_row_stride, q.activation_group_stride
        );
        specs[1].max_threads = self.tile_n * 4;
        specs[1].min_blocks = 1;
        specs[1].body = format!(
            "chop::gemm::mxfp8_fold_cluster::run<{},{},{},{},{},{},{},{},{},{},false>(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.tile_n,
            self.tile_rows,
            self.splits,
            self.stages,
            self.distributed
        );
        specs
    }
    /// The lab controls, which no serving cubin holds: the fold with its
    /// first fold's partial omitted, and admission delayed past the fold's
    /// start.
    pub fn control_specs(&self) -> Vec<EntrySpec> {
        let [admit, fold]: [EntrySpec; 2] = self.specs().try_into().unwrap();
        let mut omit = fold;
        omit.symbol.push_str("_omit_v1");
        omit.body = omit.body.replace(",false>(", ",true>(");
        let mut delayed = admit;
        delayed.symbol.push_str("_delayed_v1");
        delayed.body = delayed.body.replace(",false>(", ",true>(");
        vec![omit, delayed]
    }
}
/// The cubin of `entries`, with each entry's lab controls after it when
/// `controls` is set.
pub fn cubin_spec(name: &str, entries: &[Entry], controls: bool) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("cluster folds require entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(entry.symbol()) {
            return Err(Refusal("duplicate cluster fold entry"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_cluster.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .flat_map(|e| {
                let mut specs = e.specs();
                if controls {
                    specs.extend(e.control_specs());
                }
                specs
            })
            .collect(),
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
            return Err(Refusal("cluster fold rows exceed prepared capacity"));
        }
        self.entry.launches(b, work, counts.rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;
    fn query(n: u32, k: u32, groups: u32, rows: u32) -> Mxfp8FoldGemmQuery {
        Mxfp8FoldGemmQuery {
            n,
            k,
            groups,
            m_capacity: rows,
            activation_row_stride: k * groups,
            activation_group_stride: k,
            output: DType::F32,
            execution:
                kernel_core::precision::execution::BlockScaledExecution::NativeBlockScaledMma,
        }
    }
    fn entry(q: Mxfp8FoldGemmQuery, tile_rows: u32, splits: u32, distributed: bool) -> Entry {
        let folds = q.k / 128;
        Entry {
            base: mxfp8_fold::Entry::new(q, "cluster_fold").unwrap(),
            tile_n: 32,
            tile_rows,
            splits,
            stages: folds.div_ceil(splits).clamp(2, 8),
            distributed,
        }
    }
    #[test]
    fn cluster_splits_cover_k_once_and_launch_whole_clusters() {
        for (n, k, groups) in [(1280, 5120, 1), (1024, 4096, 2), (512, 5120, 1)] {
            for tile_rows in [16, 32] {
                for distributed in [false, true] {
                    let e = entry(query(n, k, groups, tile_rows), tile_rows, 8, distributed);
                    e.validate().unwrap();
                    assert_eq!(e.folds_per_cta() * 8, k / 128);
                    for rows in [0, 1, tile_rows] {
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
                        let work = v(8, e.workspace());
                        let calls = e.launches(&b, work, rows).unwrap();
                        assert_eq!(calls.len(), 2);
                        assert_eq!(calls[0].plan.cluster, [1, 1, 1]);
                        assert_eq!(calls[1].plan.cluster, [1, 1, 8]);
                        assert_eq!(calls[1].plan.grid, [n / 32, 1, groups * 8]);
                        assert_eq!(calls[1].plan.block, [128, 1, 1]);
                        assert_eq!(
                            calls[1].plan.dynamic_shared_bytes,
                            (k / 128 / 8) * (tile_rows + 32) * 132 + tile_rows * 4
                        );
                        let op = e.prepare(Need::rows(tile_rows)).unwrap();
                        assert_eq!(op.info().workspace, e.workspace());
                        // Hosts raise each symbol's dynamic shared limit to
                        // what the op declares.
                        assert_eq!(
                            op.info().symbols,
                            [
                                (calls[0].symbol.clone(), 0),
                                (calls[1].symbol.clone(), calls[1].plan.dynamic_shared_bytes)
                            ]
                        );
                        assert_eq!(op.launches(&b, work, Counts::rows(rows)).unwrap(), calls);
                        assert!(e.launches(&b, v(5, e.workspace()), rows).is_err());
                        assert!(e.launches(&b, v(8, e.workspace() - 1), rows).is_err());
                    }
                    assert_eq!(
                        e.workspace(),
                        if distributed {
                            256
                        } else {
                            256 + u64::from(n / 32 * groups * 8)
                                * u64::from(k / 128 / 8 * tile_rows * 32 * 4)
                        }
                    );
                    assert!(e.prepare(Need::rows(tile_rows + 1)).is_err());
                    assert!(e.plans(tile_rows + 1).is_err());
                    let specs = e.specs();
                    assert_eq!(specs.len(), 2);
                    assert!(specs[1].body.contains(&format!(
                        "run<{n},{k},{groups},{},{k},32,{tile_rows},8,{},{distributed},false>(",
                        k * groups,
                        k / 128 / 8
                    )));
                    let controls = e.control_specs();
                    assert_eq!(controls.len(), 2);
                    assert_eq!(controls[0].symbol, format!("{}_omit_v1", specs[1].symbol));
                    assert!(controls[0].body.ends_with(",true>(activations,activation_scales,weights,weight_scales,output,workspace,live_rows,rows);"));
                    assert!(controls[1].symbol.ends_with("_admit_v1_delayed_v1"));
                }
            }
        }
    }
    #[test]
    fn cluster_folds_refuse_partial_tiles_and_oversized_splits() {
        let q = query(1280, 5120, 1, 32);
        assert!(
            entry(q, 32, 4, false).validate().is_err(),
            "ten folds per CTA"
        );
        assert!(
            entry(query(1280, 5120, 1, 16), 16, 8, false)
                .validate()
                .is_ok()
        );
        assert!(entry(q, 32, 1, false).validate().is_err());
        assert!(entry(q, 32, 9, false).validate().is_err());
        assert!(
            entry(q, 16, 8, false).validate().is_err(),
            "32 rows in a 16-row tile"
        );
        // 36 folds over eight CTAs leave the last one fold; 33 leave it none.
        assert!(
            entry(query(1280, 4608, 1, 32), 32, 8, false)
                .validate()
                .is_ok()
        );
        assert!(
            entry(query(1280, 4224, 1, 32), 32, 8, false)
                .validate()
                .is_err()
        );
        let mut e = entry(q, 32, 8, false);
        e.tile_n = 64;
        assert!(e.validate().is_ok());
        assert_eq!(e.shared_bytes(), 5 * 96 * 132 + 128);
        e.tile_n = 128;
        assert!(e.validate().is_err());
        e.tile_n = 32;
        e.base.query.k = 5056;
        assert!(e.validate().is_err());
        assert!(
            entry(q, 32, 8, true)
                .symbol()
                .ends_with("_cluster32_n32_c8s5d_v1")
        );
        let mut ring = entry(query(1024, 4096, 2, 32), 32, 8, false);
        assert_eq!((ring.folds_per_cta(), ring.stages), (4, 4));
        ring.stages = 3;
        assert!(ring.validate().is_ok());
        assert_eq!(ring.shared_bytes(), 3 * 64 * 132 + 128);
        ring.distributed = true;
        assert!(
            ring.validate().is_ok(),
            "16 KB of partials in 24 KB of slots"
        );
        ring.stages = 1;
        assert!(ring.validate().is_err());
        ring.stages = 5;
        assert!(ring.validate().is_err());
        let both = [entry(q, 32, 8, false), entry(q, 32, 8, true)];
        let spec = cubin_spec("cluster", &both, false).unwrap();
        assert_eq!(spec.headers, ["#include \"gemm/mxfp8_fold_cluster.cuh\""]);
        assert_eq!(spec.entries.len(), 4);
        assert!(spec.entries.iter().all(|e| !e.body.contains(",true>(")));
        let lab = cubin_spec("cluster", &both, true).unwrap();
        assert_eq!(lab.entries.len(), 8);
        assert!(lab.entries[2].symbol.ends_with("_c8s5_v1_omit_v1"));
        assert!(
            cubin_spec(
                "cluster",
                &[entry(q, 32, 8, false), entry(q, 32, 8, false)],
                false
            )
            .is_err()
        );
    }
}
