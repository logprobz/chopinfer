//! Share one weight stage across two unchanged sixteen-row MMA groups.
use super::mxfp8_fold_ticket;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub ticket: mxfp8_fold_ticket::Entry,
    pub tile_n: u32,
}

pub fn supported_rows(rows: u32) -> bool {
    rows == 6 || (8..=32).contains(&rows)
}

impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.ticket.validate()?;
        if ![32, 64].contains(&self.tile_n) || self.ticket.query.output != DType::F32 {
            return Err(Refusal(
                "row-reuse folds require F32 outputs and 32/64 columns",
            ));
        }
        Ok(())
    }

    pub fn symbol(&self) -> String {
        format!("{}_reuse32_n{}_v1", self.ticket.symbol, self.tile_n)
    }

    pub fn workspace(&self, rows: u32) -> Result<u64, Refusal> {
        self.validate()?;
        let q = self.ticket.query;
        let tiles = u64::from(q.n.div_ceil(self.tile_n))
            * u64::from(q.groups)
            * if self.ticket.paired { 2 } else { 1 };
        Ok(256
            + (tiles * 4).div_ceil(256) * 256
            + u64::from(q.k.div_ceil(128))
                * u64::from(rows)
                * u64::from(q.n)
                * u64::from(q.groups)
                * if self.ticket.paired { 8 } else { 4 })
    }

    pub fn plans(&self, rows: u32, folds: u32) -> Result<[LaunchPlan; 2], Refusal> {
        self.validate()?;
        let q = self.ticket.query;
        if !supported_rows(rows) || rows > q.m_capacity || ![2, 4, 8].contains(&folds) {
            return Err(Refusal(
                "row-reuse folds require a supported row bucket and 2/4/8 folds",
            ));
        }
        let plan = |grid, threads| LaunchPlan {
            grid,
            block: [threads, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        Ok([
            plan([rows, 1, 1], 128),
            plan(
                [
                    q.n.div_ceil(self.tile_n),
                    1,
                    q.k.div_ceil(128).div_ceil(folds)
                        * q.groups
                        * if self.ticket.paired { 2 } else { 1 },
                ],
                self.tile_n * 4,
            ),
        ])
    }

    /// The weight reader on packed codes. Admission keeps its original symbol.
    pub fn packed_spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let q = self.ticket.query;
        super::mxfp8_fold_packed::check(q.groups, q.n, q.k)?;
        let mut spec = self.specs().remove(1);
        spec.symbol = super::mxfp8_fold_packed::symbol(&spec.symbol);
        spec.body = spec.body.replace(">(task,arena", ",true>(task,arena");
        Ok(spec)
    }

    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.ticket.query;
        let mut specs = self.ticket.specs();
        let ns = "chop::gemm::mxfp8_fold_reuse";
        specs[0].symbol = format!("{}_admit_v1", self.symbol());
        specs[0].body = format!(
            "chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();{ns}::admit_task<{},{},{},{},{},{},{}>(blockIdx.x,gridDim.x,activations,activation_scales,workspace,live_rows,status,status_row_offset,rows);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.tile_n,
            self.ticket.paired
        );
        specs[1].symbol = self.symbol();
        specs[1].max_threads = self.tile_n * 4;
        specs[1].min_blocks = 1;
        specs[1].body = format!(
            "__shared__ {ns}::Arena<{}> arena;const {ns}::Task task{{blockIdx.x,blockIdx.y,blockIdx.z,gridDim.x,gridDim.y}};chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();{ns}::task<{},{},{},{},{},{},true,{}>(task,arena,activations,activation_scales,weights,weight_scales,weights2,weight_scales2,output,output2,workspace,live_rows,rows,folds_per_cta);",
            self.tile_n,
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            self.tile_n,
            self.ticket.paired
        );
        specs
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("row-reuse folds require entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(entry.symbol()) {
            return Err(Refusal("duplicate row-reuse entry"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_reuse.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().flat_map(Entry::specs).collect(),
    })
}
