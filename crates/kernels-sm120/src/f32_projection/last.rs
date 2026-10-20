//! Ordered last-producer reduction over unchanged cached-decode partials.
use super::Entry;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub entry: Entry,
    pub tile_rows: u32,
    pub split_fast: bool,
    pub prefetch: u32,
}

impl Candidate {
    pub fn validate(&self) -> Result<(), Refusal> {
        if !self.entry.uses_cached_weights()
            || self.entry.query().weight != WeightFormat::Bf16
            || ![8, 32].contains(&self.tile_rows)
            || ![1, 16].contains(&self.prefetch)
        {
            return Err(Refusal(
                "last-producer reduction requires cached BF16 weights and fixed tiles",
            ));
        }
        self.entry.decode_plan(1)?;
        Ok(())
    }

    pub fn symbol(&self, reverse: bool) -> String {
        format!(
            "{}_last_m{}_s{}_p{}_r{}_v1",
            self.entry.stage_symbol("_decode"),
            self.tile_rows,
            u32::from(self.split_fast),
            self.prefetch,
            u32::from(reverse)
        )
    }

    pub fn prepare_symbol(&self) -> String {
        format!("{}_prepare_v1", self.symbol(false))
    }

    pub fn counter_offset(&self, rows: u32) -> u64 {
        self.entry.decode_workspace_bytes(rows).div_ceil(256) * 256
    }

    pub fn counter_bytes(&self, rows: u32) -> u64 {
        u64::from(self.entry.query().n.div_ceil(16)) * u64::from(rows.div_ceil(self.tile_rows)) * 4
    }

    pub fn workspace(&self, rows: u32) -> u64 {
        self.counter_offset(rows) + self.counter_bytes(rows)
    }

    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        self.validate()?;
        if ![1, 8, 32].contains(&rows) || self.tile_rows != if rows <= 8 { 8 } else { 32 } {
            return Err(Refusal(
                "last-producer reduction requires the measured row tile",
            ));
        }
        let q = self.entry.query();
        let mut compute = self.entry.decode_plan(rows)?;
        compute.grid = [
            q.n.div_ceil(16)
                .checked_mul(rows.div_ceil(self.tile_rows))
                .and_then(|tiles| tiles.checked_mul(q.k.div_ceil(128).div_ceil(4)))
                .ok_or(Refusal("last-producer flattened grid overflows"))?,
            1,
            1,
        ];
        compute.workspace_bytes = self.workspace(rows);
        let prepare = LaunchPlan {
            grid: [rows, 1, 1],
            ..compute
        };
        crate::backend::check_plan(&compute)?;
        crate::backend::check_plan(&prepare)?;
        Ok([prepare, compute])
    }

    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.entry.query();
        let ns = "chop::exact_f32_last";
        let mut prepare = self
            .entry
            .entry_specs()
            .into_iter()
            .find(|e| e.symbol == self.entry.stage_symbol("_rows"))
            .expect("cached rows entry");
        prepare.symbol = self.prepare_symbol();
        prepare.body = format!(
            "chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();{ns}::prepare<{},{},{}>(blockIdx.x,gridDim.x,activation,live_rows,rows,reinterpret_cast<uint8_t*>(workspace));",
            q.n, q.k, self.tile_rows
        );
        let mut result = vec![prepare];
        for reverse in [false, true] {
            let mut compute = self
                .entry
                .entry_specs()
                .into_iter()
                .find(|e| e.symbol == self.entry.stage_symbol("_decode"))
                .expect("cached decode entry");
            compute.symbol = self.symbol(reverse);
            compute.params.push(EntryParam::DevicePointer {
                name: "output".into(),
                cxx_type: "float".into(),
            });
            let tiles = q.n.div_ceil(16);
            let splits = q.k.div_ceil(128).div_ceil(4);
            let task = if self.split_fast {
                format!(
                    "{{(blockIdx.x/{splits})%{tiles},blockIdx.x/({splits}*{tiles}),blockIdx.x%{splits}}}"
                )
            } else {
                format!(
                    "{{blockIdx.x%{tiles},(blockIdx.x/{tiles})%((rows+{}-1)/{}),blockIdx.x/({tiles}*((rows+{}-1)/{}))}}",
                    self.tile_rows, self.tile_rows, self.tile_rows, self.tile_rows
                )
            };
            compute.body = format!(
                "__shared__ {ns}::Arena arena;const {ns}::Task task{task};chop::ptx::griddep_launch_dependents();chop::ptx::griddep_wait();{ns}::task<{},{},{},{},{}>(task,arena,activation,weight,reinterpret_cast<uint8_t*>(workspace),output,live_rows,rows,weight_domain);",
                q.n, q.k, self.tile_rows, self.prefetch, reverse
            );
            result.push(compute);
        }
        result
    }
}

impl Entry {
    /// Route only explicitly qualified row counts through the ordered reducer.
    pub fn with_decode_last(mut self, mut rows: Vec<u32>) -> Result<Self, Refusal> {
        rows.sort_unstable();
        if rows.is_empty() || rows.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Refusal(
                "last-producer routing needs distinct measured row buckets",
            ));
        }
        self.last_rows = rows;
        for &row in &self.last_rows {
            self.decode_last(row).expect("selected bucket").plans(row)?;
        }
        Ok(self)
    }

    pub fn decode_last(&self, rows: u32) -> Option<Candidate> {
        self.last_rows.binary_search(&rows).ok().map(|_| Candidate {
            entry: self.entry_for_rows(rows),
            tile_rows: if rows <= 8 { 8 } else { 32 },
            split_fast: true,
            prefetch: 1,
        })
    }

    /// The partial-plane layout stays fixed; only selected buckets need counters.
    pub fn runtime_workspace_bytes(&self, capacity: u32) -> u64 {
        self.last_rows
            .iter()
            .copied()
            .filter(|&rows| rows <= capacity)
            .map(|rows| {
                self.decode_last(rows)
                    .expect("selected bucket")
                    .workspace(rows)
            })
            .fold(self.decode_workspace_bytes(capacity), u64::max)
    }
}

pub fn cubin_spec(name: &str, candidates: &[Candidate]) -> Result<CubinSpec, Refusal> {
    if candidates.is_empty() {
        return Err(Refusal("last-producer reduction requires candidates"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for candidate in candidates {
        candidate.validate()?;
        if !symbols.insert(candidate.symbol(false)) {
            return Err(Refusal("duplicate last-producer candidate"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/exact_f32_last.cuh\"".into()],
        defines: vec![],
        entries: candidates.iter().flat_map(Candidate::specs).collect(),
    })
}
