//! Cached weights and four-chain CTAs for small row counts.
use super::{Entry, pointer};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

pub const MAX_ROWS: u32 = 32;

impl Entry {
    /// Opt into create-time weight classification and ordered split-K at
    /// up to 32 rows. Larger launches use the original locked entry.
    pub fn with_decode(mut self) -> Self {
        self.cached = true;
        self.symbol = format!("{}_cached_v1", self.symbol);
        self
    }

    pub fn uses_cached_weights(&self) -> bool {
        self.cached
    }

    pub fn baseline_symbol(&self) -> &str {
        if self.uses_prefill_tiles() {
            return self
                .symbol
                .strip_suffix(&format!("_tiles_n{}_v1", self.prefill_tile_n))
                .expect("prefill suffix");
        }
        if self.fused {
            self.symbol
                .strip_suffix("_split_v2")
                .expect("split entry suffix")
        } else if self.cached {
            self.symbol
                .strip_suffix("_cached_v1")
                .expect("cached entry suffix")
        } else {
            &self.symbol
        }
    }

    /// Symbols the prepared op can launch, including its original entry
    /// above the measured split-decode crossover. `symbols` instead lists
    /// the exports compiled by this entry's cubin specification.
    pub fn runtime_symbols(&self) -> Vec<String> {
        if self.uses_prefill_tiles() {
            return self.symbols();
        }
        if self.fused {
            let mut symbols = self.symbols();
            let small = self.entry_for_rows(1);
            symbols.extend(["_rows", "_decode", "_reduce"].map(|s| small.stage_symbol(s)));
            symbols.push(self.baseline_symbol().to_owned());
            return symbols;
        }
        if self.cached {
            let mut symbols = ["_weights", "_rows", "_decode", "_reduce"]
                .map(|suffix| self.stage_symbol(suffix))
                .to_vec();
            symbols.push(self.baseline_symbol().to_owned());
            symbols
        } else {
            self.symbols()
        }
    }

    pub fn symbols(&self) -> Vec<String> {
        if self.uses_prefill_tiles() {
            return ["_weights", "_prepare", ""]
                .map(|s| self.stage_symbol(s))
                .to_vec();
        }
        if self.fused {
            return ["_weights", "_prepare", ""]
                .map(|s| self.stage_symbol(s))
                .to_vec();
        }
        if self.cached {
            ["", "_weights", "_rows", "_decode", "_reduce"]
                .map(|suffix| self.stage_symbol(suffix))
                .to_vec()
        } else {
            vec![self.symbol.clone()]
        }
    }

    pub fn stage_symbol(&self, suffix: &str) -> String {
        if self.fused {
            return format!("{}{suffix}_v2", self.symbol.trim_end_matches("_v2"));
        }
        format!("{}{suffix}_v1", self.symbol.trim_end_matches("_v1"))
    }

    pub fn weight_domain_bytes(&self) -> u64 {
        u64::from(self.query.n.div_ceil(16)) * 4
    }

    pub fn decode_workspace_bytes(&self, capacity: u32) -> u64 {
        if self.uses_prefill_tiles() {
            return self.prefill_workspace_bytes(capacity);
        }
        if self.fused {
            let small = self
                .entry_for_rows(1)
                .decode_workspace_bytes(capacity.min(MAX_ROWS));
            return if capacity <= MAX_ROWS {
                small
            } else {
                self.split_workspace_bytes(capacity).max(small)
            };
        }
        128 + u64::from(capacity.min(MAX_ROWS))
            * u64::from(self.query.n)
            * u64::from(self.query.k.div_ceil(128))
            * 4
    }

    /// Splits are fixed groups of four 128-K chains, including a final
    /// partial group. Every chain has its own workspace plane.
    pub fn decode_plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if !self.cached
            || rows == 0
            || rows > self.decode_max_rows()
            || rows > self.query.m_capacity
        {
            return Err(Refusal("exact F32 decode rows exceed the declared bucket"));
        }
        self.stage_plan([
            self.query.n.div_ceil(16),
            rows.div_ceil(8),
            self.query.k.div_ceil(128).div_ceil(4),
        ])
    }

    pub(crate) fn stage_plan(&self, grid: [u32; 3]) -> Result<LaunchPlan, Refusal> {
        let plan = LaunchPlan {
            grid,
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub(crate) fn cached_spec(&self) -> EntrySpec {
        let mut old = self.clone();
        old.cached = false;
        let mut spec = old.entry_spec();
        spec.params.push(pointer("weight_domain", "const uint32_t"));
        spec.body = format!(
            "chop::exact_f32::cached<{},{},{}>(activation,weight,output,live_rows,rows,weight_domain);",
            self.query.n,
            self.query.k,
            self.query.weight == WeightFormat::F32
        );
        spec
    }

    pub fn entry_specs(&self) -> Vec<EntrySpec> {
        if self.uses_prefill_tiles() {
            return self.prefill_specs();
        }
        if self.fused {
            return self.split_specs();
        }
        let mut specs = vec![self.entry_spec()];
        if !self.cached {
            return specs;
        }
        let q = self.query;
        let f32_weight = q.weight == WeightFormat::F32;
        let weight = pointer(
            "weight",
            if f32_weight {
                "const float"
            } else {
                "const uint16_t"
            },
        );
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let live = pointer("live_rows", "const uint32_t");
        let workspace = pointer("workspace", "float");
        let stage =
            |suffix, params, body| EntrySpec::new(self.stage_symbol(suffix), params, body, 128, 1);
        specs.push(stage(
            "_weights",
            vec![weight.clone(), pointer("weight_domain", "uint32_t")],
            format!(
                "chop::exact_f32::classify_weights<{},{},{}>(weight,weight_domain);",
                q.n, q.k, f32_weight
            ),
        ));
        specs.push(stage("_rows", vec![pointer("activation", "const float"), live.clone(), rows.clone(), workspace.clone()],
            format!("chop::exact_f32::classify_rows<{}>(activation,live_rows,rows,reinterpret_cast<uint32_t*>(workspace));", q.k)));
        specs.push(stage("_decode", vec![pointer("activation", "const float"), weight, workspace.clone(), live.clone(), rows.clone(), pointer("weight_domain", "const uint32_t")],
            format!("chop::exact_f32::decode<{},{},{}>(activation,weight,workspace+32,live_rows,rows,weight_domain,reinterpret_cast<const uint32_t*>(workspace));", q.n, q.k, f32_weight)));
        specs.push(stage(
            "_reduce",
            vec![workspace, pointer("output", "float"), live, rows],
            format!(
                "chop::exact_f32::reduce<{},{}>(workspace+32,output,live_rows,rows);",
                q.n, q.k
            ),
        ));
        specs
    }
}
