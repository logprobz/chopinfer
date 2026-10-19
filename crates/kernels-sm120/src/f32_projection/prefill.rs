//! Exact prefill candidates. Admission requires measured bucket evidence.
use super::{Entry, pointer};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

impl Entry {
    pub fn with_prefill_tiles(mut self, features: u32) -> Result<Self, Refusal> {
        if !matches!(features, 32 | 64) || self.cached {
            return Err(Refusal(
                "prefill tiles require a baseline entry and 32 or 64 features",
            ));
        }
        self.cached = true;
        self.prefill_tile_n = features;
        self.symbol = format!("{}_tiles_n{features}_v1", self.symbol);
        Ok(self)
    }

    pub fn uses_prefill_tiles(&self) -> bool {
        self.prefill_tile_n != 0
    }

    pub fn prefill_workspace_bytes(&self, rows: u32) -> u64 {
        u64::from(rows).div_ceil(4) * 16 + u64::from(rows) * u64::from(self.query.k) * 6
    }

    pub fn prefill_plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if !self.uses_prefill_tiles() || rows == 0 || rows > self.query.m_capacity {
            return Err(Refusal("exact prefill rows exceed the declared capacity"));
        }
        let plan = LaunchPlan {
            grid: [
                self.query.n.div_ceil(self.prefill_tile_n),
                rows.div_ceil(32),
                1,
            ],
            block: [self.prefill_tile_n * 8, 1, 1],
            dynamic_shared_bytes: 2 * (self.prefill_tile_n * 128 * 2 + 3 * 32 * 128 * 2),
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
    }

    pub(crate) fn prefill_specs(&self) -> Vec<EntrySpec> {
        let q = self.query;
        let raw = q.weight == WeightFormat::F32;
        let weight = pointer("weight", if raw { "const float" } else { "const uint16_t" });
        let live = pointer("live_rows", "const uint32_t");
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let workspace = pointer("workspace", "void");
        let stage = |suffix, params, body, threads| {
            EntrySpec::new(self.stage_symbol(suffix), params, body, threads, 1)
        };
        vec![
            stage(
                "_weights",
                vec![weight.clone(), pointer("weight_domain", "uint32_t")],
                format!(
                    "chop::exact_f32::classify_weights<{},{},{}>(weight,weight_domain);",
                    q.n, q.k, raw
                ),
                128,
            ),
            stage(
                "_prepare",
                vec![
                    pointer("activation", "const float"),
                    live.clone(),
                    rows.clone(),
                    workspace.clone(),
                ],
                format!(
                    "chop::exact_f32::prefill::prepare<{}>(activation,live_rows,rows,workspace);",
                    q.k
                ),
                128,
            ),
            stage(
                "",
                vec![
                    pointer("activation", "const float"),
                    weight,
                    workspace,
                    pointer("output", "float"),
                    live,
                    rows,
                    pointer("weight_domain", "const uint32_t"),
                ],
                format!(
                    "chop::exact_f32::prefill::run<{},{},{},{}>(activation,weight,output,live_rows,rows,weight_domain,workspace);",
                    q.n, q.k, raw, self.prefill_tile_n
                ),
                self.prefill_tile_n * 8,
            ),
        ]
    }
}
