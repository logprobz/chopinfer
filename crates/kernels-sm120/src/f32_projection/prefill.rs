//! Exact prefill candidates. Admission requires measured bucket evidence.
use super::{Entry, pointer};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

impl Entry {
    /// Admit exact row counts chosen by the caller's measured selector.
    /// This changes runtime dispatch, never the locked cubin specification.
    pub fn with_prefill_buckets(mut self, mut buckets: Vec<(u32, u32)>) -> Result<Self, Refusal> {
        buckets.sort_unstable();
        if !self.fused
            || self.uses_prefill_tiles()
            || buckets.iter().any(|&(rows, tile)| {
                rows <= self.decode_max_rows()
                    || rows > self.query.m_capacity
                    || !matches!(tile, 32 | 64)
            })
            || buckets.windows(2).any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(Refusal(
                "prefill buckets require distinct bounded rows above split decode",
            ));
        }
        self.prefill_buckets = buckets;
        Ok(self)
    }

    pub fn prefill_buckets(&self) -> &[(u32, u32)] {
        &self.prefill_buckets
    }

    pub fn prefill_features(&self) -> u32 {
        self.prefill_tile_n
    }

    pub(crate) fn prefill_variant(&self, features: u32) -> Self {
        Entry::new(
            self.query,
            self.usage,
            self.baseline_symbol().trim_end_matches("_v1"),
        )
        .expect("same validated entry")
        .with_prefill_tiles(features)
        .expect("validated tile size")
    }

    pub(crate) fn runtime_shared_bytes(&self, symbol: &str) -> u32 {
        if self.uses_prefill_tiles() && self.symbol() == symbol {
            return self
                .prefill_plan(1)
                .expect("validated tile")
                .dynamic_shared_bytes;
        }
        for features in [32, 64] {
            if self
                .prefill_buckets
                .iter()
                .any(|&(_, tile)| tile == features)
            {
                let entry = self.prefill_variant(features);
                if entry.symbol() == symbol {
                    return entry
                        .prefill_plan(1)
                        .expect("validated tile")
                        .dynamic_shared_bytes;
                }
            }
        }
        0
    }

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
