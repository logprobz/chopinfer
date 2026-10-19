//! Two-launch ordered split-K with one activation expansion per row.
use super::{Entry, pointer};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

impl Entry {
    /// Select only between existing locked entries. At SM120, 250 W on
    /// 2026-10-19, 384x5120 measured 24.5/30.7 us at 1/32 rows with the
    /// three-launch path, versus 32.7/43.0 us fused. At 33/256 rows the
    /// fused path measured 51/145 us versus 1194/1210 us for the baseline.
    /// Keep the measured crossover at 32 rows.
    pub fn entry_for_rows(&self, rows: u32) -> Self {
        if self.fused && rows <= super::decode::MAX_ROWS {
            let mut entry = self.clone();
            entry.symbol = self.baseline_symbol().to_owned();
            entry.cached = false;
            entry.fused = false;
            entry.with_decode()
        } else {
            self.clone()
        }
    }

    pub fn with_split_decode(mut self) -> Self {
        self.cached = true;
        self.fused = true;
        self.symbol = format!("{}_split_v2", self.symbol);
        self
    }

    pub fn uses_fused_split(&self) -> bool {
        self.fused
    }

    pub fn decode_max_rows(&self) -> u32 {
        if self.fused {
            256
        } else {
            super::decode::MAX_ROWS
        }
    }

    pub(crate) fn split_workspace_bytes(&self, capacity: u32) -> u64 {
        let rows = u64::from(capacity.min(self.decode_max_rows()));
        let tiles = u64::from(self.query.n.div_ceil(16)) * rows.div_ceil(8);
        let align = |bytes: u64| bytes.div_ceil(16) * 16;
        align(4 * (rows + tiles))
            + align(rows * u64::from(self.query.k) * 6)
            + rows * u64::from(self.query.n) * u64::from(self.query.k.div_ceil(128)) * 4
    }

    pub(crate) fn prepare_plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        let tiles = u64::from(self.query.n.div_ceil(16)) * u64::from(rows.div_ceil(8));
        self.stage_plan([rows.max(tiles.div_ceil(128) as u32), 1, 1])
    }

    pub(crate) fn split_specs(&self) -> Vec<EntrySpec> {
        let q = self.query;
        let raw = q.weight == WeightFormat::F32;
        let weight = pointer("weight", if raw { "const float" } else { "const uint16_t" });
        let live = pointer("live_rows", "const uint32_t");
        let rows = EntryParam::U32 {
            name: "rows".into(),
        };
        let workspace = pointer("workspace", "void");
        let stage =
            |suffix, params, body| EntrySpec::new(self.stage_symbol(suffix), params, body, 128, 1);
        vec![
            stage(
                "_weights",
                vec![weight.clone(), pointer("weight_domain", "uint32_t")],
                format!(
                    "chop::exact_f32::classify_weights<{},{},{}>(weight,weight_domain);",
                    q.n, q.k, raw
                ),
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
                    "chop::exact_f32::split_v2::prepare<{},{}>(activation,live_rows,rows,workspace);",
                    q.n, q.k
                ),
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
                    "chop::exact_f32::split_v2::Workspace<{0},{1}> work(workspace,rows); chop::exact_f32::split_v2::compute<{0},{1},{2}>(activation,weight,work.partial,output,live_rows,rows,weight_domain,work.domain,work.components,work.tickets);",
                    q.n, q.k, raw
                ),
            ),
        ]
    }
}
