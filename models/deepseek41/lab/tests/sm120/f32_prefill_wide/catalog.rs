use bundle_deepseek41::sm120::{f32_blocked, f32_prefill, fold_cubin_for_symbol};
use kernel_build::CubinSpec;
use kernel_core::contract::LaunchPlan;
use kernels_sm120::f32_projection::{Entry, blocked, wide};
use model_deepseek41_contract::geometry::FLASH;
use serde_json::{Value, json};

pub const FAMILY: &str = "f32_prefill_wide";
pub const CUBIN: &str = "f32_prefill_wide_deepseek41_sm120";
pub const PROBES: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
pub const PATTERNS: &[&str] = &[
    "bf16",
    "tensor",
    "row_fallback",
    "weight_fallback",
    "mixed",
    "edges",
];

pub fn entries() -> Vec<Entry> {
    f32_prefill::baseline_entries(&FLASH).unwrap()
}
pub fn baseline_module() -> String {
    f32_blocked::cubin_name(&FLASH)
}
pub fn specs() -> Vec<CubinSpec> {
    let entries = entries();
    let mut spec = wide::cubin_spec(CUBIN, &entries).unwrap();
    let e = entries.iter().find(|e| e.query().n == 384).unwrap();
    for (features, name, parameters) in [
        (64, "chain", ",false,true,false>("),
        (128, "neighbor", ",false,false,true>("),
    ] {
        let mut control = wide::entry_spec(e, features).unwrap();
        control.symbol = format!("chop_f32_wide_{name}_negative_v1");
        control.body = control.body.replace(",false>(", parameters);
        assert!(control.body.contains(parameters));
        spec.entries.push(control);
    }
    vec![spec]
}

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub entry: usize,
    pub rows: u32,
    pub tile: u32,
    pub wide: bool,
    pub pattern: &'static str,
}
impl Case {
    pub fn json(self) -> Value {
        let e = &entries()[self.entry];
        json!({"n":e.query().n,"k":e.query().k,"rows":self.rows,"tile":self.tile,
            "kind":if self.wide {"wide"} else {"skip"},"pattern":self.pattern})
    }
    pub fn timed(self) -> bool {
        !matches!(self.pattern, "mixed" | "edges")
    }
    pub fn symbol(self, e: &Entry) -> String {
        if self.wide {
            wide::symbol(e, self.tile)
        } else {
            blocked::symbol(e, self.tile)
        }
    }
    pub fn plan(self, e: &Entry) -> LaunchPlan {
        if self.wide {
            wide::plan(e, self.tile, self.rows)
        } else {
            blocked::plan(e, self.tile, self.rows)
        }
        .unwrap()
    }
}
pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for (entry, e) in entries().iter().enumerate() {
        let rows: Vec<_> = match (e.query().n, e.query().k) {
            (384, 5120) => (320..=2496).step_by(128).collect(),
            (32, 5120) => vec![448, 960, 1472, 1984, 2496],
            _ => Vec::new(),
        };
        for rows in rows {
            for tile in [32, 64] {
                for &pattern in PATTERNS {
                    cases.push(Case {
                        entry,
                        rows,
                        tile,
                        wide: false,
                        pattern,
                    });
                }
            }
        }
    }
    for entry in 0..entries().len() {
        for tile in [64, 128] {
            for &pattern in PATTERNS {
                cases.push(Case {
                    entry,
                    rows: 8192,
                    tile,
                    wide: true,
                    pattern,
                });
            }
        }
    }
    cases
}
pub fn stage_plan(grid: [u32; 3]) -> LaunchPlan {
    LaunchPlan {
        grid,
        block: [128, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    }
}
pub fn generic(e: &Entry) -> (&'static str, &'static [u8]) {
    fold_cubin_for_symbol(e.baseline_symbol()).expect("serving generic reference")
}
