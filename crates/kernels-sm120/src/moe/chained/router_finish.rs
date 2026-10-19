//! Fuse cached F32 chain reduction, row selection and ordered metadata.
use super::floor::Entry;
use crate::moe::ptr;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Stage {
    Prepare,
    Finish,
}
pub const STAGES: [Stage; 2] = [Stage::Prepare, Stage::Finish];

pub fn symbol(e: &Entry, stage: Stage) -> String {
    e.symbol(super::floor::Stage::Metadata)
        .replace("_floor_", "_router_finish_")
        .replace(
            "_metadata_v1",
            match stage {
                Stage::Prepare => "_prepare_v1",
                Stage::Finish => "_finish_v1",
            },
        )
}

pub fn router_workspace_bytes(e: &Entry, rows: u32) -> u64 {
    (32 + u64::from(rows) * u64::from(e.experts()) * u64::from(e.hidden().div_ceil(128))) * 4
}

pub fn plan(e: &Entry, stage: Stage, rows: u32) -> Result<LaunchPlan, Refusal> {
    if rows == 0 || rows > 32 || e.experts() % 32 != 0 || e.top_k() > 8 {
        return Err(Refusal(
            "router epilogue requires 1..32 rows and whole expert warps",
        ));
    }
    let plan = LaunchPlan {
        grid: [rows, 1, 1],
        block: [
            match stage {
                Stage::Prepare => 128,
                Stage::Finish => e.experts().max(64 * e.top_k()),
            },
            1,
            1,
        ],
        cluster: [1; 3],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        pdl: true,
    };
    plan.validate()?;
    Ok(plan)
}

pub fn entry(e: &Entry, stage: Stage) -> EntrySpec {
    let rows = EntryParam::U32 {
        name: "rows".into(),
    };
    let (params, body, threads) = match stage {
        Stage::Prepare => (
            vec![
                ptr("input", "const float"),
                ptr("live", "const uint32_t"),
                rows,
                ptr("workspace", "float"),
                ptr("arrived", "uint32_t"),
            ],
            format!(
                "chop::moe::router_finish::prepare<{}>(input,live,rows,workspace,arrived);",
                e.hidden()
            ),
            128,
        ),
        Stage::Finish => (
            vec![
                ptr("workspace", "const float"),
                ptr("logits", "float"),
                ptr("live", "const int32_t"),
                rows,
                ptr("bias", "const float"),
                ptr("alternate_bias", "const float"),
                ptr("bias_select", "const uint8_t"),
                ptr("scores", "float"),
                ptr("ids", "int32_t"),
                ptr("weights", "float"),
                EntryParam::U32 {
                    name: "scale_bits".into(),
                },
                ptr("counts", "int32_t"),
                ptr("offsets", "int32_t"),
                ptr("sorted", "int32_t"),
                ptr("padded", "int32_t"),
                ptr("active", "int32_t"),
                ptr("scratch", "void"),
                ptr("gate_tickets", "uint32_t"),
                ptr("arrived", "uint32_t"),
            ],
            format!(
                "chop::moe::router_finish::finish<{},{},{}>(workspace,logits,live,rows,bias,alternate_bias,bias_select,scores,ids,weights,__uint_as_float(scale_bits),counts,offsets,sorted,padded,active,scratch,gate_tickets,arrived);",
                e.experts(),
                e.top_k(),
                e.hidden()
            ),
            e.experts().max(64 * e.top_k()),
        ),
    };
    EntrySpec::new(symbol(e, stage), params, body, threads, 1)
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    super::spec(
        name.into(),
        "router_finish.cuh",
        entries
            .iter()
            .flat_map(|e| STAGES.into_iter().map(move |s| entry(e, s)))
            .collect(),
    )
}
