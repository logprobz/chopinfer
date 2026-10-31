//! Lab candidate for the Flash TP4 direct MoE at 9 to 64 rows: the served
//! absorbing gate queue, bandwidth gate finisher, two-strip down queue and
//! floor down finisher of the backbone bank with L2 cache-policy knobs
//! (`lab/cuda/moe_small_l2.cuh`). Each entry keeps its served twin's ABI,
//! launch bounds and body; only the device function it calls and the policy
//! change. No serving catalog selects them.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::chained::{
    bandwidth, down_scale_tiles,
    floor::{Entry, Stage},
    lanes,
    quantized::Projection,
};
use model_deepseek41_contract::{geometry::FLASH, moe};
use std::path::PathBuf;

pub const FAMILY: &str = "moe_small_l2";
pub const CUBIN: &str = "moe_small_l2_deepseek41_sm120";
/// The timing harness's read flush (`read_flush`): data, 16-byte words,
/// sink; grid-stride, 256 threads.
pub const READ_FLUSH: &str = "chop_deepseek41_lab_l2_read_flush_v1";

/// Each knob set's symbol tag and its Stream, Keep, Retire, Loads, Fold and
/// Early knobs.
pub const VARIANTS: [(&str, [bool; 6]); 8] = [
    ("l2k", [false, true, false, false, false, false]),
    ("l2sk", [true, true, false, false, false, false]),
    ("l2skr", [true, true, true, false, false, false]),
    ("l2skrl", [true, true, true, true, false, false]),
    ("l2skrf", [true, true, true, false, true, false]),
    ("l2skre", [true, true, true, false, false, true]),
    ("l2skrlf", [true, true, true, true, true, false]),
    ("l2skrlfe", [true, true, true, true, true, true]),
];

/// The candidate twin of a served symbol under a knob set's tag.
pub fn candidate_symbol(served: &str, tag: &str) -> String {
    format!(
        "{}_{tag}_v1",
        served.strip_suffix("_v1").expect("versioned symbol")
    )
}

/// The backbone bank's direct entry, as `bundle/src/sm120/moe_direct_kernels.rs`
/// builds it from the streamed projections of `bundle/src/sm120/moe.rs`.
pub fn entry() -> Result<Entry, String> {
    let set = moe::expert_sets(&FLASH)[0];
    let projection = |kind| -> Result<Projection, String> {
        let mut p = Projection::new(moe::projection(&FLASH, 4, set, kind, 64)?, 64, 64)
            .and_then(Projection::with_streamed_chains)
            .map_err(|e| e.to_string())?;
        p.symbol = format!("{}_{}", FLASH.kernel_tag, p.symbol);
        Ok(p)
    };
    Entry::new(
        projection(GatedMlpProjection::GateUp)?,
        projection(GatedMlpProjection::Down)?,
        1e-4_f32.to_bits(),
        format!(
            "chop_{}_moe_floor_e{}_k{}",
            FLASH.kernel_tag, set.count, set.top_k
        ),
    )
    .map_err(|e| e.to_string())
}

/// The served entries of the 9- to 64-row steps with width-1 tile scales:
/// gate queue, gate finisher, down queue, down finisher.
pub fn served(e: &Entry) -> Result<[EntrySpec; 4], String> {
    Ok([
        lanes::absorb_scale_tiles_entry(e, 1).map_err(|e| e.to_string())?,
        bandwidth::entry(e, Stage::GateFinish, bandwidth::Loads::Direct),
        down_scale_tiles::strip2_entry(e, 1).map_err(|e| e.to_string())?,
        e.entry(Stage::DownFinish),
    ])
}

fn replace_once(body: &str, from: &str, to: &str) -> String {
    assert_eq!(body.matches(from).count(), 1, "{from} in {body}");
    body.replace(from, to)
}

/// `served`'s twins under one knob set.
pub fn candidates(e: &Entry, tag: &str, knobs: [bool; 6]) -> Result<Vec<EntrySpec>, String> {
    let policy = format!(
        "chop::moe::lab_l2::Policy<{}>",
        knobs.map(|k| k.to_string()).join(",")
    );
    let [gate, gate_finish, down, down_finish] = served(e)?;
    let twin = |mut spec: EntrySpec, from: &str, to: &str, call: &str, with: String| {
        spec.symbol = candidate_symbol(&spec.symbol, tag);
        spec.body = replace_once(&replace_once(&spec.body, from, to), call, &with);
        spec
    };
    Ok(vec![
        twin(
            gate,
            "chop::moe::lanes::absorbing<",
            "chop::moe::lab_l2::absorbing<",
            ">(input,",
            format!(",{policy}>(input,"),
        ),
        twin(
            gate_finish,
            "chop::moe::bandwidth::gate_finish<",
            "chop::moe::lab_l2::gate_finish<",
            "u>(scratch,",
            format!("u,{policy}>(scratch,"),
        ),
        twin(
            down,
            "chop::moe::strip2::compute<",
            "chop::moe::lab_l2::strip2<",
            ">(input,",
            format!(",{policy}>(input,"),
        ),
        twin(
            down_finish,
            "chop::moe::floor::down_finish<",
            "chop::moe::lab_l2::down_finish<",
            ",false>(scratch,",
            format!(",{policy}>(scratch,"),
        ),
    ])
}

pub fn cubins() -> Result<Vec<CubinSpec>, String> {
    let e = entry()?;
    let mut entries = Vec::new();
    for (tag, knobs) in VARIANTS {
        entries.extend(candidates(&e, tag, knobs)?);
    }
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    entries.push(EntrySpec::new(
        READ_FLUSH.into(),
        vec![
            ptr("data", "const uint8_t"),
            EntryParam::U64 {
                name: "words".into(),
            },
            ptr("sink", "uint32_t"),
        ],
        "chop::moe::lab_l2::read_flush(reinterpret_cast<const uint4*>(data),words,sink);".into(),
        256,
        1,
    ));
    Ok(vec![CubinSpec {
        name: CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec![
            "crates/kernels-sm120/cuda".into(),
            "models/deepseek41/lab/cuda".into(),
        ],
        headers: vec!["#include \"moe_small_l2.cuh\"".into()],
        defines: Vec::new(),
        entries,
    }])
}
