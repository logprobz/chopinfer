//! Frozen exact prefill entries qualified together. Serving selects one gate/up variant.
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernels_sm120::moe::chained::quantized::Projection;
use model_deepseek41_contract::{geometry::FLASH, moe};
pub const FAMILY: &str = "exact_moe_prefill";
pub const CUBIN: &str = "exact_moe_prefill_sm120";
pub const CTAS: u32 = 376;
pub const STAMP_BYTES: u64 = 4 * CTAS as u64 * 8 * 8;
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub rows: u32,
    pub columns: u32,
    pub stages: u32,
    pub async_scale: bool,
    pub predecoded: bool,
    pub dynamic: bool,
}
pub const VARIANTS: [Variant; 8] = [
    Variant {
        name: "serial64",
        rows: 64,
        columns: 64,
        stages: 1,
        async_scale: false,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "pipe64",
        rows: 64,
        columns: 64,
        stages: 2,
        async_scale: false,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "pipe128",
        rows: 128,
        columns: 64,
        stages: 2,
        async_scale: false,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "scale128",
        rows: 128,
        columns: 64,
        stages: 2,
        async_scale: true,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "stage3",
        rows: 128,
        columns: 64,
        stages: 3,
        async_scale: true,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "wide128",
        rows: 64,
        columns: 128,
        stages: 2,
        async_scale: true,
        predecoded: false,
        dynamic: false,
    },
    Variant {
        name: "decoded128",
        rows: 128,
        columns: 64,
        stages: 2,
        async_scale: true,
        predecoded: true,
        dynamic: false,
    },
    Variant {
        name: "dynamic128",
        rows: 128,
        columns: 64,
        stages: 2,
        async_scale: true,
        predecoded: false,
        dynamic: true,
    },
];
pub fn projections() -> Vec<Projection> {
    moe::expert_sets(&FLASH)
        .into_iter()
        .flat_map(|set| {
            [GatedMlpProjection::GateUp, GatedMlpProjection::Down].map(|kind| {
                Projection::new(
                    moe::projection(&FLASH, 4, set, kind, 8192).unwrap(),
                    64,
                    8192,
                )
                .unwrap()
            })
        })
        .collect()
}
pub fn projection(experts: u32, gate: bool) -> Projection {
    projections()
        .into_iter()
        .find(|p| {
            p.gate_up() == gate && p.query.projection.gemm.grouped.unwrap().experts == experts
        })
        .unwrap()
}
pub fn tile_rows(v: Variant) -> Vec<u32> {
    [8, 32, 64, 128]
        .into_iter()
        .filter(|&r| r <= v.rows)
        .collect()
}
pub fn symbol(p: &Projection, v: Variant, stage: &str, stamp: bool, negative: bool) -> String {
    format!(
        "chop_exact_prefill_e{}_before_{}_{}_{}{}{}_v1",
        p.query.projection.gemm.grouped.unwrap().experts,
        if p.gate_up() { "gateup" } else { "down" },
        v.name,
        stage,
        if stamp { "_clock" } else { "" },
        if negative { "_omit" } else { "" }
    )
}
pub fn workspace_bytes(p: &Projection, v: Variant) -> u64 {
    let g = p.query.projection.gemm.grouped.unwrap();
    ((8 + 12
        * ((u64::from(p.tokens_capacity) * u64::from(g.top_k)).div_ceil(u64::from(v.rows))
            + u64::from(g.experts)))
        * 4)
    .next_multiple_of(256)
}
pub fn shared_bytes(p: &Projection, v: Variant, rows: u32) -> u32 {
    let wr = if p.gate_up() {
        2 * v.columns
    } else {
        v.columns
    };
    let raw = (wr * 32 + rows * if v.predecoded { 128 } else { 64 } + (wr + rows) * 4)
        .next_multiple_of(16);
    (v.stages * raw + (wr + rows) * 72 * 2 + rows * 4 + 4).next_multiple_of(16)
}
#[derive(Clone, Copy)]
pub struct Buffers {
    pub h: [u64; 10],
    pub counts: u64,
    pub active: u64,
    pub scratch: u64,
    pub decoded: u64,
    pub stamps: u64,
}
pub fn launches(
    p: &Projection,
    v: Variant,
    rows: u32,
    buffers: Buffers,
    stamp: bool,
    negative: bool,
) -> Result<Vec<Launch>, String> {
    let Buffers {
        h,
        counts,
        active,
        scratch,
        decoded,
        stamps,
    } = buffers;
    if rows == 0 || rows > 8192 {
        return Err("prefill rows outside 1..8192".into());
    }
    let ptr = |h| {
        if h == 0 {
            Argument::NullDevicePointer
        } else {
            Argument::DeviceBuffer(h)
        }
    };
    let make = |stage: &str, handles: Vec<u64>, grid, threads, shared, clock, omit| Launch {
        symbol: symbol(p, v, stage, clock, omit),
        rows,
        plan: LaunchPlan {
            grid: [grid, 1, 1],
            block: [threads, 1, 1],
            cluster: [1; 3],
            dynamic_shared_bytes: shared,
            workspace_bytes: workspace_bytes(p, v),
            pdl: true,
        },
        arguments: ArgumentList(handles.into_iter().map(ptr).collect()),
        collective: false,
        persistent: false,
    };
    let mut ops = vec![make(
        "pack",
        vec![counts, h[4], active, h[7], scratch],
        1,
        32,
        0,
        false,
        false,
    )];
    if v.predecoded {
        ops.push(make(
            "predecode",
            vec![h[0], h[3], h[7], decoded],
            CTAS,
            256,
            0,
            false,
            false,
        ));
    }
    for r in tile_rows(v) {
        ops.push(make(
            &format!("r{r}"),
            vec![
                if v.predecoded { decoded } else { h[0] },
                h[1],
                h[2],
                h[3],
                h[5],
                h[6],
                h[7],
                scratch,
                h[8],
                h[9],
                stamps,
            ],
            CTAS,
            if r > 64 { 2 } else { 1 } * (v.columns / 16) * 32,
            shared_bytes(p, v, r),
            stamp,
            negative,
        ));
    }
    Ok(ops)
}
fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}
fn config(p: &Projection) -> String {
    let q = p.query.projection;
    let g = q.gemm.grouped.unwrap();
    let clamp = q.swiglu_clamp.map_or_else(
        || "chop::moe::clamp::SwigluClamp<false,0u,0u>".into(),
        |c| {
            format!(
                "chop::moe::clamp::SwigluClamp<true,{}u,{}u>",
                c.gate_max_bits, c.up_limit_bits
            )
        },
    );
    format!(
        "chop::moe::chained::quantized::Config<chop::moe::chained::Projection::{},{},{},{},{},64,8192,{clamp}>",
        if p.gate_up() { "gate_up" } else { "down" },
        g.experts,
        g.top_k,
        p.columns(),
        q.gemm.k
    )
}
pub fn cubins() -> Vec<CubinSpec> {
    let mut entries = Vec::new();
    for p in projections() {
        for v in VARIANTS {
            let c = config(&p);
            let dependency = "chop::ptx::griddep_launch_dependents(); chop::ptx::griddep_wait();";
            entries.push(EntrySpec::new(symbol(&p,v,"pack",false,false),vec![ptr("counts","const int"),ptr("offsets","const int"),ptr("active","const int"),ptr("live","const int"),ptr("scratch","int")],format!("{dependency} chop::moe::prefill_pipeline::pack<{c},{}>(counts,offsets,active,live,scratch);",v.rows),32,1));
            if v.predecoded {
                entries.push(EntrySpec::new(symbol(&p,v,"predecode",false,false),vec![ptr("input","const uint8_t"),ptr("scales","const uint8_t"),ptr("live","const int"),ptr("output","uint16_t")],format!("{dependency} chop::moe::prefill_pipeline::predecode<{c}>(input,scales,live,output);"),256,1));
            }
            for r in tile_rows(v) {
                for (stamp, negative) in [(false, false), (true, false), (false, true)] {
                    // One down mutation per tile is enough to test the bitwise verifier.
                    if negative && (p.gate_up() || v.name != "pipe64") {
                        continue;
                    }
                    let params = vec![
                        ptr("input", "const uint8_t"),
                        ptr("weight", "const uint8_t"),
                        ptr("weight_scales", "const uint8_t"),
                        ptr("activation_scales", "const uint8_t"),
                        ptr("sorted", "const int"),
                        ptr("padded", "const int"),
                        ptr("live", "const int"),
                        ptr("scratch", "int"),
                        ptr("route_weights", "const float"),
                        ptr("output", "void"),
                        ptr("stamps", "uint64_t"),
                    ];
                    let body = format!(
                        "static_assert(sizeof(chop::moe::prefill_pipeline::Shared<{c},{r},{},{},{}>) == {}); extern __shared__ __align__(16) uint8_t arena[]; {dependency} chop::moe::prefill_pipeline::task<{c},{r},{},{},{},{},{},{},{stamp},{negative}>(blockIdx.x,gridDim.x,arena,input,weight,weight_scales,activation_scales,sorted,padded,live,scratch,route_weights,output,stamps);",
                        v.columns,
                        v.stages,
                        v.predecoded,
                        shared_bytes(&p, v, r),
                        v.columns,
                        v.stages,
                        v.rows,
                        v.async_scale,
                        v.predecoded,
                        v.dynamic
                    );
                    let threads = if r > 64 { 2 } else { 1 } * (v.columns / 16) * 32;
                    entries.push(EntrySpec::new(
                        symbol(&p, v, &format!("r{r}"), stamp, negative),
                        params,
                        body,
                        threads,
                        1,
                    ));
                }
            }
        }
    }
    vec![CubinSpec {
        name: CUBIN.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["crates/kernels-sm120/cuda".into()],
        headers: vec!["#include \"moe/prefill_pipeline.cuh\"".into()],
        defines: vec![],
        entries,
    }]
}
