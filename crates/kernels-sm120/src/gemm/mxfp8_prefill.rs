//! Native MXFP8 prefill tiles over the shared TMA ring, with fold128 arithmetic.
use super::mxfp8_fold::{self, Entry};
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, LaunchPlan, Refusal, TensorMap, TensorMapElement,
    TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle, Validate,
};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmBuffers;

pub const TILE: u32 = 128;
pub const THREADS: u32 = 384;
pub const SHARED: u32 = 2 * (128 * 128 * 2) + 1024;

pub fn symbol(e: &Entry) -> String {
    format!("{}_prefill_tma_v1", e.symbol)
}
pub fn validation_symbol(e: &Entry) -> String {
    format!("{}_admit_v1", symbol(e))
}
pub fn control_symbol(e: &Entry) -> String {
    format!("{}_single_chain_control_v1", symbol(e))
}

pub fn specs(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    e.validate()?;
    let q = e.query;
    let ptr = |name: &str, ty: &str| P::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    let validation = EntrySpec::new(
        validation_symbol(e),
        vec![
            ptr("activations", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("live_rows", "const uint32_t"),
            ptr("status", "uint32_t"),
            P::U64 {
                name: "status_row_offset".into(),
            },
            P::U32 {
                name: "rows".into(),
            },
        ],
        format!(
            "chop::gemm::mxfp8_prefill::validate<{},{},{},{}>(activations,activation_scales,live_rows,status,status_row_offset,rows);",
            q.k, q.groups, q.activation_row_stride, q.activation_group_stride
        ),
        128,
        1,
    );
    let compute = EntrySpec::new(
        symbol(e),
        vec![
            P::TensorMap {
                name: "activation_map".into(),
            },
            P::TensorMap {
                name: "weight_map".into(),
            },
            ptr("activations", "const uint8_t"),
            ptr("weights", "const uint8_t"),
            ptr("activation_scales", "const uint8_t"),
            ptr("weight_scales", "const uint8_t"),
            ptr(
                "output",
                if q.output == DType::F32 {
                    "float"
                } else {
                    "uint16_t"
                },
            ),
            ptr("live_rows", "const uint32_t"),
            ptr("status", "const uint32_t"),
            P::U64 {
                name: "status_row_offset".into(),
            },
            P::U32 {
                name: "rows".into(),
            },
        ],
        format!(
            "chop::gemm::mxfp8_prefill::run<{},{},{},{},{},{},false>(activation_map,weight_map,activations,weights,activation_scales,weight_scales,output,live_rows,status,status_row_offset,rows);",
            q.n,
            q.k,
            q.groups,
            q.activation_row_stride,
            q.activation_group_stride,
            q.output == DType::F32
        ),
        THREADS,
        1,
    );
    Ok(vec![validation, compute])
}

pub fn control_spec(e: &Entry) -> Result<EntrySpec, Refusal> {
    let mut spec = specs(e)?.remove(1);
    spec.symbol = control_symbol(e);
    spec.body = spec
        .body
        .replace(",false>(activation_map", ",true>(activation_map");
    Ok(spec)
}

pub fn plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let mut p = e.plan(rows)?;
    p.grid = [
        e.query.n.div_ceil(TILE),
        rows.max(1).div_ceil(TILE),
        e.query.groups,
    ];
    p.block = [THREADS, 1, 1];
    p.dynamic_shared_bytes = SHARED;
    crate::backend::check_plan(&p)?;
    Ok(p)
}

pub fn validation_plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    let mut p = e.plan(rows)?;
    p.grid = [rows.max(1), 1, 1];
    p.block = [128, 1, 1];
    Ok(p)
}

pub fn arguments(
    e: &Entry,
    b: Mxfp8FoldGemmBuffers<u64>,
    rows: u32,
) -> Result<ArgumentList, Refusal> {
    let q = e.query;
    let map = |base_buffer, width: u32, height: u32| TensorMap {
        base_buffer,
        dims: vec![u64::from(width), u64::from(height.max(1))],
        strides_bytes: vec![1, u64::from(width)],
        box_dims: vec![128, 128],
        element_strides: vec![1, 1],
        swizzle: TensorMapSwizzle::B128,
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::B128,
        oob_fill: TensorMapOobFill::Zero,
        element: TensorMapElement::U8,
    };
    let weights = q
        .groups
        .checked_mul(q.n)
        .ok_or(Refusal("prefill weight row count overflows"))?;
    let result = ArgumentList(vec![
        Argument::TensorMap(map(b.activations, q.activation_row_stride, rows)),
        Argument::TensorMap(map(b.weights, q.k, weights)),
        Argument::DeviceBuffer(b.activations),
        Argument::DeviceBuffer(b.weights),
        Argument::DeviceBuffer(b.activation_scales),
        Argument::DeviceBuffer(b.weight_scales),
        Argument::DeviceBuffer(b.output),
        Argument::DeviceBuffer(b.live_rows),
        Argument::DeviceBuffer(b.status),
        Argument::U64(b.status_row_offset),
        Argument::U32(rows),
    ]);
    result.validate()?;
    Ok(result)
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut spec = mxfp8_fold::cubin_spec(name, entries)?;
    spec.headers = vec![r#"#include "gemm/mxfp8_prefill.cuh""#.into()];
    spec.entries = entries
        .iter()
        .map(specs)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    Ok(spec)
}
