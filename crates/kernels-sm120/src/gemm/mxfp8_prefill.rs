//! Native MXFP8 prefill tiles over the shared TMA ring, with fold128 arithmetic.
use super::mxfp8_fold::{self, Entry};
use super::mxfp8_fold_packed;
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::backend::Launch;
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

fn validate(e: &Entry) -> Result<(), Refusal> {
    e.validate()?;
    let q = e.query;
    if q.k > i32::MAX as u32 - 127
        || q.groups
            .checked_mul(q.n)
            .is_none_or(|height| height > i32::MAX as u32)
    {
        return Err(Refusal(
            "prefill TMA coordinates or rounded K exceed signed bounds",
        ));
    }
    Ok(())
}

pub fn specs(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    validate(e)?;
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

/// Refuses entries without a packed twin: the layout's N and K bounds
/// (`gemm::mxfp8_fold_packed::check`), and K below one complete 128-K fold,
/// which the packed weight map needs.
pub fn check_packed(e: &Entry) -> Result<(), Refusal> {
    validate(e)?;
    let q = e.query;
    mxfp8_fold_packed::check(q.groups, q.n, q.k)?;
    if q.k < 128 {
        return Err(Refusal("packed TMA prefill needs a complete 128-K fold"));
    }
    Ok(())
}

pub fn packed_symbol(e: &Entry) -> String {
    mxfp8_fold_packed::symbol(&symbol(e))
}

/// The compute entry reading `mxfp8-fold-packed-8x128-v1` weight codes: the
/// parameters, plan and shared memory of [`symbol`]'s entry, launched with
/// [`packed_arguments`].
pub fn packed_spec(e: &Entry) -> Result<EntrySpec, Refusal> {
    check_packed(e)?;
    let mut spec = specs(e)?.remove(1);
    spec.symbol = packed_symbol(e);
    spec.body = spec
        .body
        .replace(",false>(activation_map", ",false,true>(activation_map");
    Ok(spec)
}

/// The weight map of a packed compute entry, over the packed codes in
/// `weights`. Its box is one 128-K fold of 16 eight-row strips (the same 16
/// KB as the row-major box): dims {128, 8, K / 128, groups * N / 8}. A final
/// 64-K fold is outside the map; the kernel copies it per strip.
pub fn packed_weight_map(e: &Entry, weights: u64) -> Result<TensorMap, Refusal> {
    check_packed(e)?;
    let q = e.query;
    let strips = u64::from(q.groups) * u64::from(q.n) / 8;
    let map = TensorMap {
        base_buffer: weights,
        dims: vec![128, 8, u64::from(q.k / 128), strips],
        strides_bytes: vec![1, 128, 1024, 8 * u64::from(q.k)],
        box_dims: vec![128, 8, 1, TILE / 8],
        element_strides: vec![1, 1, 1, 1],
        swizzle: TensorMapSwizzle::B128,
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::B128,
        oob_fill: TensorMapOobFill::Zero,
        element: TensorMapElement::U8,
    };
    map.validate()?;
    Ok(map)
}

/// [`arguments`] for the packed compute entry: the packed weight map in
/// place of the row-major one; `b.weights` names the packed codes.
pub fn packed_arguments(
    e: &Entry,
    b: Mxfp8FoldGemmBuffers<u64>,
    rows: u32,
) -> Result<ArgumentList, Refusal> {
    let mut result = arguments(e, b, rows)?;
    result.0[1] = Argument::TensorMap(packed_weight_map(e, b.weights)?);
    Ok(result)
}

/// Moves one launch made with [`arguments`] to the packed entry `symbol`:
/// renamed, with the packed weight map over the codes its map named.
pub(crate) fn move_to_packed(
    e: &Entry,
    launch: &mut Launch,
    symbol: String,
) -> Result<(), Refusal> {
    let q = e.query;
    let row_major = [u64::from(q.k), u64::from(q.groups) * u64::from(q.n)];
    let Some(Argument::TensorMap(map)) = launch.arguments.0.get(1) else {
        return Err(Refusal("prefill launch has no weight map"));
    };
    if map.dims != row_major {
        return Err(Refusal("prefill launch has no row-major weight map"));
    }
    launch.arguments.0[1] = Argument::TensorMap(packed_weight_map(e, map.base_buffer)?);
    launch.symbol = symbol;
    Ok(())
}

/// `launches` of a prepared prefill op (`Sm120::with_mxfp8_prefill`) moved
/// to packed weight codes: the compute launch takes [`packed_symbol`] and the
/// packed weight map. The validation launch reads no codes and is kept.
pub fn packed_launches(e: &Entry, mut launches: Vec<Launch>) -> Result<Vec<Launch>, Refusal> {
    let compute = symbol(e);
    for launch in &mut launches {
        if launch.symbol == compute {
            move_to_packed(e, launch, packed_symbol(e))?;
        }
    }
    Ok(launches)
}

pub fn plan(e: &Entry, rows: u32) -> Result<LaunchPlan, Refusal> {
    validate(e)?;
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
    validate(e)?;
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
