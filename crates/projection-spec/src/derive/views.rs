//! View byte extents follow the shared packing plan and explicit port contracts.

use kernel_core::contract::DType;
use kernel_core::precision::format::LayoutChoice;
use kernel_core::precision::layout::PackingPlan;
use kernel_core::weight_transform::PackedSection;

use super::{RequiredView, ViewRole};
use crate::Error;

pub(super) fn view(
    id: String,
    role: ViewRole,
    dtype: DType,
    shape: Vec<u64>,
    layout: String,
) -> Result<RequiredView, Error> {
    let elements = shape
        .iter()
        .try_fold(1u64, |n, d| n.checked_mul(*d))
        .ok_or_else(|| Error("required view shape overflows".into()))?;
    let bytes = elements
        .checked_mul(kernel_core::precision::dtype_bits(dtype))
        .and_then(|v| v.checked_add(7))
        .map(|v| v / 8)
        .ok_or_else(|| Error("required view byte extent overflows".into()))?;
    Ok(RequiredView {
        id,
        role,
        dtype,
        interpretation: None,
        shape,
        bytes,
        layout,
    })
}

pub(super) fn weight_views(
    id: &str,
    plan: &PackingPlan,
    layout: &LayoutChoice,
) -> Result<Vec<RequiredView>, Error> {
    let sizes = plan.section_bytes().map_err(|e| Error(e.to_string()))?;
    let code = if plan.format == kernel_core::contract::WeightFormat::Bf16 {
        DType::Bf16
    } else {
        DType::U8
    };
    let mut result = Vec::new();
    for (section, bytes, dtype, name, layout_id) in [
        (
            PackedSection::Codes,
            sizes.codes,
            code,
            "codes",
            Some(layout.codes),
        ),
        (
            PackedSection::LocalScales,
            sizes.local_scales,
            plan.local_scale_dtype.unwrap_or(DType::U8),
            "local",
            layout.local_scales,
        ),
        (
            PackedSection::GlobalScales,
            sizes.global_scales,
            DType::F32,
            "global",
            layout.global_scales,
        ),
    ] {
        if bytes == 0 {
            continue;
        }
        let width = kernel_core::precision::dtype_bits(dtype) / 8;
        if width == 0 || bytes % width != 0 {
            return Err(Error(
                "packed section byte extent disagrees with its storage dtype".into(),
            ));
        }
        result.push(view(
            format!("{id}.{name}"),
            ViewRole::Weight(section),
            dtype,
            vec![bytes / width],
            layout_id
                .ok_or_else(|| Error("plug-in omitted required section layout".into()))?
                .into(),
        )?);
    }
    Ok(result)
}
