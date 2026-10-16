//! Resolve physical layouts through the format plug-in, retaining lookup aliases.
use kernel_core::contract::WeightFormat;
use kernel_core::precision::ScaleScope;
use kernel_core::precision::format::{KernelFamily, LayoutChoice, registry};

pub fn choice(format: WeightFormat, layout: &str) -> Result<LayoutChoice, String> {
    let family = if layout.ends_with("-fragment") {
        KernelFamily::Dense
    } else if layout.starts_with("moe-packed-") {
        KernelFamily::GroupedExpert
    } else {
        KernelFamily::Row
    };
    let global =
        (format == WeightFormat::Nvfp4).then_some(if layout == "moe-packed-ExpertProjection" {
            ScaleScope::Expert
        } else {
            ScaleScope::Tensor
        });
    let choice = registry::by_format(format)
        .map_err(|e| e.to_string())?
        .layout(family, global)
        .map_err(|e| e.to_string())?;
    if choice.weight.id != layout {
        return Err(format!("unavailable projection layout {format:?} {layout}"));
    }
    Ok(choice)
}

pub fn code_dtype(format: WeightFormat) -> &'static str {
    match format {
        WeightFormat::Bf16 => "BF16",
        WeightFormat::Nvfp4 | WeightFormat::Mxfp4 => "U8",
        _ => "F8_E4M3",
    }
}
