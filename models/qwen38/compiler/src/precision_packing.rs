//! Rank slices supplied by Qwen to the shared packed-weight builder. Physical
//! scale tiling and encoding conversion stay in the SDK.
use crate::{Result, catalog::Catalog};
use kernel_core::contract::{DType, SwigluColumns, WeightFormat};
use kernel_core::precision::execution::PrecisionDocument;
use kernel_core::precision::{DecodeOrder, ScaleScope};
use model_compiler_sdk::quantization::packing::PackingPlan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::Range;

/// Keep the export-wide loader flavor, including BF16 projections in a mixed image.
pub fn projection_record_policy(
    source: &crate::precision_source::SourcePrecision,
) -> model_compiler_sdk::quantization::projection::RecordPolicy {
    use model_compiler_sdk::quantization::{format::RecordLayout, projection::RecordPolicy};
    if source.legacy_text_layout() {
        RecordPolicy::Sections(RecordLayout::Fragment)
    } else {
        RecordPolicy::PackedPieces
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Section {
    Codes,
    LocalScales,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackedPiece {
    pub operand: String,
    pub plan: PackingPlan,
    pub section: Section,
    pub decode_order: DecodeOrder,
}

pub struct Template {
    piece: PackedPiece,
    lookup: bool,
    binding: model_qwen38_contract::projections::ports::SourceBinding,
}

pub struct PlacedPiece {
    pub packing: PackedPiece,
    pub slices: Vec<(usize, Range<u64>)>,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

pub fn templates(
    catalog: &Catalog,
    precision: &PrecisionDocument,
) -> Result<BTreeMap<String, Template>> {
    precision.validate()?;
    let source = catalog
        .precision
        .as_ref()
        .ok_or("packed precision requires imported source operands")?;
    let operations: BTreeMap<_, _> = precision
        .operations
        .iter()
        .map(|s| (s.source.id.as_str(), s))
        .collect();
    let definitions = model_qwen38_contract::projections::ports::get(
        model_qwen38_contract::contract::Shape::new(
            catalog.geometry,
            crate::placement::bundle_ranks(catalog.ranks)?,
        ),
    )?;
    let bindings = definitions.bindings()?;
    let mut result = BTreeMap::new();
    for (name, original) in &source.operands {
        let Some(binding) = bindings.get(name) else {
            continue;
        };
        let Some(selection) = operations.get(binding.operation.as_str()) else {
            continue;
        };
        let effective = selection
            .effective
            .weights
            .get(binding.operand)
            .ok_or("missing prepared fusion operand")?;
        let geometry = selection.effective.matrix_geometry()?;
        let local_scale_dtype = original.weight.local_scale.as_ref().map(|s| {
            if matches!(
                effective.format,
                WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
            ) {
                DType::F32
            } else {
                s.tensor.dtype
            }
        });
        let piece = PackedPiece {
            operand: name.clone(),
            plan: PackingPlan {
                layout: effective.layout.clone(),
                format: effective.format,
                experts: 1,
                projections: 1,
                rows: u32::try_from(geometry.rows)?,
                k: u32::try_from(geometry.k)?,
                columns: SwigluColumns::ConcatenatedHalves,
                local_scale_dtype,
            },
            section: Section::Codes,
            decode_order: selection.effective.arithmetic.decode_order,
        };
        let lookup = selection.effective.kind == kernel_core::precision::OperationKind::Lookup;
        result.insert(
            name.clone(),
            Template {
                piece: piece.clone(),
                lookup,
                binding: binding.clone(),
            },
        );
        if let Some(scale) = &original.weight.local_scale
            && !matches!(scale.scope, ScaleScope::Tensor | ScaleScope::Table)
            && result
                .insert(
                    scale.tensor.tensor.clone(),
                    Template {
                        piece: PackedPiece {
                            section: Section::LocalScales,
                            ..piece
                        },
                        lookup,
                        binding: binding.clone(),
                    },
                )
                .is_some()
        {
            return Err(format!(
                "tensor {}: shared block scales require a supported common placement",
                scale.tensor.tensor
            )
            .into());
        }
    }
    Ok(result)
}

impl Template {
    pub fn place(&self, catalog: &Catalog, rank: usize) -> Result<Option<PlacedPiece>> {
        let original = &catalog
            .precision
            .as_ref()
            .ok_or("missing imported precision")?
            .operands[&self.piece.operand];
        let definitions = model_qwen38_contract::projections::ports::get(
            model_qwen38_contract::contract::Shape::new(
                catalog.geometry,
                crate::placement::bundle_ranks(catalog.ranks)?,
            ),
        )?;
        let projection = definitions.projection(&self.binding.projection)?;
        let selected = projection_spec::derive::rank_selection(
            projection,
            &definitions.geometry,
            projection_spec::derive::Topology {
                ranks: catalog.ranks as u32,
                rank: rank as u32,
            },
            self.binding.operand as u32,
        )?;
        if self
            .binding
            .expert
            .is_some_and(|expert| !selected.experts.contains(&expert))
        {
            return Ok(None);
        }
        let slices = if let Some(shard) = &self.binding.shard_rows {
            selected
                .rows
                .into_iter()
                .filter_map(|rows| {
                    let start = rows.start.max(shard.start);
                    let end = rows.end.min(shard.end);
                    (start < end).then(|| (0, start - shard.start..end - shard.start))
                })
                .fold(
                    Vec::<(usize, Range<u64>)>::new(),
                    |mut ranges, (axis, range)| {
                        if let Some((last_axis, last)) = ranges.last_mut()
                            && *last_axis == axis
                            && last.end == range.start
                        {
                            last.end = range.end;
                        } else {
                            ranges.push((axis, range));
                        }
                        ranges
                    },
                )
        } else if selected.k != (0..original.weight.logical_shape[1]) {
            vec![(1, selected.k)]
        } else {
            selected.rows.into_iter().map(|r| (0, r)).collect()
        };
        if slices.is_empty() {
            return Ok(None);
        }
        let mut packing = self.piece.clone();
        if self.lookup {
            packing.plan.rows =
                u32::try_from(slices.iter().map(|(_, r)| r.end - r.start).sum::<u64>())?;
        }
        packing.plan.validate()?;
        let sizes = packing.plan.section_bytes()?;
        let (bytes, shape) = match packing.section {
            Section::Codes => (
                sizes.codes,
                vec![u64::from(packing.plan.rows), u64::from(packing.plan.k)],
            ),
            Section::LocalScales => (
                sizes.local_scales,
                vec![
                    sizes.local_scales
                        / (kernel_core::precision::dtype_bits(
                            packing
                                .plan
                                .local_scale_dtype
                                .ok_or("missing local scale dtype")?,
                        ) / 8),
                ],
            ),
        };
        Ok(Some(PlacedPiece {
            packing,
            slices,
            shape,
            bytes,
        }))
    }
}

impl PackedPiece {
    /// The view's safetensors dtype and layout as the bundle's prepared view
    /// contract names them (`prepared::code_layout_for` and
    /// `prepared::local_layout`), since the bundle validates each planned
    /// rank index against that contract.
    pub fn view_for(
        &self,
        shape: model_qwen38_contract::contract::Shape,
    ) -> Result<(&'static str, &'static str)> {
        use model_qwen38_contract::prepared;
        Ok(match self.section {
            Section::Codes => {
                prepared::code_layout_for(shape, self.plan.format, &self.plan.layout.id)?
            }
            Section::LocalScales => prepared::local_layout(
                self.plan.format,
                &self.plan.layout.id,
                self.plan
                    .local_scale_dtype
                    .ok_or("missing local scale dtype")?,
            )?,
        })
    }
}

impl PackedPiece {
    /// The portable record encoded by the direct loader for one placed piece.
    pub fn record(
        &self,
        weight: &kernel_core::precision::WeightSpec,
        slices: &[(usize, Range<u64>)],
    ) -> Result<kernel_core::weight_transform::WeightRecord> {
        use kernel_core::precision::format::registry;
        use kernel_core::weight_transform::PackedSection;
        use model_compiler_sdk::quantization::format;
        let [rows, k] = weight.logical_shape[..] else {
            return Err("packed pieces are matrices".into());
        };
        let axis = slices.first().ok_or("packed piece has no slices")?.0;
        if axis > 1 || slices.iter().any(|(a, _)| *a != axis) || axis == 1 && slices.len() != 1 {
            return Err("Qwen packed piece requires row ranges or one K range".into());
        }
        let (rows_kept, k_kept) = if axis == 0 {
            (slices.iter().map(|(_, r)| r.clone()).collect(), 0..k)
        } else {
            (std::iter::once(0..rows).collect(), slices[0].1.clone())
        };
        let plugin = registry::by_weight(weight)?;
        let matrix = format::matrix_source(plugin, weight, rows_kept, k_kept)?;
        let section = match self.section {
            Section::Codes => PackedSection::Codes,
            Section::LocalScales => PackedSection::LocalScales,
        };
        format::packed_records(plugin, self.plan.clone(), self.decode_order,
            vec![matrix], format::sources(weight))?
            .into_iter().find(|record| matches!(&record.transform,
                kernel_core::weight_transform::WeightTransform::PackedOperand(p) if p.section == section))
            .ok_or_else(|| "missing packed projection section".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{Placement, ScaleLayout, TensorRecord};
    use kernel_core::precision::{CodeEncoding, TensorSpan, WeightLayout, WeightSpec};
    use model_compiler_sdk::quantization::{
        import::{ExportMetadata, ImportedOperand},
        source::TensorInventory,
    };

    /// Each rank's n-gram shard rows, FP8 or NVFP4 at TP2 and TP4, are the
    /// loader's contiguous rank rows: every table row is placed once, on
    /// the rank whose heads read it.
    #[test]
    fn lookup_head_boundaries_preserve_contiguous_loader_records() {
        use kernel_core::precision::{ScaleConvention, ScaleSpec};
        for ranks in [2, 4] {
            for format in [WeightFormat::Fp8PerTensor, WeightFormat::Nvfp4] {
                lookup_shard_rows(ranks, format);
            }
        }
        fn lookup_shard_rows(ranks: usize, format: WeightFormat) {
            let mut catalog = crate::catalog::synthetic(ranks);
            let shape = crate::placement::bundle_shape_flash(ranks).unwrap();
            let ports = model_qwen38_contract::projections::ports::get(shape).unwrap();
            let bindings = ports.bindings().unwrap();
            let nvfp4 = format == WeightFormat::Nvfp4;
            let mut operands = BTreeMap::new();
            for (name, binding) in bindings.iter().filter(|(_, b)| b.shard_rows.is_some()) {
                let tensor = &catalog.tensors[name];
                let [rows, k] = tensor.shape[..] else {
                    panic!("{name} is not a table shard")
                };
                let codes = TensorSpan {
                    tensor: name.clone(),
                    dtype: if nvfp4 { DType::U8 } else { DType::Fp8E4M3 },
                    shape: vec![rows, if nvfp4 { k / 2 } else { k }],
                    offset: 0,
                    byte_length: rows * if nvfp4 { k / 2 } else { k },
                };
                let local_scale = if nvfp4 {
                    ScaleSpec {
                        tensor: TensorSpan {
                            tensor: format!("{name}_scale"),
                            dtype: DType::Fp8E4M3,
                            shape: vec![rows, k / 16],
                            offset: 0,
                            byte_length: rows * k / 16,
                        },
                        scope: ScaleScope::Block,
                        convention: ScaleConvention::Multiply,
                        groups: vec![],
                    }
                } else {
                    ScaleSpec {
                        tensor: TensorSpan {
                            tensor: format!("{}.scale", binding.operation),
                            dtype: DType::Bf16,
                            shape: vec![1],
                            offset: 0,
                            byte_length: 2,
                        },
                        scope: ScaleScope::Tensor,
                        convention: ScaleConvention::Multiply,
                        groups: vec![],
                    }
                };
                operands.insert(
                    name.clone(),
                    ImportedOperand {
                        weight: WeightSpec {
                            codes,
                            logical_shape: vec![rows, k],
                            format,
                            encoding: if nvfp4 {
                                CodeEncoding::E2M1(kernel_core::precision::NibbleOrder::EvenLow)
                            } else {
                                CodeEncoding::E4M3
                            },
                            layout: WeightLayout::row_major(),
                            global_scale: None,
                            transforms: vec![],
                            local_scale: Some(local_scale),
                        },
                        activation: None,
                        source_tensors: vec![name.clone()],
                    },
                );
            }
            catalog.precision = Some(crate::precision_source::SourcePrecision {
                inventory: TensorInventory {
                    tensors: BTreeMap::new(),
                    spans: BTreeMap::new(),
                },
                metadata: ExportMetadata::default(),
                operands,
            });
            let mut comparisons = 0;
            let mut placed_rows = vec![0_u64; ranks];
            for (name, binding) in bindings.iter().filter(|(_, b)| b.shard_rows.is_some()) {
                let template = Template {
                    piece: PackedPiece {
                        operand: name.clone(),
                        plan: PackingPlan {
                            layout: WeightLayout::row_major(),
                            format,
                            experts: 1,
                            projections: 1,
                            rows: 1,
                            k: shape.ngram_head_width(),
                            columns: SwigluColumns::ConcatenatedHalves,
                            local_scale_dtype: Some(if nvfp4 {
                                DType::Fp8E4M3
                            } else {
                                DType::Bf16
                            }),
                        },
                        section: Section::Codes,
                        decode_order: if nvfp4 {
                            DecodeOrder::ExactWeightToBf16
                        } else {
                            DecodeOrder::CodeThenScale
                        },
                    },
                    lookup: true,
                    binding: binding.clone(),
                };
                for (rank, rows) in placed_rows.iter_mut().enumerate() {
                    let expected =
                        catalog.tensors[name].rank_slices_for(catalog.geometry, rank, ranks);
                    let placed = template.place(&catalog, rank).unwrap();
                    let slices = placed.as_ref().map(|p| p.slices.as_slice()).unwrap_or(&[]);
                    assert_eq!(slices, expected.as_slice(), "{format:?} {name} rank {rank}");
                    *rows += slices.iter().map(|(_, r)| r.end - r.start).sum::<u64>();
                    comparisons += 1;
                }
            }
            assert_eq!(comparisons, 128 * ranks);
            for (rank, rows) in placed_rows.into_iter().enumerate() {
                assert_eq!(
                    rows,
                    model_qwen38_contract::views::ngram_rows(rank as u32, shape)
                        .unwrap()
                        .1,
                    "{format:?} TP{ranks} rank {rank}"
                );
            }
        }
    }

    #[test]
    fn imported_linear_qkv_uses_the_catalog_geometry() {
        let mut geometry = model_qwen38_contract::geometry::FLASH;
        geometry.id = "packing-test";
        geometry.shape.linear.key_heads /= 2;
        geometry.shape.linear.value_heads /= 2;
        let geometry = Box::leak(Box::new(geometry));
        for ranks in [2, 4] {
            let name = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight";
            let ranges = crate::catalog::linear_qkv_ranges(geometry);
            let logical_shape = vec![ranges[2].end, u64::from(geometry.hidden())];
            let codes = TensorSpan {
                tensor: name.into(),
                dtype: DType::Bf16,
                shape: logical_shape.clone(),
                offset: 0,
                byte_length: logical_shape.iter().product::<u64>() * 2,
            };
            let weight = WeightSpec {
                codes: codes.clone(),
                logical_shape: logical_shape.clone(),
                format: WeightFormat::Bf16,
                encoding: CodeEncoding::Bf16LittleEndian,
                layout: WeightLayout::row_major(),
                local_scale: None,
                global_scale: None,
                transforms: Vec::new(),
            };
            let source = crate::precision_source::SourcePrecision {
                inventory: TensorInventory {
                    tensors: BTreeMap::new(),
                    spans: BTreeMap::new(),
                },
                metadata: ExportMetadata::default(),
                operands: BTreeMap::from([(
                    name.into(),
                    ImportedOperand {
                        weight,
                        activation: None,
                        source_tensors: vec![name.into()],
                    },
                )]),
            };
            let mut catalog = crate::catalog::synthetic(ranks);
            catalog.geometry = geometry;
            catalog.tensors = BTreeMap::from([(
                name.into(),
                TensorRecord {
                    name: name.into(),
                    dtype: runtime_weights::DType::Bf16,
                    shape: logical_shape.clone(),
                    bytes: codes.byte_length,
                    file: String::new(),
                    file_start: 0,
                    placement: Placement::LinearQkv,
                    scale: ScaleLayout::None,
                },
            )]);
            catalog.precision = Some(source);
            let template = Template {
                piece: PackedPiece {
                    operand: name.into(),
                    plan: PackingPlan {
                        layout: WeightLayout::row_major(),
                        format: WeightFormat::Bf16,
                        experts: 1,
                        projections: 1,
                        rows: (logical_shape[0] / ranks as u64) as u32,
                        k: geometry.hidden(),
                        columns: SwigluColumns::ConcatenatedHalves,
                        local_scale_dtype: None,
                    },
                    section: Section::Codes,
                    decode_order: DecodeOrder::ScaleThenCode,
                },
                lookup: false,
                binding: model_qwen38_contract::projections::ports::get(
                    model_qwen38_contract::contract::Shape::new(
                        geometry,
                        crate::placement::bundle_ranks(ranks).unwrap(),
                    ),
                )
                .unwrap()
                .bindings()
                .unwrap()[name]
                    .clone(),
            };
            for rank in 0..ranks {
                let placed = template.place(&catalog, rank).unwrap().unwrap();
                assert_eq!(
                    placed.slices,
                    catalog.tensors[name].rank_slices_for(geometry, rank, ranks)
                );
                assert!(
                    placed
                        .slices
                        .iter()
                        .all(|(_, range)| range.end <= logical_shape[0])
                );
                assert_eq!(
                    placed
                        .slices
                        .iter()
                        .map(|(_, range)| range.end - range.start)
                        .sum::<u64>(),
                    logical_shape[0] / ranks as u64
                );
            }
        }
    }
}
