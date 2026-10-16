//! MiMo V2.6 Flash's tensor roles for the shared precision importers, and
//! the compiled projections that serve them.
//!
//! The family names each role, its logical geometry and the projections
//! that serve it; the shared importer
//! (`model_compiler_sdk::quantization::import`) reads the checkpoint's export
//! metadata and decides every weight's encoding, its companion tensors and
//! their dtypes. Each weight is then admitted through the capability offers
//! of the compiled kernel entries that serve its role
//! (`model_mimo26_contract::capabilities`) at every supported rank count:
//! the GEMM entries of the fused QKV, attention `o_proj`, routers, `lm_head`
//! and dense MLP, the grouped MXFP4 entries of the routed experts and the
//! row gather of the embedding; the MTP layers run the target's entries. A
//! refusal names what the export made the weight and why no offer binds it.
//! The released checkpoints store BF16 weights for the embedding, `lm_head`,
//! `o_proj`, routers and `eh_proj`, FP8 E4M3 128x128 blocks with F32
//! `weight_scale_inv` scales for `qkv_proj` and the dense MLP, and MXFP4
//! experts (E2M1 pairs, low nibble first, one UE8M0 `weight_scale` byte per
//! 32 K values); the packers read those companions by name.
//!
//! Every projection runs BF16 activation operands, whatever activation
//! recipe the export declares: MiMo reuses the stored weights and does not
//! quantize activations.
//!
//! The fused `qkv_proj` holds four independently quantized row chunks (the
//! checkpoint's TP4 rank-local Q, K and V); its FP8 block scales restart at
//! each chunk. It is imported as a bank of four parts so the importer checks
//! that scale grid.
//!
//! The export metadata describes the language model's modules. The router
//! weight is not a converted linear module and the output embedding is
//! never converted, so they are declared BF16; the MTP layers lie outside
//! the HF model the metadata names, and their unconverted `eh_proj` and
//! `o_proj` are declared BF16 as well. Any other tensor of the checkpoint
//! must be one of these roles or a companion the importer read.

#[cfg(test)]
use model_mimo26_contract::geometry::V26_FLASH as GEOMETRY;

use std::collections::{BTreeMap, BTreeSet};

use kernel_core::contract::{ActivationFormat, DType as CodeDType, WeightFormat};
use kernel_core::precision::capability::CapabilityInventory;
use kernel_core::precision::execution::{Binding, ImplementationProvider};
use kernel_core::precision::{
    ArithmeticSpec, DecodeOrder, GroupAxis, OperationKind, OperationSpec, OperationUse, Rounding,
    TensorSpan, WeightSpec,
};
use kernel_core::transform::ChunkedScaleGrid;
use model_compiler_sdk::quantization::bank::concatenate_experts;
use model_compiler_sdk::quantization::import::{
    EncodingRecipe, ExportMetadata, ImportedOperand, OperandRole, ROUTED_EXPERT_ROLE,
    import_operand,
};
use model_compiler_sdk::quantization::source::source_dtype;
use model_compiler_sdk::quantization::{ActivationPolicy, resolve_operation};
use model_mimo26_contract::backend::{Kernels, Selected};
use model_mimo26_contract::capabilities::{self, Projection};
use model_mimo26_contract::tp::QKV_SCALE_GROUP_ROWS;
use runtime_weights::DType;

use crate::MimoResult;
use crate::catalog::TensorRecord;

pub use model_compiler_sdk::quantization::import::HF_QUANT_CONFIG;

#[cfg(test)]
const HIDDEN: u64 = model_mimo26_contract::tp::HIDDEN;
/// Checkpoint data: row chunks of fused `qkv_proj`, each quantized on its own.
const QKV_PARTS: u64 = model_mimo26_contract::tp::QKV_CHUNKS;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Role {
    /// A tensor no export rule converts: a norm, bias or sink, with the
    /// dtype and shape MiMo reads.
    Side { dtype: DType, shape: Vec<u64> },
    /// A weight the importer interprets.
    Weight {
        /// Logical `[rows, k]`.
        shape: [u64; 2],
        /// Row chunks quantized on their own (1 for a plain matrix).
        parts: u64,
        /// A token lookup table.
        lookup: bool,
        /// A routed expert projection.
        routed: bool,
        /// A role the export metadata does not describe, declared BF16.
        declared_bf16: bool,
        /// The compiled projections that serve the role, which admission
        /// binds through their capability offers
        /// (`model_mimo26_contract::capabilities`).
        offered: &'static [Projection],
    },
}

#[cfg(test)]
fn expert(rows: u64, k: u64, offered: &'static [Projection]) -> Role {
    Role::Weight {
        shape: [rows, k],
        parts: 1,
        lookup: false,
        routed: true,
        declared_bf16: false,
        offered,
    }
}

/// The block-scale grid of a fused `qkv_proj` whose scale tensor has
/// `scale_rows` rows of `k_blocks` entries: the checkpoint's four independent
/// Q|K|V chunks each restart the 128-row grid (27 scale rows of 3392
/// global-attention rows, 29 of 3712 sliding-window or MTP rows).
pub fn qkv_scale_grid(
    geometry: &model_mimo26_contract::geometry::Geometry,
    global: bool,
    scale_rows: u64,
    k_blocks: usize,
) -> MimoResult<ChunkedScaleGrid> {
    let rows_per_chunk = qkv_rows(geometry, global) / QKV_PARTS;
    let grid = ChunkedScaleGrid {
        chunks: QKV_PARTS as usize,
        rows_per_chunk: rows_per_chunk as usize,
        k_blocks,
        block_rows: 128,
    };
    if grid.total_scale_rows() as u64 != scale_rows {
        return Err("QKV scale source does not match its chunked grid".into());
    }
    Ok(grid)
}

/// The source scale row of each scale row of `rank`'s fused QKV at `tp`.
/// TP4's rank is one source chunk and holds that chunk's scale rows. Other
/// rank counts regroup their Q, K and V heads from several chunks (`runs`,
/// in source rows) and hold one scale row per 64 rank rows
/// (`QKV_SCALE_GROUP_ROWS`); each such run must lie inside one source block,
/// whose scale it repeats exactly.
pub fn qkv_rank_scale_rows(
    grid: ChunkedScaleGrid,
    runs: &[std::ops::Range<u64>],
    rank: usize,
    tp: model_mimo26_contract::Tp,
) -> MimoResult<Vec<usize>> {
    if tp.ranks() == 4 {
        let per_chunk = grid.scale_rows_per_chunk();
        let first = rank * per_chunk;
        return Ok((first..first + per_chunk).collect());
    }
    let group = QKV_SCALE_GROUP_ROWS as usize;
    let mut rows = Vec::new();
    for run in runs {
        let run = run.start as usize..run.end as usize;
        if !run.start.is_multiple_of(group) || !run.len().is_multiple_of(group) {
            return Err("regrouped QKV run is not a whole number of scale groups".into());
        }
        for first in run.step_by(group) {
            let row = grid.scale_row(first)?;
            if grid.scale_row(first + group - 1)? != row {
                return Err("regrouped QKV scale group straddles a source block".into());
            }
            rows.push(row);
        }
    }
    Ok(rows)
}

/// Whether the fused QKV `name` is a global-attention layer's: MTP layers
/// run sliding-window attention.
pub(crate) fn qkv_is_global(
    geometry: &model_mimo26_contract::geometry::Geometry,
    name: &str,
) -> MimoResult<bool> {
    if name.starts_with("model.mtp.layers.") {
        return Ok(false);
    }
    let layer = name
        .strip_prefix("model.layers.")
        .and_then(|tail| tail.split('.').next())
        .and_then(|layer| layer.parse::<u32>().ok())
        .filter(|&layer| layer < geometry.layers())
        .ok_or_else(|| format!("no QKV layer geometry for {name}"))?;
    Ok(geometry.is_global(layer))
}

/// Global-attention layers' fused QKV rows (one KV head per chunk), and
/// sliding-window and MTP layers' (two).
fn qkv_rows(geometry: &model_mimo26_contract::geometry::Geometry, global: bool) -> u64 {
    let (heads, kv, qk, value) = if global {
        (
            geometry.attention_q_heads(),
            geometry.attention_kv_heads(),
            geometry.head_dim(),
            geometry.value_head_dim(),
        )
    } else {
        (
            geometry.swa_q_heads(),
            geometry.swa_kv_heads(),
            geometry.swa_head_dim(),
            geometry.swa_value_head_dim(),
        )
    };
    heads as u64 * qk as u64 + kv as u64 * (qk + value) as u64
}

/// Expand the served storage formats of the model's generated tensor roles.
pub fn expected_source_tensors(
    roles: &BTreeMap<String, Role>,
) -> MimoResult<BTreeMap<String, (DType, Vec<u64>)>> {
    let mut tensors = BTreeMap::new();
    for (name, role) in roles {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        match role {
            Role::Side { dtype, shape } => {
                tensors.insert(name.clone(), (*dtype, shape.clone()));
            }
            Role::Weight {
                shape: [rows, k],
                parts,
                offered,
                ..
            } => {
                let format = offered
                    .first()
                    .map(|projection| projection.weight(model_mimo26_contract::Tp::TP4));
                let (dtype, shape, scale) = match format {
                    Some(WeightFormat::Fp8Block128x128(_)) => (
                        DType::F8E4M3,
                        vec![*rows, *k],
                        Some((
                            "weight_scale_inv",
                            DType::F32,
                            vec![parts * (rows / parts).div_ceil(128), k.div_ceil(128)],
                        )),
                    ),
                    Some(WeightFormat::Mxfp4) => (
                        DType::U8,
                        vec![*rows, k / 2],
                        Some(("weight_scale", DType::U8, vec![*rows, k / 32])),
                    ),
                    Some(WeightFormat::Bf16) | None => (DType::Bf16, vec![*rows, *k], None),
                    other => {
                        return Err(format!("unsupported served fixture format {other:?}").into());
                    }
                };
                tensors.insert(name.clone(), (dtype, shape));
                if let Some((suffix, dtype, shape)) = scale {
                    tensors.insert(format!("{stem}.{suffix}"), (dtype, shape));
                }
            }
        }
    }
    Ok(tensors)
}

/// The text model's roles; `global` names the global-attention layers.
pub fn text_roles(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    global: impl Fn(usize) -> bool,
) -> BTreeMap<String, Role> {
    // Admission has already resolved the listed geometry and attention pattern.
    assert!(
        (0..geometry.layers()).all(|layer| global(layer as usize) == geometry.is_global(layer))
    );
    declared_roles(geometry, Scope::Text)
}

/// Materialize MTP from the same declaration as the main decoder.
pub fn mtp_roles(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> BTreeMap<String, Role> {
    declared_roles(geometry, Scope::Mtp)
}

/// The DFlash drafter's roles: its own file's BF16 weights and sides.
pub fn dflash_roles(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> BTreeMap<String, Role> {
    declared_roles(geometry, Scope::Dflash)
}

/// The weights a checkpoint file holds for one image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Scope {
    Text,
    Mtp,
    Dflash,
}

impl Scope {
    /// Whether this scope's file holds the checkpoint tensor `name`.
    pub fn holds(self, name: &str) -> bool {
        match self {
            Self::Text => {
                name.starts_with("model.layers.")
                    || matches!(
                        name,
                        "model.embed_tokens.weight" | "model.norm.weight" | "lm_head.weight"
                    )
            }
            Self::Mtp => name.starts_with("model.mtp.layers."),
            Self::Dflash => !name.starts_with("model.") && name != "lm_head.weight",
        }
    }
}

fn declared_roles(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    scope: Scope,
) -> BTreeMap<String, Role> {
    use model_mimo26_contract::projections::sources::{Source, sources};
    use projection_spec::ProjectionKind;
    let sources = sources(geometry).expect("validated projection sources");
    let dimensions = sources.geometry;
    sources
        .names
        .iter()
        .filter_map(|(name, source)| {
            if !scope.holds(name) {
                return None;
            }
            let role = match source {
                Source::Side(index) => {
                    let side = &sources.declaration.side_tensors[*index];
                    Role::Side {
                        dtype: match side.admitted_dtypes() {
                            [CodeDType::Bf16] => DType::Bf16,
                            [CodeDType::F32] => DType::F32,
                            _ => panic!("unsupported side dtype {}", side.id),
                        },
                        shape: side
                            .shape
                            .iter()
                            .map(|d| d.resolve(dimensions).expect("declared side shape"))
                            .collect(),
                    }
                }
                Source::Projection(index) => {
                    let projection = &sources.declaration.projections[*index];
                    // DFlash's projections run its own entries, outside
                    // the offers admission binds.
                    let offered =
                        model_mimo26_contract::capabilities::served(geometry, &projection.id)
                            .or((scope == Scope::Dflash).then_some(&[][..]))
                            .unwrap_or_else(|| {
                                panic!("unsupported served projection {}", projection.id)
                            });
                    Role::Weight {
                        shape: projection
                            .shape
                            .each_ref()
                            .map(|d| d.resolve(dimensions).expect("declared matrix shape")),
                        parts: projection.parts.len().max(1) as u64,
                        lookup: matches!(projection.kind, ProjectionKind::LookupTable {}),
                        routed: matches!(projection.kind, ProjectionKind::ExpertBank { .. }),
                        declared_bf16: matches!(
                            projection.id.as_str(),
                            "embedding" | "head" | "router" | "mtp.output" | "mtp.eh"
                        ),
                        offered,
                    }
                }
            };
            Some((name.clone(), role))
        })
        .collect()
}

/// The export metadata of a checkpoint: `config.json`'s
/// `quantization_config` (absent: every weight BF16) and the optional
/// `hf_quant_config.json`.
pub fn export_metadata(
    quantization_config: Option<&serde_json::Value>,
    hf_quant_config: Option<&[u8]>,
) -> MimoResult<ExportMetadata> {
    Ok(ExportMetadata::from_documents(
        quantization_config,
        hf_quant_config,
    )?)
}

/// [`export_metadata`] from `config.json`'s bytes.
pub fn export_metadata_from_config(
    config: &[u8],
    hf_quant_config: Option<&[u8]>,
) -> MimoResult<ExportMetadata> {
    let config: serde_json::Value = serde_json::from_slice(config)?;
    export_metadata(
        config
            .get("quantization_config")
            .filter(|value| !value.is_null()),
        hf_quant_config,
    )
}

fn code_dtype(tensor: &TensorRecord) -> MimoResult<CodeDType> {
    source_dtype(tensor.dtype).ok_or_else(|| {
        format!(
            "tensor {}: unsupported source dtype {}",
            tensor.name,
            tensor.dtype.safetensors_name()
        )
        .into()
    })
}

fn span(tensor: &TensorRecord) -> MimoResult<TensorSpan> {
    Ok(TensorSpan {
        tensor: tensor.name.clone(),
        dtype: code_dtype(tensor)?,
        shape: tensor.shape.clone(),
        offset: 0,
        byte_length: tensor.bytes,
    })
}

fn format_name(format: WeightFormat) -> String {
    match format {
        WeightFormat::Bf16 => "BF16".into(),
        WeightFormat::Fp8PerTensor => "per-tensor FP8".into(),
        WeightFormat::Fp8PerChannel => "per-channel FP8".into(),
        WeightFormat::Fp8Block128x128(_) => "FP8 128x128 block".into(),
        WeightFormat::Mxfp8 => "MXFP8".into(),
        WeightFormat::Mxfp4 => "MXFP4".into(),
        WeightFormat::Nvfp4 => "NVFP4".into(),
        other => format!("{other:?}"),
    }
}

/// Checks one imported weight: it binds through the capability offers of
/// the compiled projections that serve its role (once a projection's last
/// source weight is imported; `imported` holds the earlier ones), carries no
/// input transform, and has the companions MiMo's packers stream by name.
fn check_weight(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    name: &str,
    role: &Role,
    operand: &ImportedOperand,
    tensors: &BTreeMap<String, TensorRecord>,
    imported: &BTreeMap<String, ImportedOperand>,
) -> MimoResult<()> {
    let Role::Weight {
        shape: [rows, k],
        parts,
        offered,
        ..
    } = *role
    else {
        unreachable!("only weights are imported");
    };
    let stem = name.strip_suffix(".weight").expect("weight roles");
    let weight = &operand.weight;
    let found = |detail: String| -> MimoResult<()> {
        Err(format!(
            "tensor {name}: the checkpoint's export metadata makes it {} weights{detail}",
            format_name(weight.format)
        )
        .into())
    };
    if !weight.transforms.is_empty() {
        return found(" with required input transforms, which MiMo's packers do not apply".into());
    }
    if offered.is_empty() {
        return Err(format!("tensor {name}: no compiled MiMo projection serves its role").into());
    }
    for &projection in offered {
        if let Err(error) = bind_offered(geometry, name, projection, operand, imported, tensors) {
            return found(format!("; no compiled MiMo projection serves it: {error}"));
        }
    }
    // The packers stream the scales of the encodings the offers admit by
    // name; the offers refuse any other encoding.
    let scale = |suffix: &str, dtype: DType, shape: [u64; 2]| -> MimoResult<()> {
        let local = weight
            .local_scale
            .as_ref()
            .expect("validated scaled encoding");
        let expected = format!("{stem}.{suffix}");
        let record = tensors.get(&local.tensor.tensor).expect("imported tensor");
        if local.tensor.tensor != expected || record.dtype != dtype || record.shape != shape {
            return found(format!(
                " with {} scales {} of shape {:?}; MiMo's packers read {} {expected} of \
                 shape {shape:?}",
                record.dtype.safetensors_name(),
                local.tensor.tensor,
                record.shape,
                dtype.safetensors_name()
            ));
        }
        Ok(())
    };
    match weight.format {
        WeightFormat::Mxfp4 => scale("weight_scale", DType::U8, [rows, k / 32]),
        WeightFormat::Fp8Block128x128(_) => scale(
            "weight_scale_inv",
            DType::F32,
            [parts * (rows / parts).div_ceil(128), k.div_ceil(128)],
        ),
        _ => Ok(()),
    }
}

/// Binds `projection`, of which `operand` (the role `name`) is one source
/// weight, at every supported rank count through its capability offers,
/// with BF16 activation operands (MiMo never quantizes activations). The
/// projection's other weights (a fused gate's or up's, every routed expert
/// of the layer) are in `imported`; its binding waits until all are.
fn bind_offered(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    name: &str,
    projection: Projection,
    operand: &ImportedOperand,
    imported: &BTreeMap<String, ImportedOperand>,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<()> {
    let stem = name.strip_suffix(".weight").expect("weight roles");
    let suffixes = projection.operands();
    let not_operand = || format!("tensor {name} is no operand of {projection:?}");
    let (module, _) = match suffixes {
        [""] => (stem, ""),
        _ => stem
            .rsplit_once('.')
            .filter(|(_, suffix)| suffixes.contains(suffix))
            .ok_or_else(not_operand)?,
    };
    // Each operand's source weights: one, or one per routed expert.
    let (module, experts) = if projection.routed() {
        let (layer, expert) = module.rsplit_once('.').ok_or_else(not_operand)?;
        expert.parse::<usize>().map_err(|_| not_operand())?;
        (layer, Some(geometry.experts() as usize))
    } else {
        (module, None)
    };
    let members: Vec<Option<usize>> = match experts {
        Some(experts) => (0..experts).map(Some).collect(),
        None => vec![None],
    };
    let source = |suffix: &str, expert: Option<usize>| match (expert, suffix) {
        (Some(expert), suffix) => format!("{module}.{expert}.{suffix}.weight"),
        (None, "") => name.to_owned(),
        (None, suffix) => format!("{module}.{suffix}.weight"),
    };
    let get = |other: &String| {
        if other == name {
            Some(operand)
        } else {
            imported.get(other)
        }
    };
    // Stop at the first source not yet imported: every expert of a bank
    // but the last stops early.
    if suffixes
        .iter()
        .any(|suffix| members.iter().any(|&e| get(&source(suffix, e)).is_none()))
    {
        return Ok(());
    }
    let names: Vec<Vec<String>> = suffixes
        .iter()
        .map(|suffix| members.iter().map(|&e| source(suffix, e)).collect())
        .collect();
    // One imported operand per projection operand: a routed bank's
    // concatenates every expert's, which must agree in encoding.
    let mut banks = Vec::new();
    for (suffix, names) in suffixes.iter().zip(&names) {
        let sources: Vec<ImportedOperand> = names
            .iter()
            .map(|other| get(other).expect("imported").clone())
            .collect();
        banks.push(match experts {
            Some(_) => concatenate_experts(&format!("{module}.{suffix}"), &sources)?.operand,
            None => sources.into_iter().next().expect("one source"),
        });
    }
    // A lookup reads token IDs, a projection activation operands.
    let activations = if projection.lookup() {
        Vec::new()
    } else {
        banks
            .iter()
            .map(|operand| {
                operand
                    .activation
                    .clone()
                    .ok_or("a projection reads activation operands")
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let inventory = capabilities::inventory_for(geometry)?;
    for ranks in model_mimo26_contract::Tp::SUPPORTED {
        let tp = model_mimo26_contract::Tp::of_geometry(geometry, ranks.ranks())?;
        let weights = match projection {
            Projection::Qkv { .. } => vec![qkv_rank_weight(
                operand,
                tensors,
                tp,
                projection.geometry(tp).rows,
            )?],
            _ => banks.iter().map(|o| o.weight.clone()).collect(),
        };
        let operation = offered_operation(
            if suffixes.len() > 1 {
                format!("{module}.{}", suffixes.join("+"))
            } else if experts.is_some() {
                format!("{module}.{}", suffixes[0])
            } else {
                stem.to_owned()
            },
            projection,
            tp,
            weights,
            activations.clone(),
            experts,
        );
        resolve_operation(
            &operation,
            ActivationPolicy::Bf16,
            &Offers {
                inventory,
                implementation: projection.implementation(tp),
            },
        )
        .map_err(|error| format!("TP{}: {error}", tp.ranks()))?;
    }
    Ok(())
}

/// The source operation checked by checkpoint admission, before offer binding.
pub(crate) fn offered_operation(
    id: String,
    projection: Projection,
    tp: model_mimo26_contract::Tp,
    weights: Vec<WeightSpec>,
    activations: Vec<kernel_core::precision::ActivationSpec>,
    experts: Option<usize>,
) -> OperationSpec {
    let activation = activations
        .first()
        .map_or(ActivationFormat::Bf16, |a| a.format);
    let weight = weights[0].format;
    OperationSpec {
        id,
        kind: match experts {
            Some(experts) => OperationKind::Grouped {
                experts: experts as u32,
                top_k: tp.geometry.top_k(),
            },
            None if projection.lookup() => OperationKind::Lookup,
            None => OperationKind::Dense,
        },
        target: Selected::TARGET,
        weights,
        geometry: Some(projection.geometry(tp)),
        activations,
        arithmetic: ArithmeticSpec {
            weight_operand: match activation {
                ActivationFormat::Bf16 => CodeDType::Bf16,
                _ if matches!(weight, WeightFormat::Nvfp4 | WeightFormat::Mxfp4) => {
                    CodeDType::Fp4E2M1
                }
                _ => CodeDType::Fp8E4M3,
            },
            accumulator: CodeDType::F32,
            output: projection.output(),
            decode_order: match activation {
                // MiMo's block-FP8 GEMMs scale each 128-K block's BF16
                // MMA partial sum, whatever recipe the export states.
                ActivationFormat::Bf16
                    if matches!(
                        weight,
                        WeightFormat::Fp8Block128x128(_) | WeightFormat::Fp8Rows64x128(_)
                    ) =>
                {
                    DecodeOrder::BlockScaleAfterMma
                }
                ActivationFormat::Bf16 if weight == WeightFormat::Nvfp4 => {
                    DecodeOrder::ExactLocalThenGlobal
                }
                ActivationFormat::Bf16
                    if matches!(weight, WeightFormat::Mxfp8 | WeightFormat::Mxfp4) =>
                {
                    DecodeOrder::ExactScaledOperand
                }
                ActivationFormat::Bf16 if experts.is_some() || projection.lookup() => {
                    DecodeOrder::CodeThenScale
                }
                ActivationFormat::Bf16 => DecodeOrder::ScaleThenCode,
                ActivationFormat::Fp8Group128(_) if experts.is_some() => {
                    DecodeOrder::SuccessiveScalesAfterBlockMma
                }
                ActivationFormat::Fp8Group128(_) => DecodeOrder::ScaleProductAfterBlockMma,
                ActivationFormat::Mxfp8 => DecodeOrder::NativeBlockScaledMma,
                ActivationFormat::Nvfp4 => DecodeOrder::NativeBlockScaledMmaThenGlobals,
            },
            rounding: Rounding::NearestEven,
            numerical_contract: "mimo26.source-recipe.v1".into(),
        },
        epilogues: projection.epilogues(),
        uses: capabilities::uses(),
    }
}

/// One rank's fused QKV operand at `tp`, of `rank_rows` rows, from the
/// imported bank of the checkpoint's four TP4-local chunks
/// ([`qkv_scale_grid`]). At TP4 it is the rank's own chunk, a span of the
/// source codes and of its restarting block scales. Other rank counts gather
/// their heads from several chunks; block-FP8 sources become their exact
/// 64-row regroup ([`qkv_rank_scale_rows`], checked for every rank), a
/// prepared operand named after the source and the rank count.
pub(crate) fn qkv_rank_weight(
    operand: &ImportedOperand,
    tensors: &BTreeMap<String, TensorRecord>,
    tp: model_mimo26_contract::Tp,
    rank_rows: u64,
) -> MimoResult<WeightSpec> {
    let source = &operand.weight;
    let [parts, part_rows, k] = source.logical_shape[..] else {
        return Err("the fused QKV imports as a bank of row chunks".into());
    };
    let bits = kernel_core::precision::dtype_bits(source.codes.dtype);
    let mut weight = source.clone();
    weight.logical_shape = vec![rank_rows, k];
    weight.codes.shape = vec![rank_rows, k];
    weight.codes.byte_length = rank_rows * k * bits / 8;
    let tp4 = tp.ranks() == 4;
    if tp4 {
        // Rank 0's chunk; every rank's has the same geometry.
        if part_rows != rank_rows || parts != QKV_PARTS {
            return Err("a TP4 rank's QKV is one source chunk".into());
        }
    } else {
        weight.codes.tensor = format!("{}@tp{}", source.codes.tensor, tp.ranks());
        weight.codes.offset = 0;
    }
    if let Some(local) = &source.local_scale {
        let record = tensors
            .get(&local.tensor.tensor)
            .ok_or("QKV scale tensor absent")?;
        let k_blocks = *record.shape.last().ok_or("QKV scale tensor shape")?;
        let global = qkv_is_global(tp.geometry, &record.name)?;
        let grid = qkv_scale_grid(tp.geometry, global, record.shape[0], k_blocks as usize)?;
        if grid.rows_per_chunk as u64 != part_rows {
            return Err("QKV scale grid differs from its row chunks".into());
        }
        let mut scale_rows = Vec::new();
        for rank in 0..tp.count() {
            let rows = qkv_rank_scale_rows(grid, &tp.qkv_segments(global, rank), rank, tp)?;
            if rank == 0 {
                scale_rows = rows;
            }
        }
        let regroup = !tp4 && matches!(source.format, WeightFormat::Fp8Block128x128(_));
        let group_rows = if regroup {
            QKV_SCALE_GROUP_ROWS
        } else {
            u64::from(local.groups.first().map_or(1, |group| group.size))
        };
        if regroup {
            let WeightFormat::Fp8Block128x128(mode) = source.format else {
                unreachable!("block-FP8 regroup");
            };
            weight.format = WeightFormat::Fp8Rows64x128(mode);
        }
        if (scale_rows.len() as u64) != rank_rows.div_ceil(group_rows) && (tp4 || regroup) {
            return Err("a rank's QKV scale rows differ from its rows".into());
        }
        let scale = weight.local_scale.as_mut().expect("cloned scale");
        let element_bytes = kernel_core::precision::dtype_bits(local.tensor.dtype) / 8;
        let scale_count = rank_rows.div_ceil(group_rows);
        scale.tensor.shape = vec![scale_count, k_blocks];
        scale.tensor.byte_length = scale_count * k_blocks * element_bytes;
        if tp4 {
            scale.tensor.offset =
                local.tensor.offset + scale_rows[0] as u64 * k_blocks * element_bytes;
        } else {
            scale.tensor.tensor = format!("{}@tp{}", local.tensor.tensor, tp.ranks());
            scale.tensor.offset = 0;
        }
        scale.groups = local
            .groups
            .iter()
            .filter(|group| group.axis > 0)
            .map(|group| GroupAxis {
                axis: group.axis - 1,
                size: if group.axis == 1 && regroup {
                    QKV_SCALE_GROUP_ROWS as u32
                } else {
                    group.size
                },
                origin: 0,
            })
            .collect();
    }
    Ok(weight)
}

/// The offers of one projection at one rank count.
struct Offers {
    inventory: &'static CapabilityInventory,
    implementation: String,
}

impl ImplementationProvider for Offers {
    type Handle = usize;
    fn bind(
        &self,
        operation: &OperationSpec,
        usage: OperationUse,
    ) -> Result<Binding<usize>, String> {
        self.inventory.bind_matching(operation, usage, |c| {
            c.implementation == self.implementation
        })
    }
}

/// Imports the weight role `name` from `tensors` through the shared
/// importer: its own tensor and the `{stem}.*` companions the export names,
/// a fused QKV as its bank of row chunks.
pub(crate) fn import_role(
    metadata: &ExportMetadata,
    name: &str,
    role: &Role,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<ImportedOperand> {
    let Role::Weight {
        shape: [rows, k],
        parts,
        lookup,
        routed,
        declared_bf16,
        ..
    } = role
    else {
        return Err(format!("tensor {name} is not a weight role").into());
    };
    let stem = name.strip_suffix(".weight").expect("weight roles");
    // The importer reads only `{stem}.*` companions.
    let prefix = format!("{stem}.");
    let mut local = BTreeMap::new();
    for (other, record) in tensors.range(prefix.clone()..) {
        if !other.starts_with(&prefix) {
            break;
        }
        local.insert(other.clone(), span(record)?);
    }
    let logical = if *parts > 1 {
        // A bank of independently quantized row chunks, stored as
        // one `[rows, ...]` matrix (the packers read it so): only
        // that matrix is viewed as `[parts, rows / parts, ...]`.
        // The importer checks the stored K against the encoding.
        let codes = local.get_mut(name).expect("the role's own tensor");
        if codes.shape.len() != 2 || codes.shape[0] != *rows {
            return Err(format!(
                "tensor {name}: MiMo reads it as one matrix of {rows} rows \
                     ({parts} row chunks), the checkpoint stores shape {:?}",
                codes.shape
            )
            .into());
        }
        codes.shape = vec![*parts, rows / parts, codes.shape[1]];
        vec![*parts, rows / parts, *k]
    } else {
        vec![*rows, *k]
    };
    let mut targets = vec![stem.to_owned()];
    if *routed {
        targets.push(ROUTED_EXPERT_ROLE.into());
    }
    Ok(import_operand(
        metadata,
        &OperandRole {
            operation: stem.into(),
            weight: name.into(),
            targets,
            logical_shape: logical,
            lookup: *lookup,
            shared_scale_stem: None,
            declared: declared_bf16.then(EncodingRecipe::bf16),
        },
        &local,
    )?)
}

/// Imports every weight role of `roles` from `tensors` through the shared
/// importer, binds each through the capability offers of the projections
/// that serve it ([`check_weight`]), checks every side tensor's dtype and
/// shape, and requires every tensor to be a role or a companion the
/// importer read. Returns the imported weights by name.
///
/// A projection's weights share a decoder layer, so layers are admitted in
/// parallel; the first refusal in role order is returned.
pub fn admit(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    metadata: &ExportMetadata,
    tensors: &BTreeMap<String, TensorRecord>,
    roles: &BTreeMap<String, Role>,
) -> MimoResult<BTreeMap<String, ImportedOperand>> {
    let layer = |name: &str| {
        name.strip_prefix("model.layers.")
            .and_then(|tail| tail.split('.').next())
            .map(str::to_owned)
    };
    let mut groups: Vec<Vec<(&String, &Role)>> = Vec::new();
    for (name, role) in roles {
        match groups.last_mut() {
            Some(group) if layer(group[0].0) == layer(name) => group.push((name, role)),
            _ => groups.push(vec![(name, role)]),
        }
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get().min(8));
    let per_thread = groups.len().div_ceil(threads).max(1);
    let admitted: Vec<_> = std::thread::scope(|scope| {
        groups
            .chunks(per_thread)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut accounted = BTreeSet::new();
                    let mut operands = BTreeMap::new();
                    for group in chunk {
                        admit_group(
                            geometry,
                            metadata,
                            tensors,
                            group,
                            &mut accounted,
                            &mut operands,
                        )
                        .map_err(|error| error.to_string())?;
                    }
                    Ok::<_, String>((accounted, operands))
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|admitted| admitted.join().expect("an admission thread panicked"))
            .collect()
    });
    let mut accounted = BTreeSet::new();
    let mut operands = BTreeMap::new();
    for admitted in admitted {
        let (own, imported) = admitted?;
        accounted.extend(own);
        operands.extend(imported);
    }
    if let Some(name) = tensors.keys().find(|name| !accounted.contains(*name)) {
        return Err(format!(
            "tensor {name}: neither a MiMo tensor role nor a companion its export metadata declares"
        )
        .into());
    }
    Ok(operands)
}

/// Admits one layer's roles in order (see [`admit`]).
fn admit_group(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    metadata: &ExportMetadata,
    tensors: &BTreeMap<String, TensorRecord>,
    roles: &[(&String, &Role)],
    accounted: &mut BTreeSet<String>,
    operands: &mut BTreeMap<String, ImportedOperand>,
) -> MimoResult<()> {
    for &(name, role) in roles {
        let Some(tensor) = tensors.get(name) else {
            return Err(format!("missing MiMo tensor {name}").into());
        };
        match role {
            Role::Side { dtype, shape } => {
                if tensor.dtype != *dtype || tensor.shape != *shape {
                    return Err(format!(
                        "tensor {name}: MiMo reads {} {shape:?}, the checkpoint has {} {:?}",
                        dtype.safetensors_name(),
                        tensor.dtype.safetensors_name(),
                        tensor.shape
                    )
                    .into());
                }
                // A module rule cannot quantize a norm, bias or sink.
                metadata.require_unconverted(name)?;
                accounted.insert(name.clone());
            }
            Role::Weight { .. } => {
                let operand = import_role(metadata, name, role, tensors)?;
                check_weight(geometry, name, role, &operand, tensors, operands)?;
                accounted.extend(operand.source_tensors.iter().cloned());
                operands.insert(name.clone(), operand);
            }
        }
    }
    Ok(())
}

/// Imports the DFlash drafter's weights: its file carries no export
/// metadata and stores every weight BF16.
pub fn admit_dflash(
    geometry: &'static model_mimo26_contract::geometry::Geometry,
    tensors: &BTreeMap<String, TensorRecord>,
) -> MimoResult<BTreeMap<String, ImportedOperand>> {
    let metadata = export_metadata(None, None)?;
    dflash_roles(geometry)
        .iter()
        .filter(|(_, role)| matches!(role, Role::Weight { .. }))
        .map(|(name, role)| Ok((name.clone(), import_role(&metadata, name, role, tensors)?)))
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use kernel_core::codec::f32_to_bf16;
    use kernel_core::contract::ScaleMode;
    use kernel_core::precision::DecodeOrder;
    use kernel_core::quantize::{Quantized, Scales, fp8_weight_128x128, mxfp4};
    use model_compiler_sdk::quantization::payload::read_operand;

    /// How a synthetic export stores one weight.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum Stored {
        Bf16,
        /// FP8 128x128 blocks with `weight_scale_inv` of this dtype.
        Fp8Blocks(DType),
        Mxfp4,
        Mxfp8,
        Nvfp4,
    }

    /// The encoding the released checkpoints store `role` in: the one its
    /// offers read at TP4.
    pub(crate) fn served_stored(role: &Role) -> Stored {
        match role {
            Role::Weight { offered, .. } => match offered
                .first()
                .map(|projection| projection.weight(model_mimo26_contract::Tp::TP4))
            {
                Some(WeightFormat::Fp8Block128x128(_)) => Stored::Fp8Blocks(DType::F32),
                Some(WeightFormat::Mxfp4) => Stored::Mxfp4,
                _ => Stored::Bf16,
            },
            Role::Side { .. } => Stored::Bf16,
        }
    }

    /// Each role's tensors, every weight stored as `stored` says, with the
    /// companions its export names, by name.
    pub(crate) fn stored_tensors(
        roles: &BTreeMap<String, Role>,
        stored: impl Fn(&str, &Role) -> Stored,
    ) -> BTreeMap<String, (DType, Vec<u64>)> {
        let mut tensors = BTreeMap::new();
        for (name, role) in roles {
            let stem = name.strip_suffix(".weight").unwrap_or(name);
            let mut add = |suffix: &str, dtype: DType, shape: Vec<u64>| {
                tensors.insert(format!("{stem}.{suffix}"), (dtype, shape));
            };
            match role {
                Role::Side { dtype, shape } => {
                    tensors.insert(name.clone(), (*dtype, shape.clone()));
                }
                Role::Weight {
                    shape: [rows, k],
                    parts,
                    ..
                } => match stored(name, role) {
                    Stored::Bf16 => add("weight", DType::Bf16, vec![*rows, *k]),
                    Stored::Fp8Blocks(scale) => {
                        add("weight", DType::F8E4M3, vec![*rows, *k]);
                        add(
                            "weight_scale_inv",
                            scale,
                            vec![parts * (rows / parts).div_ceil(128), k.div_ceil(128)],
                        );
                    }
                    Stored::Mxfp4 => {
                        add("weight", DType::U8, vec![*rows, k / 2]);
                        add("weight_scale", DType::U8, vec![*rows, k / 32]);
                    }
                    Stored::Mxfp8 => {
                        add("weight", DType::F8E4M3, vec![*rows, *k]);
                        add("weight_scale", DType::U8, vec![*rows, k / 32]);
                    }
                    Stored::Nvfp4 => {
                        add("weight", DType::U8, vec![*rows, k / 2]);
                        add("weight_scale", DType::F8E4M3, vec![*rows, k / 16]);
                        add("weight_scale_2", DType::F32, vec![]);
                        add("input_scale", DType::F32, vec![]);
                    }
                },
            }
        }
        tensors
    }

    /// Each role's tensors as the released checkpoints store them: every
    /// weight in the encoding MiMo serves, with its companions, by name.
    pub(crate) fn served_tensors(
        roles: &BTreeMap<String, Role>,
    ) -> BTreeMap<String, (DType, Vec<u64>)> {
        stored_tensors(roles, |_, role| served_stored(role))
    }

    /// A header-only record of `name`: its bytes follow from dtype and shape.
    pub(crate) fn record(name: &str, dtype: DType, shape: &[u64]) -> TensorRecord {
        let width = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            _ => 1,
        };
        TensorRecord {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            file: "fixture.safetensors".into(),
            file_start: 0,
            bytes: shape.iter().product::<u64>() * width,
        }
    }

    fn records(tensors: BTreeMap<String, (DType, Vec<u64>)>) -> BTreeMap<String, TensorRecord> {
        tensors
            .into_iter()
            .map(|(name, (dtype, shape))| {
                let record = record(&name, dtype, &shape);
                (name, record)
            })
            .collect()
    }

    const GLOBAL: [usize; 9] = [0, 5, 11, 17, 23, 29, 35, 41, 47];

    fn released_config() -> serde_json::Value {
        serde_json::from_str(include_str!("../fixtures/config.json")).unwrap()
    }

    fn released_metadata() -> ExportMetadata {
        export_metadata(released_config().get("quantization_config"), None).unwrap()
    }

    fn weight_count(roles: &BTreeMap<String, Role>) -> usize {
        roles
            .values()
            .filter(|role| matches!(role, Role::Weight { .. }))
            .count()
    }

    #[test]
    fn released_export_imports_every_role_in_a_served_encoding() {
        let metadata = released_metadata();
        assert_eq!(metadata.expert_storage, Some(WeightFormat::Mxfp4));
        let roles = text_roles(&GEOMETRY, |layer| GLOBAL.contains(&layer));
        assert_eq!(
            expected_source_tensors(&roles).unwrap(),
            served_tensors(&roles)
        );
        let tensors = records(served_tensors(&roles));
        assert_eq!(tensors.len(), 72_574);
        let admitted = admit(&GEOMETRY, &metadata, &tensors, &roles).unwrap();
        assert_eq!(admitted.len(), weight_count(&roles));
        let format = |name: &str| admitted[name].weight.format;
        let block = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
        let qkv = "model.layers.5.self_attn.qkv_proj.weight";
        assert_eq!(format(qkv), block);
        assert_eq!(admitted[qkv].weight.logical_shape, [4, 3392, 4096]);
        assert_eq!(format("model.layers.0.mlp.gate_proj.weight"), block);
        for name in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.5.self_attn.o_proj.weight",
            "model.layers.3.mlp.gate.weight",
        ] {
            assert_eq!(format(name), WeightFormat::Bf16, "{name}");
        }
        let expert = "model.layers.3.mlp.experts.9.up_proj.weight";
        assert_eq!(format(expert), WeightFormat::Mxfp4);
        assert_eq!(
            admitted[expert].source_tensors,
            [
                expert.to_owned(),
                "model.layers.3.mlp.experts.9.up_proj.weight_scale".to_owned()
            ]
        );
        let mtp = mtp_roles(&GEOMETRY);
        let admitted = admit(&GEOMETRY, &metadata, &records(served_tensors(&mtp)), &mtp).unwrap();
        assert_eq!(admitted.len(), weight_count(&mtp));
        assert_eq!(
            admitted["model.mtp.layers.2.self_attn.qkv_proj.weight"]
                .weight
                .format,
            block
        );

        // A global layer's 108 scale rows fit four 3392-row chunks, not one
        // 13568-row matrix (106 rows).
        let mut flat = roles.clone();
        if let Some(Role::Weight { parts, .. }) =
            flat.get_mut("model.layers.0.self_attn.qkv_proj.weight")
        {
            *parts = 1;
        }
        let error = admit(&GEOMETRY, &metadata, &tensors, &flat)
            .unwrap_err()
            .to_string();
        assert!(error.contains("scale element count"), "{error}");

        // The packers read the fused QKV as one matrix: the same bytes stored
        // as a three-dimensional bank of the four chunks are refused.
        let qkv = "model.layers.0.self_attn.qkv_proj.weight";
        let mut banked = tensors.clone();
        banked.insert(qkv.into(), record(qkv, DType::F8E4M3, &[4, 3392, 4096]));
        let error = admit(&GEOMETRY, &metadata, &banked, &roles)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains(qkv)
                && error.contains("one matrix of 13568 rows")
                && error.contains("[4, 3392, 4096]"),
            "{error}"
        );
    }

    #[test]
    fn export_drift_and_unserved_encodings_are_refused_with_what_mimo_serves() {
        let config = released_config();
        for (pointer, value) in [
            ("/quantization_config/store_dtype", serde_json::json!("fp8")),
            (
                "/quantization_config/mxfp4_block_size",
                serde_json::json!(16),
            ),
            (
                "/quantization_config/weight_block_size",
                serde_json::json!([64, 128]),
            ),
            ("/quantization_config/fmt", serde_json::json!("e5m2")),
            (
                "/quantization_config/activation_scheme",
                serde_json::json!("static"),
            ),
        ] {
            let mut changed = config.clone();
            *changed.pointer_mut(pointer).unwrap() = value;
            assert!(
                export_metadata_from_config(&serde_json::to_vec(&changed).unwrap(), None).is_err(),
                "accepted {pointer}"
            );
        }

        let roles = text_roles(&GEOMETRY, |layer| GLOBAL.contains(&layer));
        let refusal = |metadata: &ExportMetadata, stored: &dyn Fn(&str, &Role) -> Stored| {
            admit(
                &GEOMETRY,
                metadata,
                &records(stored_tensors(&roles, stored)),
                &roles,
            )
            .unwrap_err()
            .to_string()
        };
        // An all-BF16 export (no quantization_config) has BF16 QKV and dense
        // MLP weights, which MiMo's FP8 GEMMs cannot read: no compiled
        // projection offers BF16 at the dense MLP's geometry.
        let error = refusal(&export_metadata(None, None).unwrap(), &|_, _| Stored::Bf16);
        assert!(
            error.contains("model.layers.0.mlp.down_proj.weight")
                && error.contains("makes it BF16 weights")
                && error.contains("no compiled MiMo projection serves it")
                && error.contains("no supplied implementation for Dense Bf16 N=4096 K=8192"),
            "{error}"
        );
        // BF16 block scales: the importer admits them, MiMo's views hold F32.
        let released = released_metadata();
        let error = refusal(&released, &|name, role| match served_stored(role) {
            Stored::Fp8Blocks(_) if name.contains("qkv_proj") => Stored::Fp8Blocks(DType::Bf16),
            other => other,
        });
        assert!(
            error.contains("model.layers.0.self_attn.qkv_proj.weight")
                && error.contains("no compiled MiMo projection serves it")
                && error.contains("supplied scale dtype differs"),
            "{error}"
        );
        // The dense MLP's offers take F32 block scales only.
        let error = refusal(&released, &|name, role| match served_stored(role) {
            Stored::Fp8Blocks(_) if name.contains("mlp.down_proj") => {
                Stored::Fp8Blocks(DType::Bf16)
            }
            other => other,
        });
        assert!(
            error.contains("model.layers.0.mlp.down_proj.weight")
                && error.contains("no compiled MiMo projection serves it")
                && error.contains("supplied scale dtype differs"),
            "{error}"
        );
        // Plain HF block FP8 without expert storage: block-FP8 experts.
        let mut plain = config["quantization_config"].clone();
        let fields = plain.as_object_mut().unwrap();
        fields.remove("store_dtype");
        fields.remove("mxfp4_block_size");
        let plain = export_metadata(Some(&plain), None).unwrap();
        let error = refusal(&plain, &|_, role| match served_stored(role) {
            Stored::Mxfp4 => Stored::Fp8Blocks(DType::F32),
            other => other,
        });
        assert!(
            error.contains("operation model.layers.1.mlp.experts.down_proj")
                && error.contains("makes it FP8 128x128 block weights")
                && error.contains("no compiled MiMo projection serves it")
                && error.contains("Grouped { experts: 256, top_k: 8 } Fp8Block128x128"),
            "{error}"
        );
    }

    #[test]
    fn qkv_scale_grid_uses_layer_geometry() {
        for global in [true, false] {
            let rows = model_mimo26_contract::tp::checkpoint_qkv_scale_rows(global);
            let grid = qkv_scale_grid(&GEOMETRY, global, rows, (HIDDEN / 128) as usize).unwrap();
            assert_eq!(
                grid.rows_per_chunk as u64,
                model_mimo26_contract::tp::checkpoint_qkv_chunk_rows(global)
            );
            let error = qkv_scale_grid(&GEOMETRY, !global, rows, (HIDDEN / 128) as usize)
                .unwrap_err()
                .to_string();
            assert!(error.contains("does not match its chunked grid"), "{error}");
        }
    }

    /// Every row of every rank's QKV operand dequantizes with its source
    /// row's own block scale: TP4 ranks keep their chunk's restarting grid,
    /// and the 64-row regroup of TP2 crosses chunk boundaries where the
    /// grid restarts.
    #[test]
    fn qkv_rank_operands_keep_every_source_rows_scale() {
        for global in [true, false] {
            let rows = qkv_rows(&GEOMETRY, global);
            let scale_rows = QKV_PARTS * (rows / QKV_PARTS).div_ceil(128);
            let grid = qkv_scale_grid(&GEOMETRY, global, scale_rows, 32).unwrap();
            // A distinct value per source scale row.
            let values: Vec<f32> = (0..scale_rows).map(|row| 1.0 + row as f32).collect();
            for ranks in model_mimo26_contract::Tp::SUPPORTED {
                let tp = model_mimo26_contract::Tp::of_geometry(&GEOMETRY, ranks.ranks()).unwrap();
                let group = if tp.ranks() == 4 {
                    128
                } else {
                    QKV_SCALE_GROUP_ROWS as usize
                };
                let mut uses = BTreeMap::<usize, usize>::new();
                let mut crossed = false;
                for rank in 0..tp.count() {
                    let runs = if tp.ranks() == 4 {
                        let chunk = rows / QKV_PARTS;
                        std::iter::once(rank as u64 * chunk..(rank as u64 + 1) * chunk).collect()
                    } else {
                        tp.qkv_segments(global, rank)
                    };
                    let rank_scales = qkv_rank_scale_rows(grid, &runs, rank, tp).unwrap();
                    let source_rows: Vec<usize> = runs
                        .iter()
                        .flat_map(|run| run.start as usize..run.end as usize)
                        .collect();
                    assert_eq!(rank_scales.len(), source_rows.len().div_ceil(group));
                    for (row, &source) in source_rows.iter().enumerate() {
                        assert_eq!(
                            values[rank_scales[row / group]],
                            values[grid.scale_row(source).unwrap()],
                            "TP{} rank {rank} row {row} (source {source})",
                            tp.ranks()
                        );
                        *uses.entry(source).or_default() += 1;
                    }
                    let chunks: std::collections::BTreeSet<_> = source_rows
                        .iter()
                        .map(|row| row / grid.rows_per_chunk)
                        .collect();
                    crossed |= chunks.len() > 1;
                }
                let shared = uses.values().filter(|&&count| count > 1).count();
                match tp.ranks() {
                    4 => assert!(!crossed && shared == 0 && uses.len() == rows as usize),
                    2 => assert!(crossed && shared == 0 && uses.len() == rows as usize),
                    _ => assert!(uses.len() == rows as usize && (shared > 0) == global),
                }
            }
        }
        // A chunk's first row starts a new scale row: the grid restarts.
        let grid = qkv_scale_grid(&GEOMETRY, true, 108, 32).unwrap();
        assert_eq!(grid.scale_row(3391).unwrap(), 26);
        assert_eq!(grid.scale_row(3392).unwrap(), 27);
    }

    /// Block-FP8 dense MLP weights admit with BF16 activations (a
    /// weight-only recipe) as with dynamic FP8 ones: the kernels run BF16
    /// activations either way.
    #[test]
    fn weight_only_and_dynamic_block_fp8_mlp_recipes_admit() {
        for fused in [false, true] {
            let roles: BTreeMap<String, Role> =
                text_roles(&GEOMETRY, |layer| GLOBAL.contains(&layer))
                    .into_iter()
                    .filter(|(name, _)| {
                        name.starts_with("model.layers.0.mlp.")
                            && if fused {
                                name.contains("gate_proj") || name.contains("up_proj")
                            } else {
                                name.contains("down_proj")
                            }
                    })
                    .collect();
            let tensors = records(stored_tensors(&roles, |_, _| Stored::Fp8Blocks(DType::F32)));
            for dynamic in [false, true] {
                let mut document = serde_json::json!({
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "config_groups": {"block_fp8": {
                        "targets": ["model.layers.0.mlp"],
                        "weights": {"dynamic": false, "num_bits": 8, "type": "float",
                                    "group_size": 128},
                    }},
                });
                if dynamic {
                    document["config_groups"]["block_fp8"]["input_activations"] = serde_json::json!(
                        {"dynamic": true, "num_bits": 8, "type": "float", "group_size": 128}
                    );
                }
                let metadata = export_metadata(Some(&document), None).unwrap();
                let admitted = admit(&GEOMETRY, &metadata, &tensors, &roles)
                    .unwrap_or_else(|e| panic!("fused={fused} dynamic={dynamic}: {e}"));
                let activation = admitted
                    .values()
                    .next()
                    .unwrap()
                    .activation
                    .as_ref()
                    .unwrap();
                assert_eq!(activation.is_bf16(), !dynamic);
            }
        }
    }

    /// A MiMo-shaped export of layers 0 and 1 in ModelOpt's mixed schema:
    /// block-FP8 attention and dense MLP with dynamic FP8 inputs, and routed
    /// experts in `experts` (`NVFP4` or `MXFP8`).
    fn modelopt_export(experts: &str, group: u32) -> ExportMetadata {
        let document = serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {"block_fp8": {
                "targets": [
                    "model.layers.0.self_attn.qkv_proj",
                    "model.layers.1.self_attn.qkv_proj",
                    "model.layers.0.mlp",
                ],
                "weights": {"dynamic": false, "num_bits": 8, "type": "float", "group_size": 128},
                "input_activations": {"dynamic": true, "num_bits": 8, "type": "float", "group_size": 128},
            }},
            "quantized_layers": {"model.layers.1.mlp.experts": {"quant_algo": experts, "group_size": group}},
        });
        export_metadata(Some(&document), None).unwrap()
    }

    #[test]
    fn few_layer_nvfp4_and_mxfp8_exports_import_but_mimo_refuses_their_experts() {
        let roles: BTreeMap<String, Role> = text_roles(&GEOMETRY, |layer| GLOBAL.contains(&layer))
            .into_iter()
            .filter(|(name, _)| {
                !name.starts_with("model.layers.")
                    || name.starts_with("model.layers.0.")
                    || name.starts_with("model.layers.1.")
            })
            .collect();
        let expert = "model.layers.1.mlp.experts.0.down_proj.weight";
        for (algorithm, group, stored, format, activation) in [
            (
                "NVFP4",
                16,
                Stored::Nvfp4,
                WeightFormat::Nvfp4,
                kernel_core::contract::ActivationFormat::Nvfp4,
            ),
            (
                "MXFP8",
                32,
                Stored::Mxfp8,
                WeightFormat::Mxfp8,
                kernel_core::contract::ActivationFormat::Mxfp8,
            ),
        ] {
            let metadata = modelopt_export(algorithm, group);
            let tensors = records(stored_tensors(&roles, |_, role| {
                match served_stored(role) {
                    Stored::Mxfp4 => stored,
                    other => other,
                }
            }));
            // The shared importer interprets the expert and its companions.
            let prefix = "model.layers.1.mlp.experts.0.down_proj.";
            let local = tensors
                .iter()
                .filter(|(name, _)| name.starts_with(prefix))
                .map(|(name, record)| (name.clone(), span(record).unwrap()))
                .collect();
            let imported = import_operand(
                &metadata,
                &OperandRole {
                    operation: "down".into(),
                    weight: expert.into(),
                    targets: vec![
                        expert.strip_suffix(".weight").unwrap().into(),
                        ROUTED_EXPERT_ROLE.into(),
                    ],
                    logical_shape: vec![HIDDEN, 2048],
                    lookup: false,
                    shared_scale_stem: None,
                    declared: None,
                },
                &local,
            )
            .unwrap();
            assert_eq!(imported.weight.format, format);
            assert_eq!(imported.activation.unwrap().format, activation);
            // Every other role of the export is one MiMo serves.
            let served: BTreeMap<String, Role> = roles
                .iter()
                .filter(|(_, role)| !matches!(served_stored(role), Stored::Mxfp4))
                .map(|(name, role)| (name.clone(), role.clone()))
                .collect();
            let others = tensors
                .iter()
                .filter(|(name, _)| !name.contains(".mlp.experts."))
                .map(|(name, record)| (name.clone(), record.clone()))
                .collect();
            admit(&GEOMETRY, &metadata, &others, &served).unwrap();
            // MiMo's MoE kernels read MXFP4 experts only: no grouped offer
            // serves the layer's bank.
            let error = admit(&GEOMETRY, &metadata, &tensors, &roles)
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("operation model.layers.1.mlp.experts.")
                    && error.contains(&format!("makes it {algorithm} weights"))
                    && error.contains("no compiled MiMo projection serves it")
                    && error.contains(&format!("Grouped {{ experts: 256, top_k: 8 }} {format:?}")),
                "{error}"
            );
        }
    }

    fn values(count: usize, seed: u32) -> Vec<f32> {
        (0..count as u32)
            .map(|n| {
                let x = n
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(seed.wrapping_mul(97))
                    >> 8;
                (x % 2001) as f32 / 250.0 - 4.0
            })
            .collect()
    }

    fn scale_bytes(quantized: &Quantized) -> Vec<u8> {
        match &quantized.scales {
            Scales::F32(values) => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Scales::Ue8M0(bytes) | Scales::Ue4M3(bytes) => bytes.clone(),
        }
    }

    /// Checks every element the shared payload reader decodes from
    /// `operand` against the quantizers' own decoding, part by part.
    fn check_decoded(
        operand: &ImportedOperand,
        payloads: &BTreeMap<String, Vec<u8>>,
        parts: &[Quantized],
    ) {
        let loaded = read_operand(operand.clone(), |span| {
            let bytes = &payloads[&span.tensor];
            Ok(bytes[span.offset as usize..][..span.byte_length as usize].to_vec())
        })
        .unwrap();
        for (index, quantized) in parts.iter().enumerate() {
            let matrix = loaded.matrix(index).unwrap();
            assert_eq!((matrix.rows, matrix.k), (quantized.rows, quantized.k));
            for row in 0..quantized.rows {
                for col in 0..quantized.k {
                    assert_eq!(
                        matrix
                            .bf16_operand(row, col, DecodeOrder::CodeThenScale)
                            .unwrap(),
                        f32_to_bf16(quantized.decode(row, col) as f32),
                        "part {index} [{row}, {col}]"
                    );
                }
            }
        }
    }

    /// Weights quantized by the shared quantizers, stored as the released
    /// export stores them, decode through the shared importer and payload
    /// reader to the quantizers' values: the chunked QKV block scales and the
    /// MXFP4 expert nibbles and UE8M0 scales mean what MiMo packs.
    #[test]
    fn synthetic_quantized_weights_decode_through_the_shared_reader() {
        let (part_rows, k) = (136_u64, 256_u64);
        let qkv = "blocks.0.self_attn.qkv_proj.weight";
        let routed = "blocks.0.mlp.experts.3.up_proj.weight";
        let roles = BTreeMap::from([
            (
                qkv.to_owned(),
                Role::Weight {
                    shape: [4 * part_rows, k],
                    parts: 4,
                    lookup: false,
                    routed: false,
                    declared_bf16: false,
                    offered: &[Projection::Qkv { global: true }],
                },
            ),
            (
                routed.to_owned(),
                expert(32, k, &[Projection::ExpertGateUp]),
            ),
        ]);
        let chunks: Vec<Quantized> = (0..4)
            .map(|part| {
                fp8_weight_128x128(
                    &values((part_rows * k) as usize, part),
                    part_rows as usize,
                    k as usize,
                    ScaleMode::ExactF32,
                )
                .unwrap()
            })
            .collect();
        let moe = mxfp4(&values(32 * k as usize, 9), 32, k as usize).unwrap();
        let payloads = BTreeMap::from([
            (
                qkv.to_owned(),
                chunks.iter().flat_map(|q| q.codes.clone()).collect(),
            ),
            (
                format!("{qkv}_scale_inv"),
                chunks.iter().flat_map(scale_bytes).collect::<Vec<u8>>(),
            ),
            (routed.to_owned(), moe.codes.clone()),
            (format!("{routed}_scale"), scale_bytes(&moe)),
        ]);
        let tensors = records(served_tensors(&roles));
        for (name, record) in &tensors {
            assert_eq!(record.bytes, payloads[name].len() as u64, "{name}");
        }
        // Their shapes are no MiMo projection's: import them alone.
        let import =
            |name: &str| import_role(&released_metadata(), name, &roles[name], &tensors).unwrap();
        check_decoded(&import(qkv), &payloads, &chunks);
        check_decoded(&import(routed), &payloads, &[moe]);
    }
}
