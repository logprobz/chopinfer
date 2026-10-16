//! Draft projection contracts. Providers decode parameters but execute no math.
use crate::backend::{Kernels, Selected};
use projection_spec::registry::{Provider, Registration, Registry, Target};
use projection_spec::{DType, Dim, Error, Geometry, Variant};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod imports;
pub mod sources;
pub mod variants;

pub const DECLARATION: &str = include_str!("../projections.json");

/// The exact source geometry and reduction identity of a legacy dense port.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseParameters {
    pub operation: String,
    pub rows: Dim,
    pub k: Dim,
    pub scale_order: ScaleOrder,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ScaleOrder {
    ScaleThenCode,
    BlockScaleAfterMma,
    ExactScaledOperand,
}

fn typed<T: for<'de> Deserialize<'de> + Serialize>(v: &Variant) -> Result<T, Error> {
    serde_json::from_value(v.parameters.clone()).map_err(|e| Error(e.to_string()))
}
fn canonical<T: Serialize>(p: &T) -> Result<Value, Error> {
    serde_json::to_value(p).map_err(|e| Error(e.to_string()))
}
fn require(condition: bool, reason: &str) -> Result<(), Error> {
    if condition {
        Ok(())
    } else {
        Err(Error(reason.into()))
    }
}
fn ports(v: &Variant, inputs: &[(&str, DType)], outputs: &[(&str, DType)]) -> Result<(), Error> {
    for (ports, expected) in [(&v.inputs, inputs), (&v.outputs, outputs)] {
        require(
            ports.len() == expected.len(),
            "port count differs from the registered contract",
        )?;
        for (port, &(id, dtype)) in ports.iter().zip(expected) {
            require(
                port.id == id && port.binding == id && port.dtype == dtype,
                "port identity or dtype differs from the registered contract",
            )?;
        }
    }
    Ok(())
}
fn dense_output(v: &Variant, g: &Geometry, output: DType) -> Result<Value, Error> {
    let p: DenseParameters = typed(v)?;
    require(
        projection_spec::registry::valid_name(&p.operation),
        "invalid dense operation",
    )?;
    p.rows.resolve(g)?;
    p.k.resolve(g)?;
    ports(v, &[("activation", DType::Bf16)], &[("output", output)])?;
    require(
        v.inputs[0].shape == vec![Dim::Literal(1), p.k.clone()]
            && v.outputs[0].shape == vec![Dim::Literal(1), p.rows.clone()],
        "dense ports differ from parameter geometry",
    )?;
    canonical(&p)
}
fn dense(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    dense_output(v, g, DType::Bf16)
}
/// Construct an explicitly admitted model namespace for declaration parsing.
pub fn registry() -> Result<Registry, Error> {
    let mut registry = Registry::default();
    registry.register(&PROVIDER)?;
    Ok(registry)
}

/// The Flash declaration at `tp` ranks.
pub fn declaration(tp: u32) -> Result<projection_spec::Declaration, Error> {
    declaration_for(crate::Tp::new(tp).map_err(Error)?)
}

/// The profile-expanded declaration and geometry of every served topology,
/// built once per process.
pub fn expanded(tp: crate::Tp) -> Result<&'static (projection_spec::Declaration, Geometry), Error> {
    type Expanded = (
        crate::Tp,
        Result<(projection_spec::Declaration, Geometry), Error>,
    );
    static EXPANDED: std::sync::OnceLock<Vec<Expanded>> = std::sync::OnceLock::new();
    EXPANDED
        .get_or_init(|| {
            crate::geometry::GEOMETRIES
                .iter()
                .flat_map(|g| crate::Tp::SUPPORTED.map(|ranks| (g, ranks.ranks())))
                .filter_map(|(g, ranks)| crate::Tp::of_geometry(g, ranks).ok())
                .map(|tp| {
                    let geometry = geometry_for(tp);
                    let expanded = declaration_for(tp)
                        .and_then(|d| d.expanded_profiles(&geometry))
                        .map(|declaration| (declaration, geometry));
                    (tp, expanded)
                })
                .collect()
        })
        .iter()
        .find(|(served, _)| *served == tp)
        .ok_or_else(|| Error("unserved projection topology".into()))?
        .1
        .as_ref()
        .map_err(Clone::clone)
}

/// Copy-only regrouping keeps four independent scale origins and explicit TP formats.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RegroupParameters {
    pub q_heads: Dim,
    pub kv_heads: Dim,
    pub qk_head_dim: Dim,
    pub value_head_dim: Dim,
    pub chunks: u32,
    pub chunk_rows: Dim,
    pub block_rows: u32,
    pub k_block: u32,
    pub output_group_rows: u32,
    pub rank_counts: Vec<u32>,
    pub bypass_ranks: Vec<u32>,
    pub member_order: Vec<String>,
}
impl RegroupParameters {
    /// Rows of one checkpoint Q|K|V chunk in the topology's geometry.
    pub fn chunk_rows(&self, tp: crate::Tp) -> Result<u64, Error> {
        self.chunk_rows.resolve(&geometry_for(tp))
    }
}
fn regroup(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    let p: RegroupParameters = typed(v)?;
    let chunks = u64::from(p.chunks);
    let [q, kv, qk, value, rows] = [
        &p.q_heads,
        &p.kv_heads,
        &p.qk_head_dim,
        &p.value_head_dim,
        &p.chunk_rows,
    ]
    .map(|d| d.resolve(g));
    let [q, kv, qk, value, rows] = [q?, kv?, qk?, value?, rows?];
    require(
        p.chunks == 4
            && p.block_rows == 128
            && p.k_block == 128
            && p.output_group_rows == 64
            && q % chunks == 0
            && kv % chunks == 0
            && rows * chunks == q * qk + kv * (qk + value),
        "regroup parameters differ from the source grid contract",
    )?;
    require(
        p.rank_counts == [2] && p.bypass_ranks == [4] && p.member_order == ["q", "k", "v"],
        "regroup topology or order differs",
    )?;
    ports(
        v,
        &[("codes", DType::Fp8E4M3), ("scales", DType::F32)],
        &[("codes", DType::Fp8E4M3), ("scales", DType::F32)],
    )?;
    use kernel_core::precision::format::FormatId;
    require(
        v.inputs[0].format == Some(FormatId::Fp8Block128)
            && v.outputs[0].format == Some(FormatId::Fp8Rows64),
        "regroup needs block128 input and rows64 output",
    )?;
    canonical(&p)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingParameters {
    pub experts: Dim,
    pub top_k: u32,
    pub capacities: Vec<u32>,
    pub score: RoutingScore,
    pub ties: RoutingTies,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RoutingScore {
    #[serde(rename = "sigmoid-correction")]
    SigmoidCorrection,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RoutingTies {
    #[serde(rename = "lower-index")]
    LowerIndex,
}
fn routing(v: &Variant, g: &Geometry) -> Result<Value, Error> {
    let p: RoutingParameters = typed(v)?;
    require(
        g.get("experts") == Some(&p.experts.resolve(g)?)
            && p.top_k == 8
            && p.capacities == [1, 2, 4, 8],
        "routing parameters differ from default served fusion",
    )?;
    ports(
        v,
        &[("activation", DType::Bf16), ("bias", DType::F32)],
        &[("ids", DType::I32), ("weights", DType::F32)],
    )?;
    canonical(&p)
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombineParameters {
    pub route_order: RouteOrder,
    pub rank_order: RankOrder,
    pub route_rounding: RouteRounding,
    pub residual_rounding: ResidualRounding,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RouteOrder {
    #[serde(rename = "ascending-route")]
    Ascending,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RankOrder {
    #[serde(rename = "ascending-rank")]
    Ascending,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum RouteRounding {
    #[serde(rename = "bf16-before-combine")]
    BeforeCombine,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum ResidualRounding {
    #[serde(rename = "once-after-sum")]
    Once,
}
fn combine(v: &Variant, _g: &Geometry) -> Result<Value, Error> {
    let p: CombineParameters = typed(v)?;
    ports(v, &[("routes", DType::Bf16)], &[("hidden", DType::Bf16)])?;
    canonical(&p)
}

pub static PROVIDER: Provider = Provider {
    namespace: "mimo26",
    registrations: &[
        Registration {
            id: "mimo26/dflash-mask.v1",
            numerical_contract: "mimo26.dflash-mask.v1",
            target: Target::Stage,
            decoder: mask,
        },
        Registration {
            id: "mimo26/qkv-scale-regroup.v1",
            numerical_contract: "mimo26.qkv-scale-regroup.v1",
            target: Target::SourceTransform,
            decoder: regroup,
        },
        Registration {
            id: "mimo26/legacy-fused-dense.v1",
            numerical_contract: "mimo26.source-recipe.v1",
            target: Target::Projection,
            decoder: dense,
        },
        Registration {
            id: "mimo26/legacy-routing.v1",
            numerical_contract: "mimo26.routing.v1",
            target: Target::Stage,
            decoder: routing,
        },
        Registration {
            id: "mimo26/legacy-combine.v1",
            numerical_contract: "mimo26.combine.v1",
            target: Target::Stage,
            decoder: combine,
        },
    ],
};

/// The Flash declaration geometry at `tp` ranks.
pub fn geometry(tp: u32) -> Result<Geometry, Error> {
    Ok(geometry_for(crate::Tp::new(tp).map_err(Error)?))
}

/// The geometry keys the declaration names: model widths, layer families
/// and the per-attention fused QKV layout. The checkpoint stores QKV as
/// four independently quantized Q|K|V chunks. `{family}-layer-{n}` keys
/// list each attention family's layers ([`declaration_for`]).
pub fn geometry_for(tp: crate::Tp) -> Geometry {
    let g = tp.geometry;
    let mut values: Geometry = [
        ("tp", tp.ranks()),
        ("hidden", g.hidden()),
        ("vocab", g.vocab()),
        ("experts", g.experts()),
        ("top_k", g.top_k()),
        ("dense_intermediate", g.dense_intermediate()),
        ("expert_intermediate", g.expert_intermediate()),
        ("layers", g.layers()),
        ("dense_layers", g.dense_layer_end()),
        ("mtp_layers", g.mtp_layers()),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), u64::from(v)))
    .collect();
    let chunks = crate::tp::QKV_CHUNKS;
    for (family, global, q, kv, qk, v) in [
        (
            "ga",
            true,
            g.attention_q_heads(),
            g.attention_kv_heads(),
            g.head_dim(),
            g.value_head_dim(),
        ),
        (
            "swa",
            false,
            g.swa_q_heads(),
            g.swa_kv_heads(),
            g.swa_head_dim(),
            g.swa_value_head_dim(),
        ),
    ] {
        let [q, kv, qk, v] = [q, kv, qk, v].map(u64::from);
        let layers: Vec<_> = (0..g.layers())
            .filter(|&i| g.is_global(i) == global)
            .collect();
        let chunk = (q * qk + kv * (qk + v)) / chunks;
        for (key, value) in [
            ("layers", layers.len() as u64),
            ("q_heads", q),
            ("kv_heads", kv),
            ("qk_dim", qk),
            ("v_dim", v),
            ("qkv_rows", chunks * chunk),
            ("chunk_rows", chunk),
            ("chunk_q_heads", q / chunks),
            ("chunk_kv_heads", kv / chunks),
            ("chunk_q_rows", q / chunks * qk),
            ("chunk_k_rows", kv / chunks * qk),
            ("chunk_v_rows", kv / chunks * v),
            ("scale_rows", chunks * chunk.div_ceil(128)),
            ("output_k", q * v),
        ] {
            values.insert(format!("{family}_{key}"), value);
        }
        for (n, layer) in layers.into_iter().enumerate() {
            values.insert(format!("{family}-layer-{n}"), u64::from(layer));
        }
    }
    let drafter = g.dflash();
    let [q, kv, head, v] = [
        drafter.num_attention_heads,
        drafter.num_key_value_heads,
        drafter.head_dim,
        drafter.v_head_dim,
    ]
    .map(u64::from);
    for (key, value) in [
        ("hidden", u64::from(drafter.hidden_size)),
        ("intermediate", u64::from(drafter.intermediate_size)),
        ("layers", u64::from(drafter.num_hidden_layers)),
        ("q_heads", q),
        ("head_dim", head),
        ("q_rows", q * head),
        ("kv_rows", kv * head),
        ("v_rows", kv * v),
        ("qkv_rows", q * head + kv * (head + v)),
        ("output_k", q * v),
        (
            "context_kv_rows",
            u64::from(drafter.num_hidden_layers) * kv * (head + v),
        ),
        (
            "fc_k",
            g.tap_count() as u64 * u64::from(drafter.target_hidden_size),
        ),
    ] {
        values.insert(format!("dflash_{key}"), value);
    }
    values
}

/// Parse the declaration for a served topology. A stem whose range ends at a
/// `{family}_layers` key with listed `{family}-layer-{n}` keys selects those
/// layers, so data-only geometries keep their own attention pattern.
pub fn declaration_for(tp: crate::Tp) -> Result<projection_spec::Declaration, Error> {
    let geometry = geometry_for(tp);
    let mut declaration: projection_spec::Declaration =
        serde_json::from_str(DECLARATION).map_err(|e| Error(e.to_string()))?;
    for stem in &mut declaration.layer_stems {
        if let projection_spec::LayerIndices::Range {
            start: Dim::Literal(0),
            end: Dim::Key(key),
        } = &stem.indices
            && let Some(family) = key.strip_suffix("_layers")
            && geometry.contains_key(&format!("{family}-layer-0"))
        {
            stem.indices = projection_spec::LayerIndices::List {
                indices: (0..geometry[key])
                    .map(|n| Dim::Key(format!("{family}-layer-{n}")))
                    .collect(),
            };
        }
    }
    // The context K/V projection fuses every drafter layer's K then V; the
    // declaration lists layer 0's pair.
    if let Some(context) = declaration
        .projections
        .iter_mut()
        .find(|p| p.id == "dflash.context-kv")
    {
        let pair = context.assembly.clone();
        context.assembly = (0..geometry["dflash_layers"])
            .flat_map(|layer| {
                pair.iter().map(move |member| {
                    let mut member = member.clone();
                    member.layer = Some(projection_spec::LayerSelector::Index(Dim::Literal(layer)));
                    member
                })
            })
            .collect();
    }
    projection_spec::parse(
        &serde_json::to_string(&declaration).map_err(|e| Error(e.to_string()))?,
        &geometry,
        &registry()?,
    )
}

/// Mask archive role is typed here; no PT payload or masking math is executed.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaskParameters {
    pub source: String,
    pub width: u32,
    pub format: MaskFormat,
    pub adapter: projection_spec::derive::StageAdapter,
}
#[derive(Debug, Serialize, Deserialize)]
pub enum MaskFormat {
    #[serde(rename = "bf16-archive")]
    Bf16Archive,
}
fn mask(v: &Variant, _g: &Geometry) -> Result<Value, Error> {
    let p: MaskParameters = typed(v)?;
    require(
        p.source == "mask_embedding.pt" && p.width == 4096,
        "mask source differs from archive contract",
    )?;
    ports(
        v,
        &[("activation", DType::Bf16), ("mask", DType::Bf16)],
        &[("activation", DType::Bf16)],
    )?;
    use projection_spec::derive::{AdapterValue, StageExecution};
    require(
        p.adapter.contract == v.id
            && p.adapter.execution
                == StageExecution::Standalone {
                    symbol: Selected::bundle_entry(crate::Tp::TP4, "dflash_mask_rows"),
                },
        "mask adapter identity or default symbol differs",
    )?;
    require(
        p.adapter.facts.len() == 4,
        "mask adapter needs the four native controls",
    )?;
    for name in ["positions", "prefix_len", "token_batch", "live_tokens"] {
        let Some(AdapterValue::Binding(binding)) = p.adapter.facts.get(name) else {
            return Err(Error(
                "mask native control is missing or not a binding".into(),
            ));
        };
        require(
            binding.view == name
                && binding.dtype == DType::I32
                && binding.offset_bytes == 0
                && binding.stride_bytes.is_none()
                && binding.interpretation
                    == if name == "live_tokens" {
                        Some(projection_spec::contracts::Interpretation::Unsigned)
                    } else {
                        None
                    },
            "mask control storage or interpretation differs from its native ABI",
        )?;
    }
    canonical(&p)
}

impl MaskParameters {
    /// Resolve the native launch target without changing logical ports or arithmetic.
    pub fn native_adapter(&self, tp: crate::Tp) -> projection_spec::derive::StageAdapter {
        let mut adapter = self.adapter.clone();
        adapter.execution = projection_spec::derive::StageExecution::Standalone {
            symbol: Selected::bundle_entry(tp, "dflash_mask_rows"),
        };
        adapter
    }
}

#[cfg(test)]
mod tests {
    use crate::{Tp, geometry::GEOMETRIES, precision};

    fn layer(name: &str) -> Option<u32> {
        name.strip_prefix("model.layers.")?
            .split('.')
            .next()?
            .parse()
            .ok()
    }

    /// Every listed geometry, data-only additions included, materializes
    /// its own layers, attention pattern and QKV layout, so its roles and
    /// text image views follow its configuration.
    #[test]
    fn listed_geometries_derive_their_own_sources_roles_and_views() {
        for geometry in GEOMETRIES {
            let roles = precision::text_roles(geometry, |i| geometry.is_global(i as u32));
            assert_eq!(
                roles.keys().filter_map(|name| layer(name)).max(),
                Some(geometry.layers() - 1)
            );
            for (name, role) in &roles {
                if let (Some(layer), precision::Role::Weight { shape, .. }) = (layer(name), role)
                    && name.ends_with(".qkv_proj.weight")
                {
                    let tp = Tp::of_geometry(geometry, 4).unwrap();
                    assert_eq!(shape[0], tp.checkpoint_rows(geometry.is_global(layer)));
                }
            }
            for ranks in Tp::SUPPORTED {
                let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                let views = crate::text_views(tp);
                assert_eq!(
                    views.keys().filter_map(|name| layer(name)).max(),
                    Some(geometry.layers() - 1)
                );
                let (declaration, _) = super::expanded(tp).unwrap();
                assert!(declaration.projections.iter().any(|p| p.id == "ga.qkv"));
            }
        }
    }

    /// Every listed geometry's text, MTP and DFlash projections bind offers
    /// of its own entries, at every rank count: header-only checkpoints of
    /// the geometry enter production admission and the component builder.
    #[test]
    fn listed_geometries_build_text_and_drafter_components() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/projections/mopd.json")).unwrap();
        for geometry in GEOMETRIES {
            let mut config = fixture["quantization_config"].clone();
            config["ignored_layers"] = (0..geometry.layers())
                .map(|layer| format!("model.layers.{layer}.self_attn.o_proj"))
                .chain(["model.decoder.self_attn.o_proj".to_owned()])
                .collect();
            let metadata = precision::export_metadata(Some(&config), None).unwrap();
            let mut roles = precision::text_roles(geometry, |i| geometry.is_global(i as u32));
            roles.extend(precision::mtp_roles(geometry));
            let drafter = precision::dflash_roles(geometry);
            let mut tensors: std::collections::BTreeMap<_, _> =
                precision::expected_source_tensors(&roles)
                    .unwrap()
                    .into_iter()
                    .map(|(name, (dtype, shape))| {
                        let width = match dtype.safetensors_name() {
                            "BF16" => 2,
                            "F32" => 4,
                            _ => 1,
                        };
                        let record = crate::catalog::TensorRecord {
                            name: name.clone(),
                            dtype,
                            bytes: shape.iter().product::<u64>() * width,
                            shape,
                            file: "fixture.safetensors".into(),
                            file_start: 0,
                        };
                        (name, record)
                    })
                    .collect();
            let mut admitted = precision::admit(geometry, &metadata, &tensors, &roles).unwrap();
            // The DFlash drafter's own BF16 file joins the same component.
            let drafter: std::collections::BTreeMap<_, _> =
                precision::expected_source_tensors(&drafter)
                    .unwrap()
                    .into_iter()
                    .map(|(name, (dtype, shape))| {
                        let record = crate::catalog::TensorRecord {
                            name: name.clone(),
                            dtype,
                            bytes: shape.iter().product::<u64>() * 2,
                            shape,
                            file: "drafter.safetensors".into(),
                            file_start: 0,
                        };
                        (name, record)
                    })
                    .collect();
            admitted.extend(precision::admit_dflash(geometry, &drafter).unwrap());
            tensors.extend(drafter);
            for ranks in Tp::SUPPORTED {
                let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                let graph = super::imports::derive(tp, 0, &tensors, &admitted).unwrap();
                let document = crate::prepared::document(&graph, tp, &admitted)
                    .unwrap_or_else(|e| panic!("{} TP{}: {e}", geometry.id, tp.ranks()));
                crate::prepared::check_ports(&document, tp).unwrap();
                crate::prepared::check_mtp_ports(&document, tp).unwrap();
                crate::prepared::check_dflash_ports(&document, tp).unwrap();
            }
        }
    }
}
