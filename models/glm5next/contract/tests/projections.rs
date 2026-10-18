//! The projection declaration against the generated tensor catalog, which
//! `catalog` pins to the checkpoint headers, and against the TP4 weight plan.
//! Only header facts are used; no payload is read.
use kernel_core::precision::ExecutionPhase;
use kernel_core::precision::format::FormatId;
use model_compiler_sdk::quantization::import::EncodingRecipe;
use model_compiler_sdk::quantization::projection::metadata::{HeaderCompanionRules, HeaderRole};
use model_glm5next_contract::catalog::{self, Group, Placement, TensorSpec};
use model_glm5next_contract::geometry::{FLASH, LayerKind, MlpKind};
use model_glm5next_contract::{projections, weight_plan};
use projection_spec::bind::{BoundDeclaration, CheckpointMetadata, TensorInfo};
use projection_spec::derive::Topology;
use projection_spec::{
    Declaration, Error, Geometry, Projection, ProjectionKind, SliceAxis, SplitClass,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

const RANKS: u32 = 4;

fn declaration() -> Declaration {
    projections::declaration(RANKS).unwrap()
}
fn geometry() -> Geometry {
    projections::geometry(RANKS).unwrap()
}

fn metadata() -> CheckpointMetadata {
    let config: Value = serde_json::from_str(include_str!("../fixtures/config.json")).unwrap();
    let tensors = catalog::catalog(&FLASH)
        .into_iter()
        .map(|t| {
            let info = TensorInfo {
                dtype: t.dtype.name().into(),
                shape: t.shape,
                file: None,
            };
            (t.name, info)
        })
        .collect();
    let mut m = CheckpointMetadata {
        tensors,
        quantization_config: config["quantization_config"].clone(),
        validated_owners: BTreeMap::new(),
    };
    attest_vision(&mut m);
    m
}

/// Vision tensors belong to the encoder declaration; until it lands, the
/// catalog's vision group stands in for its attestation.
fn attest_vision(m: &mut CheckpointMetadata) {
    let vision = catalog::catalog(&FLASH)
        .into_iter()
        .filter(|t| t.group == Group::Vision && m.tensors.contains_key(&t.name))
        .map(|t| t.name)
        .collect();
    m.validated_owners.insert("encoder-spec".into(), vision);
}

/// Expand a `{stem}` source over its stem's layers.
fn expand(d: &Declaration, g: &Geometry, source: &projection_spec::TensorSource) -> Vec<String> {
    match &source.stem {
        Some(id) => {
            let stem = d.layer_stems.iter().find(|s| s.id == *id).unwrap();
            stem.indices
                .resolve(g)
                .unwrap()
                .into_iter()
                .map(|i| {
                    let prefix = stem.template.replace("{i}", &i.to_string());
                    source.tensor.replace("{stem}", &prefix)
                })
                .collect()
        }
        None => vec![source.tensor.clone()],
    }
}

/// Weight source names of a projection, with operands and experts expanded.
fn sources(d: &Declaration, g: &Geometry, p: &Projection) -> Vec<String> {
    let source = p.source.as_ref().unwrap();
    let mut names = expand(d, g, source);
    if source.tensor.contains("{operand}") {
        names = names
            .into_iter()
            .flat_map(|n| p.operands.iter().map(move |o| n.replace("{operand}", o)))
            .collect();
    }
    if let ProjectionKind::ExpertBank { experts, .. } = &p.kind {
        let experts = experts.resolve(g).unwrap();
        names = names
            .into_iter()
            .flat_map(|n| (0..experts).map(move |e| n.replace("{expert}", &e.to_string())))
            .collect();
    }
    names
}

fn rules(d: &Declaration, g: &Geometry, m: &CheckpointMetadata) -> HeaderCompanionRules {
    let mut rules = HeaderCompanionRules::new(m, None).unwrap();
    // A BF16 source is an explicit role: the export's default recipe is FP8.
    for p in &d.projections {
        if sources(d, g, p)
            .iter()
            .all(|n| m.tensors.get(n).is_some_and(|t| t.dtype == "BF16"))
        {
            rules.roles.insert(
                p.id.clone(),
                HeaderRole {
                    declared: Some(EncodingRecipe::bf16()),
                    ..Default::default()
                },
            );
        }
    }
    rules
}

fn bind(m: &CheckpointMetadata) -> Result<BoundDeclaration, Error> {
    let (d, g) = (declaration(), geometry());
    d.bind(&g, &projections::registry().unwrap(), m, &rules(&d, &g, m))
}

/// The declaration role naming each catalog tensor.
#[derive(Clone, Debug)]
enum Role<'a> {
    Weight(&'a Projection),
    Scale(&'a Projection),
    Side(&'a projection_spec::SideTensor),
    External(&'a projection_spec::contracts::ExternalTensor),
}

fn roles<'a>(d: &'a Declaration, g: &Geometry) -> BTreeMap<String, Role<'a>> {
    let mut roles = BTreeMap::new();
    let mut claim = |name: String, role: Role<'a>| {
        if let Some(previous) = roles.insert(name.clone(), role) {
            panic!("{name} is claimed twice, first by {previous:?}");
        }
    };
    for p in &d.projections {
        for name in sources(d, g, p) {
            let scale = name.replace(".weight", ".weight_scale_inv");
            claim(name, Role::Weight(p));
            claim(scale, Role::Scale(p));
        }
    }
    for s in &d.side_tensors {
        for name in expand(d, g, &s.source) {
            claim(name, Role::Side(s));
        }
    }
    for e in &d.external_tensors {
        for name in expand(d, g, &e.source) {
            claim(name, Role::External(e));
        }
    }
    roles
}

/// A tensor family: the name with every index replaced by `{}`.
fn family(name: &str) -> String {
    name.split('.')
        .map(|part| {
            if part.bytes().all(|c| c.is_ascii_digit()) {
                "{}"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

#[test]
fn declaration_validates_at_tp4_only_and_pins_its_digest() {
    let d = declaration();
    assert_eq!(d.namespace, "glm5next");
    assert_eq!(
        d.digest(),
        include_str!("../fixtures/projections.sha256").trim()
    );
    for tp in [1, 2, 8] {
        assert!(projections::geometry(tp).is_err(), "TP{tp}");
    }
}

#[test]
fn stems_follow_the_geometry_layer_kinds() {
    let (d, g) = (declaration(), geometry());
    let indices = |id: &str| -> Vec<u64> {
        let stem = d.layer_stems.iter().find(|s| s.id == id).unwrap();
        stem.indices.resolve(&g).unwrap()
    };
    let main = 0..FLASH.layers();
    let mtp: Vec<u64> = (FLASH.layers()..FLASH.layers() + FLASH.mtp_layers())
        .map(u64::from)
        .collect();
    let select = |keep: &dyn Fn(u32) -> bool| -> Vec<u64> {
        main.clone().filter(|&i| keep(i)).map(u64::from).collect()
    };
    assert_eq!(
        indices("linear"),
        select(&|i| FLASH.layer_kind(i) == LayerKind::Linear)
    );
    assert_eq!(
        indices("latent"),
        select(&|i| FLASH.layer_kind(i) == LayerKind::Sparse)
    );
    assert_eq!(
        indices("dense-mlp"),
        select(&|i| FLASH.mlp_kind(i) == MlpKind::Dense)
    );
    assert_eq!(
        indices("moe"),
        select(&|i| FLASH.mlp_kind(i) == MlpKind::Moe)
    );
    assert_eq!(indices("layer"), select(&|_| true));
    for stem in ["mtp", "mtp-latent", "mtp-moe"] {
        assert_eq!(indices(stem), mtp, "{stem}");
    }
}

#[test]
fn binds_every_catalog_tensor_once_with_its_format() {
    let (d, g) = (declaration(), geometry());
    let m = metadata();
    let b = bind(&m).unwrap();
    let roles = roles(&d, &g);
    let catalog = catalog::catalog(&FLASH);
    let names: BTreeSet<_> = catalog.iter().map(|t| t.name.clone()).collect();
    let scales = roles
        .iter()
        .filter(|(n, r)| matches!(r, Role::Scale(_)) && !names.contains(*n))
        .map(|(n, _)| n.clone())
        .collect::<BTreeSet<_>>();
    // BF16 projections have no scale companion; every other name is a tensor.
    assert_eq!(
        roles.keys().cloned().collect::<BTreeSet<_>>(),
        &names | &scales
    );
    let mut table = BTreeMap::<(String, String), (String, String, u64)>::new();
    for t in &catalog {
        let role = match &roles[&t.name] {
            Role::Weight(p) => format!("projection {}", p.id),
            Role::Scale(p) => format!("projection {} scale", p.id),
            Role::Side(s) => format!("side {} ({:?})", s.id, s.kind),
            Role::External(e) => format!("external {} ({})", e.id, e.owner),
        };
        let entry = table
            .entry((family(&t.name), role))
            .or_insert_with(|| (t.dtype.name().into(), format!("{:?}", t.placement), 0));
        entry.2 += 1;
    }
    for ((family, role), (dtype, placement, count)) in &table {
        let placement = placement.split(" {").next().unwrap();
        println!("{count:>6} {dtype:<8} {placement:<10} {family} -> {role}");
    }

    let mut formats = BTreeMap::<&str, BTreeSet<FormatId>>::new();
    for o in &b.operands {
        formats.entry(&o.projection).or_default().insert(o.format);
    }
    for p in &d.projections {
        let fp8 = sources(&d, &g, p)
            .iter()
            .all(|n| m.tensors[n].dtype == "F8_E4M3");
        let expected = if fp8 {
            FormatId::Fp8Block128
        } else {
            FormatId::Bf16
        };
        assert_eq!(
            formats[p.id.as_str()],
            BTreeSet::from([expected]),
            "{}",
            p.id
        );
    }
    let fp8: BTreeSet<_> = formats
        .iter()
        .filter(|(_, f)| f.contains(&FormatId::Fp8Block128))
        .map(|(id, _)| *id)
        .collect();
    for id in [
        "latent.q-a",
        "latent.q-b",
        "latent.kv-a",
        "latent.output",
        "dense.gate-up",
        "dense.down",
        "moe.shared.gate-up",
        "moe.shared.down",
        "moe.experts.gate-up",
        "moe.experts.down",
        "mtp.latent.q-a",
        "mtp.moe.experts.gate-up",
    ] {
        assert!(fp8.contains(id), "{id}");
    }
    let f32_sides: BTreeSet<_> = b
        .side_tensors
        .iter()
        .filter(|n| m.tensors[*n].dtype == "F32")
        .map(|n| family(n))
        .collect();
    assert_eq!(
        f32_sides,
        [
            "self_attn.A_log",
            "self_attn.dt_bias",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_scale",
            "mlp.gate.e_score_correction_bias",
        ]
        .into_iter()
        .map(|s| format!("model.language_model.layers.{{}}.{s}"))
        .collect()
    );
}

#[test]
fn refuses_unknown_and_missing_tensors_by_name() {
    let mut m = metadata();
    let scale = "model.language_model.layers.3.self_attn.q_b_proj.weight_scale_inv";
    m.tensors.remove(scale);
    let e = bind(&m).unwrap_err().0;
    assert!(
        e.contains("layers.3.self_attn.q_b_proj.weight") && e.contains("missing companion"),
        "{e}"
    );

    let mut m = metadata();
    let conv = "model.language_model.layers.0.self_attn.v_conv1d.weight";
    m.tensors.remove(conv);
    assert!(bind(&m).unwrap_err().0.contains(conv));

    let mut m = metadata();
    m.tensors.insert(
        "model.language_model.layers.45.unexpected.weight".into(),
        TensorInfo {
            dtype: "BF16".into(),
            shape: vec![1, 1],
            file: None,
        },
    );
    assert!(bind(&m).unwrap_err().0.contains("unexpected.weight"));

    let mut m = metadata();
    let bias = "model.language_model.layers.10.mlp.gate.e_score_correction_bias";
    m.tensors.get_mut(bias).unwrap().dtype = "BF16".into();
    assert!(bind(&m).unwrap_err().0.contains(bias));
}

#[test]
fn a_text_only_checkpoint_omits_the_vision_owner() {
    let mut m = metadata();
    m.tensors
        .retain(|name, _| !name.starts_with("model.visual."));
    m.validated_owners.clear();
    bind(&m).unwrap();

    let mut m = metadata();
    m.validated_owners.clear();
    assert!(bind(&m).is_err());
}

#[test]
fn admission_follows_topology_and_format_support() {
    let (d, g) = (declaration(), geometry());
    let m = metadata();
    let b = bind(&m).unwrap();
    let admit = |projection, layer, ranks, format| {
        d.admit_checkpoint(
            &g,
            &b,
            projection_spec::Admission {
                projection,
                layer,
                ranks,
                phase: ExecutionPhase::Decode,
                format,
            },
        )
    };
    let fp8 = FormatId::Fp8Block128;
    admit("latent.q-b", Some(3), 4, fp8).unwrap();
    admit("moe.experts.gate-up", Some(3), 4, fp8).unwrap();
    admit("mtp.moe.experts.gate-up", Some(45), 4, fp8).unwrap();
    admit("linear.q", Some(0), 4, FormatId::Bf16).unwrap();
    admit("head", None, 4, FormatId::Bf16).unwrap();
    // TP4 only.
    assert!(admit("latent.q-b", Some(3), 2, fp8).is_err());
    assert!(admit("head", None, 8, FormatId::Bf16).is_err());
    // Layers outside a stem and undeclared formats are refused.
    assert!(admit("latent.q-b", Some(0), 4, fp8).is_err());
    assert!(admit("mtp.latent.q-a", Some(43), 4, fp8).is_err());
    assert!(admit("linear.q", Some(0), 4, FormatId::Nvfp4).is_err());
    // NVFP4 routed experts; MXFP8 experts only in the MTP layer.
    admit("moe.experts.down", Some(3), 4, FormatId::Nvfp4).unwrap();
    admit("mtp.moe.experts.down", Some(45), 4, FormatId::Mxfp8).unwrap();
    assert!(admit("moe.experts.down", Some(3), 4, FormatId::Mxfp8).is_err());
}

#[test]
fn gate_up_projections_declare_the_swiglu_clamp() {
    let d = declaration();
    let limit = 10.0_f32.to_bits();
    assert_eq!(geometry()["swiglu_limit_bits"], u64::from(limit));
    let gate_ups: Vec<_> = d
        .projections
        .iter()
        .filter(|p| p.epilogues.swiglu.is_some())
        .collect();
    assert_eq!(gate_ups.len(), 5);
    for p in gate_ups {
        let c = p.composition.as_ref().unwrap();
        assert!(!c.implementations.is_empty());
        assert_eq!(
            c.implementations.keys().collect::<Vec<_>>(),
            c.numerical_contracts.keys().collect::<Vec<_>>(),
            "{}",
            p.id
        );
        for (format, v) in &c.implementations {
            assert_eq!(
                v.numerical_contract,
                c.numerical_contracts[format].numerical_contract
            );
            assert_eq!(v.parameters["gate_max_bits"], "swiglu_limit_bits");
            assert_eq!(v.parameters["up_limit_bits"], "swiglu_limit_bits");
            let routed = matches!(p.kind, ProjectionKind::ExpertBank { .. });
            assert_eq!(
                v.parameters.get("route_weight").is_some(),
                routed,
                "{}",
                p.id
            );
        }
    }
    let g = geometry();
    let registry = projections::registry().unwrap();
    let p = d
        .projections
        .iter()
        .find(|p| p.id == "dense.gate-up")
        .unwrap();
    let v = &p.composition.as_ref().unwrap().implementations[&FormatId::Fp8Block128];
    for broken in ["unknown", "numerical_contract", "dtype", "limit"] {
        let mut v = v.clone();
        match broken {
            "unknown" => v.parameters["unexpected"] = Value::Bool(true),
            "numerical_contract" => v.numerical_contract = "glm5next.wrong.v1".into(),
            "dtype" => v.inputs[0].dtype = projection_spec::DType::F32,
            _ => v.parameters["gate_max_bits"] = f32::INFINITY.to_bits().into(),
        }
        assert!(
            registry
                .resolve(
                    "glm5next",
                    &v,
                    projection_spec::registry::Target::Projection,
                    &g
                )
                .is_err(),
            "{broken}"
        );
    }
}

/// Rows and K of `p` that each rank holds, through the shared derivation.
fn derived_slices(
    d: &Declaration,
    b: &BoundDeclaration,
    g: &Geometry,
    p: &Projection,
    rank: u32,
) -> Vec<projection_spec::derive::OperandSlice> {
    use kernel_core::contract::{ActivationFormat, DType};
    use kernel_core::precision::format::{A16Recipe, KernelFamily, registry};
    use kernel_core::precision::{ActivationSpec, ArithmeticSpec, Rounding};
    use model_glm5next_contract::backend::{Kernels, Selected};
    use projection_spec::derive::{
        GeometryMode, ImportedProjection, InstanceKey, ProjectionContext, SourceMatrix,
    };
    let first = b.operands.iter().find(|o| o.projection == p.id).unwrap();
    let mut operands: Vec<_> = b
        .operands
        .iter()
        .filter(|o| o.projection == p.id && o.layer == first.layer)
        .collect();
    operands.sort_by_key(|o| o.operand);
    let routed = matches!(p.kind, ProjectionKind::ExpertBank { .. });
    // A bank operand is the concatenation of its experts' bound weights.
    let weights: Vec<_> = if routed {
        (0..u64::from(p.operand_count()))
            .map(|operand| {
                let experts: Vec<_> = operands
                    .iter()
                    .filter(|o| o.operand.unwrap_or(0) == operand)
                    .map(
                        |o| model_compiler_sdk::quantization::import::ImportedOperand {
                            weight: o.imported.weight.clone(),
                            activation: Some(ActivationSpec::bf16()),
                            source_tensors: vec![o.imported.weight.codes.tensor.clone()],
                        },
                    )
                    .collect();
                model_compiler_sdk::quantization::bank::concatenate_experts(
                    &format!("{}.{operand}", p.id),
                    &experts,
                )
                .unwrap()
                .operand
                .weight
            })
            .collect()
    } else {
        operands.iter().map(|o| o.imported.weight.clone()).collect()
    };
    let recipe = p
        .composition
        .as_ref()
        .and_then(|c| c.numerical_contracts.get(&first.format));
    let a16 = recipe.map_or(A16Recipe::ScaledWeight, |r| r.a16_recipe);
    let family = match p.kind {
        ProjectionKind::LookupTable {} => KernelFamily::Lookup,
        ProjectionKind::ExpertBank { .. } => KernelFamily::GroupedExpert,
        ProjectionKind::Dense {} => KernelFamily::Dense,
    };
    let activation = ActivationFormat::Bf16;
    let imported = ImportedProjection {
        operation_id: p.id.clone(),
        target: Selected::TARGET,
        family,
        format: first.format,
        geometry: GeometryMode::RankLocal,
        activations: match family {
            KernelFamily::Lookup => Vec::new(),
            _ => vec![ActivationSpec::bf16(); weights.len()],
        },
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: p.uses[0].contract.as_ref().unwrap().output_dtype,
            decode_order: registry::by_id(first.format)
                .unwrap()
                .decode_order(family, activation, a16)
                .unwrap(),
            rounding: Rounding::NearestEven,
            numerical_contract: recipe
                .map_or(&p.numerical_contract, |r| &r.numerical_contract)
                .clone(),
        },
        a16_recipe: a16,
        matrices: operands
            .iter()
            .map(|o| (o, &o.imported.weight))
            .map(|(o, w)| SourceMatrix {
                weight: w.clone(),
                expert: o.expert.unwrap_or(0) as u32,
                operand: o.operand.unwrap_or(0) as u32,
                source_expert: 0,
                target_rows: 0..w.logical_shape[0],
                source_rows: 0..w.logical_shape[0],
                target_k: 0..w.logical_shape[1],
                source_k: 0..w.logical_shape[1],
            })
            .collect(),
        weights,
        preparation: None,
    };
    let usage = p
        .uses
        .iter()
        .find(|u| u.phase == ExecutionPhase::Decode)
        .unwrap();
    ProjectionContext {
        bound: b,
        source_instance: InstanceKey {
            projection: p.id.clone(),
            layer: first.layer,
        },
        selections: None,
        declaration: d,
        projection: p,
        geometry: g,
        topology: Topology { ranks: RANKS, rank },
        layer: first.layer,
        source_layer: first.layer,
        usage,
    }
    .ordinary(&imported)
    .unwrap_or_else(|e| panic!("{}/rank{rank}: {e}", p.id))
    .slices
}

/// The weight plan's slice of `t` on `rank`, as (start, shape) on the split
/// dimension, or `None` when the rank holds none of it.
fn planned(t: &TensorSpec, rank: u32) -> Option<(u64, Vec<u64>)> {
    weight_plan::rank_tensor(&FLASH, t, RANKS, rank)
        .unwrap()
        .map(|r| (r.start, r.shape))
}

#[test]
fn rank_slices_agree_with_the_weight_plan() {
    let (d, g) = (declaration(), geometry());
    let m = metadata();
    let b = bind(&m).unwrap();
    let d = d.expanded_profiles(&g).unwrap();
    let roles = roles(&d, &g);
    let catalog = catalog::catalog(&FLASH);
    let by_name: BTreeMap<_, _> = catalog.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut derived = BTreeSet::new();
    let mut derived_banks = BTreeSet::new();
    for p in &d.projections {
        let source = sources(&d, &g, p)[0].clone();
        let t = by_name[source.as_str()];
        let expected = match (&p.split, t.placement) {
            (SplitClass::Replicate {}, Placement::Replicated) => continue,
            (SplitClass::ExpertSharded { axis, .. }, Placement::Expert { dim, .. }) => {
                // Every rank's slice of every expert, against the derived
                // bank: the gate and up rows, the down columns.
                let along = match axis {
                    SliceAxis::Rows => 0,
                    SliceAxis::K => 1,
                };
                assert_eq!(dim, along, "{}", p.id);
                if p.composition
                    .as_ref()
                    .is_some_and(|c| !c.implementations.is_empty())
                {
                    continue;
                }
                for rank in 0..RANKS {
                    let slices = derived_slices(&d, &b, &g, p, rank);
                    assert_eq!(
                        slices.len(),
                        FLASH.shape.moe.experts as usize,
                        "{} rank {rank}",
                        p.id
                    );
                    for slice in &slices {
                        let expert = slice.source.expert;
                        let name = source.replace(".experts.0.", &format!(".experts.{expert}."));
                        let (start, shape) = planned(by_name[name.as_str()], rank).unwrap();
                        assert_eq!(slice.rows.len(), 1, "{}", p.id);
                        let (live, width) = match dim {
                            0 => (slice.rows[0].clone(), slice.k.end - slice.k.start),
                            _ => (slice.k.clone(), slice.rows[0].end - slice.rows[0].start),
                        };
                        assert_eq!(
                            (live.start, live.end - live.start, width),
                            (start, shape[dim], shape[1 - dim]),
                            "{} rank {rank} expert {expert}",
                            p.id
                        );
                    }
                    derived_banks.insert(p.id.clone());
                }
                continue;
            }
            (SplitClass::Row { .. }, Placement::Split { dim: 1 }) => 1,
            (
                SplitClass::Column { .. }
                | SplitClass::HeadGrouped { .. }
                | SplitClass::VocabSharded { .. },
                Placement::Split { dim: 0 },
            ) => 0,
            (split, placement) => panic!("{}: {split:?} against {placement:?}", p.id),
        };
        let implemented = p
            .composition
            .as_ref()
            .is_some_and(|c| !c.implementations.is_empty());
        for rank in 0..RANKS {
            let (start, shape) = planned(t, rank).unwrap();
            if implemented {
                // The clamped SwiGLU registers no derivation hook yet; its
                // split is the ordinary column split checked for `dense.down`.
                assert!(matches!(p.split, SplitClass::Column { .. }), "{}", p.id);
                continue;
            }
            for slice in derived_slices(&d, &b, &g, p, rank) {
                let (range, k) = (&slice.rows, &slice.k);
                assert_eq!(range.len(), 1, "{}", p.id);
                let (live, width) = if expected == 0 {
                    (range[0].clone(), k.end - k.start)
                } else {
                    (k.clone(), range[0].end - range[0].start)
                };
                assert_eq!(live.start, start, "{} rank {rank}", p.id);
                assert_eq!(
                    live.end - live.start,
                    shape[expected],
                    "{} rank {rank}",
                    p.id
                );
                assert_eq!(width, shape[1 - expected], "{} rank {rank}", p.id);
            }
            derived.insert(p.id.clone());
        }
    }
    for id in [
        "embedding",
        "head",
        "linear.q",
        "linear.beta",
        "linear.forget-b",
        "linear.output",
        "latent.q-b",
        "latent.output",
        "dense.down",
        "moe.shared.down",
        "mtp.latent.output",
    ] {
        assert!(derived.contains(id), "{id}");
    }

    // Both down banks derive the rank's slice of every expert; the gate-up
    // banks wait for the clamped SwiGLU hook.
    assert_eq!(
        derived_banks,
        BTreeSet::from(["moe.experts.down".into(), "mtp.moe.experts.down".into()])
    );

    let mut sides = 0;
    for (name, role) in &roles {
        let Role::Side(s) = role else { continue };
        let t = by_name[name.as_str()];
        for rank in 0..RANKS {
            let selection = s.selection(&g, Topology { ranks: RANKS, rank }).unwrap();
            let (start, shape) = planned(t, rank).unwrap();
            assert_eq!(selection.shape, shape, "{name} rank {rank}");
            match t.placement {
                Placement::Replicated => assert_eq!(selection.axis, Some(0), "{name}"),
                Placement::Split { dim } => {
                    assert_eq!(selection.axis, Some(dim), "{name}");
                    assert_eq!(selection.ranges.len(), 1, "{name}");
                    assert_eq!(selection.ranges[0].start, start, "{name} rank {rank}");
                }
                Placement::Expert { .. } => panic!("{name}: side tensors are not experts"),
            }
        }
        sides += 1;
    }
    assert!(sides > 0);
}

/// With `CHOPINFER_GLM5NEXT_CHECKPOINT` set, bind the declaration to the
/// checkpoint's own headers and quantization config.
#[test]
fn binds_the_checkpoint_headers() {
    let Some(root) = std::env::var_os("CHOPINFER_GLM5NEXT_CHECKPOINT") else {
        eprintln!("skipped: CHOPINFER_GLM5NEXT_CHECKPOINT is not set");
        return;
    };
    let root = std::path::Path::new(&root);
    let config: Value =
        serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let shards: BTreeSet<_> = index["weight_map"]
        .as_object()
        .unwrap()
        .values()
        .map(|f| f.as_str().unwrap().to_owned())
        .collect();
    let headers = shards
        .into_iter()
        .map(|shard| {
            use std::io::Read;
            let mut file = std::fs::File::open(root.join(&shard)).unwrap();
            let mut length = [0; 8];
            file.read_exact(&mut length).unwrap();
            let length = u64::from_le_bytes(length);
            assert!(length <= 64 << 20, "{shard}: header of {length} bytes");
            let mut bytes = vec![0; length as usize];
            file.read_exact(&mut bytes).unwrap();
            (shard, serde_json::from_slice(&bytes).unwrap())
        })
        .collect();
    let mut m = CheckpointMetadata::from_headers(
        &headers,
        Some(&index),
        config["quantization_config"].clone(),
    )
    .unwrap();
    assert_eq!(m.tensors.len(), catalog::catalog(&FLASH).len());
    attest_vision(&mut m);
    bind(&m).unwrap();
}

/// A metadata-only checkpoint of the NVFP4-routed mix, shaped as the plan
/// reads its headers: NVFP4 routed experts (E2M1 pairs, an E4M3
/// scale per 16 and an F32 global, with an ignored input scale), MXFP8 MTP
/// experts (an E8M0 scale per 32) and BF16 everywhere else. The real headers
/// are checked when the export is admitted.
fn mixed_nvfp4_metadata() -> CheckpointMetadata {
    let mut m = metadata();
    let mut layers = serde_json::Map::new();
    let mtp = format!("model.language_model.layers.{}.", FLASH.layers());
    for t in catalog::catalog(&FLASH) {
        if t.dtype != catalog::DType::F8E4M3 {
            continue;
        }
        let stem = t.name.strip_suffix(".weight").unwrap();
        m.tensors.remove(&format!("{stem}.weight_scale_inv"));
        let (rows, k) = (t.shape[0], t.shape[1]);
        let info = |dtype: &str, shape: Vec<u64>| TensorInfo {
            dtype: dtype.into(),
            shape,
            file: None,
        };
        let Placement::Expert { .. } = t.placement else {
            m.tensors
                .insert(t.name.clone(), info("BF16", t.shape.clone()));
            continue;
        };
        let experts = stem[..stem.find(".experts.").unwrap() + ".experts".len()].to_owned();
        if t.name.starts_with(&mtp) {
            m.tensors.insert(
                format!("{stem}.weight_scale"),
                info("U8", vec![rows, k / 32]),
            );
            layers.insert(
                experts,
                serde_json::json!({"quant_algo": "MXFP8", "group_size": 32}),
            );
        } else {
            m.tensors
                .insert(t.name.clone(), info("U8", vec![rows, k / 2]));
            m.tensors.insert(
                format!("{stem}.weight_scale"),
                info("F8_E4M3", vec![rows, k / 16]),
            );
            m.tensors
                .insert(format!("{stem}.weight_scale_2"), info("F32", Vec::new()));
            m.tensors
                .insert(format!("{stem}.input_scale"), info("F32", Vec::new()));
            layers.insert(
                experts,
                serde_json::json!({"quant_algo": "NVFP4", "group_size": 16}),
            );
        }
    }
    m.quantization_config = serde_json::json!({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "kv_cache_quant_algo": null,
        "ignore": ["lm_head", "model.visual*"],
        "quantized_layers": layers,
    });
    m
}

#[test]
fn binds_nvfp4_routed_and_mxfp8_mtp_experts_by_data() {
    let m = mixed_nvfp4_metadata();
    let b = bind(&m).unwrap();
    let mut formats = BTreeMap::<&str, BTreeSet<FormatId>>::new();
    for o in &b.operands {
        formats.entry(&o.projection).or_default().insert(o.format);
    }
    for (id, format) in formats {
        let expected = if id.starts_with("mtp.moe.experts.") {
            FormatId::Mxfp8
        } else if id.starts_with("moe.experts.") {
            FormatId::Nvfp4
        } else {
            FormatId::Bf16
        };
        assert_eq!(format, BTreeSet::from([expected]), "{id}");
    }
}

#[test]
fn use_buckets_follow_the_serving_contract_limits() {
    use model_glm5next_contract::contract::{
        DECODE_BATCH, MAX_VERIFY_ROWS, PREFILL_TOKENS, SCORED_ROWS,
    };
    let d = declaration();
    for (profile, uses) in &d.use_profiles {
        // The head scores at most a prefill chunk's scored rows.
        let prefill = if profile == "head-f32" {
            SCORED_ROWS
        } else {
            PREFILL_TOKENS
        };
        let capacities = |phase| -> Vec<u64> {
            uses.iter()
                .filter(|u| u.phase == phase)
                .map(|u| u.capacity.resolve(&geometry()).unwrap())
                .collect()
        };
        assert_eq!(capacities(ExecutionPhase::Prefill), [u64::from(prefill)]);
        assert_eq!(
            capacities(ExecutionPhase::Decode).last(),
            Some(&u64::from(DECODE_BATCH))
        );
        let rows = u64::from(DECODE_BATCH * MAX_VERIFY_ROWS);
        assert_eq!(
            capacities(ExecutionPhase::Verify).last(),
            Some(&rows),
            "{profile}"
        );
        for u in uses.iter().filter(|u| u.phase == ExecutionPhase::Verify) {
            let c = u.contract.as_ref().unwrap();
            let limit = |d: &Option<projection_spec::Dim>| d.as_ref().unwrap().resolve(&geometry());
            assert_eq!(
                limit(&c.per_item_verify).unwrap(),
                u64::from(MAX_VERIFY_ROWS)
            );
            assert_eq!(limit(&c.aggregate_verify).unwrap(), rows);
        }
    }
}

/// Every projection ID the rank views bind names a declaration that claims
/// that tensor at its layer, and every declared projection is bound.
#[test]
fn rank_view_bindings_name_the_declared_projections() {
    use model_glm5next_contract::contract::Shape;
    use model_glm5next_contract::views::{Binding, rank_views};
    let (d, g) = (declaration(), geometry());
    let roles = roles(&d, &g);
    let mut bound = BTreeSet::new();
    for rank in 0..RANKS {
        for view in rank_views(Shape::TP4, rank).unwrap() {
            let id = match view.binding {
                Binding::Projection(id) => id,
                Binding::Embedding => "embedding",
                Binding::Side(_) => continue,
            };
            match &roles[&view.name] {
                Role::Weight(p) | Role::Scale(p) => assert_eq!(p.id, id, "{}", view.name),
                // The pooling gate waits for a projection source without
                // a `.weight` suffix (see the declaration).
                Role::Side(s) if id.ends_with("indexer.pool-gate") => {
                    assert_eq!(s.id, id, "{}", view.name)
                }
                role => panic!("{}: {id} binds {role:?}", view.name),
            }
            bound.insert(id);
        }
    }
    let declared: BTreeSet<_> = d.projections.iter().map(|p| p.id.as_str()).collect();
    let pending: BTreeSet<_> = ["latent.indexer.pool-gate", "mtp.latent.indexer.pool-gate"]
        .into_iter()
        .collect();
    assert_eq!(bound, &declared | &pending);
}
