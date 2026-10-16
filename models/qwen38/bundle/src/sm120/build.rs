pub(crate) mod projection_selection;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use kernel_build::{CubinSpec, EntryParam, EntrySpec, ModelLock, Toolchain};
use kernels_sm120::arithmetic::DeclaredCubin;
use model_qwen38_contract::sm120::geometry::{self, GeometryModel, GeometryShape as Qwen38Shape};

#[allow(dead_code)]
mod precision_selection;

/// The shared row copies' cubin (`kernels_sm120::row::copy`), which every
/// geometry loads.
const ROW_COPY_CUBIN: &str = "row_copy_sm120";

fn main() {
    println!("cargo:rerun-if-env-changed=CHOPINFER_QWEN38_RECIPE_TEST");
    if env::var("PROFILE").as_deref() == Ok("release")
        && env::var("CHOPINFER_QWEN38_RECIPE_TEST").as_deref() != Ok("1")
        && let Err(error) = geometry::require_base_geometries("release bundle")
    {
        eprintln!("{error}; recipe tests alone may set CHOPINFER_QWEN38_RECIPE_TEST=1");
        std::process::exit(1);
    }
    if let Err(error) = geometry::validate_catalogs() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    let (selection, ranks_of, geometry_of) = selection_inputs(&manifest);
    let selection = projection_selection::assemble(&selection, &geometry_of)
        .expect("projection cubin placement");
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    // Model-owned: `tools.model_sdk relock` redirects it in an export.
    let lock = ModelLock::from_env(manifest.join("kernels.lock")).expect("Qwen3.8 kernel lock");
    let built = selection
        .build(&toolchain, &out, lock.path(), lock.options())
        .expect("locked Qwen3.8 cubin build and audit");
    let context_parallel = context_parallel_cubins(&toolchain, &manifest, &out);
    let embedded = built
        .iter()
        .chain(context_parallel.iter().map(|(_, cubin)| cubin))
        .cloned()
        .collect::<Vec<_>>();
    kernel_build::write_embedded(&out, &embedded, |name| {
        ranks_of.get(name).copied().unwrap_or(4)
    })
    .expect("write embedded cubin list");
    let memberships = built
        .iter()
        .flat_map(|cubin| {
            // A geometry can reuse an encoder cubin emitted for an earlier geometry.
            if cubin
                .name
                .starts_with(&format!("{}_", kernels_sm120::encoder::ROWS_GEMM_CUBIN))
                || cubin
                    .name
                    .starts_with(&format!("{}_", kernels_sm120::encoder::ATTENTION_CUBIN))
            {
                return model_qwen38_contract::geometry::GEOMETRIES
                    .iter()
                    .filter(|g| {
                        model_qwen38_contract::encoder::kernels_for(g)
                            .expect("listed encoder geometry")
                            .iter()
                            .any(|entry| {
                                cubin
                                    .entries
                                    .iter()
                                    .any(|built| built.symbol == entry.symbol)
                            })
                    })
                    .map(|g| format!("({:?}, {:?}, 0),", cubin.name, g.id))
                    .collect::<Vec<_>>();
            }
            let id = if cubin.name == "comm_sm120"
                || cubin.name == ROW_COPY_CUBIN
                || cubin.name == precision_selection::SHAPE_FREE
                || cubin.name == kernels_sm120::encoder::ROWS_GEMM_CUBIN
                || cubin.name == kernels_sm120::encoder::ATTENTION_CUBIN
            {
                "*"
            } else {
                geometry_of[&cubin.name]
            };
            vec![format!(
                "({:?}, {:?}, {}),",
                cubin.name, id, ranks_of[&cubin.name]
            )]
        })
        .chain(
            context_parallel
                .iter()
                .map(|(id, cubin)| format!("({:?}, {id:?}, 4),", cubin.name)),
        )
        .collect::<String>();
    let features = geometry::MODELS
        .iter()
        .map(|owner| format!("({:?}, {}),", owner.geometry.id, owner.hc_precompute()))
        .collect::<String>();
    let f32_norms = f32_norm_cubins()
        .iter()
        .map(|name| format!("{name:?},"))
        .collect::<String>();
    std::fs::write(
        out.join("geometry-cubins.rs"),
        format!("pub const GEOMETRY_CUBINS: &[(&str, &str, u32)] = &[{memberships}];\npub const GEOMETRY_HC_PRECOMPUTE: &[(&str, bool)] = &[{features}];\npub const F32_NORM_CUBINS: &[&str] = &[{f32_norms}];\n"),
    )
    .expect("write geometry cubin membership");
    let publication = kernels_sm120::precision::publish_capability_document(
        &model_qwen38_contract::precision::capabilities().expect("Qwen operation capabilities"),
        &selection,
        &built,
    )
    .expect("audited Qwen operation publication");
    std::fs::write(out.join("precision-capabilities.json"), publication.bytes)
        .expect("write portable capabilities");
    std::fs::write(
        out.join("precision-dependency.txt"),
        publication.binding_dependency,
    )
    .expect("write native binding identity");
    let mut document = model_qwen38_contract::note::document();
    document["implementation_contract"] = serde_json::json!({
        "selection": publication.selection, "dependencies": publication.dependencies,
        "capabilities": publication.sha256,
    });
    let note = model_qwen38_contract::note::elf_note_document(&document);
    std::fs::write(out.join("preparation-note.bin"), note).expect("write native contract note");
}

/// The context-parallel cubins of every geometry that serves a size above
/// 1 at TP4 (`model_qwen38_contract::sm120::context_parallel::sizes`),
/// each with its geometry id. They gate on their own family of `kernels.lock.d` and build
/// in a child of OUT_DIR, so the served selection keeps its `kernels.lock`
/// and its report; each cubin is then copied beside the served ones for the
/// one embedded list. An instance loads them only at a size above 1.
fn context_parallel_cubins(
    toolchain: &Toolchain,
    manifest: &Path,
    out: &Path,
) -> Vec<(&'static str, kernel_build::BuiltCubin)> {
    use model_qwen38_contract::sm120::context_parallel;
    let mut specs = Vec::new();
    let mut geometry_of = BTreeMap::new();
    for g in model_qwen38_contract::geometry::GEOMETRIES {
        let shape = model_qwen38_contract::contract::Shape::new(
            g,
            model_qwen38_contract::contract::Ranks::TP4,
        );
        // A geometry without the entries serves size 1 only.
        if context_parallel::sizes(shape).is_empty() {
            continue;
        }
        for spec in context_parallel::cubin_specs(shape).expect("Qwen3.8 context-parallel cubins") {
            geometry_of.insert(spec.name.clone(), g.id);
            specs.push(spec);
        }
    }
    let model_lock = ModelLock::from_env(manifest.join("kernels.lock.d"))
        .expect("Qwen3.8 kernel lock directory");
    let lock = kernel_build::LockDir::new(model_lock.path()).family(
        context_parallel::FAMILY,
        specs.iter().map(|spec| spec.name.clone()),
    );
    let child = out.join(context_parallel::FAMILY);
    // The report names its consumer and source root, as the served
    // selection's does.
    let context = kernel_build::BuildContext::new(
        concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION")),
        "chopinfer",
        concat!(env!("CARGO_PKG_NAME"), ".context-parallel.v1"),
    )
    .source_root("chopinfer", manifest.join("../../.."));
    kernel_build::build_in_families(
        toolchain,
        &specs,
        &child,
        &lock,
        model_lock.options(),
        Some(&context),
    )
    .expect("locked Qwen3.8 context-parallel cubin build and audit")
    .into_iter()
    .map(|mut cubin| {
        let file = cubin.path.file_name().expect("a cubin file").to_owned();
        let beside = out.join(&file);
        std::fs::copy(&cubin.path, &beside).expect("copy a context-parallel cubin");
        cubin.path = beside;
        (geometry_of[&cubin.name], cubin)
    })
    .collect()
}

pub type SelectionInputs = (
    kernel_build::KernelSelection,
    BTreeMap<String, u32>,
    BTreeMap<String, &'static str>,
);

pub fn selection_inputs(manifest: &Path) -> SelectionInputs {
    let mut specs = Vec::new();
    let mut ranks_of = BTreeMap::new();
    let mut geometry_of = BTreeMap::new();
    for owner in geometry::MODELS {
        let (owned, ranks) = geometry_specs(manifest, &owner);
        for name in ranks.keys() {
            geometry_of.insert(name.clone(), owner.geometry.id);
        }
        specs.extend(owned);
        ranks_of.extend(ranks);
    }
    // The shared stream broadcast, once for every geometry and rank count:
    // its rows' width and streams are launch arguments.
    specs.extend(compiled([kernels_sm120::row::copy::cubin_spec(
        ROW_COPY_CUBIN,
        &[kernels_sm120::row::copy::Entry::new(
            kernels_sm120::row::copy::Pattern::BroadcastStreams,
            kernels_sm120::row::copy::BROADCAST_SYMBOL,
        )],
    )]));
    ranks_of.insert(ROW_COPY_CUBIN.into(), 0);
    geometry_of.insert(ROW_COPY_CUBIN.into(), geometry::MODELS[0].geometry.id);

    let mut context = kernel_build::BuildContext::new(
        concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION")),
        "chopinfer",
        concat!(env!("CARGO_PKG_NAME"), ".catalogs.v1"),
    )
    .source_root("chopinfer", manifest.join("../../.."));
    context.tuning = model_qwen38_contract::sm120::tuning::entry_status();
    let selection = precision_selection::with_legacy(context, specs)
        .expect("Qwen legacy and mixed operation selection");
    let mixed = precision_selection::mixed_ranks(&selection).expect("mixed module rank counts");
    for spec in selection.specs() {
        // A mixed module loads at the rank counts whose operations bind it.
        ranks_of
            .entry(spec.name.clone())
            .or_insert_with(|| *mixed.get(&spec.name).expect("a bound mixed module"));
        let id = model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .find(|g| {
                spec.name
                    .starts_with(&format!("{}_precision", g.kernel_tag))
            })
            .map_or(model_qwen38_contract::geometry::FLASH.id, |g| g.id);
        geometry_of.entry(spec.name.clone()).or_insert(id);
    }
    (selection, ranks_of, geometry_of)
}

fn geometry_specs(
    manifest: &Path,
    owner: &GeometryModel,
) -> (Vec<DeclaredCubin>, BTreeMap<String, u32>) {
    let profile = Some(owner.model.at(4));
    let mut specs = compiled(kernels_sm120::row::hyper_connection::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(compiled(
        kernels_sm120::row::cubin_specs(model_qwen38_contract::sm120::catalogs(), profile)
            .into_iter()
            .filter(|spec| !spec.entries.is_empty())
            .map(|mut spec| {
                if !owner.geometry.pinned_names {
                    spec.name = format!("row_{}_engine", owner.geometry.kernel_tag);
                }
                spec
            }),
    ));
    specs.extend(kernels_sm120::attention::sparse::declared_cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(compiled(kernels_sm120::linear::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    )));
    specs.extend(compiled(kernels_sm120::attention::cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    )));
    specs.extend(compiled(model_qwen38_contract::sm120::comm::cubin_specs(
        profile,
    )));
    // TP4's decode all-reduce; the other rank counts' follow in their own
    // cubins below.
    specs.extend(compiled(
        model_qwen38_contract::sm120::comm::decode::cubin_specs_for(profile, &[4]),
    ));
    // TP4's wide decode all-reduce for graph buckets above 32 rows
    // (`comm_qwen38_decode_wide`); the other rank counts' follow in cubins
    // named for their topology below.
    specs.extend(compiled(
        model_qwen38_contract::sm120::comm::decode::wide_cubin_specs_for(profile, &[4]),
    ));
    // The F32 norm-weight forms: the prefill all-reduce and the grouped norm
    // row at every rank count, TP4's decode and wide all-reduces.
    specs.extend(compiled(
        model_qwen38_contract::sm120::comm::f32_cubin_specs(profile),
    ));
    specs.extend(compiled(
        model_qwen38_contract::sm120::comm::decode::f32_cubin_specs_for(profile, &[4]),
    ));
    specs.extend(compiled(
        model_qwen38_contract::sm120::comm::decode::f32_wide_cubin_specs_for(profile, &[4]),
    ));
    specs.extend(compiled(
        model_qwen38_contract::sm120::row::f32_norm_cubin_specs(profile),
    ));
    if owner.geometry.id == geometry::MODELS[0].geometry.id {
        specs.extend(compiled([kernels_sm120::comm::cubin_spec()]));
    }
    specs.extend(kernels_sm120::gemm::declared_cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    specs.extend(kernels_sm120::gemm::fused::declared_cubin_specs(
        model_qwen38_contract::sm120::catalogs(),
        profile,
    ));
    // The vision encoder, generated from its declaration: its rows and
    // GEMMs, then its varlen attention.
    let registered = model_qwen38_contract::geometry::GEOMETRIES
        .iter()
        .find(|g| g.id == owner.geometry.id)
        .expect("listed encoder geometry");
    let entries =
        model_qwen38_contract::encoder::kernels_for(registered).expect("vision declaration");
    let entries = if registered.pinned_names {
        entries.to_vec()
    } else {
        // Keep each shared encoder symbol in the first geometry that emits it.
        let previous = model_qwen38_contract::geometry::GEOMETRIES
            .iter()
            .take_while(|g| g.id != registered.id)
            .flat_map(|g| {
                model_qwen38_contract::encoder::kernels_for(g).expect("listed encoder geometry")
            })
            .collect::<Vec<_>>();
        entries
            .iter()
            .filter(|entry| {
                if let Some(known) = previous.iter().find(|known| known.symbol == entry.symbol) {
                    kernels_sm120::encoder::union([(**known).clone(), (*entry).clone()])
                        .expect("shared encoder symbols have identical bodies");
                    false
                } else {
                    true
                }
            })
            .cloned()
            .collect()
    };
    let encoder_specs = kernels_sm120::encoder::cubin_specs(&entries).expect("encoder bodies");
    for mut spec in encoder_specs {
        if spec.entries.is_empty() {
            continue;
        }
        if !registered.pinned_names {
            spec.name = format!("{}_{}", spec.name, registered.kernel_tag);
        }
        specs.extend(compiled([spec]));
    }
    let moe =
        kernels_sm120::moe::declared_cubin_specs(model_qwen38_contract::sm120::catalogs(), profile);
    assert!(!moe.is_empty());
    specs.extend(moe);
    let tp4 = geometry::shape(owner.model.at(4)).expect("Qwen3.8's TP4 shapes");
    specs.extend(compiled([
        bundle_spec(manifest, tp4),
        speculation_spec(manifest, tp4),
    ]));
    // Every cubin above loads at each rank count. The per-rank shapes of the
    // other topologies follow in cubins of their own, loaded only at their
    // rank count, so the TP4 cubins keep their entries and bytes.
    let mut ranks_of = specs
        .iter()
        .map(|cubin| (cubin.spec().name.clone(), 0))
        .collect::<BTreeMap<_, _>>();
    ranks_of.insert(
        format!("chop_moe_{}_route_parallel", owner.geometry.kernel_tag),
        4,
    );
    for rank_profile in owner.model.other_profiles() {
        let ranks = rank_profile
            .ranks_of(owner.model)
            .expect("a Qwen3.8 profile");
        let profile = Some(rank_profile);
        let mut tp = compiled(kernels_sm120::row::hyper_connection::cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        ));
        tp.extend(kernels_sm120::attention::sparse::declared_cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        ));
        tp.extend(compiled(kernels_sm120::linear::cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        )));
        tp.extend(compiled(kernels_sm120::attention::cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        )));
        tp.extend(kernels_sm120::gemm::declared_cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        ));
        tp.extend(kernels_sm120::gemm::fused::declared_cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        ));
        tp.extend(kernels_sm120::moe::declared_cubin_specs(
            model_qwen38_contract::sm120::catalogs(),
            profile,
        ));
        tp.extend(compiled([
            prefix_spec(manifest, geometry::shape(rank_profile).unwrap()),
            rank_speculation_spec(manifest, geometry::shape(rank_profile).unwrap()),
        ]));
        let mut decode = model_qwen38_contract::sm120::comm::decode::cubin_specs_for(
            Some(owner.model.at(4)),
            &[ranks],
        );
        for spec in &mut decode {
            spec.name = format!("comm_{}_decode", rank_profile.tag());
        }
        tp.extend(compiled(decode));
        let mut wide = model_qwen38_contract::sm120::comm::decode::wide_cubin_specs_for(
            Some(owner.model.at(4)),
            &[ranks],
        );
        for spec in &mut wide {
            spec.name = format!("comm_{}_decode_wide", rank_profile.tag());
        }
        tp.extend(compiled(wide));
        let base = Some(owner.model.at(4));
        let mut f32_decode =
            model_qwen38_contract::sm120::comm::decode::f32_cubin_specs_for(base, &[ranks]);
        for spec in &mut f32_decode {
            spec.name = model_qwen38_contract::sm120::comm::decode_cubin(owner, ranks, true, false);
        }
        tp.extend(compiled(f32_decode));
        let mut f32_wide =
            model_qwen38_contract::sm120::comm::decode::f32_wide_cubin_specs_for(base, &[ranks]);
        for spec in &mut f32_wide {
            spec.name = model_qwen38_contract::sm120::comm::decode_cubin(owner, ranks, true, true);
        }
        tp.extend(compiled(f32_wide));
        for cubin in &tp {
            let spec = cubin.spec();
            assert!(!spec.entries.is_empty(), "{} has no entries", spec.name);
            assert!(
                ranks_of.insert(spec.name.clone(), ranks).is_none(),
                "{} is built twice",
                spec.name
            );
        }
        specs.extend(tp);
    }

    (specs, ranks_of)
}

/// The cubins of the F32 norm-weight forms, which an instance loads only for
/// an image that stores F32 norm weights: each geometry's prefill
/// all-reduce and grouped norm row, and its decode and wide all-reduces at
/// TP4 and at each other rank count.
fn f32_norm_cubins() -> Vec<String> {
    let forms = model_qwen38_contract::sm120::comm::norm_weight_forms();
    let mut names = forms
        .rows
        .iter()
        .map(|row| row.f32.cubin.clone())
        .collect::<Vec<_>>();
    for transport in forms.transports {
        names.extend([
            transport.prefill[1].cubin.to_owned(),
            transport.decode_cubins[1].clone(),
            transport.wide_cubins[1].clone(),
        ]);
    }
    names.sort();
    names.dedup();
    names
}

/// Cubins whose wrappers compile in all of their arithmetic.
fn compiled(specs: impl IntoIterator<Item = CubinSpec>) -> Vec<DeclaredCubin> {
    specs
        .into_iter()
        .map(|spec| DeclaredCubin::compiled(spec).expect("compiled Qwen3.8 cubin"))
        .collect()
}

/// The bundle-owned rows (`cuda/qwen38_bundle.cuh`) with `shape`'s defines.
fn bundle_spec(manifest: &Path, shape: Qwen38Shape) -> CubinSpec {
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    let u64 = |name: &str| EntryParam::U64 { name: name.into() };
    let entry = |role: &str, params, body: &str| {
        EntrySpec::new(
            format!("chop_{}_{role}_v1", shape.geometry.kernel_tag),
            params,
            format!("chop::qwen38_bundle::{body};"),
            256,
            1,
        )
    };
    CubinSpec {
        name: format!("{}_bundle", shape.geometry.kernel_tag),
        sources_root: manifest.join("../../.."),
        include_dirs: vec![
            PathBuf::from("models/qwen38/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec!["#include \"qwen38_bundle.cuh\"".into()],
        defines: shape.bundle_defines(),
        entries: vec![
            entry(
                "ple_residual",
                vec![
                    ptr("x", "const uint16_t"),
                    ptr("gated", "const uint16_t"),
                    ptr("conv", "const uint16_t"),
                    ptr("out", "uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "ple_residual(x, gated, conv, out, live_rows)",
            ),
            entry(
                "ple_stage",
                vec![
                    ptr("pool", "const uint8_t"),
                    ptr("staging", "uint8_t"),
                    ptr("slot_ids", "const int32_t"),
                    ptr("prefix_len", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                ],
                "ple_stage(pool, staging, slot_ids, prefix_len, live_items)",
            ),
            entry(
                "copy_entries",
                vec![
                    ptr("src", "const uint8_t"),
                    ptr("dst", "uint8_t"),
                    ptr("src_ids", "const int32_t"),
                    ptr("dst_ids", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                    u64("entry_bytes"),
                    u64("plane_stride"),
                ],
                "copy_entries(src, dst, src_ids, dst_ids, live_items, entry_bytes, plane_stride)",
            ),
            entry(
                "prefix_copy",
                vec![
                    ptr("conv_pool", "uint8_t"),
                    ptr("delta_pool", "uint8_t"),
                    ptr("ple_pool", "uint8_t"),
                    ptr("ring", "uint8_t"),
                    ptr("retained", "uint8_t"),
                    ptr("pool", "uint8_t"),
                    ptr("pages", "const uint64_t"),
                    u32("ring_layers"),
                    u64("page_bytes"),
                    u32("slot"),
                    u32("index"),
                    u32("to_pages"),
                    u32("pool_entries"),
                    u32("ring_slots"),
                ],
                "prefix_copy(conv_pool, delta_pool, ple_pool, ring, retained, pool, pages, \
                 ring_layers, page_bytes, slot, index, to_pages, pool_entries, ring_slots)",
            ),
            entry(
                "prefix_copy_indexed",
                vec![
                    ptr("conv_pool", "uint8_t"),
                    ptr("delta_pool", "uint8_t"),
                    ptr("ple_pool", "uint8_t"),
                    ptr("ring", "uint8_t"),
                    ptr("retained", "uint8_t"),
                    ptr("pool", "uint8_t"),
                    ptr("pages", "const uint64_t"),
                    u32("ring_layers"),
                    u64("page_bytes"),
                    u32("slot"),
                    u32("index"),
                    u32("to_pages"),
                    u32("pool_entries"),
                    u32("ring_slots"),
                    u32("ple_entry"),
                    u32("ple_entries"),
                ],
                "prefix_copy<true>(conv_pool, delta_pool, ple_pool, ring, retained, pool, pages, \
                 ring_layers, page_bytes, slot, index, to_pages, pool_entries, ring_slots, ple_entry, ple_entries)",
            ),
        ],
    }
}

/// The prefix-cache snapshot copy of a TP2 rank, whose linear state
/// entries differ from TP4's (`cuda/qwen38_bundle.cuh`); the bundle's other
/// rows do not depend on the rank count.
fn prefix_spec(manifest: &Path, shape: Qwen38Shape) -> CubinSpec {
    let tag = shape.profile().tag();
    let mut spec = bundle_spec(manifest, shape);
    spec.name = format!("{tag}_bundle");
    spec.entries.retain(|entry| {
        ["prefix_copy", "prefix_copy_indexed"]
            .iter()
            .any(|role| entry.symbol == format!("chop_{}_{role}_v1", shape.geometry.kernel_tag))
    });
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen(shape.geometry.kernel_tag, tag, 1);
    }
    spec
}

/// The speculation rows of a TP2 rank that depend on the rank count:
/// the n-gram verify rows (the rank's hash heads) and the draft argmax (the
/// vocabulary shard and the rank count). The others are TP4's.
fn rank_speculation_spec(manifest: &Path, shape: Qwen38Shape) -> CubinSpec {
    let tag = shape.profile().tag();
    let mut spec = speculation_spec(manifest, shape);
    spec.name = format!("{tag}_speculation");
    spec.entries.retain(|entry| {
        ["ngram_verify", "draft_argmax_partial", "draft_argmax_final"]
            .iter()
            .any(|name| entry.symbol == format!("chop_{}_{name}_v1", shape.geometry.kernel_tag))
    });
    for entry in &mut spec.entries {
        entry.symbol = entry.symbol.replacen(shape.geometry.kernel_tag, tag, 1);
    }
    spec
}

/// The bundle's speculation rows (`cuda/qwen38_speculation.cuh`). The verify
/// rows call the per-sequence n-gram and PLE convolution of the Qwen3.8 row
/// catalog, so the sources root spans both crates.
fn speculation_spec(manifest: &Path, shape: Qwen38Shape) -> CubinSpec {
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u64 = |name: &str| EntryParam::U64 { name: name.into() };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    let entry = |role: &str, params, body: &str| {
        EntrySpec::new(
            format!("chop_{}_{role}_v1", shape.geometry.kernel_tag),
            params,
            format!("chop::qwen38_bundle::{body};"),
            256,
            1,
        )
    };
    CubinSpec {
        name: format!("{}_speculation", shape.geometry.kernel_tag),
        sources_root: manifest.join("../../.."),
        include_dirs: vec![
            PathBuf::from("models/qwen38/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec!["#include \"qwen38_speculation.cuh\"".into()],
        defines: shape.speculation_defines(),
        entries: vec![
            entry(
                "ngram_verify",
                vec![
                    ptr("tokens", "const uint32_t"),
                    ptr("history", "const uint32_t"),
                    ptr("multipliers", "const uint64_t"),
                    ptr("sizes", "const uint64_t"),
                    ptr("offsets", "const uint64_t"),
                    ptr("table", "const uint8_t"),
                    ptr("scale", "const uint16_t"),
                    ptr("output", "uint16_t"),
                    ptr("row_ids", "uint64_t"),
                    ptr("history_out", "uint32_t"),
                    ptr("cu_rows", "const int32_t"),
                    ptr("q_len", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                    u32("rank"),
                    u64("table_start"),
                    u64("table_rows"),
                    u64("entry_words"),
                ],
                "ngram_verify(tokens, history, multipliers, sizes, offsets, table, scale, output, row_ids, history_out, cu_rows, q_len, live_items, rank, table_start, table_rows, entry_words)",
            ),
            entry(
                "ple_conv_verify",
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("weight", "const uint16_t"),
                    ptr("state", "const uint16_t"),
                    ptr("output", "uint16_t"),
                    ptr("state_out", "uint16_t"),
                    ptr("cu_rows", "const int32_t"),
                    ptr("q_len", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                    u64("entry"),
                ],
                "ple_conv_verify(input, weight, state, output, state_out, cu_rows, q_len, live_items, entry)",
            ),
            entry(
                "ple_checkpoints",
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("tokens", "const uint32_t"),
                    ptr("staged", "const uint8_t"),
                    ptr("checkpoints", "uint8_t"),
                    ptr("cu_rows", "const int32_t"),
                    ptr("q_len", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                ],
                "ple_checkpoints(input, tokens, staged, checkpoints, cu_rows, q_len, live_items)",
            ),
            entry(
                "ple_anchor",
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("tokens", "const uint32_t"),
                    ptr("staged", "const uint8_t"),
                    ptr("anchors", "uint8_t"),
                    ptr("cu_rows", "const int32_t"),
                    ptr("prefix_len", "const int32_t"),
                    ptr("anchor_slots", "const int32_t"),
                    ptr("anchor_ends", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                ],
                "ple_anchor(input, tokens, staged, anchors, cu_rows, prefix_len, anchor_slots, anchor_ends, live_items)",
            ),
            entry(
                "mtp_gather",
                vec![
                    ptr("last", "const uint16_t"),
                    ptr("retained", "const uint16_t"),
                    ptr("sources", "const int32_t"),
                    ptr("token_sources", "const int32_t"),
                    ptr("drafts", "const int32_t"),
                    ptr("x", "uint16_t"),
                    ptr("token_ids", "int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "mtp_gather(last, retained, sources, token_sources, drafts, x, token_ids, live_rows)",
            ),
            entry(
                "mtp_norms",
                vec![
                    ptr("x", "const uint16_t"),
                    ptr("hidden_weight", "const uint16_t"),
                    ptr("embed", "const uint16_t"),
                    ptr("embed_weight", "const uint16_t"),
                    ptr("x_normed", "uint16_t"),
                    ptr("embed_normed", "uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "mtp_norms(x, hidden_weight, embed, embed_weight, x_normed, embed_normed, live_rows)",
            ),
            entry(
                "mtp_combine",
                vec![
                    ptr("hidden", "const uint16_t"),
                    ptr("token", "const uint16_t"),
                    ptr("x", "uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "mtp_combine(hidden, token, x, live_rows)",
            ),
            entry(
                "draft_argmax_partial",
                vec![
                    ptr("logits", "const uint16_t"),
                    ptr("best", "uint32_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("rank"),
                ],
                "draft_argmax_partial(logits, best, live_rows, rank)",
            ),
            entry(
                "draft_argmax_final",
                vec![
                    ptr("bests", "const uint32_t"),
                    ptr("slots", "const int32_t"),
                    ptr("targets", "const int32_t"),
                    ptr("drafts", "int32_t"),
                    ptr("token_ids", "int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "draft_argmax_final(bests, slots, targets, drafts, token_ids, live_rows)",
            ),
            entry(
                "ple_checkpoints_indexed",
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("tokens", "const uint32_t"),
                    ptr("staged", "const uint8_t"),
                    ptr("checkpoints", "uint8_t"),
                    ptr("cu_rows", "const int32_t"),
                    ptr("q_len", "const int32_t"),
                    ptr("live_items", "const uint32_t"),
                    ptr("checkpoint_ids", "const int32_t"),
                    u64("pool_entries"),
                ],
                "ple_checkpoints_indexed(input, tokens, staged, checkpoints, cu_rows, q_len, live_items, checkpoint_ids, pool_entries)",
            ),
        ],
    }
}
