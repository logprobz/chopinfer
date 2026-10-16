pub(crate) mod projection_selection;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};

use kernel_build::{CubinSpec, EntryParam, EntrySpec, KernelSelection, ModelLock, Toolchain};
use kernels_sm120::arithmetic::DeclaredCubin;
use model_mimo26_contract::sm120::geometry::{self, GeometryModel};

use model_mimo26_contract::sm120::config as mimo26;

fn main() {
    println!("cargo:rerun-if-env-changed=CHOPINFER_MIMO26_RECIPE_TEST");
    if env::var("PROFILE").as_deref() == Ok("release")
        && env::var("CHOPINFER_MIMO26_RECIPE_TEST").as_deref() != Ok("1")
        && let Err(error) = geometry::require_base_geometries("release bundle")
    {
        eprintln!("{error}; recipe tests alone may set CHOPINFER_MIMO26_RECIPE_TEST=1");
        std::process::exit(1);
    }

    if let Err(error) = geometry::validate_catalogs() {
        eprintln!("{error}");
        std::process::exit(1);
    }
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("output directory"));
    let (selection, topologies, geometries) = selection_inputs(&manifest);
    let selection = projection_selection::assemble(&selection, &geometries)
        .expect("projection cubin placement");
    // Model-owned: `tools.model_sdk relock` redirects it in an export.
    let lock = ModelLock::from_env(manifest.join("kernels.lock")).expect("MiMo kernel lock");
    let built = selection
        .build(
            &Toolchain::detect().expect("CUDA 13.2 toolchain"),
            &out,
            lock.path(),
            lock.options(),
        )
        .expect("locked MiMo cubin build and audit");
    let identity = selection
        .audited_contract_identity(&built)
        .expect("MiMo kernel contract identity");
    let dependencies = selection
        .audited_dependency_identity(&built)
        .expect("MiMo kernel dependency identity");
    // The native note publishes both as the module's implementation
    // contract.
    std::fs::write(
        out.join("kernel-contract-note.bin"),
        model_mimo26_contract::note::elf_note(&identity, &dependencies),
    )
    .expect("write MiMo native contract note");
    std::fs::write(out.join("kernel-contract-identity.txt"), identity)
        .expect("write MiMo kernel contract identity");
    // Shared cubins record rank count zero.
    kernel_build::write_embedded(&out, &built, |name| {
        topologies.get(name).copied().unwrap_or(0)
    })
    .expect("write embedded cubin list");
    let members = built
        .iter()
        .flat_map(|cubin| {
            let symbols: Vec<_> = cubin.entries.iter().map(|e| e.symbol.as_str()).collect();
            let members = model_mimo26_contract::sm120::encoders::cubin_members(&symbols)
                .expect("encoder membership");
            if !members.is_empty() {
                return members
                    .into_iter()
                    .map(|id| format!("({:?}, {:?}, 0),", cubin.name, id))
                    .collect::<Vec<_>>();
            }
            vec![format!(
                "({:?}, {:?}, {}),",
                cubin.name,
                geometries[&cubin.name],
                topologies.get(&cubin.name).copied().unwrap_or(0)
            )]
        })
        .collect::<String>();
    std::fs::write(
        out.join("geometry-cubins.rs"),
        format!("pub const GEOMETRY_CUBINS: &[(&str, &str, u32)] = &[{members}];\n"),
    )
    .expect("write geometry cubin membership");
}

/// Cubins whose wrappers compile in all of their arithmetic.
/// The shared row copy cubin: built once with the first geometry's cubins,
/// loaded at every geometry.
const ROW_COPY_CUBIN: &str = "row_copy_sm120";

fn compiled(specs: impl IntoIterator<Item = CubinSpec>) -> Vec<DeclaredCubin> {
    specs
        .into_iter()
        .map(|spec| DeclaredCubin::compiled(spec).expect("compiled MiMo cubin"))
        .collect()
}

pub type SelectionInputs = (
    kernel_build::KernelSelection,
    BTreeMap<String, u32>,
    BTreeMap<String, &'static str>,
);

pub fn selection_inputs(manifest: &Path) -> SelectionInputs {
    let mut specs = Vec::<DeclaredCubin>::new();
    let mut encoder_entries = BTreeMap::new();
    let mut topologies = BTreeMap::new();
    let mut geometries = BTreeMap::new();
    for owner in geometry::MODELS {
        let (owned, ranks) = geometry_specs(manifest, *owner);
        for name in ranks.keys() {
            geometries.insert(name.clone(), owner.geometry.id);
        }
        for cubin in owned {
            let symbols: Vec<_> = cubin
                .spec()
                .entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect();
            if !model_mimo26_contract::sm120::encoders::cubin_members(&symbols)
                .expect("encoder membership")
                .is_empty()
            {
                let mut spec = cubin.into_spec();
                model_mimo26_contract::sm120::encoders::deduplicate(
                    &mut spec.entries,
                    &mut encoder_entries,
                    |entry| &entry.symbol,
                );
                if !spec.entries.is_empty() {
                    specs.extend(compiled([spec]));
                }
            } else {
                specs.push(cubin);
            }
        }
        topologies.extend(ranks);
    }
    geometries.insert("comm_sm120".into(), "*");
    geometries.insert("comm_decode_v2_sm120".into(), "*");
    geometries.insert(ROW_COPY_CUBIN.into(), "*");
    geometries.insert(
        kernels_sm120::gemm::mid::cubin_name(Some(mimo26::PROFILE)),
        "*",
    );

    let mut context = kernel_build::BuildContext::new(
        concat!(env!("CARGO_PKG_NAME"), "@", env!("CARGO_PKG_VERSION")),
        "chopinfer",
        concat!(env!("CARGO_PKG_NAME"), ".catalogs.v1"),
    )
    .source_root("chopinfer", manifest.join("../../.."));
    context.tuning = model_mimo26_contract::sm120::tuning::entry_status();
    // Every entry declares its wrapper and, for catalog entries, the
    // arithmetic its launch arguments and Variant fields decide, as the
    // producer of its cubin states it; the build report records them and
    // the contract identity hashes them.
    let cubins = specs
        .into_iter()
        .map(|cubin| {
            cubin
                .select(
                    "mimo26.legacy-program",
                    "mimo26.compiled-entry.v1",
                    "MiMo kernel-lab operations and fake-host launch journals",
                )
                .expect("declared MiMo cubin")
        })
        .collect();
    let selection = KernelSelection::new(context, cubins).expect("MiMo kernel selection");
    (selection, topologies, geometries)
}

fn geometry_specs(
    manifest: &Path,
    owner: GeometryModel,
) -> (Vec<DeclaredCubin>, BTreeMap<String, u32>) {
    let common = owner.geometry.id == geometry::MODELS[0].geometry.id;
    let mut specs = compiled(kernels_sm120::attention::cubin_specs(
        model_mimo26_contract::sm120::catalogs(),
        Some(owner.profile(4)),
    ));
    let mut row = kernels_sm120::row::cubin_specs(
        model_mimo26_contract::sm120::catalogs(),
        Some(owner.profile(4)),
    )
    .into_iter()
    .next()
    .expect("row cubin");
    assert!(!row.entries.is_empty());
    row.name = format!("{}_row", owner.geometry.kernel_tag);
    specs.extend(compiled([row]));
    // The encoders' rows, GEMMs and attention, generated from MiMo's encoder
    // declarations.
    if owner.geometry.encoders {
        let execution = model_mimo26_contract::geometry::GEOMETRIES
            .iter()
            .find(|g| g.id == owner.geometry.id)
            .expect("registered encoder geometry");
        let mut encoders = kernels_sm120::encoder::cubin_specs(
            model_mimo26_contract::sm120::encoders::kernels_for(execution)
                .expect("MiMo encoder declarations"),
        )
        .expect("MiMo encoder entries have bodies");
        if !common {
            for cubin in &mut encoders {
                cubin.name = format!("{}_{}", execution.kernel_tag, cubin.name);
            }
        }
        specs.extend(compiled(encoders));
        // MiMo's own encoder row kernels (registered stages its declarations run).
        specs.extend(compiled(encoder_rows_spec(manifest, execution)));
    }
    specs.extend(kernels_sm120::gemm::declared_cubin_specs(
        model_mimo26_contract::sm120::catalogs(),
        Some(owner.profile(4)),
    ));
    // The medium-M kernels are shape-generic: one shared cubin serves the
    // selected entries of every topology.
    if common {
        specs.extend(kernels_sm120::gemm::mid::declared_cubin_spec_for(
            model_mimo26_contract::sm120::catalogs(),
            &kernels_sm120::gemm::mid::cubin_name(Some(owner.profile(4))),
            &model_mimo26_contract::sm120::PROFILES,
        ));
    }
    let moe = kernels_sm120::moe::declared_cubin_specs(
        model_mimo26_contract::sm120::catalogs(),
        Some(owner.profile(4)),
    );
    assert!(!moe.is_empty());
    specs.extend(moe);
    if common {
        specs.extend(compiled([
            kernels_sm120::comm::cubin_spec(),
            kernels_sm120::comm::decode_cubin_spec_v2(),
            // The shared column slice, which MTP's EH shards launch at
            // every topology (`crate::sm120::row_copy`).
            kernels_sm120::row::copy::cubin_spec(
                ROW_COPY_CUBIN,
                &[kernels_sm120::row::copy::Entry::new(
                    kernels_sm120::row::copy::Pattern::ColumnSlice,
                    kernels_sm120::row::copy::SLICE_SYMBOL,
                )],
            ),
        ]));
    }
    specs.extend(compiled([
        model_mimo26_contract::sm120::comm::verify_for(owner),
        ring_spec(manifest, owner, 4),
        ring_slots_spec(manifest, owner, 4),
        dflash_spec(manifest, owner, 4),
        retain_spec(manifest, owner),
    ]));
    // The topology whose instances load each cubin; unlisted cubins hold
    // kernels every topology runs (the GEMM and row cubins named for TP4
    // also hold the router, norms, embedding and routing rows).
    let mut topologies = specs
        .iter()
        .map(DeclaredCubin::spec)
        .filter(|spec| {
            let tag = owner.geometry.kernel_tag;
            [
                format!("attention_{tag}_sm120"),
                format!("{tag}_ring"),
                format!("{tag}_ring_slots"),
                format!("{tag}_dflash"),
            ]
            .contains(&spec.name)
                || spec.name.starts_with(&format!("chop_moe_{tag}_gateup_"))
                || spec.name.starts_with(&format!("chop_moe_{tag}_down_"))
        })
        .map(|spec| (spec.name.clone(), 4_u32))
        .collect::<BTreeMap<_, _>>();
    // The per-rank shapes of the other topologies, in cubins of their own so
    // the TP4 cubins keep their entries.
    for profile in owner.profiles().into_iter().skip(1) {
        let first = specs.len();
        let ranks = profile.ranks_of(owner.model).expect("a MiMo profile");
        specs.extend(compiled(kernels_sm120::attention::cubin_specs(
            model_mimo26_contract::sm120::catalogs(),
            Some(profile),
        )));
        let mut row = kernels_sm120::row::cubin_specs(
            model_mimo26_contract::sm120::catalogs(),
            Some(profile),
        )
        .into_iter()
        .next()
        .expect("row cubin");
        assert!(!row.entries.is_empty());
        row.name = format!("{}_row", profile.tag());
        specs.extend(compiled([row]));
        specs.extend(kernels_sm120::gemm::declared_cubin_specs(
            model_mimo26_contract::sm120::catalogs(),
            Some(profile),
        ));
        let moe = kernels_sm120::moe::declared_cubin_specs(
            model_mimo26_contract::sm120::catalogs(),
            Some(profile),
        );
        assert!(!moe.is_empty());
        specs.extend(moe);
        specs.extend(compiled([
            ring_spec(manifest, owner, ranks),
            ring_slots_spec(manifest, owner, ranks),
            dflash_spec(manifest, owner, ranks),
        ]));
        for cubin in &specs[first..] {
            topologies.insert(cubin.spec().name.clone(), ranks);
        }
    }

    for cubin in &specs {
        topologies.entry(cubin.spec().name.clone()).or_insert(0);
    }
    (specs, topologies)
}

fn dflash_kv_heads(owner: GeometryModel, ranks: u32) -> u32 {
    owner
        .rank(ranks)
        .attention(geometry::dflash(owner.geometry))
        .kv_heads
}
fn ring_threads(owner: GeometryModel, ranks: u32) -> u32 {
    mimo26::ring_threads_for(owner.geometry, owner.rank(ranks))
}

/// The DFlash drafter's bundle-owned row kernels (`cuda/mimo_dflash.cuh`).
fn dflash_spec(manifest: &Path, owner: GeometryModel, ranks: u32) -> CubinSpec {
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    let tag = owner.profile(ranks).tag();
    CubinSpec {
        name: format!("{tag}_dflash"),
        sources_root: manifest.join("../../.."),
        include_dirs: vec![
            PathBuf::from("models/mimo26/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec!["#include \"mimo_dflash.cuh\"".into()],
        defines: mimo26::dflash_defines_for(owner.geometry, owner.rank(ranks)),
        entries: vec![
            EntrySpec::new(
                format!("chop_{tag}_dflash_copy_rows_v1"),
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("output", "uint16_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("width"),
                    u32("input_stride"),
                    u32("output_stride"),
                    u32("scale_bits"),
                ],
                "chop::mimo_dflash::copy_rows(input, output, live_rows, width, input_stride, output_stride, scale_bits);".into(),
                mimo26::ROW_THREADS,
                1,
            ),
            EntrySpec::new(
                format!("chop_{tag}_dflash_mask_rows_v1"),
                vec![
                    ptr("activation", "uint16_t"),
                    ptr("mask", "const uint16_t"),
                    ptr("positions", "const int32_t"),
                    ptr("prefix_len", "const int32_t"),
                    ptr("token_batch", "const int32_t"),
                    ptr("live_rows", "const uint32_t"),
                ],
                "chop::mimo_dflash::mask_rows(activation, mask, positions, prefix_len, token_batch, live_rows);".into(),
                mimo26::ROW_THREADS,
                1,
            ),
            EntrySpec::new(
                format!("chop_{tag}_dflash_cache_write_v1"),
                vec![
                    ptr("k", "const uint16_t"),
                    ptr("v", "const uint16_t"),
                    ptr("k_cache", "uint16_t"),
                    ptr("v_cache", "uint16_t"),
                    ptr("slot_ids", "const int32_t"),
                    ptr("token_batch", "const int32_t"),
                    ptr("positions", "const int32_t"),
                    ptr("last_positions", "const int64_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("k_stride"),
                    u32("v_stride"),
                    u32("v_scale_bits"),
                ],
                "chop::mimo_dflash::cache_write(k, v, k_cache, v_cache, slot_ids, token_batch, positions, last_positions, live_rows, k_stride, v_stride, v_scale_bits);".into(),
                // A thread per 16 bytes of the K entry, then of the V entry.
                dflash_kv_heads(owner, ranks) * (geometry::dflash(owner.geometry).qk_head + geometry::dflash(owner.geometry).v_head) / 8,
                1,
            ),
            EntrySpec::new(
                format!("chop_{tag}_dflash_drafts_v1"),
                vec![
                    ptr("gathered", "const float"),
                    ptr("token_ids", "int32_t"),
                    ptr("positions", "const int32_t"),
                    ptr("prefix_len", "const int32_t"),
                    ptr("token_batch", "const int32_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("capacity"),
                ],
                "chop::mimo_dflash::drafts(gathered, token_ids, positions, prefix_len, token_batch, live_rows, capacity);".into(),
                mimo26::DFLASH_DRAFT_THREADS,
                1,
            ),
        ],
    }
}

/// The drafters' retained-row scatter and catch-up gather
/// (`cuda/mimo_retain.cuh`).
/// `model_mimo26_contract::sm120::encoders::rows`: the kernels the declarations'
/// registered stages run in MiMo's own cubin, none when they run none.
fn encoder_rows_spec(
    manifest: &Path,
    geometry: &'static model_mimo26_contract::geometry::Geometry,
) -> Option<CubinSpec> {
    use model_mimo26_contract::sm120::encoders::rows;
    let kernels = rows::kernels_for(geometry).expect("MiMo encoder declarations");
    if kernels.is_empty() {
        return None;
    }
    Some(CubinSpec {
        name: if geometry.id == model_mimo26_contract::geometry::V26_FLASH.id {
            rows::CUBIN.into()
        } else {
            format!("{}_encoder", geometry.kernel_tag)
        },
        sources_root: manifest.join("../../.."),
        include_dirs: vec![
            PathBuf::from("models/mimo26/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec![format!("#include \"{}\"", rows::HEADER)],
        defines: Vec::new(),
        entries: kernels
            .into_iter()
            .map(|kernel| {
                let params = kernel
                    .params
                    .iter()
                    .map(|&(name, cxx_type)| EntryParam::DevicePointer {
                        name: name.into(),
                        cxx_type: cxx_type.into(),
                    })
                    .collect();
                EntrySpec::new(kernel.symbol, params, kernel.body, rows::THREADS, 1)
            })
            .collect(),
    })
}

fn retain_spec(manifest: &Path, owner: GeometryModel) -> CubinSpec {
    let tag = owner.geometry.kernel_tag;
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    CubinSpec {
        name: format!("{tag}_retain"),
        sources_root: manifest.join("../../.."),
        include_dirs: vec![PathBuf::from("models/mimo26/bundle/cuda"), PathBuf::from("crates/kernels-sm120/cuda")],
        headers: vec!["#include \"mimo_retain.cuh\"".into()],
        defines: Vec::new(),
        entries: vec![
            EntrySpec::new(
                format!("chop_{tag}_retain_rows_v1"),
                vec![
                    ptr("input", "const uint16_t"),
                    ptr("history", "uint16_t"),
                    ptr("slot_ids", "const int32_t"),
                    ptr("token_batch", "const int32_t"),
                    ptr("q_start", "const int32_t"),
                    ptr("prefix_len", "const int32_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("width"),
                    u32("input_stride"),
                    u32("history_stride"),
                    u32("capacity"),
                    u32("slots"),
                ],
                "chop::mimo_retain::retain_rows(input, history, slot_ids, token_batch, q_start, prefix_len, live_rows, width, input_stride, history_stride, capacity, slots);".into(),
                mimo26::ROW_THREADS,
                1,
            ),
            EntrySpec::new(
                format!("chop_{tag}_gather_rows_v1"),
                vec![
                    ptr("history", "const uint16_t"),
                    ptr("output", "uint16_t"),
                    ptr("slot_ids", "const int32_t"),
                    ptr("token_batch", "const int32_t"),
                    ptr("positions", "const int32_t"),
                    ptr("live_rows", "const uint32_t"),
                    u32("width"),
                    u32("history_stride"),
                    u32("output_stride"),
                    u32("capacity"),
                    u32("slots"),
                ],
                "chop::mimo_retain::gather_rows(history, output, slot_ids, token_batch, positions, live_rows, width, history_stride, output_stride, capacity, slots);".into(),
                mimo26::ROW_THREADS,
                1,
            ),
        ],
    }
}

/// The bundle-owned SWA ring backup and restore kernels (`cuda/mimo_ring.cuh`).
/// Ring slot counts other than 32 that an instance may hold: one slot per
/// item of its largest graph bucket. Their entries live in the
/// `{tag}_ring_slots` cubin, which only an instance of another slot count
/// loads, so a 32-slot instance keeps its modules and launches.
const RING_SLOT_COUNTS: [u32; 8] = [1, 2, 4, 8, 16, 64, 128, 256];

fn ring_spec(manifest: &Path, owner: GeometryModel, ranks: u32) -> CubinSpec {
    let tag = owner.profile(ranks).tag();
    CubinSpec {
        name: format!("{tag}_ring"),
        entries: ring_entries(tag, ring_threads(owner, ranks), None),
        ..ring_cubin(manifest, owner, ranks)
    }
}

/// The ring entries of every other slot count, `chop_{tag}_ring_*_s{slots}_v1`.
fn ring_slots_spec(manifest: &Path, owner: GeometryModel, ranks: u32) -> CubinSpec {
    let tag = owner.profile(ranks).tag();
    CubinSpec {
        name: format!("{tag}_ring_slots"),
        entries: RING_SLOT_COUNTS
            .into_iter()
            .flat_map(|slots| ring_entries(tag, ring_threads(owner, ranks), Some(slots)))
            .collect(),
        ..ring_cubin(manifest, owner, ranks)
    }
}

fn ring_cubin(manifest: &Path, owner: GeometryModel, ranks: u32) -> CubinSpec {
    CubinSpec {
        name: String::new(),
        sources_root: manifest.join("../../.."),
        include_dirs: vec![
            PathBuf::from("models/mimo26/bundle/cuda"),
            PathBuf::from("crates/kernels-sm120/cuda"),
        ],
        headers: vec!["#include \"mimo_ring.cuh\"".into()],
        defines: mimo26::ring_defines_for(owner.geometry, owner.rank(ranks)),
        entries: Vec::new(),
    }
}

/// The ring backup, restore and prefix copy entries for `slots` ring slots,
/// or the 32-slot default.
fn ring_entries(tag: &str, threads: u32, slots: Option<u32>) -> Vec<EntrySpec> {
    let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let (suffix, template) = match slots {
        Some(slots) => (format!("_s{slots}"), format!("<{slots}>")),
        None => (String::new(), String::new()),
    };
    vec![
        EntrySpec::new(
            format!("chop_{tag}_ring_backup{suffix}_v1"),
            vec![
                pointer("state", "uint8_t"),
                pointer("slot_ids", "const int32_t"),
                pointer("q_len", "const int32_t"),
                pointer("prefix_len", "const int32_t"),
                pointer("live_batch", "const uint32_t"),
                EntryParam::U32 {
                    name: "span".into(),
                },
            ],
            format!(
                "chop::mimo_ring::backup{template}(state, slot_ids, q_len, prefix_len, live_batch, span);"
            ),
            threads,
            1,
        ),
        EntrySpec::new(
            format!("chop_{tag}_ring_restore{suffix}_v1"),
            vec![
                pointer("state", "uint8_t"),
                pointer("slot_ids", "const int32_t"),
                pointer("starts", "const int32_t"),
                pointer("counts", "const int32_t"),
                pointer("live_items", "const uint32_t"),
                EntryParam::U32 {
                    name: "span".into(),
                },
            ],
            format!(
                "chop::mimo_ring::restore{template}(state, slot_ids, starts, counts, live_items, span);"
            ),
            threads,
            1,
        ),
        EntrySpec::new(
            format!("chop_{tag}_ring_prefix_copy{suffix}_v1"),
            vec![
                pointer("state", "uint8_t"),
                pointer("drafter", "uint8_t"),
                EntryParam::U64 {
                    name: "drafter_bytes".into(),
                },
                pointer("pool", "uint8_t"),
                pointer("pages", "const uint64_t"),
                EntryParam::U32 {
                    name: "page_count".into(),
                },
                EntryParam::U64 {
                    name: "page_bytes".into(),
                },
                EntryParam::U32 {
                    name: "slot".into(),
                },
                EntryParam::U32 {
                    name: "to_pages".into(),
                },
            ],
            format!(
                "chop::mimo_ring::prefix_copy{template}(state, drafter, drafter_bytes, pool, pages, \
                 page_count, page_bytes, slot, to_pages);"
            ),
            mimo26::PREFIX_THREADS,
            1,
        ),
    ]
}
