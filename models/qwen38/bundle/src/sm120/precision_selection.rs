//! Model-owned construction of the shared mixed projection implementations.
use kernel_build::{BuildContext, KernelSelection, SelectedCubin};
use model_qwen38_contract::sm120::generated;
use std::collections::BTreeMap;
use std::path::Path;

pub fn selection(root: &Path) -> Result<KernelSelection, String> {
    let mut cubins = mixed_cubins()?;
    cubins.extend(hash_cubins()?);
    KernelSelection::new(
        BuildContext::new("bundle-qwen38", "chopinfer", "qwen38.precision.v1")
            .source_root("chopinfer", root),
        cubins,
    )
    .map_err(|e| e.to_string())
}

/// Each topology's mixed modules, the served one first: its cubins keep their
/// names and entries, and another topology's cubins, named for it, hold only
/// the entries the earlier ones lack.
fn topologies() -> impl Iterator<Item = model_qwen38_contract::contract::Ranks> {
    let served = model_qwen38_contract::contract::TOPOLOGY.ranks;
    std::iter::once(served).chain(
        model_qwen38_contract::contract::TOPOLOGIES
            .iter()
            .map(|t| t.ranks)
            .filter(move |&ranks| ranks != served),
    )
}

/// The registered hash entries of each topology that publishes the packed
/// n-gram lookup they prefix.
fn hash_cubins() -> Result<Vec<SelectedCubin>, String> {
    let mut cubins = Vec::new();
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for ranks in topologies() {
            let shape = model_qwen38_contract::contract::Shape::new(geometry, ranks);
            let prefill = generated::ngram_hash_entry(shape, "prefill");
            if model_qwen38_contract::sm120::generated::inventory(shape)?
                .document
                .operations
                .iter()
                .any(|operation| operation.stages.contains(&prefill))
            {
                cubins.push(hash_cubin(shape)?);
            }
        }
    }
    Ok(cubins)
}

fn hash_cubin(shape: model_qwen38_contract::contract::Shape) -> Result<SelectedCubin, String> {
    let geometry = shape.geometry;
    use kernel_build::{CubinSpec, EntryParam, EntrySpec};
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u32 = |name: &str| EntryParam::U32 { name: name.into() };
    let u64 = |name: &str| EntryParam::U64 { name: name.into() };
    let mut entries = Vec::new();
    for phase in ["prefill", "decode", "verify"] {
        let mut params = vec![
            ptr("tokens", "const uint32_t"),
            ptr("history", "const uint32_t"),
            ptr("multipliers", "const uint64_t"),
            ptr("sizes", "const uint64_t"),
            ptr("offsets", "const uint64_t"),
            ptr("ids", "uint64_t"),
            ptr("history_out", "uint32_t"),
        ];
        let mut args = "tokens,history,multipliers,sizes,offsets,ids,history_out,".to_owned();
        if phase == "verify" {
            params.push(ptr("cu_rows", "const int32_t"));
            args.push_str("cu_rows,");
        }
        params.push(ptr("q_len", "const uint32_t"));
        args.push_str("q_len,");
        if phase == "verify" {
            params.push(ptr("live_items", "const uint32_t"));
            args.push_str("live_items,");
        }
        params.extend([u32("rank"), u64("table_start"), u64("table_rows")]);
        args.push_str("rank,table_start,table_rows");
        if phase != "prefill" {
            params.push(u64("entry_words"));
            args.push_str(",entry_words");
        }
        entries.push(EntrySpec::new(
            generated::ngram_hash_entry(shape, phase),
            params,
            format!("chop::qwen38_hash::{phase}({args});"),
            256,
            1,
        ));
    }
    let declarations = entries
        .iter()
        .map(|entry| kernel_build::EntryDeclaration {
            symbol: entry.symbol.clone(),
            operation: "qwen38.ngram-hash-history".into(),
            contract: "qwen38.ngram-u64-history-u32.v1".into(),
            implementation: entry.symbol.clone(),
            requirements: std::collections::BTreeMap::from([(
                "typed_entry".into(),
                kernel_build::digest(format!("{entry:?}").as_bytes()),
            )]),
            correctness_case: Some("exact Qwen hash and chunked history oracle".into()),
        })
        .collect();
    SelectedCubin::new(
        CubinSpec {
            name: format!("{}_hash_v1", generated::cubin_prefix(shape)),
            sources_root: Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.."),
            include_dirs: vec![
                "models/qwen38/bundle/cuda".into(),
                "crates/kernels-sm120/cuda".into(),
            ],
            headers: vec!["#include \"qwen38_ngram_hash.cuh\"".into()],
            defines: {
                let mut defines = model_qwen38_contract::sm120::geometry::ngram_defines(*geometry);
                defines.extend([
                    ("CHOP_QWEN38_RANKS".into(), shape.ranks().to_string()),
                    (
                        "CHOP_QWEN38_NGRAM_LOCAL_HEADS".into(),
                        shape.ngram_local_heads.to_string(),
                    ),
                ]);
                defines
            },
            entries,
        },
        declarations,
    )
    .map_err(|e| e.to_string())
}

/// The launch-shape row stages (layout copies) of every geometry and
/// topology: they take their shapes at launch, so one cubin serves them all.
pub const SHAPE_FREE: &str = "qwen38_precision_shape_free_v1";

fn mixed_cubins() -> Result<Vec<SelectedCubin>, String> {
    let mut cubins = Vec::new();
    let mut shape_free = Vec::new();
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        cubins.extend(mixed_cubins_for(geometry, &mut shape_free)?);
    }
    if !shape_free.is_empty() {
        cubins.push(kernels_sm120::precision::launch_row_cubin(
            SHAPE_FREE,
            &shape_free,
        )?);
    }
    Ok(cubins)
}

/// One geometry's own precision cubins; its launch-shape row stages join
/// `shape_free`.
fn mixed_cubins_for(
    geometry: &'static model_qwen38_contract::geometry::Geometry,
    shape_free: &mut Vec<kernels_sm120::row::precision::RowStage>,
) -> Result<Vec<SelectedCubin>, String> {
    use kernels_sm120::row::precision::ShapeBinding;
    let mut cubins = Vec::new();
    let mut placed = std::collections::BTreeSet::new();
    for ranks in topologies() {
        let shape = model_qwen38_contract::contract::Shape::new(geometry, ranks);
        let prefix = generated::cubin_prefix(shape);
        let inventory = model_qwen38_contract::sm120::generated::inventory(shape)?;
        let mut offers = Vec::new();
        for offer in inventory.offers {
            let launch: Vec<_> = offer
                .handle()
                .row_stages()
                .into_iter()
                .filter(|stage| stage.shapes() == ShapeBinding::Launch)
                .collect();
            // Every topology shares the launch-shape row stages' cubin.
            let symbols = offer
                .handle()
                .entries()
                .map_err(|e| e.to_string())?
                .into_iter()
                .map(|entry| entry.symbol)
                .filter(|symbol| launch.iter().all(|s| &s.entry().symbol != symbol))
                .collect::<Vec<_>>();
            shape_free.extend(launch);
            let known = symbols.iter().filter(|s| placed.contains(*s)).count();
            if known == symbols.len() {
                continue;
            }
            if known != 0 {
                return Err(format!(
                    "{prefix}: an offer shares only some entries with another topology's cubins"
                ));
            }
            placed.extend(symbols);
            offers.push(offer);
        }
        if !offers.is_empty() {
            cubins.extend(kernels_sm120::precision::selected_cubins_with(
                &prefix,
                &offers,
                &kernels_sm120::cuda_source_root(),
                kernels_sm120::precision::LaunchRows::Shared,
            )?);
        }
        // Uses of one copy at several capacities share its entry.
        let mut stages = std::collections::BTreeMap::new();
        for copy in inventory.copies {
            if copy.stage().shapes() == ShapeBinding::Launch {
                shape_free.push(copy.stage().clone());
            } else if placed.insert(copy.stage().entry().symbol.clone()) {
                stages.insert(copy.stage().entry().symbol.clone(), copy.stage().clone());
            }
        }
        let stages: Vec<_> = stages.into_values().collect();
        if !stages.is_empty() {
            cubins.push(
                SelectedCubin::new(
                    kernels_sm120::row::precision::cubin_spec(
                        &format!("{prefix}_layouts_v1"),
                        &stages,
                    )
                    .map_err(|e| e.to_string())?,
                    stages.iter().map(|s| s.declaration()).collect(),
                )
                .map_err(|e| e.to_string())?,
            );
        }
    }
    Ok(cubins)
}

/// The rank count each mixed module loads at: one topology's, or 0 when
/// several topologies bind its entries.
pub fn mixed_ranks(selection: &KernelSelection) -> Result<BTreeMap<String, u32>, String> {
    let mut bound = BTreeMap::<String, std::collections::BTreeSet<u32>>::new();
    for geometry in model_qwen38_contract::geometry::GEOMETRIES {
        for ranks in topologies() {
            let shape = model_qwen38_contract::contract::Shape::new(geometry, ranks);
            for operation in model_qwen38_contract::sm120::generated::inventory(shape)?
                .document
                .operations
            {
                for symbol in operation.stages {
                    bound.entry(symbol).or_default().insert(shape.ranks());
                }
            }
        }
    }
    let mut result = BTreeMap::new();
    for spec in selection.specs() {
        let mut ranks = std::collections::BTreeSet::new();
        for entry in &spec.entries {
            ranks.extend(bound.get(&entry.symbol).into_iter().flatten().copied());
        }
        if !ranks.is_empty() {
            result.insert(
                spec.name.clone(),
                if ranks.len() == 1 {
                    ranks.into_iter().next().unwrap_or_default()
                } else {
                    0
                },
            );
        }
    }
    Ok(result)
}

/// Keep the existing cubin names and locked wrappers. The family declares its
/// old entries and projection ports; shared builders declare the new entries.
pub fn with_legacy(
    context: BuildContext,
    specs: Vec<kernels_sm120::arithmetic::DeclaredCubin>,
) -> Result<KernelSelection, String> {
    // Launch arguments and Variant fields that decide output bits enter
    // each entry's declaration beside its wrapper, as the producer of its
    // cubin states them.
    let mut cubins = Vec::new();
    for cubin in specs {
        cubins.push(
            cubin
                .select(
                    "qwen38.legacy-program",
                    "qwen38.compiled-entry.v1",
                    "Qwen kernel-lab operations and fixed-batch legacy execution",
                )
                .map_err(|e| e.to_string())?,
        );
    }
    cubins.extend(mixed_cubins()?);
    cubins.extend(hash_cubins()?);
    KernelSelection::new(context, cubins).map_err(|e| e.to_string())
}
