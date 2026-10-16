//! The SM120 entries GLM's precision provider (`crate::prepared`) binds a
//! lowered projection use to: the shared tiled GEMM's offer
//! (`super::precision::offer`) with its layout copy, or the stacked
//! entry's part the use runs as, and the dependency of each binding on the
//! locked cubins that build them.

use std::collections::{BTreeMap, BTreeSet};

use kernel_core::contract::WeightFormat;
use kernel_core::precision::OperationUse;
use kernels_sm120::precision::generated::GeneratedOffer;
use kernels_sm120::row::precision::LayoutCopy;
use sha2::{Digest, Sha256};

use crate::backend::Bound;
use crate::contract::Shape;

/// The digest of the projections' locked cubins `locks`, in order, and
/// every entry symbol.
fn digest(locks: &[&[u8]], symbols: impl IntoIterator<Item = String>) -> String {
    let mut digest = Sha256::new();
    for lock in locks {
        digest.update(lock);
    }
    for symbol in symbols {
        digest.update((symbol.len() as u64).to_le_bytes());
        digest.update(symbol.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

const PROJECTIONS_LOCK: &[u8] = include_bytes!("../../../bundle/kernels.lock.d/projections.lock");
const PROJECTION_FORMATS_LOCK: &[u8] =
    include_bytes!("../../../bundle/kernels.lock.d/projection_formats.lock");
const STACKED_LOCK: &[u8] =
    include_bytes!("../../../bundle/kernels.lock.d/projections_stacked.lock");
const DECODED_LOCK: &[u8] =
    include_bytes!("../../../bundle/kernels.lock.d/projections_decoded.lock");
const STAGED_LOCK: &[u8] = include_bytes!("../../../bundle/kernels.lock.d/projections_staged.lock");

/// The block-FP8 export's symbols: the GEMMs', then the copies' (each
/// once; a copy's uses at several capacities share its entry).
fn served_symbols() -> Result<Vec<String>, String> {
    let mut copies: Vec<_> = super::precision::copies(Shape::TP4)?
        .iter()
        .map(|copy| copy.stage().entry().symbol.clone())
        .collect();
    copies.dedup();
    Ok(super::precision::instantiations(Shape::TP4)?
        .into_iter()
        .map(|spec| spec.symbol)
        .chain(copies)
        .collect())
}

/// The binding dependency of the projections' kernels: the locked cubins
/// and every entry symbol, the GEMMs' then the copies'.
fn dependency() -> Result<&'static str, String> {
    use std::sync::OnceLock;
    static DEPENDENCY: OnceLock<Result<String, String>> = OnceLock::new();
    DEPENDENCY
        .get_or_init(|| Ok(digest(&[PROJECTIONS_LOCK], served_symbols()?)))
        .as_deref()
        .map_err(Clone::clone)
}

/// The NVFP4 layouts' added GEMM entries (`super::projection_formats`).
fn added_symbols() -> Result<&'static BTreeSet<String>, String> {
    use std::sync::OnceLock;
    static ADDED: OnceLock<Result<BTreeSet<String>, String>> = OnceLock::new();
    ADDED
        .get_or_init(|| {
            Ok(super::projection_formats::instantiations_added(Shape::TP4)?
                .into_iter()
                .map(|spec| spec.symbol)
                .collect())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// [`dependency`] with the NVFP4 layouts' added cubin: its lock after the
/// block-FP8 one's, and its symbols after theirs.
fn formats_dependency_with(formats_lock: &[u8]) -> Result<String, String> {
    let symbols = served_symbols()?
        .into_iter()
        .chain(added_symbols()?.iter().cloned());
    Ok(digest(&[PROJECTIONS_LOCK, formats_lock], symbols))
}

/// The decoded-tile GEMM entries (`super::precision::decoded_cubin_spec`).
fn decoded_symbols() -> Result<&'static BTreeSet<String>, String> {
    use std::sync::OnceLock;
    static DECODED: OnceLock<Result<BTreeSet<String>, String>> = OnceLock::new();
    DECODED
        .get_or_init(|| {
            Ok(super::precision::decoded_cubin_spec(Shape::TP4)?
                .entries
                .into_iter()
                .map(|entry| entry.symbol)
                .collect())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// [`dependency`] with the decoded-tile cubin's lock after the block-FP8
/// one's (its symbols are among the served ones).
fn decoded_dependency_with(decoded_lock: &[u8]) -> Result<String, String> {
    Ok(digest(&[PROJECTIONS_LOCK, decoded_lock], served_symbols()?))
}

/// The staged-tile GEMM entries (`super::precision::staged_cubin_spec`).
fn staged_symbols() -> Result<&'static BTreeSet<String>, String> {
    use std::sync::OnceLock;
    static STAGED: OnceLock<Result<BTreeSet<String>, String>> = OnceLock::new();
    STAGED
        .get_or_init(|| {
            Ok(super::precision::staged_cubin_spec(Shape::TP4)?
                .entries
                .into_iter()
                .map(|entry| entry.symbol)
                .collect())
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// [`dependency`] with the staged-tile cubin's lock after the block-FP8
/// one's (its symbols are among the served ones).
fn staged_dependency_with(staged_lock: &[u8]) -> Result<String, String> {
    Ok(digest(&[PROJECTIONS_LOCK, staged_lock], served_symbols()?))
}

/// The dependency of a binding whose capability runs `stages`: a binding
/// that runs an added entry depends on the block-FP8 and the added
/// entries' locks, one that runs a decoded-tile (or staged-tile) entry on
/// the block-FP8 and the decoded-tile (or staged-tile) locks; any other
/// keeps the block-FP8 projections' dependency.
fn dependency_of(stages: &BTreeSet<String>) -> Result<String, String> {
    use std::sync::OnceLock;
    let runs = |entries: &BTreeSet<String>| stages.iter().any(|stage| entries.contains(stage));
    if runs(staged_symbols()?) {
        if runs(added_symbols()?) || runs(decoded_symbols()?) {
            return Err(
                "a binding runs a staged-tile entry and an added or decoded-tile one".into(),
            );
        }
        static STAGED: OnceLock<Result<String, String>> = OnceLock::new();
        return STAGED
            .get_or_init(|| staged_dependency_with(STAGED_LOCK))
            .clone();
    }
    match (runs(added_symbols()?), runs(decoded_symbols()?)) {
        (true, true) => Err("a binding runs an added entry and a decoded-tile entry".into()),
        (true, false) => {
            static FORMATS: OnceLock<Result<String, String>> = OnceLock::new();
            FORMATS
                .get_or_init(|| formats_dependency_with(PROJECTION_FORMATS_LOCK))
                .clone()
        }
        (false, true) => {
            static DECODED: OnceLock<Result<String, String>> = OnceLock::new();
            DECODED
                .get_or_init(|| decoded_dependency_with(DECODED_LOCK))
                .clone()
        }
        (false, false) => dependency().map(str::to_owned),
    }
}

/// Every stacked entry's symbol (`super::stacked::entries`).
fn stacked_symbols() -> Result<&'static [String], String> {
    use std::sync::OnceLock;
    static SYMBOLS: OnceLock<Result<Vec<String>, String>> = OnceLock::new();
    SYMBOLS
        .get_or_init(|| {
            Ok(super::stacked::entries(Shape::TP4)?
                .into_iter()
                .map(|entry| entry.symbol)
                .collect())
        })
        .as_deref()
        .map_err(Clone::clone)
}

/// The dependency of a use that runs as `part` of a stacked entry, under
/// the stacked lock `stacked_lock`: the projections' lock (whose entry of
/// the use keeps the part's bits and sizes its workspace) and the stacked
/// one, every served symbol, every stacked entry's, then the fusion the
/// use runs in: the entry, its schedule, and the part's index, projection,
/// weight rows, written columns, row stride and first column.
fn stacked_dependency_with(
    stacked_lock: &[u8],
    part: &super::stacked::StackedPart,
) -> Result<String, String> {
    let entry = &part.entry;
    let placed = &part.placed;
    let fusion = format!(
        "{} capacity {} k {} strips {} splits {} prefetch {} part {} {} rows {} width {} stride {} column {}",
        entry.symbol,
        entry.capacity,
        entry.k,
        entry.strips,
        entry.splits,
        entry.prefetch,
        part.index,
        placed.id,
        placed.part.rows,
        placed.part.width,
        placed.part.row_stride,
        placed.column_offset
    );
    let symbols = served_symbols()?
        .into_iter()
        .chain(stacked_symbols()?.iter().cloned())
        .chain([fusion]);
    Ok(digest(&[PROJECTIONS_LOCK, stacked_lock], symbols))
}

/// [`stacked_dependency_with`] the bundle's stacked lock, once per part.
fn stacked_dependency(part: &super::stacked::StackedPart) -> Result<String, String> {
    use std::sync::{Mutex, OnceLock};
    static DEPENDENCIES: OnceLock<Mutex<BTreeMap<String, String>>> = OnceLock::new();
    let key = format!("{part:?}");
    let cache = DEPENDENCIES.get_or_init(Mutex::default);
    if let Some(dependency) = cache.lock().map_err(|e| e.to_string())?.get(&key) {
        return Ok(dependency.clone());
    }
    let dependency = stacked_dependency_with(STACKED_LOCK, part)?;
    cache
        .lock()
        .map_err(|e| e.to_string())?
        .insert(key, dependency.clone());
    Ok(dependency)
}

/// A derived operation's shared offer at one use and the copy of its rows
/// into the strided buffer its consumer reads, if any
/// (`super::precision::copy`), or the stacked entry's part the use is bound
/// to instead (`super::stacked::part`).
#[derive(Clone)]
pub struct Offer {
    pub generated: GeneratedOffer,
    pub copy: Option<LayoutCopy>,
    pub stacked: Option<super::stacked::StackedPart>,
}

/// The shared offer of lowered projection `id` with `weight` rows at
/// `usage`, with its copy, its capability and its dependency. A use that
/// runs as its part of a stacked entry binds that entry, not its GEMM and
/// copy, and depends on the stacked lock and its fusion.
pub fn bind(id: &str, weight: WeightFormat, usage: OperationUse) -> Result<Bound<Offer>, String> {
    let generated = super::precision::offer(id, weight, usage)?;
    let mut capability = super::precision::capability(id, weight, usage)?;
    let stacked = super::stacked::part(id, weight, usage)?;
    if let Some(part) = &stacked {
        capability.stages = vec![part.entry.symbol.clone()];
    }
    let stages: BTreeSet<String> = capability.stages.iter().cloned().collect();
    let dependency = match &stacked {
        Some(part) => stacked_dependency(part)?,
        None => dependency_of(&stages)?,
    };
    Ok(Bound {
        entry: Offer {
            generated,
            copy: super::precision::copy(id, weight, usage)?,
            stacked,
        },
        capability,
        dependency,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use kernel_core::precision::capability::{CapabilityDocument, CapabilityInventory};
    use model_compiler_sdk::quantization::import::ExportMetadata;

    use super::*;
    use crate::prepared::{
        bind, check, checkpoint_component, decode, decode_for, document, encode, instance,
        test_component,
    };
    use crate::quantization::FP8_BLOCK128;
    use kernel_core::precision::execution::ImplementationProvider;

    fn metadata() -> ExportMetadata {
        let config: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../fixtures/config.json")).unwrap();
        ExportMetadata::from_documents(config.get("quantization_config"), None).unwrap()
    }

    /// Every rank's document of the FP8 export holds each lowered
    /// projection's instances at every declared use, bound to the tiled
    /// GEMM entries the projections' cubin builds, and survives encoding.
    #[test]
    fn every_rank_binds_every_lowered_instance() {
        let symbols: BTreeSet<String> = crate::sm120::precision::cubin_specs(Shape::TP4)
            .unwrap()
            .into_iter()
            .chain([
                crate::sm120::precision::decoded_cubin_spec(Shape::TP4).unwrap(),
                crate::sm120::precision::staged_cubin_spec(Shape::TP4).unwrap(),
            ])
            .flat_map(|cubin| cubin.entries)
            .map(|entry| entry.symbol)
            .collect();
        let first = document(&FP8_BLOCK128, &metadata(), 0).unwrap();
        for rank in 0..Shape::TP4.ranks() {
            let document = document(&FP8_BLOCK128, &metadata(), rank).unwrap();
            // The package carries one document for every rank.
            assert_eq!(document, first, "rank {rank}");
            check(&document, &FP8_BLOCK128).unwrap();
            for selection in &document.operations {
                for binding in &selection.bindings {
                    let rebound = bind(&selection.effective, binding.usage).unwrap();
                    assert_eq!(rebound.identity, binding.implementation);
                    let mut stages = rebound.handle.generated.offer.capability().unwrap().stages;
                    stages.extend(
                        rebound
                            .handle
                            .copy
                            .iter()
                            .map(|copy| copy.stage().entry().symbol.clone()),
                    );
                    assert!(stages.iter().all(|stage| symbols.contains(stage)));
                }
            }
            let bytes = encode(&document).unwrap();
            assert_eq!(
                kernel_core::precision::component::decode(&bytes).unwrap(),
                document
            );
        }
    }
    /// A binding that runs an entry the NVFP4 layouts add depends on that
    /// cubin's lock too: NVIDIA's NVFP4 dense down binds under both locks,
    /// so another projection_formats lock gives another identity and the
    /// component's recorded binding no longer matches; a binding of the
    /// block-FP8 projections alone keeps their dependency.
    #[test]
    fn bindings_of_added_entries_depend_on_their_lock() {
        use crate::quantization::NVFP4_DENSE_ROUTED;
        use kernel_core::contract::WeightFormat;
        let config: serde_json::Value =
            serde_json::from_slice(include_bytes!("../../fixtures/nvidia-nvfp4-config.json"))
                .unwrap();
        let component = checkpoint_component(&NVFP4_DENSE_ROUTED, &config).unwrap();
        let document = decode_for(&component, &NVFP4_DENSE_ROUTED).unwrap();
        let selection = document
            .operations
            .iter()
            .find(|s| s.source.id == "dense.down.0.f32")
            .unwrap();
        let recorded = &selection.bindings[0];
        let usage = recorded.usage;
        let bound = bind(&selection.effective, usage).unwrap();
        assert_eq!(bound.identity, recorded.implementation);
        let capability =
            crate::sm120::precision::capability("dense.down", WeightFormat::Nvfp4, usage).unwrap();
        let stages: BTreeSet<String> = capability.stages.iter().cloned().collect();
        let formats = formats_dependency_with(PROJECTION_FORMATS_LOCK).unwrap();
        assert_eq!(dependency_of(&stages).unwrap(), formats);
        assert_ne!(formats, dependency().unwrap());
        let mut changed = PROJECTION_FORMATS_LOCK.to_vec();
        changed.extend_from_slice(b"\n");
        let changed = formats_dependency_with(&changed).unwrap();
        assert_ne!(changed, formats);
        let rebound = CapabilityInventory::new(
            CapabilityDocument {
                version: 1,
                operations: vec![capability],
            },
            stages,
            changed,
        )
        .unwrap()
        .bind(&selection.effective, usage)
        .unwrap();
        assert_ne!(rebound.identity, recorded.implementation);
        // A block-FP8 binding keeps the block-FP8 projections' dependency.
        let fp8 = decode(&test_component()).unwrap();
        let selection = &fp8.operations[0];
        let capability = crate::sm120::precision::capability(
            &instance(&selection.source.id).unwrap().0,
            selection.effective.weights[0].format,
            selection.bindings[0].usage,
        )
        .unwrap();
        let stages: BTreeSet<String> = capability.stages.iter().cloned().collect();
        assert_eq!(dependency_of(&stages).unwrap(), dependency().unwrap());
    }

    /// A KDA input projection's decode and verify uses of at most 32 rows
    /// bind their part of the stacked entry: its symbol is the only stage
    /// and the dependency covers the stacked lock and the part's fusion, so
    /// another stacked lock (or another part) gives another identity and
    /// the recorded binding no longer matches. Prefill uses keep their GEMM
    /// and copy under the projections' dependency, or the staged-tile lock's
    /// beside it when their GEMM is a staged-tile entry.
    #[test]
    fn stacked_bindings_depend_on_the_stacked_lock() {
        use crate::sm120::stacked;
        use kernel_core::contract::WeightFormat;
        use kernel_core::precision::ExecutionPhase;
        let document = decode(&test_component()).unwrap();
        let selection = |id: &str| {
            document
                .operations
                .iter()
                .find(|s| s.source.id == id)
                .unwrap()
        };
        let mut checked = 0;
        for id in stacked::PROJECTIONS {
            let selection = selection(&format!("{id}.0.bf16"));
            for recorded in &selection.bindings {
                let usage = recorded.usage;
                let identity = &recorded.implementation;
                let Some(part) = stacked::part(id, WeightFormat::Bf16, usage).unwrap() else {
                    assert!(
                        usage.phase == ExecutionPhase::Prefill || usage.capacity > 32,
                        "{id} {usage:?}"
                    );
                    assert!(identity.stages.iter().all(|s| !s.contains("_stacked_")));
                    let stages: BTreeSet<String> = identity.stages.iter().cloned().collect();
                    let want = if stages.is_disjoint(staged_symbols().unwrap()) {
                        dependency().unwrap().to_owned()
                    } else {
                        staged_dependency_with(STAGED_LOCK).unwrap()
                    };
                    assert_eq!(identity.dependency, want, "{id} {usage:?}");
                    continue;
                };
                checked += 1;
                assert_eq!(
                    identity.stages,
                    std::slice::from_ref(&part.entry.symbol),
                    "{id} {usage:?}"
                );
                let stacked = stacked_dependency_with(STACKED_LOCK, &part).unwrap();
                assert_eq!(identity.dependency, stacked);
                assert_ne!(stacked, dependency().unwrap());
                // Another stacked lock: another dependency, and the
                // recorded binding no longer matches the rebinding.
                let mut changed = STACKED_LOCK.to_vec();
                changed.extend_from_slice(b"\n");
                let changed = stacked_dependency_with(&changed, &part).unwrap();
                assert_ne!(changed, stacked);
                let mut capability =
                    crate::sm120::precision::capability(id, WeightFormat::Bf16, usage).unwrap();
                capability.stages = vec![part.entry.symbol.clone()];
                let rebound = CapabilityInventory::new(
                    CapabilityDocument {
                        version: 1,
                        operations: vec![capability],
                    },
                    BTreeSet::from([part.entry.symbol.clone()]),
                    changed,
                )
                .unwrap()
                .bind(&selection.effective, usage)
                .unwrap();
                assert_ne!(rebound.identity, *identity);
                assert_eq!(
                    bind(&selection.effective, usage).unwrap().identity,
                    *identity
                );
            }
        }
        // Six projections at six decode and six verify buckets.
        assert_eq!(checked, 6 * 12);
        // Each part binds its own fusion.
        let q = selection("linear.q.0.bf16")
            .bindings
            .iter()
            .map(|b| b.usage)
            .find(|u| u.phase == ExecutionPhase::Decode && u.capacity == 1)
            .unwrap();
        let part = |id| stacked::part(id, WeightFormat::Bf16, q).unwrap().unwrap();
        assert_ne!(
            stacked_dependency_with(STACKED_LOCK, &part("linear.q")).unwrap(),
            stacked_dependency_with(STACKED_LOCK, &part("linear.k")).unwrap()
        );
    }

    /// A binding that runs a decoded-tile entry (every FP8 prefill use
    /// `crate::sm120::precision::decoded_tiles` takes) depends on the decoded-tile
    /// cubin's lock too, and one that runs a staged-tile entry (the BF16
    /// prefill uses `crate::sm120::precision::staged_tiles` takes) on the
    /// staged-tile cubin's: another projections_decoded or
    /// projections_staged lock gives another identity and the component's
    /// recorded binding no longer matches. Every other binding but those
    /// bound to a stacked entry keeps the block-FP8 projections'
    /// dependency.
    #[test]
    fn bindings_of_decoded_tiles_depend_on_their_lock() {
        let document = decode(&test_component()).unwrap();
        let decoded = decoded_symbols().unwrap();
        assert_eq!(decoded.len(), 6);
        let staged = staged_symbols().unwrap();
        assert!(!staged.is_empty() && staged.is_disjoint(decoded));
        let lock = decoded_dependency_with(DECODED_LOCK).unwrap();
        assert_ne!(lock, dependency().unwrap());
        let staged_lock = staged_dependency_with(STAGED_LOCK).unwrap();
        assert_ne!(staged_lock, dependency().unwrap());
        assert_ne!(staged_lock, lock);
        let mut changed = DECODED_LOCK.to_vec();
        changed.extend_from_slice(b"\n");
        let changed = decoded_dependency_with(&changed).unwrap();
        assert_ne!(changed, lock);
        let mut staged_changed = STAGED_LOCK.to_vec();
        staged_changed.extend_from_slice(b"\n");
        let staged_changed = staged_dependency_with(&staged_changed).unwrap();
        assert_ne!(staged_changed, staged_lock);
        let (mut bound_decoded, mut bound_staged, mut bound_other) = (0, 0, 0);
        for selection in &document.operations {
            let id = instance(&selection.source.id).unwrap().0;
            let weight = selection.effective.weights[0].format;
            for recorded in &selection.bindings {
                let usage = recorded.usage;
                let identity = &recorded.implementation;
                // Uses bound to a stacked entry's part depend on the stacked
                // lock (`stacked_bindings_depend_on_the_stacked_lock`).
                if crate::sm120::stacked::part(&id, weight, usage)
                    .unwrap()
                    .is_some()
                {
                    continue;
                }
                let capability = crate::sm120::precision::capability(&id, weight, usage).unwrap();
                let stages: BTreeSet<String> = capability.stages.iter().cloned().collect();
                let (want, changed) = if !stages.is_disjoint(decoded) {
                    bound_decoded += 1;
                    (&lock, &changed)
                } else if !stages.is_disjoint(staged) {
                    bound_staged += 1;
                    (&staged_lock, &staged_changed)
                } else {
                    bound_other += 1;
                    assert_eq!(identity.dependency, dependency().unwrap(), "{id} {usage:?}");
                    continue;
                };
                assert_eq!(identity.dependency, *want, "{id} {usage:?}");
                let rebound = CapabilityInventory::new(
                    CapabilityDocument {
                        version: 1,
                        operations: vec![capability],
                    },
                    stages,
                    changed.clone(),
                )
                .unwrap()
                .bind(&selection.effective, usage)
                .unwrap();
                assert_ne!(rebound.identity, *identity, "{id} {usage:?}");
            }
        }
        assert!(bound_decoded > 0 && bound_staged > 0 && bound_other > 0);
    }
}
