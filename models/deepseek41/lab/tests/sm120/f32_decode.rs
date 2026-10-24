//! Bitwise baseline comparisons and paired cold-weight launch timings.
#[path = "f32_decode/cached_spans.rs"]
mod cached_spans;

use bundle_deepseek41::sm120::{
    f32_head, f32_prefill, f32_split, fold_cubin_for_symbol, head_last, projection_reuse,
    register_f32_projections,
};
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, LockDir, Toolchain};
use kernel_core::backend::{Backend, Counts, Launch, Need, Prepared, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse, buffers::*};
use kernel_harness::kit::cases::f32_projection::{self as cases, Witness};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch, poison};
use kernels_sm120::{
    backend::{F32ProjectionOffers, Sm120},
    f32_projection::{Entry, cubin_spec},
};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::GEOMETRIES;
use runtime_cuda::{Module, Stream};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

const CUBIN: &str = "f32_split_edges";
const LEGACY_CUBIN: &str = "f32_decode_edges";
const TIMING_ROWS: [u32; 12] = [1, 2, 4, 8, 16, 32, 33, 48, 64, 128, 192, 256];
const MAX_SLOWDOWN_RATIO: f32 = 1.03;
const VERIFY_GOAL_RATIO: f32 = 0.2;
// User-specified bandwidth roofline: 1.6 TB/s, with actual stored weights.
const BYTES_PER_US: f64 = 1_600_000.;
const EDGE_ROWS: [u32; 4] = [1, 32, 33, 256];
const BUCKET_ROWS: [u32; 14] = [1, 2, 4, 8, 16, 31, 32, 33, 48, 64, 128, 192, 255, 256];
const ROW_SAMPLE_SEED: u32 = 0x6c31_8a95;

fn sampled_rows() -> Vec<u32> {
    let mut rows: BTreeSet<_> = BUCKET_ROWS.into_iter().collect();
    let mut seed = ROW_SAMPLE_SEED;
    while rows.len() < BUCKET_ROWS.len() + 4 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        rows.insert(1 + seed % 256);
    }
    rows.into_iter().collect()
}

pub(super) struct HostCase {
    label: String,
    started: Instant,
}
impl HostCase {
    pub(super) fn new(label: String) -> Self {
        Self {
            label,
            started: Instant::now(),
        }
    }
}

struct ProductionCase {
    old: Entry,
    candidate: Entry,
    rows: Vec<u32>,
}

// Enumerate the same declarations the serving registration consumes, then
// deduplicate entries shared by several supported topologies.
fn production_cases() -> Vec<ProductionCase> {
    let mut selected = BTreeMap::new();
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(g, ranks).is_err() {
                continue;
            }
            let head = f32_head::entry(g, ranks)
                .map(Entry::with_split_decode)
                .unwrap();
            for entry in f32_split::entries(g, ranks).unwrap() {
                let entry = f32_prefill::serving_entry(g, entry).unwrap();
                let maximum = if entry == head {
                    bundle_deepseek41::buffers::logits_rows(
                        &bundle_deepseek41::shape::topologies()
                            .into_iter()
                            .find(|shape| shape.geometry.id == g.id && shape.ranks() == ranks)
                            .unwrap(),
                    )
                } else {
                    entry.query().m_capacity
                };
                assert!(maximum <= entry.query().m_capacity);
                if let Some((previous, previous_maximum)) =
                    selected.insert(entry.symbol().to_owned(), (entry.clone(), maximum))
                {
                    assert_eq!(previous, entry);
                    assert_eq!(previous_maximum, maximum);
                }
            }
        }
    }
    selected
        .into_values()
        .map(|(entry, maximum)| {
            let stem = entry
                .symbol()
                .strip_suffix("_v1_split_v2")
                .expect("registered cached entry retains its baseline stem");
            let old = Entry::new(entry.query(), entry.usage(), stem).unwrap();
            assert_eq!(old.symbol(), entry.baseline_symbol());
            let mut rows = sampled_rows();
            rows.extend(entry.prefill_buckets().iter().map(|&(rows, _)| rows));
            if maximum > 256 {
                rows.extend([512, 2048]);
            }
            rows.push(maximum);
            rows.sort_unstable();
            rows.dedup();
            ProductionCase {
                old,
                candidate: entry,
                rows,
            }
        })
        .collect()
}

const EDGE_WITNESSES: [Witness; 11] = [
    Witness::General,
    Witness::WeightBits,
    Witness::LowComponent,
    Witness::SeparateChains,
    Witness::MergeOrder,
    Witness::FoldBoundary,
    Witness::Fallback,
    Witness::Underflow,
    Witness::NonfiniteInput,
    Witness::NonfiniteWeight,
    Witness::Overflow,
];
const RANDOM_CASES: u32 = 4;

fn timed(rows: u32) -> bool {
    TIMING_ROWS.contains(&rows) || rows > 256
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_DECODE_ARTIFACTS"]
fn gpu_drafter_bounded_reuse() {
    use bundle_deepseek41::sm120::dspark_projection;
    use model_deepseek41_contract::geometry::FLASH;
    kernel_harness::gpu_test_with(timing_cubins(), 4 << 30, |session| {
        let mut cases = 0;
        for entry in f32_split::entries(&FLASH, 4).unwrap() {
            let old = projection_reuse::head(&FLASH, 4, entry).unwrap();
            let new = dspark_projection::route(&FLASH, 4, old.clone()).unwrap();
            if new.bounded_reuse_candidate().is_none() {
                continue;
            }
            let q = old.query();
            let (_, b) = random_operands(0, q.n, q.k, 0x7813_9231);
            let weights = ComparisonWeights::routed(session, &old, new, &b)?;
            for rows in [30, 40, 60, 80] {
                let (mut a, _) = random_operands(rows, 0, q.k, 0x7813_9231 + rows);
                // A late subnormal forces one row's complete SIMT fallback.
                a[q.k as usize - 1] = f32::from_bits(1);
                let label = HostCase::new(format!("drafter_bounded_n{}", q.n));
                compare(session, &old, rows, &a, &weights, &label, true)?;
                cases += 1;
            }
        }
        assert_eq!(cases, 8, "head and router at four forward sizes");
        println!("DSPARK_F32_COMPLETE cases={cases}");
        Ok(())
    })
    .unwrap();
}

fn coverage(matrix: &[ProductionCase]) -> serde_json::Value {
    let edge_cases: usize = edges()
        .iter()
        .map(|entry| {
            if entry.query().n == 65536 {
                1
            } else {
                (EDGE_WITNESSES.len() + RANDOM_CASES as usize) * EDGE_ROWS.len()
                    + usize::from(entry.query().k == 129)
            }
        })
        .sum();
    let production_cases: usize = matrix.iter().map(|case| case.rows.len()).sum();
    let timing_cases: usize = matrix
        .iter()
        .flat_map(|case| &case.rows)
        .filter(|&&rows| timed(rows))
        .count();
    let launch_timings: usize = matrix
        .iter()
        .map(|case| {
            case.rows
                .iter()
                .filter(|&&rows| timed(rows))
                .map(|&rows| {
                    if case.candidate.entry_for_rows(rows).uses_prefill_tiles() {
                        3
                    } else if rows <= 32 {
                        4
                    } else if rows <= 256 {
                        3
                    } else {
                        2
                    }
                })
                .sum::<usize>()
        })
        .sum();
    serde_json::json!({
        "production": matrix.iter().map(|case| serde_json::json!({
            "baseline": case.old.symbol(),
            "candidate": case.candidate.symbol(),
            "prefill_buckets": case.candidate.prefill_buckets(),
            "routed_rows": case.rows.iter().copied().filter(|&rows|
                rows <= case.candidate.decode_max_rows() || case.candidate.entry_for_rows(rows).uses_prefill_tiles()).collect::<Vec<_>>(),
            "n": case.old.query().n,
            "k": case.old.query().k,
            "weight_bytes": case.old.weight_bytes(),
            "f32_equivalent_weight_bytes": u64::from(case.old.query().n) * u64::from(case.old.query().k) * 4,
            "rows": case.rows,
            "timing_rows": case.rows.iter().copied().filter(|&rows| timed(rows)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "maximum_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "verify_goal_ratio": VERIFY_GOAL_RATIO,
        "bytes_per_us": BYTES_PER_US,
        "production_cases": production_cases,
        "edge_cases": edge_cases,
        "bitwise_cases": production_cases + edge_cases,
        "host_cases": production_cases + edge_cases,
        "poison_checks": 2,
        "reference": "old-gpu-kernel-full-output",
        "cpu_flag_sample_tiles": 32,
        "row_sample_seed": ROW_SAMPLE_SEED,
        "shards": 2,
        "shard_assignment": "case-id-popcount-modulo",
        "timing_cases": timing_cases,
        "launch_timings": launch_timings,
    })
}

#[test]
fn cpu_production_matrix_covers_registered_entries() {
    let matrix = production_cases();
    let covered: BTreeSet<_> = matrix
        .iter()
        .flat_map(|case| case.candidate.symbols())
        .collect();
    let compiled: BTreeSet<_> = f32_split::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .flat_map(|(_, spec)| spec.entries.into_iter().map(|entry| entry.symbol))
        .collect();
    assert_eq!(
        covered, compiled,
        "every compiled production entry must have bitwise comparisons"
    );
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            if check_ranks(g, ranks).is_err() {
                continue;
            }
            let backend = register_f32_projections(Sm120::new(), g, ranks).unwrap();
            let head = f32_head::entry(g, ranks)
                .map(Entry::with_split_decode)
                .unwrap();
            for entry in f32_split::entries(g, ranks).unwrap() {
                let entry = f32_prefill::serving_entry(g, entry).unwrap();
                let case = matrix.iter().find(|case| case.candidate == entry).unwrap();
                let maximum = if entry == head {
                    bundle_deepseek41::buffers::logits_rows(
                        &bundle_deepseek41::shape::topologies()
                            .into_iter()
                            .find(|shape| shape.geometry.id == g.id && shape.ranks() == ranks)
                            .unwrap(),
                    )
                } else {
                    entry.query().m_capacity
                };
                assert!(
                    case.rows.contains(&maximum),
                    "{} lacks its maximum {maximum} rows",
                    entry.symbol()
                );
                for rows in sampled_rows()
                    .into_iter()
                    .chain(entry.prefill_buckets().iter().map(|&(rows, _)| rows))
                {
                    assert!(
                        case.rows.contains(&rows),
                        "{} lacks {rows} rows",
                        entry.symbol()
                    );
                }
                assert!(
                    fold_cubin_for_symbol(case.old.symbol()).is_some(),
                    "locked baseline is missing"
                );
                // The split-family matrix keeps its own compiled-entry coverage.
                // Serving also advertises the independently qualified head variants.
                let routed = head_last::head(
                    g,
                    ranks,
                    projection_reuse::head(g, ranks, entry.clone()).unwrap(),
                )
                .unwrap();
                for symbol in routed.runtime_symbols() {
                    assert!(
                        fold_cubin_for_symbol(&symbol).is_some(),
                        "registered symbol is missing"
                    );
                }
                for &rows in &case.rows {
                    let op = backend
                        .projection(&cases::operation(&entry), entry.usage(), Need::rows(rows))
                        .unwrap()
                        .handle;
                    assert_eq!(
                        op.info()
                            .symbols
                            .iter()
                            .map(|(symbol, _)| symbol.clone())
                            .collect::<Vec<_>>(),
                        routed.runtime_symbols()
                    );
                }
            }
        }
    }
    println!("\nF32_MATRIX {}", coverage(&matrix));
}

fn edge(n: u32, k: u32, weight: WeightFormat) -> Entry {
    let q = GemmQuery {
        m_capacity: 256,
        n,
        k,
        weight,
        activation: ActivationFormat::F32,
        output: DType::F32,
        epilogues: Epilogues::default(),
        grouped: None,
    };
    Entry::new(
        q,
        OperationUse {
            phase: ExecutionPhase::Prefill,
            capacity: 256,
            graph: false,
        },
        &format!(
            "chop_f32_edge_n{n}_k{k}_{}",
            if weight == WeightFormat::F32 {
                "raw"
            } else {
                "bf16"
            }
        ),
    )
    .unwrap()
}

pub(super) fn edges() -> Vec<Entry> {
    let mut entries: Vec<_> = [1, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513]
        .into_iter()
        .map(|k| edge(17, k, WeightFormat::Bf16))
        .collect();
    entries.extend([
        edge(65536, 1, WeightFormat::Bf16),
        edge(17, 257, WeightFormat::F32),
    ]);
    entries
}

fn spec(legacy: bool) -> CubinSpec {
    let entries: Vec<_> = edges()
        .into_iter()
        .flat_map(|e| {
            [
                e.clone(),
                if legacy {
                    e.with_decode()
                } else {
                    e.with_split_decode()
                },
            ]
        })
        .collect();
    let mut spec = cubin_spec(if legacy { LEGACY_CUBIN } else { CUBIN }, &entries).unwrap();
    spec.sources_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    spec.include_dirs = vec!["crates/kernels-sm120/cuda".into()];
    spec.headers
        .push("#include \"crates/kernel-lab/cuda/probes/probe.cuh\"".into());
    let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    };
    spec.entries.extend([
        EntrySpec::new(
            "chop_probe_gate_v1".into(),
            vec![ptr("flag", "const uint32_t")],
            "chop::probe::gate(flag);".into(),
            32,
            1,
        ),
        EntrySpec::new(
            "chop_probe_flush_v1".into(),
            vec![
                ptr("buffer", "uint32_t"),
                EntryParam::U64 {
                    name: "words".into(),
                },
            ],
            "chop::probe::flush(buffer,words);".into(),
            256,
            1,
        ),
    ]);
    spec
}

fn artifacts() -> PathBuf {
    let out = PathBuf::from(
        std::env::var_os("CHOPINFER_F32_DECODE_ARTIFACTS")
            .expect("set CHOPINFER_F32_DECODE_ARTIFACTS outside the repository"),
    );
    std::fs::create_dir_all(&out).unwrap();
    let out = out.canonicalize().unwrap();
    assert!(
        !out.starts_with(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .canonicalize()
                .unwrap()
        )
    );
    out
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_DECODE_ARTIFACTS"]
fn cpu_build_edge_cases() {
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped, "CUDA compilation is required");
    // Both locked probes export the original reference symbols. Build
    // their independent modules separately; do not rename locked entries.
    for legacy in [false, true] {
        let spec = spec(legacy);
        let name = spec.name.clone();
        let built = kernel_build::build_in_families(
            &toolchain,
            &[spec],
            &artifacts(),
            &LockDir::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d"))
                .family(&name, [&name]),
            BuildOptions {
                emit_cargo_directives: false,
                profile_line_info: false,
                ..BuildOptions::from_env()
            },
            None,
        )
        .unwrap();
        assert_eq!(built.len(), 1);
        assert_eq!(
            built[0].entries.len(),
            edges().len() * if legacy { 6 } else { 4 } + 2
        );
        std::fs::copy(
            artifacts().join(kernel_build::report::BUILD_REPORT_FILE),
            artifacts().join(format!("{name}.report.json")),
        )
        .unwrap();
    }
}

fn probe_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        std::fs::read(artifacts().join(format!("{CUBIN}.cubin")))
            .expect("CPU-build edge cubin first")
    })
}

fn legacy_probe_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        std::fs::read(artifacts().join(format!("{LEGACY_CUBIN}.cubin")))
            .expect("CPU-build legacy edge cubin first")
    })
}

pub(super) fn timing_cubins() -> Cubins {
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", probe_bytes())]),
        &[(
            "probe_sm120",
            &["chop_probe_gate_v1", "chop_probe_flush_v1"],
        )],
    )
}

pub(super) fn prepared(entry: &Entry, rows: u32) -> Prepared<ProjectionBuffers<View>> {
    let backend = Sm120::new().with_f32_projections(
        F32ProjectionOffers::new(
            vec![entry.clone()],
            entry.runtime_symbols().into_iter().collect(),
            "0".repeat(64),
        )
        .unwrap(),
    );
    backend
        .projection(&cases::operation(entry), entry.usage(), Need::rows(rows))
        .unwrap()
        .handle
}

fn run_launches(
    session: &Session,
    modules: &[&Module],
    bindings: &launch::Bindings<'_>,
    launches: &[Launch],
    stream: &Stream,
) -> LabResult<()> {
    for l in launches {
        let function = modules
            .iter()
            .find_map(|module| module.function(&l.symbol).ok())
            .ok_or_else(|| format!("missing comparison entry {}", l.symbol))?;
        launch::launch(
            &session.context,
            &function,
            stream,
            l.plan,
            &l.arguments,
            bindings,
        )?;
    }
    Ok(())
}

pub(super) struct ComparisonWeights {
    candidate: Entry,
    modules: Vec<Module>,
    weight: runtime_cuda::DeviceBuffer,
    domain: runtime_cuda::DeviceBuffer,
    expected_flags: Vec<(usize, u32)>,
    initialized: std::cell::Cell<bool>,
}

fn sampled_flag_tiles(tiles: usize) -> Vec<usize> {
    if tiles <= 32 {
        return (0..tiles).collect();
    }
    let mut selected = BTreeSet::from([0, tiles - 1]);
    let mut seed = 0xa731_8539_u32;
    while selected.len() < 32 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        selected.insert(seed as usize % tiles);
    }
    selected.into_iter().collect()
}

#[test]
fn cpu_samples_keep_edges_and_bound_large_fixture_work() {
    let rows = sampled_rows();
    assert_eq!(rows.len(), BUCKET_ROWS.len() + 4);
    assert!(BUCKET_ROWS.iter().all(|r| rows.contains(r)));
    assert!(rows.iter().all(|&r| (1..=256).contains(&r)));
    assert_eq!(rows, sampled_rows());
    for tiles in [1, 2, 31, 32, 33, 4096, 8080] {
        let sample = sampled_flag_tiles(tiles);
        assert_eq!(sample.len(), tiles.min(32));
        assert_eq!(sample[0], 0);
        assert_eq!(sample.last(), Some(&(tiles - 1)));
        assert!(sample.windows(2).all(|p| p[0] < p[1]));
        assert_eq!(sample, sampled_flag_tiles(tiles));
    }
}

fn bf16_weight_code(value: f32) -> u16 {
    let bits = value.to_bits();
    if bits & 0xffff == 0 {
        // Already a BF16 code, including signaling NaNs. A numeric
        // conversion would quiet those payloads and lose edge coverage.
        (bits >> 16) as u16
    } else {
        f32_to_bf16(value)
    }
}

#[test]
fn cpu_weight_codes_preserve_all_bf16_payloads() {
    for bits in 0..=u16::MAX {
        assert_eq!(bf16_weight_code(bf16_to_f32(bits)), bits, "{bits:04x}");
    }
}

impl ComparisonWeights {
    fn new(session: &Session, old: &Entry, b: &[f32]) -> LabResult<Self> {
        Self::routed(session, old, old.clone().with_split_decode(), b)
    }

    fn routed(session: &Session, old: &Entry, new: Entry, b: &[f32]) -> LabResult<Self> {
        let legacy = if old.uses_cached_weights() {
            old.entry_for_rows(1)
        } else {
            old.clone().with_decode()
        };
        let mut modules = [old, &new, &legacy]
            .into_iter()
            .map(|entry| {
                let bytes = fold_cubin_for_symbol(entry.symbol())
                    .map(|(_, bytes)| bytes)
                    .unwrap_or_else(|| {
                        if entry.uses_cached_weights() && !entry.uses_fused_split() {
                            legacy_probe_bytes()
                        } else {
                            probe_bytes()
                        }
                    });
                session.load_cubin(bytes)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !new.prefill_buckets().is_empty() {
            for features in [32, 64] {
                let tiled = old
                    .clone()
                    .with_prefill_tiles(features)
                    .map_err(|e| e.to_string())?;
                let bytes = fold_cubin_for_symbol(tiled.symbol())
                    .ok_or("missing serving prefill cubin")?
                    .1;
                let module = session.load_cubin(bytes)?;
                module
                    .function(tiled.symbol())
                    .map_err(|e| e.to_string())?
                    .set_max_dynamic_shared_memory(
                        tiled
                            .prefill_plan(1)
                            .map_err(|e| e.to_string())?
                            .dynamic_shared_bytes,
                    )
                    .map_err(|e| e.to_string())?;
                modules.push(module);
            }
        }
        if let Some(&(rows, _)) = new.blocked_prefill_buckets().first() {
            let blocked = new.entry_for_rows(rows);
            let bytes = fold_cubin_for_symbol(blocked.symbol())
                .ok_or("missing blocked prefill cubin")?
                .1;
            let module = session.load_cubin(bytes)?;
            module
                .function(blocked.symbol())
                .map_err(|e| e.to_string())?
                .set_max_dynamic_shared_memory(
                    blocked
                        .prefill_plan(rows)
                        .map_err(|e| e.to_string())?
                        .dynamic_shared_bytes,
                )
                .map_err(|e| e.to_string())?;
            modules.push(module);
        }
        // Routed candidates may launch entries from several locked cubins.
        for symbol in old
            .runtime_symbols()
            .into_iter()
            .chain(new.runtime_symbols())
        {
            if modules.iter().all(|m| m.function(&symbol).is_err()) {
                let (_, bytes) = fold_cubin_for_symbol(&symbol)
                    .ok_or_else(|| format!("missing routed comparison cubin for {symbol}"))?;
                modules.push(session.load_cubin(bytes)?);
            }
        }
        Self::with_modules(session, old, new, b, modules)
    }

    pub(super) fn prefill(
        session: &Session,
        old: &Entry,
        features: u32,
        b: &[f32],
        bytes: [&[u8]; 2],
    ) -> LabResult<Self> {
        let new = old
            .clone()
            .with_prefill_tiles(features)
            .map_err(|e| e.to_string())?;
        let modules = bytes
            .into_iter()
            .map(|bytes| session.load_cubin(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        modules[1]
            .function(new.symbol())
            .map_err(|e| e.to_string())?
            .set_max_dynamic_shared_memory(
                new.prefill_plan(1)
                    .map_err(|e| e.to_string())?
                    .dynamic_shared_bytes,
            )
            .map_err(|e| e.to_string())?;
        Self::with_modules(session, old, new, b, modules)
    }

    fn with_modules(
        session: &Session,
        old: &Entry,
        new: Entry,
        b: &[f32],
        modules: Vec<Module>,
    ) -> LabResult<Self> {
        let weight = if old.query().weight == WeightFormat::Bf16 {
            session.upload_bf16(&b.iter().copied().map(bf16_weight_code).collect::<Vec<_>>())?
        } else {
            session.upload_f32(b)?
        };
        let domain = session.allocate(new.weight_domain_bytes() as usize)?;
        // Keep the CPU predicate exhaustive for edge fixtures, but bounded
        // for large production weights. Full outputs always use the old GPU
        // kernel, including every tile outside this independent CPU sample.
        let tile_width = old.query().k as usize * 16;
        let tiles = b.len().div_ceil(tile_width);
        let sampled = if b.len() <= 1 << 20 {
            (0..tiles).collect()
        } else {
            sampled_flag_tiles(tiles)
        };
        let expected_flags = sampled
            .into_iter()
            .map(|tile| {
                (
                    tile,
                    cases::weight_domains(
                        &b[tile * tile_width..b.len().min((tile + 1) * tile_width)],
                        old.query().k as usize,
                        old.query().weight,
                    )[0],
                )
            })
            .collect();
        Ok(Self {
            candidate: new,
            modules,
            weight,
            domain,
            expected_flags,
            initialized: false.into(),
        })
    }
}

pub(super) fn check_poison_expansion(session: &Session) -> LabResult<()> {
    // Cross the host seed size with unaligned boundaries. Read back every
    // byte before any tested projection can mask a broken device fill.
    let bytes = (16 << 20) + 137;
    let buffer = session.allocate(bytes)?;
    poison::fill_all(session, &buffer, &[0x36])?;
    let range = 7..bytes - 9;
    poison::fill(
        session,
        &buffer,
        std::slice::from_ref(&range),
        &[poison::POISON],
    )?;
    let actual = poison::read(session, &buffer)?;
    for (at, &byte) in actual.iter().enumerate() {
        let expected = if range.contains(&at) {
            poison::POISON
        } else {
            0x36
        };
        if byte != expected {
            return Err(format!(
                "poison expansion differs at byte {at}: {byte} != {expected}"
            ));
        }
    }
    println!("F32_POISON device_expansion bytes={bytes} PASS");
    Ok(())
}

pub(super) fn compare(
    session: &Session,
    old: &Entry,
    rows: u32,
    a: &[f32],
    weights: &ComparisonWeights,
    case: &HostCase,
    measure: bool,
) -> LabResult<()> {
    let fixture_ms = case.started.elapsed().as_secs_f64() * 1000.;
    let setup_start = Instant::now();
    let label = &case.label;
    let new = weights.candidate.clone();
    let q = old.query();
    let modules = &weights.modules;
    let weight = &weights.weight;
    let domain = &weights.domain;
    let x = session.upload_f32(a)?;
    let output = [
        session.allocate(rows as usize * q.n as usize * 4)?,
        session.allocate(rows as usize * q.n as usize * 4)?,
    ];
    let live = session.allocate(4)?;
    let workspace = session.allocate(
        new.runtime_workspace_bytes(rows)
            .max(old.runtime_workspace_bytes(rows)) as usize,
    )?;
    let status = session.allocate(rows as usize * 4)?;
    let bindings = launch::Bindings {
        buffers: BTreeMap::from([
            (1, &x),
            (2, weight),
            (3, &output[0]),
            (4, &output[1]),
            (5, &live),
            (6, domain),
            (7, &workspace),
            (8, &status),
        ]),
    };
    let view = |id| View {
        handle: id,
        bytes: bindings.buffers[&id].byte_len() as u64,
    };
    let buffers = |id| ProjectionBuffers {
        activation: view(1),
        weights: QuantizedWeights {
            codes: view(2),
            local_scales: None,
            global_scales: None,
        },
        output: view(id),
        live_rows: view(5),
        residual: None,
        activation_global: None,
        routing: None,
        scratch: ProjectionScratch {
            weight_domain: Some(view(6)),
            partial: View::EMPTY,
            semaphores: View::EMPTY,
            unused: View::EMPTY,
            temporary: None,
            gathered: None,
            codes: None,
            scales: None,
            activation_global: None,
            row_amax: None,
            output_codes: None,
            output_scales: None,
            quantizer_status: Some(view(8)),
            quantizer_status_row_offset: 0,
        },
        row_offset: 0,
        shard_rows: 0,
    };
    let ops = [prepared(old, rows), prepared(&new, rows)];
    if !weights.initialized.get() {
        let init = ops[1].initialize(&buffers(4)).map_err(|e| e.to_string())?;
        poison::poison(session, &[domain])?;
        run_launches(
            session,
            &modules.iter().collect::<Vec<_>>(),
            &bindings,
            &init,
            &session.stream,
        )?;
        let flags = poison::read(session, domain)?;
        for &(tile, expected) in &weights.expected_flags {
            let actual = u32::from_le_bytes(flags[tile * 4..tile * 4 + 4].try_into().unwrap());
            if actual != expected {
                return Err(format!(
                    "{label}: cached weight-domain flags differ at tile {tile}"
                ));
            }
        }
        weights.initialized.set(true);
    }
    let launches = [
        ops[0]
            .launches(&buffers(3), view(7), Counts::rows(rows))
            .unwrap(),
        ops[1]
            .launches(&buffers(4), view(7), Counts::rows(rows))
            .unwrap(),
    ];
    let runtime_modules = [
        modules.iter().collect::<Vec<_>>(),
        modules.iter().collect::<Vec<_>>(),
    ];
    let graphs = [
        session
            .capture(|s| run_launches(session, &runtime_modules[0], &bindings, &launches[0], s))?,
        session
            .capture(|s| run_launches(session, &runtime_modules[1], &bindings, &launches[1], s))?,
    ];
    let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.;
    let checks_start = Instant::now();
    let mut poison_time = Duration::ZERO;
    // Every routed row count runs eager and captured. Timed buckets and
    // edge witnesses also repeat after inactive and zero-live invocations.
    let lives = if measure || !label.starts_with("chop_") {
        vec![rows, rows.saturating_sub(1), 0, rows]
    } else {
        vec![rows]
    };
    for live_count in lives {
        live.copy_from_bytes(&live_count.to_le_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        for graph in [false, true] {
            let poison_start = Instant::now();
            poison::poison(session, &[&output[0], &output[1], &workspace, &status])?;
            poison_time += poison_start.elapsed();
            for arm in 0..2 {
                if graph {
                    graphs[arm]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                } else {
                    run_launches(
                        session,
                        &runtime_modules[arm],
                        &bindings,
                        &launches[arm],
                        &session.stream,
                    )?;
                }
            }
            let reference = poison::read(session, &output[0])?;
            let actual = poison::read(session, &output[1])?;
            if actual != reference {
                let at = actual
                    .chunks_exact(4)
                    .zip(reference.chunks_exact(4))
                    .position(|(a, b)| a != b)
                    .unwrap();
                return Err(format!(
                    "{label}: rows={rows} live={live_count} graph={graph} bits differ at {at}: {:?} != {:?}",
                    &actual[at * 4..at * 4 + 4],
                    &reference[at * 4..at * 4 + 4]
                ));
            }
            let untouched = live_count as usize * q.n as usize * 4;
            if actual[untouched..].iter().any(|&b| b != poison::POISON)
                || poison::read(session, &status)?
                    .iter()
                    .any(|&b| b != poison::POISON)
            {
                return Err(format!(
                    "{label}: inactive output or caller row status was overwritten"
                ));
            }
        }
    }
    let comparison_ms = (checks_start.elapsed() - poison_time).as_secs_f64() * 1000.;
    let timing_start = Instant::now();
    if measure {
        let paired =
            kernel_harness::time_ab(session, 2, 7, [EvictionMode::Write; 2], |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        println!(
            "F32_TIMING {label} rows={rows} old_us={} new_us={} ratio={} floor_us={} floor_ratio={}",
            paired.arms[0].median_ms * 1000.,
            paired.arms[1].median_ms * 1000.,
            paired.paired_ratio,
            old.weight_bytes() as f64 / BYTES_PER_US,
            f64::from(paired.arms[1].median_ms) * 1000.
                / (old.weight_bytes() as f64 / BYTES_PER_US)
        );
        // Each stage is measured separately as well as the full sequence.
        for (arm, entries) in launches.iter().enumerate() {
            for (index, l) in entries.iter().enumerate() {
                let graph = session.capture(|s| {
                    run_launches(
                        session,
                        &runtime_modules[arm],
                        &bindings,
                        std::slice::from_ref(l),
                        s,
                    )
                })?;
                let timing = kernel_harness::time_prepared(
                    session,
                    2,
                    7,
                    EvictionMode::Write,
                    |s| {
                        run_launches(
                            session,
                            &runtime_modules[arm],
                            &bindings,
                            &entries[..index],
                            s,
                        )
                    },
                    |s| graph.launch(s).map_err(|e| e.to_string()),
                )?;
                println!(
                    "F32_LAUNCH {label} rows={rows} arm={arm} symbol={} us={}",
                    l.symbol,
                    timing.median_ms * 1000.
                );
            }
        }
    }
    println!(
        "F32_HOST {label} rows={rows} fixture_ms={fixture_ms} setup_ms={setup_ms} poison_ms={} comparison_ms={comparison_ms} timing_ms={} total_ms={}",
        poison_time.as_secs_f64() * 1000.,
        timing_start.elapsed().as_secs_f64() * 1000.,
        case.started.elapsed().as_secs_f64() * 1000.
    );
    println!("F32_BITWISE {label} rows={rows} n={} k={} PASS", q.n, q.k);
    Ok(())
}

pub(super) fn random_operands(rows: u32, n: u32, k: u32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut value = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f32::from_bits((state & 0x807f_ffff) | (126 << 23))
    };
    (
        (0..rows * k).map(|_| value()).collect(),
        (0..n * k)
            .map(|_| bf16_to_f32(f32_to_bf16(value())))
            .collect(),
    )
}

pub(super) struct Shard {
    pub(super) ordinal: usize,
    pub(super) count: usize,
    pub(super) next: usize,
}
impl Shard {
    pub(super) fn from_env() -> LabResult<Self> {
        match (
            std::env::var("CHOPINFER_F32_SHARD"),
            std::env::var("CHOPINFER_F32_SHARDS"),
        ) {
            (Err(std::env::VarError::NotPresent), Err(std::env::VarError::NotPresent)) => {
                Ok(Self {
                    ordinal: 0,
                    count: 1,
                    next: 0,
                })
            }
            (Ok(ordinal), Ok(count)) => {
                let ordinal = ordinal.parse::<usize>().map_err(|e| e.to_string())?;
                let count = count.parse::<usize>().map_err(|e| e.to_string())?;
                if count != 2 || ordinal >= count {
                    return Err("expected shard 0 or 1 of 2".into());
                }
                Ok(Self {
                    ordinal,
                    count,
                    next: 0,
                })
            }
            _ => Err("set both exact-F32 shard variables or neither".into()),
        }
    }
    pub(super) fn take(&mut self) -> Option<usize> {
        let id = self.next;
        self.next += 1;
        (id.count_ones() as usize % self.count == self.ordinal).then_some(id)
    }
}

#[test]
fn cpu_shards_cover_every_case_once() {
    let matrix = production_cases();
    let total = coverage(&matrix)["bitwise_cases"].as_u64().unwrap() as usize;
    let mut seen = BTreeSet::new();
    for ordinal in 0..2 {
        let mut shard = Shard {
            ordinal,
            count: 2,
            next: 0,
        };
        for case in &matrix {
            let mut timed_cases = 0;
            for &rows in &case.rows {
                if let Some(id) = shard.take() {
                    assert!(seen.insert(id));
                    timed_cases += usize::from(timed(rows));
                }
            }
            assert!(
                timed_cases >= 2,
                "every entry has paired timings on both GPUs"
            );
        }
        while shard.next < total {
            if let Some(id) = shard.take() {
                assert!(seen.insert(id));
            }
        }
    }
    assert_eq!(seen, (0..total).collect());
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_DECODE_ARTIFACTS"]
fn gpu_bitwise_shapes_and_edges() {
    // Libtest can print its test prefix without a newline under one test thread.
    println!();
    let mut shard = Shard::from_env().unwrap();
    kernel_harness::gpu_test_with(timing_cubins(), 12 << 30, |session| {
        check_poison_expansion(session)?;
        for case in production_cases() {
            let old = &case.old;
            let q = old.query();
            let weights_start = Instant::now();
            let (_, b) = random_operands(0, q.n, q.k, 0x7813_9231);
            let weights = ComparisonWeights::routed(session, old, case.candidate.clone(), &b)?;
            drop(b);
            println!(
                "F32_SETUP {} total_ms={}",
                old.symbol(),
                weights_start.elapsed().as_secs_f64() * 1000.
            );
            for rows in case.rows {
                let Some(id) = shard.take() else { continue };
                let host_case = HostCase::new(format!("{} case={id}", old.symbol()));
                let (a, _) = random_operands(rows, 0, q.k, 0x7813_9231 + rows);
                compare(session, old, rows, &a, &weights, &host_case, timed(rows))?;
            }
        }
        for old in edges() {
            let q = old.query();
            if q.n == 65536 {
                let Some(id) = shard.take() else { continue };
                let host_case = HostCase::new(format!("all_bf16_codes case={id}"));
                let b: Vec<_> = (0..=u16::MAX).map(bf16_to_f32).collect();
                compare(
                    session,
                    &old,
                    8,
                    &[
                        0.,
                        -0.,
                        1.,
                        -1.,
                        f32::from_bits(1),
                        f32::MAX,
                        f32::INFINITY,
                        f32::NAN,
                    ],
                    &ComparisonWeights::new(session, &old, &b)?,
                    &host_case,
                    false,
                )?;
                continue;
            }
            for rows in EDGE_ROWS {
                for witness in EDGE_WITNESSES {
                    let Some(id) = shard.take() else { continue };
                    let host_case = HostCase::new(format!("{witness:?} case={id}"));
                    let (a, b) = cases::operands(&old, witness);
                    compare(
                        session,
                        &old,
                        rows,
                        &a[..rows as usize * q.k as usize],
                        &ComparisonWeights::new(session, &old, &b)?,
                        &host_case,
                        false,
                    )?;
                }
                for seed in 1..=RANDOM_CASES {
                    let Some(id) = shard.take() else { continue };
                    let host_case = HostCase::new(format!("random_and_late_fallback case={id}"));
                    let (mut a, mut b) = random_operands(rows, q.n, q.k, seed * 123_457);
                    if q.weight == WeightFormat::F32 {
                        for (index, value) in b.iter_mut().enumerate() {
                            *value = f32::from_bits(value.to_bits() | (index as u32 & 0xffff));
                        }
                    }
                    // One late exceptional value must govern every chain of
                    // its row/tile, including the earlier ordinary values.
                    if seed == 2 {
                        a[q.k as usize - 1] = f32::from_bits(1);
                    }
                    if seed == 3 {
                        b[16 * q.k as usize - 1] = f32::from_bits(0x0080_0000);
                    }
                    if seed == 4 {
                        a[q.k as usize - 1] = f32::from_bits(0x7f80_0001);
                    }
                    compare(
                        session,
                        &old,
                        rows,
                        &a,
                        &ComparisonWeights::new(session, &old, &b)?,
                        &host_case,
                        false,
                    )?;
                }
            }
            if q.k == 129 {
                let Some(id) = shard.take() else { continue };
                let host_case = HostCase::new(format!(
                    "every_exponent_sign_and_mantissa_boundary case={id}"
                ));
                let (mut a, b) = random_operands(32, q.n, q.k, 79);
                let mut at = 0;
                for sign in [0, 0x8000_0000] {
                    for exponent in 0..256 {
                        for mantissa in [0, 1, 0x7f, 0x80, 0x7fff, 0x8000, 0x7f0000, 0x7fffff] {
                            a[at] = f32::from_bits(sign | exponent << 23 | mantissa);
                            at += 1;
                        }
                    }
                }
                compare(
                    session,
                    &old,
                    32,
                    &a,
                    &ComparisonWeights::new(session, &old, &b)?,
                    &host_case,
                    false,
                )?;
            }
        }
        assert_eq!(
            shard.next,
            coverage(&production_cases())["bitwise_cases"]
                .as_u64()
                .unwrap() as usize
        );
        Ok(())
    })
    .unwrap()
    .expect("GPU comparison must execute on a selected idle GPU");
}
