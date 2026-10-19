//! Bitwise baseline comparisons and paired cold-weight launch timings.
use bundle_deepseek41::sm120::{
    f32_head, f32_split, fold_cubin_for_symbol, register_f32_projections,
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

const CUBIN: &str = "f32_split_edges";
const LEGACY_CUBIN: &str = "f32_decode_edges";
const TIMING_ROWS: [u32; 12] = [1, 2, 4, 8, 16, 32, 33, 48, 64, 128, 192, 256];
const MAX_SLOWDOWN_RATIO: f32 = 1.03;
const VERIFY_RATIO: f32 = 0.2;
// User-specified bandwidth roofline: 1.6 TB/s, with actual stored weights.
const BYTES_PER_US: f64 = 1_600_000.;
const EDGE_ROWS: [u32; 9] = [1, 8, 32, 33, 48, 64, 128, 192, 256];

struct ProductionCase {
    old: Entry,
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
                let maximum = if entry == head {
                    bundle_deepseek41::buffers::logits_rows()
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
            assert_eq!(old.clone().with_split_decode(), entry);
            let mut rows: Vec<_> = (1..=entry.decode_max_rows()).collect();
            if maximum > 256 {
                rows.push(2048);
            }
            rows.push(maximum);
            rows.sort_unstable();
            rows.dedup();
            ProductionCase { old, rows }
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
        .flat_map(|case| &case.rows)
        .filter(|&&rows| timed(rows))
        .map(|&rows| {
            if rows <= 32 {
                4
            } else if rows <= 256 {
                3
            } else {
                2
            }
        })
        .sum();
    serde_json::json!({
        "production": matrix.iter().map(|case| serde_json::json!({
            "baseline": case.old.symbol(),
            "candidate": case.old.clone().with_split_decode().symbol(),
            "n": case.old.query().n,
            "k": case.old.query().k,
            "weight_bytes": case.old.weight_bytes(),
            "f32_equivalent_weight_bytes": u64::from(case.old.query().n) * u64::from(case.old.query().k) * 4,
            "rows": case.rows,
            "timing_rows": case.rows.iter().copied().filter(|&rows| timed(rows)).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "maximum_slowdown_ratio": MAX_SLOWDOWN_RATIO,
        "verify_maximum_ratio": VERIFY_RATIO,
        "bytes_per_us": BYTES_PER_US,
        "production_cases": production_cases,
        "edge_cases": edge_cases,
        "bitwise_cases": production_cases + edge_cases,
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
        .flat_map(|case| case.old.clone().with_split_decode().symbols())
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
                let case = matrix
                    .iter()
                    .find(|case| case.old.clone().with_split_decode() == entry)
                    .unwrap();
                let maximum = if entry == head {
                    bundle_deepseek41::buffers::logits_rows()
                } else {
                    entry.query().m_capacity
                };
                assert!(
                    case.rows.contains(&maximum),
                    "{} lacks its maximum {maximum} rows",
                    entry.symbol()
                );
                for rows in 1..=entry.decode_max_rows() {
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
                for symbol in entry.runtime_symbols() {
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
                        entry.runtime_symbols()
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

fn edges() -> Vec<Entry> {
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

fn timing_cubins() -> Cubins {
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", probe_bytes())]),
        &[(
            "probe_sm120",
            &["chop_probe_gate_v1", "chop_probe_flush_v1"],
        )],
    )
}

fn prepared(entry: &Entry, rows: u32) -> Prepared<ProjectionBuffers<View>> {
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
    module: &Module,
    bindings: &launch::Bindings<'_>,
    launches: &[Launch],
    stream: &Stream,
) -> LabResult<()> {
    for l in launches {
        let function = module.function(&l.symbol).map_err(|e| e.to_string())?;
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

struct ComparisonWeights {
    modules: Vec<Module>,
    weight: runtime_cuda::DeviceBuffer,
    domain: runtime_cuda::DeviceBuffer,
    expected_flags: Vec<u8>,
    initialized: std::cell::Cell<bool>,
}
impl ComparisonWeights {
    fn new(session: &Session, old: &Entry, b: &[f32]) -> LabResult<Self> {
        let new = old.clone().with_split_decode();
        let legacy = old.clone().with_decode();
        let modules = [old, &new, &legacy]
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
            .collect::<Result<_, _>>()?;
        let weight = if old.query().weight == WeightFormat::Bf16 {
            session.upload_bf16(&b.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>())?
        } else {
            session.upload_f32(b)?
        };
        let domain = session.allocate(new.weight_domain_bytes() as usize)?;
        let expected_flags = cases::weight_domains(b, old.query().k as usize, old.query().weight)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        Ok(Self {
            modules,
            weight,
            domain,
            expected_flags,
            initialized: false.into(),
        })
    }
}

fn compare(
    session: &Session,
    old: &Entry,
    rows: u32,
    a: &[f32],
    weights: &ComparisonWeights,
    label: &str,
    measure: bool,
) -> LabResult<()> {
    let new = old.clone().with_split_decode();
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
    let workspace = session.allocate(new.decode_workspace_bytes(rows) as usize)?;
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
        run_launches(session, &modules[1], &bindings, &init, &session.stream)?;
        if poison::read(session, domain)? != weights.expected_flags {
            return Err(format!("{label}: cached weight-domain flags differ"));
        }
        weights.initialized.set(true);
    }
    let launches = [
        ops[0]
            .launches(&buffers(3), View::EMPTY, Counts::rows(rows))
            .unwrap(),
        ops[1]
            .launches(&buffers(4), view(7), Counts::rows(rows))
            .unwrap(),
    ];
    let runtime_modules = [
        &modules[0],
        if launches[1]
            .iter()
            .all(|launch| launch.symbol == old.symbol())
        {
            &modules[0]
        } else if rows <= 32 {
            &modules[2]
        } else {
            &modules[1]
        },
    ];
    let graphs = [
        session
            .capture(|s| run_launches(session, runtime_modules[0], &bindings, &launches[0], s))?,
        session
            .capture(|s| run_launches(session, runtime_modules[1], &bindings, &launches[1], s))?,
    ];
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
            poison::poison(session, &[&output[0], &output[1], &workspace, &status])?;
            for arm in 0..2 {
                if graph {
                    graphs[arm]
                        .launch(&session.stream)
                        .map_err(|e| e.to_string())?;
                } else {
                    run_launches(
                        session,
                        runtime_modules[arm],
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
                        runtime_modules[arm],
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
                            runtime_modules[arm],
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
    println!("F32_BITWISE {label} rows={rows} n={} k={} PASS", q.n, q.k);
    Ok(())
}

fn random_operands(rows: u32, n: u32, k: u32, seed: u32) -> (Vec<f32>, Vec<f32>) {
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

struct Shard {
    ordinal: usize,
    count: usize,
    next: usize,
}
impl Shard {
    fn from_env() -> LabResult<Self> {
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
    fn take(&mut self) -> Option<usize> {
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
fn gpu_bitwise_shapes_and_edges() {
    // Libtest can print its test prefix without a newline under one test thread.
    println!();
    let mut shard = Shard::from_env().unwrap();
    kernel_harness::gpu_test_with(timing_cubins(), 12 << 30, |session| {
        for case in production_cases() {
            let old = &case.old;
            let q = old.query();
            let (_, b) = random_operands(0, q.n, q.k, 0x7813_9231);
            let weights = ComparisonWeights::new(session, old, &b)?;
            drop(b);
            for rows in case.rows {
                let Some(id) = shard.take() else { continue };
                let (a, _) = random_operands(rows, 0, q.k, 0x7813_9231 + rows);
                compare(
                    session,
                    old,
                    rows,
                    &a,
                    &weights,
                    &format!("{} case={id}", old.symbol()),
                    timed(rows),
                )?;
            }
        }
        for old in edges() {
            let q = old.query();
            if q.n == 65536 {
                let Some(id) = shard.take() else { continue };
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
                    &format!("all_bf16_codes case={id}"),
                    false,
                )?;
                continue;
            }
            for rows in EDGE_ROWS {
                for witness in EDGE_WITNESSES {
                    let Some(id) = shard.take() else { continue };
                    let (a, b) = cases::operands(&old, witness);
                    compare(
                        session,
                        &old,
                        rows,
                        &a[..rows as usize * q.k as usize],
                        &ComparisonWeights::new(session, &old, &b)?,
                        &format!("{witness:?} case={id}"),
                        false,
                    )?;
                }
                for seed in 1..=RANDOM_CASES {
                    let Some(id) = shard.take() else { continue };
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
                        &format!("random_and_late_fallback case={id}"),
                        false,
                    )?;
                }
            }
            if q.k == 129 {
                let Some(id) = shard.take() else { continue };
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
                    &format!("every_exponent_sign_and_mantissa_boundary case={id}"),
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
