//! Candidate-only prefill tiles and their sharded bitwise gate.
use super::f32_decode::{
    ComparisonWeights, HostCase, Shard, check_poison_expansion, compare, edges, prepared,
    random_operands,
};
use bundle_deepseek41::sm120::{
    f32_prefill as registration, fold_cubin_for_symbol, register_f32_projections,
};
use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, LockDir, Toolchain};
use kernel_core::backend::{Backend, Need};
use kernel_core::codec::bf16_to_f32;
use kernel_core::contract::WeightFormat;
use kernel_harness::kit::cases::f32_projection::{self as cases, Witness};
use kernel_harness::{Cubins, LabResult};
use kernels_sm120::{
    backend::Sm120,
    f32_projection::{Entry, cubin_spec},
};
use model_deepseek41_contract::{admission::SUPPORTED_RANKS, geometry::GEOMETRIES};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Instant,
};

const EDGE_CUBIN: &str = "f32_prefill_edges";
const ROWS: [u32; 11] = [257, 511, 512, 513, 2047, 2048, 2049, 4095, 4096, 8191, 8192];
const EDGE_ROWS: [u32; 2] = [33, 65];
const WITNESSES: [Witness; 11] = [
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

fn label(old: &Entry, tile: u32) -> String {
    format!("{}_tile{tile}", old.symbol())
}

fn artifacts() -> PathBuf {
    let out = PathBuf::from(
        std::env::var_os("CHOPINFER_F32_PREFILL_ARTIFACTS")
            .expect("set CHOPINFER_F32_PREFILL_ARTIFACTS outside the repository"),
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

fn edge_spec() -> CubinSpec {
    let entries: Vec<_> = edges()
        .into_iter()
        .flat_map(|e| {
            [
                e.clone(),
                e.clone().with_prefill_tiles(32).unwrap(),
                e.with_prefill_tiles(64).unwrap(),
            ]
        })
        .collect();
    let mut spec = cubin_spec(EDGE_CUBIN, &entries).unwrap();
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

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_PREFILL_ARTIFACTS"]
fn cpu_build_prefill_candidates() {
    let mut specs = registration::cubins(GEOMETRIES).unwrap();
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d");
    let locks = LockDir::new(&dir)
        .family(registration::FAMILY, specs.iter().map(|s| &s.name))
        .family(EDGE_CUBIN, [EDGE_CUBIN]);
    specs.push(edge_spec());
    let options = BuildOptions {
        emit_cargo_directives: false,
        profile_line_info: false,
        ..BuildOptions::from_env()
    };
    if options.lock_update {
        assert!(
            !dir.join(format!("{}.lock", registration::FAMILY)).exists(),
            "cannot replace an existing lock"
        );
        assert!(
            !dir.join(format!("{EDGE_CUBIN}.lock")).exists(),
            "cannot replace an existing edge lock"
        );
    }
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let built =
        kernel_build::build_in_families(&toolchain, &specs, &artifacts(), &locks, options, None)
            .unwrap();
    assert_eq!(built.len(), GEOMETRIES.len() * 2 + 1);
    assert_eq!(built.last().unwrap().entries.len(), edges().len() * 7 + 2);
}

fn coverage() -> serde_json::Value {
    let production: Vec<_> = GEOMETRIES.iter().flat_map(|g| registration::TILE_FEATURES.map(move |tile| (g,tile)))
        .flat_map(|(g,tile)| registration::baseline_entries(g).unwrap().into_iter().map(move |old| {
            serde_json::json!({"baseline":label(&old,tile),"reference":old.symbol(),"candidate":old.clone().with_prefill_tiles(tile).unwrap().symbol(),
                "tile":tile,"n":old.query().n,"k":old.query().k,"weight_bytes":old.weight_bytes(),"rows":ROWS,"timing_rows":ROWS})
        })).collect();
    let edge_cases: usize = edges()
        .iter()
        .map(|old| {
            if old.query().n == 65536 {
                2
            } else {
                2 * (EDGE_ROWS.len() * (WITNESSES.len() + 4) + usize::from(old.query().k == 129))
            }
        })
        .sum();
    let production_cases = production.len() * ROWS.len();
    serde_json::json!({"production":production,"production_cases":production_cases,"edge_cases":edge_cases,
        "bitwise_cases":production_cases+edge_cases,"host_cases":production_cases+edge_cases,"poison_checks":2,"timing_cases":production_cases,"launch_timings":production_cases*3,
        "shards":2,"bytes_per_us":1600000,"winning_ratio":0.97,"serving_promotion":false})
}

#[test]
fn cpu_prefill_matrix_covers_candidates_and_serving_buckets() {
    let compiled: BTreeSet<_> = registration::cubins(GEOMETRIES)
        .unwrap()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    let mut covered = BTreeSet::new();
    for g in GEOMETRIES {
        for old in registration::baseline_entries(g).unwrap() {
            for tile in registration::TILE_FEATURES {
                let candidate = old.clone().with_prefill_tiles(tile).unwrap();
                covered.extend(candidate.symbols());
                assert!(
                    fold_cubin_for_symbol(candidate.symbol()).is_some(),
                    "qualified candidate is missing from the serving build"
                );
                for rows in ROWS {
                    let op = prepared(&candidate, rows);
                    assert_eq!(op.info().workspace, candidate.prefill_workspace_bytes(rows));
                    assert!(op.info().symbols.contains(&(
                        candidate.symbol().into(),
                        candidate.prefill_plan(rows).unwrap().dynamic_shared_bytes
                    )));
                }
            }
            for &ranks in SUPPORTED_RANKS {
                let backend = register_f32_projections(Sm120::new(), g, ranks).unwrap();
                let op = backend
                    .projection(&cases::operation(&old), old.usage(), Need::rows(8192))
                    .unwrap();
                let selected =
                    registration::serving_entry(g, old.clone().with_split_decode()).unwrap();
                assert_eq!(
                    op.handle.info().workspace,
                    selected.decode_workspace_bytes(8192)
                );
                for &(rows, features) in selected.prefill_buckets() {
                    let tiled = old.clone().with_prefill_tiles(features).unwrap();
                    assert_eq!(selected.entry_for_rows(rows), tiled);
                    assert!(op.handle.info().symbols.contains(&(
                        tiled.symbol().into(),
                        tiled.prefill_plan(rows).unwrap().dynamic_shared_bytes
                    )));
                }
                for rows in [258, 510, 514, 2046, 2050, 4094, 4097, 8190] {
                    assert!(!selected.entry_for_rows(rows).uses_prefill_tiles());
                }
            }
        }
    }
    assert_eq!(covered, compiled);
    assert_eq!(registration::BUCKETS.len(), 55);
    let unique: BTreeSet<_> = registration::BUCKETS
        .iter()
        .map(|b| (b.n, b.k, b.rows))
        .collect();
    assert_eq!(unique.len(), 55);
    let total = coverage()["bitwise_cases"].as_u64().unwrap() as usize;
    let mut seen = BTreeSet::new();
    for ordinal in 0..2 {
        let mut shard = Shard {
            ordinal,
            count: 2,
            next: 0,
        };
        for _ in 0..total {
            if let Some(id) = shard.take() {
                assert!(seen.insert(id));
            }
        }
    }
    assert_eq!(seen, (0..total).collect());
    println!("\nF32_PREFILL_MATRIX {}", coverage());
}

fn probe_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| std::fs::read(artifacts().join(format!("{EDGE_CUBIN}.cubin"))).unwrap())
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

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_PREFILL_ARTIFACTS"]
fn gpu_prefill_bitwise_and_timings() {
    println!();
    let mut shard = Shard::from_env().unwrap();
    kernel_harness::gpu_test_with(timing_cubins(), 4 << 30, |session| -> LabResult<()> {
        check_poison_expansion(session)?;
        for g in GEOMETRIES {
            for tile in registration::TILE_FEATURES {
                let bytes = std::fs::read(
                    artifacts().join(format!("{}.cubin", registration::cubin_name(g, tile))),
                )
                .map_err(|e| e.to_string())?;
                for old in registration::baseline_entries(g)? {
                    let q = old.query();
                    let weights_start = Instant::now();
                    let (_, b) = random_operands(0, q.n, q.k, 711_359);
                    let baseline = fold_cubin_for_symbol(old.symbol())
                        .ok_or("missing baseline")?
                        .1;
                    let weights =
                        ComparisonWeights::prefill(session, &old, tile, &b, [baseline, &bytes])?;
                    drop(b);
                    println!(
                        "F32_SETUP {} total_ms={}",
                        label(&old, tile),
                        weights_start.elapsed().as_secs_f64() * 1000.
                    );
                    for rows in ROWS {
                        let Some(id) = shard.take() else { continue };
                        let host_case = HostCase::new(format!("{} case={id}", label(&old, tile)));
                        let (a, _) = random_operands(rows, 0, q.k, 123_457 + rows);
                        compare(session, &old, rows, &a, &weights, &host_case, true)?;
                    }
                }
            }
        }
        for old in edges() {
            let q = old.query();
            for tile in registration::TILE_FEATURES {
                if q.n == 65536 {
                    let Some(id) = shard.take() else { continue };
                    let host_case = HostCase::new(format!("all_bf16_codes_tile{tile} case={id}"));
                    let b: Vec<_> = (0..=u16::MAX).map(bf16_to_f32).collect();
                    let weights =
                        ComparisonWeights::prefill(session, &old, tile, &b, [probe_bytes(); 2])?;
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
                        &weights,
                        &host_case,
                        false,
                    )?;
                    continue;
                }
                for rows in EDGE_ROWS {
                    for witness in WITNESSES {
                        let Some(id) = shard.take() else { continue };
                        let host_case = HostCase::new(format!("{witness:?}_tile{tile} case={id}"));
                        let (a, b) = cases::operands(&old, witness);
                        let weights = ComparisonWeights::prefill(
                            session,
                            &old,
                            tile,
                            &b,
                            [probe_bytes(); 2],
                        )?;
                        compare(
                            session,
                            &old,
                            rows,
                            &a[..rows as usize * q.k as usize],
                            &weights,
                            &host_case,
                            false,
                        )?;
                    }
                    for seed in 1..=4 {
                        let Some(id) = shard.take() else { continue };
                        let host_case =
                            HostCase::new(format!("late_fallback_tile{tile} case={id}"));
                        let (mut a, mut b) = random_operands(rows, q.n, q.k, seed * 123_457);
                        if q.weight == WeightFormat::F32 {
                            for (i, v) in b.iter_mut().enumerate() {
                                *v = f32::from_bits(v.to_bits() | (i as u32 & 0xffff));
                            }
                        }
                        if seed == 2 {
                            a[q.k as usize - 1] = f32::from_bits(1);
                        }
                        if seed == 3 {
                            b[16 * q.k as usize - 1] = f32::from_bits(0x0080_0000);
                        }
                        if seed == 4 {
                            a[q.k as usize - 1] = f32::from_bits(0x7f80_0001);
                        }
                        let weights = ComparisonWeights::prefill(
                            session,
                            &old,
                            tile,
                            &b,
                            [probe_bytes(); 2],
                        )?;
                        compare(session, &old, rows, &a, &weights, &host_case, false)?;
                    }
                }
                if q.k == 129 {
                    let Some(id) = shard.take() else { continue };
                    let host_case =
                        HostCase::new(format!("exponent_boundaries_tile{tile} case={id}"));
                    let (mut a, b) = random_operands(32, q.n, q.k, 79);
                    let mut at = 0;
                    for sign in [0, 0x8000_0000] {
                        for exp in 0..256 {
                            for mantissa in [0, 1, 0x7f, 0x80, 0x7fff, 0x8000, 0x7f0000, 0x7fffff] {
                                a[at] = f32::from_bits(sign | exp << 23 | mantissa);
                                at += 1;
                            }
                        }
                    }
                    let weights =
                        ComparisonWeights::prefill(session, &old, tile, &b, [probe_bytes(); 2])?;
                    compare(session, &old, 32, &a, &weights, &host_case, false)?;
                }
            }
        }
        assert_eq!(
            shard.next,
            coverage()["bitwise_cases"].as_u64().unwrap() as usize
        );
        Ok(())
    })
    .unwrap()
    .expect("prefill GPU comparison must execute");
}
