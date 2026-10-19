//! Bitwise baseline comparisons and paired cold-weight launch timings.
use bundle_deepseek41::sm120::{f32_head, fold_cubin_for_symbol};
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
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{Module, Stream};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const CUBIN: &str = "f32_decode_edges";

fn edge(n: u32, k: u32, weight: WeightFormat) -> Entry {
    let q = GemmQuery {
        m_capacity: 64,
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
            capacity: 64,
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

fn spec() -> CubinSpec {
    let entries: Vec<_> = edges()
        .into_iter()
        .flat_map(|e| [e.clone(), e.with_decode()])
        .collect();
    let mut spec = cubin_spec(CUBIN, &entries).unwrap();
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
    let spec = spec();
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped, "CUDA compilation is required");
    let built = kernel_build::build_in_families(
        &toolchain,
        &[spec],
        &artifacts(),
        &LockDir::new(Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d"))
            .family(CUBIN, [CUBIN]),
        BuildOptions {
            emit_cargo_directives: false,
            profile_line_info: false,
            ..BuildOptions::from_env()
        },
        None,
    )
    .unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), edges().len() * 6 + 2);
}

fn probe_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        std::fs::read(artifacts().join(format!("{CUBIN}.cubin")))
            .expect("CPU-build edge cubin first")
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
            entry.symbols().into_iter().collect(),
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

fn compare(
    session: &Session,
    old: &Entry,
    rows: u32,
    a: &[f32],
    b: &[f32],
    label: &str,
    measure: bool,
) -> LabResult<()> {
    let new = old.clone().with_decode();
    let q = old.query();
    let modules: Vec<_> = [old, &new]
        .into_iter()
        .map(|entry| {
            let bytes = fold_cubin_for_symbol(entry.symbol())
                .map(|(_, bytes)| bytes)
                .unwrap_or_else(probe_bytes);
            session.load_cubin(bytes)
        })
        .collect::<Result<_, _>>()?;
    let x = session.upload_f32(a)?;
    let weight = if q.weight == WeightFormat::Bf16 {
        session.upload_bf16(&b.iter().map(|&v| f32_to_bf16(v)).collect::<Vec<_>>())?
    } else {
        session.upload_f32(b)?
    };
    let output = [
        session.allocate(rows as usize * q.n as usize * 4)?,
        session.allocate(rows as usize * q.n as usize * 4)?,
    ];
    let live = session.allocate(4)?;
    let domain = session.allocate(new.weight_domain_bytes() as usize)?;
    let workspace = session.allocate(new.decode_workspace_bytes(rows) as usize)?;
    let status = session.allocate(rows as usize * 4)?;
    let bindings = launch::Bindings {
        buffers: BTreeMap::from([
            (1, &x),
            (2, &weight),
            (3, &output[0]),
            (4, &output[1]),
            (5, &live),
            (6, &domain),
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
    let init = ops[1].initialize(&buffers(4)).map_err(|e| e.to_string())?;
    poison::poison(session, &[&domain])?;
    run_launches(session, &modules[1], &bindings, &init, &session.stream)?;
    let actual_flags = poison::read(session, &domain)?;
    let expected_flags: Vec<_> = cases::weight_domains(b, q.k as usize, q.weight)
        .into_iter()
        .flat_map(u32::to_le_bytes)
        .collect();
    if actual_flags != expected_flags {
        return Err(format!("{label}: cached weight-domain flags differ"));
    }
    let launches = [
        ops[0]
            .launches(&buffers(3), View::EMPTY, Counts::rows(rows))
            .unwrap(),
        ops[1]
            .launches(&buffers(4), view(7), Counts::rows(rows))
            .unwrap(),
    ];
    let graphs = [
        session.capture(|s| run_launches(session, &modules[0], &bindings, &launches[0], s))?,
        session.capture(|s| run_launches(session, &modules[1], &bindings, &launches[1], s))?,
    ];
    for live_count in [rows, rows.saturating_sub(1), 0, rows] {
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
                        &modules[arm],
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
            "F32_TIMING {label} rows={rows} old_us={} new_us={} ratio={}",
            paired.arms[0].median_ms * 1000.,
            paired.arms[1].median_ms * 1000.,
            paired.paired_ratio
        );
        // Each stage is measured separately as well as the full sequence.
        for (arm, entries) in launches.iter().enumerate() {
            for (index, l) in entries.iter().enumerate() {
                let graph = session.capture(|s| {
                    run_launches(
                        session,
                        &modules[arm],
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
                    |s| run_launches(session, &modules[arm], &bindings, &entries[..index], s),
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

#[test]
fn gpu_bitwise_shapes_and_edges() {
    kernel_harness::gpu_test_with(timing_cubins(), 2 << 30, |session| {
        let g = &FLASH;
        for (label, n, k) in [
            ("router", g.shape.moe.experts, g.shape.hidden),
            ("wkv", g.shape.attention.head_dim, g.shape.hidden),
            ("weights_proj", g.shape.indexer.heads, g.shape.hidden),
            ("head", g.shape.vocab / 4, g.shape.hidden),
        ] {
            let old = if label == "head" {
                f32_head::entry(g, 4).unwrap()
            } else {
                let q = GemmQuery {
                    n,
                    k,
                    ..f32_head::query(g, 4)
                };
                Entry::new(
                    q,
                    f32_head::usage(),
                    &format!(
                        "chop_{}_dense_f32_m{}_n{n}_k{k}",
                        g.kernel_tag, q.m_capacity
                    ),
                )
                .unwrap()
            };
            for rows in [1, 2, 4, 8, 16, 32, 64] {
                let (a, b) = random_operands(rows, n, k, 0x7813_9231 + rows);
                compare(
                    session,
                    &old,
                    rows,
                    &a,
                    &b,
                    label,
                    matches!(rows, 1 | 32 | 64),
                )?;
            }
        }
        for old in edges() {
            let q = old.query();
            if q.n == 65536 {
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
                    &b,
                    "all_bf16_codes",
                    false,
                )?;
                continue;
            }
            for witness in [
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
            ] {
                let (a, b) = cases::operands(&old, witness);
                compare(
                    session,
                    &old,
                    32,
                    &a[..32 * q.k as usize],
                    &b,
                    &format!("{witness:?}"),
                    false,
                )?;
            }
            for seed in 1..=4 {
                let (mut a, mut b) = random_operands(32, q.n, q.k, seed * 123_457);
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
                compare(session, &old, 32, &a, &b, "random_and_late_fallback", false)?;
            }
            if q.k == 129 {
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
                    &b,
                    "every_exponent_sign_and_mantissa_boundary",
                    false,
                )?;
            }
        }
        Ok(())
    })
    .unwrap()
    .expect("GPU comparison must execute on a selected idle GPU");
}
