//! Exact prefill candidates, including admitted tensor and SIMT operands.
use bundle_deepseek41::sm120::{f32_blocked as registration, f32_prefill, fold_cubin_for_symbol};
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_core::contract::LaunchPlan;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch, poison};
use kernels_sm120::f32_projection::{Entry, blocked};
use model_deepseek41_contract::geometry::GEOMETRIES;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Stream};
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: &[u32] = &[
    1, 17, 32, 33, 64, 127, 129, 257, 511, 512, 513, 1000, 1024, 2047, 2048, 2049, 4095, 4096,
    8150, 8191, 8192,
];
const TIMED: &[u32] = &[1, 127, 129, 512, 1024, 2048, 4096, 8191, 8192];
const PROBES: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
fn root() -> PathBuf {
    std::env::var_os("CHOPINFER_F32_BLOCKED_CUBINS")
        .expect("staged F32 cubins required")
        .into()
}
fn entries() -> Vec<(String, Entry)> {
    GEOMETRIES
        .iter()
        .flat_map(|g| {
            f32_prefill::baseline_entries(g)
                .unwrap()
                .into_iter()
                .map(move |e| (registration::cubin_name(g), e))
        })
        .collect()
}
fn specs() -> Vec<kernel_build::CubinSpec> {
    let mut specs = registration::cubins(GEOMETRIES).unwrap();
    // A deliberately persistent SIMT chain must disagree on cancellation.
    let e = &entries()[0].1;
    let mut control = blocked::entry_spec(e, 32).unwrap();
    control.symbol = "chop_f32_blocked_single_chain_v1".into();
    control.body = control.body.replace(",32,false>(", ",32,false,true>(");
    assert!(control.body.contains("false,true>"));
    specs[0].entries.push(control);
    specs
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_BLOCKED_CUBINS"]
fn cpu_build_blocked_prefill() {
    let specs = specs();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d");
    let options = BuildOptions {
        emit_cargo_directives: false,
        profile_line_info: false,
        ..BuildOptions::from_env()
    };
    if options.lock_update {
        assert!(!dir.join(format!("{}.lock", registration::FAMILY)).exists());
    }
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let locks = LockDir::new(dir).family(registration::FAMILY, specs.iter().map(|s| &s.name));
    kernel_build::build_in_families(&toolchain, &specs, &root(), &locks, options, None).unwrap();
}
fn old(entry: &Entry, rows: u32) -> Entry {
    let q = entry.query();
    let tile = f32_prefill::BUCKETS
        .iter()
        .find(|b| (b.n, b.k, b.rows) == (q.n, q.k, rows))
        .map_or(64, |b| b.features);
    entry.clone().with_prefill_tiles(tile).unwrap()
}
#[test]
fn cpu_blocked_matrix_and_launch_contracts() {
    let mut symbols = BTreeMap::<String, Vec<String>>::new();
    let mut launches = Vec::new();
    for (module, e) in entries() {
        for rows in ROWS.iter().copied() {
            for m in [32, 64] {
                let plan = blocked::plan(&e, m, rows).unwrap();
                let spec = blocked::entry_spec(&e, m).unwrap();
                assert_eq!(plan.block, [spec.max_threads, 1, 1]);
                assert_eq!(plan.cluster, [1, 1, 1]);
                assert!(plan.dynamic_shared_bytes <= 101376);
                launches.push(serde_json::json!({"module":module,"symbol":spec.symbol,"rows":rows,"block":plan.block,"cluster":plan.cluster,"shared":plan.dynamic_shared_bytes,"attribute":plan.dynamic_shared_bytes}));
            }
            let before = old(&e, rows);
            let name = fold_cubin_for_symbol(before.symbol()).unwrap().0;
            if let Some(dir) = std::env::var_os("CHOPINFER_F32_BLOCKED_BASELINES") {
                std::fs::write(
                    PathBuf::from(dir).join(format!("{name}.cubin")),
                    fold_cubin_for_symbol(before.symbol()).unwrap().1,
                )
                .unwrap();
            }
            let p = before.prefill_plan(rows).unwrap();
            launches.push(serde_json::json!({"module":name,"symbol":before.symbol(),"rows":rows,"block":p.block,"cluster":p.cluster,"shared":p.dynamic_shared_bytes,"attribute":p.dynamic_shared_bytes}));
            for (symbol, p) in [
                (before.stage_symbol("_prepare"), stage_plan([rows, 1, 1])),
                (
                    before.stage_symbol("_weights"),
                    stage_plan([e.query().n.div_ceil(16), 1, 1]),
                ),
            ] {
                launches.push(serde_json::json!({"module":name,"symbol":symbol,"rows":rows,"block":p.block,"cluster":p.cluster,"shared":0,"attribute":0}));
            }
            symbols
                .entry(name.to_string())
                .or_default()
                .extend(before.symbols());
        }
    }
    for spec in specs() {
        for e in &spec.entries {
            let shared = if e.symbol.contains("blocked") {
                if e.symbol.contains("_m64_") {
                    57344
                } else {
                    32768
                }
            } else {
                0
            };
            launches.push(serde_json::json!({"module":spec.name,"symbol":e.symbol,"rows":8192,"block":[e.max_threads,1,1],"cluster":[1,1,1],"shared":shared,"attribute":shared}));
        }
        symbols
            .entry(spec.name)
            .or_default()
            .extend(spec.entries.into_iter().map(|s| s.symbol));
    }
    symbols.insert(
        "probe_sm120".into(),
        PROBES.iter().map(|s| s.to_string()).collect(),
    );
    for names in symbols.values_mut() {
        names.sort();
        names.dedup();
    }
    if let Some(path) = std::env::var_os("CHOPINFER_F32_BLOCKED_EXPORT") {
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"symbols":symbols,"launches":launches,"rows":ROWS,"shapes":entries().len(),"tiles":[32,64],"patterns":["tensor","row_fallback","weight_fallback","edges"],"timed_rows":TIMED})).unwrap()).unwrap();
    }
    println!(
        "F32_BLOCKED_MATRIX shapes={} rows={} PASS",
        entries().len(),
        ROWS.len()
    );
}
fn stage_plan(grid: [u32; 3]) -> LaunchPlan {
    LaunchPlan {
        grid,
        block: [128, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    }
}
fn call(f: &Function, stream: &Stream, plan: LaunchPlan, args: &[A]) -> LabResult<()> {
    f.launch(stream, launch::config(plan)?, args).map_err(|e| {
        format!(
            "F32 launch block={:?} grid={:?} cluster={:?} shared={} attribute={}: {e}",
            plan.block,
            plan.grid,
            plan.cluster,
            plan.dynamic_shared_bytes,
            plan.dynamic_shared_bytes
        )
    })
}
struct Buffers {
    x: DeviceBuffer,
    w: DeviceBuffer,
    domain: DeviceBuffer,
    work: DeviceBuffer,
    live: DeviceBuffer,
    out: [DeviceBuffer; 2],
}
fn compute(
    f: &Function,
    s: &Stream,
    p: LaunchPlan,
    b: &Buffers,
    rows: u32,
    arm: usize,
) -> LabResult<()> {
    call(
        f,
        s,
        p,
        &[
            A::Buffer(&b.x),
            A::Buffer(&b.w),
            A::Buffer(&b.work),
            A::Buffer(&b.out[arm]),
            A::Buffer(&b.live),
            A::U32(rows),
            A::Buffer(&b.domain),
        ],
    )
}
fn set_shared(f: &Function, p: LaunchPlan, label: &str) -> LabResult<()> {
    f.set_max_dynamic_shared_memory(p.dynamic_shared_bytes)
        .map_err(|e| {
            format!(
                "{label}: MAX_DYNAMIC_SHARED_SIZE_BYTES={} block={:?} cluster={:?}: {e}",
                p.dynamic_shared_bytes, p.block, p.cluster
            )
        })
}
fn compare_case(
    s: &Session,
    module: &runtime_cuda::Module,
    e: &Entry,
    rows: u32,
    m: u32,
    pattern: &str,
) -> LabResult<()> {
    let q = e.query();
    let before = old(e, rows);
    let candidate = blocked::symbol(e, m);
    let label = format!(
        "{candidate} n={} k={} rows={rows} pattern={pattern}",
        q.n, q.k
    );
    let old_module = s.load_cubin(fold_cubin_for_symbol(before.symbol()).unwrap().1)?;
    let f = [
        old_module
            .function(before.symbol())
            .map_err(|e| e.to_string())?,
        module.function(&candidate).map_err(|e| e.to_string())?,
    ];
    let plans = [
        before.prefill_plan(rows).unwrap(),
        blocked::plan(e, m, rows).unwrap(),
    ];
    for arm in 0..2 {
        set_shared(&f[arm], plans[arm], &label)?;
    }
    let prep = module
        .function(
            &e.clone()
                .with_prefill_tiles(32)
                .unwrap()
                .stage_symbol("_prepare"),
        )
        .map_err(|e| e.to_string())?;
    let classify = module
        .function(
            &e.clone()
                .with_prefill_tiles(32)
                .unwrap()
                .stage_symbol("_weights"),
        )
        .map_err(|e| e.to_string())?;
    let (mut a, mut w) = super::f32_decode::random_operands(rows, q.n, q.k, 9817 + rows);
    if pattern == "row_fallback" {
        for row in 0..rows as usize {
            a[row * q.k as usize + q.k as usize - 1] = f32::from_bits(1);
        }
    }
    if pattern == "weight_fallback" || pattern == "edges" {
        for col in 0..q.n as usize {
            w[col * q.k as usize + q.k as usize - 1] = f32::from_bits(0x00800000);
        }
    }
    if pattern == "edges" {
        let bits = [
            0, 0x80000000, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000, 0xff800000,
            0x7fc12345,
        ];
        for row in 0..rows as usize {
            a[row * q.k as usize + 127] = f32::from_bits(bits[row % bits.len()]);
        }
        let weight_bits = [
            0, 0x80000000, 0x00010000, 0x007f0000, 0x7f7f0000, 0x7f800000, 0xff800000, 0x7fc10000,
            0x7f810000,
        ];
        for col in 0..q.n as usize {
            w[col * q.k as usize + 127] = f32::from_bits(weight_bits[col % weight_bits.len()]);
        }
    }
    let b = Buffers {
        x: s.upload_f32(&a)?,
        w: s.upload_bf16(
            &w.iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )?,
        domain: s.allocate(e.weight_domain_bytes() as usize)?,
        work: s.allocate(before.prefill_workspace_bytes(rows) as usize)?,
        live: s.allocate(4)?,
        out: [
            s.allocate((rows as usize * q.n as usize + 16) * 4)?,
            s.allocate((rows as usize * q.n as usize + 16) * 4)?,
        ],
    };
    drop(a);
    drop(w);
    call(
        &classify,
        &s.stream,
        stage_plan([q.n.div_ceil(16), 1, 1]),
        &[A::Buffer(&b.w), A::Buffer(&b.domain)],
    )?;
    let run = |arm: usize, stream: &Stream| -> LabResult<()> {
        call(
            &prep,
            stream,
            stage_plan([rows, 1, 1]),
            &[
                A::Buffer(&b.x),
                A::Buffer(&b.live),
                A::U32(rows),
                A::Buffer(&b.work),
            ],
        )?;
        compute(&f[arm], stream, plans[arm], &b, rows, arm)
            .map_err(|e| format!("{label} arm={arm}: {e}"))
    };
    let graphs = [
        s.capture(|stream| run(0, stream))?,
        s.capture(|stream| run(1, stream))?,
    ];
    for live in [rows, rows - 1, 0, rows] {
        b.live
            .copy_from_bytes(&live.to_ne_bytes(), &s.stream)
            .map_err(|e| e.to_string())?;
        for graph in [false, true] {
            for out in &b.out {
                poison::fill_all(s, out, &[0xa5])?;
            }
            for (arm, executable) in graphs.iter().enumerate() {
                poison::fill_all(s, &b.work, &[0xa5])?;
                if graph {
                    executable.launch(&s.stream).map_err(|e| e.to_string())?;
                } else {
                    run(arm, &s.stream)?;
                }
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let output = [poison::read(s, &b.out[0])?, poison::read(s, &b.out[1])?];
            if output[0] != output[1] {
                let at = output[0]
                    .iter()
                    .zip(&output[1])
                    .position(|(a, b)| a != b)
                    .unwrap();
                return Err(format!(
                    "{label} live={live} graph={graph} mismatch byte={at}"
                ));
            }
            if live > 0
                && output[0][..live as usize * q.n as usize * 4]
                    .iter()
                    .all(|&v| v == 0xa5)
            {
                return Err(format!(
                    "{label}: reference left every live output poisoned"
                ));
            }
            if output[1][live as usize * q.n as usize * 4..]
                .iter()
                .any(|&v| v != 0xa5)
            {
                return Err(format!("{label}: inactive output or guard overwritten"));
            }
        }
    }
    println!("F32_BLOCKED_BITWISE {label} PASS");
    if pattern != "edges" && TIMED.contains(&rows) {
        let paired = kernel_harness::time_ab(s, 2, 7, [EvictionMode::Write; 2], |arm, stream| {
            graphs[arm].launch(stream).map_err(|e| e.to_string())
        })?;
        let bytes = u64::from(rows) * u64::from(q.k) * 4
            + e.weight_bytes()
            + u64::from(rows) * u64::from(q.n) * 4;
        let flops = 2. * f64::from(rows) * f64::from(q.n) * f64::from(q.k);
        println!(
            "F32_BLOCKED_TIMING {label} old_us={} new_us={} ratio={} dram_floor_us={} simt_tflops={}",
            paired.arms[0].median_ms * 1000.,
            paired.arms[1].median_ms * 1000.,
            paired.paired_ratio,
            bytes as f64 / 1_600_000.,
            flops / (f64::from(paired.arms[1].median_ms) * 1e9)
        );
    }
    Ok(())
}
fn negative(s: &Session, module: &runtime_cuda::Module, e: &Entry) -> LabResult<()> {
    let q = e.query();
    let rows = 17;
    let old = old(e, rows);
    let p = blocked::plan(e, 32, rows).unwrap();
    let good = module
        .function(&blocked::symbol(e, 32))
        .map_err(|e| e.to_string())?;
    let bad = module
        .function("chop_f32_blocked_single_chain_v1")
        .map_err(|e| e.to_string())?;
    for f in [&good, &bad] {
        set_shared(f, p, "single-chain control")?;
    }
    let (mut a, mut w) = super::f32_decode::random_operands(rows, q.n, q.k, 0x27341);
    for row in 0..rows as usize {
        a[row * q.k as usize + 127] = f32::from_bits(1);
    }
    for col in 0..q.n as usize {
        w[col * q.k as usize + q.k as usize - 1] = f32::from_bits(0x00800000);
    }
    let b = Buffers {
        x: s.upload_f32(&a)?,
        w: s.upload_bf16(
            &w.iter()
                .map(|v| (v.to_bits() >> 16) as u16)
                .collect::<Vec<_>>(),
        )?,
        domain: s.allocate(e.weight_domain_bytes() as usize)?,
        work: s.allocate(old.prefill_workspace_bytes(rows) as usize)?,
        live: s.allocate(4)?,
        out: [
            s.allocate((rows * q.n * 4) as usize)?,
            s.allocate((rows * q.n * 4) as usize)?,
        ],
    };
    b.live
        .copy_from_bytes(&rows.to_ne_bytes(), &s.stream)
        .map_err(|e| e.to_string())?;
    poison::fill_all(s, &b.domain, &[0])?;
    poison::fill_all(s, &b.work, &[0])?;
    for (arm, f) in [&good, &bad].into_iter().enumerate() {
        compute(f, &s.stream, p, &b, rows, arm)?;
    }
    s.stream.synchronize().map_err(|e| e.to_string())?;
    assert_ne!(
        poison::read(s, &b.out[0])?,
        poison::read(s, &b.out[1])?,
        "single-chain negative control was accepted"
    );
    println!("F32_BLOCKED_NEGATIVE persistent_chain PASS");
    Ok(())
}
#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_BLOCKED_CUBINS"]
fn gpu_blocked_prefill_bits_and_timings() {
    let probe = std::fs::read(root().join("probe_sm120.cubin"))
        .unwrap()
        .leak();
    let cubins = Cubins::new(
        Box::leak(Box::new([("probe_sm120", &*probe)])),
        &[("probe_sm120", PROBES)],
    );
    let shard: usize = std::env::var("CHOPINFER_F32_BLOCKED_SHARD")
        .unwrap()
        .parse()
        .unwrap();
    assert!(shard < 2);
    kernel_harness::gpu_test_with(cubins, 1 << 30, |s| {
        let mut id = 0;
        for (index, (name, e)) in entries().into_iter().enumerate() {
            let module = s.load_cubin(
                &std::fs::read(root().join(format!("{name}.cubin"))).map_err(|e| e.to_string())?,
            )?;
            if index == 0 {
                negative(s, &module, &e)?;
            }
            for rows in ROWS.iter().copied() {
                for m in [32, 64] {
                    for pattern in ["tensor", "row_fallback", "weight_fallback", "edges"] {
                        let selected = id % 2 == shard;
                        id += 1;
                        if selected {
                            compare_case(s, &module, &e, rows, m, pattern)?;
                        }
                    }
                }
            }
        }
        Ok(())
    })
    .unwrap()
    .expect("required F32 comparison skipped");
}
