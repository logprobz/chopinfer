use kernel_core::contract::{
    TensorMap as MapSpec, TensorMapElement, TensorMapInterleave, TensorMapL2Promotion,
    TensorMapOobFill, TensorMapSwizzle,
};
use kernel_lab::launch;
use kernel_lab::{LabResult, Session, gpu_test_multi, selected_ordinals, time};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, LaunchConfig, Module};
use serde_json::{Value, json};
use std::collections::BTreeMap;

const GIB: usize = 1024 * 1024 * 1024;

fn config() -> LabResult<LaunchConfig> {
    launch::simple_config(
        (1024, 1, 1),
        (256, 1, 1),
        0,
        LaunchConfig::NO_CLUSTER,
        false,
    )
}

fn measure(
    session: &Session,
    function: &Function,
    args: &[KernelArgument<'_>],
    logical_bytes: u64,
    shape: LaunchConfig,
) -> LabResult<Value> {
    let timing = time(session, 2, 3, |stream| {
        function
            .launch(stream, shape, args)
            .map_err(|e| e.to_string())
    })?;
    Ok(json!({
        "passed": true,
        "median_ms": timing.median_ms,
        "min_ms": timing.min_ms,
        "rounds": timing.rounds,
        "logical_bytes": logical_bytes,
        "ctas": shape.grid.0,
        "gb_per_second": logical_bytes as f64 / f64::from(timing.median_ms) / 1_000_000.0,
    }))
}

fn bandwidth_config(session: &Session) -> LabResult<LaunchConfig> {
    let sms = session.metadata["sm_count"]
        .as_u64()
        .ok_or("missing SM count")?;
    let ctas = u32::try_from(sms.checked_mul(4).ok_or("bandwidth grid overflow")?)
        .map_err(|e| e.to_string())?;
    launch::simple_config(
        (ctas, 1, 1),
        (256, 1, 1),
        0,
        LaunchConfig::NO_CLUSTER,
        false,
    )
}

fn read_checksum(
    session: &Session,
    function: &Function,
    args: &[KernelArgument<'_>],
    checksum: &DeviceBuffer,
    shape: LaunchConfig,
) -> LabResult<u32> {
    checksum
        .copy_from_bytes(&0_u32.to_ne_bytes(), &session.stream)
        .map_err(|e| e.to_string())?;
    function
        .launch(&session.stream, shape, args)
        .map_err(|e| e.to_string())?;
    let mut bytes = [0_u8; 4];
    checksum
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(u32::from_ne_bytes(bytes))
}

fn measured_read(
    session: &Session,
    function: &Function,
    args: &[KernelArgument<'_>],
    checksum: &DeviceBuffer,
    words: u64,
    repeats: u32,
    shape: LaunchConfig,
) -> LabResult<Value> {
    let mut measured = measure(
        session,
        function,
        args,
        words * 4 * u64::from(repeats),
        shape,
    )?;
    let actual = read_checksum(session, function, args, checksum, shape)?;
    let expected = ((u128::from(words) * u128::from(words - 1) / 2) * u128::from(repeats)) as u32;
    measured["passed"] = json!(actual == expected);
    measured["checksum"] = json!(actual);
    measured["expected_checksum"] = json!(expected);
    Ok(measured)
}

fn pattern_samples(session: &Session, buffer: &DeviceBuffer, words: u64) -> LabResult<Value> {
    let mut samples = BTreeMap::new();
    let mut passed = true;
    for index in [0, words / 2, words - 1] {
        let mut bytes = [0_u8; 4];
        buffer
            .copy_to_bytes_at_async(index as usize * 4, &mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        let actual = u32::from_ne_bytes(bytes);
        passed &= actual == index as u32;
        samples.insert(index.to_string(), json!(actual));
    }
    Ok(json!({"passed": passed, "samples": samples}))
}

struct PatternCheck<'a> {
    read: &'a Function,
    checksum: &'a DeviceBuffer,
    output: &'a DeviceBuffer,
    words: u64,
}

fn measured_pattern(
    session: &Session,
    function: &Function,
    args: &[KernelArgument<'_>],
    check: PatternCheck<'_>,
    logical_bytes: u64,
    shape: LaunchConfig,
) -> LabResult<Value> {
    let mut measured = measure(session, function, args, logical_bytes, shape)?;
    let samples = pattern_samples(session, check.output, check.words)?;
    let actual = read_checksum(
        session,
        check.read,
        &[
            KernelArgument::Buffer(check.output),
            KernelArgument::Buffer(check.checksum),
            KernelArgument::U64(check.words),
            KernelArgument::U32(1),
        ],
        check.checksum,
        shape,
    )?;
    let expected = (u128::from(check.words) * u128::from(check.words - 1) / 2) as u32;
    measured["passed"] = json!(samples["passed"] == true && actual == expected);
    measured["samples"] = samples["samples"].clone();
    measured["checksum"] = json!(actual);
    measured["expected_checksum"] = json!(expected);
    Ok(measured)
}

fn best_bandwidth(candidates: BTreeMap<String, Value>) -> Value {
    let winner = candidates
        .iter()
        .filter(|(_, value)| value["passed"] == true)
        .max_by(|(_, left), (_, right)| {
            left["gb_per_second"]
                .as_f64()
                .unwrap_or(0.0)
                .total_cmp(&right["gb_per_second"].as_f64().unwrap_or(0.0))
        });
    let mut best = winner
        .map(|(name, value)| {
            let mut result = value.clone();
            result["variant"] = json!(name);
            result
        })
        .unwrap_or_else(
            || json!({"passed": false, "error": "all bandwidth candidates failed validation"}),
        );
    best["candidates"] = json!(candidates);
    best
}

fn bandwidth(session: &Session, module: &Module) -> LabResult<BTreeMap<String, Value>> {
    let bytes = 4 * GIB;
    let words = (bytes / 4) as u64;
    let shape = bandwidth_config(session)?;
    let source = session.allocate(bytes)?;
    let destination = session.allocate(bytes)?;
    let checksum = session.allocate(4)?;
    let write = module
        .function("chop_probe_write_v1")
        .map_err(|e| e.to_string())?;
    write
        .launch(
            &session.stream,
            shape,
            &[KernelArgument::Buffer(&source), KernelArgument::U64(words)],
        )
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    if pattern_samples(session, &source, words)?["passed"] != true {
        return Err("bandwidth source initialization failed".into());
    }
    let read = module
        .function("chop_probe_read_v1")
        .map_err(|e| e.to_string())?;
    let vector_read = module
        .function("chop_probe_read_vector_v1")
        .map_err(|e| e.to_string())?;
    let tma_read = module
        .function("chop_probe_read_tma_v1")
        .map_err(|e| e.to_string())?;
    let copy = module
        .function("chop_probe_copy_v1")
        .map_err(|e| e.to_string())?;
    let vector_write = module
        .function("chop_probe_write_vector_v1")
        .map_err(|e| e.to_string())?;
    let vector_copy = module
        .function("chop_probe_copy_vector_v1")
        .map_err(|e| e.to_string())?;
    let mut results = BTreeMap::new();
    let mut read_candidates = BTreeMap::new();
    read_candidates.insert(
        "scalar".into(),
        measured_read(
            session,
            &read,
            &[
                KernelArgument::Buffer(&source),
                KernelArgument::Buffer(&checksum),
                KernelArgument::U64(words),
                KernelArgument::U32(1),
            ],
            &checksum,
            words,
            1,
            shape,
        )?,
    );
    read_candidates.insert(
        "vector_nc_128b".into(),
        measured_read(
            session,
            &vector_read,
            &[
                KernelArgument::Buffer(&source),
                KernelArgument::Buffer(&checksum),
                KernelArgument::U64(words),
                KernelArgument::U32(1),
            ],
            &checksum,
            words,
            1,
            shape,
        )?,
    );
    read_candidates.insert(
        "tma_bulk_cta_16k".into(),
        measured_read(
            session,
            &tma_read,
            &[
                KernelArgument::Buffer(&source),
                KernelArgument::Buffer(&checksum),
                KernelArgument::U64(words),
            ],
            &checksum,
            words,
            1,
            shape,
        )?,
    );
    results.insert("dram_read".into(), best_bandwidth(read_candidates));
    let mut write_candidates = BTreeMap::new();
    for (name, function) in [("scalar", &write), ("vector_cs_128b", &vector_write)] {
        write_candidates.insert(
            name.into(),
            measured_pattern(
                session,
                function,
                &[
                    KernelArgument::Buffer(&destination),
                    KernelArgument::U64(words),
                ],
                PatternCheck {
                    read: &read,
                    checksum: &checksum,
                    output: &destination,
                    words,
                },
                bytes as u64,
                shape,
            )?,
        );
    }
    results.insert("dram_write".into(), best_bandwidth(write_candidates));
    let mut copy_candidates = BTreeMap::new();
    for (name, function) in [("scalar", &copy), ("vector_nc_cs_128b", &vector_copy)] {
        copy_candidates.insert(
            name.into(),
            measured_pattern(
                session,
                function,
                &[
                    KernelArgument::Buffer(&source),
                    KernelArgument::Buffer(&destination),
                    KernelArgument::U64(words),
                ],
                PatternCheck {
                    read: &read,
                    checksum: &checksum,
                    output: &destination,
                    words,
                },
                2 * bytes as u64,
                shape,
            )?,
        );
    }
    results.insert("dram_copy".into(), best_bandwidth(copy_candidates));
    let l2_bytes = 64 * 1024 * 1024_u64;
    let l2_words = l2_bytes / 4;
    let mut l2_candidates = BTreeMap::new();
    for (name, function) in [("scalar", &read), ("vector_nc_128b", &vector_read)] {
        l2_candidates.insert(
            name.into(),
            measured_read(
                session,
                function,
                &[
                    KernelArgument::Buffer(&source),
                    KernelArgument::Buffer(&checksum),
                    KernelArgument::U64(l2_words),
                    KernelArgument::U32(32),
                ],
                &checksum,
                l2_words,
                32,
                shape,
            )?,
        );
    }
    results.insert("l2_read".into(), best_bandwidth(l2_candidates));
    Ok(results)
}

fn gather4(session: &Session, module: &Module) -> LabResult<Value> {
    let mut values = Vec::new();
    for row in 0..4 {
        values.extend(vec![(row + 1) as f32; 32]);
    }
    let source = session.upload_f32(&values)?;
    let descriptor = MapSpec {
        base_buffer: 0,
        dims: vec![32, 4],
        strides_bytes: vec![0, 128],
        box_dims: vec![32, 1],
        element_strides: vec![1, 1],
        swizzle: TensorMapSwizzle::None,
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::None,
        oob_fill: TensorMapOobFill::Zero,
        element: TensorMapElement::F32,
    };
    let map = launch::tensor_map(&session.context, &descriptor, &source)?;
    let output = session.allocate(20)?;
    let function = module
        .function("chop_probe_gather4_v1")
        .map_err(|e| e.to_string())?;
    function
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[
                KernelArgument::TensorMap(&map),
                KernelArgument::Buffer(&output),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut bytes = [0_u8; 20];
    output
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    let observed: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let passed =
        observed[0] == 1 && (0..4).all(|row| observed[row + 1] == ((row + 1) as f32).to_bits());
    Ok(json!({"passed": passed, "ready": observed[0], "rows_bits": &observed[1..]}))
}

fn read_u32(
    session: &Session,
    buffer: &runtime_cuda::DeviceBuffer,
    words: usize,
) -> LabResult<Vec<u32>> {
    let mut bytes = vec![0; words * 4];
    buffer
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().expect("four-byte word")))
        .collect())
}

fn cluster_dsm(session: &Session, module: &Module, size: u32) -> LabResult<Value> {
    let function = module
        .function("chop_probe_cluster_dsm_v1")
        .map_err(|e| e.to_string())?;
    let output = session.allocate(size as usize * 3 * 4)?;
    function
        .launch(
            &session.stream,
            launch::simple_config((size, 1, 1), (32, 1, 1), 0, (size, 1, 1), false)?,
            &[KernelArgument::Buffer(&output)],
        )
        .map_err(|e| e.to_string())?;
    let observed = read_u32(session, &output, size as usize * 3)?;
    let passed = (0..size as usize).all(|rank| {
        let previous = if rank == 0 {
            size as usize - 1
        } else {
            rank - 1
        };
        observed[rank * 3] == 1
            && observed[rank * 3 + 1] == ((previous + 1) as f32).to_bits()
            && observed[rank * 3 + 2] == (previous + 1) as u32
    });
    Ok(json!({"passed": passed, "cluster_ctas": size, "observed_words": observed}))
}

fn cluster_control(session: &Session, module: &Module) -> LabResult<Value> {
    let function = module
        .function("chop_probe_cluster_control_v1")
        .map_err(|e| e.to_string())?;
    let output = session.allocate(12)?;
    function
        .launch(
            &session.stream,
            launch::simple_config((16, 1, 1), (32, 1, 1), 0, (2, 1, 1), false)?,
            &[KernelArgument::Buffer(&output)],
        )
        .map_err(|e| e.to_string())?;
    let observed = read_u32(session, &output, 3)?;
    let passed = observed[0] == 1 && (observed[1] == 0 || observed[2] < 16);
    Ok(
        json!({"passed": passed, "ready": observed[0], "canceled": observed[1] != 0,
        "first_cta": observed[2]}),
    )
}

fn shared_limit(session: &Session, module: &Module) -> LabResult<Value> {
    let function = module
        .function("chop_probe_shared_limit_v1")
        .map_err(|e| e.to_string())?;
    function
        .set_max_dynamic_shared_memory(101376)
        .map_err(|e| e.to_string())?;
    let output = session.allocate(4)?;
    let mut low = 0_u32;
    let mut high = 101376 / 256;
    while low < high {
        let mid = (low + high).div_ceil(2);
        let bytes = mid * 256;
        let result = function.launch(
            &session.stream,
            launch::simple_config(
                (1, 1, 1),
                (32, 1, 1),
                bytes,
                LaunchConfig::NO_CLUSTER,
                false,
            )?,
            &[KernelArgument::Buffer(&output), KernelArgument::U32(bytes)],
        );
        if result.is_ok() && session.stream.synchronize().is_ok() {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    Ok(
        json!({"passed": low * 256 == 101376, "max_dynamic_shared_bytes": low * 256,
        "expected_bytes": 101376}),
    )
}

fn pdl_overlap(session: &Session, module: &Module) -> LabResult<Value> {
    let first = module
        .function("chop_probe_pdl_first_v1")
        .map_err(|e| e.to_string())?;
    let second = module
        .function("chop_probe_pdl_second_v1")
        .map_err(|e| e.to_string())?;
    let stamps = session.allocate(32)?;
    stamps
        .copy_from_bytes(&[0; 32], &session.stream)
        .map_err(|e| e.to_string())?;
    first
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[
                KernelArgument::Buffer(&stamps),
                KernelArgument::U64(200_000_000),
            ],
        )
        .map_err(|e| e.to_string())?;
    second
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, true)?,
            &[KernelArgument::Buffer(&stamps)],
        )
        .map_err(|e| e.to_string())?;
    let mut bytes = [0_u8; 32];
    stamps
        .copy_to_bytes(&mut bytes, &session.stream)
        .map_err(|e| e.to_string())?;
    let observed: Vec<u64> = bytes
        .chunks_exact(8)
        .map(|chunk| u64::from_ne_bytes(chunk.try_into().expect("eight-byte timestamp")))
        .collect();
    let passed =
        observed[0] < observed[2] && observed[2] < observed[1] && observed[1] <= observed[3];
    Ok(
        json!({"passed": passed, "a_start": observed[0], "a_end": observed[1],
        "b_pre_wait": observed[2], "b_post_wait": observed[3],
        "overlap_ns": observed[1].saturating_sub(observed[2])}),
    )
}

fn mma_throughput(session: &Session, module: &Module) -> LabResult<BTreeMap<String, Value>> {
    let sm_count = session
        .metadata
        .get("sm_count")
        .and_then(Value::as_u64)
        .and_then(|count| u32::try_from(count).ok())
        .ok_or("missing SM count")?;
    let blocks = sm_count.checked_mul(4).ok_or("MMA grid overflow")?;
    let iterations = 512_u32;
    let output = session.allocate(blocks as usize * 256 * 4)?;
    let input = session.allocate(8 * 4)?;
    input
        .copy_from_bytes(&[0; 32], &session.stream)
        .map_err(|e| e.to_string())?;
    let formats = [
        ("bf16", 16_u64),
        ("e4m3", 32),
        ("e2m1", 32),
        ("mxf8_e4m3", 32),
        ("mxf8_e2m1", 32),
        ("nvfp4_2x", 64),
        ("nvfp4_4x", 64),
        ("mxf4", 64),
    ];
    let mut results = BTreeMap::new();
    for (name, k) in formats {
        let measured = (|| -> LabResult<Value> {
            let function = module
                .function(&format!("chop_probe_mma_{name}_v1"))
                .map_err(|e| e.to_string())?;
            let timing = time(session, 2, 3, |stream| {
                function
                    .launch(
                        stream,
                        launch::simple_config(
                            (blocks, 1, 1),
                            (256, 1, 1),
                            0,
                            LaunchConfig::NO_CLUSTER,
                            false,
                        )?,
                        &[
                            KernelArgument::Buffer(&output),
                            KernelArgument::Buffer(&input),
                            KernelArgument::U32(iterations),
                        ],
                    )
                    .map_err(|e| e.to_string())
            })?;
            let flops = u64::from(blocks) * 8 * 8 * u64::from(iterations) * 16 * 8 * k * 2;
            let mut sample = [0.0_f32];
            output
                .copy_to_f32(&mut sample, &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(json!({"passed": sample[0].is_finite(), "sample": sample[0],
                "logical_flops": flops, "median_ms": timing.median_ms,
                "min_ms": timing.min_ms, "rounds": timing.rounds,
                "tflops": flops as f64 / f64::from(timing.median_ms) / 1_000_000_000.0}))
        })();
        results.insert(
            name.into(),
            match measured {
                Ok(value) => value,
                Err(error) => json!({"passed": false, "error": error}),
            },
        );
    }
    Ok(results)
}

fn probe(session: &Session) -> LabResult<()> {
    let module = session.module("probe_sm120")?;
    let output = session.allocate(4)?;
    let function = module
        .function("chop_probe_sm120_v1")
        .map_err(|e| e.to_string())?;
    function
        .launch(
            &session.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[KernelArgument::Buffer(&output)],
        )
        .map_err(|e| e.to_string())?;
    let mut data = [0_u8; 4];
    output
        .copy_to_bytes(&mut data, &session.stream)
        .map_err(|e| e.to_string())?;
    let passed = u32::from_ne_bytes(data) == 120;
    let mut receipt = BTreeMap::new();
    receipt.insert(
        "sm120_cubin".into(),
        json!({"passed": passed, "observed": u32::from_ne_bytes(data)}),
    );
    if passed {
        for (name, result) in [
            ("cluster_dsm_2", cluster_dsm(session, &module, 2)),
            ("cluster_dsm_4", cluster_dsm(session, &module, 4)),
            ("cluster_dsm_8", cluster_dsm(session, &module, 8)),
            ("cluster_launch_control", cluster_control(session, &module)),
            ("tma_gather4", gather4(session, &module)),
            ("max_dynamic_shared_memory", shared_limit(session, &module)),
            ("pdl_overlap", pdl_overlap(session, &module)),
        ] {
            receipt.insert(
                name.into(),
                match result {
                    Ok(value) => value,
                    Err(error) => json!({"passed": false, "error": error}),
                },
            );
        }
        match bandwidth(session, &module) {
            Ok(results) => receipt.extend(results),
            Err(error) => {
                receipt.insert("bandwidth".into(), json!({"passed": false, "error": error}));
            }
        }
        match mma_throughput(session, &module) {
            Ok(results) => {
                receipt.insert("mma_tflops".into(), json!(results));
            }
            Err(error) => {
                receipt.insert(
                    "mma_tflops".into(),
                    json!({"passed": false, "error": error}),
                );
            }
        }
    }
    receipt.insert(
        "pending_probes".into(),
        json!([
            "cluster_2_4_8_tma_multicast: CUDA 13.2 sm_120f shared::cluster tensor TMA lowers to undefined __cuda_syscall_cp_async_bulk_tensor_2d_tile_multicast"
        ]),
    );
    let path = session.receipt("probes", receipt)?;
    println!("{}", path.display());
    if !passed {
        return Err("SM120 cubin execution failed".into());
    }
    Ok(())
}

fn p2p_pair(source: &Session, destination: &Session) -> LabResult<BTreeMap<String, Value>> {
    let bytes = 4 * GIB;
    let words = (bytes / 4) as u64;
    let source_buffer = source.allocate(bytes)?;
    let destination_buffer = destination.allocate(bytes)?;
    let source_module = source.module("probe_sm120")?;
    let destination_module = destination.module("probe_sm120")?;
    source_module
        .function("chop_probe_write_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &source.stream,
            config()?,
            &[
                KernelArgument::Buffer(&source_buffer),
                KernelArgument::U64(words),
            ],
        )
        .map_err(|e| e.to_string())?;
    source.stream.synchronize().map_err(|e| e.to_string())?;
    let source_address = source_buffer
        .device_address_at(0, bytes)
        .map_err(|e| e.to_string())?;
    let destination_address = destination_buffer
        .device_address_at(0, bytes)
        .map_err(|e| e.to_string())?;
    let pull = destination_module
        .function("chop_probe_p2p_pull_v1")
        .map_err(|e| e.to_string())?;
    let push = source_module
        .function("chop_probe_p2p_push_v1")
        .map_err(|e| e.to_string())?;
    let pull_time = time(destination, 2, 3, |stream| {
        pull.launch(
            stream,
            config()?,
            &[
                KernelArgument::Buffer(&destination_buffer),
                KernelArgument::DevicePointerArray(&[source_address]),
                KernelArgument::U64(words),
            ],
        )
        .map_err(|e| e.to_string())
    })?;
    let pull_correct = read_u32(destination, &destination_buffer, 2)? == [0, 1];
    let push_time = time(source, 2, 3, |stream| {
        push.launch(
            stream,
            config()?,
            &[
                KernelArgument::Buffer(&source_buffer),
                KernelArgument::DevicePointerArray(&[destination_address]),
                KernelArgument::U64(words),
            ],
        )
        .map_err(|e| e.to_string())
    })?;
    let push_correct = read_u32(destination, &destination_buffer, 2)? == [0, 1];
    let mut receipt = BTreeMap::new();
    receipt.insert(
        "kernel_pull".into(),
        json!({"passed": pull_correct,
        "median_ms": pull_time.median_ms, "min_ms": pull_time.min_ms,
        "gb_per_second": bytes as f64 / f64::from(pull_time.median_ms) / 1_000_000.0}),
    );
    receipt.insert(
        "kernel_push".into(),
        json!({"passed": push_correct,
        "median_ms": push_time.median_ms, "min_ms": push_time.min_ms,
        "gb_per_second": bytes as f64 / f64::from(push_time.median_ms) / 1_000_000.0}),
    );
    let source_flag = source.allocate(4)?;
    let destination_flag = destination.allocate(4)?;
    source_flag
        .copy_from_bytes(&0_u32.to_ne_bytes(), &source.stream)
        .map_err(|e| e.to_string())?;
    destination_flag
        .copy_from_bytes(&0_u32.to_ne_bytes(), &destination.stream)
        .map_err(|e| e.to_string())?;
    let source_flag_address = source_flag
        .device_address_at(0, 4)
        .map_err(|e| e.to_string())?;
    let destination_flag_address = destination_flag
        .device_address_at(0, 4)
        .map_err(|e| e.to_string())?;
    let stamps = source.allocate(16)?;
    stamps
        .copy_from_bytes(&[0; 16], &source.stream)
        .map_err(|e| e.to_string())?;
    destination_module
        .function("chop_probe_p2p_pong_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &destination.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[
                KernelArgument::Buffer(&destination_flag),
                KernelArgument::DevicePointerArray(&[source_flag_address]),
                KernelArgument::U32(1000),
            ],
        )
        .map_err(|e| e.to_string())?;
    source_module
        .function("chop_probe_p2p_ping_v1")
        .map_err(|e| e.to_string())?
        .launch(
            &source.stream,
            launch::simple_config((1, 1, 1), (32, 1, 1), 0, LaunchConfig::NO_CLUSTER, false)?,
            &[
                KernelArgument::Buffer(&source_flag),
                KernelArgument::DevicePointerArray(&[destination_flag_address]),
                KernelArgument::Buffer(&stamps),
                KernelArgument::U32(1000),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut stamp_bytes = [0_u8; 16];
    stamps
        .copy_to_bytes(&mut stamp_bytes, &source.stream)
        .map_err(|e| e.to_string())?;
    destination
        .stream
        .synchronize()
        .map_err(|e| e.to_string())?;
    let start = u64::from_ne_bytes(stamp_bytes[..8].try_into().expect("start timestamp"));
    let end = u64::from_ne_bytes(stamp_bytes[8..].try_into().expect("end timestamp"));
    receipt.insert(
        "flag_one_way_latency".into(),
        json!({"passed": end > start,
        "round_trips": 1000, "one_way_ns": end.saturating_sub(start) / 2000}),
    );
    let mut transfers = Vec::new();
    destination_buffer
        .copy_from_bytes(&vec![0xff; bytes], &destination.stream)
        .map_err(|e| e.to_string())?;
    destination
        .stream
        .synchronize()
        .map_err(|e| e.to_string())?;
    let copy_time = time(destination, 2, 3, |stream| {
        transfers.push(
            destination_buffer
                .copy_peer_range_from_async(0, &source_buffer, 0, bytes, stream)
                .map_err(|e| e.to_string())?,
        );
        Ok(())
    })?;
    drop(transfers);
    let copy_correct = read_u32(destination, &destination_buffer, 2)? == [0, 1];
    receipt.insert(
        "copy_engine".into(),
        json!({"passed": copy_correct,
        "median_ms": copy_time.median_ms, "min_ms": copy_time.min_ms,
        "gb_per_second": bytes as f64 / f64::from(copy_time.median_ms) / 1_000_000.0}),
    );
    Ok(receipt)
}

fn probe_pairs(sessions: &[Session]) -> LabResult<()> {
    for source in sessions {
        for destination in sessions {
            if source.ordinal == destination.ordinal {
                continue;
            }
            let mut fields = BTreeMap::new();
            fields.insert("kind".into(), json!("pcie_p2p"));
            fields.insert("source_ordinal".into(), json!(source.ordinal));
            fields.insert("destination_ordinal".into(), json!(destination.ordinal));
            fields.insert("peer_device".into(), json!(destination.metadata));
            match p2p_pair(source, destination) {
                Ok(results) => fields.extend(results),
                Err(error) => {
                    fields.insert(
                        "pair_failure".into(),
                        json!({"passed": false, "error": error}),
                    );
                }
            }
            let path = source.receipt("probes", fields)?;
            println!("{}", path.display());
        }
    }
    Ok(())
}

fn run() -> LabResult<()> {
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() != Some("probe") {
        return Err("usage: kernel-lab probe [p2p]".into());
    }
    let p2p_only = match args.next().as_deref() {
        None => false,
        Some("p2p") => true,
        _ => return Err("usage: kernel-lab probe [p2p]".into()),
    };
    if args.next().is_some() {
        return Err("usage: kernel-lab probe [p2p]".into());
    }
    let Some(ordinals) = selected_ordinals()? else {
        eprintln!(
            "skipping GPU probe: set CHOPINFER_KERNEL_GPU or CHOPINFER_KERNEL_GPUS to idle GPUs"
        );
        return Ok(());
    };
    let multi = ordinals.len() > 1;
    if p2p_only && !multi {
        return Err("P2P probe requires CHOPINFER_KERNEL_GPUS with at least two ordinals".into());
    }
    if p2p_only {
        return gpu_test_multi(10 * GIB as u64, probe_pairs).map(|_| ());
    }
    for ordinal in ordinals {
        probe(&Session::open(
            kernel_lab::LAB_CUBINS,
            ordinal,
            10 * GIB as u64,
        )?)?;
    }
    if multi {
        gpu_test_multi(10 * GIB as u64, probe_pairs)?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("kernel-lab: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandwidth_peak_excludes_failed_candidates() {
        let candidates = BTreeMap::from([
            (
                "valid".into(),
                json!({"passed": true, "gb_per_second": 1500.0}),
            ),
            (
                "bad".into(),
                json!({"passed": false, "gb_per_second": 2000.0}),
            ),
        ]);
        let peak = best_bandwidth(candidates);
        assert_eq!(peak["variant"], "valid");
        assert_eq!(peak["gb_per_second"], 1500.0);
        assert!(peak["candidates"]["bad"].is_object());
    }
}
