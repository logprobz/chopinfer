use kernel_core::contract::{
    ActivationFormat, DType, DenseKernel, Epilogues, GemmQuery, ScaleMode, StaticQuery, Target,
    WeightFormat,
};
use kernels_sm120::gemm::{self, DenseRequest};
use kernels_sm120::moe::{self, GlobalScaleLayout as Globals, ScaleBindings};

fn grouped_request(k: u32, activation: ActivationFormat) -> moe::GroupedRequest {
    moe::GroupedRequest {
        shape: moe::Shape {
            hidden: 256,
            intermediate: k,
            experts: 4,
            top_k: 2,
        },
        projection: moe::Projection::Down,
        route_weight: moe::RouteWeight::AfterDown,
        swiglu_clamp: None,
        phase: moe::Phase::Decode,
        tokens_capacity: 1,
        weight: WeightFormat::Nvfp4,
        activation,
        scales: ScaleBindings {
            weight_globals: if activation == ActivationFormat::Bf16 {
                Globals::ExpertProjection
            } else {
                Globals::Scalar
            },
            activation_globals: if activation == ActivationFormat::Bf16 {
                Globals::Absent
            } else {
                Globals::Scalar
            },
            shared_gate_up_activation: true,
        },
        output_quantization: None,
        swiglu_columns: kernel_core::contract::SwigluColumns::ConcatenatedHalves,
        target: Target::Sm120f,
        symbol: "consumer_experts_v1".into(),
        knobs: None,
    }
}

#[test]
fn grouped_builder_derives_aligned_staging_and_keeps_scale_scopes_explicit() {
    for (k, bytes) in [(64, 32), (128, 64), (256, 128), (640, 64), (2560, 128)] {
        for capacity in [1, 2, 4, 8, 16, 32, 64] {
            let mut request = grouped_request(k, ActivationFormat::Bf16);
            request.tokens_capacity = capacity;
            request.phase = if capacity <= 32 {
                moe::Phase::Decode
            } else {
                moe::Phase::Prefill
            };
            let operation = moe::instantiate(request).unwrap();
            assert_eq!(
                operation.packed_contract.unwrap().decode_weight_bytes,
                bytes
            );
            assert_eq!(operation.knobs.split_k_capacity, 1);
            assert_eq!(operation.query.activation, ActivationFormat::Bf16);
            if capacity <= 32 {
                assert_eq!(
                    operation.weight_tensor_map(1).unwrap().box_dims[0],
                    u32::from(bytes)
                );
            }
        }
    }
    for projection in [moe::Projection::GateUp, moe::Projection::Down] {
        let mut request = grouped_request(640, ActivationFormat::Bf16);
        request.shape = moe::Shape {
            hidden: 2560,
            intermediate: 640,
            experts: 128,
            top_k: 10,
        };
        request.projection = projection;
        assert!(moe::instantiate(request).is_ok());
    }
}

#[test]
fn grouped_builder_refuses_unimplemented_scopes_and_incomplete_tiles() {
    let mut bad = grouped_request(256, ActivationFormat::Nvfp4);
    bad.scales.weight_globals = Globals::ExpertProjection;
    assert!(
        moe::instantiate(bad)
            .unwrap_err()
            .0
            .contains("bank-uniform")
    );
    let mut bad = grouped_request(256, ActivationFormat::Nvfp4);
    bad.projection = moe::Projection::GateUp;
    bad.scales.shared_gate_up_activation = false;
    assert!(moe::instantiate(bad).unwrap_err().0.contains("gate/up"));
    let mut bad = grouped_request(256, ActivationFormat::Nvfp4);
    bad.scales.activation_globals = Globals::Absent;
    assert!(moe::instantiate(bad).is_err());
    let mut bad = grouped_request(256, ActivationFormat::Bf16);
    bad.scales.weight_globals = Globals::Absent;
    assert!(moe::instantiate(bad).is_err());
    let mut bad = grouped_request(256, ActivationFormat::Bf16);
    bad.knobs = Some(moe::instantiate(bad.clone()).unwrap().knobs);
    bad.knobs.as_mut().unwrap().split_k_capacity = 2;
    assert!(moe::instantiate(bad).unwrap_err().0.contains("split-K=1"));
    assert!(moe::instantiate(grouped_request(32, ActivationFormat::Bf16)).is_err());
    let mut bad = grouped_request(24, ActivationFormat::Bf16);
    bad.projection = moe::Projection::GateUp;
    assert!(moe::instantiate(bad).unwrap_err().0.contains("output tile"));
    let good = moe::instantiate(grouped_request(256, ActivationFormat::Bf16)).unwrap();
    assert!(
        moe::cubin_spec("consumer", &[good.clone(), good])
            .unwrap_err()
            .0
            .contains("duplicate")
    );
}

#[test]
fn grouped_builder_exposes_actual_generic_pairs_and_fusions() {
    for weight in [
        WeightFormat::Mxfp4,
        WeightFormat::Mxfp8,
        WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
    ] {
        for activation in [
            ActivationFormat::Bf16,
            if weight == WeightFormat::Fp8Block128x128(ScaleMode::ExactF32) {
                ActivationFormat::Fp8Group128(ScaleMode::ExactF32)
            } else {
                ActivationFormat::Mxfp8
            },
        ] {
            let mut request = grouped_request(256, activation);
            request.weight = weight;
            request.scales = ScaleBindings::NONE;
            assert!(moe::instantiate(request.clone()).is_ok());
            request.projection = moe::Projection::GateUp;
            request.output_quantization = Some(kernel_core::contract::OutputQuantization::Mxfp8);
            assert_eq!(
                moe::instantiate(request).is_ok(),
                activation == ActivationFormat::Mxfp8
            );
        }
    }
}

#[test]
fn independent_grouped_entries_compile_and_audit_operand_precision() {
    use kernel_build::{BuildOptions, Toolchain, build};
    use std::{
        fs,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(!toolchain.skipped, "native compilation must run");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let retained = std::env::var_os("CHOPINFER_PRECISION_GROUPED_ARTIFACTS");
    let dir = retained.as_ref().map_or_else(
        || std::env::temp_dir().join(format!("precision-grouped-{}-{stamp}", std::process::id())),
        std::path::PathBuf::from,
    );
    fs::create_dir(&dir).unwrap();
    let mut operations = Vec::new();
    for k in [128, 640, 2560] {
        for (phase, capacity) in [(moe::Phase::Decode, 1), (moe::Phase::Prefill, 64)] {
            let mut request = grouped_request(k, ActivationFormat::Bf16);
            request.phase = phase;
            request.tokens_capacity = capacity;
            request.symbol = format!("independent_down_k{k}_t{capacity}_v1");
            operations.push(moe::instantiate(request).unwrap());
        }
    }
    for (activation, tag) in [
        (ActivationFormat::Bf16, "a16"),
        (ActivationFormat::Nvfp4, "a4"),
    ] {
        for (phase, capacity) in [(moe::Phase::Decode, 1), (moe::Phase::Prefill, 64)] {
            let mut request = grouped_request(256, activation);
            request.projection = moe::Projection::GateUp;
            request.phase = phase;
            request.tokens_capacity = capacity;
            request.symbol = format!("independent_gate_{tag}_t{capacity}_v1");
            operations.push(moe::instantiate(request).unwrap());
        }
    }
    let cubin = moe::cubin_spec("independent_precision_grouped", &operations).unwrap();
    build(
        &toolchain,
        std::slice::from_ref(&cubin),
        &dir,
        &dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let built = build(
        &toolchain,
        &[cubin],
        &dir,
        &dir.join("kernels.lock"),
        BuildOptions {
            lock_update: false,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    kernel_build::BuildReport::read(&dir.join(kernel_build::BUILD_REPORT_FILE))
        .unwrap()
        .verify_current()
        .unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), operations.len());
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(dump.status.success());
    let sass = String::from_utf8(dump.stdout).unwrap();
    for operation in operations {
        let section = sass
            .split(&format!("Function : {}", operation.symbol))
            .nth(1)
            .unwrap()
            .split("Function : ")
            .next()
            .unwrap();
        if operation.query.activation == ActivationFormat::Bf16 {
            assert!(section.contains("HMMA.16816"), "{}", operation.symbol);
            assert!(
                !section.contains("OMMA") && !section.contains("QMMA"),
                "{}",
                operation.symbol
            );
        } else {
            assert!(section.contains("OMMA"), "{}", operation.symbol);
        }
    }
    if retained.is_none() {
        fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn dense_builder_retains_every_legacy_instantiation_with_explicit_tuning() {
    for legacy in gemm::catalog(&kernels_sm120::Catalogs::base()) {
        let StaticQuery::Gemm(query) = legacy.key.query else {
            panic!("GEMM catalog query");
        };
        let shared = gemm::instantiate(DenseRequest {
            query,
            target: legacy.key.target,
            symbol: legacy.symbol.clone(),
            kernel: legacy.key.dense.unwrap(),
            knobs: Some(legacy.key.knobs),
        })
        .unwrap();
        assert_eq!(shared, legacy.instantiation);
        let old = gemm::cubin_spec(std::slice::from_ref(&legacy.instantiation)).unwrap();
        let new = gemm::cubin_spec(&[shared]).unwrap();
        assert_eq!(format!("{old:?}"), format!("{new:?}"));
    }
}

#[test]
fn dense_precision_pairs_build_without_registering_a_model() {
    let pairs = [
        (WeightFormat::Bf16, ActivationFormat::Bf16),
        (WeightFormat::Fp8PerTensor, ActivationFormat::Bf16),
        (
            WeightFormat::Fp8PerChannel,
            ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
        ),
        (
            WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
            ActivationFormat::Fp8Group128(ScaleMode::ExactF32),
        ),
        (WeightFormat::Mxfp8, ActivationFormat::Bf16),
        (WeightFormat::Mxfp8, ActivationFormat::Mxfp8),
        (WeightFormat::Mxfp4, ActivationFormat::Bf16),
        (WeightFormat::Mxfp4, ActivationFormat::Mxfp8),
        (WeightFormat::Nvfp4, ActivationFormat::Bf16),
        (WeightFormat::Nvfp4, ActivationFormat::Nvfp4),
    ];
    for (weight, activation) in pairs {
        for m_capacity in [1, 2, 4, 8, 16, 32, 64] {
            let query = GemmQuery {
                m_capacity,
                n: 256,
                k: 256,
                weight,
                activation,
                output: DType::Bf16,
                epilogues: Epilogues::default(),
                grouped: None,
            };
            // The 32-row BF16-input entries with BF16 or 128-K block FP8
            // weights take the fragment decode, the others TMA tiles.
            let fragment = m_capacity <= 32
                && activation == ActivationFormat::Bf16
                && matches!(
                    weight,
                    WeightFormat::Bf16 | WeightFormat::Fp8Block128x128(_)
                );
            let spec = gemm::instantiate(DenseRequest {
                query,
                target: Target::Sm120f,
                symbol: "consumer_projection_v1".into(),
                kernel: if fragment {
                    DenseKernel::FragmentDecode { splits: 1 }
                } else {
                    DenseKernel::Tiled { splits: 1 }
                },
                knobs: None,
            })
            .unwrap();
            assert_eq!(spec.key.query, StaticQuery::Gemm(query));
            assert!(spec.key.knobs.split_k_capacity > 0);
        }
    }
}

#[test]
fn dense_builder_checks_narrow_shapes_fusions_and_invalid_overrides() {
    let query = GemmQuery {
        m_capacity: 1,
        n: 12,
        k: 2560,
        weight: WeightFormat::Mxfp8,
        activation: ActivationFormat::Bf16,
        output: DType::Bf16,
        epilogues: Epilogues::default(),
        grouped: None,
    };
    let request = DenseRequest {
        query,
        target: Target::Sm120f,
        symbol: "small_projection_v1".into(),
        kernel: DenseKernel::Tiled { splits: 1 },
        knobs: None,
    };
    let spec = gemm::instantiate(request.clone()).unwrap();
    let mut bad = request.clone();
    bad.query.activation = ActivationFormat::Nvfp4;
    assert!(
        gemm::instantiate(bad)
            .unwrap_err()
            .0
            .contains("weight and activation pair")
    );
    let mut bad = request.clone();
    bad.query.epilogues.bias = true;
    assert!(gemm::instantiate(bad).is_err());
    let mut bad = request.clone();
    bad.knobs = Some(spec.key.knobs);
    bad.knobs.as_mut().unwrap().stages = 255;
    assert!(gemm::instantiate(bad).is_err());
    let mut bad = request;
    bad.symbol = "not a symbol".into();
    assert!(gemm::instantiate(bad).is_err());
}

#[test]
fn dense_builder_refuses_incomplete_fragment_blocks_and_packed_row_alignment() {
    let base = GemmQuery {
        m_capacity: 1,
        n: 256,
        k: 160,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::Bf16,
        output: DType::Bf16,
        epilogues: Epilogues::default(),
        grouped: None,
    };
    let request = |query, kernel| DenseRequest {
        query,
        target: Target::Sm120f,
        symbol: "consumer_v1".into(),
        kernel,
        knobs: None,
    };
    assert!(
        gemm::instantiate(request(base, DenseKernel::FragmentDecode { splits: 1 }))
            .unwrap_err()
            .0
            .contains("128-column")
    );
    let bad = GemmQuery {
        k: 16,
        weight: WeightFormat::Nvfp4,
        ..base
    };
    assert!(
        gemm::instantiate(request(bad, DenseKernel::Tiled { splits: 1 }))
            .unwrap_err()
            .0
            .contains("16-byte alignment")
    );
}

#[test]
fn independent_dense_entries_compile_and_audit_their_operand_precision() {
    use kernel_build::{BuildOptions, Toolchain, build};
    use std::{
        fs,
        process::Command,
        time::{SystemTime, UNIX_EPOCH},
    };
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(!toolchain.skipped, "native compilation must run");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("precision-dense-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let mut instances = Vec::new();
    for (tag, weight, activation, n, k, swiglu) in [
        (
            "narrow",
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16,
            12,
            2560,
            false,
        ),
        (
            "shared_gate",
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16,
            512,
            2560,
            true,
        ),
        (
            "shared_down",
            WeightFormat::Mxfp8,
            ActivationFormat::Bf16,
            2560,
            256,
            false,
        ),
        (
            "nv_a16",
            WeightFormat::Nvfp4,
            ActivationFormat::Bf16,
            256,
            256,
            false,
        ),
        (
            "nv_a4",
            WeightFormat::Nvfp4,
            ActivationFormat::Nvfp4,
            256,
            256,
            false,
        ),
    ] {
        for m_capacity in [1, 64] {
            let query = GemmQuery {
                m_capacity,
                n,
                k,
                weight,
                activation,
                output: DType::Bf16,
                epilogues: Epilogues {
                    swiglu: swiglu
                        .then_some(kernel_core::contract::SwigluColumns::ConcatenatedHalves),
                    ..Epilogues::default()
                },
                grouped: None,
            };
            instances.push(
                gemm::instantiate(DenseRequest {
                    query,
                    target: Target::Sm120f,
                    symbol: format!("precision_{tag}_m{m_capacity}_v1"),
                    kernel: DenseKernel::Tiled { splits: 1 },
                    knobs: None,
                })
                .unwrap(),
            );
        }
    }
    let mut cubin = gemm::cubin_spec(&instances).unwrap();
    cubin.name = "independent_precision_dense".into();
    let built = build(
        &toolchain,
        &[cubin],
        &dir,
        &dir.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), instances.len());
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(dump.status.success());
    let sass = String::from_utf8(dump.stdout).unwrap();
    for instance in &instances {
        let section = sass
            .split(&format!("Function : {}", instance.symbol))
            .nth(1)
            .unwrap()
            .split("Function : ")
            .next()
            .unwrap();
        let StaticQuery::Gemm(query) = instance.key.query else {
            unreachable!()
        };
        if query.activation == ActivationFormat::Bf16 {
            assert!(
                section.contains("HMMA.16816"),
                "{}: BF16 MMA absent",
                instance.symbol
            );
            assert!(
                !section.contains("QMMA") && !section.contains("OMMA"),
                "{}: lower-precision MMA in A16 entry",
                instance.symbol
            );
        } else {
            assert!(section.contains("OMMA.SF.16864"));
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

/// A dense entry's declared K splits fix every launch: a split capacity
/// override only sizes the grid, a capacity below the declaration is
/// refused, and an explicit count reproduces any reduction without a
/// width rule.
#[test]
fn dense_launches_use_their_declared_splits() {
    let query = GemmQuery {
        m_capacity: 1,
        n: 128,
        k: 4096,
        weight: WeightFormat::Bf16,
        activation: ActivationFormat::Bf16,
        output: DType::Bf16,
        epilogues: Epilogues::default(),
        grouped: None,
    };
    let request = |kernel, knobs| DenseRequest {
        query,
        target: Target::Sm120f,
        symbol: "declared_splits_v1".into(),
        kernel,
        knobs,
    };
    let whole = DenseKernel::FragmentDecode { splits: 1 };
    let mut knobs = gemm::instantiate(request(whole, None)).unwrap().key.knobs;
    knobs.split_k_capacity = 4;
    let spec = gemm::instantiate(request(whole, Some(knobs))).unwrap();
    assert_eq!(gemm::choose_split_k(&spec.key), Ok(1));
    assert_eq!(gemm::required_split_k(&spec.key), Some(1));
    let counts = |split_k| kernel_core::contract::RuntimeCounts {
        m: 1,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k,
    };
    assert!(gemm::plan(&spec.key, counts(1)).is_ok());
    assert!(gemm::plan(&spec.key, counts(4)).is_err());
    knobs.split_k_capacity = 2;
    assert!(
        gemm::instantiate(request(
            DenseKernel::FragmentDecode { splits: 4 },
            Some(knobs)
        ))
        .unwrap_err()
        .0
        .contains("below the declared K splits")
    );
    // Block-FP8 SwiGLU at N = 3712 once took 13 splits from a width rule;
    // a consumer declares them.
    let swiglu = GemmQuery {
        n: 3712,
        weight: WeightFormat::Fp8Block128x128(ScaleMode::ExactF32),
        epilogues: Epilogues {
            swiglu: Some(kernel_core::contract::SwigluColumns::ConcatenatedHalves),
            ..Epilogues::default()
        },
        ..query
    };
    let spec = gemm::instantiate(DenseRequest {
        query: swiglu,
        ..request(DenseKernel::FragmentDecode { splits: 13 }, None)
    })
    .unwrap();
    assert_eq!(gemm::choose_split_k(&spec.key), Ok(13));
    assert!(gemm::plan(&spec.key, counts(13)).is_ok());
    assert!(gemm::plan(&spec.key, counts(26)).is_err());
}
