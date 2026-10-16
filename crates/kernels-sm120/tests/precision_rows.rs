use kernel_core::contract::{DType, OutputQuantization as Q, ScaleMode, WeightFormat};
use kernels_sm120::row::{self, precision::*};

#[test]
fn quantizers_keep_supplied_and_dynamic_globals_separate() {
    let supplied = Quantizer::new(
        256,
        64,
        Q::Nvfp4,
        Nvfp4Global::Supplied,
        "consumer_nv_v1",
        ShapeBinding::Compiled,
        RowResources::SM120F,
    )
    .unwrap();
    let dynamic = Quantizer::new(
        256,
        64,
        Q::Nvfp4,
        Nvfp4Global::TensorAmax,
        "consumer_dyn_v1",
        ShapeBinding::Compiled,
        RowResources::SM120F,
    )
    .unwrap();
    assert_eq!(supplied.stages().len(), 1);
    assert_eq!(dynamic.stages().len(), 3);
    assert_eq!(supplied.scratch_bytes(), 0);
    assert_eq!(dynamic.scratch_bytes(), 260);
    assert_eq!(supplied.codes_bytes(), 8192);
    assert_eq!(supplied.scales_bytes(), 1024);
    let buffers = QuantizerBuffers {
        input: 1,
        codes: 2,
        scales: 3,
        live_rows: 4,
        global_scale: Some(5),
        row_amax: None,
    };
    assert_eq!(supplied.arguments(buffers).unwrap().len(), 1);
    assert!(dynamic.arguments(buffers).is_err());
    assert!(
        supplied
            .arguments(QuantizerBuffers {
                global_scale: None,
                ..buffers
            })
            .is_err()
    );
    assert!(
        supplied
            .arguments(QuantizerBuffers {
                row_amax: Some(6),
                ..buffers
            })
            .is_err()
    );
    assert_eq!(
        dynamic
            .arguments(QuantizerBuffers {
                row_amax: Some(6),
                ..buffers
            })
            .unwrap()
            .len(),
        3
    );
    assert!(supplied.stages()[0].plan(65).is_err());
    for (format, global) in [
        (Q::Nvfp4, Nvfp4Global::Absent),
        (Q::Mxfp8, Nvfp4Global::Supplied),
    ] {
        assert!(
            Quantizer::new(
                256,
                1,
                format,
                global,
                "bad_v1",
                ShapeBinding::Compiled,
                RowResources::SM120F
            )
            .is_err()
        );
    }
}

#[test]
fn row_quantizers_preserve_legacy_entry_specs_and_allow_new_widths() {
    for (format, label) in [
        (Q::Fp8Group128(ScaleMode::ExactF32), "fp8_exact"),
        (Q::Fp8Group128(ScaleMode::PowerOfTwoF32), "fp8_pow2"),
        (Q::Mxfp8, "mxfp8"),
        (Q::Nvfp4, "nvfp4"),
    ] {
        for width in [4096, 8192] {
            let suffix = if width == 4096 { "_std" } else { "" };
            let symbol = format!("chop_row_quant_{label}_{width}{suffix}_v1");
            let catalog = row::own_catalog();
            let old = catalog.iter().find(|r| r.symbol == symbol).unwrap();
            let new = Quantizer::new(
                width,
                64,
                format,
                if format == Q::Nvfp4 {
                    Nvfp4Global::Supplied
                } else {
                    Nvfp4Global::Absent
                },
                &symbol,
                ShapeBinding::Compiled,
                RowResources::SM120F,
            )
            .unwrap();
            let entry = new.stages()[0].entry();
            assert_eq!(entry.body, old.body);
            assert_eq!(format!("{:?}", entry.params), format!("{:?}", old.params));
            assert_eq!(entry.max_threads, old.threads);
        }
    }
    for width in [32, 160, 256, 2560] {
        let quant = Quantizer::new(
            width,
            64,
            Q::Mxfp8,
            Nvfp4Global::Absent,
            "consumer_mx_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F,
        )
        .unwrap();
        assert_eq!(quant.scales_bytes(), 64 * u64::from(width) / 32);
        assert_eq!(
            quant.stages()[0].plan(64).unwrap().grid[1],
            if width % 256 == 0 { 1 } else { width / 32 }
        );
    }
    for width in [0, 15, 129, u32::MAX - 15] {
        assert!(
            Quantizer::new(
                width,
                1,
                Q::Nvfp4,
                Nvfp4Global::Supplied,
                "bad_v1",
                ShapeBinding::Compiled,
                RowResources::SM120F
            )
            .is_err()
        );
    }
}

fn lookup_request(weight: WeightFormat, global_scale: Option<DType>) -> LookupRequest {
    LookupRequest {
        vocab: 37,
        width: 160,
        heads: 4,
        capacity: 64,
        weight,
        global_scale,
        ids: LookupIds::U64,
        shapes: ShapeBinding::Compiled,
    }
}

#[test]
fn lookups_preserve_scale_dtype_and_validate_shards() {
    for (weight, scale) in [
        (WeightFormat::Bf16, None),
        (WeightFormat::Fp8PerTensor, Some(DType::Bf16)),
        (WeightFormat::Fp8PerTensor, Some(DType::F32)),
        (WeightFormat::Nvfp4, Some(DType::F32)),
    ] {
        let lookup = Lookup::new(
            lookup_request(weight, scale),
            "table_v1",
            RowResources::SM120F,
        )
        .unwrap();
        let buffers = LookupBuffers {
            table: 1,
            ids: 2,
            local_scales: (weight == WeightFormat::Nvfp4).then_some(3),
            global_scale: scale.map(|_| 4),
            output: 5,
            live_rows: 6,
            row_offset: 16,
            shard_rows: 21,
        };
        lookup.arguments(buffers).unwrap();
        assert_eq!(lookup.stage().plan(7).unwrap().grid, [64, 4, 1]);
        assert!(
            lookup
                .arguments(LookupBuffers {
                    shard_rows: 22,
                    ..buffers
                })
                .is_err()
        );
        assert!(
            lookup
                .arguments(LookupBuffers {
                    row_offset: u64::MAX,
                    ..buffers
                })
                .is_err()
        );
        if scale.is_some() {
            assert!(
                lookup
                    .arguments(LookupBuffers {
                        global_scale: None,
                        ..buffers
                    })
                    .is_err()
            );
        }
    }
    for (weight, scale) in [
        (WeightFormat::Nvfp4, Some(DType::Bf16)),
        (WeightFormat::Fp8PerChannel, Some(DType::F32)),
        (WeightFormat::Fp8PerTensor, None),
        (WeightFormat::Mxfp8, Some(DType::Ue8M0)),
    ] {
        assert!(
            Lookup::new(
                lookup_request(weight, scale),
                "table_v1",
                RowResources::SM120F
            )
            .is_err()
        );
    }
    let mut bad = lookup_request(WeightFormat::Nvfp4, Some(DType::F32));
    bad.width = 159;
    assert!(Lookup::new(bad, "table_v1", RowResources::SM120F).is_err());
}

#[test]
fn copies_and_gathers_check_layouts_resources_and_arguments() {
    let source = CopyRows {
        row_stride: 12,
        column_stride: 1,
        column_offset: 0,
    };
    let dest = CopyRows {
        row_stride: 64,
        column_stride: 2,
        column_offset: 1,
    };
    let copy = LayoutCopy::new(
        12,
        64,
        source,
        dest,
        "copy_v1",
        ShapeBinding::Compiled,
        RowResources::SM120F,
    )
    .unwrap();
    copy.arguments(1, 2, 3).unwrap();
    assert!(copy.arguments(1, 1, 3).is_err());
    assert!(
        LayoutCopy::new(
            12,
            1,
            source,
            CopyRows {
                row_stride: 23,
                ..dest
            },
            "bad_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F
        )
        .is_err()
    );
    assert!(
        LayoutCopy::new(
            12,
            1,
            source,
            dest,
            "bad_v1",
            ShapeBinding::Compiled,
            RowResources {
                threads: 128,
                ..RowResources::SM120F
            }
        )
        .is_err()
    );
    assert!(
        LayoutCopy::new(
            12,
            1,
            source,
            dest,
            "bad_v1",
            ShapeBinding::Compiled,
            RowResources {
                shared_bytes: 0,
                ..RowResources::SM120F
            }
        )
        .is_err()
    );
    for input in [GatherInput::Tokens, GatherInput::Routed] {
        for quantized in [false, true] {
            let gather = Gather::new(
                256,
                1,
                4,
                2,
                16,
                input,
                quantized,
                "gather_v1",
                ShapeBinding::Compiled,
                RowResources::SM120F,
            )
            .unwrap();
            assert_eq!(gather.stage().capacity(), 32);
            gather
                .arguments(1, 2, 3, quantized.then_some(4), 5)
                .unwrap();
            assert!(
                gather
                    .arguments(1, 2, 3, (!quantized).then_some(4), 5)
                    .is_err()
            );
        }
    }
    assert!(
        Gather::new(
            159,
            1,
            4,
            2,
            16,
            GatherInput::Tokens,
            false,
            "bad_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F
        )
        .is_err()
    );
    assert!(
        Gather::new(
            256,
            1,
            4,
            5,
            16,
            GatherInput::Tokens,
            false,
            "bad_v1",
            ShapeBinding::Compiled,
            RowResources::SM120F
        )
        .is_err()
    );
}

#[test]
fn shared_row_entries_compile_without_a_model_profile() {
    use kernel_build::{BuildOptions, Toolchain, build};
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let mut stages = Vec::new();
    // Every stage in both forms; launch-shape stages share their header.
    for (shapes, tag) in [
        (ShapeBinding::Compiled, ""),
        (ShapeBinding::Launch, "_launch"),
    ] {
        for (format, global, label, width) in [
            (Q::Nvfp4, Nvfp4Global::Supplied, "nv", 256),
            (Q::Nvfp4, Nvfp4Global::TensorAmax, "nv_dynamic", 256),
            (
                Q::Fp8Group128(ScaleMode::ExactF32),
                Nvfp4Global::Absent,
                "fp",
                2560,
            ),
            (
                Q::Fp8Group128(ScaleMode::PowerOfTwoF32),
                Nvfp4Global::Absent,
                "fp_power",
                256,
            ),
            (Q::Mxfp8, Nvfp4Global::Absent, "mx", 256),
            (Q::Mxfp8, Nvfp4Global::Absent, "mx_tail", 160),
        ] {
            stages.extend(
                Quantizer::new(
                    width,
                    64,
                    format,
                    global,
                    &format!("quant_{label}{tag}_v1"),
                    shapes,
                    RowResources::SM120F,
                )
                .unwrap()
                .stages()
                .iter()
                .cloned(),
            );
        }
        for (weight, scale, label) in [
            (WeightFormat::Bf16, None, "bf"),
            (WeightFormat::Nvfp4, Some(DType::F32), "nv"),
            (WeightFormat::Fp8PerTensor, Some(DType::Bf16), "fp_bfscale"),
            (WeightFormat::Fp8PerTensor, Some(DType::F32), "fp_fscale"),
        ] {
            stages.push(
                Lookup::new(
                    LookupRequest {
                        shapes,
                        ..lookup_request(weight, scale)
                    },
                    &format!("lookup_{label}{tag}_v1"),
                    RowResources::SM120F,
                )
                .unwrap()
                .stage()
                .clone(),
            );
        }
        for quantized in [false, true] {
            stages.push(
                Gather::new(
                    256,
                    64,
                    4,
                    2,
                    16,
                    GatherInput::Tokens,
                    quantized,
                    &format!("gather_{}{tag}_v1", u8::from(quantized)),
                    shapes,
                    RowResources::SM120F,
                )
                .unwrap()
                .stage()
                .clone(),
            );
        }
        stages.push(
            LayoutCopy::new(
                12,
                64,
                CopyRows {
                    row_stride: 12,
                    column_stride: 1,
                    column_offset: 0,
                },
                CopyRows {
                    row_stride: 64,
                    column_stride: 2,
                    column_offset: 1,
                },
                &format!("copy_ab{tag}_v1"),
                shapes,
                RowResources::SM120F,
            )
            .unwrap()
            .stage()
            .clone(),
        );
    }
    let spec = cubin_spec("independent_precision_rows", &stages).unwrap();
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("precision-row-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let built = build(
        &toolchain,
        &[spec],
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
    assert_eq!(built[0].entries.len(), stages.len());
    for entry in &built[0].entries {
        assert_eq!(
            (entry.local_bytes, entry.stack_bytes),
            (0, 0),
            "{} spills",
            entry.symbol
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

/// A launch-shape stage serves every shape from one entry: its body,
/// arguments and declaration name no width, stride, head count or
/// vocabulary, and its builder appends those to the launch arguments.
#[test]
fn launch_shape_stages_take_their_shapes_as_arguments() {
    use kernel_core::contract::Argument;
    let rows = |row_stride, column_offset| CopyRows {
        row_stride,
        column_stride: 1,
        column_offset,
    };
    let copy = |columns, capacity, source, destination, shapes| {
        LayoutCopy::new(
            columns,
            capacity,
            source,
            destination,
            "copy_v1",
            shapes,
            RowResources::SM120F,
        )
        .unwrap()
    };
    let narrow = copy(12, 1, rows(16, 0), rows(24, 12), ShapeBinding::Launch);
    let wide = copy(
        256,
        8192,
        rows(256, 0),
        rows(3584, 3072),
        ShapeBinding::Launch,
    );
    assert_eq!(
        format!("{:?}", narrow.stage().entry()),
        format!("{:?}", wide.stage().entry())
    );
    assert_eq!(narrow.stage().declaration(), wide.stage().declaration());
    assert!(
        !narrow
            .stage()
            .declaration()
            .requirements
            .contains_key("width")
    );
    let compiled = copy(12, 1, rows(16, 0), rows(24, 12), ShapeBinding::Compiled);
    assert_eq!(compiled.stage().entry().params.len(), 3);
    assert_eq!(narrow.stage().entry().params.len(), 10);
    let args = narrow.arguments(1, 2, 3).unwrap().0;
    assert_eq!(args.len(), 10);
    assert_eq!(
        format!("{:?}", &args[3..]),
        format!("{:?}", [12, 16, 1, 0, 24, 1, 12].map(Argument::U32))
    );
    let twin = narrow
        .with_shapes(ShapeBinding::Compiled, "copy_v1")
        .unwrap();
    assert_eq!(
        format!("{:?}", twin.stage().entry()),
        format!("{:?}", compiled.stage().entry())
    );
    assert_eq!(twin.layout(), compiled.layout());
    assert_eq!(
        (twin.source_bytes(), twin.destination_bytes()),
        (compiled.source_bytes(), compiled.destination_bytes())
    );

    let lookup = |width, heads, vocab| {
        Lookup::new(
            LookupRequest {
                vocab,
                width,
                heads,
                capacity: 8,
                weight: WeightFormat::Nvfp4,
                global_scale: Some(DType::F32),
                ids: LookupIds::U64,
                shapes: ShapeBinding::Launch,
            },
            "lookup_v1",
            RowResources::SM120F,
        )
        .unwrap()
    };
    let (a, b) = (lookup(160, 4, 37), lookup(320, 8, 1 << 20));
    assert_eq!(
        format!("{:?}", a.stage().entry()),
        format!("{:?}", b.stage().entry())
    );
    assert_eq!(a.stage().declaration(), b.stage().declaration());
    assert_eq!(a.stage().plan(8).unwrap().grid, [8, 4, 1]);
    assert_eq!(b.stage().plan(8).unwrap().grid, [8, 8, 1]);
    let buffers = LookupBuffers {
        table: 1,
        ids: 2,
        local_scales: Some(3),
        global_scale: Some(4),
        output: 5,
        live_rows: 6,
        row_offset: 0,
        shard_rows: 37,
    };
    let args = a.arguments(buffers).unwrap().0;
    assert_eq!(
        format!("{:?}", &args[8..]),
        format!(
            "{:?}",
            [Argument::U32(160), Argument::U32(4), Argument::U64(37)]
        )
    );
    let compiled = a.with_shapes(ShapeBinding::Compiled, "lookup_v1").unwrap();
    assert_eq!(compiled.arguments(buffers).unwrap().0.len(), 8);
    assert!(
        compiled
            .stage()
            .entry()
            .body
            .contains("lookup<160,4,37,2,2,uint64_t>")
    );

    for (format, global, width) in [
        (Q::Nvfp4, Nvfp4Global::TensorAmax, 256),
        (Q::Mxfp8, Nvfp4Global::Absent, 256),
        (Q::Mxfp8, Nvfp4Global::Absent, 160),
        (
            Q::Fp8Group128(ScaleMode::ExactF32),
            Nvfp4Global::Absent,
            2560,
        ),
    ] {
        let quantizer = |width| {
            Quantizer::new(
                width,
                64,
                format,
                global,
                "quant_v1",
                ShapeBinding::Launch,
                RowResources::SM120F,
            )
            .unwrap()
        };
        let launch = quantizer(width);
        let other = quantizer(width * 2);
        for (a, b) in launch.stages().iter().zip(other.stages()) {
            assert_eq!(format!("{:?}", a.entry()), format!("{:?}", b.entry()));
            assert_eq!(a.declaration(), b.declaration());
        }
        let buffers = QuantizerBuffers {
            input: 1,
            codes: 2,
            scales: 3,
            live_rows: 4,
            global_scale: (global != Nvfp4Global::Absent).then_some(5),
            row_amax: (global == Nvfp4Global::TensorAmax).then_some(6),
        };
        let compiled = launch
            .with_shapes(ShapeBinding::Compiled, "quant_v1")
            .unwrap();
        for (launch, compiled) in launch
            .arguments(buffers)
            .unwrap()
            .iter()
            .zip(compiled.arguments(buffers).unwrap())
        {
            // The global-scale reduction has no shape.
            let (launch, compiled) = (&launch.0, &compiled.0);
            let extra = launch.len() - compiled.len();
            assert!(extra <= 1);
            if extra == 1 {
                assert_eq!(
                    format!("{:?}", launch.last()),
                    format!("{:?}", Some(Argument::U32(width)))
                );
            }
        }
        assert_eq!(
            launch.stages().last().unwrap().plan(64).unwrap().grid,
            compiled.stages().last().unwrap().plan(64).unwrap().grid
        );
    }

    for quantized in [false, true] {
        let gather = Gather::new(
            256,
            64,
            4,
            2,
            16,
            GatherInput::Tokens,
            quantized,
            "gather_v1",
            ShapeBinding::Launch,
            RowResources::SM120F,
        )
        .unwrap();
        let args = gather
            .arguments(1, 2, 3, quantized.then_some(4), 5)
            .unwrap()
            .0;
        assert_eq!(
            format!("{:?}", &args[args.len() - 2..]),
            format!("{:?}", [256, 2].map(Argument::U32))
        );
        let compiled = gather
            .with_shapes(ShapeBinding::Compiled, "gather_v1")
            .unwrap();
        assert_eq!(compiled.stage().capacity(), gather.stage().capacity());
        assert!(compiled.stage().entry().body.contains("<256,2>"));
    }

    let header = |shapes| {
        cubin_spec(
            "rows",
            std::slice::from_ref(copy(12, 1, rows(16, 0), rows(24, 0), shapes).stage()),
        )
        .unwrap()
        .headers
        .iter()
        .any(|h| h.contains("precision_launch.cuh"))
    };
    assert!(header(ShapeBinding::Launch));
    assert!(!header(ShapeBinding::Compiled));
}
