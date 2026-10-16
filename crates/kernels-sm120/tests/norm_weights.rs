use kernel_build::{BuildOptions, EntryParam, Toolchain};
use kernel_core::contract::DType;
use kernels_sm120::comm::{
    Epilogue,
    epilogue::{GroupedRmsNorm, HC_INJECT_NORM_EPS_BITS, NormReduction},
    hyper_connection::{self as hc, HcShape, HcTransport},
    hyper_connection_decode::{self as decode, HcDecodeTransport},
    input,
};

fn legacy() -> Epilogue {
    Epilogue {
        header: "comm/epilogue/hc_inject_norm.cuh",
        template: "chop::comm::epilogue::HcInjectNorm",
        width: 2560,
        streams: 4,
    }
}

fn transport(epilogue: Epilogue) -> HcTransport {
    HcTransport {
        shape: HcShape {
            epilogue,
            input: input::BF16,
        },
        eps_bits: HC_INJECT_NORM_EPS_BITS,
        cubin: "norm_prefill",
        symbol: "chop_norm_one_v1",
        scatter_symbol: "chop_norm_scatter_v1",
        gather_symbol: "chop_norm_gather_v1",
    }
}

#[test]
fn dtype_selection_preserves_bf16_wrappers_and_types_f32_pointers() {
    let old = legacy();
    assert_eq!(old.with_norm_weight_dtype(DType::Bf16).unwrap(), old);
    let bf16 = hc::cubin_specs(&transport(old));
    let preserved = hc::cubin_specs(&transport(old.with_norm_weight_dtype(DType::Bf16).unwrap()));
    assert_eq!(
        kernel_build::generate(&bf16[0]).unwrap().translation_unit,
        kernel_build::generate(&preserved[0])
            .unwrap()
            .translation_unit
    );
    let f32 = old.with_norm_weight_dtype(DType::F32).unwrap();
    assert_eq!(f32.weight_pointer_type(), "const float");
    let specs = hc::cubin_specs(&transport(f32));
    for entry in &specs[0].entries {
        for param in &entry.params {
            if let EntryParam::DevicePointer { name, cxx_type } = param {
                match name.as_str() {
                    "weight" => assert_eq!(cxx_type, "const float"),
                    "input" | "residual" | "logits" => assert_eq!(cxx_type, "const uint16_t"),
                    _ => (),
                }
            }
        }
    }
    for dtype in [DType::F16, DType::U8] {
        assert!(old.with_norm_weight_dtype(dtype).is_err());
    }
    assert!(
        Epilogue { streams: 2, ..old }
            .with_norm_weight_dtype(DType::F32)
            .is_err()
    );
    let mut t = transport(f32);
    t.eps_bits = 1e-5_f32.to_bits();
    assert!(hc::contract(&t, t.query(2)).is_err());
}

#[test]
fn grouped_rows_refuse_invalid_shapes_and_keep_explicit_reduction_order() {
    let base = GroupedRmsNorm {
        weight_dtype: DType::F32,
        group_width: 2560,
        groups: 4,
        reduction: NormReduction::ShuffleDown,
    };
    assert_eq!(base.plan(8, 0).unwrap().grid, [8, 4, 1]);
    assert!(base.plan(8, 9).is_err());
    assert!(base.plan(0, 0).is_err());
    let max = i32::MAX as u32;
    assert_eq!(base.plan(max, max).unwrap().grid[0], max);
    assert!(base.plan(max + 1, 0).is_err());
    assert!(base.plan(u32::MAX, 0).is_err());
    for invalid in [
        GroupedRmsNorm {
            weight_dtype: DType::F16,
            ..base
        },
        GroupedRmsNorm {
            group_width: 0,
            ..base
        },
        GroupedRmsNorm { groups: 0, ..base },
        GroupedRmsNorm {
            group_width: u32::MAX,
            ..base
        },
        GroupedRmsNorm {
            groups: 65536,
            ..base
        },
    ] {
        assert!(invalid.entry("invalid").is_err());
    }
    let serial = GroupedRmsNorm {
        reduction: NormReduction::XorThenSerial,
        ..base
    };
    assert_ne!(
        base.entry("norm").unwrap().body,
        serial.entry("norm").unwrap().body
    );
}

#[test]
fn f32_prefill_decode_and_wide_transports_compile_without_spills() {
    let t = transport(legacy().with_norm_weight_dtype(DType::F32).unwrap());
    let d = HcDecodeTransport {
        shape: t.shape,
        eps_bits: t.eps_bits,
        cubin: "norm_decode",
        wide_cubin: "norm_wide",
        symbols_v2: [
            "chop_norm_d2v2_v1",
            "chop_norm_d4v2_v1",
            "chop_norm_d8v2_v1",
        ],
        symbols_v3: [
            "chop_norm_d2v3_v1",
            "chop_norm_d4v3_v1",
            "chop_norm_d8v3_v1",
        ],
        wide_symbols: ["chop_norm_w2_v1", "chop_norm_w4_v1", "chop_norm_w8_v1"],
    };
    let specs = hc::cubin_specs(&t)
        .into_iter()
        .chain(decode::cubin_specs(&d, &hc::RANKS))
        .chain(decode::wide_cubin_specs(&d, &hc::RANKS))
        .collect::<Vec<_>>();
    for spec in &specs {
        for entry in &spec.entries {
            for param in &entry.params {
                if let EntryParam::DevicePointer { name, cxx_type } = param
                    && name == "weight"
                {
                    assert_eq!(cxx_type, "const float");
                }
            }
        }
    }
    let out = std::env::temp_dir().join(format!("norm-transports-{}", std::process::id()));
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let built = kernel_build::build(
        &toolchain,
        &specs,
        &out,
        &out.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(built.len(), 3);
    for entry in built.iter().flat_map(|c| &c.entries) {
        assert_eq!(entry.local_bytes, 0, "{}", entry.symbol);
        assert_eq!(entry.stack_bytes, 0, "{}", entry.symbol);
    }
    std::fs::remove_dir_all(out).unwrap();
}
