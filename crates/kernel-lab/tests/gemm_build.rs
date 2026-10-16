//! Compile-only dense GEMM audit. No cubin is loaded on a GPU.

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::{
    Argument, OutputQuantization, RuntimeCounts, ScaleMode, TensorMap, TensorMapElement,
    TensorMapInterleave, TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle,
};
use kernels_sm120::gemm::{
    GemmBuffers, QuantizedBuffers, arguments, catalog, cubin_spec, fragment_weight_enabled, plan,
    quantized_pipeline, tensor_maps,
};
use runtime_bundle_abi::{BundleDeviceBufferHandleV1, BundleTensorMapV1, validate_tensor_map_v1};
use std::collections::HashMap;
use std::fs;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn map(rows: u64, k: u64, box_rows: u32) -> TensorMap {
    TensorMap {
        base_buffer: 10,
        dims: vec![k, rows],
        strides_bytes: vec![2, k * 2],
        box_dims: vec![64, box_rows],
        element_strides: vec![1, 1],
        swizzle: TensorMapSwizzle::B128,
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::B128,
        oob_fill: TensorMapOobFill::Zero,
        element: TensorMapElement::Bf16,
    }
}

fn fp8_map(rows: u64, k: u64, box_rows: u32) -> TensorMap {
    let mut result = map(rows, k, box_rows);
    result.strides_bytes = vec![1, k];
    result.box_dims[0] = 128;
    result.element = TensorMapElement::Fp8E4M3;
    result
}

fn fp4_map(rows: u64, k: u64, box_rows: u32) -> TensorMap {
    let mut result = map(rows, k, box_rows);
    result.strides_bytes = vec![1, k / 2];
    result.box_dims[0] = 256;
    result.element = TensorMapElement::U4Align8B;
    result
}

fn buffers(
    weight_scales: Option<u64>,
    activation_scales: Option<u64>,
    residual: Option<u64>,
) -> GemmBuffers {
    GemmBuffers {
        output: 11,
        partial: 12,
        semaphores: 13,
        weight_scales,
        activation_scales,
        residual,
        weight_global_scale: None,
        activation_global_scale: None,
    }
}

#[test]
fn every_catalog_tensor_map_obeys_bundle_abi() {
    for item in catalog(kernel_lab::catalogs()) {
        let maps = tensor_maps(&item.key, 10, 11).unwrap();
        for map in maps {
            let mut raw = BundleTensorMapV1::empty();
            raw.rank = 2;
            raw.element_type = match map.element {
                TensorMapElement::Bf16 => 3,
                TensorMapElement::Fp8E4M3 => 6,
                TensorMapElement::U4Align8B => 9,
                _ => panic!("unexpected catalog tensor-map element"),
            };
            raw.swizzle = match map.swizzle {
                TensorMapSwizzle::None => 0,
                TensorMapSwizzle::B128 => 1,
                _ => panic!("unexpected catalog tensor-map swizzle"),
            };
            raw.l2_promotion = 2;
            raw.buffer.buffer =
                BundleDeviceBufferHandleV1::from_raw((1_u64 << 32) | map.base_buffer);
            raw.buffer.length = map.strides_bytes[1] * map.dims[1];
            raw.global_dims[..2].copy_from_slice(&map.dims);
            raw.global_strides_bytes[0] = map.strides_bytes[1];
            raw.box_dims[..2].copy_from_slice(&map.box_dims);
            raw.element_strides[..2].copy_from_slice(&map.element_strides);
            validate_tensor_map_v1(raw)
                .unwrap_or_else(|error| panic!("{} tensor map {:?}: {error}", item.symbol, map));
        }
    }
}

#[test]
fn catalog_builds_with_mma_and_tma() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is required");
    assert!(
        !toolchain.skipped,
        "SM120 compilation must run on this host"
    );
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("chopinfer-gemm-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let items = catalog(kernel_lab::catalogs());
    let spec = cubin_spec(
        &items
            .iter()
            .map(|item| item.instantiation.clone())
            .collect::<Vec<_>>(),
    )
    .unwrap();
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
    .expect("all dense GEMM entries compile and pass audits");
    assert_eq!(built.len(), 1);
    assert_eq!(built[0].entries.len(), items.len());
    for (entry, item) in built[0].entries.iter().zip(&items) {
        assert_eq!(entry.local_bytes, 0, "{} spills", entry.symbol);
        let kernel_core::contract::StaticQuery::Gemm(query) = item.key.query else {
            unreachable!()
        };
        let stack_limit = 1024;
        assert!(
            entry.stack_bytes <= stack_limit,
            "{} exceeds stack allowance",
            entry.symbol
        );
        let counts = RuntimeCounts {
            m: query.m_capacity,
            batch: query.m_capacity,
            context: 0,
            routed_rows: 0,
            split_k: kernels_sm120::gemm::required_split_k(&item.key).unwrap_or(1),
        };
        let launch = plan(&item.key, counts).unwrap();
        assert_eq!(
            entry.max_threads, launch.block[0],
            "{} block size",
            entry.symbol
        );
        assert!(
            entry.shared_static_bytes + launch.dynamic_shared_bytes <= 101_376,
            "{} exceeds shared memory",
            entry.symbol
        );
        println!(
            "{} registers={} smem_static={} smem_dynamic={}",
            entry.symbol, entry.registers, entry.shared_static_bytes, launch.dynamic_shared_bytes
        );
    }
    let dump = Command::new(toolchain.home.join("bin/cuobjdump"))
        .arg("-sass")
        .arg(&built[0].path)
        .output()
        .unwrap();
    assert!(dump.status.success());
    let sass = String::from_utf8(dump.stdout).unwrap();
    let sections = sass
        .split("Function : ")
        .skip(1)
        .map(|section| section.split_once('\n').unwrap())
        .map(|(symbol, body)| (symbol.trim(), body))
        .collect::<HashMap<_, _>>();
    for item in &items {
        let body = sections
            .get(item.symbol.as_str())
            .unwrap_or_else(|| panic!("missing SASS for {}", item.symbol));
        let kernel_core::contract::StaticQuery::Gemm(query) = item.key.query else {
            unreachable!()
        };
        let mma = match query.activation {
            kernel_core::contract::ActivationFormat::Bf16 => "HMMA.16816",
            kernel_core::contract::ActivationFormat::Mxfp8
                if query.weight == kernel_core::contract::WeightFormat::Mxfp4 =>
            {
                "QMMA.SF.16832"
            }
            kernel_core::contract::ActivationFormat::Mxfp8 => "QMMA.SF.16832.F32.E4M3.E4M3.E8",
            kernel_core::contract::ActivationFormat::Nvfp4 => "OMMA.SF.16864",
            kernel_core::contract::ActivationFormat::Fp8Group128(_) => "QMMA.16832",
        };
        assert!(body.contains(mma), "{} missing {mma}", item.symbol);
        if query.weight == kernel_core::contract::WeightFormat::Mxfp4
            && query.activation == kernel_core::contract::ActivationFormat::Mxfp8
        {
            assert!(body.contains("E2M1"), "{} missing E2M1 MMA", item.symbol);
        }
        if query.activation == kernel_core::contract::ActivationFormat::Nvfp4 {
            assert!(body.contains("E2M1"), "{} missing E2M1 MMA", item.symbol);
        }
        if fragment_weight_enabled(&item.key) && query.m_capacity <= 32 {
            assert!(
                body.contains("LDG"),
                "{} missing fragment global loads",
                item.symbol
            );
        } else if fragment_weight_enabled(&item.key) {
            assert!(
                body.contains("UTMALDG"),
                "{} missing activation TMA",
                item.symbol
            );
            assert!(
                body.contains("LDSM.16.M88.4"),
                "{} missing swizzled activation ldmatrix",
                item.symbol
            );
        } else {
            assert!(body.contains("UTMALDG.2D"), "{} missing TMA", item.symbol);
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn argument_layout_matches_entry_parameter_order() {
    // MiMo's first dense entry, the attention output projection (K 2048).
    let item = model_mimo26_contract::sm120::gemm::catalog()
        .into_iter()
        .next()
        .unwrap();
    let counts = RuntimeCounts {
        m: 1,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    // Each entry's launches pass its declared K splits.
    let at = |key: &kernel_core::contract::CompileKey| RuntimeCounts {
        split_k: kernels_sm120::gemm::required_split_k(key).unwrap_or(1),
        ..counts
    };
    let args = arguments(
        &item.key,
        at(&item.key),
        tensor_maps(&item.key, 10, 14).unwrap(),
        buffers(None, None, None),
    )
    .unwrap();
    assert_eq!(args.0.len(), 15);
    assert_eq!(args.0[0], Argument::DeviceBuffer(10));
    assert_eq!(args.0[1], Argument::DeviceBuffer(14));
    assert_eq!(
        args.0[2..5],
        [
            Argument::DeviceBuffer(11),
            Argument::DeviceBuffer(12),
            Argument::DeviceBuffer(13)
        ]
    );
    assert_eq!(
        args.0[5..],
        [
            Argument::U32(1),
            Argument::U32(4096),
            Argument::U32(2048),
            Argument::U32(1),
            Argument::U32(0),
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::NullDevicePointer,
            Argument::NullDevicePointer
        ]
    );
    let qkv = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_mimo_qkv_ga_m1_v1")
        .unwrap();
    let [qkv_weight, qkv_activation] = tensor_maps(&qkv.key, 10, 14).unwrap();
    let fp8_args = arguments(
        &qkv.key,
        at(&qkv.key),
        [qkv_weight.clone(), qkv_activation.clone()],
        buffers(Some(14), None, None),
    )
    .unwrap();
    assert_eq!(
        fp8_args.0[10..13],
        [
            Argument::DeviceBuffer(14),
            Argument::NullDevicePointer,
            Argument::NullDevicePointer
        ]
    );
    let mut qkv_bad_swizzle = qkv_activation.clone();
    qkv_bad_swizzle.swizzle = TensorMapSwizzle::B128;
    assert!(
        arguments(
            &qkv.key,
            at(&qkv.key),
            [qkv_weight, qkv_bad_swizzle],
            buffers(Some(14), None, None),
        )
        .is_err()
    );
    assert!(
        arguments(
            &qkv.key,
            at(&qkv.key),
            [map(3392, 4096, 64), qkv_activation],
            buffers(Some(14), None, None)
        )
        .is_err()
    );
    let a8 = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_std_dense_a8_m1_v1")
        .unwrap();
    let a8_args = arguments(
        &a8.key,
        at(&a8.key),
        [fp8_map(4096, 4096, 64), fp8_map(1, 4096, 8)],
        buffers(Some(14), Some(15), None),
    )
    .unwrap();
    assert_eq!(
        a8_args.0[10..13],
        [
            Argument::DeviceBuffer(14),
            Argument::DeviceBuffer(15),
            Argument::NullDevicePointer
        ]
    );
    let mx = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_format_mxfp8_decode_v1")
        .unwrap();
    let mx_args = arguments(
        &mx.key,
        at(&mx.key),
        [fp8_map(256, 128, 64), fp8_map(1, 128, 8)],
        buffers(Some(17), Some(18), None),
    )
    .unwrap();
    assert_eq!(
        mx_args.0[10..13],
        [
            Argument::DeviceBuffer(17),
            Argument::DeviceBuffer(18),
            Argument::NullDevicePointer
        ]
    );
    let nv = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_std_dense_nvfp4_m1_v1")
        .unwrap();
    let mut nv_activation = map(1, 4096, 8);
    nv_activation.box_dims[0] = 256;
    nv_activation.swizzle = TensorMapSwizzle::None;
    let nv_maps = [fp4_map(4096, 4096, 64), nv_activation];
    assert!(
        arguments(
            &nv.key,
            at(&nv.key),
            nv_maps.clone(),
            buffers(Some(19), None, None),
        )
        .is_err()
    );
    let mut nv_buffers = buffers(Some(19), None, None);
    nv_buffers.weight_global_scale = Some(20);
    let nv_args = arguments(&nv.key, at(&nv.key), nv_maps, nv_buffers).unwrap();
    assert_eq!(nv_args.0[13], Argument::DeviceBuffer(20));
    let mx4 = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_std_dense_mxfp4_m1_v1")
        .unwrap();
    let mut mx4_activation = map(1, 4096, 8);
    mx4_activation.box_dims[0] = 256;
    mx4_activation.swizzle = TensorMapSwizzle::None;
    assert!(
        arguments(
            &mx4.key,
            at(&mx4.key),
            [fp4_map(4096, 4096, 64), mx4_activation],
            buffers(Some(19), None, None),
        )
        .is_ok()
    );
    let mx4_a8 = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_std_dense_mxfp4_a8_m1_v1")
        .unwrap();
    let mut mx4_a8_activation = fp8_map(1, 4096, 8);
    mx4_a8_activation.box_dims[0] = 256;
    mx4_a8_activation.swizzle = TensorMapSwizzle::None;
    let mut mx4_a8_bad_swizzle = mx4_a8_activation.clone();
    mx4_a8_bad_swizzle.swizzle = TensorMapSwizzle::B128;
    assert!(
        arguments(
            &mx4_a8.key,
            at(&mx4_a8.key),
            [fp4_map(4096, 4096, 64), mx4_a8_bad_swizzle],
            buffers(Some(19), Some(21), None),
        )
        .is_err()
    );
    let mx4_a8_args = arguments(
        &mx4_a8.key,
        at(&mx4_a8.key),
        [fp4_map(4096, 4096, 64), mx4_a8_activation],
        buffers(Some(19), Some(21), None),
    )
    .unwrap();
    assert_eq!(mx4_a8_args.0[11], Argument::DeviceBuffer(21));
    let nv_a4 = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_std_dense_nvfp4_a4_m1_v1")
        .unwrap();
    let nv_a4_maps = [fp4_map(4096, 4096, 64), fp4_map(1, 4096, 8)];
    let mut nv_a4_buffers = buffers(Some(19), Some(21), None);
    nv_a4_buffers.weight_global_scale = Some(20);
    assert!(
        arguments(
            &nv_a4.key,
            at(&nv_a4.key),
            nv_a4_maps.clone(),
            nv_a4_buffers
        )
        .is_err()
    );
    nv_a4_buffers.activation_global_scale = Some(22);
    let nv_a4_args = arguments(&nv_a4.key, at(&nv_a4.key), nv_a4_maps, nv_a4_buffers).unwrap();
    assert_eq!(nv_a4_args.0[13], Argument::DeviceBuffer(20));
    assert_eq!(nv_a4_args.0[14], Argument::DeviceBuffer(22));
    let swiglu = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_format_swiglu_interleaved_decode_v1")
        .unwrap();
    let mut gate_up = fp8_map(128, 128, 64);
    let mut activation = map(1, 128, 8);
    activation.box_dims[0] = 128;
    activation.swizzle = TensorMapSwizzle::None;
    assert!(
        arguments(
            &swiglu.key,
            at(&swiglu.key),
            [gate_up.clone(), activation.clone()],
            buffers(Some(19), None, None),
        )
        .is_err()
    );
    gate_up.element_strides[1] = 2;
    assert!(
        arguments(
            &swiglu.key,
            at(&swiglu.key),
            [gate_up.clone(), activation.clone()],
            buffers(Some(19), None, None),
        )
        .is_err()
    );
    gate_up.box_dims[1] *= 2;
    let fused_args = arguments(
        &swiglu.key,
        at(&swiglu.key),
        [gate_up, activation],
        buffers(Some(19), None, None),
    )
    .unwrap();
    assert_eq!(fused_args.0[7], Argument::U32(128));
    let residual = catalog(kernel_lab::catalogs())
        .into_iter()
        .find(|s| s.symbol == "chop_gemm_format_residual_decode_v1")
        .unwrap();
    assert!(
        arguments(
            &residual.key,
            at(&residual.key),
            [map(4096, 4096, 64), map(1, 4096, 8)],
            buffers(None, None, None)
        )
        .is_err()
    );
    let residual_args = arguments(
        &residual.key,
        at(&residual.key),
        [map(4096, 4096, 64), map(1, 4096, 8)],
        buffers(None, None, Some(16)),
    )
    .unwrap();
    assert_eq!(residual_args.0[12], Argument::DeviceBuffer(16));
}

#[test]
fn quantized_epilogue_stages_match_row_entry_abis() {
    let items = catalog(kernel_lab::catalogs());
    let base = items
        .iter()
        .find(|item| item.symbol == "chop_gemm_mimo_o_m1_v1")
        .unwrap();
    let counts = RuntimeCounts {
        m: 1,
        batch: 1,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    let buffers = QuantizedBuffers {
        gemm: buffers(None, None, None),
        codes: 21,
        scales: 22,
        live_rows: 23,
        row_amax: Some(24),
        global_scale: Some(25),
    };
    let maps = tensor_maps(&base.key, 10, 14).unwrap();
    for (format, label) in [
        (
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
            "fp8_exact",
        ),
        (
            OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32),
            "fp8_pow2",
        ),
        (OutputQuantization::Mxfp8, "mxfp8"),
    ] {
        let stages = quantized_pipeline(base, format, counts, maps.clone(), buffers).unwrap();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].symbol, base.symbol);
        assert_eq!(stages[1].symbol, format!("chop_row_quant_{label}_4096_v1"));
        assert_eq!(stages[1].arguments.0[0], Argument::DeviceBuffer(11));
        assert_eq!(stages[1].arguments.0[1], Argument::DeviceBuffer(21));
        assert!(stages[1].launch.pdl);
    }
    let nv = quantized_pipeline(
        base,
        OutputQuantization::Nvfp4,
        counts,
        maps.clone(),
        buffers,
    )
    .unwrap();
    assert_eq!(nv.len(), 4);
    assert_eq!(nv[1].symbol, "chop_row_nvfp4_row_amax_4096_v1");
    assert_eq!(nv[2].symbol, "chop_row_nvfp4_global_scale_v1");
    assert_eq!(nv[3].symbol, "chop_row_quant_nvfp4_4096_v1");
    assert_eq!(nv[3].arguments.0[3], Argument::DeviceBuffer(25));
    let mut missing_nv = buffers;
    missing_nv.row_amax = None;
    assert!(
        quantized_pipeline(
            base,
            OutputQuantization::Nvfp4,
            counts,
            maps.clone(),
            missing_nv
        )
        .is_err()
    );
    let router = items
        .iter()
        .find(|item| item.symbol == "chop_gemm_mimo_router_m1_v1")
        .unwrap();
    assert!(
        quantized_pipeline(
            router,
            OutputQuantization::Mxfp8,
            counts,
            [map(256, 4096, 64), map(1, 4096, 8)],
            buffers,
        )
        .is_err()
    );
    assert!(quantized_pipeline(base, OutputQuantization::Mxfp4, counts, maps, buffers).is_err());
    let wide = items
        .iter()
        .find(|item| item.symbol == "chop_gemm_std_wide_m1_v1")
        .unwrap();
    let wide_stages = quantized_pipeline(
        wide,
        OutputQuantization::Mxfp8,
        RuntimeCounts {
            split_k: kernels_sm120::gemm::required_split_k(&wide.key).unwrap(),
            ..counts
        },
        tensor_maps(&wide.key, 10, 14).unwrap(),
        buffers,
    )
    .unwrap();
    assert_eq!(wide_stages[1].symbol, "chop_row_quant_mxfp8_8192_v1");
    let gate_up = items
        .iter()
        .find(|item| item.symbol == "chop_gemm_mimo_mlp_gate_up_m1_v1")
        .unwrap();
    let mut gated_buffers = buffers;
    gated_buffers.gemm.weight_scales = Some(26);
    let gated_stages = quantized_pipeline(
        gate_up,
        OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
        counts,
        tensor_maps(&gate_up.key, 10, 14).unwrap(),
        gated_buffers,
    )
    .unwrap();
    assert_eq!(gated_stages[1].symbol, "chop_row_quant_fp8_exact_4096_v1");
}
