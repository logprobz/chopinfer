//! Every planned MiMo text view's size and shape matches the
//! descriptors of the kernels that read it: the fragment GEMMs and MoE
//! variants take the planned bytes and shapes as they are.

use kernel_core::contract::{
    ActivationFormat, RuntimeCounts, StaticQuery, TensorMap, TensorMapElement, TensorMapInterleave,
    TensorMapL2Promotion, TensorMapOobFill, TensorMapSwizzle, Validate, WeightFormat,
};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};
use kernels_sm120::{gemm, moe};

/// One view of a rank image: its name, stored shape and length.
struct View<'a> {
    name: &'a str,
    shape: &'a [u64],
    bytes: u64,
}

fn dense_map(
    rows: u32,
    k: u32,
    bytes: u64,
    box_k: u32,
    box_rows: u32,
    element: TensorMapElement,
) -> TensorMap {
    TensorMap {
        base_buffer: 1,
        dims: vec![u64::from(k), u64::from(rows)],
        strides_bytes: vec![bytes, u64::from(k) * bytes],
        box_dims: vec![box_k, box_rows],
        element_strides: vec![1, 1],
        swizzle: TensorMapSwizzle::B128,
        interleave: TensorMapInterleave::None,
        l2_promotion: TensorMapL2Promotion::B256,
        oob_fill: TensorMapOobFill::Zero,
        element,
    }
}

fn verify_descriptor(view: &View) {
    if view.name.contains(".mlp.experts.") {
        let projection = if view.name.contains("gate_up_proj") {
            moe::Projection::GateUp
        } else {
            moe::Projection::Down
        };
        let variant = moe::catalog(model_mimo26_contract::sm120::catalogs())
            .into_iter()
            .find(|v| {
                v.profile == model_mimo26_contract::sm120::config::PROFILE
                    && v.projection == projection
                    && v.phase == moe::Phase::Decode
                    && v.path == moe::InputPath::A16Gather
            })
            .unwrap();
        let fragments = variant
            .weight_fragment_layout()
            .expect("MiMo fragment weights");
        assert_eq!(fragments.experts as u64, view.shape[0]);
        assert_eq!(fragments.rows_per_expert as u64, view.shape[1]);
        if view.name.ends_with(".weight") {
            assert_eq!(fragments.k as u64, view.shape[2] * 2);
            assert_eq!(view.bytes, view.shape[0] * view.shape[1] * view.shape[2]);
            let map = variant.weight_tensor_map(1).unwrap();
            assert_eq!(map.dims, [view.shape[2], view.shape[0] * view.shape[1]]);
            assert_eq!(map.strides_bytes, [1, view.shape[2]]);
            assert_eq!(map.dims[0] * map.dims[1], view.bytes);
            assert_eq!(map.box_dims, [128, 16]);
            variant.arguments([1; 15], map).unwrap();
        } else {
            assert_eq!(fragments.k as u64, view.shape[2] * 32);
            assert_eq!(
                fragments.experts * (fragments.rows_per_expert / 16) * (fragments.k / 64) * 32,
                view.bytes as usize
            );
        }
        return;
    }
    if view.name.ends_with("weight_scale_inv") {
        let n: u32 = if view.name.contains("qkv_proj") {
            if view.shape[0] == 27 { 3392 } else { 3712 }
        } else if view.name.contains("gate_up_proj") {
            8192
        } else {
            4096
        };
        assert_eq!(view.shape, [u64::from(n.div_ceil(128)), 32]);
        assert_eq!(view.bytes, u64::from(n.div_ceil(128) * 32 * 4));
        return;
    }
    let dense = if view.name == "lm_head.weight" {
        Some((38144, 4096, "mimo_head_m1"))
    } else if view.name.ends_with("qkv_proj.weight") {
        Some((
            view.shape[0] as u32,
            4096,
            if view.shape[0] == 3392 {
                "mimo_qkv_ga_m1"
            } else {
                "mimo_qkv_swa_m1"
            },
        ))
    } else if view.name.ends_with("o_proj.weight") {
        Some((4096, 2048, "mimo_o_m1"))
    } else if view.name.ends_with("gate_up_proj.weight") {
        Some((8192, 4096, "mimo_mlp_gate_up_m1"))
    } else if view.name.ends_with("down_proj.weight") {
        Some((4096, 4096, "mimo_mlp_down_m1"))
    } else if view.name.ends_with("mlp.gate.weight") {
        Some((256, 4096, "mimo_router_m1"))
    } else {
        None
    };
    if let Some((n, k, symbol)) = dense {
        let variant = gemm::catalog(model_mimo26_contract::sm120::catalogs())
            .into_iter()
            .find(|v| v.symbol.contains(symbol))
            .unwrap();
        let StaticQuery::Gemm(query) = variant.key.query else {
            panic!("not GEMM")
        };
        assert_eq!((query.n, query.k), (n, k));
        assert!(gemm::fragment_weight_enabled(&variant.key));
        assert_eq!(view.shape, [u64::from(n), u64::from(k)]);
        let weight_bytes = if query.weight == WeightFormat::Bf16 {
            2
        } else {
            1
        };
        let weight_element = if weight_bytes == 2 {
            TensorMapElement::Bf16
        } else {
            TensorMapElement::Fp8E4M3
        };
        let fragment = FragmentWeightLayout {
            rows: n as usize,
            k: k as usize,
            format: if weight_bytes == 2 {
                FragmentWeightFormat::Bf16
            } else {
                FragmentWeightFormat::Fp8E4M3
            },
        };
        assert_eq!(fragment.packed_bytes().unwrap() as u64, view.bytes);
        let mut weight = dense_map(
            n,
            k,
            weight_bytes,
            u32::from(variant.key.knobs.tile_k),
            u32::from(variant.key.knobs.tile_m),
            weight_element,
        );
        if weight_bytes == 2 {
            weight.swizzle = TensorMapSwizzle::None;
        }
        assert_eq!(weight.strides_bytes[1] * weight.dims[1], view.bytes);
        assert_eq!(
            weight.box_dims,
            [
                u32::from(variant.key.knobs.tile_k),
                u32::from(variant.key.knobs.tile_m)
            ]
        );
        weight.validate().unwrap();
        let mut activation = dense_map(
            query.m_capacity,
            k,
            2,
            u32::from(variant.key.knobs.tile_k),
            u32::from(variant.key.knobs.tile_n),
            TensorMapElement::Bf16,
        );
        if variant.key.knobs.tile_k > 64 {
            activation.swizzle = TensorMapSwizzle::None;
        }
        assert_eq!(query.activation, ActivationFormat::Bf16);
        gemm::arguments(
            &variant.key,
            RuntimeCounts {
                m: 1,
                routed_rows: 0,
                split_k: gemm::required_split_k(&variant.key).unwrap_or(1),
                batch: 1,
                context: 1,
            },
            [weight, activation],
            gemm::GemmBuffers {
                output: 1,
                partial: 1,
                semaphores: 1,
                weight_scales: (weight_bytes == 1).then_some(1),
                activation_scales: None,
                residual: None,
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
        .unwrap_or_else(|error| panic!("{}: {error}", view.name));
    } else {
        let expected = if view.name == "model.embed_tokens.weight" {
            vec![152576, 4096]
        } else if view.name.ends_with("attention_sink_bias") {
            vec![16]
        } else if view.name.ends_with("e_score_correction_bias") {
            vec![256]
        } else {
            vec![4096]
        };
        assert_eq!(view.shape, expected, "{}", view.name);
        let element_bytes = if view.name.ends_with("e_score_correction_bias") {
            4
        } else {
            2
        };
        assert_eq!(
            view.bytes,
            view.shape.iter().product::<u64>() * element_bytes
        );
    }
}

#[test]
fn fragment_engine_descriptors_match_all_mimo_views() {
    let views = model_mimo26_contract::text_views(model_mimo26_contract::Tp::TP4);
    for name in [
        "lm_head.weight",
        "model.layers.1.self_attn.qkv_proj.weight",
        "model.layers.3.self_attn.qkv_proj.weight",
        "model.layers.1.self_attn.qkv_proj.weight_scale_inv",
        "model.layers.3.self_attn.qkv_proj.weight_scale_inv",
        "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight_scale_inv",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight_scale_inv",
        "model.layers.1.mlp.experts.gate_up_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj.weight_scale",
        "model.layers.1.mlp.experts.down_proj.weight",
        "model.layers.1.mlp.experts.down_proj.weight_scale",
    ] {
        let view = &views[name];
        verify_descriptor(&View {
            name,
            bytes: view.bytes,
            shape: &view.shape,
        });
    }
}
