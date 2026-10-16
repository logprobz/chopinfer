//! The NVFP4 layout's rank views, direct rank image, fit and capacity:
//! NVFP4 routed experts (U8 E2M1 pairs, E4M3 scales per 16 K values, an F32
//! global per expert projection), MXFP8 MTP experts (E4M3, U8 E8M0 scales
//! per 32 K values), a quarter of every expert per rank, BF16 elsewhere,
//! and no rank holds an `input_scale`.
use crate::capacity::{DEVICE, Planner, Serving};
use crate::catalog::DType;
use crate::contract::Shape;
use crate::direct::{INDEX_SCHEMA, Stored, VIEW_ALIGNMENT, checkpoint_layout, plan, test_support};
use crate::geometry::FLASH;
use crate::quantization::NVFP4_ROUTED;
use crate::views::{Binding, checkpoint_rank_views};
use kernel_core::latent_attention::LatentKvFormat;

const CONFIG: &[u8] = include_bytes!("../fixtures/nvfp4-config.json");
const LAYERS: &str = "model.language_model.layers";

fn find<'a>(layout: &'a [Stored], name: &str) -> &'a Stored {
    let full = format!("{LAYERS}.{name}");
    layout
        .iter()
        .find(|entry| entry.view.name == full)
        .unwrap_or_else(|| panic!("{name}"))
}

#[test]
fn rank_views_bind_nvfp4_and_mxfp8_experts_with_their_scales() {
    let shape = Shape::TP4;
    for rank in 0..4 {
        let views = checkpoint_rank_views(shape, &NVFP4_ROUTED, rank).unwrap();
        assert!(!views.iter().any(|v| v.name.ends_with(".input_scale")));
        let text = views.iter().map(|v| v.bytes()).sum::<u64>();
        // The TP4 weight plan's 47.38 GiB less the vision tower.
        assert_eq!(text, 50_875_242_744 - 1_127_254_016, "rank {rank}");
        // Every rank holds a slice of every expert: the last one's.
        let first = 287;
        let view = |name: String| {
            let name = format!("{LAYERS}.{name}");
            views
                .iter()
                .find(|v| v.name == name)
                .unwrap_or_else(|| panic!("{name}"))
        };
        let expert = format!("5.mlp.experts.{first}.gate_proj");
        for (suffix, dtype, dims) in [
            ("weight", DType::U8, vec![512, 2048]),
            ("weight_scale", DType::F8E4M3, vec![512, 256]),
            ("weight_scale_2", DType::F32, vec![]),
        ] {
            let v = view(format!("{expert}.{suffix}"));
            assert_eq!((v.dtype, v.shape.clone()), (dtype, dims), "{suffix}");
            assert_eq!(v.binding, Binding::Projection("moe.experts.gate-up"));
        }
        let mtp = format!("45.mlp.experts.{first}.down_proj");
        let v = view(format!("{mtp}.weight_scale"));
        assert_eq!((v.dtype, v.shape.clone()), (DType::U8, vec![4096, 16]));
        assert_eq!(v.start, 16 * u64::from(rank));
        assert_eq!(v.binding, Binding::Projection("mtp.moe.experts.down"));
        assert_eq!(view(format!("{mtp}.weight")).dtype, DType::F8E4M3);
        let q_b = view("3.self_attn.q_b_proj.weight".into());
        assert_eq!(
            (q_b.dtype, q_b.shape.clone()),
            (DType::Bf16, vec![4096, 1536])
        );
    }
}

#[test]
fn the_rank_image_packs_every_experts_codes_scales_and_globals() {
    let layout = checkpoint_layout(Shape::TP4, &NVFP4_ROUTED, 1).unwrap();
    let packed = |name: &str| {
        let entry = find(&layout, name);
        assert_eq!(entry.layout_id, "chained_rows_v1", "{name}");
        (entry.dtype, entry.view.shape.clone())
    };
    assert_eq!(
        packed("5.mlp.experts.gate_up.weight"),
        (DType::U8, vec![288, 1024, 2048])
    );
    assert_eq!(
        packed("5.mlp.experts.gate_up.weight_scale"),
        (DType::F8E4M3, vec![288, 1024, 256])
    );
    assert_eq!(
        packed("5.mlp.experts.gate_up.weight_scale_2"),
        (DType::F32, vec![288, 2])
    );
    assert_eq!(
        packed("5.mlp.experts.down.weight"),
        (DType::U8, vec![288, 4096, 256])
    );
    assert_eq!(
        packed("5.mlp.experts.down.weight_scale"),
        (DType::F8E4M3, vec![288, 4096, 32])
    );
    assert_eq!(
        packed("5.mlp.experts.down.weight_scale_2"),
        (DType::F32, vec![288, 1])
    );
    assert_eq!(
        packed("45.mlp.experts.gate_up.weight"),
        (DType::F8E4M3, vec![288, 1024, 4096])
    );
    assert_eq!(
        packed("45.mlp.experts.gate_up.weight_scale"),
        (DType::U8, vec![288, 1024, 128])
    );
    assert_eq!(
        packed("45.mlp.experts.down.weight_scale"),
        (DType::U8, vec![288, 4096, 16])
    );
    assert_eq!(
        packed("0.mlp.gate_up.weight"),
        (DType::Bf16, vec![6144, 4096])
    );
    // Each expert's gate global then its up global, every expert, each
    // global whole on every rank.
    let globals = find(&layout, "5.mlp.experts.gate_up.weight_scale_2");
    let names: Vec<_> = globals.pieces[..3]
        .iter()
        .map(|p| p.tensor.as_str())
        .collect();
    assert_eq!(
        names,
        [
            "model.language_model.layers.5.mlp.experts.0.gate_proj.weight_scale_2",
            "model.language_model.layers.5.mlp.experts.0.up_proj.weight_scale_2",
            "model.language_model.layers.5.mlp.experts.1.gate_proj.weight_scale_2",
        ]
    );
    assert_eq!(globals.pieces[1].offset - globals.pieces[0].offset, 4);
    let q_b = find(&layout, "3.self_attn.q_b_proj.weight");
    assert_eq!(
        (q_b.dtype, q_b.layout_id),
        (DType::Bf16, "checkpoint_rows_v1")
    );
    assert!(!layout.iter().any(|e| e.view.name.contains("input_scale")
        || e.pieces.iter().any(|p| p.tensor.contains("input_scale"))));
    for pair in layout.windows(2) {
        assert_eq!(pair[0].offset % VIEW_ALIGNMENT, 0);
        assert!(pair[0].offset + pair[0].bytes() <= pair[1].offset);
    }
}

#[test]
fn the_plan_follows_the_admitted_layout() {
    let request = test_support::checkpoint_request(&NVFP4_ROUTED, CONFIG, 2, |_| {});
    let planned = plan(&request).unwrap().unwrap();
    let layout = checkpoint_layout(Shape::TP4, &NVFP4_ROUTED, 2).unwrap();
    let pieces: usize = layout.iter().map(|entry| entry.pieces.len()).sum();
    assert_eq!(planned.records.len(), pieces);
    let index =
        bundle_runtime::index::parse(&planned.components[0].bytes, 2, INDEX_SCHEMA).unwrap();
    assert_eq!(index.views.len(), layout.len());
    assert_eq!(index.image_bytes, planned.arena_bytes);
    // 46.36 GiB: the text views at 4096-byte alignment. Re-pinned when the
    // lowered BF16 projections' weight views took their zero rows to the
    // row tile (49_751_158_784 before).
    assert_eq!(planned.arena_bytes, 49_778_946_048);
    // NVFP4 headers under the block-FP8 config are refused by tensor, the
    // first being the BF16 dense MLP.
    let mut request = request;
    request.documents[0].bytes = include_bytes!("../fixtures/config.json").to_vec();
    let error = plan(&request).unwrap_err();
    assert_eq!(
        error,
        format!(
            "{LAYERS}.0.mlp.gate_proj.weight is BF16 in model.safetensors, the catalog F8_E4M3"
        )
    );
}

#[test]
fn nvfp4_weights_free_kv_capacity() {
    let fit =
        crate::fit::check_checkpoint(Shape::TP4, &NVFP4_ROUTED, DEVICE.bytes, 32, true, 4).unwrap();
    assert_eq!(fit.weight_bytes, 50_875_242_744);
    let planner = Planner::for_checkpoint(&FLASH, &NVFP4_ROUTED, 4).unwrap();
    assert_eq!(planner.weights(true, true), 50_875_242_744);
    let serving = |context_parallel| Serving {
        ranks: 4,
        context_parallel,
        slots: 32,
        draft_tokens: 3,
        kv: LatentKvFormat::Fp8E4M3Block128,
        vision: false,
    };
    // A text package; block FP8 holds 665,792 and 2,249,472 tokens here.
    let one = planner.plan(serving(1), DEVICE).unwrap();
    let four = planner.plan(serving(4), DEVICE).unwrap();
    assert_eq!((one.pages, one.kv_tokens), (74_729, 4_782_656));
    assert_eq!(four.kv_tokens, 18_716_928);
    assert!(
        planner
            .table(DEVICE)
            .starts_with("flash nvfp4-routed TP4, ")
    );
}

/// NVIDIA's export: NVFP4 dense MLPs (row-split down projection), BF16 MTP
/// experts, F32 KDA convolutions stored as read, and BF16 mHC vectors
/// widened to the F32 the Sinkhorn engine reads.
#[test]
fn the_nvidia_rank_image_stores_dense_nvfp4_and_widens_bf16_vectors() {
    use crate::quantization::NVFP4_DENSE_ROUTED;
    use kernel_core::weight_transform::WeightTransform;
    let layout = checkpoint_layout(Shape::TP4, &NVFP4_DENSE_ROUTED, 1).unwrap();
    let entry = |name: &str| {
        let entry = find(&layout, name);
        (entry.dtype, entry.view.shape.clone(), entry.layout_id)
    };
    assert_eq!(
        entry("0.mlp.gate_up.weight"),
        (DType::U8, vec![6144, 2048], "chained_rows_v1")
    );
    assert_eq!(
        entry("0.mlp.gate_up.weight_scale"),
        (DType::F8E4M3, vec![6144, 256], "chained_rows_v1")
    );
    assert_eq!(
        entry("0.mlp.gate_up.weight_scale_2"),
        (DType::F32, vec![1, 2], "chained_rows_v1")
    );
    assert_eq!(
        entry("0.mlp.down_proj.weight"),
        (DType::U8, vec![4096, 1536], "checkpoint_rows_v1")
    );
    assert_eq!(
        find(&layout, "0.mlp.down_proj.weight").slice,
        Some((1, 1536..3072))
    );
    assert_eq!(
        entry("0.mlp.down_proj.weight_scale"),
        (DType::F8E4M3, vec![4096, 192], "checkpoint_rows_v1")
    );
    assert_eq!(
        entry("45.mlp.experts.gate_up.weight"),
        (DType::Bf16, vec![288, 1024, 4096], "chained_rows_v1")
    );
    let conv = find(&layout, "0.self_attn.q_conv1d.weight");
    assert_eq!(
        (conv.transform.clone(), conv.dtype, conv.layout_id),
        (WeightTransform::Copy, DType::F32, "f32_row_major_v1")
    );
    let base = find(&layout, "0.hc_attn_base");
    assert_eq!(
        (base.transform.clone(), base.dtype, base.layout_id),
        (
            WeightTransform::WidenBf16F32,
            DType::F32,
            "f32_row_major_v1"
        )
    );
    assert_eq!(base.bytes(), 24 * 4);
    const NVIDIA: &[u8] = include_bytes!("../fixtures/nvidia-nvfp4-config.json");
    let request = test_support::checkpoint_request(&NVFP4_DENSE_ROUTED, NVIDIA, 1, |_| {});
    let planned = plan(&request).unwrap().unwrap();
    let pieces: usize = layout.iter().map(|entry| entry.pieces.len()).sum();
    assert_eq!(planned.records.len(), pieces);
    // 47.84 GiB: the text views at 4096-byte alignment. Re-pinned when the
    // lowered BF16 projections' weight views took their zero rows to the
    // row tile (51_343_708_160 before).
    assert_eq!(planned.arena_bytes, 51_371_495_424);
}
