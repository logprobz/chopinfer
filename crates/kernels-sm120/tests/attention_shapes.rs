use kernel_core::contract::{AttentionPhase, DType};
use kernels_sm120::Profile;
use kernels_sm120::attention::{self, CacheLayout, Shape, gated, sparse};

fn geometry(head_dim: u32, rotary: u32) -> gated::Config {
    gated::Config {
        q_heads: 6,
        kv_heads: 1,
        head_dim,
        rotary,
        theta: 1e7,
        sections: if rotary == 32 {
            [6, 5, 5]
        } else {
            [11, 11, 10]
        },
        eps: 1e-6,
        page: 64,
        max_rows: 8192,
    }
}

#[test]
fn gated_rows_refuse_dimensions_outside_the_device_paths() {
    for dim in [64, 192] {
        for entry in gated::entries(Profile::StdDenseMoe, geometry(dim, 64)) {
            assert!(entry.contract().is_err(), "{} at head {dim}", entry.symbol);
        }
    }
    for dim in [128, 256] {
        for rotary in [32, 64] {
            for entry in gated::entries(Profile::StdDenseMoe, geometry(dim, rotary)) {
                entry.contract().unwrap();
            }
        }
    }
}

#[test]
fn gated_rows_refuse_unsupported_rotary_sections_and_warp_counts() {
    let mut invalid = Vec::new();
    for rotary in [16, 48, 128] {
        let mut g = geometry(128, rotary);
        g.sections = [rotary / 2, 0, 0];
        invalid.push(g);
    }
    for sections in [[0, 0, 0], [11, 11, 9]] {
        invalid.push(gated::Config {
            sections,
            ..geometry(128, 64)
        });
    }
    invalid.push(gated::Config {
        q_heads: 56,
        kv_heads: 2,
        ..geometry(128, 64)
    });
    for g in invalid {
        for entry in gated::entries(Profile::StdDenseMoe, g) {
            if matches!(
                entry.query,
                kernel_core::contract::GatedAttentionQuery::QkNormRopeKvWrite { .. }
            ) {
                assert!(entry.contract().is_err(), "{}: {g:?}", entry.symbol);
            }
        }
    }
    // The boundary still fits eight warps, including two KV heads.
    for entry in gated::entries(
        Profile::StdDenseMoe,
        gated::Config {
            q_heads: 48,
            kv_heads: 2,
            ..geometry(128, 64)
        },
    ) {
        entry.contract().unwrap();
    }
}

#[test]
fn gated_fold_refuses_unsupported_dimensions_rotary_and_sections() {
    for (qk_dim, v_dim, rotary, sections) in [
        (64, 64, 32, [6, 5, 5]),
        (192, 192, 64, [11, 11, 10]),
        (128, 64, 64, [11, 11, 10]),
        (128, 128, 16, [3, 3, 2]),
        (128, 128, 48, [8, 8, 8]),
        (128, 128, 128, [22, 21, 21]),
        (128, 128, 64, [0, 0, 0]),
        (128, 128, 64, [11, 11, 9]),
    ] {
        for kv_dtype in [DType::Bf16, DType::Fp8E4M3] {
            let mut variant = attention::variant(
                "shape",
                Profile::StdDenseMoe,
                AttentionPhase::Decode,
                Shape {
                    q_heads: 6,
                    kv_heads: 1,
                    qk_dim,
                    v_dim,
                    page_size: 64,
                    kv_dtype,
                    layout: CacheLayout::Paged,
                    window_left: None,
                    sink: false,
                    softcap: None,
                    causal: true,
                },
            );
            variant.validate().unwrap();
            variant.gated_fold = Some(
                gated::Config {
                    sections,
                    ..geometry(qk_dim, rotary)
                }
                .fold(),
            );
            assert_eq!(
                variant.instantiation().unwrap_err().0,
                "gated fold needs paged decode of 128- or 256-wide heads with 32 or 64 rotary dimensions in M-RoPE sections"
            );
        }
    }
}

#[test]
fn gated_decode_accepts_both_head_and_rotary_widths() {
    for dim in [128, 256] {
        for rotary in [32, 64] {
            for dtype in [DType::Bf16, DType::Fp8E4M3] {
                let g = geometry(dim, rotary);
                let mut variant = attention::variant(
                    "shape",
                    Profile::StdDenseMoe,
                    AttentionPhase::Decode,
                    Shape {
                        q_heads: g.q_heads,
                        kv_heads: g.kv_heads,
                        qk_dim: dim,
                        v_dim: dim,
                        page_size: g.page,
                        kv_dtype: dtype,
                        layout: CacheLayout::Paged,
                        window_left: None,
                        sink: false,
                        softcap: None,
                        causal: true,
                    },
                );
                variant.gated_fold = Some(g.fold());
                variant.validate().unwrap();
                variant.query.q_heads = 16;
                if dim == 128 {
                    assert_eq!(
                        variant.validate().unwrap_err().0,
                        "paged decode key tile partials exceed K tile fold scratch"
                    );
                    variant.gated_fold = None;
                    assert!(variant.validate().is_err());
                } else {
                    variant.validate().unwrap();
                }
            }
        }
    }
}

#[test]
fn paged_decode_refuses_every_shape_that_exceeds_fold_scratch() {
    for qk_dim in [64, 128, 192, 256] {
        for v_dim in [64, 128, 192, 256] {
            for group in [1, 6, 8, 16] {
                for kv_dtype in [DType::Bf16, DType::Fp8E4M3] {
                    let variant = attention::variant(
                        "shape",
                        Profile::StdDenseMoe,
                        AttentionPhase::Decode,
                        Shape {
                            q_heads: group * 2,
                            kv_heads: 2,
                            qk_dim,
                            v_dim,
                            page_size: 64,
                            kv_dtype,
                            layout: CacheLayout::Paged,
                            window_left: None,
                            sink: false,
                            softcap: None,
                            causal: true,
                        },
                    );
                    // Two key tiles need max, denominator and V partials
                    // in the K tile. With one key tile no fold runs.
                    let fits =
                        qk_dim + v_dim > 320 || 8 * (2 + v_dim / 2) * group.max(8) <= 64 * qk_dim;
                    let result = variant.instantiation();
                    assert_eq!(result.is_ok(), fits, "{qk_dim}/{v_dim}, group {group}");
                    if !fits {
                        assert_eq!(
                            result.unwrap_err().0,
                            "paged decode key tile partials exceed K tile fold scratch"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_attention_accepts_head_128_in_each_phase() {
    let g = sparse::Geometry {
        index_heads: 4,
        index_dim: 128,
        rotary_dim: 64,
        theta: 1e7,
        mrope: [11, 11, 10],
        eps: 1e-6,
        block: 4,
        ring: 16,
        top_blocks: 512,
        max_context: 262144,
        q_heads: 6,
        kv_heads: 1,
        head_dim: 128,
        page_size: 64,
        softmax_scale: 1.0 / 128_f32.sqrt(),
    };
    for entry in sparse::attention_entries(Profile::StdDenseMoe, g, 2) {
        entry.contract().unwrap();
    }
    for head_dim in [64, 192] {
        for entry in sparse::attention_entries(
            Profile::StdDenseMoe,
            sparse::Geometry {
                head_dim,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                ..g
            },
            2,
        ) {
            assert!(
                entry.contract().is_err(),
                "{} at head {head_dim}",
                entry.symbol()
            );
        }
    }
}
