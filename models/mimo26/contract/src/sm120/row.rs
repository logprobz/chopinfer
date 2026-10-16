//! MiMo V2.6 Flash's row selections and their declarations.

use super::geometry::{self, GeometryModel};
use kernels_sm120::names::intern;

use kernel_core::contract::{OutputQuantization, RotaryStyle, ScaleMode};
use kernels_sm120::invariance::{Anchors, Invariance, Ties};
use kernels_sm120::row::builders::*;
use kernels_sm120::row::{RowInstantiation, row_local};
use kernels_sm120::{Profile, RankShape};

/// The row entries served prefill and exact verify steps launch, with
/// their declarations; every other row entry claims nothing. These are
/// row-local: each gives every row the bits it has alone, in any chunk and
/// beside any rows. MiMo's router breaks ties to the lower expert, its ring
/// write leaves the window ring only at the end of its rows. Its tiled
/// expert permutation writes route metadata that depends on every route of
/// the launch, so its claims are the routed operation's
/// (`Scope::RoutedOperation`), which the expert adapter tests composed.
pub fn served() -> Vec<(String, Invariance)> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| served_for(*owner))
        .collect()
}

fn served_for(owner: GeometryModel) -> Vec<(String, Invariance)> {
    let g = owner.geometry;
    let kernel_tag = g.kernel_tag;
    let local = row_local();
    let mut served = vec![
        (
            if geometry::pinned_names(g) {
                "chop_row_embedding_full_v1".to_owned()
            } else {
                format!("chop_row_{}_embedding_full_v1", g.kernel_tag)
            },
            local,
        ),
        (
            intern(
                [
                    "chop_row_rms_",
                    g.kernel_tag,
                    "_hf",
                    intern((g.hidden()).to_string()),
                    "_v1",
                ]
                .concat(),
            )
            .to_owned(),
            local,
        ),
        (
            intern(
                [
                    "chop_row_combine_",
                    g.kernel_tag,
                    intern((g.hidden()).to_string()),
                    "_v1",
                ]
                .concat(),
            )
            .to_owned(),
            local,
        ),
        (
            intern(
                [
                    "chop_row_router_",
                    g.kernel_tag,
                    "_sigmoid",
                    intern((g.experts()).to_string()),
                    "_v1",
                ]
                .concat(),
            )
            .to_owned(),
            local.with_ties(Ties::LowerIndex),
        ),
    ];
    for stage in ["count", "prefix", "scan", "scatter"] {
        served.push((
            format!("chop_row_permute_tile_{stage}_{kernel_tag}_b64_v1"),
            Invariance::PREFILL.of_routed_operation(),
        ));
    }
    for tag in ["", "_tp2"] {
        served.push((
            format!("chop_row_ring_write_{kernel_tag}{tag}_bf16_pos32_v1"),
            local.with_anchors(Anchors::None),
        ));
        for rope in [
            "ga_bf16_table_p64",
            "ga_fp8_table_p64",
            "swa_prefill_table_p32",
        ] {
            served.push((
                format!("chop_row_rope_{kernel_tag}{tag}_{rope}_pos32_v1"),
                local,
            ));
        }
    }
    served
}

/// Sets each served row entry's declaration (`served`).
fn declare(owner: GeometryModel, rows: &mut [RowInstantiation]) {
    let served = served_for(owner);
    for row in rows {
        if let Some((_, invariance)) = served.iter().find(|(symbol, _)| *symbol == row.symbol) {
            row.invariance = *invariance;
        }
    }
}

/// MiMo's selected rows, in their serving cubin order.
pub fn catalog() -> Vec<RowInstantiation> {
    geometry::MODELS
        .iter()
        .flat_map(|owner| catalog_for(*owner))
        .collect()
}

pub fn catalog_for(owner: GeometryModel) -> Vec<RowInstantiation> {
    let g = owner.geometry;
    let kernel_tag = g.kernel_tag;
    let mut rows = Vec::new();
    for (name, width, head, kind, residual, profile) in [
        (
            intern([g.kernel_tag, "_hf", intern((g.hidden()).to_string())].concat()),
            g.hidden(),
            0,
            3,
            false,
            owner.profile(4),
        ),
        (
            intern(
                [
                    g.kernel_tag,
                    if geometry::pinned_names(g) {
                        "_plain"
                    } else {
                        "_vision_plain"
                    },
                    intern((g.vision().hidden_size).to_string()),
                ]
                .concat(),
            ),
            g.vision().hidden_size,
            0,
            0,
            false,
            owner.profile(4),
        ),
        (
            intern(
                [
                    g.kernel_tag,
                    if geometry::pinned_names(g) {
                        "_plain"
                    } else {
                        "_audio_plain"
                    },
                    intern((g.audio().input_local_dim).to_string()),
                ]
                .concat(),
            ),
            g.audio().input_local_dim,
            0,
            0,
            false,
            owner.profile(4),
        ),
        (
            intern(
                [
                    g.kernel_tag,
                    "_hf",
                    intern((g.hidden()).to_string()),
                    "_residual",
                ]
                .concat(),
            ),
            g.hidden(),
            0,
            3,
            true,
            owner.profile(4),
        ),
    ] {
        rows.push(
            rmsnorm_entry(profile, name, width, head, kind, residual)
                .expect("MiMo's RMSNorm rows fit their templates"),
        );
    }
    for (name, output_quantization) in [
        (
            "fp8_exact",
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
        ),
        (
            "fp8_pow2",
            OutputQuantization::Fp8Group128(ScaleMode::PowerOfTwoF32),
        ),
        ("mxfp8", OutputQuantization::Mxfp8),
        ("nvfp4", OutputQuantization::Nvfp4),
    ] {
        rows.push(
            rmsnorm_quant_entry(
                owner.profile(4),
                &format!("rms_{kernel_tag}_fused_{name}"),
                g.hidden(),
                g.eps() as f32,
                true,
                output_quantization,
            )
            .expect("MiMo's fused RMSNorm rows are whole scale groups"),
        );
    }
    rows.push(
        rmsnorm_quant_entry(
            owner.profile(4),
            intern(
                [
                    "rms_",
                    g.kernel_tag,
                    "_hf",
                    intern((g.hidden()).to_string()),
                    "_fp8_exact",
                ]
                .concat(),
            ),
            g.hidden(),
            g.eps() as f32,
            false,
            OutputQuantization::Fp8Group128(ScaleMode::ExactF32),
        )
        .expect("MiMo's fused RMSNorm rows are whole scale groups"),
    );
    rows.extend(
        quantize_entries(
            owner.profile(4),
            g.hidden(),
            intern((g.hidden()).to_string()),
        )
        .expect("MiMo's hidden rows are whole quantizer groups"),
    );
    // The shared router, permute and combine builders compile top-k 8.
    assert!(g.top_k() == 8);
    // The shared NVFP4 global-scale builder compiles width 4096.
    assert!(g.hidden() == 4096);
    rows.push(nvfp4_global_scale(owner.profile(4)).expect("the row tuning's global scale"));
    rows.push(
        router_entry(
            owner.profile(4),
            intern([g.kernel_tag, "_sigmoid", intern((g.experts()).to_string())].concat()),
            g.experts(),
            1,
            false,
        )
        .expect("MiMo's sigmoid router ranks 256 experts"),
    );
    for (name, experts, block, profile) in [
        (
            intern([g.kernel_tag, "_b16"].concat()),
            g.experts(),
            16,
            owner.profile(4),
        ),
        (
            intern([g.kernel_tag].concat()),
            g.experts(),
            32,
            owner.profile(4),
        ),
        (
            intern([g.kernel_tag, "_b64"].concat()),
            g.experts(),
            64,
            owner.profile(4),
        ),
        (
            intern([g.kernel_tag, "_b128"].concat()),
            g.experts(),
            128,
            owner.profile(4),
        ),
    ] {
        rows.extend(
            expert_permute_entries(profile, name, experts, block)
                .expect("MiMo's permutation shape"),
        );
    }
    rows.extend(
        permute_tile_entries(
            owner.profile(4),
            &format!("{kernel_tag}_b64"),
            g.experts(),
            g.top_k(),
            64,
        )
        .expect("MiMo's tiled permutation shape"),
    );
    rows.push(
        expert_combine_entry(
            owner.profile(4),
            intern([g.kernel_tag, intern((g.hidden()).to_string())].concat()),
            g.hidden(),
            g.experts(),
            true,
        )
        .expect("MiMo's combine shape"),
    );
    for (name, sharded) in [("full", false), ("shard", true)] {
        rows.push(kernels_sm120::row::embedding(
            owner.profile(4),
            &format!("embedding_{name}"),
            g.vocab(),
            g.hidden(),
            sharded,
        ));
    }
    // Policy: generic SwiGLU laboratory widths, independent of model geometry.
    for width in [512, 1536, 3584, 4096, 7168] {
        rows.extend(swiglu_entries(owner.profile(4), width).expect("the SwiGLU laboratory widths"));
    }
    // TP4's vocabulary shard, and the full vocabulary.
    let mimo_tp4_vocab = mimo_rank_tag(owner, 4).1.vocab_shard();
    for (name, vocab, k) in [
        ("argmax_full", g.vocab(), 1),
        ("top20_full", g.vocab(), 20),
        ("argmax_shard", mimo_tp4_vocab, 1),
        ("top20_shard", mimo_tp4_vocab, 20),
    ] {
        rows.push(kernels_sm120::row::logits(owner.profile(4), name, vocab, k));
    }
    rows.extend(rope_catalog(owner, 4));
    rows.extend(dflash_catalog(owner, 4));
    // MiMo's per-rank rows at 2 ranks, in their own profiles.
    for profile in owner.profiles().into_iter().skip(1) {
        let ranks = profile.ranks().expect("MiMo profile has ranks");
        rows.extend(rope_catalog(owner, ranks));
        rows.extend(dflash_catalog(owner, ranks));
    }
    if !geometry::pinned_names(g) {
        for row in &mut rows {
            if [
                "quant_",
                "nvfp4_",
                "swiglu_",
                "embedding_",
                "logits_",
                "dflash",
            ]
            .iter()
            .any(|stem| row.symbol.starts_with(&format!("chop_row_{stem}")))
            {
                row.symbol =
                    row.symbol
                        .replacen("chop_row_", &format!("chop_row_{}_", g.kernel_tag), 1);
            }
        }
    }
    declare(owner, &mut rows);
    rows
}

/// MiMo's profile at `ranks`, its per-rank shapes and the name tag after
/// `mimo`: TP4 keeps its names, the others insert `_tp{ranks}`.
fn mimo_rank_tag(owner: GeometryModel, ranks: u32) -> (Profile, RankShape, &'static str) {
    let profile = owner.model.at(ranks);
    let g = owner.geometry;
    let rank = profile.rank_shape().expect("a MiMo profile");
    let tag = profile
        .tag()
        .strip_prefix(g.kernel_tag)
        .expect("MiMo tags start with mimo");
    (profile, rank, tag)
}

/// The DFlash drafter heads split over the ranks, and the vocabulary shard's argmax.
fn dflash_catalog(owner: GeometryModel, ranks: u32) -> Vec<RowInstantiation> {
    let g = owner.geometry;
    let (profile, rank, tag) = mimo_rank_tag(owner, ranks);
    let (ranks, vocab) = (rank.ranks(), rank.vocab_shard());
    let heads = rank.attention(geometry::dflash(g));
    let head_dim = geometry::dflash(g).qk_head;
    let mut rows = Vec::new();
    for (name, heads) in [("q", heads.q_heads), ("k", heads.kv_heads)] {
        rows.push(
            qk_norm_entry(
                profile,
                &format!("dflash{tag}_{name}_norm"),
                heads,
                head_dim,
                g.dflash().rms_norm_eps as f32,
            )
            .expect("MiMo's DFlash QK norm shape"),
        );
        rows.extend(
            rope_entries(
                profile,
                &format!("dflash{tag}_{name}"),
                heads,
                head_dim,
                g.dflash().rotary_dim,
                g.dflash().rope_theta as f32,
            )
            .expect("MiMo's DFlash RoPE shape"),
        );
    }
    rows.extend(
        sharded_argmax_entries(profile, &format!("dflash{tag}"), vocab, ranks)
            .expect("MiMo's vocabulary shard"),
    );
    rows
}

/// RoPE, KV writes and SWA ring writes of one MiMo rank at `ranks` (64 / ranks
/// Q heads; 4 GA and 8 SWA KV heads split over the ranks, one copied head
/// when fewer), plus the standard GPT-J rows at TP4.
fn rope_catalog(owner: GeometryModel, ranks: u32) -> Vec<RowInstantiation> {
    let g = owner.geometry;
    let kernel_tag = g.kernel_tag;
    let (profile, rank, tag) = mimo_rank_tag(owner, ranks);
    let qh = rank.attention(geometry::ga(g)).q_heads;
    let swa_qh = rank.attention(geometry::swa(g)).q_heads;
    let (qk, v) = (geometry::swa(g).qk_head, geometry::swa(g).v_head);
    let (ga, swa) = (
        rank.attention(geometry::ga(g)).kv_heads,
        rank.attention(geometry::swa(g)).kv_heads,
    );
    let mut rows = Vec::new();
    for (base, qh, kvh, qk, v, rotary, theta, ring, prefill, fp8) in [
        (
            "ga_bf16",
            qh,
            ga,
            geometry::ga(g).qk_head,
            geometry::ga(g).v_head,
            g.rotary(),
            g.rope_theta(),
            false,
            false,
            false,
        ),
        (
            "ga_fp8",
            qh,
            ga,
            geometry::ga(g).qk_head,
            geometry::ga(g).v_head,
            g.rotary(),
            g.rope_theta(),
            false,
            false,
            true,
        ),
        (
            "swa_decode",
            swa_qh,
            swa,
            geometry::swa(g).qk_head,
            geometry::swa(g).v_head,
            g.rotary(),
            g.swa_rope_theta(),
            true,
            false,
            false,
        ),
        (
            "swa_prefill",
            swa_qh,
            swa,
            geometry::swa(g).qk_head,
            geometry::swa(g).v_head,
            g.rotary(),
            g.swa_rope_theta(),
            true,
            true,
            false,
        ),
        (
            "swa_decode_fp8",
            swa_qh,
            swa,
            geometry::swa(g).qk_head,
            geometry::swa(g).v_head,
            g.rotary(),
            g.swa_rope_theta(),
            true,
            false,
            true,
        ),
        (
            "swa_prefill_fp8",
            swa_qh,
            swa,
            geometry::swa(g).qk_head,
            geometry::swa(g).v_head,
            g.rotary(),
            g.swa_rope_theta(),
            true,
            true,
            true,
        ),
        // Policy: the independent GPT-J laboratory shape.
        ("std_gptj", 16, 2, 128, 128, 128, 1e4, false, false, false),
    ] {
        if base == "std_gptj" && (ranks != 4 || !geometry::pinned_names(g)) {
            continue;
        }
        let name = if base == "std_gptj" {
            base.to_string()
        } else {
            format!("{kernel_tag}{tag}_{base}")
        };
        rows.extend(
            rope_kv_entries(
                if name == "std_gptj" {
                    Profile::StdDenseMoe
                } else {
                    profile
                },
                &name,
                RopeKvShape {
                    q_heads: qh,
                    kv_heads: kvh,
                    qk_dim: qk,
                    v_dim: v,
                    rotary_dim: rotary,
                    theta: theta as f32,
                    style: if name == "std_gptj" {
                        RotaryStyle::GptJ
                    } else {
                        RotaryStyle::Neox
                    },
                    ring,
                    prefill,
                    fp8,
                },
            )
            .expect("MiMo's RoPE KV write shape"),
        );
    }
    rows.extend(
        ring_write_entries(
            profile,
            &format!("{kernel_tag}{tag}"),
            swa,
            qk,
            v,
            g.window(),
        )
        .expect("MiMo's ring write shape"),
    );
    rows
}
