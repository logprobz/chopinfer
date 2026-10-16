//! Qwen3.8 Flash dense GEMMs at their per-rank shapes: TP4's under
//! `crate::sm120::config::PROFILE`, with every rank-invariant shape, and the shapes
//! that change with the rank count under the TP2 profile.
//!
//! Every Qwen dense weight is BF16 in the fragment-major layout, with BF16
//! activations. Shapes that fit the fragment decode and prefill families are
//! ordinary `GemmVariant` entries ([`catalog`]). [`fused_catalog`] holds the
//! entries of the model-shaped kernels in `kernels_sm120::gemm::fused`:
//! cluster decode buckets, short-K and skinny entries.

use kernel_core::contract::{DType, DenseKernel, GemmQuery, StaticQuery, SwigluColumns};
use kernels_sm120::Profile;
use kernels_sm120::gemm::fused::{ClusterShape, Entry, Kind, cluster_kind};
use kernels_sm120::gemm::{GemmVariant, WHOLE_K_DECODE, bf16_query, instance};
use kernels_sm120::invariance::{Declared, Invariance, OpKind};

pub use crate::contract::{HC_RANK, HIDDEN, WIDE as STREAMS_WIDTH};
/// Entry symbol of the hyper-connection up projection and stream mix.
pub const HC_UP_MIX: &str = "chop_gemm_qwen38_hc_up_mix_decode_v1";
/// Hyper-connection geometry supplied to the shared fused kernels.
pub const HC: kernels_sm120::gemm::fused::HcGeometry = kernels_sm120::gemm::fused::HcGeometry {
    width: HIDDEN,
    streams: crate::contract::STREAMS,
    rank: HC_RANK,
    up_mix_symbol: HC_UP_MIX,
};

/// TP4's per-rank shapes: the TP4 entries are the rank operation list
/// evaluated at four ranks.
const TP4: crate::sm120::config::RankShape = crate::sm120::config::TP4;
/// Per-rank vocabulary shard of 248,320 at TP4.
pub const VOCAB_SHARD: u32 = TP4.vocab_shard();
/// Prefill chunk capacity.
pub const PREFILL_ROWS: u32 = crate::contract::PREFILL_TOKENS;
/// Value heads per TP4 rank, the row count of in_proj_a and in_proj_b.
pub const VALUE_HEADS: u32 = TP4.linear_value_heads();
/// The shared expert on one TP4 rank. Each rank's slice of the intermediate
/// 640 (160 columns) is zero-padded to 256, which the SwiGLU output tiles
/// and the prefill K stages require; the padded gate and up rows are zero,
/// so they add nothing through the down projection.
pub const SHARED_INTERMEDIATE: u32 = TP4.shared_padded();

const DECODE_BUCKETS: [u32; 6] = crate::contract::DECODE_ENTRY_BUCKETS;

/// Qwen3.8 profiles besides TP4 whose per-rank GEMM shapes have entries of
/// their own.
pub const OTHER_RANKS: [u32; 1] = [2];

// PLE value and MTP fc reuse TP4's lin_qkv entry only at the same shape.
const _: () = assert!(TP4.linear_qkv() == HIDDEN && crate::contract::PLE_EMBED_DIM == HIDDEN);

/// The F32 head's prefill takes one K segment at every rank count: its K
/// chain is then the `head_f32_m{M}` decode entries' (128-K blocks from
/// zero, folded in order), so a target logits row has the same bits
/// whichever head entry its step selects (decode buckets, more rows than
/// the largest bucket, or scored prompt rows).
const HEAD_F32_SEGMENTS: u8 = 1;

/// The K splits kernel-lab launched these fragment decode projections with,
/// at every decode bucket, before each entry declared the single split
/// Qwen3.8 serves them with (`kernels_sm120::gemm::lab_split_catalog`).
pub const LAB_SPLITS: [(&str, u8); 6] = [
    ("chop_gemm_qwen38_head", 2),
    ("chop_gemm_qwen38_head_f32", 2),
    ("chop_gemm_qwen38_shared_gate_up", 20),
    ("chop_gemm_qwen38_shared_down", 2),
    ("chop_gemm_qwen38_tp2_shared_gate_up", 20),
    ("chop_gemm_qwen38_tp2_shared_down", 3),
];

pub fn lab_splits() -> Vec<(String, u8)> {
    let mut splits = Vec::new();
    for owner in super::geometry::MODELS {
        for &(stem, count) in &LAB_SPLITS {
            let profile = owner.model.at(if stem.contains("_tp2_") { 2 } else { 4 });
            let shape = super::geometry::shape(profile).unwrap();
            let k = if stem.contains("shared_down") {
                shape.shared_padded()
            } else {
                owner.geometry.hidden()
            };
            let count = if super::tuning::measured(owner.geometry)
                || k.is_multiple_of(u32::from(count) * 128)
            {
                count
            } else {
                1
            };
            splits.push((stem.replacen("qwen38", owner.geometry.kernel_tag, 1), count));
        }
    }
    splits
}

/// A head decode bucket: prefill steps run the prompt-ending rows through
/// it, launched unsplit, so each row has its one-row bits whatever rows
/// share the launch (`qwen_head_rows_match_across_entries`).
fn head_rows(item: GemmVariant) -> GemmVariant {
    let StaticQuery::Gemm(query) = item.key.query else {
        unreachable!("dense entries hold GEMM queries")
    };
    item.declare(Invariance::any_rows(query.m_capacity))
}

/// The shared expert's gate/up query of `m` rows over `shared` columns.
fn shared_gate_up(g: crate::geometry::Geometry, m: u32, shared: u32) -> GemmQuery {
    let mut query = bf16_query(m, 2 * shared, g.hidden(), DType::Bf16);
    query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
    query
}

/// The fragment-family entries: TP4's shared expert decode buckets and
/// prefill, the head's BF16 and F32 decode buckets, the 8192-row prefill of
/// every dense role and the F32 head's, then TP2's
/// (`rank_dense_entries`).
///
/// 8192-row prefill for every fragment-family shape, the F32 head for
/// scored prompt logits (as MiMo selects its head), and the head's decode
/// buckets in BF16 (the MTP drafter's argmax) and F32 (every target logits
/// row, as MiMo's `head_f32_m{M}`). The other decode buckets use the
/// cluster decode ([`fused_catalog`]), which was faster for each of them on
/// a shared GPU 0; the fragment decode head is faster up to 16 tokens.
pub fn catalog() -> Vec<GemmVariant> {
    super::geometry::MODELS
        .iter()
        .flat_map(catalog_for)
        .collect()
}

fn catalog_for(owner: &super::geometry::GeometryModel) -> Vec<GemmVariant> {
    let g = owner.geometry;
    let profile = owner.model.at(4);
    let shape = super::geometry::shape(profile).unwrap();
    let tag = profile.tag();
    let mut items = Vec::new();
    for m in DECODE_BUCKETS {
        let tn = m.max(8) as u16;
        items.push(instance(
            profile,
            &format!("{tag}_shared_gate_up_m{m}"),
            shared_gate_up(g, m, shape.shared_padded()),
            64,
            tn,
            WHOLE_K_DECODE,
        ));
        items.push(instance(
            profile,
            &format!("{tag}_shared_down_m{m}"),
            bf16_query(m, g.hidden(), shape.shared_padded(), DType::Bf16),
            64,
            tn,
            WHOLE_K_DECODE,
        ));
    }
    items.push(instance(
        profile,
        &format!("{tag}_shared_gate_up_prefill"),
        shared_gate_up(g, PREFILL_ROWS, shape.shared_padded()),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: super::tuning::prefill(g, "shared_gate_up", g.hidden()),
        },
    ));
    items.push(instance(
        profile,
        &format!("{tag}_shared_down_prefill"),
        bf16_query(PREFILL_ROWS, g.hidden(), shape.shared_padded(), DType::Bf16),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: super::tuning::prefill(g, "shared_down", shape.shared_padded()),
        },
    ));
    for m in DECODE_BUCKETS {
        for (name, output) in [("head", DType::Bf16), ("head_f32", DType::F32)] {
            items.push(head_rows(instance(
                profile,
                &format!("{tag}_{name}_m{m}"),
                bf16_query(m, shape.vocab_shard(), g.hidden(), output),
                64,
                m.max(8) as u16,
                WHOLE_K_DECODE,
            )));
        }
    }
    for (name, n, k, output, segments) in super::dense::shapes(profile) {
        items.push(instance(
            profile,
            &format!("{tag}_{name}_prefill"),
            bf16_query(PREFILL_ROWS, n, k, output),
            128,
            128,
            DenseKernel::FragmentPrefill { segments },
        ));
    }
    items.push(instance(
        profile,
        &format!("{tag}_head_f32_prefill"),
        bf16_query(PREFILL_ROWS, shape.vocab_shard(), g.hidden(), DType::F32),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: HEAD_F32_SEGMENTS,
        },
    ));
    for ranks in OTHER_RANKS {
        rank_dense_entries(&mut items, owner.model.at(ranks));
    }
    items
}

/// The fragment-family entries of `ranks` ranks, named as TP4's with
/// `qwen38_tp{ranks}_` for `qwen38_`: the head's BF16 and F32 decode
/// buckets, the prefill of every rank-dependent shape, the F32 head,
/// and the shared expert's decode buckets and prefill, with TP4's tiles.
fn rank_dense_entries(items: &mut Vec<GemmVariant>, profile: Profile) {
    let shape = super::geometry::shape(profile).unwrap();
    let g = shape.geometry;
    let tag = profile.tag();
    let vocab = shape.vocab_shard();
    for m in DECODE_BUCKETS {
        for (name, output) in [("head", DType::Bf16), ("head_f32", DType::F32)] {
            items.push(head_rows(instance(
                profile,
                &format!("{tag}_{name}_m{m}"),
                bf16_query(m, vocab, g.hidden(), output),
                64,
                m.max(8) as u16,
                WHOLE_K_DECODE,
            )));
        }
    }
    for (name, n, k, output, segments) in super::dense::shapes(profile) {
        items.push(instance(
            profile,
            &format!("{tag}_{name}_prefill"),
            bf16_query(PREFILL_ROWS, n, k, output),
            128,
            128,
            DenseKernel::FragmentPrefill { segments },
        ));
    }
    items.push(instance(
        profile,
        &format!("{tag}_head_f32_prefill"),
        bf16_query(PREFILL_ROWS, vocab, g.hidden(), DType::F32),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: HEAD_F32_SEGMENTS,
        },
    ));
    let shared = shape.shared_padded();
    for m in DECODE_BUCKETS {
        let tn = m.max(8) as u16;
        items.push(instance(
            profile,
            &format!("{tag}_shared_gate_up_m{m}"),
            shared_gate_up(g, m, shared),
            64,
            tn,
            WHOLE_K_DECODE,
        ));
        items.push(instance(
            profile,
            &format!("{tag}_shared_down_m{m}"),
            bf16_query(m, g.hidden(), shared, DType::Bf16),
            64,
            tn,
            WHOLE_K_DECODE,
        ));
    }
    items.push(instance(
        profile,
        &format!("{tag}_shared_gate_up_prefill"),
        shared_gate_up(g, PREFILL_ROWS, shared),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: super::tuning::prefill(g, "shared_gate_up", g.hidden()),
        },
    ));
    items.push(instance(
        profile,
        &format!("{tag}_shared_down_prefill"),
        bf16_query(PREFILL_ROWS, g.hidden(), shared, DType::Bf16),
        128,
        128,
        DenseKernel::FragmentPrefill {
            segments: super::tuning::prefill(g, "shared_down", shared),
        },
    ));
}

/// The fragment prefill entries with a medium-M stand-in
/// (`kernels_sm120::gemm::mid`): TP4's and each other rank count's.
pub fn mid_references() -> Vec<(Profile, String)> {
    let mut references = Vec::new();
    for profile in super::PROFILES {
        for name in super::dense::shapes(profile)
            .into_iter()
            .map(|(name, ..)| name)
            .chain(["head_f32", "shared_gate_up", "shared_down"])
        {
            references.push((
                profile,
                format!("chop_gemm_{}_{name}_prefill_v1", profile.tag()),
            ));
        }
    }
    references
}

/// The invariance a model-shaped entry declares: skinny and short-K
/// prefill entries are chunk invariant (their prefill schedule fixes each
/// row's K order at every live M), and the cluster decode buckets K3
/// verify steps run (`EXACT_VERIFY_ROWS` rows) are one-row exact. Those
/// buckets share the one-row bucket's reduction geometry: the declared
/// blocks per warp are the same up to four rows, so the strips, cluster
/// split and 8-token tile match, and each row sums its K blocks in the same
/// order whatever rows share the launch.
fn fused_invariance(query: &GemmQuery, kind: Kind) -> Invariance {
    match kind {
        Kind::Skinny { .. } => Invariance::PREFILL,
        Kind::ShortK { .. } if query.m_capacity == PREFILL_ROWS => Invariance::PREFILL,
        Kind::ClusterDecode { .. }
            if query.m_capacity <= crate::sm120::config::EXACT_VERIFY_ROWS =>
        {
            Invariance::one_row(query.m_capacity)
        }
        _ => Invariance::NONE,
    }
}

fn fused(profile: Profile, name: &str, query: GemmQuery, kind: Kind, segments: &[u32]) -> Entry {
    let invariance = fused_invariance(&query, kind);
    let mut entry = Entry::new(profile, name, query, kind, segments, invariance);
    entry.hc = Some(super::geometry::owner(profile).expect("Qwen profile").hc);
    entry
}

/// Cluster decode entries: (name, rows of each weight, K, output). Weights
/// that read the same activation share one launch:
///
/// - `lin_in`: linear attention in_proj_qkv, in_proj_z, in_proj_a and
///   in_proj_b;
/// - `attn_in`: attention q_proj, k_proj, v_proj and the indexer's
///   index_qk_proj;
/// - `hc_in`: a hyper-connection's input_mix_weight_down and
///   block_inject_weight.
///
/// The single-weight entries serve the final mixer (`hc_down`), the PLE
/// value projection and the MTP fc projections (`lin_qkv`), both output
/// projections (`out`), the router, the PLE key projection, and the shared
/// expert on Q-5d's zero-padded views (gate/up with a SwiGLU epilogue).
/// The last column is each entry's `WarpBlocks`.
/// The cluster decode shapes of one rank that change with the rank count,
/// derived from the geometry. `lin_qkv`, `hc_in`, `router`,
/// `hc_down` and `ple_key` are TP4's entries at every rank count.
pub fn cluster_shapes(
    shape: super::geometry::GeometryShape,
    invariant: bool,
) -> Result<Vec<ClusterShape<Vec<u32>>>, String> {
    let g = shape.geometry;
    let h = g.hidden();
    let wide = h * g.streams();
    let heads = shape.linear_value_heads();
    let shared = shape.shared_padded();
    let mut shapes = vec![
        (
            "lin_in",
            vec![shape.linear_qkv(), shape.linear_value(), heads, heads],
            h,
            DType::Bf16,
            false,
            (1, 2),
        ),
        (
            "attn_in",
            vec![
                shape.q_proj_rows(),
                shape.kv_heads() * g.head_dim(),
                shape.kv_heads() * g.head_dim(),
                (g.index_heads() + g.index_kv_heads()) * g.index_dim(),
            ],
            h,
            DType::Bf16,
            false,
            (1, 2),
        ),
    ];
    if invariant {
        shapes.extend([(
            "hc_in",
            vec![g.hc_rank(), g.streams()],
            wide,
            DType::Bf16,
            false,
            (2, 2),
        )]);
    }
    shapes.push((
        "out",
        vec![h],
        shape.linear_value(),
        DType::Bf16,
        false,
        if shape.ranks == 2 { (1, 2) } else { (1, 1) },
    ));
    if invariant {
        shapes.extend([
            ("router", vec![g.experts()], h, DType::F32, false, (1, 2)),
            (
                "hc_down",
                vec![g.hc_rank()],
                wide,
                DType::Bf16,
                false,
                (2, 2),
            ),
            (
                "ple_key",
                vec![wide],
                g.ple_embed_dim(),
                DType::Bf16,
                false,
                (1, 2),
            ),
        ]);
    }
    shapes.extend([
        (
            "shared_gate_up",
            vec![2 * shared],
            h,
            DType::Bf16,
            true,
            (1, 2),
        ),
        ("shared_down", vec![h], shared, DType::Bf16, false, (1, 1)),
    ]);
    for role in [
        super::dense::Role::AttentionOutput,
        super::dense::Role::PleValue,
        super::dense::Role::MtpFc,
    ] {
        let (owner, name) = super::dense::alias_for(shape, role);
        if owner != shape.ranks || shapes.iter().any(|s| s.0 == name) {
            continue;
        }
        let (n, k, output) = role.dimensions(shape);
        shapes.push((name, vec![n], k, output, false, (1, 2)));
    }
    if invariant && let Some(index) = shapes.iter().position(|s| s.0 == "lin_qkv") {
        let entry = shapes.remove(index);
        shapes.insert(3, entry);
    }
    for (name, _, k, _, _, blocks) in &mut shapes {
        *blocks = super::tuning::cluster(g, name, shape.ranks, *k)?;
    }
    Ok(shapes)
}

/// The cluster decode buckets of `shapes` at `profile`.
fn cluster_entries<Rows: AsRef<[u32]>>(
    profile: Profile,
    shapes: &[ClusterShape<Rows>],
) -> Vec<Entry> {
    let mut entries = Vec::new();
    for m in DECODE_BUCKETS {
        for (name, segments, k, output, swiglu, warp_blocks) in shapes {
            let segments = segments.as_ref();
            let mut query = bf16_query(m, segments.iter().sum(), *k, *output);
            if *swiglu {
                query.epilogues.swiglu = Some(SwigluColumns::ConcatenatedHalves);
            }
            entries.push(fused(
                profile,
                &format!("{name}_c{m}"),
                query,
                cluster_kind(*k, m, *warp_blocks),
                segments,
            ));
        }
    }
    entries
}

fn rank_fused_for(profile: Profile) -> Vec<Entry> {
    let shape = super::geometry::shape(profile).unwrap();
    let g = shape.geometry;
    let heads = shape.linear_value_heads();
    let mut entries = cluster_entries(
        profile,
        &cluster_shapes(shape, false).unwrap_or_else(|e| panic!("{e}")),
    );
    entries.push(fused(
        profile,
        "lin_ab",
        bf16_query(PREFILL_ROWS, 2 * heads, g.hidden(), DType::Bf16),
        Kind::Skinny {
            rows: heads,
            weights: 2,
            prefill_splits: super::tuning::skinny(g, "lin_ab", g.hidden()),
        },
        &[heads, heads],
    ));
    entries
}

/// Cluster decode buckets, then the short-K and skinny entries, at TP4 and
/// then at TP2. Skinny entries serve decode
/// too: their split count follows live M.
pub fn fused_catalog() -> Vec<Entry> {
    super::geometry::MODELS.iter().flat_map(fused_for).collect()
}

pub(super) fn fused_for(owner: &super::geometry::GeometryModel) -> Vec<Entry> {
    let profile = owner.model.at(4);
    let shape = super::geometry::shape(profile).unwrap();
    let g = shape.geometry;
    let mut entries = cluster_entries(
        profile,
        &cluster_shapes(shape, true).unwrap_or_else(|e| panic!("{e}")),
    );
    entries.extend([
        fused(
            profile,
            "hc_up_decode",
            bf16_query(
                crate::contract::DECODE_ENTRY_ROWS,
                g.hidden() * g.streams(),
                g.hc_rank(),
                DType::Bf16,
            ),
            // Four warps per CTA spread one token tile's strips over 160 SMs.
            Kind::ShortK {
                tokens: 32,
                warps: 4,
            },
            &[g.hidden() * g.streams()],
        ),
        fused(
            profile,
            "hc_up_prefill",
            bf16_query(
                PREFILL_ROWS,
                g.hidden() * g.streams(),
                g.hc_rank(),
                DType::Bf16,
            ),
            Kind::ShortK {
                tokens: 64,
                warps: 8,
            },
            &[g.hidden() * g.streams()],
        ),
        fused(
            profile,
            "lin_ab",
            bf16_query(
                PREFILL_ROWS,
                2 * shape.linear_value_heads(),
                g.hidden(),
                DType::Bf16,
            ),
            Kind::Skinny {
                rows: shape.linear_value_heads(),
                weights: 2,
                prefill_splits: super::tuning::skinny(g, "lin_ab", g.hidden()),
            },
            &[shape.linear_value_heads(), shape.linear_value_heads()],
        ),
        fused(
            profile,
            "hc_inject",
            bf16_query(
                PREFILL_ROWS,
                g.streams(),
                g.hidden() * g.streams(),
                DType::Bf16,
            ),
            Kind::Skinny {
                rows: g.streams(),
                weights: 1,
                prefill_splits: super::tuning::skinny(g, "hc_inject", g.hidden() * g.streams()),
            },
            &[g.streams()],
        ),
    ]);
    for ranks in OTHER_RANKS {
        entries.extend(rank_fused_for(owner.model.at(ranks)));
    }
    entries
}

/// The declaration of `HC_UP_MIX`, which is not a catalog entry: K3 verify
/// steps launch it at up to `EXACT_VERIFY_ROWS` rows. Its token tile is 32
/// rows at every live count, each row is one MMA column with its own
/// bottleneck, up projection and stream sum, and rows past the live count
/// are zero, so a row has its one-row bits.
pub fn hc_up_mix_declaration() -> Declared {
    Declared {
        symbol: HC_UP_MIX.into(),
        profile: None,
        op: OpKind::Gemm,
        invariance: Invariance::one_row(crate::sm120::config::EXACT_VERIFY_ROWS),
    }
}

/// Resolve the model's fused GEMM headers beside the shared CUDA engines.
pub fn sources_for(profile: Profile, spec: &mut kernel_build::CubinSpec) {
    let g = super::geometry::owner(profile)
        .expect("Qwen profile")
        .geometry;
    spec.defines.extend(super::geometry::hc_defines(g));
    spec.sources_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..");
    spec.include_dirs = vec![
        "crates/kernels-sm120/cuda".into(),
        "models/qwen38/bundle/cuda".into(),
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::Argument;

    use kernels_sm120::gemm::fused::{
        Buffers, HC_PRECOMPUTE_CUBIN, cubin_specs, hc_precompute_spec,
    };

    fn find(symbol: &str) -> Entry {
        fused_catalog()
            .into_iter()
            .find(|entry| entry.symbol == symbol)
            .unwrap()
    }

    #[test]
    fn fused_entries_use_owned_hc_geometry() {
        let entries = fused_catalog();
        assert!(!entries.is_empty());
        for entry in entries {
            assert_eq!(
                entry.hc,
                Some(super::super::geometry::owner(entry.profile).unwrap().hc)
            );
        }
    }

    #[test]
    fn hc_precompute_is_append_only_and_refuses_unqualified_entries() {
        let specs = cubin_specs(super::super::catalogs(), None);
        let candidate = specs
            .iter()
            .find(|s| s.name == HC_PRECOMPUTE_CUBIN)
            .unwrap();
        assert_eq!(candidate.entries.len(), 6);
        for entry in fused_catalog() {
            if let Ok(spec) = hc_precompute_spec(&entry) {
                let old = entry.entry_spec();
                assert_eq!(spec.params[..old.params.len()], old.params);
                assert_eq!(spec.max_threads, old.max_threads);
                assert_eq!(spec.min_blocks, old.min_blocks);
                assert!(
                    !specs
                        .iter()
                        .filter(|s| s.name != HC_PRECOMPUTE_CUBIN)
                        .any(|s| s.entries.iter().any(|e| e.symbol == spec.symbol))
                );
                for capacity in [1, 8, 64] {
                    let mut changed = entry.clone();
                    changed.query.m_capacity = capacity;
                    assert!(hc_precompute_spec(&changed).is_err());
                }
                for profile in crate::sm120::PROFILES[1..].iter().copied() {
                    let mut changed = entry.clone();
                    changed.profile = profile;
                    assert!(hc_precompute_spec(&changed).is_err());
                }
            }
        }
        for profile in crate::sm120::PROFILES[1..].iter().copied() {
            assert!(
                !cubin_specs(super::super::catalogs(), Some(profile))
                    .iter()
                    .any(|s| s.name == HC_PRECOMPUTE_CUBIN)
            );
        }
    }

    #[test]
    fn dense_entries_select_fragment_families() {
        let items = catalog();
        let qwen = items
            .iter()
            .filter(|item| item.profile == crate::sm120::config::PROFILE)
            .filter(|item| item.symbol.starts_with("chop_gemm_qwen38_"))
            .filter(|item| !item.symbol.contains("_shared_"))
            .collect::<Vec<_>>();
        // BF16 and F32 head decode buckets, prefill, F32 prefill head.
        assert_eq!(
            qwen.len(),
            2 * 6 + super::super::dense::shapes(crate::sm120::config::PROFILE).len() + 1
        );
        for item in qwen {
            assert!(
                kernels_sm120::gemm::fragment_weight_enabled(&item.key),
                "{}",
                item.symbol
            );
        }
    }

    #[test]
    fn cluster_decode_entries_split_k_over_their_segments() {
        // (symbol, strips over all segments, grid, cluster)
        for (symbol, total, grid, cluster) in [
            ("chop_gemm_qwen38_lin_in_c1_v1", 160 + 96 + 2, 129 * 5, 5),
            ("chop_gemm_qwen38_lin_in_c8_v1", 258, 65 * 5, 5),
            (
                "chop_gemm_qwen38_attn_in_c1_v1",
                192 + 16 + 16 + 40,
                132 * 5,
                5,
            ),
            ("chop_gemm_qwen38_hc_in_c1_v1", 20 + 1, 21 * 5, 5),
            ("chop_gemm_qwen38_out_c32_v1", 160, 40 * 6, 6),
            ("chop_gemm_qwen38_hc_down_c32_v1", 20, 20 * 5, 5),
        ] {
            let e = find(symbol);
            assert_eq!(e.strips(), total, "{symbol}");
            let plan = e.plan(e.query.m_capacity).unwrap();
            assert_eq!(plan.grid, [grid, 1, 1], "{symbol}");
            assert_eq!(plan.cluster, [cluster, 1, 1], "{symbol}");
        }
        let lin = find("chop_gemm_qwen38_lin_in_c1_v1");
        let mut buffers = Buffers {
            output_strides: [2560, 1536, 24, 24],
            ..Buffers::default()
        };
        assert_eq!(lin.arguments(1, buffers).unwrap().0.len(), 16);
        buffers.output_strides[2] = 8;
        assert!(lin.arguments(1, buffers).is_err());
    }

    #[test]
    fn short_k_items_fill_the_device() {
        let decode = find("chop_gemm_qwen38_hc_up_decode_v1");
        let plan = decode.plan(1).unwrap();
        assert_eq!(plan.grid, [160, 1, 1]);
        assert_eq!(plan.block, [128, 1, 1]);
        assert_eq!(plan.dynamic_shared_bytes, 32 * 656);
        let prefill = find("chop_gemm_qwen38_hc_up_prefill_v1");
        assert_eq!(prefill.plan(8192).unwrap().grid, [188, 1, 1]);
        assert_eq!(prefill.plan(100).unwrap().grid, [2 * 80, 1, 1]);
        assert!(prefill.plan(8193).is_err());
    }

    #[test]
    fn skinny_splits_ignore_live_m() {
        let table = super::super::tuning::dense(crate::geometry::FLASH);
        let ab_splits = table["skinny_splits"]["lin_ab"].as_u64().unwrap() as u32;
        let inject_splits = table["skinny_splits"]["hc_inject"].as_u64().unwrap() as u32;
        let ab = find("chop_gemm_qwen38_lin_ab_v1");
        let one = ab.plan(1).unwrap();
        assert_eq!(one.grid, [10, 1, 1]);
        assert_eq!(one.workspace_bytes, 10 * 2 * 16 * 32 * 4);
        let full = ab.plan(8192).unwrap();
        assert_eq!(full.grid, [256 * 10, 1, 1]);
        assert_eq!(ab.workspace_capacity().unwrap(), full.workspace_bytes);
        let inject = find("chop_gemm_qwen38_hc_inject_v1");
        assert_eq!(inject.plan(1).unwrap().grid, [40, 1, 1]);
        assert_eq!(inject.plan(512).unwrap().grid, [16 * 40, 1, 1]);
        assert_eq!(inject.plan(8192).unwrap().grid, [256 * 40, 1, 1]);
        // Prefill phases take the entry's own count, also at every live M.
        for (entry, splits) in [(&ab, ab_splits), (&inject, inject_splits)] {
            for m in [1, 33, 512, 2048, 8192] {
                let plan = entry.prefill_plan(m).unwrap();
                assert_eq!(plan.grid, [m.div_ceil(32) * splits, 1, 1]);
                assert!(plan.workspace_bytes <= entry.workspace_capacity().unwrap());
            }
        }
        let buffers = Buffers {
            weights: [1, 2, 0, 0],
            activation: 3,
            outputs: [4, 0, 0, 0],
            output_strides: [24, 0, 0, 0],
            partial: 5,
            semaphores: 6,
        };
        assert_eq!(ab.arguments(5, buffers).unwrap().0.len(), 10);
        assert_eq!(
            ab.prefill_arguments(5, buffers).unwrap().0[8],
            Argument::U32(ab_splits)
        );
        let short = Buffers {
            output_strides: [12, 0, 0, 0],
            ..buffers
        };
        assert!(ab.arguments(5, short).is_err());
    }

    #[test]
    fn derived_tp4_shapes_equal_the_former_literals() {
        assert_eq!(SHARED_INTERMEDIATE, 256);
        assert_eq!((VOCAB_SHARD, VALUE_HEADS), (62080, 12));
    }

    /// As for MiMo: Qwen's rank-dependent shapes at four ranks are TP4's.
    #[test]
    fn qwen_rank_shapes_at_tp4_match_tp4_entries() {
        let items = catalog();
        for (name, n, k, ..) in super::super::dense::shapes(crate::sm120::config::PROFILE) {
            let symbol = format!("chop_gemm_qwen38_{name}_prefill_v1");
            let item = items
                .iter()
                .find(|item| item.symbol == symbol)
                .unwrap_or_else(|| panic!("{symbol}"));
            let StaticQuery::Gemm(query) = item.key.query else {
                panic!("{symbol} is a GEMM");
            };
            assert_eq!((query.n, query.k), (n, k), "{symbol}");
            assert_eq!(item.profile, crate::sm120::config::PROFILE);
        }
    }

    /// Every fragment prefill entry carries its declared K segments.
    #[test]
    fn prefill_entries_carry_their_declared_segments() {
        let segments = |symbol: &str| {
            let item = catalog()
                .into_iter()
                .find(|item| item.symbol == symbol)
                .unwrap_or_else(|| panic!("{symbol}"));
            kernels_sm120::gemm::prefill_segments(&item.key).unwrap()
        };
        let table = super::super::tuning::dense(crate::geometry::FLASH);
        for name in ["out", "hc_down", "head"] {
            assert_eq!(
                segments(&format!("chop_gemm_qwen38_{name}_prefill_v1")),
                table["prefill_segments"][name].as_u64().unwrap() as u8
            );
        }
        for tag in ["qwen38", "qwen38_tp2"] {
            assert_eq!(segments(&format!("chop_gemm_{tag}_head_f32_prefill_v1")), 1);
        }
    }
}
