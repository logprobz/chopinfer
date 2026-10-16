//! The rank-image views the program reads, with the dtype, layout and size
//! each kernel expects. Names follow the checkpoint (`model-qwen38`
//! placement); shapes are per rank.

pub use bundle_runtime::index::{RankIndex, View};
use kernel_core::transform::fragment::{FragmentWeightFormat, FragmentWeightLayout};

use crate::contract::*;

/// The schema every Qwen3.8 rank image index is written under.
pub const SCHEMA: &str = "chopinfer.qwen38.rank-image.v1";

pub const BF16_FRAGMENT: &str = "bf16_mma_fragments_16x16_v1";
pub const FP8_FRAGMENT: &str = "fp8_e4m3_mma_fragments_16x32_v1";
pub const BF16_ROWS: &str = "bf16_row_major_v1";
pub const F32_ROWS: &str = "f32_row_major_v1";
pub const I64_ROWS: &str = "i64_row_major_v1";
pub const F32_BLOCK_SCALES: &str = "f32_block_scale_128x128_v1";
pub const NGRAM: &str = "fp8_e4m3_ngram_head_rows_160_v1";

pub const MODEL: &str = "model.language_model";

pub fn parse(bytes: &[u8], rank: u32) -> Result<RankIndex, String> {
    bundle_runtime::index::parse(bytes, rank, SCHEMA)
}

/// One view the program binds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Required {
    pub name: String,
    pub dtype: &'static str,
    pub layout_id: &'static str,
    pub bytes: u64,
}

fn fragment(rows: u32, k: u32, format: FragmentWeightFormat) -> u64 {
    FragmentWeightLayout {
        rows: rows as usize,
        k: k as usize,
        format,
    }
    .packed_bytes()
    .expect("catalog fragment shapes are aligned") as u64
}

fn bf16_weight(name: String, rows: u32, k: u32) -> Required {
    Required {
        name,
        dtype: "BF16",
        layout_id: BF16_FRAGMENT,
        bytes: fragment(rows, k, FragmentWeightFormat::Bf16),
    }
}

fn rows(name: String, dtype: &'static str, elements: u64) -> Required {
    let (layout_id, element) = match dtype {
        "F32" => (F32_ROWS, 4),
        "I64" => (I64_ROWS, 8),
        _ => (BF16_ROWS, 2),
    };
    Required {
        name,
        dtype,
        layout_id,
        bytes: elements * element,
    }
}

fn hyper_connection(result: &mut Vec<Required>, stem: &str, inject: bool, shape: Shape) {
    let hc_rank = shape.hc_rank();
    let streams = shape.streams();
    let wide = shape.wide();
    result.push(rows(format!("{stem}.hc_norm.weight"), "BF16", wide.into()));
    result.push(bf16_weight(
        format!("{stem}.input_mix_weight_down.weight"),
        hc_rank,
        wide,
    ));
    result.push(bf16_weight(
        format!("{stem}.input_mix_weight_up.weight"),
        wide,
        hc_rank,
    ));
    if inject {
        result.push(bf16_weight(
            format!("{stem}.block_inject_weight.weight"),
            streams,
            wide,
        ));
    }
}

/// Rows of rank `rank`'s share of the n-gram table: its hash heads (four
/// at TP4), and on the last rank the checkpoint's alignment padding.
pub fn ngram_rows(rank: u32, shape: Shape) -> Result<(u64, u64), &'static str> {
    let tables =
        crate::contract::ngram_tables_for(shape.geometry).map_err(|_| "n-gram table geometry")?;
    if !shape.ranks.contains(rank) {
        return Err("rank outside the topology");
    }
    let first = (rank * shape.ngram_local_heads) as usize;
    let start = tables.offsets[first];
    let end = if rank + 1 == shape.ranks() {
        tables.rows
    } else {
        tables.offsets[first + shape.ngram_local_heads as usize]
    };
    Ok((start, end - start))
}

/// The index name of the rank's n-gram table.
pub fn ngram_view_flash() -> String {
    ngram_view_for(Shape::flash(crate::contract::Ranks::TP4))
}

pub fn ngram_view_for(shape: Shape) -> String {
    let ple_layer = shape.ple_layer();
    format!("{MODEL}.layers.{ple_layer}.ple.ple_embedding.ngram_embedding.weight")
}

/// The views of one decoder layer under `stem`.
fn layer_views(result: &mut Vec<Required>, layer: u32, stem: &str, shape: Shape) {
    let attn_head = shape.attn_head();
    let conv_taps = shape.conv_taps();
    let experts = shape.experts();
    let expert_intermediate = shape.expert_intermediate();
    let hidden = shape.hidden();
    let index_dim = shape.index_dim();
    let index_row = shape.index_row();
    let kv_heads = shape.kv_heads();
    let linear_head = shape.linear_head();
    hyper_connection(
        result,
        &format!("{stem}.attn_hyper_connection"),
        true,
        shape,
    );
    hyper_connection(result, &format!("{stem}.mlp_hyper_connection"), true, shape);
    if shape.is_full(layer) {
        let attn = format!("{stem}.self_attn");
        for (name, rows) in [
            ("q_proj", shape.q_heads * 2 * attn_head),
            ("k_proj", kv_heads * attn_head),
            ("v_proj", kv_heads * attn_head),
            ("indexer.index_qk_proj", index_row),
        ] {
            result.push(bf16_weight(format!("{attn}.{name}.weight"), rows, hidden));
        }
        result.push(bf16_weight(
            format!("{attn}.o_proj.weight"),
            hidden,
            shape.attn_value,
        ));
        for (name, width) in [
            ("q_norm", attn_head),
            ("k_norm", attn_head),
            ("indexer.q_layernorm", index_dim),
            ("indexer.k_layernorm", index_dim),
        ] {
            result.push(rows(format!("{attn}.{name}.weight"), "BF16", width.into()));
        }
    } else {
        let linear = format!("{stem}.linear_attn");
        for (name, rows) in [
            ("in_proj_qkv", shape.linear_qkv),
            ("in_proj_z", shape.linear_value),
            ("in_proj_a", shape.value_heads),
            ("in_proj_b", shape.value_heads),
        ] {
            result.push(bf16_weight(format!("{linear}.{name}.weight"), rows, hidden));
        }
        result.push(bf16_weight(
            format!("{linear}.out_proj.weight"),
            hidden,
            shape.linear_value,
        ));
        result.push(rows(
            format!("{linear}.conv1d.weight"),
            "BF16",
            u64::from(shape.linear_qkv * conv_taps),
        ));
        result.push(rows(
            format!("{linear}.norm.weight"),
            "BF16",
            linear_head.into(),
        ));
        for name in ["A_log", "dt_bias"] {
            result.push(rows(
                format!("{linear}.{name}"),
                "F32",
                shape.value_heads.into(),
            ));
        }
    }
    let mlp = format!("{stem}.mlp");
    result.push(bf16_weight(format!("{mlp}.gate.weight"), experts, hidden));
    result.push(bf16_weight(
        format!("{mlp}.shared_expert.gate_up_proj.weight"),
        2 * shape.shared_intermediate,
        hidden,
    ));
    result.push(bf16_weight(
        format!("{mlp}.shared_expert.down_proj.weight"),
        hidden,
        shape.shared_intermediate,
    ));
    result.push(rows(
        format!("{mlp}.shared_expert_gate.weight"),
        "BF16",
        hidden.into(),
    ));
    for (name, n, k) in [
        ("gate_up_proj", 2 * expert_intermediate, hidden),
        ("down_proj", hidden, expert_intermediate),
    ] {
        result.push(Required {
            name: format!("{mlp}.experts.{name}.weight"),
            dtype: "F8_E4M3",
            layout_id: FP8_FRAGMENT,
            bytes: fragment(shape.local_experts * n, k, FragmentWeightFormat::Fp8E4M3),
        });
        result.push(Required {
            name: format!("{mlp}.experts.{name}.weight_scale_inv"),
            dtype: "F32",
            layout_id: F32_BLOCK_SCALES,
            bytes: u64::from(shape.local_experts * n / 128 * (k / 128)) * 4,
        });
    }
}

/// Every view the MTP drafter reads on each rank: its input projections and
/// norms, its layer (placed as a main full layer) and its mixer. The
/// embedding and head are the target's.
pub fn drafter_required(shape: Shape) -> Vec<Required> {
    if shape.mtp_layers() == 0 {
        return Vec::new();
    }
    let hidden = shape.hidden();
    let mtp_layer = shape.mtp_layer();
    let wide = shape.wide();
    let mut result = Vec::new();
    for name in ["fc_hidden", "fc_embedding"] {
        result.push(bf16_weight(format!("mtp.{name}.weight"), hidden, hidden));
    }
    result.push(rows(
        "mtp.pre_fc_norm_hidden.weight".into(),
        "BF16",
        wide.into(),
    ));
    result.push(rows(
        "mtp.pre_fc_norm_embedding.weight".into(),
        "BF16",
        hidden.into(),
    ));
    layer_views(
        &mut result,
        mtp_layer,
        &layer_stem_for(mtp_layer, shape),
        shape,
    );
    hyper_connection(&mut result, "mtp.hyper_connection_mixer", false, shape);
    result
}

/// Total bytes of the complete text image, including its MTP views and padding.
pub fn image_bytes(rank: u32, shape: Shape) -> u64 {
    required(rank, shape)
        .into_iter()
        .chain(drafter_required(shape))
        .map(|view| view.bytes.div_ceil(4096) * 4096)
        .sum()
}

/// Every view the text program reads on `rank` of `shape`'s topology.
pub fn required(rank: u32, shape: Shape) -> Vec<Required> {
    let hidden = shape.hidden();
    let layers = shape.layers();
    let ngram_heads = shape.ngram_heads();
    let ngram_head_width = shape.ngram_head_width();
    let ple_conv_taps = shape.ple_conv_taps();
    let ple_embed_dim = shape.ple_embed_dim();
    let ple_layer = shape.ple_layer();
    let vocab = shape.vocab();
    let wide = shape.wide();
    let mut result = vec![
        rows(
            format!("{MODEL}.embed_tokens.weight"),
            "BF16",
            u64::from(vocab) * u64::from(hidden),
        ),
        bf16_weight("lm_head.weight".into(), shape.vocab_per_rank, hidden),
    ];
    hyper_connection(
        &mut result,
        &format!("{MODEL}.hyper_connection_mixer"),
        false,
        shape,
    );
    for layer in 0..layers {
        layer_views(&mut result, layer, &layer_stem_for(layer, shape), shape);
    }
    let ple = format!("{MODEL}.layers.{ple_layer}.ple");
    result.push(bf16_weight(
        format!("{ple}.key_proj.weight"),
        wide,
        ple_embed_dim,
    ));
    result.push(bf16_weight(
        format!("{ple}.value_proj.weight"),
        hidden,
        ple_embed_dim,
    ));
    for name in ["norm_key", "norm_query", "norm_conv"] {
        result.push(rows(format!("{ple}.{name}.weight"), "BF16", wide.into()));
    }
    result.push(rows(
        format!("{ple}.conv1d.weight"),
        "BF16",
        u64::from(wide * ple_conv_taps),
    ));
    for (name, count) in [
        (
            "layer_multipliers",
            u64::from(shape.geometry.ngram_orders()),
        ),
        ("ngram_heads_offsets", u64::from(ngram_heads)),
        ("ngram_heads_vocab_sizes", u64::from(ngram_heads)),
    ] {
        result.push(rows(format!("{ple}.ple_embedding.{name}"), "I64", count));
    }
    result.push(rows(
        format!("{ple}.ple_embedding.ngram_embedding.weight_scale"),
        "BF16",
        1,
    ));
    let (_, table_rows) = ngram_rows(rank, shape).expect("fixed n-gram geometry");
    result.push(Required {
        name: ngram_view_for(shape),
        dtype: "F8_E4M3",
        layout_id: ngram_layout(shape),
        bytes: table_rows * u64::from(ngram_head_width),
    });
    result
}

/// Checks that `index` holds every required view with its expected dtype,
/// layout and size, and that each full layer's q, k and v projections lie
/// back to back: one `[ATTN_ROW, HIDDEN]` fragment view from q's offset, so
/// one GEMM writes the packed row. The error names the first view that
/// differs.
pub fn validate(index: &RankIndex, shape: Shape) -> Result<(), String> {
    let layers = shape.layers();
    if !shape.ranks.contains(index.rank) {
        return Err(format!(
            "rank {} is outside the {}-rank topology",
            index.rank,
            shape.ranks()
        ));
    }
    check(
        index,
        (0..layers).filter(|&layer| shape.is_full(layer)),
        required(index.rank, shape),
        shape,
    )
}

/// Checks the MTP drafter's views as `validate` checks the target's.
pub fn validate_drafter(index: &RankIndex, shape: Shape) -> Result<(), String> {
    let mtp_layer = shape.mtp_layer();
    check(index, [mtp_layer], drafter_required(shape), shape)
}

pub fn check(
    index: &RankIndex,
    full_layers: impl IntoIterator<Item = u32>,
    required: Vec<Required>,
    shape: Shape,
) -> Result<(), String> {
    for layer in full_layers {
        let attn = format!("{}.self_attn", layer_stem_for(layer, shape));
        let [q, k, v] = ["q_proj", "k_proj", "v_proj"].map(|name| {
            index
                .views
                .get(&format!("{attn}.{name}.weight"))
                .map(|view| (view.offset, view.bytes))
        });
        if let (Some(q), Some(k), Some(v)) = (q, k, v)
            && (q.0 + q.1 != k.0 || k.0 + k.1 != v.0)
        {
            return Err(format!(
                "rank {} layer {layer} q, k and v projections are not contiguous",
                index.rank
            ));
        }
    }
    for required in required {
        let Some(view) = index.views.get(&required.name) else {
            return Err(format!("rank {} lacks view {}", index.rank, required.name));
        };
        if view.dtype != required.dtype
            || view.layout_id != required.layout_id
            || view.bytes != required.bytes
        {
            return Err(format!(
                "rank {} view {} is {} {} of {} bytes; the program reads {} {} of {} bytes",
                index.rank,
                required.name,
                view.dtype,
                view.layout_id,
                view.bytes,
                required.dtype,
                required.layout_id,
                required.bytes
            ));
        }
    }
    Ok(())
}

pub fn layer_stem_for(layer: u32, shape: Shape) -> String {
    if layer >= shape.layers() {
        format!("mtp.layers.{}", layer - shape.layers())
    } else {
        format!("{MODEL}.layers.{layer}")
    }
}

/// The rank table layout for this geometry's head width.
pub fn ngram_layout(shape: Shape) -> &'static str {
    static LAYOUTS: std::sync::OnceLock<Vec<(u32, String)>> = std::sync::OnceLock::new();
    let layouts = LAYOUTS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| {
                (
                    g.ngram_head_width(),
                    format!("fp8_e4m3_ngram_head_rows_{}_v1", g.ngram_head_width()),
                )
            })
            .collect()
    });
    &layouts
        .iter()
        .find(|(width, _)| *width == shape.ngram_head_width())
        .expect("registered geometry")
        .1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_requires_packed_qkv_projections() {
        let mut offset = 0;
        let views = required(0, Shape::TP4)
            .into_iter()
            .map(|view| {
                let entry = (
                    view.name.clone(),
                    View {
                        offset,
                        bytes: view.bytes,
                        dtype: view.dtype.into(),
                        shape: vec![view.bytes],
                        layout_id: view.layout_id.into(),
                    },
                );
                offset += view.bytes.next_multiple_of(4096);
                entry
            })
            .collect();
        let mut index = RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: offset,
            sha256: Some("0".repeat(64)),
            direct_identity: None,
            views,
        };
        validate(&index, Shape::TP4).unwrap();
        let v = format!("{MODEL}.layers.7.self_attn.v_proj.weight");
        index.views.get_mut(&v).unwrap().offset += 4096;
        assert!(
            validate(&index, Shape::TP4)
                .unwrap_err()
                .contains("layer 7")
        );
    }

    #[test]
    fn required_views_cover_every_layer_once() {
        let views = required(0, Shape::TP4);
        let names = views
            .iter()
            .map(|view| view.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(names.len(), views.len());
        let count = |needle: &str| views.iter().filter(|v| v.name.contains(needle)).count();
        assert_eq!(count(".self_attn.q_proj."), 12);
        assert_eq!(count(".linear_attn.in_proj_qkv."), 36);
        assert_eq!(count(".experts.gate_up_proj.weight_scale_inv"), 48);
        assert_eq!(count("hc_norm"), 97);
        assert_eq!(count("mtp."), 0);
        // The drafter's layer reads a main full layer's views under its
        // own names.
        let drafter = drafter_required(Shape::TP4);
        let layer = |stem: &str| {
            views
                .iter()
                .chain(&drafter)
                .filter(|view| view.name.starts_with(stem))
                .map(|view| (view.name[stem.len()..].to_owned(), view.bytes))
                .collect::<Vec<_>>()
        };
        assert_eq!(layer("mtp.layers.0."), layer(&format!("{MODEL}.layers.3.")));
        assert_eq!(drafter.len(), layer("mtp.layers.0.").len() + 7);
        // Fragment views pad rows to 16: inject is 4 rows, stored as 16.
        let inject = views
            .iter()
            .find(|view| {
                view.name
                    .ends_with("0.attn_hyper_connection.block_inject_weight.weight")
            })
            .unwrap();
        assert_eq!(inject.bytes, 16 * 10240 * 2);
    }

    #[test]
    fn ngram_shares_cover_the_padded_table() {
        for shape in Shape::ALL {
            let shares = (0..shape.ranks())
                .map(|rank| ngram_rows(rank, shape).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(shares[0].0, 0);
            for pair in shares.windows(2) {
                assert_eq!(pair[0].0 + pair[0].1, pair[1].0);
            }
            // The checkpoint's 128 shards of 2,500,012 rows.
            let last = shares.last().unwrap();
            assert_eq!(last.0 + last.1, 128 * 2_500_012);
            assert!(ngram_rows(shape.ranks(), shape).is_err());
        }
        assert_eq!(
            ngram_rows(1, Shape::TP2).unwrap().0,
            ngram_rows(2, Shape::TP4).unwrap().0
        );
    }
}
