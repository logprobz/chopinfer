//! Qwen3.8 Flash's row entries on the shared engines (`kernels_sm120::row`):
//! its hyper-connection, n-gram and PLE rows (`row::hyper_connection`), and
//! its embedding and logits rows of the shared row engine.

use kernel_core::contract::RecurrentPhase;
use kernels_sm120::Profile;
use kernels_sm120::row::hyper_connection::{
    Entry, Ngram, Streams, hc_entries, ngram_decode_entry, ngram_entry, ple_conv_entry,
    ple_conv_items_entry, ple_gate_entry,
};
use kernels_sm120::row::{self, RowInstantiation};

/// Each hyper-connection stream's width.
pub const WIDTH: u32 = crate::contract::HIDDEN;
/// Hyper-connection streams.
pub const STREAMS: u32 = crate::contract::STREAMS;
/// The bottleneck's rank.
pub const RANK: u32 = crate::contract::HC_RANK;
/// Rows of one launch: a prefill chunk.
pub const MAX_ROWS: u32 = crate::contract::PREFILL_TOKENS;

/// The hyper-connection streams of every row entry.
pub const HC: Streams = Streams {
    streams: STREAMS,
    width: WIDTH,
    rank: RANK,
    eps_bits: crate::contract::EPS.to_bits(),
    capacity: MAX_ROWS,
};

/// The n-gram table's geometry: maximum order three, eight 160-wide heads per order.
pub const NGRAM: Ngram = Ngram {
    orders: crate::contract::NGRAM_SIZE,
    heads_per_order: crate::contract::NGRAM_HEADS_PER_ORDER,
    head_width: crate::contract::NGRAM_HEAD_WIDTH,
    capacity: MAX_ROWS,
};

/// The per-sequence n-gram rows of a Qwen3.8 profile: the rank's local hash
/// heads, derived from the rank shape.
pub fn ngram(profile: Profile) -> Entry {
    let shape = super::geometry::shape(profile).expect("Qwen profile");
    let g = shape.geometry;
    let ngram = Ngram {
        orders: g.ngram_orders(),
        heads_per_order: g.ngram_heads_per_order(),
        head_width: g.ngram_head_width(),
        capacity: MAX_ROWS,
    };
    ngram_entry(profile, ngram, shape.ngram_heads(), g.ngram_heads() / 4)
        .expect("supported row selection")
}

/// The hyper-connection, n-gram and PLE rows: TP4's (every row the bundle
/// runs at each rank count), then the n-gram rows of TP2, whose
/// local hash heads differ.
pub fn hyper_connection_rows() -> Vec<Entry> {
    super::geometry::MODELS.iter().flat_map(hc_rows).collect()
}

fn hc_rows(model: &super::geometry::GeometryModel) -> Vec<Entry> {
    let g = model.geometry;
    let hc = Streams {
        streams: g.streams(),
        width: g.hidden(),
        rank: g.hc_rank(),
        eps_bits: g.eps().to_bits(),
        capacity: MAX_ROWS,
    };
    let profile = model.model.at(4);
    let mut out = hc_entries(profile, hc).expect("supported row selection");
    let ngram_tp4 = ngram(profile);
    let conv =
        ple_conv_entry(profile, hc, RecurrentPhase::Decode).expect("supported row selection");
    out.extend([
        ngram_tp4.clone(),
        ple_gate_entry(profile, hc).expect("supported row selection"),
        conv.clone(),
        ple_conv_entry(profile, hc, RecurrentPhase::Prefill).expect("supported row selection"),
        // Every decode bucket item in one launch, with the per-sequence
        // entries' queries.
        ngram_decode_entry(&ngram_tp4, g.ngram_heads() / 4).expect("supported row selection"),
        ple_conv_items_entry(&conv).expect("supported row selection"),
    ]);
    for profile in model.model.other_profiles() {
        let ngram = ngram(profile);
        let decode =
            ngram_decode_entry(&ngram, g.ngram_heads() / 4).expect("supported row selection");
        out.extend([ngram, decode]);
    }
    out
}

/// The per-stream grouped norm over F32 weights of each geometry
/// (`GeometryModel::f32_norm_row`), in a cubin of its own that only an
/// image with F32 norm weights loads. The BF16 norm stays in
/// [`hyper_connection_rows`].
pub fn f32_norm_cubin_specs(profile: Option<Profile>) -> Vec<kernel_build::CubinSpec> {
    super::geometry::MODELS
        .iter()
        .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
        .map(|owner| {
            let names = owner.f32_norms;
            owner
                .f32_norm_row()
                .and_then(|row| row.cubin_spec(names.row_cubin, names.row_symbol))
                .expect("F32 grouped norm row")
        })
        .collect()
}

/// The embedding and logits rows of the shared row engine: the replicated
/// 248,320 x 2560 table, and F32 log-probabilities over the full vocabulary
/// or one rank's 62,080-entry shard. The embedding is row-local, which
/// served prefill rows rely on.
pub fn rows() -> Vec<RowInstantiation> {
    super::geometry::MODELS
        .iter()
        .flat_map(model_rows)
        .collect()
}

fn model_rows(model: &super::geometry::GeometryModel) -> Vec<RowInstantiation> {
    let profile = model.model.at(4);
    let vocab = model.geometry.vocab();
    let shard = super::geometry::shape(profile).unwrap().vocab_shard();
    let mut embedding = row::embedding(
        profile,
        &format!("embedding_{}", profile.tag()),
        vocab,
        model.geometry.hidden(),
        false,
    );
    embedding.invariance = row::row_local();
    let mut rows = vec![embedding];
    let tag = profile.tag();
    for (name, vocab, k) in [
        (format!("argmax_{tag}_full"), vocab, 1),
        (format!("top20_{tag}_full"), vocab, 20),
        (format!("argmax_{tag}_shard"), shard, 1),
        (format!("top20_{tag}_shard"), shard, 20),
    ] {
        rows.push(row::logits(profile, &name, vocab, k));
    }
    rows
}

/// Bind the owned row headers and their geometry beside the shared engines.
pub fn sources_for(profile: Profile, spec: &mut kernel_build::CubinSpec) {
    let g = super::geometry::owner(profile)
        .expect("Qwen profile")
        .geometry;
    super::gemm::sources_for(profile, spec);
    spec.defines.extend(super::geometry::ple_defines(g));
    spec.defines.extend(super::geometry::ngram_defines(g));
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use kernel_core::contract::{RecurrentQuery, StaticQuery};

    #[test]
    fn ple_entries_match_contract_history_geometry() {
        let mut count = 0;
        for entry in hyper_connection_rows() {
            if let StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                kernel,
                dilation,
                channels,
                ..
            }) = entry.query
            {
                count += 1;
                let g = super::super::geometry::owner(entry.profile)
                    .unwrap()
                    .geometry;
                assert_eq!(kernel, g.ple_conv_taps());
                assert_eq!(dilation, g.ple_conv_dilation());
                assert_eq!(channels, g.hidden() * g.streams());
            }
        }
        assert_eq!(count, 3 * super::super::geometry::MODELS.len());
    }
}
