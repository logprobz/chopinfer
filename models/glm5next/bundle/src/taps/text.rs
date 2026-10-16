//! The text program's taps (`models/glm5next/taps.json`): which scratch
//! buffer each tap reads after which stage. Every tap reads a buffer the
//! program already writes (no kernel of its own), and every one is
//! replicated, so rank 0's copy is written: the residual streams, the
//! mixers' collapsed rows, the sublayer inputs after their norms, the
//! all-reduced sublayer outputs (each the BF16 sum the expansion reads,
//! rounded once), the normalized latents, the indexer's key lists and the
//! router's F32 logits and routes, and the head's collapse and final norm.
//! The Sinkhorn post and comb taps are not written: the engine holds them
//! in BF16, the reference in F32.

use model_glm5next_contract::contract::Shape;
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapSharding, TapSpec};

use crate::buffers::Buffer;
use crate::lower::streams;
use crate::program::{Phase, Stage, Sublayer};

/// One tap and the buffer it reads.
#[derive(Clone, Debug)]
pub struct Tap {
    pub spec: TapSpec,
    pub buffer: Buffer,
}

fn tap(name: String, buffer: Buffer, cols: u32, dtype: TapDType, compare: TapCompare) -> Tap {
    Tap {
        spec: TapSpec {
            name,
            layer: None,
            detail: false,
            cols,
            dtype,
            sharding: TapSharding::Replicated,
            compare,
        },
        buffer,
    }
}

fn bf16(name: String, buffer: Buffer, cols: u32) -> Tap {
    tap(name, buffer, cols, TapDType::Bf16, TapCompare::Dense)
}

fn scope(layer: u32) -> String {
    format!("L{layer:02}")
}

const fn sublayer_name(sublayer: Sublayer) -> &'static str {
    match sublayer {
        Sublayer::Attention => "attn",
        Sublayer::Mlp => "mlp",
    }
}

/// The taps to copy after `stage`, before the environment's selection.
pub fn taps(stage: Stage, shape: Shape) -> Vec<Tap> {
    let p = shape.profile;
    let mut taps = Vec::new();
    match stage {
        Stage::ExpandStreams => taps.push(bf16(
            "embed.out".into(),
            streams(0, Sublayer::Attention).0,
            p.hc_width,
        )),
        Stage::HcMix(layer, sublayer) => {
            let name = scope(layer);
            let input = streams(layer, sublayer).0;
            match sublayer {
                Sublayer::Attention => {
                    taps.push(bf16(format!("{name}.in.streams"), input, p.hc_width));
                }
                Sublayer::Mlp => {
                    let mut streams = bf16(format!("{name}.x.hc.mlp.streams"), input, p.hc_width);
                    streams.spec.detail = true;
                    taps.push(streams);
                }
            }
            let sub = sublayer_name(sublayer);
            taps.push(bf16(
                format!("{name}.x.hc.{sub}"),
                Buffer::Collapsed,
                p.hidden,
            ));
        }
        Stage::InputNorm(layer, sublayer) => taps.push(bf16(
            format!("{}.{}.in", scope(layer), sublayer_name(sublayer)),
            Buffer::Normed,
            p.hidden,
        )),
        Stage::QueryNorm(layer) => {
            let mut query = bf16(
                format!("{}.x.q_latent", scope(layer)),
                Buffer::QueryNormed,
                p.q_rank,
            );
            query.spec.detail = true;
            taps.push(query);
        }
        Stage::LatentNorm(layer) => taps.push(bf16(
            format!("{}.x.latent", scope(layer)),
            Buffer::Latent,
            p.kv_rank,
        )),
        // The key lists' I32 positions, -1 past the selection, read as
        // U32 (2^32 - 1 where none).
        Stage::IndexSelect(layer) => taps.push(tap(
            format!("{}.x.indexer.ids", scope(layer)),
            Buffer::KeyLists,
            shape.latent_keys(),
            TapDType::U32,
            TapCompare::Set,
        )),
        // The route reads the compacted F32 logits and writes the global
        // expert ids and their normalized, scaled weights.
        Stage::Route(layer) => {
            let name = format!("{}.router", scope(layer));
            taps.push(tap(
                format!("{name}.logits"),
                Buffer::RouterLogits,
                shape.geometry.shape.moe.experts,
                TapDType::F32,
                TapCompare::Dense,
            ));
            taps.push(tap(
                format!("{name}.weights"),
                Buffer::RouteWeights,
                p.top_k,
                TapDType::F32,
                TapCompare::Paired,
            ));
            taps.push(tap(
                format!("{name}.ids"),
                Buffer::RouteIds,
                p.top_k,
                TapDType::U32,
                TapCompare::Set,
            ));
        }
        Stage::Reduce(layer, sublayer) => taps.push(bf16(
            format!("{}.{}.out", scope(layer), sublayer_name(sublayer)),
            Buffer::Reduced,
            p.hidden,
        )),
        Stage::HcExpand(layer, Sublayer::Mlp) => taps.push(bf16(
            format!("{}.out", scope(layer)),
            streams(layer, Sublayer::Mlp).1,
            p.hc_width,
        )),
        Stage::HeadCollapse => {
            taps.push(bf16("final.x.hc.head".into(), Buffer::Collapsed, p.hidden))
        }
        Stage::FinalNorm => taps.push(bf16("final.norm".into(), Buffer::Normed, p.hidden)),
        _ => {}
    }
    for tap in &mut taps {
        tap.spec.layer = stage.layer();
    }
    taps
}

/// Every text tap, once per name, in program order.
pub fn specs(shape: Shape) -> Vec<TapSpec> {
    crate::program::stages(shape, Phase::Prefill { rows: 1 })
        .into_iter()
        .flat_map(|stage| taps(stage, shape))
        .map(|tap| tap.spec)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The declaration declares every text tap as the program writes it:
    /// 45 layers' streams, sublayer inputs and outputs, the 11 latent
    /// layers' latents and key lists, the 42 MoE layers' routes, and the
    /// head's taps.
    #[test]
    fn taps_json_declares_every_text_tap() {
        let declaration =
            runtime_bundle_sdk::taps::Declaration::parse(include_str!("../../../taps.json"))
                .unwrap();
        let specs = specs(Shape::TP4);
        let mut names = specs.iter().map(|spec| &spec.name).collect::<Vec<_>>();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), specs.len(), "a tap is written twice");
        assert_eq!(specs.len(), 1 + 45 * 9 + 11 * 3 + 42 * 3 + 2);
        for spec in &specs {
            declaration.check(spec, 4).unwrap();
        }
    }

    /// The key lists' columns are the reference's: 2048 pooled positions
    /// and at most three tail positions.
    #[test]
    fn key_lists_hold_the_reference_columns() {
        let taps = taps(Stage::IndexSelect(3), Shape::TP4);
        assert_eq!(taps.len(), 1);
        assert_eq!(taps[0].spec.cols, 2051);
    }
}
