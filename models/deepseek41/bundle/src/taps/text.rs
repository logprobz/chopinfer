//! The program's taps (`models/deepseek41/taps.json`): which scratch buffer
//! each tap reads after which stage. Every tap reads a buffer the program
//! already writes (no kernel of its own), and every one is replicated, so
//! rank 0's copy is written: the residual streams entering and leaving each
//! layer, each mixer's `pre`, `post` and `comb` and collapsed row, the
//! sublayer inputs after their norms, the window latent after `kv_norm`,
//! the all-reduced sublayer outputs, the routes, and the head's F32
//! collapse and final norm (the rows the LM head reads). The names and values follow the reference's `taps`
//! (`models/deepseek41/reference.py`). The declaration leaves the columns
//! to the geometry.

use runtime_bundle_abi::DeviceBufferSpanV1;
use runtime_bundle_sdk::taps::{TapCompare, TapDType, TapSharding, TapSpec};

use crate::buffers::Scratch;
use crate::lower::{pre, streams, sublayer_index};
use crate::program::{Phase, Stage, Sublayer, stages};
use crate::shape::Shape;

/// The scratch buffer a tap reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Buffer {
    Embedded,
    /// The streams sublayer `index` reads.
    Streams(u32),
    /// The `pre` sublayer `index`'s mixer writes.
    Pre(u32),
    Post,
    Comb,
    Collapsed,
    Normed,
    /// The last streams' F32 collapse, the final norm's input.
    HeadCollapsed,
    /// F32 normed rows: the final norm's output, the LM head's input.
    NormedWide,
    WindowLatent,
    Reduced,
    ExpertWeights,
    ExpertIds,
}

impl Buffer {
    pub fn span(self, scratch: &Scratch<DeviceBufferSpanV1>) -> DeviceBufferSpanV1 {
        match self {
            Self::Embedded => scratch.embedded,
            Self::Streams(index) => streams(scratch, index),
            Self::Pre(index) => pre(scratch, index),
            Self::Post => scratch.post,
            Self::Comb => scratch.comb,
            Self::Collapsed => scratch.collapsed,
            Self::Normed => scratch.normed,
            Self::HeadCollapsed => scratch.head_collapsed,
            Self::NormedWide => scratch.normed_wide,
            Self::WindowLatent => scratch.window_latent,
            Self::Reduced => scratch.reduced,
            Self::ExpertWeights => scratch.expert_weights,
            Self::ExpertIds => scratch.expert_ids,
        }
    }
}

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

fn dense(name: String, buffer: Buffer, cols: u32, dtype: TapDType) -> Tap {
    tap(name, buffer, cols, dtype, TapCompare::Dense)
}

fn scope(layer: u32) -> String {
    format!("L{layer:02}")
}

/// The reference's name of a sublayer's mixer and of its input.
const fn names(sublayer: Sublayer) -> (&'static str, &'static str) {
    match sublayer {
        Sublayer::Attention => ("attn", "attn"),
        Sublayer::Mlp => ("ffn", "mlp"),
    }
}

/// The taps to copy after `stage`, before the environment's selection.
pub fn taps(stage: Stage, shape: &Shape) -> Vec<Tap> {
    let d = shape.dims();
    let hidden = d.hidden;
    let width = d.hc.streams * hidden;
    let mut taps = Vec::new();
    let mut layer_tap = |tap: Tap, layer: u32| {
        let mut tap = tap;
        tap.spec.layer = Some(layer);
        taps.push(tap);
    };
    match stage {
        Stage::EmbeddingReduce => {
            return vec![dense(
                "embed.out".into(),
                Buffer::Embedded,
                hidden,
                TapDType::Bf16,
            )];
        }
        Stage::HcMix(layer, sublayer) => {
            let (name, index) = (scope(layer), sublayer_index(layer, sublayer));
            let (mixer, input) = names(sublayer);
            if sublayer == Sublayer::Attention {
                let streams = Buffer::Streams(index);
                layer_tap(
                    dense(format!("{name}.in.streams"), streams, width, TapDType::Bf16),
                    layer,
                );
            }
            let hc = format!("{name}.x.hc.{mixer}");
            let streams = d.hc.streams;
            for (site, buffer, cols) in [
                ("pre", Buffer::Pre(index), streams),
                ("post", Buffer::Post, streams),
                ("comb", Buffer::Comb, streams * streams),
            ] {
                layer_tap(
                    dense(format!("{hc}.{site}"), buffer, cols, TapDType::F32),
                    layer,
                );
            }
            let collapsed = format!("{name}.x.hc.{input}");
            layer_tap(
                dense(collapsed, Buffer::Collapsed, hidden, TapDType::Bf16),
                layer,
            );
        }
        Stage::InputNorm(layer, sublayer) => {
            let name = format!("{}.{}.in", scope(layer), names(sublayer).1);
            layer_tap(dense(name, Buffer::Normed, hidden, TapDType::Bf16), layer);
        }
        Stage::WindowRingWrite(layer) => {
            // A prefill's window write's normalized latent, its tail
            // rotated.
            let name = format!("{}.x.latent", scope(layer));
            let cols = d.attention.head_dim;
            layer_tap(
                dense(name, Buffer::WindowLatent, cols, TapDType::F32),
                layer,
            );
        }
        Stage::Route(layer) => {
            let name = format!("{}.router", scope(layer));
            let top_k = d.moe.top_k;
            let weights = format!("{name}.weights");
            let weights = tap(
                weights,
                Buffer::ExpertWeights,
                top_k,
                TapDType::F32,
                TapCompare::Paired,
            );
            layer_tap(weights, layer);
            let ids = format!("{name}.ids");
            let ids = tap(
                ids,
                Buffer::ExpertIds,
                top_k,
                TapDType::U32,
                TapCompare::Set,
            );
            layer_tap(ids, layer);
        }
        Stage::Reduce(layer, sublayer) => {
            let name = format!("{}.{}.out", scope(layer), names(sublayer).1);
            layer_tap(dense(name, Buffer::Reduced, hidden, TapDType::Bf16), layer);
        }
        Stage::HcExpand(layer, Sublayer::Mlp) => {
            let next = Buffer::Streams(sublayer_index(layer, Sublayer::Mlp) + 1);
            let name = format!("{}.out", scope(layer));
            layer_tap(dense(name, next, width, TapDType::Bf16), layer);
        }
        Stage::HeadCollapse => {
            let name = "final.x.hc.head".into();
            return vec![dense(name, Buffer::HeadCollapsed, hidden, TapDType::F32)];
        }
        Stage::FinalNorm => {
            return vec![dense(
                "final.norm".into(),
                Buffer::NormedWide,
                hidden,
                TapDType::F32,
            )];
        }
        _ => {}
    }
    taps
}

/// Every tap of a prefill's program, once per name.
pub fn specs(shape: &Shape) -> Vec<TapSpec> {
    stages(shape.geometry, Phase::Prefill { rows: 1 })
        .into_iter()
        .flat_map(|stage| taps(stage, shape))
        .map(|tap| tap.spec)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};
    use runtime_bundle_sdk::taps::{Declaration, TapFilter, Taps};

    use super::*;
    use crate::buffers::scratch_rows;

    fn declaration() -> Declaration {
        Declaration::parse(include_str!("../../../taps.json")).unwrap()
    }

    /// The scratch of `shape`, each buffer a distinct fake allocation.
    fn bound(shape: &Shape) -> Scratch<DeviceBufferSpanV1> {
        let mut next = 0;
        bundle_runtime::scratch::bind(Scratch::of(shape), |bytes| {
            next += 1;
            Ok::<_, runtime_bundle_abi::BundleAbiError>(DeviceBufferSpanV1::new(
                runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(next, 1),
                0,
                bytes,
            ))
        })
        .unwrap()
    }

    /// Each tap reads whole rows of its buffer, at every geometry and rank
    /// count.
    #[test]
    fn taps_read_whole_rows_of_their_buffers() {
        for g in GEOMETRIES {
            for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
                let Ok(shape) = Shape::new(g, ranks) else {
                    continue;
                };
                let scratch = bound(&shape);
                for stage in stages(g, Phase::Prefill { rows: 1 }) {
                    for tap in taps(stage, &shape) {
                        let row = tap.buffer.span(&scratch).length / u64::from(scratch_rows());
                        assert_eq!(
                            row,
                            u64::from(tap.spec.cols) * tap.spec.dtype.bytes(),
                            "{}",
                            tap.spec.name
                        );
                    }
                }
            }
        }
    }

    /// The head's taps read the F32 rows its collapse and final norm write
    /// (`lower::head` into `head_collapsed`, `lower::final_norm` into
    /// `normed_wide`), never the last sublayer's BF16 collapse and norm,
    /// which those stages leave holding the last layer's FFN input.
    #[test]
    fn the_head_taps_read_the_rows_the_head_reads() {
        for g in GEOMETRIES {
            let shape = Shape::new(g, Ranks::TP1).unwrap();
            let scratch = bound(&shape);
            let head = taps(Stage::HeadCollapse, &shape);
            let norm = taps(Stage::FinalNorm, &shape);
            let [head] = &head[..] else {
                panic!("one head collapse tap")
            };
            let [norm] = &norm[..] else {
                panic!("one final norm tap")
            };
            assert_eq!(head.buffer.span(&scratch), scratch.head_collapsed);
            assert_eq!(norm.buffer.span(&scratch), scratch.normed_wide);
            assert_eq!(head.spec.dtype, TapDType::F32);
            assert_eq!(norm.spec.dtype, TapDType::F32);
        }
    }

    /// Every tap's entry declares its dtype, sharding and comparison, and
    /// every entry declares a tap. The text entries leave the columns to
    /// the geometry, which is the only difference `check` may report; the
    /// encoder entries (`crate::encoders::taps`) declare theirs.
    #[test]
    fn taps_json_declares_every_tap() {
        let declaration = declaration();
        let mut used = vec![false; declaration.entries().len()];
        let encoder = crate::encoders::taps().unwrap();
        // The aligner's cells and 32 blocks of the 1024-wide F32 residual
        // stream, with the transformer's input.
        assert_eq!(encoder.len(), 1 + 1 + 32);
        for spec in &encoder {
            used[declaration.check(spec, 4).unwrap()] = true;
        }
        for g in GEOMETRIES {
            let shape = Shape::new(g, Ranks::TP1).unwrap();
            for spec in specs(&shape) {
                let index = match declaration.check(&spec, 1) {
                    Ok(index) => index,
                    Err(error) => {
                        let (index, _) = declaration.entry(&spec.name).unwrap();
                        assert!(
                            error.ends_with("its cols differs")
                                && declaration.entries()[index].get("cols").is_none(),
                            "{error}"
                        );
                        index
                    }
                };
                used[index] = true;
            }
        }
        for (entry, used) in declaration.entries().iter().zip(used) {
            assert!(used, "{} matches no tap", entry["name"]);
        }
    }

    /// The layers' taps follow the reference's sites in program order, and
    /// the filter keeps a layer's.
    #[test]
    fn every_layer_taps_the_reference_sites_in_program_order() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let names: Vec<String> = specs(&shape).into_iter().map(|spec| spec.name).collect();
        assert_eq!(names[0], "embed.out");
        let layer: Vec<&str> = names
            .iter()
            .filter_map(|name| name.strip_prefix("L03."))
            .collect();
        assert_eq!(
            layer,
            [
                "in.streams",
                "x.hc.attn.pre",
                "x.hc.attn.post",
                "x.hc.attn.comb",
                "x.hc.attn",
                "attn.in",
                "x.latent",
                "attn.out",
                "x.hc.ffn.pre",
                "x.hc.ffn.post",
                "x.hc.ffn.comb",
                "x.hc.mlp",
                "mlp.in",
                "router.weights",
                "router.ids",
                "mlp.out",
                "out",
            ]
        );
        assert_eq!(names[names.len() - 2..], ["final.x.hc.head", "final.norm"]);
        let filter = TapFilter {
            layers: None,
            sites: Some(vec!["out".into()]),
            from: None,
        };
        let taps = Taps::new(PathBuf::new(), filter);
        let kept: Vec<String> = specs(&shape)
            .into_iter()
            .filter(|spec| taps.selected(spec))
            .map(|spec| spec.name)
            .collect();
        let mut expected = vec!["embed.out".to_owned()];
        expected.extend((0..FLASH.layers()).map(|layer| format!("L{layer:02}.out")));
        assert_eq!(kept, expected);
    }
}
