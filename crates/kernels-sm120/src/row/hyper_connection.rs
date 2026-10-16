//! Rows of hyper-connection models (`cuda/row/qwen38_ple.cuh`): the
//! hyper-connection grouped norm, bottleneck, mix and inject, the n-gram hash
//! gather and the PLE gate and causal convolution. Inputs and outputs are
//! BF16, reductions F32. Live counts are device scalars; prepared capacity
//! is a launch bound. A model builds its rows from these builders and lists
//! them in its catalog (`ModelCatalog::hyper_connection_rows`).

use crate::invariance::{Anchors, Invariance};
use crate::{Catalogs, Profile};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan, OpFamily,
    Refusal, SigmoidMeanHyperConnectionQuery as H, StaticQuery, Target, Validate,
};
use kernel_core::contract::{ConvActivation, NgramQuery, RecurrentPhase, RecurrentQuery};
use std::path::PathBuf;

/// Most rows of one launch.
pub const MAX_ROWS: u32 = 8192;

/// The hyper-connection streams of a model's rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Streams {
    pub streams: u32,
    /// Each stream's width.
    pub width: u32,
    /// The bottleneck's rank.
    pub rank: u32,
    /// The grouped norm's epsilon, as F32 bits.
    pub eps_bits: u32,
    /// Prepared rows of one launch.
    pub capacity: u32,
}

/// The n-gram hash gather's table geometry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Ngram {
    pub orders: u32,
    pub heads_per_order: u32,
    pub head_width: u32,
    pub capacity: u32,
}

// These bounds mirror the fixed row implementations in qwen38.cuh and
// qwen38_ple.cuh. Refuse other computations before publishing an entry.
fn check_capacity(capacity: u32) -> Result<(), Refusal> {
    if capacity == 0 || capacity > MAX_ROWS {
        return Err(Refusal("row capacity must be between 1 and 8192"));
    }
    Ok(())
}

impl Streams {
    fn validate(self) -> Result<(), Refusal> {
        if self.streams != 4
            || self.width == 0
            || !self.width.is_multiple_of(256)
            || self.rank != 320
            || self.eps_bits != 1e-6_f32.to_bits()
        {
            return Err(Refusal(
                "unsupported hyper-connection row geometry or epsilon",
            ));
        }
        check_capacity(self.capacity)
    }
}

impl Ngram {
    /// SM120's n-gram template hashes orders 2 and 3 (`orders` 3) with 8
    /// heads each; the contract takes any orders from 2.
    fn validate(self, local: u32) -> Result<(), Refusal> {
        if self.orders != 3
            || self.heads_per_order != 8
            || self.head_width == 0
            || !self.head_width.is_multiple_of(16)
            || local == 0
            || !16_u32.is_multiple_of(local)
        {
            return Err(Refusal("unsupported n-gram row geometry or local heads"));
        }
        check_capacity(self.capacity)
    }
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub query: StaticQuery,
    pub symbol: String,
    pub params: Vec<EntryParam>,
    pub body: String,
    pub(crate) groups: u32,
    /// Decode entries over every bucket item at once: the grid covers the
    /// items and each reads its own row count.
    pub(crate) items: bool,
    /// The hyper-connection rows and the PLE gate are row-local in prefill
    /// and verify alike. The prefill n-gram hashes each row's token history,
    /// which it carries along its sequence, and the PLE convolution carries
    /// its window like the linear convolution, with its anchor state derived
    /// by the bundle's PLE anchor launch.
    pub invariance: Invariance,
}

impl Entry {
    fn checked_capacity(&self) -> Result<u32, Refusal> {
        let (capacity, groups) = match self.query {
            StaticQuery::HyperConnection(H::GroupedNorm {
                streams: 4,
                width,
                eps_bits,
                capacity,
            }) if width > 0
                && width.is_multiple_of(256)
                && eps_bits == 1e-6_f32.to_bits()
                && !self.items =>
            {
                (capacity, 4)
            }
            StaticQuery::HyperConnection(H::Bottleneck {
                streams: 4,
                rank: 320,
                capacity,
            }) if !self.items => (capacity, 1),
            StaticQuery::HyperConnection(H::Mix {
                streams: 4,
                width,
                capacity,
            }) if width > 0 && width.is_multiple_of(256) && !self.items => (capacity, width / 256),
            StaticQuery::HyperConnection(H::Inject {
                streams: 4,
                width,
                capacity,
            }) if width > 0 && width.is_multiple_of(256) && !self.items => (capacity, 4),
            StaticQuery::Ngram(NgramQuery::HashGather {
                orders,
                heads_per_order,
                head_width,
                local_heads,
                capacity,
            }) => {
                Ngram {
                    orders,
                    heads_per_order,
                    head_width,
                    capacity,
                }
                .validate(local_heads)?;
                (capacity, local_heads)
            }
            StaticQuery::Ngram(NgramQuery::PleGate {
                streams: 4,
                width,
                capacity,
            }) if width > 0 && width.is_multiple_of(256) && !self.items => (capacity, 4),
            StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                channels,
                kernel: 4,
                dilation: 3,
                activation: ConvActivation::Silu,
                phase,
                capacity,
            }) if channels > 0
                && channels.is_multiple_of(1024)
                && (phase == RecurrentPhase::Decode
                    || (phase == RecurrentPhase::Prefill && !self.items)) =>
            {
                (capacity, channels / 256)
            }
            _ => {
                return Err(Refusal(
                    "unsupported row query for the compiled computation",
                ));
            }
        };
        check_capacity(capacity)?;
        if self.groups != groups {
            return Err(Refusal("row query and launch groups disagree"));
        }
        Ok(capacity)
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.checked_capacity()?;
        let spec = InstantiationSpec {
            family: if matches!(self.query, StaticQuery::Recurrent(_)) {
                OpFamily::Recurrent
            } else {
                OpFamily::Row
            },
            key: CompileKey {
                query: self.query.clone(),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: 256,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: 8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        Ok(spec)
    }

    pub fn plan(&self, capacity: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if capacity == 0 || capacity > self.checked_capacity()? || live > capacity {
            return Err(Refusal("row count exceeds prepared capacity"));
        }
        if !self.items
            && matches!(
                self.query,
                StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                    phase: RecurrentPhase::Decode,
                    ..
                })
            )
            && live > 1
        {
            return Err(Refusal("PLE decode accepts one row per sequence"));
        }
        let plan = LaunchPlan {
            grid: [capacity, self.groups, 1],
            block: [256, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if values.len() != self.params.len()
            || !self.params.iter().zip(&values).all(|(p, a)| {
                matches!(
                    (p, a),
                    (EntryParam::DevicePointer { .. }, Argument::DeviceBuffer(_))
                        | (EntryParam::U32 { .. }, Argument::U32(_))
                        | (EntryParam::U64 { .. }, Argument::U64(_))
                )
            })
        {
            return Err(Refusal("row argument ABI mismatch"));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }
}

pub(crate) fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

/// Every model's rows in `catalogs`, in order.
pub fn catalog(catalogs: &Catalogs) -> Vec<Entry> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.hyper_connection_rows)())
        .collect()
}

/// The grouped norm, bottleneck, mix and inject of `streams`, named
/// `chop_row_{tag}_hc_{kind}_v1` after `profile`. Four streams with width
/// divisible by 256, bottleneck rank 320 and epsilon 1e-6 are implemented.
pub fn hc_entries(profile: Profile, streams: Streams) -> Result<Vec<Entry>, Refusal> {
    streams.validate()?;
    let Streams {
        streams,
        width,
        rank,
        eps_bits,
        capacity,
    } = streams;
    let mut out = Vec::new();
    for (name, query, groups, operands, body) in [
        (
            "hc_norm",
            H::GroupedNorm {
                streams,
                width,
                eps_bits,
                capacity,
            },
            streams,
            vec!["x", "weight"],
            "hc_norm(x, weight, output, live_rows)",
        ),
        (
            "hc_bottleneck",
            H::Bottleneck {
                streams,
                rank,
                capacity,
            },
            1,
            vec!["x"],
            "hc_bottleneck(x, output, live_rows)",
        ),
        (
            "hc_mix",
            H::Mix {
                streams,
                width,
                capacity,
            },
            width / 256,
            vec!["up", "normed"],
            "hc_mix(up, normed, output, live_rows)",
        ),
        (
            "hc_inject",
            H::Inject {
                streams,
                width,
                capacity,
            },
            streams,
            vec!["x", "y", "logits"],
            "hc_inject(x, y, logits, output, live_rows)",
        ),
    ] {
        let mut params = operands
            .into_iter()
            .map(|n| ptr(n, "const uint16_t"))
            .collect::<Vec<_>>();
        params.extend([
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ]);
        out.push(Entry {
            profile,
            query: StaticQuery::HyperConnection(query),
            symbol: format!("chop_row_{}_{name}_v1", profile.tag()),
            params,
            body: format!("chop::row::qwen38::{body};"),
            groups,
            items: false,
            invariance: Invariance::any_rows(capacity),
        });
    }
    Ok(out)
}

/// The row cubin `row_{tag}` of `owner`'s entries among `entries`.
pub fn cubin_spec(owner: Profile, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: format!("row_{}", owner.tag()),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"row/qwen38_ple.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .filter(|e| e.profile == owner)
            .map(|e| EntrySpec::new(e.symbol.clone(), e.params.clone(), e.body.clone(), 256, 1))
            .collect(),
    }
}

/// One cubin per profile of `catalogs` with rows (`cubin_spec`), or
/// `profile`'s alone.
pub fn cubin_specs(catalogs: &Catalogs, profile: Option<Profile>) -> Vec<CubinSpec> {
    let entries = catalog(catalogs);
    catalogs
        .profiles()
        .into_iter()
        .filter(|&owner| profile.is_none_or(|selected| selected == owner))
        .filter(|&owner| entries.iter().any(|e| e.profile == owner))
        .map(|owner| {
            let mut spec = cubin_spec(owner, &entries);
            let model = catalogs
                .models()
                .iter()
                .find(|model| model.profiles.contains(&owner))
                .expect("hyper-connection rows have an owning catalog");
            (model.hyper_connection_sources)(owner, &mut spec);
            spec
        })
        .collect()
}

/// The n-gram template argument for `local` hash heads per rank: none when
/// it equals the catalog-supplied default in `row/qwen38_ple.cuh`, so that
/// entry's body (and cubin) keeps its spelling.
fn ngram_template(local: u32, header_local: u32) -> String {
    if local == header_local {
        String::new()
    } else {
        format!("<{local}>")
    }
}

/// The per-sequence n-gram rows of `profile`: `local` hash heads per rank
/// (grid y), head `rank * local + y`, named `chop_row_{tag}_ngram_v1`.
/// The hash policy uses orders 2 and 3. The model supplies the EOS token
/// and head geometry; local heads must divide the total head count.
pub fn ngram_entry(
    profile: Profile,
    ngram: Ngram,
    local: u32,
    header_local: u32,
) -> Result<Entry, Refusal> {
    ngram.validate(local)?;
    let template = ngram_template(local, header_local);
    Ok(Entry {
        profile,
        query: StaticQuery::Ngram(NgramQuery::HashGather {
            orders: ngram.orders,
            heads_per_order: ngram.heads_per_order,
            head_width: ngram.head_width,
            local_heads: local,
            capacity: ngram.capacity,
        }),
        symbol: format!("chop_row_{}_ngram_v1", profile.tag()),
        groups: local,
        params: vec![
            ptr("tokens", "const uint32_t"),
            ptr("history", "const uint32_t"),
            ptr("multipliers", "const uint64_t"),
            ptr("sizes", "const uint64_t"),
            ptr("offsets", "const uint64_t"),
            ptr("table", "const uint8_t"),
            ptr("scale", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("row_ids", "uint64_t"),
            ptr("history_out", "uint32_t"),
            ptr("live_rows", "const uint32_t"),
            EntryParam::U32 {
                name: "rank".into(),
            },
            EntryParam::U64 {
                name: "table_start".into(),
            },
            EntryParam::U64 {
                name: "table_rows".into(),
            },
        ],
        body: format!(
            "chop::row::qwen38::ngram{template}(tokens, history, multipliers, sizes, offsets, table, scale, output, row_ids, history_out, live_rows, rank, table_start, table_rows);"
        ),
        items: false,
        invariance: Invariance::PREFILL,
    })
}

/// The n-gram over every decode bucket item of `ngram`'s profile, with its
/// query and local heads.
pub fn ngram_decode_entry(ngram: &Entry, header_local: u32) -> Result<Entry, Refusal> {
    ngram.contract()?;
    if ngram.items {
        return Err(Refusal("expected a per-sequence n-gram entry"));
    }
    let StaticQuery::Ngram(NgramQuery::HashGather { local_heads, .. }) = ngram.query else {
        return Err(Refusal("expected an n-gram entry"));
    };
    let template = ngram_template(local_heads, header_local);
    Ok(Entry {
        profile: ngram.profile,
        query: ngram.query.clone(),
        symbol: format!("chop_row_{}_ngram_decode_v1", ngram.profile.tag()),
        params: with_items(&ngram.params)?,
        body: format!(
            "chop::row::qwen38::ngram_decode{template}(tokens, history, multipliers, sizes, offsets, table, scale, output, row_ids, history_out, q_len, rank, table_start, table_rows, entry);"
        ),
        groups: ngram.groups,
        items: true,
        invariance: Invariance::NONE,
    })
}

/// A per-sequence entry's parameters for every decode item: `q_len` in
/// place of `live_rows`, then the staged state stride `entry`.
fn with_items(params: &[EntryParam]) -> Result<Vec<EntryParam>, Refusal> {
    let mut params = params.to_vec();
    let live = params
        .iter()
        .position(|p| matches!(p, EntryParam::DevicePointer { name, .. } if name == "live_rows"))
        .ok_or(Refusal("per-sequence row entry has no live_rows parameter"))?;
    params[live] = ptr("q_len", "const uint32_t");
    params.push(EntryParam::U64 {
        name: "entry".into(),
    });
    Ok(params)
}

/// The PLE gate of `streams` (one group per stream), named
/// `chop_row_{tag}_ple_gate_v1`. Refuses unsupported stream geometry.
pub fn ple_gate_entry(profile: Profile, streams: Streams) -> Result<Entry, Refusal> {
    streams.validate()?;
    Ok(Entry {
        profile,
        query: StaticQuery::Ngram(NgramQuery::PleGate {
            streams: streams.streams,
            width: streams.width,
            capacity: streams.capacity,
        }),
        symbol: format!("chop_row_{}_ple_gate_v1", profile.tag()),
        groups: streams.streams,
        params: vec![
            ptr("key", "const uint16_t"),
            ptr("query", "const uint16_t"),
            ptr("value", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        body: "chop::row::qwen38::ple_gate(key, query, value, output, live_rows);".into(),
        items: false,
        invariance: Invariance::any_rows(streams.capacity),
    })
}

/// The PLE convolution over every stream's columns at `phase`: a SiLU
/// causal convolution of 4 taps at dilation 3 in 256-column groups, named
/// `chop_row_{tag}_ple_conv_{phase}_v1`. Prefill rows carry the window as
/// the linear convolution does, with the anchor state the bundle's PLE
/// anchor launch derives. Refuses unsupported stream geometry and verify phases.
pub fn ple_conv_entry(
    profile: Profile,
    streams: Streams,
    phase: RecurrentPhase,
) -> Result<Entry, Refusal> {
    streams.validate()?;
    let channels = streams.width * streams.streams;
    let suffix = match phase {
        RecurrentPhase::Decode => "decode",
        RecurrentPhase::Prefill => "prefill",
        RecurrentPhase::Verify { .. } => {
            return Err(Refusal("the PLE convolution has no verify entry"));
        }
    };
    Ok(Entry {
        profile,
        query: StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
            channels,
            kernel: 4,
            dilation: 3,
            activation: ConvActivation::Silu,
            phase,
            capacity: streams.capacity,
        }),
        symbol: format!("chop_row_{}_ple_conv_{suffix}_v1", profile.tag()),
        groups: channels / 256,
        params: vec![
            ptr("input", "const uint16_t"),
            ptr("weight", "const uint16_t"),
            ptr("state", "const uint16_t"),
            ptr("output", "uint16_t"),
            ptr("state_out", "uint16_t"),
            ptr("live_rows", "const uint32_t"),
        ],
        body: "chop::row::qwen38::ple_conv(input, weight, state, output, state_out, live_rows);"
            .into(),
        items: false,
        invariance: match phase {
            RecurrentPhase::Prefill => Invariance::PREFILL.with_anchors(Anchors::Companion),
            _ => Invariance::NONE,
        },
    })
}

/// The PLE convolution over every decode bucket item in one launch (grid x
/// is the bucket), with the per-sequence decode entry `conv`'s query.
/// `q_len` holds each item's row count (zero or one) and `entry` the
/// distance between consecutive items' staged state, in elements.
pub fn ple_conv_items_entry(conv: &Entry) -> Result<Entry, Refusal> {
    conv.contract()?;
    if conv.items
        || !matches!(
            conv.query,
            StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                phase: RecurrentPhase::Decode,
                ..
            })
        )
    {
        return Err(Refusal("expected a per-sequence decode convolution entry"));
    }
    Ok(Entry {
        profile: conv.profile,
        query: conv.query.clone(),
        symbol: format!("chop_row_{}_ple_conv_items_decode_v1", conv.profile.tag()),
        params: with_items(&conv.params)?,
        body: "chop::row::qwen38::ple_conv_decode(input, weight, state, output, state_out, q_len, entry);".into(),
        groups: conv.groups,
        items: true,
        invariance: Invariance::NONE,
    })
}

#[cfg(test)]
mod geometry_regression {
    use super::*;

    const SUPPORTED: Streams = Streams {
        streams: 4,
        width: 2560,
        rank: 320,
        eps_bits: 1e-6_f32.to_bits(),
        capacity: 32,
    };

    const HASH: Ngram = Ngram {
        orders: 3,
        heads_per_order: 8,
        head_width: 160,
        capacity: 32,
    };

    #[test]
    fn ngram_bodies_follow_the_catalog_header_default() {
        for local in [4, 8] {
            let entry = ngram_entry(Profile::StdDenseMoe, HASH, local, 8).unwrap();
            let decode = ngram_decode_entry(&entry, 8).unwrap();
            let suffix = if local == 8 { "" } else { "<4>" };
            assert!(entry.body.contains(&format!("::ngram{suffix}(")));
            assert!(decode.body.contains(&format!("::ngram_decode{suffix}(")));
        }
    }

    #[test]
    fn refuses_geometry_the_cuda_cannot_emit() {
        for other in [
            Streams {
                streams: 2,
                width: 1024,
                rank: 64,
                eps_bits: 1e-5_f32.to_bits(),
                ..SUPPORTED
            },
            Streams {
                streams: 2,
                ..SUPPORTED
            },
            Streams {
                width: 1025,
                ..SUPPORTED
            },
            Streams {
                rank: 64,
                ..SUPPORTED
            },
            Streams {
                eps_bits: 1e-5_f32.to_bits(),
                ..SUPPORTED
            },
        ] {
            assert!(hc_entries(Profile::StdDenseMoe, other).is_err());
            assert!(ple_gate_entry(Profile::StdDenseMoe, other).is_err());
            for phase in [RecurrentPhase::Prefill, RecurrentPhase::Decode] {
                assert!(ple_conv_entry(Profile::StdDenseMoe, other, phase).is_err());
            }
        }
    }

    #[test]
    fn ngram_refuses_other_hash_geometry_and_invalid_head_splits() {
        for other in [
            Ngram {
                orders: 4,
                heads_per_order: 4,
                head_width: 128,
                ..HASH
            },
            Ngram { orders: 4, ..HASH },
            Ngram { orders: 9, ..HASH },
            Ngram {
                heads_per_order: 4,
                ..HASH
            },
            Ngram {
                head_width: 127,
                ..HASH
            },
        ] {
            assert!(ngram_entry(Profile::StdDenseMoe, other, 4, 4).is_err());
        }
        for local in [0, 3, 17] {
            assert!(ngram_entry(Profile::StdDenseMoe, HASH, local, 4).is_err());
        }
        for local in [1, 2, 4, 8, 16] {
            let entry = ngram_entry(Profile::StdDenseMoe, HASH, local, 4).unwrap();
            let decode = ngram_decode_entry(&entry, 4).unwrap();
            assert_eq!(entry.plan(32, 1).unwrap().grid, [32, local, 1]);
            assert_eq!(decode.plan(32, 1).unwrap().grid, [32, local, 1]);
            let template = if local == 4 {
                String::new()
            } else {
                format!("<{local}>")
            };
            assert!(entry.body.contains(&format!("::ngram{template}(")));
            assert!(decode.body.contains(&format!("::ngram_decode{template}(")));
        }
    }

    #[test]
    fn parameterized_widths_use_matching_launch_groups() {
        let streams = Streams {
            width: 1024,
            ..SUPPORTED
        };
        for entry in hc_entries(Profile::StdDenseMoe, streams).unwrap() {
            entry.plan(32, 1).unwrap();
        }
        let mix = hc_entries(Profile::StdDenseMoe, streams).unwrap().remove(2);
        assert_eq!(mix.plan(32, 1).unwrap().grid, [32, 4, 1]);
        let conv = ple_conv_entry(Profile::StdDenseMoe, streams, RecurrentPhase::Prefill).unwrap();
        assert_eq!(conv.plan(32, 1).unwrap().grid, [32, 16, 1]);
        let hash = ngram_entry(
            Profile::StdDenseMoe,
            Ngram {
                head_width: 64,
                ..HASH
            },
            4,
            4,
        )
        .unwrap();
        assert_eq!(hash.plan(32, 1).unwrap().grid, [32, 4, 1]);
    }

    #[test]
    fn builders_refuse_invalid_capacity() {
        for capacity in [0, MAX_ROWS + 1] {
            let streams = Streams {
                capacity,
                ..SUPPORTED
            };
            assert!(hc_entries(Profile::StdDenseMoe, streams).is_err());
            assert!(ple_gate_entry(Profile::StdDenseMoe, streams).is_err());
            assert!(
                ple_conv_entry(Profile::StdDenseMoe, streams, RecurrentPhase::Prefill).is_err()
            );
            assert!(ngram_entry(Profile::StdDenseMoe, Ngram { capacity, ..HASH }, 4, 4).is_err());
        }
    }

    #[test]
    fn plans_obey_the_query_capacity() {
        let profile = Profile::StdDenseMoe;
        let mut entries = hc_entries(profile, SUPPORTED).unwrap();
        let ngram = ngram_entry(profile, HASH, 4, 4).unwrap();
        let conv = ple_conv_entry(profile, SUPPORTED, RecurrentPhase::Decode).unwrap();
        entries.extend([
            ngram_decode_entry(&ngram, 4).unwrap(),
            ngram,
            ple_gate_entry(profile, SUPPORTED).unwrap(),
            ple_conv_entry(profile, SUPPORTED, RecurrentPhase::Prefill).unwrap(),
            ple_conv_items_entry(&conv).unwrap(),
            conv,
        ]);
        for entry in entries {
            assert!(entry.plan(32, 1).is_ok(), "{}", entry.symbol);
            assert!(entry.plan(64, 1).is_err(), "{}", entry.symbol);
            assert!(entry.plan(0, 0).is_err());
            assert!(entry.plan(16, 17).is_err());
        }
    }

    #[test]
    fn decode_adapters_refuse_other_entry_kinds() {
        let profile = Profile::StdDenseMoe;
        let gate = ple_gate_entry(profile, SUPPORTED).unwrap();
        assert!(ngram_decode_entry(&gate, 4).is_err());
        assert!(ple_conv_items_entry(&gate).is_err());
        let prefill = ple_conv_entry(profile, SUPPORTED, RecurrentPhase::Prefill).unwrap();
        assert!(ple_conv_items_entry(&prefill).is_err());
        assert!(
            ple_conv_entry(profile, SUPPORTED, RecurrentPhase::Verify { max_rows: 4 }).is_err()
        );
        let ngram = ngram_entry(profile, HASH, 4, 4).unwrap();
        let decode = ngram_decode_entry(&ngram, 4).unwrap();
        assert!(ngram_decode_entry(&decode, 4).is_err());
        let mut missing_count = ngram;
        missing_count.params.clear();
        assert!(ngram_decode_entry(&missing_count, 4).is_err());
    }

    #[test]
    fn changed_queries_cannot_claim_another_compiled_geometry() {
        let mut norm = hc_entries(Profile::StdDenseMoe, SUPPORTED)
            .unwrap()
            .remove(0);
        let StaticQuery::HyperConnection(H::GroupedNorm {
            ref mut eps_bits, ..
        }) = norm.query
        else {
            unreachable!()
        };
        *eps_bits = 1e-5_f32.to_bits();
        assert!(norm.contract().is_err());
        let mut hash = ngram_entry(Profile::StdDenseMoe, HASH, 4, 4).unwrap();
        let StaticQuery::Ngram(NgramQuery::HashGather {
            ref mut local_heads,
            ..
        }) = hash.query
        else {
            unreachable!()
        };
        *local_heads = 2;
        assert!(hash.contract().is_err());
    }
}
