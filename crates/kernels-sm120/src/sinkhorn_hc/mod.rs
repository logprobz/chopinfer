//! Sinkhorn-mixed hyper-connection engine (`cuda/sinkhorn_hc`): the
//! mix-collapse before each sublayer, the expand after it and the
//! mean-of-streams head of `kernel_core::sinkhorn_hyper_connection`. Every
//! entry is row-local with a capacity-sized grid and the live row count on
//! the device, so one entry per op serves decode, verify and prefill
//! buckets up to its capacity. A different op from the sigmoid-mean
//! hyper-connection rows (`row::hyper_connection`). See
//! `cuda/sinkhorn_hc/README.md` for the ABI, the schedule and the
//! numerics.
//!
//! Declared exactness ([`Entry::invariance`] and the split, tiled and
//! fused entries' own): each row's outputs are a function of that row's
//! streams and the shared mixer alone. A CTA (or a split's CTAs, or a
//! tile's lanes) reduces each row in an order fixed by the mixer's width:
//! 128-K chains from zero folded in K order, the squares' fixed thread
//! mapping and butterfly, the Sinkhorn per row. No order depends on the
//! launch's rows, the row's place in it or the other rows' values, and
//! no entry hands state to a later launch. So every served entry claims
//! one-row bits up to its capacity, any chunk split and row independence
//! (`Invariance::any_rows`), the one-row bits being the one-CTA
//! mix-collapse's for every mix-collapse form (the split and tiled
//! entries write them at every row count). The lab's probes declare
//! nothing: they serve no op.

pub mod breakdown;
pub mod completion;
pub mod expand_mix;
pub mod finish_norm;
pub mod finish_split;
pub mod finish_twin;
pub mod lagged;
pub mod lagged_split;
pub mod lagged_tiled;
pub mod lagged_warp;
pub mod mix_norm;
pub mod mix_norm_batched;
pub mod mix_norm_exact;
pub mod mix_norm_tree;
pub mod mixer_probe;
pub mod prefill_expand_mix;
pub mod prefill_expand_mix_norm;
pub mod prefill_fused;
pub mod prefill_warp;
pub mod round_chain_partials;
pub mod round_cols;
pub mod round_expand_mix;
pub mod round_rows;
pub mod split;
pub mod tiled;

use std::path::PathBuf;

use crate::{Profile, ProfileNames};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    MAX_COMM_RANKS, NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery as Query, SinkhornMix, SinkhornOrder,
    StreamHeadMode,
};

/// Most rows of one launch: a prefill chunk.
pub const MAX_ROWS: u32 = 8192;
/// Threads of every entry.
pub const THREADS: u32 = 256;
/// Columns of one expand or head CTA: two per thread.
pub const COLUMNS: u32 = 2 * THREADS;
/// K of one projection chain.
pub const CHAIN: u32 = 128;
/// Bytes of a row's flat streams the mix-collapse keeps in static shared
/// memory (BF16).
pub const STREAM_SHARED_LIMIT: u32 = 40_960;

/// Which kernel an entry runs. `MixCollapse` with `probe` also writes the
/// raw projection and rstd, for the lab's accumulation gate, and the
/// Sinkhorn input cells, for its bitwise check of the normalizations.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Op {
    MixCollapse { probe: bool },
    Expand,
    Head,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    pub query: Query,
    pub op: Op,
    pub params: Vec<EntryParam>,
}

/// Most streams of a mixer: one thread per comb cell of 256 threads, as
/// `sinkhorn_hc.cuh` asserts.
pub const MAX_STREAMS: u32 = 8;
/// Widest mixer row.
pub const MAX_WIDTH: u32 = 16_384;
/// Most Sinkhorn iterations.
pub const MAX_ITERATIONS: u32 = 64;

/// The limits of the per-sublayer kernels (this module's entries and the
/// split, tiled and fused ones): [`shape_limits`], and the per-sublayer
/// placement's BF16 mixing. The lagged placement's mix-collapse, the F32
/// expand and the sigmoid-weighted head are the lagged engine's
/// ([`lagged`]) and are refused here by name.
pub fn limits(query: Query) -> Result<(), Refusal> {
    query.validate()?;
    match query {
        Query::MixCollapse { mix, .. }
        | Query::MixCollapseNorm { mix, .. }
        | Query::AllReduceExpandMixNorm { mix, .. }
            if mix.placement != HyperConnectionPlacement::PerSublayer =>
        {
            return Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels take the per-sublayer placement; the lagged engine serves the lagged one",
            ));
        }
        Query::ExpandF32 { .. } => {
            return Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels serve the BF16 expand; the lagged engine serves the F32 one",
            ));
        }
        Query::Head {
            mode: StreamHeadMode::SigmoidWeighted | StreamHeadMode::SigmoidWeightedF32,
            ..
        } => {
            return Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels serve the mean-of-streams head; the lagged engine serves the sigmoid-weighted one",
            ));
        }
        _ => {}
    }
    shape_limits(query)
}

/// The limits every SM120 Sinkhorn kernel shares, on top of the contract's
/// math (the static assertions of `sinkhorn_hc.cuh`): 2 to [`MAX_STREAMS`]
/// streams; a mixer's width a multiple of [`CHAIN`] up to [`MAX_WIDTH`]
/// with 1 to [`MAX_ITERATIONS`] iterations; and a fused epilogue's ranks
/// within the comm pointer arrays' `MAX_COMM_RANKS` slots.
pub fn shape_limits(query: Query) -> Result<(), Refusal> {
    query.validate()?;
    let streams = match query {
        Query::MixCollapse { mix, .. }
        | Query::MixCollapseNorm { mix, .. }
        | Query::AllReduceExpandMixNorm { mix, .. } => {
            if !mix.width.is_multiple_of(CHAIN)
                || mix.width > MAX_WIDTH
                || mix.iterations > MAX_ITERATIONS
            {
                return Err(Refusal(
                    "SM120 Sinkhorn mixers take widths of whole 128-K chains up to 16384 and at most 64 iterations",
                ));
            }
            mix.streams
        }
        Query::Expand { streams, .. }
        | Query::ExpandF32 { streams, .. }
        | Query::Head { streams, .. } => streams,
    };
    if !(2..=MAX_STREAMS).contains(&streams) {
        return Err(Refusal("SM120 Sinkhorn kernels take 2 to 8 streams"));
    }
    if let Query::AllReduceExpandMixNorm { ranks, .. } = query
        && ranks > MAX_COMM_RANKS as u32
    {
        return Err(Refusal(
            "the Sinkhorn comm epilogue's pointer arrays hold 8 ranks",
        ));
    }
    Ok(())
}

/// The engine's own refusals on top of the contract's.
fn check(query: Query, op: Op) -> Result<u32, Refusal> {
    limits(query)?;
    let capacity = match (query, op) {
        (Query::MixCollapse { mix, capacity }, Op::MixCollapse { .. }) => {
            let SinkhornOrder::RowSoftmaxThenColumnFirst = mix.order;
            if mix.flat_width() * 2 > STREAM_SHARED_LIMIT {
                return Err(Refusal(
                    "Sinkhorn mixer streams exceed the 40 KiB shared-memory row",
                ));
            }
            capacity
        }
        (
            Query::Expand {
                width, capacity, ..
            },
            Op::Expand,
        )
        | (
            Query::Head {
                width, capacity, ..
            },
            Op::Head,
        ) => {
            if !width.is_multiple_of(2) {
                return Err(Refusal("Sinkhorn expand and head widths must be even"));
            }
            capacity
        }
        (Query::AllReduceExpandMixNorm { .. }, _) => {
            return Err(Refusal(
                "the fused Sinkhorn all-reduce epilogue is not implemented on the unfused path",
            ));
        }
        _ => return Err(Refusal("Sinkhorn query and kernel disagree")),
    };
    if capacity > MAX_ROWS {
        return Err(Refusal("Sinkhorn row capacity exceeds 8192"));
    }
    Ok(capacity)
}

impl Entry {
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        check(self.query, self.op)?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::SinkhornHyperConnection(self.query),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: if matches!(self.op, Op::MixCollapse { .. }) {
                        CHAIN as u16
                    } else {
                        COLUMNS as u16
                    },
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (THREADS / 32) as u8,
                    split_k_capacity: 1,
                },
                dense: None,
                target: Target::Sm120f,
            },
            symbol: self.symbol.clone(),
        };
        spec.validate()?;
        self.query.numerical_contract_id()?;
        Ok(spec)
    }

    /// The prepared row maximum.
    pub fn capacity(&self) -> Result<u32, Refusal> {
        check(self.query, self.op)
    }

    /// The entry's declared exactness (see the module docs): one-row bits
    /// up to its capacity, any chunk split and row independence for the
    /// served mix-collapse, expand and head; nothing for the probe, or for
    /// an entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        match (self.op, self.capacity()) {
            (Op::MixCollapse { probe: true }, _) | (_, Err(_)) => Invariance::NONE,
            (_, Ok(capacity)) => Invariance::any_rows(capacity),
        }
    }

    /// Launch plan for a prepared bucket of `rows` (at most the capacity).
    /// The grid depends only on the bucket; the kernels read the live count
    /// on the device, so one plan serves every graph replay of the bucket
    /// and `live_rows` is only checked. Rows sit on the slowest axis.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if rows == 0 || rows > self.capacity()? {
            return Err(Refusal("Sinkhorn bucket exceeds the entry capacity"));
        }
        if live_rows > rows {
            return Err(Refusal("Sinkhorn live rows exceed the prepared bucket"));
        }
        let grid = match self.query {
            Query::MixCollapse { .. } => [rows, 1, 1],
            Query::MixCollapseNorm { .. } => unreachable!("refused by check"),
            Query::Expand { width, .. } | Query::Head { width, .. } => {
                [width.div_ceil(COLUMNS), rows, 1]
            }
            Query::AllReduceExpandMixNorm { .. } | Query::ExpandF32 { .. } => {
                unreachable!("refused by check")
            }
        };
        let plan = LaunchPlan {
            grid,
            block: [THREADS, 1, 1],
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
                )
            })
        {
            return Err(Refusal("Sinkhorn hyper-connection argument ABI mismatch"));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn body(&self) -> String {
        match (self.query, self.op) {
            (Query::MixCollapse { mix, .. }, Op::MixCollapse { probe }) => {
                let traits = format!(
                    "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
                    mix.streams,
                    mix.width,
                    mix.iterations,
                    mix.eps_bits,
                    mix.norm_eps_bits,
                    mix.fn_dtype == DType::F32
                );
                let probes = if probe {
                    "projected,rstd,cells"
                } else {
                    "nullptr,nullptr,nullptr"
                };
                format!(
                    "chop::sinkhorn_hc::mix_collapse<{traits},{probe}>(x,fn,base,scale,collapsed,post,comb,{probes},live_rows);"
                )
            }
            (Query::Expand { streams, width, .. }, _) => format!(
                "chop::sinkhorn_hc::expand<{streams},{width}>(x,y,post,comb,out,live_rows);"
            ),
            (Query::Head { streams, width, .. }, _) => {
                format!("chop::sinkhorn_hc::head<{streams},{width}>(x,out,live_rows);")
            }
            _ => String::new(),
        }
    }
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

/// The mix-collapse entry ABI: x, fn (BF16 or F32), base, scale, collapsed,
/// post, comb, the probe's projection, rstd and Sinkhorn input cells, and
/// the live row count.
pub fn mix_params(fn_dtype: DType, probe: bool) -> Vec<EntryParam> {
    let mut params = vec![
        ptr("x", "const uint16_t"),
        ptr(
            "fn",
            if fn_dtype == DType::F32 {
                "const float"
            } else {
                "const uint16_t"
            },
        ),
        ptr("base", "const float"),
        ptr("scale", "const float"),
        ptr("collapsed", "uint16_t"),
        ptr("post", "uint16_t"),
        ptr("comb", "uint16_t"),
    ];
    if probe {
        params.extend([
            ptr("projected", "float"),
            ptr("rstd", "float"),
            ptr("cells", "float"),
        ]);
    }
    params.push(ptr("live_rows", "const uint32_t"));
    params
}

/// The expand entry ABI.
pub fn expand_params() -> Vec<EntryParam> {
    vec![
        ptr("x", "const uint16_t"),
        ptr("y", "const uint16_t"),
        ptr("post", "const uint16_t"),
        ptr("comb", "const uint16_t"),
        ptr("out", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ]
}

/// The head entry ABI.
pub fn head_params() -> Vec<EntryParam> {
    vec![
        ptr("x", "const uint16_t"),
        ptr("out", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ]
}

/// One mixer shape and the row capacity its entries are prepared for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Geometry {
    pub mix: SinkhornMix,
    pub capacity: u32,
}

/// The mix-collapse, expand and head of `geometry` under `profile`, with
/// the probe mix-collapse when `probe`. Symbols are
/// `chop_sinkhorn_hc_{tag}_<entry>_v1`.
pub fn entries(profile: Profile, geometry: Geometry, probe: bool) -> Vec<Entry> {
    let Geometry { mix, capacity } = geometry;
    let symbol = |name: &str| format!("chop_sinkhorn_hc_{}_{name}_v1", profile.tag());
    let mut list = vec![
        Entry {
            profile,
            symbol: symbol("mix_collapse"),
            query: Query::MixCollapse { mix, capacity },
            op: Op::MixCollapse { probe: false },
            params: mix_params(mix.fn_dtype, false),
        },
        Entry {
            profile,
            symbol: symbol("expand"),
            query: Query::Expand {
                streams: mix.streams,
                width: mix.width,
                capacity,
            },
            op: Op::Expand,
            params: expand_params(),
        },
        Entry {
            profile,
            symbol: symbol("head"),
            query: Query::Head {
                streams: mix.streams,
                width: mix.width,
                mode: StreamHeadMode::MeanOfStreams,
                capacity,
            },
            op: Op::Head,
            params: head_params(),
        },
    ];
    if probe {
        list.push(Entry {
            profile,
            symbol: symbol("mix_collapse_probe"),
            query: Query::MixCollapse { mix, capacity },
            op: Op::MixCollapse { probe: true },
            params: mix_params(mix.fn_dtype, true),
        });
    }
    list
}

/// The served mixer at four ranks (replicated, so the same at any rank
/// count): 4 streams of 4096, 20 iterations, eps 1e-6, norm eps 1e-5, BF16
/// fn, a mixer around every sublayer. One capacity of 8192 rows serves
/// decode batches 1-32, verify buckets of up to 6 rows per sequence and
/// prefill chunks up to 8192 tokens.
pub const SERVED: Geometry = Geometry {
    mix: SinkhornMix {
        streams: 4,
        width: 4096,
        iterations: 20,
        eps_bits: 1e-6_f32.to_bits(),
        norm_eps_bits: 1e-5_f32.to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::PerSublayer,
        fn_dtype: DType::Bf16,
    },
    capacity: MAX_ROWS,
};

/// The served shape's profile, named by its numbers.
pub const SERVED_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "sinkhorn-hc-4x4096",
    tag: "hc4x4096",
});

/// The engine's own catalog: the served shape's entries.
pub fn catalog() -> Vec<Entry> {
    entries(SERVED_PROFILE, SERVED, false)
}

/// The served entries plus the lab's probe mix-collapse.
pub fn lab_catalog() -> Vec<Entry> {
    entries(SERVED_PROFILE, SERVED, true)
}

/// A cubin of `entries` named `name`.
pub fn cubin_spec(name: String, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/sinkhorn_hc.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|entry| {
                EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    entry.body(),
                    THREADS,
                    1,
                )
            })
            .collect(),
    }
}

/// The served cubin, `sinkhorn_hc_{tag}_sm120`.
pub fn cubin_specs() -> Vec<CubinSpec> {
    let name = format!("sinkhorn_hc_{}_sm120", SERVED_PROFILE.tag());
    vec![cubin_spec(name, &catalog())]
}

/// The lab's cubins: the served cubin and `sinkhorn_hc_{tag}_probe_sm120`
/// with the probe mix-collapse.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let probes = lab_catalog()
        .into_iter()
        .filter(|entry| entry.op == Op::MixCollapse { probe: true })
        .collect::<Vec<_>>();
    let name = format!("sinkhorn_hc_{}_probe_sm120", SERVED_PROFILE.tag());
    cubin_specs()
        .into_iter()
        .chain([cubin_spec(name, &probes)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_entries_validate_and_plan_every_bucket() {
        assert_eq!(f32::from_bits(SERVED.mix.eps_bits), 1e-6);
        assert_eq!(f32::from_bits(SERVED.mix.norm_eps_bits), 1e-5);
        let entries = lab_catalog();
        assert_eq!(entries.len(), 4);
        for entry in &entries {
            entry.contract().unwrap();
            assert_eq!(entry.capacity().unwrap(), 8192);
            // Decode batches, verify buckets of 6 rows per sequence and
            // prefill chunks.
            for rows in [1, 32, 6 * 32, 8192] {
                let plan = entry.plan(rows, rows).unwrap();
                assert!(plan.pdl);
                assert_eq!(plan.block, [256, 1, 1]);
                assert_eq!(plan.dynamic_shared_bytes, 0);
                assert_eq!(plan.workspace_bytes, 0);
                let expected = match entry.op {
                    Op::MixCollapse { .. } => [rows, 1, 1],
                    _ => [8, rows, 1],
                };
                assert_eq!(plan.grid, expected, "{}", entry.symbol);
                entry.plan(rows, 0).unwrap();
            }
            assert!(entry.plan(8193, 1).is_err());
            assert!(entry.plan(0, 0).is_err());
            assert!(entry.plan(32, 33).is_err());
            let args = (0..entry.params.len() as u64)
                .map(Argument::DeviceBuffer)
                .collect::<Vec<_>>();
            entry.pack_arguments(args).unwrap();
            assert!(entry.pack_arguments(vec![Argument::U32(1)]).is_err());
        }
        let symbols = entries
            .iter()
            .map(|e| e.symbol.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            [
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_v1",
                "chop_sinkhorn_hc_hc4x4096_expand_v1",
                "chop_sinkhorn_hc_hc4x4096_head_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_probe_v1",
            ]
        );
        assert_eq!(
            entries[0].body(),
            "chop::sinkhorn_hc::mix_collapse<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>,false>(x,fn,base,scale,collapsed,post,comb,nullptr,nullptr,nullptr,live_rows);"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_sm120",
                "sinkhorn_hc_hc4x4096_probe_sm120"
            ]
        );
        assert_eq!(specs[0].entries.len(), 3);
        assert_eq!(specs[1].entries.len(), 1);
    }

    #[test]
    fn the_kernels_refuse_valid_mixers_past_their_limits() {
        for (streams, width, iterations) in [
            (4, 4128, 20),
            (2, 16_512, 20),
            (4, 4096, 65),
            (9, 1024, 20),
            (1, 1024, 20),
        ] {
            let mix = SinkhornMix {
                streams,
                width,
                iterations,
                ..SERVED.mix
            };
            let geometry = Geometry { mix, capacity: 32 };
            Query::MixCollapse { mix, capacity: 32 }.validate().unwrap();
            // Expand and head take any even width and need no iterations.
            let all = !(2..=MAX_STREAMS).contains(&streams);
            for entry in entries(SERVED_PROFILE, geometry, true) {
                let refused = all || matches!(entry.op, Op::MixCollapse { .. });
                assert_eq!(entry.contract().is_err(), refused, "{}", entry.symbol);
                assert_eq!(entry.plan(1, 1).is_err(), refused);
            }
            assert!(split::entry(SERVED_PROFILE, geometry).contract().is_err());
        }
        assert_eq!(
            limits(Query::AllReduceExpandMixNorm {
                ranks: 9,
                dtype: DType::F32,
                mix: SERVED.mix,
                rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                norm: kernel_core::rmsnorm::RmsNormReduction::Ordered,
                capacity: 1,
            }),
            Err(Refusal(
                "the Sinkhorn comm epilogue's pointer arrays hold 8 ranks"
            ))
        );
    }

    #[test]
    fn expand_and_head_serve_stream_rows_past_u32_elements() {
        // Eight streams of 2^29 columns: 2^32 elements a row, which the
        // u64 buffer sizes and offsets hold. The mixer's K does not fit u32.
        let mix = SinkhornMix {
            streams: 8,
            width: 1 << 29,
            ..SERVED.mix
        };
        let geometry = Geometry { mix, capacity: 1 };
        for entry in entries(SERVED_PROFILE, geometry, false) {
            match entry.op {
                Op::MixCollapse { .. } => assert!(entry.contract().is_err()),
                _ => {
                    entry.contract().unwrap();
                    assert_eq!(entry.plan(1, 1).unwrap().grid, [1 << 20, 1, 1]);
                }
            }
        }
    }

    #[test]
    fn unsupported_queries_refuse_by_name() {
        let wide = SinkhornMix {
            streams: 8,
            ..SERVED.mix
        };
        assert_eq!(
            check(
                Query::MixCollapse {
                    mix: wide,
                    capacity: 1
                },
                Op::MixCollapse { probe: false }
            ),
            Err(Refusal(
                "Sinkhorn mixer streams exceed the 40 KiB shared-memory row"
            ))
        );
        assert_eq!(
            check(
                Query::AllReduceExpandMixNorm {
                    ranks: 4,
                    dtype: DType::F32,
                    mix: SERVED.mix,
                    rmsnorm_eps_bits: 1e-5_f32.to_bits(),
                    norm: kernel_core::rmsnorm::RmsNormReduction::Ordered,
                    capacity: 256,
                },
                Op::Expand
            ),
            Err(Refusal(
                "the fused Sinkhorn all-reduce epilogue is not implemented on the unfused path"
            ))
        );
        assert_eq!(
            check(
                Query::Head {
                    streams: 4,
                    width: 4095,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity: 1
                },
                Op::Head
            ),
            Err(Refusal("Sinkhorn expand and head widths must be even"))
        );
        assert_eq!(
            check(
                Query::Expand {
                    streams: 4,
                    width: 4096,
                    capacity: 8193
                },
                Op::Expand
            ),
            Err(Refusal("Sinkhorn row capacity exceeds 8192"))
        );
        assert!(
            check(
                Query::MixCollapse {
                    mix: SERVED.mix,
                    capacity: 1
                },
                Op::Head
            )
            .is_err()
        );
        let lagged = SinkhornMix {
            placement: HyperConnectionPlacement::LaggedSinglePass,
            ..SERVED.mix
        };
        assert_eq!(
            check(
                Query::MixCollapse {
                    mix: lagged,
                    capacity: 1
                },
                Op::MixCollapse { probe: false }
            ),
            Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels take the per-sublayer placement; the lagged engine serves the lagged one"
            ))
        );
        assert_eq!(
            check(
                Query::ExpandF32 {
                    streams: 4,
                    width: 4096,
                    capacity: 1
                },
                Op::Expand
            ),
            Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels serve the BF16 expand; the lagged engine serves the F32 one"
            ))
        );
        assert_eq!(
            check(
                Query::Head {
                    streams: 4,
                    width: 4096,
                    mode: StreamHeadMode::SigmoidWeighted,
                    capacity: 1
                },
                Op::Head
            ),
            Err(Refusal(
                "SM120's per-sublayer Sinkhorn kernels serve the mean-of-streams head; the lagged engine serves the sigmoid-weighted one"
            ))
        );
    }
}
