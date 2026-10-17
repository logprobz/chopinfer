//! The lagged Sinkhorn hyper-connection engine (`cuda/sinkhorn_hc/lagged.cuh`):
//! the mix-collapse of `HyperConnectionPlacement::LaggedSinglePass`, which
//! collapses with the previous sublayer's F32 `pre` and stores its own
//! `pre`, `post` and `comb` in F32, the F32 expand (`ExpandF32`) and the
//! sigmoid-weighted head. One CTA per row for the mix-collapse, as the
//! per-sublayer one-CTA entry, whose projection, rstd and Sinkhorn it
//! shares statement for statement; the expand and head give each thread two
//! columns. See `cuda/sinkhorn_hc/README.md`, "Lagged placement".
//!
//! Declared exactness, as the per-sublayer entries' (`sinkhorn_hc`'s module
//! docs): each row's outputs are a function of that row's streams, its
//! collapse weights and the shared mixer, reduced in orders fixed by the
//! width, so every served entry claims one-row bits up to its capacity, any
//! chunk split and row independence. The probe declares nothing.

use std::path::PathBuf;

use super::{CHAIN, COLUMNS, Geometry, MAX_ROWS, STREAM_SHARED_LIMIT, THREADS, ptr, shape_limits};
use crate::{Profile, ProfileNames};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::{
    HyperConnectionPlacement, SinkhornHyperConnectionQuery as Query, SinkhornMix, SinkhornOrder,
    StreamHeadMode,
};

/// Which kernel an entry runs. `MixCollapse` with `probe` also writes the
/// raw projection, rstd and Sinkhorn input cells, as the per-sublayer probe.
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

/// The engine's refusals on top of the contract's: SM120's shared limits
/// (`shape_limits`), the lagged placement for the mix-collapse with its row
/// in 40 KiB of shared memory, the F32 expand, the sigmoid-weighted head,
/// even stream widths and at most [`MAX_ROWS`] rows.
fn check(query: Query, op: Op) -> Result<u32, Refusal> {
    shape_limits(query)?;
    let capacity = match (query, op) {
        (Query::MixCollapse { mix, capacity }, Op::MixCollapse { .. }) => {
            let SinkhornOrder::RowSoftmaxThenColumnFirst = mix.order;
            if mix.placement != HyperConnectionPlacement::LaggedSinglePass {
                return Err(Refusal(
                    "the lagged Sinkhorn mix-collapse takes the lagged placement",
                ));
            }
            if mix.flat_width() * 2 > STREAM_SHARED_LIMIT {
                return Err(Refusal(
                    "Sinkhorn mixer streams exceed the 40 KiB shared-memory row",
                ));
            }
            capacity
        }
        (
            Query::ExpandF32 {
                width, capacity, ..
            },
            Op::Expand,
        )
        | (
            Query::Head {
                width,
                mode: StreamHeadMode::SigmoidWeighted,
                capacity,
                ..
            },
            Op::Head,
        ) => {
            if !width.is_multiple_of(2) {
                return Err(Refusal("Sinkhorn expand and head widths must be even"));
            }
            capacity
        }
        _ => return Err(Refusal("lagged Sinkhorn query and kernel disagree")),
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

    /// The entry's declared exactness (see the module docs).
    pub fn invariance(&self) -> Invariance {
        match (self.op, self.capacity()) {
            (Op::MixCollapse { probe: true }, _) | (_, Err(_)) => Invariance::NONE,
            (_, Ok(capacity)) => Invariance::any_rows(capacity),
        }
    }

    /// Launch plan for a prepared bucket of `rows` (at most the capacity),
    /// as the per-sublayer entries plan theirs: the grid depends only on
    /// the bucket and the kernels read the live count on the device.
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
            Query::ExpandF32 { width, .. } | Query::Head { width, .. } => {
                [width.div_ceil(COLUMNS), rows, 1]
            }
            _ => unreachable!("refused by check"),
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
            return Err(Refusal(
                "lagged Sinkhorn hyper-connection argument ABI mismatch",
            ));
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
                    "chop::sinkhorn_hc::mix_collapse_lagged<{traits},{probe}>(x,fn,base,scale,pre_in,collapsed,pre,post,comb,{probes},live_rows);"
                )
            }
            (Query::ExpandF32 { streams, width, .. }, _) => format!(
                "chop::sinkhorn_hc::expand_f32<{streams},{width}>(x,y,post,comb,out,live_rows);"
            ),
            (Query::Head { streams, width, .. }, _) => {
                format!("chop::sinkhorn_hc::weighted_head<{streams},{width}>(x,pre,out,live_rows);")
            }
            _ => String::new(),
        }
    }
}

/// The lagged mix-collapse ABI: x, fn (BF16 or F32), base, scale, the F32
/// collapse weights pre_in, collapsed, the F32 pre, post and comb, the
/// probe's projection, rstd and Sinkhorn input cells, and the live row
/// count.
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
        ptr("pre_in", "const float"),
        ptr("collapsed", "uint16_t"),
        ptr("pre", "float"),
        ptr("post", "float"),
        ptr("comb", "float"),
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

/// The F32 expand ABI.
pub fn expand_params() -> Vec<EntryParam> {
    vec![
        ptr("x", "const uint16_t"),
        ptr("y", "const uint16_t"),
        ptr("post", "const float"),
        ptr("comb", "const float"),
        ptr("out", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ]
}

/// The sigmoid-weighted head ABI.
pub fn head_params() -> Vec<EntryParam> {
    vec![
        ptr("x", "const uint16_t"),
        ptr("pre", "const float"),
        ptr("out", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
    ]
}

/// The lagged mix-collapse, F32 expand and weighted head of `geometry`
/// (a lagged mixer) under `profile`, with the probe mix-collapse when
/// `probe`. Symbols are `chop_sinkhorn_hc_{tag}_<entry>_v1`.
pub fn entries(profile: Profile, geometry: Geometry, probe: bool) -> Vec<Entry> {
    let Geometry { mix, capacity } = geometry;
    let symbol = |name: &str| format!("chop_sinkhorn_hc_{}_{name}_v1", profile.tag());
    let mut list = vec![
        Entry {
            profile,
            symbol: symbol("lagged_mix_collapse"),
            query: Query::MixCollapse { mix, capacity },
            op: Op::MixCollapse { probe: false },
            params: mix_params(mix.fn_dtype, false),
        },
        Entry {
            profile,
            symbol: symbol("expand_f32"),
            query: Query::ExpandF32 {
                streams: mix.streams,
                width: mix.width,
                capacity,
            },
            op: Op::Expand,
            params: expand_params(),
        },
        Entry {
            profile,
            symbol: symbol("weighted_head"),
            query: Query::Head {
                streams: mix.streams,
                width: mix.width,
                mode: StreamHeadMode::SigmoidWeighted,
                capacity,
            },
            op: Op::Head,
            params: head_params(),
        },
    ];
    if probe {
        list.push(Entry {
            profile,
            symbol: symbol("lagged_mix_collapse_probe"),
            query: Query::MixCollapse { mix, capacity },
            op: Op::MixCollapse { probe: true },
            params: mix_params(mix.fn_dtype, true),
        });
    }
    list
}

/// The lab's lagged mixer: 4 streams of 5120 (the row's streams fill the
/// 40 KiB shared-memory row exactly), 20 iterations, eps 1e-6, norm eps
/// 1e-20, F32 fn, at a capacity of 8192 rows.
pub const LAB: Geometry = Geometry {
    mix: SinkhornMix {
        streams: 4,
        width: 5120,
        iterations: 20,
        eps_bits: 1e-6_f32.to_bits(),
        norm_eps_bits: 1e-20_f32.to_bits(),
        order: SinkhornOrder::RowSoftmaxThenColumnFirst,
        placement: HyperConnectionPlacement::LaggedSinglePass,
        fn_dtype: DType::F32,
    },
    capacity: MAX_ROWS,
};

/// The lab mixer's profile, named by its numbers.
pub const LAB_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "sinkhorn-hc-lagged-4x5120",
    tag: "hcl4x5120",
});

/// The per-sublayer twin of [`LAB`] (the same mixer, the per-sublayer
/// placement), whose probe the lab compares the lagged probe with.
pub const TWIN: Geometry = Geometry {
    mix: SinkhornMix {
        placement: HyperConnectionPlacement::PerSublayer,
        ..LAB.mix
    },
    capacity: MAX_ROWS,
};

/// The twin's profile.
pub const TWIN_PROFILE: Profile = Profile::synthetic(ProfileNames {
    name: "sinkhorn-hc-lagged-twin-4x5120",
    tag: "hclt4x5120",
});

/// The lab mixer's served entries.
pub fn catalog() -> Vec<Entry> {
    entries(LAB_PROFILE, LAB, false)
}

/// The served entries plus the lab's probe mix-collapse.
pub fn lab_catalog() -> Vec<Entry> {
    entries(LAB_PROFILE, LAB, true)
}

/// The twin's per-sublayer probe mix-collapse.
pub fn twin_probe() -> super::Entry {
    super::entries(TWIN_PROFILE, TWIN, true)
        .into_iter()
        .find(|entry| entry.op == super::Op::MixCollapse { probe: true })
        .expect("the per-sublayer catalog holds a probe")
}

/// A cubin of `entries` named `name`.
pub fn cubin_spec(name: String, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/lagged.cuh\"".into()],
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

/// The served cubin's name, `sinkhorn_hc_{tag}_sm120`.
pub fn served_cubin() -> String {
    format!("sinkhorn_hc_{}_sm120", LAB_PROFILE.tag())
}

/// The lagged probe cubin's name, `sinkhorn_hc_{tag}_probe_sm120`.
pub fn probe_cubin() -> String {
    format!("sinkhorn_hc_{}_probe_sm120", LAB_PROFILE.tag())
}

/// The twin probe cubin's name, `sinkhorn_hc_{twin tag}_probe_sm120`.
pub fn twin_cubin() -> String {
    format!("sinkhorn_hc_{}_probe_sm120", TWIN_PROFILE.tag())
}

/// The lab's cubins: the served entries, the lagged probe and the twin's
/// per-sublayer probe.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let probes = lab_catalog()
        .into_iter()
        .filter(|entry| entry.op == Op::MixCollapse { probe: true })
        .collect::<Vec<_>>();
    vec![
        cubin_spec(served_cubin(), &catalog()),
        cubin_spec(probe_cubin(), &probes),
        super::cubin_spec(twin_cubin(), &[twin_probe()]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lab_entries_validate_and_plan_every_bucket() {
        assert_eq!(f32::from_bits(LAB.mix.eps_bits), 1e-6);
        assert_eq!(f32::from_bits(LAB.mix.norm_eps_bits), 1e-20);
        assert_eq!(LAB.mix.flat_width() * 2, STREAM_SHARED_LIMIT);
        let entries = lab_catalog();
        assert_eq!(entries.len(), 4);
        for entry in &entries {
            entry.contract().unwrap();
            assert_eq!(entry.capacity().unwrap(), 8192);
            for rows in [1, 32, 6 * 32, 8192] {
                let plan = entry.plan(rows, rows).unwrap();
                assert!(plan.pdl);
                assert_eq!(plan.block, [256, 1, 1]);
                assert_eq!(plan.dynamic_shared_bytes, 0);
                let expected = match entry.op {
                    Op::MixCollapse { .. } => [rows, 1, 1],
                    _ => [10, rows, 1],
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
            assert!(entry.pack_arguments(vec![]).is_err());
            assert!(!entry.body().is_empty());
            let declared = entry.invariance();
            assert_eq!(
                declared == Invariance::NONE,
                entry.op == Op::MixCollapse { probe: true }
            );
        }
        // The twin is the per-sublayer engine's probe at the same mixer.
        let twin = twin_probe();
        twin.contract().unwrap();
        assert_eq!(
            twin.query,
            Query::MixCollapse {
                mix: TWIN.mix,
                capacity: 8192
            }
        );
    }

    #[test]
    fn the_engine_refuses_other_placements_heads_and_widths() {
        let refused = |query: Query, op: Op| check(query, op).unwrap_err();
        // The per-sublayer queries are the per-sublayer engine's.
        assert_eq!(
            refused(
                Query::MixCollapse {
                    mix: TWIN.mix,
                    capacity: 8
                },
                Op::MixCollapse { probe: false }
            ),
            Refusal("the lagged Sinkhorn mix-collapse takes the lagged placement")
        );
        for (query, op) in [
            (
                Query::Expand {
                    streams: 4,
                    width: 5120,
                    capacity: 8,
                },
                Op::Expand,
            ),
            (
                Query::Head {
                    streams: 4,
                    width: 5120,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity: 8,
                },
                Op::Head,
            ),
            (
                Query::ExpandF32 {
                    streams: 4,
                    width: 5120,
                    capacity: 8,
                },
                Op::Head,
            ),
        ] {
            assert_eq!(
                refused(query, op),
                Refusal("lagged Sinkhorn query and kernel disagree")
            );
        }
        // Streams past the 40 KiB row, odd widths, too many rows or
        // streams.
        let wide = SinkhornMix {
            width: 5248,
            ..LAB.mix
        };
        wide.validate().unwrap();
        assert!(
            check(
                Query::MixCollapse {
                    mix: wide,
                    capacity: 8
                },
                Op::MixCollapse { probe: false }
            )
            .is_err()
        );
        assert!(
            check(
                Query::ExpandF32 {
                    streams: 4,
                    width: 5121,
                    capacity: 8,
                },
                Op::Expand
            )
            .is_err()
        );
        assert!(
            check(
                Query::ExpandF32 {
                    streams: 4,
                    width: 5120,
                    capacity: 8193,
                },
                Op::Expand
            )
            .is_err()
        );
        assert!(
            check(
                Query::Head {
                    streams: 9,
                    width: 5120,
                    mode: StreamHeadMode::SigmoidWeighted,
                    capacity: 8,
                },
                Op::Head
            )
            .is_err()
        );
        // The per-sublayer engine refuses the lagged queries by name.
        assert!(
            super::super::limits(Query::MixCollapse {
                mix: LAB.mix,
                capacity: 8
            })
            .is_err()
        );
    }

    #[test]
    fn lab_cubins_name_their_entries() {
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hcl4x5120_sm120",
                "sinkhorn_hc_hcl4x5120_probe_sm120",
                "sinkhorn_hc_hclt4x5120_probe_sm120",
            ]
        );
        assert_eq!(
            specs.iter().map(|s| s.entries.len()).collect::<Vec<_>>(),
            [3, 1, 1]
        );
    }
}
