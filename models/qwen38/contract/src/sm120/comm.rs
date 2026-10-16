//! Qwen3.8 Flash's hyper-connection all-reduces on the shared transports
//! (`kernels_sm120::comm::hyper_connection` and `hyper_connection_decode`):
//! its shape (the inject-norm epilogue of four 2560-wide streams over BF16
//! partials), its cubins and entries, their declarations, and the
//! transports' plans, states and launches at that shape.

use crate::sm120::config::EXACT_VERIFY_ROWS;
use kernel_build::CubinSpec;
use kernel_core::contract::{
    InstantiationSpec, LaunchPlan, Refusal, SigmoidMeanHyperConnectionQuery,
};
use kernels_sm120::Profile;
use kernels_sm120::comm::hyper_connection::{self as hc, HcTransport};
use kernels_sm120::comm::{Epilogue, input};
use kernels_sm120::invariance::{Declared, Invariance, OpKind};

pub use hc::{
    Buffers, Call, HcShape, Launch, MAX_CAPACITY, ONE_SHOT_ROWS, RANKS, arguments, buffers_with,
    phased, standalone_arguments,
};

pub const SYMBOL: &str = "chop_comm_qwen38_allreduce_hc_v1";
pub const SCATTER_SYMBOL: &str = "chop_comm_qwen38_reduce_scatter_v2";
pub const GATHER_SYMBOL: &str = "chop_comm_qwen38_allgather_hc_v2";

/// The fused all-reduce epilogue: the hyper-connection inject and per-stream
/// Gemma norm of four 2560-wide streams, which every hyper-connection entry,
/// here and in [`decode`], instantiates its transport with.
pub const EPILOGUE: Epilogue = Epilogue {
    header: "comm/epilogue/hc_inject_norm.cuh",
    template: "chop::comm::epilogue::HcInjectNorm",
    width: crate::contract::HIDDEN,
    streams: crate::contract::STREAMS,
};

/// The inject-norm epilogue over BF16 partials.
pub const SHAPE: HcShape = HcShape {
    epilogue: EPILOGUE,
    input: input::BF16,
};

/// The prefill transport: the one-shot entry and the phased pair.
pub const TRANSPORT: HcTransport = HcTransport {
    shape: SHAPE,
    eps_bits: crate::contract::EPS.to_bits(),
    cubin: "comm_qwen38",
    symbol: SYMBOL,
    scatter_symbol: SCATTER_SYMBOL,
    gather_symbol: GATHER_SYMBOL,
};

/// A hyper-connection all-reduce at one rank count.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub query: SigmoidMeanHyperConnectionQuery,
    /// The one-shot and phased entries give every row the same bits at any
    /// row count, so prefill rows ignore the chunk split.
    pub invariance: Invariance,
}

/// The all-reduce at `ranks` ranks with the prefill capacity.
pub fn query(ranks: u32) -> SigmoidMeanHyperConnectionQuery {
    TRANSPORT.query(ranks)
}

/// The all-reduce at every rank count of `RANKS`.
pub fn catalog() -> Vec<Entry> {
    RANKS
        .into_iter()
        .map(|ranks| Entry {
            profile: crate::sm120::config::PROFILE,
            query: query(ranks),
            invariance: Invariance::PREFILL,
        })
        .collect()
}

pub fn contract(query: SigmoidMeanHyperConnectionQuery) -> Result<InstantiationSpec, Refusal> {
    hc::contract(&TRANSPORT, query)
}

pub fn plan(query: SigmoidMeanHyperConnectionQuery, live: u32) -> Result<LaunchPlan, Refusal> {
    hc::plan(&TRANSPORT, query, live)
}

pub fn buffers(query: SigmoidMeanHyperConnectionQuery) -> Result<Buffers, Refusal> {
    hc::buffers(&TRANSPORT, query)
}

pub fn one_shot_capacity(query: SigmoidMeanHyperConnectionQuery) -> Result<u32, Refusal> {
    hc::one_shot_capacity(&TRANSPORT, query)
}

pub fn prefill_plan(
    query: SigmoidMeanHyperConnectionQuery,
    rows: u32,
) -> Result<LaunchPlan, Refusal> {
    hc::prefill_plan(&TRANSPORT, query, rows)
}

pub fn launches(
    query: SigmoidMeanHyperConnectionQuery,
    rows: u32,
    call: Call,
) -> Result<Vec<Launch>, Refusal> {
    hc::launches(&TRANSPORT, query, rows, call)
}

/// Inbox bytes of a one-shot state of its own at `capacity` rows.
pub fn standalone_inbox_bytes(ranks: u32, capacity: u32) -> u64 {
    hc::standalone_inbox_bytes(SHAPE, ranks, capacity)
}

/// The prefill transport's cubin, for Qwen3.8's profiles.
pub fn cubin_specs(profile: Option<Profile>) -> Vec<CubinSpec> {
    super::geometry::MODELS
        .iter()
        .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
        .flat_map(|owner| owned_sources_for(owner.model.at(4), hc::cubin_specs(&owner.transport)))
        .collect()
}

/// The F32 norm-weight form of the prefill transport, for Qwen3.8's
/// profiles: its own cubin and entries, which only an image with F32 norm
/// weights loads. [`cubins`] and the kernel lab keep the BF16 forms alone.
pub fn f32_cubin_specs(profile: Option<Profile>) -> Vec<CubinSpec> {
    super::geometry::MODELS
        .iter()
        .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
        .flat_map(|owner| {
            let transport = owner
                .transport_for(kernel_core::contract::DType::F32)
                .expect("F32 inject-norm epilogue");
            owned_sources_for(owner.model.at(4), hc::cubin_specs(&transport))
        })
        .collect()
}

/// The cubin of `owner`'s decode (or `wide`) all-reduce entries at `ranks`
/// ranks, of the BF16 or `f32` norm-weight form. TP4's carry the base tag
/// and load at every rank count; another rank count's carry its profile's.
pub fn decode_cubin(
    owner: &super::geometry::GeometryModel,
    ranks: u32,
    f32: bool,
    wide: bool,
) -> String {
    if ranks == 4 {
        return match (f32, wide) {
            (false, false) => owner.decode.cubin,
            (false, true) => owner.decode.wide_cubin,
            (true, false) => owner.f32_norms.decode_cubin,
            (true, true) => owner.f32_norms.wide_cubin,
        }
        .into();
    }
    let profile = owner.model.try_at(ranks).expect("a Qwen3.8 rank count");
    format!(
        "comm_{}{}_decode{}",
        profile.tag(),
        if f32 { "_f32" } else { "" },
        if wide { "_wide" } else { "" }
    )
}

/// The F32 norm-weight forms Qwen3.8's bundle embeds beside their BF16
/// forms: each geometry's per-stream grouped norm row, and its all-reduces
/// at two and four ranks, for kernel-lab cases.
pub fn norm_weight_forms() -> kernels_sm120::comm::norm_weights::NormWeightForms {
    use kernel_core::contract::DType;
    use kernels_sm120::comm::norm_weights::{
        Embedded, NormRowForms, NormTransportForms, NormWeightForms,
    };
    let mut forms = NormWeightForms::default();
    for owner in super::geometry::MODELS {
        let tag = owner.model.at(4).tag();
        let names = owner.f32_norms;
        forms.rows.push(NormRowForms {
            row: owner.f32_norm_row().expect("F32 grouped norm row"),
            f32: Embedded {
                cubin: names.row_cubin.into(),
                symbol: names.row_symbol.into(),
            },
            bf16: Embedded {
                cubin: format!("row_{tag}"),
                symbol: format!("chop_row_{tag}_hc_norm_v1"),
            },
        });
        for ranks in [2, 4] {
            let dtypes = [DType::Bf16, DType::F32];
            forms.transports.push(NormTransportForms {
                ranks,
                prefill: dtypes.map(|d| owner.transport_for(d).expect("inject-norm epilogue")),
                decode: dtypes.map(|d| owner.decode_for(d).expect("inject-norm epilogue")),
                decode_cubins: [false, true].map(|f32| decode_cubin(&owner, ranks, f32, false)),
                wide_cubins: [false, true].map(|f32| decode_cubin(&owner, ranks, f32, true)),
            });
        }
    }
    forms
}

/// The prefill transport's cubin instantiated with another `shape`.
pub fn cubin_specs_with(shape: HcShape) -> Vec<CubinSpec> {
    owned_sources(hc::cubin_specs(&HcTransport { shape, ..TRANSPORT }))
}

/// Every comm cubin: the prefill transport's, the decode entries' and the
/// wide entries', at every rank count.
pub fn cubins() -> Vec<CubinSpec> {
    let mut cubins = cubin_specs(None);
    cubins.extend(decode::cubin_specs(None));
    cubins.extend(decode::wide_cubin_specs(None));
    cubins
}

/// The comm entries' declarations, in their former order:
///
/// - The one-shot and phased hyper-connection all-reduces give the same
///   bits at every row count.
/// - The decode all-reduces (`decode::SYMBOLS_V2` and
///   `decode::SYMBOLS_V3`), which the
///   MTP K3 verify steps launch, give a row of a call of up to
///   `EXACT_VERIFY_ROWS` rows the bits of a one-row call: both protocols,
///   low latency up to `decode::LL_MAX_ROWS` prepared rows and the one-shot
///   above, sum a column's ranks in rank order from zero and finish each
///   (row, stream) group in a CTA of its own. The eight-rank entry runs
///   only on a host of eight distinct devices: elsewhere kernel-lab reports
///   its claims as not qualified on that host, not as tested.
/// - The all-reduce of each rank count (`catalog`), in Qwen3.8's profile.
pub fn declared() -> Vec<Declared> {
    super::geometry::MODELS
        .iter()
        .flat_map(declared_for)
        .collect()
}

fn declared_for(owner: &super::geometry::GeometryModel) -> Vec<Declared> {
    let declare = |symbol: &str, profile, invariance| Declared {
        symbol: symbol.to_owned(),
        profile,
        op: OpKind::Comm,
        invariance,
    };
    let mut per_rank = vec![
        (owner.transport.symbol, Invariance::PREFILL),
        (owner.transport.scatter_symbol, Invariance::PREFILL),
        (owner.transport.gather_symbol, Invariance::PREFILL),
    ];
    for symbol in owner.decode.symbols_v2 {
        per_rank.push((symbol, Invariance::one_row(EXACT_VERIFY_ROWS)));
    }
    if owner.decode.symbols_v3 != owner.decode.symbols_v2 {
        for symbol in owner.decode.symbols_v3 {
            per_rank.push((symbol, Invariance::one_row(EXACT_VERIFY_ROWS)));
        }
    }
    let mut declared = per_rank
        .iter()
        .map(|&(symbol, invariance)| declare(symbol, None, invariance))
        .collect::<Vec<_>>();
    for _ in RANKS {
        declared.push(declare(
            owner.transport.symbol,
            Some(owner.model.at(4)),
            Invariance::PREFILL,
        ));
    }
    declared
}

/// The decode transport: an entry per rank count, its whole-row LL form,
/// and the wide entries of graph buckets above 32 rows.
pub mod decode {
    use super::*;
    use kernels_sm120::comm::hyper_connection_decode::{self as hcd, HcDecodeTransport};

    pub use hcd::{
        Buffers, Call, LL_MAX_ROWS, MAX_ROWS, WIDE_MAX_ROWS, WideProtocol, arguments, buffers_with,
        defines_with, graph_takes_wide, low_latency_rows, plan_with, wide_arguments,
        wide_buffers_with, wide_defines_with, wide_plan_with, wide_protocol,
    };

    /// One entry per rank count of `RANKS`, in its order.
    pub const SYMBOLS_V2: [&str; 3] = [
        "chop_comm_qwen38_allreduce_hc_decode_r2_v2",
        "chop_comm_qwen38_allreduce_hc_decode_r4_v2",
        "chop_comm_qwen38_allreduce_hc_decode_r8_v2",
    ];
    /// The whole-row LL inbox readers, with v2's one-shot and output bits.
    pub const SYMBOLS_V3: [&str; 3] = [
        "chop_comm_qwen38_allreduce_hc_decode_r2_v3",
        "chop_comm_qwen38_allreduce_hc_decode_r4_v3",
        "chop_comm_qwen38_allreduce_hc_decode_r8_v3",
    ];
    /// The wide entries, in `RANKS` order.
    pub const WIDE_SYMBOLS: [&str; 3] = [
        "chop_comm_qwen38_allreduce_hc_wide_r2_v1",
        "chop_comm_qwen38_allreduce_hc_wide_r4_v1",
        "chop_comm_qwen38_allreduce_hc_wide_r8_v1",
    ];

    pub const TRANSPORT: HcDecodeTransport = HcDecodeTransport {
        shape: SHAPE,
        eps_bits: crate::contract::EPS.to_bits(),
        cubin: "comm_qwen38_decode",
        wide_cubin: "comm_qwen38_decode_wide",
        symbols_v2: SYMBOLS_V2,
        symbols_v3: SYMBOLS_V3,
        wide_symbols: WIDE_SYMBOLS,
    };

    pub fn symbol_v2(ranks: u32) -> Result<&'static str, Refusal> {
        hcd::symbol_v2(&TRANSPORT, ranks)
    }

    pub fn symbol_v3(ranks: u32) -> Result<&'static str, Refusal> {
        hcd::symbol_v3(&TRANSPORT, ranks)
    }

    pub fn wide_symbol(ranks: u32) -> Result<&'static str, Refusal> {
        hcd::wide_symbol(&TRANSPORT, ranks)
    }

    pub fn query(ranks: u32) -> SigmoidMeanHyperConnectionQuery {
        hcd::query(&TRANSPORT, ranks)
    }

    pub fn contract(ranks: u32) -> Result<InstantiationSpec, Refusal> {
        hcd::contract(&TRANSPORT, ranks)
    }

    pub fn wide_query(ranks: u32) -> SigmoidMeanHyperConnectionQuery {
        hcd::wide_query(&TRANSPORT, ranks)
    }

    pub fn wide_contract(ranks: u32) -> Result<InstantiationSpec, Refusal> {
        hcd::wide_contract(&TRANSPORT, ranks)
    }

    /// Geometry for `rows` prepared rows (the graph bucket).
    pub fn plan(rows: u32) -> Result<LaunchPlan, Refusal> {
        plan_with(SHAPE, rows)
    }

    /// The wide entries' geometry for `rows` prepared rows.
    pub fn wide_plan(rows: u32) -> Result<LaunchPlan, Refusal> {
        wide_plan_with(SHAPE, rows)
    }

    /// The decode state at `ranks` ranks.
    pub fn buffers(ranks: u32) -> Result<Buffers, Refusal> {
        contract(ranks)?;
        buffers_with(SHAPE, ranks)
    }

    /// A decode state that also takes wide calls.
    pub fn wide_buffers(ranks: u32) -> Result<Buffers, Refusal> {
        contract(ranks)?;
        wide_buffers_with(SHAPE, ranks)
    }

    pub fn defines() -> Vec<(String, String)> {
        defines_with(SHAPE)
    }

    /// The decode cubin with the entries of every rank count.
    pub fn cubin_specs(profile: Option<Profile>) -> Vec<CubinSpec> {
        cubin_specs_for(profile, &RANKS)
    }

    /// The decode cubin with the entries of `ranks` only, for a bundle
    /// built for one topology.
    pub fn cubin_specs_for(profile: Option<Profile>, ranks: &[u32]) -> Vec<CubinSpec> {
        super::super::geometry::MODELS
            .iter()
            .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
            .flat_map(|owner| {
                super::owned_sources_for(owner.model.at(4), hcd::cubin_specs(&owner.decode, ranks))
            })
            .collect()
    }

    /// The decode entries of `ranks` instantiated with another `shape`.
    pub fn cubin_specs_with(shape: HcShape, ranks: &[u32]) -> Vec<CubinSpec> {
        super::owned_sources(hcd::cubin_specs(
            &HcDecodeTransport { shape, ..TRANSPORT },
            ranks,
        ))
    }

    /// The wide cubin with the entries of every rank count.
    pub fn wide_cubin_specs(profile: Option<Profile>) -> Vec<CubinSpec> {
        wide_cubin_specs_for(profile, &RANKS)
    }

    pub fn wide_cubin_specs_for(profile: Option<Profile>, ranks: &[u32]) -> Vec<CubinSpec> {
        super::super::geometry::MODELS
            .iter()
            .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
            .flat_map(|owner| {
                super::owned_sources_for(
                    owner.model.at(4),
                    hcd::wide_cubin_specs(&owner.decode, ranks),
                )
            })
            .collect()
    }

    /// The F32 norm-weight forms of the decode entries of `ranks`.
    pub fn f32_cubin_specs_for(profile: Option<Profile>, ranks: &[u32]) -> Vec<CubinSpec> {
        super::super::geometry::MODELS
            .iter()
            .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
            .flat_map(|owner| {
                let decode = owner
                    .decode_for(kernel_core::contract::DType::F32)
                    .expect("F32 inject-norm epilogue");
                super::owned_sources_for(owner.model.at(4), hcd::cubin_specs(&decode, ranks))
            })
            .collect()
    }

    /// The F32 norm-weight forms of the wide entries of `ranks`.
    pub fn f32_wide_cubin_specs_for(profile: Option<Profile>, ranks: &[u32]) -> Vec<CubinSpec> {
        super::super::geometry::MODELS
            .iter()
            .filter(|owner| profile.is_none_or(|p| p.ranks_of(owner.model).is_some()))
            .flat_map(|owner| {
                let decode = owner
                    .decode_for(kernel_core::contract::DType::F32)
                    .expect("F32 inject-norm epilogue");
                super::owned_sources_for(owner.model.at(4), hcd::wide_cubin_specs(&decode, ranks))
            })
            .collect()
    }

    /// The wide entries of `ranks` instantiated with another `shape`.
    pub fn wide_cubin_specs_with(shape: HcShape, ranks: &[u32]) -> Vec<CubinSpec> {
        super::owned_sources(hcd::wide_cubin_specs(
            &HcDecodeTransport { shape, ..TRANSPORT },
            ranks,
        ))
    }
}

/// Bind the model-owned epilogue beside the shared transport engines.
fn owned_sources(specs: Vec<CubinSpec>) -> Vec<CubinSpec> {
    owned_sources_for(super::config::PROFILE, specs)
}

fn owned_sources_for(profile: Profile, mut specs: Vec<CubinSpec>) -> Vec<CubinSpec> {
    for spec in &mut specs {
        super::gemm::sources_for(profile, spec);
    }
    specs
}

#[cfg(test)]
mod tests {
    #[test]
    fn epilogue_names_its_owned_header_and_instance() {
        let hc = super::EPILOGUE;
        assert_eq!(hc.functor(), "chop::comm::epilogue::HcInjectNorm<2560, 4>");
        assert_eq!(
            hc.include(),
            "#include \"comm/epilogue/hc_inject_norm.cuh\""
        );
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../bundle/cuda");
        assert!(root.join(hc.header).is_file(), "{hc:?}");
    }

    use super::*;

    /// Each declared BF16 norm row is the catalog's grouped norm entry with
    /// the F32 row's groups and arguments.
    #[test]
    fn norm_weight_forms_pair_the_catalog_rows() {
        let forms = norm_weight_forms();
        assert_eq!(forms.rows.len(), super::super::geometry::MODELS.len());
        assert_eq!(forms.transports.len(), 2 * forms.rows.len());
        let rows = super::super::row::hyper_connection_rows();
        for form in &forms.rows {
            let bf16 = rows
                .iter()
                .find(|entry| entry.symbol == form.bf16.symbol)
                .unwrap();
            assert_eq!(form.bf16.cubin, format!("row_{}", bf16.profile.tag()));
            assert!(matches!(
                bf16.query,
                kernel_core::contract::StaticQuery::HyperConnection(
                    kernel_core::contract::SigmoidMeanHyperConnectionQuery::GroupedNorm {
                        streams,
                        width,
                        ..
                    }
                ) if streams == form.row.groups && width == form.row.group_width
            ));
            assert_eq!(
                bf16.plan(8, 5).unwrap(),
                form.row.plan(8, 5).unwrap(),
                "{}",
                form.f32.symbol
            );
        }
        for form in &forms.transports {
            assert_eq!(
                form.prefill[0].shape.epilogue.weight_pointer_type(),
                "const uint16_t"
            );
            assert_eq!(
                form.prefill[1].shape.epilogue.weight_pointer_type(),
                "const float"
            );
            assert_eq!(form.decode[1].shape, form.prefill[1].shape);
        }
    }

    /// Each profile's rank count is a transport rank count.
    #[test]
    fn rank_counts_match_the_profiles() {
        for profile in crate::sm120::config::MODEL.profiles() {
            assert!(RANKS.contains(&profile.ranks().unwrap()));
        }
    }
}
