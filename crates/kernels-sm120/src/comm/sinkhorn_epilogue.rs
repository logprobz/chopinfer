//! The Sinkhorn mHC comm epilogue
//! (`cuda/comm/epilogue/sinkhorn_expand_mix_norm.cuh`): after the all-reduce
//! of a sublayer's F32 rank partials, the rounding of the sum, the Sinkhorn
//! expand into the streams, the next sublayer's mix-collapse and its input
//! RMSNorm, one 256-thread CTA per row of the fused width
//! (`super::FUSED_WIDTH`, 4096), bitwise equal to the unfused launches
//! (`kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery::AllReduceExpandMixNorm`).
//!
//! Two transports run it, rank-generic up to `MAX_COMM_RANKS`:
//! [`Transport::OneShot`] is the F32 one-shot all-reduce
//! (`chop_comm_allreduce_oneshot_f32_v1`'s protocol and state, one tile per
//! row) with the epilogue on each reduced row, and [`Transport::Rows`] runs
//! the epilogue alone over rows the phased reduce-scatter and all-gather
//! summed. Both read the live row count on the device; the one-shot sums
//! every row of its bucket, as the unfused entry does.

use std::path::PathBuf;

use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    MAX_COMM_RANKS, NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

/// The functor's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/epilogue/sinkhorn_expand_mix_norm.cuh";
/// Threads of every entry: one CTA per row.
pub const THREADS: u32 = 256;
/// Most rows of one launch: a prefill chunk.
pub const MAX_ROWS: u32 = 8192;

/// How an entry's rows reach the epilogue.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Transport {
    /// The F32 one-shot all-reduce, then the epilogue on each reduced row.
    OneShot,
    /// The epilogue alone over F32 sums another all-reduce wrote.
    Rows,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub symbol: String,
    pub mix: SinkhornMix,
    /// The next sublayer's input RMSNorm eps.
    pub rmsnorm_eps_bits: u32,
    pub capacity: u32,
    pub transport: Transport,
    pub params: Vec<EntryParam>,
}

impl Entry {
    /// The contract of a call over `ranks` ranks: its input norm sums its
    /// squares in column order (one thread, `sinkhorn_expand_mix_norm.cuh`).
    pub fn query(&self, ranks: u32) -> Query {
        Query::AllReduceExpandMixNorm {
            ranks,
            dtype: DType::F32,
            mix: self.mix,
            rmsnorm_eps_bits: self.rmsnorm_eps_bits,
            norm: kernel_core::rmsnorm::RmsNormReduction::Ordered,
            capacity: self.capacity,
        }
    }

    /// The engine's refusals on top of the contract's, at `ranks` ranks.
    fn check(&self, ranks: u32) -> Result<(), Refusal> {
        crate::sinkhorn_hc::limits(self.query(ranks))?;
        if self.mix.width != super::FUSED_WIDTH {
            return Err(Refusal(
                "the Sinkhorn comm epilogue runs at the fused width only",
            ));
        }
        if self.mix.flat_width() * 2 > crate::sinkhorn_hc::STREAM_SHARED_LIMIT {
            return Err(Refusal(
                "Sinkhorn mixer streams exceed the 40 KiB shared-memory row",
            ));
        }
        if self.capacity > MAX_ROWS {
            return Err(Refusal("Sinkhorn comm epilogue capacity exceeds 8192"));
        }
        Ok(())
    }

    /// The instantiation, compiled for every rank count up to
    /// `MAX_COMM_RANKS` (its pointer arrays have that many slots).
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let ranks = MAX_COMM_RANKS as u32;
        self.check(ranks)?;
        let spec = InstantiationSpec {
            family: OpFamily::Comm,
            key: CompileKey {
                query: StaticQuery::SinkhornHyperConnection(self.query(ranks)),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: crate::sinkhorn_hc::CHAIN as u16,
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
        self.query(ranks).numerical_contract_id()?;
        Ok(spec)
    }

    /// The plan of a bucket of `rows` rows over `ranks` ranks: one CTA per
    /// row (the one-shot's tile), the grid fixed by the bucket.
    pub fn plan(&self, rows: u32, ranks: u32) -> Result<LaunchPlan, Refusal> {
        self.check(ranks)?;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal(
                "Sinkhorn comm epilogue bucket exceeds the entry capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [rows, 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn body(&self) -> String {
        let mix = self.mix;
        let mixer = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            mix.streams,
            mix.width,
            mix.iterations,
            mix.eps_bits,
            mix.norm_eps_bits,
            mix.fn_dtype == DType::F32
        );
        let epilogue = format!(
            "const chop::comm::epilogue::SinkhornExpandMixNorm<{mixer}> epilogue{{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,live_rows,__uint_as_float(eps_bits)}};"
        );
        let call = match self.transport {
            Transport::OneShot => {
                "chop::comm::epilogue::allreduce_oneshot_sinkhorn(input,output,inboxes,flags,epoch,elements,rank,ranks,epilogue);"
            }
            Transport::Rows => {
                "chop::comm::epilogue::sinkhorn_expand_mix_norm_rows(summed,epilogue);"
            }
        };
        format!("{epilogue}{call}")
    }

    fn spec(&self) -> EntrySpec {
        EntrySpec::new(
            self.symbol.clone(),
            self.params.clone(),
            self.body(),
            THREADS,
            1,
        )
    }
}

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

/// The epilogue's parameters, after the transport's.
pub fn epilogue_params(fn_dtype: DType) -> Vec<EntryParam> {
    vec![
        ptr("streams", "const uint16_t"),
        ptr("post", "uint16_t"),
        ptr("comb", "uint16_t"),
        ptr("reduced", "uint16_t"),
        ptr("expanded", "uint16_t"),
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
        ptr("weight", "const uint16_t"),
        ptr("normed", "uint16_t"),
        ptr("live_rows", "const uint32_t"),
        EntryParam::U32 {
            name: "eps_bits".into(),
        },
    ]
}

/// The one-shot entry's ABI: `chop_comm_allreduce_oneshot_f32_v1`'s, then
/// the epilogue's.
pub fn one_shot_params(fn_dtype: DType) -> Vec<EntryParam> {
    let array = |name: &str, cxx_type: &str| EntryParam::PointerArray {
        name: name.into(),
        slots: MAX_COMM_RANKS as u8,
        cxx_type: cxx_type.into(),
    };
    let mut params = vec![
        ptr("input", "const float"),
        ptr("output", "float"),
        array("inboxes", "float"),
        array("flags", "unsigned int"),
        ptr("epoch", "unsigned int"),
        EntryParam::U64 {
            name: "elements".into(),
        },
        EntryParam::U32 {
            name: "rank".into(),
        },
        EntryParam::U32 {
            name: "ranks".into(),
        },
    ];
    params.extend(epilogue_params(fn_dtype));
    params
}

/// The rows entry's ABI: the F32 sums, then the epilogue's.
pub fn rows_params(fn_dtype: DType) -> Vec<EntryParam> {
    let mut params = vec![ptr("summed", "const float")];
    params.extend(epilogue_params(fn_dtype));
    params
}

/// The one-shot and rows entries of `mix` with the input norm's eps under
/// `profile`: `chop_comm_sinkhorn_{tag}_oneshot_v1`, a collective, and
/// `chop_sinkhorn_epilogue_{tag}_rows_v1`, which waits for no peer and so
/// does not carry the collectives' `chop_comm_` prefix.
pub fn entries(
    profile: Profile,
    mix: SinkhornMix,
    rmsnorm_eps_bits: u32,
    capacity: u32,
) -> Result<Vec<Entry>, Refusal> {
    let tag = profile.tag();
    let entries = vec![
        Entry {
            symbol: format!("chop_comm_sinkhorn_{tag}_oneshot_v1"),
            mix,
            rmsnorm_eps_bits,
            capacity,
            transport: Transport::OneShot,
            params: one_shot_params(mix.fn_dtype),
        },
        Entry {
            symbol: format!("chop_sinkhorn_epilogue_{tag}_rows_v1"),
            mix,
            rmsnorm_eps_bits,
            capacity,
            transport: Transport::Rows,
            params: rows_params(mix.fn_dtype),
        },
    ];
    for entry in &entries {
        entry.contract()?;
    }
    Ok(entries)
}

/// A cubin of `entries` named `name`.
pub fn cubin_spec(name: String, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: super::push_defines(),
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// The served mixer's entries (`crate::sinkhorn_hc::SERVED`) under the
/// Sinkhorn engine's served profile, with the input norm's eps 1e-5.
pub fn catalog() -> Vec<Entry> {
    let served = crate::sinkhorn_hc::SERVED;
    entries(
        crate::sinkhorn_hc::SERVED_PROFILE,
        served.mix,
        1e-5_f32.to_bits(),
        served.capacity,
    )
    .expect("the served Sinkhorn comm epilogue")
}

/// The lab's reference of the unfused path's input norm after the
/// mix-collapse: row.cuh's HF-ordered RMSNorm of the fused width (eps is an
/// argument), and its symbol.
pub fn reference_norm() -> (crate::row::rmsnorm::RmsNorm, String) {
    use crate::row::rmsnorm::{RmsNorm, RmsNormTarget};
    use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
    let requirement = RmsNormRequirement::bf16(
        super::FUSED_WIDTH,
        1e-5,
        kernel_core::contract::RmsNormKind::HfBf16Intermediate,
        false,
        RmsNormNumerics::HF_ORDERED,
    );
    let norm = RmsNorm::new(requirement, MAX_ROWS, RmsNormTarget::SM120F)
        .expect("the HF-ordered RMSNorm of the fused width");
    let symbol = format!(
        "chop_row_rmsnorm_{}_w{}_v1",
        crate::sinkhorn_hc::SERVED_PROFILE.tag(),
        super::FUSED_WIDTH
    );
    (norm, symbol)
}

/// The lab's cubins of [`catalog`]: `comm_sinkhorn_{tag}_sm120` with the
/// one-shot entry alone (a collective cubin, every entry `chop_comm_`),
/// `sinkhorn_epilogue_{tag}_sm120` with the rows entry and
/// [`reference_norm`], and `comm_sinkhorn_late_wait_{tag}_sm120`, the
/// one-shot entry with its dependency wait moved after its input reads
/// (`CHOP_COMM_TEST_LATE_WAIT`, symbol prefix [`LATE_WAIT_PREFIX`]), a
/// mutant the lab's checks must catch.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = crate::sinkhorn_hc::SERVED_PROFILE.tag();
    let (one_shot, rows): (Vec<_>, Vec<_>) = catalog()
        .into_iter()
        .partition(|entry| entry.transport == Transport::OneShot);
    let mut rows = cubin_spec(format!("sinkhorn_epilogue_{tag}_sm120"), &rows);
    let (norm, symbol) = reference_norm();
    rows.headers.push("#include \"row/row.cuh\"".into());
    rows.entries.push(norm.entry(symbol));
    let mut late_wait = cubin_spec(format!("comm_sinkhorn_late_wait_{tag}_sm120"), &one_shot);
    for entry in &mut late_wait.entries {
        entry.symbol = entry.symbol.replacen("chop_comm_", LATE_WAIT_PREFIX, 1);
    }
    late_wait
        .defines
        .push(("CHOP_COMM_TEST_LATE_WAIT".into(), "1".into()));
    vec![
        cubin_spec(format!("comm_sinkhorn_{tag}_sm120"), &one_shot),
        rows,
        late_wait,
    ]
}

/// The symbol prefix of the lab's late-wait mutant, in place of `chop_comm_`.
pub const LATE_WAIT_PREFIX: &str = "chop_comm_late_wait_";

/// The epilogue's buffers of one call, as handles the caller resolves.
#[derive(Clone, Copy, Debug)]
pub struct EpilogueBuffers {
    pub streams: u64,
    pub post: u64,
    pub comb: u64,
    pub reduced: u64,
    pub expanded: u64,
    pub fn_: u64,
    pub base: u64,
    pub scale: u64,
    pub collapsed: u64,
    pub weight: u64,
    pub normed: u64,
    pub live_rows: u64,
    pub eps_bits: u32,
}

impl EpilogueBuffers {
    fn arguments(self) -> Vec<Argument> {
        let mut arguments = [
            self.streams,
            self.post,
            self.comb,
            self.reduced,
            self.expanded,
            self.fn_,
            self.base,
            self.scale,
            self.collapsed,
            self.weight,
            self.normed,
            self.live_rows,
        ]
        .map(Argument::DeviceBuffer)
        .to_vec();
        arguments.push(Argument::U32(self.eps_bits));
        arguments
    }
}

/// The one-shot entry's arguments: `super::arguments(call)`, then the
/// epilogue's. `call` must reduce whole rows of the fused width.
pub fn one_shot_arguments(
    call: super::OneShotCall,
    epilogue: EpilogueBuffers,
) -> Result<ArgumentList, Refusal> {
    if call.elements == 0 || !call.elements.is_multiple_of(u64::from(super::FUSED_WIDTH)) {
        return Err(Refusal(
            "the Sinkhorn comm epilogue reduces whole 4096-wide rows",
        ));
    }
    let mut args = super::arguments(call)?;
    args.0.extend(epilogue.arguments());
    args.validate()?;
    Ok(args)
}

/// The rows entry's arguments.
pub fn rows_arguments(summed: u64, epilogue: EpilogueBuffers) -> Result<ArgumentList, Refusal> {
    let mut args = vec![Argument::DeviceBuffer(summed)];
    args.extend(epilogue.arguments());
    let args = ArgumentList(args);
    args.validate()?;
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ProfileNames, sinkhorn_hc};

    const PROFILE: Profile = Profile::synthetic(ProfileNames {
        name: "sinkhorn-comm-test",
        tag: "test",
    });

    fn served() -> Vec<Entry> {
        entries(
            PROFILE,
            sinkhorn_hc::SERVED.mix,
            1e-5_f32.to_bits(),
            MAX_ROWS,
        )
        .unwrap()
    }

    #[test]
    fn entries_plan_every_bucket_at_every_rank_count() {
        let entries = served();
        assert_eq!(
            entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                "chop_comm_sinkhorn_test_oneshot_v1",
                "chop_sinkhorn_epilogue_test_rows_v1"
            ]
        );
        for entry in &entries {
            for ranks in 1..=MAX_COMM_RANKS as u32 {
                for rows in [1, 32, 6 * 32, MAX_ROWS] {
                    let plan = entry.plan(rows, ranks).unwrap();
                    assert_eq!(plan.grid, [rows, 1, 1]);
                    assert_eq!(plan.block, [THREADS, 1, 1]);
                    assert!(plan.pdl);
                }
            }
            assert!(entry.plan(0, 4).is_err());
            assert!(entry.plan(MAX_ROWS + 1, 4).is_err());
            assert!(entry.plan(1, 0).is_err());
            assert!(entry.plan(1, MAX_COMM_RANKS as u32 + 1).is_err());
        }
        assert_eq!(
            entries[0].body(),
            "const chop::comm::epilogue::SinkhornExpandMixNorm<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>> epilogue{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,live_rows,__uint_as_float(eps_bits)};chop::comm::epilogue::allreduce_oneshot_sinkhorn(input,output,inboxes,flags,epoch,elements,rank,ranks,epilogue);"
        );
        let spec = cubin_spec("sinkhorn_comm_test".into(), &entries);
        assert!(spec.sources_root.join(HEADER).is_file());
        assert_eq!(spec.entries.len(), 2);
        assert!(
            spec.defines
                .contains(&("CHOP_COMM_MAX_RANKS".into(), "8".into()))
        );
    }

    #[test]
    fn the_lab_cubins_split_the_collective_from_the_rows() {
        let specs = lab_cubin_specs();
        let names = |spec: &CubinSpec| {
            spec.entries
                .iter()
                .map(|entry| entry.symbol.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(specs[0].name, "comm_sinkhorn_hc4x4096_sm120");
        assert_eq!(names(&specs[0]), ["chop_comm_sinkhorn_hc4x4096_oneshot_v1"]);
        assert_eq!(specs[1].name, "sinkhorn_epilogue_hc4x4096_sm120");
        assert_eq!(
            names(&specs[1]),
            [
                "chop_sinkhorn_epilogue_hc4x4096_rows_v1",
                "chop_row_rmsnorm_hc4x4096_w4096_v1"
            ]
        );
        assert!(specs[1].entries[1].body.contains("rmsnorm<4096,0,3,false>"));
        assert_eq!(specs[2].name, "comm_sinkhorn_late_wait_hc4x4096_sm120");
        assert_eq!(
            names(&specs[2]),
            ["chop_comm_late_wait_sinkhorn_hc4x4096_oneshot_v1"]
        );
        assert_eq!(specs[2].entries[0].body, specs[0].entries[0].body);
        let mut defines = specs[2].defines.clone();
        assert_eq!(
            defines.pop(),
            Some(("CHOP_COMM_TEST_LATE_WAIT".into(), "1".into()))
        );
        assert_eq!(defines, specs[0].defines);
        assert!(
            !specs[0]
                .defines
                .iter()
                .any(|(name, _)| name == "CHOP_COMM_TEST_LATE_WAIT")
        );
    }

    #[test]
    fn other_widths_and_capacities_refuse_by_name() {
        let wide = SinkhornMix {
            width: 5120,
            ..sinkhorn_hc::SERVED.mix
        };
        assert_eq!(
            entries(PROFILE, wide, 1e-5_f32.to_bits(), 8).unwrap_err(),
            Refusal("the Sinkhorn comm epilogue runs at the fused width only")
        );
        assert_eq!(
            entries(
                PROFILE,
                sinkhorn_hc::SERVED.mix,
                1e-5_f32.to_bits(),
                MAX_ROWS + 1
            )
            .unwrap_err(),
            Refusal("Sinkhorn comm epilogue capacity exceeds 8192")
        );
    }

    #[test]
    fn arguments_follow_the_entry_abi() {
        let entries = served();
        let epilogue = EpilogueBuffers {
            streams: 10,
            post: 11,
            comb: 12,
            reduced: 13,
            expanded: 14,
            fn_: 15,
            base: 16,
            scale: 17,
            collapsed: 18,
            weight: 19,
            normed: 20,
            live_rows: 21,
            eps_bits: 1e-5_f32.to_bits(),
        };
        let call = |elements| super::super::OneShotCall {
            input: 1,
            output: 2,
            peer_inboxes: vec![3, 4, 5, 6],
            peer_flags: vec![7, 8, 9, 22],
            epoch: 23,
            elements,
            rank: 1,
            ranks: 4,
        };
        let one_shot = one_shot_arguments(call(2 * 4096), epilogue).unwrap();
        assert_eq!(one_shot.0.len(), entries[0].params.len());
        assert!(one_shot_arguments(call(4096 + 8), epilogue).is_err());
        assert!(one_shot_arguments(call(0), epilogue).is_err());
        let rows = rows_arguments(2, epilogue).unwrap();
        assert_eq!(rows.0.len(), entries[1].params.len());
        assert_eq!(rows.0.last(), Some(&Argument::U32(1e-5_f32.to_bits())));
    }
}
