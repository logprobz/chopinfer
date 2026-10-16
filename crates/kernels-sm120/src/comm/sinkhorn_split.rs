//! The Sinkhorn mHC comm epilogue split over CTAs
//! (`cuda/comm/epilogue/sinkhorn_split.cuh`): bitwise
//! [`super::sinkhorn_epilogue`]'s rows entry, with a 4096-wide row spread
//! over [`SLICES`] CTAs of [`SLICE_COLUMNS`] columns. Each CTA rounds and
//! expands its columns of the F32 sums another all-reduce wrote (the
//! one-shot or the phased pair) and computes the next mixer's 128-K chains
//! over them; the row's last CTA (elected by a ticket) adds the chains in K
//! order and runs the rest of the mix-collapse and the input RMSNorm. No
//! CTA waits on another, so the entry runs in a launch of its own after the
//! all-reduce and needs no co-residency of its grid.

use std::path::PathBuf;

use super::FUSED_WIDTH;
use super::sinkhorn_epilogue::{EpilogueBuffers, epilogue_params};
use crate::Profile;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    MAX_COMM_RANKS, NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::RmsNormReduction;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/epilogue/sinkhorn_split.cuh";
/// Threads of the entry.
pub const THREADS: u32 = 256;
/// Columns of one CTA's slice of a row.
pub const SLICE_COLUMNS: u32 = 256;
/// CTAs per 4096-wide row.
pub const SLICES: u32 = FUSED_WIDTH / SLICE_COLUMNS;
/// Most rows of one launch: a prefill chunk.
pub const MAX_ROWS: u32 = 8192;

fn ptr(name: &str, cxx_type: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    }
}

/// The entry's parameters: the F32 sums, then the one-CTA epilogue's, with
/// the chain sums and tickets before the live row count.
pub fn params(fn_dtype: DType) -> Vec<EntryParam> {
    let mut params = vec![ptr("summed", "const float")];
    params.extend(epilogue_params(fn_dtype));
    let at = params
        .iter()
        .position(|p| matches!(p, EntryParam::DevicePointer { name, .. } if name == "live_rows"))
        .expect("the live row count");
    params.splice(
        at..at,
        [ptr("partials", "float"), ptr("tickets", "uint32_t")],
    );
    params
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub symbol: String,
    pub mix: SinkhornMix,
    /// The next sublayer's input RMSNorm eps.
    pub rmsnorm_eps_bits: u32,
    /// The order the entry's input RMSNorm sums its squares in: column
    /// order ([`entry`], `sinkhorn_split.cuh`) or the 256-lane tree
    /// (`super::sinkhorn_split_tree::entry`), which its query states.
    pub norm: RmsNormReduction,
    pub capacity: u32,
    pub params: Vec<EntryParam>,
}

impl Entry {
    /// What every launch of the entry keeps: one-row bits up to its
    /// capacity, any chunk split and row independence. Row `r` is grid row
    /// `r` alone: each of its slices reads only that row's sums, streams,
    /// `post` and `comb`, its chains go to the row's own partials, and the
    /// slice that takes the row's own ticket adds them in K order and
    /// finishes the row (the ticket elects which CTA does, not what it
    /// computes); rows at or past the live count return.
    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.capacity)
    }

    /// The contract of the epilogue after an all-reduce over `ranks` ranks.
    pub fn query(&self, ranks: u32) -> Query {
        Query::AllReduceExpandMixNorm {
            ranks,
            dtype: DType::F32,
            mix: self.mix,
            rmsnorm_eps_bits: self.rmsnorm_eps_bits,
            norm: self.norm,
            capacity: self.capacity,
        }
    }

    fn check(&self, ranks: u32) -> Result<(), Refusal> {
        crate::sinkhorn_hc::limits(self.query(ranks))?;
        if self.mix.width != FUSED_WIDTH {
            return Err(Refusal(
                "the Sinkhorn comm epilogue runs at the fused width only",
            ));
        }
        // A slice holds one 128-K chain per warp: two chain blocks of every
        // stream.
        if self.mix.streams * SLICE_COLUMNS / crate::sinkhorn_hc::CHAIN
            != crate::sinkhorn_hc::split::CHAINS_PER_CTA
        {
            return Err(Refusal(
                "the split Sinkhorn comm epilogue takes 4 streams (one chain per warp)",
            ));
        }
        crate::sinkhorn_hc::split::fits(self.mix)?;
        if self.capacity > MAX_ROWS {
            return Err(Refusal("Sinkhorn comm epilogue capacity exceeds 8192"));
        }
        Ok(())
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let ranks = MAX_COMM_RANKS as u32;
        self.check(ranks)?;
        let spec = InstantiationSpec {
            family: OpFamily::Comm,
            key: CompileKey {
                query: StaticQuery::SinkhornHyperConnection(self.query(ranks)),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: SLICE_COLUMNS as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (THREADS / 32) as u8,
                    split_k_capacity: SLICES as u8,
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

    /// The plan of a bucket of `rows` rows after an all-reduce over `ranks`
    /// ranks: `SLICES` CTAs per row, grid (slice, row).
    pub fn plan(&self, rows: u32, ranks: u32) -> Result<LaunchPlan, Refusal> {
        self.check(ranks)?;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal(
                "Sinkhorn comm epilogue bucket exceeds the entry capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [SLICES, rows, 1],
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
        format!(
            "const chop::comm::epilogue::SinkhornSplitExpandMixNorm<{mixer}> epilogue{{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)}};chop::comm::epilogue::sinkhorn_split_rows(summed,epilogue);"
        )
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

/// The split rows entry of `mix` under `profile`,
/// `chop_sinkhorn_epilogue_{tag}_rows_split_v1`, which waits for no peer
/// and so does not carry the collectives' `chop_comm_` prefix. Its input
/// norm sums its squares in column order, as one thread of the finishing
/// CTA adds them.
pub fn entry(
    profile: Profile,
    mix: SinkhornMix,
    rmsnorm_eps_bits: u32,
    capacity: u32,
) -> Result<Entry, Refusal> {
    let entry = Entry {
        symbol: format!("{SYMBOL_PREFIX}{}_rows_split_v1", profile.tag()),
        mix,
        rmsnorm_eps_bits,
        norm: RmsNormReduction::Ordered,
        capacity,
        params: params(mix.fn_dtype),
    };
    entry.contract()?;
    Ok(entry)
}

/// A cubin of `entries` named `name`.
pub fn cubin_spec(name: String, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// The served mixer's split entry under the Sinkhorn engine's served
/// profile, with the input norm's eps 1e-5.
pub fn catalog() -> Entry {
    let served = crate::sinkhorn_hc::SERVED;
    entry(
        crate::sinkhorn_hc::SERVED_PROFILE,
        served.mix,
        1e-5_f32.to_bits(),
        served.capacity,
    )
    .expect("the served split Sinkhorn comm epilogue")
}

/// The lab's cubins of [`catalog`]: `sinkhorn_epilogue_split_{tag}_sm120`,
/// and `sinkhorn_epilogue_split_late_wait_{tag}_sm120`, the entry with its
/// dependency wait moved after its reads of the sums and streams
/// (`CHOP_COMM_TEST_LATE_WAIT`, symbol prefix [`LATE_WAIT_PREFIX`]), a
/// mutant the lab's checks must catch.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = crate::sinkhorn_hc::SERVED_PROFILE.tag();
    let mut late_wait = cubin_spec(
        format!("sinkhorn_epilogue_split_late_wait_{tag}_sm120"),
        &[catalog()],
    );
    for entry in &mut late_wait.entries {
        entry.symbol = entry.symbol.replacen(SYMBOL_PREFIX, LATE_WAIT_PREFIX, 1);
    }
    late_wait
        .defines
        .push(("CHOP_COMM_TEST_LATE_WAIT".into(), "1".into()));
    vec![
        cubin_spec(format!("sinkhorn_epilogue_split_{tag}_sm120"), &[catalog()]),
        late_wait,
    ]
}

/// The served entry's symbol prefix.
const SYMBOL_PREFIX: &str = "chop_sinkhorn_epilogue_";

/// The symbol prefix of the lab's late-wait mutant, in place of the served
/// entry's.
pub const LATE_WAIT_PREFIX: &str = "chop_sinkhorn_epilogue_late_wait_";

/// The split epilogue's buffers: the one-CTA epilogue's, the chain sums
/// (`crate::sinkhorn_hc::split::partials_bytes`) and the zeroed tickets.
#[derive(Clone, Copy, Debug)]
pub struct SplitBuffers {
    pub epilogue: EpilogueBuffers,
    pub partials: u64,
    pub tickets: u64,
}

/// The entry's arguments over the F32 sums in `summed`.
pub fn rows_arguments(summed: u64, split: SplitBuffers) -> Result<ArgumentList, Refusal> {
    let e = split.epilogue;
    let mut args = [
        summed,
        e.streams,
        e.post,
        e.comb,
        e.reduced,
        e.expanded,
        e.fn_,
        e.base,
        e.scale,
        e.collapsed,
        e.weight,
        e.normed,
        split.partials,
        split.tickets,
        e.live_rows,
    ]
    .map(Argument::DeviceBuffer)
    .to_vec();
    args.push(Argument::U32(e.eps_bits));
    let args = ArgumentList(args);
    args.validate()?;
    Ok(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sinkhorn_hc;

    #[test]
    fn the_split_entry_plans_every_bucket_at_every_rank_count() {
        let entry = catalog();
        assert_eq!(SLICES, 16);
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_epilogue_hc4x4096_rows_split_v1"
        );
        for ranks in 1..=MAX_COMM_RANKS as u32 {
            for rows in [1, 2, 7, 16, 32, 6 * 32, MAX_ROWS] {
                let plan = entry.plan(rows, ranks).unwrap();
                assert_eq!(plan.grid, [16, rows, 1]);
                assert_eq!(plan.block, [THREADS, 1, 1]);
                assert!(plan.pdl);
            }
        }
        assert!(entry.plan(0, 4).is_err());
        assert!(entry.plan(MAX_ROWS + 1, 4).is_err());
        assert!(entry.plan(1, 0).is_err());
        assert_eq!(
            entry.body(),
            "const chop::comm::epilogue::SinkhornSplitExpandMixNorm<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>> epilogue{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)};chop::comm::epilogue::sinkhorn_split_rows(summed,epilogue);"
        );
        let specs = lab_cubin_specs();
        assert_eq!(specs.len(), 2);
        assert_eq!(specs[0].name, "sinkhorn_epilogue_split_hc4x4096_sm120");
        assert_eq!(specs[0].entries.len(), 1);
        assert!(specs[0].defines.is_empty());
        assert!(specs[0].sources_root.join(HEADER).is_file());
        assert_eq!(
            specs[1].name,
            "sinkhorn_epilogue_split_late_wait_hc4x4096_sm120"
        );
        assert_eq!(
            specs[1]
                .entries
                .iter()
                .map(|entry| entry.symbol.as_str())
                .collect::<Vec<_>>(),
            ["chop_sinkhorn_epilogue_late_wait_hc4x4096_rows_split_v1"]
        );
        assert_eq!(specs[1].entries[0].body, specs[0].entries[0].body);
        assert_eq!(
            specs[1].defines,
            [("CHOP_COMM_TEST_LATE_WAIT".to_string(), "1".to_string())]
        );
    }

    #[test]
    fn arguments_follow_the_entry_abi() {
        let entry = catalog();
        let split = SplitBuffers {
            epilogue: EpilogueBuffers {
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
            },
            partials: 24,
            tickets: 25,
        };
        let args = rows_arguments(2, split).unwrap();
        assert_eq!(args.0.len(), entry.params.len());
        let names = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } | EntryParam::U32 { name } => name.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>();
        assert_eq!(
            names[12..],
            ["partials", "tickets", "live_rows", "eps_bits"]
        );
        assert_eq!(args.0[12], Argument::DeviceBuffer(24));
        assert_eq!(args.0[13], Argument::DeviceBuffer(25));
        assert_eq!(args.0[14], Argument::DeviceBuffer(21));
        assert_eq!(args.0.last(), Some(&Argument::U32(1e-5_f32.to_bits())));
    }

    #[test]
    fn other_shapes_refuse_by_name() {
        let refuse = |mix| entry(sinkhorn_hc::SERVED_PROFILE, mix, 1e-5_f32.to_bits(), 8);
        assert!(
            refuse(SinkhornMix {
                streams: 8,
                ..sinkhorn_hc::SERVED.mix
            })
            .is_err()
        );
        assert!(
            refuse(SinkhornMix {
                width: 5120,
                ..sinkhorn_hc::SERVED.mix
            })
            .is_err()
        );
        assert_eq!(
            entry(
                sinkhorn_hc::SERVED_PROFILE,
                sinkhorn_hc::SERVED.mix,
                1e-5_f32.to_bits(),
                MAX_ROWS + 1
            )
            .unwrap_err(),
            Refusal("Sinkhorn comm epilogue capacity exceeds 8192")
        );
    }
}
