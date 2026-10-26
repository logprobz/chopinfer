//! The split mix-collapse fused with the sublayer's input RMSNorm, with
//! batched loads (`cuda/sinkhorn_hc/mix_norm_batched.cuh`): the outputs of
//! [`super::mix_norm_exact`]'s entry, bit for bit, in the same launch
//! geometry and ABI, with the same per-element operations on a shorter
//! critical path. A live row's fn values go to registers before the
//! dependency wait, the chain sums' butterfly runs transposed, the ticket
//! releases once per CTA, the finishing CTA issues all its loads at once,
//! and the Sinkhorn exchanges cells by shuffles. See
//! `cuda/sinkhorn_hc/README.md` ("Split mix-collapse with batched loads").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan, Refusal,
};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{Geometry, SERVED, SERVED_PROFILE, THREADS, mix_norm, mix_norm_exact};
use crate::Profile;
use crate::row::ordered_sum;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_norm_batched.cuh";
/// Lab-only mutant: the dependency wait after the chain sums' reads of the
/// streams and the live count.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MIX_NORM_BATCHED_TEST_LATE_WAIT";
/// Lab-only mutant: the chain sums' butterfly levels from distance 1 up.
pub const LEVELS_REVERSED_DEFINE: &str = "CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED";
/// Which kernel an entry runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    /// The batched schedule.
    Batched,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    /// The mix-collapse, as the split entry declares it.
    pub query: Query,
    /// The input RMSNorm of the collapsed row, as [`mix_norm::Entry`]
    /// declares it.
    pub norm: RmsNormRequirement,
    pub kernel: Kernel,
    pub params: Vec<EntryParam>,
}

impl Entry {
    /// The exact-sum fused entry of the same mix-collapse and norm, whose
    /// outputs, grid and ABI this entry shares.
    pub fn exact(&self) -> mix_norm_exact::Entry {
        mix_norm_exact::Entry {
            profile: self.profile,
            symbol: self.symbol.clone(),
            query: self.query,
            norm: self.norm,
            params: self.params.clone(),
        }
    }

    fn check(&self) -> Result<SinkhornMix, Refusal> {
        self.exact().contract()?;
        let Query::MixCollapse { mix, .. } = self.query else {
            return Err(Refusal(
                "the batched Sinkhorn mix and norm entry is a mix-collapse",
            ));
        };
        if mix.fn_dtype != DType::Bf16 {
            return Err(Refusal(
                "the batched Sinkhorn mix holds BF16 fn values in registers",
            ));
        }
        let flat = mix.streams * mix.width;
        let mixes = (2 + mix.streams) * mix.streams;
        let chains = flat / 128;
        if flat % (8 * THREADS) != 0 || (mixes * chains) % THREADS != 0 {
            return Err(Refusal(
                "the batched Sinkhorn mix's finisher loads whole vectors and chain sums per thread",
            ));
        }
        Ok(mix)
    }

    /// The mix-collapse's contract, as the split entry declares it.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        self.exact().contract()
    }

    /// The prepared row maximum.
    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.check()?;
        self.exact().capacity()
    }

    /// The split entry's plan for a prepared bucket of `rows`.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.check()?;
        self.exact().plan(rows, live_rows)
    }

    /// The launch arguments of the exact fused entry.
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        self.exact().pack_arguments(values)
    }

    pub fn body(&self) -> Result<String, Refusal> {
        let mix = self.check()?;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            mix.streams,
            mix.width,
            mix.iterations,
            mix.eps_bits,
            mix.norm_eps_bits,
            mix.fn_dtype == DType::F32
        );
        let args = "x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)";
        Ok(match self.kernel {
            Kernel::Batched => format!(
                "chop::sinkhorn_hc::mix_collapse_norm_batched<{traits},false>({args},nullptr);"
            ),
        })
    }
}

/// The batched fused entry of `geometry` and `norm` under `profile`, symbol
/// `chop_sinkhorn_hc_{tag}_mix_collapse_norm_batched{suffix}_v1`.
pub fn entry_with_suffix(
    profile: Profile,
    geometry: Geometry,
    norm: RmsNormRequirement,
    suffix: &str,
) -> Entry {
    let Geometry { mix, capacity } = geometry;
    Entry {
        profile,
        symbol: format!(
            "chop_sinkhorn_hc_{}_mix_collapse_norm_batched{suffix}_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        norm,
        kernel: Kernel::Batched,
        params: mix_norm::params(mix),
    }
}

/// The batched fused entry of `geometry` and `norm` under `profile`.
pub fn entry(profile: Profile, geometry: Geometry, norm: RmsNormRequirement) -> Entry {
    entry_with_suffix(profile, geometry, norm, "")
}

/// A cubin of `entries` named `name`, compiled with each of `defines` set
/// to 1.
pub fn cubin_spec(name: String, entries: &[Entry], defines: &[&str]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: defines
            .iter()
            .map(|define| ((*define).to_owned(), "1".to_owned()))
            .collect(),
        entries: entries
            .iter()
            .map(|entry| {
                Ok(EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    entry.body()?,
                    THREADS,
                    1,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

/// The served shape's batched fused entry, normalizing at the mixer's norm
/// eps.
pub fn served() -> Entry {
    entry(SERVED_PROFILE, SERVED, mix_norm::served_norm())
}

/// The lab's mutants of the served entry: (suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_levels_reversed", LEVELS_REVERSED_DEFINE),
    ("_ties_away", ordered_sum::TIES_AWAY_DEFINE),
];

/// The lab's cubins: the served entry
/// (`sinkhorn_hc_{tag}_mix_norm_batched_sm120`), each mutant in a cubin of
/// its own (`sinkhorn_hc_{tag}_mix_norm_batched{suffix}_sm120`). The lab
/// compares the entry with [`mix_norm_exact`]'s lab cubin.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = SERVED_PROFILE.tag();
    let mut specs = vec![
        cubin_spec(
            format!("sinkhorn_hc_{tag}_mix_norm_batched_sm120"),
            &[served()],
            &[],
        )
        .expect("the served batched entry is valid"),
    ];
    for (suffix, define) in MUTANTS {
        let mutant = entry_with_suffix(SERVED_PROFILE, SERVED, mix_norm::served_norm(), suffix);
        specs.push(
            cubin_spec(
                format!("sinkhorn_hc_{tag}_mix_norm_batched{suffix}_sm120"),
                &[mutant],
                &[define],
            )
            .expect("the batched mutants are valid"),
        );
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::RmsNormKind;
    use kernel_core::rmsnorm::RmsNormNumerics;

    #[test]
    fn served_batched_entry_shares_the_exact_entrys_plan_and_abi() {
        let entry = served();
        let exact = mix_norm_exact::served();
        assert_eq!(entry.contract().unwrap().key, exact.contract().unwrap().key);
        assert_eq!(entry.capacity().unwrap(), 8192);
        for rows in [1, 2, 4, 8, 16, 32, 6 * 32, 8192] {
            assert_eq!(
                entry.plan(rows, rows).unwrap(),
                exact.plan(rows, rows).unwrap()
            );
        }
        assert!(entry.plan(8193, 1).is_err());
        assert_eq!(entry.params, exact.params);
        assert_eq!(entry.norm, exact.norm);
        let mut args = (0..12).map(Argument::DeviceBuffer).collect::<Vec<_>>();
        args.push(Argument::U32(1e-5_f32.to_bits()));
        assert_eq!(
            entry.pack_arguments(args.clone()).unwrap(),
            exact.pack_arguments(args.clone()).unwrap()
        );
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_batched_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_norm_batched<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>,false>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits),nullptr);"
        );

        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_mix_norm_batched_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_late_wait_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_levels_reversed_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_ties_away_sm120",
            ]
        );
        assert_eq!(specs[0].headers, [format!("#include \"{HEADER}\"")]);
        assert_eq!(
            specs[3].defines,
            [(ordered_sum::TIES_AWAY_DEFINE.into(), "1".into())]
        );
    }

    #[test]
    fn batched_entry_refuses_what_the_exact_entry_or_its_loads_refuse() {
        let mix = SERVED.mix;
        let refuse = |norm: RmsNormRequirement, mix: SinkhornMix| {
            let entry = entry(SERVED_PROFILE, Geometry { mix, capacity: 32 }, norm);
            assert!(entry.contract().is_err(), "{norm:?} {mix:?}");
            assert!(entry.plan(1, 1).is_err());
            assert!(entry.body().is_err());
        };
        refuse(
            RmsNormRequirement {
                fused_residual: true,
                ..mix_norm::served_norm()
            },
            mix,
        );
        refuse(
            RmsNormRequirement::bf16(
                4096,
                1e-5,
                RmsNormKind::Plain,
                false,
                RmsNormNumerics::TREE_256,
            ),
            mix,
        );
        // F32 fn values do not fit the registers.
        let f32_fn = SinkhornMix {
            fn_dtype: DType::F32,
            ..mix
        };
        refuse(mix_norm::norm_requirement(f32_fn, 1e-5), f32_fn);
        // 128 terms over 224 workers: fewer terms than workers.
        let narrow = SinkhornMix { width: 128, ..mix };
        refuse(mix_norm::norm_requirement(narrow, 1e-5), narrow);
    }
}
