//! The split mix-collapse fused with the sublayer's input RMSNorm, the
//! norm's sum of squares in the 256-lane tree of
//! [`RmsNormNumerics::HF_TREE_256`] (`cuda/sinkhorn_hc/mix_norm_tree.cuh`):
//! the split mix-collapse's outputs and chain sums, as
//! [`super::mix_norm_batched`]'s entry gives them, and the row engine's
//! tree norm ([`crate::row::rmsnorm_tree`]) of the collapsed row, bit for
//! bit, in the batched entry's grid and ABI. See
//! `cuda/sinkhorn_hc/README.md` ("Split mix-collapse with the tree norm").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan, Refusal,
};
use kernel_core::properties::Invariance;
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::mix_norm_batched::{self, LATE_WAIT_DEFINE, LEVELS_REVERSED_DEFINE};
use super::{Geometry, SERVED, SERVED_PROFILE, THREADS, mix_norm};
use crate::Profile;
use crate::row::rmsnorm_tree;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_norm_tree.cuh";

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    /// The mix-collapse, as the split entry declares it.
    pub query: Query,
    /// The input RMSNorm of the collapsed row, with the tree numerics.
    pub norm: RmsNormRequirement,
    pub params: Vec<EntryParam>,
}

impl Entry {
    /// The batched entry of the same mix-collapse with the norm's ordered
    /// numerics, whose grid, ABI and outputs other than the normed row this
    /// entry shares.
    pub fn batched(&self) -> mix_norm_batched::Entry {
        mix_norm_batched::Entry {
            profile: self.profile,
            symbol: self.symbol.clone(),
            query: self.query,
            norm: RmsNormRequirement {
                numerics: Some(RmsNormNumerics::HF_ORDERED),
                ..self.norm
            },
            kernel: mix_norm_batched::Kernel::Batched,
            params: self.params.clone(),
        }
    }

    fn check(&self) -> Result<SinkhornMix, Refusal> {
        if self.norm.numerics != Some(RmsNormNumerics::HF_TREE_256) {
            return Err(Refusal(
                "the tree mixer norm sums its squares in the 256-lane tree",
            ));
        }
        // The finisher's seven worker warps run the tree's eight.
        if THREADS != rmsnorm_tree::LANES as u32 {
            return Err(Refusal(
                "the tree mixer norm runs its finisher in 256 threads",
            ));
        }
        self.batched().contract()?;
        let Query::MixCollapse { mix, .. } = self.query else {
            return Err(Refusal("the tree mixer norm entry is a mix-collapse"));
        };
        Ok(mix)
    }

    /// The mix-collapse's contract, as the split entry declares it.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        self.batched().contract()
    }

    /// The prepared row maximum.
    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.check()?;
        self.batched().capacity()
    }

    /// The entry's declared exactness, compiled without the lab's mutant
    /// defines: one-row bits up to its capacity, any chunk split and row
    /// independence. Its mix-collapse outputs are the split entry's (one
    /// row's chain sums folded in K order by the row's finisher) and its
    /// norm is the row engine's tree over the collapsed row, whose order
    /// depends on the width alone (`super`'s module docs). Nothing for an
    /// entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        match self.contract().and_then(|_| self.capacity()) {
            Ok(capacity) => Invariance::any_rows(capacity),
            Err(_) => Invariance::NONE,
        }
    }

    /// The split entry's plan for a prepared bucket of `rows`.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.check()?;
        self.batched().plan(rows, live_rows)
    }

    /// The launch's arguments: [`mix_norm::Entry::pack_arguments`]'s.
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        self.check()?;
        self.batched().pack_arguments(values)
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
        Ok(format!(
            "chop::sinkhorn_hc::mix_collapse_norm_tree<{traits}>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        ))
    }
}

/// [`mix_norm::norm_requirement`] with the tree numerics.
pub fn norm_requirement(mix: SinkhornMix, eps: f32) -> RmsNormRequirement {
    RmsNormRequirement {
        numerics: Some(RmsNormNumerics::HF_TREE_256),
        ..mix_norm::norm_requirement(mix, eps)
    }
}

/// The served shape's input norm, at the mixer's norm eps.
pub fn served_norm() -> RmsNormRequirement {
    norm_requirement(SERVED.mix, f32::from_bits(SERVED.mix.norm_eps_bits))
}

/// The tree fused entry of `geometry` and `norm` under `profile`, symbol
/// `chop_sinkhorn_hc_{tag}_mix_collapse_norm_tree{suffix}_v1`.
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
            "chop_sinkhorn_hc_{}_mix_collapse_norm_tree{suffix}_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        norm,
        params: mix_norm::params(mix),
    }
}

/// The tree fused entry of `geometry` and `norm` under `profile`.
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

/// The served shape's tree fused entry, normalizing at the mixer's norm eps.
pub fn served() -> Entry {
    entry(SERVED_PROFILE, SERVED, served_norm())
}

/// The lab's mutants of the served entry: (suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_levels_reversed", LEVELS_REVERSED_DEFINE),
    (
        "_distances_reversed",
        rmsnorm_tree::DISTANCES_REVERSED_DEFINE,
    ),
];

/// The lab's cubins: the served entry
/// (`sinkhorn_hc_{tag}_mix_norm_tree_sm120`) and each mutant in a cubin of
/// its own (`sinkhorn_hc_{tag}_mix_norm_tree{suffix}_sm120`). The lab
/// compares the entry with the batched entry's lab cubin and the row
/// engine's tree norm.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = SERVED_PROFILE.tag();
    let mut specs = vec![
        cubin_spec(
            format!("sinkhorn_hc_{tag}_mix_norm_tree_sm120"),
            &[served()],
            &[],
        )
        .expect("the served tree entry is valid"),
    ];
    for (suffix, define) in MUTANTS {
        let mutant = entry_with_suffix(SERVED_PROFILE, SERVED, served_norm(), suffix);
        specs.push(
            cubin_spec(
                format!("sinkhorn_hc_{tag}_mix_norm_tree{suffix}_sm120"),
                &[mutant],
                &[define],
            )
            .expect("the tree mutants are valid"),
        );
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::RmsNormKind;

    #[test]
    fn served_tree_entry_shares_the_batched_entrys_plan_and_abi() {
        let entry = served();
        let batched = mix_norm_batched::served();
        assert_eq!(
            entry.contract().unwrap().key,
            batched.contract().unwrap().key
        );
        assert_eq!(entry.capacity().unwrap(), 8192);
        for rows in [1, 2, 4, 8, 16, 32, 6 * 32, 8192] {
            assert_eq!(
                entry.plan(rows, rows).unwrap(),
                batched.plan(rows, rows).unwrap()
            );
        }
        assert!(entry.plan(8193, 1).is_err());
        assert_eq!(entry.params, batched.params);
        assert_eq!(entry.batched().norm, batched.norm);
        assert_eq!(entry.norm.numerics, Some(RmsNormNumerics::HF_TREE_256));
        assert_eq!(
            entry.norm,
            RmsNormRequirement {
                numerics: Some(RmsNormNumerics::HF_TREE_256),
                ..batched.norm
            }
        );
        let mut args = (0..12).map(Argument::DeviceBuffer).collect::<Vec<_>>();
        args.push(Argument::U32(1e-5_f32.to_bits()));
        assert_eq!(
            entry.pack_arguments(args.clone()).unwrap(),
            batched.pack_arguments(args).unwrap()
        );
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_tree_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_norm_tree<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_mix_norm_tree_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_tree_late_wait_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_tree_levels_reversed_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_tree_distances_reversed_sm120",
            ]
        );
        assert_eq!(specs[0].headers, [format!("#include \"{HEADER}\"")]);
        assert_eq!(
            specs[3].defines,
            [(rmsnorm_tree::DISTANCES_REVERSED_DEFINE.into(), "1".into())]
        );
    }

    #[test]
    fn tree_entry_refuses_other_numerics_and_what_the_batched_entry_refuses() {
        let mix = SERVED.mix;
        let refuse = |norm: RmsNormRequirement, mix: SinkhornMix| {
            let entry = entry(SERVED_PROFILE, Geometry { mix, capacity: 32 }, norm);
            assert!(entry.contract().is_err(), "{norm:?} {mix:?}");
            assert!(entry.plan(1, 1).is_err());
            assert!(entry.body().is_err());
        };
        // The ordered numerics are the batched entry's.
        refuse(mix_norm::served_norm(), mix);
        refuse(
            RmsNormRequirement {
                fused_residual: true,
                ..served_norm()
            },
            mix,
        );
        refuse(
            RmsNormRequirement {
                kind: RmsNormKind::Plain,
                ..served_norm()
            },
            mix,
        );
        let f32_fn = SinkhornMix {
            fn_dtype: DType::F32,
            ..mix
        };
        refuse(norm_requirement(f32_fn, 1e-5), f32_fn);
        // 128 terms over 224 workers: fewer columns than the batched entry
        // takes.
        let narrow = SinkhornMix { width: 128, ..mix };
        refuse(norm_requirement(narrow, 1e-5), narrow);
    }
}
