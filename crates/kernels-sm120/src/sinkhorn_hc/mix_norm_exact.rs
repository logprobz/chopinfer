//! The split mix-collapse fused with the sublayer's input RMSNorm, with the
//! norm's ordered sum computed by the exact parallel method
//! (`cuda/sinkhorn_hc/mix_norm_exact.cuh`, `row::ordered_sum`): the outputs
//! of [`super::mix_norm`]'s entry (the split entry's followed by the row
//! engine's HF-ordered RMSNorm of the collapsed row), bit for bit, in the
//! same launch geometry and ABI. The row's finishing CTA runs the Sinkhorn
//! in warp 0 from the comb logits on, while warps 1 to 7 collapse the row
//! and normalize it, adding the norm's squares in about 20 steps per worker
//! instead of one thread's 4096 dependent adds. See
//! `cuda/sinkhorn_hc/README.md` ("Split mix-collapse with the exact input
//! norm").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan, Refusal,
};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{Geometry, SERVED, SERVED_PROFILE, THREADS, mix_norm};
use crate::Profile;
use crate::row::ordered_sum;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_norm_exact.cuh";
/// Lab-only mutant: the dependency wait after the chain sums' reads of the
/// streams and the live count.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MIX_NORM_EXACT_TEST_LATE_WAIT";
/// The threads that finish a row beside the Sinkhorn warp, and so the
/// ordered sum's workers.
pub const WORKERS: u32 = THREADS - 32;

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    /// The mix-collapse, as the split entry declares it.
    pub query: Query,
    /// The input RMSNorm of the collapsed row, as [`mix_norm::Entry`]
    /// declares it.
    pub norm: RmsNormRequirement,
    pub params: Vec<EntryParam>,
}

impl Entry {
    /// The serial-sum fused entry of the same mix-collapse and norm, whose
    /// outputs, grid and ABI this entry shares.
    pub fn serial(&self) -> mix_norm::Entry {
        mix_norm::Entry {
            profile: self.profile,
            symbol: self.symbol.clone(),
            query: self.query,
            norm: self.norm,
            params: self.params.clone(),
        }
    }

    fn check(&self) -> Result<SinkhornMix, Refusal> {
        self.serial().contract()?;
        let Query::MixCollapse { mix, .. } = self.query else {
            return Err(Refusal(
                "the fused Sinkhorn mix and exact norm entry is a mix-collapse",
            ));
        };
        ordered_sum::Layout {
            terms: mix.width as usize,
            workers: WORKERS as usize,
        }
        .check()?;
        // The squares and the ordered sum's state live in the staged
        // streams: H F32 squares, then three words per worker and two per
        // worker warp.
        let state = 3 * WORKERS + 2 * (WORKERS / 32);
        if (mix.width + state) * 4 > mix.streams * mix.width * 2 {
            return Err(Refusal(
                "the fused Sinkhorn mix's squares and ordered sum fit its staged streams",
            ));
        }
        Ok(mix)
    }

    /// The mix-collapse's contract, as the split entry declares it.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        self.serial().contract()
    }

    /// The prepared row maximum.
    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.check()?;
        self.serial().capacity()
    }

    /// The split entry's plan for a prepared bucket of `rows`.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.check()?;
        self.serial().plan(rows, live_rows)
    }

    /// The launch's arguments: [`mix_norm::Entry::pack_arguments`]'s.
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        self.serial().pack_arguments(values)
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
            "chop::sinkhorn_hc::mix_collapse_norm_exact<{traits}>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        ))
    }
}

/// The exact-sum fused entry of `geometry` and `norm` under `profile`,
/// symbol `chop_sinkhorn_hc_{tag}_mix_collapse_norm_exact{suffix}_v1`.
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
            "chop_sinkhorn_hc_{}_mix_collapse_norm_exact{suffix}_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        norm,
        params: mix_norm::params(mix),
    }
}

/// The exact-sum fused entry of `geometry` and `norm` under `profile`.
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

/// The served shape's exact-sum fused entry, normalizing at the mixer's
/// norm eps.
pub fn served() -> Entry {
    entry(SERVED_PROFILE, SERVED, mix_norm::served_norm())
}

/// The lab's mutants of the served entry: (suffix, define).
pub const MUTANTS: [(&str, &str); 2] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_ties_away", ordered_sum::TIES_AWAY_DEFINE),
];

/// The lab's cubins: the served entry (`sinkhorn_hc_{tag}_mix_norm_exact_sm120`)
/// and each mutant in a cubin of its own
/// (`sinkhorn_hc_{tag}_mix_norm_exact{suffix}_sm120`). The lab compares the
/// entry with [`mix_norm`]'s lab cubin.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = SERVED_PROFILE.tag();
    let mut specs = vec![
        cubin_spec(
            format!("sinkhorn_hc_{tag}_mix_norm_exact_sm120"),
            &[served()],
            &[],
        )
        .expect("the served exact-sum entry is valid"),
    ];
    for (suffix, define) in MUTANTS {
        let mutant = entry_with_suffix(SERVED_PROFILE, SERVED, mix_norm::served_norm(), suffix);
        specs.push(
            cubin_spec(
                format!("sinkhorn_hc_{tag}_mix_norm_exact{suffix}_sm120"),
                &[mutant],
                &[define],
            )
            .expect("the exact-sum mutants are valid"),
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
    fn served_exact_entry_shares_the_serial_entrys_plan_and_abi() {
        let entry = served();
        let serial = mix_norm::served();
        entry.contract().unwrap();
        assert_eq!(
            entry.contract().unwrap().key,
            serial.contract().unwrap().key
        );
        assert_eq!(entry.capacity().unwrap(), 8192);
        for rows in [1, 2, 4, 8, 16, 32, 6 * 32, 8192] {
            assert_eq!(
                entry.plan(rows, rows).unwrap(),
                serial.plan(rows, rows).unwrap()
            );
        }
        assert!(entry.plan(8193, 1).is_err());
        assert_eq!(entry.params, serial.params);
        assert_eq!(entry.norm, serial.norm);
        let mut args = (0..12).map(Argument::DeviceBuffer).collect::<Vec<_>>();
        args.push(Argument::U32(1e-5_f32.to_bits()));
        entry.pack_arguments(args).unwrap();
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_exact_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_norm_exact<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_mix_norm_exact_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_exact_late_wait_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_exact_ties_away_sm120",
            ]
        );
        assert_eq!(
            specs[2].defines,
            [(ordered_sum::TIES_AWAY_DEFINE.into(), "1".into())]
        );
    }

    #[test]
    fn exact_entry_refuses_what_the_serial_entry_or_the_sum_refuses() {
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
        // 128 terms over 224 workers: fewer terms than workers.
        let narrow = SinkhornMix { width: 128, ..mix };
        refuse(mix_norm::norm_requirement(narrow, 1e-5), narrow);
    }
}
