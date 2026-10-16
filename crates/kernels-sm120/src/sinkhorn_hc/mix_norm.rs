//! The split mix-collapse fused with the sublayer's input RMSNorm
//! (`cuda/sinkhorn_hc/mix_norm_split.cuh`): in one launch, the outputs of
//! [`super::split`]'s entry followed by the row engine's HF-ordered RMSNorm
//! (`row::rmsnorm`, `RmsNormKind::HfBf16Intermediate`) of the collapsed
//! row, bit for bit. The chain sums, tickets and grid are the split
//! entry's. The row's finishing CTA runs the Sinkhorn normalizations in one
//! warp while one thread adds the norm's squares in column order, so the
//! norm's serial sum hides the Sinkhorn and the second launch goes away.
//! See `cuda/sinkhorn_hc/README.md` ("Split mix-collapse with the input
//! norm").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan, Refusal, RmsNormKind, Validate,
};
use kernel_core::rmsnorm::{RmsNormNumerics, RmsNormRequirement};
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{COLUMNS, Geometry, SERVED, SERVED_PROFILE, THREADS, ptr, split};
use crate::Profile;
use crate::row::rmsnorm::{RmsNorm, RmsNormTarget};

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_norm_split.cuh";
/// Lab-only mutant: the dependency wait after the chain sums' reads of the
/// streams and the live count.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MIX_NORM_TEST_LATE_WAIT";
/// Lab-only mutant: the norm's squares added in reverse column order.
pub const REVERSE_SUM_DEFINE: &str = "CHOP_MIX_NORM_TEST_REVERSE_SUM";

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    /// The mix-collapse, as the split entry declares it.
    pub query: Query,
    /// The input RMSNorm of the collapsed row: HF order with a BF16
    /// intermediate, no residual, the mixer's width. Its eps is the launch's
    /// `eps_bits`, as the row engine's entry takes it.
    pub norm: RmsNormRequirement,
    pub params: Vec<EntryParam>,
}

/// The input RMSNorm the entry fuses for `mix` at `eps`.
pub fn norm_requirement(mix: SinkhornMix, eps: f32) -> RmsNormRequirement {
    RmsNormRequirement::bf16(
        mix.width,
        eps,
        RmsNormKind::HfBf16Intermediate,
        false,
        RmsNormNumerics::HF_ORDERED,
    )
}

impl Entry {
    /// The split entry of the same mix-collapse, whose chain sums, tickets
    /// and outputs this entry shares.
    pub fn split(&self) -> split::Entry {
        split::Entry {
            profile: self.profile,
            symbol: self.symbol.clone(),
            query: self.query,
            params: split::params(self.mix().unwrap_or(SERVED.mix)),
        }
    }

    fn mix(&self) -> Result<SinkhornMix, Refusal> {
        match self.query {
            Query::MixCollapse { mix, .. } => Ok(mix),
            _ => Err(Refusal(
                "the fused Sinkhorn mix and norm entry is a mix-collapse",
            )),
        }
    }

    fn check(&self) -> Result<SinkhornMix, Refusal> {
        let mix = self.mix()?;
        let capacity = self.split().capacity()?;
        split::fits(mix)?;
        // The row engine's validation of the same norm, then this kernel's
        // own form of it.
        RmsNorm::new(self.norm, capacity, RmsNormTarget::SM120F)?;
        if self.norm.kind != RmsNormKind::HfBf16Intermediate
            || self.norm.numerics != Some(RmsNormNumerics::HF_ORDERED)
            || self.norm.fused_residual
        {
            return Err(Refusal(
                "the fused Sinkhorn mix runs the HF-ordered input RMSNorm without a residual",
            ));
        }
        if self.norm.width != mix.width {
            return Err(Refusal(
                "the fused Sinkhorn mix normalizes the collapsed row of the mixer's width",
            ));
        }
        if !mix.width.is_multiple_of(COLUMNS) {
            return Err(Refusal(
                "the fused Sinkhorn mix keeps whole 512-column collapse passes",
            ));
        }
        if mix.streams * mix.streams > 32 {
            return Err(Refusal(
                "the fused Sinkhorn mix runs every Sinkhorn cell in one warp",
            ));
        }
        Ok(mix)
    }

    /// The mix-collapse's contract, as the split entry declares it; the
    /// norm is [`Entry::norm`], validated by the row engine's rules.
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.check()?;
        self.split().contract()
    }

    /// The prepared row maximum.
    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.check()?;
        self.split().capacity()
    }

    /// The split entry's plan for a prepared bucket of `rows`.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.check()?;
        self.split().plan(rows, live_rows)
    }

    /// The launch's arguments: device buffers in parameter order, then the
    /// norm's eps bits.
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if values.len() != self.params.len()
            || !self.params.iter().zip(&values).all(|(p, a)| {
                matches!(
                    (p, a),
                    (EntryParam::DevicePointer { .. }, Argument::DeviceBuffer(_))
                        | (EntryParam::U32 { .. }, Argument::U32(_))
                )
            })
        {
            return Err(Refusal("fused Sinkhorn mix and norm argument ABI mismatch"));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
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
            "chop::sinkhorn_hc::mix_collapse_norm_split<{traits}>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        ))
    }
}

/// The entry ABI: the split entry's, with the norm's weight after `scale`,
/// the normed row after `comb`, and the norm's eps bits last.
pub fn params(mix: SinkhornMix) -> Vec<EntryParam> {
    let mut params = split::params(mix);
    let at = |params: &[EntryParam], name: &str| {
        params
            .iter()
            .position(|p| matches!(p, EntryParam::DevicePointer { name: n, .. } if n == name))
            .expect("a split entry parameter")
    };
    let scale = at(&params, "scale");
    params.insert(scale + 1, ptr("weight", "const uint16_t"));
    let comb = at(&params, "comb");
    params.insert(comb + 1, ptr("normed", "uint16_t"));
    params.push(EntryParam::U32 {
        name: "eps_bits".into(),
    });
    params
}

/// The fused entry of `geometry` and `norm` under `profile`, symbol
/// `chop_sinkhorn_hc_{tag}_mix_collapse_norm_split{suffix}_v1`.
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
            "chop_sinkhorn_hc_{}_mix_collapse_norm_split{suffix}_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        norm,
        params: params(mix),
    }
}

/// The fused entry of `geometry` and `norm` under `profile`.
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

/// The served shape's fused entry, normalizing at the mixer's norm eps.
pub fn served() -> Entry {
    entry(SERVED_PROFILE, SERVED, served_norm())
}

/// The served shape's input RMSNorm.
pub fn served_norm() -> RmsNormRequirement {
    norm_requirement(SERVED.mix, f32::from_bits(SERVED.mix.norm_eps_bits))
}

/// The lab's mutants of the served entry: (suffix, define).
pub const MUTANTS: [(&str, &str); 2] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_reverse_sum", REVERSE_SUM_DEFINE),
];

/// The lab's cubins: the served fused entry
/// (`sinkhorn_hc_{tag}_mix_norm_sm120`) and each mutant in a cubin of its
/// own (`sinkhorn_hc_{tag}_mix_norm{suffix}_sm120`). The lab compares the
/// norm with the row engine's entry the comm epilogue's lab cubin holds
/// (`comm::sinkhorn_epilogue::reference_norm`).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = SERVED_PROFILE.tag();
    let mut specs = vec![
        cubin_spec(
            format!("sinkhorn_hc_{tag}_mix_norm_sm120"),
            &[served()],
            &[],
        )
        .expect("the served fused entry is valid"),
    ];
    for (suffix, define) in MUTANTS {
        let mutant = entry_with_suffix(SERVED_PROFILE, SERVED, served_norm(), suffix);
        specs.push(
            cubin_spec(
                format!("sinkhorn_hc_{tag}_mix_norm{suffix}_sm120"),
                &[mutant],
                &[define],
            )
            .expect("the fused mutants are valid"),
        );
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_fused_entry_validates_and_plans_the_split_grid() {
        let entry = served();
        entry.contract().unwrap();
        assert_eq!(entry.capacity().unwrap(), 8192);
        for rows in [1, 2, 4, 8, 16, 32, 6 * 32, 8192] {
            let plan = entry.plan(rows, rows).unwrap();
            assert_eq!(plan, split::served().plan(rows, rows).unwrap());
            assert_eq!(plan.grid, [16, rows, 1]);
            assert!(plan.pdl);
        }
        assert!(entry.plan(8193, 1).is_err());
        assert!(entry.plan(32, 33).is_err());
        let names = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } | EntryParam::U32 { name } => name.as_str(),
                _ => "",
            })
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                "x",
                "fn",
                "base",
                "scale",
                "weight",
                "collapsed",
                "post",
                "comb",
                "normed",
                "partials",
                "tickets",
                "live_rows",
                "eps_bits"
            ]
        );
        let mut args = (0..12).map(Argument::DeviceBuffer).collect::<Vec<_>>();
        args.push(Argument::U32(1e-5_f32.to_bits()));
        entry.pack_arguments(args.clone()).unwrap();
        args[12] = Argument::DeviceBuffer(12);
        assert!(entry.pack_arguments(args).is_err());
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_split_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_norm_split<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>>(x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits));"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_mix_norm_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_late_wait_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_reverse_sum_sm120",
            ]
        );
        assert_eq!(specs[1].defines, [(LATE_WAIT_DEFINE.into(), "1".into())]);
        assert_eq!(specs[2].defines, [(REVERSE_SUM_DEFINE.into(), "1".into())]);
    }

    #[test]
    fn fused_entry_refuses_norms_and_mixers_it_does_not_run() {
        let refuse = |norm: RmsNormRequirement, mix: SinkhornMix| {
            let entry = entry(SERVED_PROFILE, Geometry { mix, capacity: 32 }, norm);
            assert!(entry.contract().is_err(), "{norm:?} {mix:?}");
            assert!(entry.plan(1, 1).is_err());
            assert!(entry.body().is_err());
        };
        let mix = SERVED.mix;
        refuse(
            RmsNormRequirement {
                fused_residual: true,
                ..served_norm()
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
        refuse(
            RmsNormRequirement {
                width: 2048,
                ..norm_requirement(mix, 1e-5)
            },
            mix,
        );
        // Five streams: 25 cells fit a warp but 4096 x 5 exceeds the split
        // entry's shared memory.
        refuse(
            norm_requirement(mix, 1e-5),
            SinkhornMix { streams: 5, ..mix },
        );
        // A width of 3 x 128 is no whole 512-column pass.
        let narrow = SinkhornMix { width: 384, ..mix };
        refuse(norm_requirement(narrow, 1e-5), narrow);
        entry(
            SERVED_PROFILE,
            Geometry { mix, capacity: 32 },
            norm_requirement(mix, 1e-6),
        )
        .contract()
        .unwrap();
    }
}
