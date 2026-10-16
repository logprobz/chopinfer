//! The split mix-collapse fused with the sublayer's input RMSNorm, with
//! batched loads (`cuda/sinkhorn_hc/mix_norm_batched.cuh`): the outputs of
//! [`super::mix_norm_exact`]'s entry, bit for bit, in the same launch
//! geometry and ABI, with the same per-element operations on a shorter
//! critical path. A live row's fn values go to registers before the
//! dependency wait, the chain sums' butterfly runs transposed, the ticket
//! releases once per CTA, the finishing CTA issues all its loads at once,
//! and the Sinkhorn exchanges cells by shuffles. See
//! `cuda/sinkhorn_hc/README.md` ("Split mix-collapse with batched loads").
//!
//! The lab also builds probe entries, stamped copies of this kernel and of
//! the exact entry (`cuda/sinkhorn_hc/mix_norm_batched_probe.cuh` and
//! `mix_norm_exact_probe.cuh`, lab only), which take the phase stamps of
//! `cuda/sinkhorn_hc/mix_phase_probe.cuh` as one more argument.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, DType, InstantiationSpec, LaunchPlan, Refusal,
};
use kernel_core::rmsnorm::RmsNormRequirement;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{Geometry, SERVED, SERVED_PROFILE, THREADS, mix_norm, mix_norm_exact, ptr};
use crate::Profile;
use crate::row::ordered_sum;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_norm_batched.cuh";
/// The exact entry's probed copy (lab only).
pub const EXACT_PROBE_HEADER: &str = "sinkhorn_hc/mix_norm_exact_probe.cuh";
/// This entry's probed copy (lab only).
pub const BATCHED_PROBE_HEADER: &str = "sinkhorn_hc/mix_norm_batched_probe.cuh";
/// Lab-only mutant: the dependency wait after the chain sums' reads of the
/// streams and the live count.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MIX_NORM_BATCHED_TEST_LATE_WAIT";
/// Lab-only mutant: the chain sums' butterfly levels from distance 1 up.
pub const LEVELS_REVERSED_DEFINE: &str = "CHOP_MIX_NORM_BATCHED_TEST_LEVELS_REVERSED";
/// Stamps per CTA of a probe entry (`PhaseProbe::kSlots`).
pub const PROBE_SLOTS: usize = 32;
/// The u64 stamps of a probe entry in slots 0 to 12, by slot
/// (`PhaseProbe`): global-timer stamps, then two SM clocks.
pub const PROBE_SLOT_NAMES: [&str; 13] = [
    "entry",
    "waited",
    "ticket",
    "staged",
    "logits",
    "sinkhorn",
    "pre_post",
    "collapse",
    "squares",
    "sum",
    "inv",
    "waited_clock",
    "logits_clock",
];
/// The first of every CTA's per-warp chain-sum stamps, one per warp
/// (`PhaseProbe::kChainWarps`).
pub const PROBE_CHAIN_WARPS: usize = 16;
/// The first of the finishing CTA's per-warp output stamps, one per worker
/// warp (`PhaseProbe::kOutputWarps`).
pub const PROBE_OUTPUT_WARPS: usize = 24;
/// Warps per CTA, and the finisher's worker warps.
pub const PROBE_WARPS: usize = (THREADS / 32) as usize;
pub const PROBE_WORKER_WARPS: usize = PROBE_WARPS - 1;
// The named slots, then the chain warps' and the output warps' stamps fit a
// CTA's slots.
const _: () = assert!(
    PROBE_SLOT_NAMES.len() <= PROBE_CHAIN_WARPS
        && PROBE_CHAIN_WARPS + PROBE_WARPS == PROBE_OUTPUT_WARPS
        && PROBE_OUTPUT_WARPS + PROBE_WORKER_WARPS <= PROBE_SLOTS
);

/// Which kernel an entry runs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    /// The batched schedule.
    Batched,
    /// Its probed copy (lab only).
    BatchedProbe,
    /// The exact entry's probed copy (lab only).
    ExactProbe,
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
    /// outputs, grid and ABI this entry shares (a probe entry adds its
    /// stamps).
    pub fn exact(&self) -> mix_norm_exact::Entry {
        let mut params = self.params.clone();
        if self.probes() {
            params.pop();
        }
        mix_norm_exact::Entry {
            profile: self.profile,
            symbol: self.symbol.clone(),
            query: self.query,
            norm: self.norm,
            params,
        }
    }

    fn probes(&self) -> bool {
        self.kernel != Kernel::Batched
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

    /// The launch's arguments: [`mix_norm::Entry::pack_arguments`]'s, then
    /// a probe entry's stamps.
    pub fn pack_arguments(&self, mut values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        if !self.probes() {
            return self.exact().pack_arguments(values);
        }
        let stamps = match values.pop() {
            Some(stamps @ Argument::DeviceBuffer(_)) => stamps,
            _ => return Err(Refusal("a probe entry's last argument is its stamps")),
        };
        let mut args = self.exact().pack_arguments(values)?;
        args.0.push(stamps);
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
        let args = "x,fn,base,scale,weight,collapsed,post,comb,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)";
        Ok(match self.kernel {
            Kernel::Batched => format!(
                "chop::sinkhorn_hc::mix_collapse_norm_batched<{traits},false>({args},nullptr);"
            ),
            Kernel::BatchedProbe => format!(
                "chop::sinkhorn_hc::mix_collapse_norm_batched_probed<{traits}>({args},stamps);"
            ),
            Kernel::ExactProbe => {
                format!(
                    "chop::sinkhorn_hc::mix_collapse_norm_exact_probed<{traits}>({args},stamps);"
                )
            }
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

/// A probe entry of `kernel` (the batched kernel's or the exact entry's
/// probed copy) for `geometry` and `norm`: the fused ABI plus `stamps`
/// [rows, groups, [`PROBE_SLOTS`]] u64, symbol
/// `chop_sinkhorn_hc_{tag}_mix_collapse_norm_{batched|exact}_probe_v1`.
pub fn probe_entry(
    profile: Profile,
    geometry: Geometry,
    norm: RmsNormRequirement,
    kernel: Kernel,
) -> Entry {
    let Geometry { mix, capacity } = geometry;
    let (name, kernel) = match kernel {
        Kernel::Batched | Kernel::BatchedProbe => ("batched", Kernel::BatchedProbe),
        Kernel::ExactProbe => ("exact", Kernel::ExactProbe),
    };
    let mut params = mix_norm::params(mix);
    params.push(ptr("stamps", "uint64_t"));
    Entry {
        profile,
        symbol: format!(
            "chop_sinkhorn_hc_{}_mix_collapse_norm_{name}_probe_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        norm,
        kernel,
        params,
    }
}

/// A cubin of `entries` named `name`, compiled with each of `defines` set
/// to 1.
pub fn cubin_spec(name: String, entries: &[Entry], defines: &[&str]) -> Result<CubinSpec, Refusal> {
    let mut headers = vec![format!("#include \"{HEADER}\"")];
    for (kernel, header) in [
        (Kernel::BatchedProbe, BATCHED_PROBE_HEADER),
        (Kernel::ExactProbe, EXACT_PROBE_HEADER),
    ] {
        if entries.iter().any(|entry| entry.kernel == kernel) {
            headers.push(format!("#include \"{header}\""));
        }
    }
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers,
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

/// The served shape's probe entries: the batched kernel's and the exact
/// entry's probed copy.
pub fn served_probes() -> [Entry; 2] {
    [Kernel::BatchedProbe, Kernel::ExactProbe]
        .map(|kernel| probe_entry(SERVED_PROFILE, SERVED, mix_norm::served_norm(), kernel))
}

/// The lab's mutants of the served entry: (suffix, define).
pub const MUTANTS: [(&str, &str); 3] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    ("_levels_reversed", LEVELS_REVERSED_DEFINE),
    ("_ties_away", ordered_sum::TIES_AWAY_DEFINE),
];

/// The lab's cubins: the served entry
/// (`sinkhorn_hc_{tag}_mix_norm_batched_sm120`), each mutant in a cubin of
/// its own (`sinkhorn_hc_{tag}_mix_norm_batched{suffix}_sm120`) and the two
/// probe entries (`sinkhorn_hc_{tag}_mix_norm_probe_sm120`). The lab
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
    specs.push(
        cubin_spec(
            format!("sinkhorn_hc_{tag}_mix_norm_probe_sm120"),
            &served_probes(),
            &[],
        )
        .expect("the probe entries are valid"),
    );
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

        // The probes take their stamps last.
        let [batched, exact_probe] = served_probes();
        assert_eq!(
            batched.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_batched_probe_v1"
        );
        assert_eq!(
            exact_probe.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_norm_exact_probe_v1"
        );
        for probe in [&batched, &exact_probe] {
            assert_eq!(probe.params.len(), entry.params.len() + 1);
            assert_eq!(probe.plan(4, 4).unwrap(), entry.plan(4, 4).unwrap());
            assert!(probe.pack_arguments(args.clone()).is_err());
            let mut with_stamps = args.clone();
            with_stamps.push(Argument::DeviceBuffer(99));
            assert_eq!(probe.pack_arguments(with_stamps).unwrap().0.len(), 14);
            assert!(probe.body().unwrap().ends_with(",stamps);"));
        }
        assert!(
            exact_probe
                .body()
                .unwrap()
                .starts_with("chop::sinkhorn_hc::mix_collapse_norm_exact_probed<")
        );
        assert!(
            batched
                .body()
                .unwrap()
                .starts_with("chop::sinkhorn_hc::mix_collapse_norm_batched_probed<")
        );

        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_mix_norm_batched_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_late_wait_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_levels_reversed_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_batched_ties_away_sm120",
                "sinkhorn_hc_hc4x4096_mix_norm_probe_sm120",
            ]
        );
        assert_eq!(specs[0].headers, [format!("#include \"{HEADER}\"")]);
        assert_eq!(
            specs[4].headers,
            [HEADER, BATCHED_PROBE_HEADER, EXACT_PROBE_HEADER].map(|h| format!("#include \"{h}\""))
        );
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
