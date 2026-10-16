//! The mix-collapse with rows tiled over a CTA
//! (`cuda/sinkhorn_hc/mix_tiled.cuh`): the same `MixCollapse` query, ABI
//! and output bits as the one-CTA entry, with [`Entry::rows`] rows per CTA
//! so each warp loads its fn values of a chain once for all of them. Every
//! per-element operation and order is the one-CTA entry's: the lane
//! products and butterfly of each (row, mix) chain, the chains added in K
//! order, the sum of squares' thread mapping, rstd, pre/post, Sinkhorn and
//! collapse. See `cuda/sinkhorn_hc/README.md` ("Tiled mix-collapse").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, DType, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{CHAIN, Geometry, Op, SERVED, SERVED_PROFILE, THREADS, check, mix_params};
use crate::Profile;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "sinkhorn_hc/mix_tiled.cuh";
/// Rows per CTA the kernel takes (`Tiled<M, Rows>`'s assertion).
pub const ROWS: [u32; 3] = [4, 8, 16];
/// The tiles of rows per CTA [`served_prefill_mix`] picks.
pub const SERVED_TILES: [u32; 2] = [4, 8];
/// Lab-only mutant: the butterfly levels after the halvings in reverse
/// order (offset 1 first), the same lanes and rows in another sum tree.
pub const LEVELS_REVERSED_DEFINE: &str = "CHOP_MIX_TILED_TEST_LEVELS_REVERSED";
/// Lab-only mutant: the live count and the first chunk of streams read
/// before the dependency wait.
pub const LATE_WAIT_DEFINE: &str = "CHOP_MIX_TILED_TEST_LATE_WAIT";
/// The lab's mutants of each served tile: (suffix, define).
pub const MUTANTS: [(&str, &str); 2] = [
    ("_levels_reversed", LEVELS_REVERSED_DEFINE),
    ("_late_wait", LATE_WAIT_DEFINE),
];

/// Which mix-collapse a prefill launch of the served mixer runs: the split
/// over CTAs, or the tiled entry at a tile of rows per CTA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefillMix {
    Split,
    Tiled(u32),
}

/// The fewest prefill rows that run a tiled entry rather than the split.
/// Kernel-lab `sinkhorn_hc_tiled` timing (lab job g2-280, 2026-10-09, cold
/// L2, ms per launch): the split grows with the rows (128: 0.078, 144:
/// 0.086, 160: 0.094) while the 4-row tile stays at 0.084 to 512 rows, so
/// the split is faster at 128 rows and the 4-row tile from 144.
pub const TILED_FROM_ROWS: u32 = 144;
/// The fewest prefill rows that run the 8-row tile rather than the 4-row
/// one. Two 4-row CTAs fit an SM, so on the 188 SMs of the served GPU the
/// 4-row tile runs in one round to 1504 rows (376 CTAs) and needs a second
/// from 1505. The same timing: at 1504 rows 0.111 against the 8-row tile's
/// 0.121, at 1505 rows 0.176 against 0.139 (2048: 0.201 against 0.156;
/// 8192: 0.633 against 0.510).
pub const EIGHT_ROWS_FROM: u32 = 1505;

/// The served mixer's prefill schedule for a launch of `rows` rows.
pub const fn served_prefill_mix(rows: u32) -> PrefillMix {
    if rows < TILED_FROM_ROWS {
        PrefillMix::Split
    } else if rows < EIGHT_ROWS_FROM {
        PrefillMix::Tiled(4)
    } else {
        PrefillMix::Tiled(8)
    }
}

/// Chains of one staged chunk at `rows` rows per CTA: 16 KiB of BF16
/// streams.
pub fn chunk_chains(rows: u32) -> u32 {
    64 / rows
}

/// Refuses a mixer or tile the tiled kernel cannot hold: 4, 8 or 16 rows
/// per CTA, whole staged chunks of chains, and every Sinkhorn cell of a
/// row in one warp, as `mix_tiled.cuh` asserts.
pub fn fits(mix: SinkhornMix, rows: u32) -> Result<(), Refusal> {
    if !ROWS.contains(&rows) {
        return Err(Refusal(
            "the tiled Sinkhorn mix takes 4, 8 or 16 rows per CTA",
        ));
    }
    if !(mix.flat_width() / CHAIN).is_multiple_of(chunk_chains(rows)) {
        return Err(Refusal(
            "the tiled Sinkhorn mix stages whole chunks of 128-K chains",
        ));
    }
    if mix.streams * mix.streams > 32 {
        return Err(Refusal(
            "the tiled Sinkhorn mix runs every Sinkhorn cell of a row in one warp",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    pub query: Query,
    /// Rows per CTA.
    pub rows: u32,
    /// Whether the entry also writes the one-CTA probe's raw projection,
    /// rstd and Sinkhorn input cells.
    pub probe: bool,
    pub params: Vec<EntryParam>,
}

impl Entry {
    fn mix(&self) -> Result<SinkhornMix, Refusal> {
        check(self.query, Op::MixCollapse { probe: self.probe })?;
        match self.query {
            Query::MixCollapse { mix, .. } => {
                fits(mix, self.rows)?;
                Ok(mix)
            }
            _ => Err(Refusal("the tiled Sinkhorn entry is a mix-collapse")),
        }
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        self.mix()?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::SinkhornHyperConnection(self.query),
                knobs: CompileKnobs {
                    tile_m: self.rows as u16,
                    tile_n: CHAIN as u16,
                    tile_k: (chunk_chains(self.rows) * CHAIN) as u16,
                    stages: 2,
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
        self.mix()?;
        check(self.query, Op::MixCollapse { probe: self.probe })
    }

    /// The entry's declared exactness, compiled without the lab's mutant
    /// defines: the one-CTA mix-collapse's bits for every row at every row
    /// count up to its capacity, wherever the row
    /// sits in its tile (each warp-sum addition combines one row's slot
    /// across lanes, the lower lanes' sum first, and every row's leader
    /// holds lane 0's tree; rows past the live count are never mixed into
    /// a live one), any chunk split and row independence (`super`'s module
    /// docs). Nothing for a probe or an entry its kernel refuses.
    pub fn invariance(&self) -> Invariance {
        match self.contract().and_then(|_| self.capacity()) {
            Ok(capacity) if !self.probe => Invariance::any_rows(capacity),
            _ => Invariance::NONE,
        }
    }

    /// Launch plan for a prepared bucket of `rows`: one CTA per
    /// [`Entry::rows`] rows. The kernel reads the live count on the device,
    /// so `live_rows` is only checked.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        if rows == 0 || rows > self.capacity()? {
            return Err(Refusal("Sinkhorn bucket exceeds the entry capacity"));
        }
        if live_rows > rows {
            return Err(Refusal("Sinkhorn live rows exceed the prepared bucket"));
        }
        let plan = LaunchPlan {
            grid: [rows.div_ceil(self.rows), 1, 1],
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
            return Err(Refusal("Sinkhorn tiled mix argument ABI mismatch"));
        }
        let args = ArgumentList(values);
        args.validate()?;
        Ok(args)
    }

    pub fn body(&self) -> Result<String, Refusal> {
        let mix = self.mix()?;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            mix.streams,
            mix.width,
            mix.iterations,
            mix.eps_bits,
            mix.norm_eps_bits,
            mix.fn_dtype == DType::F32
        );
        let probes = if self.probe {
            "projected,rstd,cells"
        } else {
            "nullptr,nullptr,nullptr"
        };
        Ok(format!(
            "chop::sinkhorn_hc::mix_collapse_tiled<{traits},{},{}>(x,fn,base,scale,collapsed,post,comb,{probes},live_rows);",
            self.rows, self.probe
        ))
    }
}

/// The tiled mix-collapse of `geometry` under `profile` at `rows` rows per
/// CTA, symbol `chop_sinkhorn_hc_{tag}_mix_collapse_tiled_r{rows}{suffix}_v1`
/// (`_probe` before the suffix with `probe`). The ABI is the one-CTA
/// entry's.
pub fn entry_with_suffix(
    profile: Profile,
    geometry: Geometry,
    rows: u32,
    probe: bool,
    suffix: &str,
) -> Entry {
    let Geometry { mix, capacity } = geometry;
    let probe_suffix = if probe { "_probe" } else { "" };
    Entry {
        profile,
        symbol: format!(
            "chop_sinkhorn_hc_{}_mix_collapse_tiled_r{rows}{probe_suffix}{suffix}_v1",
            profile.tag()
        ),
        query: Query::MixCollapse { mix, capacity },
        rows,
        probe,
        params: mix_params(mix.fn_dtype, probe),
    }
}

/// The tiled mix-collapse of `geometry` under `profile` at `rows` rows per
/// CTA.
pub fn entry(profile: Profile, geometry: Geometry, rows: u32) -> Entry {
    entry_with_suffix(profile, geometry, rows, false, "")
}

/// A cubin of `entries` named `name`, compiled with each of `defines` set
/// to 1. Two CTAs fit an SM.
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
                    2,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

/// The served shape's tiled entry at `rows` rows per CTA (one of
/// [`SERVED_TILES`]).
pub fn served(rows: u32) -> Entry {
    entry(SERVED_PROFILE, SERVED, rows)
}

/// The lab's entries: every tile of [`ROWS`], then their probes.
pub fn lab_entries() -> Vec<Entry> {
    [false, true]
        .into_iter()
        .flat_map(|probe| {
            ROWS.map(|rows| entry_with_suffix(SERVED_PROFILE, SERVED, rows, probe, ""))
        })
        .collect()
}

/// The mutant `suffix` of the served tile of `rows` rows per CTA.
pub fn mutant(rows: u32, suffix: &str) -> Entry {
    entry_with_suffix(SERVED_PROFILE, SERVED, rows, false, suffix)
}

/// The lab's cubins: `sinkhorn_hc_{tag}_tiled_sm120` with [`lab_entries`],
/// and each mutant, at every served tile, in a cubin of its own
/// (`sinkhorn_hc_{tag}_tiled{suffix}_sm120`).
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = SERVED_PROFILE.tag();
    let mut specs = vec![
        cubin_spec(
            format!("sinkhorn_hc_{tag}_tiled_sm120"),
            &lab_entries(),
            &[],
        )
        .expect("the lab's tiled entries are valid"),
    ];
    for (suffix, define) in MUTANTS {
        specs.push(
            cubin_spec(
                format!("sinkhorn_hc_{tag}_tiled{suffix}_sm120"),
                &SERVED_TILES.map(|rows| mutant(rows, suffix)),
                &[define],
            )
            .expect("the tiled mutants are valid"),
        );
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_tiled_entry_validates_and_plans_every_bucket() {
        // Every tile the schedule picks is served.
        for rows in 1..=8192 {
            if let PrefillMix::Tiled(tile) = served_prefill_mix(rows) {
                assert!(SERVED_TILES.contains(&tile), "{rows} rows");
            }
        }
        for tile in SERVED_TILES {
            served(tile).contract().unwrap();
            assert_eq!(
                served(tile).plan(960, 960).unwrap().grid,
                [960 / tile, 1, 1]
            );
        }
        let entry = served(8);
        entry.contract().unwrap();
        assert_eq!(entry.capacity().unwrap(), 8192);
        for (rows, ctas) in [
            (1, 1),
            (7, 1),
            (8, 1),
            (9, 2),
            (952, 119),
            (2048, 256),
            (8191, 1024),
            (8192, 1024),
        ] {
            let plan = entry.plan(rows, rows).unwrap();
            assert_eq!(plan.grid, [ctas, 1, 1]);
            assert_eq!(plan.block, [256, 1, 1]);
            assert_eq!(plan.dynamic_shared_bytes, 0);
            assert!(plan.pdl);
            entry.plan(rows, 0).unwrap();
        }
        assert!(entry.plan(8193, 1).is_err());
        assert!(entry.plan(0, 0).is_err());
        assert!(entry.plan(32, 33).is_err());
        assert_eq!(entry.params, mix_params(DType::Bf16, false));
        let args = (0..entry.params.len() as u64)
            .map(Argument::DeviceBuffer)
            .collect::<Vec<_>>();
        entry.pack_arguments(args).unwrap();
        assert!(entry.pack_arguments(vec![Argument::U32(1)]).is_err());
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r8_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_tiled<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>,8,false>(x,fn,base,scale,collapsed,post,comb,nullptr,nullptr,nullptr,live_rows);"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_hc_hc4x4096_tiled_sm120",
                "sinkhorn_hc_hc4x4096_tiled_levels_reversed_sm120",
                "sinkhorn_hc_hc4x4096_tiled_late_wait_sm120",
            ]
        );
        assert_eq!(
            specs[0]
                .entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r4_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r8_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r16_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r4_probe_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r8_probe_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r16_probe_v1",
            ]
        );
        assert_eq!(
            specs[1].defines,
            [(LEVELS_REVERSED_DEFINE.into(), "1".into())]
        );
        assert_eq!(specs[2].defines, [(LATE_WAIT_DEFINE.into(), "1".into())]);
        assert_eq!(
            specs[2]
                .entries
                .iter()
                .map(|e| e.symbol.as_str())
                .collect::<Vec<_>>(),
            [
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r4_late_wait_v1",
                "chop_sinkhorn_hc_hc4x4096_mix_collapse_tiled_r8_late_wait_v1",
            ]
        );
        let probe = &lab_entries()[3];
        assert_eq!(probe.params, mix_params(DType::Bf16, true));
        assert!(probe.body().unwrap().contains(",4,true>("));
        assert!(
            probe
                .body()
                .unwrap()
                .ends_with("(x,fn,base,scale,collapsed,post,comb,projected,rstd,cells,live_rows);")
        );
    }

    #[test]
    fn the_served_prefill_mix_steps_through_the_split_and_both_tiles() {
        assert_eq!(served_prefill_mix(1), PrefillMix::Split);
        assert_eq!(served_prefill_mix(143), PrefillMix::Split);
        assert_eq!(served_prefill_mix(144), PrefillMix::Tiled(4));
        assert_eq!(served_prefill_mix(952), PrefillMix::Tiled(4));
        assert_eq!(served_prefill_mix(1504), PrefillMix::Tiled(4));
        assert_eq!(served_prefill_mix(1505), PrefillMix::Tiled(8));
        assert_eq!(served_prefill_mix(1920), PrefillMix::Tiled(8));
        assert_eq!(served_prefill_mix(8192), PrefillMix::Tiled(8));
        for rows in 1..=8192 {
            if let PrefillMix::Tiled(tile) = served_prefill_mix(rows) {
                entry(SERVED_PROFILE, SERVED, tile)
                    .plan(rows, rows)
                    .unwrap();
            }
        }
    }

    #[test]
    fn tiled_entry_refuses_tiles_and_mixers_it_does_not_run() {
        for rows in [0, 1, 2, 12, 32] {
            let entry = entry(SERVED_PROFILE, SERVED, rows);
            assert!(entry.contract().is_err(), "{rows} rows per CTA");
            assert!(entry.plan(1, 1).is_err());
            assert!(entry.body().is_err());
        }
        for rows in ROWS {
            entry(SERVED_PROFILE, SERVED, rows).contract().unwrap();
        }
        let geometry = |mix: SinkhornMix| Geometry { mix, capacity: 32 };
        // Two streams of 256: four chains, a chunk at 16 rows per CTA but
        // half of one at 8.
        let narrow = SinkhornMix {
            streams: 2,
            width: 256,
            ..SERVED.mix
        };
        assert!(
            entry(SERVED_PROFILE, geometry(narrow), 8)
                .contract()
                .is_err()
        );
        assert!(
            entry(SERVED_PROFILE, geometry(narrow), 16)
                .contract()
                .is_ok()
        );
        // Six streams: 36 Sinkhorn cells exceed a warp.
        let six = SinkhornMix {
            streams: 6,
            width: 1024,
            ..SERVED.mix
        };
        assert!(entry(SERVED_PROFILE, geometry(six), 8).contract().is_err());
        // What the one-CTA entry refuses.
        let wide = SinkhornMix {
            streams: 8,
            ..SERVED.mix
        };
        assert!(entry(SERVED_PROFILE, geometry(wide), 8).contract().is_err());
        let f32_fn = entry(
            SERVED_PROFILE,
            geometry(SinkhornMix {
                fn_dtype: DType::F32,
                ..SERVED.mix
            }),
            8,
        );
        f32_fn.contract().unwrap();
        assert!(f32_fn.body().unwrap().contains(",true>,8,false>("));
    }
}
