//! The mix-collapse split over CTAs (`cuda/sinkhorn_hc/mix_split.cuh`):
//! the same `MixCollapse` query and output bits as the one-CTA entry, with
//! a row's 128-K chains spread over [`groups`] CTAs that write per-chain
//! sums to a partials table, and the row's last CTA (elected by a ticket)
//! adding them in K order and finishing the row. See
//! `cuda/sinkhorn_hc/README.md` ("Split mix-collapse").

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, CompileKey, CompileKnobs, InstantiationSpec, LaunchPlan,
    NumericalContract, OpFamily, Refusal, StaticQuery, Target, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::{SinkhornHyperConnectionQuery as Query, SinkhornMix};

use super::{CHAIN, Geometry, Op, SERVED, SERVED_PROFILE, THREADS, check, mix_params, ptr};
use crate::Profile;

/// Chains of one CTA: one per warp.
pub const CHAINS_PER_CTA: u32 = THREADS / 32;

/// CTAs per row of `mix`: its 128-K chains, eight per CTA.
pub fn groups(mix: SinkhornMix) -> u32 {
    mix.flat_width() / CHAIN / CHAINS_PER_CTA
}

/// Static shared bytes the finishing CTA's row and staged chain sums may
/// take (`Split<M>`'s assertion in `mix_split.cuh`).
pub const SHARED_LIMIT: u32 = 46 * 1024;

/// Static shared bytes of `mix`'s finishing CTA: the row's streams (BF16)
/// and the staged chain sums, one padding word per mix.
pub fn shared_bytes(mix: SinkhornMix) -> u64 {
    let chains = u64::from(mix.flat_width() / CHAIN);
    u64::from(mix.flat_width()) * 2 + u64::from(mix.mixes()) * (chains + 1) * 4
}

/// Refuses a mixer the split kernels cannot hold: whole groups of eight
/// 128-K chains, and the finishing CTA's static shared memory within
/// [`SHARED_LIMIT`], as `mix_split.cuh` asserts.
pub fn fits(mix: SinkhornMix) -> Result<(), Refusal> {
    if !mix.flat_width().is_multiple_of(CHAIN * CHAINS_PER_CTA) {
        return Err(Refusal(
            "the split Sinkhorn mix needs whole groups of eight 128-K chains",
        ));
    }
    if shared_bytes(mix) > u64::from(SHARED_LIMIT) {
        return Err(Refusal(
            "the split Sinkhorn mix's row and chain sums exceed 46 KiB of shared memory",
        ));
    }
    Ok(())
}

/// Bytes of the F32 chain sums of `rows` rows (`[rows, mixes, chains]`),
/// which every launch rewrites before it reads them.
pub fn partials_bytes(mix: SinkhornMix, rows: u32) -> u64 {
    u64::from(rows) * u64::from(mix.mixes()) * u64::from(mix.flat_width() / CHAIN) * 4
}

/// Bytes of the tickets of `rows` rows (`[rows]` u32): zeroed once at
/// allocation; every launch leaves them zeroed.
pub fn tickets_bytes(rows: u32) -> u64 {
    u64::from(rows) * 4
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub profile: Profile,
    pub symbol: String,
    pub query: Query,
    pub params: Vec<EntryParam>,
}

impl Entry {
    fn mix(&self) -> Result<SinkhornMix, Refusal> {
        super::limits(self.query)?;
        match self.query {
            Query::MixCollapse { mix, .. } => {
                fits(mix)?;
                Ok(mix)
            }
            _ => Err(Refusal("the split Sinkhorn entry is a mix-collapse")),
        }
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        check(self.query, Op::MixCollapse { probe: false })?;
        let mix = self.mix()?;
        let spec = InstantiationSpec {
            family: OpFamily::Row,
            key: CompileKey {
                query: StaticQuery::SinkhornHyperConnection(self.query),
                knobs: CompileKnobs {
                    tile_m: 1,
                    tile_n: CHAIN as u16,
                    tile_k: 1,
                    stages: 1,
                    swizzle: 0,
                    warps: (THREADS / 32) as u8,
                    split_k_capacity: u8::try_from(groups(mix))
                        .map_err(|_| Refusal("the split Sinkhorn mix has over 255 groups"))?,
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
        check(self.query, Op::MixCollapse { probe: false })
    }

    /// The entry's declared exactness: the one-CTA mix-collapse's bits for
    /// every row at every row count up to its capacity (the finisher folds
    /// a row's chain sums from zero in K order, whichever CTA arrives
    /// last; the ticket only elects it), any chunk split and row
    /// independence (`super`'s module docs). Nothing for an entry its
    /// kernel refuses.
    pub fn invariance(&self) -> Invariance {
        match self.contract().and_then(|_| self.capacity()) {
            Ok(capacity) => Invariance::any_rows(capacity),
            Err(_) => Invariance::NONE,
        }
    }

    /// Launch plan for a prepared bucket of `rows`: `groups` CTAs per row,
    /// rows on the y axis. The kernels read the live count on the device,
    /// so `live_rows` is only checked.
    pub fn plan(&self, rows: u32, live_rows: u32) -> Result<LaunchPlan, Refusal> {
        let mix = self.mix()?;
        self.contract()?;
        if rows == 0 || rows > self.capacity()? {
            return Err(Refusal("Sinkhorn bucket exceeds the entry capacity"));
        }
        if live_rows > rows {
            return Err(Refusal("Sinkhorn live rows exceed the prepared bucket"));
        }
        let plan = LaunchPlan {
            grid: [groups(mix), rows, 1],
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
            return Err(Refusal("Sinkhorn split mix argument ABI mismatch"));
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
            mix.fn_dtype == kernel_core::contract::DType::F32
        );
        Ok(format!(
            "chop::sinkhorn_hc::mix_collapse_split<{traits}>(x,fn,base,scale,collapsed,post,comb,partials,tickets,live_rows);"
        ))
    }
}

/// The split entry ABI: the mix-collapse's, then the chain sums and the
/// tickets before the live row count.
pub fn params(mix: SinkhornMix) -> Vec<EntryParam> {
    let mut params = mix_params(mix.fn_dtype, false);
    let live = params.pop().expect("the live row count");
    params.extend([ptr("partials", "float"), ptr("tickets", "uint32_t"), live]);
    params
}

/// The split mix-collapse of `geometry` under `profile`, symbol
/// `chop_sinkhorn_hc_{tag}_mix_collapse_split_v1`.
pub fn entry(profile: Profile, geometry: Geometry) -> Entry {
    let Geometry { mix, capacity } = geometry;
    Entry {
        profile,
        symbol: format!("chop_sinkhorn_hc_{}_mix_collapse_split_v1", profile.tag()),
        query: Query::MixCollapse { mix, capacity },
        params: params(mix),
    }
}

/// A cubin of `entries` named `name`.
pub fn cubin_spec(name: String, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/mix_split.cuh\"".into()],
        defines: vec![],
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

/// The served shape's split entry.
pub fn served() -> Entry {
    entry(SERVED_PROFILE, SERVED)
}

/// The lab's cubin of the served split entry,
/// `sinkhorn_hc_{tag}_split_sm120`.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let name = format!("sinkhorn_hc_{}_split_sm120", SERVED_PROFILE.tag());
    vec![cubin_spec(name, &[served()]).expect("the served split entry is valid")]
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::DType;

    #[test]
    fn served_split_entry_validates_and_plans_every_bucket() {
        let entry = served();
        entry.contract().unwrap();
        assert_eq!(entry.capacity().unwrap(), 8192);
        assert_eq!(groups(SERVED.mix), 16);
        assert_eq!(partials_bytes(SERVED.mix, 1), 24 * 128 * 4);
        assert_eq!(tickets_bytes(32), 128);
        for rows in [1, 2, 4, 16, 32, 6 * 32, 2048, 8192] {
            let plan = entry.plan(rows, rows).unwrap();
            assert_eq!(plan.grid, [16, rows, 1]);
            assert_eq!(plan.block, [256, 1, 1]);
            assert_eq!(plan.dynamic_shared_bytes, 0);
            assert!(plan.pdl);
            entry.plan(rows, 0).unwrap();
        }
        assert!(entry.plan(8193, 1).is_err());
        assert!(entry.plan(0, 0).is_err());
        assert!(entry.plan(32, 33).is_err());
        let names = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } => name.as_str(),
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
                "collapsed",
                "post",
                "comb",
                "partials",
                "tickets",
                "live_rows"
            ]
        );
        let args = (0..entry.params.len() as u64)
            .map(Argument::DeviceBuffer)
            .collect::<Vec<_>>();
        entry.pack_arguments(args).unwrap();
        assert!(entry.pack_arguments(vec![Argument::U32(1)]).is_err());
        assert_eq!(
            entry.symbol,
            "chop_sinkhorn_hc_hc4x4096_mix_collapse_split_v1"
        );
        assert_eq!(
            entry.body().unwrap(),
            "chop::sinkhorn_hc::mix_collapse_split<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>>(x,fn,base,scale,collapsed,post,comb,partials,tickets,live_rows);"
        );
        let specs = lab_cubin_specs();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].name, "sinkhorn_hc_hc4x4096_split_sm120");
        assert_eq!(specs[0].entries.len(), 1);
    }

    #[test]
    fn split_refuses_mixers_whose_chain_sums_exceed_shared_memory() {
        assert_eq!(shared_bytes(SERVED.mix), 32_768 + 24 * 129 * 4);
        fits(SERVED.mix).unwrap();
        // 8 streams of 2048 pass the one-CTA entry's 40 KiB stream row but
        // need 74,048 bytes with the staged chain sums.
        let wide = SinkhornMix {
            streams: 8,
            width: 2048,
            ..SERVED.mix
        };
        assert_eq!(shared_bytes(wide), 74_048);
        let one_cta = super::super::entries(
            SERVED_PROFILE,
            Geometry {
                mix: wide,
                capacity: 32,
            },
            false,
        );
        one_cta[0].contract().unwrap();
        let split = entry_with(wide);
        assert_eq!(
            split.contract().unwrap_err(),
            Refusal("the split Sinkhorn mix's row and chain sums exceed 46 KiB of shared memory")
        );
        assert!(split.plan(1, 1).is_err());
        assert!(split.body().is_err());
    }

    #[test]
    fn split_refuses_what_the_one_cta_entry_refuses() {
        let wide = SinkhornMix {
            streams: 8,
            ..SERVED.mix
        };
        let entry = entry(
            SERVED_PROFILE,
            Geometry {
                mix: wide,
                capacity: 1,
            },
        );
        assert!(entry.contract().is_err());
        let f32_fn = entry_with(SinkhornMix {
            fn_dtype: DType::F32,
            ..SERVED.mix
        });
        f32_fn.contract().unwrap();
        assert_eq!(f32_fn.params[1], ptr("fn", "const float"));
    }

    fn entry_with(mix: SinkhornMix) -> Entry {
        entry(SERVED_PROFILE, Geometry { mix, capacity: 32 })
    }
}
