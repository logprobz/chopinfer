//! Lagged mix-collapse split over independent 128-K chains. The finishing
//! CTA folds chain sums in K order and preserves the one-CTA entry's bits.

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{
    Argument, ArgumentList, InstantiationSpec, LaunchPlan, Refusal, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;

use super::{CHAIN, Geometry, THREADS, lagged, ptr, split};
use crate::Profile;

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: lagged::Entry,
}

impl Entry {
    pub fn query(&self) -> Query {
        self.reference.query
    }

    pub fn symbol(&self) -> String {
        self.reference.symbol.clone()
    }

    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let mut spec = self.reference.contract()?;
        let Query::MixCollapse { mix, .. } = self.query() else {
            return Err(Refusal("the lagged split entry requires a mix-collapse"));
        };
        if !matches!(self.reference.op, lagged::Op::MixCollapse { .. }) {
            return Err(Refusal("the lagged split entry requires a mix-collapse"));
        }
        if !mix
            .flat_width()
            .is_multiple_of(CHAIN * split::CHAINS_PER_CTA)
        {
            return Err(Refusal(
                "the lagged split needs whole groups of eight chains",
            ));
        }
        if u64::from(mix.mixes()) * u64::from(mix.flat_width() / CHAIN + 1) * 4
            > u64::from(split::SHARED_LIMIT)
        {
            return Err(Refusal("the lagged split chain table exceeds 46 KiB"));
        }
        spec.key.knobs.split_k_capacity = u8::try_from(split::groups(mix))
            .map_err(|_| Refusal("the lagged split has over 255 groups"))?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.contract()?;
        self.reference.capacity()
    }

    pub fn invariance(&self) -> Invariance {
        if self.contract().is_ok() {
            self.reference.invariance()
        } else {
            Invariance::NONE
        }
    }

    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let mut plan = self.reference.plan(rows, live)?;
        let Query::MixCollapse { mix, .. } = self.query() else {
            unreachable!("validated mix-collapse");
        };
        plan.grid = [split::groups(mix), rows, 1];
        Ok(plan)
    }

    pub fn params(&self) -> Vec<EntryParam> {
        let mut params = self.reference.params.clone();
        let live = params.pop().expect("live count");
        params.extend([ptr("partials", "float"), ptr("tickets", "uint32_t"), live]);
        params
    }

    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        let mut entry = self.reference.clone();
        entry.params = self.params();
        entry.pack_arguments(values)
    }

    pub fn body(&self) -> Result<String, Refusal> {
        self.contract()?;
        Ok(self
            .reference
            .body()
            .replace("mix_collapse_lagged<", "mix_collapse_lagged_split<")
            .replace(",live_rows);", ",partials,tickets,live_rows);"))
    }
}

pub fn entry(profile: Profile, geometry: Geometry, probe: bool) -> Entry {
    let mut reference = lagged::entries(profile, geometry, probe)
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .expect("lagged mix-collapse");
    reference.symbol = reference
        .symbol
        .replace("lagged_mix_collapse", "lagged_mix_collapse_split");
    Entry { reference }
}

pub fn cubin_spec(name: String, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/lagged_split.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(|e| {
                Ok(EntrySpec::new(
                    e.symbol(),
                    e.params(),
                    e.body()?,
                    THREADS,
                    1,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_plans_the_full_row_without_staging_streams() {
        let entry = entry(lagged::LAB_PROFILE, lagged::LAB, false);
        assert!(split::fits(lagged::LAB.mix).is_err());
        entry.contract().unwrap();
        for rows in [1, 2, 31, 32, 33, 128, 8192] {
            assert_eq!(entry.plan(rows, 0).unwrap().grid, [20, rows, 1]);
        }
        assert!(entry.plan(0, 0).is_err());
        assert!(entry.plan(32, 33).is_err());
        assert!(entry.plan(8193, 0).is_err());
        assert_eq!(entry.invariance(), Invariance::any_rows(8192));
        assert_eq!(entry.params().len(), 12);
        assert!(entry.pack_arguments(vec![]).is_err());
        assert_eq!(split::partials_bytes(lagged::LAB.mix, 32), 491_520);
    }
}
