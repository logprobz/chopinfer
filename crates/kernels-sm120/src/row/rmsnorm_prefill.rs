//! Prefill-only BF16 norm/ceil fusion with its original materialized outputs.
use super::rmsnorm_quantize;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal};
use kernel_core::rmsnorm_quantize::RmsNormQuantizeQuery;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    RoundBeforeWeight,
    ClearStatus,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: rmsnorm_quantize::Entry,
    pub cache_groups: bool,
    pub mutation: Mutation,
}
impl Entry {
    pub fn new(
        query: RmsNormQuantizeQuery,
        capacity: u32,
        symbol: impl Into<String>,
        cache_groups: bool,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            reference: rmsnorm_quantize::Entry::new(query, capacity, symbol)?,
            cache_groups,
            mutation: Mutation::None,
        };
        entry.validate()?;
        Ok(entry)
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        if self.reference.query.norm.input != DType::Bf16 {
            return Err(Refusal(
                "prefill norm fusion requires the BF16 input boundary",
            ));
        }
        let width = u64::from(self.reference.query.norm.width);
        let shared = width * 2
            + if self.cache_groups {
                width + width / 32
            } else {
                2
            };
        if shared + 2048 > 48 * 1024 {
            return Err(Refusal(
                "prefill norm staging exceeds default shared memory",
            ));
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.reference.plan(rows)
    }
    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let mut spec = self.reference.spec();
        let width = self.reference.query.norm.width;
        let floor = self.reference.query.amax_floor_bits;
        let cached = self.cache_groups;
        let mutation = match self.mutation {
            Mutation::None => "none",
            Mutation::RoundBeforeWeight => "round_before_weight",
            Mutation::ClearStatus => "clear_status",
        };
        spec.body = format!(
            "chop::row::rmsnorm_prefill::run<{width},{floor:#x}u,{cached},chop::row::rmsnorm_prefill::Mutation::{mutation}>(input,weight,normed,widened,codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits));"
        );
        Ok(spec)
    }
}
pub fn cubin_spec(name: String, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("prefill norm family requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.reference.symbol) {
            return Err(Refusal("duplicate prefill norm entry"));
        }
    }
    Ok(CubinSpec {
        name,
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"row/rmsnorm_prefill.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{contract::RmsNormKind, rmsnorm::RmsNormRequirement};
    fn query() -> RmsNormQuantizeQuery {
        RmsNormQuantizeQuery {
            norm: RmsNormRequirement::bf16_backend_chosen(5120, 1e-20, RmsNormKind::Plain, false),
            amax_floor_bits: 1e-4f32.to_bits(),
        }
    }
    #[test]
    fn prefill_norm_keeps_materialized_abi_and_fixed_tree() {
        for cached in [false, true] {
            let e = Entry::new(query(), 8192, "prefill_norm", cached).unwrap();
            assert_eq!(e.spec().unwrap().params, e.reference.spec().params);
            for rows in [1, 64, 192, 320, 1024, 2048, 2496, 2624, 4096, 8191, 8192] {
                let p = e.plan(rows).unwrap();
                assert_eq!(p, e.reference.plan(rows).unwrap());
                assert_eq!(p.block, [256, 1, 1]);
                assert_eq!(p.grid, [rows, 1, 1]);
                assert_eq!(p.cluster, [1, 1, 1]);
                assert_eq!(p.dynamic_shared_bytes, 0);
                assert_eq!(p.workspace_bytes, 0);
                assert!(p.pdl);
            }
            assert!(e.plan(0).is_err());
            assert!(e.plan(8193).is_err());
            let mut wrong = query();
            wrong.norm.input = DType::F32;
            wrong.norm.output = DType::F32;
            assert!(Entry::new(wrong, 8192, "wrong", cached).is_err());
            let mut wide = query();
            wide.norm.width = 32768;
            assert!(Entry::new(wide, 8192, "wide", cached).is_err());
            wide.norm.width = u32::MAX - 31;
            assert!(Entry::new(wide, 8192, "overflow", cached).is_err());
        }
    }
}
