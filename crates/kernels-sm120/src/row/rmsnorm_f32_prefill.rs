//! F32 RMSNorm and ceil with a row-local shared arena and the original tree.
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
    pub cache_input: bool,
    pub mutation: Mutation,
}

impl Entry {
    pub fn new(
        query: RmsNormQuantizeQuery,
        capacity: u32,
        symbol: impl Into<String>,
        cache_input: bool,
    ) -> Result<Self, Refusal> {
        let entry = Self {
            reference: rmsnorm_quantize::Entry::new(query, capacity, symbol)?,
            cache_input,
            mutation: Mutation::None,
        };
        entry.validate()?;
        Ok(entry)
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        self.reference.validate()?;
        if self.reference.query.norm.input != DType::F32 {
            return Err(Refusal("F32 prefill fusion requires F32 input and output"));
        }
        let width = u64::from(self.reference.query.norm.width);
        // Include compiler-owned storage for CTA vote collectives.
        if width * 5 + width / 32 + 2048 > 48 * 1024 {
            return Err(Refusal("F32 prefill arena exceeds default shared memory"));
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
        let cached = self.cache_input;
        let mutation = match self.mutation {
            Mutation::None => "none",
            Mutation::RoundBeforeWeight => "round_before_weight",
            Mutation::ClearStatus => "clear_status",
        };
        spec.body = format!(
            "chop::row::rmsnorm_f32_prefill::run<{width},{floor:#x}u,{cached},chop::row::rmsnorm_f32_prefill::Mutation::{mutation}>(input,weight,normed,widened,codes,scales,live_rows,status,status_row_offset,__uint_as_float(eps_bits));"
        );
        Ok(spec)
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("F32 prefill family requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for entry in entries {
        entry.validate()?;
        if !symbols.insert(&entry.reference.symbol) {
            return Err(Refusal("duplicate F32 prefill entry"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"row/rmsnorm_f32_prefill.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{contract::RmsNormKind, rmsnorm::RmsNormRequirement};

    #[test]
    fn admission_keeps_the_original_tree_abi_and_private_arena() {
        let mut query = RmsNormQuantizeQuery {
            norm: RmsNormRequirement::bf16_backend_chosen(1280, 1e-20, RmsNormKind::Plain, false),
            amax_floor_bits: 1e-4f32.to_bits(),
        };
        assert!(Entry::new(query, 8192, "wrong", true).is_err());
        query.norm.input = DType::F32;
        query.norm.output = DType::F32;
        for cached in [false, true] {
            let entry = Entry::new(query, 8192, "fused", cached).unwrap();
            assert_eq!(entry.spec().unwrap().params, entry.reference.spec().params);
            for rows in [1, 6, 32, 64, 8191, 8192] {
                let plan = entry.plan(rows).unwrap();
                assert_eq!(plan.grid, [rows, 1, 1]);
                assert_eq!(plan.block, [256, 1, 1]);
                assert_eq!(plan.workspace_bytes, 0);
                assert_eq!(plan.dynamic_shared_bytes, 0);
            }
            assert!(entry.plan(0).is_err());
            assert!(entry.plan(8193).is_err());
        }
        query.norm.width = 32768;
        assert!(Entry::new(query, 8192, "too_wide", true).is_err());
    }
}
