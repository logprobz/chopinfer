//! The prefill expansion-then-mix tile followed by the input norm and ceil
//! quantizer of its rows in one launch
//! (`cuda/sinkhorn_hc/prefill_expand_mix_norm.cuh`). The ABI is
//! `prefill_fused`'s.
use super::{THREADS, prefill_expand_mix, ptr};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{
    contract::{DType, LaunchPlan, Refusal, Validate},
    sinkhorn_expand_mix::{SinkhornExpandMixQuery, SinkhornRoundNormQuery},
};

/// Lab controls; each changes an output on finite rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    /// The stream sum in reverse stream order.
    ReverseExpansion,
    /// The normalized value rounded to BF16 before the weight multiply.
    RoundBeforeWeight,
    /// The prior row status cleared before the quantizer's commit.
    ClearStatus,
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub query: SinkhornRoundNormQuery,
    pub capacity: u32,
    pub tile_rows: u32,
    pub chains: u32,
    pub symbol: String,
    pub mutation: Mutation,
}
impl Entry {
    /// The expansion-then-mix tile this entry runs first.
    pub fn mixer(&self) -> prefill_expand_mix::Entry {
        prefill_expand_mix::Entry {
            query: SinkhornExpandMixQuery {
                mix: self.query.mixer.mix,
            },
            capacity: self.capacity,
            tile_rows: self.tile_rows,
            chains: self.chains,
            symbol: self.symbol.clone(),
            mutation: prefill_expand_mix::Mutation::None,
        }
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        self.mixer().validate()?;
        crate::row::rmsnorm_prefill::Entry::new(
            self.query.norm,
            self.capacity,
            &self.symbol,
            true,
        )?;
        let n = self.query.norm.norm;
        let m = self.query.mixer.mix;
        // The tail stages a row's codes and scales; its arena and the
        // pipeline's stay within static shared memory.
        let tail = u64::from(m.width) + u64::from(m.width / 32) + 64;
        let pipeline = 2 * u64::from(self.tile_rows) * u64::from(self.chains * super::CHAIN) * 2
            + u64::from(self.tile_rows) * u64::from(9 + m.mixes() + m.streams) * 4
            + 256;
        if n.input != DType::Bf16
            || n.width != m.width
            || !m.width.is_multiple_of(16)
            || tail + pipeline > 48 * 1024
        {
            return Err(Refusal(
                "prefill expand-mix norm needs the mixer's BF16 width within static shared memory",
            ));
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        self.mixer().plan(rows, live)
    }
    pub fn spec(&self) -> Result<EntrySpec, Refusal> {
        self.validate()?;
        let m = self.query.mixer.mix;
        let traits = format!(
            "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
            m.streams,
            m.width,
            m.iterations,
            m.eps_bits,
            m.norm_eps_bits,
            m.fn_dtype == DType::F32
        );
        let mutation = match self.mutation {
            Mutation::None => "none",
            Mutation::ReverseExpansion => "reverse_expansion",
            Mutation::RoundBeforeWeight => "round_before_weight",
            Mutation::ClearStatus => "clear_status",
        };
        let mut params = [
            ("x", "const uint16_t"),
            ("y", "const uint16_t"),
            ("post_in", "const float"),
            ("comb_in", "const float"),
            ("expanded", "uint16_t"),
            ("fn", "const void"),
            ("base", "const float"),
            ("scale", "const float"),
            ("pre_in", "const float"),
            ("collapsed", "uint16_t"),
            ("pre", "float"),
            ("post", "float"),
            ("comb", "float"),
            ("norm_weight", "const uint16_t"),
            ("normed", "uint16_t"),
            ("widened", "float"),
            ("codes", "uint8_t"),
            ("scales", "uint8_t"),
            ("live_rows", "const uint32_t"),
            ("status", "uint32_t"),
        ]
        .into_iter()
        .map(|(n, t)| ptr(n, t))
        .collect::<Vec<_>>();
        params.extend([
            EntryParam::U64 {
                name: "status_offset".into(),
            },
            EntryParam::U32 {
                name: "norm_eps_bits".into(),
            },
        ]);
        Ok(EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::sinkhorn_hc::prefill_expand_mix_norm::run<{traits},{},{},{:#x}u,chop::sinkhorn_hc::prefill_expand_mix_norm::Mutation::{mutation}>(x,y,post_in,comb_in,expanded,fn,base,scale,pre_in,collapsed,pre,post,comb,norm_weight,normed,widened,codes,scales,live_rows,status,status_offset,norm_eps_bits);",
                self.tile_rows, self.chains, self.query.norm.amax_floor_bits
            ),
            THREADS,
            1,
        ))
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("prefill expand-mix norm family requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for e in entries {
        if !symbols.insert(&e.symbol) {
            return Err(Refusal("duplicate prefill expand-mix norm symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/prefill_expand_mix_norm.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        contract::RmsNormKind, rmsnorm::RmsNormRequirement, rmsnorm_quantize::RmsNormQuantizeQuery,
    };
    fn entry(tile_rows: u32, chains: u32) -> Entry {
        let mix = super::super::lagged::LAB.mix;
        Entry {
            query: SinkhornRoundNormQuery {
                mixer: SinkhornExpandMixQuery { mix },
                norm: RmsNormQuantizeQuery {
                    norm: RmsNormRequirement::bf16_backend_chosen(
                        mix.width,
                        1e-20,
                        RmsNormKind::Plain,
                        false,
                    ),
                    amax_floor_bits: 1e-4f32.to_bits(),
                },
            },
            capacity: 8192,
            tile_rows,
            chains,
            symbol: "expand_mix_norm".into(),
            mutation: Mutation::None,
        }
    }
    #[test]
    fn tiles_keep_the_fused_abi_and_the_mixer_plan() {
        for (tile_rows, chains) in [(4, 4), (8, 4), (8, 8)] {
            let e = entry(tile_rows, chains);
            assert_eq!(e.spec().unwrap().params.len(), 22);
            for rows in [1, 7, 64, 192, 320, 2496, 8191, 8192] {
                for live in [0, rows - 1, rows] {
                    assert_eq!(
                        e.plan(rows, live).unwrap(),
                        e.mixer().plan(rows, live).unwrap()
                    );
                }
                assert!(e.plan(rows, rows + 1).is_err());
            }
        }
        let mut narrow = entry(8, 8);
        narrow.query.norm.norm.width = 4096;
        assert!(entry(8, 8).validate().is_ok());
        assert!(narrow.validate().is_err());
        assert!(cubin_spec("m", &[entry(8, 8), entry(8, 8)]).is_err());
    }
}
