//! Exact prefill expansion, lagged mixing and BF16 norm/ceil in a private CTA arena.
use super::{CHAIN, THREADS, ptr, shape_limits};
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::{
    contract::{DType, LaunchPlan, Refusal, Validate},
    sinkhorn_expand_mix::SinkhornRoundNormQuery,
    sinkhorn_hyper_connection::SinkhornHyperConnectionQuery,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    ReverseExpansion,
    RoundBeforeWeight,
    ClearStatus,
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub query: SinkhornRoundNormQuery,
    pub capacity: u32,
    pub tile_rows: u32,
    pub symbol: String,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        let m = self.query.mixer.mix;
        shape_limits(SinkhornHyperConnectionQuery::MixCollapse {
            mix: m,
            capacity: self.capacity,
        })?;
        crate::row::rmsnorm_prefill::Entry::new(
            self.query.norm,
            self.capacity,
            &self.symbol,
            true,
        )?;
        let r = u64::from(self.tile_rows);
        let projection = 4 * r * 4 * u64::from(CHAIN)
            + r * u64::from(9 + m.mixes() + m.streams + m.streams * m.streams) * 4;
        let tail = u64::from(m.width) * 3 + u64::from(m.width / 32) + 64;
        if ![1, 4, 8].contains(&self.tile_rows)
            || self.capacity == 0
            || self.capacity > 8192
            || m.streams > 4
            || !m.flat_width().is_multiple_of(CHAIN * 4)
            || projection.max(tail) + r * u64::from(m.streams) * 4 + 2048 > 48 * 1024
        {
            return Err(Refusal(
                "prefill fused tile exceeds the private arena contract",
            ));
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if rows == 0 || rows > self.capacity || live > rows {
            return Err(Refusal(
                "prefill fused live rows must fit the grid capacity",
            ));
        }
        let plan = LaunchPlan {
            grid: [rows.div_ceil(self.tile_rows), 1, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: true,
        };
        crate::backend::check_plan(&plan)?;
        Ok(plan)
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
                "chop::sinkhorn_hc::prefill_fused::run<{traits},{},4,{:#x}u,chop::sinkhorn_hc::prefill_fused::Mutation::{mutation}>(x,y,post_in,comb_in,expanded,fn,base,scale,pre_in,collapsed,pre,post,comb,norm_weight,normed,widened,codes,scales,live_rows,status,status_offset,norm_eps_bits);",
                self.tile_rows, self.query.norm.amax_floor_bits
            ),
            THREADS,
            1,
        ))
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("prefill fused family requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for e in entries {
        if !symbols.insert(&e.symbol) {
            return Err(Refusal("duplicate prefill fused symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/prefill_fused.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::{
        contract::RmsNormKind, rmsnorm::RmsNormRequirement, rmsnorm_quantize::RmsNormQuantizeQuery,
        sinkhorn_expand_mix::SinkhornExpandMixQuery,
    };
    #[test]
    fn private_tiles_reject_impossible_rows_and_keep_the_full_abi() {
        for tile_rows in [1, 4, 8] {
            let e = Entry {
                query: SinkhornRoundNormQuery {
                    mixer: SinkhornExpandMixQuery {
                        mix: super::super::lagged::LAB.mix,
                    },
                    norm: RmsNormQuantizeQuery {
                        norm: RmsNormRequirement::bf16_backend_chosen(
                            5120,
                            1e-20,
                            RmsNormKind::Plain,
                            false,
                        ),
                        amax_floor_bits: 1e-4f32.to_bits(),
                    },
                },
                capacity: 8192,
                tile_rows,
                symbol: "fused".into(),
                mutation: Mutation::None,
            };
            assert_eq!(e.spec().unwrap().params.len(), 22);
            for rows in [1, 7, 64, 192, 320, 2240, 2496, 2624, 8191, 8192] {
                for live in [0, rows - 1, rows] {
                    let p = e.plan(rows, live).unwrap();
                    assert_eq!(p.grid, [rows.div_ceil(tile_rows), 1, 1]);
                    assert_eq!(p.workspace_bytes, 0);
                    assert_eq!(p.dynamic_shared_bytes, 0);
                }
                assert!(e.plan(rows, rows + 1).is_err());
            }
            assert!(e.plan(0, 0).is_err());
            assert!(e.plan(8193, 8193).is_err());
            let mut invalid = e.clone();
            invalid.tile_rows = 16;
            assert!(invalid.validate().is_err());
        }
    }
    #[test]
    fn panel_mapping_preserves_each_rms_lane_chain() {
        for lane in 0..256 {
            let original: Vec<_> = (lane..20480 / 8).step_by(256).collect();
            let panels: Vec<_> = (0..40)
                .filter(|chunk| lane / 64 == chunk % 4)
                .map(|chunk| chunk * 64 + lane % 64)
                .collect();
            assert_eq!(original, panels);
        }
    }
    #[test]
    fn row_groups_keep_the_original_projection_expression_tree() {
        let add = |a: &str, b: &str| format!("({a}+{b})");
        for rows in [1usize, 4, 8] {
            let initial: Vec<Vec<String>> = (0..32)
                .map(|lane| (0..rows).map(|r| format!("row{r}:lane{lane}")).collect())
                .collect();
            let mut reference = initial.clone();
            for offset in [16, 8, 4, 2, 1] {
                reference = (0..32)
                    .map(|lane| {
                        (0..rows)
                            .map(|r| add(&reference[lane][r], &reference[lane ^ offset][r]))
                            .collect()
                    })
                    .collect();
            }
            let mut candidate = initial;
            let levels = rows.ilog2() as usize;
            for level in 0..levels {
                let offset = 16 >> level;
                let half = rows >> (level + 1);
                candidate = (0..32)
                    .map(|lane| {
                        (0..half)
                            .map(|r| {
                                if lane & offset == 0 {
                                    add(&candidate[lane][r], &candidate[lane ^ offset][r])
                                } else {
                                    add(
                                        &candidate[lane ^ offset][r + half],
                                        &candidate[lane][r + half],
                                    )
                                }
                            })
                            .collect()
                    })
                    .collect();
            }
            for level in levels..5 {
                let offset = 16 >> level;
                candidate = (0..32)
                    .map(|lane| vec![add(&candidate[lane][0], &candidate[lane ^ offset][0])])
                    .collect();
            }
            for r in 0..rows {
                assert_eq!(candidate[r * (32 / rows)][0], reference[0][r]);
            }
        }
    }
}
