//! Exact prefill expansion followed by the unchanged prefill lagged mixer in
//! one launch (`cuda/sinkhorn_hc/prefill_expand_mix.cuh`): each CTA expands
//! its own rows with the F32 expand's expressions, then mixes them.
use super::{CHAIN, THREADS, ptr, shape_limits};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::{
    contract::{DType, LaunchPlan, Refusal, Validate},
    sinkhorn_expand_mix::SinkhornExpandMixQuery,
    sinkhorn_hyper_connection::SinkhornHyperConnectionQuery,
};

/// Lab controls; each changes expanded bits on finite rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    None,
    /// The stream sum in reverse stream order.
    ReverseExpansion,
    /// `post * y` contracted into the final add.
    FusedPost,
    /// The stream sum rounded to BF16 before `post * y` is added.
    RoundMixed,
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub query: SinkhornExpandMixQuery,
    pub capacity: u32,
    pub tile_rows: u32,
    pub chains: u32,
    pub symbol: String,
    pub mutation: Mutation,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        let m = self.query.mix;
        shape_limits(SinkhornHyperConnectionQuery::MixCollapse {
            mix: m,
            capacity: self.capacity,
        })?;
        // The pipeline stages two panels of `chains` 128-K chains per row.
        let panel = self.chains * CHAIN;
        let stage = 2 * u64::from(self.tile_rows) * u64::from(panel) * 2;
        if ![4, 8].contains(&self.tile_rows)
            || ![4, 8].contains(&self.chains)
            || self.capacity == 0
            || self.capacity > 8192
            || !m.width.is_multiple_of(8)
            || !m.flat_width().is_multiple_of(panel)
            || !THREADS.is_multiple_of(panel / 8)
            || stage > 32 * 1024
        {
            return Err(Refusal(
                "prefill expand-mix tile exceeds the pipeline contract",
            ));
        }
        Ok(())
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.validate()?;
        if rows == 0 || rows > self.capacity || live > rows {
            return Err(Refusal(
                "prefill expand-mix live rows must fit the grid capacity",
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
        let m = self.query.mix;
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
            Mutation::FusedPost => "fused_post",
            Mutation::RoundMixed => "round_mixed",
        };
        let params = [
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
            ("live_rows", "const uint32_t"),
        ]
        .into_iter()
        .map(|(n, t)| ptr(n, t))
        .collect();
        Ok(EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::sinkhorn_hc::prefill_expand_mix::run<{traits},{},{},chop::sinkhorn_hc::prefill_expand_mix::Mutation::{mutation}>(x,y,post_in,comb_in,expanded,fn,base,scale,pre_in,collapsed,pre,post,comb,live_rows);",
                self.tile_rows, self.chains
            ),
            THREADS,
            1,
        ))
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("prefill expand-mix family requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for e in entries {
        if !symbols.insert(&e.symbol) {
            return Err(Refusal("duplicate prefill expand-mix symbol"));
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/prefill_expand_mix.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect::<Result<_, _>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn entry(tile_rows: u32, chains: u32) -> Entry {
        Entry {
            query: SinkhornExpandMixQuery {
                mix: super::super::lagged::LAB.mix,
            },
            capacity: 8192,
            tile_rows,
            chains,
            symbol: "expand_mix".into(),
            mutation: Mutation::None,
        }
    }
    #[test]
    fn tiles_keep_the_mixer_abi_and_reject_impossible_rows() {
        for (tile_rows, chains) in [(4, 4), (8, 4), (8, 8), (4, 8)] {
            let e = entry(tile_rows, chains);
            let spec = e.spec().unwrap();
            assert_eq!(spec.params.len(), 14);
            for rows in [1, 7, 64, 192, 320, 2240, 2496, 2624, 8191, 8192] {
                for live in [0, rows - 1, rows] {
                    let p = e.plan(rows, live).unwrap();
                    assert_eq!(p.grid, [rows.div_ceil(tile_rows), 1, 1]);
                    assert_eq!(p.block, [THREADS, 1, 1]);
                    assert_eq!(p.workspace_bytes, 0);
                    assert_eq!(p.dynamic_shared_bytes, 0);
                    assert!(p.pdl);
                }
                assert!(e.plan(rows, rows + 1).is_err());
            }
            assert!(e.plan(0, 0).is_err());
            assert!(e.plan(8193, 8193).is_err());
        }
        for (tile_rows, chains) in [(1, 4), (16, 4), (8, 2), (8, 16)] {
            assert!(entry(tile_rows, chains).validate().is_err());
        }
    }
    /// The expansion loop of `run`: thread t takes items t, t + 256 in one
    /// iteration, then advances by 512. Every live row's packs are written
    /// once and nothing past the live count, for every tile fill.
    #[test]
    fn expansion_items_cover_each_live_pack_once() {
        let packs = 5120 / 8;
        for tile_rows in [4usize, 8] {
            for count in 1..=tile_rows {
                let limit = count * packs;
                let mut seen = vec![0u8; tile_rows * packs];
                for t in 0..THREADS as usize {
                    let mut item = t;
                    while item < limit {
                        seen[item] += 1;
                        let next = item + THREADS as usize;
                        if next < limit {
                            seen[next] += 1;
                        }
                        item += 2 * THREADS as usize;
                    }
                }
                assert!(seen[..limit].iter().all(|&n| n == 1), "{tile_rows} {count}");
                assert!(seen[limit..].iter().all(|&n| n == 0));
            }
        }
    }
    #[test]
    fn duplicate_symbols_are_refused() {
        let e = entry(8, 4);
        assert!(cubin_spec("m", &[e.clone(), e.clone()]).is_err());
        assert!(cubin_spec("m", &[]).is_err());
        assert_eq!(cubin_spec("m", &[e]).unwrap().entries.len(), 1);
    }
}
