//! Expansion and exact projection partials followed by tiled lagged mixing.
use super::{CHAIN, COLUMNS, THREADS, ptr, shape_limits};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, LaunchPlan, Refusal, Validate};
use kernel_core::sinkhorn_expand_mix::SinkhornExpandMixQuery;
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery;

#[derive(Clone, Debug)]
pub struct Entry {
    pub query: SinkhornExpandMixQuery,
    pub capacity: u32,
    pub symbol: String,
    pub probe: bool,
}
impl Entry {
    pub fn validate(&self) -> Result<(), Refusal> {
        self.query.validate()?;
        let m = self.query.mix;
        shape_limits(SinkhornHyperConnectionQuery::MixCollapse {
            mix: m,
            capacity: self.capacity,
        })?;
        if self.capacity == 0
            || self.capacity > 64
            || !m.flat_width().is_multiple_of(CHAIN * 8)
            || u64::from(m.mixes()) * u64::from(m.flat_width() / CHAIN + 1) * 4 > 46 * 1024
        {
            return Err(Refusal(
                "expand-mix requires at most 64 rows and complete chain groups",
            ));
        }
        Ok(())
    }
    pub fn workspace(&self, rows: u32) -> Result<u64, Refusal> {
        self.validate()?;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("expand-mix rows exceed capacity"));
        }
        let m = self.query.mix;
        Ok(u64::from(rows)
            * (u64::from(m.mixes()) * u64::from(m.flat_width() / CHAIN) + u64::from(THREADS))
            * 4)
    }
    pub fn symbols(&self) -> [String; 2] {
        [
            format!("{}_partials_v1", self.symbol),
            format!("{}_finish_v1", self.symbol),
        ]
    }
    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 2], Refusal> {
        let workspace_bytes = self.workspace(rows)?;
        let m = self.query.mix;
        let plans =
            [m.flat_width() / (CHAIN * 8), m.width.div_ceil(COLUMNS)].map(|tiles| LaunchPlan {
                grid: [tiles, rows, 1],
                block: [THREADS, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes,
                cluster: [1; 3],
                pdl: true,
            });
        for p in plans {
            crate::backend::check_plan(&p)?;
        }
        Ok(plans)
    }
    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
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
        let [partials, finish] = self.symbols();
        let mut tail = vec![
            ptr("expanded", "const uint16_t"),
            ptr("base", "const float"),
            ptr("scale", "const float"),
            ptr("pre_in", "const float"),
            ptr("collapsed", "uint16_t"),
            ptr("pre", "float"),
            ptr("post", "float"),
            ptr("comb", "float"),
        ];
        if self.probe {
            tail.extend([
                ptr("projected", "float"),
                ptr("rstd", "float"),
                ptr("cells", "float"),
            ]);
        }
        tail.extend([
            ptr("workspace", "const float"),
            ptr("live_rows", "const uint32_t"),
        ]);
        let probes = if self.probe {
            "projected,rstd,cells"
        } else {
            "nullptr,nullptr,nullptr"
        };
        Ok(vec![
            EntrySpec::new(
                partials,
                vec![
                    ptr("x", "const uint16_t"),
                    ptr("y", "const uint16_t"),
                    ptr("post_in", "const float"),
                    ptr("comb_in", "const float"),
                    ptr("expanded", "uint16_t"),
                    ptr("fn", "const void"),
                    ptr("workspace", "float"),
                    ptr("live_rows", "const uint32_t"),
                ],
                format!(
                    "chop::sinkhorn_hc::expand_mix_partials<{traits}>(x,y,post_in,comb_in,expanded,fn,workspace,live_rows);"
                ),
                THREADS,
                1,
            ),
            EntrySpec::new(
                finish,
                tail,
                format!(
                    "chop::sinkhorn_hc::expand_mix_finish<{traits},{}>(expanded,base,scale,pre_in,collapsed,pre,post,comb,{probes},workspace,live_rows);",
                    self.probe
                ),
                THREADS,
                1,
            ),
        ])
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("expand-mix cubin needs entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec!["#include \"sinkhorn_hc/expand_mix.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .map(Entry::specs)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wider_rows_change_only_grid_and_workspace_bounds() {
        let old = Entry {
            query: SinkhornExpandMixQuery {
                mix: super::super::lagged::LAB.mix,
            },
            capacity: 32,
            symbol: "expand_mix_rows".into(),
            probe: false,
        };
        let wide = Entry {
            capacity: 64,
            ..old.clone()
        };
        assert_eq!(
            format!("{:?}", old.specs().unwrap()),
            format!("{:?}", wide.specs().unwrap())
        );
        for rows in 1..=32 {
            assert_eq!(old.plans(rows).unwrap(), wide.plans(rows).unwrap());
        }
        for rows in 33..=64 {
            assert!(old.plans(rows).is_err());
            let plans = wide.plans(rows).unwrap();
            assert!(plans.iter().all(|p| p.grid[1] == rows));
            assert_eq!(
                wide.workspace(rows).unwrap(),
                wide.workspace(1).unwrap() * u64::from(rows)
            );
        }
        assert!(wide.plans(0).is_err());
        assert!(wide.plans(65).is_err());
        assert!(
            Entry {
                capacity: 65,
                ..wide
            }
            .validate()
            .is_err()
        );
    }
}
