//! Prefill lane layouts with the original per-output arithmetic trees.
use super::{CHAIN, Geometry, THREADS, lagged, tiled};
use crate::Profile;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, InstantiationSpec, LaunchPlan, Refusal, Validate};
use kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery as Query;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kind {
    Mix {
        rows: u32,
        staged: bool,
    },
    Pipeline {
        rows: u32,
        chains: u32,
    },
    Expand {
        threads: u32,
        row_warps: bool,
        broadcast: bool,
    },
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub reference: lagged::Entry,
    pub kind: Kind,
}
impl Entry {
    pub fn contract(&self) -> Result<InstantiationSpec, Refusal> {
        let mut spec = self.reference.contract()?;
        match (self.kind, self.reference.query, self.reference.op) {
            (
                Kind::Mix { rows, staged },
                Query::MixCollapse { mix, .. },
                lagged::Op::MixCollapse { .. },
            ) => {
                tiled::fits(mix, rows)?;
                let per_warp = mix.mixes().div_ceil(8);
                let element = if mix.fn_dtype == DType::F32 { 4 } else { 2 };
                let shared = if staged {
                    2 * 8 * per_warp * CHAIN * element
                } else {
                    0
                } + rows * (9 + mix.mixes() + mix.streams) * 4;
                if shared > 48 * 1024 {
                    return Err(Refusal("prefill warp mixer exceeds default shared memory"));
                }
                spec.key.knobs.tile_m = rows as u16;
                spec.key.knobs.tile_k = CHAIN as u16;
                spec.key.knobs.stages = if staged { 2 } else { 1 };
            }
            (
                Kind::Pipeline { rows, chains },
                Query::MixCollapse { mix, .. },
                lagged::Op::MixCollapse { .. },
            ) => {
                tiled::fits(mix, rows)?;
                if ![4, 8].contains(&chains)
                    || !(mix.flat_width() / CHAIN).is_multiple_of(chains)
                    || !(256u32).is_multiple_of(chains * CHAIN / 8)
                {
                    return Err(Refusal(
                        "prefill panels require whole four- or eight-chain groups",
                    ));
                }
                let shared = 4 * rows * chains * CHAIN + rows * (9 + mix.mixes() + mix.streams) * 4;
                if shared > 48 * 1024 {
                    return Err(Refusal("prefill panels exceed default shared memory"));
                }
                spec.key.knobs.tile_m = rows as u16;
                spec.key.knobs.tile_k = (chains * CHAIN) as u16;
                spec.key.knobs.stages = 2;
            }
            (
                Kind::Expand {
                    threads,
                    row_warps,
                    broadcast,
                },
                Query::ExpandF32 { width, streams, .. },
                lagged::Op::Expand,
            ) => {
                if ![128, 256].contains(&threads)
                    || !width.is_multiple_of(4)
                    || (broadcast && streams * (streams + 1) > 32)
                {
                    return Err(Refusal("invalid prefill expansion layout"));
                }
                spec.key.knobs.tile_m = if row_warps { (threads / 32) as u16 } else { 1 };
                spec.key.knobs.tile_n = if row_warps { 128 } else { (threads * 4) as u16 };
                spec.key.knobs.warps = (threads / 32) as u8;
                spec.key.knobs.swizzle = u8::from(broadcast);
            }
            _ => return Err(Refusal("prefill warp candidate and reference disagree")),
        }
        spec.validate()?;
        Ok(spec)
    }
    pub fn shared_bytes(&self) -> u32 {
        0
    }
    pub fn invariance(&self) -> kernel_core::properties::Invariance {
        if self.contract().is_ok() {
            self.reference.invariance()
        } else {
            kernel_core::properties::Invariance::NONE
        }
    }
    pub fn plan(&self, rows: u32, live: u32) -> Result<LaunchPlan, Refusal> {
        self.contract()?;
        let mut p = self.reference.plan(rows, live)?;
        match (self.kind, self.reference.query) {
            (Kind::Mix { rows: tile, .. } | Kind::Pipeline { rows: tile, .. }, _) => {
                p.grid[0] = rows.div_ceil(tile)
            }
            (
                Kind::Expand {
                    threads, row_warps, ..
                },
                Query::ExpandF32 { width, .. },
            ) => {
                p.grid = [
                    width.div_ceil(if row_warps { 128 } else { threads * 4 }),
                    if row_warps {
                        rows.div_ceil(threads / 32)
                    } else {
                        rows
                    },
                    1,
                ];
                p.block = [threads, 1, 1];
            }
            _ => unreachable!("validated candidate"),
        }
        p.dynamic_shared_bytes = 0;
        crate::backend::check_plan(&p)?;
        Ok(p)
    }
    pub fn body(&self) -> Result<String, Refusal> {
        self.contract()?;
        let original = self.reference.body();
        Ok(match self.kind {
            Kind::Mix { rows, staged } => {
                let probe = matches!(self.reference.op, lagged::Op::MixCollapse { probe: true });
                original
                    .replace("mix_collapse_lagged<", "mix_collapse_prefill_warp<")
                    .replace(
                        &format!(",{probe}>("),
                        &format!(",{rows},{staged},{probe}>("),
                    )
            }
            Kind::Pipeline { rows, chains } => {
                let probe = matches!(self.reference.op, lagged::Op::MixCollapse { probe: true });
                original
                    .replace("mix_collapse_lagged<", "mix_collapse_prefill_pipeline<")
                    .replace(
                        &format!(",{probe}>("),
                        &format!(",{rows},{chains},{probe}>("),
                    )
            }
            Kind::Expand {
                threads,
                row_warps,
                broadcast,
            } => original
                .replace("expand_f32<", "expand_prefill_warp<")
                .replace(">(x,y", &format!(",{threads},{row_warps},{broadcast}>(x,y")),
        })
    }
}
pub fn mix(profile: Profile, geometry: Geometry, rows: u32, staged: bool, probe: bool) -> Entry {
    let mut reference = lagged::entries(profile, geometry, probe)
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .unwrap();
    reference.symbol = reference.symbol.replace(
        "lagged_mix_collapse",
        &format!("prefill_mix_warp_r{rows}_s{}", u8::from(staged)),
    );
    Entry {
        reference,
        kind: Kind::Mix { rows, staged },
    }
}
pub fn pipeline(
    profile: Profile,
    geometry: Geometry,
    rows: u32,
    chains: u32,
    probe: bool,
) -> Entry {
    let mut reference = lagged::entries(profile, geometry, probe)
        .into_iter()
        .find(|e| e.op == lagged::Op::MixCollapse { probe })
        .unwrap();
    reference.symbol = reference.symbol.replace(
        "lagged_mix_collapse",
        &format!("prefill_mix_pipeline_r{rows}_c{chains}"),
    );
    Entry {
        reference,
        kind: Kind::Pipeline { rows, chains },
    }
}
pub fn expand(
    profile: Profile,
    geometry: Geometry,
    threads: u32,
    row_warps: bool,
    broadcast: bool,
) -> Entry {
    let mut reference = lagged::entries(profile, geometry, false)
        .into_iter()
        .find(|e| e.op == lagged::Op::Expand)
        .unwrap();
    reference.symbol = reference.symbol.replace(
        "expand_f32",
        &format!(
            "prefill_expand_warp_t{threads}_r{}_b{}",
            u8::from(row_warps),
            u8::from(broadcast)
        ),
    );
    Entry {
        reference,
        kind: Kind::Expand {
            threads,
            row_warps,
            broadcast,
        },
    }
}
pub fn cubin_spec(name: String, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("prefill warp family requires entries"));
    }
    let mut names = std::collections::BTreeSet::new();
    for e in entries {
        if !names.insert(&e.reference.symbol) {
            return Err(Refusal("duplicate prefill warp symbol"));
        }
    }
    Ok(CubinSpec {
        name,
        sources_root: crate::cuda_source_root(),
        include_dirs: vec![],
        headers: vec![
            "#include \"sinkhorn_hc/prefill_warp.cuh\"".into(),
            "#include \"sinkhorn_hc/expand_prefill_warp.cuh\"".into(),
            "#include \"sinkhorn_hc/prefill_pipeline.cuh\"".into(),
        ],
        defines: vec![],
        entries: entries
            .iter()
            .map(|e| {
                Ok(EntrySpec::new(
                    e.reference.symbol.clone(),
                    e.reference.params.clone(),
                    e.body()?,
                    match e.kind {
                        Kind::Mix { .. } | Kind::Pipeline { .. } => THREADS,
                        Kind::Expand { threads, .. } => threads,
                    },
                    1,
                ))
            })
            .collect::<Result<_, Refusal>>()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn add(a: &str, b: &str) -> String {
        format!("({a}+{b})")
    }
    #[test]
    fn local_butterfly_has_the_exact_original_expression_tree() {
        let mut reference: Vec<_> = (0..32).map(|n| n.to_string()).collect();
        for offset in [16, 8, 4, 2, 1] {
            reference = (0..32)
                .map(|n| add(&reference[n], &reference[n ^ offset]))
                .collect();
        }
        for rows in [4, 8, 16] {
            let group = 32 / rows;
            let mut lanes = Vec::new();
            for low in 0..group {
                let mut values: Vec<_> = (0..rows).map(|r| (low + r * group).to_string()).collect();
                let mut offset = rows / 2;
                while offset > 0 {
                    for n in 0..offset {
                        values[n] = add(&values[n], &values[n + offset]);
                    }
                    offset /= 2;
                }
                lanes.push(values[0].clone());
            }
            let mut offset = group / 2;
            while offset > 0 {
                lanes = (0..group)
                    .map(|n| add(&lanes[n], &lanes[n ^ offset]))
                    .collect();
                offset /= 2;
            }
            assert_eq!(lanes[0], reference[0]);
        }
        for chains in [32, 160, 256] {
            for thread in 0..256 {
                let expected: Vec<_> = (thread..chains * 16).step_by(256).collect();
                let actual: Vec<_> = (0..chains)
                    .filter(|c| c % 16 == thread / 16)
                    .map(|c| c * 16 + thread % 16)
                    .collect();
                assert_eq!(actual, expected);
            }
        }
    }
    #[test]
    fn warp_plans_preserve_abis_and_cover_each_output_once() {
        for rows in [4, 8, 16] {
            for staged in [false, true] {
                for probe in [false, true] {
                    let e = mix(lagged::LAB_PROFILE, lagged::LAB, rows, staged, probe);
                    for capacity in [1, 17, 64, 192, 320, 1024, 2496, 2624, 8191, 8192] {
                        let p = e.plan(capacity, 0).unwrap();
                        assert_eq!(p.grid, [capacity.div_ceil(rows), 1, 1]);
                        assert_eq!(p.block, [256, 1, 1]);
                        assert_eq!(p.dynamic_shared_bytes, 0);
                        assert_eq!(p.workspace_bytes, 0);
                        assert!(p.pdl);
                    }
                    assert!(e.plan(0, 0).is_err());
                    assert!(e.plan(8193, 0).is_err());
                    assert!(e.plan(64, 65).is_err());
                    assert_eq!(
                        e.invariance(),
                        if probe {
                            kernel_core::properties::Invariance::NONE
                        } else {
                            kernel_core::properties::Invariance::any_rows(8192)
                        }
                    );
                }
            }
        }
        for chains in [4, 8] {
            let e = pipeline(lagged::LAB_PROFILE, lagged::LAB, 8, chains, false);
            let p = e.plan(8192, 8191).unwrap();
            assert_eq!(p.grid, [1024, 1, 1]);
            assert_eq!(p.dynamic_shared_bytes, 0);
            assert_eq!(e.contract().unwrap().key.knobs.tile_k, chains as u16 * 128);
        }
        for threads in [128, 256] {
            for row_warps in [false, true] {
                for broadcast in [false, true] {
                    let e = expand(
                        lagged::LAB_PROFILE,
                        lagged::LAB,
                        threads,
                        row_warps,
                        broadcast,
                    );
                    let p = e.plan(17, 17).unwrap();
                    let mut hits = vec![0; 17 * 5120];
                    for y in 0..p.grid[1] {
                        for x in 0..p.grid[0] {
                            for t in 0..threads {
                                let row = if row_warps {
                                    y * (threads / 32) + t / 32
                                } else {
                                    y
                                };
                                let col = x * if row_warps { 128 } else { threads * 4 }
                                    + if row_warps { t % 32 } else { t } * 4;
                                if row < 17 && col < 5120 {
                                    for i in 0..4 {
                                        hits[(row * 5120 + col + i) as usize] += 1;
                                    }
                                }
                            }
                        }
                    }
                    assert!(hits.into_iter().all(|n| n == 1));
                    assert_eq!(p.cluster, [1; 3]);
                    assert_eq!(p.dynamic_shared_bytes, 0);
                }
            }
        }
    }
}
