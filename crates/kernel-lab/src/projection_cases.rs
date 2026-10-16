//! Oracle and invariance cases enumerated from the generated offer inventory.
use kernel_core::precision::capability::OperationCapability;
use kernel_core::precision::{OperationKind, OperationUse};
use kernels_sm120::precision::KernelOffer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    pub capability: OperationCapability,
    pub oracle_rows: Vec<u32>,
    pub replay: bool,
    pub chunk_invariant: bool,
    pub empty_routes: bool,
    pub zero_live_refused: bool,
}

fn rows(usage: OperationUse) -> Vec<u32> {
    [
        0,
        1,
        15,
        16,
        17,
        31,
        32,
        33,
        usage.capacity.saturating_sub(1),
        usage.capacity,
    ]
    .into_iter()
    .filter(|n| *n <= usage.capacity)
    .collect::<BTreeSet<_>>()
    .into_iter()
    .collect()
}

/// Numerical output rows used by the dense oracle.
pub fn numerical_rows(live: u32, tiles: &[u32]) -> Vec<usize> {
    if live == 0 {
        return Vec::new();
    }
    let mut rows = BTreeSet::from([0, live / 2, live - 1]);
    // Fragment kernels advance in 8-row MMA strips within their larger tiles.
    for tile in std::iter::once(8)
        .chain(tiles.iter().copied())
        .filter(|t| *t > 0)
    {
        for boundary in (tile..live).step_by(tile as usize) {
            rows.extend([boundary - 1, boundary]);
        }
    }
    rows.into_iter().map(|r| r as usize).collect()
}

/// Exercise partial strips and tiles on both sides of a chunk split.
pub fn chunk_splits(capacity: u32, tiles: &[u32]) -> (u32, Vec<u32>) {
    let largest = tiles.iter().copied().max().unwrap_or(64).max(64);
    let rows = capacity.min(largest.saturating_mul(2).saturating_add(1));
    let mut splits = BTreeSet::from([rows / 2]);
    for tile in [8, 32, 64]
        .into_iter()
        .chain(tiles.iter().copied())
        .filter(|t| *t > 0)
    {
        for boundary in (tile..=rows).step_by(tile as usize) {
            splits.extend([boundary - 1, boundary, boundary + 1]);
        }
    }
    (
        rows,
        splits.into_iter().filter(|s| *s > 0 && *s < rows).collect(),
    )
}

pub fn typed(offer: &KernelOffer) -> Result<Case, String> {
    let capability = offer.capability().map_err(|e| e.to_string())?;
    let chunk_invariant = offer.invariance().chunk != kernels_sm120::invariance::Chunk::None;
    Ok(Case {
        oracle_rows: rows(capability.usage),
        zero_live_refused: capability.kind == OperationKind::Dense,
        replay: true,
        chunk_invariant,
        empty_routes: matches!(capability.kind, OperationKind::Grouped { .. }),
        capability,
    })
}

/// Retained catalog entries inherit their declared invariance claim. Every use
/// still gets an oracle and replay case, including decode and verify uses of a
/// prefill-shaped implementation.
pub fn catalog(capability: OperationCapability) -> Result<Case, String> {
    let declared = kernels_sm120::invariance::declarations(crate::catalogs());
    let mut chunk_invariant = capability.kind == OperationKind::Lookup;
    for stage in &capability.stages {
        let entry = declared
            .iter()
            .find(|d| &d.symbol == stage)
            .ok_or_else(|| format!("missing invariance declaration for {stage}"))?;
        chunk_invariant |= entry
            .invariance
            .claims()
            .contains(&kernels_sm120::invariance::Property::Chunk);
    }
    Ok(Case {
        oracle_rows: if capability.kind == OperationKind::Lookup {
            vec![0, 1, capability.usage.capacity]
        } else {
            vec![1]
        },
        zero_live_refused: false,
        replay: true,
        chunk_invariant,
        empty_routes: false,
        capability,
    })
}

pub fn inventory() -> Result<Vec<Case>, String> {
    let (offers, retained, _) = crate::models::projection_inventory()?;
    let mut cases = offers.iter().map(typed).collect::<Result<Vec<_>, _>>()?;
    cases.extend(
        retained
            .into_iter()
            .map(catalog)
            .collect::<Result<Vec<_>, _>>()?,
    );
    if cases.is_empty() {
        return Err("generated projection inventory is empty".into());
    }
    Ok(cases)
}

/// Pick the largest row set for each stage, breaking ties by capacity and then
/// inventory order. A multi-stage case may represent several symbols.
pub fn representatives(cases: &[(Vec<String>, Vec<u32>)]) -> Vec<usize> {
    let mut best = std::collections::BTreeMap::new();
    for (index, (stages, rows)) in cases.iter().enumerate() {
        let rank = (
            rows.iter().collect::<BTreeSet<_>>().len(),
            rows.iter().max().copied().unwrap_or(0),
        );
        for stage in stages {
            let selected = best.entry(stage).or_insert((rank, index));
            if rank > selected.0 {
                *selected = (rank, index);
            }
        }
    }
    best.into_values()
        .map(|(_, i)| i)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub struct Selection(BTreeSet<usize>);
impl Selection {
    pub fn read() -> Result<Self, String> {
        let cases = inventory()?;
        let (_, _, copies) = crate::models::projection_inventory()?;
        let mut rows = cases
            .iter()
            .map(|c| (c.capability.stages.clone(), c.oracle_rows.clone()))
            .collect::<Vec<_>>();
        rows.extend(copies.iter().map(|c| {
            (
                vec![c.stage().entry().symbol.clone()],
                vec![
                    0,
                    1,
                    c.stage().capacity().saturating_sub(1),
                    c.stage().capacity(),
                ],
            )
        }));
        let indices = match std::env::var("CHOPINFER_PROJECTION_CASES") {
            Ok(value) => {
                let parsed = value
                    .split(',')
                    .map(|i| i.parse::<usize>().map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                let unique = parsed.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != parsed.len()
                    || unique.is_empty()
                    || unique.iter().any(|i| *i >= rows.len())
                {
                    return Err("invalid prepared case selection".into());
                }
                unique
            }
            Err(std::env::VarError::NotPresent) => representatives(&rows).into_iter().collect(),
            Err(e) => return Err(e.to_string()),
        };
        Ok(Self(indices))
    }
    pub fn contains(&self, case_index: usize) -> bool {
        self.0.contains(&case_index)
    }
}

/// Prebuilt typed entries. GPU jobs read this report and never invoke nvcc.
pub struct Prepared {
    report: kernel_build::BuildReport,
    modules:
        std::cell::RefCell<std::collections::BTreeMap<std::path::PathBuf, runtime_cuda::Module>>,
}

impl Prepared {
    pub fn read() -> Result<Self, String> {
        let path = std::env::var_os("CHOPINFER_PROJECTION_REPORT")
            .ok_or("missing prepared projection report")?;
        let report = kernel_build::BuildReport::read(std::path::Path::new(&path))
            .map_err(|e| e.to_string())?;
        report.verify_current().map_err(|e| e.to_string())?;
        Ok(Self {
            report,
            modules: Default::default(),
        })
    }
    pub fn contains(&self, symbol: &str) -> bool {
        self.report
            .cubins
            .iter()
            .any(|c| c.declarations.iter().any(|d| d.symbol == symbol))
    }
    pub fn function(
        &self,
        session: &crate::Session,
        symbol: &str,
        shared: u32,
    ) -> Result<runtime_cuda::Function, String> {
        let cubin = self
            .report
            .cubins
            .iter()
            .find(|c| c.declarations.iter().any(|d| d.symbol == symbol))
            .ok_or_else(|| format!("prepared selection has no {symbol}"))?;
        let mut modules = self.modules.borrow_mut();
        if !modules.contains_key(&cubin.path) {
            modules.insert(
                cubin.path.clone(),
                session.load_cubin(&std::fs::read(&cubin.path).map_err(|e| e.to_string())?)?,
            );
        }
        let module = &modules[&cubin.path];
        if module.context() != session.context {
            return Err("prepared module belongs to another CUDA context".into());
        }
        let function = module.function(symbol).map_err(|e| e.to_string())?;
        if shared >= 48 * 1024 {
            function
                .set_max_dynamic_shared_memory(shared)
                .map_err(|e| e.to_string())?;
        }
        Ok(function)
    }
}

/// The compute entry an offer launches: its dense or grouped GEMM, or its
/// lookup stage.
pub fn compute_symbol(offer: &KernelOffer) -> String {
    use kernels_sm120::precision::Compute;
    match &offer.handle().compute {
        Compute::Dense(spec) => spec.symbol.clone(),
        Compute::Grouped(g) => g.symbol.clone(),
        Compute::Lookup(l) => l.stage().entry().symbol.clone(),
    }
}

/// What an offer computes, apart from the use it serves: its capability
/// without the use, the stages and the composition naming them.
fn operation_key(offer: &KernelOffer) -> Result<String, String> {
    let mut capability = offer.capability().map_err(|e| e.to_string())?;
    capability.usage = OperationUse {
        phase: kernel_core::precision::ExecutionPhase::Prefill,
        capacity: 1,
        graph: false,
    };
    capability.stages.clear();
    capability.implementation.clear();
    serde_json::to_string(&capability).map_err(|e| e.to_string())
}

/// The offer a rows claim of `offers[index]` refers to: the one-row decode
/// offer of the same operation (same capability apart from its use), by
/// inventory index.
pub fn one_row_offer(offers: &[KernelOffer], index: usize) -> Result<usize, String> {
    let key = operation_key(&offers[index])?;
    for (other, candidate) in offers.iter().enumerate() {
        let usage = candidate.usage();
        if usage.phase == kernel_core::precision::ExecutionPhase::Decode
            && usage.capacity == 1
            && operation_key(candidate)? == key
        {
            return Ok(other);
        }
    }
    Err(format!(
        "{}: no one-row decode offer of its operation",
        compute_symbol(&offers[index])
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernels_sm120::precision::Compute;
    #[test]
    fn dense_samples_detect_a_wrong_later_strip_row() {
        let rows = numerical_rows(64, &[128]);
        let reference = kernel_core::oracle::OracleOutput::new(
            vec![1.0; rows.len()],
            vec![1.0; rows.len()],
            128,
        )
        .unwrap();
        let mut actual = [1.0_f32; 64];
        actual[47] = 0.0;
        assert!(actual.iter().all(|x| x.is_finite()));
        let sampled = rows.iter().map(|r| actual[*r]).collect::<Vec<_>>();
        assert!(
            crate::check_output(
                &sampled,
                &reference,
                kernel_core::contract::DType::Bf16,
                &mut Default::default()
            )
            .is_err()
        );
    }

    #[test]
    fn dense_sampling_covers_every_strip_and_tile_boundary() {
        for capacity in [1, 8, 16, 33, 64, 128, 8192] {
            let rows = numerical_rows(capacity, &[64, 128]);
            for tile in [8, 64, 128] {
                for boundary in (tile..capacity).step_by(tile as usize) {
                    assert!(rows.contains(&((boundary - 1) as usize)));
                    assert!(rows.contains(&(boundary as usize)));
                }
            }
            assert!(rows.iter().all(|r| *r < capacity as usize));
            let (live, splits) = chunk_splits(capacity, &[128]);
            assert!(live <= capacity);
            assert!(splits.iter().all(|s| *s > 0 && *s < live));
            if capacity >= 128 {
                for split in [63, 64, 65, 127] {
                    assert!(splits.contains(&split));
                }
            }
        }
        assert!(numerical_rows(0, &[128]).is_empty());
        assert!(chunk_splits(1, &[128]).1.is_empty());
    }

    #[test]
    fn representatives_cover_symbols_with_the_largest_row_set() {
        let cases = vec![
            (vec!["a".into()], vec![0, 1]),
            (vec!["a".into(), "b".into()], vec![0, 1, 16, 32]),
            (vec!["b".into()], vec![0, 1, 64]),
            (vec!["c".into()], vec![0, 1]),
            (vec!["c".into()], vec![0, 64]),
        ];
        assert_eq!(representatives(&cases), vec![1, 4]);
        let inventory = inventory().unwrap();
        let cases = inventory
            .iter()
            .map(|c| (c.capability.stages.clone(), c.oracle_rows.clone()))
            .collect::<Vec<_>>();
        let chosen = representatives(&cases);
        for (stages, rows) in &cases {
            for stage in stages {
                assert!(
                    chosen
                        .iter()
                        .any(|i| cases[*i].0.contains(stage) && cases[*i].1.len() >= rows.len())
                );
            }
        }
    }

    #[test]
    fn catalog_without_a_chunk_claim_does_not_inherit_one_from_its_kernel() {
        let (offers, _, _) = crate::models::projection_inventory().unwrap();
        let old = offers.iter().find(|o| matches!(&o.handle().compute,
            Compute::Dense(spec) if matches!(spec.key.dense, Some(kernel_core::contract::DenseKernel::FragmentPrefill { .. })))).unwrap();
        let Compute::Dense(spec) = &old.handle().compute else {
            unreachable!()
        };
        let undeclared = KernelOffer::catalog_dense(spec.clone(), old.usage()).unwrap();
        assert!(!typed(&undeclared).unwrap().chunk_invariant);
    }

    #[test]
    fn every_generated_offer_has_oracle_and_invariance_cases() {
        let (offers, retained, _) = crate::models::projection_inventory().unwrap();
        let cases = inventory().unwrap();
        assert_eq!(cases.len(), offers.len() + retained.len());
        for case in cases {
            assert!(case.replay);
            assert!(case.oracle_rows.first().is_some_and(|n| *n <= 1));
            assert!(case.oracle_rows.contains(&1));

            assert!(!case.capability.stages.is_empty());
        }
    }
}
