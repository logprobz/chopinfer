//! The cached reuse kernel with a separate, bounded row-domain plane.
//! Each output retains its original 128-K chains and ordered reduction.
use super::Entry;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, WeightFormat};

#[derive(Clone, Debug)]
pub struct Candidate {
    pub entry: Entry,
    pub capacity: u32,
}

impl Candidate {
    pub fn validate(&self) -> Result<(), Refusal> {
        if !self.entry.uses_cached_weights()
            || self.entry.uses_fused_split()
            || self.entry.uses_prefill_tiles()
            || self.entry.query().weight != WeightFormat::Bf16
            || !(33..=256).contains(&self.capacity)
            || self.capacity > self.entry.query().m_capacity
        {
            return Err(Refusal(
                "bounded reuse requires cached BF16 weights and 33 to 256 rows",
            ));
        }
        Ok(())
    }

    pub fn domain_words(&self) -> u32 {
        self.capacity.div_ceil(32) * 32
    }

    pub fn symbol(&self, reduce: bool) -> String {
        format!(
            "{}_reuse32_m{}_{}_v1",
            self.entry.stage_symbol("_decode"),
            self.capacity,
            if reduce { "reduce" } else { "compute" }
        )
    }

    pub fn workspace(&self, rows: u32) -> u64 {
        let q = self.entry.query();
        u64::from(self.domain_words()) * 4
            + u64::from(rows) * u64::from(q.n) * u64::from(q.k.div_ceil(128)) * 4
    }

    pub fn plans(&self, rows: u32) -> Result<[LaunchPlan; 3], Refusal> {
        self.validate()?;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("bounded reuse rows exceed capacity"));
        }
        let q = self.entry.query();
        let elements = rows
            .checked_mul(q.n)
            .ok_or(Refusal("bounded reuse grid overflows"))?;
        Ok([
            self.entry.stage_plan([rows, 1, 1])?,
            self.entry.stage_plan([
                q.n.div_ceil(16),
                rows.div_ceil(32),
                q.k.div_ceil(128).div_ceil(4),
            ])?,
            self.entry.stage_plan([elements.div_ceil(128), 1, 1])?,
        ])
    }

    pub fn specs(&self) -> Result<Vec<EntrySpec>, Refusal> {
        self.validate()?;
        let q = self.entry.query();
        let mut compute = super::reuse::spec(&self.entry)?;
        compute.symbol = self.symbol(false);
        compute.body = compute.body.replace(
            "workspace+32",
            &format!("workspace+{}", self.domain_words()),
        );
        let mut reduce = self
            .entry
            .entry_specs()
            .into_iter()
            .find(|s| s.symbol == self.entry.stage_symbol("_reduce"))
            .ok_or(Refusal("bounded reuse requires a cached reducer"))?;
        reduce.symbol = self.symbol(true);
        reduce.body = format!(
            "chop::exact_f32::reduce<{},{}>(workspace+{},output,live_rows,rows);",
            q.n,
            q.k,
            self.domain_words()
        );
        Ok(vec![compute, reduce])
    }
}

impl Entry {
    /// Route these explicit rows through cached reuse with room for every
    /// row's domain word before the partial planes. Other rows keep their path.
    pub fn with_bounded_reuse(mut self, mut rows: Vec<u32>) -> Result<Self, Refusal> {
        rows.sort_unstable();
        if rows.is_empty() || rows.windows(2).any(|w| w[0] == w[1]) || rows[0] <= 32 {
            return Err(Refusal("bounded reuse needs distinct row counts above 32"));
        }
        self.bounded_reuse_rows = rows;
        let candidate = self.bounded_reuse_candidate().expect("nonempty rows");
        for &rows in &self.bounded_reuse_rows {
            candidate.plans(rows)?;
        }
        Ok(self)
    }

    pub fn bounded_reuse_candidate(&self) -> Option<Candidate> {
        let &capacity = self.bounded_reuse_rows.last()?;
        let mut entry = self.entry_for_rows(1);
        entry.bounded_reuse_rows.clear();
        Some(Candidate { entry, capacity })
    }

    pub fn uses_bounded_reuse(&self, rows: u32) -> bool {
        self.bounded_reuse_rows.binary_search(&rows).is_ok()
    }
}

pub fn cubin_spec(name: &str, candidates: &[Candidate]) -> Result<CubinSpec, Refusal> {
    if candidates.is_empty() {
        return Err(Refusal("bounded reuse requires entries"));
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/exact_f32_reuse.cuh\"".into()],
        defines: vec![],
        entries: candidates
            .iter()
            .map(Candidate::specs)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ActivationFormat, DType, Epilogues, GemmQuery};
    use kernel_core::precision::{ExecutionPhase, OperationUse};

    #[test]
    fn bounded_rows_keep_domain_words_out_of_partial_planes() {
        let entry = Entry::new(
            GemmQuery {
                m_capacity: 8192,
                n: 32320,
                k: 5120,
                weight: WeightFormat::Bf16,
                activation: ActivationFormat::F32,
                output: DType::F32,
                epilogues: Epilogues::default(),
                grouped: None,
            },
            OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            },
            "chop_test_bounded",
        )
        .unwrap()
        .with_split_decode();
        let routed = entry
            .clone()
            .with_bounded_reuse((35..=80).step_by(5).collect())
            .unwrap();
        let candidate = routed.bounded_reuse_candidate().unwrap();
        assert_eq!(candidate.domain_words(), 96);
        for rows in [35, 40, 60, 80] {
            let plans = candidate.plans(rows).unwrap();
            assert_eq!(plans[1].grid, [2020, rows.div_ceil(32), 10]);
            assert!(candidate.workspace(rows) >= u64::from(rows) * (4 + 32320 * 40 * 4));
            assert!(routed.runtime_workspace_bytes(rows) >= candidate.workspace(rows));
            assert!(routed.uses_bounded_reuse(rows));
        }
        for rows in [1, 30, 32, 33, 64, 81, 160] {
            assert!(!routed.uses_bounded_reuse(rows));
            assert_eq!(
                routed.entry_for_rows(rows).symbol(),
                entry.entry_for_rows(rows).symbol()
            );
        }
        assert!(candidate.plans(81).is_err());
        for rows in [vec![], vec![32], vec![40, 40], vec![260]] {
            assert!(entry.clone().with_bounded_reuse(rows).is_err());
        }
        for spec in candidate.specs().unwrap() {
            assert!(spec.body.contains("workspace+96"));
            assert!(!spec.body.contains("workspace+32"));
            assert!(routed.runtime_symbols().contains(&spec.symbol));
        }
    }
}
