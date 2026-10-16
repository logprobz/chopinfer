//! The declared arithmetic of catalog entries, for numerical identities.
//!
//! A compiled wrapper does not carry every choice that decides an entry's
//! output bits: fragment prefill K segments and dense K splits are launch
//! arguments, and the route weight point, SwiGLU clamps and engine schedule
//! of a grouped entry live in its `moe::Variant`. A [`DeclaredCubin`] states
//! them per entry, from the instantiations that built the cubin, so a
//! bundle's entry declarations, and so its contract identity, change when
//! they do, even with an identical cubin.

use crate::{gemm, moe};
use kernel_build::{CubinSpec, EntryDeclaration, EntryParam, EntrySpec, SelectedCubin};
use kernel_core::contract::{InstantiationSpec, Refusal, StaticQuery};
use std::collections::BTreeMap;

/// A dense engine entry's declared kernel (K splits or segments) and query.
pub fn dense(spec: &InstantiationSpec) -> Option<String> {
    let StaticQuery::Gemm(query) = spec.key.query else {
        return None;
    };
    let kernel = spec.key.dense?;
    Some(format!("dense.v1; kernel={kernel:?}; query={query:?}"))
}

/// A fused-weight GEMM entry's kind (cluster blocks per warp, skinny
/// prefill splits) and query.
pub fn fused_gemm(entry: &gemm::fused::Entry) -> String {
    format!(
        "qwen38-gemm.v1; kind={:?}; segments={:?}; query={:?}",
        entry.kind, entry.segments, entry.query
    )
}

/// A grouped expert entry's route weight point, SwiGLU clamps, engine
/// schedule, knobs and query.
pub fn grouped(v: &moe::Variant) -> String {
    format!(
        "grouped.v1; route_weight={:?}; swiglu_clamp={:?}; engine={:?}; knobs={:?}; \
         weight_stages={}; query={:?}",
        v.route_weight, v.swiglu_clamp, v.engine, v.knobs, v.weight_stages, v.query
    )
}

/// A medium-M kernel's declared arithmetic: the reference entries and K
/// segments it serves, which its callers pass at launch.
pub fn mid(references: &[(String, u8)]) -> String {
    format!("mid.v1; references={references:?}")
}

/// A cubin with the declared arithmetic of each of its entries, stated by
/// the producer from the instantiations it built the cubin from.
#[derive(Clone, Debug)]
pub struct DeclaredCubin {
    spec: CubinSpec,
    /// By entry symbol: the declared arithmetic, or `None` for an entry
    /// whose wrapper compiles in all of its arithmetic.
    arithmetic: BTreeMap<String, Option<String>>,
}

impl DeclaredCubin {
    /// `spec` with `arithmetic`, which states each entry once. An entry that
    /// takes its K splits at launch must declare them.
    pub fn new(
        spec: CubinSpec,
        arithmetic: BTreeMap<String, Option<String>>,
    ) -> Result<Self, Refusal> {
        if arithmetic.len() != spec.entries.len()
            || spec
                .entries
                .iter()
                .any(|entry| !arithmetic.contains_key(&entry.symbol))
        {
            return Err(Refusal(
                "declared arithmetic must state each entry of its cubin once",
            ));
        }
        if spec
            .entries
            .iter()
            .any(|entry| arithmetic[&entry.symbol].is_none() && launch_splits(entry))
        {
            return Err(Refusal(
                "an entry that takes its K splits at launch must declare its arithmetic",
            ));
        }
        Ok(Self { spec, arithmetic })
    }

    /// A cubin whose entries' wrappers compile in all of their arithmetic.
    pub fn compiled(spec: CubinSpec) -> Result<Self, Refusal> {
        let arithmetic = spec
            .entries
            .iter()
            .map(|entry| (entry.symbol.clone(), None))
            .collect();
        Self::new(spec, arithmetic)
    }

    /// `specs` with the arithmetic in `declared`, by symbol, which they
    /// consume; entries it omits compile theirs in.
    pub(crate) fn all(
        specs: Vec<CubinSpec>,
        mut declared: BTreeMap<String, String>,
    ) -> Result<Vec<Self>, Refusal> {
        let cubins = specs
            .into_iter()
            .map(|spec| {
                let arithmetic = spec
                    .entries
                    .iter()
                    .map(|entry| (entry.symbol.clone(), declared.remove(&entry.symbol)))
                    .collect();
                Self::new(spec, arithmetic)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !declared.is_empty() {
            return Err(Refusal("declared arithmetic names an entry of no cubin"));
        }
        Ok(cubins)
    }

    pub fn spec(&self) -> &CubinSpec {
        &self.spec
    }

    pub fn into_spec(self) -> CubinSpec {
        self.spec
    }

    /// The same cubin under another name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.spec.name = name.into();
        self
    }

    /// Entry `symbol`'s declared arithmetic; `None` when its wrapper
    /// compiles it in or the cubin has no such entry.
    pub fn arithmetic(&self, symbol: &str) -> Option<&str> {
        self.arithmetic.get(symbol)?.as_deref()
    }

    /// Declarations of this legacy cubin's entries: each typed wrapper's
    /// digest and, where declared, its arithmetic.
    pub fn legacy_declarations(
        &self,
        operation: &str,
        contract: &str,
        correctness_case: &str,
    ) -> Vec<EntryDeclaration> {
        self.spec
            .entries
            .iter()
            .map(|entry| {
                let mut requirements = BTreeMap::from([(
                    "typed_entry".to_owned(),
                    kernel_build::digest(format!("{entry:?}").as_bytes()),
                )]);
                if let Some(declared) = &self.arithmetic[&entry.symbol] {
                    requirements.insert("arithmetic".into(), declared.clone());
                }
                EntryDeclaration {
                    symbol: entry.symbol.clone(),
                    operation: operation.into(),
                    contract: contract.into(),
                    implementation: entry.symbol.clone(),
                    requirements,
                    correctness_case: Some(correctness_case.into()),
                }
            })
            .collect()
    }

    /// This legacy cubin, under its existing name, with
    /// [`Self::legacy_declarations`].
    pub fn select(
        self,
        operation: &str,
        contract: &str,
        correctness_case: &str,
    ) -> Result<SelectedCubin, kernel_build::BuildError> {
        let declarations = self.legacy_declarations(operation, contract, correctness_case);
        SelectedCubin::existing(self.spec, declarations)
    }
}

/// Whether `entry` takes its K splits or segments as a launch argument.
fn launch_splits(entry: &EntrySpec) -> bool {
    entry
        .params
        .iter()
        .any(|param| matches!(param, EntryParam::U32 { name } if name == "splits"))
}
