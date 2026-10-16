//! Structural validation and checked geometry binding.

use std::collections::BTreeSet;

use kernel_core::precision::shape_elements;

use crate::registry::{Registry, valid_name};
use crate::*;

/// Maximum source instances in one declaration binding.
pub const MAX_INSTANCES: usize = 1_000_000;

impl Dim {
    /// Resolve a positive dimension, refusing missing keys, overflow and
    /// fractional quotients before any slices or allocations are derived.
    pub fn resolve(&self, geometry: &Geometry) -> Result<u64, Error> {
        self.resolve_at(geometry, 0, false)
    }

    /// Resolve an offset expression, allowing zero but never a zero divisor.
    pub fn resolve_offset(&self, geometry: &Geometry) -> Result<u64, Error> {
        self.resolve_at(geometry, 0, true)
    }

    fn resolve_at(
        &self,
        geometry: &Geometry,
        depth: usize,
        allow_zero: bool,
    ) -> Result<u64, Error> {
        if depth > 32 {
            return Err(Error("dimension expression exceeds depth 32".into()));
        }
        let child = |d: &Dim| d.resolve_at(geometry, depth + 1, allow_zero);
        let value = match self {
            Dim::Literal(value) => *value,
            Dim::Key(key) => {
                if !valid_name(key) {
                    return Err(Error(format!("invalid geometry key `{key}`")));
                }
                *geometry
                    .get(key)
                    .ok_or_else(|| Error(format!("missing geometry key `{key}`")))?
            }
            Dim::Expr(expression) => match expression.as_ref() {
                DimExpr::Sum { terms } => {
                    if terms.len() < 2 {
                        return Err(Error("dimension sum needs at least two terms".into()));
                    }
                    terms.iter().try_fold(0u64, |acc, d| {
                        acc.checked_add(child(d)?)
                            .ok_or_else(|| Error("dimension sum overflows".into()))
                    })?
                }
                DimExpr::Product { factors } => {
                    if factors.len() < 2 {
                        return Err(Error("dimension product needs at least two factors".into()));
                    }
                    factors.iter().try_fold(1u64, |acc, d| {
                        acc.checked_mul(child(d)?)
                            .ok_or_else(|| Error("dimension product overflows".into()))
                    })?
                }
                DimExpr::Quotient {
                    numerator,
                    denominator,
                } => {
                    let n = child(numerator)?;
                    let d = child(denominator)?;
                    if d == 0 {
                        return Err(Error("dimension divisor must be positive".into()));
                    }
                    if n % d != 0 {
                        return Err(Error(format!("dimension quotient {n}/{d} is not exact")));
                    }
                    n / d
                }
            },
        };
        if value == 0 && !allow_zero {
            return Err(Error("dimension must be positive".into()));
        }
        Ok(value)
    }
}

impl LayerIndices {
    /// Concrete indices in declaration order, with a bounded expansion.
    pub fn resolve(&self, geometry: &Geometry) -> Result<Vec<u64>, Error> {
        match self {
            Self::Range { start, end } => {
                let start = start.resolve_offset(geometry)?;
                let end = end.resolve(geometry)?;
                if end <= start || end - start > MAX_INSTANCES as u64 {
                    return Err(Error("layer range must be nonempty and bounded".into()));
                }
                Ok((start..end).collect())
            }
            Self::List { indices } => {
                let indices = indices
                    .iter()
                    .map(|d| d.resolve_offset(geometry))
                    .collect::<Result<Vec<_>, _>>()?;
                if indices.is_empty()
                    || indices.len() > MAX_INSTANCES
                    || indices.windows(2).any(|w| w[0] >= w[1])
                {
                    return Err(Error(
                        "layer indices must be nonempty, bounded and strictly increasing".into(),
                    ));
                }
                Ok(indices.clone())
            }
        }
    }
}

fn name(id: &str, ids: &mut BTreeSet<String>) -> Result<(), Error> {
    if !valid_name(id) {
        return Err(Error(format!("invalid id `{id}`")));
    }
    if !ids.insert(id.into()) {
        return Err(Error(format!("duplicate id `{id}`")));
    }
    Ok(())
}

/// Refuse empty/control names and any placeholder outside `allowed`.
pub(crate) fn pattern(text: &str, allowed: &[&str]) -> Result<(), Error> {
    if text.is_empty() || text.len() > 1024 || text.chars().any(char::is_control) {
        return Err(Error(format!("invalid tensor pattern `{text}`")));
    }
    let mut rest = text.to_owned();
    for placeholder in allowed {
        let token = format!("{{{placeholder}}}");
        if rest.matches(&token).count() > 1 {
            return Err(Error(format!("tensor pattern `{text}` repeats {token}")));
        }
        rest = rest.replace(&token, "");
    }
    if rest.contains(['{', '}']) {
        return Err(Error(format!(
            "tensor pattern `{text}` has an unknown placeholder"
        )));
    }
    Ok(())
}

pub(crate) fn check_source(
    source: &TensorSource,
    stems: &[LayerStem],
    allowed: &[&str],
) -> Result<(), Error> {
    pattern(&source.tensor, allowed)?;
    match &source.stem {
        Some(id) => {
            if !stems.iter().any(|s| s.id == *id) {
                return Err(Error(format!(
                    "tensor `{}` names missing stem `{id}`",
                    source.tensor
                )));
            }
            if !source.tensor.contains("{stem}") {
                return Err(Error(format!("tensor `{}` needs {{stem}}", source.tensor)));
            }
        }
        None if source.tensor.contains("{stem}") => {
            return Err(Error(format!(
                "tensor `{}` has {{stem}} without a stem",
                source.tensor
            )));
        }
        None => {}
    }
    Ok(())
}

fn checked_shape(id: &str, shape: &[Dim], geometry: &Geometry) -> Result<Vec<u64>, Error> {
    let shape = shape
        .iter()
        .map(|d| d.resolve(geometry))
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() {
        return Err(Error(format!("{id}: shape must be nonempty")));
    }
    shape_elements(id, &shape).map_err(|e| Error(e.to_string()))?;
    Ok(shape)
}

pub(crate) fn multiply(a: u64, b: u64) -> Result<u64, Error> {
    a.checked_mul(b)
        .ok_or_else(|| Error("source dimensions overflow".into()))
}

type SourceGeometry = (Vec<u64>, Vec<(&'static str, u64)>);

pub(crate) fn source_geometry(
    projection: &Projection,
    geometry: &Geometry,
) -> Result<SourceGeometry, Error> {
    let source = projection
        .source
        .as_ref()
        .ok_or_else(|| Error("materialized projection needs a source".into()))?;
    let mut rows = projection.shape[0].resolve(geometry)?;
    let k = projection.shape[1].resolve(geometry)?;
    let mut counts = vec![("operand", u64::from(projection.operand_count()))];
    if !source.tensor.contains("{operand}") {
        rows = multiply(rows, projection.operand_count().into())?;
    }
    let mut shape = Vec::new();
    match &projection.kind {
        crate::ProjectionKind::Dense {} => {}
        crate::ProjectionKind::ExpertBank { experts, .. } => {
            let experts = experts.resolve(geometry)?;
            counts.push(("expert", experts));
            if !source.tensor.contains("{expert}") {
                shape.push(experts);
            }
        }
        crate::ProjectionKind::LookupTable {} => {}
    }
    shape.extend([rows, k]);
    kernel_core::precision::shape_elements(&projection.id, &shape)
        .map_err(|e| Error(e.to_string()))?;
    Ok((shape, counts))
}

fn input_split(split: &SplitClass, geometry: &Geometry) -> Result<(u8, Option<u64>), Error> {
    Ok(match split {
        SplitClass::Row { tile } => (0, Some(tile.resolve(geometry)?)),
        SplitClass::ExpertOwned { .. } => (1, None),
        SplitClass::ExpertSharded {
            axis: SliceAxis::Rows,
            ..
        } => (2, None),
        SplitClass::ExpertSharded {
            axis: SliceAxis::K,
            tile,
        } => (2, Some(tile.resolve(geometry)?)),
        _ => (0, None),
    })
}

type ResolvedUses = std::collections::HashSet<(
    kernel_core::precision::ExecutionPhase,
    u64,
    bool,
    kernel_core::contract::DType,
)>;

fn resolved_uses(uses: &[Use], geometry: &Geometry) -> Result<ResolvedUses, Error> {
    uses.iter()
        .map(|u| {
            Ok((
                u.phase,
                u.capacity.resolve(geometry)?,
                u.graph,
                u.contract
                    .as_ref()
                    .map_or(kernel_core::contract::DType::Bf16, |c| c.output_dtype),
            ))
        })
        .collect()
}

impl Declaration {
    /// Validate every geometry-dependent structural contract. Registry
    /// parameters are typed even when callers construct a declaration by hand.
    pub fn validate(&self, geometry: &Geometry, registry: &Registry) -> Result<(), Error> {
        self.expanded_profiles(geometry)?
            .validate_expanded(geometry, registry)
    }

    fn validate_expanded(&self, geometry: &Geometry, registry: &Registry) -> Result<(), Error> {
        if self.schema != SCHEMA {
            return Err(Error(format!(
                "unsupported projection schema `{}`",
                self.schema
            )));
        }
        if !valid_name(&self.namespace) {
            return Err(Error(format!("invalid namespace `{}`", self.namespace)));
        }
        if self.projections.is_empty() {
            return Err(Error("declaration has no projections".into()));
        }
        let mut ids = BTreeSet::new();
        let mut stem_instances = 0usize;
        for stem in &self.layer_stems {
            name(&stem.id, &mut ids)?;
            pattern(&stem.template, &["i"])?;
            if !stem.template.contains("{i}") {
                return Err(Error(format!("stem `{}` needs {{i}}", stem.id)));
            }
            stem_instances = stem_instances
                .checked_add(stem.indices.resolve(geometry)?.len())
                .filter(|n| *n <= MAX_INSTANCES)
                .ok_or_else(|| Error("stem expansion exceeds instance limit".into()))?;
        }
        for p in &self.projections {
            let result = (|| {
                name(&p.id, &mut ids)?;
                if let Some(id) = &p.stem
                    && !self.layer_stems.iter().any(|s| s.id == *id)
                {
                    return Err(Error("projection has an unknown owning stem".into()));
                }
                let mut operands = BTreeSet::new();
                for operand in &p.operands {
                    name(operand, &mut operands)?;
                }
                if p.operands.len() > MAX_INSTANCES {
                    return Err(Error("operand count exceeds instance limit".into()));
                }
                if p.operand_count() == 0 {
                    return Err(Error("operand_count must be positive".into()));
                }
                let [rows, _k] = checked_shape(&p.id, &p.shape, geometry)?
                    .try_into()
                    .unwrap();
                let mut allowed = vec!["stem", "operand"];
                match &p.kind {
                    ProjectionKind::Dense {} => {}
                    ProjectionKind::ExpertBank { experts, top_k } => {
                        allowed.push("expert");
                        let experts = experts.resolve(geometry)?;
                        let top_k = top_k.resolve(geometry)?;
                        if experts > u32::MAX.into() || top_k > experts {
                            return Err(Error(
                                "expert top_k must be within the u32 expert count".into(),
                            ));
                        }
                    }
                    ProjectionKind::LookupTable {} => {
                        if p.operand_count() != 1 {
                            return Err(Error("lookup table requires one operand".into()));
                        }
                    }
                }
                if p.lookup.is_some() {
                    if p.source.is_some() {
                        return Err(Error(
                            "lookup assembly forbids a redundant projection source".into(),
                        ));
                    }
                } else if p.source.is_some() || p.assembly.is_empty() {
                    check_source(
                        p.source
                            .as_ref()
                            .ok_or_else(|| Error("projection needs a source".into()))?,
                        &self.layer_stems,
                        &allowed,
                    )?;
                }
                if p.lookup.is_none() && p.assembly.is_empty() {
                    source_geometry(p, geometry)?;
                }
                let mut part_ids = BTreeSet::new();
                let mut total = 0u64;
                for part in &p.parts {
                    name(&part.id, &mut part_ids)?;
                    total = total
                        .checked_add(part.rows.resolve(geometry)?)
                        .ok_or_else(|| Error("part rows overflow".into()))?;
                }
                if !p.parts.is_empty() && total != rows {
                    return Err(Error(format!("part rows total {total}, expected {rows}")));
                }
                let legal = match &p.split {
                    SplitClass::Replicate {} => true,
                    SplitClass::Column { tile } | SplitClass::Row { tile } => {
                        tile.resolve(geometry)?;
                        matches!(p.kind, ProjectionKind::Dense {})
                    }
                    SplitClass::HeadGrouped {
                        heads,
                        head_dim,
                        tile,
                    } => {
                        tile.resolve(geometry)?;
                        let grouped = heads
                            .resolve(geometry)?
                            .checked_mul(head_dim.resolve(geometry)?)
                            .ok_or_else(|| Error("head-grouped rows overflow".into()))?;
                        let chunks: Vec<_> = if p.parts.is_empty() {
                            vec![(rows, grouped)]
                        } else {
                            p.parts
                                .iter()
                                .map(|part| {
                                    let part_heads =
                                        part.heads.as_ref().unwrap_or(heads).resolve(geometry)?;
                                    let group = part_heads
                                        .checked_mul(head_dim.resolve(geometry)?)
                                        .ok_or_else(|| {
                                            Error("part head-grouped rows overflow".into())
                                        })?;
                                    Ok((part.rows.resolve(geometry)?, group))
                                })
                                .collect::<Result<_, Error>>()?
                        };
                        if chunks.iter().any(|(rows, group)| rows != group) {
                            return Err(Error(
                                "rows must equal head count times head dimension".into(),
                            ));
                        }
                        matches!(p.kind, ProjectionKind::Dense {})
                    }
                    SplitClass::SegmentedHeads {
                        segments,
                        order,
                        tile,
                    } => {
                        tile.resolve(geometry)?;
                        crate::segments::validate(segments, order, rows, geometry)?;
                        matches!(p.kind, ProjectionKind::Dense {})
                    }
                    SplitClass::ExpertOwned { tile } | SplitClass::ExpertSharded { tile, .. } => {
                        tile.resolve(geometry)?;
                        matches!(p.kind, ProjectionKind::ExpertBank { .. })
                    }
                    SplitClass::VocabSharded { tile } => {
                        tile.resolve(geometry)?;
                        matches!(p.kind, ProjectionKind::LookupTable { .. })
                    }
                };
                if !legal {
                    return Err(Error("split class is illegal for projection kind".into()));
                }
                if !matches!(p.split, SplitClass::HeadGrouped { .. })
                    && p.parts.iter().any(|p| p.heads.is_some())
                {
                    return Err(Error("part heads require a head-grouped split".into()));
                }
                if !valid_name(&p.numerical_contract) {
                    return Err(Error("invalid numerical contract or format id".into()));
                }
                if p.epilogues.gelu.is_some() && p.epilogues.swiglu.is_some() {
                    return Err(Error("GELU and SwiGLU epilogues cannot coexist".into()));
                }
                if p.epilogues.swiglu.is_some() && p.operand_count() != 2 {
                    return Err(Error("SwiGLU requires two operands".into()));
                }
                if p.epilogues.weighted_moe_combine
                    && !matches!(p.kind, ProjectionKind::ExpertBank { .. })
                {
                    return Err(Error("weighted combine requires an expert bank".into()));
                }
                if matches!(p.kind, ProjectionKind::LookupTable { .. })
                    && p.epilogues != Epilogues::default()
                {
                    return Err(Error(
                        "lookup table does not admit projection epilogues".into(),
                    ));
                }
                if p.uses.is_empty()
                    && !p
                        .stem()
                        .as_ref()
                        .and_then(|id| self.layer_stems.iter().find(|s| s.id == *id))
                        .is_some_and(|s| s.execution.as_ref().is_some_and(|e| e.indices.is_empty()))
                {
                    return Err(Error("projection has no uses".into()));
                }
                let mut uses = BTreeSet::new();
                for use_ in &p.uses {
                    let capacity = use_.capacity.resolve(geometry)?;
                    if capacity > u32::MAX.into() {
                        return Err(Error("use capacity exceeds u32".into()));
                    }
                    let phase = match use_.phase {
                        kernel_core::precision::ExecutionPhase::Prefill => 0,
                        kernel_core::precision::ExecutionPhase::Decode => 1,
                        kernel_core::precision::ExecutionPhase::Verify => 2,
                    };
                    let output = use_
                        .contract
                        .as_ref()
                        .map(|c| format!("{:?}/{}", c.output_dtype, c.numerical_contract));
                    if !uses.insert((phase, capacity, use_.graph, output)) {
                        return Err(Error("duplicate projection use".into()));
                    }
                }
                Ok(())
            })();
            result.map_err(|e: Error| Error(format!("projection `{}`: {e}", p.id)))?;
        }
        for group in &self.fusion_groups {
            name(&group.id, &mut ids)?;
            if !valid_name(&group.input) || group.members.len() < 2 {
                return Err(Error(format!(
                    "fusion `{}` needs an input and at least two members",
                    group.id
                )));
            }
            if !group.uses.is_empty() {
                let first = self
                    .projections
                    .iter()
                    .find(|p| p.id == group.members[0])
                    .ok_or_else(|| Error("fusion has a missing member".into()))?;
                let group_uses = resolved_uses(&group.uses, geometry)?;
                if group_uses.len() != group.uses.len()
                    || group_uses != resolved_uses(&first.uses, geometry)?
                {
                    return Err(Error(
                        "fusion use buckets must match member use buckets without duplicates"
                            .into(),
                    ));
                }
            }
            let mut members = BTreeSet::new();
            let mut first: Option<&Projection> = None;
            for id in &group.members {
                if !members.insert(id) {
                    return Err(Error(format!(
                        "fusion `{}` repeats member `{id}`",
                        group.id
                    )));
                }
                let member = self
                    .projections
                    .iter()
                    .find(|p| p.id == *id)
                    .ok_or_else(|| {
                        Error(format!("fusion `{}` has missing member `{id}`", group.id))
                    })?;
                if let Some(first) = first {
                    if input_split(&first.split, geometry)? != input_split(&member.split, geometry)?
                    {
                        return Err(Error(format!(
                            "fusion `{}` member `{id}` has another input split",
                            group.id
                        )));
                    }
                    let same_kind = match (&first.kind, &member.kind) {
                        (ProjectionKind::Dense {}, ProjectionKind::Dense {}) => true,
                        (
                            ProjectionKind::ExpertBank {
                                experts: a,
                                top_k: b,
                            },
                            ProjectionKind::ExpertBank {
                                experts: c,
                                top_k: d,
                            },
                        ) => {
                            a.resolve(geometry)? == c.resolve(geometry)?
                                && b.resolve(geometry)? == d.resolve(geometry)?
                        }
                        _ => false,
                    };
                    if !same_kind
                        || first.shape[1].resolve(geometry)? != member.shape[1].resolve(geometry)?
                        || first.stem() != member.stem()
                        || resolved_uses(&first.uses, geometry)?
                            != resolved_uses(&member.uses, geometry)?
                    {
                        return Err(Error(format!(
                            "fusion `{}` member `{id}` does not share the input contract",
                            group.id
                        )));
                    }
                }
                first = Some(member);
            }
        }
        for side in &self.side_tensors {
            name(&side.id, &mut ids)?;
            check_source(&side.source, &self.layer_stems, &["stem"])?;
            crate::contract_validation::shape(&side.shape, geometry)?;
            side.validate_placement(geometry)?;
            if side
                .dtypes
                .iter()
                .any(|d| !matches!(d, DType::Bf16 | DType::F32 | DType::I64))
            {
                return Err(Error("side sources admit only Bf16, F32 and I64".into()));
            }
            if side.admitted_dtypes().is_empty()
                || side
                    .admitted_dtypes()
                    .iter()
                    .enumerate()
                    .any(|(i, d)| side.admitted_dtypes()[..i].contains(d))
            {
                return Err(Error(format!(
                    "side `{}` needs distinct admitted dtypes",
                    side.id
                )));
            }
            let consumer_stem = if let Some(p) = self
                .projections
                .iter()
                .find(|p| side.consumer.matches_declared(&p.id))
            {
                p.stem()
            } else if let Some(g) = self
                .fusion_groups
                .iter()
                .find(|g| side.consumer.matches_declared(&g.id))
            {
                self.projections
                    .iter()
                    .find(|p| p.id == g.members[0])
                    .unwrap()
                    .stem()
            } else if self.consumer_exists(&side.consumer) {
                side.source.stem.as_deref()
            } else {
                return Err(Error(format!(
                    "side `{}` has missing consumer `{}`",
                    side.id, side.consumer
                )));
            };
            if side.source.stem.as_deref() != consumer_stem {
                return Err(Error(format!(
                    "side `{}` and its consumer use different layer stems",
                    side.id
                )));
            }
        }
        for external in &self.external_tensors {
            name(&external.id, &mut ids)?;
        }
        self.validate_drafters()?;
        self.validate_offers()?;
        for c in self
            .projections
            .iter()
            .filter_map(|p| p.composition.as_ref())
            .chain(
                self.fusion_groups
                    .iter()
                    .filter_map(|g| g.composition.as_ref()),
            )
        {
            for stage in c.pre.iter().chain(&c.post) {
                name(&stage.id, &mut ids)?;
            }
        }
        self.validate_contracts(geometry, registry)?;
        Ok(())
    }
}
