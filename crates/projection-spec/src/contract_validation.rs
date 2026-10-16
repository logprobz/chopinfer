//! Structural checks for typed composition and ownership contracts.

use std::collections::BTreeSet;

use kernel_core::precision::format::registry as formats;

use crate::contracts::*;
use crate::registry::{Registry, Target, valid_name};
use crate::*;

pub(crate) fn offset(dim: &Dim, geometry: &Geometry) -> Result<u64, Error> {
    dim.resolve_offset(geometry)
}

pub(crate) fn range(range: &RowRange, geometry: &Geometry) -> Result<(u64, u64), Error> {
    let start = offset(&range.start, geometry)?;
    let end = start
        .checked_add(range.length.resolve(geometry)?)
        .ok_or_else(|| Error("logical row range overflows".into()))?;
    Ok((start, end))
}

pub(crate) fn shape(shape: &[Dim], geometry: &Geometry) -> Result<Vec<u64>, Error> {
    let values = shape
        .iter()
        .map(|d| d.resolve(geometry))
        .collect::<Result<Vec<_>, _>>()?;
    values.iter().try_fold(1u64, |n, d| {
        n.checked_mul(*d)
            .filter(|n| *n <= kernel_core::precision::MAX_TENSOR_BYTES * 2)
            .ok_or_else(|| Error("side/port shape exceeds supported element bound".into()))
    })?;
    Ok(values)
}

fn distinct<T: Ord>(values: impl IntoIterator<Item = T>, at: &str) -> Result<(), Error> {
    let mut seen = BTreeSet::new();
    for value in values {
        if !seen.insert(value) {
            return Err(Error(format!("{at}: duplicate entry")));
        }
    }
    Ok(())
}

fn support(rules: &[SupportRule]) -> Result<(), Error> {
    for rule in rules {
        if rule.ranks.contains(&0) {
            return Err(Error("support rank must be positive".into()));
        }
        distinct(rule.ranks.iter(), "support ranks")?;
        let mut ids = std::collections::HashSet::new();
        for id in &rule.formats {
            formats::by_id(*id).map_err(|e| Error(e.to_string()))?;
            if !ids.insert(id) {
                return Err(Error("duplicate support format".into()));
            }
        }
        let mut phases = std::collections::HashSet::new();
        if rule.phases.iter().any(|p| !phases.insert(p)) {
            return Err(Error("duplicate support phase".into()));
        }
    }
    Ok(())
}

fn variant(
    v: &Variant,
    target: Target,
    d: &Declaration,
    geometry: &Geometry,
    registry: &Registry,
) -> Result<(), Error> {
    if !valid_name(&v.numerical_contract) || v.inputs.is_empty() || v.outputs.is_empty() {
        return Err(Error(format!(
            "variant `{}` needs typed inputs/outputs and a numerical contract",
            v.id
        )));
    }
    for ports in [&v.inputs, &v.outputs] {
        distinct(ports.iter().map(|p| &p.id), "variant ports")?;
        for port in ports {
            port.validate(geometry)?;
        }
    }
    if target == Target::SourceTransform
        && (!v.inputs.iter().any(|p| p.format.is_some())
            || !v.outputs.iter().any(|p| p.format.is_some()))
    {
        return Err(Error(
            "source transform needs declared input/output weight formats".into(),
        ));
    }
    if registry.resolve(&d.namespace, v, target, geometry)? != v.parameters {
        return Err(Error(format!(
            "variant `{}` must supply canonical parameters",
            v.id
        )));
    }
    Ok(())
}

fn composition(
    c: &Composition,
    target: Target,
    d: &Declaration,
    geometry: &Geometry,
    registry: &Registry,
) -> Result<(), Error> {
    for (format, recipe) in &c.numerical_contracts {
        kernel_core::precision::format::registry::by_id(*format)
            .map_err(|e| Error(e.to_string()))?;
        if !valid_name(&recipe.numerical_contract) {
            return Err(Error("invalid format numerical contract".into()));
        }
    }
    for (format, implementation) in &c.implementations {
        kernel_core::precision::format::registry::by_id(*format)
            .map_err(|e| Error(e.to_string()))?;
        variant(implementation, target, d, geometry, registry)?;
        if c.numerical_contracts
            .get(format)
            .is_some_and(|r| r.numerical_contract != implementation.numerical_contract)
        {
            return Err(Error(
                "format implementation disagrees with its immutable numerical contract".into(),
            ));
        }
    }
    let mut stages = BTreeSet::new();
    for stage in c.pre.iter().chain(&c.post) {
        if !valid_name(&stage.id) || !stages.insert(&stage.id) || stage.phases.is_empty() {
            return Err(Error(
                "composition stages need unique ids and explicit phases".into(),
            ));
        }
        let mut phases = std::collections::HashSet::new();
        if stage.phases.iter().any(|p| !phases.insert(p)) {
            return Err(Error("duplicate stage phase".into()));
        }
        variant(&stage.contract, Target::Stage, d, geometry, registry)?;
        distinct(stage.mutates.iter(), "stage state mutations")?;
        for mutation in &stage.mutates {
            let input = stage
                .contract
                .inputs
                .iter()
                .find(|p| p.id == *mutation && p.state)
                .ok_or_else(|| {
                    Error(format!(
                        "stage `{}` mutates undeclared input state `{mutation}`",
                        stage.id
                    ))
                })?;
            if !stage.contract.outputs.iter().any(|p| {
                p.state
                    && p.binding == input.binding
                    && p.dtype == input.dtype
                    && p.interpretation == input.interpretation
                    && p.shape == input.shape
            }) {
                return Err(Error(format!(
                    "stage `{}` mutation `{mutation}` has no matching state output",
                    stage.id
                )));
            }
        }
    }
    if let Some(v) = &c.implementation {
        variant(v, target, d, geometry, registry)?;
    }
    Ok(())
}

fn validate_uses(
    uses: &[Use],
    default: Option<&Variant>,
    target: Target,
    d: &Declaration,
    geometry: &Geometry,
    registry: &Registry,
) -> Result<(), Error> {
    for use_ in uses {
        if let Some(c) = &use_.contract {
            let compiled = c.compiled_capacity.resolve(geometry)?;
            if compiled > u32::MAX.into() || !valid_name(&c.numerical_contract) {
                return Err(Error(
                    "invalid compiled capacity or per-use numerical contract".into(),
                ));
            }
            let expanded = crate::validate::multiply(
                use_.capacity.resolve(geometry)?,
                c.row_multiplier.resolve(geometry)?,
            )?;
            if let Some(chunk) = &c.chunk_rows {
                if chunk.resolve(geometry)? > compiled {
                    return Err(Error("chunk rows exceed compiled capacity".into()));
                }
            } else if expanded > compiled {
                return Err(Error(
                    "expanded rows need an explicit chunking capacity".into(),
                ));
            }
            match (&c.per_item_verify, &c.aggregate_verify) {
                (Some(item), Some(aggregate))
                    if use_.phase == kernel_core::precision::ExecutionPhase::Verify =>
                {
                    if aggregate.resolve(geometry)? < use_.capacity.resolve(geometry)? {
                        return Err(Error(
                            "aggregate verify limit is below requested capacity".into(),
                        ));
                    }
                    if item.resolve(geometry)? > aggregate.resolve(geometry)? {
                        return Err(Error(
                            "per-item verify limit exceeds aggregate limit".into(),
                        ));
                    }
                }
                (None, None) if use_.phase != kernel_core::precision::ExecutionPhase::Verify => {}
                _ => {
                    return Err(Error(
                        "verify use needs both per-item and aggregate limits".into(),
                    ));
                }
            }
            support(&c.support)?;
            if let Some(v) = &c.implementation {
                if default == Some(v) {
                    return Err(Error(
                        "per-use override repeats the default implementation".into(),
                    ));
                }
                variant(v, target, d, geometry, registry)?;
            }
        }
    }
    Ok(())
}

impl Declaration {
    pub(crate) fn validate_contracts(
        &self,
        geometry: &Geometry,
        registry: &Registry,
    ) -> Result<(), Error> {
        self.validate_conditions()?;
        for stem in &self.layer_stems {
            support(&stem.support)?;
            if let Some(execution) = &stem.execution {
                let indices = stem.indices.resolve(geometry)?;
                let executed = execution
                    .indices
                    .iter()
                    .map(|d| offset(d, geometry))
                    .collect::<Result<Vec<_>, _>>()?;
                let aliases = execution
                    .aliases
                    .iter()
                    .map(|a| {
                        Ok((
                            offset(&a.execution_index, geometry)?,
                            offset(&a.source_index, geometry)?,
                        ))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                distinct(executed.iter(), "execution layers")?;
                distinct(aliases.iter().map(|a| a.0), "layer aliases")?;
                for ((execution_index, source_index), alias) in
                    aliases.iter().zip(&execution.aliases)
                {
                    let source_indices = if let Some(id) = &alias.source_stem {
                        self.layer_stems
                            .iter()
                            .find(|s| s.id == *id)
                            .ok_or_else(|| {
                                Error("layer alias names an unknown source stem".into())
                            })?
                            .indices
                            .resolve(geometry)?
                    } else {
                        indices.clone()
                    };
                    if !executed.contains(execution_index) || !source_indices.contains(source_index)
                    {
                        return Err(Error(format!(
                            "stem `{}` has an invalid execution alias",
                            stem.id
                        )));
                    }
                }
                if executed
                    .iter()
                    .any(|i| !indices.contains(i) && !aliases.iter().any(|a| a.0 == *i))
                {
                    return Err(Error(format!(
                        "stem `{}` executes an unmaterialized layer",
                        stem.id
                    )));
                }
            }
        }
        for p in &self.projections {
            distinct(
                p.inputs.iter().map(|input| &input.operand),
                "activation input operands",
            )?;
            for input in &p.inputs {
                if p.operand_index(&input.operand).is_none()
                    || !valid_name(&input.binding)
                    || input.segments.is_empty()
                {
                    return Err(Error(
                        "activation input needs an operand, binding and segments".into(),
                    ));
                }
                for segment in &input.segments {
                    if !valid_name(&segment.source) {
                        return Err(Error("invalid activation source binding".into()));
                    }
                    segment.range(
                        self,
                        geometry,
                        crate::derive::Topology { ranks: 1, rank: 0 },
                    )?;
                }
            }
            if matches!(p.kind, crate::ProjectionKind::LookupTable {})
                && p.uses.iter().filter_map(|u| u.contract.as_ref()).any(|c| {
                    c.activation
                        .is_some_and(|a| a != kernel_core::contract::ActivationFormat::Bf16)
                })
            {
                return Err(Error("lookup kind has no activation quantizer".into()));
            }
            for v in &p.source_transforms {
                variant(v, Target::SourceTransform, self, geometry, registry)?;
            }
            if let Some(c) = &p.composition {
                composition(c, Target::Projection, self, geometry, registry)?;
            }
            distinct(
                p.operand_splits.iter().map(|s| &s.operand),
                "operand splits",
            )?;
            for split in &p.operand_splits {
                if p.operand_index(&split.operand).is_none() {
                    return Err(Error("operand split index is out of range".into()));
                }
                split.tile.resolve(geometry)?;
                let legal = match p.split {
                    SplitClass::Replicate {} => false,
                    SplitClass::Row { .. } => split.axis == SliceAxis::K,
                    SplitClass::ExpertSharded { axis, .. } => split.axis == axis,
                    _ => split.axis == SliceAxis::Rows,
                };
                if !legal {
                    return Err(Error(
                        "per-operand split axis disagrees with split class".into(),
                    ));
                }
            }
            let mut next = 0;
            for part in &p.parts {
                if let Some(rows) = &part.range {
                    let (start, end) = range(rows, geometry)?;
                    if start != next || end - start != part.rows.resolve(geometry)? {
                        return Err(Error(
                            "part row ranges must cover ordered logical rows".into(),
                        ));
                    }
                }
                next += part.rows.resolve(geometry)?;
                let mut run_end = 0;
                distinct(part.head_runs.iter().map(|r| &r.id), "head runs")?;
                for run in &part.head_runs {
                    if let SplitClass::HeadGrouped { head_dim, .. } = &p.split
                        && run.head_dim.resolve(geometry)? != head_dim.resolve(geometry)?
                    {
                        return Err(Error(
                            "head run head dimension differs from split head dimension".into(),
                        ));
                    }
                    let (start, end) = range(&run.rows, geometry)?;
                    let heads = crate::validate::multiply(
                        run.heads.resolve(geometry)?,
                        run.head_dim.resolve(geometry)?,
                    )?;
                    if start != run_end
                        || end - start != heads
                        || end > part.rows.resolve(geometry)?
                    {
                        return Err(Error(
                            "head runs must cover ordered whole heads within a part".into(),
                        ));
                    }
                    run_end = end;
                }
                if !part.head_runs.is_empty() && run_end != part.rows.resolve(geometry)? {
                    return Err(Error("head runs do not cover the part".into()));
                }
                for v in &part.transforms {
                    variant(v, Target::SourceTransform, self, geometry, registry)?;
                }
            }
            validate_uses(
                &p.uses,
                p.composition
                    .as_ref()
                    .and_then(|c| c.implementation.as_ref()),
                Target::Projection,
                self,
                geometry,
                registry,
            )?;
            if let Some(lookup) = &p.lookup {
                if !matches!(p.kind, ProjectionKind::LookupTable { .. }) || !p.assembly.is_empty() {
                    return Err(Error(
                        "lookup assembly requires a materialized lookup projection".into(),
                    ));
                }
                for format in lookup.companions.formats.keys() {
                    formats::by_id(*format).map_err(|e| Error(e.to_string()))?;
                }
                let shards = lookup.shards.resolve(geometry)?;
                if shards.is_empty() || lookup.heads.is_empty() {
                    return Err(Error(
                        "lookup assembly needs shards and ordered heads".into(),
                    ));
                }
                distinct(shards.iter().map(|s| &s.id), "lookup shards")?;
                distinct(lookup.heads.iter().map(|h| &h.id), "lookup heads")?;
                let rows = p.shape[0].resolve(geometry)?;
                let mut end = 0;
                let mut shard_sources = std::collections::HashSet::new();
                for shard in &shards {
                    let (start, next) = range(&shard.rows, geometry)?;
                    if start != end {
                        return Err(Error(
                            "lookup checkpoint shards must cover contiguous rows".into(),
                        ));
                    }
                    crate::validate::check_source(&shard.source, &self.layer_stems, &["stem"])?;
                    if !shard_sources
                        .insert((shard.source.stem.as_deref(), shard.source.tensor.as_str()))
                    {
                        return Err(Error("duplicate lookup shard source".into()));
                    }
                    if shard.source.stem.as_deref() != p.stem() {
                        return Err(Error(
                            "lookup shards and projection need the same stem".into(),
                        ));
                    }
                    end = next;
                }
                if end != rows {
                    return Err(Error(
                        "lookup checkpoint shards do not cover table rows".into(),
                    ));
                }
                end = 0;
                for head in &lookup.heads {
                    let (start, next) = range(&head.vocabulary, geometry)?;
                    if start != end || next > rows {
                        return Err(Error(
                            "lookup head vocabularies must be ordered within table rows".into(),
                        ));
                    }
                    end = next;
                }
            }
            if !p.assembly.is_empty() {
                if matches!(p.kind, ProjectionKind::ExpertBank { .. })
                    && p.assembly
                        .iter()
                        .any(|m| m.inherit_split != p.assembly[0].inherit_split)
                {
                    return Err(Error(
                        "expert inheritance must agree across assembly members".into(),
                    ));
                }
                let mut rows = 0u64;
                for member in &p.assembly {
                    let source = self
                        .projections
                        .iter()
                        .find(|s| s.id == member.projection)
                        .ok_or_else(|| {
                            Error(format!(
                                "assembly `{}` names missing source `{}`",
                                p.id, member.projection
                            ))
                        })?;
                    if source.id == p.id || !source.assembly.is_empty() {
                        return Err(Error(
                            "assembly members must refer to materialized projections".into(),
                        ));
                    }
                    if source.shape[1].resolve(geometry)? != p.shape[1].resolve(geometry)? {
                        return Err(Error("assembly source K differs".into()));
                    }
                    let (_, end) = range(&member.rows, geometry)?;
                    if end > source.shape[0].resolve(geometry)?
                        || member
                            .operand
                            .as_ref()
                            .is_some_and(|name| source.operand_index(name).is_none())
                    {
                        return Err(Error("assembly source selection is out of bounds".into()));
                    }
                    if member.layer.is_none()
                        && let Some(stem) = source
                            .stem()
                            .and_then(|id| self.layer_stems.iter().find(|s| s.id == id))
                        && stem.indices.resolve(geometry)?.len() > 1
                    {
                        return Err(Error(
                            "assembly needs an explicit source layer for a multilayer source"
                                .into(),
                        ));
                    }
                    if let Some(selector) = &member.layer {
                        let source_stem = source
                            .stem()
                            .and_then(|id| self.layer_stems.iter().find(|s| s.id == id))
                            .ok_or_else(|| Error("assembly source layer has no stem".into()))?;
                        let currents = p
                            .stem()
                            .map(|id| {
                                self.layer_stems
                                    .iter()
                                    .find(|s| s.id == id)
                                    .unwrap()
                                    .indices
                                    .resolve(geometry)
                            })
                            .transpose()?
                            .map_or_else(
                                || vec![None],
                                |layers| layers.into_iter().map(Some).collect(),
                            );
                        for current in currents {
                            if !source_stem
                                .indices
                                .resolve(geometry)?
                                .contains(&selector.resolve(geometry, current)?)
                            {
                                return Err(Error(
                                    "assembly source layer is not materialized".into(),
                                ));
                            }
                        }
                    }
                    let operands = member
                        .operand
                        .as_ref()
                        .map_or(u64::from(source.operand_count()), |_| 1);
                    let member_rows =
                        crate::validate::multiply(member.rows.length.resolve(geometry)?, operands)?;
                    rows = rows
                        .checked_add(member_rows)
                        .ok_or_else(|| Error("assembly rows overflow".into()))?;
                }
                if rows
                    != crate::validate::multiply(
                        p.shape[0].resolve(geometry)?,
                        p.operand_count().into(),
                    )?
                {
                    return Err(Error("assembly rows do not match projection rows".into()));
                }
            }
        }
        for group in &self.fusion_groups {
            validate_uses(
                &group.uses,
                group
                    .composition
                    .as_ref()
                    .and_then(|c| c.implementation.as_ref()),
                Target::FusionGroup,
                self,
                geometry,
                registry,
            )?;
            if let Eligibility::All {
                formats: ids,
                phases,
                ..
            } = &group.eligibility
            {
                if ids.is_empty() {
                    return Err(Error("fusion predicate needs admitted formats".into()));
                }
                for id in ids {
                    formats::by_id(*id).map_err(|e| Error(e.to_string()))?;
                }
                let mut seen = std::collections::HashSet::new();
                if phases.iter().any(|p| !seen.insert(p)) {
                    return Err(Error("duplicate fusion predicate phase".into()));
                }
            }
            if let Some(c) = &group.composition {
                composition(c, Target::FusionGroup, self, geometry, registry)?;
            }
            if !group.outputs.is_empty() {
                distinct(
                    group.outputs.iter().map(|s| &s.member),
                    "fusion output segments",
                )?;
                if group.outputs.len() != group.members.len() {
                    return Err(Error("fusion needs one output segment per member".into()));
                }
                let mut occupied = std::collections::BTreeMap::<String, BTreeSet<u64>>::new();
                let mut row_strides = std::collections::BTreeMap::new();
                for segment in &group.outputs {
                    if !group.members.contains(&segment.member) {
                        return Err(Error("fusion output segment names a nonmember".into()));
                    }
                    let start = offset(&segment.offset, geometry)?;
                    if !valid_name(&segment.buffer) {
                        return Err(Error("invalid fusion output buffer".into()));
                    }
                    let row_stride = segment.row_stride.resolve(geometry)?;
                    if row_strides
                        .insert(segment.buffer.as_str(), row_stride)
                        .is_some_and(|previous| previous != row_stride)
                    {
                        return Err(Error(
                            "fusion segments in one buffer need a shared row stride".into(),
                        ));
                    }
                    let occupied = occupied.entry(segment.buffer.clone()).or_default();
                    let stride = 1;
                    let live = segment.live_columns.resolve(geometry)?;
                    let member = self
                        .projections
                        .iter()
                        .find(|p| p.id == segment.member)
                        .ok_or_else(|| Error("fusion output member is missing".into()))?;
                    let width = crate::validate::multiply(
                        member.shape[0].resolve(geometry)?,
                        member.operand_count().into(),
                    )?;
                    let width = if member.epilogues.swiglu.is_some() {
                        width / 2
                    } else {
                        width
                    };
                    if live != width {
                        return Err(Error(
                            "fusion segment live columns differ from member output width".into(),
                        ));
                    }
                    if live > crate::validate::MAX_INSTANCES as u64 {
                        return Err(Error("fusion output segment exceeds instance limit".into()));
                    }
                    let end = start
                        .checked_add(
                            (live - 1)
                                .checked_mul(stride)
                                .ok_or_else(|| Error("output stride overflows".into()))?,
                        )
                        .ok_or_else(|| Error("output segment overflows".into()))?;
                    if end >= row_stride {
                        return Err(Error("fusion output segment exceeds row stride".into()));
                    }
                    for column in 0..live {
                        if !occupied.insert(start + column * stride) {
                            return Err(Error("fusion output segments overlap".into()));
                        }
                    }
                }
            }
        }
        for side in &self.side_tensors {
            for view in &side.views {
                if !self.consumer_exists(&view.consumer) {
                    return Err(Error("side view has an unknown consumer".into()));
                }
                let legal = side
                    .admitted_dtypes()
                    .iter()
                    .all(|source| match view.conversion {
                        ViewConversion::Identity => *source == view.dtype,
                        ViewConversion::WidenBf16F32 => {
                            view.dtype == DType::F32 && matches!(source, DType::Bf16 | DType::F32)
                        }
                    });
                if !legal {
                    return Err(Error(
                        "side source-to-view conversion would narrow or change dtype".into(),
                    ));
                }
            }
            if let Some(norm) = &side.norm {
                if side.kind != SideKind::Norm {
                    return Err(Error("norm contract requires a norm side role".into()));
                }
                let width = norm.reduction_width.resolve(geometry)?;
                let groups = norm.groups.resolve(geometry)?;
                let epsilon = f32::from_bits(norm.epsilon_bits);
                if width > u64::from(i32::MAX as u32 - 256)
                    || groups > 65535
                    || crate::validate::multiply(width, groups)? > u32::MAX.into()
                    || !epsilon.is_finite()
                    || epsilon <= 0.0
                {
                    return Err(Error(
                        "unsupported norm reduction geometry or epsilon".into(),
                    ));
                }
                if norm.epsilon_bits != 1e-6f32.to_bits()
                    || !norm.one_plus_weight
                    || (groups > 1 && norm.affine == NormAffine::Shared)
                {
                    return Err(Error("shared norm kernel requires epsilon 1e-6, 1+w and per-group affine weights".into()));
                }
                distinct(
                    norm.weight_dtypes.iter().map(|d| format!("{d:?}")),
                    "norm weight dtypes",
                )?;
                if norm.weight_dtypes.is_empty()
                    || norm
                        .weight_dtypes
                        .iter()
                        .any(|d| !matches!(d, DType::Bf16 | DType::F32))
                    || side
                        .admitted_dtypes()
                        .iter()
                        .any(|d| !norm.weight_dtypes.contains(d))
                {
                    return Err(Error(
                        "norm source dtypes must be admitted BF16 or F32 weights".into(),
                    ));
                }
                let affine_elements = match norm.affine {
                    NormAffine::Shared => width,
                    NormAffine::PerGroup => crate::validate::multiply(width, groups)?,
                };
                let elements = shape(&side.shape, geometry)?.iter().product::<u64>();
                if affine_elements != elements {
                    return Err(Error(
                        "norm reduction and affine storage disagree with side shape".into(),
                    ));
                }
            }
        }
        distinct(
            self.external_tensors.iter().map(|e| &e.id),
            "external tensor ids",
        )?;
        let mut owners = std::collections::BTreeMap::new();
        for external in &self.external_tensors {
            if owners
                .insert(external.owner.as_str(), external.optional)
                .is_some_and(|previous| previous != external.optional)
            {
                return Err(Error(
                    "external owner optionality must agree across every owned role".into(),
                ));
            }
            if !valid_name(&external.owner) || !valid_name(&external.image_role) {
                return Err(Error(
                    "external tensor needs an owner and image role".into(),
                ));
            }
            crate::validate::check_source(&external.source, &self.layer_stems, &["stem"])?;
            shape(&external.shape, geometry)?;
        }
        Ok(())
    }

    pub(crate) fn consumer_exists(&self, consumer: &crate::Consumer) -> bool {
        let id = match consumer {
            crate::Consumer::Kernel { kernel } => return valid_name(kernel),
            crate::Consumer::Declared(id) => id.as_str(),
        };
        self.projections.iter().any(|p| p.id == id)
            || self.fusion_groups.iter().any(|g| g.id == id)
            || self
                .projections
                .iter()
                .filter_map(|p| p.composition.as_ref())
                .flat_map(|c| c.pre.iter().chain(&c.post))
                .any(|s| s.id == id)
            || self
                .fusion_groups
                .iter()
                .filter_map(|g| g.composition.as_ref())
                .flat_map(|c| c.pre.iter().chain(&c.post))
                .any(|s| s.id == id)
    }

    /// Refuse an unsupported execution without dropping materialized checkpoint roles.
    pub fn admit(
        &self,
        geometry: &Geometry,
        projection: &str,
        layer: Option<u64>,
        ranks: u32,
        phase: kernel_core::precision::ExecutionPhase,
        format: kernel_core::precision::format::FormatId,
    ) -> Result<(), Error> {
        self.expanded_profiles(geometry)?.admit_expanded(
            geometry,
            crate::Admission {
                projection,
                layer,
                ranks,
                phase,
                format,
            },
            None,
        )
    }

    /// [`Self::admit`] for a declaration already returned by
    /// `expanded_profiles`, with checkpoint conditions when bound.
    pub fn admit_expanded(
        &self,
        geometry: &Geometry,
        request: crate::Admission<'_>,
        bound: Option<&crate::bind::BoundDeclaration>,
    ) -> Result<(), Error> {
        let crate::Admission {
            projection,
            layer,
            ranks,
            phase,
            format,
        } = request;
        let p = self
            .projections
            .iter()
            .find(|p| p.id == projection)
            .ok_or_else(|| Error("unknown projection".into()))?;
        if ranks == 0 {
            return Err(Error("execution rank count must be positive".into()));
        }
        let check = |rules: &[SupportRule]| -> Result<(), Error> {
            let rules = crate::support::active(rules, bound)?;
            let matches = |rule: &SupportRule| {
                (rule.ranks.is_empty() || rule.ranks.contains(&ranks))
                    && (rule.formats.is_empty() || rule.formats.contains(&format))
                    && (rule.phases.is_empty() || rule.phases.contains(&phase))
            };
            if rules.iter().any(|r| !r.enabled && matches(r))
                || (rules.iter().any(|r| r.enabled)
                    && !rules.iter().any(|r| r.enabled && matches(r)))
            {
                return Err(Error(format!(
                    "projection `{projection}` is unsupported for this topology/use"
                )));
            }
            Ok(())
        };
        if let Some(id) = p.stem() {
            let stem = self
                .layer_stems
                .iter()
                .find(|s| s.id == id)
                .ok_or_else(|| Error("missing execution stem".into()))?;
            if let Some(execution) = &stem.execution {
                let executed = execution
                    .indices
                    .iter()
                    .map(|d| offset(d, geometry))
                    .collect::<Result<Vec<_>, _>>()?;
                if layer.is_none_or(|i| !executed.contains(&i)) {
                    return Err(Error("layer is materialized but not executable".into()));
                }
            } else if layer.is_none_or(|i| {
                !stem
                    .indices
                    .resolve(geometry)
                    .is_ok_and(|indices| indices.contains(&i))
            }) {
                return Err(Error("execution layer is not materialized".into()));
            }
            check(&stem.support)?;
        } else if layer.is_some() {
            return Err(Error("unstemmed projection cannot select a layer".into()));
        }
        let uses: Vec<_> = p.uses.iter().filter(|u| u.phase == phase).collect();
        if uses.is_empty() {
            return Err(Error("projection has no requested execution phase".into()));
        }
        for use_ in uses {
            if let Some(c) = &use_.contract {
                check(&c.support)?;
            }
        }
        Ok(())
    }
}

impl Declaration {
    pub(crate) fn normalize_contracts(
        &mut self,
        geometry: &Geometry,
        registry: &Registry,
    ) -> Result<(), Error> {
        let namespace = self.namespace.clone();
        let normalize = |v: &mut Variant, target: Target| -> Result<(), Error> {
            v.parameters = registry.resolve(&namespace, v, target, geometry)?;
            Ok(())
        };
        let composition = |c: &mut Composition, target: Target| -> Result<(), Error> {
            for implementation in c.implementations.values_mut() {
                normalize(implementation, target)?;
            }
            for stage in c.pre.iter_mut().chain(&mut c.post) {
                normalize(&mut stage.contract, Target::Stage)?;
            }
            if let Some(v) = &mut c.implementation {
                normalize(v, target)?;
            }
            Ok(())
        };
        for uses in self.use_profiles.values_mut() {
            for usage in uses {
                if let Some(v) = usage
                    .contract
                    .as_mut()
                    .and_then(|c| c.implementation.as_mut())
                {
                    let target = registry
                        .providers
                        .iter()
                        .flat_map(|p| p.registrations)
                        .find(|r| r.id == v.id)
                        .map(|r| r.target)
                        .ok_or_else(|| Error(format!("variant `{}` is not registered", v.id)))?;
                    normalize(v, target)?;
                }
            }
        }
        for p in &mut self.projections {
            if let Some(profile) = &mut p.use_profile {
                for override_ in profile.overrides_mut() {
                    if let Some(v) = &mut override_.implementation {
                        normalize(v, Target::Projection)?;
                    }
                }
            }
            for v in &mut p.source_transforms {
                normalize(v, Target::SourceTransform)?;
            }
            for part in &mut p.parts {
                for v in &mut part.transforms {
                    normalize(v, Target::SourceTransform)?;
                }
            }
            if let Some(c) = &mut p.composition {
                composition(c, Target::Projection)?;
            }
            for use_ in &mut p.uses {
                if let Some(c) = &mut use_.contract
                    && let Some(v) = &mut c.implementation
                {
                    normalize(v, Target::Projection)?;
                }
            }
        }
        for group in &mut self.fusion_groups {
            if let Some(profile) = &mut group.use_profile {
                for override_ in profile.overrides_mut() {
                    if let Some(v) = &mut override_.implementation {
                        normalize(v, Target::FusionGroup)?;
                    }
                }
            }
            for use_ in &mut group.uses {
                if let Some(c) = &mut use_.contract
                    && let Some(v) = &mut c.implementation
                {
                    normalize(v, Target::FusionGroup)?;
                }
            }
            if let Some(c) = &mut group.composition {
                composition(c, Target::FusionGroup)?;
            }
        }
        Ok(())
    }
}
