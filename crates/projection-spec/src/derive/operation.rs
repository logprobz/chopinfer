//! Derive operations from validated declaration data and explicit imported recipes.

use kernel_core::contract::{DType, SwigluColumns};
use kernel_core::precision::format::{KernelFamily, LayoutVariant, registry as formats};
use kernel_core::precision::layout::PackingPlan;
use kernel_core::precision::{OperationSpec, OperationUse};
use std::collections::BTreeSet;

use super::selection::{self, Selection};
use super::views;
use super::*;
use crate::contracts::{Composition, SupportRule};
use crate::registry::Target as VariantTarget;
use crate::{Error, ProjectionKind, Variant};

fn u32_dim(n: u64, at: &str) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| Error(format!("{at} exceeds u32")))
}

fn support(
    rules: &[SupportRule],
    t: Topology,
    usage: OperationUse,
    format: FormatId,
    bound: &crate::bind::BoundDeclaration,
) -> Result<(), Error> {
    let rules = crate::support::active(rules, Some(bound))?;
    let matches = |r: &SupportRule| {
        (r.ranks.is_empty() || r.ranks.contains(&t.ranks))
            && (r.formats.is_empty() || r.formats.contains(&format))
            && (r.phases.is_empty() || r.phases.contains(&usage.phase))
    };
    if rules.iter().any(|r| !r.enabled && matches(r))
        || (rules.iter().any(|r| r.enabled) && !rules.iter().any(|r| r.enabled && matches(r)))
    {
        return Err(Error(
            "requested topology/format/use is not supported".into(),
        ));
    }
    Ok(())
}

type LayerInstance = (Option<u64>, InstanceKey);

fn instances(
    declaration: &crate::Declaration,
    bound: &crate::bind::BoundDeclaration,
    geometry: &crate::Geometry,
    p: &Projection,
) -> Result<Vec<LayerInstance>, Error> {
    if let Some(id) = p.stem() {
        let stem = declaration
            .layer_stems
            .iter()
            .find(|s| s.id == id)
            .ok_or_else(|| Error("missing layer stem".into()))?;
        if let Some(execution) = &stem.execution {
            execution
                .indices
                .iter()
                .map(|d| {
                    let i = d.resolve_offset(geometry)?;
                    let alias = execution
                        .aliases
                        .iter()
                        .find(|a| a.execution_index.resolve_offset(geometry).ok() == Some(i));
                    let source_index = alias
                        .map(|a| a.source_index.resolve_offset(geometry))
                        .transpose()?
                        .unwrap_or(i);
                    let source_stem = alias
                        .and_then(|a| a.source_stem.as_deref())
                        .or_else(|| p.source.as_ref().and_then(|s| s.stem.as_deref()))
                        .unwrap_or(id);
                    let source = if source_stem != id {
                        super::execution::source(
                            declaration,
                            bound,
                            geometry,
                            p,
                            source_stem,
                            source_index,
                        )?
                    } else {
                        InstanceKey {
                            projection: p.id.clone(),
                            layer: Some(source_index),
                        }
                    };
                    Ok((Some(i), source))
                })
                .collect()
        } else {
            Ok(stem
                .indices
                .resolve(geometry)?
                .into_iter()
                .map(|i| {
                    (
                        Some(i),
                        InstanceKey {
                            projection: p.id.clone(),
                            layer: Some(i),
                        },
                    )
                })
                .collect())
        }
    } else {
        Ok(vec![(
            None,
            InstanceKey {
                projection: p.id.clone(),
                layer: None,
            },
        )])
    }
}

pub(super) fn import_admitted(
    input: &DerivationInput<'_>,
    p: &Projection,
    source_layer: Option<u64>,
    imported: &ImportedProjection,
) -> Result<(), Error> {
    if imported.weights.len() != p.operand_count() as usize || imported.matrices.is_empty() {
        return Err(Error(
            "import must provide every operand and its source matrices".into(),
        ));
    }
    let plugin = formats::by_id(imported.format).map_err(|e| Error(e.to_string()))?;
    if p.format.is_some_and(|id| id != imported.format)
        || imported.weights.iter().any(|w| !plugin.accepts(w.format))
    {
        return Err(Error(
            "imported format id disagrees with the declared source operands".into(),
        ));
    }
    sources_admitted(
        input.bound,
        p,
        input.geometry,
        source_layer,
        imported,
        input.declaration,
        input.topology,
    )
}

fn sources_admitted(
    bound_declaration: &crate::bind::BoundDeclaration,
    p: &Projection,
    geometry: &crate::Geometry,
    source_layer: Option<u64>,
    imported: &ImportedProjection,
    declaration: &Declaration,
    topology: Topology,
) -> Result<(), Error> {
    let instance = Some((p.id.as_str(), source_layer));
    let bound = super::index::operands(bound_declaration, instance, &BTreeSet::new());
    let assembled = bound_declaration
        .assemblies
        .iter()
        .find(|a| a.projection == p.id && a.layer == source_layer);
    if bound.is_empty() && assembled.is_none() {
        return Err(Error(
            "projection instance has no checkpoint binding".into(),
        ));
    }
    let names = assembled
        .map(|a| {
            a.tensors
                .iter()
                .flatten()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let mut owners = std::collections::BTreeMap::<&str, Vec<&crate::bind::BoundOperand>>::new();
    for binding in super::index::operands(bound_declaration, instance, &names) {
        owners
            .entry(binding.imported.weight.codes.tensor.as_str())
            .or_default()
            .push(binding);
    }
    let direct_names = bound
        .iter()
        .map(|b| b.imported.weight.codes.tensor.as_str())
        .collect::<BTreeSet<_>>();
    for source in &imported.matrices {
        source.weight.validate().map_err(|e| Error(e.to_string()))?;
        let admitted = owners
            .get(source.weight.codes.tensor.as_str())
            .is_some_and(|bindings| {
                bindings.iter().any(|b| {
                    b.imported.weight == source.weight || b.imported.parts.contains(&source.weight)
                })
            });
        if !admitted {
            return Err(Error(format!(
                "source `{}` does not match an authenticated binding",
                source.weight.codes.tensor
            )));
        }
        let scoped = direct_names.contains(source.weight.codes.tensor.as_str())
            || names.contains(source.weight.codes.tensor.as_str());
        if !scoped {
            return Err(Error(format!(
                "source `{}` is outside projection `{}` binding",
                source.weight.codes.tensor, p.id
            )));
        }
        if source.operand >= p.operand_count()
            || source.target_rows.is_empty()
            || source.target_k.is_empty()
            || source.source_rows.is_empty()
            || source.source_k.is_empty()
            || source.target_rows.end - source.target_rows.start
                != source.source_rows.end - source.source_rows.start
            || source.target_k.end - source.target_k.start
                != source.source_k.end - source.source_k.start
        {
            return Err(Error(
                "source matrix coordinate mapping is inconsistent".into(),
            ));
        }
        let rank = source.weight.logical_shape.len();
        if source.source_rows.end > source.weight.logical_shape[rank - 2]
            || source.source_k.end > source.weight.logical_shape[rank - 1]
            || source.target_rows.end > p.shape[0].resolve(geometry)?
            || source.target_k.end > p.shape[1].resolve(geometry)?
        {
            return Err(Error(
                "source matrix selection exceeds logical bounds".into(),
            ));
        }
        let source_experts = if rank == 3 {
            source.weight.logical_shape[0]
        } else {
            1
        };
        if source.source_expert >= source_experts {
            return Err(Error("source expert is outside the imported bank".into()));
        }
    }
    super::authentication::check(
        declaration,
        bound_declaration,
        p,
        geometry,
        source_layer,
        topology,
        imported,
    )
}

fn transform(
    v: &Variant,
    part: Option<&crate::Part>,
    ctx: &ProjectionContext<'_>,
    imported: ImportedProjection,
    input: &DerivationInput<'_>,
) -> Result<ImportedProjection, Error> {
    input.registry.resolve(
        &input.declaration.namespace,
        v,
        VariantTarget::SourceTransform,
        input.geometry,
    )?;
    let expected: Vec<_> = v.inputs.iter().filter_map(|p| p.format).collect();
    if !expected.contains(&imported.format) {
        return Err(Error(
            "source transform input format id disagrees with its ports".into(),
        ));
    }

    if !expected.is_empty()
        && imported.weights.iter().any(|w| {
            !expected
                .iter()
                .any(|id| formats::by_id(*id).is_ok_and(|p| p.accepts(w.format)))
        })
    {
        return Err(Error(format!(
            "variant `{}` input format disagrees with imported operands",
            v.id
        )));
    }
    let original = imported.clone();
    let result = match input.variants.transform(v, part, ctx, imported)? {
        SourceTransformResult::Bypass => return Ok(original),
        SourceTransformResult::Applied(result) => result,
    };
    if result.matrices != original.matrices {
        return Err(Error(
            "source transform changes authenticated source matrices".into(),
        ));
    }
    if (result.format != original.format || result.weights != original.weights)
        && result.preparation.is_none()
    {
        return Err(Error(
            "effective source operands require explicit rank preparation".into(),
        ));
    }
    if let Some(preparation) = &result.preparation {
        check_preparation(ctx, &result, preparation)?;
    }
    let expected: Vec<_> = v.outputs.iter().filter_map(|p| p.format).collect();
    if !expected.contains(&result.format) {
        return Err(Error(
            "source transform output format id disagrees with its ports".into(),
        ));
    }

    if !expected.is_empty()
        && result.weights.iter().any(|w| {
            !expected
                .iter()
                .any(|id| formats::by_id(*id).is_ok_and(|p| p.accepts(w.format)))
        })
    {
        return Err(Error(format!(
            "variant `{}` output format disagrees with effective operands",
            v.id
        )));
    }
    Ok(result)
}

fn record_sources(
    ctx: &ProjectionContext<'_>,
    imported: &ImportedProjection,
) -> Vec<kernel_core::precision::TensorSpan> {
    let mut result = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let names = imported
        .matrices
        .iter()
        .map(|m| m.weight.codes.tensor.as_str())
        .collect::<BTreeSet<_>>();
    for weight in imported.matrices.iter().map(|m| &m.weight).chain(
        super::index::operands(ctx.bound, None, &names)
            .into_iter()
            .map(|b| &b.imported.weight),
    ) {
        for span in std::iter::once(&weight.codes)
            .chain(weight.local_scale.iter().map(|s| &s.tensor))
            .chain(weight.global_scale.iter().map(|s| &s.tensor))
        {
            if seen.insert(span) {
                result.push(span.clone());
            }
        }
    }
    result
}

fn check_preparation(
    ctx: &ProjectionContext<'_>,
    imported: &ImportedProjection,
    preparation: &SourcePreparation,
) -> Result<(), Error> {
    if preparation.topology != ctx.topology
        || preparation.geometry.rows == 0
        || preparation.geometry.k == 0
        || preparation.records.is_empty()
    {
        return Err(Error(
            "source preparation has invalid topology, geometry or records".into(),
        ));
    }
    let sources = record_sources(ctx, imported);
    for record in &preparation.records {
        if record.record.sources.len() != record.tensors.len() {
            return Err(Error(
                "source preparation record source count disagrees".into(),
            ));
        }
        for (source, name) in record.record.sources.iter().zip(&record.tensors) {
            let admitted = sources.iter().any(|span| {
                span.tensor == *name
                    && span.byte_length == source.byte_length
                    && (span.dtype.checkpoint_dtype() == Some(source.dtype.as_str())
                        || (span.dtype == DType::Ue8M0 && source.dtype == "U8"))
                    && span.elements().is_ok_and(|n| {
                        source.shape.iter().try_fold(1u64, |v, d| v.checked_mul(*d)) == Some(n)
                    })
            });
            if !admitted {
                return Err(Error(
                    "source preparation record is outside authenticated matrices".into(),
                ));
            }
        }
    }
    Ok(())
}

fn transform_parts(
    ctx: &ProjectionContext<'_>,
    input: &DerivationInput<'_>,
    imported: ImportedProjection,
) -> Result<ImportedProjection, Error> {
    if !ctx
        .projection
        .parts
        .iter()
        .any(|p| !p.transforms.is_empty())
    {
        return Ok(imported);
    }
    let mut result = imported.clone();
    result.matrices.clear();
    result.preparation = None;
    let mut format = None;
    let mut first = true;
    let mut offset = 0;
    for part in &ctx.projection.parts {
        let start = part
            .range
            .as_ref()
            .map(|r| r.start.resolve_offset(ctx.geometry))
            .transpose()?
            .unwrap_or(offset);
        let end = start
            .checked_add(part.rows.resolve(ctx.geometry)?)
            .ok_or_else(|| Error("transform part extent overflows".into()))?;
        let mut scoped = imported.clone();
        scoped.matrices = imported
            .matrices
            .iter()
            .filter_map(|m| {
                let begin = m.target_rows.start.max(start);
                let end = m.target_rows.end.min(end);
                if begin >= end {
                    return None;
                }
                let mut m = m.clone();
                m.source_rows = m.source_rows.start + begin - m.target_rows.start
                    ..m.source_rows.start + end - m.target_rows.start;
                m.target_rows = begin..end;
                Some(m)
            })
            .collect();
        if scoped.matrices.is_empty() {
            offset = end;
            continue;
        }
        for v in &part.transforms {
            scoped = transform(v, Some(part), ctx, scoped, input)?;
        }
        if scoped
            .matrices
            .iter()
            .any(|m| m.target_rows.start < start || m.target_rows.end > end)
        {
            return Err(Error(
                "part transform changes another part's target rows".into(),
            ));
        }
        if format.is_some_and(|f| f != scoped.format) {
            return Err(Error("parts select incompatible effective formats".into()));
        }
        format = Some(scoped.format);
        if first {
            result.weights = scoped.weights.clone();
            result.arithmetic = scoped.arithmetic.clone();
            result.a16_recipe = scoped.a16_recipe;
            first = false;
        }
        if result.arithmetic != scoped.arithmetic || result.a16_recipe != scoped.a16_recipe {
            return Err(Error("parts select incompatible arithmetic recipes".into()));
        }
        if let Some(preparation) = scoped.preparation {
            if let Some(previous) = &mut result.preparation {
                if previous.topology != preparation.topology
                    || previous.geometry != preparation.geometry
                {
                    return Err(Error(
                        "parts select incompatible preparation geometry".into(),
                    ));
                }
                for record in preparation.records {
                    if !previous.records.contains(&record) {
                        previous.records.push(record);
                    }
                }
            } else {
                result.preparation = Some(preparation);
            }
        }
        result.matrices.extend(scoped.matrices);
        offset = end;
    }
    result.format =
        format.ok_or_else(|| Error("part transforms have no source matrices".into()))?;
    Ok(result)
}

fn stages(
    c: Option<&Composition>,
    ctx: &ProjectionContext<'_>,
    input: &DerivationInput<'_>,
) -> Result<Vec<DerivedStage>, Error> {
    let Some(c) = c else { return Ok(Vec::new()) };
    let mut result = Vec::new();
    for stage in c
        .pre
        .iter()
        .chain(&c.post)
        .filter(|s| s.phases.contains(&ctx.usage.phase))
    {
        input.registry.resolve(
            &input.declaration.namespace,
            &stage.contract,
            VariantTarget::Stage,
            input.geometry,
        )?;
        let derived = input.variants.stage(stage, ctx)?;
        if let Some(adapter) = &derived.adapter {
            adapter.validate(&stage.contract)?;
        }
        for (views, ports) in [
            (&derived.inputs, &stage.contract.inputs),
            (&derived.outputs, &stage.contract.outputs),
        ] {
            if views.len() != ports.len()
                || views.iter().zip(ports).any(|(v, p)| {
                    v.dtype != p.dtype
                        || v.interpretation != p.interpretation
                        || crate::contract_validation::shape(&p.shape, input.geometry)
                            .is_ok_and(|s| s != v.shape)
                })
            {
                return Err(Error(format!(
                    "stage `{}` behavior changes its declared ports",
                    stage.id
                )));
            }
        }
        if derived.id != stage.id {
            return Err(Error("stage behavior changes its declared identity".into()));
        }
        if derived.numerical_contract != stage.contract.numerical_contract
            || derived.mutations != stage.mutates
        {
            return Err(Error(format!(
                "stage `{}` behavior changes its declared numerical/state contract",
                stage.id
            )));
        }
        result.push(derived);
    }
    Ok(result)
}

fn implementation<'a>(p: &'a Projection, usage: &'a Use, format: FormatId) -> Option<&'a Variant> {
    usage
        .contract
        .as_ref()
        .and_then(|c| c.implementation.as_ref())
        .or_else(|| {
            p.composition
                .as_ref()
                .and_then(|c| c.implementation_for(format))
        })
}

pub(super) fn checked_ordinary(
    ctx: &ProjectionContext<'_>,
    imported: &ImportedProjection,
) -> Result<DerivedOperation, Error> {
    selection::topology(ctx.topology)?;
    if !ctx.declaration.projections.contains(ctx.projection)
        || !ctx.projection.uses.contains(ctx.usage)
        || ctx.source_layer != ctx.source_instance.layer
        || imported.weights.len() != ctx.projection.operand_count() as usize
        || imported.matrices.is_empty()
    {
        return Err(Error(
            "ordinary context disagrees with its declaration/import".into(),
        ));
    }
    if !instances(ctx.declaration, ctx.bound, ctx.geometry, ctx.projection)?
        .iter()
        .any(|(layer, source)| *layer == ctx.layer && *source == ctx.source_instance)
    {
        return Err(Error(
            "ordinary context has an undeclared execution/source identity".into(),
        ));
    }
    let source = ctx
        .declaration
        .projections
        .iter()
        .find(|p| p.id == ctx.source_instance.projection)
        .ok_or_else(|| Error("ordinary context has no declared source projection".into()))?;
    let (admitted, layer) = if !ctx.projection.assembly.is_empty() {
        (
            ctx.projection,
            super::execution::assembly_layer(ctx.bound, ctx.projection, ctx.layer)?,
        )
    } else {
        (source, ctx.source_layer)
    };
    sources_admitted(
        ctx.bound,
        admitted,
        ctx.geometry,
        layer,
        imported,
        ctx.declaration,
        ctx.topology,
    )?;
    ordinary(ctx, imported, ctx.bound, ctx.selections)
}

fn numerical_recipe<'a>(
    projection: &'a Projection,
    bound: &crate::bind::BoundDeclaration,
    source: &InstanceKey,
) -> Result<Option<&'a crate::contracts::NumericalRecipe>, Error> {
    projection
        .composition
        .as_ref()
        .map(|c| c.numerical_for_sources(super::execution::formats(bound, source)))
        .transpose()
        .map(Option::flatten)
}

pub(super) fn ordinary(
    ctx: &ProjectionContext<'_>,
    imported: &ImportedProjection,
    bound: &crate::bind::BoundDeclaration,
    selections: Option<&[Selection]>,
) -> Result<DerivedOperation, Error> {
    let p = ctx.projection;
    let g = ctx.geometry;
    let first = imported
        .weights
        .first()
        .ok_or_else(|| Error("import contains no weights".into()))?;
    for weight in &imported.weights {
        weight.validate().map_err(|e| Error(e.to_string()))?;
    }
    let plugin = formats::by_id(imported.format).map_err(|e| Error(e.to_string()))?;
    if imported.weights.iter().any(|w| !plugin.accepts(w.format)) {
        return Err(Error(
            "effective format id disagrees with transformed weights".into(),
        ));
    }
    let family = imported.family;
    let legal_family = match p.kind {
        ProjectionKind::Dense {} => matches!(family, KernelFamily::Dense | KernelFamily::Row),
        ProjectionKind::GroupedDense { .. } => family == KernelFamily::Dense,
        ProjectionKind::ExpertBank { .. } => family == KernelFamily::GroupedExpert,
        ProjectionKind::LookupTable { .. } => family == KernelFamily::Lookup,
    };
    if !legal_family {
        return Err(Error(
            "imported kernel family disagrees with projection kind".into(),
        ));
    }
    let usage = OperationUse {
        phase: ctx.usage.phase,
        capacity: u32_dim(ctx.usage.capacity.resolve(g)?, "requested capacity")?,
        graph: ctx.usage.graph,
    };
    if let Some(id) = p.stem() {
        support(
            &ctx.declaration
                .layer_stems
                .iter()
                .find(|s| s.id == id)
                .unwrap()
                .support,
            ctx.topology,
            usage,
            plugin.id(),
            bound,
        )?;
    }
    let mut arithmetic = imported.arithmetic.clone();
    let numerical = numerical_recipe(p, bound, &ctx.source_instance)?;
    if arithmetic.numerical_contract
        != numerical.map_or(p.numerical_contract.as_str(), |n| {
            n.numerical_contract.as_str()
        })
    {
        return Err(Error(
            "import arithmetic does not match declaration numerical contract".into(),
        ));
    }
    let (compiled, row_multiplier, chunk_rows, item, aggregate, exact) =
        if let Some(c) = &ctx.usage.contract {
            support(&c.support, ctx.topology, usage, plugin.id(), bound)?;
            arithmetic.output = c.output_dtype;
            arithmetic.numerical_contract = numerical.map_or_else(
                || c.numerical_contract.clone(),
                |n| n.numerical_contract.clone(),
            );
            if let Some(format) = c.activation
                && imported.activations.iter().any(|a| a.format != format)
            {
                return Err(Error(
                    "declared activation disagrees with the imported recipe".into(),
                ));
            }
            (
                u32_dim(c.compiled_capacity.resolve(g)?, "compiled capacity")?,
                c.row_multiplier.resolve(g)?,
                c.chunk_rows.as_ref().map(|d| d.resolve(g)).transpose()?,
                c.per_item_verify
                    .as_ref()
                    .map(|d| d.resolve(g))
                    .transpose()?,
                c.aggregate_verify
                    .as_ref()
                    .map(|d| d.resolve(g))
                    .transpose()?,
                c.exact_rows,
            )
        } else {
            (usage.capacity, 1, None, None, None, false)
        };
    let expected_activations = if family == KernelFamily::Lookup {
        0
    } else {
        imported.weights.len()
    };
    if imported.activations.len() != expected_activations {
        return Err(Error(
            "import activation count disagrees with projection operands".into(),
        ));
    }
    let activation = imported
        .activations
        .first()
        .map_or(kernel_core::contract::ActivationFormat::Bf16, |a| a.format);
    let decode = plugin
        .decode_order(family, activation, imported.a16_recipe)
        .map_err(|e| Error(e.to_string()))?;
    if arithmetic.decode_order != decode {
        return Err(Error(
            "imported arithmetic disagrees with the format plug-in's scale order".into(),
        ));
    }
    let mut layout = plugin
        .layout_for_weight(first, family, LayoutVariant::Standard)
        .map_err(|e| Error(e.to_string()))?;
    // A grouped scalar-global layout packs one global for every expert and
    // operand. That holds only when every operand reads the same scalar
    // span; otherwise each expert and operand keeps its own global.
    fn global(
        w: &kernel_core::precision::WeightSpec,
    ) -> Option<&kernel_core::precision::TensorSpan> {
        w.global_scale.as_ref().map(|s| &s.tensor)
    }
    if family == KernelFamily::GroupedExpert
        && first
            .global_scale
            .as_ref()
            .is_some_and(|s| s.scope != kernel_core::precision::ScaleScope::Expert)
        && imported.weights.iter().any(|w| global(w) != global(first))
    {
        let separate = plugin
            .layout_variant(
                family,
                Some(kernel_core::precision::ScaleScope::Expert),
                LayoutVariant::Standard,
            )
            .map_err(|e| Error(e.to_string()))?;
        layout.weight = separate.weight;
        layout.global_scales = separate.global_scales;
    }
    let mut resolved = Vec::new();
    let mut all_slices = Vec::new();
    for operand in 0..p.operand_count() {
        let mut selected = match selections {
            Some(selections) => selections
                .get(operand as usize)
                .ok_or_else(|| Error("missing assembly operand selection".into()))?
                .clone(),
            None => selection::select(p, g, ctx.topology, operand)?,
        };
        let padding = plugin.padding_tile(family);
        selected.physical.rows = selection::pad(selected.physical.rows, padding.rows as u64)?;
        selected.physical.k = selection::pad(selected.physical.k, padding.k as u64)?;
        if let Some(preparation) = &imported.preparation {
            check_preparation(ctx, imported, preparation)?;
            if preparation.geometry.rows < selected.physical.rows
                || preparation.geometry.k < selected.physical.k
            {
                return Err(Error(
                    "prepared geometry removes declared live elements or padding".into(),
                ));
            }
            selected.physical = preparation.geometry;
        }
        // Fragment K alignment comes from the shared layout, not source-format guesses.
        let probe = PackingPlan {
            layout: layout.weight.clone(),
            format: first.format,
            experts: 1,
            projections: 1,
            rows: 1,
            k: 1,
            columns: SwigluColumns::ConcatenatedHalves,
            local_scale_dtype: None,
        };
        if let Some(format) = probe.fragment() {
            let tile = kernel_core::transform::fragment::FragmentWeightLayout {
                rows: 1,
                k: 1,
                format,
            }
            .tile_k() as u64;
            selected.physical.k = selection::pad(selected.physical.k, tile)?;
        }
        if imported
            .preparation
            .as_ref()
            .is_some_and(|p| p.geometry != selected.physical)
        {
            return Err(Error(
                "prepared geometry disagrees with format alignment".into(),
            ));
        }
        let mut slices = selection::slices(&imported.matrices, &selected, plugin.id(), operand)?;
        if imported.preparation.is_some() {
            for slice in &mut slices {
                slice.format = formats::by_weight(&slice.source.weight)
                    .map_err(|e| Error(e.to_string()))?
                    .id();
            }
        }
        validate_coverage(&slices, &selected)?;
        all_slices.extend(slices);
        resolved.push(selected);
    }
    let selection = &resolved[0];
    if resolved.iter().any(|s| {
        s.logical != selection.logical
            || s.physical != selection.physical
            || s.experts != selection.experts
    }) {
        return Err(Error(
            "fused operands have incompatible rank geometry".into(),
        ));
    }
    let experts = selection.experts.end - selection.experts.start;
    let kind = p.operation_kind(g, selection)?;
    let output_name = match arithmetic.output {
        DType::Bf16 => "bf16",
        DType::F32 => "f32",
        _ => return Err(Error("unsupported projection output dtype".into())),
    };
    let id = ctx.operation_id(&imported.operation_id, arithmetic.output)?;
    let operation = OperationSpec {
        id: id.clone(),
        kind,
        target: imported.target,
        weights: imported.weights.clone(),
        geometry: match imported.geometry {
            GeometryMode::RankLocal => Some(selection.physical),
            GeometryMode::SourceShape => None,
        },
        activations: imported.activations.clone(),
        arithmetic,
        epilogues: p.epilogues,
        uses: vec![usage],
    };
    operation.validate().map_err(|e| Error(e.to_string()))?;
    let packing = PackingPlan {
        layout: layout.weight.clone(),
        format: first.format,
        experts,
        projections: p.operand_count(),
        rows: u32_dim(selection.physical.rows, "packed rows")?,
        k: u32_dim(selection.physical.k, "packed K")?,
        columns: p
            .epilogues
            .swiglu
            .unwrap_or(SwigluColumns::ConcatenatedHalves),
        local_scale_dtype: plugin
            .companions(first)
            .map_err(|e| Error(e.to_string()))?
            .iter()
            .find(|b| b.role == kernel_core::precision::format::CompanionRole::LocalScale)
            .map(|b| b.launch_dtype),
    };
    packing.validate().map_err(|e| Error(e.to_string()))?;
    let mut required = views::weight_views(&id, &packing, &layout)?;
    if let Some(preparation) = &imported.preparation {
        for view in &mut required {
            let ViewRole::Weight(section) = view.role else {
                continue;
            };
            let names: BTreeSet<_> = preparation
                .records
                .iter()
                .filter(|r| r.section == section)
                .map(|r| r.view.as_str())
                .collect();
            if names.len() != 1 {
                return Err(Error(
                    "prepared section needs one explicit resident view name".into(),
                ));
            }
            view.id = (*names.first().expect("one prepared view")).into();
        }
    }
    let rows = u64::from(usage.capacity)
        .checked_mul(row_multiplier)
        .ok_or_else(|| Error("requested row multiplier overflows".into()))?;
    // A grouped dense row holds every local group's K-wide input and output
    // side by side, in group order.
    let groups = match p.kind {
        ProjectionKind::GroupedDense { .. } => u64::from(experts),
        _ => 1,
    };
    let input_width = selection
        .physical
        .k
        .checked_mul(groups)
        .ok_or_else(|| Error("input width overflows".into()))?;
    for (index, activation) in operation.activations.iter().enumerate() {
        required.push(views::view(
            format!("{id}.input.{index}.{:?}.{}", usage.phase, usage.capacity),
            ViewRole::Input,
            activation.storage,
            vec![rows, input_width],
            "row-major".into(),
        )?);
    }
    let output_width = selection
        .physical
        .rows
        .checked_mul(p.operand_count().into())
        .and_then(|width| width.checked_mul(groups))
        .ok_or_else(|| Error("output width overflows".into()))?;
    let output_width = if family == KernelFamily::Lookup {
        selection.physical.k
    } else if p.epilogues.swiglu.is_some() {
        output_width / 2
    } else {
        output_width
    };
    required.push(views::view(
        format!(
            "{id}.output.{:?}.{}.{}",
            usage.phase, usage.capacity, output_name
        ),
        ViewRole::Output,
        operation.arithmetic.output,
        vec![rows, output_width],
        "row-major".into(),
    )?);
    Ok(DerivedOperation {
        instance: InstanceKey {
            projection: p.id.clone(),
            layer: ctx.layer,
        },
        source_layer: ctx.source_layer,
        source_instance: ctx.source_instance.clone(),
        requested: usage,
        compiled_capacity: compiled,
        row_multiplier,
        chunk_rows,
        per_item_verify: item,
        aggregate_verify: aggregate,
        exact_rows: exact,
        operation,
        logical_geometry: selection.logical,
        family,
        a16_recipe: imported.a16_recipe,
        slices: all_slices,
        packing,
        views: required,
        stages: Vec::new(),
        inputs: Vec::new(),
        variant_records: imported
            .preparation
            .as_ref()
            .map_or_else(Vec::new, |p| p.records.clone()),
        source_preparation: imported.preparation.clone(),
        record_sources: record_sources(ctx, imported),
    })
}

fn validate_coverage(slices: &[OperandSlice], selection: &Selection) -> Result<(), Error> {
    for expert in selection.experts.clone() {
        let local = expert - selection.experts.start;
        let mut area = 0u64;
        let mut rectangles: Vec<(std::ops::Range<u64>, std::ops::Range<u64>)> = Vec::new();
        for slice in slices.iter().filter(|s| s.expert == local) {
            let target_k = slice.source.target_k.start + slice.k.start - slice.source.source_k.start
                ..slice.source.target_k.start + slice.k.end - slice.source.source_k.start;
            for range in &slice.rows {
                let target = slice.source.target_rows.start + range.start
                    - slice.source.source_rows.start
                    ..slice.source.target_rows.start + range.end - slice.source.source_rows.start;
                if rectangles.iter().any(|(rows, k)| {
                    rows.start < target.end
                        && target.start < rows.end
                        && k.start < target_k.end
                        && target_k.start < k.end
                }) {
                    return Err(Error(
                        "source matrices overlap in a rank's logical operand".into(),
                    ));
                }
                rectangles.push((target, target_k.clone()));
            }
            let rows = slice.rows.iter().try_fold(0u64, |n, r| {
                n.checked_add(r.end - r.start)
                    .ok_or_else(|| Error("source selection row count overflows".into()))
            })?;
            area = area
                .checked_add(
                    rows.checked_mul(slice.k.end - slice.k.start)
                        .ok_or_else(|| Error("source selection area overflows".into()))?,
                )
                .ok_or_else(|| Error("source coverage overflows".into()))?;
        }
        if area
            != selection
                .logical
                .rows
                .checked_mul(selection.logical.k)
                .ok_or_else(|| Error("logical matrix area overflows".into()))?
        {
            return Err(Error(
                "source matrices do not cover each rank's logical operand".into(),
            ));
        }
    }
    Ok(())
}

/// Derive every supported execution instance and use in declaration order.
/// Missing source/variant behavior is an error; no shape-driven fallback occurs.
#[derive(Clone)]
pub(super) struct SourceNode {
    pub imported: ImportedProjection,
    pub derived: DerivedOperation,
}
type NodeKey = (InstanceKey, usize);

fn derive_node(
    input: &DerivationInput<'_>,
    p: &Projection,
    layer: Option<u64>,
    source_instance: InstanceKey,
    usage: &crate::Use,
    cache: &mut std::collections::BTreeMap<NodeKey, SourceNode>,
    active: &mut BTreeSet<InstanceKey>,
) -> Result<SourceNode, Error> {
    let identity = InstanceKey {
        projection: p.id.clone(),
        layer,
    };
    let use_index = p
        .uses
        .iter()
        .position(|u| u == usage)
        .ok_or_else(|| Error("undeclared assembly source use".into()))?;
    let key = (identity.clone(), use_index);
    if let Some(node) = cache.get(&key) {
        return Ok(node.clone());
    }
    if !active.insert(identity.clone()) {
        return Err(Error("assembly source dependency cycle".into()));
    }
    let source_layer = source_instance.layer;
    let source_projection = input
        .declaration
        .projections
        .iter()
        .find(|s| s.id == source_instance.projection)
        .ok_or_else(|| Error("execution alias source projection is missing".into()))?;
    let prototype = super::execution::prototype(input, p, &source_instance)?;
    let (composition, binding_layer) = if !p.assembly.is_empty() {
        (p, super::execution::assembly_layer(input.bound, p, layer)?)
    } else {
        (source_projection, source_layer)
    };
    let mut sources = std::collections::BTreeMap::new();
    for member in &composition.assembly {
        let role = input
            .declaration
            .projections
            .iter()
            .find(|p| p.id == member.projection)
            .ok_or_else(|| Error("assembly source role is missing".into()))?;
        let member_layer = member
            .layer
            .as_ref()
            .map(|l| l.resolve(input.geometry, binding_layer))
            .transpose()?;
        let behavior = !role.source_transforms.is_empty()
            || role.parts.iter().any(|p| !p.transforms.is_empty())
            || !role.assembly.is_empty()
            || role.composition.as_ref().is_some_and(|c| {
                c.implementation.is_some()
                    || !c.implementations.is_empty()
                    || !c.pre.is_empty()
                    || !c.post.is_empty()
            });
        // Plain checkpoint weights do not require a source execution offer.
        // Their assembly consumer owns its own phases and buckets.
        if !behavior {
            continue;
        }
        let source_use = role
            .uses
            .iter()
            .find(|u| {
                u.phase == usage.phase
                    && u.graph == usage.graph
                    && u.capacity.resolve(input.geometry).ok()
                        == usage.capacity.resolve(input.geometry).ok()
            })
            .ok_or_else(|| {
                Error(format!(
                    "assembly source `{}` has no matching declared use for `{}` {:?}",
                    role.id, p.id, usage.phase
                ))
            })?;
        let source_key = instances(input.declaration, input.bound, input.geometry, role)?
            .into_iter()
            .find(|(l, _)| *l == member_layer)
            .map(|(_, key)| key)
            .unwrap_or(InstanceKey {
                projection: role.id.clone(),
                layer: member_layer,
            });
        let node = derive_node(
            input,
            role,
            member_layer,
            source_key,
            source_use,
            cache,
            active,
        )?;
        sources.insert(
            InstanceKey {
                projection: role.id.clone(),
                layer: member_layer,
            },
            node,
        );
    }
    let assembled = super::assembly::prepare_with_sources(
        input,
        composition,
        binding_layer,
        &prototype,
        &sources,
    )?;
    let imported = assembled.as_ref().map_or(&prototype, |a| &a.imported);
    import_admitted(input, composition, binding_layer, imported)?;
    let numerical = numerical_recipe(p, input.bound, &source_instance)?;
    let mut selected = imported.clone();
    if let Some(recipe) = numerical {
        selected.a16_recipe = recipe.a16_recipe;
        selected.arithmetic.numerical_contract = recipe.numerical_contract.clone();
        let activation = selected
            .activations
            .first()
            .map_or(kernel_core::contract::ActivationFormat::Bf16, |a| a.format);
        selected.arithmetic.decode_order = formats::by_id(selected.format)
            .map_err(|e| Error(e.to_string()))?
            .decode_order(selected.family, activation, recipe.a16_recipe)
            .map_err(|e| Error(e.to_string()))?;
    }
    let ctx = ProjectionContext {
        bound: input.bound,
        source_instance: source_instance.clone(),
        selections: assembled.as_ref().map(|a| a.selections.as_slice()),
        declaration: input.declaration,
        projection: p,
        geometry: input.geometry,
        topology: input.topology,
        layer,
        source_layer,
        usage,
    };
    let mut effective = selected.clone();
    for v in &p.source_transforms {
        effective = transform(v, None, &ctx, effective, input)?;
    }
    effective = transform_parts(&ctx, input, effective)?;
    let requested = OperationUse {
        phase: usage.phase,
        capacity: u32_dim(
            usage.capacity.resolve(input.geometry)?,
            "requested capacity",
        )?,
        graph: usage.graph,
    };
    if let Some(stem) = p
        .stem()
        .and_then(|id| input.declaration.layer_stems.iter().find(|s| s.id == id))
    {
        support(
            &stem.support,
            input.topology,
            requested,
            effective.format,
            input.bound,
        )?;
    }
    if let Some(contract) = &usage.contract {
        support(
            &contract.support,
            input.topology,
            requested,
            effective.format,
            input.bound,
        )?;
    }
    let formats = super::execution::formats(input.bound, &source_instance);
    let source_format = formats.first().copied().unwrap_or(imported.format);
    if formats
        .windows(2)
        .any(|f| implementation(p, usage, f[0]) != implementation(p, usage, f[1]))
    {
        return Err(Error(
            "source formats select incompatible projection implementations".into(),
        ));
    }
    let mut derived = if let Some(v) = implementation(p, usage, source_format) {
        input.registry.resolve(
            &input.declaration.namespace,
            v,
            VariantTarget::Projection,
            input.geometry,
        )?;
        if numerical.is_some_and(|n| n.numerical_contract != v.numerical_contract) {
            return Err(Error(
                "format numerical mapping disagrees with the selected implementation contract"
                    .into(),
            ));
        }
        let derived = input.variants.implementation(v, &ctx, &effective)?;
        if derived.operation.arithmetic.numerical_contract != v.numerical_contract {
            return Err(Error(format!(
                "variant `{}` changes its registered numerical contract",
                v.id
            )));
        }
        if derived.instance
            != (InstanceKey {
                projection: p.id.clone(),
                layer,
            })
            || derived.source_layer != source_layer
            || derived.source_instance != source_instance
            || derived.requested != requested
        {
            return Err(Error(
                "implementation changes the requested instance/use identity".into(),
            ));
        }
        let c = usage.contract.as_ref();
        let compiled = c
            .map(|c| c.compiled_capacity.resolve(input.geometry))
            .transpose()?
            .unwrap_or(requested.capacity.into());
        let multiplier = c
            .map(|c| c.row_multiplier.resolve(input.geometry))
            .transpose()?
            .unwrap_or(1);
        let resolve = |d: Option<&crate::Dim>| d.map(|d| d.resolve(input.geometry)).transpose();
        if u64::from(derived.compiled_capacity) != compiled
            || derived.row_multiplier != multiplier
            || derived.chunk_rows != resolve(c.and_then(|c| c.chunk_rows.as_ref()))?
            || derived.per_item_verify != resolve(c.and_then(|c| c.per_item_verify.as_ref()))?
            || derived.aggregate_verify != resolve(c.and_then(|c| c.aggregate_verify.as_ref()))?
            || derived.exact_rows != c.is_some_and(|c| c.exact_rows)
            || derived.operation.arithmetic.output
                != c.map_or(effective.arithmetic.output, |c| c.output_dtype)
        {
            return Err(Error(
                "implementation changes its declared capacity/output contract".into(),
            ));
        }
        derived
            .operation
            .validate()
            .map_err(|e| Error(e.to_string()))?;
        derived
    } else {
        ordinary(
            &ctx,
            &effective,
            input.bound,
            assembled.as_ref().map(|a| a.selections.as_slice()),
        )?
    };
    // Preparation survives adapters that replace their own final packing records.
    derived.source_preparation = effective.preparation.clone();
    derived.record_sources = record_sources(&ctx, &effective);
    if derived.variant_records.is_empty() {
        derived.variant_records = assembled
            .as_ref()
            .map_or_else(Vec::new, |a| a.records.clone());
    }
    if derived.variant_records.is_empty() {
        derived.variant_records = effective
            .preparation
            .as_ref()
            .map_or_else(Vec::new, |p| p.records.clone());
    }
    if let Some(assembled) = &assembled {
        for record in &mut derived.variant_records {
            record.view = derived
                .views
                .iter()
                .find(|v| v.role == ViewRole::Weight(record.section))
                .ok_or_else(|| Error("assembled record has no destination view".into()))?
                .id
                .clone();
        }
        let mut inherited = assembled.stages.clone();
        inherited.append(&mut derived.stages);
        derived.stages = inherited;
        if let Some(preparation) = &mut derived.source_preparation {
            preparation.records = derived.variant_records.clone();
        }
    }
    super::inputs::lower(&ctx, &mut derived)?;
    derived
        .stages
        .extend(stages(p.composition.as_ref(), &ctx, input)?);

    let node = SourceNode {
        imported: effective,
        derived,
    };
    active.remove(&identity);
    cache.insert(key, node.clone());
    Ok(node)
}

pub fn derive(input: DerivationInput<'_>) -> Result<RankDerivation, Error> {
    let expanded = input.declaration.expanded_profiles(input.geometry)?;
    let input = DerivationInput {
        declaration: &expanded,
        ..input
    };
    let _index = super::index::scope(input.bound);
    selection::topology(input.topology)?;
    input.declaration.validate(input.geometry, input.registry)?;
    let mut result = RankDerivation::default();
    let mut cache = std::collections::BTreeMap::new();
    let mut active = BTreeSet::new();
    for p in &input.declaration.projections {
        for (layer, source_instance) in
            instances(input.declaration, input.bound, input.geometry, p)?
        {
            for usage in &p.uses {
                result.operations.push(
                    derive_node(
                        &input,
                        p,
                        layer,
                        source_instance.clone(),
                        usage,
                        &mut cache,
                        &mut active,
                    )?
                    .derived,
                );
            }
        }
    }
    derive_fusions(&input, &mut result)?;
    derive_sides(&input, &mut result)?;
    Ok(result)
}

fn derive_fusions(input: &DerivationInput<'_>, result: &mut RankDerivation) -> Result<(), Error> {
    for group in &input.declaration.fusion_groups {
        let first = &group.members[0];
        let keys: Vec<_> = result
            .operations
            .iter()
            .filter(|o| o.instance.projection == *first)
            .map(|o| (o.instance.layer, o.requested))
            .collect();
        for (layer, requested) in keys {
            let mut members: Vec<_> = group
                .members
                .iter()
                .map(|id| {
                    result
                        .operations
                        .iter()
                        .find(|o| {
                            o.instance.projection == *id
                                && o.instance.layer == layer
                                && o.requested == requested
                        })
                        .cloned()
                        .ok_or_else(|| Error("fusion member lacks a matching requested use".into()))
                })
                .collect::<Result<_, _>>()?;
            if !input
                .bound
                .fusion_groups
                .iter()
                .any(|f| f.id == group.id && f.members == group.members)
            {
                return Err(Error(
                    "declared fusion has no bound eligibility decision".into(),
                ));
            }
            let group_usage = group.uses.iter().find(|u| {
                u.phase == requested.phase
                    && u.capacity.resolve(input.geometry).ok() == Some(requested.capacity.into())
                    && u.graph == requested.graph
            });
            let source_formats = members
                .iter()
                .flat_map(|m| super::execution::formats(input.bound, &m.source_instance))
                .collect::<std::collections::BTreeSet<_>>();
            let activations = members
                .iter()
                .map(|m| {
                    input
                        .declaration
                        .projections
                        .iter()
                        .find(|p| p.id == m.instance.projection)
                        .unwrap()
                        .uses
                        .iter()
                        .find(|u| {
                            u.phase == requested.phase
                                && u.capacity.resolve(input.geometry).ok()
                                    == Some(requested.capacity.into())
                                && u.graph == requested.graph
                        })
                        .and_then(|u| u.contract.as_ref())
                        .and_then(|c| c.activation)
                })
                .collect::<Vec<_>>();
            let fused = group
                .eligibility
                .admits(&source_formats, requested.phase, &activations);
            if !fused && group.fallback == crate::contracts::Fallback::Refuse {
                return Err(Error(format!(
                    "fusion `{}` eligibility failed and fallback refuses",
                    group.id
                )));
            }
            let numerical = if fused {
                group
                    .composition
                    .as_ref()
                    .map(|c| c.numerical_for_sources(source_formats.iter().copied()))
                    .transpose()?
                    .flatten()
            } else {
                None
            };
            let source_format = source_formats.iter().next().copied();
            if fused
                && group.composition.as_ref().is_some_and(|c| {
                    source_formats.iter().any(|f| {
                        c.implementation_for(*f)
                            != source_format.and_then(|f| c.implementation_for(f))
                    })
                })
            {
                return Err(Error(
                    "fusion source formats select incompatible implementations".into(),
                ));
            }
            let variant = group_usage
                .and_then(|u| u.contract.as_ref())
                .and_then(|c| c.implementation.as_ref())
                .or_else(|| {
                    group.composition.as_ref().and_then(|c| {
                        source_format
                            .and_then(|f| c.implementation_for(f))
                            .or(c.implementation.as_ref())
                    })
                });
            let mut fusion = DerivedFusion {
                id: group.id.clone(),
                layer,
                requested,
                members: members.iter().map(|o| o.instance.clone()).collect(),
                fused,
                operations: Vec::new(),
                views: Vec::new(),
                stages: Vec::new(),
            };
            if fused && let Some(recipe) = numerical {
                for member in &mut members {
                    let format =
                        formats::by_weight(member.operation.weights.first().ok_or_else(|| {
                            Error("fusion member has no effective weight".into())
                        })?)
                        .map_err(|e| Error(e.to_string()))?
                        .id();
                    let activation = member
                        .operation
                        .activations
                        .first()
                        .map_or(kernel_core::contract::ActivationFormat::Bf16, |a| a.format);
                    member.a16_recipe = recipe.a16_recipe;
                    member.operation.arithmetic.numerical_contract =
                        recipe.numerical_contract.clone();
                    member.operation.arithmetic.decode_order = formats::by_id(format)
                        .map_err(|e| Error(e.to_string()))?
                        .decode_order(member.family, activation, recipe.a16_recipe)
                        .map_err(|e| Error(e.to_string()))?;
                    member
                        .operation
                        .validate()
                        .map_err(|e| Error(e.to_string()))?;
                }
                if variant.is_none() {
                    fusion.operations = members.clone();
                }
            }
            let p = input
                .declaration
                .projections
                .iter()
                .find(|p| p.id == *first)
                .unwrap();
            let usage = group_usage.unwrap_or_else(|| {
                p.uses
                    .iter()
                    .find(|u| {
                        u.phase == requested.phase
                            && u.capacity.resolve(input.geometry).ok()
                                == Some(u64::from(requested.capacity))
                            && u.graph == requested.graph
                    })
                    .unwrap()
            });
            let ctx = ProjectionContext {
                bound: input.bound,
                source_instance: members[0].source_instance.clone(),
                selections: None,
                declaration: input.declaration,
                projection: p,
                geometry: input.geometry,
                topology: input.topology,
                layer,
                source_layer: members[0].source_layer,
                usage,
            };
            if fused && let Some(v) = variant {
                input.registry.resolve(
                    &input.declaration.namespace,
                    v,
                    VariantTarget::FusionGroup,
                    input.geometry,
                )?;
                if numerical.is_some_and(|n| n.numerical_contract != v.numerical_contract) {
                    return Err(Error("format numerical mapping disagrees with the fusion implementation contract".into()));
                }
                let context = FusionContext {
                    projection: ctx.clone(),
                    group,
                    numerical,
                };
                fusion = input.variants.fusion(v, &context, &members)?;
                if fusion.id != group.id
                    || fusion.layer != layer
                    || fusion.requested != requested
                    || !fusion.fused
                    || fusion.members
                        != members
                            .iter()
                            .map(|m| m.instance.clone())
                            .collect::<Vec<_>>()
                {
                    return Err(Error(
                        "fusion provider changes its declared identity/membership/use".into(),
                    ));
                }
                for op in &fusion.operations {
                    if op.operation.arithmetic.numerical_contract != v.numerical_contract {
                        return Err(Error(
                            "fusion behavior changes its registered numerical contract".into(),
                        ));
                    }
                    if let Some(recipe) = numerical {
                        let format =
                            formats::by_weight(op.operation.weights.first().ok_or_else(|| {
                                Error("fusion operation has no effective weight".into())
                            })?)
                            .map_err(|e| Error(e.to_string()))?
                            .id();
                        let activation = op
                            .operation
                            .activations
                            .first()
                            .map_or(kernel_core::contract::ActivationFormat::Bf16, |a| a.format);
                        let expected = formats::by_id(format)
                            .map_err(|e| Error(e.to_string()))?
                            .decode_order(op.family, activation, recipe.a16_recipe)
                            .map_err(|e| Error(e.to_string()))?;
                        if op.a16_recipe != recipe.a16_recipe
                            || op.operation.arithmetic.decode_order != expected
                        {
                            return Err(Error(
                                "fusion provider changes its selected A16 recipe".into(),
                            ));
                        }
                    }
                    op.operation.validate().map_err(|e| Error(e.to_string()))?;
                }
            }
            fusion
                .stages
                .extend(stages(group.composition.as_ref(), &ctx, input)?);
            result.fusions.push(fusion);
        }
    }
    Ok(())
}

fn derive_sides(input: &DerivationInput<'_>, result: &mut RankDerivation) -> Result<(), Error> {
    let mut seen = BTreeSet::new();
    for bound in &input.bound.side_views {
        let side = input
            .declaration
            .side_tensors
            .iter()
            .find(|s| s.id == bound.side)
            .ok_or_else(|| Error("bound side view has no declared role".into()))?;
        let shape = side.selection(input.geometry, input.topology)?.shape;
        let id = format!("{}.{}", bound.tensor, bound.view.consumer);
        if seen.insert(id.clone()) {
            result.side_views.push(views::view(
                id,
                ViewRole::Side,
                bound.view.dtype,
                shape,
                "row-major".into(),
            )?);
        }
    }
    Ok(())
}

impl DerivedOperation {
    /// The local group count of a grouped dense operation
    /// (`ProjectionKind::GroupedDense`) with more than one group on this
    /// rank; `None` for every other operation. Its packed weights are
    /// `[groups, rows, K]`, and group `g` reads input columns
    /// `g * K..(g + 1) * K` and writes output columns
    /// `g * rows..(g + 1) * rows` of each row. A rank with one group runs it
    /// as the dense matrix it is.
    pub fn dense_groups(&self) -> Option<u32> {
        (self.family == KernelFamily::Dense && self.packing.experts > 1)
            .then_some(self.packing.experts)
    }

    /// The grouped fold GEMM query (`kernel_core::mxfp8_gemm`) of this
    /// operation for `execution`: the rows of its input view, its groups, the
    /// packed rows and K of one group, and the K-wide group slices of each
    /// input row. A dense operation is the one-group query.
    pub fn fold_gemm_query(
        &self,
        execution: kernel_core::precision::execution::BlockScaledExecution,
    ) -> Result<kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery, Error> {
        use kernel_core::contract::Validate;
        if self.family != KernelFamily::Dense {
            return Err(Error(format!(
                "operation `{}` is not a dense or grouped dense operation",
                self.operation.id
            )));
        }
        let groups = self.dense_groups().unwrap_or(1);
        let row_stride = self
            .packing
            .k
            .checked_mul(groups)
            .ok_or_else(|| Error("grouped activation row overflows".into()))?;
        let query = kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery {
            m_capacity: u32::try_from(
                u64::from(self.requested.capacity)
                    .checked_mul(self.row_multiplier)
                    .ok_or_else(|| Error("fold GEMM rows overflow".into()))?,
            )
            .map_err(|_| Error("fold GEMM rows exceed u32".into()))?,
            n: self.packing.rows,
            k: self.packing.k,
            groups,
            activation_row_stride: row_stride,
            activation_group_stride: self.packing.k,
            output: self.operation.arithmetic.output,
            execution,
        };
        query
            .validate()
            .map_err(|e| Error(format!("operation `{}`: {}", self.operation.id, e.0)))?;
        Ok(query)
    }
}

impl RankDerivation {
    /// Operations selected by explicit per-use fusion replacements.
    pub fn selected_operations(&self) -> impl Iterator<Item = &DerivedOperation> {
        self.operations
            .iter()
            .filter(|operation| {
                !self.fusions.iter().any(|fusion| {
                    fusion.fused
                        && !fusion.operations.is_empty()
                        && fusion.requested == operation.requested
                        && fusion.members.contains(&operation.instance)
                })
            })
            .chain(
                self.fusions
                    .iter()
                    .filter(|f| f.fused)
                    .flat_map(|f| &f.operations),
            )
    }

    /// Collect equal operation recipes into their original multi-use coverage.
    /// A grouped dense operation is refused: its spec describes one group, and
    /// it runs on the grouped fold GEMM contract, never a dense kernel.
    pub fn operation_specs(&self) -> Result<Vec<OperationSpec>, Error> {
        let mut result: Vec<OperationSpec> = Vec::new();
        for derived in self.selected_operations() {
            if let Some(groups) = derived.dense_groups() {
                return Err(Error(format!(
                    "operation `{}` is grouped dense ({groups} groups) and runs on the grouped \
                     fold GEMM contract, not a dense kernel",
                    derived.operation.id
                )));
            }
            let mut comparable = derived.operation.clone();
            comparable.uses.clear();
            if let Some(existing) = result.iter_mut().find(|s| {
                let mut candidate = (*s).clone();
                candidate.uses.clear();
                candidate == comparable
            }) {
                for use_ in &derived.operation.uses {
                    if !existing.uses.contains(use_) {
                        existing.uses.push(*use_);
                    }
                }
            } else {
                result.push(derived.operation.clone());
            }
        }
        for spec in &result {
            spec.validate().map_err(|e| Error(e.to_string()))?;
        }
        Ok(result)
    }
}
