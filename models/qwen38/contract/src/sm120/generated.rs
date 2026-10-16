//! Publish Qwen's generated SM120 offers from the declaration, the committed
//! schedules and the committed symbol adoptions.
use crate::contract::*;
use crate::precision::*;
use crate::projections::ports::Ports;
use kernel_core::contract::{ActivationFormat, DType, Target, WeightFormat};
use kernel_core::precision::capability::{CapabilityDocument, OperationCapability};
use kernel_core::precision::format::{KernelFamily, registry};
use kernel_core::precision::{ActivationSpec, ArithmeticSpec, OperationKind, OperationSpec};
use kernel_core::precision::{ExecutionPhase, MatrixGeometry, OperationUse};
use kernels_sm120::precision::generated::identity::shape_free;
use kernels_sm120::precision::generated::{self as shared, Adoption};
use kernels_sm120::{precision::KernelOffer, row::precision as row};
use projection_spec::derive::{Topology, rank_selection};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

pub fn schedules() -> Result<&'static [shared::schedule::Override], String> {
    static RECORDS: OnceLock<Result<Vec<shared::schedule::Override>, String>> = OnceLock::new();
    RECORDS
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../../bundle/projection-schedules.json"))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// The append-only symbol adoptions. A later record supersedes an earlier
/// one for the same symbol.
pub fn ledger() -> Result<&'static [Adoption], String> {
    static RECORDS: OnceLock<Result<Vec<Adoption>, String>> = OnceLock::new();
    RECORDS
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json"))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// A content-named entry's symbol under a non-Flash geometry's kernel tag.
fn tagged(shape: Shape, generated: &str) -> String {
    format!(
        "chop_{}_{}",
        shape.geometry.kernel_tag,
        generated.trim_start_matches("chop_")
    )
}

/// The adoptions a geometry serves. The ledger names Flash's entries; any
/// other geometry builds its own cubins, so it serves the same code under its
/// kernel tag and never shares a symbol with Flash. Launch-shape row stages
/// (`identity::shape_free`) serve every geometry from one shared cubin under
/// their content names.
fn aliases(shape: Shape) -> Result<&'static [Adoption], String> {
    let ledger = ledger()?;
    if shape.geometry.id == crate::geometry::FLASH.id {
        return Ok(ledger);
    }
    type Scoped = BTreeMap<&'static str, &'static [Adoption]>;
    static SCOPED: OnceLock<Mutex<Scoped>> = OnceLock::new();
    let mut scoped = SCOPED
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| "adoption scope cache")?;
    Ok(*scoped.entry(shape.geometry.id).or_insert_with(|| {
        let tagged = ledger.iter().map(|record| Adoption {
            symbol: if record.symbol == record.generated && shape_free(&record.generated) {
                record.symbol.clone()
            } else if record.symbol == record.generated {
                tagged(shape, &record.generated)
            } else {
                record
                    .symbol
                    .replacen("qwen38", shape.geometry.kernel_tag, 1)
            },
            ..record.clone()
        });
        Box::leak(tagged.collect::<Vec<_>>().into_boxed_slice())
    }))
}

/// Adoptions naming a non-Flash geometry's unadopted entries, which keep
/// their content names, under its kernel tag. Launch-shape row stages keep
/// their content names in every geometry.
fn unadopted<'a>(
    shape: Shape,
    entries: impl IntoIterator<Item = (&'a Adoption, &'a str)>,
) -> Vec<Adoption> {
    if shape.geometry.id == crate::geometry::FLASH.id {
        return vec![];
    }
    entries
        .into_iter()
        .filter(|(identity, symbol)| identity.generated == *symbol && !shape_free(symbol))
        .map(|(identity, _)| Adoption {
            symbol: tagged(shape, &identity.generated),
            ..identity.clone()
        })
        .collect()
}

pub fn scheduled_offer(
    shape: Shape,
    operation: &OperationSpec,
    usage: OperationUse,
    aliases: &[Adoption],
) -> Result<shared::GeneratedOffer, String> {
    let mut inputs = shared::schedule::key(operation, usage)?;
    if operation.kind == OperationKind::Lookup {
        inputs.lookup = Some(shared::schedule::LookupAddressing {
            heads: shape.ngram_local_heads,
            ids: row::LookupIds::U64,
            scale_dtype: Some(DType::F32),
        });
    }
    shared::schedule::offer(operation, &inputs, schedules()?, aliases)
}

/// The rank-local operation domain of a published projection: the declared
/// projection after rank selection and format padding, the declared
/// composition's recipe for the format, and BF16 activations.
pub fn domain(
    shape: Shape,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<OperationSpec, String> {
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    domain_in(ports, id, weight, usage)
}

fn domain_in(
    ports: &Ports,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<OperationSpec, String> {
    let shape = ports.shape;
    let p = ports.projection(id).map_err(|e| e.to_string())?;
    let plugin = registry::by_format(weight).map_err(|e| e.to_string())?;
    let family = p.family();
    let (numerical_contract, recipe) = match p
        .composition
        .as_ref()
        .and_then(|c| c.numerical_contracts.get(&plugin.id()))
    {
        Some(recipe) => (recipe.numerical_contract.clone(), recipe.a16_recipe),
        None if family == KernelFamily::Lookup => (
            p.numerical_contract.clone(),
            kernel_core::precision::format::A16Recipe::ScaledWeight,
        ),
        None => return Err(format!("projection {id} composes no {weight:?} recipe")),
    };
    let selection = rank_selection(
        p,
        &ports.geometry,
        Topology {
            ranks: shape.ranks(),
            rank: 0,
        },
        0,
    )
    .map_err(|e| e.to_string())?;
    let kind = p
        .operation_kind(&ports.geometry, &selection)
        .map_err(|e| e.to_string())?;
    let resolve = |dim: &projection_spec::Dim| dim.resolve(&ports.geometry);
    let geometry = match family {
        KernelFamily::GroupedExpert => selection.physical,
        // A lookup addresses its rank's rows inside the whole table.
        KernelFamily::Lookup => MatrixGeometry {
            rows: resolve(&p.shape[0]).map_err(|e| e.to_string())?,
            k: resolve(&p.shape[1]).map_err(|e| e.to_string())?,
        },
        _ => ports
            .weight_geometry(id, weight)
            .map_err(|e| e.to_string())?,
    };
    let operands = u8::try_from(p.operand_count()).map_err(|e| e.to_string())?;
    shared::domain(shared::Domain {
        kind,
        target: Target::Sm120f,
        geometry,
        operands,
        weight,
        local_scale: None,
        global_scale: (weight == WeightFormat::Nvfp4).then_some(
            if family == KernelFamily::GroupedExpert {
                kernel_core::precision::ScaleScope::Expert
            } else {
                kernel_core::precision::ScaleScope::Tensor
            },
        ),
        activations: if family == KernelFamily::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16(); usize::from(operands)]
        },
        arithmetic: ArithmeticSpec {
            weight_operand: DType::Bf16,
            accumulator: DType::F32,
            output: DType::Bf16,
            decode_order: plugin
                .decode_order(family, ActivationFormat::Bf16, recipe)
                .map_err(|e| e.to_string())?,
            rounding: kernel_core::precision::Rounding::NearestEven,
            numerical_contract,
        },
        epilogues: p.epilogues,
        usage,
    })
}

/// The portable operation `capability` executes, over canonical row-major
/// operands with local scales of `local_scale` (the format's canonical dtype
/// when `None`), the capability's global scale scope and BF16 activations.
pub fn capability_domain(
    capability: &OperationCapability,
    local_scale: Option<DType>,
) -> Result<OperationSpec, String> {
    use kernel_core::precision::ScaleScope;
    use kernel_core::precision::capability::GlobalScaleCapability;
    shared::domain(shared::Domain {
        kind: capability.kind,
        target: capability.target,
        geometry: capability.geometry,
        operands: capability.operands,
        weight: capability.weight,
        local_scale,
        global_scale: match capability.globals {
            GlobalScaleCapability::Absent => None,
            GlobalScaleCapability::SharedScalar => Some(ScaleScope::Tensor),
            GlobalScaleCapability::ExpertProjection => Some(ScaleScope::Expert),
        },
        activations: if capability.kind == OperationKind::Lookup {
            vec![]
        } else {
            vec![ActivationSpec::bf16(); usize::from(capability.operands)]
        },
        arithmetic: capability.arithmetic.clone(),
        epilogues: capability.epilogues,
        usage: capability.usage,
    })
}

/// One published projection use: the generated offer, the exact copy into a
/// fused family's output row, the registered stages before the shared
/// compute, its portable capability and its symbol identities.
pub struct Published {
    pub offer: KernelOffer,
    pub copy: Option<row::LayoutCopy>,
    pub prefix: Vec<String>,
    pub capability: OperationCapability,
    pub adoptions: Vec<Adoption>,
}

type Key = (usize, u32, String, String, String);

/// Generate one published use once per process.
pub fn publish(
    shape: Shape,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Arc<Published>, String> {
    static CACHE: OnceLock<Mutex<BTreeMap<Key, Arc<Published>>>> = OnceLock::new();
    let key = (
        std::ptr::from_ref(shape.geometry) as usize,
        shape.ranks(),
        id.to_owned(),
        format!("{weight:?}"),
        format!("{usage:?}"),
    );
    let cache = CACHE.get_or_init(Mutex::default);
    if let Some(published) = cache.lock().map_err(|e| e.to_string())?.get(&key) {
        return Ok(Arc::clone(published));
    }
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let published = Arc::new(generate(ports, id, weight, usage)?);
    cache
        .lock()
        .map_err(|e| e.to_string())?
        .insert(key, Arc::clone(&published));
    Ok(published)
}

/// Generate one published use of a declared port. A format the declaration
/// or the shared kernels cannot publish is refused with the port and format.
pub fn generate(
    ports: &Ports,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Published, String> {
    generate_offer(ports, id, weight, usage)
        .map_err(|e| format!("projection {id} cannot publish {weight:?}: {e}"))
}

fn generate_offer(
    ports: &Ports,
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Published, String> {
    let shape = ports.shape;
    let ledger = aliases(shape)?;
    let operation = domain_in(ports, id, weight, usage)?;
    let mut generated = scheduled_offer(shape, &operation, usage, ledger)?;
    let mut capability = generated.offer.capability().map_err(|e| e.to_string())?;
    let scoped = unadopted(
        shape,
        generated
            .identities
            .iter()
            .zip(capability.stages.iter().map(String::as_str)),
    );
    if !scoped.is_empty() {
        generated = scheduled_offer(shape, &operation, usage, &[ledger, &scoped].concat())?;
        capability = generated.offer.capability().map_err(|e| e.to_string())?;
    }
    let mut adoptions: Vec<_> = generated
        .identities
        .iter()
        .zip(&capability.stages)
        .map(|(identity, symbol)| Adoption {
            symbol: symbol.clone(),
            ..identity.clone()
        })
        .collect();
    let mut copy = None;
    let mut prefix = vec![];
    match operation.kind {
        OperationKind::Dense => {
            let role = ports
                .role(id)
                .ok_or_else(|| format!("projection {id} is not a published dense port"))?;
            capability.implementation = role.mixed_implementation_for(shape, weight);
            if let Some((layout, identity)) = layout_copy_for(ports, role, weight, usage)? {
                let symbol = layout.stage().entry().symbol.clone();
                capability.stages.push(symbol.clone());
                adoptions.push(Adoption { symbol, ..identity });
                copy = Some(layout);
            }
        }
        OperationKind::Grouped { .. } => {}
        OperationKind::Lookup => {
            capability.implementation = lookup_implementation(shape);
            prefix.push(ngram_hash_symbol(shape, usage.phase));
            capability.stages.insert(0, prefix[0].clone());
        }
    }
    Ok(Published {
        offer: generated.offer,
        copy,
        prefix,
        capability,
        adoptions,
    })
}

/// The exact copy of a dense port's output columns into its fused family's
/// output row at one use, and its identity, or `None` when the port writes
/// its own row.
pub fn layout_copy_for(
    ports: &Ports,
    role: DenseRole,
    weight: WeightFormat,
    usage: OperationUse,
) -> Result<Option<(row::LayoutCopy, Adoption)>, String> {
    let shape = ports.shape;
    let Some((columns, source_stride, destination_stride, offset)) = role
        .copy_layout_in(ports, weight)
        .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let ledger = aliases(shape)?;
    let layout_copy = |aliases: &[Adoption]| {
        shared::layout_copy(
            columns,
            projection_capacity(usage).map_err(|e| e.to_string())?,
            row::CopyRows {
                row_stride: source_stride,
                column_stride: 1,
                column_offset: 0,
            },
            row::CopyRows {
                row_stride: destination_stride,
                column_stride: 1,
                column_offset: offset,
            },
            row::ShapeBinding::Launch,
            aliases,
        )
    };
    let (mut layout, identity) = layout_copy(ledger)?;
    let scoped = unadopted(shape, [(&identity, layout.stage().entry().symbol.as_str())]);
    if !scoped.is_empty() {
        layout = layout_copy(&[ledger, &scoped].concat())?.0;
    }
    Ok(Some((layout, identity)))
}

/// Publisher construction, never a checkpoint-time compilation request.
/// The registered n-gram hash entry of one topology and phase:
/// `chop_{tag}_ngram_hash_{phase}_v1`, with a `_tp{N}` tag suffix outside
/// TP4.
pub fn ngram_hash_symbol(shape: Shape, phase: ExecutionPhase) -> String {
    ngram_hash_entry(
        shape,
        match phase {
            ExecutionPhase::Prefill => "prefill",
            ExecutionPhase::Decode => "decode",
            ExecutionPhase::Verify => "verify",
        },
    )
}

/// [`ngram_hash_symbol`] by the phase's entry label.
pub fn ngram_hash_entry(shape: Shape, phase: &str) -> String {
    format!("chop_{}_ngram_hash_{phase}_v1", topology_tag(shape))
}

/// A topology's symbol tag: the geometry's kernel tag at TP4, with a
/// `_tp{N}` suffix at another rank count.
pub fn topology_tag(shape: Shape) -> String {
    if shape.ranks == crate::contract::TOPOLOGY.ranks {
        shape.geometry.kernel_tag.to_owned()
    } else {
        format!("{}_tp{}", shape.geometry.kernel_tag, shape.ranks())
    }
}

/// The name prefix of a topology's mixed cubins: `{kernel_tag}_precision`
/// at TP4, `{kernel_tag}_precision_tp{N}` at another rank count.
pub fn cubin_prefix(shape: Shape) -> String {
    if shape.ranks == crate::contract::TOPOLOGY.ranks {
        format!("{}_precision", shape.geometry.kernel_tag)
    } else {
        format!(
            "{}_precision_tp{}",
            shape.geometry.kernel_tag,
            shape.ranks()
        )
    }
}

/// Whether the cubin `name` is one of a geometry's mixed cubins (its TP4
/// [`cubin_prefix`], or another rank count's, followed by `_`).
pub fn mixed_cubin(name: &str) -> bool {
    crate::geometry::GEOMETRIES.iter().any(|geometry| {
        let prefix = cubin_prefix(Shape::new(geometry, crate::contract::TOPOLOGY.ranks));
        name.strip_prefix(&prefix)
            .is_some_and(|suffix| suffix.starts_with('_'))
    })
}

/// The registered hash entries of every geometry and topology that
/// publishes the packed n-gram lookup, as the bundle embeds them
/// (`{cubin_prefix}_hash_v1`).
pub fn ngram_hash_forms() -> Result<Vec<kernels_sm120::row::ngram_hash::NgramHashForm>, String> {
    use kernels_sm120::row::ngram_hash::{NgramHashForm, NgramTable};
    let mut forms = Vec::new();
    for geometry in crate::geometry::GEOMETRIES {
        for topology in TOPOLOGIES {
            let shape = Shape::new(geometry, topology.ranks);
            let prefill = ngram_hash_symbol(shape, kernel_core::precision::ExecutionPhase::Prefill);
            if !inventory(shape)?
                .document
                .operations
                .iter()
                .any(|operation| operation.stages.contains(&prefill))
            {
                continue;
            }
            forms.push(NgramHashForm {
                cubin: format!("{}_hash_v1", cubin_prefix(shape)),
                prefill,
                decode: ngram_hash_symbol(shape, kernel_core::precision::ExecutionPhase::Decode),
                verify: ngram_hash_symbol(shape, kernel_core::precision::ExecutionPhase::Verify),
                ranks: shape.ranks(),
                local_heads: shape.ngram_local_heads,
                table: NgramTable {
                    vocab: u64::from(geometry.vocab()),
                    orders: geometry.ngram_orders() as usize,
                    heads_per_order: geometry.ngram_heads_per_order() as usize,
                    base: u64::from(geometry.ngram_vocab_base()),
                    seed: 0,
                    layer_index: 0,
                    alignment: u64::from(geometry.ngram_alignment()),
                },
                eos: shape.ngram_eos(),
            });
        }
    }
    Ok(forms)
}

pub struct Inventory {
    pub offers: Vec<KernelOffer>,
    pub document: CapabilityDocument,
    /// Every distinct layout copy use: uses of one entry at several
    /// capacities share its symbol, so cubins take each symbol once.
    pub copies: Vec<row::LayoutCopy>,
    pub adoptions: Vec<Adoption>,
}

fn append(records: &mut BTreeMap<String, Adoption>, incoming: &[Adoption]) -> Result<(), String> {
    for record in incoming {
        if let Some(old) = records.insert(record.symbol.clone(), record.clone())
            && old != *record
        {
            return Err(format!("conflicting adoption {}", record.symbol));
        }
    }
    Ok(())
}

/// A layout copy use: its entry, capacity and layout.
type CopyUse = (String, u32, [u32; 7]);

fn copy_use(copy: &row::LayoutCopy) -> CopyUse {
    let (columns, s, d) = copy.layout();
    (
        copy.stage().entry().symbol.clone(),
        copy.stage().capacity(),
        [
            columns,
            s.row_stride,
            s.column_stride,
            s.column_offset,
            d.row_stride,
            d.column_stride,
            d.column_offset,
        ],
    )
}

/// The layout copies and n-gram lookups of every offered port and use at
/// one topology, past the declaration's refusals (for example a format no
/// admitted export uses). Kernel-lab compares their row stages in both shape
/// forms.
pub fn row_uses(shape: Shape) -> Result<(Vec<KernelOffer>, Vec<row::LayoutCopy>), String> {
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    let offered = offered_ports(ports)?;
    let mut offers = Vec::new();
    let mut copies = BTreeMap::new();
    for usage in uses().map_err(|e| e.to_string())? {
        for &(id, weight) in &offered {
            if let Some(role) = ports.role(id) {
                if let Some((copy, _)) = layout_copy_for(ports, role, weight, usage)? {
                    copies.insert(copy_use(&copy), copy);
                }
            } else if ports.projection(id).map_err(|e| e.to_string())?.family()
                == KernelFamily::Lookup
            {
                offers.push(publish(shape, id, weight, usage)?.offer.clone());
            }
        }
    }
    Ok((offers, copies.into_values().collect()))
}

/// Every port and weight format the shared kernels could publish at one
/// topology, before the declaration's refusals.
pub fn published_ports(shape: Shape) -> Result<Vec<(&'static str, WeightFormat)>, String> {
    offered_ports(crate::projections::ports::get(shape).map_err(|e| e.to_string())?)
}

/// The declared offers of each port: the dense ports in role order, then
/// the expert banks and lookups that publish their own port.
pub fn offered_ports(ports: &Ports) -> Result<Vec<(&str, WeightFormat)>, String> {
    let declared = |id| -> Result<Vec<(&str, WeightFormat)>, String> {
        let p = ports.projection(id).map_err(|e| e.to_string())?;
        let weights = p.offered_weights().map_err(|e| e.to_string())?;
        Ok(weights.into_iter().map(|w| (p.id.as_str(), w)).collect())
    };
    let mut offers = Vec::new();
    for role in ports.roles() {
        offers.extend(declared(role.projection_id(ports.shape, None))?);
    }
    for family in [KernelFamily::GroupedExpert, KernelFamily::Lookup] {
        for p in &ports.declaration.projections {
            if p.family() == family && p.publishes_as.is_none() {
                offers.extend(declared(&p.id)?);
            }
        }
    }
    Ok(offers)
}

/// Every published use at one topology. A topology publishes the mixed
/// composition only where the declaration admits a checkpoint that uses it.
pub fn inventory(shape: Shape) -> Result<Inventory, String> {
    let ports = crate::projections::ports::get(shape).map_err(|e| e.to_string())?;
    inventory_in(ports, |id, weight, usage| publish(shape, id, weight, usage))
}

/// [`inventory`] of one topology's ports, publishing each use through
/// `publish`.
pub fn inventory_in(
    ports: &'static Ports,
    publish: impl Fn(&str, WeightFormat, OperationUse) -> Result<Arc<Published>, String>,
) -> Result<Inventory, String> {
    let ports_formats = offered_ports(ports)?;
    let support =
        crate::projections::support::Support::in_ports(ports, ports_formats.iter().copied())
            .map_err(|e| e.to_string())?;
    let mut offers = Vec::new();
    let mut operations = Vec::new();
    let mut copies = BTreeMap::new();
    let mut records = BTreeMap::new();
    for usage in uses().map_err(|e| e.to_string())? {
        // Ports whose generated kernels coincide share one offer: a format
        // a later port offers alone is still built.
        let mut built = std::collections::BTreeSet::new();
        for &(id, weight) in &ports_formats {
            let format = registry::by_format(weight).map_err(|e| e.to_string())?.id();
            if support.admit_format(id, usage.phase, format).is_err() {
                continue;
            }
            let published = publish(id, weight, usage)?;
            let first = built.insert(
                published
                    .offer
                    .capability()
                    .map_err(|e| e.to_string())?
                    .stages,
            );
            operations.push(published.capability.clone());
            if let Some(copy) = &published.copy {
                copies.insert(copy_use(copy), copy.clone());
            }
            append(&mut records, &published.adoptions)?;
            if first {
                offers.push(published.offer.clone());
            }
        }
    }
    let document = CapabilityDocument {
        version: 1,
        operations,
    };
    if !document.operations.is_empty() {
        document.validate()?;
    }
    Ok(Inventory {
        offers,
        document,
        copies: copies.into_values().collect(),
        adoptions: records.into_values().collect(),
    })
}

/// The generated capabilities of every topology of one geometry.
pub fn generated_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<Vec<OperationCapability>, String> {
    let mut operations = Vec::new();
    for topology in TOPOLOGIES {
        operations.extend(
            inventory(Shape::new(geometry, topology.ranks))?
                .document
                .operations,
        );
    }
    Ok(operations)
}

pub fn all_capabilities() -> Result<Vec<OperationCapability>, String> {
    let mut operations = Vec::new();
    for geometry in crate::geometry::GEOMETRIES {
        operations.extend(generated_for(geometry)?);
    }
    Ok(operations)
}

pub fn adoptions() -> Result<Vec<Adoption>, String> {
    let mut result = BTreeMap::new();
    for geometry in crate::geometry::GEOMETRIES {
        for topology in TOPOLOGIES {
            append(
                &mut result,
                &inventory(Shape::new(geometry, topology.ranks))?.adoptions,
            )?;
        }
    }
    let current = shared::placement::capture(
        result.into_values(),
        include_str!("../../../bundle/kernels.lock"),
    )?;
    Ok(shared::append_adoptions(ledger()?, current))
}

/// The complete portable document, retaining registered legacy compositions.
pub fn document_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<CapabilityDocument, String> {
    let mut operations = generated_for(geometry)?;
    operations.extend(legacy_capabilities_for(geometry).map_err(|e| e.to_string())?);
    let document = CapabilityDocument {
        version: 1,
        operations,
    };
    document.validate()?;
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    /// Every entry symbol of one geometry's TP4 inventory, and its row
    /// stages (copies and lookups) whose shapes are compiled.
    fn symbols(geometry: &'static crate::geometry::Geometry) -> (Vec<String>, Vec<String>) {
        symbols_at(geometry, crate::contract::Ranks::TP4)
    }

    /// [`symbols`] at one topology.
    fn symbols_at(
        geometry: &'static crate::geometry::Geometry,
        ranks: crate::contract::Ranks,
    ) -> (Vec<String>, Vec<String>) {
        let inventory = inventory(Shape::new(geometry, ranks)).unwrap();
        let mut rows: Vec<_> = inventory
            .copies
            .iter()
            .map(|copy| copy.stage().clone())
            .collect();
        let mut symbols = std::collections::BTreeSet::new();
        for offer in &inventory.offers {
            symbols.extend(
                offer
                    .handle()
                    .entries()
                    .unwrap()
                    .into_iter()
                    .map(|e| e.symbol),
            );
            rows.extend(offer.handle().row_stages());
        }
        symbols.extend(rows.iter().map(|stage| stage.entry().symbol.clone()));
        let compiled = rows
            .iter()
            .filter(|stage| stage.shapes() == row::ShapeBinding::Compiled)
            .map(|stage| stage.entry().symbol.clone())
            .collect::<std::collections::BTreeSet<_>>();
        (
            symbols.into_iter().collect(),
            compiled.into_iter().collect(),
        )
    }

    fn flash_clone() -> &'static crate::geometry::Geometry {
        let mut copy = crate::geometry::FLASH;
        copy.id = "flash-copy";
        copy.kernel_tag = "flash_copy";
        Box::leak(Box::new(copy))
    }

    #[test]
    fn another_geometry_serves_flash_code_under_its_own_symbols() {
        // Flash's dimensions under another tag: every entry has adopted or
        // generated code identical to Flash's. Its own cubins hold every
        // entry but the launch-shape layout copy, which both serve from the
        // shared cubin. At every topology, adopted symbols and content-named
        // ones alike.
        for topology in TOPOLOGIES {
            let (flash, _) = symbols_at(&crate::geometry::FLASH, topology.ranks);
            let (other, _) = symbols_at(flash_clone(), topology.ranks);
            assert_eq!(flash.len(), other.len());
            let (shared, own): (Vec<_>, Vec<_>) = other.iter().partition(|s| shape_free(s));
            assert_eq!(shared.len(), 1, "{shared:?}");
            assert!(shared.iter().all(|s| flash.contains(s)));
            assert!(own.iter().all(|s| s.starts_with("chop_flash_copy_")));
            assert!(own.iter().all(|s| !flash.contains(s)));
        }
    }

    /// A registered or added geometry adds no launch-shape copy: each serves
    /// Flash's, and its own rows cubin holds only the compiled fallback
    /// stages, its lookups (launch-shape lookups measured slower).
    #[test]
    fn geometries_share_every_launch_shape_row_stage() {
        let (flash, _) = symbols(&crate::geometry::FLASH);
        let shared = flash.iter().filter(|s| shape_free(s)).collect::<Vec<_>>();
        for geometry in crate::geometry::GEOMETRIES.iter().chain([flash_clone()]) {
            let (all, compiled) = symbols(geometry);
            let launch = all.iter().filter(|s| shape_free(s)).collect::<Vec<_>>();
            assert_eq!(launch, shared, "{}", geometry.id);
            assert!(
                !compiled.is_empty() && compiled.iter().all(|s| s.contains("_lookup_")),
                "{}: {compiled:?}",
                geometry.id
            );
            eprintln!(
                "{}: {} TP4 projection entries, {} shared launch-shape row stages, {} compiled lookups",
                geometry.id,
                all.len(),
                launch.len(),
                compiled.len()
            );
        }
    }

    /// The mixed copies at TP2 are the TP4 inventory's launch-shape entry
    /// at other shapes; the compiled lookup's TP2 shape is one new entry.
    /// TP2 adds one copy or lookup entry (59 before shapes were launch
    /// arguments).
    #[test]
    fn tp2_copies_add_no_entry_and_lookups_one() {
        let (tp4, _) = symbols(&crate::geometry::FLASH);
        let (offers, copies) = row_uses(Shape::TP2).unwrap();
        let mut stages = copies
            .iter()
            .map(|copy| copy.stage().clone())
            .collect::<Vec<_>>();
        stages.extend(offers.iter().flat_map(|offer| offer.handle().row_stages()));
        assert!(!copies.is_empty() && !offers.is_empty());
        let new = stages
            .iter()
            .map(|stage| stage.entry().symbol.clone())
            .filter(|symbol| !tp4.contains(symbol))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            copies
                .iter()
                .all(|copy| copy.stage().shapes() == row::ShapeBinding::Launch)
        );
        assert!(
            offers
                .iter()
                .flat_map(|offer| offer.handle().row_stages())
                .all(|stage| stage.shapes() == row::ShapeBinding::Compiled)
        );
        assert_eq!(new.len(), 1, "TP2 adds {new:?}");
        assert!(new.iter().all(|symbol| symbol.contains("_lookup_")));
        eprintln!(
            "TP2: {} copy and {} lookup uses, {} new entries",
            copies.len(),
            offers.len(),
            new.len()
        );
    }

    /// The ledger's active records are exactly current entries, and each
    /// retired symbol's `replaced_by` chain ends at a current entry of its
    /// kind and format.
    #[test]
    fn retired_adoptions_are_replaced_by_current_entries_of_their_kind() {
        let mut current = BTreeMap::new();
        for geometry in crate::geometry::GEOMETRIES {
            for topology in TOPOLOGIES {
                for record in inventory(Shape::new(geometry, topology.ranks))
                    .unwrap()
                    .adoptions
                {
                    current.insert(record.symbol.clone(), record);
                }
            }
        }
        // `chop_proj_{format}_{kind}_`: formats may contain underscores.
        let family = |generated: &str| {
            let rest = generated.strip_prefix("chop_proj_").unwrap();
            ["_dense_", "_grouped_", "_lookup_", "_copy_"]
                .iter()
                .find_map(|kind| rest.find(kind).map(|at| rest[..at + kind.len()].to_owned()))
                .unwrap()
        };
        let mut latest = BTreeMap::new();
        for record in ledger().unwrap() {
            latest.insert(record.symbol.as_str(), record);
        }
        let mut retired = 0;
        for (symbol, record) in &latest {
            if record.replaced_by.is_none() {
                assert!(
                    current.contains_key(*symbol),
                    "active {symbol} is not current"
                );
                continue;
            }
            assert!(
                !current.contains_key(*symbol),
                "retired {symbol} is current"
            );
            let mut next = *record;
            let mut seen = std::collections::BTreeSet::from([*symbol]);
            while let Some(successor) = &next.replaced_by {
                assert!(seen.insert(successor), "{symbol} is replaced in a cycle");
                next = latest[successor.as_str()];
            }
            assert_eq!(
                family(&record.generated),
                family(&current[&next.symbol].generated),
                "{symbol}"
            );
            retired += 1;
        }
        assert!(retired > 0);
    }

    /// A use that binds by composition, kind, geometry, format and
    /// epilogues finds one implementation: topologies whose ports share a
    /// geometry never publish the same composition with other stages.
    #[test]
    fn topologies_publish_unambiguous_compositions() {
        let mut seen = BTreeMap::new();
        for topology in TOPOLOGIES {
            let shape = Shape::new(&crate::geometry::FLASH, topology.ranks);
            for capability in inventory(shape).unwrap().document.operations {
                let key = format!(
                    "{} {:?} {:?} {:?} {:?} {:?}",
                    capability.implementation,
                    capability.kind,
                    capability.geometry,
                    capability.weight,
                    capability.usage,
                    capability.epilogues
                );
                if let Some((ranks, stages)) =
                    seen.insert(key.clone(), (shape.ranks(), capability.stages.clone()))
                {
                    assert!(
                        stages == capability.stages,
                        "TP{ranks} and TP{} differ for {key}",
                        shape.ranks()
                    );
                }
            }
        }
    }

    /// The hash forms the catalog declares are the bundle's: the tables
    /// the contract hashes into, each topology's local heads, and an entry
    /// per phase that each published lookup names as its prefix.
    #[test]
    fn declared_ngram_hash_forms_follow_each_published_lookup() {
        let forms = ngram_hash_forms().unwrap();
        assert_eq!(forms.len(), TOPOLOGIES.len());
        for form in &forms {
            let shape = Shape::flash(crate::contract::Ranks::new(form.ranks).unwrap());
            let tables = crate::contract::ngram_tables_for(shape.geometry).unwrap();
            let t = form.table;
            let declared = kernel_core::ngram_oracle::ngram_tables(
                t.vocab,
                t.orders,
                t.heads_per_order,
                t.base,
                t.seed,
                t.layer_index,
                t.alignment,
            )
            .unwrap();
            assert_eq!(declared, tables);
            assert_eq!(form.local_heads * form.ranks, shape.ngram_heads());
            let stages: BTreeSet<String> = inventory(shape)
                .unwrap()
                .document
                .operations
                .into_iter()
                .filter(|c| c.kind == OperationKind::Lookup)
                .map(|c| c.stages[0].clone())
                .collect();
            assert_eq!(
                stages,
                BTreeSet::from([
                    form.prefill.clone(),
                    form.decode.clone(),
                    form.verify.clone()
                ])
            );
        }
    }

    /// Every generated offer a topology serves keeps what serving relies
    /// on: a prefill use's rows over chunks at the manifest's granule, an
    /// exact verify use's (`EXACT_VERIFY`) rows their one-row bits. Prepared
    /// projections ask these needs, so a lost claim would refuse creation.
    #[test]
    fn served_offers_keep_what_serving_relies_on() {
        use kernel_core::precision::ExecutionPhase;
        let mut checked = 0;
        for geometry in crate::geometry::GEOMETRIES {
            for topology in TOPOLOGIES {
                for offer in inventory(Shape::new(geometry, topology.ranks))
                    .unwrap()
                    .offers
                {
                    let usage = offer.usage();
                    let claim = offer.invariance();
                    let kept = match usage.phase {
                        ExecutionPhase::Prefill => {
                            claim.chunks_at(crate::contract::PREFILL_GRANULE)
                        }
                        ExecutionPhase::Verify
                            if usage.capacity <= crate::contract::EXACT_VERIFY.rows =>
                        {
                            claim.one_row_at(usage.capacity)
                        }
                        _ => continue,
                    };
                    assert!(
                        kept,
                        "{:?} at {usage:?} claims {claim:?}",
                        offer.capability().unwrap().stages
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0);
    }

    #[test]
    fn generated_adoptions_match_the_committed_ledger() {
        let committed = ledger().unwrap();
        for expected in adoptions().unwrap() {
            assert!(
                committed.contains(&expected),
                "missing adoption {}",
                expected.symbol
            );
        }
    }
}
