//! Generated comparison inventory over the retained catalog schedules.
use super::*;
use crate::sm120::workspace::moe::Selection as ExpertSelection;
use kernels_sm120::precision::KernelOffer;
use kernels_sm120::precision::generated::{self as shared, Adoption, comparison};
use std::collections::BTreeMap;

/// The token capacity of the entry serving `usage`: decode and verify
/// buckets above the decode entries' rows run the prefill-shaped entries
/// at the bucket's rows.
fn entry_capacity(usage: OperationUse) -> u32 {
    if usage.phase == ExecutionPhase::Prefill || usage.capacity > DECODE_ENTRY_ROWS {
        PREFILL_TOKENS
    } else {
        usage.capacity
    }
}

/// The SM120 GEMM catalog symbol of `projection` at `tp` for `usage`.
pub fn symbol(projection: Projection, tp: Tp, usage: OperationUse) -> String {
    let tag = if projection == Projection::Router {
        crate::topology::kernel_tag(Tp::of_geometry(tp.geometry, 4).expect("listed geometry"))
    } else {
        crate::topology::kernel_tag(tp)
    };
    let bucket = match entry_capacity(usage) {
        PREFILL_TOKENS => "prefill".to_owned(),
        capacity => format!("m{capacity}"),
    };
    format!("chop_gemm_{tag}_{}_{bucket}_v1", projection.entry())
}

/// The committed schedules record the exact expert entries. The inventory
/// never reads the process environment: the inexact alternative binds its
/// own entry, and the bundle selects between the two at execution.
const CAPTURED: ExpertSelection = ExpertSelection { exact: true };

pub fn schedules() -> Result<&'static [shared::schedule::Override], String> {
    static RECORDS: std::sync::OnceLock<Result<Vec<shared::schedule::Override>, String>> =
        std::sync::OnceLock::new();
    RECORDS
        .get_or_init(|| {
            serde_json::from_str(include_str!("../../../bundle/projection-schedules.json"))
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

fn schedule_key(
    operation: &kernel_core::precision::OperationSpec,
    usage: OperationUse,
) -> Result<shared::schedule::Key, String> {
    let mut inputs = shared::schedule::key(operation, usage)?;
    let mut named = inputs.clone();
    named.projection = Some(operation.id.clone());
    if schedules()?.iter().any(|r| r.inputs == named) {
        inputs = named;
    }
    Ok(inputs)
}

fn draft_operation(old: &KernelOffer) -> Result<kernel_core::precision::OperationSpec, String> {
    let cap = old.capability().map_err(|e| e.to_string())?;
    let mut operation = comparison::operation(&cap)?;
    let symbol = &cap.stages[0];
    operation.id = [
        ("q", "dflash.q"),
        ("qkv", "dflash.qkv"),
        ("context_kv", "dflash.context-kv"),
        ("o", "dflash.output"),
        ("gate_up", "dflash.mlp.gate-up"),
        ("down", "dflash.down"),
        ("fc", "dflash.fc"),
    ]
    .into_iter()
    .find(|(role, _)| symbol.contains(&format!("_dflash_{role}_")))
    .ok_or("missing draft declaration projection ID")?
    .1
    .into();
    Ok(operation)
}

// Export-only detection: name a draft projection only if another served bucket
// has the same portable inputs but different launch or reduction knobs.
fn draft_schedule_collides(
    old: &KernelOffer,
    operation: &kernel_core::precision::OperationSpec,
    catalog: &[kernels_sm120::gemm::GemmVariant],
) -> Result<bool, String> {
    let inputs = shared::schedule::key(operation, old.usage())?;
    let actual = comparison::implementation(old)?;
    let kernels_sm120::precision::Compute::Dense(spec) = &old.handle().compute else {
        return Err("draft is not dense".into());
    };
    let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
        return Err("draft GEMM query".into());
    };
    for peer in catalog {
        let kernel_core::contract::StaticQuery::Gemm(q) = peer.key.query else {
            continue;
        };
        if q.m_capacity != query.m_capacity {
            continue;
        }
        let candidate =
            KernelOffer::catalog_variant(peer, old.usage()).map_err(|e| e.to_string())?;
        let domain = comparison::operation(&candidate.capability().map_err(|e| e.to_string())?)?;
        if shared::schedule::key(&domain, old.usage())? == inputs
            && comparison::implementation(&candidate)? != actual
        {
            return Ok(true);
        }
    }
    Ok(false)
}

pub fn scheduled_offer(
    operation: &kernel_core::precision::OperationSpec,
    usage: OperationUse,
    aliases: &[Adoption],
) -> Result<shared::GeneratedOffer, String> {
    shared::schedule::offer(
        operation,
        &schedule_key(operation, usage)?,
        schedules()?,
        aliases,
    )
}

pub fn captured_schedules() -> Result<Vec<shared::schedule::Override>, String> {
    let mut rows = Vec::new();
    let catalog = crate::sm120::gemm::catalog();
    for geometry in crate::geometry::GEOMETRIES {
        for ranks in Tp::SUPPORTED {
            let tp = Tp::of_geometry(geometry, ranks.ranks()).map_err(|e| e.to_string())?;
            // Operations come from the entries the bundle selects, never
            // from the committed schedules, so a capture can add a row.
            for projection in Projection::of(geometry) {
                for usage in uses() {
                    if projection.routed() {
                        let variant = crate::sm120::workspace::moe::entry(
                            tp,
                            if projection == Projection::ExpertGateUp {
                                kernels_sm120::moe::Projection::GateUp
                            } else {
                                kernels_sm120::moe::Projection::Down
                            },
                            usage.phase,
                            usage.capacity,
                            CAPTURED,
                        )
                        .map_err(|e| e.to_string())?;
                        let cap =
                            kernels_sm120::precision::catalog_grouped_capability(variant, usage)
                                .map_err(|e| e.to_string())?;
                        rows.push(shared::schedule::capture_grouped(
                            &comparison::operation(&cap)?,
                            usage,
                            variant,
                        )?);
                    } else if projection.lookup() {
                        let entry = lookup_entry(tp)?;
                        let cap = kernels_sm120::precision::catalog_lookup_capability(entry, usage)
                            .map_err(|e| e.to_string())?;
                        rows.push(shared::schedule::capture_row(
                            &comparison::operation(&cap)?,
                            usage,
                            entry,
                        )?);
                    } else {
                        let entry = catalog
                            .iter()
                            .find(|e| e.symbol == symbol(projection, tp, usage))
                            .ok_or("dense entry")?;
                        let old = KernelOffer::catalog_variant(entry, usage)
                            .map_err(|e| e.to_string())?;
                        let operation =
                            comparison::operation(&old.capability().map_err(|e| e.to_string())?)?;
                        if let Some(record) = shared::schedule::capture(&operation, &old)? {
                            rows.push(record);
                        }
                    }
                }
            }
            for entry in catalog
                .iter()
                .filter(|e| e.profile == crate::sm120::profile(tp) && e.symbol.contains("_dflash_"))
            {
                let kernel_core::contract::StaticQuery::Gemm(q) = entry.key.query else {
                    return Err("draft GEMM query".into());
                };
                let usage = OperationUse {
                    phase: if q.m_capacity > 32 {
                        ExecutionPhase::Prefill
                    } else {
                        ExecutionPhase::Verify
                    },
                    capacity: q.m_capacity,
                    graph: q.m_capacity <= 32,
                };
                let old = KernelOffer::catalog_variant(entry, usage).map_err(|e| e.to_string())?;
                let operation = draft_operation(&old)?;
                let record = if draft_schedule_collides(&old, &operation, &catalog)? {
                    let mut inputs = shared::schedule::key(&operation, usage)?;
                    inputs.projection = Some(operation.id.clone());
                    Some(shared::schedule::Override {
                        inputs,
                        schedule: shared::schedule::Schedule::Typed {
                            implementation: comparison::implementation(&old)?,
                        },
                    })
                } else {
                    shared::schedule::capture(&operation, &old)?
                };
                if let Some(record) = record {
                    rows.push(record);
                }
            }
        }
    }
    shared::schedule::append(schedules()?, rows)
}

/// The embedding row entry of the topology's vocabulary and hidden width.
fn lookup_entry(tp: Tp) -> Result<&'static kernels_sm120::row::RowInstantiation, String> {
    crate::sm120::workspace::row::entries_for(tp)
        .find(|e| {
            matches!(e.query,
                kernel_core::contract::RowQuery::EmbeddingGather { vocab, width }
                if vocab == tp.geometry.vocab() && width == tp.geometry.hidden())
        })
        .ok_or_else(|| "missing lookup entry".into())
}

/// The generated offer that reproduces a catalog entry, with its adoption
/// aliases. A data-only geometry adds no schedule records, and its catalog
/// may tune a shape another profile records; where generation does not
/// reproduce the entry its bundle selects, that entry is the offer. The
/// committed adoption ledger and schedule capture keep the listed geometries
/// on their generated offers.
fn adopted(
    operation: &kernel_core::precision::OperationSpec,
    usage: OperationUse,
    existing: &KernelOffer,
) -> (KernelOffer, Vec<Adoption>) {
    let generated = || {
        let aliases = shared::adopt(&scheduled_offer(operation, usage, &[])?, existing)?;
        let adopted = scheduled_offer(operation, usage, &aliases)?;
        (format!("{:?}", adopted.offer.handle()) == format!("{:?}", existing.handle()))
            .then_some((adopted.offer, aliases))
            .ok_or_else(|| "generated handle differs from the catalog entry".to_string())
    };
    generated().unwrap_or_else(|_: String| (existing.clone(), vec![]))
}

pub struct Inventory {
    pub offers: Vec<KernelOffer>,
    pub catalog: Vec<OperationCapability>,
    pub document: CapabilityDocument,
    pub adoptions: Vec<Adoption>,
}

pub fn inventory(geometry: &'static crate::geometry::Geometry) -> Result<Inventory, String> {
    let catalogs = crate::sm120::catalogs();
    let dense = crate::sm120::gemm::catalog();
    let mut operations = Vec::new();
    let mut offers = Vec::new();
    let mut catalog = Vec::new();
    let mut records = BTreeMap::new();
    for ranks in Tp::SUPPORTED {
        let tp = Tp::of_geometry(geometry, ranks.ranks()).map_err(|e| e.to_string())?;
        let profile = Some(crate::sm120::profile(tp));
        let mut specs = kernels_sm120::moe::cubin_specs(catalogs, profile);
        specs.extend(kernels_sm120::row::cubin_specs(catalogs, None));
        for projection in Projection::of(geometry) {
            for usage in uses() {
                let (mut capability, aliases) = if !projection.routed() && !projection.lookup() {
                    let entry = dense
                        .iter()
                        .find(|v| v.symbol == symbol(projection, tp, usage))
                        .ok_or("missing dense catalog entry")?;
                    let existing =
                        KernelOffer::catalog_variant(entry, usage).map_err(|e| e.to_string())?;
                    let operation =
                        comparison::operation(&existing.capability().map_err(|e| e.to_string())?)?;
                    let (offer, aliases) = adopted(&operation, usage, &existing);
                    offers.push(offer.clone().with_invariance(existing.invariance()));
                    let mut capability = offer.capability().map_err(|e| e.to_string())?;
                    if capability.weight != WeightFormat::Bf16 {
                        capability.local_scale_dtypes = vec![DType::F32];
                    }
                    (capability, aliases)
                } else {
                    let (capability, declaration) = if projection.routed() {
                        let variant = crate::sm120::workspace::moe::entry(
                            tp,
                            if projection == Projection::ExpertGateUp {
                                kernels_sm120::moe::Projection::GateUp
                            } else {
                                kernels_sm120::moe::Projection::Down
                            },
                            usage.phase,
                            usage.capacity,
                            CAPTURED,
                        )
                        .map_err(|e| e.to_string())?;
                        let cap =
                            kernels_sm120::precision::catalog_grouped_capability(variant, usage)
                                .map_err(|e| e.to_string())?;
                        let operation = comparison::operation(&cap)?;
                        let chosen =
                            shared::schedule::grouped(&operation, usage, variant, schedules()?)?;
                        let declaration = shared::identity::retained_grouped(&chosen, &cap)?;
                        if declaration != shared::identity::retained_grouped(variant, &cap)? {
                            return Err("grouped schedule policy differs from hand list".into());
                        }
                        (cap, declaration)
                    } else {
                        let entry = lookup_entry(tp)?;
                        let cap = kernels_sm120::precision::catalog_lookup_capability(entry, usage)
                            .map_err(|e| e.to_string())?;
                        let operation = comparison::operation(&cap)?;
                        let chosen = shared::schedule::row(&operation, usage, entry, schedules()?)?;
                        let declaration = shared::identity::retained_row(&chosen, &cap)?;
                        if declaration != shared::identity::retained_row(entry, &cap)? {
                            return Err("row schedule policy differs from hand list".into());
                        }
                        (cap, declaration)
                    };
                    catalog.push(capability.clone());
                    let operation = comparison::operation(&capability)?;
                    comparison::catalog(
                        &operation,
                        capability.clone(),
                        &specs,
                        &BTreeMap::from([(capability.stages[0].clone(), declaration)]),
                    )?
                };
                if capability.geometry != projection.geometry(tp)
                    || capability.operands as usize != projection.operands().len()
                    || capability.weight != projection.weight(tp)
                    || capability.arithmetic.output != projection.output()
                    || capability.epilogues != projection.epilogues()
                {
                    return Err("generated capability differs from model rank operation".into());
                }
                capability.implementation = projection.implementation(tp);
                operations.push(capability);
                for alias in aliases {
                    if let Some(old) = records.insert(alias.symbol.clone(), alias.clone())
                        && old != alias
                    {
                        return Err(format!("conflicting adoption {}", alias.symbol));
                    }
                }
            }
        }
    }
    // Draft projections share the locked dense cubins with text projections.
    for entry in &dense {
        if !entry.symbol.contains("_dflash_")
            || !Tp::SUPPORTED.iter().any(|ranks| {
                Tp::of_geometry(geometry, ranks.ranks())
                    .is_ok_and(|tp| crate::sm120::profile(tp) == entry.profile)
            })
        {
            continue;
        }
        let kernel_core::contract::StaticQuery::Gemm(query) = entry.key.query else {
            return Err("draft entry is not dense".into());
        };
        let usage = OperationUse {
            phase: if query.m_capacity > 32 {
                ExecutionPhase::Prefill
            } else {
                ExecutionPhase::Verify
            },
            capacity: query.m_capacity,
            graph: query.m_capacity <= 32,
        };
        let old = KernelOffer::catalog_variant(entry, usage).map_err(|e| e.to_string())?;
        let (offer, aliases) = adopted(&draft_operation(&old)?, usage, &old);
        offers.push(offer.with_invariance(old.invariance()));
        for alias in aliases {
            records.insert(alias.symbol.clone(), alias);
        }
    }
    let document = CapabilityDocument {
        version: 1,
        operations,
    };
    document.validate()?;
    Ok(Inventory {
        offers,
        catalog,
        document,
        adoptions: records.into_values().collect(),
    })
}

pub fn ledger() -> Result<Vec<Adoption>, String> {
    serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json"))
        .map_err(|e| e.to_string())
}

pub fn adoptions() -> Result<Vec<Adoption>, String> {
    let mut result = BTreeMap::new();
    for geometry in crate::geometry::GEOMETRIES {
        for record in inventory(geometry)?.adoptions {
            if let Some(old) = result.insert(record.symbol.clone(), record.clone())
                && old != record
            {
                return Err(format!("conflicting adoption {}", record.symbol));
            }
        }
    }
    let history = ledger()?;
    let current = shared::placement::capture(
        result.into_values(),
        include_str!("../../../bundle/kernels.lock"),
    )?;
    Ok(shared::append_adoptions(&history, current))
}

/// The catalog offer of a drafter projection's dense entry for `usage`,
/// which the generated inventory does not list.
pub(crate) fn drafter_capability(
    projection: Projection,
    tp: Tp,
    usage: OperationUse,
) -> Result<OperationCapability, Refusal> {
    let entry = crate::sm120::gemm::entry(&symbol(projection, tp, usage))
        .ok_or(Refusal("missing drafter projection entry"))?;
    KernelOffer::catalog_variant(entry, usage)
        .and_then(|offer| offer.capability())
        .map_err(|_| Refusal("invalid drafter projection offer"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// The inventory and every offer stay available with the exact verify
    /// switch off: it selects between bound alternatives only in the bundle.
    /// The check runs in a child process that sets the switch.
    #[test]
    fn inventory_ignores_the_exact_verify_switch() {
        if std::env::var_os(crate::serving::VERIFY_EXACT).is_some() {
            for geometry in crate::geometry::GEOMETRIES {
                inventory(geometry).unwrap();
                for ranks in Tp::SUPPORTED {
                    let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                    for projection in Projection::of(geometry) {
                        for usage in uses() {
                            super::super::capability(projection, tp, usage).unwrap();
                        }
                    }
                }
            }
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "capabilities::sm120::tests::inventory_ignores_the_exact_verify_switch",
            ])
            .env(crate::serving::VERIFY_EXACT, "0")
            .status()
            .unwrap();
        assert!(status.success());
    }
    #[test]
    fn draft_projection_ids_disambiguate_identical_prefill_shapes() {
        let declaration: serde_json::Value =
            serde_json::from_str(include_str!("../../projections.json")).unwrap();
        let catalog = crate::sm120::gemm::catalog();
        let mut checked = 0;
        for entry in catalog.iter().filter(|e| {
            e.symbol.contains("_dflash_qkv_prefill_")
                || e.symbol.contains("_dflash_context_kv_prefill_")
                || e.symbol.contains("_dflash_o_prefill_")
                || e.symbol.contains("_dflash_down_prefill_")
        }) {
            let usage = OperationUse {
                phase: ExecutionPhase::Prefill,
                capacity: 8192,
                graph: false,
            };
            let old = KernelOffer::catalog_variant(entry, usage).unwrap();
            let operation = draft_operation(&old).unwrap();
            assert!(
                declaration["projections"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|p| p["id"] == operation.id)
            );
            let fresh = scheduled_offer(&operation, usage, &[]).unwrap();
            shared::adopt(&fresh, &old).unwrap_or_else(|e| panic!("{}: {e}", operation.id));
            checked += 1;
        }
        assert_eq!(checked, 8);
    }

    #[test]
    fn named_overrides_only_disambiguate_real_schedule_collisions() {
        let catalog = crate::sm120::gemm::catalog();
        let records = schedules().unwrap();
        let mut count = 0;
        for record in records.iter().filter(|r| r.inputs.projection.is_some()) {
            let mut base = record.inputs.clone();
            base.projection = None;
            let matched = catalog
                .iter()
                .filter(|e| e.symbol.contains("_dflash_"))
                .any(|entry| {
                    let Ok(old) = KernelOffer::catalog_variant(entry, record.inputs.usage) else {
                        return false;
                    };
                    let operation = draft_operation(&old).unwrap();
                    record.inputs.projection.as_deref() == Some(operation.id.as_str())
                        && shared::schedule::key(&operation, old.usage()).unwrap() == base
                        && comparison::implementation(&old).unwrap()
                            == match &record.schedule {
                                shared::schedule::Schedule::Typed { implementation } => {
                                    implementation.clone()
                                }
                                _ => unreachable!(),
                            }
                        && draft_schedule_collides(&old, &operation, &catalog).unwrap()
                });
            assert!(
                matched,
                "unnecessary or unknown selector: {:?}",
                record.inputs
            );
            count += 1;
        }
        assert!(count >= 4);
        assert_eq!(captured_schedules().unwrap(), records);
    }

    /// The published offers equal, operation for operation, the capability
    /// of the catalog entry the bundle selects for each projection and use,
    /// checked against the rank shape without the generated schedules.
    #[test]
    fn public_offers_are_the_catalog_entries_the_bundle_selects() {
        let catalog = crate::sm120::gemm::catalog();
        for geometry in crate::geometry::GEOMETRIES {
            let mut expected = vec![];
            for ranks in Tp::SUPPORTED {
                let tp = Tp::of_geometry(geometry, ranks.ranks()).unwrap();
                for projection in Projection::of(geometry) {
                    for usage in uses() {
                        let mut capability = if projection.routed() {
                            let variant = crate::sm120::workspace::moe::entry(
                                tp,
                                if projection == Projection::ExpertGateUp {
                                    kernels_sm120::moe::Projection::GateUp
                                } else {
                                    kernels_sm120::moe::Projection::Down
                                },
                                usage.phase,
                                usage.capacity,
                                CAPTURED,
                            )
                            .unwrap();
                            kernels_sm120::precision::catalog_grouped_capability(variant, usage)
                                .unwrap()
                        } else if projection.lookup() {
                            kernels_sm120::precision::catalog_lookup_capability(
                                lookup_entry(tp).unwrap(),
                                usage,
                            )
                            .unwrap()
                        } else {
                            let entry = catalog
                                .iter()
                                .find(|e| e.symbol == symbol(projection, tp, usage))
                                .unwrap();
                            let mut capability =
                                KernelOffer::catalog_dense(entry.instantiation.clone(), usage)
                                    .unwrap()
                                    .capability()
                                    .unwrap();
                            if capability.weight != WeightFormat::Bf16 {
                                capability.local_scale_dtypes = vec![DType::F32];
                            }
                            capability
                        };
                        assert_eq!(capability.geometry, projection.geometry(tp));
                        assert_eq!(capability.weight, projection.weight(tp));
                        assert_eq!(capability.epilogues, projection.epilogues());
                        capability.implementation = projection.implementation(tp);
                        expected.push(capability);
                    }
                }
            }
            assert_eq!(document_for(geometry).unwrap().operations, expected);
        }
    }
    #[test]
    fn generated_adoptions_match_the_committed_ledger() {
        let committed: Vec<Adoption> =
            serde_json::from_str(include_str!("../../../bundle/projection-adoptions.json"))
                .unwrap();
        for expected in adoptions().unwrap() {
            assert!(
                committed.contains(&expected),
                "missing adoption {}",
                expected.symbol
            );
        }
    }

    /// Every offer names one entry of a MiMo GEMM or MoE cubin the bundle
    /// builds, and covers every projection, rank count and use.
    #[test]
    fn every_offer_is_a_compiled_mimo_entry() {
        let compiled: BTreeSet<String> = Tp::SUPPORTED
            .into_iter()
            .flat_map(|tp| {
                let profile = Some(crate::sm120::profile(tp));
                let catalogs = crate::sm120::catalogs();
                let mut specs = kernels_sm120::gemm::cubin_specs(catalogs, profile);
                specs.extend(kernels_sm120::moe::cubin_specs(catalogs, profile));
                specs
            })
            .chain(kernels_sm120::row::cubin_specs(
                crate::sm120::catalogs(),
                Some(crate::sm120::config::PROFILE),
            ))
            .flat_map(|spec| spec.entries)
            .map(|entry| entry.symbol)
            .collect();
        let document = document_for(&crate::geometry::V26_FLASH).unwrap();
        assert_eq!(
            document.operations.len(),
            Tp::SUPPORTED.len() * Projection::ALL.len() * uses().len()
        );
        for capability in &document.operations {
            assert_eq!(capability.stages.len(), 1);
            assert!(
                compiled.contains(&capability.stages[0]),
                "{:?}",
                capability.stages
            );
        }
        inventory_for(&crate::geometry::V26_FLASH).unwrap();
    }

    /// Expert offers name the entry each use runs: the decode entries up to
    /// 16 rows, exact entries for exact verify rows, and the prefill
    /// entries for prefill and every wider decode and verify bucket, at
    /// every rank count.
    #[test]
    fn expert_offers_follow_the_expert_selection() {
        let usage = |phase, capacity| OperationUse {
            phase,
            capacity,
            graph: phase != ExecutionPhase::Prefill,
        };
        let stage = |tp, phase, capacity| {
            let requested = usage(phase, capacity);
            let entry = crate::sm120::workspace::moe::entry(
                tp,
                kernels_sm120::moe::Projection::GateUp,
                phase,
                capacity,
                ExpertSelection { exact: true },
            )
            .unwrap();
            kernels_sm120::precision::catalog_grouped_capability(entry, requested)
                .unwrap()
                .stages
                .remove(0)
        };
        for tp in [Tp::TP4, Tp::TP2] {
            let prefill = stage(tp, ExecutionPhase::Prefill, PREFILL_TOKENS);
            assert!(prefill.contains("_prefill_mxfp4_a16_t8192_"), "{prefill}");
            for capacity in [1, 2, 4, 8, 16] {
                let decode = stage(tp, ExecutionPhase::Decode, capacity);
                assert!(
                    decode.contains(&format!("_decode_mxfp4_a16_t{capacity}_")),
                    "{decode}"
                );
            }
            for capacity in [32, 64, 128, 256] {
                for phase in [ExecutionPhase::Decode, ExecutionPhase::Verify] {
                    assert_eq!(stage(tp, phase, capacity), prefill);
                }
            }
            // Exact verify rows take the one-token reduction order.
            let exact = stage(tp, ExecutionPhase::Verify, 4);
            assert!(exact.contains("_exact_mxfp4_a16_t4_"), "{exact}");
        }
    }
}
