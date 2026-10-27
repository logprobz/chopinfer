//! Recipes with dependency builds, alternate lock roots or qualification exports.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn probes() -> Step {
    family(
        Root::Shared,
        "dependency_probes",
        vec![kernels_sm120::probes::cubin_spec()],
    )
    .pinned()
}

fn fold_baselines(decode: bool, ticket: bool) -> Result<Vec<Step>> {
    let mut steps = vec![
        ordered(Root::Bundle, bundle::mxfp8_fold::cubins(GEOMETRIES)?)
            .at("baseline")
            .pinned(),
    ];
    if decode {
        steps.push(
            ordered(
                Root::Lab,
                crate::dense_decode_reference::decode_cubins(GEOMETRIES)?,
            )
            .at("decode")
            .pinned(),
        );
    }
    if ticket {
        steps.push(
            ordered(
                Root::Lab,
                crate::dense_decode_reference::ticket_cubins(GEOMETRIES)?,
            )
            .at("ticket")
            .pinned(),
        );
    }
    Ok(steps)
}

fn attention() -> Result<Vec<Step>> {
    let mut references: BTreeMap<&str, Vec<CubinSpec>> = BTreeMap::new();
    for (name, spec) in bundle::latent::cubins(GEOMETRIES)?
        .into_iter()
        .filter(|(name, _)| *name != "latent_sources_bf16")
        .chain(bundle::latent_glue::cubins(GEOMETRIES)?)
    {
        references.entry(name).or_default().push(spec);
    }
    let mut steps: Vec<_> = references
        .into_iter()
        .map(|(name, specs)| family(Root::Bundle, name, specs).pinned())
        .collect();
    let g = GEOMETRIES
        .iter()
        .find(|g| g.kernel_tag == "deepseek41")
        .unwrap();
    steps.extend([
        family(
            Root::Bundle,
            bundle::latent_scheduled::FAMILY,
            bundle::latent_scheduled::cubins(GEOMETRIES)?,
        ),
        family(
            Root::Bundle,
            bundle::latent_scheduled::QUERY_FAMILY,
            vec![bundle::latent_scheduled::query_cubin()],
        ),
        probes(),
        family(
            Root::Lab,
            lab::attention_probe::FAMILY,
            vec![lab::attention_probe::cubin()?],
        ),
        family(
            Root::Bundle,
            bundle::latent_decode::FAMILY,
            vec![bundle::latent_decode::cubin(g)?],
        ),
        family(
            Root::Lab,
            lab::cache_catalog::FAMILY,
            vec![lab::cache_catalog::cubin(g)?],
        ),
        family(
            Root::Lab,
            lab::grouped_cache::FAMILY,
            vec![lab::grouped_cache::cubin(g)?],
        ),
    ]);
    Ok(steps)
}

fn streamed() -> Result<Vec<Step>> {
    let mut specs = bundle::moe::streamed_cubins(GEOMETRIES)?;
    let symbols: BTreeSet<_> = specs
        .iter()
        .flat_map(|(_, spec)| spec.entries.iter().map(|e| e.symbol.clone()))
        .collect();
    // The standalone wrapper predates the bundle's lane-layout conversion.
    // Apply that same in-place conversion to reproduce the locked entries.
    for geometry in GEOMETRIES {
        let builds = bundle_deepseek41::sm120::moe_row_tiles::builds(geometry)?
            .into_iter()
            .filter(|(name, _)| symbols.contains(name))
            .collect();
        bundle_deepseek41::sm120::moe_row_tiles::serve(&mut specs, &builds)?;
    }
    Ok(vec![ordered(Root::Bundle, specs)])
}

fn lab_prefill_tile() -> Result<Vec<Step>> {
    let mut spec = bundle::indexer_mma_prefill_tile::cubins(&[FLASH])?
        .remove(0)
        .1;
    // The lab lock keeps the original module name; bundle adoption renamed it
    // without changing the cubin bytes or entry declarations.
    spec.name = kernels_sm120::pooled_indexer::position_mma_prefill_tile::CUBIN.into();
    Ok(vec![family(
        Root::Lab,
        bundle::indexer_mma_prefill_tile::FAMILY,
        vec![spec],
    )])
}

pub fn recipes() -> Vec<Recipe> {
    macro_rules! own {
        ($entries:ident, $cubin:ident) => {
            recipe(
                Root::Bundle,
                bundle::$entries::FAMILY,
                || {
                    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../bundle")
                        .canonicalize()?;
                    Ok(vec![family(
                        Root::Bundle,
                        bundle::$entries::FAMILY,
                        vec![bundle::$cubin::cubin_spec(&path)],
                    )])
                },
                Export::None,
            )
        };
    }
    vec![
        recipe(
            Root::Shared,
            "dependency_probes",
            || Ok(vec![probes()]),
            Export::None,
        ),
        own!(acceptance_entries, acceptance_cubin),
        own!(host_acceptance_entries, host_acceptance_cubin),
        own!(ring_gather_entries, ring_gather_cubin),
        own!(retain_rows_entries, retain_rows_cubin),
        own!(engram_hash_entries, engram_hash_cubin),
        own!(confidence_parallel, confidence_parallel),
        recipe(
            Root::Lab,
            crate::dense_decode_reference::DECODE_FAMILY,
            || {
                let mut steps = vec![
                    ordered(
                        Root::Lab,
                        crate::dense_decode_reference::decode_cubins(GEOMETRIES)?,
                    )
                    .at("decode")
                    .pinned(),
                ];
                steps.extend(fold_baselines(false, false)?);
                Ok(steps)
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            crate::dense_decode_reference::TICKET_FAMILY,
            || {
                let mut steps = vec![
                    ordered(
                        Root::Lab,
                        crate::dense_decode_reference::ticket_cubins(GEOMETRIES)?,
                    )
                    .at("ticket")
                    .pinned(),
                ];
                steps.extend(fold_baselines(true, false)?);
                Ok(steps)
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            crate::dense_decode_reference::PREFETCH_FAMILY,
            || {
                let mut steps = vec![
                    ordered(
                        Root::Lab,
                        crate::dense_decode_reference::prefetch_cubins(GEOMETRIES)?,
                    )
                    .at("prefetch")
                    .pinned(),
                ];
                steps.extend(fold_baselines(true, true)?);
                Ok(steps)
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            crate::dense_decode_reference::REUSE_FAMILY,
            || {
                Ok(vec![
                    ordered(
                        Root::Lab,
                        crate::dense_decode_reference::reuse_cubins(GEOMETRIES)?,
                    )
                    .pinned(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::mxfp8_fold::DENSE_FAMILY,
            || {
                Ok(vec![
                    ordered(Root::Bundle, bundle::mxfp8_fold::dense_cubins(GEOMETRIES)?).pinned(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::latent_window_bf16::FAMILY,
            || {
                Ok(vec![
                    ordered(
                        Root::Bundle,
                        bundle::latent_window_bf16::cubins(GEOMETRIES)?,
                    ),
                    probes(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::latent_glue::FAMILY,
            || {
                Ok(vec![
                    ordered(Root::Bundle, bundle::latent_glue::cubins(GEOMETRIES)?),
                    probes(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::latent_window_stage::FAMILY,
            || {
                Ok(vec![
                    ordered(
                        Root::Bundle,
                        bundle::latent_window_stage::cubins(GEOMETRIES)?
                            .into_iter()
                            .chain(bundle::latent_window_stage::key_cubins(GEOMETRIES)?)
                            .collect(),
                    ),
                    probes(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::latent_window_stage::KEY_FAMILY,
            || {
                Ok(vec![
                    ordered(
                        Root::Bundle,
                        bundle::latent_window_stage::key_cubins(GEOMETRIES)?,
                    ),
                    probes(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::query_prefill::FAMILY,
            || {
                Ok(vec![
                    ordered(Root::Bundle, bundle::query_prefill::cubins(GEOMETRIES)?),
                    family(
                        Root::Output("control-locks"),
                        "controls",
                        vec![kernels_sm120::row::rmsnorm_f32_prefill::cubin_spec(
                            "query_prefill_controls_sm120",
                            &bundle::query_prefill::controls()?,
                        )?],
                    ),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            bundle::latent_scheduled::FAMILY,
            attention,
            Export::Attention,
        ),
        recipe(
            Root::Bundle,
            bundle::moe::STREAMED_FAMILY,
            streamed,
            Export::None,
        ),
        recipe(
            Root::Lab,
            bundle::indexer_mma_prefill_tile::FAMILY,
            lab_prefill_tile,
            Export::Modules("candidate-modules.json"),
        ),
        recipe(
            Root::Lab,
            bundle::indexer_select_early::FAMILY,
            || {
                let (name, spec) = bundle::indexer_select_early::cubins(&[FLASH])?.remove(0);
                assert_eq!(name, bundle::indexer_select_early::FAMILY);
                Ok(vec![family(Root::Lab, name, vec![spec])])
            },
            Export::Modules("select-early-modules.json"),
        ),
        recipe(
            Root::Lab,
            "position_candidates_cached",
            || {
                use kernels_sm120::pooled_indexer::{
                    position_blocks::CandidateEntry, position_candidates_cached,
                };
                let entry = CandidateEntry {
                    name: FLASH.kernel_tag.into(),
                    query: model_deepseek41_contract::indexer::candidates(
                        &FLASH,
                        model_deepseek41_contract::serving::PREFILL_TOKENS,
                    ),
                };
                let spec = position_candidates_cached::cubin_spec(
                    "position_candidates_cached_sm120",
                    vec![position_candidates_cached::spec(&entry)?],
                );
                Ok(vec![family(
                    Root::Lab,
                    "position_candidates_cached",
                    vec![spec],
                )])
            },
            Export::Modules("candidate-modules.json"),
        ),
        recipe(
            Root::Lab,
            "position_mma_tile",
            || {
                use kernels_sm120::pooled_indexer::position_mma_tile;
                let entries = bundle::indexer::entries(&FLASH);
                let scores = entries
                    .iter()
                    .flat_map(|e| position_mma_tile::VARIANTS.map(|v| v.spec(e)))
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                Ok(vec![family(
                    Root::Lab,
                    "position_mma_tile",
                    vec![position_mma_tile::cubin_spec(scores)],
                )])
            },
            Export::Modules("candidate-modules.json"),
        ),
        recipe(
            Root::Lab,
            "indexer_prefill_probe",
            || {
                Ok(vec![family(
                    Root::Lab,
                    "indexer_prefill_probe",
                    vec![lab::indexer_prefill_probe::spec()],
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            exact::family::FAMILY,
            || {
                Ok(vec![family(
                    Root::Lab,
                    exact::family::FAMILY,
                    exact::family::cubins(),
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            exact::family::FAMILY,
            || {
                Ok(vec![family(
                    Root::Bundle,
                    exact::family::FAMILY,
                    exact::family::cubins(),
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            exact::vector::FAMILY,
            || {
                Ok(vec![family(
                    Root::Lab,
                    exact::vector::FAMILY,
                    exact::vector::cubins(),
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            exact::vector::FAMILY,
            || {
                Ok(vec![family(
                    Root::Bundle,
                    exact::vector::FAMILY,
                    exact::vector::cubins(),
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            lab::fold_bandwidth::FAMILY,
            || {
                Ok(vec![family(
                    Root::Lab,
                    lab::fold_bandwidth::FAMILY,
                    vec![lab::fold_bandwidth::cubin()?],
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            lab::fold_bandwidth::FAMILY,
            || {
                Ok(vec![family(
                    Root::Bundle,
                    lab::fold_bandwidth::FAMILY,
                    vec![lab::fold_bandwidth::cubin()?],
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            lab::fold_cluster::FAMILY,
            || {
                Ok(vec![family(
                    Root::Lab,
                    lab::fold_cluster::FAMILY,
                    lab::fold_cluster::cubins(),
                )])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            "fold_local_rows_reference",
            || {
                let (_, spec) = bundle_deepseek41::sm120::fold_local_rows::cubins(
                    std::slice::from_ref(&FLASH),
                )?
                .pop()
                .ok_or("missing retained fold reference")?;
                Ok(vec![
                    family(Root::Lab, "fold_local_rows_reference", vec![spec]).pinned(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            "mxfp8_fold_cluster_reference",
            || {
                let (_, spec) =
                    bundle_deepseek41::sm120::fold_cluster::cubins(std::slice::from_ref(&FLASH))?
                        .pop()
                        .ok_or("missing retained fold reference")?;
                Ok(vec![
                    family(Root::Lab, "mxfp8_fold_cluster_reference", vec![spec]).pinned(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Lab,
            "mxfp8_prefill_tma_matrix_reference",
            || {
                let (_, spec) = bundle_deepseek41::sm120::prefill_tma_matrix::cubins(
                    std::slice::from_ref(&FLASH),
                )?
                .pop()
                .ok_or("missing retained fold reference")?;
                Ok(vec![
                    family(Root::Lab, "mxfp8_prefill_tma_matrix_reference", vec![spec]).pinned(),
                ])
            },
            Export::None,
        ),
        recipe(
            Root::Bundle,
            "routed_moe_router_finish",
            || {
                let spec = kernels_sm120::moe::chained::router_finish::cubin_spec(
                    "router_finish_moe_deepseek41_sm120",
                    &bundle::moe::floor_entries(&FLASH)?,
                );
                Ok(vec![family(
                    Root::Bundle,
                    "routed_moe_router_finish",
                    vec![spec],
                )])
            },
            Export::RouterReferences,
        ),
        recipe(
            Root::Output("kernels.lock.d"),
            "routed_moe_router_finish64",
            || {
                let entries = bundle::moe::floor_entries(&FLASH)?
                    .into_iter()
                    .filter(|e| e.experts() == 384 && e.top_k() == 6)
                    .collect::<Vec<_>>();
                if entries.len() != 1 {
                    return Err("expected one backbone router geometry".into());
                }
                let spec = kernels_sm120::moe::chained::router_finish64::cubin_spec(
                    "router_finish_moe_deepseek41_m64_sm120",
                    &entries,
                );
                if spec.entries.len() != 2 {
                    return Err("missing wide router stages".into());
                }
                Ok(vec![family(
                    Root::Output("kernels.lock.d"),
                    "routed_moe_router_finish64",
                    vec![spec],
                )])
            },
            Export::None,
        ),
    ]
}
