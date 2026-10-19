//! The model's native attention entries use the admitted geometry and
//! prepare through Backend at every supported rank count.

use bundle_deepseek41::sm120::latent;
use kernel_core::backend::{Backend, Need};
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::GEOMETRIES;

#[test]
fn latent_registration_prepares_all_rank_counts() {
    let mut prepared = 0;
    for geometry in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            check_ranks(geometry, ranks).unwrap();
            let entries = latent::entries(geometry, ranks).unwrap();
            assert_eq!(
                entries.len(),
                7 + 3 * usize::from(geometry.shape.compress_ratios.contains(&2))
            );
            let backend =
                latent::register(kernels_sm120::backend::Sm120::new(), geometry, ranks).unwrap();
            for entry in entries {
                let op = backend
                    .latent_attention_sources(&entry.query, Need::rows(4))
                    .unwrap();
                assert_eq!(op.info().symbols.len(), 2);
                assert!(op.info().bytes_of("workspace").unwrap() > 0);
                prepared += 1;
            }
        }
        let cubin = latent::cubin(geometry).unwrap();
        assert_eq!(cubin.entries.len(), 42);
        let symbols = cubin
            .entries
            .iter()
            .map(|e| &e.symbol)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(symbols.len(), cubin.entries.len());
    }
    assert!(!GEOMETRIES.is_empty());
    assert_eq!(
        prepared,
        GEOMETRIES
            .iter()
            .map(|g| 21 + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
    );
}

pub(super) struct Modules;

impl kernel_harness::kit::Modules for Modules {
    fn load<'s>(
        &'s self,
        session: &'s kernel_harness::Session,
    ) -> kernel_harness::LabResult<kernel_harness::kit::modules::Functions<'s>> {
        let loaded = bundle_deepseek41::sm120::latent_cubins()
            .map(|(_, bytes, symbols)| Ok((symbols, session.load_cubin(bytes)?)))
            .collect::<kernel_harness::LabResult<Vec<_>>>()?;
        Ok(Box::new(move |symbol| {
            let (_, module) = loaded
                .iter()
                .find(|(symbols, _)| symbols.contains(&symbol))
                .ok_or_else(|| format!("no model latent cubin holds {symbol}"))?;
            module.function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(
        &self,
        session: &kernel_harness::Session,
    ) -> kernel_harness::LabResult<Option<kernel_harness::conformance::Probes>> {
        use kernel_harness::poison::{CopyProducer, LiveProducer};
        let (bytes, _) = bundle_deepseek41::sm120::dependency_probes();
        let module = session.load_cubin(bytes)?;
        let live = module
            .function(kernels_sm120::probes::LIVE)
            .map_err(|e| e.to_string())?;
        let copy = module
            .function(kernels_sm120::probes::COPY)
            .map_err(|e| e.to_string())?;
        Ok(Some(kernel_harness::conformance::Probes {
            live: LiveProducer::with_function(session, live)?,
            copy: Some(CopyProducer::with_function(copy)),
        }))
    }
}

fn conform(
    session: Option<&kernel_harness::Session>,
    ratio2_only: bool,
) -> kernel_harness::LabResult<()> {
    use kernel_harness::kit::{self, OpResult, Outcome, cases};
    let mut prepared = 0;
    for g in GEOMETRIES {
        for &ranks in SUPPORTED_RANKS {
            let entries = latent::entries(g, ranks)?
                .into_iter()
                .filter(|e| {
                    !ratio2_only
                        || matches!(
                            e.query,
                            kernel_core::latent_attention::LatentAttentionQuery::Attention {
                                page_size: 32,
                                ..
                            }
                        )
                })
                .collect::<Vec<_>>();
            let backend = latent::register(kernels_sm120::backend::Sm120::new(), g, ranks)?;
            let report = kit::conform(
                &backend,
                &Modules,
                session,
                &entries
                    .iter()
                    .flat_map(|e| {
                        if matches!(
                            e.query,
                            kernel_core::latent_attention::LatentAttentionQuery::Attention {
                                second_fp4_keys: true,
                                ..
                            }
                        ) {
                            cases::latent_sources::paged_family(std::slice::from_ref(e), true)
                        } else {
                            cases::latent_sources::numerical_family(std::slice::from_ref(e))
                        }
                    })
                    .collect::<Vec<_>>(),
            )?;
            if let Some(session) = session {
                session.receipt(
                    "model-native-latent",
                    std::collections::BTreeMap::from([
                        ("geometry".into(), serde_json::json!(g.id)),
                        ("ranks".into(), serde_json::json!(ranks)),
                        ("report".into(), report.json()),
                        ("power_cap_watts".into(), serde_json::json!(250)),
                    ]),
                )?;
            }
            assert!(report.failures().is_empty(), "{:?}", report.failures());
            assert_eq!(
                entries.len(),
                (if ratio2_only { 0 } else { 7 })
                    + 3 * usize::from(g.shape.compress_ratios.contains(&2))
            );
            assert_eq!(report.ops.len(), entries.len());
            assert_eq!(report.prepared().len(), entries.len());
            for op in &report.ops {
                let OpResult::Prepared { checks, .. } = &op.result else {
                    panic!("required latent op refused")
                };
                if session.is_some() {
                    assert!(
                        checks
                            .iter()
                            .all(|c| matches!(c.outcome, Outcome::Passed(_))),
                        "{}: {checks:?}",
                        op.name
                    );
                    for required in ["conformance", kit::PDL] {
                        assert!(
                            checks
                                .iter()
                                .any(|c| c.name == required
                                    && matches!(c.outcome, Outcome::Passed(_)))
                        );
                    }
                }
            }
            prepared += report.prepared().len();
        }
    }
    assert_eq!(
        prepared,
        GEOMETRIES
            .iter()
            .map(|g| (if ratio2_only { 0 } else { 21 })
                + 9 * usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
    );
    assert!(prepared > 0);
    Ok(())
}

#[test]
fn latent_required_cases_prepare_and_symbols_are_embedded() {
    let (bytes, symbols) = bundle_deepseek41::sm120::dependency_probes();
    assert!(!bytes.is_empty());
    assert_eq!(
        symbols,
        &[kernels_sm120::probes::COPY, kernels_sm120::probes::LIVE]
    );
    for symbol in symbols {
        assert!(bytes.windows(symbol.len()).any(|s| s == symbol.as_bytes()));
    }
    let embedded = bundle_deepseek41::sm120::latent_cubins().collect::<Vec<_>>();
    let scheduled = bundle_deepseek41::sm120::latent_scheduled::cubins(GEOMETRIES).unwrap();
    assert_eq!(
        embedded.len(),
        GEOMETRIES
            .iter()
            .map(|g| 2 + usize::from(g.shape.compress_ratios.contains(&2)))
            .sum::<usize>()
            + scheduled.len()
    );
    for spec in scheduled {
        let (_, bytes, symbols) = embedded
            .iter()
            .find(|(name, _, _)| *name == spec.name)
            .unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(symbols.len(), spec.entries.len());
        for entry in spec.entries {
            assert!(symbols.contains(&entry.symbol.as_str()));
            assert!(
                bytes
                    .windows(entry.symbol.len())
                    .any(|s| s == entry.symbol.as_bytes())
            );
        }
    }
    for g in GEOMETRIES {
        let spec = latent::cubin(g).unwrap();
        let (_, bytes, symbols) = embedded
            .iter()
            .find(|(name, _, _)| *name == spec.name)
            .unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(symbols.len(), 42);
        for entry in spec.entries {
            assert!(symbols.contains(&entry.symbol.as_str()));
            assert!(
                bytes
                    .windows(entry.symbol.len())
                    .any(|s| s == entry.symbol.as_bytes())
            );
        }
    }
    for g in GEOMETRIES
        .iter()
        .filter(|g| g.shape.compress_ratios.contains(&2))
    {
        let spec = latent::ratio2_cubin(g).unwrap();
        let (_, bytes, symbols) = embedded
            .iter()
            .find(|(name, _, _)| *name == spec.name)
            .unwrap();
        assert_eq!(symbols.len(), 18);
        for entry in spec.entries {
            assert!(symbols.contains(&entry.symbol.as_str()));
            assert!(
                bytes
                    .windows(entry.symbol.len())
                    .any(|s| s == entry.symbol.as_bytes())
            );
        }
    }
    for g in GEOMETRIES {
        let spec = latent::bf16_cubin(g).unwrap();
        let (_, bytes, symbols) = embedded
            .iter()
            .find(|(name, _, _)| *name == spec.name)
            .unwrap();
        assert_eq!(
            symbols.len(),
            42 + 18 * usize::from(g.shape.compress_ratios.contains(&2))
        );
        for entry in spec.entries {
            assert!(symbols.contains(&entry.symbol.as_str()));
            assert!(
                bytes
                    .windows(entry.symbol.len())
                    .any(|s| s == entry.symbol.as_bytes())
            );
        }
    }
    conform(None, false).unwrap();
}

#[test]
fn latent_native_instances_meet_f64_bounds_and_replay() {
    kernel_harness::gpu_test(1 << 30, |session| conform(Some(session), false))
        .expect("model native latent instances");
}

#[test]
fn latent_ratio2_instances_meet_f64_bounds_and_replay() {
    kernel_harness::gpu_test(1 << 30, |session| conform(Some(session), true))
        .expect("model ratio-2 latent instances");
}
