//! Header-only checkpoints of every listed geometry enter the same
//! admission and component builder as production.
use model_mimo26_contract::{Tp, catalog::TensorRecord, precision, weight_plan::Drafter};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

/// The served export's quantization rules, with the BF16 attention outputs
/// of the geometry's own layers.
fn quantization_config(tp: Tp) -> serde_json::Value {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../contract/tests/fixtures/projections/mopd.json"
    ))
    .unwrap();
    let mut config = fixture["quantization_config"].clone();
    let mut ignored: BTreeSet<_> = (0..tp.geometry.layers())
        .map(|layer| format!("model.layers.{layer}.self_attn.o_proj"))
        .collect();
    ignored.insert("model.decoder.self_attn.o_proj".into());
    config["ignored_layers"] = serde_json::json!(ignored);
    config
}

/// Header records of the tensors `roles` read.
fn records(roles: &BTreeMap<String, precision::Role>) -> BTreeMap<String, TensorRecord> {
    precision::expected_source_tensors(roles)
        .unwrap()
        .into_iter()
        .map(|(name, (dtype, shape))| {
            let width = match dtype.safetensors_name() {
                "BF16" => 2,
                "F32" => 4,
                _ => 1,
            };
            let record = TensorRecord {
                name: name.clone(),
                dtype,
                bytes: shape.iter().product::<u64>() * width,
                shape,
                file: "fixture.safetensors".into(),
                file_start: 0,
            };
            (name, record)
        })
        .collect()
}

fn build(tp: Tp, drafter: Option<Drafter>) -> Vec<u8> {
    let geometry = tp.geometry;
    let mut roles = precision::text_roles(geometry, |i| geometry.is_global(i as u32));
    if drafter == Some(Drafter::Mtp) {
        roles.extend(precision::mtp_roles(geometry));
    }
    let mut tensors = records(&roles);
    let metadata = precision::export_metadata(Some(&quantization_config(tp)), None).unwrap();
    let mut admitted = precision::admit(geometry, &metadata, &tensors, &roles).unwrap();
    if drafter == Some(Drafter::Dflash) {
        let drafter = records(&precision::dflash_roles(geometry));
        admitted.extend(precision::admit_dflash(geometry, &drafter).unwrap());
        tensors.extend(drafter);
    }
    let graph =
        model_mimo26_contract::projections::imports::derive(tp, 0, &tensors, &admitted).unwrap();
    model_mimo26_contract::prepared::encode(&graph, tp, &admitted).unwrap()
}

/// The topology's component with the packaged drafter's projections,
/// built once per process.
pub(super) fn component(tp: Tp, drafter: Option<Drafter>) -> &'static [u8] {
    static SLOTS: OnceLock<Vec<OnceLock<Vec<u8>>>> = OnceLock::new();
    let geometries = model_mimo26_contract::geometry::GEOMETRIES;
    let slots = SLOTS.get_or_init(|| (0..geometries.len() * 6).map(|_| OnceLock::new()).collect());
    let geometry = geometries
        .iter()
        .position(|g| g.id == tp.geometry.id)
        .unwrap();
    let slot = geometry * 6 + usize::from(tp == Tp::of_geometry(tp.geometry, 4).unwrap()) * 3;
    let kind = match drafter {
        None => 0,
        Some(Drafter::Mtp) => 1,
        Some(Drafter::Dflash) => 2,
    };
    slots[slot + kind].get_or_init(|| build(tp, drafter))
}

fn span(slot: u32) -> runtime_bundle_abi::DeviceBufferSpanV1 {
    runtime_bundle_abi::DeviceBufferSpanV1::new(
        runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(slot, 1),
        0,
        1 << 30,
    )
}

/// The topology's text component, authenticated.
fn authenticated(tp: Tp) -> bundle_runtime::projection::Authenticated {
    crate::projection::Prepared::authenticate(
        component(tp, None),
        tp,
        crate::switches::Switches::configured(),
        &crate::sm120::own_ops::Sm120Own::new(tp),
    )
    .unwrap()
}

/// Every selection bound to distinct placeholder views.
fn store(
    authenticated: &bundle_runtime::projection::Authenticated,
) -> bundle_runtime::projection::Store {
    authenticated
        .bind(|selection| {
            Ok(Some(bundle_runtime::projection::Weights {
                codes: span(100),
                local_scales: selection.effective.weights[0]
                    .local_scale
                    .as_ref()
                    .map(|_| span(101)),
                global_scales: None,
                activation_global: None,
            }))
        })
        .unwrap()
}

/// The choices of the configured switches with exact verify and the
/// medium-M entries as given.
fn choices(
    authenticated: &bundle_runtime::projection::Authenticated,
    exact_verify: bool,
    dense_mid: bool,
) -> crate::projection::Choices {
    crate::projection::Choices::new(
        authenticated.document(),
        |id, usage| authenticated.locate(id, usage),
        crate::switches::Switches {
            exact_verify,
            dense_mid,
            ..crate::switches::Switches::configured()
        },
        // The medium-M entries are the same at every topology.
        &crate::sm120::own_ops::Sm120Own::new(crate::topology::Tp::TP4),
    )
    .unwrap()
}

fn launch(
    rank: crate::projection::Rank<'_>,
    id: &str,
    phase: crate::selection::Phase,
    rows: u32,
) -> Result<Vec<crate::pipeline::PreparedLaunch>, kernel_core::contract::Refusal> {
    crate::projection::launch(
        rank,
        id,
        crate::projection::Launch {
            phase,
            rows,
            activation: span(1),
            live_rows: span(2),
            routing: Some(bundle_runtime::projection::Routing {
                weights: span(3),
                offsets: span(4),
                sorted_rows: span(5),
                padded_rows: span(6),
                active_experts: span(7),
            }),
            output: span(8),
            partial: span(9),
            semaphores: span(10),
            unit_scale: span(11),
        },
    )
    .map(crate::program_lower::launches)
}

/// With the exact verify switch off, a verify-8 expert step lowers the
/// authenticated inexact entry; the switch is a parameter, so the check
/// runs in process.
#[test]
fn inexact_expert_policy_lowers_its_authenticated_entry() {
    use kernel_core::precision::ExecutionPhase;
    use model_mimo26_contract::sm120::workspace::moe::{Selection, entry};
    let tp = Tp::TP4;
    let authenticated = authenticated(tp);
    let store = store(&authenticated);
    let symbol = |exact| {
        entry(
            tp,
            crate::sm120::launch::moe::Projection::GateUp,
            ExecutionPhase::Verify,
            8,
            Selection { exact },
        )
        .unwrap()
        .symbol
        .clone()
    };
    assert_ne!(symbol(true), symbol(false));
    for exact_verify in [true, false] {
        let choices = choices(&authenticated, exact_verify, true);
        let calls = launch(
            crate::projection::Rank {
                store: &store,
                choices: &choices,
            },
            "moe.experts.gate-up.1.bf16",
            crate::selection::Phase::Verify(8),
            8,
        )
        .unwrap();
        assert_eq!(calls[0].symbol, symbol(exact_verify));
    }
}

/// Medium-M entries replace the prefill-shaped entry of every decode and
/// verify use at exactly the rows the shared catalog selects them for
/// (`gemm::mid::select`), as the instance resolved when it was created;
/// a component without them is refused.
#[test]
fn medium_dense_alternatives_follow_the_catalog_and_are_required() {
    use crate::selection::MimoPhase as _;
    use kernel_core::precision::ExecutionPhase;
    let tp = Tp::TP4;
    let authenticated = authenticated(tp);
    let store = store(&authenticated);
    let mut medium = 0;
    for dense_mid in [true, false] {
        let choices = choices(&authenticated, true, dense_mid);
        let rank = crate::projection::Rank {
            store: &store,
            choices: &choices,
        };
        for id in ["dense.down.0.bf16", "ga.qkv.0.bf16", "swa.output.1.bf16"] {
            let position = authenticated
                .document()
                .operations
                .iter()
                .position(|s| s.source.id == id)
                .unwrap();
            for binding in &authenticated.document().operations[position].bindings {
                let usage = binding.usage;
                let phase = match usage.phase {
                    ExecutionPhase::Prefill => continue,
                    ExecutionPhase::Decode => crate::selection::Phase::Decode(usage.capacity),
                    ExecutionPhase::Verify => crate::selection::Phase::Verify(usage.capacity),
                };
                let reference = &binding.implementation.stages[0];
                for rows in 1..=usage.capacity {
                    let calls = launch(rank, id, phase, rows).unwrap();
                    let selected = crate::sm120::reference::medium_symbol(reference, rows)
                        .filter(|_| dense_mid && phase.prefill_kernels());
                    medium += usize::from(selected.is_some());
                    assert_eq!(
                        calls[0].symbol,
                        selected.unwrap_or(reference.clone()),
                        "{id} {usage:?} at {rows}"
                    );
                }
            }
        }
    }
    assert!(medium > 0);
    let id = "dense.down.0.bf16";
    let mut document = kernel_core::precision::component::decode(component(tp, None)).unwrap();
    document
        .operations
        .retain(|s| s.source.id != format!("{id}.mid"));
    let bytes = kernel_core::precision::component::encode(&document).unwrap();
    assert_eq!(
        crate::projection::Prepared::authenticate(
            &bytes,
            tp,
            crate::switches::Switches::configured(),
            &crate::sm120::own_ops::Sm120Own::new(tp)
        )
        .err()
        .unwrap(),
        format!("missing declared model port {id}.mid")
    );
    // Choices refuse a medium alternative missing at a use on their own.
    let missing = authenticated.document().operations[authenticated
        .document()
        .operations
        .iter()
        .position(|s| s.source.id == format!("{id}.mid"))
        .unwrap()]
    .bindings
    .last()
    .unwrap()
    .usage;
    let switches = crate::switches::Switches {
        dense_mid: true,
        ..crate::switches::Switches::configured()
    };
    let alternative = format!("{id}.mid");
    assert_eq!(
        crate::projection::Choices::new(
            authenticated.document(),
            |located, usage| {
                (located != alternative || usage != missing)
                    .then(|| authenticated.locate(located, usage))
                    .flatten()
            },
            switches,
            &crate::sm120::own_ops::Sm120Own::new(tp),
        )
        .err()
        .unwrap(),
        format!("missing row alternative {id}.mid")
    );
}

/// Every stage a component binds is in a cubin its topology loads, and
/// every dense binding, text or drafter, has the N and K of the rank view it
/// reads.
#[test]
fn bound_stages_are_loaded_and_match_their_rank_views() {
    use kernel_core::precision::OperationKind;
    use model_mimo26_contract::weight_plan::{assembled_resident, resident};
    for tp in Tp::SUPPORTED {
        let loaded: BTreeSet<_> = crate::CUBINS
            .iter()
            .filter(|cubin| crate::encoder_availability::cubin_serves(cubin.name, tp))
            .flat_map(|cubin| cubin.symbols.iter().copied())
            .collect();
        let mut views: BTreeMap<String, Vec<u64>> = model_mimo26_contract::text_views(tp)
            .into_iter()
            .map(|(name, view)| (name, view.shape))
            .collect();
        for layer in 0..tp.geometry.mtp_layers() {
            for (suffix, shape, _, _) in model_mimo26_contract::drafters::mtp_fields(tp) {
                views.insert(format!("model.mtp.layers.{layer}.{suffix}"), shape);
            }
        }
        views.extend(
            model_mimo26_contract::drafters::fields(tp).map(|(name, shape, _)| (name, shape)),
        );
        for drafter in [None, Some(Drafter::Mtp), Some(Drafter::Dflash)] {
            let document =
                kernel_core::precision::component::decode(component(tp, drafter)).unwrap();
            let mut dense = 0;
            for selection in &document.operations {
                for binding in &selection.bindings {
                    for stage in &binding.implementation.stages {
                        assert!(loaded.contains(stage.as_str()), "{stage} at {tp:?}");
                    }
                }
                if selection.effective.kind != OperationKind::Dense {
                    continue;
                }
                let id = &selection.source.id;
                let weights = &selection.effective.weights;
                let name = assembled_resident(tp, id)
                    .unwrap()
                    .unwrap_or_else(|| resident(&weights[0].codes.tensor, weights.len() == 2));
                let shape = views
                    .get(&name)
                    .unwrap_or_else(|| panic!("{id} reads no rank view {name} at {tp:?}"));
                let geometry = selection.effective.geometry.unwrap();
                assert_eq!(
                    shape[..],
                    [geometry.rows * weights.len() as u64, geometry.k],
                    "{id} at {tp:?}"
                );
                dense += 1;
            }
            assert!(dense > 0);
            if drafter == Some(Drafter::Dflash) {
                for port in [
                    "dflash.fc",
                    "dflash.context-kv",
                    "dflash.qkv.0",
                    "dflash.output.0",
                ] {
                    assert!(
                        document
                            .operations
                            .iter()
                            .any(|s| s.source.id.starts_with(port)),
                        "{port} at {tp:?}"
                    );
                }
            }
        }
    }
}

/// Every dense use a component binds launches through the SM120 backend,
/// a medium-M use wider than its entry's rows included, what MiMo's own
/// dense planner launched (`sm120::dense::plan_entry`):
/// symbol, rows, plan and resolved arguments. Decode and verify uses are
/// compared at every live row count, medium-M uses at every row their entry
/// serves, and prefill at its first 64 rows, every power of two and its
/// neighbours, a stride of 61 and its capacity. Uses of one identity launch
/// alike over their own weight views, so each identity and use is compared
/// once, over both drafter components (each holds the text program's).
#[test]
fn dense_projections_launch_what_mimo_planned() {
    use kernel_core::precision::ExecutionPhase;
    use kernel_core::precision::execution::ImplementationProvider as _;
    use model_mimo26_contract::sm120::prepared::Entry;
    let resolved = |launch: &crate::pipeline::PreparedLaunch| {
        crate::dispatch::arguments(&launch.arguments, &launch.bindings).unwrap()
    };
    // Lengths no capacity check reaches; each region its own buffer.
    let wide = |slot| {
        runtime_bundle_abi::DeviceBufferSpanV1::new(
            runtime_bundle_abi::BundleDeviceBufferHandleV1::from_parts(slot, 1),
            0,
            1 << 40,
        )
    };
    for tp in Tp::SUPPORTED {
        let contract = model_mimo26_contract::prepared::Provider { tp };
        let mut seen = BTreeSet::new();
        let mut compared = 0;
        let mut past = 0;
        for drafter in [Drafter::Mtp, Drafter::Dflash] {
            let authenticated = crate::projection::Prepared::authenticate(
                component(tp, Some(drafter)),
                tp,
                crate::switches::Switches::configured(),
                &crate::sm120::own_ops::Sm120Own::new(tp),
            )
            .unwrap();
            let store = authenticated
                .bind(|selection| {
                    Ok(Some(bundle_runtime::projection::Weights {
                        codes: wide(100),
                        local_scales: selection.effective.weights[0]
                            .local_scale
                            .as_ref()
                            .map(|_| wide(101)),
                        global_scales: None,
                        activation_global: None,
                    }))
                })
                .unwrap();
            // Each use launches its own selection, a medium-M one included.
            let choices = choices(&authenticated, true, false);
            for selection in &authenticated.document().operations {
                for binding in &selection.bindings {
                    let usage = binding.usage;
                    let bound = contract.bind(&selection.effective, usage).unwrap();
                    let Entry::Dense {
                        entry,
                        medium,
                        head,
                    } = bound.handle
                    else {
                        continue;
                    };
                    if !seen.insert(format!("{:?} {usage:?}", bound.identity)) {
                        continue;
                    }
                    // The backend prepares every dense use, a medium-M one
                    // wider than its entry's rows at the entry's rows.
                    let recipe = store.handle(&selection.source.id, usage).unwrap().recipe();
                    assert!(
                        matches!(recipe, bundle_runtime::projection::Recipe::Prepared { .. }),
                        "{} {usage:?}",
                        selection.source.id
                    );
                    let past_rows = medium
                        .as_ref()
                        .is_some_and(|medium| usage.capacity > medium.max_rows());
                    past += usize::from(past_rows);
                    let phase = match usage.phase {
                        ExecutionPhase::Prefill => crate::selection::Phase::Prefill,
                        ExecutionPhase::Decode => crate::selection::Phase::Decode(usage.capacity),
                        ExecutionPhase::Verify => crate::selection::Phase::Verify(usage.capacity),
                    };
                    let rows: Vec<u32> = match (&medium, usage.phase) {
                        (Some(medium), _) => medium.rows.clone().collect(),
                        (None, ExecutionPhase::Prefill) => {
                            let mut rows: BTreeSet<u32> = (1..=64).collect();
                            for shift in 6..14 {
                                rows.extend([(1 << shift) - 1, 1 << shift, (1 << shift) + 1]);
                            }
                            rows.extend((1..usage.capacity).step_by(61));
                            rows.insert(usage.capacity);
                            rows.into_iter().filter(|&r| r <= usage.capacity).collect()
                        }
                        (None, _) => (1..=usage.capacity).collect(),
                    };
                    let id = &selection.source.id;
                    for live in rows {
                        let prepared = crate::projection::launch(
                            crate::projection::Rank {
                                store: &store,
                                choices: &choices,
                            },
                            id,
                            crate::projection::Launch {
                                phase,
                                rows: live,
                                activation: wide(1),
                                live_rows: wide(2),
                                routing: None,
                                output: wide(8),
                                partial: wide(9),
                                semaphores: wide(10),
                                unit_scale: wide(11),
                            },
                        )
                        .map(crate::program_lower::launches)
                        .unwrap_or_else(|e| panic!("{id} {usage:?} at {live} rows: {e:?}"));
                        let planned = crate::sm120::dense::plan_entry(
                            entry.clone(),
                            medium.clone(),
                            live,
                            crate::sm120::dense::ProjectViews {
                                activation: wide(1),
                                weight: wide(100),
                                weight_scales: selection.effective.weights[0]
                                    .local_scale
                                    .as_ref()
                                    .map(|_| wide(101)),
                                output: wide(8),
                                partial: wide(9),
                                semaphores: wide(10),
                            },
                            head,
                        )
                        .unwrap();
                        let [prepared] = &prepared[..] else {
                            panic!("{id} {usage:?}: one dense launch")
                        };
                        let at = format!("{id} {usage:?} at {live} rows on {tp:?}");
                        assert_eq!(prepared.symbol, planned.symbol, "{at}");
                        assert_eq!(prepared.rows, planned.rows, "{at}");
                        assert_eq!(prepared.plan, planned.plan, "{at}");
                        assert_eq!(resolved(prepared), resolved(&planned), "{at}");
                        compared += 1;
                    }
                }
            }
        }
        assert!(compared > 1000, "{compared} launches at {tp:?}");
        assert!(past > 0, "wide medium-M uses at {tp:?}");
    }
}

/// The BF16 head's decode buckets also serve a prefill's few unscored
/// logits rows, so they need, and their prepared ops provide, each row's
/// bits in any chunk; other decode uses need only their rows, and the F32
/// head nothing beyond its phase's need.
#[test]
fn bf16_head_decode_uses_need_chunk_invariance() {
    use kernel_core::precision::ExecutionPhase;
    use kernel_core::properties::{Chunk, Invariance};
    let chunk = Invariance::NONE.with_chunk(Chunk::Any);
    let switches = crate::switches::Switches::configured();
    let authenticated = authenticated(Tp::TP4);
    let mut heads = 0;
    for selection in &authenticated.document().operations {
        for binding in &selection.bindings {
            let usage = binding.usage;
            let id = selection.source.id.as_str();
            let need = crate::projection::need(id, usage, switches).unwrap();
            let reached = id == crate::projection::BF16_HEAD
                && usage.phase == ExecutionPhase::Decode
                && usage.capacity <= crate::contract::DECODE_ENTRY_ROWS;
            if usage.phase == ExecutionPhase::Decode {
                assert_eq!(need.exact.chunk == Chunk::Any, reached, "{id} {usage:?}");
            }
            if reached {
                heads += 1;
            }
        }
    }
    assert!(heads > 0);
    let store = store(&authenticated);
    for capacity in [1, 2, 4, 8, 16, 32] {
        let usage =
            model_mimo26_contract::prepared::usage(crate::selection::Phase::Decode(capacity))
                .unwrap();
        let handle = store.handle(crate::projection::BF16_HEAD, usage).unwrap();
        let bundle_runtime::projection::Recipe::Prepared { op, .. } = handle.recipe() else {
            panic!("the BF16 head is prepared")
        };
        assert!(op.info().guarantees.provides(chunk), "decode {capacity}");
    }
}
