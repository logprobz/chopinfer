//! Fake host coverage of the MTP drafter through a mixed component's
//! recorded MTP layer: the NVIDIA shape (FP8 block MTP experts on the
//! registered compositions) and the local-lab shape (NVFP4 MTP experts on
//! the main layers' generated entries), at TP2 and TP4.
use super::precision::{document_bound, published, selection};
use super::speculation::{
    assert_drafter_rows, batch, committed_state, drafted, drafter_host_at, execute, kept, plain,
    truth, truths,
};
use super::*;
use kernel_core::contract::WeightFormat;
use kernel_core::precision::execution::{PrecisionDocument, PrecisionSelection};
use model_qwen38_contract::precision::{self as ports, Projection};

/// The MTP layer's recorded operations: its BF16 dense ports on their
/// registered compositions, and its experts in `experts` on the registered
/// FP8 composition or the main layers' NVFP4 entries.
fn drafter_records(
    shape: Shape,
    experts: WeightFormat,
    inventory: &kernel_core::precision::capability::CapabilityInventory,
) -> Vec<PrecisionSelection> {
    let usage = ports::uses().unwrap()[0];
    let layer = shape.mtp_layer();
    let mut records: Vec<_> = ports::ports(shape)
        .unwrap()
        .layer(layer)
        .iter()
        .map(|(role, id)| {
            selection(
                id.clone(),
                role.legacy_capability(shape, usage).unwrap(),
                inventory,
            )
        })
        .collect();
    for projection in [Projection::GateUp, Projection::Down] {
        let (name, main) = if projection == Projection::GateUp {
            ("gate_up_proj", "main.experts.gate-up")
        } else {
            ("down_proj", "main.experts.down")
        };
        let capability = if experts == WeightFormat::Nvfp4 {
            published(shape, main, WeightFormat::Nvfp4, usage)
        } else {
            ports::legacy_expert_capability(shape, projection, usage).unwrap()
        };
        records.push(selection(
            format!("mtp.layers.0.mlp.experts.{name}"),
            capability,
            inventory,
        ));
    }
    records
}

/// A component of one export shape: NVFP4 experts and the NVFP4 lookup,
/// MXFP8 dense ports when `mx`, and the MTP layer recorded with `mtp`
/// experts (or not at all).
pub(super) fn component(shape: Shape, mx: bool, mtp: Option<WeightFormat>) -> PrecisionDocument {
    let inventory = crate::precision::inventory().unwrap();
    let mut document = document_bound(shape, true, mx, false, &inventory);
    if let Some(experts) = mtp {
        document
            .operations
            .extend(drafter_records(shape, experts, &inventory));
    }
    document.validate().unwrap();
    document
}

/// The NVIDIA export shape (BF16 dense, FP8 block MTP experts) and the
/// local-lab shape (MXFP8 dense, NVFP4 MTP experts).
const EXPORTS: [(&str, bool, WeightFormat); 2] = [
    (
        "NVIDIA",
        false,
        WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32),
    ),
    ("local-lab", true, WeightFormat::Nvfp4),
];

fn mixed_drafter_host(shape: Shape, document: PrecisionDocument, fp8_kv: bool) -> Rc<FakeHost> {
    let host = drafter_host_at(3, shape);
    host.fp8_kv.set(fp8_kv);
    *host.precision.borrow_mut() = Some(document);
    host
}

fn create_mixed_drafter(host: &Rc<FakeHost>) -> BundleInstanceV1 {
    let instance = create(host);
    host.model(instance.handle());
    instance
}

/// Every registered FP8 fragment expert entry of `shape`: the fixed
/// drafter program's, and the NVIDIA shape's recorded MTP experts'.
fn fragment_experts(shape: Shape) -> BTreeSet<String> {
    let mut symbols = BTreeSet::new();
    for usage in ports::uses().unwrap() {
        for projection in [Projection::GateUp, Projection::Down] {
            symbols.extend(
                ports::legacy_expert_capability(shape, projection, usage)
                    .unwrap()
                    .stages,
            );
        }
    }
    symbols
}

/// Both export shapes at TP2 and TP4, with BF16 and FP8 KV, prefill, draft,
/// verify, accept all drafts, reject some, cancel a drafted step and
/// decode: drafts follow the model, the committed state is the accepted
/// tokens', the MTP experts run their recorded entries and every resource
/// is released.
#[test]
fn recorded_mtp_layers_draft_verify_accept_reject_and_cancel() {
    for shape in [Shape::TP4, Shape::TP2] {
        for (export, mx, experts) in EXPORTS {
            for fp8_kv in [false, true] {
                let label = format!("{export} TP{} fp8_kv={fp8_kv}", shape.ranks());
                let document = component(shape, mx, Some(experts));
                let fragments = fragment_experts(shape);
                let host = mixed_drafter_host(shape, document, fp8_kv);
                let mut instance = create_mixed_drafter(&host);
                let prompt = truths(0..5);
                execute(&host, &mut instance, &batch(vec![plain(10, 1, 0, &prompt)]));
                instance.commit().unwrap();
                // Draft three and accept them all with the bonus row.
                let before = host.launches.borrow().len();
                let sink = execute(
                    &host,
                    &mut instance,
                    &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
                );
                assert_eq!(sink.draft_tokens(), [(10, truths(6..9))], "{label}");
                // Only the MTP layer can launch a fragment expert entry: the
                // NVIDIA shape records them, the local-lab shape runs the
                // main layers' NVFP4 entries instead.
                let names = launched(&host, before);
                assert_eq!(
                    names.iter().any(|name| fragments.contains(name)),
                    experts != WeightFormat::Nvfp4,
                    "{label}"
                );
                instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
                // Draft two and keep only the bonus row.
                let sink = execute(
                    &host,
                    &mut instance,
                    &batch(vec![drafted(10, 1, 9, truth(9), 2)]),
                );
                assert_eq!(sink.draft_tokens(), [(10, truths(10..12))], "{label}");
                instance.commit_accepted(&kept(&[(10, 1)])).unwrap();
                // A plain decode, then a drafted step from it.
                execute(
                    &host,
                    &mut instance,
                    &batch(vec![plain(10, 1, 10, &[truth(10)])]),
                );
                instance.commit().unwrap();
                let sink = execute(
                    &host,
                    &mut instance,
                    &batch(vec![drafted(10, 1, 11, truth(11), 1)]),
                );
                assert_eq!(sink.draft_tokens(), [(10, truths(12..13))], "{label}");
                instance.commit_accepted(&kept(&[(10, 2)])).unwrap();
                let mut sequences = BTreeMap::new();
                sequences.insert(1, truths(0..32));
                assert_drafter_rows(&host, &sequences);
                assert_eq!(
                    committed_state(&host, &instance, 1)[0],
                    truths(0..13),
                    "{label}"
                );
                // A cancelled drafted step leaves the committed state.
                execute(
                    &host,
                    &mut instance,
                    &batch(vec![drafted(10, 1, 13, truth(13), 3)]),
                );
                instance.cancel().unwrap();
                assert_eq!(
                    committed_state(&host, &instance, 1)[0],
                    truths(0..13),
                    "{label}"
                );
                instance.retire_request(10).unwrap();
                instance.destroy().unwrap();
                assert_eq!(host.released.get(), host.acquired.get(), "{label}");
            }
        }
    }
}

/// The local-lab export shape with its F32 norm weights: MXFP8 dense,
/// NVFP4 experts and lookup, the MTP layer recorded on the main layers'
/// NVFP4 entries, and F32 main and PLE norms, with the MTP norms BF16 as
/// local-lab stores them or with F32 MTP hyper-connection norms. The
/// drafter binds its projections through the recorded precision and its
/// norms by their stored dtypes: the MTP layer's HC norm row reads BF16 or
/// F32 accordingly, drafts follow the model and its experts run NVFP4.
#[test]
fn local_lab_shaped_drafters_bind_recorded_projections_and_stored_norms() {
    let mtp_norms = [
        "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
        "mtp.hyper_connection_mixer.hc_norm.weight",
    ];
    for shape in [Shape::TP4, Shape::TP2] {
        let fragments = fragment_experts(shape);
        for f32_mtp in [false, true] {
            let label = format!("TP{} F32 MTP norms {f32_mtp}", shape.ranks());
            let host = mixed_drafter_host(
                shape,
                component(shape, true, Some(WeightFormat::Nvfp4)),
                false,
            );
            let mut norms = super::norms::local_lab_norms(shape);
            if f32_mtp {
                norms.extend(mtp_norms.iter().map(|n| n.to_string()));
            }
            *host.f32_norms.borrow_mut() = norms;
            let names = drafted_step_launches(&host);
            // The target's norms are all F32; only the drafter's BF16 MTP
            // hyper-connection norm row takes the BF16 entry.
            assert!(
                names.iter().any(|n| n == "chop_row_qwen38_hc_norm_f32_v1"),
                "{label}"
            );
            assert_eq!(
                names.iter().any(|n| n == crate::sm120::entries::HC_NORM),
                !f32_mtp,
                "{label}"
            );
            assert!(!names.iter().any(|n| fragments.contains(n)), "{label}");
            assert!(
                names
                    .iter()
                    .any(|n| n.starts_with("chop_qwen38_precision_nvfp4_gate_up")
                        || n.starts_with("chop_proj_nvfp4_grouped_n640_k2560_")),
                "{label}"
            );
        }
    }
}

/// The drafted step's launches of `host`'s instance after its prompt.
fn drafted_step_launches(host: &Rc<FakeHost>) -> Vec<String> {
    let mut instance = create_mixed_drafter(host);
    execute(
        host,
        &mut instance,
        &batch(vec![plain(10, 1, 0, &truths(0..5))]),
    );
    instance.commit().unwrap();
    let before = host.launches.borrow().len();
    let sink = execute(
        host,
        &mut instance,
        &batch(vec![drafted(10, 1, 5, truth(5), 3)]),
    );
    assert_eq!(sink.draft_tokens(), [(10, truths(6..9))]);
    let names = launched(host, before);
    instance.commit_accepted(&kept(&[(10, 4)])).unwrap();
    instance.retire_request(10).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.released.get(), host.acquired.get());
    names
}

/// Recorded FP8 block MTP experts and BF16 MTP dense ports launch exactly
/// what the fixed drafter program of an FP8 package launches: a component
/// of the registered compositions with the MTP layer recorded drafts and
/// verifies with the same launches as one without.
#[test]
fn recorded_registered_mtp_layers_launch_the_fixed_drafter_program() {
    for shape in [Shape::TP4, Shape::TP2] {
        let fixed = drafted_step_launches(&drafter_host_at(3, shape));
        let inventory = crate::precision::inventory().unwrap();
        let mut document = document_bound(shape, false, false, false, &inventory);
        let unrecorded = drafted_step_launches(&mixed_drafter_host(shape, document.clone(), false));
        document.operations.extend(drafter_records(
            shape,
            WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32),
            &inventory,
        ));
        let recorded = drafted_step_launches(&mixed_drafter_host(shape, document, false));
        assert_eq!(unrecorded, fixed, "TP{}", shape.ranks());
        assert_eq!(recorded, fixed, "TP{}", shape.ranks());
    }
}

/// A drafter is refused at create, before any device resource, for a mixed
/// component without MTP records (asking to regenerate it), one recording
/// part of the MTP layer, one with an altered MTP binding, and MTP dense
/// ports in a format the MTP layer does not compose. The same components
/// still serve without the drafter where they bind.
#[test]
fn unrecorded_partial_and_altered_mtp_layers_refuse_the_drafter() {
    let fp8 = WeightFormat::Fp8Block128x128(kernel_core::contract::ScaleMode::ExactF32);
    for shape in [Shape::TP4, Shape::TP2] {
        let recorded = component(shape, true, Some(WeightFormat::Nvfp4));
        let ids = model_qwen38_contract::prepared::drafter_operations(shape).unwrap();
        let mut partial = recorded.clone();
        partial.operations.retain(|s| s.source.id != ids[0]);
        let mut altered = recorded.clone();
        let last = altered.operations.len() - 1;
        altered.operations[last].bindings[0]
            .implementation
            .dependency = "0".repeat(64);
        // MXFP8 MTP attention ports: published compositions exist for the
        // main layers' geometry, but the MTP layer composes BF16 alone.
        let mut quantized = component(shape, true, None);
        let inventory = crate::precision::inventory().unwrap();
        let usage = ports::uses().unwrap()[0];
        for (role, id) in ports::ports(shape).unwrap().layer(shape.mtp_layer()) {
            quantized.operations.push(selection(
                id.clone(),
                published(
                    shape,
                    role.projection_id(shape, None),
                    WeightFormat::Mxfp8,
                    usage,
                ),
                &inventory,
            ));
        }
        quantized.operations.extend(
            drafter_records(shape, fp8, &inventory)
                .into_iter()
                .rev()
                .take(2)
                .rev(),
        );
        for (label, document, serves_text, logged) in [
            (
                "unrecorded",
                component(shape, true, None),
                true,
                Some("regenerate"),
            ),
            ("partial", partial, false, Some("part of the MTP layer")),
            ("altered", altered, false, None),
            (
                "quantized MTP dense",
                quantized,
                false,
                Some("composes no Mxfp8 MTP operation"),
            ),
        ] {
            let label = format!("{label} at TP{}", shape.ranks());
            let host = mixed_drafter_host(shape, document.clone(), false);
            let config = configuration_at(
                shape,
                host.layout(),
                &[model_qwen38_contract::prepared::COMPONENT_ROLE
                    .as_bytes()
                    .to_vec()],
            );
            assert!(
                api()
                    .create(HostServicesTableV1::new(Rc::clone(&host)), &config)
                    .is_err(),
                "{label}"
            );
            assert!(host.symbol_names.borrow().is_empty(), "{label}");
            assert!(host.extents.borrow().is_empty(), "{label}");
            assert_eq!(host.released.get(), host.acquired.get(), "{label}");
            if let Some(logged) = logged {
                assert!(
                    host.logs.borrow().iter().any(|line| line.contains(logged)),
                    "{label}: {:?}",
                    host.logs.borrow()
                );
            }
            // Without the drafter only the text operations decide.
            let text = Rc::new(FakeHost::at(shape));
            text.ngram_host.set(host.ngram_host.get());
            *text.precision.borrow_mut() = Some(document);
            let config = configuration_at(
                shape,
                text.layout(),
                &[model_qwen38_contract::prepared::COMPONENT_ROLE
                    .as_bytes()
                    .to_vec()],
            );
            let created = api().create(HostServicesTableV1::new(Rc::clone(&text)), &config);
            assert_eq!(created.is_ok(), serves_text, "{label} without the drafter");
            if let Ok(mut instance) = created {
                instance.destroy().unwrap();
            }
            assert_eq!(text.released.get(), text.acquired.get(), "{label}");
        }
    }
}

/// Startup warmup waves of 1, 2, 4, ... sequences up to the package's
/// served count, as `serve` runs them, for both export shapes at TP2 and
/// TP4: each wave prefills and decodes every sequence (the drafter catches
/// up after the decode, at the decode wave's bucket), then verifies three
/// drafts for as many sequences as draft in one step within the verify
/// rows `serve` allows. Every recorded MTP projection and every fixed
/// drafter GEMM launches in each bucket.
#[test]
fn recorded_mtp_drafters_run_every_warmup_wave() {
    use model_qwen38_contract::package::{Encoders, maximum_sequences};
    for shape in [Shape::TP4, Shape::TP2] {
        let served = maximum_sequences(shape, true, Encoders::VISION);
        for (export, mx, experts) in EXPORTS {
            let host = mixed_drafter_host(shape, component(shape, mx, Some(experts)), false);
            host.served.set(Some(served));
            // `serve` narrows a step's verify rows to 32 when it serves at
            // most 32 sequences.
            let verify_rows = if served <= 32 { 32 } else { MAX_VERIFY_TOKENS };
            host.speculation.set((3, verify_rows));
            let mut instance = create_mixed_drafter(&host);
            let prompt = truths(0..5);
            let mut request = 100;
            for wave in (0..=served.ilog2()).map(|power| 1_u64 << power) {
                let label = format!("{export} TP{} wave {wave}", shape.ranks());
                let ids: Vec<u64> = (request..request + wave).collect();
                request += wave;
                // Four KV pages per slot, so every served slot fits the
                // fake pool (the shared helpers give each slot 64).
                let paged = |(item, _, slot): (BatchItemV1, BatchAppendV1, u64),
                             previous: u64,
                             count: u64| {
                    let end = previous + count;
                    let pages = (0..end.div_ceil(PAGE_TOKENS))
                        .map(|page| slot * 4 + page)
                        .collect();
                    (
                        item,
                        BatchAppendV1::new(previous, count, end, pages, None),
                        slot,
                    )
                };
                let step = |ids: &[u64],
                            item: &dyn Fn(u64, u64) -> (BatchItemV1, BatchAppendV1, u64),
                            previous: u64,
                            count: u64| {
                    batch(
                        ids.iter()
                            .zip(0..)
                            .map(|(&id, slot)| paged(item(id, slot), previous, count))
                            .collect(),
                    )
                };
                execute(
                    &host,
                    &mut instance,
                    &step(&ids, &|id, slot| plain(id, slot, 0, &prompt), 0, 5),
                );
                instance.commit().unwrap();
                execute(
                    &host,
                    &mut instance,
                    &step(&ids, &|id, slot| plain(id, slot, 5, &[truth(5)]), 5, 1),
                );
                instance.commit().unwrap();
                let drafting = ids
                    .len()
                    .min(model_qwen38_contract::contract::DRAFT_ITEMS as usize)
                    .min(verify_rows as usize / 4);
                let sink = execute(
                    &host,
                    &mut instance,
                    &step(
                        &ids[..drafting],
                        &|id, slot| drafted(id, slot, 6, truth(6), 3),
                        6,
                        4,
                    ),
                );
                assert_eq!(sink.draft_tokens().len(), drafting, "{label}");
                let accepted: Vec<_> = ids[..drafting].iter().map(|&id| (id, 4)).collect();
                instance.commit_accepted(&kept(&accepted)).unwrap();
                for id in ids {
                    instance.retire_request(id).unwrap();
                }
            }
            instance.destroy().unwrap();
            assert_eq!(host.released.get(), host.acquired.get());
        }
    }
}
