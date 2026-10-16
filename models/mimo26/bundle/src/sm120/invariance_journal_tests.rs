//! The launch journal against the kernels' declared invariance
//! (`kernels_sm120::invariance::Program`): each launch's symbol resolves to
//! its catalog entry (or a bundle kernel's declaration), and the entry must
//! declare what MiMo promises of that phase. Prefill entries keep a row's
//! bits over chunks at the manifest's granule; exact verify steps
//! (`EXACT_VERIFY`) run only entries one-row exact at the rows each launch
//! processes, as its plan records them (`PreparedLaunch::rows`) and
//! checked against a one-row decode step of one item. MiMo's package
//! declares no prefill granule and no anchors, so chunks may start at any
//! row and no prefill entry needs anchor support.

use super::*;
use kernels_sm120::invariance::{Launched, Program};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

const PROGRAM: Program = Program {
    prefill_granule: 1,
    exact_verify: model_mimo26_contract::sm120::EXACT_VERIFY,
    anchor_items: 0,
};

/// A journal step's phase.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    Prefill,
    Decode { items: u32 },
    Verify { rows: u32 },
}

fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_cache(crate::BUNDLE_MIMO26_EXTENDED_API_V1).expect("bundle API")
}

/// The pages of rows `[0, end)` of a request whose pages start at `first`.
fn pages(first: u64, end: u64) -> Vec<u64> {
    (first..first + end.div_ceil(PAGE_TOKENS)).collect()
}

/// A request's prefill item: `rows` prompt rows from position 0.
fn prefill(request: u64, first_page: u64, rows: u64) -> (BatchItemV1, BatchAppendV1) {
    (
        BatchItemV1::new(
            request,
            false,
            rows as u32,
            true,
            (0..rows as u32).collect(),
        ),
        BatchAppendV1::new(0, rows, rows, pages(first_page, rows), None),
    )
}

fn batch(items: Vec<(BatchItemV1, BatchAppendV1)>) -> BatchRequestV1 {
    let (items, appends) = items.into_iter().unzip();
    BatchRequestV1::new(BatchWireVersionV1::V3, items, appends)
}

/// Every launch of eager steps at `tp`, unpipelined or pipelined, with its
/// phase: prefills of 3 and 300 rows together, 20 rows (the exact comm
/// entries) and 1,100 rows (the phased pair), a decode of two items, a
/// one-row decode of one item (the AR baseline), and exact verify steps of
/// 2, 4 and 8 rows. Logits rows take the F32 head, or with `f32_logits`
/// off the BF16 head (`CHOPINFER_MIMO_F32_LOGITS=0`), whose verify rows
/// the BF16 row exchange moves.
fn journal(tp: Tp, pipelined: bool, f32_logits: bool) -> Vec<(Launched, String)> {
    crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
    crate::pipeline::take_recorded();
    let host = Rc::new(FakeHost::at(tp));
    host.speculation.set(Some((7, 32)));
    let mut instance = api()
        .create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &configuration_for_at(tp, host.kv(), false),
        )
        .expect("create");
    crate::instance::force_f32_logits(instance.handle(), f32_logits);
    crate::instance::force_overlap(
        instance.handle(),
        pipelined.then_some(crate::overlap::Config {
            sms: 188,
            reserved_sms: 16,
            comm_ctas: Some(24),
            min_tokens: 1024,
        }),
    );
    let mut steps = Vec::new();
    let mut step = |phase: Step, request: BatchRequestV1| {
        let before = host.launch_records.borrow().len();
        crate::instance::PLANNED.with(|planned| planned.borrow_mut().clear());
        instance
            .execute(&request.encode().unwrap(), &mut ViewSink::default())
            .unwrap_or_else(|error| panic!("{phase:?}: {error:?}"));
        let launched = host.launch_records.borrow()[before..]
            .iter()
            .map(|(_, symbol, ..)| symbol.clone())
            .collect::<Vec<_>>();
        let planned = crate::instance::PLANNED.with(|planned| planned.take());
        steps.push((phase, journaled(phase, launched, planned)));
        match phase {
            Step::Verify { .. } => instance.commit_accepted(&[kept(50, 1)]).unwrap(),
            _ => instance.commit().unwrap(),
        }
    };
    step(
        Step::Prefill,
        batch(vec![prefill(50, 0, 3), prefill(51, 10, 300)]),
    );
    for (request, first_page, rows) in [(52, 20, 20), (53, 30, 1100)] {
        step(
            Step::Prefill,
            batch(vec![prefill(request, first_page, rows)]),
        );
    }
    step(
        Step::Decode { items: 2 },
        batch(vec![
            (
                BatchItemV1::new(50, true, 1, true, vec![7]),
                BatchAppendV1::new(3, 1, 4, pages(0, 4), None),
            ),
            (
                BatchItemV1::new(51, true, 1, true, vec![8]),
                BatchAppendV1::new(300, 1, 301, pages(10, 301), None),
            ),
        ]),
    );
    step(
        Step::Decode { items: 1 },
        batch(vec![(
            BatchItemV1::new(50, true, 1, true, vec![9]),
            BatchAppendV1::new(4, 1, 5, pages(0, 5), None),
        )]),
    );
    let mut at = 5;
    for rows in [2_u64, 4, 8] {
        step(
            Step::Verify { rows: rows as u32 },
            batch(vec![(
                BatchItemV1::verify(50, rows as u32, false, (10..10 + rows as u32).collect()),
                BatchAppendV1::new(at, rows, at + rows, pages(0, at + rows), None),
            )]),
        );
        at += 1;
    }
    instance.destroy().expect("destroy");
    crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
    super::launch_journal::export_rows(
        &format!(
            "invariance-tp{}-pipelined-{pipelined}-f32-{f32_logits}",
            tp.ranks()
        ),
        &[&host],
    );
    launches(&steps, tp)
}

/// One step's launches, each with the rows its plan processes. The rows
/// come from the stages' lowered launches (`PreparedLaunch::rows`), which
/// must be the launches the host ran.
fn journaled(
    phase: Step,
    launched: Vec<String>,
    planned: Vec<(usize, String, u32)>,
) -> Vec<(String, u32)> {
    let mut ran = launched.clone();
    let mut lowered = planned
        .iter()
        .map(|(_, s, _)| s.clone())
        .collect::<Vec<_>>();
    ran.sort();
    lowered.sort();
    assert_eq!(ran, lowered, "{phase:?}: the host ran the lowered launches");
    planned
        .iter()
        .map(|(_, s, rows)| (s.clone(), *rows))
        .collect()
}

/// Expected verify operations per AR row. Bucketed projections and MoE
/// keep their operations; attention unfolds decode's RoPE and KV write
/// and selects exact row entries (`attention_plan::plan`, `selection::select`).
fn verify_baseline(baseline: &[(&str, u32)], rows: u32, tp: Tp) -> Vec<(String, u32)> {
    let (plain, combine) = crate::sm120::launch::comm::push_symbols(tp.ranks(), 1);
    let (verify_plain, verify_combine) =
        crate::sm120::launch::comm::exact_symbols(tp.ranks(), rows)
            .expect("exact verify collectives");
    let mut comm = std::collections::BTreeMap::new();
    for (from, to) in [(plain, verify_plain), (combine, verify_combine)] {
        comm.insert(from.to_owned(), to.to_owned());
    }
    let mut expected = Vec::new();
    for &(symbol, count) in baseline {
        let mut replacements = vec![symbol.to_owned()];
        if let Some(replacement) = comm.get(symbol) {
            // Verify's one-row norm order, for every supported topology.
            replacements[0] = replacement.clone();
        } else if symbol.starts_with("chop_gemm_mimo") {
            replacements[0] = symbol.replace("_m1_v1", &format!("_m{rows}_v1"));
        } else if symbol.starts_with("chop_moe_mimo") {
            replacements[0] = symbol.replace("_t1_v1", &format!("_t{rows}_v1"));
            // Gate/up needs the exact K order at 2, 4 and 8 rows; down
            // already has it at every bucket (`kernels::moe::EXACT_BUCKETS`).
            if symbol.contains("_gateup_decode_mxfp4_a16_") {
                replacements[0] = replacements[0].replace("_decode_", "_exact_");
            }
        } else if let Some(name) = symbol.strip_prefix("chop_attn_") {
            if let Some(tag) = name.strip_suffix("_ga_bf16_decode_rope_table_v1") {
                let entry = if crate::selection::exact_attention_grouped() {
                    "rows_shared"
                } else {
                    "rows"
                };
                replacements = vec![
                    format!("chop_attn_{tag}_ga_bf16_decode_{entry}_v1"),
                    format!("chop_row_rope_{tag}_ga_bf16_table_p64_pos32_v1"),
                ];
            } else if let Some(tag) = name.strip_suffix("_swa_decode_rope_table_v1") {
                replacements = vec![
                    format!("chop_attn_{tag}_swa_decode_rows_v1"),
                    format!("chop_row_rope_{tag}_swa_prefill_table_p32_pos32_v1"),
                ];
            } else if name.ends_with("_ga_bf16_decode_merge_v1") {
                replacements[0] = symbol.replace("_merge_v1", "_rows_merge_v1");
            }
        }
        expected.extend(
            replacements
                .into_iter()
                .map(|replacement| (replacement, count)),
        );
    }
    expected
}

/// Each step's launches with their phases; verify launches carry the rows
/// their plans process, checked against the one-row decode step of one
/// item (`Program::verify_launches`).
fn launches(steps: &[(Step, Vec<(String, u32)>)], tp: Tp) -> Vec<(Launched, String)> {
    fn records(launches: &[(String, u32)]) -> Vec<(&str, u32)> {
        launches
            .iter()
            .map(|(s, rows)| (s.as_str(), *rows))
            .collect::<Vec<_>>()
    }
    let baseline = steps
        .iter()
        .find(|(phase, _)| *phase == Step::Decode { items: 1 })
        .map(|(_, launches)| records(launches))
        .expect("a one-row decode step");
    let mut launches = Vec::new();
    for (phase, step) in steps {
        match *phase {
            Step::Prefill => {
                launches.extend(step.iter().map(|(s, _)| (Launched::Prefill, s.clone())))
            }
            Step::Decode { .. } => {
                launches.extend(step.iter().map(|(s, _)| (Launched::Decode, s.clone())))
            }
            Step::Verify { rows } => launches.extend(
                Program::verify_launches(
                    rows,
                    1,
                    &records(step),
                    &records(&verify_baseline(&baseline, rows, tp)),
                )
                .unwrap_or_else(|errors| panic!("{phase:?}: {}", errors.join("\n")))
                .into_iter()
                .map(|(launched, s)| (launched, s.to_owned())),
            ),
        }
    }
    launches
}

/// Every rank count's launches, unpipelined and pipelined, with the F32
/// logits head and with the BF16 one, satisfy the
/// declarations of the entries they launch.
#[test]
fn launches_keep_the_declared_invariance() {
    let extra = crate::sm120::declared_kernels();
    let mut violations = Vec::new();
    // Every rank count, pipelined or not, then the BF16 head.
    let runs = Tp::SUPPORTED
        .into_iter()
        .flat_map(|tp| [(tp, false, true), (tp, true, true)])
        .chain([(Tp::TP4, false, false)]);
    for (tp, pipelined, f32_logits) in runs {
        crate::sm120::overlap::ADJUSTED.with(|adjusted| adjusted.set([0; 4]));
        let launches = journal(tp, pipelined, f32_logits);
        assert!(launches.iter().any(|(l, _)| *l == Launched::Prefill));
        // Each pipelined launch's grid moved by its op's facts
        // (`sm120::overlap::count`): persistent grids shrank and phased
        // pairs ran among them.
        let [all, shrunk, phased, _] = crate::sm120::overlap::ADJUSTED.with(|c| c.get());
        if pipelined {
            assert!(
                all > 0 && shrunk > 0 && phased > 0,
                "{tp:?}: {all} {shrunk} {phased}"
            );
        } else {
            assert_eq!(all, 0);
        }
        for violation in PROGRAM.check_journal(
            model_mimo26_contract::sm120::catalogs(),
            launches
                .iter()
                .map(|(launched, symbol)| (*launched, symbol.as_str())),
            &extra,
        ) {
            violations.push(format!(
                "{tp:?} pipelined={pipelined} f32_logits={f32_logits}: \
                 {violation}"
            ));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

/// A 66-token prefill under a 2-token overlap minimum splits 64/2: its
/// first half's all-reduces run the phased pair and its second half's the
/// one-shot entries, each launch's grid moved by its op's facts
/// (`sm120::overlap::count`).
#[test]
fn a_short_pipelined_prefill_runs_phased_and_one_shot_halves() {
    assert_eq!(crate::overlap::split_point(66), 64);
    for tp in Tp::SUPPORTED {
        crate::sm120::overlap::ADJUSTED.with(|adjusted| adjusted.set([0; 4]));
        let host = Rc::new(FakeHost::at(tp));
        let mut instance = api()
            .create(
                HostServicesTableV1::new(Rc::clone(&host)),
                &configuration_for_at(tp, host.kv(), false),
            )
            .expect("create");
        crate::instance::force_overlap(
            instance.handle(),
            Some(crate::overlap::Config {
                sms: 188,
                reserved_sms: 16,
                comm_ctas: Some(24),
                min_tokens: 2,
            }),
        );
        let before = host.launch_records.borrow().len();
        instance
            .execute(
                &batch(vec![prefill(50, 0, 66)]).encode().unwrap(),
                &mut ViewSink::default(),
            )
            .unwrap();
        instance.commit().unwrap();
        let launched = host.launch_records.borrow()[before..]
            .iter()
            .map(|(_, symbol, ..)| symbol.clone())
            .collect::<Vec<_>>();
        let [all, _, phased, comm] = crate::sm120::overlap::ADJUSTED.with(|c| c.get());
        // Both halves reduce every layer on the comm lane: the phased pair
        // for the first, the one-shot entries for the second.
        assert!(
            all > 0 && phased > 0 && comm > phased,
            "{tp:?}: {all} {phased} {comm}"
        );
        for symbol in crate::sm120::comm::PHASED_SYMBOLS {
            assert!(launched.iter().any(|launched| launched == symbol), "{tp:?}");
        }
    }
}
