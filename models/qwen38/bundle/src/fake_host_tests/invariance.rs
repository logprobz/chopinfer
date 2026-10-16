//! The launch journal against the kernels' declared invariance
//! (`sm120::Program`, the kernels' declarations): each prefill launch's symbol
//! resolves to its catalog entry (or an owned kernel's declaration), which
//! must keep a row's bits over chunks at the manifest's granule
//! (`PREFILL_GRANULE`) and, since steps may keep `ANCHOR_ITEMS` prefix
//! anchors, support anchors where it writes per-slot state. Verify steps in
//! the exact domain (`EXACT_VERIFY`: the MTP drafter's K3 steps of one
//! request) run only entries one-row exact at the rows each launch
//! processes, as its plan records them (`PreparedLaunch::rows`) and
//! checked against a one-row decode step of one item. The
//! steps here are host-drafted: the target launches are those of a drafted
//! step, and the drafter's own launches only choose drafts.

use super::sm120::{Declared, Launched, OpKind, Program};
use super::speculation::{batch, plain};
use super::*;
use kernel_core::precision::ExecutionPhase;
use kernel_core::precision::execution::PrecisionDocument;

/// A journal step's phase.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    Prefill,
    Decode { items: u32 },
    Verify { rows: u32 },
}

pub(super) const PROGRAM: Program = Program {
    prefill_granule: crate::contract::PREFILL_GRANULE,
    exact_verify: crate::contract::EXACT_VERIFY,
    anchor_items: crate::contract::ANCHOR_ITEMS,
};

/// Expected verify operations per AR row. Projection and MoE buckets
/// follow `lower::RankStage`; n-gram and PLE convolution replace their
/// single-row decode entries with the owned multirow verify entries.
/// A precision package's projections verify through their operation's
/// stages bound at the verify bucket (`verify_stages`).
fn verify_baseline(
    baseline: &[(&str, u32)],
    rows: u32,
    shape: Shape,
    document: Option<&PrecisionDocument>,
) -> Vec<(String, u32)> {
    let bound = document
        .map(|document| verify_stages(document, rows))
        .unwrap_or_default();
    baseline
        .iter()
        .map(|&(symbol, count)| {
            let replacement = if let Some(verify) = bound.get(symbol) {
                verify.clone()
            } else if symbol == super::decode_comm(shape.ranks(), 1) {
                super::decode_comm(shape.ranks(), rows).to_owned()
            } else if symbol.starts_with("chop_gemm_qwen38") {
                symbol
                    .replace("_c1_v1", &format!("_c{rows}_v1"))
                    .replace("_m1_v1", &format!("_m{rows}_v1"))
            } else if symbol.starts_with("chop_moe_qwen38") {
                symbol.replace("_t1_v1", &format!("_t{rows}_v1"))
            } else if symbol == "chop_row_qwen38_ple_conv_items_decode_v1" {
                model_qwen38_contract::sm120::owned::PLE_CONV_VERIFY.to_owned()
            } else if symbol == crate::launch::per_rank("chop_row_qwen38_ngram_decode_v1", shape) {
                crate::launch::per_rank(model_qwen38_contract::sm120::owned::NGRAM_VERIFY, shape)
            } else if symbol == ngram_hash(shape, ExecutionPhase::Decode) {
                ngram_hash(shape, ExecutionPhase::Verify)
            } else {
                symbol.to_owned()
            };
            (replacement, count)
        })
        .collect()
}

/// Each stage a precision package's operations bind at the one-row decode
/// use, mapped to the stage at the same place of the same operation's
/// binding at the verify use of `rows`: the entry whose rows must have the
/// one-row decode entry's bits.
fn verify_stages(document: &PrecisionDocument, rows: u32) -> BTreeMap<String, String> {
    let mut stages = BTreeMap::new();
    for selection in &document.operations {
        let at = |phase, capacity| {
            selection
                .bindings
                .iter()
                .find(|b| b.usage.phase == phase && b.usage.capacity == capacity)
                .map(|b| &b.implementation.stages)
        };
        let (Some(one), Some(verify)) = (
            at(ExecutionPhase::Decode, 1),
            at(ExecutionPhase::Verify, rows),
        ) else {
            continue;
        };
        assert_eq!(one.len(), verify.len(), "{}", selection.source.id);
        for (one, verify) in one.iter().zip(verify) {
            let held = stages.entry(one.clone()).or_insert_with(|| verify.clone());
            assert_eq!(held, verify, "{one} verifies through two entries");
        }
    }
    stages
}

/// The registered n-gram hash entry of `phase` at `shape`.
fn ngram_hash(shape: Shape, phase: ExecutionPhase) -> String {
    crate::launch::tagged(
        &model_qwen38_contract::sm120::generated::ngram_hash_symbol(shape, phase),
        shape,
    )
}

/// Every launch of eager steps at `shape`, with its phase: a prefill of 3
/// and 2,050 rows together, one of 20 rows (the one-shot all-reduce), one of
/// 150 rows that keeps a prefix anchor, a decode of two items, a one-row
/// decode of one item (the AR baseline), and verify steps of 4 and 2 rows;
/// with `precision`, of an instance serving that precision document.
fn journal(shape: Shape, precision: Option<PrecisionDocument>) -> Vec<(Launched, String)> {
    crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
    let host = Rc::new(FakeHost::at(shape));
    let document = precision.clone();
    *host.precision.borrow_mut() = precision;
    host.ngram_host.set(ngram_in_host_memory(shape));
    host.speculation.set((3, MAX_VERIFY_TOKENS));
    crate::pipeline::take_recorded();
    let mut instance = crate::instance::with_row_owners(false, || create(&host));
    host.model(instance.handle());
    let mut steps = Vec::new();
    let mut step = |phase: Step, request: BatchRequestV1| {
        let before = host.launches.borrow().len();
        crate::instance::PLANNED.with(|planned| planned.borrow_mut().clear());
        run(&host, &mut instance, &format!("{phase:?}"), request);
        let launched = host.launches.borrow()[before..]
            .iter()
            .map(|(_, symbol, _)| symbol.clone())
            .collect::<Vec<_>>();
        let planned = crate::instance::PLANNED.with(|planned| planned.take());
        steps.push((phase, journaled(phase, launched, planned)));
        match phase {
            Step::Verify { .. } => instance
                .commit_accepted(&speculation::kept(&[(10, 2)]))
                .unwrap(),
            _ => instance.commit().unwrap(),
        }
    };
    let long = long_prompt();
    let short = (0..20).map(|token| token + 5).collect::<Vec<_>>();
    let anchored = (0..150).map(|token| token + 9).collect::<Vec<_>>();
    step(
        Step::Prefill,
        batch(vec![plain(10, 0, 0, &PROMPT), plain(11, 1, 0, &long)]),
    );
    step(Step::Prefill, batch(vec![plain(12, 2, 0, &short)]));
    let (item, append, slot) = plain(13, 3, 0, &anchored);
    step(
        Step::Prefill,
        batch(vec![(item.with_anchor(true), append, slot)]),
    );
    step(
        Step::Decode { items: 2 },
        batch(vec![plain(10, 0, 3, &[4]), plain(11, 1, 2050, &[5])]),
    );
    step(
        Step::Decode { items: 1 },
        batch(vec![plain(10, 0, 4, &[5])]),
    );
    let verify = |at: u64, tokens: Vec<u32>| {
        let rows = tokens.len() as u64;
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(10, rows as u32, false, tokens)],
            vec![BatchAppendV1::new(at, rows, at + rows, vec![0], None)],
        )
        .with_state_slots(vec![0])
    };
    step(Step::Verify { rows: 4 }, verify(5, vec![6, 7, 8, 9]));
    step(Step::Verify { rows: 2 }, verify(7, vec![10, 11]));
    instance.destroy().unwrap();
    crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
    if document.is_none() {
        super::export_rows(&format!("invariance-{}", shape.tag()), &[&host]);
    }
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
    for (phase, step) in &steps {
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
                    &records(&verify_baseline(&baseline, rows, shape, document.as_ref())),
                )
                .unwrap_or_else(|errors| panic!("{phase:?}: {}", errors.join("\n")))
                .into_iter()
                .map(|(launched, s)| (launched, s.to_owned())),
            ),
        }
    }
    launches
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

/// Every served topology's launches satisfy the declarations of the
/// entries they launch, and the anchored prefill launches the PLE anchor;
/// so do those of packed packages of both export shapes (NVFP4 experts;
/// also MXFP8 dense ports and the NVFP4 lookup), whose generated
/// projections are checked against what their prepared ops guarantee and
/// verify through their operations' verify-bucket entries.
#[test]
fn launches_keep_the_declared_invariance() {
    let owned = model_qwen38_contract::owned::declared();
    let mut violations = Vec::new();
    // Every served rank count runs on one device per rank.
    for shape in [Shape::TP2, Shape::TP4] {
        for mx in [None, Some(false), Some(true)] {
            let document = mx.map(|mx| super::precision::document_for(shape, true, mx, false));
            // What the prepared generated projections guarantee at each use.
            let generated = document.as_ref().map_or_else(Vec::new, |document| {
                let bytes = kernel_core::precision::component::encode(document).unwrap();
                crate::precision::Prepared::decode(
                    &bytes,
                    shape,
                    crate::instance::target(shape, false).as_ref(),
                )
                .unwrap()
                .guarantees
            });
            assert_eq!(generated.is_empty(), mx.is_none());
            let launches = journal(shape, document);
            assert!(
                launches
                    .iter()
                    .any(|(_, symbol)| symbol == model_qwen38_contract::sm120::owned::PLE_ANCHOR),
                "{shape:?}: the anchored prefill launches the PLE anchor"
            );
            // Each launch is checked against the declarations of its step's
            // use: the catalogs', the owned kernels' and, for a generated
            // projection, its op's at that use (the check reads only the
            // claims).
            let at = |launched: Launched| {
                let (phase, capacity) = match launched {
                    Launched::Prefill => (ExecutionPhase::Prefill, PREFILL_TOKENS),
                    Launched::Verify { rows, .. } => (ExecutionPhase::Verify, rows),
                    Launched::Decode => (ExecutionPhase::Decode, 0),
                };
                let mut declared = owned.clone();
                declared.extend(
                    generated
                        .iter()
                        .filter(|(usage, _, _)| usage.phase == phase && usage.capacity == capacity)
                        .map(|(_, symbol, invariance)| Declared {
                            symbol: symbol.clone(),
                            profile: None,
                            op: OpKind::Gemm,
                            invariance: *invariance,
                        }),
                );
                declared
            };
            let mut checked = [false; 2];
            let mut steps = Vec::new();
            for (launched, _) in &launches {
                if *launched != Launched::Decode && !steps.contains(launched) {
                    steps.push(*launched);
                }
            }
            for step in steps {
                let declared = at(step);
                let exact = usize::from(step != Launched::Prefill);
                checked[exact] |= launches.iter().any(|(launched, symbol)| {
                    *launched == step && generated.iter().any(|(_, s, _)| s == symbol)
                });
                for violation in PROGRAM.check_journal(
                    model_qwen38_contract::sm120::catalogs(),
                    launches
                        .iter()
                        .filter(|(launched, _)| *launched == step)
                        .map(|(launched, symbol)| (*launched, symbol.as_str())),
                    &declared,
                ) {
                    violations.push(format!("{:?} {mx:?}: {violation}", shape.ranks()));
                }
            }
            // The packed journals launch generated projections in prefill
            // and in exact verify steps, so the check is not vacuous.
            assert_eq!(checked, [mx.is_some(); 2], "{shape:?} {mx:?}");
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}

/// The DCP2 arm: every launch of eager steps of a TP4 instance at
/// context-parallel size 2, by rank, with its phase: a prefill of 5 and
/// 2,050 rows (two selection pieces), one of 150 rows that keeps a prefix
/// anchor, a one-row decode of one item (each rank's AR baseline), and
/// host-drafted verify steps of 4 and 2 rows. A verify step's launches are
/// checked against the baseline with the decode partials replaced by the
/// verify partials, which verify graphs run (the same instantiation).
fn dcp2_journal() -> Vec<Vec<(Launched, String)>> {
    use super::context_parallel::{create_at, item, pages};
    let shape = Shape::TP4;
    let size = 2;
    crate::instance::FORCE_EAGER.with(|eager| eager.set(Some(true)));
    let host = Rc::new(FakeHost::at(shape));
    host.ngram_host.set(ngram_in_host_memory(shape));
    host.speculation.set((3, MAX_VERIFY_TOKENS));
    crate::pipeline::take_recorded();
    let mut instance = create_at(&host, size);
    let n = u64::from(size);
    let mut steps = Vec::new();
    let mut step = |phase: Step, request: BatchRequestV1| {
        let before = host.launches.borrow().len();
        crate::instance::PLANNED.with(|planned| planned.borrow_mut().clear());
        run(&host, &mut instance, &format!("dcp2 {phase:?}"), request);
        let launched = host.launches.borrow()[before..]
            .iter()
            .map(|(_, symbol, _)| symbol.clone())
            .collect::<Vec<_>>();
        let planned = crate::instance::PLANNED.with(|planned| planned.take());
        let lowered = planned
            .iter()
            .map(|(_, s, rows)| (0, s.clone(), *rows))
            .collect();
        journaled(phase, launched, lowered);
        steps.push((phase, planned));
        match phase {
            Step::Verify { .. } => instance
                .commit_accepted(&speculation::kept(&[(10, 2)]))
                .unwrap(),
            _ => instance.commit().unwrap(),
        }
    };
    let long = long_prompt();
    let anchored = (0..150).map(|token| token + 9).collect::<Vec<_>>();
    step(
        Step::Prefill,
        batch(vec![item(10, 0, 0, &PROMPT, n), item(11, 1, 0, &long, n)]),
    );
    let (anchor, append, slot) = item(13, 3, 0, &anchored, n);
    step(
        Step::Prefill,
        batch(vec![(anchor.with_anchor(true), append, slot)]),
    );
    let at = PROMPT.len() as u64;
    step(
        Step::Decode { items: 1 },
        batch(vec![item(10, 0, at, &[5], n)]),
    );
    let verify = |at: u64, tokens: Vec<u32>| {
        let rows = tokens.len() as u64;
        BatchRequestV1::new(
            BatchWireVersionV1::V3,
            vec![BatchItemV1::verify(10, rows as u32, false, tokens)],
            vec![BatchAppendV1::new(
                at,
                rows,
                at + rows,
                pages(0, at + rows, n),
                None,
            )],
        )
        .with_state_slots(vec![0])
    };
    step(Step::Verify { rows: 4 }, verify(at + 1, vec![6, 7, 8, 9]));
    step(Step::Verify { rows: 2 }, verify(at + 3, vec![10, 11]));
    instance.destroy().unwrap();
    crate::instance::FORCE_EAGER.with(|eager| eager.set(None));
    let partials = model_qwen38_contract::sm120::context_parallel::partial_entries(shape);
    let verify_partials = |symbol: &str| {
        let decode = partials.iter().find(|e| {
            e.phase == kernel_core::contract::AttentionPhase::Decode && e.symbol() == symbol
        })?;
        partials
            .iter()
            .find(|e| e.phase == kernel_core::contract::AttentionPhase::Verify && e.kv == decode.kv)
            .map(|e| e.symbol())
    };
    (0..shape.ranks() as usize)
        .map(|rank| {
            let of_rank = |planned: &[(usize, String, u32)]| {
                planned
                    .iter()
                    .filter(|(r, _, _)| *r == rank)
                    .map(|(_, s, rows)| (s.clone(), *rows))
                    .collect::<Vec<_>>()
            };
            let records = |launches: &[(String, u32)]| {
                launches
                    .iter()
                    .map(|(s, rows)| (s.clone(), *rows))
                    .collect::<Vec<_>>()
            };
            let baseline = steps
                .iter()
                .find(|(phase, _)| *phase == Step::Decode { items: 1 })
                .map(|(_, planned)| of_rank(planned))
                .expect("a one-row decode step");
            let mut lane = Vec::new();
            for (phase, planned) in &steps {
                let launches = of_rank(planned);
                match *phase {
                    Step::Prefill => {
                        lane.extend(launches.into_iter().map(|(s, _)| (Launched::Prefill, s)))
                    }
                    Step::Decode { .. } => {
                        lane.extend(launches.into_iter().map(|(s, _)| (Launched::Decode, s)))
                    }
                    Step::Verify { rows } => {
                        let base = records(&baseline);
                        let base = base
                            .iter()
                            .map(|(s, r)| (s.as_str(), *r))
                            .collect::<Vec<_>>();
                        let expected = verify_baseline(&base, rows, shape, None)
                            .into_iter()
                            .map(|(s, r)| (verify_partials(&s).unwrap_or(s), r))
                            .collect::<Vec<_>>();
                        let expected = expected
                            .iter()
                            .map(|(s, r)| (s.as_str(), *r))
                            .collect::<Vec<_>>();
                        let ran = launches
                            .iter()
                            .map(|(s, r)| (s.as_str(), *r))
                            .collect::<Vec<_>>();
                        lane.extend(
                            Program::verify_launches(rows, 1, &ran, &expected)
                                .unwrap_or_else(|errors| {
                                    panic!("rank {rank} {phase:?}: {}", errors.join("\n"))
                                })
                                .into_iter()
                                .map(|(launched, s)| (launched, s.to_owned())),
                        );
                    }
                }
            }
            lane
        })
        .collect()
}

/// Every rank's launches at context-parallel size 2 satisfy the
/// declarations of the entries they launch, each rank's lane checked
/// alone (a composed selection is followed by its split merge on its
/// rank), and they run the context-parallel entries the declarations
/// cover: the owned key write, the head-sum selection with its merge, and
/// the verify partials in the exact verify steps.
#[test]
fn dcp2_launches_keep_the_declared_invariance() {
    let extra = model_qwen38_contract::owned::declared();
    let shape = Shape::TP4;
    let cp = model_qwen38_contract::sm120::context_parallel::symbols(shape);
    let mut violations = Vec::new();
    for (rank, lane) in dcp2_journal().into_iter().enumerate() {
        let ran = |phase: fn(&Launched) -> bool| {
            cp.iter()
                .filter(|symbol| lane.iter().any(|(l, s)| phase(l) && s == *symbol))
                .count()
        };
        assert!(
            ran(|l| *l == Launched::Prefill) >= 6,
            "rank {rank}: the prefill steps run the context-parallel entries"
        );
        assert!(
            lane.iter()
                .any(|(l, s)| matches!(l, Launched::Verify { rows: 4, .. })
                    && s.contains("selected_partials")
                    && s.ends_with("_verify_v1")),
            "rank {rank}: the exact verify step runs the verify partials"
        );
        for violation in PROGRAM.check_journal(
            model_qwen38_contract::sm120::catalogs(),
            lane.iter()
                .map(|(launched, symbol)| (*launched, symbol.as_str())),
            &extra,
        ) {
            violations.push(format!("rank {rank}: {violation}"));
        }
    }
    assert!(violations.is_empty(), "{}", violations.join("\n"));
}
