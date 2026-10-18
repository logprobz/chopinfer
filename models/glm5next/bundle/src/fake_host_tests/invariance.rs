//! The invariance journal: what GLM asks of each op at create and what the
//! backend grants it (`ops::exact`, `Ops::asked`), and every launch of a
//! served instance's steps against the declarations of the entries it
//! launches (the SM120 invariance `Program`).
//!
//! The asks of every serving configuration (each checkpoint layout at
//! context-parallel size 1, 2 and 4) are one pinned journal: an ask or a
//! grant that changes changes its digest. The asks choose what the rows
//! alone choose, so asking moves no launch.
//!
//! The launch journal runs eager and graph steps of an instance of each
//! latent KV format with the projections' precision component and without
//! the drafter (the drafter's launches only choose drafts, which the
//! target's verify accepts or rejects; its verify steps here are
//! host-drafted): prefills of 3 and 2,050 rows together (two indexer
//! pieces), one of 130 rows that keeps a prefix anchor, decodes of two
//! items and of one, and verify steps of four rows of one item and of
//! three and two rows of two. Each distinct (phase, symbol) is a line with
//! what the ops that launched it asked and were granted, or the declaration
//! a launch outside the ops resolves to; the lines are pinned. The program
//! check (prefill launches keep rows over chunks at `PREFILL_GRANULE` and
//! anchors there, verify launches one-row bits at the verify bucket's rows)
//! finds the claims GLM relies on that no launch gets; they are pinned in a
//! journal of their own, so a claim lost or gained fails the suite until
//! the pin moves.

use super::sm120::{Declared, ExactVerify, Launched, OpKind, Program};
use kernel_core::properties::{Anchors, Batch, Chunk, Invariance, Rows, Scope, Ties};
use model_glm5next_contract::contract::{
    ANCHOR_ITEMS, Glm, PAGE_TOKENS, PREFILL_GRANULE, latent_format,
};
use model_glm5next_contract::quantization::{
    FP8_BLOCK128, Layout, representative_layouts, test_support::config_of,
};
use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};

use super::*;
use crate::ops::{Asked, Ops, Plan};

/// What GLM's program promises: chunks and anchors on the granule, and
/// one-row bits at every verify step an instance serves (at most the
/// widest step's rows over every item).
fn program() -> Program {
    Program {
        prefill_granule: PREFILL_GRANULE,
        exact_verify: ExactVerify {
            rows: crate::buffers::widest_step_rows(),
            items: DECODE_BATCH,
        },
        anchor_items: ANCHOR_ITEMS,
    }
}

/// `invariance` in a line: its claims, `none` for none.
fn claims(invariance: Invariance) -> String {
    let mut text = Vec::new();
    match invariance.chunk {
        Chunk::None => {}
        Chunk::Any => text.push("chunk any".to_owned()),
        Chunk::Granule(granule) => text.push(format!("chunk {granule}")),
    }
    if let Rows::OneRow { max } = invariance.rows {
        text.push(format!("one-row {max}"));
    }
    if invariance.batch == Batch::Independent {
        text.push("batch".into());
    }
    if invariance.ties == Ties::LowerIndex {
        text.push("ties lower".into());
    }
    match invariance.anchors {
        Anchors::NotApplicable => {}
        anchors => text.push(format!("anchors {anchors:?}")),
    }
    if invariance.scope == Scope::RoutedOperation && !text.is_empty() {
        text.push("of the routed operation".into());
    }
    if text.is_empty() {
        "none".into()
    } else {
        text.join(", ")
    }
}

/// An ask as a line: the op, its rows, what it asked and was granted.
fn ask_line(asked: &Asked) -> String {
    format!(
        "{} | rows {} | asks {} | granted {}",
        asked.what,
        asked.need.capacity,
        claims(asked.need.exact),
        claims(asked.granted)
    )
}

/// Every serving configuration's plan: each representative layout at each
/// group size.
fn plans() -> Vec<Plan> {
    let shape = Shape::TP4;
    representative_layouts()
        .into_iter()
        .flat_map(|layout| {
            model_glm5next_contract::contract::CONTEXT_PARALLEL_SIZES.map(|size| Plan {
                shape,
                layout,
                index_topk: shape.profile.index_topk_max,
                context_parallel: size,
            })
        })
        .collect()
}

/// The ops of `plan`, with their asks or (`without_exact`) with each ask's
/// rows alone.
fn ops(plan: Plan, without_exact: bool) -> Ops {
    crate::ops::WITHOUT_EXACT.with(|cell| cell.set(without_exact));
    let ops = Ops::for_plan(plan);
    crate::ops::WITHOUT_EXACT.with(|cell| cell.set(false));
    ops
}

/// Every ask of every serving configuration is granted, and the asks
/// choose the implementations and symbols the rows alone choose: asking
/// moves no launch.
#[test]
fn the_asks_choose_what_the_rows_alone_choose() {
    for plan in plans() {
        let at = format!("{} size {}", plan.layout.id, plan.context_parallel);
        let asked = ops(plan, false).asked;
        let plain = ops(plan, true).asked;
        assert_eq!(asked.len(), plain.len(), "{at}");
        assert!(
            asked.iter().any(|a| a.need.exact != Invariance::NONE),
            "{at}"
        );
        for (asked, plain) in asked.iter().zip(&plain) {
            assert!(
                asked.granted.provides(asked.need.exact),
                "{at}: {}",
                ask_line(asked)
            );
            assert_eq!(plain.need.exact, Invariance::NONE);
            assert_eq!(
                (&asked.what, &asked.implementation, &asked.symbols),
                (&plain.what, &plain.implementation, &plain.symbols),
                "{at}"
            );
        }
    }
}

/// The precision component of a checkpoint of `layout`: the pinned FP8
/// test checkpoint's, or one of a synthetic config of its class mix.
fn component(layout: &Layout) -> Vec<u8> {
    if *layout == FP8_BLOCK128 {
        return model_glm5next_contract::prepared::test_component();
    }
    let config = serde_json::from_slice(&config_of(layout)).unwrap();
    model_glm5next_contract::prepared::checkpoint_component(layout, &config).unwrap()
}

/// What each projection use asks and is granted when create authenticates
/// `layout`'s component on the backend of `ops`' target, with its asks or
/// (`without_exact`) with each ask's rows alone.
fn projection_asks(ops: &Ops, layout: &'static Layout, without_exact: bool) -> Vec<Asked> {
    let (backend, _) = ops.target.backend(layout).unwrap();
    let asks = crate::ops::Asks::default();
    crate::ops::WITHOUT_EXACT.with(|cell| cell.set(without_exact));
    let authenticated =
        crate::projections::authenticate(&component(layout), layout, backend.as_ref(), &asks);
    crate::ops::WITHOUT_EXACT.with(|cell| cell.set(false));
    authenticated.unwrap();
    asks.asked()
}

/// The ask lines of `asked`, each line once with the count of asks it
/// stands for, in first-ask order: the projections of every layer ask
/// alike, so a projection's layer index reads `N`.
fn by_layers(asked: &[Asked]) -> Vec<String> {
    let mut lines = Vec::<(String, usize)>::new();
    for asked in asked {
        let what = asked
            .what
            .split(' ')
            .map(|word| match word.contains('.') {
                false => word.to_owned(),
                true => word
                    .split('.')
                    .map(|part| match part.bytes().all(|b| b.is_ascii_digit()) {
                        true => "N",
                        false => part,
                    })
                    .collect::<Vec<_>>()
                    .join("."),
            })
            .collect::<Vec<_>>()
            .join(" ");
        let line = ask_line(&Asked {
            what,
            ..asked.clone()
        });
        match lines.iter_mut().find(|(seen, _)| *seen == line) {
            Some((_, count)) => *count += 1,
            None => lines.push((line, 1)),
        }
    }
    lines
        .into_iter()
        .map(|(line, count)| format!("{line} | {count} uses"))
        .collect()
}

/// Every projection use of each layout asks what its phase needs, granted,
/// and the asks bind the implementations and symbols the rows alone bind.
#[test]
fn the_projections_ask_what_their_phase_needs() {
    for plan in plans()
        .into_iter()
        .filter(|plan| plan.context_parallel == 1)
    {
        let ops = ops(plan, false);
        let asked = projection_asks(&ops, plan.layout, false);
        let plain = projection_asks(&ops, plan.layout, true);
        assert_eq!(asked.len(), plain.len(), "{}", plan.layout.id);
        assert!(
            asked.iter().any(|a| a.need.exact.rows != Rows::None),
            "{}",
            plan.layout.id
        );
        for (asked, plain) in asked.iter().zip(&plain) {
            assert!(
                asked.granted.provides(asked.need.exact),
                "{}: {}",
                plan.layout.id,
                ask_line(asked)
            );
            assert_eq!(
                (&asked.what, &asked.implementation, &asked.symbols),
                (&plain.what, &plain.implementation, &plain.symbols),
                "{}",
                plan.layout.id
            );
        }
    }
}

/// Every serving configuration's asks and grants, and each layout's
/// projections', pinned. The journal in the temporary directory on a
/// mismatch lists each configuration's ops.
#[test]
fn the_asks_and_grants_are_pinned() {
    let mut lines = Vec::new();
    for plan in plans() {
        lines.push(format!(
            "plan {} context-parallel {}",
            plan.layout.id, plan.context_parallel
        ));
        let ops = ops(plan, false);
        lines.extend(ops.asked.iter().map(ask_line));
        if plan.context_parallel == 1 {
            lines.push(format!("projections {}", plan.layout.id));
            lines.extend(by_layers(&projection_asks(&ops, plan.layout, false)));
        }
    }
    // Re-pinned when the stream expansion and score compaction moved onto
    // the shared row copies: the row ops ask them, the own ops no longer.
    // Re-pinned when the drafter's argmax moved onto the shared sharded
    // argmax: the candidates' all-reduces, which asked nothing, are gone.
    // Re-pinned when the own row ops and the draft argmax were asked at
    // create: each plan lists them, and the own ops' list goes; and when
    // the all-reduces without the phased entries asked what every phase
    // needs, which the composed form now keeps; and when the
    // context-parallel localizations and partial merges asked it too; and
    // when each layout's projections asked what their phase needs; and
    // when the comm epilogues asked what every phase needs. Re-pinned when
    // layouts took ids derived from their encodings: with the earlier ids
    // put back the journal is the earlier pin's, (3429, 0x11d59ba6cad32f95).
    // Re-pinned when the block-FP8 and NVFP4 mix joined the plans: its
    // lines follow the earlier 3429, (3429, 0xd4cff0ebdbfa4477), unchanged.
    check_journal(
        &lines,
        (4578, 0x97f7cc82cc063865),
        "glm5next-invariance-asks.txt",
    )
    .unwrap();
}

/// The phase of a journal step.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Step {
    Prefill,
    Decode { items: u32 },
    Verify { rows: u32, items: u32 },
}

/// The pages of `end` tokens of a request whose pages start at `first`.
fn pages(first: u64, end: u64) -> Vec<u64> {
    (first..first + end.div_ceil(PAGE_TOKENS)).collect()
}

/// One request's rows of a step: (request, slot, first page, previous
/// tokens, inputs, anchored). One input of a request with previous tokens
/// is a decode row, several a host-drafted verify item.
type Request<'a> = (u64, u64, u64, u64, &'a [u32], bool);

fn batch(items: &[Request<'_>]) -> Vec<u8> {
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, first, previous, inputs, anchored) in items {
        let count = inputs.len() as u64;
        let end = previous + count;
        batch.push(match (previous, count) {
            (0, _) => BatchItemV1::new(request, false, count as u32, true, inputs.to_vec())
                .with_anchor(anchored),
            (_, 1) => BatchItemV1::new(request, true, 1, true, inputs.to_vec()),
            _ => BatchItemV1::verify(request, count as u32, false, inputs.to_vec()),
        });
        appends.push(BatchAppendV1::new(
            previous,
            count,
            end,
            pages(first, end),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// One step of the journal: its phase, each stream's launches in issue
/// order with the phase the program check takes, and each symbol the ops
/// lowered with the implementation and guarantees of each op that
/// launched it.
struct Journaled {
    step: Step,
    streams: BTreeMap<u32, Vec<(Launched, String)>>,
    lowered: BTreeMap<String, Vec<(String, Invariance)>>,
}

/// Every launch of the journal's steps on an instance of `format` (see
/// the module docs): a verify launch at its step's bucket rows over its
/// items. With the instance's asks and the admission create left.
fn journal(format: KvFormat) -> (Vec<Journaled>, Vec<Asked>, crate::ops::reduce::Admission) {
    let host = FakeHost::with_format((3, 32), format);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let record = crate::config::tests::record_naming(DECODE_BATCH, &options(2048), format, &roles);
    let mut instance = {
        api()
            .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
            .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
    };
    let handle = instance.handle();
    let (asked, admission) = crate::instance::asked(handle);
    let items = bundle_runtime::geometry::Items::new(DECODE_BATCH).unwrap();
    let mut steps = Vec::new();
    let mut step = |step: Step, request: Vec<u8>, kept: &[(u64, u32)]| {
        let from = host.device.journal.borrow().len();
        crate::ops::LOWERED.with(|lowered| *lowered.borrow_mut() = Some(Vec::new()));
        instance
            .execute(&request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{step:?}: {error:?} {:?}", host.logs.borrow()));
        let launched = match step {
            Step::Prefill => Launched::Prefill,
            Step::Decode { .. } => Launched::Decode,
            Step::Verify { rows, items: count } => {
                let bucket = bundle_runtime::graphs::verify_bucket_within::<Glm>(rows, items)
                    .expect("a verify bucket");
                Launched::Verify {
                    rows: bucket,
                    items: count,
                    launch_rows: bucket,
                }
            }
        };
        let streams = launches_by_stream(&host, from)
            .into_iter()
            .map(|(stream, lines)| {
                let launches = lines
                    .iter()
                    .map(|line| (launched, line.split(' ').nth(2).unwrap().to_owned()))
                    .collect();
                (stream, launches)
            })
            .collect::<BTreeMap<_, _>>();
        assert!(!streams.is_empty(), "{step:?} launches nothing");
        let mut lowered = BTreeMap::<String, Vec<(String, Invariance)>>::new();
        let recorded = crate::ops::LOWERED.with(|lowered| lowered.take().unwrap_or_default());
        for (symbol, implementation, granted) in recorded {
            let ops = lowered.entry(symbol).or_default();
            if !ops.contains(&(implementation.clone(), granted)) {
                ops.push((implementation, granted));
            }
        }
        steps.push(Journaled {
            step,
            streams,
            lowered,
        });
        if kept.is_empty() {
            instance.commit().unwrap();
        } else {
            let kept = kept.iter().map(|&(r, rows)| super::kept(r, rows));
            instance.commit_accepted(&kept.collect::<Vec<_>>()).unwrap();
        }
    };
    let prompt = |count: u32, salt: u32| (0..count).map(|t| (t * 7 + salt) % 1000).collect();
    let (short, long, anchored): (Vec<u32>, Vec<u32>, Vec<u32>) =
        (prompt(3, 1), prompt(2050, 2), prompt(130, 3));
    step(
        Step::Prefill,
        batch(&[(10, 0, 0, 0, &short, false), (11, 1, 8, 0, &long, false)]),
        &[],
    );
    step(
        Step::Prefill,
        batch(&[(12, 2, 48, 0, &anchored, true)]),
        &[],
    );
    step(
        Step::Decode { items: 2 },
        batch(&[(10, 0, 0, 3, &[21], false), (11, 1, 8, 2050, &[22], false)]),
        &[],
    );
    step(
        Step::Decode { items: 1 },
        batch(&[(10, 0, 0, 4, &[23], false)]),
        &[],
    );
    step(
        Step::Verify { rows: 4, items: 1 },
        batch(&[(10, 0, 0, 5, &[24, 25, 26, 27], false)]),
        &[(10, 2)],
    );
    step(
        Step::Verify { rows: 5, items: 2 },
        batch(&[
            (10, 0, 0, 7, &[28, 29, 30], false),
            (11, 1, 8, 2051, &[31, 32], false),
        ]),
        &[(10, 1), (11, 2)],
    );
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    (steps, asked, admission)
}

/// The claims of a launch of `symbol` in a step: the meet of the
/// guarantees of the ops that launched it there, or the declaration of
/// GLM's own kernels or of GLM's catalogs and the kernel base's (the
/// projections), none for an undeclared symbol.
fn claimed(
    symbol: &str,
    lowered: &BTreeMap<String, Vec<(String, Invariance)>>,
    declared: &BTreeMap<String, Declared>,
) -> Option<Declared> {
    match lowered.get(symbol) {
        Some(ops) => Some(Declared {
            symbol: symbol.to_owned(),
            profile: None,
            op: OpKind::Bundle,
            invariance: ops
                .iter()
                .map(|&(_, granted)| granted)
                .reduce(Invariance::meet)
                .unwrap(),
        }),
        None => declared.get(symbol).cloned(),
    }
}

/// Whether the ask `what` names a phase other than `phase` (an op of
/// that phase only, such as a decode and a verify bucket's stacked
/// projections, which launch the same symbol).
fn names_another_phase(what: &str, phase: &str) -> bool {
    what.split(|c: char| !c.is_alphanumeric())
        .map(str::to_lowercase)
        .any(|word| ["prefill", "decode", "verify"].contains(&word.as_str()) && word != phase)
}

/// Where a launch of `symbol` in `phase` got its claims: what each op
/// that launched it asked and was granted, or its declaration (for a
/// launch no asked op lowered).
fn source(
    phase: &str,
    symbol: &str,
    ops: Option<&Vec<(String, Invariance)>>,
    asked: &[Asked],
    declared: &BTreeMap<String, Declared>,
) -> String {
    let from_asks = ops.map(|ops| {
        ops.iter()
            .flat_map(|(implementation, granted)| {
                asked
                    .iter()
                    .filter(move |asked| {
                        asked.implementation == *implementation
                            && asked.symbols.iter().any(|s| s == symbol)
                            && asked.granted == *granted
                            && !names_another_phase(&asked.what, phase)
                    })
                    .map(|asked| {
                        format!(
                            "op asks {} granted {}",
                            claims(asked.need.exact),
                            claims(asked.granted)
                        )
                    })
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
            .join(" | ")
    });
    match from_asks {
        Some(from) if !from.is_empty() => from,
        _ => match declared.get(symbol) {
            Some(declared) => format!("declared {}", claims(declared.invariance)),
            None => "undeclared".into(),
        },
    }
}

/// Every distinct launch of the journals of both latent KV formats with
/// where its claims come from, pinned, and every claim GLM's program
/// relies on that a launch does not get, pinned apart (the GLM-F HANDOFF
/// lists them).
#[test]
fn the_launches_keep_the_declared_invariance() {
    let declared = super::sm120::declarations();
    let program = program();
    let mut lines = Vec::new();
    let mut gaps = BTreeSet::new();
    let formats = Shape::TP4
        .kv_formats()
        .into_iter()
        .filter(|format| !has_mtp_layer(format));
    for format in formats {
        let kv = format!("{:?}", latent_format(&format));
        let (steps, asked, admission) = journal(format);
        // The fake devices admit both spinning entries, as serving does.
        assert_eq!(
            (admission.low_latency, admission.phased),
            (true, true),
            "{kv}"
        );
        lines.push(format!("format {kv}"));
        let mut seen = BTreeMap::<(&str, String), Option<Vec<(String, Invariance)>>>::new();
        let mut verified = 0;
        for Journaled {
            step,
            streams,
            lowered,
        } in &steps
        {
            let phase = match step {
                Step::Prefill => "prefill",
                Step::Decode { .. } => "decode",
                Step::Verify { .. } => "verify",
            };
            for (launched, symbol) in streams.values().flatten() {
                let claims = claimed(symbol, lowered, &declared);
                if let Err(violation) = program.check(*launched, symbol, claims.as_ref()) {
                    gaps.insert(format!("{kv} {phase}: {violation}"));
                }
                verified += usize::from(matches!(launched, Launched::Verify { .. }));
                let ops = seen.entry((phase, symbol.clone())).or_insert(None);
                if let Some(launched) = lowered.get(symbol) {
                    let ops = ops.get_or_insert_with(Vec::new);
                    for op in launched {
                        if !ops.contains(op) {
                            ops.push(op.clone());
                        }
                    }
                }
            }
        }
        assert!(verified > 0, "{kv}: the verify steps are checked");
        lines.extend(seen.iter().map(|((phase, symbol), ops)| {
            let from = source(phase, symbol, ops.as_ref(), &asked, &declared);
            format!("{phase} {symbol} | {from}")
        }));
    }
    let gaps = gaps.into_iter().collect::<Vec<_>>();
    // Re-pinned when the stream expansion and score compaction moved onto
    // the shared row copies (their symbols, now asked by the row ops), and
    // when the own row ops were asked at create (their launches take the
    // asks, no longer the declarations), and when the projections asked
    // at create (their launches take the asks, no longer undeclared).
    check_journal(
        &lines,
        (362, 0x4459ee58aaec84e5),
        "glm5next-invariance-launches.txt",
    )
    .unwrap();
    // Empty since the projections claim what their phase needs.
    check_journal(
        &gaps,
        (0, 0xcbf29ce484222325),
        "glm5next-invariance-gaps.txt",
    )
    .unwrap();
}
