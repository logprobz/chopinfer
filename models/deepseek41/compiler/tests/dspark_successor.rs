//! DSpark successor reuse against inline drafting on a real TP4 instance,
//! bit for bit on every rank. Build before the GPU job, with `--release`:
//! the host loads and uploads the checkpoint in this binary, and the first
//! debug run (2026-11-07) spent about 16 minutes before its first step,
//! whose two arms of a case then took under a second.
//!
//! The package must be built with the bundle's `state-capture` feature,
//! and `CHOPINFER_DSPARK_STATE_DIR` must name an empty directory: the test
//! drives the capture controls there (`instance::dspark::successor_capture`).
//! Each case runs twice through the same host steps: a reuse arm, whose
//! queueing step queues the successor its `successor.json` names, and an
//! inline arm, where `successors-off` queues none and the consumer drafts
//! inline. Every drafted step's saved status, draft IDs, confidence and
//! item rings on every rank, its target logits and draft tokens, must be
//! identical across the arms, and the reuse arm's consumer must have
//! reused the successor. A case either forces its acceptance or takes each
//! item's greedy acceptance from the queueing step's own target logits.
use model_deepseek41_contract::execution_options::DSPARK_GRAPH_HANDLES_PER_RANK;
use runtime_bundle_abi::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[path = "dspark_successor/device.rs"]
mod device;
#[path = "../../bundle/src/fake_host_tests/dspark/successor_protocol.rs"]
mod protocol;
use device::{Capture, Fixture, Greedy, Item, Scheduled};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    bundle: PathBuf,
    checkpoint: PathBuf,
    output: PathBuf,
}

fn gpu_guard() -> Result<()> {
    let selected = std::env::var("CHOPINFER_KERNEL_GPUS")?;
    let ordinals = selected.split(',').map(str::trim).collect::<Vec<_>>();
    if ordinals != ["0", "1", "2", "3"] {
        return Err("this test requires the job's four GPUs, numbered 0,1,2,3".into());
    }
    Ok(())
}

/// How the queueing step's items accept their drafts.
#[derive(Clone, Copy, Debug)]
enum Acceptance {
    /// Every item accepts this many drafts, with bonus [`FORCED_BONUS`].
    Forced(u32),
    /// Each item's greedy acceptance of its own target logits.
    Greedy,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    label: &'static str,
    decoders: u32,
    /// Each decoder's prompt tokens.
    prompt: usize,
    /// The drafts of every verify step.
    depth: u32,
    acceptance: Acceptance,
    /// Whether the successor is an allocated step's, of five heads.
    allocated: bool,
}

const fn case(
    label: &'static str,
    decoders: u32,
    prompt: usize,
    depth: u32,
    acceptance: Acceptance,
    allocated: bool,
) -> Case {
    Case {
        label,
        decoders,
        prompt,
        depth,
        acceptance,
        allocated,
    }
}

use Acceptance::{Forced, Greedy as Natural};

/// Every case, each a decode step consuming the successor: depth five
/// accepts all five drafts, at C5 within one 32-row projection and at C16
/// with its saved status at the 1,408-byte bound (C6 to C8 queue none); a
/// 125-token prompt's successor projects across the 128-cell ring wrap; 9
/// to 16 items take the wide successor. Every kind but all-accepted, which
/// forcing defines, also has a greedy case.
const CASES: &[Case] = &[
    case("c5-all-accepted", 5, 40, 5, Forced(5), false),
    case("c16-all-accepted", 16, 40, 5, Forced(5), false),
    case("c8-ring-wrap", 8, 125, 3, Forced(3), false),
    case("c8-ring-wrap-greedy", 8, 125, 3, Natural, false),
    case("c9-wide", 9, 40, 3, Forced(3), false),
    case("c12-wide", 12, 40, 3, Forced(3), false),
    case("c13-wide", 13, 40, 3, Forced(3), false),
    case("c16-wide", 16, 40, 3, Forced(3), false),
    case("c12-wide-greedy", 12, 40, 3, Natural, false),
];

const FORCED_BONUS: u32 = 13;
/// The input tokens of the warm, queueing and follow-up steps.
const WARM_INPUT: u32 = 7;
const QUEUE_INPUT: u32 = 11;
const FOLLOW_INPUT: u32 = 17;

fn prompt(decoder: u64, tokens: usize) -> Vec<u32> {
    (0..tokens as u64)
        .map(|at| 100 + ((at * 7 + decoder * 131) % 4000) as u32)
        .collect()
}

/// One arm's captures and decided acceptance.
struct Arm {
    captures: BTreeMap<&'static str, Capture>,
    records: BTreeMap<&'static str, Value>,
    /// The queueing step's `successor.json`.
    feedback: Vec<protocol::Accepted>,
}

fn verify(decoders: u32, inputs: &BTreeMap<u64, u32>, drafts: u32) -> Vec<Scheduled> {
    (1..=u64::from(decoders))
        .map(|request| Scheduled {
            request,
            item: Item::Verify {
                bonus: inputs[&request],
                drafts,
            },
        })
        .collect()
}

fn each(decoders: u32, value: u32) -> BTreeMap<u64, u32> {
    (1..=u64::from(decoders)).map(|r| (r, value)).collect()
}

fn run_arm(fixture: &mut Fixture, case: &Case, inline: bool, output: &Path) -> Result<Arm> {
    protocol::select_arm(&fixture.root, inline)?;
    let arm = protocol::arm_directory(output, inline);
    let mut captures = BTreeMap::new();
    let mut records = BTreeMap::new();
    let decoders = u64::from(case.decoders);
    let mut prompts = Vec::new();
    for request in 1..=decoders {
        fixture.admit(request, request - 1)?;
        prompts.push(Scheduled {
            request,
            item: Item::Prefill {
                tokens: prompt(request, case.prompt),
            },
        });
    }
    fixture.step(&prompts, None)?;
    fixture.commit(&BTreeMap::new())?;
    let mut drafted = |fixture: &mut Fixture,
                       name: &'static str,
                       items: &[Scheduled],
                       greedy: Option<Greedy>|
     -> Result<()> {
        let capture = fixture.step(items, greedy)?;
        records.insert(name, fixture.take_dump(&arm.join(name))?);
        captures.insert(name, capture);
        Ok(())
    };
    drafted(
        fixture,
        "warm",
        &verify(case.decoders, &each(case.decoders, WARM_INPUT), case.depth),
        None,
    )?;
    fixture.commit(&each(case.decoders, 1))?;
    let greedy = match case.acceptance {
        Forced(accepted) => {
            let items: Vec<_> = (1..=decoders)
                .map(|request| (request, accepted, FORCED_BONUS))
                .collect();
            protocol::write_feedback(&fixture.root, case.allocated, &items)?;
            None
        }
        Natural => Some(Greedy {
            root: fixture.root.clone(),
            allocated: case.allocated,
        }),
    };
    drafted(
        fixture,
        "queue",
        &verify(case.decoders, &each(case.decoders, QUEUE_INPUT), case.depth),
        greedy,
    )?;
    let feedback = protocol::take_feedback(&fixture.root)?;
    let kept: BTreeMap<u64, u32> = feedback
        .iter()
        .map(|&(request, accepted, _)| (request, accepted + 1))
        .collect();
    let bonus: BTreeMap<u64, u32> = feedback
        .iter()
        .map(|&(request, _, bonus)| (request, bonus))
        .collect();
    fixture.commit(&kept)?;
    drafted(
        fixture,
        "consume",
        &verify(case.decoders, &bonus, case.depth),
        None,
    )?;
    fixture.commit(&each(case.decoders, 1))?;
    drafted(
        fixture,
        "follow",
        &verify(
            case.decoders,
            &each(case.decoders, FOLLOW_INPUT),
            case.depth,
        ),
        None,
    )?;
    fixture.commit(&each(case.decoders, 1))?;
    for request in 1..=decoders {
        fixture.retire(request)?;
    }
    Ok(Arm {
        captures,
        records,
        feedback,
    })
}

/// Whether the arms match: the dumps of every step (the shared protocol's
/// comparison) and each step's draft tokens and F32 logits.
fn compare(output: &Path, case: &Case, reuse: &Arm, inline: &Arm) -> Result<()> {
    let label = case.label;
    if reuse.feedback != inline.feedback {
        return Err(format!("{label}: the arms decided different acceptance").into());
    }
    for step in protocol::STEPS {
        let (a, b) = (&reuse.captures[step], &inline.captures[step]);
        if a.drafts != b.drafts {
            return Err(format!("{label} {step}: draft tokens differ").into());
        }
        if a.rows.len() != b.rows.len() {
            return Err(format!("{label} {step}: logits row counts differ").into());
        }
        for (row, (x, y)) in a.rows.iter().zip(&b.rows).enumerate() {
            if let Some(at) = protocol::first_difference(x, y) {
                return Err(
                    format!("{label} {step}: logits row {row} differs at byte {at}").into(),
                );
            }
        }
    }
    Ok(protocol::compare_dumps(output)?)
}

fn run(extra: &[(&str, &str)]) -> Result<()> {
    gpu_guard()?;
    let path = PathBuf::from(
        std::env::var_os("CHOPINFER_DEEPSEEK41_SUCCESSOR_TEST")
            .ok_or("CHOPINFER_DEEPSEEK41_SUCCESSOR_TEST must name the test configuration")?,
    );
    let config: Config = serde_json::from_slice(&protocol::read(&path)?)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    std::fs::create_dir(&config.output)
        .map_err(|error| format!("{}: {error}", config.output.display()))?;
    let options: BTreeMap<_, _> = extra
        .iter()
        .map(|&(name, value)| (name.to_owned(), value.to_owned()))
        .collect();
    let mut fixture = Fixture::new(&config, &options)?;
    let mut receipts = Vec::new();
    for case in CASES {
        let output = config.output.join(case.label);
        let reuse = run_arm(&mut fixture, case, false, &output)?;
        let inline = run_arm(&mut fixture, case, true, &output)?;
        compare(&output, case, &reuse, &inline)?;
        let heads = &reuse.records["consume"]["computed_positions"];
        receipts.push(json!({
            "case": case.label, "decoders": case.decoders, "prompt": case.prompt,
            "depth": case.depth, "acceptance": format!("{:?}", case.acceptance),
            "allocated": case.allocated,
            "feedback": reuse.feedback, "consumer_heads": heads,
        }));
        eprintln!(
            "DSpark successor {}: reuse matches inline bitwise ({heads} heads)",
            case.label
        );
    }
    fixture.finish()?;
    let receipt = config.output.join("receipt.json");
    std::fs::write(
        &receipt,
        serde_json::to_vec_pretty(&json!({
            "schema": "chopinfer.deepseek41.dspark-successor-gpu.v1", "passed": true,
            "geometry": "flash", "tensor_parallel": 4, "model_options": options,
            "comparison": "bitwise per-rank saved status, draft IDs, confidence and item \
                rings, rank-0 draft tokens and F32 target logits, reuse against inline",
            "cases": receipts,
        }))?,
    )
    .map_err(|error| format!("{}: {error}", receipt.display()))?;
    Ok(())
}

#[test]
#[ignore = "requires a state-capture package, its checkpoint, an empty CHOPINFER_DSPARK_STATE_DIR and four idle GPUs; run alone"]
fn dspark_successor_graphs_gpu() -> Result<()> {
    run(&[])
}

#[test]
#[ignore = "requires a state-capture package, its checkpoint, an empty CHOPINFER_DSPARK_STATE_DIR and four idle GPUs; run alone"]
fn dspark_successor_eager_gpu() -> Result<()> {
    run(&[(DSPARK_GRAPH_HANDLES_PER_RANK, "0")])
}

#[test]
fn greedy_acceptance_breaks_ties_by_id_and_refuses_nonfinite_rows() {
    let row = [1_f32, 3.0, 3.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(device::maximum(&row).unwrap(), 1);
    assert!(device::maximum(&f32::NAN.to_le_bytes()).is_err());
    assert!(device::maximum(&[]).is_err());
    assert_eq!(protocol::first_difference(b"abc", b"abd"), Some(2));
    assert_eq!(protocol::first_difference(b"ab", b"abc"), Some(2));
    assert_eq!(protocol::first_difference(b"ab", b"ab"), None);
}
