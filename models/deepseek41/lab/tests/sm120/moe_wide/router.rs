//! Compare complete router projection, selection and metadata with poison replay.
use super::device::{Pool, Programs};
use kernel_build::{EntryParam, EntrySpec};
use kernel_core::backend::{Launch, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_harness::{LabResult, Session};
use kernels_sm120::f32_projection::cached_spans::Buffers;
use kernels_sm120::moe::chained::{bandwidth, floor, router_finish64};
use model_deepseek41_contract::{geometry::FLASH, moe};

const EXPERTS: u32 = 384;
const TOP: u32 = 6;
const HIDDEN: u32 = 5120;
const TABLE_BYTES: u64 = (2 + 4 * EXPERTS as u64) * 4;

fn expert(row: u32, k: u32, pattern: u32, rank: u32) -> u32 {
    let slot = match pattern {
        0 => k,
        1 => (TOP * row + k) % 24,
        2 => TOP * row + k,
        _ => unreachable!(),
    };
    (37 * slot + 11 + 17 * rank) % EXPERTS
}

struct Input {
    activation: View,
    weight: View,
    weight_domain: View,
    live: View,
    bias: View,
    select: View,
}

fn input(p: &mut Pool<'_>, rows: u32, pattern: u32, rank: u32, fallback: bool) -> LabResult<Input> {
    let activation = p.zeros(u64::from(rows * HIDDEN) * 4)?;
    let weight = p.zeros(u64::from(EXPERTS * HIDDEN) * 2)?;
    if p.session.is_some() {
        let mut a = vec![0_f32; (rows * HIDDEN) as usize];
        for row in 0..rows {
            let start = (row * HIDDEN) as usize;
            a[start..start + EXPERTS as usize].fill(-16.);
            for k in 0..TOP {
                a[start + expert(row, k, pattern, rank) as usize] = 8. - k as f32 * 0.5;
            }
            for k in EXPERTS..HIDDEN {
                a[start + k as usize] = ((k + 3 * row + 7 * rank) % 11) as f32 / 16. - 0.25;
            }
            if fallback {
                a[start + 4096] = 2_f32.powi(-40);
            }
        }
        p.write(
            activation,
            &a.into_iter().flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
        )?;
        let weights = (0..EXPERTS)
            .flat_map(|e| {
                (0..HIDDEN).flat_map(move |k| {
                    let value = if k < EXPERTS {
                        if k == e { 1. } else { 0. }
                    } else {
                        ((k + rank * 3) as i32 % 11 - 5) as f32 / 128.
                    };
                    f32_to_bf16(value).to_le_bytes()
                })
            })
            .collect::<Vec<_>>();
        p.write(weight, &weights)?;
    }
    Ok(Input {
        activation,
        weight,
        weight_domain: p.bytes(
            &vec![1_u32; (EXPERTS / 16) as usize]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?,
        live: p.bytes(&rows.to_le_bytes())?,
        bias: p.zeros(u64::from(EXPERTS) * 4)?,
        select: p.zeros(u64::from(rows))?,
    })
}

pub fn bind(
    spec: EntrySpec,
    plan: LaunchPlan,
    rows: u32,
    mut value: impl FnMut(&str) -> LabResult<Argument>,
) -> LabResult<Launch> {
    let arguments = spec
        .params
        .iter()
        .map(|p| match p {
            EntryParam::DevicePointer { name, .. } | EntryParam::U32 { name } => value(name),
            _ => Err("unsupported router entry parameter".into()),
        })
        .collect::<LabResult<Vec<_>>>()?;
    Ok(Launch {
        symbol: spec.symbol,
        rows,
        plan,
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    })
}

struct Arm {
    calls: Vec<Launch>,
    workspace: View,
    domain: View,
    partials: View,
    arrived: View,
    gaps: [View; 2],
    semantic: Vec<(&'static str, View)>,
}

fn arm(
    p: &mut Pool<'_>,
    e: &floor::Entry,
    rows: u32,
    input: &Input,
    candidate: bool,
) -> LabResult<Arm> {
    let layout = router_finish64::Layout::new(e, rows).map_err(|e| e.to_string())?;
    let ticket_offset = layout.routed + TABLE_BYTES.next_multiple_of(256);
    let workspace = p.zeros(ticket_offset + 256)?;
    let domain = p.subspan(workspace, 0, 256)?;
    let partials = p.subspan(workspace, layout.partials, layout.partial_bytes)?;
    let arrived = p.subspan(workspace, layout.arrival, 4)?;
    let table = p.subspan(workspace, layout.routed, TABLE_BYTES)?;
    let tickets = p.subspan(workspace, ticket_offset, 256)?;
    let gaps = [
        p.subspan(workspace, layout.arrival + 4, 252)?,
        p.subspan(
            workspace,
            layout.routed + TABLE_BYTES,
            ticket_offset - layout.routed - TABLE_BYTES,
        )?,
    ];
    let mut semantic = Vec::new();
    for (name, bytes) in [
        ("logits", u64::from(rows * EXPERTS) * 4),
        ("scores", u64::from(rows * EXPERTS) * 4),
        ("ids", u64::from(rows * TOP) * 4),
        ("weights", u64::from(rows * TOP) * 4),
        ("counts", u64::from(EXPERTS) * 4),
        ("offsets", u64::from(EXPERTS + 1) * 4),
        ("sorted", u64::from(EXPERTS * 64) * 4),
        ("padded", 4),
        ("active", u64::from(EXPERTS + 1) * 4),
    ] {
        semantic.push((name, p.zeros(bytes)?));
    }
    semantic.extend([("scratch", table), ("tickets", tickets)]);
    let view = |name: &str| {
        semantic
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
            .ok_or_else(|| format!("missing router role {name}"))
    };
    let cached = bundle_deepseek41::sm120::f32_verify::router().map_err(|e| e.to_string())?;
    let stages = cached
        .stages(
            Buffers {
                activation: input.activation,
                weight: input.weight,
                row_domain: domain,
                partials,
                live_rows: input.live,
                weight_domain: input.weight_domain,
                output: view("logits")?,
            },
            rows,
        )
        .map_err(|e| e.to_string())?;
    let scale_bits = moe::router(&FLASH, moe::expert_sets(&FLASH)[0])
        .scaling_factor
        .to_bits();
    let value = |name: &str| {
        Ok(match name {
            "rows" => Argument::U32(rows),
            "scale_bits" => Argument::U32(scale_bits),
            _ => Argument::DeviceBuffer(
                match name {
                    "input" => input.activation,
                    "live" | "live_tokens" => input.live,
                    "bias" | "alternate_bias" => input.bias,
                    "bias_select" => input.select,
                    "row_domain" => domain,
                    "partials" => partials,
                    "arrived" => arrived,
                    "gate_tickets" => tickets,
                    other => view(other)?,
                }
                .handle,
            ),
        })
    };
    let calls = if candidate {
        let prepare = router_finish64::Stage::Prepare;
        let finish = router_finish64::Stage::Finish;
        vec![
            bind(
                router_finish64::entry(e, prepare),
                router_finish64::plan(e, prepare, rows).map_err(|e| e.to_string())?,
                rows,
                value,
            )?,
            stages.into_producer(rows).map_err(|e| e.to_string())?,
            bind(
                router_finish64::entry(e, finish),
                router_finish64::plan(e, finish, rows).map_err(|e| e.to_string())?,
                rows,
                value,
            )?,
        ]
    } else {
        let mut calls = stages.into_launches().to_vec();
        let route = bandwidth::router(e);
        calls.push(bind(
            route.entry().map_err(|e| e.to_string())?,
            route.plan(rows).map_err(|e| e.to_string())?,
            rows,
            value,
        )?);
        calls.push(bind(
            e.entry(floor::Stage::Metadata),
            e.plan(floor::Stage::Metadata, rows)
                .map_err(|e| e.to_string())?,
            rows,
            value,
        )?);
        calls
    };
    Ok(Arm {
        calls,
        workspace,
        domain,
        partials,
        arrived,
        gaps,
        semantic,
    })
}

fn entry() -> LabResult<floor::Entry> {
    super::super::moe::registration::floor_entries(&FLASH)?
        .into_iter()
        .find(|e| e.experts() == EXPERTS)
        .ok_or("missing backbone floor entry".into())
}

pub fn cpu_inventory() -> LabResult<()> {
    let e = entry()?;
    let mut count = 0;
    for rows in 33..=64 {
        for rank in 0..4 {
            for pattern in 0..3 {
                let mut p = Pool::new(None);
                let input = input(&mut p, rows, pattern, rank, false)?;
                let old = arm(&mut p, &e, rows, &input, false)?;
                let new = arm(&mut p, &e, rows, &input, true)?;
                assert_eq!(old.calls.len(), 5);
                assert_eq!(new.calls.len(), 3);
                assert_eq!(old.partials.bytes, 61440 * u64::from(rows));
                assert_eq!(old.semantic.len(), new.semantic.len());
                for call in old.calls.iter().chain(&new.calls) {
                    assert!(
                        bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol).is_some(),
                        "{}",
                        call.symbol
                    );
                    kernel_harness::launch::config(call.plan)?;
                }
                for row in 0..rows {
                    let ids = (0..TOP)
                        .map(|k| expert(row, k, pattern, rank))
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(ids.len(), TOP as usize);
                }
                count += 1;
            }
        }
    }
    assert_eq!(count, 384);
    println!("MOE_ROUTER64_CPU cases={count} PASS");
    Ok(())
}

pub fn cpu_worker(worker: u32, checked: bool) -> LabResult<()> {
    if worker > 1 {
        return Err("worker must be zero or one".into());
    }
    let e = entry()?;
    let widths: Vec<u32> = if checked {
        vec![33, 64]
    } else {
        (33..=64).collect()
    };
    let ranks = if checked {
        vec![worker]
    } else {
        vec![worker, worker + 2]
    };
    let mut cases = 0;
    for rows in widths {
        for &rank in &ranks {
            for pattern in 0..3 {
                for fallback in [false, true] {
                    let mut p = Pool::new(None);
                    let input = input(&mut p, rows, pattern, rank, fallback)?;
                    let old = arm(&mut p, &e, rows, &input, false)?;
                    let new = arm(&mut p, &e, rows, &input, true)?;
                    assert_eq!((old.calls.len(), new.calls.len()), (5, 3));
                    cases += 1;
                    println!(
                        "MOE_ROUTER64_DRY_PAIR rows={rows} rank={rank} pattern={pattern} fallback={fallback} PASS"
                    );
                }
            }
        }
    }
    assert_eq!(cases, if checked { 12 } else { 384 });
    let scheduled_replays = cases * if checked { 4 } else { 20 };
    println!(
        "MOE_ROUTER64_DRY worker={worker} checked={checked} cases={cases} scheduled_replays={scheduled_replays} PASS"
    );
    Ok(())
}

fn assert_bytes(label: &str, old: &[u8], new: &[u8]) -> LabResult<()> {
    if old != new {
        let at = old.iter().zip(new).position(|(a, b)| a != b);
        return Err(format!("router64 bitwise mismatch in {label} at {at:?}"));
    }
    Ok(())
}

pub fn run(s: &Session, worker: u32, checked: bool) -> LabResult<()> {
    let e = entry()?;
    let widths: Vec<u32> = if checked {
        vec![33, 64]
    } else {
        (33..=64).collect()
    };
    let ranks = if checked {
        vec![worker]
    } else {
        vec![worker, worker + 2]
    };
    let poisons = if checked {
        vec![0x5a]
    } else {
        vec![0x5a, 0xa5]
    };
    let modes = if checked {
        vec![false]
    } else {
        vec![false, true]
    };
    let mut cases = 0;
    let mut replays = 0;
    for rows in widths {
        for &rank in &ranks {
            for pattern in 0..3 {
                for fallback in [false, true] {
                    let mut p = Pool::new(Some(s));
                    let input = input(&mut p, rows, pattern, rank, fallback)?;
                    let arms = [
                        arm(&mut p, &e, rows, &input, false)?,
                        arm(&mut p, &e, rows, &input, true)?,
                    ];
                    let calls = arms
                        .iter()
                        .flat_map(|a| a.calls.iter().cloned())
                        .collect::<Vec<_>>();
                    let programs = Programs::load(&p, &calls)?;
                    let graphs = if checked {
                        Vec::new()
                    } else {
                        arms.iter()
                            .map(|arm| s.capture(|stream| programs.issue(&p, &arm.calls, stream)))
                            .collect::<LabResult<Vec<_>>>()?
                    };
                    for &poison in &poisons {
                        for &graph in &modes {
                            for arm in &arms {
                                p.fill(arm.workspace, poison)?;
                                for (_, view) in &arm.semantic {
                                    p.fill(*view, poison)?;
                                }
                            }
                            let lives = if checked {
                                vec![rows, rows / 2, 0, rows]
                            } else {
                                vec![rows, rows / 2, 1, 0, rows]
                            };
                            for live in lives {
                                p.write(input.live, &live.to_le_bytes())?;
                                let before = arms
                                    .iter()
                                    .map(|a| p.read(a.partials))
                                    .collect::<LabResult<Vec<_>>>()?;
                                for (index, arm) in arms.iter().enumerate() {
                                    if graph {
                                        graphs[index]
                                            .launch(&s.stream)
                                            .map_err(|e| e.to_string())?;
                                    } else {
                                        programs.issue(&p, &arm.calls, &s.stream)?;
                                    }
                                }
                                p.sync()?;
                                for arm in &arms {
                                    for gap in arm.gaps {
                                        assert_bytes(
                                            "workspace padding",
                                            &vec![poison; gap.bytes as usize],
                                            &p.read(gap)?,
                                        )?;
                                    }
                                }
                                for ((name, old), (_, new)) in
                                    arms[0].semantic.iter().zip(&arms[1].semantic)
                                {
                                    assert_bytes(name, &p.read(*old)?, &p.read(*new)?)?;
                                }
                                let domains = [p.read(arms[0].domain)?, p.read(arms[1].domain)?];
                                assert_bytes("row domain", &domains[0], &domains[1])?;
                                for row in 0..64 {
                                    let at = row as usize * 4;
                                    let expected = if row < rows {
                                        u32::from(row >= live || !fallback).to_le_bytes()
                                    } else {
                                        [poison; 4]
                                    };
                                    assert_bytes(
                                        "domain classification",
                                        &expected,
                                        &domains[1][at..at + 4],
                                    )?;
                                }
                                let after = [p.read(arms[0].partials)?, p.read(arms[1].partials)?];
                                assert_bytes("chain partials", &after[0], &after[1])?;
                                for index in 0..2 {
                                    for chain in 0..40 {
                                        let start = ((chain * rows + live) * EXPERTS * 4) as usize;
                                        let end = ((chain + 1) * rows * EXPERTS * 4) as usize;
                                        assert_bytes(
                                            "inactive partials",
                                            &before[index][start..end],
                                            &after[index][start..end],
                                        )?;
                                    }
                                }
                                let ids = p.read(
                                    arms[1]
                                        .semantic
                                        .iter()
                                        .find(|(n, _)| *n == "ids")
                                        .unwrap()
                                        .1,
                                )?;
                                for row in 0..live {
                                    for k in 0..TOP {
                                        let at = ((row * TOP + k) * 4) as usize;
                                        let got =
                                            u32::from_le_bytes(ids[at..at + 4].try_into().unwrap());
                                        if got != expert(row, k, pattern, rank) {
                                            return Err(format!(
                                                "router selected wrong expert at {row}/{k}: {got}"
                                            ));
                                        }
                                    }
                                }
                                assert_bytes(
                                    "arrival reset and election",
                                    &rows.to_le_bytes(),
                                    &p.read(arms[1].arrived)?,
                                )?;
                                replays += 1;
                            }
                        }
                    }
                    p.guards()?;
                    cases += 1;
                    println!(
                        "MOE_ROUTER64_PAIR rows={rows} rank={rank} pattern={pattern} fallback={fallback} PASS"
                    );
                }
            }
        }
    }
    let expected = if checked { (12, 48) } else { (384, 7680) };
    if (cases, replays) != expected {
        return Err(format!(
            "router inventory differs: {cases}/{replays}, expected {expected:?}"
        ));
    }
    println!(
        "MOE_ROUTER64_COMPLETE worker={worker} checked={checked} cases={cases} replays={replays} PASS"
    );
    Ok(())
}
