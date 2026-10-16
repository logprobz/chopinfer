//! One captured graph per bucket replayed at several live counts, as serving
//! replays one graph per bucket with the live counts read on device: the
//! conv and delta-rule decode and verify entries at 32 sequences (verify with
//! up to eight rows each) and the gated norm at 32 rows. Every live count
//! gets new contents. Outputs and the checkpoints a launch only writes start
//! poisoned; live decode slots are restored before every launch; source
//! slots, and everything belonging to sequences past the live count, must
//! keep their bytes. The eager launch at each live count meets the recurrent
//! oracles and the graph replay matches it bit for bit.

use super::*;
use kernel_lab::poison::{POISON, Replay, WriteMap, live_counts, write};

/// Sequences (and decode or norm rows) of every bucket.
const BUCKET: usize = 32;

#[test]
fn linear_replays_one_graph_at_every_live_count() {
    gpu_test(4 << 30, |session| {
        let module = module(session)?;
        for kind in [
            "conv_decode",
            "delta_decode",
            "conv_verify",
            "delta_verify",
            "gated_norm",
        ] {
            replay_kind(session, &module, kind)?;
        }
        Ok(())
    })
    .expect("linear live-count graph replay");
}

/// The capacity layout of one kind: pool entries, slots and checkpoints.
struct Layout {
    verify: bool,
    norm: bool,
    conv: bool,
    /// Output row capacity.
    row_capacity: usize,
    /// Pool entries: a source slot per sequence at odd entries, and for
    /// verify eight checkpoints per sequence after them.
    entries: usize,
    slot_ids: Vec<i32>,
    checkpoint_ids: Vec<i32>,
    /// Output elements per row.
    width: usize,
    /// Pool entry bytes and element type.
    entry_bytes: usize,
    element: DType,
}

impl Layout {
    fn new(kind: &str) -> Self {
        let g = geometry();
        let max = verify_row_limit() as usize;
        let verify = kind.ends_with("_verify");
        let norm = kind == "gated_norm";
        let conv = kind.starts_with("conv");
        let slots = 2 * BUCKET + 1;
        let checkpoint_ids = if verify {
            (0..BUCKET * max).map(|i| (slots + i) as i32).collect()
        } else {
            Vec::new()
        };
        Self {
            verify,
            norm,
            conv,
            row_capacity: if verify { BUCKET * max } else { BUCKET },
            entries: slots + checkpoint_ids.len(),
            slot_ids: (0..BUCKET).map(|s| (2 * s + 1) as i32).collect(),
            checkpoint_ids,
            width: if conv { g.channels } else { g.norm_width },
            entry_bytes: if conv {
                HISTORY * g.channels * 2
            } else {
                g.delta_state * 4
            },
            element: if conv { DType::Bf16 } else { DType::F32 },
        }
    }

    fn entry(&self, id: i32) -> std::ops::Range<usize> {
        let at = id as usize * self.entry_bytes;
        at..at + self.entry_bytes
    }
}

fn replay_kind(session: &Session, module: &Module, kind: &str) -> LabResult<()> {
    let g = geometry();
    let layout = Layout::new(kind);
    let max = verify_row_limit() as usize;
    let kernel =
        Kernel::load(module, &symbol(kind))?.with_bucket(layout.row_capacity as u32, BUCKET as u32);
    // Every buffer at capacity, allocated once: the graph keeps their
    // addresses, and each live count rewrites their contents in place.
    let mut rng = Rng::new(0x11fe_0000);
    let mut op = Operands::new();
    if layout.norm {
        let data = NormCase::new(&mut rng, layout.row_capacity);
        op.bf16(session, "input", &data.input)?;
        op.bf16(session, "gate", &data.gate)?;
        op.bf16(session, "weight", &data.weight)?;
        op.u32(session, "live_rows", 0)?;
        op.scalars.insert("gate_stride", g.z_stride as u32);
    } else if layout.conv {
        let data = ConvCase::new(&mut rng, layout.row_capacity, layout.entries);
        op.bf16(session, "input", &data.input)?;
        op.bf16(session, "weight", &data.weight)?;
        op.bf16(session, "state", &data.pool)?;
        op.scalars.insert("input_stride", g.qkv_stride as u32);
    } else {
        let data = DeltaCase::new(&mut rng, layout.row_capacity, layout.entries);
        op.bf16(session, "qkv", &data.qkv)?;
        op.bf16(session, "a", &data.a)?;
        op.bf16(session, "b", &data.b)?;
        op.f32(session, "a_log", &data.a_log)?;
        op.f32(session, "dt_bias", &data.dt_bias)?;
        op.f32(session, "state", &data.pool)?;
        op.scalars.insert("qkv_stride", g.qkv_stride as u32);
        op.scalars.insert("gate_stride", g.gate_stride as u32);
    }
    if !layout.norm {
        op.i32(session, "slot_ids", &layout.slot_ids)?;
        op.u32(session, "live_sequences", 0)?;
    }
    if layout.verify {
        op.i32(session, "cu_rows", &[0; BUCKET + 1])?;
        op.i32(session, "checkpoint_ids", &layout.checkpoint_ids)?;
    }
    op.bytes(
        session,
        "output",
        &vec![POISON; layout.row_capacity * layout.width * 2],
    )?;
    let launch = |stream: &Stream| {
        kernel.launch(
            session,
            stream,
            &op,
            layout.row_capacity as u32,
            BUCKET as u32,
            true,
        )
    };
    let graph = session.capture(launch)?;
    for live in live_counts(BUCKET as u32) {
        let live = live as usize;
        let mut rng = Rng::new(0x5eed_0000 ^ ((live as u64) << 16) ^ kind.len() as u64);
        let lengths: Vec<usize> = (0..BUCKET)
            .map(|_| {
                if layout.verify {
                    1 + (rng.next() % max as u64) as usize
                } else {
                    1
                }
            })
            .collect();
        let cu = cu(&lengths);
        let live_rows = cu[live] as usize;
        let mut report = Report::new(format!("{kind}_live{live}_of{BUCKET}"));
        // New contents for this live count.
        let (conv, delta, norm) = if layout.norm {
            (
                None,
                None,
                Some(NormCase::new(&mut rng, layout.row_capacity)),
            )
        } else if layout.conv {
            (
                Some(ConvCase::new(&mut rng, layout.row_capacity, layout.entries)),
                None,
                None,
            )
        } else {
            (
                None,
                Some(DeltaCase::new(
                    &mut rng,
                    layout.row_capacity,
                    layout.entries,
                )),
                None,
            )
        };
        if let Some(data) = &norm {
            write(session, &op.buffers["input"], 0, &bytes_u16(&data.input))?;
            write(session, &op.buffers["gate"], 0, &bytes_u16(&data.gate))?;
            write(session, &op.buffers["weight"], 0, &bytes_u16(&data.weight))?;
            write(
                session,
                &op.buffers["live_rows"],
                0,
                &(live as u32).to_ne_bytes(),
            )?;
        }
        if let Some(data) = &conv {
            write(session, &op.buffers["input"], 0, &bytes_u16(&data.input))?;
            write(session, &op.buffers["weight"], 0, &bytes_u16(&data.weight))?;
        }
        if let Some(data) = &delta {
            for (name, bytes) in [
                ("qkv", bytes_u16(&data.qkv)),
                ("a", bytes_u16(&data.a)),
                ("b", bytes_u16(&data.b)),
                ("a_log", bytes_f32(&data.a_log)),
                ("dt_bias", bytes_f32(&data.dt_bias)),
            ] {
                write(session, &op.buffers[name], 0, &bytes)?;
            }
        }
        if !layout.norm {
            write(
                session,
                &op.buffers["live_sequences"],
                0,
                &(live as u32).to_ne_bytes(),
            )?;
        }
        if layout.verify {
            write(session, &op.buffers["cu_rows"], 0, &bytes_i32(&cu))?;
        }
        // The state image: live decode slots are read and rewritten, live
        // checkpoints only written; the entries of sequences past the live
        // count start poisoned and must stay poisoned.
        let mut checks = Replay::new(session).output(
            "output",
            &op.buffers["output"],
            WriteMap::of(DType::Bf16).prefix(live_rows * layout.width)?,
        );
        let initial = conv
            .as_ref()
            .map(|data| bytes_u16(&data.pool))
            .or_else(|| delta.as_ref().map(|data| bytes_f32(&data.pool)));
        let mut pool_before = initial.clone();
        if let Some(mut image) = initial {
            let (mut written, mut updated) = (Vec::new(), Vec::new());
            for s in 0..BUCKET {
                let checkpoints: &[i32] = if layout.verify {
                    &layout.checkpoint_ids[s * max..(s + 1) * max]
                } else {
                    &[]
                };
                if s >= live {
                    image[layout.entry(layout.slot_ids[s])].fill(POISON);
                    if layout.verify {
                        for &id in &checkpoints[..max] {
                            image[layout.entry(id)].fill(POISON);
                        }
                    }
                } else if layout.verify {
                    for &id in &checkpoints[..lengths[s]] {
                        written.push(layout.entry(id));
                    }
                    for &id in &checkpoints[lengths[s]..max] {
                        image[layout.entry(id)].fill(POISON);
                    }
                } else {
                    updated.push(layout.entry(layout.slot_ids[s]));
                }
            }
            pool_before = Some(image.clone());
            checks = checks.state(
                "state",
                &op.buffers["state"],
                image,
                WriteMap::of(layout.element).byte_ranges(written)?,
                updated,
            );
        }
        let labelled = |error: String| format!("{kind} live {live}: {error}");
        checks.eager(|| launch(&session.stream)).map_err(labelled)?;
        let output = u16s(checks.expected("output").ok_or("missing output")?);
        let after = checks.expected("state").map(<[u8]>::to_vec);
        oracle_checks(
            &mut report,
            &layout,
            kind,
            live,
            &lengths,
            &cu,
            (conv.as_ref(), delta.as_ref(), norm.as_ref()),
            pool_before.as_deref(),
            after.as_deref(),
            &output,
        )?;
        checks.graph(&graph).map_err(labelled)?;
        report.finish(session)?;
        let mut receipt = BTreeMap::new();
        receipt.insert("family".into(), json!("linear"));
        receipt.insert("kind".into(), json!(kind));
        receipt.insert("bucket_sequences".into(), json!(BUCKET));
        receipt.insert("bucket_rows".into(), json!(layout.row_capacity));
        receipt.insert("live".into(), json!(live));
        receipt.insert("live_rows".into(), json!(live_rows));
        receipt.insert("correct".into(), json!(true));
        checks.record(&mut receipt);
        session.receipt("live_replay", receipt)?;
    }
    Ok(())
}

/// The eager launch against the recurrent oracles: every live sequence's
/// output rows and the state it leaves (the rewritten slot for decode, one
/// checkpoint per row for verify).
#[allow(clippy::too_many_arguments)]
fn oracle_checks(
    report: &mut Report,
    layout: &Layout,
    kind: &str,
    live: usize,
    lengths: &[usize],
    cu: &[i32],
    data: (Option<&ConvCase>, Option<&DeltaCase>, Option<&NormCase>),
    pool_before: Option<&[u8]>,
    after: Option<&[u8]>,
    output: &[u16],
) -> LabResult<()> {
    let g = geometry();
    let max = verify_row_limit() as usize;
    let width = layout.width;
    let query = phase_query(&symbol(kind));
    match data {
        (_, _, Some(norm)) => {
            if live > 0 {
                let rows = live;
                let gate = (0..rows)
                    .flat_map(|r| decode(&norm.gate[r * g.z_stride..r * g.z_stride + width]))
                    .collect::<Vec<_>>();
                let expected = recurrent_oracle::gated_rmsnorm(
                    query,
                    &decode(&norm.input[..rows * width]),
                    &gate,
                    &decode(&norm.weight),
                )
                .map_err(|e| e.to_string())?;
                report.check(
                    "output",
                    &decode(&output[..rows * width]),
                    &expected,
                    DType::Bf16,
                );
            }
        }
        (Some(conv), _, _) => {
            let weight = decode(&conv.weight);
            let before = u16s(pool_before.ok_or("missing pool")?);
            let after = u16s(after.ok_or("missing pool")?);
            let n = HISTORY * g.channels;
            for s in 0..live {
                let start = cu[s] as usize;
                let len = lengths[s];
                let slot = layout.slot_ids[s] as usize;
                let expected = recurrent_oracle::causal_conv1d(
                    query,
                    &conv.rows(start, len),
                    &weight,
                    &decode(&before[slot * n..(slot + 1) * n]),
                )
                .map_err(|e| e.to_string())?;
                report.check(
                    &format!("seq{s}"),
                    &decode(&output[start * width..(start + len) * width]),
                    &expected.output,
                    DType::Bf16,
                );
                if layout.verify {
                    for (r, state) in expected.row_states.iter().enumerate() {
                        let id = layout.checkpoint_ids[s * max + r] as usize;
                        report.exact(
                            "checkpoint bits",
                            compare_state_bits(&after[id * n..(id + 1) * n], state),
                        );
                    }
                } else {
                    report.exact(
                        "state bits",
                        compare_state_bits(&after[slot * n..(slot + 1) * n], &expected.state),
                    );
                }
            }
        }
        (_, Some(delta), _) => {
            let before = f32s(pool_before.ok_or("missing pool")?);
            let after = f32s(after.ok_or("missing pool")?);
            let n = g.delta_state;
            for s in 0..live {
                let start = cu[s] as usize;
                let len = lengths[s];
                let slot = layout.slot_ids[s] as usize;
                let expected =
                    delta.oracle(query, start, len, &f64s(&before[slot * n..(slot + 1) * n]))?;
                report.check(
                    &format!("seq{s}_output"),
                    &decode(&output[start * width..(start + len) * width]),
                    &expected.output,
                    DType::Bf16,
                );
                if layout.verify {
                    for (r, state) in expected.row_states.iter().enumerate() {
                        let id = layout.checkpoint_ids[s * max + r] as usize;
                        report.check(
                            &format!("seq{s}_row{r}_state"),
                            &f64s(&after[id * n..(id + 1) * n]),
                            state,
                            DType::F32,
                        );
                    }
                } else {
                    report.check(
                        &format!("seq{s}_state"),
                        &f64s(&after[slot * n..(slot + 1) * n]),
                        &expected.state,
                        DType::F32,
                    );
                }
            }
        }
        _ => return Err("linear live case has no data".into()),
    }
    Ok(())
}
