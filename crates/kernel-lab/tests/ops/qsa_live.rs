//! One captured QSA step per bucket replayed at several live row counts, as
//! serving replays one graph per bucket with the live count read on device:
//! the key write, query, scores, top-k and attention of decode at a 32-row
//! bucket, verify at 256 rows and prefill at 2048 rows, with BF16 KV and
//! contexts up to 8,195 tokens. The attention grid takes the effective split
//! count (`Fixture::splits`) and the merge runs exactly when it exceeds one.
//! Every live count gets new sequences, contexts, page order, KV, queries and
//! history: live sequences cover exactly the live rows, and padding sequences
//! with their own slots, pages and positions cover the rest of the bucket.
//! The caches start poisoned and take every sequence's history eagerly, so
//! rows past the live count see valid caches, but the ring rows and block
//! keys those rows would write hold poison. The step's poisoned checks
//! (`Fixture::replay`) then run the eager launch and the graph replay, and
//! the outputs, workspace partials, ring rows and block keys of rows past
//! the live count must keep their poison. The eager step meets the QSA oracles (`check`) and the
//! graph replay matches it bit for bit. A poisoned check fails at once;
//! oracle misses are collected so every live count still replays, and fail
//! the test at the end.

use super::*;
use kernel_lab::poison::{POISON, live_counts};

/// The largest context of every bucket: the plans, the score stride and the
/// cos/sin table cover it.
const CONTEXT: usize = 8195;

/// Sequence contexts, taken in turn from a seeded start and never below a
/// sequence's rows. Long and short alternate, so adjacent sequences mix
/// sparse and dense rows; 2,051 and 2,052 tokens sit on the two sides of the
/// dense limit, and 1, 4 and 5 tokens hold no complete block, one block, and
/// one block and a tail.
const CONTEXTS: [usize; 12] = [8195, 1, 2052, 5, 4097, 17, 2051, 130, 3001, 4, 8192, 2055];

#[test]
fn qsa_chain_replays_one_graph_at_every_live_count() {
    gpu_test(8 << 30, |session| {
        let modules = Modules::load(session)?;
        let mut misses = Vec::new();
        for (phase, bucket) in [
            (AttentionPhase::Decode, 32),
            (AttentionPhase::Verify, 256),
            (AttentionPhase::Prefill, 2048),
        ] {
            misses.extend(replay_bucket(session, &modules, phase, bucket)?);
        }
        if misses.is_empty() {
            Ok(())
        } else {
            Err(format!("oracle misses:\n{}", misses.join("\n")))
        }
    })
    .expect("QSA live-count graph replay");
}

/// A decode step whose attention partial reads a stale live count (the
/// bucket's, as do the key write, query, scores and top-k before it, so every
/// row it reads is valid) while the merge reads the true count of 0: the
/// partial writes every row's split partials and the output stays untouched.
/// Unbounded workspace scratch accepts it; the live-row bound rejects it, and
/// passes the true step.
#[test]
fn qsa_stale_live_count_partial_fails_the_workspace_bound() {
    gpu_test(8 << 30, |session| {
        let modules = Modules::load(session)?;
        let bucket = 32;
        let (case, live_sequences) = live_case(AttentionPhase::Decode, bucket, 0);
        let needed = Capacity::of(&case, modules.topology.geometry);
        let capacity = Capacity {
            rows: bucket,
            sequences: needed.sequences,
            pages: needed.pages,
            context: CONTEXT,
        };
        let mut fixture = Fixture::allocate(session, &case, modules.topology, capacity)?;
        let kernels = Kernels::load(&modules, &case)?;
        let splits = fixture.splits(&kernels.attention.0);
        if splits < 2 {
            return Err("the negative control needs split partials".into());
        }
        stage(session, &mut fixture, &kernels, &case, 0, live_sequences)?;
        let bound: Vec<_> = std::iter::once(0..fixture.workspace_partials(&kernels)).collect();
        let checks = |bounded: bool| {
            let replay = Replay::new(session).output(
                "attention",
                &fixture.d_out,
                WriteMap::of(DType::Bf16).none()?,
            );
            Ok::<_, String>(if bounded {
                replay.scratch_within("attention workspace", &fixture.d_workspace, bound.clone())
            } else {
                replay.scratch("attention workspace", &fixture.d_workspace)
            })
        };
        let stream = &session.stream;
        checks(true)?
            .eager(|| fixture.launch_step(stream, &kernels))
            .map_err(|e| format!("the true step fails the workspace bound: {e}"))?;
        let set_live =
            |live: usize| poison::write(session, &fixture.d_live, 0, &(live as u32).to_ne_bytes());
        let stale = || -> LabResult<()> {
            set_live(bucket)?;
            fixture.launch_key_write(stream, &kernels, fixture.prepared)?;
            fixture.launch_query(stream, &kernels)?;
            fixture.launch_scores(stream, &kernels)?;
            fixture.launch_topk(stream, &kernels)?;
            fixture.launch_attention_partial(stream, &kernels)?;
            stream.synchronize().map_err(|e| e.to_string())?;
            set_live(0)?;
            fixture.launch_merge(stream, &kernels)
        };
        checks(false)?
            .eager(stale)
            .map_err(|e| format!("the unbounded check already fails: {e}"))?;
        let error = match checks(true)?.eager(stale) {
            Ok(()) => return Err("the workspace bound passed a stale live count".into()),
            Err(error) => error,
        };
        eprintln!("stale live count failed the workspace bound as required: {error}");
        let mut receipt = BTreeMap::from([
            ("case".into(), json!(case.name)),
            ("live".into(), json!(0)),
            ("stale_live".into(), json!(bucket)),
            ("splits".into(), json!(splits)),
            ("unbounded_check_passed".into(), json!(true)),
            ("bounded_check_failed".into(), json!(true)),
            ("bounded_error".into(), json!(error)),
            ("true_live_passed".into(), json!(true)),
        ]);
        fixture.topology.mark(&mut receipt);
        session.receipt("qsa_negative_control", receipt)?;
        Ok(())
    })
    .expect("QSA stale live count negative control");
}

/// The step at one live count and its number of live sequences: live
/// sequences cover exactly the first `live` rows and padding sequences the
/// rest of the bucket. Decode sequences have one row, verify items one to
/// eight and prefill chunks one to 1024; a generator seeded by the bucket and
/// the live count draws the row counts and the first context.
fn live_case(phase: AttentionPhase, bucket: usize, live: usize) -> (Case, usize) {
    let mut rng = Rng(0x11fe_5eed_0000_0001 ^ (bucket as u64) << 32 ^ (live as u64) << 8);
    let mut context = (rng.next() % CONTEXTS.len() as u64) as usize;
    let mut split = |rows: usize, sequences: &mut Vec<(usize, usize)>| {
        let mut left = rows;
        while left > 0 {
            let take = match phase {
                AttentionPhase::Decode => 1,
                AttentionPhase::Verify => 1 + (rng.next() % 8) as usize,
                AttentionPhase::Prefill => 1 + (rng.next() % 1024) as usize,
            }
            .min(left);
            let total = CONTEXTS[context % CONTEXTS.len()].max(take);
            context += 1;
            sequences.push((total - take, take));
            left -= take;
        }
    };
    let mut sequences = Vec::new();
    split(live, &mut sequences);
    let live_sequences = sequences.len();
    split(bucket - live, &mut sequences);
    let phase_name = format!("{phase:?}").to_lowercase();
    let case = Case {
        name: format!("{phase_name}_live{live}_of{bucket}"),
        phase,
        fp8: false,
        rounded: false,
        sequences,
        pattern: Pattern::Random,
    };
    (case, live_sequences)
}

/// Captures one bucket's step once and replays it at every live count, and
/// returns the oracle misses.
fn replay_bucket(
    session: &Session,
    modules: &Modules,
    phase: AttentionPhase,
    bucket: usize,
) -> LabResult<Vec<String>> {
    let cases: Vec<(usize, Case, usize)> = live_counts(bucket as u32)
        .into_iter()
        .map(|live| {
            let (case, live_sequences) = live_case(phase, bucket, live as usize);
            (live as usize, case, live_sequences)
        })
        .collect();
    // Every buffer at the bucket's capacity, allocated once: the graph keeps
    // their addresses, and each live count rewrites their contents.
    let mut capacity = Capacity {
        rows: bucket,
        sequences: 0,
        pages: 0,
        context: CONTEXT,
    };
    for (_, case, _) in &cases {
        let needed = Capacity::of(case, modules.topology.geometry);
        capacity.sequences = capacity.sequences.max(needed.sequences);
        capacity.pages = capacity.pages.max(needed.pages);
    }
    let first = &cases[0].1;
    let phase_name = format!("{phase:?}").to_lowercase();
    let mut fixture = Fixture::allocate(session, first, modules.topology, capacity)?;
    let kernels = Kernels::load(modules, first)?;
    let splits = fixture.splits(&kernels.attention.0);
    let workspace = kernels.attention.0.workspace_bytes(fixture.prepared);
    if fixture.prepared as usize != bucket || workspace > fixture.d_workspace.byte_len() as u64 {
        return Err(format!(
            "{phase_name}: a {bucket} row bucket prepares {} rows and needs {workspace} of the fixture's {} workspace bytes",
            fixture.prepared,
            fixture.d_workspace.byte_len()
        ));
    }
    let mut graph = None;
    let mut misses = Vec::new();
    for (live, case, live_sequences) in &cases {
        let (live, live_sequences) = (*live, *live_sequences);
        let started = std::time::Instant::now();
        stage(session, &mut fixture, &kernels, case, live, live_sequences)?;
        // One graph per bucket, captured at the first live count (the whole
        // bucket) with every row in place.
        if graph.is_none() {
            graph = Some(session.capture(|stream| fixture.launch_step(stream, &kernels))?);
        }
        let labelled = |error: String| format!("{} ({splits} splits): {error}", case.name);
        let mut checks = fixture.replay(session, &kernels)?;
        checks
            .eager(|| fixture.launch_step(&session.stream, &kernels))
            .map_err(labelled)?;
        let oracle = if live > 0 {
            check(session, &fixture)
        } else {
            Ok(BTreeMap::new())
        };
        checks
            .graph(graph.as_ref().ok_or("no captured graph")?)
            .map_err(labelled)?;
        let (mut receipt, miss) = match oracle {
            Ok(summary) => (summary, None),
            Err(miss) => (BTreeMap::new(), Some(miss)),
        };
        receipt.insert("family".into(), json!("qsa"));
        receipt.insert("case".into(), json!(case.name));
        receipt.insert("phase".into(), json!(phase_name));
        receipt.insert("bucket_rows".into(), json!(bucket));
        receipt.insert("live".into(), json!(live));
        receipt.insert("live_sequences".into(), json!(live_sequences));
        receipt.insert("sequences".into(), json!(case.sequences));
        receipt.insert("context".into(), json!(CONTEXT));
        receipt.insert("kv_dtype".into(), json!("bf16"));
        receipt.insert("splits".into(), json!(splits));
        receipt.insert("merge".into(), json!(splits > 1));
        receipt.insert("correct".into(), json!(miss.is_none()));
        if let Some(miss) = &miss {
            receipt.insert("oracle_miss".into(), json!(miss));
        }
        checks.record(&mut receipt);
        fixture.topology.mark(&mut receipt);
        session.receipt("live_replay", receipt)?;
        eprintln!(
            "{}: {live} of {bucket} rows live in {live_sequences} sequences, {splits} splits, {} and graph bitwise in {:.1}s",
            case.name,
            if miss.is_none() {
                "oracles"
            } else {
                "an oracle miss"
            },
            started.elapsed().as_secs_f64()
        );
        misses.extend(miss);
    }
    Ok(misses)
}

/// Uploads one live count's step into the captured buffers: `load` draws new
/// sequences, KV, page order and queries; the caches start poisoned and take
/// every sequence's history, written eagerly; then every row of the bucket
/// goes up, padding rows included, and the live count. The host view keeps
/// the live sequences and rows, so the write maps, states and oracles cover
/// exactly them.
fn stage(
    session: &Session,
    fixture: &mut Fixture,
    kernels: &Kernels,
    case: &Case,
    live: usize,
    live_sequences: usize,
) -> LabResult<()> {
    fixture.load(session, case)?;
    let heads = fixture.topology.heads;
    let all = std::mem::take(&mut fixture.step);
    let live_rows: usize = fixture.sequences[..live_sequences]
        .iter()
        .map(|sequence| sequence.rows)
        .sum();
    if all.rows != fixture.padded_rows || live_rows != live {
        return Err(format!(
            "{}: {} rows fill a {} row bucket and the live sequences hold {live_rows} of {live} live rows",
            case.name, all.rows, fixture.padded_rows
        ));
    }
    // The padding sequences' history too, so rows past the live count see
    // valid caches, such as the ring positions a leader pools from. The
    // blocks those rows would complete still hold poison, and the ring rows
    // they would only write are poisoned again; ring rows a leader would read
    // first keep their history (`Fixture::replay` models the live rows the
    // same way).
    poison::poison(
        session,
        &[
            &fixture.d_ring,
            &fixture.d_ring_positions,
            &fixture.d_block_keys,
        ],
    )?;
    fixture.write_history(session, kernels)?;
    let (mut keys, mut rope) = (Vec::new(), Vec::new());
    for sequence in &fixture.sequences[live_sequences..] {
        let (first, end) = (sequence.cached, sequence.total());
        let start = first - first % BLOCK;
        let pooled = start < first && start + BLOCK - 1 < end;
        for position in end.saturating_sub(RING).max(first)..end {
            if pooled && (start..first).any(|read| read % RING == position % RING) {
                continue;
            }
            let row = sequence.slot as usize * RING + position % RING;
            keys.push(row * INDEX_DIM * 2..(row + 1) * INDEX_DIM * 2);
            rope.push(row * 12..(row + 1) * 12);
        }
    }
    poison::fill(session, &fixture.d_ring, &keys, &[POISON])?;
    poison::fill(session, &fixture.d_ring_positions, &rope, &[POISON])?;
    fixture.step = Step {
        rows: live,
        row_batch: all.row_batch[..live].to_vec(),
        positions: all.positions[..live].to_vec(),
        rope: all.rope[..live].to_vec(),
        projection: all.projection[..live * WIDTH].to_vec(),
        q: all.q[..live * heads * DIM].to_vec(),
    };
    fixture.sequences.truncate(live_sequences);
    fixture.upload_rows(
        session,
        &all.row_batch,
        &all.positions,
        &all.rope,
        &all.projection,
    )?;
    poison::write(session, &fixture.d_live, 0, &(live as u32).to_ne_bytes())
}
