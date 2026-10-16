//! The decode item entries (n-gram and PLE convolution over every bucket
//! item in one launch) against one per-sequence launch per item: outputs,
//! row ids and staged state must match bit for bit, items with no row stay
//! untouched, and PDL graph replay repeats the result. The bucket's rows are
//! the contract's own-op oracles' (`own_ops::{ngram, ple}`): the n-gram bit
//! for bit, the convolution within its bounds and its state exactly.
//! Buckets run at 8 items and, for the wide Qwen decode steps, at 33
//! to 256 live items over the power-of-two bucket, whose tail items have no
//! row.

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::row::hyper_connection::Entry;
use model_qwen38_contract::{
    own_ops::{ItemForm, ngram as own, ple},
    sm120::row as qwen38,
};
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison as replay_checks;
use kernel_lab::poison::{Phase, WriteMap};

/// Bucket items; items 2 and 6 have no row this step.
const Q_LEN: [u32; 8] = [1, 1, 0, 1, 1, 1, 0, 1];
/// Live items of the wide decode steps.
const WIDE_COUNTS: [usize; 6] = [33, 64, 100, 128, 255, 256];
/// The wide step whose bucket also replays as a PDL graph.
const WIDE_REPLAY: usize = 255;
const WIDE: usize = 10240;
const SENTINEL: u8 = 0xff;

fn entry(symbol: &str) -> Entry {
    qwen38::hyper_connection_rows()
        .into_iter()
        .find(|e| e.symbol == symbol)
        .unwrap()
}

fn raw<T: Copy>(values: &[T], bytes: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|&v| bytes(v)).collect()
}

fn bf16_codes(len: usize, seed: usize) -> Vec<u8> {
    raw(
        &(0..len)
            .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
            .collect::<Vec<_>>(),
        |v| v.to_ne_bytes().to_vec(),
    )
}

struct Buffers<'a> {
    session: &'a Session,
    named: BTreeMap<String, (DeviceBuffer, usize)>,
}

impl<'a> Buffers<'a> {
    fn new(session: &'a Session) -> Self {
        Self {
            session,
            named: BTreeMap::new(),
        }
    }

    fn put(&mut self, name: &str, data: &[u8]) -> LabResult<()> {
        let buffer = self.session.allocate(data.len())?;
        buffer
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.named.insert(name.into(), (buffer, data.len()));
        Ok(())
    }

    fn get(&self, name: &str) -> LabResult<Vec<u8>> {
        let (buffer, len) = &self.named[name];
        let mut out = vec![0; *len];
        buffer
            .copy_to_bytes(&mut out, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }

    fn checks(
        &self,
        q_len: &[u32],
        specs: &[(&str, usize, usize, DType)],
    ) -> LabResult<replay_checks::Checks<'_>> {
        let mut checks = replay_checks::Checks::new(self.session);
        for &(name, stride, written, element) in specs {
            let map = WriteMap::of(element).byte_ranges(
                q_len
                    .iter()
                    .enumerate()
                    .filter(|&(_, &rows)| rows != 0)
                    .map(|(item, _)| item * stride..item * stride + written),
            )?;
            checks = checks.output(name, &self.named[name].0, map);
        }
        Ok(checks)
    }

    /// Launch `entry` with its pointer parameters bound by name through
    /// `names` (parameter to buffer) and `scalars`.
    fn launch(
        &self,
        module: &Module,
        entry: &Entry,
        rows: u32,
        names: &BTreeMap<&str, String>,
        scalars: &BTreeMap<&str, Argument>,
        pdl: bool,
    ) -> LabResult<()> {
        let ids: BTreeMap<_, _> = self
            .named
            .keys()
            .enumerate()
            .map(|(i, n)| (n.clone(), i as u64 + 1))
            .collect();
        let args = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } => {
                    let buffer = names.get(name.as_str()).cloned().unwrap_or(name.clone());
                    ids.get(&buffer)
                        .map(|&id| Argument::DeviceBuffer(id))
                        .ok_or(format!("no buffer {buffer}"))
                }
                EntryParam::U32 { name } | EntryParam::U64 { name } => {
                    Ok(scalars[name.as_str()].clone())
                }
                _ => Err("unsupported parameter".to_owned()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = entry.pack_arguments(args).map_err(|e| e.to_string())?;
        let mut plan = entry.plan(rows, rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        let function = module.function(&entry.symbol).map_err(|e| e.to_string())?;
        launch::launch(
            &self.session.context,
            &function,
            &self.session.stream,
            plan,
            &args,
            &launch::Bindings {
                buffers: self
                    .named
                    .values()
                    .enumerate()
                    .map(|(i, (b, _))| (i as u64 + 1, b))
                    .collect(),
            },
        )
    }
}

/// Byte ranges `[item * stride, item * stride + len)` of every item.
fn item(bytes: &[u8], index: usize, stride: usize, len: usize) -> &[u8] {
    &bytes[index * stride..index * stride + len]
}

/// Row counts of a wide step's bucket: `count` live items, every ninth
/// from the fifth with no row this step, then the bucket's tail with none.
fn wide_q_len(count: usize) -> Vec<u32> {
    (0..count.next_power_of_two())
        .map(|i| u32::from(i < count && i % 9 != 4))
        .collect()
}

/// Runs both item entries over one bucket of `q_len` row counts (its
/// length is the bucket) against a per-sequence launch per item with a row,
/// then, when `replay`, replays two item launches as a PDL graph. Returns
/// the n-gram ranks checked.
fn items_match(session: &Session, q_len: &[u32], replay: bool) -> LabResult<Vec<usize>> {
    use kernel_core::ngram_oracle as ng;
    let module = session
        .context
        .load_module(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/row_qwen38.cubin"
        )))
        .map_err(|e| e.to_string())?;
    let items = q_len.len();
    let q_len_bytes = raw(q_len, |v| v.to_ne_bytes().to_vec());
    let one = 1_u32.to_ne_bytes();

    // n-gram: staged history words sit `entry` words apart. The rank's
    // local hash heads follow CHOPINFER_QWEN38_RANKS (4 at TP4), in the
    // topology's row cubin.
    let ranks = kernel_lab::qwen38_ranks()?;
    let tag = kernel_lab::qwen38_profile()?.tag();
    let local = kernel_lab::qwen38_shape()?.ngram_heads() as usize;
    let ngram_module = session.module(&format!("row_{tag}"))?;
    let ngram = entry(&format!("chop_row_{tag}_ngram_v1"));
    let ngram_items = entry(&format!("chop_row_{tag}_ngram_decode_v1"));
    let tables = ng::ngram_tables(248064, 3, 8, 17, 0, 1, 1).map_err(|e| e.to_string())?;
    // Lean runs (TP2) check every rank; TP4 rank 1.
    let lean = kernel_lab::qwen38_lean()?;
    let ngram_ranks = if lean {
        (0..ranks as usize).collect()
    } else {
        vec![1_usize]
    };
    let mut failures = replay_checks::Failures::default();
    for rank in ngram_ranks.iter().copied() {
        let result = (|| -> LabResult<()> {
            let start = tables.offsets[rank * local];
            let table_rows = tables.sizes[rank * local..rank * local + local]
                .iter()
                .sum::<u64>();
            let table: Vec<u8> = (0..table_rows as usize * 160)
                .map(|i| (i % 251) as u8)
                .collect();
            let entry_words = 5_usize;
            let tokens: Vec<u32> = (0..items as u32)
                .map(|i| if i == 3 { 248044 } else { 100 + i })
                .collect();
            let histories: Vec<u32> = (0..items * entry_words)
                .map(|i| match (i / entry_words, i % entry_words) {
                    (4, _) => 248044,
                    (item, word) => 20 + (item * 7 + word) as u32,
                })
                .collect();
            let scalars = BTreeMap::from([
                ("rank", Argument::U32(rank as u32)),
                ("table_start", Argument::U64(start)),
                ("table_rows", Argument::U64(table_rows)),
                ("entry", Argument::U64(entry_words as u64)),
            ]);
            let mut b = Buffers::new(session);
            for (name, values) in [
                ("multipliers", &tables.multipliers),
                ("sizes", &tables.sizes),
                ("offsets", &tables.offsets),
            ] {
                b.put(name, &raw(values, |v| v.to_ne_bytes().to_vec()))?;
            }
            b.put("table", &table)?;
            b.put("scale", &f32_to_bf16(0.125).to_ne_bytes())?;
            b.put("q_len", &q_len_bytes)?;
            b.put("tokens", &raw(&tokens, |v| v.to_ne_bytes().to_vec()))?;
            b.put("history", &raw(&histories, |v| v.to_ne_bytes().to_vec()))?;
            b.put("output", &vec![SENTINEL; items * local * 160 * 2])?;
            b.put("row_ids", &vec![SENTINEL; items * local * 8])?;
            b.put("history_out", &vec![SENTINEL; items * entry_words * 4])?;
            b.put("one", &one)?;
            for i in 0..items {
                b.put(&format!("tokens{i}"), &tokens[i].to_ne_bytes())?;
                b.put(
                    &format!("history{i}"),
                    &raw(&histories[i * entry_words..i * entry_words + 2], |v| {
                        v.to_ne_bytes().to_vec()
                    }),
                )?;
                b.put(&format!("output{i}"), &vec![SENTINEL; local * 160 * 2])?;
                b.put(&format!("row_ids{i}"), &vec![SENTINEL; local * 8])?;
                b.put(&format!("history_out{i}"), &[SENTINEL; 2 * 4])?;
            }
            for (name, len) in [
                ("output_graph", items * local * 160 * 2),
                ("row_ids_graph", items * local * 8),
                ("history_out_graph", items * entry_words * 4),
            ] {
                b.put(name, &vec![SENTINEL; len])?;
            }
            let names = BTreeMap::new();
            let mut checks = b.checks(
                q_len,
                &[
                    ("output", local * 160 * 2, local * 160 * 2, DType::Bf16),
                    ("row_ids", local * 8, local * 8, DType::I64),
                    ("history_out", entry_words * 4, 8, DType::I32),
                ],
            )?;
            checks.phase(Phase::Eager, false, || {
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &names,
                    &scalars,
                    false,
                )
            })?;
            for (i, &rows) in q_len.iter().enumerate() {
                if rows == 0 {
                    continue;
                }
                let names = BTreeMap::from([
                    ("tokens", format!("tokens{i}")),
                    ("history", format!("history{i}")),
                    ("output", format!("output{i}")),
                    ("row_ids", format!("row_ids{i}")),
                    ("history_out", format!("history_out{i}")),
                    ("live_rows", "one".into()),
                ]);
                let mut reference = replay_checks::Checks::new(session)
                    .output(
                        "reference ngram",
                        &b.named[&format!("output{i}")].0,
                        WriteMap::of(DType::Bf16).prefix(local * 160)?,
                    )
                    .output(
                        "reference IDs",
                        &b.named[&format!("row_ids{i}")].0,
                        WriteMap::of(DType::I64).prefix(local)?,
                    )
                    .output(
                        "reference history",
                        &b.named[&format!("history_out{i}")].0,
                        WriteMap::of(DType::I32).prefix(2)?,
                    );
                reference.phase(Phase::Eager, false, || {
                    b.launch(&ngram_module, &ngram, 1, &names, &scalars, false)
                })?;
            }
            let (output_bytes, row_bytes) = (local * 160 * 2, local * 8);
            let check_ngram = |b: &Buffers| -> LabResult<()> {
                let (output, row_ids, history_out) =
                    (b.get("output")?, b.get("row_ids")?, b.get("history_out")?);
                for (i, &rows) in q_len.iter().enumerate() {
                    let expected = if rows == 0 {
                        [
                            vec![SENTINEL; output_bytes],
                            vec![SENTINEL; row_bytes],
                            vec![SENTINEL; 8],
                        ]
                    } else {
                        [
                            b.get(&format!("output{i}"))?,
                            b.get(&format!("row_ids{i}"))?,
                            b.get(&format!("history_out{i}"))?,
                        ]
                    };
                    assert_eq!(
                        item(&output, i, output_bytes, output_bytes),
                        expected[0],
                        "n-gram output {i}"
                    );
                    assert_eq!(
                        item(&row_ids, i, row_bytes, row_bytes),
                        expected[1],
                        "n-gram rows {i}"
                    );
                    assert_eq!(
                        item(&history_out, i, entry_words * 4, 8),
                        expected[2],
                        "n-gram history {i}"
                    );
                }
                Ok(())
            };
            check_ngram(&b)?;
            // The bucket's rows are the contract's n-gram of every item.
            let op = own::Ngram {
                orders: 3,
                heads_per_order: 8,
                head_width: 160,
                local_heads: local as u32,
                rank: rank as u32,
                eos: 248044,
                entry_words: entry_words as u64,
                writes: own::NgramWrites::Gather,
                items: ItemForm::Decode,
            };
            let counts = q_len.iter().map(|&rows| rows as i32).collect::<Vec<_>>();
            let shard = own::Shard {
                start,
                rows: table_rows,
                codes: Some(&table),
                scale: f32_to_bf16(0.125),
            };
            let oracle = own::ngram_launch(
                &op,
                &tables,
                &tokens,
                &histories,
                &counts,
                None,
                None,
                items as u32,
                shard,
            )
            .map_err(|e| format!("n-gram oracle: {}", e.0))?;
            let (output, row_ids, history_out) =
                (b.get("output")?, b.get("row_ids")?, b.get("history_out")?);
            for (at, expected) in &oracle.written {
                let i = at.item as usize;
                let ids = expected
                    .row_ids
                    .iter()
                    .flat_map(|id| id.to_ne_bytes())
                    .collect::<Vec<_>>();
                assert_eq!(
                    item(&output, i, output_bytes, output_bytes),
                    raw(&expected.output, |v| v.to_ne_bytes().to_vec()),
                    "n-gram output {i} against the oracle"
                );
                assert_eq!(
                    item(&row_ids, i, row_bytes, row_bytes),
                    ids,
                    "n-gram rows {i}"
                );
                assert_eq!(
                    item(&history_out, i, entry_words * 4, 8),
                    raw(&expected.history, |v| v.to_ne_bytes().to_vec()),
                    "n-gram history {i}"
                );
            }
            if !replay {
                return Ok(());
            }
            let second_names = BTreeMap::from([
                ("output", "output_graph".into()),
                ("row_ids", "row_ids_graph".into()),
                ("history_out", "history_out_graph".into()),
            ]);
            let mut second = b.checks(
                q_len,
                &[
                    (
                        "output_graph",
                        local * 160 * 2,
                        local * 160 * 2,
                        DType::Bf16,
                    ),
                    ("row_ids_graph", local * 8, local * 8, DType::I64),
                    ("history_out_graph", entry_words * 4, 8, DType::I32),
                ],
            )?;
            second.phase(Phase::Eager, false, || {
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    false,
                )
            })?;
            for name in ["output", "row_ids", "history_out"] {
                assert_eq!(
                    b.get(&format!("{name}_graph"))?,
                    b.get(name)?,
                    "second ngram call {name}"
                );
            }
            let graph = session.capture(|_| {
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &names,
                    &scalars,
                    true,
                )?;
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    true,
                )
            })?;
            let mut calls = vec![checks, second];
            replay_checks::phase_all(&mut calls, Phase::Graph, true, || {
                graph.launch(&session.stream).map_err(|e| e.to_string())
            })?;
            replay_checks::phase_all(&mut calls, Phase::Pdl, true, || {
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &names,
                    &scalars,
                    true,
                )?;
                b.launch(
                    &ngram_module,
                    &ngram_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    true,
                )
            })?;
            check_ngram(&b)?;
            for call in calls {
                let mut fields = BTreeMap::from([
                    ("kind".into(), json!("ngram")),
                    ("items".into(), json!(items)),
                    ("rank".into(), json!(rank)),
                ]);
                call.record(&mut fields);
                session.receipt("ple-items-replay", fields)?;
            }
            Ok(())
        })();
        failures.check(&format!("ngram rank {rank} items {items}"), result);
    }

    let convolution = (|| -> LabResult<()> {
        // PLE convolution: staged state rows sit `entry` elements apart.
        let conv = entry("chop_row_qwen38_ple_conv_decode_v1");
        let conv_items = entry("chop_row_qwen38_ple_conv_items_decode_v1");
        let state_len = 9 * WIDE;
        let entry_elements = state_len + 64;
        let scalars = BTreeMap::from([("entry", Argument::U64(entry_elements as u64))]);
        let input = bf16_codes(items * WIDE, 1);
        let states = bf16_codes(items * entry_elements, 3);
        let mut b = Buffers::new(session);
        b.put("weight", &bf16_codes(4 * WIDE, 2))?;
        b.put("q_len", &q_len_bytes)?;
        b.put("input", &input)?;
        b.put("state", &states)?;
        b.put("output", &vec![SENTINEL; items * WIDE * 2])?;
        b.put("state_out", &vec![SENTINEL; items * entry_elements * 2])?;
        b.put("one", &one)?;
        for i in 0..items {
            b.put(&format!("input{i}"), item(&input, i, WIDE * 2, WIDE * 2))?;
            b.put(
                &format!("state{i}"),
                item(&states, i, entry_elements * 2, state_len * 2),
            )?;
            b.put(&format!("output{i}"), &vec![SENTINEL; WIDE * 2])?;
            b.put(&format!("state_out{i}"), &vec![SENTINEL; state_len * 2])?;
        }
        b.put("output_graph", &vec![SENTINEL; items * WIDE * 2])?;
        b.put(
            "state_out_graph",
            &vec![SENTINEL; items * entry_elements * 2],
        )?;
        let names = BTreeMap::new();
        let mut checks = b.checks(
            q_len,
            &[
                ("output", WIDE * 2, WIDE * 2, DType::Bf16),
                ("state_out", entry_elements * 2, state_len * 2, DType::Bf16),
            ],
        )?;
        checks.phase(Phase::Eager, false, || {
            b.launch(&module, &conv_items, items as u32, &names, &scalars, false)
        })?;
        for (i, &rows) in q_len.iter().enumerate() {
            if rows == 0 {
                continue;
            }
            let names = BTreeMap::from([
                ("input", format!("input{i}")),
                ("state", format!("state{i}")),
                ("output", format!("output{i}")),
                ("state_out", format!("state_out{i}")),
                ("live_rows", "one".into()),
            ]);
            let mut reference = replay_checks::Checks::new(session)
                .output(
                    "reference convolution",
                    &b.named[&format!("output{i}")].0,
                    WriteMap::of(DType::Bf16).prefix(WIDE)?,
                )
                .output(
                    "reference state",
                    &b.named[&format!("state_out{i}")].0,
                    WriteMap::of(DType::Bf16).prefix(state_len)?,
                );
            reference.phase(Phase::Eager, false, || {
                b.launch(&module, &conv, 1, &names, &scalars, false)
            })?;
        }
        let check_conv = |b: &Buffers| -> LabResult<()> {
            let (output, state_out) = (b.get("output")?, b.get("state_out")?);
            for (i, &rows) in q_len.iter().enumerate() {
                let expected = if rows == 0 {
                    [vec![SENTINEL; WIDE * 2], vec![SENTINEL; state_len * 2]]
                } else {
                    [
                        b.get(&format!("output{i}"))?,
                        b.get(&format!("state_out{i}"))?,
                    ]
                };
                assert_eq!(
                    item(&output, i, WIDE * 2, WIDE * 2),
                    expected[0],
                    "conv output {i}"
                );
                assert_eq!(
                    item(&state_out, i, entry_elements * 2, state_len * 2),
                    expected[1],
                    "conv state {i}"
                );
            }
            Ok(())
        };
        check_conv(&b)?;
        // Each item's row within the contract's bounds, its state exact.
        let op = ple::PleConv {
            layout: ple::PleLayout {
                wide: WIDE as u32,
                conv_rows: 9,
                eos: 248044,
            },
            taps: 4,
            dilation: 3,
            items: ItemForm::Decode,
        };
        let values = |bytes: &[u8]| {
            bytes
                .chunks_exact(2)
                .map(|pair| u16::from_ne_bytes([pair[0], pair[1]]))
                .collect::<Vec<_>>()
        };
        let weight = values(&bf16_codes(4 * WIDE, 2));
        let (output, state_out) = (b.get("output")?, b.get("state_out")?);
        for (i, _) in q_len.iter().enumerate().filter(|&(_, &rows)| rows != 0) {
            let oracle = ple::ple_conv_item(
                &op,
                &values(item(&input, i, WIDE * 2, WIDE * 2)),
                &weight,
                &values(item(&states, i, entry_elements * 2, state_len * 2)),
                1,
            )
            .map_err(|e| format!("convolution oracle: {}", e.0))?;
            oracle
                .output
                .check_bf16(&values(item(&output, i, WIDE * 2, WIDE * 2)))
                .map_err(|failure| format!("conv output {i} outside the oracle: {failure:?}"))?;
            assert_eq!(
                values(item(&state_out, i, entry_elements * 2, state_len * 2)),
                oracle.state,
                "conv state {i} against the oracle"
            );
        }
        if replay {
            let second_names = BTreeMap::from([
                ("output", "output_graph".into()),
                ("state_out", "state_out_graph".into()),
            ]);
            let mut second = b.checks(
                q_len,
                &[
                    ("output_graph", WIDE * 2, WIDE * 2, DType::Bf16),
                    (
                        "state_out_graph",
                        entry_elements * 2,
                        state_len * 2,
                        DType::Bf16,
                    ),
                ],
            )?;
            second.phase(Phase::Eager, false, || {
                b.launch(
                    &module,
                    &conv_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    false,
                )
            })?;
            for name in ["output", "state_out"] {
                assert_eq!(
                    b.get(&format!("{name}_graph"))?,
                    b.get(name)?,
                    "second convolution call {name}"
                );
            }
            let graph = session.capture(|_| {
                b.launch(&module, &conv_items, items as u32, &names, &scalars, true)?;
                b.launch(
                    &module,
                    &conv_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    true,
                )
            })?;
            let mut calls = vec![checks, second];
            replay_checks::phase_all(&mut calls, Phase::Graph, true, || {
                graph.launch(&session.stream).map_err(|e| e.to_string())
            })?;
            replay_checks::phase_all(&mut calls, Phase::Pdl, true, || {
                b.launch(&module, &conv_items, items as u32, &names, &scalars, true)?;
                b.launch(
                    &module,
                    &conv_items,
                    items as u32,
                    &second_names,
                    &scalars,
                    true,
                )
            })?;
            check_conv(&b)?;
            for call in calls {
                let mut fields = BTreeMap::from([
                    ("kind".into(), json!("convolution")),
                    ("items".into(), json!(items)),
                ]);
                call.record(&mut fields);
                session.receipt("ple-items-replay", fields)?;
            }
        }
        Ok(())
    })();
    failures.check("convolution items", convolution);
    failures.finish()?;
    Ok(ngram_ranks)
}

#[test]
fn qwen38_ple_items_match_per_item_launches() {
    gpu_test(64 << 20, |session| {
        let ngram_ranks = items_match(session, &Q_LEN, true)?;
        let (ranks, lean) = (kernel_lab::qwen38_ranks()?, kernel_lab::qwen38_lean()?);
        let tag = kernel_lab::qwen38_profile()?.tag();
        let mut receipt = BTreeMap::from([
            (
                "symbols".into(),
                json!([
                    format!("chop_row_{tag}_ngram_decode_v1"),
                    "chop_row_qwen38_ple_conv_items_decode_v1"
                ]),
            ),
            ("q_len".into(), json!(Q_LEN)),
            (
                "bitwise_equal_to_per_item".into(),
                json!([
                    format!("chop_row_{tag}_ngram_v1"),
                    "chop_row_qwen38_ple_conv_decode_v1"
                ]),
            ),
            ("graph_determinism_pdl".into(), json!(true)),
        ]);
        if ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(ranks));
        }
        if lean {
            receipt.insert("lean".into(), json!(true));
            receipt.insert("ngram_ranks".into(), json!(ngram_ranks));
        }
        session.receipt("qwen38-ple-items-g1", receipt)?;
        Ok(())
    })
    .expect("Qwen PLE items G1");
}

/// Qwen decode buckets of up to 256 items. At 33, 64, 100, 128, 255
/// and 256 live items, each launched over its power-of-two bucket (the tail
/// items have no row), the item entries match one per-sequence launch per
/// item bit for bit and leave the items with no row untouched; the
/// 255-item step's 256-item bucket also replays as a PDL graph.
#[test]
fn qwen38_ple_items_wide_buckets_match_per_item_launches() {
    // A 256-item bucket stages 0.4 MB of convolution state and output per
    // item, twice (the bucket's and the per-item launches').
    gpu_test(512 << 20, |session| {
        let mut ngram_ranks = Vec::new();
        let mut failures = replay_checks::Failures::default();
        for count in WIDE_COUNTS {
            match items_match(session, &wide_q_len(count), count == WIDE_REPLAY) {
                Ok(ranks) => ngram_ranks = ranks,
                Err(error) => failures.check(&format!("wide items {count}"), Err(error)),
            }
        }
        failures.finish()?;
        let (ranks, lean) = (kernel_lab::qwen38_ranks()?, kernel_lab::qwen38_lean()?);
        let tag = kernel_lab::qwen38_profile()?.tag();
        let mut receipt = BTreeMap::from([
            (
                "symbols".into(),
                json!([
                    format!("chop_row_{tag}_ngram_decode_v1"),
                    "chop_row_qwen38_ple_conv_items_decode_v1"
                ]),
            ),
            ("live_items".into(), json!(WIDE_COUNTS)),
            (
                "buckets".into(),
                json!(WIDE_COUNTS.map(usize::next_power_of_two)),
            ),
            (
                "bitwise_equal_to_per_item".into(),
                json!([
                    format!("chop_row_{tag}_ngram_v1"),
                    "chop_row_qwen38_ple_conv_decode_v1"
                ]),
            ),
            (
                "graph_determinism_pdl_live_items".into(),
                json!(WIDE_REPLAY),
            ),
        ]);
        if ranks != 4 {
            receipt.insert("qwen38_ranks".into(), json!(ranks));
        }
        if lean {
            receipt.insert("lean".into(), json!(true));
            receipt.insert("ngram_ranks".into(), json!(ngram_ranks));
        }
        session.receipt("qwen38-ple-items-wide", receipt)?;
        Ok(())
    })
    .expect("Qwen PLE items at wide buckets");
}
