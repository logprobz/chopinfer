//! The Qwen3.8 n-gram entries reading a rank's table from page-locked,
//! device-mapped host memory (`--ngram-placement host`) against the same
//! table in device memory: outputs, row ids and history must match bit for
//! bit, at a prefill phase of 8192 rows and at every decode bucket: 1 to 32
//! items, and 33 to 256 live Qwen items in the power-of-two bucket,
//! whose tail items have no row and must stay untouched. Each placement is
//! also timed; the report is one JSON line per shape.
//!
//! The table has the serving geometry of rank 1 (12.8 GB), so the test needs
//! that much device memory and pinned host memory.

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, DType};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use kernels_sm120::row::hyper_connection::Entry;
use model_qwen38_contract::sm120::row as qwen38;
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::json;
use std::collections::BTreeMap;
use std::time::Instant;

use kernel_lab::poison as replay_checks;
use kernel_lab::poison::{Phase, WriteMap};

const VOCAB: u64 = 248_064;
const RANK: usize = 1;
const PREFILL_ROWS: u32 = 8192;
const BUCKETS: [u32; 6] = [1, 2, 4, 8, 16, 32];
/// Live items of the wide decode steps, each in its power-of-two bucket.
const WIDE_COUNTS: [u32; 6] = [33, 64, 100, 128, 255, 256];
/// The wide step whose bucket also replays as a PDL graph.
const WIDE_REPLAY: u32 = 255;
/// Items of the largest decode bucket.
const MAX_ITEMS: usize = 256;
/// Bytes one item's row writes: rank 1's four local heads of 160 BF16
/// values, and their row ids.
const ITEM_OUTPUT: usize = 4 * 160 * 2;
const ITEM_ROW_IDS: usize = 4 * 8;
const ENTRY_WORDS: u64 = 5;

fn entry(symbol: &str) -> Entry {
    qwen38::hyper_connection_rows()
        .into_iter()
        .find(|e| e.symbol == symbol)
        .unwrap()
}

fn words<T: Copy>(values: &[T], bytes: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|&v| bytes(v)).collect()
}

/// Tokens spread over the vocabulary, so rows land all over the table.
fn tokens(count: usize, seed: u64) -> Vec<u32> {
    (0..count as u64)
        .map(|i| ((i * 2_654_435_761 + seed * 40_503) % VOCAB) as u32)
        .collect()
}

struct Launcher<'a> {
    session: &'a Session,
    module: &'a Module,
    buffers: BTreeMap<String, DeviceBuffer>,
    scalars: BTreeMap<&'static str, Argument>,
}

impl Launcher<'_> {
    fn put(&mut self, name: &str, data: &[u8]) -> LabResult<()> {
        let buffer = self.session.allocate(data.len())?;
        buffer
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.insert(name.into(), buffer);
        Ok(())
    }

    fn get(&self, name: &str) -> LabResult<Vec<u8>> {
        let buffer = &self.buffers[name];
        let mut out = vec![0; buffer.byte_len()];
        buffer
            .copy_to_bytes(&mut out, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }

    /// Launches `entry` over `rows` blocks with `table` bound to `table`.
    fn launch(&self, entry: &Entry, rows: u32, table: &DeviceBuffer) -> LabResult<()> {
        self.launch_bound(entry, rows, table, "")
    }

    fn launch_bound(
        &self,
        entry: &Entry,
        rows: u32,
        table: &DeviceBuffer,
        suffix: &str,
    ) -> LabResult<()> {
        let mut bound: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
        let args = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } => {
                    let buffer = if name == "table" {
                        table
                    } else {
                        let key = if ["output", "row_ids", "history_out"].contains(&name.as_str()) {
                            format!("{name}{suffix}")
                        } else {
                            name.clone()
                        };
                        self.buffers
                            .get(key.as_str())
                            .ok_or(format!("no buffer {name}"))?
                    };
                    let id = bound.len() as u64 + 1;
                    bound.insert(id, buffer);
                    Ok(Argument::DeviceBuffer(id))
                }
                EntryParam::U32 { name } | EntryParam::U64 { name } => {
                    Ok(self.scalars[name.as_str()].clone())
                }
                _ => Err("unsupported parameter".to_owned()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = entry.pack_arguments(args).map_err(|e| e.to_string())?;
        let plan = entry.plan(rows, rows).map_err(|e| e.to_string())?;
        let function = self
            .module
            .function(&entry.symbol)
            .map_err(|e| e.to_string())?;
        launch::launch(
            &self.session.context,
            &function,
            &self.session.stream,
            plan,
            &args,
            &launch::Bindings { buffers: bound },
        )
    }

    fn checks(
        &self,
        live: usize,
        decode: bool,
        suffix: &str,
    ) -> LabResult<replay_checks::Checks<'_>> {
        let mut checks = replay_checks::Checks::new(self.session);
        for (name, stride, written, element) in [
            ("output", ITEM_OUTPUT, ITEM_OUTPUT, DType::Bf16),
            ("row_ids", ITEM_ROW_IDS, ITEM_ROW_IDS, DType::I64),
            ("history_out", ENTRY_WORDS as usize * 4, 8, DType::I32),
        ] {
            let key = format!("{name}{suffix}");
            let map = if decode {
                WriteMap::of(element)
                    .byte_ranges((0..live).map(|item| item * stride..item * stride + written))?
            } else {
                WriteMap::of(element).byte_prefix(if name == "history_out" {
                    8 * usize::from(live != 0)
                } else {
                    live * stride
                })?
            };
            checks = checks.output(&key, &self.buffers[&key], map);
        }
        Ok(checks)
    }

    /// Mean microseconds per launch over `iterations` back-to-back launches.
    fn time(
        &self,
        entry: &Entry,
        rows: u32,
        table: &DeviceBuffer,
        iterations: u32,
    ) -> LabResult<f64> {
        self.launch(entry, rows, table)?;
        self.session
            .stream
            .synchronize()
            .map_err(|e| e.to_string())?;
        let start = Instant::now();
        for _ in 0..iterations {
            self.launch(entry, rows, table)?;
        }
        self.session
            .stream
            .synchronize()
            .map_err(|e| e.to_string())?;
        Ok(start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations))
    }
}

/// Runs the decode entry at each of `lives` live items (in the
/// power-of-two bucket, the tail items with no row) and, when
/// `prefill_phase`, the prefill entry, from the device and the mapped host
/// table; when a step has `replay` live items, two launches from the host
/// table also replay as a PDL graph. Returns the timing report.
fn host_table_matches(
    session: &Session,
    lives: &[u32],
    prefill_phase: bool,
    replay: Option<u32>,
) -> LabResult<Vec<serde_json::Value>> {
    use kernel_core::ngram_oracle as ng;
    let tables = ng::ngram_tables(VOCAB, 3, 8, 20_000_000, 0, 0, 128).map_err(|e| e.to_string())?;
    let start = tables.offsets[RANK * 4];
    let table_rows = tables.offsets[RANK * 4 + 4] - start;
    let table_bytes = table_rows as usize * 160;
    let module = session
        .context
        .load_module(include_bytes!(concat!(
            env!("OUT_DIR"),
            "/row_qwen38.cubin"
        )))
        .map_err(|e| e.to_string())?;
    let mut host_table = DeviceBuffer::allocate_mapped_host(&session.stream, table_bytes)
        .map_err(|e| e.to_string())?;
    let bytes = host_table
        .mapped_host_bytes_mut()
        .map_err(|e| e.to_string())?;
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (i % 251) as u8 ^ (i >> 20) as u8;
    }
    let device_table = session.allocate(table_bytes)?;
    device_table
        .copy_from_bytes(bytes, &session.stream)
        .map_err(|e| e.to_string())?;

    let prefill = entry("chop_row_qwen38_ngram_v1");
    let decode = entry("chop_row_qwen38_ngram_decode_v1");
    let rows = PREFILL_ROWS as usize;
    let mut lab = Launcher {
        session,
        module: &module,
        buffers: BTreeMap::new(),
        scalars: BTreeMap::from([
            ("rank", Argument::U32(RANK as u32)),
            ("table_start", Argument::U64(start)),
            ("table_rows", Argument::U64(table_rows)),
            ("entry", Argument::U64(ENTRY_WORDS)),
        ]),
    };
    for (name, values) in [
        ("multipliers", &tables.multipliers),
        ("sizes", &tables.sizes),
        ("offsets", &tables.offsets),
    ] {
        lab.put(name, &words(values, |v| v.to_ne_bytes().to_vec()))?;
    }
    lab.put("scale", &f32_to_bf16(0.125).to_ne_bytes())?;
    let token_words = tokens(rows, 1);
    lab.put("tokens", &words(&token_words, |v| v.to_ne_bytes().to_vec()))?;
    let history = tokens(MAX_ITEMS * ENTRY_WORDS as usize, 2);
    lab.put("history", &words(&history, |v| v.to_ne_bytes().to_vec()))?;
    lab.put("output", &vec![0; rows * 640 * 2])?;
    lab.put("row_ids", &vec![0; rows * 4 * 8])?;
    lab.put(
        "history_out",
        &vec![0; MAX_ITEMS * ENTRY_WORDS as usize * 4],
    )?;

    for name in ["output", "row_ids", "history_out"] {
        let len = lab.buffers[name].byte_len();
        lab.put(&format!("{name}_graph"), &vec![0xff; len])?;
    }
    let mut report = Vec::new();
    let outputs = |lab: &Launcher<'_>| -> LabResult<[Vec<u8>; 3]> {
        Ok([
            lab.get("output")?,
            lab.get("row_ids")?,
            lab.get("history_out")?,
        ])
    };
    // (entry, planned rows, live rows, timed launches)
    let mut shapes: Vec<(&Entry, u32, u32, u32)> = lives
        .iter()
        .map(|&live| (&decode, live.next_power_of_two(), live, 2000))
        .collect();
    if prefill_phase {
        shapes.push((&prefill, PREFILL_ROWS, PREFILL_ROWS, 50));
    }
    let mut failures = replay_checks::Failures::default();
    for (entry, rows, live, iterations) in shapes {
        let result = (|| -> LabResult<()> {
            lab.put(
                "live_rows",
                &words(&[rows], |v: u32| v.to_ne_bytes().to_vec()),
            )?;
            let q_len = (0..MAX_ITEMS as u32)
                .map(|item| u32::from(item < live))
                .collect::<Vec<_>>();
            lab.put("q_len", &words(&q_len, |v| v.to_ne_bytes().to_vec()))?;
            let mut checks = lab.checks(live as usize, entry.symbol == decode.symbol, "")?;
            checks.phase(Phase::Eager, false, || {
                lab.launch(entry, rows, &device_table)
            })?;
            let mut results = vec![outputs(&lab)?];
            checks.phase(Phase::Relaunch, false, || {
                lab.launch(entry, rows, &host_table)
            })?;
            results.push(outputs(&lab)?);
            assert!(
                results[0] == results[1],
                "{} at {live} of {rows} rows differs between device and host tables",
                entry.symbol
            );
            if entry.symbol == decode.symbol {
                // Live items wrote row ids; the others, output, row ids and
                // history included, are untouched.
                let [output, row_ids, history_out] = &results[0];
                for item in 0..rows as usize {
                    let at = |stride: usize, len: usize| item * stride..item * stride + len;
                    let written = row_ids[at(ITEM_ROW_IDS, ITEM_ROW_IDS)] != [0xff; ITEM_ROW_IDS];
                    let untouched = output[at(ITEM_OUTPUT, ITEM_OUTPUT)]
                        .iter()
                        .chain(&history_out[at(ENTRY_WORDS as usize * 4, 8)])
                        .all(|&byte| byte == 0xff);
                    let live_item = item < live as usize;
                    assert!(
                        written == live_item && (live_item || untouched),
                        "{} item {item} of {live} live in {rows}: written {written}",
                        entry.symbol
                    );
                }
            }
            if replay == Some(live) {
                let mut second = lab.checks(live as usize, true, "_graph")?;
                second.phase(Phase::Eager, false, || {
                    lab.launch_bound(entry, rows, &host_table, "_graph")
                })?;
                for (index, name) in ["output", "row_ids", "history_out"].iter().enumerate() {
                    assert_eq!(
                        lab.get(&format!("{name}_graph"))?,
                        results[1][index],
                        "second call {name}"
                    );
                }
                let graph = session.capture(|_| {
                    lab.launch(entry, rows, &host_table)?;
                    lab.launch_bound(entry, rows, &host_table, "_graph")
                })?;
                let mut calls = vec![checks, second];
                replay_checks::phase_all(&mut calls, Phase::Graph, true, || {
                    graph.launch(&session.stream).map_err(|e| e.to_string())
                })?;
                replay_checks::phase_all(&mut calls, Phase::Pdl, true, || {
                    lab.launch(entry, rows, &host_table)?;
                    lab.launch_bound(entry, rows, &host_table, "_graph")
                })?;
                assert!(
                    outputs(&lab)? == results[1],
                    "{} graph replay",
                    entry.symbol
                );
                for call in calls {
                    let mut fields = BTreeMap::from([
                        ("symbol".into(), json!(entry.symbol)),
                        ("live".into(), json!(live)),
                    ]);
                    call.record(&mut fields);
                    session.receipt("ngram-host-replay", fields)?;
                }
            } else {
                let mut fields = BTreeMap::from([
                    ("symbol".into(), json!(entry.symbol)),
                    ("live".into(), json!(live)),
                ]);
                checks.record(&mut fields);
                session.receipt("ngram-host-replay", fields)?;
            }
            let device_us = lab.time(entry, rows, &device_table, iterations)?;
            let host_us = lab.time(entry, rows, &host_table, iterations)?;
            report.push(json!({
                "symbol": entry.symbol, "rows": rows, "live": live,
                "device_us": device_us, "host_us": host_us,
                "added_us": host_us - device_us,
            }));
            Ok(())
        })();
        failures.check(&format!("{} rows {rows} live {live}", entry.symbol), result);
    }
    failures.finish()?;
    for line in &report {
        println!("{line}");
    }
    Ok(report)
}

#[test]
fn qwen38_ngram_reads_a_mapped_host_table_bit_for_bit() {
    // The rank's 12.8 GB table plus working buffers.
    gpu_test(14 << 30, |session| {
        host_table_matches(session, &BUCKETS, true, None)
    })
    .unwrap();
}

/// Qwen decode buckets of up to 256 items from the host table, at
/// 33, 64, 100, 128, 255 and 256 live items; the 255-item step's bucket
/// also replays as a PDL graph.
#[test]
fn qwen38_ngram_host_table_wide_buckets_bit_for_bit() {
    gpu_test(14 << 30, |session| {
        host_table_matches(session, &WIDE_COUNTS, false, Some(WIDE_REPLAY))
    })
    .unwrap();
}
