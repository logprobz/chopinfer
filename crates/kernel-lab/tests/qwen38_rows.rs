//! The Qwen3.8 row catalog (`row::qwen38`) against the f64 oracles, with
//! graph replay, PDL and inactive rows. `CHOPINFER_QWEN38_RANKS` (2 or 4, default 4) selects the n-gram rows' topology; every other row is
//! TP4's at each rank count and runs as TP4's.

use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, SigmoidMeanHyperConnectionQuery as H, StaticQuery};
use kernel_core::hyper_connection_oracle as hc;
use kernel_core::oracle::OracleOutput;
use kernel_lab::{LabResult, Session, check_output, gpu_test, launch};
use kernels_sm120::Profile;
use kernels_sm120::row::hyper_connection::Entry;
use model_qwen38_contract::sm120::row as qwen38;
use runtime_cuda::{DeviceBuffer, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison as replay;
use kernel_lab::poison::{Phase, WriteMap};

struct Fixture<'a> {
    session: &'a Session,
    /// The row cubin of the fixture's profile, `row_{tag}`.
    artifact: String,
    module: Module,
    buffers: BTreeMap<String, DeviceBuffer>,
    scalars: BTreeMap<String, Argument>,
}
impl<'a> Fixture<'a> {
    fn new(session: &'a Session, rows: usize) -> LabResult<Self> {
        Self::for_profile(session, rows, model_qwen38_contract::sm120::config::PROFILE)
    }
    fn for_profile(session: &'a Session, rows: usize, profile: Profile) -> LabResult<Self> {
        let artifact = format!("row_{}", profile.tag());
        let mut f = Self {
            session,
            module: session.module(&artifact)?,
            artifact,
            buffers: BTreeMap::new(),
            scalars: BTreeMap::new(),
        };
        f.bytes("live_rows", &(rows as u32).to_ne_bytes())?;
        Ok(f)
    }
    fn bytes(&mut self, name: &str, data: &[u8]) -> LabResult<()> {
        let b = self.session.allocate(data.len())?;
        b.copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.insert(name.into(), b);
        Ok(())
    }
    fn input(&mut self, name: &str, len: usize, seed: usize) -> LabResult<Vec<f64>> {
        let codes: Vec<_> = (0..len)
            .map(|i| f32_to_bf16((((i * 31 + seed * 73) % 509) as f32 - 254.0) / 128.0))
            .collect();
        self.bytes(
            name,
            &codes
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(codes.iter().map(|&v| f64::from(bf16_to_f32(v))).collect())
    }
    fn launch(
        &self,
        entry: &Entry,
        capacity: u32,
        rows: u32,
        stream: &Stream,
        pdl: bool,
    ) -> LabResult<()> {
        self.launch_bound(entry, capacity, rows, stream, pdl, &BTreeMap::new())
    }

    fn launch_bound(
        &self,
        entry: &Entry,
        capacity: u32,
        rows: u32,
        stream: &Stream,
        pdl: bool,
        outputs: &BTreeMap<String, DeviceBuffer>,
    ) -> LabResult<()> {
        let function = self
            .module
            .function(&entry.symbol)
            .map_err(|e| e.to_string())?;
        let buffers = self
            .buffers
            .iter()
            .enumerate()
            .map(|(i, (name, b))| (i as u64 + 1, outputs.get(name).unwrap_or(b)))
            .collect();
        let ids: BTreeMap<_, _> = self
            .buffers
            .keys()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i as u64 + 1))
            .collect();
        let args = entry
            .params
            .iter()
            .map(|p| match p {
                EntryParam::DevicePointer { name, .. } => {
                    Ok(Argument::DeviceBuffer(ids[name.as_str()]))
                }
                EntryParam::U32 { name } | EntryParam::U64 { name } => {
                    Ok(self.scalars[name].clone())
                }
                _ => Err("unsupported test scalar".to_owned()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        let args = entry.pack_arguments(args).map_err(|e| e.to_string())?;
        let mut plan = entry.plan(capacity, rows).map_err(|e| e.to_string())?;
        plan.pdl = pdl;
        launch::launch(
            &self.session.context,
            &function,
            stream,
            plan,
            &args,
            &launch::Bindings { buffers },
        )
    }
    fn output(&self, name: &str, len: usize) -> LabResult<Vec<u16>> {
        self.session.download_bf16(&self.buffers[name], len)
    }

    fn checks<'b>(
        &'b self,
        buffers: &'b BTreeMap<String, DeviceBuffer>,
        rows: usize,
        output_words: usize,
    ) -> LabResult<replay::Checks<'b>> {
        let mut checks = replay::Checks::new(self.session);
        for name in ["output", "state_out", "row_ids", "history_out"] {
            if let Some(buffer) = buffers.get(name) {
                let element = WriteMap::of(match name {
                    "row_ids" => DType::I64,
                    "history_out" => DType::I32,
                    _ => DType::Bf16,
                });
                // The row entries guard every live output.
                // Convolution copies its nine state rows; ngram writes two
                // history words. Inputs are separate immutable allocations.
                let bytes = if rows == 0 {
                    0
                } else if name == "output" {
                    output_words * 2
                } else if name == "history_out" {
                    8
                } else {
                    buffer.byte_len()
                };
                checks = checks.output(name, buffer, element.byte_prefix(bytes)?);
            }
        }
        Ok(checks)
    }
    fn check(&self, entry: &Entry, rows: usize, reference: &OracleOutput) -> LabResult<()> {
        let len = reference.values.len();
        let capacity = rows as u32 + 1;
        let mut receipt = BTreeMap::from([
            ("symbol".into(), json!(entry.symbol)),
            ("artifact".into(), evidence::identity(&self.artifact)?),
            ("rows".into(), json!(rows)),
            ("profile".into(), json!(entry.profile.as_str())),
        ]);
        record_ranks(&mut receipt)?;
        if entry.profile != model_qwen38_contract::sm120::config::PROFILE
            && kernel_lab::qwen38_lean()?
        {
            receipt.insert("lean".into(), json!(true));
        }
        let mut second = BTreeMap::new();
        for name in ["output", "state_out", "row_ids", "history_out"] {
            if let Some(buffer) = self.buffers.get(name) {
                second.insert(name.to_owned(), self.session.allocate(buffer.byte_len())?);
            }
        }
        let mut checks = vec![
            self.checks(&self.buffers, rows, len)?,
            self.checks(&second, rows, len)?,
        ];
        replay::phase_all(&mut checks, Phase::Eager, false, || {
            self.launch(entry, capacity, rows as u32, &self.session.stream, false)?;
            self.launch_bound(
                entry,
                capacity,
                rows as u32,
                &self.session.stream,
                false,
                &second,
            )
        })?;
        for (name, buffer) in &second {
            if kernel_lab::poison::read(self.session, buffer)?
                != kernel_lab::poison::read(self.session, &self.buffers[name])?
            {
                return Err(format!(
                    "{} second call {name} differs from eager",
                    entry.symbol
                ));
            }
        }
        let initial = self.output("output", len)?;
        let actual: Vec<_> = initial.iter().map(|&v| bf16_to_f32(v)).collect();
        check_output(&actual, reference, DType::Bf16, &mut receipt)?;
        let graph = self.session.capture(|stream| {
            self.launch(entry, capacity, rows as u32, stream, true)?;
            self.launch_bound(entry, capacity, rows as u32, stream, true, &second)
        })?;
        let mut failures = replay::Failures::default();
        for iteration in 0..3 {
            failures.check(
                &format!("{} graph {iteration}", entry.symbol),
                replay::phase_all(&mut checks, Phase::Graph, true, || {
                    graph
                        .launch(&self.session.stream)
                        .map_err(|e| e.to_string())
                }),
            );
        }
        failures.check(
            "PDL",
            replay::phase_all(&mut checks, Phase::Pdl, true, || {
                self.launch(entry, capacity, rows as u32, &self.session.stream, true)?;
                self.launch_bound(
                    entry,
                    capacity,
                    rows as u32,
                    &self.session.stream,
                    true,
                    &second,
                )
            }),
        );
        for (call, check) in checks.iter().enumerate() {
            let mut evidence = BTreeMap::new();
            check.record(&mut evidence);
            receipt.insert(format!("call_{call}_replay"), json!(evidence));
        }
        self.buffers["live_rows"]
            .copy_from_bytes(&0_u32.to_ne_bytes(), &self.session.stream)
            .map_err(|e| e.to_string())?;
        let mut empty = vec![
            self.checks(&self.buffers, 0, 0)?,
            self.checks(&second, 0, 0)?,
        ];
        failures.check(
            "zero-live eager",
            replay::phase_all(&mut empty, Phase::Eager, false, || {
                self.launch(entry, capacity, rows as u32, &self.session.stream, false)?;
                self.launch_bound(
                    entry,
                    capacity,
                    rows as u32,
                    &self.session.stream,
                    false,
                    &second,
                )
            }),
        );
        failures.check(
            "zero-live graph",
            replay::phase_all(&mut empty, Phase::Graph, true, || {
                graph
                    .launch(&self.session.stream)
                    .map_err(|e| e.to_string())
            }),
        );
        // Restore checked live results for the caller's side-output oracle.
        self.buffers["live_rows"]
            .copy_from_bytes(&(rows as u32).to_ne_bytes(), &self.session.stream)
            .map_err(|e| e.to_string())?;
        failures.check(
            "restore live outputs",
            replay::phase_all(&mut checks, Phase::Relaunch, false, || {
                self.launch(entry, capacity, rows as u32, &self.session.stream, false)?;
                self.launch_bound(
                    entry,
                    capacity,
                    rows as u32,
                    &self.session.stream,
                    false,
                    &second,
                )
            }),
        );
        failures.finish()?;
        receipt.insert(
            "graph_replay_deterministic_pdl_inactive".into(),
            json!(true),
        );
        self.session.receipt("qwen38-row-g1", receipt)?;
        Ok(())
    }
}

/// Records the topology in a receipt when it is not TP4's.
fn record_ranks(receipt: &mut BTreeMap<String, serde_json::Value>) -> LabResult<()> {
    let ranks = kernel_lab::qwen38_ranks()?;
    if ranks != 4 {
        receipt.insert("qwen38_ranks".into(), json!(ranks));
    }
    Ok(())
}

/// TP4's rows; `ngram_g1` runs the n-gram row of the selected topology.
fn tp4_catalog() -> Vec<Entry> {
    qwen38::hyper_connection_rows()
        .into_iter()
        .filter(|entry| entry.profile == model_qwen38_contract::sm120::config::PROFILE)
        .collect()
}

/// The decode item entries, checked against the per-sequence entries in
/// qwen38_ple_items.rs.
fn items_entry(entry: &Entry) -> bool {
    entry
        .params
        .iter()
        .any(|param| matches!(param, EntryParam::DevicePointer { name, .. } if name == "q_len"))
}

#[test]
fn hyper_connection_g1() {
    gpu_test(128 << 20, |session| {
        let mut failures = replay::Failures::default();
        for rows in [1, 2, 8, 16, 32, 257] {
            for entry in qwen38::hyper_connection_rows() {
                let StaticQuery::HyperConnection(query) = entry.query else {
                    continue;
                };
                let mut f = Fixture::new(session, rows)?;
                let wide = rows * 10240;
                let reference = match query {
                    H::GroupedNorm { .. } => {
                        let x = f.input("x", wide, 1)?;
                        let w = f.input("weight", 10240, 2)?;
                        hc::grouped_norm(query, &x, &w)
                    }
                    H::Bottleneck { .. } => {
                        let x = f.input("x", rows * 320, 1)?;
                        hc::bottleneck(query, &x)
                    }
                    H::Mix { .. } => {
                        let up = f.input("up", wide, 1)?;
                        let normed = f.input("normed", wide, 2)?;
                        hc::mix(query, &up, &normed)
                    }
                    H::Inject { .. } => {
                        let x = f.input("x", wide, 1)?;
                        let y = f.input("y", rows * 2560, 2)?;
                        let logits = f.input("logits", rows * 4, 3)?;
                        hc::inject(query, &x, &y, &logits)
                    }
                    _ => unreachable!(),
                }
                .map_err(|e| e.to_string())?;
                let n = reference.values.len();
                f.bytes(
                    "output",
                    &vec![0x55aa_u16; n + n / rows]
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
                if let Err(error) = f.check(&entry, rows, &reference) {
                    failures.check(&format!("{} rows {rows}", entry.symbol), Err(error));
                    continue;
                }
            }
        }
        failures.finish()
    })
    .expect("Qwen HC G1");
}

/// Sigmoid logits from -150 to 7, including the review's -88, -89 and -90,
/// against amplified multiplicands. `1 / (1 + exp(-x))` overflowed below
/// about -88 and returned 0 where the products are representable; the mix and
/// inject bounds are relative to them. Mix column c takes logit
/// `TAIL_LOGITS[c % 15]` on every stream; inject streams take z = logit / 4
/// of -90, -100, -150 and 2, then -88, 0, -80 and -81, with residuals near
/// the injected products.
#[test]
fn hyper_connection_sigmoid_tail_g1() {
    const TAIL_LOGITS: [f32; 15] = [
        -150.0, -110.0, -100.0, -95.0, -90.0, -89.0, -88.0, -85.0, -80.0, -79.5, -20.0, -3.0, 0.0,
        1.5, 7.0,
    ];
    fn put(f: &mut Fixture<'_>, name: &str, values: &[f32]) -> LabResult<Vec<f64>> {
        let codes: Vec<u16> = values.iter().map(|&v| f32_to_bf16(v)).collect();
        f.bytes(
            name,
            &codes
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        Ok(codes.iter().map(|&v| f64::from(bf16_to_f32(v))).collect())
    }
    let amplitudes = [
        2_f32.powi(40),
        -(2_f32.powi(60)),
        3.0 * 2_f32.powi(20),
        2_f32.powi(100),
    ];
    gpu_test(64 << 20, |session| {
        let mut failures = replay::Failures::default();
        let rows = 2;
        for entry in qwen38::hyper_connection_rows() {
            let StaticQuery::HyperConnection(query) = entry.query else {
                continue;
            };
            let mut f = Fixture::new(session, rows)?;
            let reference = match query {
                H::Mix { .. } => {
                    let up: Vec<f32> = (0..rows * 10240)
                        .map(|i| TAIL_LOGITS[i % 2560 % TAIL_LOGITS.len()])
                        .collect();
                    let normed: Vec<f32> = (0..rows * 10240)
                        .map(|i| {
                            let column = i % 2560 / TAIL_LOGITS.len();
                            amplitudes[column % 4] * [1.0, 0.5, -0.25, 2.0][i / 2560 % 4]
                        })
                        .collect();
                    let up = put(&mut f, "up", &up)?;
                    let normed = put(&mut f, "normed", &normed)?;
                    hc::mix(query, &up, &normed)
                }
                H::Inject { .. } => {
                    let logits = put(
                        &mut f,
                        "logits",
                        &[-360.0, -400.0, -600.0, 8.0, -352.0, 0.0, -320.0, -324.0],
                    )?;
                    let x: Vec<f32> = (0..rows * 10240)
                        .map(|i| [0.0, 2_f32.powi(-90), -3.0 * 2_f32.powi(-95), 1.0][i % 4])
                        .collect();
                    let y: Vec<f32> = (0..rows * 2560)
                        .map(|i| {
                            [
                                2_f32.powi(40),
                                2_f32.powi(60),
                                -(2_f32.powi(50)),
                                2_f32.powi(100),
                            ][i / 4 % 4]
                        })
                        .collect();
                    let x = put(&mut f, "x", &x)?;
                    let y = put(&mut f, "y", &y)?;
                    hc::inject(query, &x, &y, &logits)
                }
                _ => continue,
            }
            .map_err(|e| e.to_string())?;
            let n = reference.values.len();
            f.bytes(
                "output",
                &vec![0x55aa_u16; n + n / rows]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            if let Err(error) = f.check(&entry, rows, &reference) {
                failures.check(&format!("{} rows {rows}", entry.symbol), Err(error));
                continue;
            }
        }
        failures.finish()
    })
    .expect("Qwen HC sigmoid tails");
}

#[test]
fn ple_g1() {
    use kernel_core::contract::{NgramQuery, RecurrentPhase, RecurrentQuery};
    use kernel_core::{ngram_oracle, recurrent_oracle};
    gpu_test(192 << 20, |session| {
        let mut failures = replay::Failures::default();
        for rows in [1, 2, 8, 32, 257] {
            for entry in tp4_catalog() {
                if items_entry(&entry) {
                    continue;
                }
                if matches!(
                    entry.query,
                    StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                        phase: RecurrentPhase::Decode,
                        ..
                    })
                ) && rows != 1
                {
                    continue;
                }
                let mut f = Fixture::new(session, rows)?;
                let (reference, state) = match entry.query {
                    StaticQuery::Ngram(query @ NgramQuery::PleGate { .. }) => {
                        let mut key = f.input("key", rows * 10240, 1)?;
                        // Include exact zero and both signs of the signed-root gate.
                        for (i, v) in key.iter_mut().enumerate() {
                            if i / 2560 % 4 == 0 {
                                *v = 0.0;
                            }
                        }
                        let codes: Vec<_> = key
                            .iter()
                            .flat_map(|v| f32_to_bf16(*v as f32).to_ne_bytes())
                            .collect();
                        f.bytes("key", &codes)?;
                        let q = f.input("query", rows * 10240, 2)?;
                        let v = f.input("value", rows * 2560, 3)?;
                        (
                            ngram_oracle::ple_gate(query, &key, &q, &v)
                                .map_err(|e| e.to_string())?,
                            None,
                        )
                    }
                    StaticQuery::Recurrent(query) => {
                        let input = f.input("input", rows * 10240, 1)?;
                        let weight = f.input("weight", 10240 * 4, 2)?;
                        let state = f.input("state", 10240 * 9, 3)?;
                        let reference =
                            recurrent_oracle::causal_conv1d(query, &input, &weight, &state)
                                .map_err(|e| e.to_string())?;
                        f.bytes("state_out", &vec![0; 10240 * 9 * 2])?;
                        (reference.output, Some(reference.state))
                    }
                    _ => continue,
                };
                let n = reference.values.len();
                f.bytes(
                    "output",
                    &vec![0x55aa_u16; n + n / rows]
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
                if let Err(error) = f.check(&entry, rows, &reference) {
                    failures.check(&format!("{} rows {rows}", entry.symbol), Err(error));
                    continue;
                }
                if let Some(state) = state {
                    let expected: Vec<_> = state.iter().map(|v| f32_to_bf16(*v as f32)).collect();
                    assert_eq!(
                        f.output("state_out", expected.len())?,
                        expected,
                        "conv state copies"
                    );
                }
            }
        }
        failures.finish()
    })
    .expect("Qwen PLE G1");
}

#[test]
fn ngram_g1() {
    use kernel_core::{contract::NgramQuery, ngram_oracle as ng};
    gpu_test(64 << 20, |session| {
        let mut failures = replay::Failures::default();
        // The selected topology's per-sequence row: `local` hash heads per
        // rank, head `rank * local + y`, ranks `0..16 / local`.
        let profile = kernel_lab::qwen38_profile()?;
        let symbol = format!("chop_row_{}_ngram_v1", profile.tag());
        let entry = qwen38::hyper_connection_rows()
            .into_iter()
            .find(|e| e.symbol == symbol)
            .ok_or_else(|| format!("missing Qwen row {symbol}"))?;
        let StaticQuery::Ngram(query @ NgramQuery::HashGather { local_heads, .. }) = entry.query
        else {
            unreachable!()
        };
        let local = local_heads as usize;
        let mut tables = ng::ngram_tables(248064, 3, 8, 17, 0, 1, 1).map_err(|e| e.to_string())?;
        // Exercise addressing above 32 bits without allocating a checkpoint-sized table.
        for offset in &mut tables.offsets {
            *offset += 1_u64 << 33;
        }
        // Lean (TP2): one row, a full 32 and a multi-block 257, still
        // at every history case and rank.
        let row_counts: &[usize] = if kernel_lab::qwen38_lean()? {
            &[1, 32, 257]
        } else {
            &[1, 2, 8, 32, 257]
        };
        for &rows in row_counts {
            for history in [[248044, 248044], [19, 248044], [248044, 23], [19, 23]] {
                let tokens: Vec<u32> = (0..rows)
                    .map(|i| if i % 5 == 2 { 248044 } else { (100 + i) as u32 })
                    .collect();
                let expected_rows = ng::ngram_rows(&tables, 8, &history, &tokens, 248044)
                    .map_err(|e| e.to_string())?;
                for rank in 0..16 / local {
                    let mut f = Fixture::for_profile(session, rows, profile)?;
                    let first = rank * local;
                    let start = tables.offsets[first];
                    let table_rows = tables.sizes[first..first + local].iter().sum::<u64>();
                    let table: Vec<u8> = (0..table_rows as usize * 160)
                        .map(|i| {
                            let code = (i % 127) as u8;
                            if i % 2 == 0 { code } else { code | 128 }
                        })
                        .collect();
                    for (name, values) in [
                        ("tokens", tokens.as_slice()),
                        ("history", history.as_slice()),
                    ] {
                        f.bytes(
                            name,
                            &values
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    for (name, values) in [
                        ("multipliers", &tables.multipliers),
                        ("sizes", &tables.sizes),
                        ("offsets", &tables.offsets),
                    ] {
                        f.bytes(
                            name,
                            &values
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                    }
                    f.bytes("table", &table)?;
                    f.bytes("scale", &f32_to_bf16(0.125).to_ne_bytes())?;
                    f.bytes("history_out", &[0; 8])?;
                    f.bytes("row_ids", &vec![0; rows * local * 8])?;
                    f.scalars.insert("rank".into(), Argument::U32(rank as u32));
                    f.scalars.insert("table_start".into(), Argument::U64(start));
                    f.scalars
                        .insert("table_rows".into(), Argument::U64(table_rows));
                    let values =
                        ng::ngram_gather(query, &expected_rows.rows, rank, &table, start, 0.125)
                            .map_err(|e| e.to_string())?;
                    let n = values.len();
                    let bounds = values.iter().map(|v| v.abs()).collect();
                    let reference =
                        OracleOutput::new(values, bounds, 1).map_err(|e| e.to_string())?;
                    f.bytes(
                        "output",
                        &vec![0x55aa_u16; n + n / rows]
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    if let Err(error) = f.check(&entry, rows, &reference) {
                        failures.check(&format!("{} rows {rows}", entry.symbol), Err(error));
                        continue;
                    }
                    let mut actual = vec![0; rows * local * 8];
                    f.buffers["row_ids"]
                        .copy_to_bytes(&mut actual, &session.stream)
                        .map_err(|e| e.to_string())?;
                    let ids: Vec<_> = actual
                        .chunks_exact(8)
                        .map(|v| u64::from_ne_bytes(v.try_into().unwrap()))
                        .collect();
                    let expected: Vec<_> = expected_rows
                        .rows
                        .chunks_exact(16)
                        .flat_map(|r| r[first..first + local].iter().copied())
                        .collect();
                    assert_eq!(ids, expected, "rank {rank} exact hash rows");
                    let mut tail = [0; 8];
                    f.buffers["history_out"]
                        .copy_to_bytes(&mut tail, &session.stream)
                        .map_err(|e| e.to_string())?;
                    assert_eq!(
                        tail.to_vec(),
                        expected_rows
                            .history
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
        failures.finish()
    })
    .expect("Qwen ngram G1");
}

#[path = "common/evidence.rs"]
mod evidence;
#[path = "common/pool_offsets.rs"]
mod pool_offsets;

/// The n-gram gather with its table rows past the 32-bit lines
/// (`pool_offsets`), as a rank's 6 to 24 GiB table places them: the last
/// rank's first heads start just past `i32::MAX` bytes into its table and
/// the others at 2^32 bytes, in a sentinel-filled table. The low run packs
/// the same rows from the table's start and meets the oracle; the high
/// run's output equals it bit for bit, and each run's row ids equal the
/// oracle's for its offsets.
#[test]
fn ngram_high_ids_g1() {
    use kernel_core::{contract::NgramQuery, ngram_oracle as ng};
    gpu_test(pool_offsets::need_bytes(5 << 30), |session| {
        let profile = kernel_lab::qwen38_profile()?;
        let symbol = format!("chop_row_{}_ngram_v1", profile.tag());
        let entry = qwen38::hyper_connection_rows()
            .into_iter()
            .find(|e| e.symbol == symbol)
            .ok_or_else(|| format!("missing Qwen row {symbol}"))?;
        let StaticQuery::Ngram(query @ NgramQuery::HashGather { local_heads, .. }) = entry.query
        else {
            unreachable!()
        };
        let local = local_heads as usize;
        let base = ng::ngram_tables(248064, 3, 8, 17, 0, 1, 1).map_err(|e| e.to_string())?;
        let rank = 16 / local - 1;
        let first = rank * local;
        let sizes = base.sizes[first..first + local].to_vec();
        let rows = 257;
        let history = [19_u32, 23];
        let tokens: Vec<u32> = (0..rows)
            .map(|i| if i % 5 == 2 { 248044 } else { (100 + i) as u32 })
            .collect();
        // Rows of the rank's table where each head starts: packed from its
        // start, or the first heads in group A and the rest in group B.
        let packed: Vec<u64> = sizes
            .iter()
            .scan(0, |at, &size| {
                let start = *at;
                *at += size;
                Some(start)
            })
            .collect();
        let half = local.div_ceil(2);
        let span = sizes.iter().sum::<u64>() as usize;
        let ids = pool_offsets::high_ids(160, 1, span);
        let high: Vec<u64> = (0..local)
            .map(|h| {
                if h < half {
                    ids.a[0] + packed[h]
                } else {
                    ids.b[0] + packed[h] - packed[half]
                }
            })
            .collect();
        // Each head's FP8 codes, the same in both placements.
        let head_rows: Vec<Vec<u8>> = (0..local)
            .map(|h| {
                (0..sizes[h] as usize * 160)
                    .map(|i| {
                        let code = ((i + 37 * h) % 127) as u8;
                        if i % 2 == 0 { code } else { code | 128 }
                    })
                    .collect()
            })
            .collect();
        let table_start = 1_u64 << 33;
        let mut results = Vec::new();
        for (placement, starts) in [("low", &packed), ("high", &high)] {
            let mut tables = base.clone();
            for (offset, &start) in tables.offsets[first..first + local].iter_mut().zip(starts) {
                *offset = table_start + start;
            }
            let table_rows = starts
                .iter()
                .zip(&sizes)
                .map(|(start, size)| start + size)
                .max()
                .unwrap_or(0);
            let expected_rows =
                ng::ngram_rows(&tables, 8, &history, &tokens, 248044).map_err(|e| e.to_string())?;
            let mut f = Fixture::for_profile(session, rows, profile)?;
            for (name, values) in [("tokens", tokens.as_slice()), ("history", &history[..])] {
                f.bytes(
                    name,
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            for (name, values) in [
                ("multipliers", &tables.multipliers),
                ("sizes", &tables.sizes),
                ("offsets", &tables.offsets),
            ] {
                f.bytes(
                    name,
                    &values
                        .iter()
                        .flat_map(|v| v.to_ne_bytes())
                        .collect::<Vec<_>>(),
                )?;
            }
            f.bytes("scale", &f32_to_bf16(0.125).to_ne_bytes())?;
            f.bytes("history_out", &[0; 8])?;
            f.bytes("row_ids", &vec![0; rows * local * 8])?;
            f.scalars.insert("rank".into(), Argument::U32(rank as u32));
            f.scalars
                .insert("table_start".into(), Argument::U64(table_start));
            f.scalars
                .insert("table_rows".into(), Argument::U64(table_rows));
            let n = rows * local * 160;
            f.bytes(
                "output",
                &vec![0x55aa_u16; n + n / rows]
                    .iter()
                    .flat_map(|v| v.to_ne_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let mut pool = None;
            if placement == "low" {
                let mut table = vec![0_u8; table_rows as usize * 160];
                for (&start, rows) in starts.iter().zip(&head_rows) {
                    let at = start as usize * 160;
                    table[at..at + rows.len()].copy_from_slice(rows);
                }
                f.bytes("table", &table)?;
                let values =
                    ng::ngram_gather(query, &expected_rows.rows, rank, &table, table_start, 0.125)
                        .map_err(|e| e.to_string())?;
                let bounds = values.iter().map(|v| v.abs()).collect();
                let reference = OracleOutput::new(values, bounds, 1).map_err(|e| e.to_string())?;
                f.check(&entry, rows, &reference)?;
            } else {
                let mut table = pool_offsets::SentinelPool::new(session, table_rows * 160)?;
                for (&start, rows) in starts.iter().zip(&head_rows) {
                    table.put(session, start * 160, rows)?;
                }
                f.buffers.insert("table".into(), table.buffer.clone());
                let mut checks = f.checks(&f.buffers, rows, n)?;
                checks.phase(Phase::Eager, false, || {
                    f.launch(&entry, rows as u32 + 1, rows as u32, &session.stream, false)
                })?;
                pool = Some(table);
            }
            let mut actual = vec![0; rows * local * 8];
            f.buffers["row_ids"]
                .copy_to_bytes(&mut actual, &session.stream)
                .map_err(|e| e.to_string())?;
            let ids: Vec<u64> = actual
                .chunks_exact(8)
                .map(|v| u64::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            let expected: Vec<u64> = expected_rows
                .rows
                .chunks_exact(16)
                .flat_map(|r| r[first..first + local].iter().copied())
                .collect();
            if ids != expected {
                return Err(format!("{placement} n-gram row ids differ from the oracle"));
            }
            results.push((f.output("output", n)?, table_rows, pool));
        }
        let (low, high_run) = (&results[0], &results[1]);
        if low.0 != high_run.0 {
            let at = low.0.iter().zip(&high_run.0).position(|(a, b)| a != b);
            return Err(format!(
                "n-gram output at high table rows differs from the low run at {at:?}"
            ));
        }
        let scanned = high_run
            .2
            .as_ref()
            .map_or(Ok(0), |table| table.check_untouched(session, "ngram"))?;
        let mut receipt = json!({
            "symbol": entry.symbol,
            "rank": rank,
            "rows": rows,
            "head_starts": high,
            "highest_offset": high_run.1 * 160,
            "table_bytes": high_run.1 * 160,
            "ids": ids.describe(),
            "outputs_bitwise": true,
            "row_ids_exact": true,
            "sentinel_bytes_checked": scanned,
        });
        let ranks = kernel_lab::qwen38_ranks()?;
        if ranks != 4 {
            receipt["qwen38_ranks"] = json!(ranks);
        }
        pool_offsets::receipt(session, "ngram", &symbol, receipt)
    })
    .expect("Qwen n-gram gather at high table rows");
}

#[test]
#[ignore = "requires the parent's exclusive KERNEL window on GPU 1"]
fn rows_g4() {
    use kernel_core::contract::{NgramQuery, RecurrentPhase, RecurrentQuery};
    gpu_test(15 * 1024 * 1024 * 1024, |session| {
        evidence::exclusive(session)?;
        let (peak, peak_source) = evidence::peak(session, "dram_read")?;
        let probe = session.module("probe_sm120")?;
        let empty = probe
            .function("chop_probe_sm120_v1")
            .map_err(|e| e.to_string())?;
        let scratch = session.allocate(4)?;
        let floor = kernel_lab::time(session, 2, 7, |stream| {
            empty
                .launch(
                    stream,
                    launch::simple_config((1, 1, 1), (32, 1, 1), 0, (1, 1, 1), false)?,
                    &[runtime_cuda::KernelArgument::Buffer(&scratch)],
                )
                .map_err(|e| e.to_string())
        })?;
        for rows in [1, 8, 32, 512, 2048, 8192] {
            for entry in tp4_catalog() {
                if items_entry(&entry) {
                    continue;
                }
                let mut f = Fixture::new(session, rows)?;
                let (output_len, traffic) = match entry.query {
                    StaticQuery::HyperConnection(H::GroupedNorm { .. }) => {
                        f.input("x", rows * 10240, 1)?;
                        f.input("weight", 10240, 2)?;
                        (rows * 10240, (3 * rows + 1) * 10240 * 2)
                    }
                    StaticQuery::HyperConnection(H::Bottleneck { .. }) => {
                        f.input("x", rows * 320, 1)?;
                        (rows * 320, rows * 320 * 4)
                    }
                    StaticQuery::HyperConnection(H::Mix { .. }) => {
                        f.input("up", rows * 10240, 1)?;
                        f.input("normed", rows * 10240, 2)?;
                        (rows * 2560, rows * (2 * 10240 + 2560) * 2)
                    }
                    StaticQuery::HyperConnection(H::Inject { .. }) => {
                        f.input("x", rows * 10240, 1)?;
                        f.input("y", rows * 2560, 2)?;
                        f.input("logits", rows * 4, 3)?;
                        (rows * 10240, rows * (2 * 10240 + 2560 + 4) * 2)
                    }
                    StaticQuery::Ngram(NgramQuery::PleGate { .. }) => {
                        f.input("key", rows * 10240, 1)?;
                        f.input("query", rows * 10240, 2)?;
                        f.input("value", rows * 2560, 3)?;
                        (rows * 10240, rows * (3 * 10240 + 2560) * 2)
                    }
                    StaticQuery::Ngram(NgramQuery::HashGather { .. }) => {
                        let tables = kernel_core::ngram_oracle::ngram_tables(
                            248064, 3, 8, 20_000_000, 0, 1, 128,
                        )
                        .map_err(|e| e.to_string())?;
                        let table_rows = tables.sizes[..4].iter().sum::<u64>();
                        for (name, values) in [
                            ("multipliers", &tables.multipliers),
                            ("sizes", &tables.sizes),
                            ("offsets", &tables.offsets),
                        ] {
                            f.bytes(
                                name,
                                &values
                                    .iter()
                                    .flat_map(|v| v.to_ne_bytes())
                                    .collect::<Vec<_>>(),
                            )?;
                        }
                        let table = session.allocate(table_rows as usize * 160)?;
                        table
                            .zero_at_async(0, table.byte_len(), &session.stream)
                            .map_err(|e| e.to_string())?;
                        f.buffers.insert("table".into(), table);
                        let tokens: Vec<u32> = (0..rows)
                            .map(|i| ((i * 104729 + 37) % 248064) as u32)
                            .collect();
                        f.bytes(
                            "tokens",
                            &tokens
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        f.bytes(
                            "history",
                            &[248044_u32, 248044]
                                .iter()
                                .flat_map(|v| v.to_ne_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        f.bytes("history_out", &[0; 8])?;
                        f.bytes("row_ids", &vec![0; rows * 32])?;
                        f.bytes("scale", &f32_to_bf16(0.125).to_ne_bytes())?;
                        f.scalars.insert("rank".into(), Argument::U32(0));
                        f.scalars.insert("table_start".into(), Argument::U64(0));
                        f.scalars
                            .insert("table_rows".into(), Argument::U64(table_rows));
                        (rows * 640, rows * (640 * 3 + 32 + 4))
                    }
                    StaticQuery::Recurrent(RecurrentQuery::CausalConv1d { phase, .. }) => {
                        if phase == RecurrentPhase::Decode && rows != 1 {
                            continue;
                        }
                        f.input("input", rows * 10240, 1)?;
                        f.input("weight", 4 * 10240, 2)?;
                        f.input("state", 9 * 10240, 3)?;
                        f.bytes("state_out", &vec![0; 9 * 10240 * 2])?;
                        (rows * 10240, (2 * rows + 22) * 10240 * 2)
                    }
                    _ => unreachable!(),
                };
                f.bytes("output", &vec![0; output_len * 2])?;
                let timing = kernel_lab::time(session, 2, 7, |stream| {
                    f.launch(&entry, rows as u32, rows as u32, stream, true)
                })?;
                let lower_ms = traffic as f64 / peak * 1000.0;
                let data = BTreeMap::from([
                    ("artifact".into(), evidence::identity("row_qwen38")?),
                    ("symbol".into(), json!(entry.symbol)),
                    ("rows".into(), json!(rows)),
                    ("median_ms".into(), json!(timing.median_ms)),
                    ("min_ms".into(), json!(timing.min_ms)),
                    ("rounds".into(), json!(timing.rounds)),
                    ("logical_bytes".into(), json!(traffic)),
                    ("dram_lower_bound_ms".into(), json!(lower_ms)),
                    (
                        "percent_dram_roofline".into(),
                        json!(100.0 * lower_ms / f64::from(timing.median_ms)),
                    ),
                    ("launch_floor_ms".into(), json!(floor.median_ms)),
                    ("peak_source".into(), json!(peak_source)),
                ]);
                evidence::receipt(session, "qwen38-row-g4", data)?;
            }
        }
        Ok(())
    })
    .expect("Qwen row G4")
    .expect("GPU selection is required for G4");
}

#[test]
fn hyper_connection_composite_g1() {
    use kernel_core::oracle::{self, GemmEpilogue, GemmShape};
    gpu_test(64 << 20, |session| {
        let entries = qwen38::hyper_connection_rows();
        let find = |suffix: &str| {
            entries
                .iter()
                .find(|e| e.symbol == format!("chop_row_qwen38_{suffix}_v1"))
                .unwrap()
        };
        let mut f = Fixture::new(session, 2)?;
        let x = f.input("x", 2 * 10240, 1)?;
        let norm = f.input("weight", 10240, 2)?;
        let weights = |len: usize, seed: usize| -> Vec<f64> {
            (0..len)
                .map(|i| {
                    f64::from(bf16_to_f32(f32_to_bf16(
                        (((i * 31 + seed * 73) % 509) as f32 - 254.0) / 65536.0,
                    )))
                })
                .collect()
        };
        let down = weights(320 * 10240, 3);
        let up = weights(10240 * 320, 4);
        let reference = hc::hyper_connection_input(
            &x,
            hc::HyperConnectionWeights {
                norm: &norm,
                down: &down,
                up: &up,
                inject: None,
                streams: 4,
                width: 2560,
                rank: 320,
                eps: f64::from(1e-6_f32),
            },
        )
        .map_err(|e| e.to_string())?;
        f.bytes("output", &vec![0; 2 * 10240 * 2])?;
        {
            let mut checks = f.checks(&f.buffers, 2, 2 * 10240)?;
            checks.phase(Phase::Eager, false, || {
                f.launch(find("hc_norm"), 2, 2, &session.stream, true)
            })?;
        }
        let normed = f.output("output", 2 * 10240)?;
        let decoded = |values: &[u16]| {
            values
                .iter()
                .map(|&v| f64::from(bf16_to_f32(v)))
                .collect::<Vec<_>>()
        };
        let project = |input: &[f64], weight: &[f64], n, k| -> LabResult<Vec<u8>> {
            let projected = oracle::gemm(
                input,
                weight,
                GemmShape {
                    m: 2,
                    n,
                    k,
                    experts: 1,
                },
                &[],
                GemmEpilogue {
                    residual: None,
                    swiglu: None,
                },
            )
            .map_err(|e| e.to_string())?;
            Ok(projected
                .values
                .iter()
                .flat_map(|v| f32_to_bf16(*v as f32).to_ne_bytes())
                .collect())
        };
        f.bytes("x", &project(&decoded(&normed), &down, 320, 10240)?)?;
        f.bytes("output", &vec![0; 2 * 320 * 2])?;
        {
            let mut checks = f.checks(&f.buffers, 2, 2 * 320)?;
            checks.phase(Phase::Eager, false, || {
                f.launch(find("hc_bottleneck"), 2, 2, &session.stream, true)
            })?;
        }
        let hidden = f.output("output", 2 * 320)?;
        f.bytes("up", &project(&decoded(&hidden), &up, 10240, 320)?)?;
        f.bytes(
            "normed",
            &normed
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )?;
        f.bytes("output", &vec![0; 2 * 2560 * 2])?;
        {
            let mut checks = f.checks(&f.buffers, 2, 2 * 2560)?;
            checks.phase(Phase::Eager, false, || {
                f.launch(find("hc_mix"), 2, 2, &session.stream, true)
            })?;
        }
        let actual = decoded(&f.output("output", 2 * 2560)?);
        oracle::check_layer_cosine(&actual, &reference.mixed, 2560)
            .map_err(|e| format!("{e:?}"))?;
        session.receipt("qwen38-row-g1", {
            let mut receipt = BTreeMap::from([
                ("case".into(), json!("hyper_connection_composite")),
                ("artifact".into(), evidence::identity("row_qwen38")?),
                (
                    "projection_backend".into(),
                    json!("CPU f64 GEMMs rounded to BF16; row maps on GPU"),
                ),
                ("per_token_cosine_at_least".into(), json!(0.9999)),
            ]);
            record_ranks(&mut receipt)?;
            receipt
        })?;
        Ok(())
    })
    .expect("Qwen HC composite G1");
}

#[test]
fn ple_conv_chunk_state_g1() {
    use kernel_core::contract::{RecurrentPhase, RecurrentQuery};
    gpu_test(64 << 20, |session| {
        let entry = qwen38::hyper_connection_rows()
            .into_iter()
            .find(|e| {
                matches!(
                    e.query,
                    StaticQuery::Recurrent(RecurrentQuery::CausalConv1d {
                        phase: RecurrentPhase::Prefill,
                        ..
                    })
                )
            })
            .unwrap();
        let mut full = Fixture::new(session, 17)?;
        let input = full.input("input", 17 * 10240, 1)?;
        let weights = full.input("weight", 4 * 10240, 2)?;
        let history = full.input("state", 9 * 10240, 3)?;
        full.bytes("output", &vec![0; 17 * 10240 * 2])?;
        full.bytes("state_out", &vec![0; 9 * 10240 * 2])?;
        let mut full_checks = full.checks(&full.buffers, 17, 17 * 10240)?;
        full_checks.phase(Phase::Eager, false, || {
            full.launch(&entry, 17, 17, &session.stream, true)
        })?;
        let reference = full.output("output", 17 * 10240)?;
        let final_state = full.output("state_out", 9 * 10240)?;
        let encode = |values: &[f64]| {
            values
                .iter()
                .map(|v| f32_to_bf16(*v as f32))
                .collect::<Vec<_>>()
        };
        for chunks in [vec![1; 17], vec![2, 5, 10], vec![9, 8]] {
            let mut state = encode(&history);
            let mut start = 0;
            let mut output = Vec::new();
            for rows in chunks {
                let mut f = Fixture::new(session, rows)?;
                for (name, codes) in [
                    (
                        "input",
                        encode(&input[start * 10240..(start + rows) * 10240]),
                    ),
                    ("weight", encode(&weights)),
                    ("state", state.clone()),
                ] {
                    f.bytes(
                        name,
                        &codes
                            .iter()
                            .flat_map(|v| v.to_ne_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
                f.bytes("output", &vec![0; rows * 10240 * 2])?;
                f.bytes("state_out", &vec![0; 9 * 10240 * 2])?;
                let mut checks = f.checks(&f.buffers, rows, rows * 10240)?;
                checks.phase(Phase::Eager, false, || {
                    f.launch(&entry, rows as u32, rows as u32, &session.stream, true)
                })?;
                output.extend(f.output("output", rows * 10240)?);
                state = f.output("state_out", 9 * 10240)?;
                start += rows;
            }
            assert_eq!(output, reference, "chunked convolution output");
            assert_eq!(state, final_state, "chunked convolution history");
        }
        session.receipt("qwen38-row-g1", {
            let mut receipt = BTreeMap::from([
                ("case".into(), json!("ple_conv_chunk_continuity")),
                ("artifact".into(), evidence::identity("row_qwen38")?),
                ("bit_exact_chunk_continuity".into(), json!(true)),
            ]);
            record_ranks(&mut receipt)?;
            receipt
        })?;
        Ok(())
    })
    .expect("PLE conv chunk continuity");
}
