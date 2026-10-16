use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, SigmoidMeanHyperConnectionQuery as H, StaticQuery};
use kernel_core::hyper_connection_oracle as hc;
use kernel_core::oracle::OracleOutput;
use kernel_lab::{LabResult, Session, check_output, gpu_test, launch};
use kernels_sm120::row::hyper_connection::Entry;
use model_qwen38_contract::sm120::row as qwen38;
use runtime_cuda::{DeviceBuffer, Module, Stream};
use serde_json::json;
use std::collections::BTreeMap;

use kernel_lab::poison::{Phase, WriteMap};

struct Fixture<'a> {
    session: &'a Session,
    module: Module,
    buffers: BTreeMap<String, DeviceBuffer>,
    scalars: BTreeMap<String, Argument>,
}
impl<'a> Fixture<'a> {
    fn new(session: &'a Session, rows: usize) -> LabResult<Self> {
        let mut f = Self {
            session,
            module: session
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_qwen38.cubin"
                )))
                .map_err(|e| e.to_string())?,
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
    ) -> LabResult<conc::replay::Checks<'b>> {
        let mut checks = conc::replay::Checks::new(self.session);
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
            ("artifact".into(), evidence::identity("row_qwen38")?),
            ("rows".into(), json!(rows)),
            ("profile".into(), json!(entry.profile.as_str())),
        ]);
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
        conc::replay::phase_all(&mut checks, Phase::Eager, false, || {
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
        let mut failures = conc::replay::Failures::default();
        for iteration in 0..3 {
            failures.check(
                &format!("{} graph {iteration}", entry.symbol),
                conc::replay::phase_all(&mut checks, Phase::Graph, true, || {
                    graph
                        .launch(&self.session.stream)
                        .map_err(|e| e.to_string())
                }),
            );
        }
        failures.check(
            "PDL",
            conc::replay::phase_all(&mut checks, Phase::Pdl, true, || {
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
            conc::replay::phase_all(&mut empty, Phase::Eager, false, || {
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
            conc::replay::phase_all(&mut empty, Phase::Graph, true, || {
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
            conc::replay::phase_all(&mut checks, Phase::Relaunch, false, || {
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

#[path = "common/concurrency.rs"]
mod conc;
use conc::evidence;
#[test]
fn m3_qwen_hc_rows() {
    if !conc::enabled("m3") {
        return;
    }
    gpu_test(128 << 20, |session| {
        conc::exclusive(session)?;
        let mut failures = conc::replay::Failures::default();
        for rows in conc::ROWS {
            let result = (|| -> LabResult<()> {
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
                    f.check(&entry, rows, &reference)?;
                    f.buffers["live_rows"]
                        .copy_from_bytes(&(rows as u32).to_ne_bytes(), &session.stream)
                        .map_err(|e| e.to_string())?;
                    let mut checks = f.checks(&f.buffers, rows, n)?;
                    checks.phase(Phase::Eager, false, || {
                        f.launch(&entry, rows as u32, rows as u32, &session.stream, true)
                    })?;
                    conc::measure(
                        session,
                        "M3",
                        &entry.symbol,
                        rows,
                        "row_qwen38",
                        || f.launch(&entry, rows as u32, rows as u32, &session.stream, true),
                        || {
                            check_output(
                                &f.output("output", n)?
                                    .into_iter()
                                    .map(bf16_to_f32)
                                    .collect::<Vec<_>>(),
                                &reference,
                                DType::Bf16,
                                &mut BTreeMap::new(),
                            )
                        },
                        |graph| {
                            checks.phase(Phase::Graph, true, || {
                                graph.launch(&session.stream).map_err(|e| e.to_string())
                            })?;
                            let mut fields = BTreeMap::new();
                            checks.record(&mut fields);
                            Ok(fields)
                        },
                    )?;
                }

                Ok(())
            })();
            failures.check(&format!("rows={rows}"), result);
        }
        failures.finish()?;
        Ok(())
    })
    .expect("Qwen HC G1");
}
