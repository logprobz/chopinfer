use super::{
    NORM, OFFSET, ROLES, WIDTH, artifacts, fusion_checks, norm, quant, registration, validate,
};
use kernel_harness::{LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::row::rmsnorm_f32_prefill::Entry;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Stream};
use serde_json::{Value, json};

const PATTERNS: [&str; 12] = [
    "finite",
    "zero",
    "denormal",
    "ties",
    "maximum",
    "nan_first",
    "nan_last",
    "inf",
    "weight_nan",
    "weight_inf",
    "sticky",
    "cancellation",
];
fn upload(buffer: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    buffer
        .copy_from_bytes(bytes, stream)
        .map_err(|e| e.to_string())
}
struct Input {
    x: DeviceBuffer,
    weight: DeviceBuffer,
    live: DeviceBuffer,
}
struct Output {
    buffers: [DeviceBuffer; 4],
    sizes: [usize; 4],
}
impl Output {
    fn new(s: &Session, rows: u32) -> LabResult<Self> {
        let n = rows as usize * WIDTH;
        let sizes = [n * 4, n, n / 32, (rows as usize + OFFSET + 2) * 4];
        Ok(Self {
            buffers: [
                s.allocate(sizes[0])?,
                s.allocate(sizes[1])?,
                s.allocate(sizes[2])?,
                s.allocate(sizes[3])?,
            ],
            sizes,
        })
    }
    fn poison(&self, s: &Session, rows: u32, sticky: bool) -> LabResult<()> {
        for i in 0..3 {
            upload(&self.buffers[i], &vec![0xa5; self.sizes[i]], &s.stream)?;
        }
        let mut status = vec![0x1357_2468u32; rows as usize + OFFSET + 2];
        status[OFFSET..OFFSET + rows as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.buffers[3],
            &status
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
            &s.stream,
        )
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(self.sizes)
            .map(|(buffer, size)| {
                let mut bytes = vec![0; size];
                buffer
                    .copy_to_bytes(&mut bytes, &s.stream)
                    .map_err(|e| e.to_string())?;
                Ok(bytes)
            })
            .collect()
    }
}
struct Device<'a> {
    s: &'a Session,
    entry: Entry,
    functions: Vec<Function>,
    // Keep modules alive through every graph replay.
    _modules: Vec<runtime_cuda::Module>,
    input: Input,
    outputs: [Output; 2],
    rows: u32,
}
impl<'a> Device<'a> {
    fn new(s: &'a Session, entry: &Entry, rows: u32, x: &[u8], w: &[u8]) -> LabResult<Self> {
        validate(rows, rows, entry)?;
        if x.len() != rows as usize * WIDTH * 4 || w.len() != WIDTH * 2 {
            return Err("fixture allocation differs from its launch shape".into());
        }
        let symbols = [entry.reference.symbol.clone(), NORM.into(), quant().symbol];
        let modules = symbols
            .iter()
            .map(|symbol| s.load_cubin(&artifacts::module_for(symbol)?))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = modules
            .iter()
            .zip(symbols)
            .map(|(m, name)| m.function(&name).map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let input = Input {
            x: s.allocate(x.len())?,
            weight: s.allocate(w.len())?,
            live: s.allocate(4)?,
        };
        upload(&input.x, x, &s.stream)?;
        upload(&input.weight, w, &s.stream)?;
        upload(&input.live, &rows.to_ne_bytes(), &s.stream)?;
        Ok(Self {
            s,
            entry: entry.clone(),
            functions,
            _modules: modules,
            input,
            outputs: [Output::new(s, rows)?, Output::new(s, rows)?],
            rows,
        })
    }
    fn launch(&self, arm: usize, stream: &Stream) -> LabResult<()> {
        let [y, codes, scales, status] = &self.outputs[arm].buffers;
        let i = &self.input;
        let eps = self.entry.reference.query.norm.epsilon_bits;
        if arm == 1 {
            if self.entry.reference.symbol == "chop_prefill_query_norm_ceil_cached_v1" {
                use kernel_core::{backend::View, contract::Argument};
                let buffers = [&i.x, &i.weight, y, codes, scales, &i.live, status];
                let views = buffers
                    .iter()
                    .enumerate()
                    .map(|(n, b)| View {
                        handle: n as u64 + 1,
                        bytes: b.byte_len() as u64,
                    })
                    .collect::<Vec<_>>();
                let actual = super::served::plan(self.rows, &views)?;
                if actual.symbol != self.entry.reference.symbol {
                    return Err("served symbol differs".into());
                }
                let args = actual
                    .arguments
                    .0
                    .iter()
                    .map(|a| match *a {
                        Argument::DeviceBuffer(id) => Ok(A::Buffer(buffers[id as usize - 1])),
                        Argument::NullDevicePointer => Ok(A::NullBuffer),
                        Argument::U64(value) => Ok(A::U64(value)),
                        Argument::U32(value) if value == eps => Ok(A::U32(value)),
                        _ => Err("unexpected served query argument".to_owned()),
                    })
                    .collect::<LabResult<Vec<_>>>()?;
                return self.functions[0]
                    .launch(stream, launch::config(actual.plan)?, &args)
                    .map_err(|e| e.to_string());
            }
            self.functions[0]
                .launch(
                    stream,
                    launch::config(self.entry.plan(self.rows).unwrap())?,
                    &[
                        A::Buffer(&i.x),
                        A::Buffer(&i.weight),
                        A::Buffer(y),
                        A::NullBuffer,
                        A::Buffer(codes),
                        A::Buffer(scales),
                        A::Buffer(&i.live),
                        A::Buffer(status),
                        A::U64(OFFSET as u64),
                        A::U32(eps),
                    ],
                )
                .map_err(|e| e.to_string())
        } else {
            let mut plan = norm().plan(self.rows).map_err(|e| e.to_string())?;
            plan.grid[0] = self.rows;
            self.functions[1]
                .launch(
                    stream,
                    launch::config(plan)?,
                    &[
                        A::Buffer(&i.x),
                        A::Buffer(&i.weight),
                        A::NullBuffer,
                        A::NullBuffer,
                        A::Buffer(y),
                        A::Buffer(&i.live),
                        A::U32(eps),
                        A::Buffer(status),
                        A::U64(OFFSET as u64),
                    ],
                )
                .map_err(|e| e.to_string())?;
            self.functions[2]
                .launch(
                    stream,
                    launch::config(quant().plan(self.rows).unwrap())?,
                    &[
                        A::Buffer(y),
                        A::Buffer(codes),
                        A::Buffer(scales),
                        A::Buffer(&i.live),
                        A::Buffer(status),
                        A::U64(OFFSET as u64),
                    ],
                )
                .map_err(|e| e.to_string())
        }
    }
    fn once(&self, arm: usize, sticky: bool) -> LabResult<Vec<Vec<u8>>> {
        self.outputs[arm].poison(self.s, self.rows, sticky)?;
        self.launch(arm, &self.s.stream)?;
        self.s.stream.synchronize().map_err(|e| e.to_string())?;
        self.outputs[arm].read(self.s)
    }
}
fn compare(a: &[Vec<u8>], b: &[Vec<u8>]) -> LabResult<()> {
    fusion_checks::compare_roles(&ROLES, a, b)
}
fn fixture(pattern: &str) -> (Vec<u8>, Vec<u8>) {
    let mut state = 0x1234_5678_9abc_def1u64;
    let x = (0..64 * WIDTH)
        .flat_map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let sign = (state as u32) & 0x8000_0000;
            let bits = match pattern {
                "zero" => sign,
                "denormal" => sign | ((state as u32 & 0x7f_ffff).max(1)),
                "ties" => sign | 0x3f80_8000 | (((i as u32) % 7) << 16),
                "maximum" => sign | 0x7f7f_ffff,
                "nan_first" if i % WIDTH == 0 => 0x7fc1_0000,
                "nan_last" if i % WIDTH == WIDTH - 1 => 0xff81_0000,
                "inf" if i % WIDTH == WIDTH - 1 => 0xff80_0000,
                "cancellation" => sign | if i % 8 == 0 { 0x4700_0000 } else { 0x3780_0000 },
                _ => sign | (((state >> 32) as u32 % 20 + 117) << 23) | (state as u32 & 0x7f_ffff),
            };
            bits.to_ne_bytes()
        })
        .collect();
    let w = (0..WIDTH)
        .flat_map(|i| {
            let bits: u16 = match pattern {
                "weight_nan" if i == WIDTH - 1 => 0x7fc1,
                "weight_inf" if i == 0 => 0xff80,
                _ => 0x3f00 + (i as u16 % 128),
            };
            bits.to_ne_bytes()
        })
        .collect();
    (x, w)
}
fn active(output: &[Vec<u8>], rows: usize) -> Vec<Vec<u8>> {
    let mut out = output.to_vec();
    out[3] = output[3][OFFSET * 4..(OFFSET + rows) * 4].to_vec();
    out
}

pub fn run(s: &Session, memcheck: bool) -> LabResult<()> {
    let mut failures = fusion_checks::Failures::new("QUERY_PREFILL");
    let mut reports = Vec::<Value>::new();
    let mut timings = Vec::<Value>::new();
    let mut invariances = 0;
    let mut negatives = 0;
    let corpus = artifacts::corpus()?;
    for entry in registration::entries()? {
        for case in &corpus {
            let label = format!("real-L{}:{}", case.layer, entry.reference.symbol);
            let device = Device::new(s, &entry, 8192, &case.input, &case.weight)?;
            check(
                &device,
                &label,
                false,
                memcheck,
                &mut failures,
                &mut reports,
            )?;
            if !memcheck {
                let expected = active(&device.once(0, false)?, 8192);
                // Independent buffers give each partition a local row zero.
                let mut joined = vec![Vec::new(); 4];
                for (begin, rows) in [(0, 64), (64, 8128)] {
                    let part = Device::new(
                        s,
                        &entry,
                        rows as u32,
                        &case.input[begin * WIDTH * 4..(begin + rows) * WIDTH * 4],
                        &case.weight,
                    )?;
                    for (into, values) in joined.iter_mut().zip(active(&part.once(1, false)?, rows))
                    {
                        into.extend(values);
                    }
                }
                failures.record(&format!("{label}:chunk"), compare(&expected, &joined));
                invariances += 1;
                // Reverse batch positions without changing any row's arithmetic.
                let reversed = case
                    .input
                    .chunks_exact(WIDTH * 4)
                    .rev()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>();
                let moved = Device::new(s, &entry, 8192, &reversed, &case.weight)?;
                let moved_output = active(&moved.once(1, false)?, 8192);
                let restored = moved_output
                    .iter()
                    .zip([WIDTH * 4, WIDTH, WIDTH / 32, 4])
                    .map(|(bytes, stride)| {
                        bytes
                            .chunks_exact(stride)
                            .rev()
                            .flatten()
                            .copied()
                            .collect()
                    })
                    .collect::<Vec<_>>();
                failures.record(&format!("{label}:position"), compare(&expected, &restored));
                invariances += 1;
                for d in [&device, &moved] {
                    d.outputs[1].poison(s, 8192, false)?;
                }
                s.stream.synchronize().map_err(|e| e.to_string())?;
                let other = s.context.stream().map_err(|e| e.to_string())?;
                device.launch(1, &s.stream)?;
                moved.launch(1, &other)?;
                other.synchronize().map_err(|e| e.to_string())?;
                s.stream.synchronize().map_err(|e| e.to_string())?;
                failures.record(
                    &format!("{label}:concurrent-original"),
                    compare(&expected, &active(&device.outputs[1].read(s)?, 8192)),
                );
                failures.record(
                    &format!("{label}:concurrent-reversed"),
                    compare(&moved_output, &active(&moved.outputs[1].read(s)?, 8192)),
                );
                invariances += 1;
                // Whole-chain intervals include both baseline launches and their dependency.
                for hot in [false, true] {
                    for output in &device.outputs {
                        output.poison(s, 8192, false)?;
                    }
                    let graphs = (0..2)
                        .map(|arm| s.capture(|stream| device.launch(arm, stream)))
                        .collect::<LabResult<Vec<_>>>()?;
                    let timing =
                        kernel_harness::with_timing_label(&format!("{label}:hot={hot}"), || {
                            kernel_harness::time_ab_prepared(
                                s,
                                3,
                                15,
                                [EvictionMode::Read; 2],
                                |arm, stream| {
                                    if hot {
                                        graphs[arm].launch(stream).map_err(|e| e.to_string())
                                    } else {
                                        Ok(())
                                    }
                                },
                                |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
                            )
                        })?;
                    let row = json!({"case":label,"hot":hot,"rows":8192,
                        "old_us":timing.arms[0].median_ms*1000.,"new_us":timing.arms[1].median_ms*1000.,
                        "ratio":timing.paired_ratio});
                    eprintln!("QUERY_PREFILL_TIMING {row}");
                    timings.push(row);
                }
            }
        }
        if !memcheck {
            for pattern in PATTERNS {
                let (x, w) = fixture(pattern);
                let device = Device::new(s, &entry, 64, &x, &w)?;
                check(
                    &device,
                    &format!("{pattern}:{}", entry.reference.symbol),
                    pattern == "sticky",
                    false,
                    &mut failures,
                    &mut reports,
                )?;
            }
        }
    }
    if !memcheck {
        for entry in registration::controls()? {
            let sticky = entry.reference.symbol.contains("clear_status");
            let (x, w) = fixture("finite");
            let device = Device::new(s, &entry, 64, &x, &w)?;
            let original = device.once(0, sticky)?;
            let mutated = device.once(1, sticky)?;
            failures.require(
                &entry.reference.symbol,
                compare(&original, &mutated).is_err(),
                "arithmetic/status control escaped equality gate",
            );
            negatives += 1;
        }
    }
    let expected = if memcheck { 4 } else { 28 };
    failures.require("coverage", reports.len() == expected, "case count differs");
    failures.require(
        "invariance coverage",
        invariances == if memcheck { 0 } else { 12 },
        "invariance count differs",
    );
    failures.require(
        "negative coverage",
        negatives == if memcheck { 0 } else { 2 },
        "negative count differs",
    );
    failures.require(
        "timing coverage",
        timings.len() == if memcheck { 0 } else { 8 },
        "timing count differs",
    );
    let value = json!({"schema":"chopinfer.query-prefill-result.v1","memcheck":memcheck,
        "reports":reports,"timings":timings,"invariances":invariances,"negatives":negatives,
        "failures":failures.messages(),"passed":failures.len()==0});
    s.receipt(
        "query-prefill",
        value
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    )?;
    eprintln!(
        "QUERY_PREFILL_COMPLETE memcheck={memcheck} cases={expected} invariances={invariances} negatives={negatives} timings={} passed={}",
        if memcheck { 0 } else { 8 },
        failures.len() == 0
    );
    failures.finish()
}

fn check(
    device: &Device<'_>,
    label: &str,
    sticky: bool,
    memcheck: bool,
    failures: &mut fusion_checks::Failures,
    reports: &mut Vec<Value>,
) -> LabResult<()> {
    let before = failures.len();
    let mut output_sha256 = String::new();
    let s = device.s;
    let graphs = (0..2)
        .map(|arm| s.capture(|stream| device.launch(arm, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    let lives = if memcheck {
        vec![device.rows]
    } else {
        fusion_checks::live_counts(device.rows).to_vec()
    };
    for (step, live) in lives.into_iter().enumerate() {
        validate(device.rows, live, &device.entry)?;
        upload(&device.input.live, &live.to_ne_bytes(), &s.stream)?;
        let mut previous: Option<Vec<Vec<u8>>> = None;
        for replay in 0..if memcheck { 1 } else { 2 } {
            let context = format!("{label}:step={step}:live={live}:replay={replay}");
            for (arm, graph) in graphs.iter().enumerate() {
                device.outputs[arm].poison(s, device.rows, sticky)?;
                if replay == 0 {
                    device.launch(arm, &s.stream)?;
                } else {
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            let old = device.outputs[0].read(s)?;
            let new = device.outputs[1].read(s)?;
            failures.record(&context, compare(&old, &new));
            for output in [&old, &new] {
                failures.record(
                    &context,
                    fusion_checks::guards(&ROLES, output, device.rows, live, &[], OFFSET, sticky),
                );
            }
            if let Some(previous) = &previous {
                failures.record(&context, compare(previous, &new));
            }
            if memcheck || (step == 4 && replay == 1) {
                output_sha256 =
                    kernel_build::digest(&new.iter().flatten().copied().collect::<Vec<_>>());
            }
            previous = Some(old);
        }
    }
    let record = json!({"case":label,"rows":device.rows,"passed":failures.len()==before,"output_sha256":output_sha256});
    eprintln!("QUERY_PREFILL_CASE {record}");
    reports.push(record);
    Ok(())
}

#[test]
fn fixtures_contain_the_declared_negative_values() {
    for pattern in PATTERNS {
        let (x, w) = fixture(pattern);
        assert_eq!(x.len(), 64 * WIDTH * 4);
        assert_eq!(w.len(), WIDTH * 2);
    }
    let (x, _) = fixture("nan_last");
    assert!(f32::from_ne_bytes(x[(WIDTH - 1) * 4..WIDTH * 4].try_into().unwrap()).is_nan());
    let (x, _) = fixture("finite");
    assert!(
        x.chunks_exact(4)
            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite())
    );
}
