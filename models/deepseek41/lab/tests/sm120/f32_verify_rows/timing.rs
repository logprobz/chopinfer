//! Pair-lab timing of every form at the verify row counts, with the
//! weight-read floor. Each row count pairs the form a verify step serves
//! today (arm A) with a second graph of itself on the same operands (the
//! in-job A/A control) and with every other admitted form, AB/BA with a
//! 256 MiB write eviction before every sample, so weights start cold as
//! they do after the layers between two calls. The enforced power limit
//! and SM clock are read at the start, before every row count and at the
//! end, never set; a job whose limit changes is void.
use super::catalog::{self, Shape};
use super::fixture::{self, Form, Modules, Step, Weights, err};
use kernel_harness::kit::cases::f32_projection::Witness;
use kernel_harness::{LabResult, Session, eviction::EvictionMode, launch};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub const ROWS: [u32; 7] = [32, 48, 64, 96, 128, 160, 192];
const WARMUP: usize = 2;
const ROUNDS: usize = 15;
pub const RECEIPT_KIND: &str = "deepseek41-f32-verify-rows";

fn list<T: std::str::FromStr>(name: &str) -> Option<Vec<T>> {
    std::env::var(name).ok().map(|v| {
        v.split(',')
            .map(|s| s.trim().parse().ok().expect("listed value"))
            .collect()
    })
}

/// The shapes `CHOPINFER_F32_ROWS_SHAPES` names (default head, router).
pub fn selected_shapes() -> Vec<Shape> {
    list::<String>("CHOPINFER_F32_ROWS_SHAPES")
        .unwrap_or_else(|| vec!["head".into(), "router".into()])
        .iter()
        .map(|name| catalog::shape(name).expect("known shape"))
        .collect()
}

pub fn selected_rows() -> Vec<u32> {
    list("CHOPINFER_F32_ROWS_ROWS").unwrap_or_else(|| ROWS.to_vec())
}

/// The enforced power limit and SM clock of this session's GPU.
fn reading(s: &Session) -> LabResult<[f64; 2]> {
    let uuid = s.metadata["uuid"].as_str().ok_or("missing device UUID")?;
    let output = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=uuid,enforced.power.limit,clocks.sm")
        .arg("--format=csv,noheader,nounits")
        .output()
        .map_err(err)?;
    let text = String::from_utf8(output.stdout).map_err(err)?;
    let fields: Vec<&str> = text
        .lines()
        .map(|l| l.split(',').map(str::trim).collect::<Vec<_>>())
        .find(|f| f.len() == 3 && f[0] == uuid)
        .ok_or("nvidia-smi does not report this GPU")?
        .into_iter()
        .skip(1)
        .collect();
    let value = |f: &str| {
        f.parse::<f64>()
            .ok()
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| format!("no reading in {f:?}"))
    };
    Ok([value(fields[0])?, value(fields[1])?])
}

#[derive(Default)]
struct Power {
    first: Option<f64>,
    samples: Vec<Value>,
    changes: Vec<String>,
}
impl Power {
    fn sample(&mut self, s: &Session, label: &str) -> LabResult<()> {
        let [limit, sm] = reading(s)?;
        let first = *self.first.get_or_insert(limit);
        if limit != first {
            self.changes
                .push(format!("{label}: enforced {limit} W, first {first} W"));
        }
        self.samples
            .push(json!({"label": label, "enforced_power_limit_w": limit, "sm_clock_mhz": sm}));
        Ok(())
    }
}

/// The weight-read floor per call: one cold pass of 16-byte loads over
/// each weight copy in turn, paired with itself.
fn floor(s: &Session, lab: &runtime_cuda::Module, copies: &[&DeviceBuffer]) -> LabResult<Value> {
    let function = lab.function("chop_probe_read_vector_v1").map_err(err)?;
    let checksum = s.allocate(4)?;
    let bytes = copies[0].byte_len() as u64;
    let blocks = kernels_sm120::gemm::SM_COUNT * 8;
    let graph = s.capture(|stream| {
        for copy in copies {
            function
                .launch(
                    stream,
                    launch::simple_config(
                        (blocks, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &[
                        KernelArgument::Buffer(copy),
                        KernelArgument::Buffer(&checksum),
                        KernelArgument::U64(bytes / 4),
                        KernelArgument::U32(1),
                    ],
                )
                .map_err(err)?;
        }
        Ok(())
    })?;
    let paired =
        kernel_harness::time_ab(s, WARMUP, ROUNDS, [EvictionMode::Write; 2], |_, stream| {
            graph.launch(stream).map_err(err)
        })?;
    let us = f64::from(paired.arms[0].median_ms) * 1000. / copies.len() as f64;
    Ok(json!({"bytes": bytes, "copies": copies.len(), "us": us,
        "aa_ratio": paired.paired_ratio, "gb_per_s": bytes as f64 / us / 1000.}))
}

/// Independent register MMAs on every SM: the tensor rate ceiling.
fn mma_rate(s: &Session, lab: &runtime_cuda::Module) -> LabResult<Value> {
    let function = lab.function("chop_lab_mma_rate_v1").map_err(err)?;
    let output = s.allocate(4 * 256)?;
    let iterations = 4096_u32;
    let ctas = kernels_sm120::gemm::SM_COUNT * 2;
    let paired =
        kernel_harness::time_ab(s, WARMUP, ROUNDS, [EvictionMode::Write; 2], |_, stream| {
            function
                .launch(
                    stream,
                    launch::simple_config(
                        (ctas, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &[
                        KernelArgument::Buffer(&output),
                        KernelArgument::U32(iterations),
                    ],
                )
                .map_err(err)
        })?;
    let us = f64::from(paired.arms[0].median_ms) * 1000.;
    let mmas = f64::from(ctas) * 8. * 24. * f64::from(iterations);
    Ok(
        json!({"us": us, "mmas": mmas, "tflops": mmas * 4096. / us / 1e6,
        "aa_ratio": paired.paired_ratio}),
    )
}

/// The weight copies a timing graph reads: one for the head, whose
/// weights exceed L2, and one per layer for the small projections.
fn copies(shape: Shape) -> usize {
    if u64::from(shape.n) * u64::from(shape.k) * 2 > 64 << 20 {
        1
    } else {
        40
    }
}

/// The forms timed at `rows`: today's served form (twice, for the A/A
/// control), then every other admitted form.
fn forms(shape: Shape, rows: u32) -> Vec<Form> {
    let current = fixture::current(shape, rows);
    let mut forms = vec![current.clone(), current.clone()];
    let mut others = vec![
        Form::Served,
        Form::Split,
        Form::Spans,
        Form::Bounded,
        Form::Reuse,
    ];
    others.extend(
        catalog::candidates(shape)
            .into_iter()
            .map(|c| Form::Rows(Box::new(c))),
    );
    others.extend(
        catalog::packed(shape)
            .into_iter()
            .map(|(c, served)| Form::Packed(Box::new(c), served)),
    );
    for form in others {
        if form.admits(shape, rows) && form.name() != current.name() {
            forms.push(form);
        }
    }
    forms
}

pub fn bench(s: &Session, lab: &'static [u8]) -> LabResult<BTreeMap<String, Value>> {
    let mut power = Power::default();
    power.sample(s, "start")?;
    let lab_module = s.load_cubin(lab)?;
    let mma = mma_rate(s, &lab_module)?;
    println!("F32_ROWS_MMA {mma}");
    let mut results = vec![json!({"mma_rate": mma})];
    for shape in selected_shapes() {
        let modules = Modules::load(s, lab, shape)?;
        let (a, b) = fixture::operands(shape, Witness::General);
        let weights = Weights::new(s, &modules, shape, &b)?;
        let extra = (1..copies(shape))
            .map(|_| s.allocate(weights.weight.byte_len()))
            .collect::<LabResult<Vec<_>>>()?;
        for copy in &extra {
            let bytes = weights.weight.byte_len();
            copy.copy_range_from_async(0, &weights.weight, 0, bytes, &s.stream)
                .map_err(err)?;
        }
        drop(b);
        let copies: Vec<&DeviceBuffer> = std::iter::once(&weights.weight).chain(&extra).collect();
        let floor = floor(s, &lab_module, &copies)?;
        println!(
            "F32_ROWS_FLOOR {}",
            json!({"shape": shape.name, "floor": floor})
        );
        let floor_us = floor["us"].as_f64().unwrap();
        for rows in selected_rows() {
            let step = Step::new(s, &weights, rows, &a, forms(shape, rows))?;
            step.compare(s, &modules, shape.name)?;
            let graphs = (0..step.forms.len())
                .map(|f| step.repeated(s, &modules, f, &copies))
                .collect::<LabResult<Vec<_>>>()?;
            power.sample(s, &format!("{} rows={rows}", shape.name))?;
            let mut timed = Vec::new();
            let mut aa = 0.;
            let calls = copies.len() as f64;
            for f in 1..step.forms.len() {
                let label = format!("{} rows={rows} {}", shape.name, step.forms[f].name());
                let paired = kernel_harness::with_timing_label(&label, || {
                    kernel_harness::time_ab(
                        s,
                        WARMUP,
                        ROUNDS,
                        [EvictionMode::Write; 2],
                        |arm, stream| {
                            graphs[if arm == 0 { 0 } else { f }]
                                .launch(stream)
                                .map_err(err)
                        },
                    )
                })?;
                if f == 1 {
                    aa = f64::from(paired.paired_ratio);
                }
                let us = f64::from(paired.arms[1].median_ms) * 1000. / calls;
                let flops = 6. * f64::from(rows) * f64::from(shape.n) * f64::from(shape.k);
                let record = json!({
                    "shape": shape.name, "n": shape.n, "k": shape.k, "rows": rows,
                    "calls_per_graph": copies.len(),
                    "baseline": step.forms[0].name(), "form": step.forms[f].name(),
                    "control": f == 1,
                    "launches": step.launches(f).iter().map(|l| json!({"symbol": l.symbol,
                        "grid": l.plan.grid, "block": l.plan.block,
                        "shared": l.plan.dynamic_shared_bytes})).collect::<Vec<_>>(),
                    "baseline_us": f64::from(paired.arms[0].median_ms) * 1000. / calls,
                    "us": us, "min_us": f64::from(paired.arms[1].min_ms) * 1000. / calls,
                    "paired_ratio": paired.paired_ratio,
                    "floor_us": floor_us, "floor_ratio": us / floor_us,
                    "mma_tflops": flops / us / 1e6,
                });
                println!("F32_ROWS_TIME {record}");
                timed.push(record);
            }
            results.push(json!({"shape": shape.name, "rows": rows, "aa_ratio": aa,
                "aa_spread": (aa - 1.).abs(), "timed": timed}));
        }
        results.push(json!({"shape": shape.name, "floor": floor}));
    }
    power.sample(s, "end")?;
    let void = !power.changes.is_empty();
    let mut data = BTreeMap::new();
    data.insert("results".into(), json!(results));
    data.insert("power_samples".into(), json!(power.samples));
    data.insert("power_limit_changes".into(), json!(power.changes));
    data.insert("void".into(), json!(void));
    data.insert(
        "scope".into(),
        json!(
            "Single-GPU CUDA event medians of one graph per form, AB/BA with a 256 MiB write \
            eviction before each sample, 15 pairs after 2 warmups; arm A is the form a verify \
            step serves today and the first pairing is its A/A control. General witness operands. \
            The floor is one cold 16-byte-load pass over the weight bytes. Pair-lab timings \
            locate work and never decide adoption."
        ),
    );
    if void {
        return Err(format!(
            "an enforced power limit changed during the job, so it is void: {}",
            power.changes[0]
        ));
    }
    Ok(data)
}
