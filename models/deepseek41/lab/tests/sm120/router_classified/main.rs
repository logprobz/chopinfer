//! The classifying router producer
//! (`bundle_deepseek41::sm120::router_classified`) against the served pair
//! it replaces at 1 to 32 rows: the router finish's prepare (row domain and
//! ticket reset), then the router's cached exact F32 decode producer. Every
//! partial and the ticket, bitwise, over activation patterns that put rows
//! in and out of the tensor domain (in one chain, in every chain, at chain
//! edges, non-finite values), weight tiles in and out of it, row and live
//! counts and graph replays, with output-flip negatives, a sanitized matrix
//! and paired timing. All modules are the bundle's embedded, locked cubins.
use bundle_deepseek41::sm120::{fold_cubin_for_symbol, router_classified as registration};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::f32_projection::{Entry, classified_decode};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

const PREPARE: &str = "chop_deepseek41_moe_router_finish_e384_k6_prepare_v1";
const EXPERTS: usize = 384;
const HIDDEN: usize = 5120;
const CHAINS: usize = HIDDEN / 128;
const MAX_ROWS: usize = 32;
const ROWS: [u32; 13] = [1, 2, 3, 4, 5, 6, 8, 9, 12, 16, 24, 31, 32];
const TIMED_ROWS: [u32; 6] = [1, 4, 6, 8, 16, 32];
/// Partials start after the 32 row-domain words the prepare writes.
const DOMAIN_WORDS: usize = 32;
const TICKET_GUARD: u32 = 0x5eed_7123;
const STALE_TICKET: u32 = 0xdead_beef;
const PATTERNS: [&str; 10] = [
    "finite",
    "zero",
    "negative_zero",
    "row_nan",
    "row_inf",
    "row_tiny",
    "row_huge",
    "chain_edge",
    "all_out",
    "ties",
];
const WEIGHTS: [&str; 2] = ["tensor", "tile_out"];
const REPLAYED: [&str; 2] = ["finite", "chain_edge"];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
const CHAIN: usize = 16;
const CASE_SECONDS: u64 = 10;
const SANITIZED_CASE_SECONDS: u64 = 60;
const ROLES: [&str; 2] = ["partials", "ticket"];

fn entry() -> Entry {
    registration::entry()
}
/// The served producer, the classifying one and the create-time weight
/// classifier.
fn symbols() -> [String; 4] {
    let e = entry();
    [
        PREPARE.into(),
        e.stage_symbol("_decode"),
        classified_decode::symbol(&e),
        e.stage_symbol("_weights"),
    ]
}
fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_ROUTER_CLASSIFIED").expect("frozen router artifacts required"),
    )
}
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
fn modules() -> LabResult<Modules> {
    let mut modules = Modules::new();
    for symbol in symbols() {
        let (name, bytes) = fold_cubin_for_symbol(&symbol).ok_or("entry must be embedded")?;
        let module = modules
            .entry(name.into())
            .or_insert_with(|| (bytes.to_vec(), Vec::new()));
        if !module.1.contains(&symbol) {
            module.1.push(symbol);
        }
    }
    let probe = std::fs::read(artifacts().join("probe_sm120.cubin")).map_err(|e| e.to_string())?;
    let kernel_lab: serde_json::Value = serde_json::from_str(include_str!(
        "../../../../../../crates/kernel-lab/kernels.lock"
    ))
    .map_err(|e| e.to_string())?;
    if !kernel_lab
        .as_array()
        .unwrap()
        .iter()
        .any(|c| c["name"] == "probe_sm120" && c["cubin_sha256"] == kernel_build::digest(&probe))
    {
        return Err("timing probe differs from the kernel-lab lock".into());
    }
    modules.insert(
        "probe_sm120".into(),
        (probe, PROBE_SYMBOLS.iter().map(|s| (*s).into()).collect()),
    );
    Ok(modules)
}
fn coverage() -> serde_json::Value {
    json!({"symbols": symbols(), "rows": ROWS, "timed_rows": TIMED_ROWS, "patterns": PATTERNS,
        "weights": WEIGHTS, "replayed": REPLAYED, "chain": CHAIN,
        "bitwise_cases": ROWS.len() * PATTERNS.len() * WEIGHTS.len(),
        "timing_cases": TIMED_ROWS.len(), "negative_controls": ROLES.len(),
        "memcheck_cases": 2 * 2})
}

#[test]
fn classifying_producer_pairs_with_the_served_prepare_and_producer() {
    let e = entry();
    let q = e.query();
    assert_eq!((q.n as usize, q.k as usize), (EXPERTS, HIDDEN));
    for symbol in symbols() {
        assert!(fold_cubin_for_symbol(&symbol).is_some(), "{symbol}");
    }
    let decode = e
        .entry_specs()
        .into_iter()
        .find(|s| s.symbol == e.stage_symbol("_decode"))
        .unwrap();
    let classified = classified_decode::entry_spec(&e).unwrap();
    assert_eq!(classified.params.len(), decode.params.len() + 1);
    for rows in ROWS {
        let old = e.decode_plan(rows).unwrap();
        let new = classified_decode::plan(&e, rows).unwrap();
        assert_eq!(old.grid[..2], new.grid[..2]);
        assert_eq!(new.cluster, [1, 1, 8]);
    }
    let original = vec![vec![0u8; 8]; ROLES.len()];
    for role in 0..ROLES.len() {
        let mut bad = original.clone();
        bad[role][0] = 1;
        assert!(compare(&original, &bad).is_err());
    }
    assert_eq!(coverage()["bitwise_cases"], 13 * 10 * 2);
    assert!(TIMED_ROWS.iter().all(|r| ROWS.contains(r)));
}

/// The router finish's workspace (`router_finish::router_workspace_bytes`):
/// the 32 domain words and every chain's partial rows.
fn workspace_bytes(rows: u32) -> usize {
    (DOMAIN_WORDS + rows as usize * EXPERTS * CHAINS) * 4
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_ROUTER_CLASSIFIED"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_ROUTER_CLASSIFIED_EXPORT").is_some() {
        let mut manifest = Vec::new();
        for (name, (bytes, symbols)) in &modules {
            std::fs::write(dir.join(format!("{name}.cubin")), bytes).unwrap();
            manifest
                .push(json!({"name":name,"sha256":kernel_build::digest(bytes),"symbols":symbols}));
        }
        std::fs::write(
            dir.join("modules.json"),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("coverage.json"),
            serde_json::to_vec_pretty(&coverage()).unwrap(),
        )
        .unwrap();
    }
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("modules.json")).unwrap()).unwrap();
    assert_eq!(manifest.as_array().unwrap().len(), modules.len());
    for (name, (bytes, symbols)) in modules {
        let record = manifest
            .as_array()
            .unwrap()
            .iter()
            .find(|v| v["name"] == name)
            .unwrap();
        assert_eq!(record["sha256"], kernel_build::digest(&bytes));
        assert_eq!(record["symbols"], json!(symbols));
        assert_eq!(
            std::fs::read(dir.join(format!("{name}.cubin"))).unwrap(),
            bytes
        );
    }
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(
            &std::fs::read(dir.join("coverage.json")).unwrap()
        )
        .unwrap(),
        coverage()
    );
    eprintln!(
        "ROUTER_CLASSIFIED_MODULES {} PASS",
        manifest.as_array().unwrap().len()
    );
}

fn upload(buffer: &DeviceBuffer, bytes: &[u8], stream: &Stream) -> LabResult<()> {
    buffer
        .copy_from_bytes(bytes, stream)
        .map_err(|e| e.to_string())
}
fn download(buffer: &DeviceBuffer, len: usize, stream: &Stream) -> LabResult<Vec<u8>> {
    let mut bytes = vec![0; len];
    buffer
        .copy_to_bytes(&mut bytes, stream)
        .map_err(|e| e.to_string())?;
    Ok(bytes)
}
fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}
fn compare(expected: &[Vec<u8>], actual: &[Vec<u8>]) -> LabResult<()> {
    for ((role, a), b) in ROLES.iter().zip(expected).zip(actual) {
        if a != b {
            return Err(format!(
                "{role} mismatch at {:?}",
                a.iter().zip(b).position(|(a, b)| a != b)
            ));
        }
    }
    Ok(())
}

struct Inputs {
    activation: DeviceBuffer,
    weight: DeviceBuffer,
    weight_domain: DeviceBuffer,
    live: DeviceBuffer,
}
/// One arm's workspace and ticket, with a guard word each side.
struct Output {
    workspace: DeviceBuffer,
    ticket: DeviceBuffer,
}
impl Output {
    fn new(s: &Session) -> LabResult<Self> {
        Ok(Self {
            workspace: s.allocate(workspace_bytes(MAX_ROWS as u32))?,
            ticket: s.allocate(12)?,
        })
    }
    fn reset(&self, s: &Session) -> LabResult<()> {
        upload(
            &self.workspace,
            &vec![0xa5; workspace_bytes(MAX_ROWS as u32)],
            &s.stream,
        )?;
        upload(
            &self.ticket,
            &words(&[TICKET_GUARD, STALE_TICKET, TICKET_GUARD]),
            &s.stream,
        )
    }
    /// The partials of `rows` rows (the domain words are the prepare's
    /// alone) and the ticket with its guards.
    fn read(&self, s: &Session, rows: u32) -> LabResult<Vec<Vec<u8>>> {
        let all = download(&self.workspace, workspace_bytes(rows), &s.stream)?;
        Ok(vec![
            all[DOMAIN_WORDS * 4..].to_vec(),
            download(&self.ticket, 12, &s.stream)?,
        ])
    }
}
struct Functions {
    prepare: Function,
    producer: Function,
    classified: Function,
    weights: Function,
}
/// The ticket word between its guards.
fn ticket(o: &Output) -> A<'_> {
    A::Range {
        buffer: &o.ticket,
        offset: 4,
        byte_len: 4,
    }
}
/// Arm 0 the served prepare then producer, arm 1 the classifying producer.
fn step(
    f: &Functions,
    arm: usize,
    rows: u32,
    i: &Inputs,
    o: &Output,
    stream: &Stream,
) -> LabResult<()> {
    let e = entry();
    let producer = [
        A::Buffer(&i.activation),
        A::Buffer(&i.weight),
        A::Buffer(&o.workspace),
        A::Buffer(&i.live),
        A::U32(rows),
        A::Buffer(&i.weight_domain),
    ];
    let ticket = ticket(o);
    if arm == 0 {
        let plan = kernel_core::contract::LaunchPlan {
            grid: [rows, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        f.prepare
            .launch(
                stream,
                launch::config(plan)?,
                &[
                    A::Buffer(&i.activation),
                    A::Buffer(&i.live),
                    A::U32(rows),
                    A::Buffer(&o.workspace),
                    ticket,
                ],
            )
            .map_err(|e| e.to_string())?;
        f.producer
            .launch(
                stream,
                launch::config(e.decode_plan(rows).map_err(|e| e.to_string())?)?,
                &producer,
            )
            .map_err(|e| e.to_string())
    } else {
        let mut arguments = producer.to_vec();
        arguments.push(ticket);
        f.classified
            .launch(
                stream,
                launch::config(classified_decode::plan(&e, rows).map_err(|e| e.to_string())?)?,
                &arguments,
            )
            .map_err(|e| e.to_string())
    }
}
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
fn activation(pattern: &str) -> Vec<u8> {
    let mut state = 0x726f_7574_6572_5f61u64;
    let mut values = vec![0u32; MAX_ROWS * HIDDEN];
    for (at, value) in values.iter_mut().enumerate() {
        let (row, k) = (at / HIDDEN, at % HIDDEN);
        let r = xorshift(&mut state);
        let sign = r as u32 & 0x8000_0000;
        let mantissa = r as u32 & 0x7f_ffff;
        let normal = sign | (((r >> 32) as u32 % 40 + 107) << 23) | mantissa;
        *value = match pattern {
            "zero" => 0,
            "negative_zero" => sign,
            "row_nan" if row == 0 && k == 4097 => 0x7fc1_0000,
            "row_inf" if row == MAX_ROWS - 1 && k == 0 => 0xff80_0000,
            "row_tiny" if row % 4 == 1 && k == 100 + 640 * (row % 8) => 0x2b80_0000,
            "row_huge" if row % 4 == 2 && k == HIDDEN - 1 => 0x5380_0000,
            // A non-tensor value in one chain owned by one CTA of the
            // cluster: odd rows at the end of chain 17, even rows at the
            // start of chain 39.
            "chain_edge" if row % 2 == 1 && k == 17 * 128 + 127 => 0x2b80_0001,
            "chain_edge" if row % 2 == 0 && row % 3 == 0 && k == 39 * 128 => 0x5390_0000,
            "all_out" if k == (row * 613) % HIDDEN => 0x2b00_0000 | (row as u32),
            "ties" => sign | 0x3f80_8000 | ((k as u32 % 7) << 16),
            _ => normal,
        };
    }
    words(&values)
}
fn weights(pattern: &str) -> Vec<u8> {
    let mut state = 0x7765_6967_6874_735fu64;
    (0..EXPERTS * HIDDEN)
        .flat_map(|at| {
            let r = xorshift(&mut state);
            let bits = if pattern == "tile_out" && at == 37 * HIDDEN + 2049 {
                0x7f00
            } else {
                ((r as u16) & 0x807f) | ((((r >> 16) as u16) % 30 + 112) << 7)
            };
            bits.to_ne_bytes()
        })
        .collect()
}

struct Loaded {
    functions: Functions,
    _modules: Vec<Module>,
}
fn load(s: &Session) -> LabResult<Loaded> {
    let modules = modules()?;
    let mut loaded = BTreeMap::new();
    for (name, (bytes, _)) in &modules {
        if name != "probe_sm120" {
            loaded.insert(name.clone(), s.load_cubin(bytes)?);
        }
    }
    let function = |symbol: &str| -> LabResult<Function> {
        let (name, _) = modules
            .iter()
            .find(|(_, (_, symbols))| symbols.iter().any(|s| s == symbol))
            .ok_or_else(|| format!("unstaged symbol {symbol}"))?;
        loaded[name].function(symbol).map_err(|e| e.to_string())
    };
    let [prepare, producer, classified, weights] = symbols();
    Ok(Loaded {
        functions: Functions {
            prepare: function(&prepare)?,
            producer: function(&producer)?,
            classified: function(&classified)?,
            weights: function(&weights)?,
        },
        _modules: loaded.into_values().collect(),
    })
}
fn probes() -> Cubins {
    let probe = Box::leak(
        std::fs::read(artifacts().join("probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    Cubins::new(
        Box::leak(Box::new([("probe_sm120", &probe[..])])),
        &[("probe_sm120", PROBE_SYMBOLS)],
    )
}
fn finish(s: &Session, done: &runtime_cuda::Event, seconds: u64, what: &str) -> LabResult<()> {
    done.record(&s.stream).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !done.query().map_err(|e| e.to_string())? {
        if Instant::now() > deadline {
            eprintln!("ROUTER_CLASSIFIED_WATCHDOG {what} did not finish within {seconds} s");
            std::process::exit(86);
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    Ok(())
}
/// Both arms' partials and tickets agree; the ticket is reset between its
/// guards; partials past the live rows keep their poison and a live row's
/// first partial is written.
fn check(outputs: &[Vec<Vec<u8>>], rows: u32, live: u32) -> LabResult<()> {
    compare(&outputs[0], &outputs[1])?;
    if outputs[0][1] != words(&[TICKET_GUARD, 0, TICKET_GUARD]) {
        return Err("ticket not reset between its guards".into());
    }
    let partials = &outputs[0][0];
    let active = live.min(rows) as usize;
    for chain in 0..CHAINS {
        for token in 0..rows as usize {
            let at = ((chain * rows as usize + token) * EXPERTS) * 4;
            let row = &partials[at..at + EXPERTS * 4];
            let poisoned = row.iter().all(|b| *b == 0xa5);
            if poisoned != (token >= active) {
                return Err(format!("chain {chain} token {token} written {}", !poisoned));
            }
        }
    }
    Ok(())
}
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let f = &k.functions;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    let done = s.context.event().map_err(|e| e.to_string())?;
    let seconds = if memcheck {
        SANITIZED_CASE_SECONDS
    } else {
        CASE_SECONDS
    };
    let inputs = Inputs {
        activation: s.allocate(MAX_ROWS * HIDDEN * 4)?,
        weight: s.allocate(EXPERTS * HIDDEN * 2)?,
        weight_domain: s.allocate(EXPERTS / 16 * 4)?,
        live: s.allocate(4)?,
    };
    let outputs = [Output::new(s)?, Output::new(s)?];
    let weight_patterns: &[&str] = if memcheck { &["tile_out"] } else { &WEIGHTS };
    for &weight in weight_patterns {
        upload(&inputs.weight, &weights(weight), &s.stream)?;
        let plan = kernel_core::contract::LaunchPlan {
            grid: [(EXPERTS / 16) as u32, 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        f.weights
            .launch(
                &s.stream,
                launch::config(plan)?,
                &[A::Buffer(&inputs.weight), A::Buffer(&inputs.weight_domain)],
            )
            .map_err(|e| e.to_string())?;
        finish(s, &done, seconds, "weight domain")?;
        let domain = download(&inputs.weight_domain, EXPERTS / 16 * 4, &s.stream)?;
        let tensor = domain
            .chunks(4)
            .filter(|w| u32::from_ne_bytes((*w).try_into().unwrap()) != 0)
            .count();
        if tensor != if weight == "tile_out" { 23 } else { 24 } {
            return Err(format!("{weight}: {tensor} tensor tiles"));
        }
        let row_counts: &[u32] = if memcheck { &[6, 32] } else { &ROWS };
        for &rows in row_counts {
            let graphs = if memcheck {
                Vec::new()
            } else {
                (0..2)
                    .map(|arm| {
                        s.capture(|stream| step(f, arm, rows, &inputs, &outputs[arm], stream))
                    })
                    .collect::<LabResult<Vec<_>>>()?
            };
            let patterns: &[&str] = if memcheck {
                &["finite", "chain_edge"]
            } else {
                &PATTERNS
            };
            for &pattern in patterns {
                upload(&inputs.activation, &activation(pattern), &s.stream)?;
                let lives: Vec<u32> = if memcheck {
                    vec![rows]
                } else {
                    vec![rows, 0, rows.div_ceil(2), rows + 5]
                };
                for &live in &lives {
                    upload(&inputs.live, &live.to_ne_bytes(), &s.stream)?;
                    let replays = if !memcheck && live == rows && REPLAYED.contains(&pattern) {
                        2
                    } else {
                        0
                    };
                    for replay in 0..=replays {
                        for output in &outputs {
                            output.reset(s)?;
                        }
                        for arm in 0..2 {
                            if replay > 0 {
                                graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                            } else {
                                step(f, arm, rows, &inputs, &outputs[arm], &s.stream)?;
                            }
                        }
                        let what =
                            format!("{weight} rows={rows} {pattern} live={live} replay={replay}");
                        finish(s, &done, seconds, &what)?;
                        let read = outputs
                            .iter()
                            .map(|o| o.read(s, rows))
                            .collect::<LabResult<Vec<_>>>()?;
                        check(&read, rows, live).map_err(|e| format!("{what}: {e}"))?;
                    }
                }
                bitwise += 1;
            }
            eprintln!(
                "ROUTER_CLASSIFIED_BITWISE {}",
                json!({"weights":weight,"rows":rows,"patterns":patterns.len()})
            );
            if memcheck || weight != "tensor" {
                continue;
            }
            if rows == 32 {
                upload(&inputs.activation, &activation("chain_edge"), &s.stream)?;
                upload(&inputs.live, &rows.to_ne_bytes(), &s.stream)?;
                for (output, graph) in outputs.iter().zip(&graphs) {
                    output.reset(s)?;
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                finish(s, &done, seconds, "negatives")?;
                let expected = outputs[0].read(s, rows)?;
                compare(&expected, &outputs[1].read(s, rows)?)?;
                let flips = [
                    (&outputs[1].workspace, DOMAIN_WORDS * 4, 0usize),
                    (&outputs[1].ticket, 4, 1usize),
                ];
                for (buffer, at, role) in flips {
                    let mut bytes = download(
                        buffer,
                        if role == 0 { workspace_bytes(rows) } else { 12 },
                        &s.stream,
                    )?;
                    bytes[at] ^= 1;
                    upload(buffer, &bytes, &s.stream)?;
                    if compare(&expected, &outputs[1].read(s, rows)?).is_ok() {
                        return Err("negative control escaped the comparison".into());
                    }
                    bytes[at] ^= 1;
                    upload(buffer, &bytes, &s.stream)?;
                    negative += 1;
                }
            }
            if !TIMED_ROWS.contains(&rows) {
                continue;
            }
            upload(&inputs.activation, &activation("finite"), &s.stream)?;
            upload(&inputs.live, &rows.to_ne_bytes(), &s.stream)?;
            for output in &outputs {
                output.reset(s)?;
            }
            let chains = (0..2)
                .map(|arm| {
                    s.capture(|stream| {
                        (0..CHAIN)
                            .try_for_each(|_| step(f, arm, rows, &inputs, &outputs[arm], stream))
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            let label = format!("router-classified:{rows}");
            let single = kernel_harness::with_timing_label(&label, || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                    graphs[arm].launch(stream).map_err(|e| e.to_string())
                })
            })?;
            let chained = kernel_harness::with_timing_label(&format!("{label}:chain"), || {
                kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                    chains[arm].launch(stream).map_err(|e| e.to_string())
                })
            })?;
            finish(s, &done, seconds, &format!("{label} timing"))?;
            compare(&outputs[0].read(s, rows)?, &outputs[1].read(s, rows)?)
                .map_err(|e| format!("{label} after timing: {e}"))?;
            let per = |ms: f32| ms * 1000. / CHAIN as f32;
            let record = json!({"rows":rows,
                "old_us":single.arms[0].median_ms * 1000.,"new_us":single.arms[1].median_ms * 1000.,
                "ratio":single.paired_ratio,"chain":CHAIN,"old_chain_us":per(chained.arms[0].median_ms),
                "new_chain_us":per(chained.arms[1].median_ms),"chain_ratio":chained.paired_ratio});
            eprintln!("ROUTER_CLASSIFIED_TIMING {record}");
            timings.push(record);
        }
    }
    let c = coverage();
    let expected = |key: &str| c[key].as_u64().unwrap();
    if bitwise as u64
        != if memcheck {
            expected("memcheck_cases")
        } else {
            expected("bitwise_cases")
        }
        || negative as u64
            != if memcheck {
                0
            } else {
                expected("negative_controls")
            }
        || timings.len() as u64
            != if memcheck {
                0
            } else {
                expected("timing_cases")
            }
    {
        return Err("executed coverage differs".into());
    }
    eprintln!(
        "ROUTER_CLASSIFIED_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_ROUTER_CLASSIFIED"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "router-classified",
            value
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    });
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}

/// Run under compute-sanitizer memcheck, racecheck and synccheck.
#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_ROUTER_CLASSIFIED"]
fn sanitized_producer_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
