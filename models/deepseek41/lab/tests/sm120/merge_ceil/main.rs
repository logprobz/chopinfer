//! The quantizing attention merges
//! (`bundle_deepseek41::sm120::latent_merge_ceil`) against the served pair
//! each replaces where Flash TP4's WO_a is a status fold: the served latent
//! merge, then WO_a's served row-sliced quantizer, in both compiled forms,
//! on the same status rows. Every BF16 output, code, scale and the status,
//! bitwise, over partial, sink and RoPE patterns (non-finite and
//! overflowing values, empty partials, floors, ties and a sticky status),
//! row counts through 64, live counts, graph replays and output-flip
//! negatives, then paired timing of one step and of a chain at decode and
//! verify widths. All modules are the bundle's embedded, locked cubins.
use bundle_deepseek41::sm120::{
    fold_cubin_for_symbol, fold_fused, latent_merge_ceil as registration,
};
use kernel_core::latent_attention::{LatentAttentionQuery, LatentKvFormat};
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::gemm::mxfp8_fold_cluster_fused::Quantizer;
use kernels_sm120::latent::sources::Entry;
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::PathBuf,
    time::{Duration, Instant},
};

/// Flash TP4's query heads on a rank, and the latent width.
const HEADS: usize = 16;
const LATENT: usize = 512;
const WIDTH: usize = HEADS * LATENT;
/// The widest status-fold bucket; buffers hold one more poisoned row.
const MAX_ROWS: usize = 64;
const ROWS: [u32; 14] = [1, 2, 3, 4, 5, 6, 8, 12, 16, 31, 32, 33, 60, 64];
const TIMED_ROWS: [u32; 6] = [1, 4, 6, 8, 32, 64];
const OFFSET: usize = 3;
const GUARD: u32 = 0x1357_2468;
const PATTERNS: [&str; 14] = [
    "finite",
    "zero",
    "wide",
    "tiny",
    "ties",
    "empty",
    "split_empty",
    "num_nan",
    "num_inf",
    "overflow",
    "sink_nan",
    "sink_dominant",
    "rope_nan",
    "sticky",
];
/// Patterns replayed through graphs.
const REPLAYED: [&str; 3] = ["finite", "num_nan", "sticky"];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
/// Steps per chained timing graph.
const CHAIN: usize = 16;
const CASE_SECONDS: u64 = 10;
const SANITIZED_CASE_SECONDS: u64 = 60;
const ROLES: [&str; 4] = ["out", "codes", "scales", "status"];

fn twins() -> Vec<Entry> {
    registration::entries()
}
/// The matrix's row counts within `twin`'s capacity: decode entries serve
/// at most the decode bucket.
fn rows_of(twin: &Entry, rows: &[u32]) -> Vec<u32> {
    rows.iter()
        .copied()
        .filter(|&r| twin.merge_plan(r).is_ok())
        .collect()
}
/// WO_a's compiled sliced quantizers, one per symbol.
fn quantizers() -> Vec<Quantizer> {
    let mut all: Vec<Quantizer> = Vec::new();
    for q in fold_fused::quantizers() {
        if !all.iter().any(|a| a.symbol == q.symbol) {
            all.push(q);
        }
    }
    all
}
fn bf16(entry: &Entry) -> bool {
    matches!(
        entry.query,
        LatentAttentionQuery::Attention {
            kv: LatentKvFormat::Bf16,
            ..
        }
    )
}
/// The merge's sink and RoPE tail operands, by its parameter names.
fn operands(entry: &Entry) -> (bool, bool) {
    let params = entry.merge_params().unwrap();
    let named = |name: &str| {
        params.iter().any(
            |p| matches!(p, kernel_build::EntryParam::DevicePointer { name: n, .. } if n == name),
        )
    };
    (named("sinks"), named("cos_sin"))
}
fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_MERGE_CEIL").expect("frozen merge artifacts required"),
    )
}
/// Module name to its bytes and the symbols the matrix uses.
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
fn symbols() -> Vec<String> {
    let mut symbols: Vec<String> = Vec::new();
    for twin in twins() {
        symbols.push(twin.merge_symbol());
        symbols.push(twin.merge_ceil_symbol());
    }
    symbols.extend(quantizers().into_iter().map(|q| q.symbol));
    symbols
}
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
fn timed(entry: &Entry) -> bool {
    !bf16(entry)
}
fn coverage() -> serde_json::Value {
    let twins = twins();
    let quantizers = quantizers();
    let bitwise: usize = twins.iter().map(|t| rows_of(t, &ROWS).len()).sum();
    let timed: usize = twins
        .iter()
        .filter(|t| timed(t))
        .map(|t| rows_of(t, &TIMED_ROWS).len())
        .sum();
    json!({"entries": twins.iter().map(|t| json!({"twin":t.merge_ceil_symbol(),
            "merge":t.merge_symbol(),"bf16":bf16(t),"splits":t.splits})).collect::<Vec<_>>(),
        "quantizers": quantizers.iter().map(|q| q.symbol.clone()).collect::<Vec<_>>(),
        "rows": ROWS, "timed_rows": TIMED_ROWS, "patterns": PATTERNS, "replayed": REPLAYED,
        "chain": CHAIN,
        "bitwise_cases": bitwise * PATTERNS.len(),
        "timing_cases": timed * quantizers.len(),
        "negative_controls": twins.len() * ROLES.len(),
        "memcheck_cases": twins.len()})
}

#[test]
fn twins_pair_with_the_served_merge_and_sliced_quantizer() {
    let twins = twins();
    assert_eq!(twins.len(), 14);
    assert!(twins.iter().any(bf16) && !twins.iter().all(bf16));
    let quantizers = quantizers();
    assert_eq!(quantizers.len(), 2);
    for q in &quantizers {
        assert_eq!(q.width as usize, WIDTH);
        assert!(!q.f32_input);
        assert_eq!(q.spec().params.len(), 6);
        assert!(fold_cubin_for_symbol(&q.symbol).is_some(), "{}", q.symbol);
    }
    for twin in &twins {
        let LatentAttentionQuery::Attention { heads, latent, .. } = twin.query else {
            panic!("{}: an attention query", twin.merge_symbol());
        };
        assert_eq!((heads as usize, latent as usize), (HEADS, LATENT));
        assert_eq!(operands(twin), (true, true), "{}", twin.merge_symbol());
        let rows = rows_of(twin, &ROWS);
        let widest = *rows.last().unwrap();
        assert!(widest == 32 || widest == 64, "{}", twin.merge_symbol());
        assert_eq!(rows.len(), ROWS.iter().filter(|&&r| r <= widest).count());
        let plan = twin.merge_plan(widest).unwrap();
        assert_eq!(plan.grid, [widest, HEADS as u32, 1]);
        assert_eq!(
            twin.workspace_bytes(1),
            (twin.splits as usize * HEADS * (LATENT + 2) * 4) as u64
        );
        for symbol in [twin.merge_symbol(), twin.merge_ceil_symbol()] {
            assert!(fold_cubin_for_symbol(&symbol).is_some(), "{symbol}");
        }
    }
    assert_eq!(FLASH.shape.attention.heads as usize / 4, HEADS);
    let original = vec![vec![0u8; 8]; ROLES.len()];
    for role in 0..ROLES.len() {
        let mut bad = original.clone();
        bad[role][0] = 1;
        assert!(compare(&original, &bad).is_err());
    }
    assert!(coverage()["bitwise_cases"].as_u64().unwrap() > 14 * 10 * 14);
    assert!(TIMED_ROWS.iter().all(|r| ROWS.contains(r)));
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_MERGE_CEIL"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_MERGE_CEIL_EXPORT").is_some() {
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
        "MERGE_CEIL_MODULES {} PASS",
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

/// The merge's inputs, shared by every arm.
struct Inputs {
    workspace: DeviceBuffer,
    sinks: DeviceBuffer,
    cos_sin: DeviceBuffer,
    live: DeviceBuffer,
}
struct Output {
    buffers: [DeviceBuffer; 4],
}
const LENGTHS: [usize; 4] = [
    (MAX_ROWS + 1) * WIDTH * 2,
    (MAX_ROWS + 1) * WIDTH,
    (MAX_ROWS + 1) * WIDTH / 32,
    (MAX_ROWS + OFFSET + 2) * 4,
];
impl Output {
    fn new(s: &Session) -> LabResult<Self> {
        let buffers = LENGTHS
            .iter()
            .map(|&len| s.allocate(len))
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            buffers: buffers.try_into().map_err(|_| "four buffers")?,
        })
    }
    /// Poison everywhere, and each status row zero or sticky.
    fn reset(&self, s: &Session, sticky: bool) -> LabResult<()> {
        for (buffer, &len) in self.buffers.iter().zip(&LENGTHS).take(3) {
            upload(buffer, &vec![0xa5; len], &s.stream)?;
        }
        let mut flags = vec![GUARD; MAX_ROWS + OFFSET + 2];
        flags[OFFSET..OFFSET + MAX_ROWS].fill(if sticky { 0x80 } else { 0 });
        upload(&self.buffers[3], &words(&flags), &s.stream)
    }
    /// `rows` rows and the poisoned row after them, and every status word.
    fn read(&self, s: &Session, rows: u32) -> LabResult<Vec<Vec<u8>>> {
        let n = rows as usize + 1;
        let lengths = [n * WIDTH * 2, n * WIDTH, n * WIDTH / 32, LENGTHS[3]];
        self.buffers
            .iter()
            .zip(lengths)
            .map(|(b, n)| download(b, n, &s.stream))
            .collect()
    }
}
fn merge_arguments<'a>(entry: &Entry, i: &'a Inputs, o: &'a Output) -> Vec<A<'a>> {
    let (sinks, rope) = operands(entry);
    let mut arguments = vec![
        A::Buffer(&i.workspace),
        A::Buffer(&i.live),
        A::Buffer(&o.buffers[0]),
    ];
    if sinks {
        arguments.push(A::Buffer(&i.sinks));
    }
    if rope {
        arguments.push(A::Buffer(&i.cos_sin));
    }
    arguments
}
/// The functions of one twin: its merge, the twin, and each quantizer
/// with its registration.
struct Functions {
    merge: Function,
    twin: Function,
    quantizers: Vec<(Function, Quantizer)>,
}
/// Arm `q` the served merge then quantizer `q`, the last arm the twin.
fn step(
    entry: &Entry,
    f: &Functions,
    arm: usize,
    rows: u32,
    i: &Inputs,
    o: &Output,
    stream: &Stream,
) -> LabResult<()> {
    let plan = launch::config(entry.merge_plan(rows).map_err(|e| e.to_string())?)?;
    let mut arguments = merge_arguments(entry, i, o);
    if let Some((quantizer, q)) = f.quantizers.get(arm) {
        f.merge
            .launch(stream, plan, &arguments)
            .map_err(|e| e.to_string())?;
        let [out, codes, scales, status] = &o.buffers;
        quantizer
            .launch(
                stream,
                launch::config(q.plan(rows).map_err(|e| e.to_string())?)?,
                &[
                    A::Buffer(out),
                    A::Buffer(codes),
                    A::Buffer(scales),
                    A::Buffer(&i.live),
                    A::Buffer(status),
                    A::U64(OFFSET as u64),
                ],
            )
            .map_err(|e| e.to_string())
    } else {
        let [_, codes, scales, status] = &o.buffers;
        arguments.extend([
            A::Buffer(codes),
            A::Buffer(scales),
            A::Buffer(status),
            A::U64(OFFSET as u64),
        ]);
        f.twin
            .launch(stream, plan, &arguments)
            .map_err(|e| e.to_string())
    }
}
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}
/// Each (row, split, head) partial: its maximum, denominator and
/// numerator.
fn workspace(pattern: &str, splits: usize) -> Vec<u8> {
    let mut state = 0x6d65_7267_655f_6365u64;
    let partial = LATENT + 2;
    let mut values = vec![0u32; MAX_ROWS * splits * HEADS * partial];
    for (pair, block) in values.chunks_mut(partial).enumerate() {
        let (row, split, head) = (pair / (splits * HEADS), pair / HEADS % splits, pair % HEADS);
        let r = xorshift(&mut state);
        let max = ((r % 800) as f32 - 400.) / 100.;
        let empty = match pattern {
            "empty" => head % 3 == 0,
            "split_empty" => split % 2 == 1 || (head == 2 && split != 0),
            _ => false,
        };
        block[0] = if empty { f32::NEG_INFINITY } else { max }.to_bits();
        let den = 0.5 + (r >> 20) as f32 % 4000. / 100.;
        block[1] = if pattern == "overflow" {
            1e-30_f32
        } else {
            den
        }
        .to_bits();
        for (d, value) in block[2..].iter_mut().enumerate() {
            let r = xorshift(&mut state);
            let sign = r as u32 & 0x8000_0000;
            let mantissa = r as u32 & 0x7f_ffff;
            *value = match pattern {
                "zero" => sign,
                "wide" => sign | (((r >> 32) as u32 % 200 + 27) << 23) | mantissa,
                "tiny" => sign | (((r >> 32) as u32 % 24 + 90) << 23) | mantissa,
                "ties" => sign | 0x3f80_8000 | ((d as u32 % 7) << 16),
                "num_nan" if row == 0 && split == 0 && head == 3 && d == 17 => 0x7fc1_0000,
                "num_inf" if row == 1 && split + 1 == splits && head == 15 && d == 511 => {
                    0xff80_0000
                }
                "overflow" => sign | (250 << 23) | mantissa,
                _ => sign | (((r >> 32) as u32 % 16 + 119) << 23) | mantissa,
            };
        }
    }
    words(&values)
}
fn set_inputs(s: &Session, pattern: &str, i: &Inputs, splits: usize) -> LabResult<()> {
    upload(&i.workspace, &workspace(pattern, splits), &s.stream)?;
    let sinks: Vec<u32> = (0..HEADS)
        .map(|h| match pattern {
            "sink_nan" if h == 5 => f32::NAN.to_bits(),
            "sink_dominant" => 40.0_f32.to_bits(),
            _ => (h as f32 / 4. - 2.).to_bits(),
        })
        .collect();
    upload(&i.sinks, &words(&sinks), &s.stream)?;
    let mut state = 0x726f_7065_7461_696cu64;
    let table: Vec<u32> = (0..MAX_ROWS * 64)
        .map(|at| {
            if pattern == "rope_nan" && at == 2 * 64 + 9 {
                return f32::NAN.to_bits();
            }
            let angle = (xorshift(&mut state) % 6283) as f32 / 1000.;
            if at % 2 == 0 {
                angle.cos()
            } else {
                angle.sin()
            }
            .to_bits()
        })
        .collect();
    upload(&i.cos_sin, &words(&table), &s.stream)
}

struct Loaded {
    functions: Vec<Functions>,
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
    let functions = twins()
        .iter()
        .map(|t| {
            Ok(Functions {
                merge: function(&t.merge_symbol())?,
                twin: function(&t.merge_ceil_symbol())?,
                quantizers: quantizers()
                    .into_iter()
                    .map(|q| Ok((function(&q.symbol)?, q)))
                    .collect::<LabResult<_>>()?,
            })
        })
        .collect::<LabResult<Vec<_>>>()?;
    Ok(Loaded {
        functions,
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
/// Waits for the stream's work through `done`, or stops the process when
/// it takes longer than `seconds`.
fn finish(s: &Session, done: &runtime_cuda::Event, seconds: u64, what: &str) -> LabResult<()> {
    done.record(&s.stream).map_err(|e| e.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    while !done.query().map_err(|e| e.to_string())? {
        if Instant::now() > deadline {
            eprintln!("MERGE_CEIL_WATCHDOG {what} did not finish within {seconds} s");
            std::process::exit(86);
        }
        std::thread::sleep(Duration::from_micros(100));
    }
    Ok(())
}
/// Checks one case's outputs: every served arm equals the first and the
/// twin equals them; the status guard words, the poisoned row after the
/// grid and every idle row keep their poison; finite partials leave each
/// live row's status as it started and a non-finite input flags its row.
fn check(outputs: &[Vec<Vec<u8>>], rows: u32, live: u32, pattern: &str) -> LabResult<()> {
    let served = &outputs[0];
    for (arm, other) in outputs.iter().enumerate().skip(1) {
        compare(served, other).map_err(|e| format!("arm {arm}: {e}"))?;
    }
    let status = &served[3];
    let word = |r: usize| u32::from_ne_bytes(status[r * 4..r * 4 + 4].try_into().unwrap());
    if (0..OFFSET)
        .chain(OFFSET + MAX_ROWS..OFFSET + MAX_ROWS + 2)
        .any(|r| word(r) != GUARD)
    {
        return Err("status guard changed".into());
    }
    let active = live.min(rows) as usize;
    for (role, bytes) in served[..3].iter().enumerate() {
        let row = bytes.len() / (rows as usize + 1);
        if bytes[active * row..].iter().any(|b| *b != 0xa5) {
            return Err(format!("{} wrote past the live rows", ROLES[role]));
        }
        if active > 0 && bytes[..row].iter().all(|b| *b == 0xa5) {
            return Err(format!("{} stored nothing for row 0", ROLES[role]));
        }
    }
    let start = if pattern == "sticky" { 0x80 } else { 0 };
    for r in 0..MAX_ROWS {
        let value = word(OFFSET + r);
        if r >= active && value != start {
            return Err(format!("idle row {r} status {value:#x}"));
        }
    }
    if active > 0 {
        let first = word(OFFSET);
        if pattern == "finite" && (0..active).any(|r| word(OFFSET + r) != 0) {
            return Err("finite partials flagged a row".into());
        }
        if matches!(pattern, "num_nan" | "sink_nan" | "overflow") && first == 0 {
            return Err(format!("{pattern} left row 0 unflagged"));
        }
    }
    Ok(())
}
/// Every twin at every row count and pattern against both served pairs,
/// eager at full, zero, partial and excess live counts, and through two
/// graph replays for the replayed patterns; `memcheck` runs two patterns
/// at three row counts per twin, eagerly, without timing.
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let quantizers = quantizers();
    let arms = quantizers.len() + 1;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    let done = s.context.event().map_err(|e| e.to_string())?;
    let seconds = if memcheck {
        SANITIZED_CASE_SECONDS
    } else {
        CASE_SECONDS
    };
    let splits = twins().iter().map(|t| t.splits as usize).max().unwrap();
    let inputs = Inputs {
        workspace: s.allocate(MAX_ROWS * splits * HEADS * (LATENT + 2) * 4)?,
        sinks: s.allocate(HEADS * 4)?,
        cos_sin: s.allocate(MAX_ROWS * 64 * 4)?,
        live: s.allocate(4)?,
    };
    let outputs = (0..arms)
        .map(|_| Output::new(s))
        .collect::<LabResult<Vec<_>>>()?;
    for (twin, functions) in twins().iter().zip(&k.functions) {
        let symbol = twin.merge_ceil_symbol();
        let served_rows = rows_of(twin, &ROWS);
        let widest = *served_rows.last().unwrap();
        let row_counts = if memcheck {
            vec![1, widest]
        } else {
            served_rows
        };
        for rows in row_counts {
            let graphs = if memcheck {
                Vec::new()
            } else {
                (0..arms)
                    .map(|arm| {
                        s.capture(|stream| {
                            step(twin, functions, arm, rows, &inputs, &outputs[arm], stream)
                        })
                    })
                    .collect::<LabResult<Vec<_>>>()?
            };
            let patterns: &[&str] = if memcheck {
                &["finite", "num_nan"]
            } else {
                &PATTERNS
            };
            for &pattern in patterns {
                set_inputs(s, pattern, &inputs, twin.splits as usize)?;
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
                            output.reset(s, pattern == "sticky")?;
                        }
                        for arm in 0..arms {
                            if replay > 0 {
                                graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                            } else {
                                step(
                                    twin,
                                    functions,
                                    arm,
                                    rows,
                                    &inputs,
                                    &outputs[arm],
                                    &s.stream,
                                )?;
                            }
                        }
                        let what =
                            format!("{symbol} rows={rows} {pattern} live={live} replay={replay}");
                        finish(s, &done, seconds, &what)?;
                        let read = outputs
                            .iter()
                            .map(|o| o.read(s, rows))
                            .collect::<LabResult<Vec<_>>>()?;
                        check(&read, rows, live, pattern).map_err(|e| format!("{what}: {e}"))?;
                    }
                }
                bitwise += 1;
            }
            eprintln!(
                "MERGE_CEIL_BITWISE {}",
                json!({"symbol":symbol,"rows":rows,"patterns":patterns.len()})
            );
            if memcheck {
                continue;
            }
            if rows == widest {
                // Output flips on the twin's buffers escape no comparison.
                set_inputs(s, "finite", &inputs, twin.splits as usize)?;
                upload(&inputs.live, &rows.to_ne_bytes(), &s.stream)?;
                for (output, graph) in outputs.iter().zip(&graphs) {
                    output.reset(s, false)?;
                    graph.launch(&s.stream).map_err(|e| e.to_string())?;
                }
                finish(s, &done, seconds, &format!("{symbol} negatives"))?;
                let expected = outputs[0].read(s, rows)?;
                let twin_out = &outputs[arms - 1];
                compare(&expected, &twin_out.read(s, rows)?)?;
                for role in 0..ROLES.len() {
                    let mut bad = expected[role].clone();
                    let at = if role == 3 { OFFSET * 4 } else { 0 };
                    bad[at] ^= 1;
                    upload(&twin_out.buffers[role], &bad, &s.stream)?;
                    if compare(&expected, &twin_out.read(s, rows)?).is_ok() {
                        return Err("negative control escaped the comparison".into());
                    }
                    upload(&twin_out.buffers[role], &expected[role], &s.stream)?;
                    negative += 1;
                }
            }
            if !timed(twin) || !TIMED_ROWS.contains(&rows) {
                continue;
            }
            set_inputs(s, "finite", &inputs, twin.splits as usize)?;
            upload(&inputs.live, &rows.to_ne_bytes(), &s.stream)?;
            for output in &outputs {
                output.reset(s, false)?;
            }
            let twin_arm = arms - 1;
            let chain = |arm: usize| {
                s.capture(|stream| {
                    (0..CHAIN).try_for_each(|_| {
                        step(twin, functions, arm, rows, &inputs, &outputs[arm], stream)
                    })
                })
            };
            let twin_chain = chain(twin_arm)?;
            for (q, quantizer) in quantizers.iter().enumerate() {
                let served_chain = chain(q)?;
                let label = format!("merge-ceil:{symbol}:{}:{rows}", quantizer.symbol);
                let single = kernel_harness::with_timing_label(&label, || {
                    kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        graphs[if arm == 0 { q } else { twin_arm }]
                            .launch(stream)
                            .map_err(|e| e.to_string())
                    })
                })?;
                let chained = kernel_harness::with_timing_label(&format!("{label}:chain"), || {
                    kernel_harness::time_ab(s, 3, 15, [EvictionMode::Read; 2], |arm, stream| {
                        if arm == 0 { &served_chain } else { &twin_chain }
                            .launch(stream)
                            .map_err(|e| e.to_string())
                    })
                })?;
                finish(s, &done, seconds, &format!("{label} timing"))?;
                compare(
                    &outputs[q].read(s, rows)?,
                    &outputs[twin_arm].read(s, rows)?,
                )
                .map_err(|e| format!("{label} after timing: {e}"))?;
                let per = |ms: f32| ms * 1000. / CHAIN as f32;
                let record = json!({"symbol":symbol,"merge":twin.merge_symbol(),
                    "quantizer":quantizer.symbol,"rows":rows,
                    "old_us":single.arms[0].median_ms * 1000.,"new_us":single.arms[1].median_ms * 1000.,
                    "ratio":single.paired_ratio,"chain":CHAIN,"old_chain_us":per(chained.arms[0].median_ms),
                    "new_chain_us":per(chained.arms[1].median_ms),"chain_ratio":chained.paired_ratio});
                eprintln!("MERGE_CEIL_TIMING {record}");
                timings.push(record);
            }
        }
    }
    let c = coverage();
    let expected = |key: &str| c[key].as_u64().unwrap();
    if bitwise as u64
        != if memcheck {
            expected("memcheck_cases") * 2 * 2
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
        "MERGE_CEIL_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_MERGE_CEIL"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "merge-ceil",
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
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_MERGE_CEIL"]
fn sanitized_twin_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
