//! The clustered query RoPE and window write twins
//! (`bundle_deepseek41::sm120::query_window`) against the served pair they
//! would replace in Flash TP4's one-row decode: the served four-group query
//! RoPE, then the served fused window write, on the same status row. Every
//! rotated query, native query head, latent, code, scale, ring cell and the
//! status, bitwise, over query, KV, weight and table patterns (non-finite
//! values in either half, and in both), ring addressing cases, live counts,
//! graph replays and output-flip negatives, then paired timing of one step
//! and of a PDL chain. All modules are the bundle's embedded, locked cubins.
use bundle_deepseek41::sm120::{
    fold_cubin_for_symbol, latent_scheduled, query_window as registration, window_fused,
};
use kernel_core::latent_attention::LatentKvFormat;
use kernel_harness::{Cubins, LabResult, Session, eviction::EvictionMode, launch};
use kernels_sm120::latent::query_window_twin::Entry;
use model_deepseek41_contract::{
    geometry::FLASH,
    own_ops::rope::{self, RopeTable},
};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument as A, Module, Stream};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

const ROWS: u32 = registration::ROWS;
/// Flash TP4's query heads on a rank.
const HEADS: u32 = 16;
const SLOTS: usize = 136;
const OFFSET: usize = 3;
const EPS: f32 = 1e-6;
const PATTERNS: [&str; 18] = [
    "finite",
    "zero",
    "wide",
    "tiny",
    "query_nan_first",
    "query_nan_tail",
    "query_inf_head",
    "kv_nan",
    "kv_inf",
    "weight_nan",
    "rope_nan",
    "query_and_kv_nan",
    "sticky",
    "negative",
    "bad_sequence",
    "stale",
    "future",
    "ties",
];
const LIVES: [u32; 4] = [ROWS, 0, ROWS + 7, ROWS];
const PROBE_SYMBOLS: &[&str] = &[
    "chop_probe_gate_v1",
    "chop_probe_flush_v1",
    "chop_probe_read_v1",
];
/// Steps per chained timing graph.
const CHAIN: usize = 16;
const ROLES: [&str; 8] = [
    "query",
    "native_q",
    "latent",
    "quantized_codes",
    "quantized_scales",
    "window_codes",
    "window_scales",
    "status",
];

/// Each twin with the served RoPE and fused window entries of its format.
struct Pair {
    twin: Entry,
    rope: kernels_sm120::latent::query_parallel::Entry,
    window: kernels_sm120::latent::window_fused::Entry,
}
impl Pair {
    fn bf16(&self) -> bool {
        self.twin.kv == LatentKvFormat::Bf16
    }
    fn code_stride(&self) -> usize {
        SLOTS * if self.bf16() { 1024 } else { 512 } + 66
    }
    fn scale_stride(&self) -> usize {
        SLOTS * 16 + 7
    }
    fn symbols(&self) -> [&str; 3] {
        [&self.rope.symbol, &self.window.symbol, &self.twin.symbol]
    }
}
fn pairs() -> Vec<Pair> {
    registration::entries()
        .into_iter()
        .map(|twin| Pair {
            rope: latent_scheduled::query(),
            window: window_fused::entries()
                .into_iter()
                .find(|w| w.kv == twin.kv)
                .unwrap(),
            twin,
        })
        .collect()
}
fn artifacts() -> PathBuf {
    PathBuf::from(
        std::env::var_os("CHOPINFER_QUERY_WINDOW").expect("frozen twin artifacts required"),
    )
}
/// Module name to its bytes and the symbols the matrix uses.
type Modules = BTreeMap<String, (Vec<u8>, Vec<String>)>;
fn modules() -> LabResult<Modules> {
    let mut modules = Modules::new();
    for pair in pairs() {
        for symbol in pair.symbols() {
            let (name, bytes) = fold_cubin_for_symbol(symbol).ok_or("entry must be embedded")?;
            let module = modules
                .entry(name.into())
                .or_insert_with(|| (bytes.to_vec(), Vec::new()));
            if !module.1.iter().any(|s| s == symbol) {
                module.1.push(symbol.into());
            }
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
    let pairs = pairs();
    json!({"entries": pairs.iter().map(|p| json!({"twin":p.twin.symbol,"rope":p.rope.symbol,
            "window":p.window.symbol,"bf16":p.bf16()})).collect::<Vec<_>>(),
        "rows": ROWS, "heads": HEADS, "patterns": PATTERNS, "lives": LIVES, "chain": CHAIN,
        "bitwise_cases": pairs.len() * PATTERNS.len(), "timing_cases": pairs.len(),
        "negative_controls": pairs.len() * ROLES.len(), "memcheck_cases": pairs.len()})
}

#[test]
fn twins_pair_with_the_served_one_row_rope_and_window_write() {
    let pairs = pairs();
    assert_eq!(pairs.len(), 2);
    for pair in &pairs {
        assert_eq!(pair.twin.rope().spec().params, pair.rope.spec().params);
        assert_eq!(pair.twin.window().spec().params, pair.window.spec().params);
        assert_eq!(pair.twin.window().kv, pair.window.kv);
        assert_eq!(pair.twin.capacity, ROWS);
        assert!(pair.rope.capacity >= ROWS && pair.window.capacity >= ROWS);
        assert_eq!(
            pair.rope.symbol,
            "chop_deepseek41_latent_query_rope_heads_v1"
        );
        let plan = pair.twin.plan(ROWS, HEADS).unwrap();
        assert_eq!((plan.grid, plan.cluster), ([1, 2, 1], [1, 2, 1]));
        for symbol in pair.symbols() {
            assert!(fold_cubin_for_symbol(symbol).is_some(), "{symbol}");
        }
    }
    assert!(pairs.iter().any(Pair::bf16) && !pairs.iter().all(Pair::bf16));
    assert_eq!(FLASH.shape.attention.heads / 4, HEADS);
    let original = vec![vec![0u8; 8]; ROLES.len()];
    for role in 0..ROLES.len() {
        let mut bad = original.clone();
        bad[role][0] = 1;
        assert!(compare(&original, &bad).is_err());
    }
    assert_eq!(coverage()["bitwise_cases"], 2 * 18);
}

/// Explicit artifact staging and checking, never part of an ordinary run.
#[test]
#[ignore = "requires an external frozen artifact directory named by CHOPINFER_QUERY_WINDOW"]
fn frozen_modules_are_complete() {
    let dir = artifacts();
    let modules = modules().unwrap();
    if std::env::var_os("CHOPINFER_QUERY_WINDOW_EXPORT").is_some() {
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
        "QUERY_WINDOW_MODULES {} PASS",
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
fn bytes<T: Copy>(values: &[T], f: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|v| f(*v)).collect()
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

/// One arm's inputs (the query is rotated in place, so each arm owns one).
struct Inputs {
    kv: DeviceBuffer,
    weight: DeviceBuffer,
    table: DeviceBuffer,
    live: DeviceBuffer,
    positions: DeviceBuffer,
    batches: DeviceBuffer,
    last: DeviceBuffer,
}
struct Output {
    buffers: [DeviceBuffer; 8],
    lengths: [usize; 8],
}
impl Output {
    fn new(s: &Session, p: &Pair) -> LabResult<Self> {
        let n = ROWS as usize;
        let q = n * HEADS as usize * 512;
        let lengths = [
            q * 4,
            q * 2,
            n * 512 * 4,
            n * 512,
            n * 16,
            n * p.code_stride(),
            n * p.scale_stride(),
            (n + OFFSET + 2) * 4,
        ];
        let buffers = lengths
            .iter()
            .map(|&len| s.allocate(len))
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            buffers: buffers.try_into().map_err(|_| "eight buffers")?,
            lengths,
        })
    }
    /// The query's initial values, and poison everywhere else.
    fn reset(&self, s: &Session, query: &[u8], sticky: bool) -> LabResult<()> {
        upload(&self.buffers[0], query, &s.stream)?;
        for i in 1..7 {
            upload(&self.buffers[i], &vec![0xa5; self.lengths[i]], &s.stream)?;
        }
        let mut flags = vec![0x1357_2468u32; ROWS as usize + OFFSET + 2];
        flags[OFFSET..OFFSET + ROWS as usize].fill(if sticky { 0x80 } else { 0 });
        upload(
            &self.buffers[7],
            &bytes(&flags, |v| v.to_ne_bytes().to_vec()),
            &s.stream,
        )
    }
    fn read(&self, s: &Session) -> LabResult<Vec<Vec<u8>>> {
        self.buffers
            .iter()
            .zip(self.lengths)
            .map(|(b, n)| download(b, n, &s.stream))
            .collect()
    }
}
/// The window write's arguments after its live count and status.
fn window_arguments<'a>(p: &Pair, i: &'a Inputs, o: &'a Output) -> Vec<A<'a>> {
    let [_, _, latent, qcodes, qscales, codes, scales, status] = &o.buffers;
    vec![
        A::Buffer(&i.kv),
        A::Buffer(&i.weight),
        A::Buffer(latent),
        A::Buffer(&i.table),
        A::Buffer(&i.live),
        A::U32(EPS.to_bits()),
        A::Buffer(status),
        A::U64(OFFSET as u64),
        A::Buffer(qcodes),
        A::Buffer(qscales),
        A::Buffer(&i.positions),
        A::Buffer(&i.batches),
        A::NullBuffer,
        A::Buffer(&i.last),
        A::Buffer(codes),
        if p.bf16() {
            A::NullBuffer
        } else {
            A::Buffer(scales)
        },
        A::U32(ROWS),
        A::U32(SLOTS as u32),
        A::U32(1),
        A::U64(p.code_stride() as u64),
        A::U64(if p.bf16() { 0 } else { p.scale_stride() as u64 }),
    ]
}
/// Arm 0 the served RoPE then the served window write, arm 1 the twin.
fn step(
    p: &Pair,
    f: &[Function; 3],
    arm: usize,
    i: &Inputs,
    o: &Output,
    stream: &Stream,
) -> LabResult<()> {
    let [query, native_q, .., status] = &o.buffers;
    if arm == 0 {
        let rope = [
            A::Buffer(query),
            A::Buffer(native_q),
            A::Buffer(&i.table),
            A::Buffer(&i.live),
            A::U32(HEADS),
            A::Buffer(status),
            A::U64(OFFSET as u64),
        ];
        f[0].launch(
            stream,
            launch::config(p.rope.plan(ROWS, HEADS).map_err(|e| e.to_string())?)?,
            &rope,
        )
        .map_err(|e| e.to_string())?;
        f[1].launch(
            stream,
            launch::config(p.window.plan(ROWS).map_err(|e| e.to_string())?)?,
            &window_arguments(p, i, o),
        )
        .map_err(|e| e.to_string())
    } else {
        let mut twin = vec![
            A::Buffer(query),
            A::Buffer(native_q),
            A::Buffer(&i.table),
            A::U32(HEADS),
        ];
        twin.extend(window_arguments(p, i, o));
        f[2].launch(
            stream,
            launch::config(p.twin.plan(ROWS, HEADS).map_err(|e| e.to_string())?)?,
            &twin,
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
fn query_values(pattern: &str) -> Vec<u8> {
    let mut state = 0x51ce_4a11_0bad_cafeu64;
    let n = ROWS as usize * HEADS as usize * 512;
    let values: Vec<u32> = (0..n)
        .map(|i| {
            let r = xorshift(&mut state);
            let sign = r as u32 & 0x8000_0000;
            let (head, column) = (i / 512 % HEADS as usize, i % 512);
            match pattern {
                "query_nan_first" | "query_and_kv_nan" if i == 0 => 0x7fc1_0000,
                "query_nan_tail" if head == HEADS as usize - 1 && column == 500 => 0xff81_0000,
                "query_inf_head" if head == 7 && column == 3 => 0x7f80_0000,
                _ => sign | (((r >> 32) as u32 % 16 + 119) << 23) | (r as u32 & 0x7f_ffff),
            }
        })
        .collect();
    bytes(&values, |v| v.to_ne_bytes().to_vec())
}
fn kv_values(pattern: &str) -> Vec<u8> {
    let mut state = 0x9ac1_04d2_1122_3344u64;
    let values: Vec<u32> = (0..ROWS as usize * 512)
        .map(|i| {
            let r = xorshift(&mut state);
            let sign = r as u32 & 0x8000_0000;
            match pattern {
                "zero" => sign,
                "tiny" => sign | ((r as u32 & 0x7f_ffff).max(1)),
                "wide" => sign | (((r >> 32) as u32 % 254 + 1) << 23) | (r as u32 & 0x7f_ffff),
                "kv_nan" | "query_and_kv_nan" if i == 511 => 0xff81_0000,
                "kv_inf" if i == 0 => 0x7f80_0000,
                "ties" => sign | 0x3f80_8000 | ((i as u32 % 7) << 16),
                _ => sign | (((r >> 32) as u32 % 20 + 117) << 23) | (r as u32 & 0x7f_ffff),
            }
        })
        .collect();
    bytes(&values, |v| v.to_ne_bytes().to_vec())
}
fn set_inputs(s: &Session, pattern: &str, i: &Inputs) -> LabResult<()> {
    upload(&i.kv, &kv_values(pattern), &s.stream)?;
    let weights: Vec<u16> = (0..512)
        .map(|n| {
            if pattern == "weight_nan" && n == 511 {
                0x7fc1
            } else {
                0x3f00 + n as u16 % 128
            }
        })
        .collect();
    upload(
        &i.weight,
        &bytes(&weights, |v| v.to_ne_bytes().to_vec()),
        &s.stream,
    )?;
    let positions: Vec<i32> = (0..ROWS)
        .map(|r| match pattern {
            "negative" => -1,
            "stale" => 17,
            "future" => 200,
            _ => 135 + r as i32,
        })
        .collect();
    let mut table = rope::reference(
        RopeTable::Window,
        &(0..ROWS).map(|r| 135 + r).collect::<Vec<_>>(),
    )
    .unwrap();
    if pattern == "rope_nan" {
        table[63] = f32::NAN;
    }
    upload(
        &i.table,
        &bytes(&table, |v| v.to_ne_bytes().to_vec()),
        &s.stream,
    )?;
    let last: Vec<i32> = positions
        .iter()
        .map(|p| match pattern {
            "stale" => p + SLOTS as i32,
            "future" => p - 1,
            _ => *p,
        })
        .collect();
    let batches: Vec<i32> = (0..ROWS)
        .map(|r| {
            if pattern == "bad_sequence" {
                ROWS as i32
            } else {
                r as i32
            }
        })
        .collect();
    for (buffer, values) in [
        (&i.positions, positions),
        (&i.batches, batches),
        (&i.last, last),
    ] {
        upload(
            buffer,
            &bytes(&values, |v| v.to_ne_bytes().to_vec()),
            &s.stream,
        )?;
    }
    Ok(())
}

struct Loaded {
    functions: Vec<[Function; 3]>,
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
    let functions = pairs()
        .iter()
        .map(|p| {
            let [a, b, c] = p.symbols();
            Ok([function(a)?, function(b)?, function(c)?])
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
/// Every pair at every pattern, eager and two graph replays, at full, zero
/// and excess live counts; `memcheck` runs a finite case and one with a
/// non-finite query and KV per pair, eagerly, without timing.
fn matrix(s: &Session, memcheck: bool) -> LabResult<serde_json::Value> {
    let k = load(s)?;
    let (mut bitwise, mut negative) = (0, 0);
    let mut timings = Vec::new();
    for (pair, functions) in pairs().iter().zip(&k.functions) {
        let inputs = Inputs {
            kv: s.allocate(ROWS as usize * 512 * 4)?,
            weight: s.allocate(512 * 2)?,
            table: s.allocate(ROWS as usize * 64 * 4)?,
            live: s.allocate(4)?,
            positions: s.allocate(ROWS as usize * 4)?,
            batches: s.allocate(ROWS as usize * 4)?,
            last: s.allocate(ROWS as usize * 4)?,
        };
        let outputs = [Output::new(s, pair)?, Output::new(s, pair)?];
        let capture = |arm: usize, steps: usize| {
            s.capture(|stream| {
                (0..steps)
                    .try_for_each(|_| step(pair, functions, arm, &inputs, &outputs[arm], stream))
            })
        };
        let graphs = [capture(0, 1)?, capture(1, 1)?];
        let chains = [capture(0, CHAIN)?, capture(1, CHAIN)?];
        let patterns: &[&str] = if memcheck {
            &["finite", "query_and_kv_nan"]
        } else {
            &PATTERNS
        };
        for &pattern in patterns {
            set_inputs(s, pattern, &inputs)?;
            let query = query_values(pattern);
            let lives: &[u32] = if memcheck { &[ROWS] } else { &LIVES };
            for &live_rows in lives {
                upload(&inputs.live, &live_rows.to_ne_bytes(), &s.stream)?;
                let mut eager: Option<Vec<Vec<u8>>> = None;
                for graph in if memcheck {
                    vec![false]
                } else {
                    vec![false, true, true]
                } {
                    for output in &outputs {
                        output.reset(s, &query, pattern == "sticky")?;
                    }
                    for arm in 0..2 {
                        if graph {
                            graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                        } else {
                            step(pair, functions, arm, &inputs, &outputs[arm], &s.stream)?;
                        }
                    }
                    s.stream.synchronize().map_err(|e| e.to_string())?;
                    let old = outputs[0].read(s)?;
                    let new = outputs[1].read(s)?;
                    compare(&old, &new).map_err(|e| {
                        format!(
                            "{} live={live_rows} {pattern} graph={graph}: {e}",
                            pair.twin.symbol
                        )
                    })?;
                    if let Some(ref previous) = eager {
                        compare(previous, &new)?;
                    } else {
                        eager = Some(old);
                    }
                    let status = &new[7];
                    if (0..OFFSET)
                        .chain(OFFSET + ROWS as usize..OFFSET + ROWS as usize + 2)
                        .any(|r| status[r * 4..r * 4 + 4] != 0x1357_2468u32.to_ne_bytes())
                    {
                        return Err(format!("{}: status guard changed", pair.twin.symbol));
                    }
                    let row =
                        u32::from_ne_bytes(status[OFFSET * 4..OFFSET * 4 + 4].try_into().unwrap());
                    let flagged = pattern == "sticky"
                        || pattern.contains("_nan")
                        || pattern.ends_with("_inf")
                        || pattern.ends_with("_inf_head");
                    if live_rows > 0 && (pattern == "finite" || flagged) && (row != 0) != flagged {
                        return Err(format!(
                            "{} {pattern}: row status {row:#x} against the pattern",
                            pair.twin.symbol
                        ));
                    }
                    if live_rows > 0 && pattern == "finite" && new[5].iter().all(|b| *b == 0xa5) {
                        return Err("finite row stored no window cell".into());
                    }
                    if live_rows == 0 && new[1..7].iter().any(|r| r.iter().any(|b| *b != 0xa5)) {
                        return Err("an idle row wrote an output".into());
                    }
                }
            }
            bitwise += 1;
            eprintln!(
                "QUERY_WINDOW_BITWISE {}",
                json!({"symbol":pair.twin.symbol,"pattern":pattern})
            );
        }
        if memcheck {
            continue;
        }
        set_inputs(s, "finite", &inputs)?;
        let query = query_values("finite");
        upload(&inputs.live, &ROWS.to_ne_bytes(), &s.stream)?;
        for output in &outputs {
            output.reset(s, &query, false)?;
        }
        for graph in &graphs {
            graph.launch(&s.stream).map_err(|e| e.to_string())?;
        }
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let expected = outputs[0].read(s)?;
        compare(&expected, &outputs[1].read(s)?)?;
        for role in 0..ROLES.len() {
            let mut bad = expected[role].clone();
            let at = if role == 7 { OFFSET * 4 } else { 0 };
            bad[at] ^= 1;
            upload(&outputs[1].buffers[role], &bad, &s.stream)?;
            if compare(&expected, &outputs[1].read(s)?).is_ok() {
                return Err("negative control escaped the comparison".into());
            }
            upload(&outputs[1].buffers[role], &expected[role], &s.stream)?;
            negative += 1;
        }
        // The timed graphs rotate the query again on every replay; both arms
        // start from the same values and must end equal.
        for output in &outputs {
            output.reset(s, &query, false)?;
        }
        let label = format!("query-window:{}", pair.twin.symbol);
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
        s.stream.synchronize().map_err(|e| e.to_string())?;
        compare(&outputs[0].read(s)?, &outputs[1].read(s)?)
            .map_err(|e| format!("{} after timing: {e}", pair.twin.symbol))?;
        let per = |ms: f32| ms * 1000. / CHAIN as f32;
        let record = json!({"symbol":pair.twin.symbol,"rope":pair.rope.symbol,"window":pair.window.symbol,
            "old_us":single.arms[0].median_ms * 1000.,"new_us":single.arms[1].median_ms * 1000.,
            "ratio":single.paired_ratio,"chain":CHAIN,"old_chain_us":per(chained.arms[0].median_ms),
            "new_chain_us":per(chained.arms[1].median_ms),"chain_ratio":chained.paired_ratio});
        eprintln!("QUERY_WINDOW_TIMING {record}");
        timings.push(record);
    }
    let c = coverage();
    let expected = |key: &str| c[key].as_u64().unwrap();
    if bitwise as u64
        != if memcheck {
            expected("memcheck_cases") * 2
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
        "QUERY_WINDOW_COMPLETE memcheck={memcheck} bitwise={bitwise} negative={negative} timings={}",
        timings.len()
    );
    Ok(json!({"memcheck":memcheck,"bitwise":bitwise,"negative":negative,"timings":timings}))
}

#[test]
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_QUERY_WINDOW"]
fn bitwise_graph_live_rows_and_timing_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| {
        let value = matrix(s, false)?;
        s.receipt(
            "query-window",
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
#[ignore = "requires CHOPINFER_KERNEL_GPU and CHOPINFER_QUERY_WINDOW"]
fn sanitized_twin_on_gpu() {
    let result = kernel_harness::gpu_test_with(probes(), 1 << 30, |s| matrix(s, true).map(|_| ()));
    assert!(result.as_ref().is_ok_and(Option::is_some), "{result:?}");
}
