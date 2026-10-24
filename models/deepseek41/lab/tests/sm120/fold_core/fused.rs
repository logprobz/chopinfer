//! Folds that admit rows by status (`fused_family`) against the served
//! launches they replace, with the bundle's selector, quantizers and cubins
//! (a cubin digest per served module is printed with every case).
//!
//! WO chain at 16 and 32 rows, every arm on its own buffers from the same
//! BF16 attention output (the decoded captured WO_a codes of each captured
//! layer, or synthetic) and WO_a/WO_b weights:
//! - served: the cached 8192-column quantizer, WO_a's admission and cluster
//!   fold, the 2048-column F32 quantizer, WO_b's admission and cluster fold;
//! - `fused`: the cached quantizer, WO_a quantizing WO_b's input in its
//!   merge, WO_b failing rows by status and WO_a's block flags;
//! - `sliced<S>w<W>`: `fused` after the sliced quantizer.
//!
//! Bit for bit against served: WO_a and WO_b outputs and every status word
//! (guards included) on every row, and both quantized inputs on every row
//! whose final status is zero (other rows' codes are dead). Rejected rows: a
//! NaN input element (row 0), an input row whose WO_a outputs overflow (row
//! 1), a preset status (row 2); their status bits must be the served ones
//! and must be set. Controls: WO_a and WO_b admitting every live row must
//! change the result.
//!
//! Status folds (WQ_b, WQ_a, WKV): served admission and fold against the
//! fold alone failing rows by status (WQ_b: the served row-local fold given
//! the row status as its failed flags, and the same fold on 48-column tiles
//! (`fused_family::WIDE`); WQ_a and WKV: `fused_family`). Rows rejected as a
//! producer quantizer leaves them (zero codes, scale 1 and a nonzero
//! status); the control admitting every row must differ.
//!
//! `CHOPINFER_FOLD_FUSED_MODE`: `check` (every row count up to the tile,
//! eager and graph, full, short with rejected rows, empty and full live
//! counts, then the captured inputs), `memcheck` and `racecheck` (the tile
//! only, eager), `time` (`check` at the tile, then graph replays of every
//! arm, warm and cold). `CHOPINFER_FOLD_FUSED_ROWS` keeps one row tile and
//! `CHOPINFER_FOLD_FUSED_CASES` (`wo` or `status`) one group.
use super::*;
use kernel_core::contract::{Argument, Refusal};
use kernels_sm120::{
    gemm::mxfp8_fold_cluster_fused::{Quantized, Quantizer},
    row::block_quantize,
};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Check,
    Memcheck,
    Racecheck,
    Time,
}

fn mode() -> LabResult<Mode> {
    Ok(
        match std::env::var("CHOPINFER_FOLD_FUSED_MODE")
            .map_err(|e| e.to_string())?
            .as_str()
        {
            "check" => Mode::Check,
            "memcheck" => Mode::Memcheck,
            "racecheck" => Mode::Racecheck,
            "time" => Mode::Time,
            _ => {
                return Err("CHOPINFER_FOLD_FUSED_MODE: check, memcheck, racecheck or time".into());
            }
        },
    )
}
fn tiles() -> Vec<u32> {
    let kept = std::env::var("CHOPINFER_FOLD_FUSED_ROWS")
        .ok()
        .map(|v| v.parse::<u32>().expect("CHOPINFER_FOLD_FUSED_ROWS"));
    if kept == Some(64) {
        return vec![64];
    }
    [16, 32]
        .into_iter()
        .filter(|r| kept.is_none_or(|k| k == *r))
        .collect()
}
fn group(name: &str) -> bool {
    std::env::var("CHOPINFER_FOLD_FUSED_CASES")
        .ok()
        .is_none_or(|g| g == name)
}
const WIDE_ROWS: [u32; 6] = [33, 40, 48, 56, 63, 64];
fn row_counts(tile: u32, mode: Mode) -> Vec<u32> {
    if tile == 64 {
        return if matches!(mode, Mode::Memcheck | Mode::Racecheck) {
            vec![33, 64]
        } else {
            WIDE_ROWS.to_vec()
        };
    }
    if mode == Mode::Check {
        (1..=tile).collect()
    } else {
        vec![tile]
    }
}
fn base_of(n: u32, k: u32, groups: u32) -> mxfp8_fold::Entry {
    family::bases()
        .into_iter()
        .find(|b| (b.query.n, b.query.k, b.query.groups) == (n, k, groups))
        .expect("served base of a fused shape")
}
fn served_quantizer(parallel: &str) -> block_quantize::Entry {
    let entry = sm120::ceil_parallel::quantizers(&FLASH, 4)
        .unwrap()
        .into_iter()
        .find(|e| e.symbol == parallel)
        .expect("a served dense quantizer");
    // The bundle registers the cached twin where it has one.
    sm120::ceil_cached::cached(&entry).unwrap_or(entry)
}
fn quantize_launch(
    e: &block_quantize::Entry,
    input: View,
    codes: View,
    scales: View,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Launch> {
    Ok(Launch {
        symbol: e.symbol.clone(),
        plan: e.plan(rows).map_err(|e| e.to_string())?,
        arguments: e
            .arguments(
                input.handle,
                codes.handle,
                Some(scales.handle),
                b.live_rows.handle,
                b.status.handle,
                b.status_row_offset,
            )
            .map_err(|e| e.to_string())?,
        rows,
        collective: false,
        persistent: false,
    })
}

/// Functions of `calls`: a bundle cubin's where the bundle holds the
/// symbol, otherwise the fused family's.
pub(super) fn load(s: &Session, calls: &[Launch]) -> LabResult<Functions> {
    struct Cache(std::cell::RefCell<BTreeMap<String, Module>>);
    let cache = s.state(|| Ok(Cache(std::cell::RefCell::new(BTreeMap::new()))))?;
    let mut loaded = cache.0.borrow_mut();
    let mut map = BTreeMap::new();
    for call in calls {
        if map.contains_key(&call.symbol) {
            continue;
        }
        let (name, bytes) = match fold_cubin_for_symbol(&call.symbol) {
            Some((name, bytes)) => (name.to_owned(), bytes.to_vec()),
            None if call.symbol.starts_with("chop_fold_fused_") => {
                let name = fused_family::CUBIN.to_owned();
                let bytes = std::fs::read(root().join("cubins").join(format!("{name}.cubin")))
                    .map_err(|e| format!("{name}: {e}"))?;
                (name, bytes)
            }
            None => return Err(format!("no cubin holds {}", call.symbol)),
        };
        if !loaded.contains_key(&name) {
            loaded.insert(name.clone(), s.load_cubin(&bytes)?);
        }
        let f = loaded[&name]
            .function(&call.symbol)
            .map_err(|e| e.to_string())?;
        f.set_max_dynamic_shared_memory(call.plan.dynamic_shared_bytes)
            .map_err(|e| e.to_string())?;
        map.insert(call.symbol.clone(), f);
    }
    Ok(Functions {
        map,
        _modules: Vec::new(),
    })
}
pub(super) fn eager(calls: &[Launch]) -> Vec<Launch> {
    let mut v = calls.to_vec();
    for c in &mut v {
        c.plan.pdl = false;
    }
    v
}
pub(super) fn served_modules(calls: &[Launch]) -> Value {
    let mut map = serde_json::Map::new();
    for c in calls {
        if let Some((name, bytes)) = fold_cubin_for_symbol(&c.symbol) {
            map.insert(name.into(), json!(kernel_build::digest(bytes)));
        }
    }
    Value::Object(map)
}
fn symbols(calls: &[Launch]) -> Vec<&str> {
    calls.iter().map(|c| c.symbol.as_str()).collect()
}
fn guarded_status(rows: u32, offset: u32, preset: &[(u32, u32)]) -> Vec<u8> {
    (0..rows + offset + 2)
        .flat_map(|i| {
            let v = if i < offset || i >= rows + offset {
                0xaabbccdd_u32
            } else {
                preset
                    .iter()
                    .find(|(r, _)| *r + offset == i)
                    .map_or(0, |(_, v)| *v)
            };
            v.to_le_bytes()
        })
        .collect()
}
fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

// Host E4M3 and BF16, for inputs only.
fn e4m3(code: u8) -> f32 {
    let sign = if code & 128 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((code >> 3) & 15);
    let mantissa = f32::from(code & 7);
    sign * if exponent == 0 {
        mantissa / 8.0 * 2f32.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}
fn bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// The BF16 values the captured or synthetic WO_a codes stand for, which
/// admission guarantees are exact.
fn attention_rows(input: &data::Input) -> Vec<u16> {
    input
        .codes
        .iter()
        .enumerate()
        .map(|(i, &c)| {
            let scale = 2f32.powi(i32::from(input.scales[i / 32]) - 127);
            bf16(e4m3(c) * scale)
        })
        .collect()
}

struct WoArm {
    label: String,
    a: Buffers<View>,
    b: Buffers<View>,
    flags: Option<View>,
    calls: Vec<Launch>,
    /// The fused arms' WO_a and WO_b launches with their lab controls.
    controls: Option<[Vec<Launch>; 2]>,
}

struct Wo {
    rows: u32,
    input: View,
    arms: Vec<WoArm>,
}

const ATTENTION: u32 = 8192;
const OUTPUT_A: u32 = 2048;

fn wo_buffers(
    a: &mut Arena<'_>,
    oa: &mxfp8_fold::Entry,
    ob: &mxfp8_fold::Entry,
    weights: &[View; 4],
    rows: u32,
) -> LabResult<(Buffers<View>, Buffers<View>)> {
    let live = a.bytes(&rows.to_le_bytes())?;
    let status = a.filled((u64::from(rows) + 5) * 4, 0)?;
    let size_a = oa.bytes(rows);
    let size_b = ob.bytes(rows);
    let fold_a = Buffers {
        activations: a.filled(size_a.activations, 0xa5)?,
        activation_scales: a.filled(size_a.activation_scales, 0xa5)?,
        weights: weights[0],
        weight_scales: weights[1],
        output: a.filled(size_a.output + 64, 255)?,
        live_rows: live,
        status,
        status_row_offset: 3,
    };
    let fold_b = Buffers {
        activations: a.filled(size_b.activations, 0xa5)?,
        activation_scales: a.filled(size_b.activation_scales, 0xa5)?,
        weights: weights[2],
        weight_scales: weights[3],
        output: a.filled(size_b.output + 64, 255)?,
        live_rows: live,
        status,
        status_row_offset: 3,
    };
    Ok((fold_a, fold_b))
}

fn wide_reference(
    a: &mut Arena<'_>,
    base: &mxfp8_fold::Entry,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Vec<Launch>> {
    let packed = packed_views(a, base, b)?;
    let query = sm120::mxfp8_fold::routed_query(base.query, rows);
    let op = backend()
        .mxfp8_fold_gemm(&query, Need::rows(rows))
        .map_err(|e| e.to_string())?;
    let work = a.filled(op.info().workspace.max(4), 0)?;
    op.launches(&packed, work, Counts::rows(rows))
        .map_err(|e| e.to_string())
}

fn wo_build(
    a: &mut Arena<'_>,
    tile: u32,
    rows: u32,
    weights: [&[u8]; 4],
    input: &[u16],
) -> LabResult<Wo> {
    let oa = base_of(1024, 4096, 2);
    let ob = base_of(5120, 2048, 1);
    let weights = if a.session.is_some() {
        [
            a.bytes(weights[0])?,
            a.bytes(weights[1])?,
            a.bytes(weights[2])?,
            a.bytes(weights[3])?,
        ]
    } else {
        let (sa, sb) = (oa.bytes(1), ob.bytes(1));
        [
            a.filled(sa.weights, 0)?,
            a.filled(sa.weight_scales, 0)?,
            a.filled(sb.weights, 0)?,
            a.filled(sb.weight_scales, 0)?,
        ]
    };
    let input_bytes: Vec<u8> = input.iter().flat_map(|v| v.to_le_bytes()).collect();
    let input = if a.session.is_some() {
        a.bytes(&input_bytes)?
    } else {
        a.filled(u64::from(rows) * u64::from(ATTENTION) * 2, 0)?
    };
    let q8 = served_quantizer("chop_deepseek41_dense_ceil_w8192_parallel_v1");
    let q2 = served_quantizer("chop_deepseek41_dense_ceil_f32_w2048_parallel_v1");
    let served_a = sm120::fold_cluster::served("attention.o-a", tile);
    let served_b = sm120::fold_cluster::served("attention.o-b", tile);
    let fused_a = if tile == 64 {
        sm120::fold_verify::entry("attention.o-a").unwrap()
    } else {
        fused_family::fused("attention.o-a", tile)
    };
    let fused_b = if tile == 64 {
        sm120::fold_verify::entry("attention.o-b").unwrap()
    } else {
        fused_family::fused("attention.o-b", tile)
    };
    let err = |e: Refusal| e.to_string();
    let mut arms = Vec::new();
    {
        let (fa, fb) = wo_buffers(a, &oa, &ob, &weights, rows)?;
        let old_a = if tile == 64 {
            wide_reference(a, &oa, &fa, rows)?
        } else {
            let entry = served_a.as_ref().ok_or("served o-a")?;
            let work = a.filled(entry.workspace(), 0)?;
            entry.launches(&fa, work, rows).map_err(err)?
        };
        let old_b = if tile == 64 {
            wide_reference(a, &ob, &fb, rows)?
        } else {
            let entry = served_b.as_ref().ok_or("served o-b")?;
            let work = a.filled(entry.workspace(), 0)?;
            entry.launches(&fb, work, rows).map_err(err)?
        };
        let mut calls = vec![quantize_launch(
            &q8,
            input,
            fa.activations,
            fa.activation_scales,
            &fa,
            rows,
        )?];
        calls.extend(old_a);
        calls.push(quantize_launch(
            &q2,
            fa.output,
            fb.activations,
            fb.activation_scales,
            &fb,
            rows,
        )?);
        calls.extend(old_b);
        arms.push(WoArm {
            label: "served".into(),
            a: fa,
            b: fb,
            flags: None,
            calls,
            controls: None,
        });
    }
    let first: Vec<Option<Quantizer>> = if tile == 64 {
        let mut sliced = sm120::fold_fused::sliced(32).unwrap();
        sliced.capacity = 64;
        vec![None, Some(sliced)]
    } else {
        std::iter::once(None)
            .chain(fused_family::quantizers().into_iter().map(Some))
            .collect()
    };
    for q in first {
        let (mut fa, mut fb) = wo_buffers(a, &oa, &ob, &weights, rows)?;
        let raw_weights = [fa.weights, fb.weights];
        if tile == 64 {
            fa = packed_views(a, &oa, &fa)?;
            fb = packed_views(a, &ob, &fb)?;
        }
        let flags = a.filled(u64::from(rows) * u64::from(OUTPUT_A / 32), 0xa5)?;
        let work_a = a.filled(fused_a.workspace(), 0)?;
        let work_b = a.filled(fused_b.workspace(), 0)?;
        let quantize = match &q {
            None => quantize_launch(&q8, input, fa.activations, fa.activation_scales, &fa, rows)?,
            Some(q) => q
                .launch(
                    input,
                    fa.activations,
                    fa.activation_scales,
                    fa.live_rows,
                    fa.status,
                    fa.status_row_offset,
                    rows,
                )
                .map_err(err)?,
        };
        let out = Quantized {
            codes: fb.activations,
            scales: fb.activation_scales,
            flags,
        };
        let mut launch_a = fused_a
            .launch(&fa, Some(out), None, work_a, rows)
            .map_err(err)?;
        let mut launch_b = fused_b
            .launch(&fb, None, Some(flags), work_b, rows)
            .map_err(err)?;
        if tile == 64 {
            launch_a.symbol = fused_a.packed_spec().map_err(err)?.symbol;
            launch_b.symbol = fused_b.packed_spec().map_err(err)?.symbol;
        }
        let ignore = |l: &Launch,
                      entry: &kernels_sm120::gemm::mxfp8_fold_cluster_fused::Entry,
                      raw: View| {
            let mut l = l.clone();
            l.symbol = entry.control_spec().symbol;
            // The negative control reads the same weights in row-major form.
            l.arguments.0[2] = Argument::DeviceBuffer(raw.handle);
            l
        };
        let controls = [
            vec![
                quantize.clone(),
                ignore(&launch_a, &fused_a, raw_weights[0]),
                launch_b.clone(),
            ],
            vec![
                quantize.clone(),
                launch_a.clone(),
                ignore(&launch_b, &fused_b, raw_weights[1]),
            ],
        ];
        arms.push(WoArm {
            label: q.as_ref().map_or("fused".into(), fused_family::label),
            a: fa,
            b: fb,
            flags: Some(flags),
            calls: vec![quantize, launch_a, launch_b],
            controls: Some(controls),
        });
    }
    Ok(Wo { rows, input, arms })
}

/// A row of one power of two signed as WO_a's output column 0 (group 0)
/// weights: that column overflows to infinity while columns whose weights
/// are not aligned with it stay finite.
fn overflow_row(weights: &[u8], scales: &[u8]) -> Vec<u16> {
    let sum: f64 = (0..4096)
        .map(|k| f64::from(e4m3(weights[k]).abs()) * 2f64.powi(i32::from(scales[k / 32]) - 127))
        .sum();
    // Twice the F32 maximum over the column's absolute weight sum.
    let power = (6.9e38 / sum.max(1e-30)).log2().ceil().clamp(-126.0, 127.0) as i32;
    let value = bf16(2f32.powi(power));
    (0..ATTENTION as usize)
        .map(|k| {
            if k < 4096 && weights[k] & 128 != 0 {
                value | 0x8000
            } else {
                value
            }
        })
        .collect()
}

fn wo_reset(
    a: &Arena<'_>,
    wo: &Wo,
    input: &[u16],
    weights_a: [&[u8]; 2],
    live: u32,
    bad: bool,
) -> LabResult<()> {
    let rows = wo.rows;
    let mut values = input.to_vec();
    let width = ATTENTION as usize;
    if bad && live >= 1 {
        values[5] = 0x7fc0;
    }
    if bad && live >= 2 {
        values[width..2 * width].copy_from_slice(&overflow_row(weights_a[0], weights_a[1]));
    }
    if bad && live > 32 {
        values[(live as usize - 1) * width + 5] = 0x7fc0;
    }
    if bad && live > 33 {
        values[(live as usize - 2) * width..(live as usize - 1) * width]
            .copy_from_slice(&overflow_row(weights_a[0], weights_a[1]));
    }
    let mut preset: Vec<(u32, u32)> = if bad && live >= 3 {
        vec![(2, 0x80)]
    } else {
        vec![]
    };
    if bad && live > 34 {
        preset.push((live - 3, 0x80));
    }
    a.write(
        wo.input,
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    for arm in &wo.arms {
        a.write(arm.a.live_rows, &live.to_le_bytes())?;
        a.write(arm.a.status, &guarded_status(rows, 3, &preset))?;
        for v in [
            arm.a.activations,
            arm.a.activation_scales,
            arm.b.activations,
            arm.b.activation_scales,
        ]
        .into_iter()
        .chain(arm.flags)
        {
            a.fill(v, 0xa5)?;
        }
        a.fill(arm.a.output, 255)?;
        a.fill(arm.b.output, 255)?;
    }
    Ok(())
}

/// Every arm against the served one; the served arm's rejected rows must
/// carry their status.
fn wo_compare(a: &Arena<'_>, wo: &Wo, live: u32, bad: bool, arms: &[usize]) -> LabResult<()> {
    let served = &wo.arms[0];
    let status = words(&a.read(served.a.status)?);
    let row_status = |r: u32| status[(r + 3) as usize];
    if bad {
        for r in (0..live.min(3)).chain((32..live).rev().take(3)) {
            if row_status(r) == 0 {
                return Err(format!("rejected row {r} kept a zero status"));
            }
        }
    }
    let out_b = a.read(served.b.output)?;
    let width_b = 5120 * 4;
    for r in 0..live {
        let row = &out_b[r as usize * width_b..(r as usize + 1) * width_b];
        let finite = row
            .chunks_exact(4)
            .all(|v| f32::from_le_bytes(v.try_into().unwrap()).is_finite());
        if row_status(r) != 0 && row.iter().any(|&v| v != 0) {
            return Err(format!(
                "served row {r} with status {:#x} not cleared",
                row_status(r)
            ));
        }
        if row_status(r) == 0 && !finite {
            return Err(format!("served live row {r} not finite"));
        }
    }
    if out_b[live as usize * width_b..].iter().any(|&v| v != 255) {
        return Err("served inactive output or guard overwritten".into());
    }
    let dead = |r: usize| (r as u32) < live && row_status(r as u32) != 0;
    for &i in arms {
        let arm = &wo.arms[i];
        let label = &arm.label;
        if a.read(arm.a.status)? != a.read(served.a.status)? {
            return Err(format!(
                "{label}: status differs: {:x?} vs {:x?}",
                words(&a.read(arm.a.status)?),
                status
            ));
        }
        for (name, x, y) in [
            ("WO_a output", served.a.output, arm.a.output),
            ("WO_b output", served.b.output, arm.b.output),
        ] {
            let (x, y) = (a.read(x)?, a.read(y)?);
            if x != y {
                return Err(format!(
                    "{label}: {name} differs at byte {:?}",
                    x.iter().zip(&y).position(|(p, q)| p != q)
                ));
            }
        }
        for (name, x, y, width) in [
            (
                "WO_a codes",
                served.a.activations,
                arm.a.activations,
                ATTENTION as usize,
            ),
            (
                "WO_a scales",
                served.a.activation_scales,
                arm.a.activation_scales,
                ATTENTION as usize / 32,
            ),
            (
                "WO_b codes",
                served.b.activations,
                arm.b.activations,
                OUTPUT_A as usize,
            ),
            (
                "WO_b scales",
                served.b.activation_scales,
                arm.b.activation_scales,
                OUTPUT_A as usize / 32,
            ),
        ] {
            let (x, y) = (a.read(x)?, a.read(y)?);
            for (r, (p, q)) in x.chunks(width).zip(y.chunks(width)).enumerate() {
                if !dead(r) && p != q {
                    return Err(format!("{label}: {name} of row {r} differ"));
                }
            }
        }
    }
    Ok(())
}

fn wo_case(
    s: &Session,
    tile: u32,
    rows: u32,
    source: &str,
    weights: [&[u8]; 4],
    input: &[u16],
    mode: Mode,
) -> LabResult<Value> {
    let mut a = Arena::new(Some(s));
    let wo = wo_build(&mut a, tile, rows, weights, input)?;
    let all: Vec<Launch> = wo
        .arms
        .iter()
        .flat_map(|arm| {
            arm.calls
                .iter()
                .chain(arm.controls.iter().flatten().flatten())
                .cloned()
        })
        .collect();
    let f = load(s, &all)?;
    let mut record = json!({"case":"wo","tile":tile,"rows":rows,"source":source,
        "served":symbols(&wo.arms[0].calls),"served_modules":served_modules(&wo.arms[0].calls),
        "arms":wo.arms.iter().map(|arm| json!({"label":arm.label,"calls":arm.calls.iter().map(|c| json!({"symbol":c.symbol,"grid":c.plan.grid,"block":c.plan.block,"cluster":c.plan.cluster})).collect::<Vec<_>>()})).collect::<Vec<_>>()});
    let graphs =
        if mode == Mode::Check || mode == Mode::Time || tile == 64 && mode == Mode::Memcheck {
            Some(
                wo.arms
                    .iter()
                    .map(|arm| s.capture(|stream| issue(s, &a, &f, &arm.calls, stream)))
                    .collect::<LabResult<Vec<_>>>()?,
            )
        } else {
            None
        };
    let configurations: Vec<(u32, bool)> = match mode {
        Mode::Racecheck | Mode::Memcheck => vec![(rows, true), (rows.saturating_sub(1), false)],
        _ => vec![
            (rows, false),
            (rows.saturating_sub(1), true),
            (0, false),
            (rows, true),
            (rows, false),
        ],
    };
    let every: Vec<usize> = (1..wo.arms.len()).collect();
    let mut comparisons = 0;
    for &(live, bad) in &configurations {
        for graph in [false, true] {
            if graph && graphs.is_none() {
                continue;
            }
            wo_reset(&a, &wo, input, [weights[0], weights[1]], live, bad)?;
            for (i, arm) in wo.arms.iter().enumerate() {
                match (&graphs, graph) {
                    (Some(g), true) => g[i].launch(&s.stream).map_err(|e| e.to_string())?,
                    _ => issue(s, &a, &f, &eager(&arm.calls), &s.stream)?,
                }
            }
            s.stream.synchronize().map_err(|e| e.to_string())?;
            wo_compare(&a, &wo, live, bad, &every).map_err(|e| {
                format!(
                    "wo tile {tile} rows {rows} live {live} bad {bad} graph {graph} {source}: {e}"
                )
            })?;
            comparisons += 1;
        }
    }
    record["comparisons"] = json!(comparisons);
    record["bitwise"] = json!(true);
    if mode == Mode::Racecheck || mode == Mode::Memcheck || rows < 3 {
        return Ok(record);
    }
    // Admitting every live row must change WO_a's output (the NaN and preset
    // rows) and WO_b's (the overflow row). The sliced arm runs the controls:
    // the cached quantizer zeros its rejected rows' codes, so admitting them
    // there leaves the outputs zero.
    for (which, buffer) in [(0, "WO_a output"), (1, "WO_b output")] {
        wo_reset(&a, &wo, input, [weights[0], weights[1]], rows, true)?;
        let arm = &wo.arms[2];
        issue(s, &a, &f, &eager(&wo.arms[0].calls), &s.stream)?;
        issue(
            s,
            &a,
            &f,
            &eager(&arm.controls.as_ref().unwrap()[which]),
            &s.stream,
        )?;
        let (x, y) = if which == 0 {
            (wo.arms[0].a.output, arm.a.output)
        } else {
            (wo.arms[0].b.output, arm.b.output)
        };
        if a.read(x)? == a.read(y)? {
            return Err(format!(
                "wo tile {tile} rows {rows}: the {buffer} control was ineffective"
            ));
        }
    }
    record["negative"] = json!(true);
    if mode == Mode::Time
        && let Some(graphs) = &graphs
    {
        wo_reset(&a, &wo, input, [weights[0], weights[1]], rows, false)?;
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::arms(s, cold, graphs.len(), |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        }
        record["labels"] = json!(wo.arms.iter().map(|arm| &arm.label).collect::<Vec<_>>());
    }
    Ok(record)
}

/// A WO chain input: source, WO_a and WO_b weights and scales, BF16 rows.
type WoInput = (String, [Vec<u8>; 4], Vec<u16>);

/// Synthetic weights and rows, then with `captured` the WO_a and WO_b
/// weights of each captured layer and its decoded attention output rows
/// (repeated to `rows`).
fn wo_inputs(rows: u32, captured: bool) -> LabResult<Vec<WoInput>> {
    let oa = base_of(1024, 4096, 2);
    let ob = base_of(5120, 2048, 1);
    let mut result = Vec::new();
    let a = data::synthetic(&oa, rows);
    let b = data::synthetic(&ob, rows);
    result.push((
        "synthetic".into(),
        [
            a.weights.clone(),
            a.weight_scales.clone(),
            b.weights,
            b.weight_scales,
        ],
        attention_rows(&a),
    ));
    if captured {
        let outputs_b = data::captured(&ob, true)?;
        for input in data::captured(&oa, true)? {
            if input.rows == 1 || rows > 32 && input.rows != 32 {
                continue;
            }
            let layer = input
                .id
                .strip_suffix("-o-a-prefix")
                .or_else(|| input.id.strip_suffix("-o-a-repeated"));
            let Some(layer) = layer else { continue };
            let Some(b) = outputs_b
                .iter()
                .find(|i| i.id.starts_with(&format!("{layer}-o-b")))
            else {
                continue;
            };
            let source = attention_rows(&input);
            let width = ATTENTION as usize;
            let have = source.len() / width;
            let values = (0..rows as usize)
                .flat_map(|r| {
                    source[(r % have) * width..(r % have + 1) * width]
                        .iter()
                        .copied()
                })
                .collect();
            result.push((
                format!("{layer}-x{rows}"),
                [
                    input.weights.clone(),
                    input.weight_scales.clone(),
                    b.weights.clone(),
                    b.weight_scales.clone(),
                ],
                values,
            ));
        }
    }
    Ok(result)
}

/// A status fold case: served admission and fold, then each candidate's
/// fold alone, every arm on its own buffers.
struct StatusCase {
    id: &'static str,
    served: Vec<Launch>,
    candidates: Vec<(String, Vec<Launch>)>,
    /// The first candidate admitting every live row.
    control: Vec<Launch>,
    b: Vec<Buffers<View>>,
}

fn status_views(a: &mut Arena<'_>, input: &data::Input, rows: u32) -> LabResult<Buffers<View>> {
    let size = base_of(input.n, input.k, input.groups).bytes(rows);
    let session = a.session.is_some();
    Ok(Buffers {
        activations: if session {
            a.bytes(&input.codes)?
        } else {
            a.filled(size.activations, 0)?
        },
        activation_scales: if session {
            a.bytes(&input.scales)?
        } else {
            a.filled(size.activation_scales, 0)?
        },
        weights: if session {
            a.bytes(&input.weights)?
        } else {
            a.filled(size.weights, 0)?
        },
        weight_scales: if session {
            a.bytes(&input.weight_scales)?
        } else {
            a.filled(size.weight_scales, 0)?
        },
        output: a.filled(size.output + 64, 255)?,
        live_rows: a.bytes(&rows.to_le_bytes())?,
        // The row-local folds read the status from its start.
        status: a.filled((u64::from(rows) + 2) * 4, 0)?,
        status_row_offset: 0,
    })
}

/// `run` reading its failed flags from `flags` instead of its workspace.
fn reading(run: &Launch, work: View, flags: View) -> Launch {
    let mut run = run.clone();
    for arg in &mut run.arguments.0 {
        if *arg == Argument::DeviceBuffer(work.handle) {
            *arg = Argument::DeviceBuffer(flags.handle);
        }
    }
    run
}

fn status_build(
    a: &mut Arena<'_>,
    id: &'static str,
    tile: u32,
    rows: u32,
    input: &data::Input,
) -> LabResult<StatusCase> {
    let err = |e: Refusal| e.to_string();
    let arms = if id == "attention.q-b" && tile != 64 {
        2 + fused_family::WIDE.len()
    } else {
        2
    };
    let b = (0..arms)
        .map(|_| status_views(a, input, rows))
        .collect::<LabResult<Vec<_>>>()?;
    let (served, candidates, control) = if tile == 64 {
        let base = base_of(input.n, input.k, input.groups);
        let served = wide_reference(a, &base, &b[0], rows)?;
        if id == "attention.q-b" {
            let candidate = sm120::fold_verify_local::entry().map_err(err)?;
            let packed = packed_views(a, &base, &b[1])?;
            let mut run = candidate.launch(&packed, rows).map_err(err)?;
            run.symbol = candidate.packed_symbol();
            let zeros = a.filled(64 * 4, 0)?;
            let mut control = run.clone();
            control.arguments.0[8] = Argument::DeviceBuffer(zeros.handle);
            return Ok(StatusCase {
                id,
                served,
                candidates: vec![("tiled32".into(), vec![run])],
                control: vec![control],
                b,
            });
        }
        let f = sm120::fold_verify::entry(id).ok_or("wide status candidate absent")?;
        let work = a.filled(f.workspace(), 0)?;
        let packed = packed_views(a, &base, &b[1])?;
        let mut run = f.launch(&packed, None, None, work, rows).map_err(err)?;
        run.symbol = f.packed_spec().map_err(err)?.symbol;
        let mut control = f.launch(&b[1], None, None, work, rows).map_err(err)?;
        control.symbol = f.control_spec().symbol;
        (served, vec![("status".into(), vec![run])], vec![control])
    } else if id == "attention.q-b" {
        let e = sm120::fold_local_rows::served(id, tile).ok_or("served q-b")?;
        let work = a.filled(e.workspace(), 0)?;
        let served = e.launches(&b[0], work, rows).map_err(err)?;
        let mut candidates = Vec::new();
        let mut control = Vec::new();
        for (arm, buffers) in b.iter().enumerate().skip(1) {
            let work = a.filled(e.workspace(), 0)?;
            let [_, run]: [Launch; 2] = e
                .launches(buffers, work, rows)
                .map_err(err)?
                .try_into()
                .map_err(|_| "a row-local fold is admission then fold")?;
            if arm == 1 {
                let zeros = a.filled(256, 0)?;
                control = vec![reading(&run, work, zeros)];
                candidates.push((
                    "status".to_string(),
                    vec![reading(&run, work, buffers.status)],
                ));
                continue;
            }
            let stages = fused_family::WIDE[arm - 2];
            let mut wide = reading(&run, work, buffers.status);
            wide.symbol = fused_family::wide_symbol(stages);
            wide.plan.grid[0] = input.n.div_ceil(fused_family::WIDE_TILE);
            wide.plan.block[0] = fused_family::WIDE_TILE * 4;
            candidates.push((fused_family::wide_label(stages), vec![wide]));
        }
        (served, candidates, control)
    } else {
        let e = sm120::fold_cluster::served(id, tile).ok_or("served cluster fold")?;
        let work = a.filled(e.workspace(), 0)?;
        let served = e.launches(&b[0], work, rows).map_err(err)?;
        let f = fused_family::fused(id, tile);
        let work = a.filled(f.workspace(), 0)?;
        let run = f.launch(&b[1], None, None, work, rows).map_err(err)?;
        let mut control = run.clone();
        control.symbol.push_str("_ignore_v1");
        (
            served,
            vec![("status".to_string(), vec![run])],
            vec![control],
        )
    };
    Ok(StatusCase {
        id,
        served,
        candidates,
        control,
        b,
    })
}

/// With `bad`, rows 0 and `live - 1` rejected: a status, and with `zeroed`
/// zero codes and scale 1 as a producer quantizer leaves them (otherwise
/// their codes stay, so admitting them changes their outputs).
fn status_reset(
    a: &Arena<'_>,
    c: &StatusCase,
    input: &data::Input,
    rows: u32,
    live: u32,
    bad: bool,
    zeroed: bool,
) -> LabResult<()> {
    let q = base_of(input.n, input.k, input.groups).query;
    let stride = q.activation_row_stride as usize;
    let rejected: Vec<(u32, u32)> = if bad && live > 1 {
        vec![(0, 1), (live - 1, 0x80)]
    } else {
        vec![]
    };
    let mut codes = input.codes.clone();
    let mut scales = input.scales.clone();
    for &(r, _) in rejected.iter().filter(|_| zeroed) {
        let r = r as usize;
        codes[r * stride..(r + 1) * stride].fill(0);
        scales[r * stride / 32..(r + 1) * stride / 32].fill(127);
    }
    let status: Vec<u8> = (0..rows + 2)
        .flat_map(|i| {
            let v = if i >= rows {
                0xaabbccdd_u32
            } else {
                rejected
                    .iter()
                    .find(|(r, _)| *r == i)
                    .map_or(0, |(_, v)| *v)
            };
            v.to_le_bytes()
        })
        .collect();
    for b in &c.b {
        a.write(b.activations, &codes)?;
        a.write(b.activation_scales, &scales)?;
        a.write(b.live_rows, &live.to_le_bytes())?;
        a.write(b.status, &status)?;
        a.fill(b.output, 255)?;
    }
    Ok(())
}

fn status_case(
    s: &Session,
    id: &'static str,
    tile: u32,
    input: &data::Input,
    mode: Mode,
) -> LabResult<Value> {
    let rows = input.rows;
    let mut a = Arena::new(Some(s));
    let c = status_build(&mut a, id, tile, rows, input)?;
    let arms: Vec<&Vec<Launch>> = std::iter::once(&c.served)
        .chain(c.candidates.iter().map(|(_, l)| l))
        .collect();
    let all: Vec<Launch> = arms
        .iter()
        .copied()
        .flatten()
        .chain(&c.control)
        .cloned()
        .collect();
    let f = load(s, &all)?;
    let mut record = json!({"case":"status","projection":id,"tile":tile,"rows":rows,"source":input.id,
        "served":symbols(&c.served),"served_modules":served_modules(&c.served),
        "candidates":c.candidates.iter().map(|(label, calls)| json!({"label":label,"calls":symbols(calls)})).collect::<Vec<_>>()});
    let graphs =
        if mode == Mode::Check || mode == Mode::Time || tile == 64 && mode == Mode::Memcheck {
            Some(
                arms.iter()
                    .map(|calls| s.capture(|stream| issue(s, &a, &f, calls, stream)))
                    .collect::<LabResult<Vec<_>>>()?,
            )
        } else {
            None
        };
    let configurations: Vec<(u32, bool)> = match mode {
        Mode::Racecheck | Mode::Memcheck => vec![(rows, true)],
        _ => vec![
            (rows, false),
            (rows.saturating_sub(1), true),
            (0, false),
            (rows, true),
            (rows, false),
        ],
    };
    let width = (input.n * input.groups * 4) as usize;
    let mut comparisons = 0;
    for &(live, bad) in &configurations {
        for graph in [false, true] {
            if graph && graphs.is_none() {
                continue;
            }
            status_reset(&a, &c, input, rows, live, bad, true)?;
            for (i, calls) in arms.iter().enumerate() {
                match (&graphs, graph) {
                    (Some(g), true) => g[i].launch(&s.stream).map_err(|e| e.to_string())?,
                    _ => issue(s, &a, &f, &eager(calls), &s.stream)?,
                }
            }
            let context = format!(
                "{id} tile {tile} rows {rows} live {live} bad {bad} graph {graph} {}",
                input.id
            );
            let out = a.read(c.b[0].output)?;
            let status = a.read(c.b[0].status)?;
            for (i, (label, _)) in c.candidates.iter().enumerate() {
                if a.read(c.b[i + 1].output)? != out || a.read(c.b[i + 1].status)? != status {
                    return Err(format!("{context}: {label} output or status differs"));
                }
            }
            if bad && live > 1 {
                for r in [0, live as usize - 1] {
                    if out[r * width..(r + 1) * width].iter().any(|&v| v != 0) {
                        return Err(format!("{context}: rejected row {r} not cleared"));
                    }
                }
            }
            if out[live as usize * width..].iter().any(|&v| v != 255) {
                return Err(format!("{context}: inactive output or guard overwritten"));
            }
            comparisons += 1;
        }
    }
    record["comparisons"] = json!(comparisons);
    record["bitwise"] = json!(true);
    if mode == Mode::Racecheck || mode == Mode::Memcheck || rows < 2 {
        return Ok(record);
    }
    // Rejected rows that keep their codes: every arm still clears them by
    // status, bit for bit, and admitting every row must not.
    status_reset(&a, &c, input, rows, rows, true, false)?;
    for calls in &arms {
        issue(s, &a, &f, &eager(calls), &s.stream)?;
    }
    for (i, (label, _)) in c.candidates.iter().enumerate() {
        if a.read(c.b[i + 1].output)? != a.read(c.b[0].output)?
            || a.read(c.b[i + 1].status)? != a.read(c.b[0].status)?
        {
            return Err(format!(
                "{id} tile {tile} rows {rows}: {label} differs on rejected rows with codes"
            ));
        }
    }
    status_reset(&a, &c, input, rows, rows, true, false)?;
    issue(s, &a, &f, &eager(&c.served), &s.stream)?;
    issue(s, &a, &f, &eager(&c.control), &s.stream)?;
    if a.read(c.b[0].output)? == a.read(c.b[1].output)? {
        return Err(format!(
            "{id} tile {tile} rows {rows}: the admit-all control was ineffective"
        ));
    }
    record["negative"] = json!(true);
    if mode == Mode::Time
        && let Some(graphs) = &graphs
    {
        status_reset(&a, &c, input, rows, rows, false, true)?;
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::arms(s, cold, graphs.len(), |arm, stream| {
                graphs[arm].launch(stream).map_err(|e| e.to_string())
            })?;
        }
    }
    Ok(record)
}

const STATUS: &[(&str, u32, u32, u32)] = &[
    ("attention.q-b", 8192, 1280, 1),
    ("attention.q-a", 1280, 5120, 1),
    ("attention.kv", 512, 5120, 1),
];

pub fn gpu(s: &Session) -> LabResult<()> {
    let mode = mode()?;
    let mut cases = 0;
    for tile in tiles() {
        if group("wo") {
            for rows in row_counts(tile, mode) {
                let captured =
                    matches!(mode, Mode::Check | Mode::Time) && (rows == tile || tile == 64);
                let inputs = wo_inputs(rows, captured)?;
                if tile == 64 && captured && inputs.len() != 3 {
                    return Err("wide WO capture coverage differs".into());
                }
                for (source, weights, input) in inputs {
                    let w = [
                        &weights[0][..],
                        &weights[1][..],
                        &weights[2][..],
                        &weights[3][..],
                    ];
                    let record = wo_case(s, tile, rows, &source, w, &input, mode)?;
                    println!("FOLD_FUSED_CASE {record}");
                    cases += 1;
                }
            }
        }
        if group("status") {
            for &(id, n, k, groups) in STATUS {
                if tile == 64 && id == "attention.q-b" {
                    continue;
                }
                let base = base_of(n, k, groups);
                let mut inputs: Vec<data::Input> = row_counts(tile, mode)
                    .into_iter()
                    .map(|r| data::synthetic(&base, r))
                    .collect();
                if tile == 64 && matches!(mode, Mode::Check | Mode::Time) {
                    for rows in WIDE_ROWS {
                        inputs.extend(data::captured_wide(&base, rows)?);
                    }
                } else if mode == Mode::Check {
                    inputs.extend(
                        data::captured(&base, true)?
                            .into_iter()
                            .filter(|i| i.rows <= tile),
                    );
                }
                for input in &inputs {
                    let record = status_case(s, id, tile, input, mode)?;
                    println!("FOLD_FUSED_CASE {record}");
                    cases += 1;
                }
            }
        }
    }
    if tiles() == [64] && group("wo") && group("status") {
        let expected = if matches!(mode, Mode::Memcheck | Mode::Racecheck) {
            6
        } else {
            42
        };
        if cases != expected {
            return Err(format!(
                "wide dense ran {cases} cases instead of {expected}"
            ));
        }
    }
    if cases == 0 {
        return Err("no fused case ran".into());
    }
    println!(
        "FOLD_FUSED_DONE {}",
        json!({"mode":std::env::var("CHOPINFER_FOLD_FUSED_MODE").unwrap_or_default(),"cases":cases})
    );
    Ok(())
}

/// Every case's served and candidate calls without opening CUDA: the plans
/// the GPU run must reproduce.
#[test]
fn fused_catalog() {
    let specs = fused_family::cubins();
    let lab: std::collections::BTreeSet<String> = specs
        .iter()
        .flat_map(|s| s.entries.iter().map(|e| e.symbol.clone()))
        .collect();
    assert_eq!(
        lab.len(),
        2 * fused_family::entries().len()
            + fused_family::quantizers().len()
            + fused_family::WIDE.len()
    );
    let mut cases = Vec::new();
    for tile in [16, 32] {
        for rows in 1..=tile {
            let mut a = Arena::new(None);
            let oa = base_of(1024, 4096, 2);
            let size = oa.bytes(rows);
            let weights: [Vec<u8>; 4] = Default::default();
            let w = [
                &weights[0][..],
                &weights[1][..],
                &weights[2][..],
                &weights[3][..],
            ];
            let input = vec![0u16; (size.activations) as usize];
            let wo = wo_build(&mut a, tile, rows, w, &input).unwrap();
            assert_eq!(wo.arms.len(), 2 + fused_family::SLICES.len());
            assert_eq!(wo.arms[0].calls.len(), 6);
            for arm in &wo.arms {
                for c in arm
                    .calls
                    .iter()
                    .chain(arm.controls.iter().flatten().flatten())
                {
                    c.plan.validate_geometry().unwrap();
                    assert!(
                        fold_cubin_for_symbol(&c.symbol).is_some() || lab.contains(&c.symbol),
                        "{}",
                        c.symbol
                    );
                }
            }
            for arm in &wo.arms[1..] {
                assert_eq!(arm.calls.len(), 3);
                assert_eq!(arm.calls[1].plan.cluster, wo.arms[0].calls[2].plan.cluster);
                assert_eq!(arm.calls[2].plan.grid, wo.arms[0].calls[5].plan.grid);
            }
            assert!(
                wo.arms[0]
                    .calls
                    .iter()
                    .all(|c| fold_cubin_for_symbol(&c.symbol).is_some())
            );
            cases.push(json!({"case":"wo","tile":tile,"rows":rows,"arms":wo.arms.iter().map(|arm| json!({"label":arm.label,"calls":symbols(&arm.calls)})).collect::<Vec<_>>(),"served_modules":served_modules(&wo.arms[0].calls)}));
            for &(id, n, k, groups) in STATUS {
                let input = data::empty(&base_of(n, k, groups), rows);
                let c = status_build(&mut a, id, tile, rows, &input).unwrap();
                assert_eq!(c.served.len(), 2);
                assert_eq!(c.candidates[0].1.len(), 1);
                assert_eq!(c.candidates[0].1[0].plan, c.served[1].plan);
                assert_eq!(c.b.len(), c.candidates.len() + 1);
                for (_, calls) in &c.candidates[1..] {
                    assert_eq!(calls[0].plan.grid, [171, 1, 1]);
                    assert_eq!(calls[0].plan.block, [192, 1, 1]);
                    assert_eq!(
                        calls[0].arguments.0.len(),
                        c.candidates[0].1[0].arguments.0.len()
                    );
                    assert!(lab.contains(&calls[0].symbol));
                }
                for call in c
                    .served
                    .iter()
                    .chain(c.candidates.iter().flat_map(|(_, l)| l))
                    .chain(&c.control)
                {
                    call.plan.validate_geometry().unwrap();
                    assert!(
                        fold_cubin_for_symbol(&call.symbol).is_some() || lab.contains(&call.symbol),
                        "{}",
                        call.symbol
                    );
                }
                cases.push(json!({"case":"status","projection":c.id,"tile":tile,"rows":rows,"served":symbols(&c.served),"served_modules":served_modules(&c.served),
                    "candidates":c.candidates.iter().map(|(label, calls)| json!({"label":label,"calls":symbols(calls)})).collect::<Vec<_>>()}));
            }
        }
    }
    println!(
        "FOLD_FUSED_CATALOG {}",
        json!({"family":fused_family::FAMILY,"cubin":fused_family::CUBIN,"symbols":lab,"cases":cases})
    );
}

#[test]
fn wide_fused_catalog_keeps_full_row_references_and_packed_candidates() {
    for rows in WIDE_ROWS {
        let mut a = Arena::new(None);
        let wo = wo_build(&mut a, 64, rows, [&[]; 4], &[]).unwrap();
        assert_eq!(wo.arms.len(), 3);
        assert_eq!(wo.arms[0].calls.len(), 6);
        assert!(
            wo.arms[0]
                .calls
                .iter()
                .all(|c| !c.symbol.contains("cluster") && !c.symbol.contains("m8192"))
        );
        for arm in &wo.arms[1..] {
            for (call, id) in arm.calls[1..]
                .iter()
                .zip(["attention.o-a", "attention.o-b"])
            {
                let entry = sm120::fold_verify::entry(id).unwrap();
                assert_eq!(call.symbol, entry.packed_spec().unwrap().symbol);
                assert_eq!(call.plan, entry.plan(rows).unwrap());
                assert!(fold_cubin_for_symbol(&call.symbol).is_some());
            }
        }
        for &(id, n, k, groups) in &STATUS[1..] {
            let input = data::empty(&base_of(n, k, groups), rows);
            let c = status_build(&mut a, id, 64, rows, &input).unwrap();
            assert!(c.served.len() >= 2);
            assert!(
                c.served
                    .iter()
                    .all(|c| !c.symbol.contains("cluster") && !c.symbol.contains("m8192"))
            );
            assert_eq!(c.candidates[0].1.len(), 1);
            assert_eq!(
                c.candidates[0].1[0].symbol,
                sm120::fold_verify::entry(id)
                    .unwrap()
                    .packed_spec()
                    .unwrap()
                    .symbol
            );
            assert!(fold_cubin_for_symbol(&c.control[0].symbol).is_some());
        }
    }
}

#[test]
fn host_bf16_and_e4m3_round_trip() {
    assert_eq!(e4m3(0x7e), 448.0);
    assert_eq!(e4m3(0x01), 2f32.powi(-9));
    assert_eq!(e4m3(0xb8), -1.0);
    assert_eq!(bf16(1.0), 0x3f80);
    assert_eq!(bf16(f32::from_bits(0x3f80_8000)), 0x3f80);
    assert_eq!(bf16(f32::from_bits(0x3f81_8000)), 0x3f82);
    // 4096 weights of -1: 2^118 is the first power of two over 6.9e38 / 4096.
    let row = overflow_row(&[0x80 | 0x38; 4096], &[127; 128]);
    assert_eq!(
        (row.len(), row[0], row[4095], row[4096]),
        (8192, 0xfa80, 0xfa80, 0x7a80)
    );
}

#[test]
#[ignore = "requires CHOPINFER_FOLD_CORE_ROOT, CHOPINFER_FOLD_FUSED_MODE and assigned idle GPUs"]
fn gpu_fused() {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    let bytes =
        BYTES.get_or_init(|| std::fs::read(root().join("cubins/probe_sm120.cubin")).unwrap());
    let cubins = kernel_harness::Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", bytes)]),
        &[(
            "probe_sm120",
            &["chop_probe_read_v1", "chop_probe_flush_v1"],
        )],
    );
    assert!(
        kernel_harness::gpu_test_with(cubins, 6 << 30, gpu)
            .unwrap()
            .is_some()
    );
}

#[test]
fn local64_catalog_keeps_the_full_row_reference() {
    for rows in WIDE_ROWS {
        let mut a = Arena::new(None);
        let input = data::empty(&base_of(8192, 1280, 1), rows);
        let case = status_build(&mut a, "attention.q-b", 64, rows, &input).unwrap();
        assert_eq!(case.b.len(), 2);
        assert!(case.served.len() >= 2);
        assert_eq!(case.candidates[0].1[0].plan.grid, [171, 2, 1]);
        assert_eq!(
            case.candidates[0].1[0].symbol,
            sm120::fold_verify_local::entry().unwrap().packed_symbol()
        );
        assert_ne!(
            case.control[0].arguments.0[8],
            case.candidates[0].1[0].arguments.0[8]
        );
    }
}

#[test]
#[ignore = "requires frozen wide local fold inputs and assigned GPUs"]
fn gpu_local64() {
    let bytes = Box::leak(
        std::fs::read(root().join("cubins/probe_sm120.cubin"))
            .unwrap()
            .into_boxed_slice(),
    );
    let cubins = kernel_harness::Cubins::new(
        Box::leak(Box::new([("probe_sm120", &bytes[..])])),
        &[(
            "probe_sm120",
            &["chop_probe_read_v1", "chop_probe_flush_v1"],
        )],
    );
    let ran = kernel_harness::gpu_test_with(cubins, 6 << 30, |s| {
        let mode = mode()?;
        let mut count = 0;
        let base = base_of(8192, 1280, 1);
        for rows in row_counts(64, mode) {
            let mut inputs = vec![data::synthetic(&base, rows)];
            if matches!(mode, Mode::Check | Mode::Time) {
                inputs.extend(data::captured_wide(&base, rows)?);
            }
            for input in inputs {
                let record = status_case(s, "attention.q-b", 64, &input, mode)?;
                println!("FOLD_LOCAL64_CASE {record}");
                count += 1;
            }
        }
        let expected = if matches!(mode, Mode::Memcheck | Mode::Racecheck) {
            2
        } else {
            18
        };
        assert_eq!(count, expected);
        println!(
            "FOLD_LOCAL64_DONE {}",
            json!({"cases":count,"mode":std::env::var("CHOPINFER_FOLD_FUSED_MODE").unwrap()})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
