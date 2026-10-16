//! Fragment prefill GEMMs under prompt chunking. The generated invariance
//! tests (`gemm_invariance.rs`) prove each entry's prompt rows ignore the
//! chunk split; here a Qwen3.8 logits row must also give the same bits from
//! each F32 head entry, and the timing case runs each entry at the rows and
//! grids of prefill chunks.
//!
//! Environment: `CHOPINFER_KERNEL_BENCH_FILTER` keeps entries whose symbol
//! contains any comma-separated part; `CHOPINFER_KERNEL_BENCH_POINTS`
//! replaces the timed `rows@grid` points; `CHOPINFER_KERNEL_BENCH=1`
//! enables the timing case.

use super::invariance::{launch_rows, resize, wide_values};
use super::*;

/// Timed `rows@grid` points: a 2,048-token chunk, its pipelined halves, and
/// small chunks.
const BENCH_POINTS: [(u32, u32); 5] = [(2048, 188), (1024, 172), (512, 188), (64, 188), (1, 188)];

/// Every MiMo and Qwen3.8 TP4 fragment prefill entry a prompt runs.
fn chunk_entries() -> Vec<String> {
    let filter = std::env::var("CHOPINFER_KERNEL_BENCH_FILTER").ok();
    gemm::catalog(kernel_lab::catalogs())
        .into_iter()
        .filter(|item| {
            matches!(
                item.profile,
                model_mimo26_contract::sm120::config::PROFILE
                    | model_qwen38_contract::sm120::config::PROFILE
            ) && gemm::fragment_weight_enabled(&item.key)
                && query(item).m_capacity == 8192
        })
        .map(|item| item.instantiation.symbol)
        .filter(|symbol| {
            filter
                .as_deref()
                .is_none_or(|f| f.split(',').any(|part| symbol.contains(part)))
        })
        .collect()
}

/// A Qwen3.8 target logits row has the same bits from every F32 head entry
/// a step may select: a decode bucket (a lone prompt-ending row, or up to 32
/// head rows), and the prefill head (more than 32 rows, or scored prompt
/// rows beside it), at every topology.
#[test]
fn qwen_head_rows_match_across_entries() {
    watched_gpu_test(4 * 1024 * 1024 * 1024, 900, |session| {
        let catalog = gemm::catalog(kernel_lab::catalogs());
        let symbol = |tag: &str, name: &str| {
            let symbol = format!("chop_gemm_{tag}_{name}_v1");
            catalog
                .iter()
                .any(|item| item.symbol == symbol)
                .then_some(symbol)
        };
        let mut failures = Vec::new();
        let mut compared = 0;
        for tag in ["qwen38", "qwen38_tp2"] {
            let (Some(prefill), Some(single), Some(widest)) = (
                symbol(tag, "head_f32_prefill"),
                symbol(tag, "head_f32_m1"),
                symbol(tag, "head_f32_m32"),
            ) else {
                continue;
            };
            let mut prefill = Case::prepare(session, &prefill, 64, false)?;
            let mut single = Case::prepare(session, &single, 1, false)?;
            let mut widest = Case::prepare(session, &widest, 32, false)?;
            // The bundle launches its decode entries unsplit, one K chain.
            single.counts.split_k = 1;
            widest.counts.split_k = 1;
            let q = query(&prefill.variant);
            let n = q.n as usize;
            let all = wide_values(64 * q.k as usize, 0x4845_4144_524f_5753);
            let native = |case: &Case, m: u32| -> LabResult<u32> {
                let mut counts = case.counts;
                counts.m = m;
                counts.batch = m;
                Ok(gemm::plan(&case.variant.key, counts)
                    .map_err(|e| e.to_string())?
                    .grid[0])
            };
            let k = q.k as usize;
            let window = |case: &mut Case, m: u32, grid: u32| -> LabResult<Vec<u8>> {
                Ok(launch_rows(session, case, &all[..m as usize * k], Some(grid))?.concat())
            };
            let grid = native(&widest, 32)?;
            let reference = window(&mut widest, 32, grid)?;
            let grid = native(&single, 1)?;
            let alone = window(&mut single, 1, grid)?;
            let mut arms = vec![("head_f32_m1 M=1".to_owned(), alone)];
            for m in [1, 32, 33, 64] {
                arms.push((
                    format!("head_f32_prefill M={m}"),
                    window(&mut prefill, m, 188)?,
                ));
            }
            for (arm, rows) in arms {
                // Four bytes per F32 output element.
                let shared = rows.len().min(reference.len());
                let differ = rows[..shared]
                    .chunks(4)
                    .zip(reference[..shared].chunks(4))
                    .filter(|(a, b)| a != b)
                    .count();
                let shared = shared / 4;
                let mut receipt = BTreeMap::new();
                receipt.insert("case".into(), json!("qwen_head_rows_match_across_entries"));
                receipt.insert("tag".into(), json!(tag));
                receipt.insert("arm".into(), json!(arm));
                receipt.insert("rows".into(), json!(shared / n));
                receipt.insert("elements".into(), json!(shared));
                receipt.insert("differ".into(), json!(differ));
                session.receipt("check", receipt)?;
                eprintln!("{tag} {arm} against head_f32_m32 M=32: {differ} of {shared} differ");
                compared += 1;
                if differ != 0 {
                    failures.push(format!("{tag} {arm}: {differ}"));
                }
            }
        }
        if compared == 0 {
            return Err("no Qwen3.8 F32 head entries in the catalog".into());
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "head rows depend on the entry: {}",
                failures.join("; ")
            ))
        }
    })
    .expect("Qwen3.8 head rows match across entries");
}

/// Kernel time of each entry at the rows and grids of prefill chunks.
#[test]
fn gemm_prefill_chunk_bench() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        kernel_lab::skip("GEMM chunk benchmarks: set CHOPINFER_KERNEL_BENCH=1 on idle GPUs")
            .unwrap();
        return;
    }
    let points = std::env::var("CHOPINFER_KERNEL_BENCH_POINTS")
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(|point| {
                    let (m, grid) = point.split_once('@').expect("points are rows@grid");
                    (m.parse().expect("rows"), grid.parse().expect("grid"))
                })
                .collect::<Vec<(u32, u32)>>()
        })
        .unwrap_or_else(|| BENCH_POINTS.to_vec());
    watched_gpu_test(4 * 1024 * 1024 * 1024, 1500, |session| {
        let most = points.iter().map(|&(m, _)| m).max().unwrap_or(1);
        for symbol in chunk_entries() {
            bench_points(
                session,
                &mut Case::prepare(session, &symbol, most, false)?,
                &points,
            )?;
        }
        Ok(())
    })
    .expect("fragment prefill chunk benchmarks");
}

/// Times `case` at every `rows@grid` point.
fn bench_points(session: &Session, case: &mut Case, points: &[(u32, u32)]) -> LabResult<()> {
    let symbol = case.variant.symbol.clone();
    for &(m, grid) in points {
        resize(case, m, Some(grid))?;
        let function = case.function(session)?;
        let timing = time(session, 2, 5, |stream| {
            case.launch(session, &function, stream)
        })?;
        let us = f64::from(timing.median_ms) * 1000.0;
        eprintln!("{symbol} M={m} grid={grid}: {us:.2} us");
        let mut receipt = BTreeMap::new();
        receipt.insert("case".into(), json!("gemm_prefill_chunk_bench"));
        receipt.insert("symbol".into(), json!(symbol));
        receipt.insert("m".into(), json!(m));
        receipt.insert("grid".into(), json!(grid));
        receipt.insert("median_us".into(), json!(us));
        receipt.insert("min_us".into(), json!(f64::from(timing.min_ms) * 1000.0));
        receipt.insert("rounds".into(), json!(timing.rounds));
        session.receipt("bench", receipt)?;
    }
    Ok(())
}
