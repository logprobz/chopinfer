//! The router tile pair (../../../router_wide.md): the certified router
//! projection at its first 32-expert by 32-row tiles (served until the
//! 64-expert binding, now from the lab cubin) against the same projection
//! at 32-, 64- and 128-expert tiles, bit for bit on the logits and timed as
//! paired graphs. Worker w takes captured layer 14 or 39 (the
//! captured MoE input codes widened to BF16, with the layer's router
//! weights) and two synthetic cases whose uncertified rows and experts
//! reach the SIMT path.
use super::*;
use std::path::{Path, PathBuf};

pub const DATA: &str = "CHOPINFER_DEEPSEEK41_ROUTER_WIDE_DATA";
pub const WORKER: &str = "CHOPINFER_DEEPSEEK41_ROUTER_WIDE_WORKER";
/// Captured layers by worker.
pub const LAYERS: [u32; 2] = [14, 39];

#[derive(Clone, Copy, Debug)]
pub enum Source {
    /// The worker's captured layer.
    Captured,
    Synthetic(Pattern),
}

/// (label, source, live rows) of `worker`'s cases.
pub fn cases(worker: u32) -> Vec<(String, Source, u32)> {
    let layer = LAYERS[worker as usize];
    let mut out = vec![(format!("captured-l{layer}"), Source::Captured, CAPACITY)];
    let synthetic = if worker == 0 {
        [(Pattern::Outliers, CAPACITY), (Pattern::Outliers, 65)]
    } else {
        [(Pattern::Outliers, 2049), (Pattern::Clean, CAPACITY)]
    };
    for (pattern, live) in synthetic {
        let name = format!("{pattern:?}").to_lowercase();
        out.push((format!("{name}-{live}"), Source::Synthetic(pattern), live));
    }
    out
}

/// A staged file named by DATA.json, checked against its size and digest.
fn data_file(dir: &Path, manifest: &serde_json::Value, name: &str) -> LabResult<Vec<u8>> {
    let item = &manifest["files"][name];
    let path = dir.join(name);
    let meta = std::fs::symlink_metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if !meta.is_file() || Some(meta.len()) != item["bytes"].as_u64() {
        return Err(format!("invalid data file {}", path.display()));
    }
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    if item["sha256"] != kernel_build::digest(&bytes) {
        return Err(format!("digest mismatch {}", path.display()));
    }
    Ok(bytes)
}

fn halves(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// The case's BF16 rows (`CAPACITY` by K) and router weights.
pub fn rows_and_weights(source: Source, worker: u32) -> LabResult<(Vec<u16>, Vec<u16>)> {
    match source {
        Source::Synthetic(pattern) => Ok((input_rows(pattern), router_weight())),
        Source::Captured => {
            let dir = PathBuf::from(std::env::var(DATA).map_err(|e| format!("{DATA}: {e}"))?);
            let manifest: serde_json::Value = serde_json::from_slice(
                &std::fs::read(dir.join("DATA.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let layer = LAYERS[worker as usize];
            let rows = halves(&data_file(
                &dir,
                &manifest,
                &format!("input-l{layer}.bf16"),
            )?);
            let weights = halves(&data_file(
                &dir,
                &manifest,
                &format!("router-l{layer}.bf16"),
            )?);
            if rows.len() != (CAPACITY * K) as usize || weights.len() != (N * K) as usize {
                return Err("captured rows or router weights have the wrong size".into());
            }
            Ok((rows, weights))
        }
    }
}

/// The projection's inputs on the device and its host flags.
pub struct Projection {
    pub activation: View,
    pub normed: View,
    pub certified: View,
    pub weight: View,
    pub weight_domain: View,
    pub live: View,
    pub uncertified: usize,
    pub simt_groups: usize,
}

pub fn projection(
    a: &mut Arena<'_>,
    rows: &[u16],
    weights: &[u16],
    live: u32,
) -> LabResult<Projection> {
    let k = K as usize;
    // The certifying norm's flag: every widened element a tensor operand.
    let flags: Vec<u32> = (0..CAPACITY as usize)
        .map(|r| {
            u32::from(
                r < live as usize
                    && rows[r * k..(r + 1) * k]
                        .iter()
                        .all(|&x| tensor_operand(widen(x))),
            )
        })
        .collect();
    let domain = weight_domain(weights);
    let bytes16 = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let bytes32 = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let widened: Vec<u8> = rows.iter().flat_map(|&x| widen(x).to_le_bytes()).collect();
    Ok(Projection {
        activation: a.bytes(&widened)?,
        normed: a.bytes(&bytes16(rows))?,
        certified: a.bytes(&bytes32(&flags))?,
        weight: a.bytes(&bytes16(weights))?,
        weight_domain: a.bytes(&bytes32(&domain))?,
        live: a.bytes(&live.to_le_bytes())?,
        uncertified: flags[..live as usize].iter().filter(|&&f| f == 0).count(),
        simt_groups: domain.iter().filter(|&&d| d == 0).count(),
    })
}

/// The 32-expert certified projection (`features` None) or a wide tile, into
/// `logits`, over a step of `rows` rows.
pub fn launch(p: &Projection, logits: View, rows: u32, features: Option<u32>) -> LabResult<Launch> {
    let e = family::router();
    let (symbol, plan) = match features {
        None => (
            family::certified_symbol(),
            certified::plan(&e, family::TILE_ROWS, rows),
        ),
        Some(f) => (
            family::wide_symbol(f),
            certified::wide_plan(&e, family::TILE_ROWS, f, rows),
        ),
    };
    Ok(make(
        symbol,
        rows,
        plan.map_err(|e| e.to_string())?,
        vec![
            ptr(p.activation),
            ptr(p.normed),
            ptr(p.certified),
            ptr(p.weight),
            ptr(logits),
            ptr(p.live),
            Argument::U32(rows),
            ptr(p.weight_domain),
        ],
    ))
}

fn case(
    runner: &mut Runner<'_>,
    session: &Session,
    worker: u32,
    label: &str,
    source: Source,
    live: u32,
) -> LabResult<(serde_json::Value, bool)> {
    let mut a = Arena::device(session);
    let (rows, weights) = rows_and_weights(source, worker)?;
    let p = projection(&mut a, &rows, &weights, live)?;
    let logits_bytes = u64::from(CAPACITY) * u64::from(N) * 4;
    let reference_logits = a.filled(logits_bytes, 0xa5)?;
    let reference = vec![launch(&p, reference_logits, live, None)?];
    runner.run(&mut a, &reference)?;
    let expected = a.read(reference_logits)?;
    let graph = |runner: &mut Runner<'_>, a: &Arena<'_>, ops: &[Launch]| {
        runner.ensure(ops)?;
        session.capture(|st| runner.issue(a, ops, st))
    };
    let reference_graph = graph(runner, &a, &reference)?;
    let mut passed = true;
    let mut arms = Vec::new();
    for features in family::WIDE_FEATURES {
        let logits = a.filled(logits_bytes, 0xa5)?;
        let ops = vec![launch(&p, logits, live, Some(features))?];
        runner.run(&mut a, &ops)?;
        let bitwise = a.read(logits)? == expected;
        passed &= bitwise;
        // Paired timing at the served step only.
        let timing = if live == CAPACITY {
            let candidate = graph(runner, &a, &ops)?;
            let graphs = [&reference_graph, &candidate];
            timing::pair(session, |arm, st| {
                graphs[arm].launch(st).map_err(|e| e.to_string())
            })?
        } else {
            serde_json::Value::Null
        };
        arms.push(serde_json::json!({
            "arm": format!("f{features}"),
            "bitwise": bitwise,
            "timing": timing,
        }));
    }
    Ok((
        serde_json::json!({
            "case": label,
            "live": live,
            "uncertified_rows": p.uncertified,
            "simt_weight_groups": p.simt_groups,
            "arms": arms,
            "passed": passed,
        }),
        passed,
    ))
}

pub fn run(session: &Session) -> LabResult<()> {
    let worker: u32 = std::env::var(WORKER)
        .map_err(|e| format!("{WORKER}: {e}"))?
        .parse()
        .map_err(|_| "worker must be 0 or 1")?;
    if worker > 1 {
        return Err("worker must be 0 or 1".into());
    }
    let mut runner = Runner::new(Some(session))?;
    let mut executed = 0;
    let mut failed = 0;
    for (label, source, live) in cases(worker) {
        let (record, passed) = case(&mut runner, session, worker, &label, source, live)?;
        println!("router_wide_record {record}");
        executed += 1;
        failed += usize::from(!passed);
    }
    println!(
        "router_wide_done {}",
        serde_json::json!({"worker": worker, "cases": executed, "failed": failed})
    );
    if executed != cases(worker).len() || failed > 0 {
        return Err(format!("{failed} of {executed} router tile cases failed"));
    }
    Ok(())
}

#[test]
fn router_wide_cpu_plan() {
    // Every arm names a lab cubin entry with the served arguments, block and
    // rows; only the feature grid and the shared memory differ. The cases
    // reach uncertified rows and SIMT experts.
    let lab: std::collections::BTreeSet<String> = family::cubins()
        .into_iter()
        .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
        .collect();
    let mut a = Arena::host();
    let (rows, weights) = rows_and_weights(Source::Synthetic(Pattern::Outliers), 0).unwrap();
    let p = projection(&mut a, &rows, &weights, 2049).unwrap();
    assert!(p.uncertified > 0 && p.simt_groups > 0);
    let logits = a.filled(4, 0).unwrap();
    for live in [CAPACITY, 2049, 65] {
        let served = launch(&p, logits, live, None).unwrap();
        assert!(lab.contains(&served.symbol), "{}", served.symbol);
        assert!(!Runner::serving(&served.symbol), "{}", served.symbol);
        for features in family::WIDE_FEATURES {
            let wide = launch(&p, logits, live, Some(features)).unwrap();
            assert!(lab.contains(&wide.symbol), "{}", wide.symbol);
            // The bundle serves the 64-expert tile, which the runner loads
            // from the bundle.
            assert_eq!(
                Runner::serving(&wide.symbol),
                features == family::SERVED_FEATURES,
                "{}",
                wide.symbol
            );
            assert_eq!(wide.arguments, served.arguments);
            assert_eq!(wide.plan.block, served.plan.block);
            assert_eq!(wide.plan.grid[1], served.plan.grid[1]);
            assert_eq!(wide.plan.grid[0], N.div_ceil(features));
        }
    }
    for worker in [0, 1] {
        let c = cases(worker);
        assert_eq!(c.len(), 3);
        assert!(matches!(c[0].1, Source::Captured));
        assert_eq!(c[0].2, CAPACITY);
    }
    println!(
        "router_wide_cpu_plan {}",
        serde_json::json!({"arms": family::WIDE_FEATURES.len()})
    );
}

#[test]
#[ignore = "requires the staged lab cubin, the data named by CHOPINFER_DEEPSEEK41_ROUTER_WIDE_DATA and a worker"]
fn router_wide_gpu() {
    kernel_harness::gpu_test_with(kernel_harness::Cubins::NONE, 8 << 30, run).unwrap();
}
