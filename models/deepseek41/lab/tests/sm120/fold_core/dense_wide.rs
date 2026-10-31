//! Every schedule a wide verify fold row (`dense_fold.wide.verify`) may
//! choose against the served fold: bitwise at every row each wide bucket
//! serves, timed at the middle and widest of them.
use super::*;
#[allow(dead_code)]
#[path = "../../../../../../crates/kernel-lab/families/tests/support/dense_decode.rs"]
mod support;
use kernels_sm120::gemm::dense_decode::{self, Finish, Knobs};
use model_deepseek41_contract::verify_bounds::WIDE;

const CUBIN: &str = "dense_wide_schedules";
const DIRECTORY: &str = "CHOPINFER_DENSE_WIDE_SCHEDULES";
/// Bytes written before each cold sample, as the harness's write eviction.
const EVICTION_BYTES: usize = 256 << 20;

/// Every schedule a wide row admits: one ticket row group of three
/// stages at either column tile, with or without weight prefetch.
fn schedules() -> Vec<Knobs> {
    [32, 64]
        .into_iter()
        .flat_map(|tile_n| {
            [false, true].map(|weight_prefetch| Knobs {
                tile_n,
                row_groups: 1,
                stages: 3,
                weight_prefetch,
                finish: Finish::Ticket,
            })
        })
        .collect()
}

fn bases() -> Vec<dense_decode::Entry> {
    let bases = sm120::mxfp8_fold::wide_bases(&FLASH, 4).unwrap();
    assert!(!bases.is_empty() && bases.iter().all(|b| !b.paired));
    bases
}

/// The rows wide bucket `bucket` serves: those above the next narrower
/// bucket, or above the decode buckets.
fn served_rows(bucket: u32) -> std::ops::RangeInclusive<u32> {
    let below = WIDE
        .into_iter()
        .filter(|&b| b < bucket)
        .max()
        .unwrap_or(dense_decode::DECODE_ROWS);
    below + 1..=bucket
}

/// The timed rows of `bucket`: the middle and the widest it serves.
fn timed(bucket: u32) -> [u32; 2] {
    let rows = served_rows(bucket);
    [(rows.start() + rows.end()) / 2, bucket]
}

/// The wide fold of decode base `base` at `bucket` with schedule `knobs`,
/// named as `sm120::mxfp8_fold::wide_entry` names it.
fn entry(base: &dense_decode::Entry, bucket: u32, knobs: Knobs) -> dense_decode::Entry {
    let q = base.query;
    dense_decode::Entry::wide(
        sm120::mxfp8_fold::routed_query(q, bucket),
        &format!(
            "chop_deepseek41_dense_decode_n{}_k{}_g{}_single",
            q.n, q.k, q.groups
        ),
        false,
        knobs,
        dense_decode::WIDE_ROWS,
    )
    .unwrap()
}

fn specs() -> Vec<kernel_build::CubinSpec> {
    let mut entries = BTreeMap::new();
    for base in bases() {
        for knobs in schedules() {
            let e = entry(&base, *WIDE.last().unwrap(), knobs);
            entries.insert(e.symbol.clone(), e);
        }
    }
    let entries: Vec<_> = entries.into_values().collect();
    vec![dense_decode::cubin_spec(CUBIN, &entries, true).unwrap()]
}

/// The served backend's wide folds, both buckets registered.
fn served() -> &'static Sm120 {
    static BACKEND: OnceLock<Sm120> = OnceLock::new();
    BACKEND.get_or_init(|| {
        sm120::mxfp8_fold::register_wide(Sm120::new(), &FLASH, 4, &WIDE)
            .unwrap()
            .with_packed_fold_weights()
    })
}

/// The launches of `base`'s fold at `rows` of `bucket`: the served one,
/// or schedule `knobs`'s through the same backend selection.
fn launches(
    a: &mut Arena<'_>,
    base: &dense_decode::Entry,
    bucket: u32,
    knobs: Option<Knobs>,
    b: &Buffers<View>,
    rows: u32,
) -> LabResult<Vec<Launch>> {
    let q = sm120::mxfp8_fold::routed_query(base.query, bucket);
    let op = match knobs {
        None => served().mxfp8_fold_gemm(&q, Need::rows(bucket)),
        Some(knobs) => Sm120::new()
            .with_dense_decode(entry(base, bucket, knobs), None)
            .with_packed_fold_weights()
            .mxfp8_fold_gemm(&q, Need::rows(bucket)),
    }
    .map_err(|e| e.to_string())?;
    let work = a.filled(op.info().workspace.max(4), 0)?;
    op.launches(b, work, Counts::rows(rows))
        .map_err(|e| e.to_string())
}

/// Functions of `calls`: the bundle's where it holds the symbol, so the
/// served schedule's arm is an A/A twin of the served fold, otherwise
/// the staged schedule module's.
fn load(s: &Session, lab: &Module, calls: &[Launch]) -> LabResult<Functions> {
    struct Cache(std::cell::RefCell<BTreeMap<String, Module>>);
    let cache = s.state(|| Ok(Cache(std::cell::RefCell::new(BTreeMap::new()))))?;
    let mut bundle = cache.0.borrow_mut();
    let mut map = BTreeMap::new();
    for call in calls {
        if map.contains_key(&call.symbol) {
            continue;
        }
        let f = match fold_cubin_for_symbol(&call.symbol) {
            Some((name, bytes)) => {
                if !bundle.contains_key(name) {
                    bundle.insert(name.to_owned(), s.load_cubin(bytes)?);
                }
                bundle[name].function(&call.symbol)
            }
            None => lab.function(&call.symbol),
        }
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

fn symbols(calls: &[Launch]) -> Vec<&str> {
    calls.iter().map(|c| c.symbol.as_str()).collect()
}

/// Every schedule against the served fold at `rows` of `bucket`, on
/// synthetic inputs: live and rejected rows, eager and two graph
/// replays, and a changed activation the comparison must see. `timed`
/// also times every arm warm and cold.
fn check(
    s: &Session,
    lab: &Module,
    flush: &DeviceBuffer,
    base: &dense_decode::Entry,
    bucket: u32,
    rows: u32,
    timed: bool,
) -> LabResult<Value> {
    let q = base.query;
    let legacy = entry(base, bucket, Knobs::ticket(q.n))
        .legacy()
        .map_err(|e| e.to_string())?;
    let mut input = data::synthetic(&legacy, rows);
    // Every arm reads the packed codes the served fold reads.
    input.weights = kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout {
        groups: q.groups as usize,
        n: q.n as usize,
        k: q.k as usize,
    }
    .pack(&input.weights)
    .map_err(|e| e.to_string())?;
    let schedules = schedules();
    let arms = 1 + schedules.len();
    let mut a = Arena::new(Some(s));
    let b = (0..arms)
        .map(|_| common(&mut a, &legacy, rows, &input))
        .collect::<LabResult<Vec<_>>>()?;
    let mut calls = Vec::new();
    for (arm, knobs) in std::iter::once(None)
        .chain(schedules.iter().copied().map(Some))
        .enumerate()
    {
        calls.push(launches(&mut a, base, bucket, knobs, &b[arm], rows)?);
    }
    if !calls[1..].iter().any(|c| symbols(c) == symbols(&calls[0])) {
        return Err("the served fold is not one of the schedules".into());
    }
    let all: Vec<_> = calls.iter().flatten().cloned().collect();
    let f = load(s, lab, &all)?;
    let graphs = calls
        .iter()
        .map(|v| s.capture(|stream| issue(s, &a, &f, v, stream)))
        .collect::<LabResult<Vec<_>>>()?;
    let mut comparisons = 0;
    for (live, bad) in [
        (rows, false),
        (rows - 1, true),
        (0, false),
        (rows, true),
        (rows, false),
    ] {
        for replay in 0..3 {
            for arm in 0..arms {
                reset(&a, &b[arm], &input, rows, live, bad)?;
                if replay == 0 {
                    issue(s, &a, &f, &fused::eager(&calls[arm]), &s.stream)?;
                } else {
                    graphs[arm].launch(&s.stream).map_err(|e| e.to_string())?;
                }
                let actual = a.read(b[arm].status)?;
                let expected = status(rows, live, bad);
                if actual[..12] != expected[..12]
                    || actual[(rows as usize + 3) * 4..] != expected[(rows as usize + 3) * 4..]
                {
                    return Err("status guard changed".into());
                }
            }
            for arm in 1..arms {
                compare(&a, &[b[0], b[arm]], &input, rows, live, bad, None)?;
                comparisons += 1;
            }
        }
    }
    // The last round left the served output of every live row in `b[0]`.
    let mut changed = input.codes.clone();
    changed[(rows as usize - 1) * q.k as usize] ^= 1;
    for arm in 1..arms {
        reset(&a, &b[arm], &input, rows, rows, false)?;
        a.write(b[arm].activations, &changed)?;
        issue(s, &a, &f, &fused::eager(&calls[arm]), &s.stream)?;
        if a.read(b[0].output)? == a.read(b[arm].output)? {
            return Err("a changed activation left the output equal".into());
        }
    }
    let mut record = json!({
        "n":q.n, "k":q.k, "groups":q.groups, "bucket":bucket, "rows":rows,
        "comparisons":comparisons, "bitwise":true, "negative_controls":arms - 1,
        "graph_replays":2,
        "arms":std::iter::once("served".to_owned())
            .chain(schedules.iter().map(|k| k.tag()))
            .collect::<Vec<_>>(),
        "symbols":calls.iter().map(|c| symbols(c)).collect::<Vec<_>>(),
        "served_modules":fused::served_modules(&all),
    });
    if timed {
        for buffer in &b {
            reset(&a, buffer, &input, rows, rows, false)?;
        }
        for (name, cold) in [("warm", false), ("cold", true)] {
            record[name] = timing::sampled(
                s,
                arms,
                |stream| {
                    if cold {
                        flush
                            .zero_at_async(0, flush.byte_len(), stream)
                            .map_err(|e| e.to_string())?;
                    }
                    Ok(())
                },
                |arm, stream| graphs[arm].launch(stream).map_err(|e| e.to_string()),
            )?;
        }
    }
    Ok(record)
}

#[test]
fn wide_schedules_keep_the_served_launch_shape() {
    let lab: std::collections::BTreeSet<_> = specs()[0]
        .entries
        .iter()
        .map(|e| e.symbol.clone())
        .collect();
    // Two launches (admission and fold) per base and schedule.
    assert_eq!(lab.len(), 2 * bases().len() * schedules().len());
    for base in bases() {
        for knobs in schedules() {
            // One symbol serves every capacity of a schedule.
            let [narrow, wide] =
                [128, 192].map(|bucket| entry(&base, bucket, knobs).packed_specs().unwrap());
            assert_eq!(
                narrow
                    .iter()
                    .map(|e| (&e.symbol, &e.body))
                    .collect::<Vec<_>>(),
                wide.iter()
                    .map(|e| (&e.symbol, &e.body))
                    .collect::<Vec<_>>()
            );
        }
        for bucket in WIDE {
            let served = sm120::mxfp8_fold::wide_entry(&FLASH, &base, bucket, bucket).unwrap();
            assert_eq!(entry(&base, bucket, served.knobs).symbol, served.symbol);
            let legacy = served.legacy().unwrap();
            for rows in [*served_rows(bucket).start(), timed(bucket)[0], bucket] {
                let mut a = Arena::new(None);
                let b = common(&mut a, &legacy, rows, &data::empty(&legacy, rows)).unwrap();
                let reference = launches(&mut a, &base, bucket, None, &b, rows).unwrap();
                assert!(
                    reference
                        .iter()
                        .all(|c| fold_cubin_for_symbol(&c.symbol).is_some())
                );
                for knobs in schedules() {
                    let calls = launches(&mut a, &base, bucket, Some(knobs), &b, rows).unwrap();
                    assert_eq!(calls.len(), reference.len());
                    for (call, served) in calls.iter().zip(&reference) {
                        call.plan.validate_geometry().unwrap();
                        assert!(lab.contains(&call.symbol), "{}", call.symbol);
                        assert_eq!(call.arguments.0.len(), served.arguments.0.len());
                    }
                    if knobs == served.knobs {
                        assert_eq!(symbols(&calls), symbols(&reference));
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires an external directory named by CHOPINFER_DENSE_WIDE_SCHEDULES"]
fn build_wide_schedules() {
    let directory = PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    support::build(&specs(), &directory).unwrap();
}

/// Every base, wide bucket and served row, or the bases of shard
/// `CHOPINFER_DENSE_WIDE_SHARD` of `CHOPINFER_DENSE_WIDE_SHARDS`.
#[test]
#[ignore = "requires CHOPINFER_DENSE_WIDE_SCHEDULES staged cubins and assigned GPUs"]
fn gpu_wide_schedules() {
    let directory = PathBuf::from(std::env::var_os(DIRECTORY).expect(DIRECTORY));
    let count = |name: &str, default: usize| {
        std::env::var(name).map_or(default, |v| v.parse::<usize>().expect(name))
    };
    let (shard, shards) = (
        count("CHOPINFER_DENSE_WIDE_SHARD", 0),
        count("CHOPINFER_DENSE_WIDE_SHARDS", 1),
    );
    assert!(shard < shards, "invalid shard");
    let ran = kernel_harness::gpu_test(6 << 30, |s| {
        let lab = support::load(s, &directory, CUBIN)?;
        let flush = s.allocate(EVICTION_BYTES)?;
        let start = std::time::Instant::now();
        let mut cases = 0;
        for (index, base) in bases().iter().enumerate() {
            if index % shards != shard {
                continue;
            }
            for bucket in WIDE {
                for rows in served_rows(bucket) {
                    let timed = timed(bucket).contains(&rows);
                    let record = check(s, &lab, &flush, base, bucket, rows, timed)?;
                    println!("DENSE_WIDE_CASE {record}");
                    cases += 1;
                }
            }
        }
        println!(
            "DENSE_WIDE_DONE {}",
            json!({"cases":cases, "shard":shard, "shards":shards,
                "seconds":start.elapsed().as_secs()})
        );
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
