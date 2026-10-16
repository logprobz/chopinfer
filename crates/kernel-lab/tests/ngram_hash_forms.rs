//! The registered n-gram hash entries every model catalog declares
//! (`ModelCatalog::ngram_hashes`), run from the cubins its bundle embeds:
//! ids and history against the exact oracle on every rank of each declared
//! rank count, for chunked prefill histories with EOS resets, decode buckets
//! with inactive slots and irregular verify items, with graph replays. The
//! cubins come from the bundles' kernel-build reports
//! (`CHOPINFER_BUNDLE_REPORTS`, paths separated by `:`), each checked
//! against its lock and sources.

use kernel_core::contract::{Argument as A, ArgumentList, LaunchPlan};
use kernel_core::ngram_oracle as ng;
use kernel_core::precision::ExecutionPhase;
use kernel_lab::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use kernels_sm120::row::ngram_hash::NgramHashForm;
use runtime_cuda::{DeviceBuffer, Module};
use serde_json::json;
use std::collections::BTreeMap;

const TOKENS: u64 = 1;
const HISTORY: u64 = 2;
const MULTIPLIERS: u64 = 3;
const SIZES: u64 = 4;
const OFFSETS: u64 = 5;
const IDS: u64 = 6;
const HISTORY_OUT: u64 = 7;
const LIVE: u64 = 8;
const CU_ROWS: u64 = 9;
const LIVE_ITEMS: u64 = 10;
const ENTRY_WORDS: usize = 11;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn upload(s: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = s.allocate(bytes.len().max(4))?;
    if !bytes.is_empty() {
        buffer.copy_from_bytes(bytes, &s.stream).map_err(err)?;
    }
    Ok(buffer)
}

fn words<T: Copy>(values: &[T], bytes: impl Fn(T) -> Vec<u8>) -> Vec<u8> {
    values.iter().flat_map(|&v| bytes(v)).collect()
}

/// A reported cubin's file and entry symbols.
type Reported = BTreeMap<String, (std::path::PathBuf, Vec<String>)>;

/// Every reported bundle cubin by name, from lock-verified current reports.
fn reported() -> LabResult<Reported> {
    let reports = std::env::var_os("CHOPINFER_BUNDLE_REPORTS").ok_or(
        "this GPU test needs CHOPINFER_BUNDLE_REPORTS: the kernel-build-report.json of each \
             bundle build holding the declared forms (in the bundle crate's build script \
             output directory), separated by `:`; no in-repo script sets it, so a GPU run \
             fails without it",
    )?;
    let mut cubins = BTreeMap::new();
    for path in std::env::split_paths(&reports) {
        let report = kernel_build::BuildReport::read(&path).map_err(err)?;
        report.verify_current().map_err(err)?;
        for cubin in report.cubins {
            let symbols = cubin.declarations.iter().map(|d| d.symbol.clone());
            if cubins
                .insert(cubin.name.clone(), (cubin.path, symbols.collect()))
                .is_some()
            {
                return Err(format!("two reports build {}", cubin.name));
            }
        }
    }
    Ok(cubins)
}

/// Every catalog's declared hash forms.
fn forms() -> Vec<NgramHashForm> {
    kernel_lab::catalogs()
        .models()
        .iter()
        .flat_map(|model| (model.ngram_hashes)())
        .collect()
}

/// With reports listed, each declared form's cubin is reported with its
/// entries: the CPU check of a GPU job's inputs.
#[test]
fn reported_bundles_build_every_declared_hash_form() {
    if std::env::var_os("CHOPINFER_BUNDLE_REPORTS").is_none() {
        eprintln!("SKIP: set CHOPINFER_BUNDLE_REPORTS to check the reported cubins");
        return;
    }
    let reported = reported().unwrap();
    for form in forms() {
        let (_, symbols) = reported
            .get(&form.cubin)
            .unwrap_or_else(|| panic!("no listed report builds {}", form.cubin));
        for symbol in [&form.prefill, &form.decode, &form.verify] {
            assert!(symbols.contains(symbol), "{symbol}");
        }
    }
}

/// One launch of `form`'s `phase` entry on `rank` over items of `lengths`
/// tokens after `initial` histories, checked against the oracle.
fn hash(
    s: &Session,
    module: &Module,
    form: &NgramHashForm,
    phase: ExecutionPhase,
    rank: u32,
    lengths: &[u32],
    initial: &[[u32; 2]],
) -> LabResult<()> {
    let t = form.table;
    let tables = ng::ngram_tables(
        t.vocab,
        t.orders,
        t.heads_per_order,
        t.base,
        t.seed,
        t.layer_index,
        t.alignment,
    )
    .map_err(err)?;
    let local = form.local_heads as usize;
    let heads = local * form.ranks as usize;
    let per_order = t.heads_per_order;
    // A rank owns the table rows of its heads.
    let first = rank as usize * local;
    let start = tables.offsets[first];
    let end = if rank + 1 == form.ranks {
        tables.rows
    } else {
        tables.offsets[first + local]
    };
    let rows = end - start;
    let eos = form.eos;
    let mut cu_rows = vec![0_i32];
    for &n in lengths {
        cu_rows.push(cu_rows.last().unwrap() + n as i32);
    }
    let decode = phase == ExecutionPhase::Decode;
    let total = if decode {
        lengths.len()
    } else {
        *cu_rows.last().unwrap() as usize
    };
    let vocab = t.vocab as u32;
    let tokens: Vec<u32> = (0..total)
        .map(|i| {
            if i % 7 == 2 {
                eos
            } else {
                (i as u32 * 7919 + 11) % vocab
            }
        })
        .collect();
    let mut expected_ids = vec![u64::MAX; total * local + 8];
    let mut history = vec![u32::MAX; lengths.len() * ENTRY_WORDS];
    let mut expected_history = history.clone();
    for (item, &n) in lengths.iter().enumerate() {
        history[item * ENTRY_WORDS..item * ENTRY_WORDS + 2].copy_from_slice(&initial[item]);
        let at = if decode { item } else { cu_rows[item] as usize };
        if n == 0 {
            if decode {
                expected_ids[at * local..(at + 1) * local].fill(start);
            }
            continue;
        }
        let reference = ng::ngram_rows(
            &tables,
            per_order,
            &initial[item],
            &tokens[at..at + n as usize],
            eos,
        )
        .map_err(err)?;
        for row in 0..n as usize {
            let first = row * heads + rank as usize * local;
            expected_ids[(at + row) * local..(at + row + 1) * local]
                .copy_from_slice(&reference.rows[first..first + local]);
        }
        expected_history[item * ENTRY_WORDS..item * ENTRY_WORDS + 2]
            .copy_from_slice(&reference.history);
    }
    let u32s = |values: &[u32]| words(values, |v| v.to_le_bytes().to_vec());
    let u64s = |values: &[u64]| words(values, |v| v.to_le_bytes().to_vec());
    let buffers = BTreeMap::from([
        (TOKENS, upload(s, &u32s(&tokens))?),
        (HISTORY, upload(s, &u32s(&history))?),
        (MULTIPLIERS, upload(s, &u64s(&tables.multipliers))?),
        (SIZES, upload(s, &u64s(&tables.sizes))?),
        (OFFSETS, upload(s, &u64s(&tables.offsets))?),
        (IDS, upload(s, &u64s(&vec![u64::MAX; expected_ids.len()]))?),
        (
            HISTORY_OUT,
            upload(s, &u32s(&vec![u32::MAX; expected_history.len()]))?,
        ),
        (LIVE, upload(s, &u32s(lengths))?),
        (
            CU_ROWS,
            upload(s, &words(&cu_rows, |v| v.to_le_bytes().to_vec()))?,
        ),
        (
            LIVE_ITEMS,
            upload(s, &(lengths.len() as u32).to_le_bytes())?,
        ),
    ]);
    let mut args = vec![
        A::DeviceBuffer(TOKENS),
        A::DeviceBuffer(HISTORY),
        A::DeviceBuffer(MULTIPLIERS),
        A::DeviceBuffer(SIZES),
        A::DeviceBuffer(OFFSETS),
        A::DeviceBuffer(IDS),
        A::DeviceBuffer(HISTORY_OUT),
    ];
    if phase == ExecutionPhase::Verify {
        args.push(A::DeviceBuffer(CU_ROWS));
    }
    args.push(A::DeviceBuffer(LIVE));
    if phase == ExecutionPhase::Verify {
        args.push(A::DeviceBuffer(LIVE_ITEMS));
    }
    args.extend([A::U32(rank), A::U64(start), A::U64(rows)]);
    if phase != ExecutionPhase::Prefill {
        args.push(A::U64(ENTRY_WORDS as u64));
    }
    let grid = match phase {
        ExecutionPhase::Prefill => [total.max(1) as u32, local as u32, 1],
        ExecutionPhase::Decode => [lengths.len() as u32, local as u32, 1],
        ExecutionPhase::Verify => [
            lengths.iter().copied().max().unwrap_or(0).max(1),
            local as u32,
            lengths.len() as u32,
        ],
    };
    let symbol = match phase {
        ExecutionPhase::Prefill => &form.prefill,
        ExecutionPhase::Decode => &form.decode,
        ExecutionPhase::Verify => &form.verify,
    };
    let function = module.function(symbol).map_err(err)?;
    let plan = LaunchPlan {
        grid,
        block: [256, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: true,
    };
    let arguments = ArgumentList(args);
    let bindings = Bindings {
        buffers: buffers.iter().map(|(id, b)| (*id, b)).collect(),
    };
    let graph =
        s.capture(|stream| launch(&s.context, &function, stream, plan, &arguments, &bindings))?;
    // The outputs as the oracle expects them: rows and history the launch
    // leaves unwritten keep the 0xFF fill.
    let outputs = [&buffers[&IDS], &buffers[&HISTORY_OUT]];
    let matches = || -> LabResult<bool> {
        let mut ids = vec![0; expected_ids.len() * 8];
        buffers[&IDS]
            .copy_to_bytes(&mut ids, &s.stream)
            .map_err(err)?;
        let ids: Vec<u64> = ids
            .chunks_exact(8)
            .map(|v| u64::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let mut written = vec![0; expected_history.len() * 4];
        buffers[&HISTORY_OUT]
            .copy_to_bytes(&mut written, &s.stream)
            .map_err(err)?;
        Ok(ids == expected_ids && written == u32s(&expected_history))
    };
    // Each replay rewrites poisoned outputs, so none passes on bytes an
    // earlier launch left behind.
    for _ in 0..3 {
        kernel_lab::poison::poison(s, &outputs)?;
        graph.launch(&s.stream).map_err(err)?;
        if !matches()? {
            return Err(format!(
                "{symbol} rank {rank} {lengths:?}: ids or history differ from the exact oracle"
            ));
        }
    }
    // Negative control: poisoned outputs without a launch must not pass,
    // or the comparison could not see a replay that skips its writes.
    kernel_lab::poison::poison(s, &outputs)?;
    // A launch that writes nothing (an empty prefill) has no stale output.
    let stale_refused = total > 0;
    if stale_refused && matches()? {
        return Err(format!(
            "{symbol} rank {rank} {lengths:?}: unlaunched poisoned outputs pass the oracle"
        ));
    }
    s.receipt(
        "registered-ngram-hash",
        BTreeMap::from([
            ("symbol".into(), json!(symbol)),
            ("ranks".into(), json!(form.ranks)),
            ("rank".into(), json!(rank)),
            ("lengths".into(), json!(lengths)),
            ("bitwise".into(), json!(true)),
            ("graph_replays".into(), json!(3)),
            ("poisoned_before_each_replay".into(), json!(true)),
            ("stale_output_refused".into(), json!(stale_refused)),
        ]),
    )?;
    Ok(())
}

#[test]
fn declared_ngram_hash_forms_match_the_oracle_on_every_rank() {
    let result = kernel_lab::gpu_test(64 * 1024 * 1024, |s| {
        let reported = reported()?;
        for form in forms() {
            let (path, _) = reported
                .get(&form.cubin)
                .ok_or_else(|| format!("no listed bundle report builds {}", form.cubin))?;
            let module = s.load_cubin(&std::fs::read(path).map_err(err)?)?;
            let eos = form.eos;
            for rank in 0..form.ranks {
                for n in [0, 1, 2, 3, 17, 129, 8192] {
                    for initial in [[eos, eos], [11, eos], [11, 42]] {
                        hash(
                            s,
                            &module,
                            &form,
                            ExecutionPhase::Prefill,
                            rank,
                            &[n],
                            &[initial],
                        )?;
                    }
                }
                for capacity in [1_u32, 2, 4, 8, 16, 32, 64, 128, 256] {
                    let lengths: Vec<u32> = (0..capacity).map(|i| u32::from(i % 3 != 1)).collect();
                    let history: Vec<[u32; 2]> = (0..capacity).map(|i| [i, eos]).collect();
                    hash(
                        s,
                        &module,
                        &form,
                        ExecutionPhase::Decode,
                        rank,
                        &lengths,
                        &history,
                    )?;
                }
                hash(
                    s,
                    &module,
                    &form,
                    ExecutionPhase::Verify,
                    rank,
                    &[0, 1, 3, 2, 0, 4],
                    &[[11, 42]; 6],
                )?;
            }
        }
        Ok(())
    })
    .expect("registered n-gram hash GPU cases");
    if result.is_none() {
        eprintln!("UNRUN: the registered n-gram hash forms require an idle assigned GPU");
    }
}
