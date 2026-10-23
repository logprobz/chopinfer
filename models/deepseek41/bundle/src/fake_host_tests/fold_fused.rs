//! Audit the status folds (`sm120::fold_fused`): in the 16- and 32-row
//! decode and verify buckets of every backbone layer each replaces its
//! projection's admission and fold (WO_a also WO_b's input quantizer) on
//! the same operands, the sliced quantizer replaces WO_a's input quantizer
//! on its spans, and every other launch is retained. Then the producers
//! whose rejected-row zeroing the status folds rely on.
use super::*;
use crate::sm120::fold_fused;
use std::collections::BTreeSet;

/// A run's records and its allocations (slot, rank, bytes) in order.
struct Run {
    records: Vec<String>,
    allocations: Vec<(u32, u32, u64)>,
}

fn run(kv: KvLayout) -> Run {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, kv);
    if kv.storage == KvStorage::Native {
        host.speculation.set(Some((5, 32)));
    }
    let mut instance = create(&host, 32);
    let mut records = Vec::new();
    let mut ends = [1_u64; 32];
    for slot in 0..32 {
        let count = if slot == 0 { 32 } else { 1 };
        let from = host.device.journal.borrow().len();
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, count, true),
            &mut Views::default(),
        );
        if slot == 0 {
            records.push("case prefill 32".into());
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            ends[0] = 32;
        }
    }
    for count in [8, 16, 32] {
        let from = host.device.journal.borrow().len();
        let items = (0..count)
            .map(|i| (i + 1, i, ends[i as usize], 7))
            .collect::<Vec<_>>();
        step(&mut instance, &decode(&items), &mut Views::default());
        records.push(format!("case decode {count}"));
        records.extend_from_slice(&host.device.journal.borrow()[from..]);
        for end in &mut ends[..count as usize] {
            *end += 1;
        }
    }
    if kv.storage == KvStorage::Native {
        // Four rows per item: verify 32 on slots 0..8, then verify 16 on
        // slots 8..12.
        for (first, count) in [(0_u64, 8_u64), (8, 4)] {
            let from = host.device.journal.borrow().len();
            let tokens = vec![7; 4];
            let items = (first..first + count)
                .map(|i| (i + 1, i, ends[i as usize], tokens.as_slice()))
                .collect::<Vec<_>>();
            instance
                .execute(&verify(&items), &mut Views::default())
                .unwrap();
            instance
                .commit_accepted(
                    &(first + 1..=first + count)
                        .map(|i| kept(i, 4))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            records.push(format!("case verify {}", count * 4));
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            for end in &mut ends[first as usize..(first + count) as usize] {
                *end += 4;
            }
        }
    }
    let allocations = {
        let ranks = host.device.allocation_ranks.borrow();
        host.device
            .buffer_order
            .borrow()
            .iter()
            .map(|slot| {
                let (rank, bytes) = ranks.get(slot).copied().unwrap_or((u32::MAX, 0));
                (*slot, rank, bytes)
            })
            .collect()
    };
    for id in 1..=32 {
        instance.retire_request(id).unwrap();
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    Run {
        records,
        allocations,
    }
}

/// WO_a's block flags: one byte per 32 columns of each scratch row.
fn flags_bytes() -> u64 {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let output_a = u64::from(shape.local_groups() * shape.dims().attention.o_rank);
    u64::from(crate::buffers::scratch_rows()) * output_a / 32
}

/// Buffer names both runs share: `buffer<i>` for the reference's `i`th
/// allocation and its counterpart, and `flags<rank>` for the one
/// allocation per rank only the status folds make.
fn names(before: &Run, after: &Run) -> [BTreeMap<u32, String>; 2] {
    let flags = flags_bytes();
    let mut names = [BTreeMap::new(), BTreeMap::new()];
    let extra = |(slot, rank, bytes): (u32, u32, u64), names: &mut BTreeMap<u32, String>| {
        assert_eq!(
            bytes, flags,
            "the status folds add an allocation other than their flags"
        );
        names.insert(slot, format!("flags{rank}"));
    };
    let mut j = 0;
    for (i, &(slot, rank, bytes)) in before.allocations.iter().enumerate() {
        while (after.allocations[j].1, after.allocations[j].2) != (rank, bytes) {
            extra(after.allocations[j], &mut names[1]);
            j += 1;
        }
        names[0].insert(slot, format!("buffer{i}"));
        names[1].insert(after.allocations[j].0, format!("buffer{i}"));
        j += 1;
    }
    for &allocation in &after.allocations[j..] {
        extra(allocation, &mut names[1]);
    }
    let added: BTreeSet<_> = names[1]
        .values()
        .filter(|n| n.starts_with("flags"))
        .collect();
    assert_eq!(added.len(), 4, "one flags allocation per rank");
    names
}

/// `records` with buffers named and streams, events and graphs numbered by
/// first appearance, so the shifted handle counter compares equal.
fn canonical(records: &[String], names: &BTreeMap<u32, String>) -> Vec<String> {
    let mut ids: [BTreeMap<String, usize>; 3] = Default::default();
    let mut id = |kind: usize, raw: &str| {
        let next = ids[kind].len();
        let n = *ids[kind].entry(raw.to_owned()).or_insert(next);
        format!("{}{n}", ["stream", "event", "graph"][kind])
    };
    let marker = "BundleDeviceBufferHandleV1 { raw: ";
    records
        .iter()
        .map(|line| {
            let mut parts = line.split(' ').map(str::to_owned).collect::<Vec<_>>();
            match parts[0].as_str() {
                "launch" | "upload" | "copy" | "download" | "zero" | "synchronize" | "capture" => {
                    parts[1] = id(0, &parts[1])
                }
                "record" | "wait" => {
                    parts[1] = id(0, &parts[1]);
                    parts[2] = id(1, &parts[2]);
                }
                "captured" | "replay" => {
                    parts[1] = id(0, &parts[1]);
                    parts[2] = id(2, &parts[2]);
                }
                _ => {}
            }
            let line = parts.join(" ");
            let mut out = String::new();
            let mut rest = line.as_str();
            while let Some((head, tail)) = rest.split_once(marker) {
                out.push_str(head);
                out.push_str(marker);
                let (raw, tail) = tail.split_once(' ').unwrap();
                let slot = raw.parse::<u64>().unwrap() as u32;
                out.push_str(names.get(&slot).map_or(raw, String::as_str));
                out.push(' ');
                rest = tail;
            }
            out.push_str(rest);
            out
        })
        .collect()
}

type Span = (String, u64, u64);
fn spans(line: &str) -> Vec<Span> {
    line.split("DeviceBufferSpanV1 { buffer: BundleDeviceBufferHandleV1 { raw: ")
        .skip(1)
        .map(|part| {
            let (name, part) = part.split_once(" }, offset: ").unwrap();
            let (offset, part) = part.split_once(", length: ").unwrap();
            let length = part.split_once(" }").unwrap().0;
            (
                name.to_owned(),
                offset.parse().unwrap(),
                length.parse().unwrap(),
            )
        })
        .collect()
}
fn symbol(line: &str) -> &str {
    line.split_whitespace().nth(2).unwrap_or("")
}

/// The projections a status fold replaces at each bucket.
const PROJECTIONS: [&str; 5] = [
    "attention.q-a",
    "attention.kv",
    "attention.q-b",
    "attention.o-a",
    "attention.o-b",
];

/// The projection whose served admission (`true`) or fold `line` launches
/// at `rows` rows.
fn served(line: &str, rows: u32) -> Option<(&'static str, bool)> {
    let s = symbol(line);
    let (fold, admit) = match s.strip_suffix("_admit_v1") {
        Some(fold) => (fold, true),
        None => (s, false),
    };
    PROJECTIONS.into_iter().find_map(|id| {
        let served = if id == "attention.q-b" {
            crate::sm120::fold_local_rows::served(id, rows).map(|e| e.symbol())
        } else {
            crate::sm120::fold_cluster::served(id, rows).map(|e| e.symbol())
        };
        (served.as_deref() == Some(fold)).then_some((id, admit))
    })
}

/// The status fold symbol of `id` at `rows` rows.
fn status_fold(id: &str, rows: u32) -> String {
    if id == "attention.q-b" {
        fold_fused::wide(rows).unwrap().symbol()
    } else {
        fold_fused::fused(id, rows).unwrap().symbol()
    }
}

/// WO_a's served input quantizer, and WO_b's, which WO_a's merge replaces.
const WO_A_INPUT: &str = "chop_deepseek41_dense_ceil_w8192_cached_v1";
const WO_B_INPUT: &str = "chop_deepseek41_dense_ceil_f32_w2048_parallel_v1";

/// A journal's records without its capture markers, whether each lies
/// inside a capture, and each case's markers with the captures' node total.
/// The generic capture cuts a program into pieces at fixed fractions of its
/// launches (`bundle_runtime::graphs::pieces`), so removed launches may move
/// a cut; only the markers' sequence and the total are compared.
struct Split {
    records: Vec<String>,
    captured: Vec<bool>,
    markers: BTreeMap<String, (Vec<String>, i64)>,
}

fn split(journal: &[String]) -> Split {
    let mut out = Split {
        records: Vec::new(),
        captured: Vec::new(),
        markers: BTreeMap::new(),
    };
    let (mut case, mut inside) = (String::new(), false);
    for line in journal {
        if let Some(name) = line.strip_prefix("case ") {
            case = name.to_owned();
        }
        if line.starts_with("capture ") {
            inside = true;
            out.markers
                .entry(case.clone())
                .or_default()
                .0
                .push(line.clone());
            continue;
        }
        if let Some(rest) = line.strip_prefix("captured ") {
            inside = false;
            let (head, count) = rest.rsplit_once(' ').unwrap();
            let entry = out.markers.entry(case.clone()).or_default();
            entry.0.push(format!("captured {head}"));
            entry.1 += count.parse::<i64>().unwrap();
            continue;
        }
        out.records.push(line.clone());
        out.captured.push(inside);
    }
    out
}

/// The pieces each case replaced, by (case, piece).
fn audit(before: &[String], after: &[String]) -> Result<Vec<(String, String)>, String> {
    let (old_journal, new_journal) = (split(before), split(after));
    let (before, after) = (&old_journal.records, &new_journal.records);
    let (mut i, mut j) = (0, 0);
    let mut case = String::new();
    let mut changed = Vec::new();
    // Launches the replacements removed from each case's captures.
    let mut removed = BTreeMap::<String, i64>::new();
    while i < before.len() && j < after.len() {
        let (a, b) = (&before[i], &after[j]);
        if a == b {
            if let Some(name) = a.strip_prefix("case ") {
                case = name.to_owned();
            }
            i += 1;
            j += 1;
            continue;
        }
        let rows = match case.split_once(' ') {
            Some(("decode" | "verify", rows)) => rows.parse::<u32>().unwrap(),
            _ => return Err(format!("{case} changed: {a}\n{b}")),
        };
        if !fold_fused::ROWS.contains(&rows)
            || !a.starts_with("launch ")
            || !b.starts_with("launch ")
        {
            return Err(format!("{case} changed: {a}\n{b}"));
        }
        if symbol(a) == WO_A_INPUT {
            let sliced = fold_fused::sliced(rows).unwrap();
            if symbol(b) != sliced.symbol || spans(a) != spans(b) {
                return Err(format!(
                    "WO_a's input at {case} is not sliced on its spans: {b}"
                ));
            }
            changed.push((case.clone(), "attention.o-a input".into()));
            i += 1;
            j += 1;
            continue;
        }
        let Some((id, true)) = served(a, rows) else {
            return Err(format!("{case} changed an unrelated operation: {a}\n{b}"));
        };
        let fold = before.get(i + 1).ok_or("an admission without its fold")?;
        if served(fold, rows) != Some((id, false)) || symbol(b) != status_fold(id, rows) {
            return Err(format!(
                "{id} at {case} is not admission and fold to its status fold: {b}"
            ));
        }
        // The admission binds codes, scales, the workspace (its flags, then a
        // cluster fold's slabs), the live rows and the status.
        let admit = spans(a);
        let workspace = admit[2].clone();
        let status = admit[4].clone();
        let mut old: Vec<Span> = admit.iter().chain(&spans(fold)).cloned().collect();
        let mut consumed = 2;
        if id == "attention.o-a" {
            // WO_a's merge quantizes WO_b's input in place of its quantizer.
            let quantizer = before.get(i + 2).filter(|q| symbol(q) == WO_B_INPUT);
            old.extend(spans(
                quantizer.ok_or("WO_a without WO_b's input quantizer")?,
            ));
            consumed = 3;
        }
        if old_journal.captured[i..i + consumed]
            .iter()
            .any(|&c| c != new_journal.captured[j])
        {
            return Err(format!("{id} at {case} moved across a capture"));
        }
        let new = spans(b);
        let flags = new.iter().filter(|s| s.0.starts_with("flags")).count();
        let wo = matches!(id, "attention.o-a" | "attention.o-b");
        for span in &new {
            let slabs = (&span.0, span.1) == (&workspace.0, workspace.1)
                && span.2 + 256 == workspace.2
                && id != "attention.q-b";
            let kept = old.contains(span) && *span != workspace;
            if !(kept || slabs || wo && span.0.starts_with("flags")) {
                return Err(format!("{id} at {case} binds another operand: {span:?}"));
            }
        }
        for span in spans(fold).iter().chain([&status]) {
            if *span != workspace && !new.contains(span) {
                return Err(format!("{id} at {case} drops an operand: {span:?}"));
            }
        }
        if flags != usize::from(wo) {
            return Err(format!("{id} at {case} binds {flags} block flags"));
        }
        changed.push((case.clone(), id.to_owned()));
        if new_journal.captured[j] {
            *removed.entry(case.clone()).or_default() += consumed as i64 - 1;
        }
        i += consumed;
        j += 1;
    }
    if i != before.len() || j != after.len() {
        return Err("operation count changed".into());
    }
    let cases: BTreeSet<_> = old_journal
        .markers
        .keys()
        .chain(new_journal.markers.keys())
        .collect();
    for case in cases {
        let (old, new) = (
            &old_journal.markers.get(case),
            &new_journal.markers.get(case),
        );
        let fewer = removed.get(case).copied().unwrap_or(0);
        if !matches!((old, new), (Some(old), Some(new)) if old.0 == new.0 && new.1 == old.1 - fewer)
        {
            return Err(format!(
                "the captures of {case} changed more than its folds"
            ));
        }
    }
    Ok(changed)
}

/// The canonical records of the reference and of the status folds.
fn journals(kv: KvLayout) -> [Vec<String>; 2] {
    let before = fold_fused::with_reference(|| run(kv));
    let after = run(kv);
    let names = names(&before, &after);
    let journals = [
        canonical(&before.records, &names[0]),
        canonical(&after.records, &names[1]),
    ];
    if let Some(root) = std::env::var_os("CHOPINFER_STATUS_FOLD_JOURNALS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        for (arm, lines) in ["reference", "status"].into_iter().zip(&journals) {
            let name = format!("status.{:?}.{arm}.journal", kv.storage);
            std::fs::write(root.join(name), lines.join("\n")).unwrap();
        }
    }
    journals
}

#[test]
fn status_folds_change_only_their_decode_and_verify_projections() {
    for kv in KV_LAYOUTS {
        let [before, after] = journals(kv);
        let changed = audit(&before, &after).unwrap();
        let mut cases = vec!["decode 16", "decode 32"];
        if kv.storage == KvStorage::Native {
            cases.extend(["verify 32", "verify 16"]);
        }
        // Every backbone layer on each of the four ranks, not the drafter's.
        let mut expected = Vec::new();
        for case in cases {
            for piece in PROJECTIONS.into_iter().chain(["attention.o-a input"]) {
                expected.push((
                    (case.to_owned(), piece.to_owned()),
                    4 * FLASH.layers() as usize,
                ));
            }
        }
        expected.sort();
        let mut counted = BTreeMap::new();
        for key in changed {
            *counted.entry(key).or_insert(0) += 1;
        }
        assert_eq!(counted.into_iter().collect::<Vec<_>>(), expected, "{kv:?}");
        // The audit refuses a moved operand, a lost launch, a kept admission
        // and an operand bound in place of the flags.
        let at = after
            .iter()
            .position(|l| symbol(l) == status_fold("attention.o-a", 32))
            .unwrap();
        let mut bad = after.clone();
        bad[at] = bad[at].replacen("offset: 0,", "offset: 64,", 1);
        assert!(audit(&before, &bad).is_err());
        let mut bad = after.clone();
        bad.remove(at);
        assert!(audit(&before, &bad).is_err());
        let fold = crate::sm120::fold_cluster::served("attention.o-a", 32).unwrap();
        let admit = before
            .iter()
            .find(|l| symbol(l) == format!("{}_admit_v1", fold.symbol()))
            .unwrap();
        let mut bad = after.clone();
        bad.insert(at, admit.clone());
        assert!(audit(&before, &bad).is_err());
        let mut bad = after.clone();
        bad[at] = bad[at].replacen("raw: flags", "raw: buffer", 1);
        assert!(audit(&before, &bad).is_err());
        // And a capture that lost a node more than its folds.
        let captured = after
            .iter()
            .position(|l| l.starts_with("captured "))
            .unwrap();
        let mut bad = after.clone();
        let (head, count) = bad[captured].rsplit_once(' ').unwrap();
        bad[captured] = format!("{head} {}", count.parse::<u64>().unwrap() - 1);
        assert!(audit(&before, &bad).is_err());
    }
}

/// Each audited header's statement that stores a rejected row (nonzero
/// row flags) as zero codes with scale 1, the block quantizers' protocol,
/// or that ends an HC boundary in the cached norm quantizer.
const ZEROING: &[(&str, &str)] = &[
    ("row/block_quantize.cuh", "uint16_t codes = 0;"),
    ("row/ceil_epilogue.cuh", "else value.scale_code = 127;"),
    (
        "row/rmsnorm_quantize.cuh",
        "ceil_epilogue<Width,F32,FloorBits>(normed,codes,scales,status,status_row_offset);",
    ),
    (
        "row/rmsnorm_quantize_cached.cuh",
        "const uint16_t value = flags == 0 ? saved_codes[slot] : 0;",
    ),
    (
        "row/rmsnorm_quantize_cached.cuh",
        "= flags == 0 ? saved_scales[slot] : 127;",
    ),
    (
        "row/rmsnorm_quantize_cached_f32.cuh",
        "const uint32_t value = group_flags == 0 ? packed[slot] : 127u << 16;",
    ),
    (
        "row/rmsnorm_quantize_tiles.cuh",
        "else value.scale_code = 127;",
    ),
    (
        "row/ceil_cached.cuh",
        "const uint32_t value = flags == 0 ? packed[k] : 127u << 16;",
    ),
    (
        "row/ceil_parallel.cuh",
        "else value.scale_code = Format == 1 ? 0x38 : 127; // scale 1",
    ),
    (
        "sinkhorn_hc/finish_norm.cuh",
        "chop::row::rmsnorm_quantize_cached::task<H,Warps,FloorBits>(collapsed,weight,",
    ),
    (
        "sinkhorn_hc/finish_twin.cuh",
        "chop::row::rmsnorm_quantize_cached::task<H,Warps,Floor>(collapsed,weight,",
    ),
    (
        "sinkhorn_hc/completion.cuh",
        "finish_norm::task<M,Warps,Floor>(expanded,base,scale,pre_in,collapsed,",
    ),
    (
        "sinkhorn_hc/completion.cuh",
        "chop::row::rmsnorm_quantize_cached::task<M::kWidth,Warps,Floor>(collapsed,weight,",
    ),
    (
        "sinkhorn_hc/finish_split.cuh",
        "finish_twin::task<M,Warps,FloorBits,NormMutation<Mut>::value>(expanded,base,scale,pre_in,",
    ),
];

/// The audited engines by namespace, and the headers whose statements
/// above make their rejected rows zero.
const ENGINES: &[(&str, &[&str])] = &[
    (
        "chop::row::rmsnorm_quantize",
        &[
            "row/rmsnorm_quantize.cuh",
            "row/ceil_epilogue.cuh",
            "row/block_quantize.cuh",
        ],
    ),
    (
        "chop::row::rmsnorm_quantize_cached",
        &["row/rmsnorm_quantize_cached.cuh"],
    ),
    (
        "chop::row::rmsnorm_quantize_cached_f32",
        &["row/rmsnorm_quantize_cached_f32.cuh"],
    ),
    (
        "chop::row::rmsnorm_quantize_tiles",
        &["row/rmsnorm_quantize_tiles.cuh", "row/block_quantize.cuh"],
    ),
    ("chop::row::block_quantize_cached", &["row/ceil_cached.cuh"]),
    (
        "chop::row::block_quantize_parallel",
        &["row/ceil_parallel.cuh", "row/block_quantize.cuh"],
    ),
    (
        "chop::sinkhorn_hc::finish_norm",
        &[
            "sinkhorn_hc/finish_norm.cuh",
            "row/rmsnorm_quantize_cached.cuh",
        ],
    ),
    (
        "chop::sinkhorn_hc::finish_twin",
        &[
            "sinkhorn_hc/finish_twin.cuh",
            "row/rmsnorm_quantize_cached.cuh",
        ],
    ),
    (
        "chop::sinkhorn_hc::completion",
        &[
            "sinkhorn_hc/completion.cuh",
            "sinkhorn_hc/finish_norm.cuh",
            "row/rmsnorm_quantize_cached.cuh",
        ],
    ),
    (
        "chop::sinkhorn_hc::finish_split",
        &[
            "sinkhorn_hc/finish_split.cuh",
            "sinkhorn_hc/finish_twin.cuh",
            "row/rmsnorm_quantize_cached.cuh",
        ],
    ),
];

fn kernel_source(header: &str) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../crates/kernels-sm120/cuda")
        .join(header);
    std::fs::read_to_string(&root).unwrap_or_else(|e| panic!("{}: {e}", root.display()))
}

/// The entry bodies of the norm, ceil and HC boundary families, by symbol.
fn bodies() -> BTreeMap<String, String> {
    use crate::sm120 as s;
    let g = std::slice::from_ref(&FLASH);
    let mut specs = Vec::new();
    for family in [
        s::ceil_cached::cubins(g),
        s::ceil_parallel::cubins(g),
        s::norm_quantize::cubins(g),
        s::norm_quantize_cached::cubins(g),
        s::norm_quantize_tiles::cubins(g),
        s::norm_query_cached::cubins(g),
        s::hc_finish_norm::cubins(g),
    ] {
        specs.extend(family.unwrap().into_iter().map(|(_, spec)| spec));
    }
    specs.push(s::hc_completion::cubin().unwrap());
    specs.push(s::hc_finish_twin::cubin().unwrap());
    specs.extend(s::hc_cols::cubins().unwrap());
    specs
        .into_iter()
        .flat_map(|spec| spec.entries)
        .map(|e| (e.symbol, e.body))
        .collect()
}

/// The audited engines `body` calls: by full name, inside the engine's own
/// namespace, or by its task inside the parent namespace.
fn engines(body: &str) -> Vec<&'static str> {
    ENGINES
        .iter()
        .map(|(namespace, _)| *namespace)
        .filter(|ns| {
            let (parent, leaf) = ns.rsplit_once("::").unwrap();
            body.contains(&format!("{ns}::"))
                || body.contains(&format!("using namespace {ns};"))
                || body.contains(&format!("using namespace {parent};"))
                    && body.contains(&format!("{leaf}::task<"))
        })
        .collect()
}

/// WQ_a's, WKV's and WQ_b's status folds fail exactly the rows the
/// admission would only because their codes come from producers that store
/// rejected rows as zeros; WO_a's come from the sliced quantizer and WO_b's
/// from WO_a's merge. This fails if a status fold reads another producer
/// or an audited producer stops zeroing.
#[test]
fn status_folds_read_codes_only_from_zeroing_producers() {
    for (header, statement) in ZEROING {
        assert!(
            kernel_source(header).contains(statement),
            "{header} no longer stores rejected rows as zeros: {statement}"
        );
    }
    for (engine, headers) in ENGINES {
        for header in *headers {
            assert!(
                ZEROING.iter().any(|(h, _)| h == header),
                "{engine}: {header}"
            );
        }
    }
    let bodies = bodies();
    let mut producers = BTreeSet::new();
    for kv in KV_LAYOUTS {
        let [_, after] = journals(kv);
        let mut case = String::new();
        let mut first = 0;
        for (at, line) in after.iter().enumerate() {
            if let Some(name) = line.strip_prefix("case ") {
                case = name.to_owned();
                first = at;
                continue;
            }
            let Some(rows) = case
                .split_once(' ')
                .filter(|(kind, _)| *kind != "prefill")
                .and_then(|(_, rows)| rows.parse::<u32>().ok())
                .filter(|rows| fold_fused::ROWS.contains(rows))
            else {
                continue;
            };
            let Some(id) = PROJECTIONS
                .into_iter()
                .find(|id| line.starts_with("launch ") && symbol(line) == status_fold(id, rows))
            else {
                continue;
            };
            // The step's latest launch before the fold that binds its codes,
            // past the twin fold that reads them too (WQ_a beside WKV).
            let codes = spans(line)[0].clone();
            let overlaps =
                |s: &Span| s.0 == codes.0 && s.1 < codes.1 + codes.2 && codes.1 < s.1 + s.2;
            let producer = after[first..at]
                .iter()
                .rev()
                .filter(|l| l.starts_with("launch ") && spans(l).iter().any(overlaps))
                .find(|l| {
                    !["attention.q-a", "attention.kv"]
                        .into_iter()
                        .any(|twin| symbol(l) == status_fold(twin, rows))
                })
                .unwrap_or_else(|| {
                    panic!("{id} at {case} reads codes no launch of the step wrote")
                });
            let p = symbol(producer);
            match id {
                "attention.o-a" => {
                    assert_eq!(p, fold_fused::sliced(rows).unwrap().symbol, "{case}")
                }
                "attention.o-b" => assert_eq!(p, status_fold("attention.o-a", rows), "{case}"),
                _ => {
                    let body = bodies.get(p).unwrap_or_else(|| {
                        panic!(
                            "{id} at {case} reads codes of {p}, whose rejected rows are not audited"
                        )
                    });
                    assert!(
                        !engines(body).is_empty(),
                        "{id}'s producer {p} calls no audited engine: {body}"
                    );
                    producers.insert((id, p.to_owned()));
                }
            }
        }
    }
    for id in ["attention.q-a", "attention.kv", "attention.q-b"] {
        assert!(
            producers.iter().any(|(of, _)| *of == id),
            "{id}: {producers:?}"
        );
    }
    println!("status fold producers: {producers:#?}");
}

#[test]
fn status_folds_keep_the_workspace_regions() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let kernels = crate::Kernels::new(&shape).unwrap();
    let experts = model_deepseek41_contract::moe::Experts {
        count: FLASH.shape.moe.experts,
        top_k: FLASH.shape.moe.top_k,
    };
    for kv in KV_LAYOUTS {
        let sizes = || {
            let ops = crate::ops::Ops::prepare_set(&kernels, &shape, kv.storage, experts);
            (ops.workspace(), ops.shared_workspace())
        };
        let before = fold_fused::with_reference(sizes);
        assert_eq!(sizes(), before, "{kv:?}");
    }
}
