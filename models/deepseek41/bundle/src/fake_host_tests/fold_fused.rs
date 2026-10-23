//! Audit the status folds (`sm120::fold_fused`): in the 1-, 2-, 4-, 8-, 16-
//! and 32-row decode and verify buckets of every backbone layer each
//! replaces its projection's admission and fold (WO_a also WO_b's input
//! quantizer) on the same operands, the sliced quantizer replaces WO_a's
//! input quantizer on its spans, and every other launch is retained. Every
//! fold reads the plan's packed codes, so without the status folds the
//! backend's decode folds serve every bucket (the cluster and row-local
//! folds read row-major codes and serve no instance): a prefetch or ticket
//! fold's initialize and fold, a split decode or row-reuse fold, or one
//! stream or direct fold, which binds the layer's weights as its packed
//! bank and whose state the status fold drops (WO_a's and WO_b's also take
//! the block flags and slabs of the 16-row fold). The one-row decode step
//! keeps the dual fold for WQ_a and WKV. Then the producers whose
//! rejected-row zeroing the status folds rely on.
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
    for count in [1, 2, 4, 8, 16, 32] {
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
        // Four rows per item: verify 32 on slots 0..8, verify 16 on slots
        // 8..12, verify 8 on slots 12..14 and verify 4 on slot 14; then
        // verify 2, two rows on slot 15.
        for (first, count, per) in [
            (0_u64, 8_u64, 4_u32),
            (8, 4, 4),
            (12, 2, 4),
            (14, 1, 4),
            (15, 1, 2),
        ] {
            let from = host.device.journal.borrow().len();
            let tokens = vec![7; per as usize];
            let items = (first..first + count)
                .map(|i| (i + 1, i, ends[i as usize], tokens.as_slice()))
                .collect::<Vec<_>>();
            instance
                .execute(&verify(&items), &mut Views::default())
                .unwrap();
            instance
                .commit_accepted(
                    &(first + 1..=first + count)
                        .map(|i| kept(i, per))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            records.push(format!("case verify {}", count * u64::from(per)));
            records.extend_from_slice(&host.device.journal.borrow()[from..]);
            for end in &mut ends[first as usize..(first + count) as usize] {
                *end += u64::from(per);
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

/// The projection's fold shape: N, K and groups.
fn shape(id: &str) -> (u32, u32, u32) {
    let q = if id == "attention.q-b" {
        fold_fused::wide(16).unwrap().base.query
    } else {
        fold_fused::fused(id, 16).unwrap().cluster.base.query
    };
    (q.n, q.k, q.groups)
}

/// What a served launch of a projection is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// An admission (a prefetch or ticket fold's initialize, or a split
    /// decode or row-reuse fold's admission).
    Admit,
    /// A launch of the fold after its admission.
    Fold,
    /// A stream or direct fold, which admits rows itself and binds the
    /// layer's weights as its packed bank with its own state: `states` spans
    /// of it.
    Packed { states: usize },
}

/// The projection whose served launch `line` is without the status folds,
/// and what it is. Launches that read no codes keep their names; the rest
/// take their packed entries.
fn served(line: &str) -> Option<(&'static str, Kind)> {
    static SERVED: std::sync::OnceLock<BTreeMap<String, (&'static str, Kind)>> =
        std::sync::OnceLock::new();
    SERVED
        .get_or_init(served_symbols)
        .get(symbol(line))
        .copied()
}

/// Every symbol `served` names, built once: each catalog is a full
/// registration, and the audit asks for thousands of lines.
fn served_symbols() -> BTreeMap<String, (&'static str, Kind)> {
    let prefetch = crate::sm120::mxfp8_fold::prefetch_entries(&FLASH, 4).unwrap();
    let ticket = crate::sm120::mxfp8_fold::ticket_entries(&FLASH, 4).unwrap();
    let split = crate::sm120::mxfp8_fold::decode_entries(&FLASH, 4).unwrap();
    let reuse = crate::sm120::projection_reuse::entries().unwrap();
    let streams = crate::sm120::mxfp8_fold_stream::catalog(&FLASH).unwrap();
    let directs = crate::sm120::mxfp8_fold_direct_entries::catalog().unwrap();
    let mut map = BTreeMap::new();
    for id in PROJECTIONS {
        let shape = shape(id);
        let matches = |q: kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery, paired: bool| {
            !paired && (q.n, q.k, q.groups) == shape
        };
        let mut symbols: Vec<(String, Kind)> = Vec::new();
        for e in streams
            .iter()
            .filter(|e| matches(e.ticket.query, e.ticket.paired))
        {
            symbols.push((e.ticket.symbol.clone(), Kind::Packed { states: 2 }));
        }
        for e in directs
            .iter()
            .filter(|e| matches(e.packed.ticket.query, e.packed.ticket.paired))
        {
            symbols.push((e.symbol.clone(), Kind::Packed { states: 1 }));
        }
        for e in prefetch
            .iter()
            .chain(&ticket)
            .filter(|e| matches(e.query, e.paired))
        {
            let [initialize, fold] = e.symbols();
            symbols.extend([(initialize, Kind::Admit), (fold, Kind::Fold)]);
        }
        for e in split.iter().filter(|e| matches(e.query, e.paired)) {
            let [admit, partials, merge] = e.symbols();
            symbols.extend([
                (admit, Kind::Admit),
                (partials, Kind::Fold),
                (merge, Kind::Fold),
            ]);
        }
        for e in reuse
            .iter()
            .filter(|e| matches(e.ticket.query, e.ticket.paired))
        {
            symbols.extend([
                (format!("{}_admit_v1", e.symbol()), Kind::Admit),
                (e.symbol(), Kind::Fold),
            ]);
        }
        for (symbol, kind) in symbols {
            let packed = kernels_sm120::gemm::mxfp8_fold_packed::symbol(&symbol);
            for name in [symbol, packed] {
                map.entry(name).or_insert((id, kind));
            }
        }
    }
    map
}

/// A projection's status fold at a bucket.
struct Fold {
    id: &'static str,
    rows: u32,
    status: String,
    workspace: Option<u64>,
}

/// Every projection's status fold at every bucket, built once: each entry
/// builds fold catalogs, which the audits would otherwise rebuild for every
/// changed line.
fn folds() -> &'static [Fold] {
    static TABLE: std::sync::OnceLock<Vec<Fold>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        PROJECTIONS
            .into_iter()
            .flat_map(|id| {
                fold_fused::ROWS.into_iter().map(move |rows| Fold {
                    id,
                    rows,
                    status: kernels_sm120::gemm::mxfp8_fold_packed::symbol(
                        &if id == "attention.q-b" {
                            fold_fused::wide(rows).unwrap().symbol()
                        } else {
                            fold_fused::fused(id, rows).unwrap().symbol()
                        },
                    ),
                    workspace: fold_fused::fused(id, rows).map(|e| e.workspace()),
                })
            })
            .collect()
    })
}

/// The status fold of `id` at `rows` rows.
fn status_entry(id: &str, rows: u32) -> &'static Fold {
    folds()
        .iter()
        .find(|f| f.id == id && f.rows == rows)
        .unwrap_or_else(|| panic!("no status fold of {id} at {rows} rows"))
}

/// The status fold symbol of `id` at `rows` rows: its packed entry.
fn status_fold(id: &str, rows: u32) -> String {
    status_entry(id, rows).status.clone()
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
    // Each projection's status folds at 16 rows, in launch order: each case
    // launches the same projections of the same layers and ranks in the
    // same order, and the 16-row folds are audited against the reference's
    // admissions and folds, which bind their layer weights.
    let folds16: Vec<_> = PROJECTIONS
        .into_iter()
        .map(|id| (id, status_fold(id, 16)))
        .collect();
    let mut in_decode_16 = false;
    let mut templates = BTreeMap::<&str, Vec<Vec<Span>>>::new();
    for l in after {
        if let Some(name) = l.strip_prefix("case ") {
            in_decode_16 = name == "decode 16";
        }
        if !in_decode_16 || !l.starts_with("launch ") {
            continue;
        }
        if let Some((id, _)) = folds16.iter().find(|(_, s)| symbol(l) == s) {
            templates.entry(id).or_default().push(spans(l));
        }
    }
    let mut packed_folds = BTreeMap::<&str, usize>::new();
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
                packed_folds.clear();
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
        if let Some((id, Kind::Packed { states })) = served(a) {
            // A stream or direct fold, which admits rows itself and binds the
            // layer's weights as its packed bank, becomes the status fold
            // over those weights, WO_a's also in place of WO_b's input
            // quantizer, whose codes and scales it writes: the fold's state
            // goes; for WO_a and WO_b the block flags and the 16-row fold's
            // slabs come in; every other operand is kept.
            let old = spans(a);
            if old.len() != 7 + states {
                return Err(format!("{id} at {case}: an unknown packed fold: {a}"));
            }
            let mut kept: Vec<Span> = old
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != 2 && !(5..5 + states).contains(k))
                .map(|(_, span)| span.clone())
                .collect();
            let mut consumed = 1;
            if id == "attention.o-a" {
                let quantizer = before.get(i + 1).filter(|q| symbol(q) == WO_B_INPUT);
                let q = spans(quantizer.ok_or("WO_a without WO_b's input quantizer")?);
                if q[0] != old[4] {
                    return Err(format!("{id} at {case}: WO_b's input is not WO_a's output"));
                }
                kept.extend([q[1].clone(), q[2].clone()]);
                consumed = 2;
            }
            if symbol(b) != status_fold(id, rows)
                || old_journal.captured[i..i + consumed]
                    .iter()
                    .any(|&c| c != new_journal.captured[j])
            {
                return Err(format!("{id} at {case} is not its status fold: {b}"));
            }
            let k = packed_folds.entry(id).or_default();
            let template = templates
                .get(id)
                .and_then(|t| t.get(*k))
                .ok_or("more packed folds than 16-row status folds")?;
            *k += 1;
            let wo = matches!(id, "attention.o-a" | "attention.o-b");
            let weights = &template[2];
            if old[2] != *weights {
                return Err(format!(
                    "{id} at {case} does not bind the layer's weights as its packed bank: {a}"
                ));
            }
            let slab = template
                .iter()
                .position(|s| s.0.starts_with("flags"))
                .map(|f| &template[f + 1]);
            let new = spans(b);
            if new.get(2) != Some(weights) {
                return Err(format!(
                    "{id} at {case} does not bind the layer's weights: {b}"
                ));
            }
            for span in &new {
                let added =
                    span == weights || wo && (span.0.starts_with("flags") || Some(span) == slab);
                if !(kept.contains(span) || added) {
                    return Err(format!("{id} at {case} binds another operand: {span:?}"));
                }
            }
            if let Some(span) = kept.iter().find(|s| !new.contains(s)) {
                return Err(format!("{id} at {case} drops an operand: {span:?}"));
            }
            let flags = new.iter().filter(|s| s.0.starts_with("flags")).count();
            if flags != usize::from(wo) || wo != slab.is_some_and(|s| new.contains(s)) {
                return Err(format!(
                    "{id} at {case} binds {flags} block flags or no slabs"
                ));
            }
            changed.push((case.clone(), id.to_owned()));
            if new_journal.captured[j] {
                *removed.entry(case.clone()).or_default() += consumed as i64 - 1;
            }
            i += consumed;
            j += 1;
            continue;
        }
        let Some((id, Kind::Admit)) = served(a) else {
            return Err(format!("{case} changed an unrelated operation: {a}\n{b}"));
        };
        // The admission's fold: one launch, or a split fold's partials and
        // merge.
        let folds = before[i + 1..]
            .iter()
            .take_while(|l| served(l) == Some((id, Kind::Fold)))
            .count();
        if folds == 0 || symbol(b) != status_fold(id, rows) {
            return Err(format!(
                "{id} at {case} is not admission and fold to its status fold: {b}"
            ));
        }
        let fold: Vec<Span> = before[i + 1..i + 1 + folds]
            .iter()
            .flat_map(|l| spans(l))
            .collect();
        // The admission binds codes, scales, the served fold's workspace,
        // the live rows and the status.
        let admit = spans(a);
        let workspace = admit[2].clone();
        let status = admit[4].clone();
        let mut old: Vec<Span> = admit.iter().chain(&fold).cloned().collect();
        let mut consumed = 1 + folds;
        if id == "attention.o-a" {
            // WO_a's merge quantizes WO_b's input in place of its quantizer.
            let quantizer = before.get(i + consumed).filter(|q| symbol(q) == WO_B_INPUT);
            old.extend(spans(
                quantizer.ok_or("WO_a without WO_b's input quantizer")?,
            ));
            consumed += 1;
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
        // A cluster status fold takes its own slabs at the served fold's
        // workspace offset.
        let slab_bytes = status_entry(id, rows).workspace;
        for span in &new {
            let slabs = (&span.0, span.1) == (&workspace.0, workspace.1)
                && Some(span.2) == slab_bytes
                && id != "attention.q-b";
            let kept = old.contains(span) && *span != workspace;
            if !(kept || slabs || wo && span.0.starts_with("flags")) {
                return Err(format!("{id} at {case} binds another operand: {span:?}"));
            }
        }
        // Every operand but the served fold's workspace stays bound.
        for span in fold.iter().chain([&status]) {
            if span.0 != workspace.0 && !new.contains(span) {
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

/// The serving layout of `storage`. Fails if the layouts change, so none
/// drops out of the split audits.
fn layout(storage: KvStorage) -> KvLayout {
    assert_eq!(
        KV_LAYOUTS.map(|kv| kv.storage),
        [KvStorage::Native, KvStorage::Bf16]
    );
    KV_LAYOUTS
        .into_iter()
        .find(|kv| kv.storage == storage)
        .unwrap()
}

#[test]
fn status_folds_change_only_native_decode_and_verify_projections() {
    check_projection_changes(layout(KvStorage::Native));
}

#[test]
fn status_folds_change_only_bf16_decode_projections() {
    check_projection_changes(layout(KvStorage::Bf16));
}

fn check_projection_changes(kv: KvLayout) {
    let [before, after] = journals(kv);
    let changed = audit(&before, &after).unwrap();
    let mut cases = vec![
        "decode 1",
        "decode 2",
        "decode 4",
        "decode 8",
        "decode 16",
        "decode 32",
    ];
    if kv.storage == KvStorage::Native {
        cases.extend(["verify 32", "verify 16", "verify 8", "verify 4", "verify 2"]);
    }
    // Every backbone layer on each of the four ranks, not the drafter's;
    // the one-row decode step keeps the dual fold for WQ_a and WKV.
    let mut expected = Vec::new();
    for case in cases {
        for piece in PROJECTIONS
            .into_iter()
            .chain(["attention.o-a input"])
            .filter(|p| case != "decode 1" || !matches!(*p, "attention.q-a" | "attention.kv"))
        {
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
    let admit = before
        .iter()
        .find(|l| served(l) == Some(("attention.o-a", Kind::Admit)))
        .unwrap();
    let mut bad = after.clone();
    bad.insert(at, admit.clone());
    assert!(audit(&before, &bad).is_err());
    let mut bad = after.clone();
    bad[at] = bad[at].replacen("raw: flags", "raw: buffer", 1);
    assert!(audit(&before, &bad).is_err());
    // At one row, a WO_b status fold in place of the direct fold over
    // other weights, or without WO_a's block flags.
    let one = after.iter().position(|l| l == "case decode 1").unwrap();
    let wo_b = one
        + after[one..]
            .iter()
            .position(|l| symbol(l) == status_fold("attention.o-b", 1))
            .unwrap();
    let weights = spans(&after[wo_b])[2].clone();
    let mut bad = after.clone();
    bad[wo_b] = bad[wo_b].replacen(
        &format!("offset: {}, length: {}", weights.1, weights.2),
        &format!("offset: {}, length: {}", weights.1 + 64, weights.2),
        1,
    );
    assert!(audit(&before, &bad).is_err());
    let mut bad = after.clone();
    bad[wo_b] = bad[wo_b].replacen("raw: flags", "raw: buffer", 1);
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
fn status_folds_read_native_codes_only_from_zeroing_producers() {
    check_zeroing_producers(layout(KvStorage::Native));
}

#[test]
fn status_folds_read_bf16_codes_only_from_zeroing_producers() {
    check_zeroing_producers(layout(KvStorage::Bf16));
}

fn check_zeroing_producers(kv: KvLayout) {
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
    let folds: BTreeMap<u32, Vec<(&str, String)>> = fold_fused::ROWS
        .iter()
        .map(|&rows| {
            let folds = PROJECTIONS.map(|id| (id, status_fold(id, rows)));
            (rows, folds.to_vec())
        })
        .collect();
    let mut producers = BTreeSet::new();
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
        let Some(&(id, _)) = folds[&rows]
            .iter()
            .find(|(_, fold)| line.starts_with("launch ") && symbol(line) == fold)
        else {
            continue;
        };
        // The step's latest launch before the fold that binds its codes,
        // past the twin fold that reads them too (WQ_a beside WKV).
        let codes = spans(line)[0].clone();
        let overlaps = |s: &Span| s.0 == codes.0 && s.1 < codes.1 + codes.2 && codes.1 < s.1 + s.2;
        let producer = after[first..at]
            .iter()
            .rev()
            .filter(|l| l.starts_with("launch ") && spans(l).iter().any(overlaps))
            .find(|l| {
                !folds[&rows].iter().any(|(twin, fold)| {
                    matches!(*twin, "attention.q-a" | "attention.kv") && symbol(l) == fold
                })
            })
            .unwrap_or_else(|| panic!("{id} at {case} reads codes no launch of the step wrote"));
        let p = symbol(producer);
        match id {
            "attention.o-a" => {
                assert_eq!(p, fold_fused::sliced(rows).unwrap().symbol, "{case}")
            }
            "attention.o-b" => assert_eq!(p, status_fold("attention.o-a", rows), "{case}"),
            _ => {
                let body = bodies.get(p).unwrap_or_else(|| {
                    panic!("{id} at {case} reads codes of {p}, whose rejected rows are not audited")
                });
                assert!(
                    !engines(body).is_empty(),
                    "{id}'s producer {p} calls no audited engine: {body}"
                );
                producers.insert((id, p.to_owned()));
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
