//! The Sinkhorn mHC comm epilogue model option
//! (`runtime.sinkhorn_comm_epilogue`) on an instance: the same requests on
//! two hosts, the option false on one and true on the other, issue the same launches in the same order on every
//! stream, except that each sublayer boundary's expand, next mix-collapse
//! and next input norm (in the decode and verify graphs, the next mixer
//! fused with its norm) become the widening of the all-reduce's rounded
//! sums and one split epilogue launch after the same all-reduce launch (the
//! low-latency or a phased BF16 entry; a MoE sublayer's rounded all-reduce
//! of F32 partials and the launches after it stay as they are). Only the
//! switched instance loads the epilogue's cubin, so
//! with the switch off every pinned journal stays as it was; the switched
//! instance's later buffer handles shift by that load, so the launches are
//! compared under one consistent renaming of handles per stream. The
//! binding of each epilogue launch is `crate::comm_epilogue_tests`'.

use std::collections::BTreeMap;

use super::*;
use model_glm5next_contract::sm120::engines::{self as selected, ModelRow};

const FUSED_ROWS: &str = "chop_sinkhorn_epilogue_glm5next_rows_split_tree_v1";
/// A boundary in the journal.
const EPILOGUE: &str = "<comm epilogue>";

/// The launch journal lines from `from` on, by stream in issue order.
fn by_stream(host: &FakeHost, from: usize) -> BTreeMap<u32, Vec<String>> {
    let mut streams = BTreeMap::<u32, Vec<String>>::new();
    for line in &host.device.journal.borrow()[from..] {
        let mut words = line.splitn(3, ' ');
        if words.next() != Some("launch") {
            continue;
        }
        let stream = words.next().unwrap().parse().unwrap();
        streams.entry(stream).or_default().push(line.clone());
    }
    streams
}

fn symbol(line: &str) -> &str {
    line.split(' ').nth(2).unwrap()
}

/// `lines` with each boundary's row launches after its BF16 all-reduce (the
/// low-latency or a phased entry) replaced by [`EPILOGUE`]: the expand,
/// split mix-collapse and norm, or in the decode and verify graphs the
/// expand and fused mixer and norm. Also the launches the boundaries took
/// beyond one each.
fn unfused_boundaries(lines: &[String]) -> (Vec<String>, usize) {
    let sinkhorn = |op: &str| format!("chop_sinkhorn_hc_glm5next_{op}_v1");
    let forms = [
        vec![
            sinkhorn("expand"),
            sinkhorn("mix_collapse_split"),
            selected::rmsnorm_symbol(4096),
        ],
        vec![sinkhorn("expand"), sinkhorn("mix_collapse_norm_tree")],
    ];
    let rows_at = |at: usize| {
        forms.iter().find(|rows| {
            lines.len() >= at + rows.len()
                && rows
                    .iter()
                    .zip(&lines[at..])
                    .all(|(row, line)| symbol(line) == row)
        })
    };
    let bf16 =
        |reduce: &str| reduce == selected::LL_SYMBOL || selected::PHASED_SYMBOLS.contains(&reduce);
    let rounded = selected::rounded_phased_admission_plans().unwrap();
    let rounded = |reduce: &str| {
        reduce == selected::LL_ROUND_SYMBOL || rounded.iter().any(|(symbol, _)| *symbol == reduce)
    };
    let mut out = Vec::new();
    let mut extra = 0;
    let mut at = 0;
    while at < lines.len() {
        let rows = rows_at(at);
        let reduce = rows.map(|_| symbol(&lines[at - 1]));
        if let Some(rows) = rows.filter(|_| bf16(reduce.unwrap())) {
            out.push(EPILOGUE.to_owned());
            at += rows.len();
            extra += rows.len() - 1;
        } else {
            assert!(reduce.is_none_or(rounded), "{reduce:?}");
            out.push(lines[at].clone());
            at += 1;
        }
    }
    (out, extra)
}

/// `lines` with each boundary's widening and epilogue replaced by
/// [`EPILOGUE`].
fn fused_boundaries(lines: &[String]) -> Vec<String> {
    let widen = ModelRow::Widen.symbol();
    let mut out = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        match symbol(line) {
            FUSED_ROWS => {
                assert_eq!(symbol(&lines[at - 1]), widen);
                out.pop();
                out.push(EPILOGUE.to_owned());
            }
            _ => out.push(line.clone()),
        }
    }
    out
}

/// `lines` with every buffer handle (`raw: N`) renamed by the order of its
/// first appearance: two sequences that are equal under one consistent
/// renaming become equal. The switched instance's extra module load shifts
/// the handles create allocates after it.
fn relabel(lines: &[String]) -> Vec<String> {
    let mut names = BTreeMap::<String, usize>::new();
    lines
        .iter()
        .map(|line| {
            let mut out = String::new();
            let mut rest = line.as_str();
            while let Some(at) = rest.find("raw: ") {
                let (head, tail) = rest.split_at(at + "raw: ".len());
                out.push_str(head);
                let digits = tail
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(tail.len());
                let next = names.len();
                let name = *names.entry(tail[..digits].to_owned()).or_insert(next);
                out.push_str(&format!("#{name}"));
                rest = &tail[digits..];
            }
            out.push_str(rest);
            out
        })
        .collect()
}

/// Runs `requests` on a fresh one-slot instance, with the comm epilogue
/// when `fused`, and returns each request's launches by stream.
fn run(fused: bool, requests: &[Vec<u8>]) -> Vec<BTreeMap<u32, Vec<String>>> {
    let host = FakeHost::new((0, 0));
    host.items.set(1);
    let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
    let options = crate::config::tests::options_comm_epilogue(2048, &fused.to_string());
    let record = crate::config::tests::record_naming(1, &options, host.format, &roles);
    // One stream per rank: the shared gate/up overlap off, so each stream's
    // launches are the whole program's (`fake_host_tests::shared_overlap`
    // proves the overlap with the epilogue on).
    let mut instance =
        crate::shared_overlap::with_settings_in_tests(crate::shared_overlap::Settings::OFF, || {
            api().create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        })
        .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    let logged = host
        .logs
        .borrow()
        .iter()
        .any(|line| line.contains("runtime.sinkhorn_comm_epilogue=true"));
    assert_eq!(logged, fused);
    // Create loads the epilogue's cubin, and resolves its entries, with the
    // switch alone.
    let resolved = host
        .device
        .journal
        .borrow()
        .iter()
        .filter(|line| line.starts_with("resolve ") && line.ends_with(FUSED_ROWS))
        .count();
    assert_eq!(resolved, if fused { 4 } else { 0 });
    let mut out = Vec::new();
    for request in requests {
        let start = host.device.journal.borrow().len();
        instance
            .execute(request, &mut NoRows::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        out.push(by_stream(&host, start));
    }
    instance.destroy().unwrap();
    out
}

/// An eager prefill of 40 rows and one of 9 rows (the phased two-shot)
/// and the first decode step (the low-latency entry, captured): every
/// stream issues the unfused launches but for the 48 boundaries that reduce
/// BF16 partials, each the widening and one epilogue launch after the same
/// all-reduce launch, so a
/// prefill step saves one launch per boundary and stream (decode, which
/// already fuses the mixer and its norm, none).
#[test]
fn the_switch_replaces_only_the_boundary_launches() {
    let requests = [
        prompt_chunk(7, 0, 40, false),
        prompt_chunk(7, 40, 9, true),
        step(&[(7, 0, 49, &[1])]),
    ];
    let unfused = run(false, &requests);
    let fused = run(true, &requests);
    for (index, (unfused, fused)) in unfused.iter().zip(&fused).enumerate() {
        assert_eq!(unfused.len(), 4, "request {index}: one stream per rank");
        assert_eq!(
            unfused.keys().collect::<Vec<_>>(),
            fused.keys().collect::<Vec<_>>()
        );
        for (stream, lines) in unfused {
            let (marked, extra) = unfused_boundaries(lines);
            let boundaries = marked.iter().filter(|line| *line == EPILOGUE).count();
            assert_eq!(boundaries, 48, "request {index}, stream {stream}");
            assert_eq!(
                relabel(&marked),
                relabel(&fused_boundaries(&fused[stream])),
                "request {index}, stream {stream}"
            );
            let per_boundary = if index < 2 { 1 } else { 0 };
            assert_eq!(
                extra,
                (per_boundary + 1) * boundaries,
                "request {index}, stream {stream}"
            );
            assert_eq!(
                lines.len() - fused[stream].len(),
                per_boundary * boundaries,
                "request {index}, stream {stream}"
            );
        }
    }
}

/// An explicit false option creates and steps exactly as a record without
/// the option (the journals before the option existed).
#[test]
fn the_option_off_issues_what_no_option_issues() {
    let journal = |options: Option<Vec<u8>>| {
        let host = FakeHost::new((0, 0));
        let mut instance = match options {
            None => create_one_slot(&host),
            Some(options) => {
                host.items.set(1);
                let roles = [crate::projections::COMPONENT_ROLE.as_bytes().to_vec()];
                let record = crate::config::tests::record_naming(1, &options, host.format, &roles);
                api()
                    .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
                    .unwrap()
            }
        };
        instance
            .execute(&prompt_chunk(7, 0, 9, true), &mut NoRows::default())
            .unwrap();
        instance.commit().unwrap();
        instance.destroy().unwrap();
        host.device.journal.borrow().clone()
    };
    let off = crate::config::tests::options_comm_epilogue(2048, "false");
    assert_eq!(journal(Some(off)), journal(None));
}
