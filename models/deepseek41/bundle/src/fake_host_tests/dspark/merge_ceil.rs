//! The `attention_merge_ceil` candidate: wherever WO_a runs as a status
//! fold, the target's layers and the drafter's measured blocks, at decode
//! and verify widths, each attention merge launches its quantizing twin
//! with the merge's grid and arguments followed by the codes, scales,
//! status and status row offset of WO_a's sliced quantizer, which no longer
//! launches. Every other launch keeps its symbol, grid and arguments, and
//! steps above the status folds' 64 rows keep both launches.
use super::*;
use model_deepseek41_contract::execution_options::{
    ATTENTION_MERGE_CEIL, DSPARK_GRAPH_HANDLES_PER_RANK,
};

const SLICED: &str = "chop_deepseek41_dense_ceil_w8192_sliced_";
const MERGE: &str = "_merge_v1";
const TWIN: &str = "_merge_ceil_v1";

/// `served`'s launch lines with each sliced quantizer folded into the
/// merge before it: the merge's symbol as its twin's, and the quantizer's
/// codes, scales, status and status row offset appended to its arguments.
/// Returns the lines and the number of twins.
fn twinned(served: &[String]) -> (Vec<String>, usize) {
    use super::super::query_window::{arguments, symbol};
    let mut lines: Vec<String> = Vec::with_capacity(served.len());
    let mut merge = None;
    let mut twins = 0;
    for line in served {
        let name = symbol(line);
        if name.ends_with(MERGE) {
            merge = Some(lines.len());
        } else if name.starts_with(SLICED) {
            // Input, codes, scales, live rows, status, status row offset.
            let quantizer = arguments(line);
            assert_eq!(quantizer.len(), 6, "{line}");
            let at = merge.take().expect("a merge before each sliced quantizer");
            let old = &lines[at];
            let mut items = arguments(old);
            let head = &old[..old.len() - items.join(", ").len() - 2];
            items.extend([1, 2, 4, 5].map(|i| quantizer[i].clone()));
            let merged = symbol(old);
            lines[at] = format!(
                "{}[{}]",
                head.replacen(merged, &merged.replace(MERGE, TWIN), 1),
                items.join(", ")
            );
            twins += 1;
            continue;
        }
        lines.push(line.clone());
    }
    (lines, twins)
}

/// `lines` with each stream and buffer handle renamed by its first
/// appearance: the instance with the option loads more functions, which
/// shifts the handles of streams and buffers created after them.
pub(super) fn renamed(lines: &[String]) -> Vec<String> {
    const RAW: &str = "raw: ";
    let mut names = std::collections::HashMap::new();
    let mut streams = std::collections::HashMap::new();
    lines
        .iter()
        .map(|line| {
            let mut words = line.splitn(3, ' ');
            let (kind, stream, line) = (
                words.next().unwrap(),
                words.next().unwrap(),
                words.next().unwrap(),
            );
            let next = streams.len();
            let stream = *streams.entry(stream.to_owned()).or_insert(next);
            let mut out = format!("{kind} #{stream} ");
            let mut rest = line;
            while let Some(at) = rest.find(RAW) {
                let digits = rest[at + RAW.len()..]
                    .bytes()
                    .take_while(u8::is_ascii_digit)
                    .count();
                let raw = &rest[at + RAW.len()..at + RAW.len() + digits];
                let next = names.len();
                let name = *names.entry(raw.to_owned()).or_insert(next);
                out.push_str(&rest[..at]);
                out.push_str(&format!("{RAW}#{name}"));
                rest = &rest[at + RAW.len() + digits..];
            }
            out.push_str(rest);
            out
        })
        .collect()
}

/// Asserts `actual` is `expected` up to buffer handle names, naming the
/// first launch that differs.
pub(super) fn same(expected: &[String], actual: &[String], case: &str) {
    let (expected, actual) = (renamed(expected), renamed(actual));
    if let Some(at) = (0..expected.len().min(actual.len())).find(|&i| expected[i] != actual[i]) {
        panic!(
            "{case}: launch {at} differs\nexpected {}\nactual   {}",
            expected[at], actual[at]
        );
    }
    assert_eq!(expected.len(), actual.len(), "{case}");
}

fn count(lines: &[String], suffix: &str) -> usize {
    use super::super::query_window::symbol;
    lines.iter().filter(|l| symbol(l).ends_with(suffix)).count()
}

#[test]
fn quantizing_merges_replace_each_sliced_quantizer_at_verify_widths() {
    let create = |enabled: bool| {
        create_with_options(
            Shape::new(&FLASH, Ranks::TP4).unwrap(),
            16,
            128,
            &[
                (DSPARK_GRAPH_HANDLES_PER_RANK, 0.into()),
                (ATTENTION_MERGE_CEIL, enabled.into()),
            ],
        )
    };
    let (served_host, mut served) = create(false);
    let (fused_host, mut fused) = create(true);
    for (host, instance) in [(&served_host, &mut served), (&fused_host, &mut fused)] {
        for slot in 0..16 {
            execute(
                instance,
                host,
                &super::super::prompt(slot + 1, slot, 0, 3, true),
            );
            instance.commit().unwrap();
        }
    }
    let launches = |step: &[String]| -> Vec<String> {
        step.iter()
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect()
    };
    let mut drafted = 0;
    for (items, rows) in [
        (1_u32, 6_u32),
        (2, 12),
        (8, 32),
        (16, 33),
        (16, 64),
        (16, 65),
    ] {
        let (a, step_a, accepted_a) =
            super::verify_rows::verify_step(&mut served, &served_host, items, rows);
        let (b, step_b, accepted_b) =
            super::verify_rows::verify_step(&mut fused, &fused_host, items, rows);
        let (all_a, all_b) = (launches(&step_a), launches(&step_b));
        let (expected, twins) = twinned(&all_a);
        same(&expected, &all_b, &format!("{rows} rows"));
        assert_eq!(count(&all_a, TWIN), 0);
        // The target's 40 layers on each of the 4 ranks fold WO_a through
        // 64 rows; the drafter's blocks where their rows are measured.
        let (target, _) = twinned(&a);
        same(&target, &b, &format!("{rows} target rows"));
        let target_twins = count(&b, TWIN);
        assert_eq!(
            target_twins,
            if rows <= 64 { 40 * 4 } else { 0 },
            "{rows} rows"
        );
        drafted += twins - target_twins;
        served.commit_accepted(&accepted_a).unwrap();
        fused.commit_accepted(&accepted_b).unwrap();
    }
    assert!(drafted > 0);
    for (host, instance) in [(&served_host, &mut served), (&fused_host, &mut fused)] {
        for request in 1..=16 {
            instance.retire_request(request).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}

/// Autoregressive decode steps of 1, 2 and 8 rows, and the cubin loads:
/// only an instance with the option loads the quantizing merges.
#[test]
fn quantizing_merges_replace_each_sliced_quantizer_at_decode() {
    let run = |enabled: bool| {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        let mut instance =
            super::super::create_with_options(&host, 8, &[(ATTENTION_MERGE_CEIL, enabled.into())]);
        let loaded = host
            .device
            .module_cubins
            .borrow()
            .values()
            .any(|name| *name == crate::sm120::latent_merge_ceil::CUBIN);
        assert_eq!(loaded, enabled);
        for slot in 0..8 {
            execute(
                &mut instance,
                &host,
                &super::super::prompt(slot + 1, slot, 0, 1, true),
            );
            instance.commit().unwrap();
        }
        let mut ends = [1_u64; 8];
        let mut steps = Vec::new();
        for rows in [1_u32, 2, 8] {
            let from = host.device.journal.borrow().len();
            let items = (0..rows)
                .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
                .collect::<Vec<_>>();
            execute(&mut instance, &host, &super::super::decode(&items));
            instance.commit().unwrap();
            steps.push(
                host.device.journal.borrow()[from..]
                    .iter()
                    .filter(|l| l.starts_with("launch "))
                    .cloned()
                    .collect::<Vec<_>>(),
            );
            for end in &mut ends[..rows as usize] {
                *end += 1;
            }
        }
        for id in 1..=8 {
            instance.retire_request(id).unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        steps
    };
    let (served, fused) = (run(false), run(true));
    for (rows, (a, b)) in [1, 2, 8].into_iter().zip(served.iter().zip(&fused)) {
        let (expected, twins) = twinned(a);
        same(&expected, b, &format!("{rows} decode rows"));
        assert_eq!(twins, 40 * 4, "{rows} rows");
        assert_eq!(count(b, TWIN), twins);
    }
}
