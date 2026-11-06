//! The `prefill_norm_ceil_rows` candidate: a prefill step of 33 to 8192
//! rows runs each sublayer input norm as the fused prefill norm-ceil, in
//! place of the row norm, the widening of its rows and the serial input
//! quantizer; every other launch, and every decode step, is unchanged.
use super::*;
use model_deepseek41_contract::execution_options::PREFILL_NORM_CEIL_ROWS;

const FUSED: &str = "chop_prefill_norm_ceil_deepseek41_cached_v1";
const NORM: &str = "chop_row_rmsnorm_deepseek41_w5120_v1";
const WIDEN: &str = "chop_row_rounding_widen_deepseek41_w5120_v1";
const QUANTIZER: &str = "chop_deepseek41_moe_ceil_w5120_v1";

/// The launches of a prompt of `rows` rows and of one decode step after
/// it, with the candidate `enabled`.
fn launches(rows: u64, enabled: bool) -> [Vec<String>; 2] {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create_with_options(&host, 4, &[(PREFILL_NORM_CEIL_ROWS, enabled.into())]);
    let since = |from: usize| -> Vec<String> {
        host.device.journal.borrow()[from..]
            .iter()
            .filter(|line| line.starts_with("launch "))
            .cloned()
            .collect()
    };
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &prompt(7, 0, 0, rows, true),
        &mut Views::default(),
    );
    let prefill = since(from);
    let from = host.device.journal.borrow().len();
    step(
        &mut instance,
        &decode(&[(7, 0, rows, 9)]),
        &mut Views::default(),
    );
    let decode = since(from);
    instance.retire_request(7).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    [prefill, decode]
}

fn stream(line: &str) -> &str {
    line.strip_prefix("launch ")
        .and_then(|rest| rest.split_once(' '))
        .expect("a launch line")
        .0
}

/// `lines` without the unfused input norms: each row norm, the widening
/// that follows it on its stream, and each serial input quantizer.
fn without_chain(lines: &[String]) -> Vec<String> {
    let mut after_norm = std::collections::BTreeSet::<String>::new();
    let mut out = Vec::new();
    for line in lines {
        let stream = stream(line).to_owned();
        let norm = line.contains(&format!("{NORM} "));
        let widen = line.contains(&format!("{WIDEN} ")) && after_norm.contains(&stream);
        if norm {
            after_norm.insert(stream);
        } else {
            after_norm.remove(&stream);
        }
        if !(norm || widen || line.contains(&format!("{QUANTIZER} "))) {
            out.push(line.clone());
        }
    }
    out
}

/// `lines` with each launch stream and host handle renumbered by its first
/// use, and the `skip` symbols' launches dropped.
fn canonical(lines: &[String], skip: &[&str]) -> Vec<String> {
    let mut streams = std::collections::BTreeMap::<String, usize>::new();
    let mut handles = std::collections::BTreeMap::<String, usize>::new();
    lines
        .iter()
        .filter(|line| {
            !skip
                .iter()
                .any(|symbol| line.contains(&format!("{symbol} ")))
        })
        .map(|line| {
            let (stream, mut rest) = line
                .strip_prefix("launch ")
                .and_then(|rest| rest.split_once(' '))
                .expect("a launch line");
            let next = streams.len();
            let stream = *streams.entry(stream.to_owned()).or_insert(next);
            let mut out = format!("launch #{stream} ");
            while let Some(at) = rest.find("raw: ") {
                let (head, tail) = rest.split_at(at + "raw: ".len());
                let digits = tail
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(tail.len());
                let next = handles.len();
                let id = *handles.entry(tail[..digits].to_owned()).or_insert(next);
                out.push_str(head);
                out.push_str(&format!("#{id}"));
                rest = &tail[digits..];
            }
            out.push_str(rest);
            out
        })
        .collect()
}

/// Fails at the first launch where `candidate` and `default` differ.
fn same(what: &str, candidate: &[String], default: &[String]) {
    if let Some(at) =
        (0..candidate.len().max(default.len())).find(|&at| candidate.get(at) != default.get(at))
    {
        panic!(
            "{what}: launch {at} differs\ncandidate: {:?}\ndefault: {:?}",
            candidate.get(at),
            default.get(at)
        );
    }
}

fn count(lines: &[String], symbol: &str) -> usize {
    lines
        .iter()
        .filter(|line| line.contains(&format!("{symbol} ")))
        .count()
}

/// Below 33 rows and at the measured rows the candidate changes nothing. At
/// every other count the norm and quantizer of each sublayer input become
/// one fused launch of one CTA per row, and the journal without them is the
/// default's without the chain: the row norm, the widening after it and
/// the serial quantizer.
#[test]
fn the_candidate_fuses_every_input_norm_from_33_rows() {
    let calls = 2 * FLASH.layers() as usize * 4;
    let mut fused_rows = 0;
    for rows in [1, 32, 33, 64, 65, 128, 1500, 2305, 4096, 8191] {
        let [base_prefill, base_decode] = launches(rows, false);
        let [prefill, decode] = launches(rows, true);
        let measured = count(&base_prefill, FUSED) > 0;
        if rows <= 32 || measured {
            same(
                &format!("prefill of {rows} rows"),
                &canonical(&prefill, &[]),
                &canonical(&base_prefill, &[]),
            );
        } else {
            assert_eq!(count(&prefill, FUSED), calls, "rows={rows}");
            assert_eq!(count(&prefill, NORM), 0, "rows={rows}");
            assert_eq!(count(&prefill, QUANTIZER), 0, "rows={rows}");
            assert_eq!(count(&base_prefill, NORM), calls, "rows={rows}");
            assert_eq!(count(&base_prefill, QUANTIZER), calls, "rows={rows}");
            for launch in prefill.iter().filter(|l| l.contains(&format!("{FUSED} "))) {
                assert!(launch.contains(&format!("grid_x: {rows},")), "{launch}");
                assert!(launch.contains("block_x: 256,"), "{launch}");
            }
            same(
                &format!("prefill of {rows} rows without the norms"),
                &canonical(&prefill, &[FUSED]),
                &canonical(&without_chain(&base_prefill), &[]),
            );
            fused_rows += 1;
        }
        same(
            &format!("decode after {rows} rows"),
            &canonical(&decode, &[]),
            &canonical(&base_decode, &[]),
        );
    }
    assert_eq!(fused_rows, 7);
}
