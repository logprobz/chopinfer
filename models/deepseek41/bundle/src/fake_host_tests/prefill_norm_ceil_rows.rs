//! The `prefill_norm_ceil_rows` option, adopted: the default admits the
//! fused prefill norm-ceil at the option's rows, 33 to 8192, so the
//! option changes no launch (`norm_prefill` pins the route).
use super::*;
use model_deepseek41_contract::execution_options::PREFILL_NORM_CEIL_ROWS;

/// The launches of a prompt of `rows` rows and of one decode step after
/// it, with the option `enabled`.
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

/// `lines` with each launch stream and host handle renumbered by its first
/// use.
fn canonical(lines: &[String]) -> Vec<String> {
    let mut streams = std::collections::BTreeMap::<String, usize>::new();
    let mut handles = std::collections::BTreeMap::<String, usize>::new();
    lines
        .iter()
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

/// Fails at the first launch where `option` and `default` differ.
fn same(what: &str, option: &[String], default: &[String]) {
    if let Some(at) =
        (0..option.len().max(default.len())).find(|&at| option.get(at) != default.get(at))
    {
        panic!(
            "{what}: launch {at} differs\noption: {:?}\ndefault: {:?}",
            option.get(at),
            default.get(at)
        );
    }
}

/// The option's prefill and decode journals are the default's at every
/// row count.
#[test]
fn the_option_changes_nothing() {
    for rows in [1, 32, 33, 64, 65, 128, 1500, 2305, 2624, 4096, 8191, 8192] {
        let [prefill, decode] = launches(rows, false);
        let [option_prefill, option_decode] = launches(rows, true);
        same(
            &format!("prefill of {rows} rows"),
            &canonical(&option_prefill),
            &canonical(&prefill),
        );
        same(
            &format!("decode after {rows} rows"),
            &canonical(&option_decode),
            &canonical(&decode),
        );
    }
}
