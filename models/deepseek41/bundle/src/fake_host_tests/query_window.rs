//! Flash TP4's one-row decode runs each layer's query RoPE and window write
//! as one twin launch (`sm120::query_window`) with exactly the served
//! pair's operands; every other launch, and every other step, keeps its
//! bindings.
use super::*;

const ROPE: &str = "chop_deepseek41_latent_query_rope_heads_v1";
const WINDOW: &str = "chop_deepseek41_window_fused_";
const TWIN: &str = "chop_deepseek41_query_window_twin_";

pub(super) fn symbol(line: &str) -> &str {
    line.split_whitespace().nth(2).unwrap_or_default()
}

/// The top-level items of a journal line's argument list (its last
/// bracketed list).
pub(super) fn arguments(line: &str) -> Vec<String> {
    let bytes = line.as_bytes();
    assert_eq!(bytes.last(), Some(&b']'), "{line}");
    let mut depth = 0_i32;
    let mut start = None;
    for (at, byte) in bytes.iter().enumerate().rev() {
        match byte {
            b']' | b')' | b'}' => depth += 1,
            b'[' | b'(' | b'{' => {
                depth -= 1;
                if depth == 0 {
                    start = Some(at);
                    break;
                }
            }
            _ => {}
        }
    }
    let list = &line[start.unwrap() + 1..line.len() - 1];
    let (mut items, mut depth, mut from) = (Vec::new(), 0_i32, 0);
    for (at, byte) in list.bytes().enumerate() {
        match byte {
            b'[' | b'(' | b'{' => depth += 1,
            b']' | b')' | b'}' => depth -= 1,
            b',' if depth == 0 => {
                items.push(list[from..at].trim().to_owned());
                from = at + 1;
            }
            _ => {}
        }
    }
    items.push(list[from..].trim().to_owned());
    items
}

/// Each decode step's launch lines, steps of 1, 2, 8 and again 1 live
/// rows, on an instance of `layout` with the twins served or apart.
fn run_case(layout: KvLayout, served: bool) -> Vec<Vec<String>> {
    crate::sm120::query_window::with_served(served, || {
        let (cubins, steps) = decode_steps(layout);
        assert_eq!(
            cubins.contains(&crate::sm120::query_window::CUBIN.to_owned()),
            served
        );
        steps
    })
}

/// The cubins a Flash TP4 instance of `layout` loads, and each decode
/// step's launch lines, steps of 1, 2, 8 and again 1 live rows (the last
/// replaying the first's graph).
pub(super) fn decode_steps(layout: KvLayout) -> (Vec<String>, Vec<Vec<String>>) {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, layout);
    let mut instance = create(&host, 8);
    let cubins = host
        .device
        .module_cubins
        .borrow()
        .values()
        .map(|name| name.to_string())
        .collect();
    for slot in 0..8 {
        step(
            &mut instance,
            &prompt(slot + 1, slot, 0, 1, true),
            &mut Views::default(),
        );
    }
    let mut ends = [1_u64; 8];
    let mut steps = Vec::new();
    for rows in [1_u32, 2, 8, 1] {
        let from = host.device.journal.borrow().len();
        let items = (0..rows)
            .map(|slot| (u64::from(slot) + 1, u64::from(slot), ends[slot as usize], 7))
            .collect::<Vec<_>>();
        instance
            .execute(&decode(&items), &mut Views::default())
            .unwrap_or_else(|e| panic!("rows={rows}: {e:?}: {:?}", host.logs.borrow()));
        instance.commit().unwrap();
        steps.push(
            host.device.journal.borrow()[from..]
                .iter()
                .filter(|l| l.starts_with("launch "))
                .cloned()
                .collect(),
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
    assert_launches_fit_configured_shared(&host.device.journal.borrow(), &shape.topology());
    (cubins, steps)
}

pub(super) fn replaced(line: &str) -> bool {
    let s = symbol(line);
    s == ROPE || s.starts_with(WINDOW) || s.starts_with(TWIN)
}

pub(super) fn stream(line: &str) -> &str {
    line.split_whitespace().nth(1).unwrap_or_default()
}

/// A run's steps with every stream and buffer handle renumbered by first
/// use: first over the launches a twin does not replace (`replaced`), then
/// over the rest, in order. A twin's module load takes fake handles, so raw
/// numbers shift between runs while the streams and buffers they name do
/// not.
pub(super) fn canonical(
    steps: &[Vec<String>],
    replaced: impl Fn(&str) -> bool,
) -> Vec<Vec<String>> {
    let mut streams = BTreeMap::<String, usize>::new();
    let mut buffers = BTreeMap::<String, usize>::new();
    let mut number = |line: &str| {
        let next = streams.len();
        streams.entry(stream(line).to_owned()).or_insert(next);
        for raw in line.split("raw: ").skip(1) {
            let raw: String = raw.chars().take_while(char::is_ascii_digit).collect();
            let next = buffers.len();
            buffers.entry(raw).or_insert(next);
        }
    };
    steps
        .iter()
        .flatten()
        .filter(|l| !replaced(l))
        .for_each(|l| number(l));
    steps.iter().flatten().for_each(|l| number(l));
    steps
        .iter()
        .map(|lines| {
            lines
                .iter()
                .map(|line| {
                    let rest = line.splitn(3, ' ').nth(2).unwrap_or_default();
                    let mut parts = rest.split("raw: ");
                    let mut out = format!(
                        "launch s{} {}",
                        streams[stream(line)],
                        parts.next().unwrap()
                    );
                    for part in parts {
                        let digits = part.chars().take_while(char::is_ascii_digit).count();
                        out.push_str(&format!("raw: #{}", buffers[&part[..digits]]));
                        out.push_str(&part[digits..]);
                    }
                    out
                })
                .collect()
        })
        .collect()
}

pub(super) fn first_difference(what: &str, a: &[String], b: &[String]) {
    if let Some(at) = (0..a.len().max(b.len())).find(|&i| a.get(i) != b.get(i)) {
        panic!(
            "{what} differ at {at} of {}/{}:\n{:?}\n{:?}",
            a.len(),
            b.len(),
            a.get(at),
            b.get(at)
        );
    }
}

#[test]
fn one_row_decode_runs_each_layers_rope_and_window_write_as_one_twin_launch() {
    for layout in KV_LAYOUTS {
        let reference = canonical(&run_case(layout, false), replaced);
        let serving = canonical(&run_case(layout, true), replaced);
        let count = |lines: &[String], prefix: &str| {
            lines
                .iter()
                .filter(|l| symbol(l).starts_with(prefix))
                .count()
        };
        // The first one-row step records its graph; the last replays it.
        let [first, two, eight, last] = [0, 1, 2, 3].map(|i| (&reference[i], &serving[i]));
        let layers = 40 * 4;
        assert_eq!(count(first.0, ROPE), layers);
        assert_eq!(count(first.0, WINDOW), layers);
        assert_eq!(count(first.0, TWIN), 0);
        assert_eq!(count(first.1, ROPE), 0);
        assert_eq!(count(first.1, WINDOW), 0);
        assert_eq!(count(first.1, TWIN), layers);
        let suffix = if layout.storage == KvStorage::Bf16 {
            "bf16_v1"
        } else {
            "native_v1"
        };
        assert!(
            first
                .1
                .iter()
                .filter(|l| symbol(l).starts_with(TWIN))
                .all(|l| symbol(l).ends_with(suffix))
        );
        // Every other launch keeps its place and bindings.
        let others = |lines: &[String]| {
            lines
                .iter()
                .filter(|l| !replaced(l))
                .cloned()
                .collect::<Vec<_>>()
        };
        first_difference("one-row launches", &others(first.0), &others(first.1));
        // Each twin binds its layer's served RoPE and window write operands.
        let ropes: Vec<_> = first.0.iter().filter(|l| symbol(l) == ROPE).collect();
        let windows: Vec<_> = first
            .0
            .iter()
            .filter(|l| symbol(l).starts_with(WINDOW))
            .collect();
        let twins: Vec<_> = first
            .1
            .iter()
            .filter(|l| symbol(l).starts_with(TWIN))
            .collect();
        for ((rope, window), twin) in ropes.iter().zip(&windows).zip(&twins) {
            assert_eq!(stream(rope), stream(twin));
            let rope = arguments(rope);
            let mut expected = vec![
                rope[0].clone(),
                rope[1].clone(),
                rope[2].clone(),
                rope[4].clone(),
            ];
            expected.extend(arguments(window));
            assert_eq!(arguments(twin), expected);
        }
        // Wider steps keep the served pair; the replayed step launches none.
        first_difference("two-row launches", two.0, two.1);
        first_difference("eight-row launches", eight.0, eight.1);
        first_difference("replayed one-row launches", last.0, last.1);
        assert_eq!(count(two.1, TWIN) + count(eight.1, TWIN), 0);
    }
}
