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

#[test]
fn one_row_decode_uses_twins_and_wider_steps_keep_the_pair() {
    for layout in KV_LAYOUTS {
        let (cubins, steps) = decode_steps(layout);
        assert!(cubins.contains(&crate::sm120::query_window::CUBIN.to_owned()));
        let count = |lines: &[String], prefix: &str| {
            lines
                .iter()
                .filter(|line| symbol(line).starts_with(prefix))
                .count()
        };
        assert_eq!(count(&steps[0], TWIN), 40 * 4);
        assert_eq!(count(&steps[0], ROPE), 0);
        assert_eq!(count(&steps[0], WINDOW), 0);
        for step in &steps[1..3] {
            assert_eq!(count(step, TWIN), 0);
            assert_eq!(count(step, ROPE), 40 * 4);
            assert_eq!(count(step, WINDOW), 40 * 4);
        }
        assert!(
            steps[3].is_empty(),
            "the final one-row step replays its graph"
        );
    }
}
