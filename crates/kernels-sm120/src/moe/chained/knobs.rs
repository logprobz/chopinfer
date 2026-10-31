//! Cache-policy and schedule knobs of the direct queues and finishers of
//! the absorbing path (the 32- to 64-row buckets and the wide verify
//! steps: [`lanes`], [`strip2`], [`down_scale_tiles`], [`bandwidth`],
//! [`floor`]). Every knob keeps each loaded value, product, add and stored
//! output of its body (`chained_knobs.cuh`):
//!
//! - stream: the queues' weight code loads skip L1 and carry an L2
//!   evict_first policy.
//! - keep: the queues' K-block partial stores carry an L2 evict_last policy.
//! - retire: whoever folds a partial line discards it from L2 once read
//!   (`discard.global.L2`), so the dead line is never written back.
//! - loads: the gate finisher issues every partial load before its first
//!   add, the served all-loads-first finisher
//!   ([`bandwidth::finish_loads_entry`]).
//!
//! No knob is the served set. Each entry here keeps the parameters, launch
//! bounds and launch of the served entry it is built from; an entry a knob
//! set does not change is the served one.
//!
//! [`floor`]: super::floor
use super::{
    bandwidth, down_scale_tiles,
    floor::{Entry, Stage},
    lanes, strip2,
};
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::Refusal;

/// The header of every knob entry.
pub const HEADER: &str = "chained_knob_entries.cuh";

/// Offset widths of the tile-order encoded scale entries: the package's.
pub const SCALE_WIDTHS: [u32; 1] = [1];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Knobs {
    pub stream: bool,
    pub keep: bool,
    pub retire: bool,
    pub loads: bool,
}

impl Knobs {
    /// Whether no knob is set.
    pub fn served(self) -> bool {
        self == Self::default()
    }

    /// Every knob set, the served one first.
    pub fn all() -> impl Iterator<Item = Knobs> {
        (0..16u32).map(|bits| Knobs {
            stream: bits & 1 != 0,
            keep: bits & 2 != 0,
            retire: bits & 4 != 0,
            loads: bits & 8 != 0,
        })
    }

    fn set(self, retire: bool) -> String {
        format!(
            "chop::moe::knobs::Set<{},{},{retire}>",
            self.stream, self.keep
        )
    }

    fn letters(self, retire: bool) -> String {
        [(self.stream, 's'), (self.keep, 'k'), (retire, 'r')]
            .into_iter()
            .filter_map(|(on, c)| on.then_some(c))
            .collect()
    }
}

/// `symbol` with `tag` before its version.
fn tagged(symbol: &str, tag: &str) -> String {
    format!(
        "{}_{tag}_v1",
        symbol.strip_suffix("_v1").expect("versioned symbol")
    )
}

fn replace_once(body: &str, from: &str, to: &str) -> String {
    assert_eq!(body.matches(from).count(), 1, "{from} in {body}");
    body.replace(from, to)
}

/// `raw` reading a tile-order encoded scale bank of `width` in place of
/// `raw_scales`, named as the served encoded entries are.
fn encoded(
    mut raw: EntrySpec,
    raw_scales: &str,
    scales: &str,
    width: Option<u32>,
) -> Result<EntrySpec, Refusal> {
    let Some(width) = width else {
        return Ok(raw);
    };
    if !SCALE_WIDTHS.contains(&width) {
        return Err(Refusal("knob entries read raw or width-1 tile scales"));
    }
    raw.symbol = format!(
        "{}_scale_tiles_w{width}_v1",
        raw.symbol.strip_suffix("_v1").expect("versioned symbol")
    );
    raw.body = replace_once(&raw.body, raw_scales, &format!("{scales}<{width}>"));
    Ok(raw)
}

const GATE_RAW: &str = "chop::moe::lanes::RawScales";
const GATE_ENCODED: &str = "chop::moe::lanes::EncodedScales";
const DOWN_RAW: &str = "chop::moe::streamed::RawScales";
const DOWN_ENCODED: &str = "chop::moe::encoded_down::Scales";

/// The absorbing gate queue under `k`'s stream and keep, with raw scale
/// tiles (`None`) or a tile-order encoded scale bank of `Some(width)`.
pub fn gate_entry(e: &Entry, k: Knobs, width: Option<u32>) -> Result<EntrySpec, Refusal> {
    if !k.stream && !k.keep {
        return match width {
            None => lanes::absorb_entry(e),
            Some(width) => lanes::absorb_scale_tiles_entry(e, width),
        };
    }
    let mut spec = lanes::absorb_entry(e)?;
    spec.symbol = tagged(&spec.symbol, &format!("policy_{}", k.letters(false)));
    spec.body = replace_once(
        &spec.body,
        ">(input,",
        &format!(",{GATE_RAW},{}>(input,", k.set(false)),
    );
    encoded(spec, GATE_RAW, GATE_ENCODED, width)
}

/// The gate finisher under `k`: the served bandwidth or all-loads-first
/// finisher by loads, under retire.
pub fn gate_finish_entry(e: &Entry, k: Knobs) -> EntrySpec {
    let mut spec = if k.loads {
        bandwidth::finish_loads_entry(e)
    } else {
        bandwidth::entry(e, Stage::GateFinish, bandwidth::Loads::Direct)
    };
    if k.retire {
        let (from, to) = if k.loads {
            (
                "chop::moe::bandwidth_loads::gate_finish<",
                "chop::moe::knobs::gate_finish_loads<",
            )
        } else {
            (
                "chop::moe::bandwidth::gate_finish<",
                "chop::moe::knobs::gate_finish<",
            )
        };
        spec.symbol = tagged(&spec.symbol, "policy_r");
        spec.body = replace_once(&spec.body, from, to);
        spec.body = replace_once(
            &spec.body,
            "u>(scratch,",
            &format!("u,{}>(scratch,", k.set(true)),
        );
    }
    spec
}

/// The two-strip down queue under `k`'s stream and keep, with raw scales
/// (`None`) or a tile-order encoded scale bank of `Some(width)`.
pub fn down_entry(e: &Entry, k: Knobs, width: Option<u32>) -> Result<EntrySpec, Refusal> {
    if !k.stream && !k.keep {
        return match width {
            None => Ok(strip2::entry(e)),
            Some(width) => down_scale_tiles::strip2_entry(e, width),
        };
    }
    let mut spec = strip2::entry(e);
    spec.symbol = tagged(&spec.symbol, &format!("policy_{}", k.letters(false)));
    spec.body = replace_once(
        &spec.body,
        ">(input,",
        &format!(",{DOWN_RAW},{}>(input,", k.set(false)),
    );
    encoded(spec, DOWN_RAW, DOWN_ENCODED, width)
}

/// The down finisher under `k`'s retire.
pub fn down_finish_entry(e: &Entry, k: Knobs) -> EntrySpec {
    let mut spec = e.entry(Stage::DownFinish);
    if k.retire {
        spec.symbol = tagged(&spec.symbol, "policy_r");
        spec.body = replace_once(
            &spec.body,
            "chop::moe::floor::down_finish<",
            "chop::moe::knobs::down_finish<",
        );
        spec.body = replace_once(
            &spec.body,
            ",false>(scratch,",
            &format!(",{}>(scratch,", k.set(true)),
        );
    }
    spec
}

/// Whether `symbol` is an entry of this module: one some knob set takes in
/// place of a served entry.
pub fn owns(symbol: &str) -> bool {
    symbol.contains("_policy_")
}

/// Every entry of `e` that some knob set takes and no served module has,
/// raw then encoded at each [`SCALE_WIDTHS`] width, once each.
pub fn entries(e: &Entry) -> Result<Vec<EntrySpec>, Refusal> {
    let mut out: Vec<EntrySpec> = Vec::new();
    let widths = std::iter::once(None).chain(SCALE_WIDTHS.map(Some));
    for k in Knobs::all() {
        let mut specs = Vec::new();
        for width in widths.clone() {
            specs.push(gate_entry(e, k, width)?);
            specs.push(down_entry(e, k, width)?);
        }
        specs.push(gate_finish_entry(e, k));
        specs.push(down_finish_entry(e, k));
        for spec in specs {
            if owns(&spec.symbol) && !out.iter().any(|s| s.symbol == spec.symbol) {
                out.push(spec);
            }
        }
    }
    Ok(out)
}

/// [`entries`] of each bank in `entries`, in one module.
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    let mut specs = Vec::new();
    for e in entries {
        specs.extend(self::entries(e)?);
    }
    Ok(super::spec(name.into(), HEADER, specs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::chained::wide::tests::fixture;

    const WIDTHS: [Option<u32>; 2] = [None, Some(1)];

    /// Each knob entry is the served entry it replaces with only its symbol
    /// and the device function's knob set changed; the served knob set
    /// changes nothing.
    #[test]
    fn knob_entries_keep_the_served_abi() {
        for capacity in [64, 192] {
            let e = fixture(capacity);
            let served_gate = |w: Option<u32>| match w {
                None => lanes::absorb_entry(&e).unwrap(),
                Some(w) => lanes::absorb_scale_tiles_entry(&e, w).unwrap(),
            };
            let served_down = |w: Option<u32>| match w {
                None => strip2::entry(&e),
                Some(w) => down_scale_tiles::strip2_entry(&e, w).unwrap(),
            };
            for k in Knobs::all() {
                for width in WIDTHS {
                    let gate = gate_entry(&e, k, width).unwrap();
                    let served = served_gate(width);
                    assert_eq!(
                        (gate.max_threads, gate.min_blocks),
                        (served.max_threads, served.min_blocks)
                    );
                    if k.stream || k.keep {
                        assert_eq!(gate.params, served.params);
                        let set = format!("chop::moe::knobs::Set<{},{},false>", k.stream, k.keep);
                        let scales = match width {
                            None => GATE_RAW.to_string(),
                            Some(w) => format!("{GATE_ENCODED}<{w}>"),
                        };
                        let served_call = match width {
                            None => ">(input,".to_string(),
                            Some(_) => format!(",{scales}>(input,"),
                        };
                        assert_eq!(
                            gate.body,
                            served
                                .body
                                .replace(&served_call, &format!(",{scales},{set}>(input,"))
                        );
                    } else {
                        assert!(same(&gate, &served));
                    }
                    let down = down_entry(&e, k, width).unwrap();
                    let served = served_down(width);
                    assert_eq!(
                        (&down.params, down.max_threads, down.min_blocks),
                        (&served.params, served.max_threads, served.min_blocks)
                    );
                    if !k.stream && !k.keep {
                        assert!(same(&down, &served));
                    }
                    for spec in [&gate, &down] {
                        assert_eq!(owns(&spec.symbol), k.stream || k.keep);
                        assert_eq!(spec.symbol.contains("_scale_tiles_w1_v1"), width.is_some());
                    }
                }
                let finish = gate_finish_entry(&e, k);
                let served = bandwidth::entry(&e, Stage::GateFinish, bandwidth::Loads::Direct);
                assert_eq!(
                    (&finish.params, finish.max_threads, finish.min_blocks),
                    (&served.params, served.max_threads, served.min_blocks)
                );
                assert_eq!(owns(&finish.symbol), k.retire);
                assert_eq!(
                    finish.body.contains("bandwidth_loads::gate_finish<")
                        || finish.body.contains("knobs::gate_finish_loads<"),
                    k.loads
                );
                let down_finish = down_finish_entry(&e, k);
                let served = e.entry(Stage::DownFinish);
                assert_eq!(
                    (
                        &down_finish.params,
                        down_finish.max_threads,
                        down_finish.min_blocks
                    ),
                    (&served.params, served.max_threads, served.min_blocks)
                );
                assert_eq!(owns(&down_finish.symbol), k.retire);
                assert_eq!(same(&down_finish, &served), !k.retire);
            }
            let all = entries(&e).unwrap();
            assert_eq!(all.len(), 15);
            assert!(all.iter().all(|s| owns(&s.symbol)));
        }
    }

    fn same(a: &EntrySpec, b: &EntrySpec) -> bool {
        (&a.symbol, &a.body, &a.params, a.max_threads, a.min_blocks)
            == (&b.symbol, &b.body, &b.params, b.max_threads, b.min_blocks)
    }

    /// The encoded entries pair with their raw twins as the served encoded
    /// entries do: the raw symbol with `_scale_tiles_w1` before its version.
    #[test]
    fn encoded_knob_entries_name_their_raw_twins() {
        let e = fixture(64);
        let all = entries(&e).unwrap();
        for spec in all.iter().filter(|s| s.symbol.contains("_scale_tiles_w")) {
            let (base, width) = spec.symbol.rsplit_once("_scale_tiles_w").unwrap();
            assert_eq!(width, "1_v1");
            let raw = all
                .iter()
                .find(|s| s.symbol == format!("{base}_v1"))
                .expect("raw twin");
            assert_eq!(raw.params, spec.params);
            assert_eq!(
                raw.body
                    .replace(GATE_RAW, &format!("{GATE_ENCODED}<1>"))
                    .replace(DOWN_RAW, &format!("{DOWN_ENCODED}<1>"))
                    .replace(&raw.symbol, &spec.symbol),
                spec.body
            );
        }
        assert_eq!(
            all.iter()
                .filter(|s| s.symbol.contains("_scale_tiles_w"))
                .count(),
            6
        );
    }
}
