//! Replay frozen launches exported by each revision's actual MoE lowerer.
use super::device::Pool;
use kernel_core::backend::{Launch, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_harness::{LabResult, launch};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

fn number(value: &Value) -> LabResult<u64> {
    value
        .as_u64()
        .ok_or_else(|| "expected unsigned graph field".into())
}
fn string(value: &Value) -> LabResult<&str> {
    value
        .as_str()
        .ok_or_else(|| "expected string graph field".into())
}
fn array(value: &Value) -> LabResult<&Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| "expected graph array".into())
}
pub fn document(path: &Path) -> LabResult<Value> {
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

pub struct Weights(BTreeMap<String, View>);
impl Weights {
    pub fn load(pool: &mut Pool<'_>, root: &Path, rank: u32) -> LabResult<Self> {
        let root = root.join(format!("rank{rank}"));
        let doc = document(&root.join("weights.json"))?;
        if doc["schema"] != "chopinfer.moe-wide-weights.v1"
            || doc["rank"] != rank
            || array(&doc["files"])?.len() != 20
        {
            return Err("invalid weight inventory".into());
        }
        let mut files = BTreeMap::new();
        for file in array(&doc["files"])? {
            let name = string(&file["name"])?;
            if Path::new(name).components().count() != 1 {
                return Err("weight name is not a single component".into());
            }
            let bytes = number(&file["bytes"])?;
            if std::fs::metadata(root.join(name))
                .map_err(|e| e.to_string())?
                .len()
                != bytes
            {
                return Err(format!("wrong weight length {name}"));
            }
            let view = pool.zeros(bytes)?;
            if pool.session.is_some() {
                pool.write(
                    view,
                    &std::fs::read(root.join(name)).map_err(|e| e.to_string())?,
                )?;
            }
            if files.insert(name.into(), view).is_some() {
                return Err("duplicate weight file".into());
            }
        }
        Ok(Self(files))
    }
    fn role(&self, name: &str, width: u32) -> LabResult<View> {
        let role = name
            .strip_prefix("layers.0.ffn.")
            .ok_or("unknown weight role")?;
        let scale = if width == 0 {
            "raw".to_owned()
        } else {
            format!("w{width}")
        };
        let file = match role {
            "gate.weight" => "router-bf16.bin".into(),
            "gate.weight.f32_domain" => "router-domain.bin".into(),
            "gate.bias" => "router-bias.bin".into(),
            "gate.bias_vl" => "router-bias-vl.bin".into(),
            "experts.gate_up.weight" => "gate-codes.bin".into(),
            "experts.gate_up.scale" => format!("gate-scale-{scale}.bin"),
            "experts.down.weight" => "down-codes.bin".into(),
            "experts.down.scale" => format!("down-scale-{scale}.bin"),
            _ => {
                let (which, field) = role
                    .strip_prefix("shared_experts.")
                    .and_then(|s| s.split_once('.'))
                    .ok_or("unknown shared weight role")?;
                if !["w1", "w2", "w3"].contains(&which) {
                    return Err("unknown shared bank".into());
                }
                let suffix = match field {
                    "weight" => "codes",
                    "scale" => "scale",
                    _ => return Err("unknown shared field".into()),
                };
                format!("shared-{which}-{suffix}.bin")
            }
        };
        self.0
            .get(&file)
            .copied()
            .ok_or_else(|| format!("missing weight {file}"))
    }
}

pub enum Unit {
    Fork,
    Join,
    Side(std::ops::Range<usize>),
    Compute(std::ops::Range<usize>),
}
pub struct Arm {
    pub rows: u32,
    pub prefix: Vec<Launch>,
    pub calls: Vec<Launch>,
    pub units: Vec<Unit>,
    /// Each lowered stage's calls, by stage name (`Route(0)`, ...).
    pub scopes: Vec<(String, std::ops::Range<usize>)>,
    pub roles: BTreeMap<String, View>,
    pub mutable: Vec<View>,
}
impl Arm {
    pub fn load(
        pool: &mut Pool<'_>,
        weights: &Weights,
        root: &Path,
        rows: u32,
        rank: u32,
        width: u32,
    ) -> LabResult<Self> {
        let doc = document(&root.join(format!("r{rows}-rank{rank}-w{width}.json")))?;
        if doc["schema"] != "chopinfer.moe-wide-graph.v1"
            || doc["rows"] != rows
            || doc["rank"] != rank
            || doc["scale_width"] != width
            || doc["bounded_row_views"] != true
        {
            return Err("wrong graph identity".into());
        }
        let recorded = doc["roles"].as_object().ok_or("missing graph roles")?;
        let mut owners = BTreeMap::new();
        for (name, span) in recorded
            .iter()
            .filter(|(name, _)| name.starts_with("layers."))
        {
            if number(&span["offset"])? != 0 {
                return Err("weight offset is not zero".into());
            }
            let view = weights.role(name, width)?;
            if view.bytes < number(&span["bytes"])? {
                return Err(format!("weight too short {name}"));
            }
            if owners.insert(number(&span["buffer"])?, view).is_some() {
                return Err("weight roles alias".into());
            }
        }
        let mut mutable = Vec::new();
        for (owner, bytes) in doc["allocations"]
            .as_object()
            .ok_or("missing allocations")?
        {
            let owner: u64 = owner
                .parse()
                .map_err(|e: std::num::ParseIntError| e.to_string())?;
            let bytes = number(bytes)?;
            if let Some(view) = owners.get(&owner) {
                if view.bytes < bytes {
                    return Err("weight owner too short".into());
                }
            } else {
                let view = pool.zeros(bytes)?;
                mutable.push(view);
                owners.insert(owner, view);
            }
        }
        let mut span = |value: &Value| {
            let root = *owners
                .get(&number(&value["buffer"])?)
                .ok_or("missing buffer owner")?;
            pool.subspan(root, number(&value["offset"])?, number(&value["bytes"])?)
        };
        let mut roles = recorded
            .iter()
            .map(|(name, value)| Ok((name.clone(), span(value)?)))
            .collect::<LabResult<BTreeMap<_, _>>>()?;
        let mut decode = |items: &Value| {
            array(items)?
                .iter()
                .map(|call| {
                    let arguments = array(&call["arguments"])?
                        .iter()
                        .map(|arg| {
                            if arg.as_object().is_none_or(|m| m.len() != 1) {
                                return Err("invalid argument record".into());
                            }
                            if let Some(value) = arg.get("pointer") {
                                return Ok(Argument::DeviceBuffer(span(value)?.handle));
                            }
                            if let Some(value) = arg.get("u32") {
                                return Ok(Argument::U32(
                                    number(value)?.try_into().map_err(|_| "u32 overflow")?,
                                ));
                            }
                            if let Some(value) = arg.get("u64") {
                                return Ok(Argument::U64(number(value)?));
                            }
                            if let Some(value) = arg.get("i32") {
                                return Ok(Argument::I32(
                                    value
                                        .as_i64()
                                        .ok_or("invalid i32")?
                                        .try_into()
                                        .map_err(|_| "i32 overflow")?,
                                ));
                            }
                            if arg["null"] == true {
                                return Ok(Argument::NullDevicePointer);
                            }
                            Err("unsupported recorded argument".into())
                        })
                        .collect::<LabResult<Vec<_>>>()?;
                    let triple = |name: &str| -> LabResult<[u32; 3]> {
                        array(&call[name])?
                            .iter()
                            .map(|v| {
                                number(v)?
                                    .try_into()
                                    .map_err(|_| "launch extent overflow".into())
                            })
                            .collect::<LabResult<Vec<_>>>()?
                            .try_into()
                            .map_err(|_| "launch extent is not a triple".into())
                    };
                    let plan = LaunchPlan {
                        grid: triple("grid")?,
                        block: triple("block")?,
                        cluster: triple("cluster")?,
                        dynamic_shared_bytes: number(&call["shared_bytes"])?
                            .try_into()
                            .map_err(|_| "shared byte overflow")?,
                        workspace_bytes: 0,
                        pdl: call["pdl"].as_bool().ok_or("missing pdl")?,
                    };
                    launch::config(plan)?;
                    let symbol = string(&call["symbol"])?;
                    if bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).is_none() {
                        return Err(format!("missing embedded symbol {symbol}"));
                    }
                    Ok(Launch {
                        symbol: symbol.into(),
                        rows: number(&call["rows"])?
                            .try_into()
                            .map_err(|_| "row overflow")?,
                        plan,
                        arguments: ArgumentList(arguments),
                        collective: false,
                        persistent: false,
                    })
                })
                .collect::<LabResult<Vec<_>>>()
        };
        let prefix = decode(&doc["prefix"])?;
        let calls = decode(&doc["calls"])?;
        // Legacy route storage includes a spare count and a prefill-sized
        // sorted tail. Both algorithms publish this same bounded metadata:
        // a 64-row tile per active expert through 64 rows, and above them
        // the routes plus each expert's padding to whole tiles.
        let sorted = (rows * 6 + 384 * 63).next_multiple_of(64).max(384 * 64);
        for (name, bytes) in [
            ("scratch.route_counts", 384 * 4),
            ("sorted_rows", u64::from(sorted) * 4),
        ] {
            let view = *roles.get(name).ok_or("missing routed metadata")?;
            roles.insert(name.into(), pool.subspan(view, 0, bytes)?);
        }
        let mut coverage = vec![0; calls.len()];
        let mut fork = 0;
        let mut join = 0;
        let units = array(&doc["units"])?
            .iter()
            .map(|unit| match string(&unit["kind"])? {
                "fork" => {
                    fork += 1;
                    Ok(Unit::Fork)
                }
                "join" => {
                    join += 1;
                    Ok(Unit::Join)
                }
                kind @ ("side" | "compute") => {
                    let start = number(&unit["start"])? as usize;
                    let end = number(&unit["end"])? as usize;
                    if start >= end || end > calls.len() {
                        return Err("invalid unit range".into());
                    }
                    for n in &mut coverage[start..end] {
                        *n += 1;
                    }
                    Ok(if kind == "side" {
                        Unit::Side(start..end)
                    } else {
                        Unit::Compute(start..end)
                    })
                }
                _ => Err("unknown graph unit".into()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        if prefix.is_empty() || fork != 1 || join != 1 || coverage.iter().any(|&n| n != 1) {
            return Err("graph does not cover one complete fork/join MoE".into());
        }
        let scopes = array(&doc["scopes"])?
            .iter()
            .map(|scope| {
                let start = number(&scope["start"])? as usize;
                let end = number(&scope["end"])? as usize;
                if start > end || end > calls.len() {
                    return Err("invalid stage scope".into());
                }
                Ok((string(&scope["stage"])?.to_owned(), start..end))
            })
            .collect::<LabResult<Vec<_>>>()?;
        if scopes.iter().map(|(_, r)| r.len()).sum::<usize>() != calls.len() {
            return Err("stage scopes do not cover the calls once".into());
        }
        Ok(Self {
            rows,
            prefix,
            calls,
            units,
            scopes,
            roles,
            mutable,
        })
    }
    /// The calls of the stage named `stage`.
    pub fn scope(&self, stage: &str) -> LabResult<&[Launch]> {
        self.scopes
            .iter()
            .find(|(name, _)| name == stage)
            .map(|(_, range)| &self.calls[range.clone()])
            .ok_or_else(|| format!("missing stage scope {stage}"))
    }
    /// The symbols each routed expert stage launches, in stage order.
    pub fn expert_symbols(&self) -> LabResult<Vec<Vec<&str>>> {
        EXPERT_STAGES
            .iter()
            .map(|stage| {
                Ok(self
                    .scope(stage)?
                    .iter()
                    .map(|c| c.symbol.as_str())
                    .collect())
            })
            .collect()
    }
    pub fn role(&self, name: &str) -> LabResult<View> {
        self.roles
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing semantic role {name}"))
    }
}

/// The routed expert stages: gate/up, down and the routed sum.
const EXPERT_STAGES: [&str; 3] = ["ExpertsGateUp(0)", "ExpertsDown(0)", "RoutedSum(0)"];

/// Above 64 rows a pair must compare the 8192-row expert chains (the
/// reference: gate/up chain and its ceil quantizer, down chain, routed sum)
/// with the backbone bank's wide direct queues (the candidate: pack,
/// absorbing gate queue and finisher, two-strip down queue, down finisher,
/// all compiled at one capacity of at least `rows`). Two exports of the
/// same selection would otherwise compare a path with itself. Pairs of 64
/// rows or fewer compare revisions and are not checked here.
pub fn check_wide_pair(
    rows: u32,
    reference: &[Vec<&str>],
    candidate: &[Vec<&str>],
) -> LabResult<()> {
    if rows <= 64 {
        return Ok(());
    }
    let chain = match reference {
        [gate, down, sum] => {
            let chain = |s: &str, kind: &str| {
                s.starts_with("deepseek41_chop_moe_chain_mxfp4_a8ceil_before_")
                    && s.contains(kind)
                    && s.contains("_e384k6_")
                    && s.contains("_t8192_")
            };
            matches!(gate.as_slice(), [g, q] if chain(g, "_gateup_")
                && q.starts_with("chop_deepseek41_moe_ceil_w"))
                && matches!(down.as_slice(), [d] if chain(d, "_down_"))
                && matches!(sum.as_slice(), [s] if s.starts_with("chop_moe_deepseek41_sum_add_")
                    && s.contains("_l384_t8192_"))
        }
        _ => false,
    };
    if !chain {
        return Err(format!(
            "{rows} rows: the reference does not run the 8192-row expert chains: {reference:?}"
        ));
    }
    let capacity = match candidate {
        [gate, ..] => gate
            .first()
            .and_then(|s| s.strip_prefix("chop_deepseek41_moe_wide_t"))
            .and_then(|s| s.strip_suffix("_e384_k6_pack_v1"))
            .and_then(|t| t.parse::<u32>().ok()),
        _ => None,
    };
    let wide = capacity.is_some_and(|t| {
        let bank = format!("t{t}_e384_k6_");
        let named = |s: &str, prefix: &str, stage: &str| {
            s.strip_prefix(prefix)
                .and_then(|s| s.strip_prefix(bank.as_str()))
                .is_some_and(|s| s.starts_with(stage))
        };
        t >= rows
            && matches!(candidate, [gate, down, sum]
                if matches!(gate.as_slice(), [_, g, f]
                    if named(g, "chop_deepseek41_moe_absorb_lanes_", "gate_small")
                        && named(f, "chop_deepseek41_moe_bandwidth_direct_", "gate_finish"))
                && matches!(down.as_slice(), [d]
                    if named(d, "chop_deepseek41_moe_strip2_", "down_small"))
                && matches!(sum.as_slice(), [s]
                    if named(s, "chop_deepseek41_moe_floor_", "down_finish")))
    });
    if !wide {
        return Err(format!(
            "{rows} rows: the candidate does not run the wide direct queues: {candidate:?}"
        ));
    }
    Ok(())
}
