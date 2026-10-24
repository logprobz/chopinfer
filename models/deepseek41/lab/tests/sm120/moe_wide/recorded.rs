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
        // sorted tail. Both algorithms publish this same bounded metadata.
        for (name, bytes) in [
            ("scratch.route_counts", 384 * 4),
            ("sorted_rows", 384 * 64 * 4),
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
        Ok(Self {
            rows,
            prefix,
            calls,
            units,
            roles,
            mutable,
        })
    }
    pub fn role(&self, name: &str) -> LabResult<View> {
        self.roles
            .get(name)
            .copied()
            .ok_or_else(|| format!("missing semantic role {name}"))
    }
}
