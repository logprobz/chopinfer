use kernel_core::codec::bf16_to_f32;
use kernel_harness::LabResult;
use serde_json::Value;
use std::path::{Component, Path, PathBuf};
pub const H: usize = 5120;

pub struct Case {
    pub id: String,
    pub rows: usize,
    pub root: PathBuf,
    pub record: Value,
}
impl Case {
    pub fn widening(&self) -> bool {
        let output = &self.record["outputs"]["widened"];
        output.is_object() && output["null"] != true
    }
    pub fn operand(&self, direction: &str, role: &str) -> LabResult<Vec<u8>> {
        let operand = &self.record[direction][role];
        let name = operand["file"]
            .as_str()
            .ok_or_else(|| format!("missing {direction}/{role}"))?;
        let path = self.root.join(name);
        regular(&self.root, Path::new(name))?;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        if operand["bytes"].as_u64() != Some(bytes.len() as u64)
            || operand["sha256"].as_str() != Some(&kernel_build::digest(&bytes))
        {
            return Err(format!("{} {role} operand changed", self.id));
        }
        Ok(bytes)
    }
    pub fn values(&self, direction: &str, role: &str) -> LabResult<Vec<f64>> {
        let data = self.operand(direction, role)?;
        let dtype = self.record[direction][role]["dtype"]
            .as_str()
            .ok_or("missing dtype")?;
        let v = match dtype {
            "bf16" => bf16(&data)
                .into_iter()
                .map(|x| f64::from(bf16_to_f32(x)))
                .collect(),
            "f32" => f32s(&data).into_iter().map(f64::from).collect(),
            _ => return Err("expected a floating operand".into()),
        };
        Ok(v)
    }
    pub fn eps_bits(&self) -> LabResult<u32> {
        let args = self.record["norm_arguments"]
            .as_array()
            .ok_or("missing norm arguments")?;
        let values = args
            .iter()
            .filter_map(|v| v.as_str()?.strip_prefix("U32(")?.strip_suffix(')'))
            .map(str::parse::<u32>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;
        if values.len() != 1
            || !f32::from_bits(values[0]).is_finite()
            || f32::from_bits(values[0]) <= 0.0
        {
            return Err("invalid captured norm epsilon".into());
        }
        Ok(values[0])
    }
}

#[test]
fn absent_or_null_widen_is_disabled() {
    for (outputs, expected) in [
        (serde_json::json!({}), false),
        (serde_json::json!({"widened":{"null":true}}), false),
        (
            serde_json::json!({"widened":{"file":"output.bin","dtype":"f32"}}),
            true,
        ),
    ] {
        let case = Case {
            id: "widen-control".into(),
            rows: 1,
            root: PathBuf::new(),
            record: serde_json::json!({"outputs":outputs}),
        };
        assert_eq!(case.widening(), expected);
    }
}
pub fn bf16(bytes: &[u8]) -> Vec<u16> {
    assert!(bytes.len().is_multiple_of(2));
    bytes
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
pub fn f32s(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn regular(root: &Path, name: &Path) -> LabResult<()> {
    if name.is_absolute()
        || name
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("corpus file escaped its root".into());
    }
    let file = root.join(name);
    if file.ancestors().any(|p| p.is_symlink()) || !file.is_file() {
        return Err("corpus file is not regular".into());
    }
    Ok(())
}
pub fn load() -> LabResult<(Vec<Case>, bool, String)> {
    let path =
        PathBuf::from(std::env::var_os("CHOPINFER_HC_GRAM_CORPUS").ok_or("missing corpus path")?);
    let bytes = std::fs::read(&path).map_err(|e| e.to_string())?;
    let sha = kernel_build::digest(&bytes);
    if sha != std::env::var("CHOPINFER_HC_GRAM_CORPUS_SHA256").map_err(|e| e.to_string())? {
        return Err("corpus manifest digest differs".into());
    }
    let value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let complete = value["complete"]
        .as_bool()
        .ok_or("missing completeness declaration")?;
    if !complete && std::env::var("CHOPINFER_HC_GRAM_ALLOW_PARTIAL").as_deref() != Ok("1") {
        return Err("real-input corpus is incomplete; diagnostic opt-in required".into());
    }
    let root = path.parent().ok_or("corpus parent missing")?;
    for (name, expected) in value["files"]
        .as_object()
        .ok_or("missing corpus file hashes")?
    {
        regular(root, Path::new(name))?;
        let data = std::fs::read(root.join(name)).map_err(|e| e.to_string())?;
        if expected.as_str() != Some(&kernel_build::digest(&data)) {
            return Err("corpus file changed".into());
        }
    }
    let mut cases = Vec::new();
    let mut counts = std::collections::BTreeMap::new();
    for capture in value["captures"].as_array().ok_or("missing captures")? {
        if capture["request_origin"] != "real" {
            return Err("warmup cannot qualify numerical outputs".into());
        }
        let phase = capture["phase"].as_str().ok_or("missing phase")?;
        let rows = capture["rows"].as_u64().ok_or("missing rows")? as usize;
        if !matches!(
            (phase, rows),
            ("decode-1", 1) | ("decode-8", 8) | ("verify-6", 6) | ("verify-32", 32)
        ) {
            return Err("unsupported captured phase".into());
        }
        *counts.entry(phase.to_owned()).or_insert(0) += 1;
        let records = capture["cases"].as_array().ok_or("missing cases")?;
        if records.len() != 80
            || records
                .iter()
                .filter(|r| r["eligible_post_mix"] == true)
                .count()
                != 77
        {
            return Err("missing complete HC boundary coverage".into());
        }
        for record in records.iter().filter(|r| r["eligible_post_mix"] == true) {
            let id = format!(
                "{phase}:{}:{}:{}",
                capture["request_ids"], record["site"][0], record["site"][1]
            );
            let group = capture["group"].as_str().ok_or("missing group")?;
            if !["ar", "verify6", "verify32"].contains(&group) {
                return Err("invalid group".into());
            }
            cases.push(Case {
                id,
                rows,
                root: root.join(group).join("operands"),
                record: record.clone(),
            });
        }
    }
    if cases.is_empty() || (complete && (counts.len() != 4 || counts.values().any(|&n| n != 2))) {
        return Err("zero cases or incomplete declared coverage".into());
    }
    Ok((cases, complete, sha))
}
