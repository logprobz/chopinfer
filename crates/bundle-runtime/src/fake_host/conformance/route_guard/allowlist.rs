//! Bounded allowlist TOML: one assignment per line, JSON-compatible strings
//! and arrays. Refuse unknown fields, duplicate fields and empty justifications.
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug)]
pub(super) struct Allow {
    pub entry: Option<String>,
    pub profiles: Vec<String>,
    pub phase: String,
    pub family: String,
    pub symbol: String,
    pub rows: Vec<(u32, u32)>,
    pub reason: String,
    pub task: String,
}

pub(super) fn glob(pattern: &str, value: &str) -> bool {
    let mut rest = value;
    let mut parts = pattern.split('*').peekable();
    let first = parts.next().unwrap();
    if !rest.starts_with(first) {
        return false;
    }
    rest = &rest[first.len()..];
    if parts.peek().is_none() {
        return rest.is_empty();
    }
    while let Some(part) = parts.next() {
        if parts.peek().is_none() {
            return rest.ends_with(part);
        }
        let Some(at) = rest.find(part) else {
            return false;
        };
        rest = &rest[at + part.len()..];
    }
    true
}

impl Allow {
    pub fn applies(&self, profile: &str) -> bool {
        self.profiles.iter().any(|p| glob(p, profile))
    }
    pub fn matches(
        &self,
        profile: &str,
        phase: &str,
        family: &str,
        entry: &str,
        symbol: &str,
        rows: u32,
    ) -> bool {
        self.entry.as_ref().is_none_or(|p| glob(p, entry))
            && self.applies(profile)
            && self.phase == phase
            && self.family == family
            && glob(&self.symbol, symbol)
            && self.rows.iter().any(|&(lo, hi)| lo <= rows && rows <= hi)
    }
}

pub(super) fn parse(text: &str) -> Result<Vec<Allow>, String> {
    fn entry(mut fields: BTreeMap<String, Value>) -> Result<Allow, String> {
        let identity: Option<String> = fields
            .remove("entry")
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| e.to_string())?;
        if identity.as_ref().is_some_and(|s| s.is_empty()) {
            return Err("empty entry identity".into());
        }
        let mut take = |name: &str| {
            fields
                .remove(name)
                .ok_or_else(|| format!("allowlist needs {name}"))
        };
        let profiles: Vec<String> =
            serde_json::from_value(take("profiles")?).map_err(|e| e.to_string())?;
        let phase: String = serde_json::from_value(take("phase")?).map_err(|e| e.to_string())?;
        let family: String = serde_json::from_value(take("family")?).map_err(|e| e.to_string())?;
        let symbol: String = serde_json::from_value(take("symbol")?).map_err(|e| e.to_string())?;
        let rows: Vec<(u32, u32)> =
            serde_json::from_value(take("rows")?).map_err(|e| e.to_string())?;
        let reason: String = serde_json::from_value(take("reason")?).map_err(|e| e.to_string())?;
        let task: String = serde_json::from_value(take("task")?).map_err(|e| e.to_string())?;
        if reason.trim().is_empty() || task.trim().is_empty() {
            return Err("allowlist needs nonempty reason and task".into());
        }
        if !fields.is_empty()
            || profiles.is_empty()
            || profiles.iter().any(|s| s.is_empty())
            || !["prefill", "decode", "verify"].contains(&phase.as_str())
            || family.is_empty()
            || symbol.is_empty()
            || rows.is_empty()
            || rows.iter().any(|&(a, b)| a == 0 || a > b)
        {
            return Err("invalid allowlist fields or ranges".into());
        }
        Ok(Allow {
            entry: identity,
            profiles,
            phase,
            family,
            symbol,
            rows,
            reason,
            task,
        })
    }
    let mut entries = Vec::new();
    let mut fields = None::<BTreeMap<String, Value>>;
    for (index, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line == "[[allow]]" {
            if let Some(previous) = fields.replace(BTreeMap::new()) {
                entries.push(entry(previous)?);
            }
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("allowlist line {}: expected assignment", index + 1))?;
        let fields = fields.as_mut().ok_or("assignment outside [[allow]]")?;
        let value = serde_json::from_str(value.trim())
            .map_err(|e| format!("allowlist line {}: {e}", index + 1))?;
        if fields.insert(key.trim().to_owned(), value).is_some() {
            return Err(format!("duplicate allowlist key {}", key.trim()));
        }
    }
    if let Some(fields) = fields {
        entries.push(entry(fields)?);
    }
    Ok(entries)
}
