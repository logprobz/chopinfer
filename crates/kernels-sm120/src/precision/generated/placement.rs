//! Reassemble adopted entries at their recorded cubin positions.
use super::*;
use kernel_build::{KernelSelection, SelectedCubin};

pub fn capture(
    records: impl IntoIterator<Item = Adoption>,
    locked: &str,
) -> Result<Vec<Adoption>, String> {
    let lock: serde_json::Value = serde_json::from_str(locked).map_err(|e| e.to_string())?;
    let mut positions = BTreeMap::new();
    for cubin in lock.as_array().ok_or("lock is not a cubin list")? {
        let name = cubin["name"].as_str().ok_or("lock cubin has no name")?;
        for (position, entry) in cubin["entries"]
            .as_array()
            .ok_or("lock cubin has no entries")?
            .iter()
            .enumerate()
        {
            let symbol = entry["symbol"].as_str().ok_or("lock entry has no symbol")?;
            if positions
                .insert(
                    symbol,
                    Placement {
                        cubin: name.into(),
                        position,
                    },
                )
                .is_some()
            {
                return Err(format!("ambiguous locked placement for {symbol}"));
            }
        }
    }
    records
        .into_iter()
        .map(|mut record| {
            record.placement = Some(
                positions
                    .get(record.symbol.as_str())
                    .ok_or_else(|| format!("no locked placement for {}", record.symbol))?
                    .clone(),
            );
            Ok(record)
        })
        .collect()
}

pub fn assemble(
    selection: &KernelSelection,
    replacements: &[EntrySpec],
    identities: &[Adoption],
    ledger: &[Adoption],
) -> Result<KernelSelection, String> {
    let active = active(ledger);
    let fresh: BTreeMap<_, _> = identities.iter().map(|a| (a.symbol.as_str(), a)).collect();
    let mut entries = BTreeMap::new();
    for entry in replacements {
        if let Some(old) = entries.insert(entry.symbol.as_str(), entry)
            && identity::entry(old) != identity::entry(entry)
        {
            return Err(format!("conflicting generated body {}", entry.symbol));
        }
    }
    for spec in selection.specs() {
        for (position, entry) in spec.entries.iter().enumerate() {
            if let Some(record) = active.get(entry.symbol.as_str()) {
                let placement = record.placement.as_ref().ok_or("missing cubin placement")?;
                if placement.cubin != spec.name || placement.position != position {
                    return Err(format!("recorded placement moved {}", entry.symbol));
                }
            }
        }
    }
    let mut used = BTreeSet::new();
    let mut cubins = Vec::new();
    for baseline in selection.specs() {
        let mut slots = BTreeMap::new();
        for (&symbol, record) in &active {
            let place = record
                .placement
                .as_ref()
                .ok_or_else(|| format!("missing placement for {symbol}"))?;
            if place.cubin != baseline.name {
                continue;
            }
            let new = fresh
                .get(symbol)
                .ok_or_else(|| format!("missing generated identity {symbol}"))?;
            if new.generated != record.generated || new.entry_sha256 != record.entry_sha256 {
                return Err(format!(
                    "generated declaration differs from adoption {symbol}"
                ));
            }
            let entry = entries
                .get(symbol)
                .ok_or_else(|| format!("missing generated entry {symbol}"))?;
            if slots.insert(place.position, (*entry).clone()).is_some() {
                return Err("adoptions occupy the same cubin slot".into());
            }
            if let Some((position, old)) = baseline
                .entries
                .iter()
                .enumerate()
                .find(|(_, e)| e.symbol == symbol)
                && (position != place.position || identity::entry(old) != identity::entry(entry))
            {
                return Err(format!("locked position or body differs for {symbol}"));
            }
            used.insert(symbol);
        }
        let remaining = baseline
            .entries
            .iter()
            .filter(|entry| !active.contains_key(entry.symbol.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        let total = remaining.len() + slots.len();
        if slots.keys().any(|position| *position >= total) {
            return Err("adoption position exceeds cubin length".into());
        }
        let mut remaining = remaining.into_iter();
        let mut spec = baseline.clone();
        spec.entries = (0..total)
            .map(|position| {
                slots
                    .remove(&position)
                    .unwrap_or_else(|| remaining.next().expect("checked cubin slots"))
            })
            .collect();
        cubins.push(
            SelectedCubin::existing(spec, selection.declarations()[&baseline.name].clone())
                .map_err(|e| e.to_string())?,
        );
    }
    if used.len() != active.len() {
        return Err("adoption names a missing cubin".into());
    }
    KernelSelection::new(selection.context().clone(), cubins).map_err(|e| e.to_string())
}
