//! The device bytes each rank allocates at create, by category, logged once
//! per rank as a `startup memory` JSON line. The host sizes the serving KV
//! pool from the memory create leaves free, so the line attributes what is
//! not KV. `after_kv` counts what create allocated once the host had sized
//! the pool, which comes out of the host's reserve instead.

use std::collections::BTreeMap;

use bundle_runtime::resources::allocated_bytes;
use runtime_bundle_abi::BundleMemoryPoolHandleV1;

#[derive(Debug, Default)]
pub(super) struct Ledger {
    pools: Vec<BundleMemoryPoolHandleV1>,
    /// Each rank's allocated bytes when the ledger opened, when the host
    /// sized the KV pool, and when the last category was attributed.
    opened: Vec<u64>,
    sized: Option<Vec<u64>>,
    marks: Vec<u64>,
    categories: Vec<BTreeMap<&'static str, u64>>,
    /// A breakdown of one category per rank, such as scratch by region.
    details: Vec<BTreeMap<&'static str, Vec<(&'static str, u64)>>>,
}

impl Ledger {
    /// A ledger of the ranks allocating from `pools`, in rank order.
    pub(super) fn open(pools: impl IntoIterator<Item = BundleMemoryPoolHandleV1>) -> Self {
        let pools: Vec<_> = pools.into_iter().collect();
        let opened: Vec<_> = pools.iter().map(|&pool| allocated_bytes(pool)).collect();
        Self {
            categories: vec![BTreeMap::new(); pools.len()],
            details: vec![BTreeMap::new(); pools.len()],
            marks: opened.clone(),
            opened,
            sized: None,
            pools,
        }
    }

    /// Attributes to `category` what each rank allocated since the last
    /// attribution.
    pub(super) fn take(&mut self, category: &'static str) {
        for rank in 0..self.pools.len() {
            let now = allocated_bytes(self.pools[rank]);
            let bytes = now.saturating_sub(self.marks[rank]);
            self.marks[rank] = self.marks[rank].max(now);
            if bytes > 0 {
                *self.categories[rank].entry(category).or_default() += bytes;
            }
        }
    }

    /// Attributes to `category` the `bytes` that rank `rank` allocated since
    /// the last attribution, as the allocating step reported them. A later
    /// `take` attributes any bytes beyond them.
    pub(super) fn add(&mut self, rank: usize, category: &'static str, bytes: u64) {
        if bytes > 0 {
            *self.categories[rank].entry(category).or_default() += bytes;
            self.marks[rank] += bytes;
        }
    }

    /// Breaks rank `rank`'s `category` down by `parts`.
    pub(super) fn detail(
        &mut self,
        rank: usize,
        category: &'static str,
        parts: Vec<(&'static str, u64)>,
    ) {
        self.details[rank].insert(category, parts);
    }

    /// Marks the host's sizing of the serving KV pool.
    pub(super) fn sized(&mut self) {
        self.take("other");
        self.sized = Some(self.marks.clone());
    }

    /// Each rank's JSON line: its rank, total, bytes allocated after the
    /// KV pool was sized, and bytes by category.
    pub(super) fn lines(&mut self) -> Vec<String> {
        self.take("other");
        (0..self.pools.len())
            .map(|rank| {
                let total = self.marks[rank] - self.opened[rank];
                let after = self
                    .sized
                    .as_ref()
                    .map_or(0, |sized| self.marks[rank] - sized[rank]);
                let object = |parts: &mut dyn Iterator<Item = (&str, u64)>| {
                    let parts: Vec<_> = parts
                        .map(|(name, bytes)| format!("\"{name}\":{bytes}"))
                        .collect();
                    format!("{{{}}}", parts.join(","))
                };
                let mut line = format!(
                    "{{\"rank\":{rank},\"total\":{total},\"after_kv\":{after},\"bytes\":{}",
                    object(&mut self.categories[rank].iter().map(|(&k, &v)| (k, v)))
                );
                for (category, parts) in &self.details[rank] {
                    line += &format!(
                        ",\"{category}_parts\":{}",
                        object(&mut parts.iter().copied())
                    );
                }
                line + "}"
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_json_objects_by_rank() {
        let mut ledger = Ledger::open([
            BundleMemoryPoolHandleV1::from_parts(9, 1),
            BundleMemoryPoolHandleV1::from_parts(9, 2),
        ]);
        ledger.take("inputs");
        ledger.add(1, "scratch", 7);
        ledger.detail(1, "scratch", vec![("attention", 3), ("other", 4)]);
        ledger.sized();
        ledger.add(0, "overlap", 8);
        let lines = ledger.lines();
        assert_eq!(
            lines,
            [
                "{\"rank\":0,\"total\":8,\"after_kv\":8,\"bytes\":{\"overlap\":8}}",
                "{\"rank\":1,\"total\":7,\"after_kv\":0,\"bytes\":{\"scratch\":7},\"scratch_parts\":{\"attention\":3,\"other\":4}}",
            ]
        );
        for line in lines {
            serde_json::from_str::<serde_json::Value>(&line).unwrap();
        }
    }
}
