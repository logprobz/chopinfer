//! Batch resident hits without fetching, inserting or evicting a row.
use super::{ENTRY_META, Ordering, Plane, SHARDS, WAYS, put, word};

impl Plane {
    /// Fill complete resident rows at their original output positions. Return
    /// every unfilled position, sorted in input order, for the existing miss
    /// path. A contended shard is a miss here, as in the single-row lookup.
    /// The caller validates row IDs and keeps the mapping alive.
    #[allow(dead_code)]
    pub(crate) fn copy_resident_rows(&self, rows: &[u32], out: &mut [u8]) -> Vec<usize> {
        assert_eq!(Some(out.len()), rows.len().checked_mul(self.width));
        let cache = &self.cache;
        let mut groups: [Vec<(usize, usize)>; SHARDS] = std::array::from_fn(|_| Vec::new());
        for (position, &row) in rows.iter().enumerate() {
            let (shard, bucket) = cache.bucket(row);
            groups[shard].push((position, bucket));
        }
        let (offset, width) = cache.planes[self.index];
        let bit = 1_u64 << self.index;
        let mut misses = Vec::new();
        for (shard_index, group) in groups.iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            // Cache storage is locked memory. Neither the original mapping
            // nor a potentially faulting miss destination is read under this lock.
            let Ok(mut shard) = cache.shards[shard_index].try_lock() else {
                misses.extend(group.iter().map(|&(position, _)| position));
                continue;
            };
            for &(position, bucket) in group {
                let tag = u64::from(rows[position]) + 1;
                let bytes = shard.memory.bytes();
                let way = (0..WAYS).find(|&way| {
                    let meta = bucket + way * ENTRY_META;
                    word(bytes, meta) == tag && word(bytes, meta + 16) & bit != 0
                });
                let Some(way) = way else {
                    misses.push(position);
                    continue;
                };
                // Keep the single-row hit's exact copy and LRU update.
                shard.clock = shard.clock.saturating_add(1);
                let clock = shard.clock;
                let bytes = shard.memory.bytes();
                let from = bucket + WAYS * ENTRY_META + way * cache.row_bytes + offset;
                out[position * width..][..width].copy_from_slice(&bytes[from..from + width]);
                put(bytes, bucket + way * ENTRY_META + 8, clock);
            }
        }
        misses.sort_unstable();
        let hits = (rows.len() - misses.len()) as u64;
        cache.hits.fetch_add(hits, Ordering::Relaxed);
        let before = cache.copies.fetch_add(hits, Ordering::Relaxed);
        if before / 65536 != before.saturating_add(hits) / 65536 {
            cache.report("sample");
        }
        // Miss accounting belongs to the fallback, which may find a hit
        // inserted by another worker after this lookup.
        misses
    }
}
