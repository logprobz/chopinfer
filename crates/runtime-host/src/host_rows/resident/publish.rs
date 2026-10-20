//! Publish complete raw row planes with the existing replacement rule.
use super::*;

impl Plane {
    pub(super) fn store_full_row(&self, row: u32, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.width);
        let cache = &self.cache;
        let (shard, bucket) = cache.bucket(row);
        let tag = u64::from(row) + 1;
        let (offset, width) = cache.planes[self.index];
        if let Ok(mut shard) = cache.shards[shard].try_lock() {
            shard.clock = shard.clock.saturating_add(1);
            let clock = shard.clock;
            let target = shard.memory.bytes();
            let same = (0..WAYS).find(|&way| word(target, bucket + way * ENTRY_META) == tag);
            let way = same.unwrap_or_else(|| {
                (0..WAYS)
                    .min_by_key(|&way| {
                        let at = bucket + way * ENTRY_META;
                        if word(target, at) == 0 {
                            0
                        } else {
                            word(target, at + 8)
                        }
                    })
                    .expect("cache ways")
            });
            let meta = bucket + way * ENTRY_META;
            if same.is_none() {
                if word(target, meta) != 0 {
                    cache.evictions.fetch_add(1, Ordering::Relaxed);
                }
                put(target, meta, tag);
                put(target, meta + 16, 0);
            }
            let to = bucket + WAYS * ENTRY_META + way * cache.row_bytes + offset;
            target[to..to + width].copy_from_slice(bytes);
            put(target, meta + 8, clock);
            let valid = word(target, meta + 16);
            put(target, meta + 16, valid | (1_u64 << self.index));
            cache.insertions.fetch_add(1, Ordering::Relaxed);
        } else {
            cache.contention.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn publish_fetched(&self, row: u32, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.width);
        let cache = &self.cache;
        let (shard, bucket) = cache.bucket(row);
        let tag = u64::from(row) + 1;
        let bit = 1_u64 << self.index;
        // A duplicate or another gather may have filled this plane after
        // the bulk lookup. Account for that hit as the scalar path does.
        let hit = if let Ok(mut shard) = cache.shards[shard].try_lock() {
            shard.clock = shard.clock.saturating_add(1);
            let clock = shard.clock;
            let target = shard.memory.bytes();
            let way = (0..WAYS).find(|&way| {
                let meta = bucket + way * ENTRY_META;
                word(target, meta) == tag && word(target, meta + 16) & bit != 0
            });
            if let Some(way) = way {
                put(target, bucket + way * ENTRY_META + 8, clock);
            }
            way.is_some()
        } else {
            cache.contention.fetch_add(1, Ordering::Relaxed);
            false
        };
        if hit {
            cache.hits.fetch_add(1, Ordering::Relaxed);
        } else {
            cache.misses.fetch_add(1, Ordering::Relaxed);
            self.store_full_row(row, bytes);
        }
        if cache.copies.fetch_add(1, Ordering::Relaxed) % 65536 == 65535 {
            cache.report("sample");
        }
    }
}
