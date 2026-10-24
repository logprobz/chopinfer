//! Prepared images, by their source.
//!
//! A conversation sends every earlier image again with each turn, and
//! preparing one decodes, resizes and normalizes it. The cache returns the
//! item a source prepared before, payload included, so a repeated image costs
//! a hash of its source bytes. The key covers the source's MIME type, which
//! picks the decoder, and the recipe that prepared it.
//!
//! An image whose recipe normalizes an 8-bit canvas is kept as that canvas,
//! a quarter of its BF16 payload and an eighth of its F32 one.
//!
//! Its payloads keep their leases in the process media pool. An entry whose
//! payload no request holds is idle, and the pool reclaims idle entries, least
//! recently used first, when a reservation does not fit
//! ([`crate::media_memory::ReclaimV1`]). The pool bounds the payload bytes;
//! [`MAX_ENTRIES`] bounds the entries, so many small images cannot grow the
//! map without limit: keeping another drops the least recently used idle one.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::generation::PreparedMediaInputV1;
use crate::media_memory::ReclaimV1;

/// The most entries kept with none held by a request; entries that requests
/// hold are kept beyond it until they are idle.
pub(crate) const MAX_ENTRIES: usize = 4096;

#[derive(Default)]
pub(crate) struct PreparedImageCacheV1 {
    inner: Mutex<CacheInner>,
}

#[derive(Default)]
struct CacheInner {
    /// Each entry's item, last use and the pool bytes its payload keeps.
    entries: HashMap<[u8; 32], (PreparedMediaInputV1, u64, u64)>,
    tick: u64,
}

impl PreparedImageCacheV1 {
    /// The key of the image `source` prepares to under `modality`'s recipe.
    pub(crate) fn key(
        mime: &str,
        modality: &runtime_frontend::FrontendMediaModalityV1,
        source: &[u8],
    ) -> [u8; 32] {
        let preparation = serde_json::json!({
            "mime": mime,
            "recipe": modality.recipe.tag,
            "parameters": modality.recipe.parameters,
            "encoder": modality.encoder,
            "maximum_tokens": modality.maximum_tokens,
        });
        runtime_media::content_key(
            "chopinfer.prepared-image.v1",
            &serde_json::to_vec(&preparation).unwrap_or_default(),
            source,
        )
    }

    pub(crate) fn get(&self, key: &[u8; 32]) -> Option<PreparedMediaInputV1> {
        let mut inner = self.lock();
        inner.tick += 1;
        let tick = inner.tick;
        inner.entries.get_mut(key).map(|(prepared, last_use, _)| {
            *last_use = tick;
            prepared.clone()
        })
    }

    /// Keeps `prepared`, whose payload holds `retained` pool bytes, dropping
    /// the least recently used idle entry when the cache is full.
    pub(crate) fn insert(&self, key: [u8; 32], prepared: PreparedMediaInputV1, retained: u64) {
        let dropped = {
            let mut inner = self.lock();
            inner.tick += 1;
            let tick = inner.tick;
            let dropped = if inner.entries.len() >= MAX_ENTRIES && !inner.entries.contains_key(&key)
            {
                inner.remove_least_recently_used_idle()
            } else {
                None
            };
            inner.entries.insert(key, (prepared, tick, retained));
            dropped
        };
        // Dropped without the cache's lock: its lease gives the bytes back
        // to the pool.
        drop(dropped);
    }

    /// The entries no request holds and the pool bytes they keep.
    pub(crate) fn idle(&self) -> (usize, u64) {
        self.lock()
            .entries
            .values()
            .filter(|(prepared, _, _)| !prepared.input.payload.is_shared())
            .fold((0, 0), |(entries, bytes), (_, _, retained)| {
                (entries + 1, bytes + retained)
            })
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Drops every idle entry, as a short pool would one at a time.
    #[cfg(test)]
    pub(crate) fn reclaim_idle(&self) {
        while self.reclaim_one() {}
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, CacheInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl CacheInner {
    fn remove_least_recently_used_idle(&mut self) -> Option<(PreparedMediaInputV1, u64, u64)> {
        let idle = self
            .entries
            .iter()
            .filter(|(_, (prepared, _, _))| !prepared.input.payload.is_shared())
            .min_by_key(|(_, (_, last_use, _))| *last_use)
            .map(|(key, _)| *key);
        idle.and_then(|key| self.entries.remove(&key))
    }
}

impl ReclaimV1 for PreparedImageCacheV1 {
    fn reclaim_one(&self) -> bool {
        let removed = self.lock().remove_least_recently_used_idle();
        // Dropped without the cache's lock: its lease gives the bytes back
        // to the pool.
        removed.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_ENTRIES, PreparedImageCacheV1};
    use crate::generation::PreparedMediaInputV1;
    use runtime_server::MediaInputV1;

    fn key(index: usize) -> [u8; 32] {
        let mut key = [0; 32];
        key[..8].copy_from_slice(&(index as u64).to_le_bytes());
        key
    }

    fn image(index: usize) -> PreparedMediaInputV1 {
        PreparedMediaInputV1::from(MediaInputV1 {
            key: key(index),
            encoder: 0,
            geometry: [1, 1, 1, 1],
            output_rows: 1,
            payload: vec![index as u8; 16].into(),
        })
    }

    #[test]
    fn a_full_cache_drops_its_least_recently_used_idle_entry() {
        let cache = PreparedImageCacheV1::default();
        for index in 0..MAX_ENTRIES {
            cache.insert(key(index), image(index), 16);
        }
        // A request holds the oldest entry and has just used the second.
        let held = cache.get(&key(0)).unwrap();
        assert!(cache.get(&key(1)).is_some());
        for index in MAX_ENTRIES..MAX_ENTRIES + 3 {
            cache.insert(key(index), image(index), 16);
        }
        assert_eq!(cache.len(), MAX_ENTRIES);
        assert!(cache.get(&key(0)).is_some(), "a held entry stays");
        assert!(cache.get(&key(1)).is_some(), "a recent entry stays");
        for dropped in 2..5 {
            assert!(cache.get(&key(dropped)).is_none(), "entry {dropped}");
        }
        assert!(cache.get(&key(5)).is_some());
        assert!(cache.get(&key(MAX_ENTRIES + 2)).is_some());
        drop(held);
    }
}
