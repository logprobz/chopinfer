//! Prepared images, by their source.
//!
//! A conversation sends every earlier image again with each turn, and
//! preparing one decodes, resizes and normalizes it. The cache returns the
//! item a source prepared before, payload included, so a repeated image costs
//! a hash of its source bytes. The key covers the source's MIME type, which
//! picks the decoder, and the recipe that prepared it.
//!
//! Its payloads keep their leases in the process media pool. An entry whose
//! payload no request holds is idle, and the pool reclaims idle entries, least
//! recently used first, when a reservation does not fit
//! ([`crate::media_memory::ReclaimV1`]).

use std::collections::HashMap;
use std::sync::Mutex;

use crate::generation::PreparedMediaInputV1;
use crate::media_memory::ReclaimV1;

#[derive(Default)]
pub(crate) struct PreparedImageCacheV1 {
    inner: Mutex<CacheInner>,
}

#[derive(Default)]
struct CacheInner {
    entries: HashMap<[u8; 32], (PreparedMediaInputV1, u64)>,
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
        inner.entries.get_mut(key).map(|(prepared, last_use)| {
            *last_use = tick;
            prepared.clone()
        })
    }

    pub(crate) fn insert(&self, key: [u8; 32], prepared: PreparedMediaInputV1) {
        let mut inner = self.lock();
        inner.tick += 1;
        let tick = inner.tick;
        inner.entries.insert(key, (prepared, tick));
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

impl ReclaimV1 for PreparedImageCacheV1 {
    fn reclaim_one(&self) -> bool {
        let removed = {
            let mut inner = self.lock();
            let idle = inner
                .entries
                .iter()
                .filter(|(_, (prepared, _))| !prepared.input.payload.is_shared())
                .min_by_key(|(_, (_, last_use))| *last_use)
                .map(|(key, _)| *key);
            idle.and_then(|key| inner.entries.remove(&key))
        };
        // Dropped without the cache's lock: its lease gives the bytes back
        // to the pool.
        removed.is_some()
    }
}
