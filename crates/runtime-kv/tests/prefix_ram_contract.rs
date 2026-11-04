//! The RAM prefix tier across both tiers: copies in flight meeting GPU and
//! host evictions and logical events, failure and quarantine, pending
//! anchors and their promotion, the demotion headroom rule, and a seeded
//! model check. Both audits run after every operation, and a byte model of
//! device and host pages checks what every copy moved. Each capture writes
//! snapshot bytes of its own, and the model keeps the full bytes of every
//! page each capture, demotion and promotion took: a GPU anchor holds its
//! key's latest capture, a host anchor what its copy found at plan time,
//! and a usable host entry or a publish the latest capture again.
use std::collections::BTreeMap;
use std::sync::Arc;

use runtime_kv::{
    BlockId, HostAnchorId, HostPageId, HostTier, KvError, KvGeometry, KvPool, KvTransferClass,
    KvTransferDirection, KvTransferId, KvTransferPlan, KvTransferStatus, KvTransferSubmission,
    PendingAnchorId, PrefixCacheKeyV1, PrefixChainV1, PrefixHostInvalidationV1,
    PrefixPendingCopyV1, PrefixPublishV1, SequenceHandle,
};

const BLOCK: usize = 4;
/// Snapshot pages per anchor.
const STATE: usize = 1;
/// The bytes of a row no write or completed copy defined.
const STALE: u32 = u32::MAX;

type Page = [u32; BLOCK];

fn chain(tokens: &[u32]) -> PrefixChainV1 {
    PrefixChainV1::new(BLOCK, Arc::from(tokens), Vec::new()).unwrap()
}

fn counting(n: u32, len: u32) -> PrefixChainV1 {
    chain(&(n * 1000..n * 1000 + len).collect::<Vec<_>>())
}

/// Checks an anchor's full page bytes, KV pages then snapshot pages,
/// against its prompt: committed rows hold the prompt's tokens and each
/// snapshot page one capture's bytes.
fn assert_rows(prompt: &PrefixChainV1, tokens: usize, bytes: &[Page], what: &str) {
    let kv = tokens.div_ceil(BLOCK);
    assert_eq!(bytes.len(), kv + STATE, "{what} pages");
    for (position, token) in prompt.tokens()[..tokens].iter().enumerate() {
        assert_eq!(
            bytes[position / BLOCK][position % BLOCK],
            *token,
            "{what} row {position}"
        );
    }
    assert!(
        bytes[kv..]
            .iter()
            .all(|page| page[0] != STALE && page.iter().all(|row| *row == page[0])),
        "{what} snapshot"
    );
}

/// A live sequence and the prompt whose rows it holds.
struct Live {
    handle: SequenceHandle,
    prompt: PrefixChainV1,
    tokens: usize,
}

/// Both tiers, an executor holding submitted copies, and the bytes of
/// every page.
struct Tiers {
    pool: KvPool,
    tier: HostTier,
    device: BTreeMap<BlockId, Page>,
    host: BTreeMap<HostPageId, Page>,
    copies: BTreeMap<KvTransferId, KvTransferSubmission>,
    /// The prompt and tokens of every public key captured.
    keys: BTreeMap<PrefixCacheKeyV1, (PrefixChainV1, usize)>,
    /// The snapshot bytes of the last capture.
    stamp: u32,
    /// The full page bytes of each key's latest capture.
    latest: BTreeMap<PrefixCacheKeyV1, Vec<Page>>,
    /// The full source page bytes each host anchor's copy out found when it
    /// was planned.
    copied: BTreeMap<HostAnchorId, Vec<Page>>,
    /// The full source page bytes each promotion found when it was planned.
    promoted: BTreeMap<PendingAnchorId, Vec<Page>>,
    /// Pending anchors published or abandoned, whose ids must stay dead.
    retired: Vec<PendingAnchorId>,
}

impl Tiers {
    fn new(pool: KvPool, host_pages: usize) -> Self {
        let tier = HostTier::new(&pool, host_pages).unwrap();
        let tiers = Self {
            pool,
            tier,
            device: BTreeMap::new(),
            host: BTreeMap::new(),
            copies: BTreeMap::new(),
            keys: BTreeMap::new(),
            stamp: 0,
            latest: BTreeMap::new(),
            copied: BTreeMap::new(),
            promoted: BTreeMap::new(),
            retired: Vec::new(),
        };
        tiers.check();
        tiers
    }

    fn ram(pages: usize, host_pages: usize) -> Self {
        let pool = KvPool::new(KvGeometry::new(BLOCK, pages).unwrap(), 8)
            .unwrap()
            .with_prefix_ram(true);
        Self::new(pool, host_pages)
    }

    fn device(&self, page: BlockId) -> Page {
        self.device.get(&page).copied().unwrap_or([STALE; BLOCK])
    }

    fn host(&self, page: HostPageId) -> Page {
        self.host.get(&page).copied().unwrap_or([STALE; BLOCK])
    }

    fn free(&self) -> usize {
        self.pool.stats().free_blocks()
    }

    /// The full bytes of `key`'s GPU anchor pages, KV then snapshot.
    fn anchor_bytes(&self, key: PrefixCacheKeyV1) -> Option<Vec<Page>> {
        let (_, blocks, state) = self.pool.anchor_pages(key)?;
        Some(
            blocks
                .iter()
                .chain(&state)
                .map(|page| self.device(*page))
                .collect(),
        )
    }

    /// The full bytes of a host anchor's pages, KV then snapshot.
    fn host_anchor_bytes(&self, host: HostAnchorId) -> Option<Vec<Page>> {
        let anchor = self.tier.anchor(host)?;
        Some(
            anchor
                .blocks
                .iter()
                .chain(&anchor.state)
                .map(|page| self.host(*page))
                .collect(),
        )
    }

    /// Both audits, no retired pending anchor id answering, every GPU
    /// anchor holding its key's latest capture, every ready host anchor
    /// holding what its copy out found, and every usable host entry holding
    /// its key's latest capture.
    fn check(&self) {
        self.pool.audit().unwrap();
        self.tier.audit(&self.pool).unwrap();
        for id in &self.retired {
            assert_eq!(self.pool.pending_anchor(*id), None, "retired {id:?}");
        }
        for (key, (prompt, tokens)) in &self.keys {
            if let Some(bytes) = self.anchor_bytes(*key) {
                assert_rows(prompt, *tokens, &bytes, "GPU anchor");
                assert_eq!(Some(&bytes), self.latest.get(key), "GPU anchor {key:?}");
            }
            let usable = self
                .pool
                .host_entry(*key)
                .filter(|entry| entry.ready && self.pool.host_invalidation(*key).is_none());
            if let Some(entry) = usable {
                assert_eq!(
                    self.copied.get(&entry.host),
                    self.latest.get(key),
                    "usable host entry {key:?}"
                );
            }
        }
        for (host, bytes) in &self.copied {
            if self.tier.anchor(*host).is_some_and(|anchor| anchor.ready) {
                assert_eq!(
                    self.host_anchor_bytes(*host).as_ref(),
                    Some(bytes),
                    "host anchor {host:?}"
                );
            }
        }
    }

    /// Appends rows of the sequence's prompt, after its tail copy if any;
    /// new pages start stale.
    fn append(&mut self, live: &mut Live, more: usize) -> Result<(), KvError> {
        let report = self.pool.append(live.handle, more);
        self.check();
        let report = report?;
        for page in report.allocated() {
            self.device.insert(*page, [STALE; BLOCK]);
        }
        if let Some(copy) = report.copy() {
            let source = self.device(copy.source());
            self.device.insert(copy.destination(), source);
        }
        let blocks = self.pool.snapshot(live.handle).unwrap().into_blocks();
        for position in live.tokens..live.tokens + more {
            self.device
                .get_mut(&blocks[position / BLOCK])
                .expect("written pages are allocated")[position % BLOCK] =
                live.prompt.tokens()[position];
        }
        live.tokens += more;
        self.check();
        Ok(())
    }

    /// A cold sequence holding the first `tokens` rows of `prompt`.
    fn start(&mut self, prompt: &PrefixChainV1, tokens: usize) -> Result<Live, KvError> {
        let handle = self.pool.create_sequence(prompt.tokens().len())?;
        let mut live = Live {
            handle,
            prompt: prompt.clone(),
            tokens: 0,
        };
        if let Err(error) = self.append(&mut live, tokens) {
            self.release(live);
            return Err(error);
        }
        Ok(live)
    }

    fn release(&mut self, live: Live) {
        self.pool.release(live.handle).unwrap();
        self.check();
    }

    /// Captures `live` at `depth` and writes snapshot bytes of its own,
    /// returning the key when a new anchor was made.
    fn capture(&mut self, live: &Live, depth: usize) -> Option<PrefixCacheKeyV1> {
        let key = live.prompt.key_at(depth).unwrap();
        let state = self
            .pool
            .insert_content_anchor(&live.prompt, live.handle, depth, STATE)
            .unwrap()?;
        self.stamp += 1;
        for page in state {
            self.device.insert(page, [self.stamp; BLOCK]);
        }
        let bytes = self.anchor_bytes(key).unwrap();
        assert_rows(&live.prompt, depth, &bytes, "capture");
        self.keys.insert(key, (live.prompt.clone(), depth));
        self.latest.insert(key, bytes);
        self.check();
        Some(key)
    }

    /// A cold sequence over `prompt`, captured at each depth and released.
    fn capture_prompt(&mut self, prompt: &PrefixChainV1, depths: &[usize]) {
        let live = self.start(prompt, *depths.iter().max().unwrap()).unwrap();
        for depth in depths {
            self.capture(&live, *depth).unwrap();
        }
        self.release(live);
    }

    /// Keeps the full source bytes a copy out found when it was planned.
    fn copy_out(&mut self, host: HostAnchorId, plan: KvTransferPlan) -> KvTransferId {
        let found = plan
            .pages()
            .iter()
            .map(|pair| self.device(pair.device))
            .collect();
        self.copied.insert(host, found);
        self.submit(plan)
    }

    fn submit(&mut self, plan: KvTransferPlan) -> KvTransferId {
        let id = plan.id();
        self.copies.insert(id, plan.into_submission());
        self.check();
        id
    }

    /// Completes a copy: a success moves its bytes, and any other status
    /// leaves stale bytes wherever it may have written.
    fn complete(&mut self, id: KvTransferId, status: KvTransferStatus) {
        let copy = self.copies.remove(&id).unwrap();
        let success = status == KvTransferStatus::Complete;
        for pair in copy.pages() {
            match copy.direction() {
                KvTransferDirection::Out => {
                    let bytes = if success {
                        self.device(pair.device)
                    } else {
                        [STALE; BLOCK]
                    };
                    self.host.insert(pair.host, bytes);
                }
                KvTransferDirection::In => {
                    let bytes = if success {
                        self.host(pair.host)
                    } else {
                        [STALE; BLOCK]
                    };
                    self.device.insert(pair.device, bytes);
                }
            }
        }
        self.tier
            .complete_transfer(&mut self.pool, copy.completion(status))
            .unwrap();
        self.check();
    }

    fn demote(&mut self, key: PrefixCacheKeyV1) -> Result<(HostAnchorId, KvTransferId), KvError> {
        let result = self.tier.begin_offload(&mut self.pool, key, None);
        self.check();
        let (host, plan) = result?;
        assert_eq!(plan.class(), KvTransferClass::Demotion);
        let copy = self.copy_out(host, plan);
        assert_eq!(
            self.copied.get(&host),
            self.latest.get(&key),
            "demotion source"
        );
        Ok((host, copy))
    }

    /// Demotes `key`, completes the copy and removes the GPU anchor.
    fn move_to_host(&mut self, key: PrefixCacheKeyV1) {
        let (_, copy) = self.demote(key).unwrap();
        self.complete(copy, KvTransferStatus::Complete);
        assert!(self.remove(key));
    }

    fn promote(
        &mut self,
        key: PrefixCacheKeyV1,
    ) -> Result<(PendingAnchorId, KvTransferId), KvError> {
        let pending = self.pool.create_pending_anchor(key, STATE);
        self.check();
        let pending = pending?;
        match self.start_promotion(pending) {
            Ok(copy) => Ok((pending, copy)),
            Err(error) => {
                self.abandon(pending);
                Err(error)
            }
        }
    }

    /// Promotes a pending anchor's source entry, keeping the full source
    /// bytes the promotion found, which must be what the entry's copy out
    /// found.
    fn start_promotion(&mut self, pending: PendingAnchorId) -> Result<KvTransferId, KvError> {
        let source = self.pool.host_entry(pending.key()).map(|entry| entry.host);
        let plan = self.tier.begin_promotion(&mut self.pool, pending);
        self.check();
        let plan = plan?;
        assert_eq!(plan.class(), KvTransferClass::Promotion);
        assert_eq!(plan.direction(), KvTransferDirection::In);
        let found: Vec<_> = plan
            .pages()
            .iter()
            .map(|pair| self.host(pair.host))
            .collect();
        assert_eq!(
            source.and_then(|host| self.copied.get(&host)),
            Some(&found),
            "promotion source"
        );
        self.promoted.insert(pending, found);
        // Fresh pages hold whatever they held before.
        for pair in plan.pages() {
            self.device.insert(pair.device, [STALE; BLOCK]);
        }
        Ok(self.submit(plan))
    }

    fn publish(&mut self, pending: PendingAnchorId) -> PrefixPublishV1 {
        let key = pending.key();
        let prompt = self.keys[&key].0.clone();
        let outcome = self.pool.publish_pending_anchor(pending, &prompt).unwrap();
        self.retired.push(pending);
        let found = self.promoted.remove(&pending);
        if outcome == PrefixPublishV1::Published {
            // The anchor holds what the promotion found, and that is the
            // key's latest capture: a publish never resurrects an older one.
            assert_eq!(self.anchor_bytes(key), found, "published anchor");
            assert_eq!(found.as_ref(), self.latest.get(&key), "published capture");
        }
        self.check();
        outcome
    }

    fn abandon(&mut self, pending: PendingAnchorId) {
        self.pool.abandon_pending_anchor(pending).unwrap();
        self.retired.push(pending);
        self.promoted.remove(&pending);
        self.check();
    }

    /// A retired attempt's id answers to nothing: a lookup, an abandon, a
    /// publish and a promotion all refuse it.
    fn assert_retired(&mut self, id: PendingAnchorId) {
        let prompt = self.keys[&id.key()].0.clone();
        assert_eq!(self.pool.pending_anchor(id), None);
        assert_eq!(
            self.pool.abandon_pending_anchor(id),
            Err(KvError::StaleHandle)
        );
        assert_eq!(
            self.pool.publish_pending_anchor(id, &prompt),
            Err(KvError::StaleHandle)
        );
        assert_eq!(
            self.tier.begin_promotion(&mut self.pool, id).map(|_| ()),
            Err(KvError::StaleHandle)
        );
        self.check();
    }

    /// Resumes `key`'s GPU anchor, checks the bytes the new sequence starts
    /// from, and finishes its restore.
    fn resume(&mut self, key: PrefixCacheKeyV1) -> Live {
        let (prompt, tokens) = self.keys[&key].clone();
        let hit = self.pool.resume_anchor(key, prompt.tokens().len()).unwrap();
        let blocks = self.pool.snapshot(hit.handle()).unwrap().into_blocks();
        let bytes: Vec<_> = blocks
            .iter()
            .chain(hit.state())
            .map(|page| self.device(*page))
            .collect();
        assert_rows(&prompt, tokens, &bytes, "resumed sequence");
        assert_eq!(Some(&bytes), self.latest.get(&key), "resumed sequence");
        self.pool.finish_restore(hit.handle()).unwrap();
        self.check();
        Live {
            handle: hit.handle(),
            prompt,
            tokens,
        }
    }

    fn drop_invalidated(&mut self) -> Vec<(PrefixCacheKeyV1, PrefixHostInvalidationV1)> {
        let dropped = self.tier.drop_invalidated_entries(&mut self.pool).unwrap();
        self.check();
        dropped
    }

    fn drop_entry(&mut self, key: PrefixCacheKeyV1) -> bool {
        let dropped = self.tier.drop_public_entry(&mut self.pool, key).unwrap();
        self.check();
        dropped
    }

    fn remove(&mut self, key: PrefixCacheKeyV1) -> bool {
        let removed = self.pool.remove_anchor(key).unwrap();
        self.check();
        removed
    }

    /// Allocates a scratch sequence of `tokens` and releases it.
    fn pressure(&mut self, tokens: usize) -> Result<(), KvError> {
        let handle = self.pool.create_sequence(tokens)?;
        let result = self.pool.append(handle, tokens);
        if let Ok(report) = &result {
            for page in report.allocated() {
                self.device.insert(*page, [STALE; BLOCK]);
            }
        }
        self.pool.release(handle).unwrap();
        self.check();
        result.map(|_| ())
    }
}

#[test]
fn copies_survive_a_gpu_eviction_during_demotion_and_a_host_eviction_during_promotion() {
    let prompt = counting(1, 12);
    let key = prompt.key_at(8).unwrap();
    let mut tiers = Tiers::ram(32, 16);
    tiers.capture_prompt(&prompt, &[8]);
    let (_, copy) = tiers.demote(key).unwrap();
    // The GPU anchor goes, but its leased pages stay until the copy
    // completes.
    assert!(tiers.remove(key));
    assert_eq!(tiers.free(), 29);
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(tiers.free(), 32);
    assert!(tiers.pool.host_entry(key).unwrap().ready);
    let (pending, copy) = tiers.promote(key).unwrap();
    // The host tier drops the source entry, and the promotion goes on: the
    // copy holds the host pages until it completes.
    assert!(tiers.drop_entry(key));
    assert_eq!(tiers.tier.used_pages(), 3);
    assert_eq!(tiers.pool.lookup_host_anchor(&prompt, 12), None);
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(tiers.tier.used_pages(), 0);
    assert_eq!(
        tiers.pool.pending_anchor(pending).unwrap().copy,
        PrefixPendingCopyV1::Copied
    );
    assert_eq!(tiers.publish(pending), PrefixPublishV1::Published);
    assert_eq!(tiers.pool.lookup_anchor(&prompt, 12), Some((key, 8)));
    let live = tiers.resume(key);
    tiers.release(live);
}

#[test]
fn touches_and_recaptures_drop_a_shadow_while_its_copy_is_in_flight() {
    use PrefixHostInvalidationV1::{Recaptured, Resumed};
    let prompt = counting(2, 12);
    let key = prompt.key_at(8).unwrap();
    for reason in [Resumed, Recaptured] {
        let mut tiers = Tiers::ram(32, 16);
        tiers.capture_prompt(&prompt, &[8]);
        let (_, copy) = tiers.demote(key).unwrap();
        if reason == Resumed {
            let live = tiers.resume(key);
            tiers.release(live);
        } else {
            let live = tiers.start(&prompt, 8).unwrap();
            assert_eq!(tiers.capture(&live, 8), None);
            tiers.release(live);
        }
        assert_eq!(tiers.pool.host_invalidation(key), Some(reason));
        assert_eq!(tiers.drop_invalidated(), [(key, reason)]);
        assert!(tiers.pool.host_entry(key).is_none());
        // The copy finishes under its references and is discarded.
        tiers.complete(copy, KvTransferStatus::Complete);
        assert_eq!(tiers.tier.used_pages(), 0);
        assert!(tiers.pool.host_entry(key).is_none() && tiers.pool.has_anchor(key));
    }
    // A recapture before a promotion's copy starts leaves nothing to copy,
    // and one during the copy invalidates it; the new GPU anchor stays.
    let mut tiers = Tiers::ram(32, 16);
    tiers.capture_prompt(&prompt, &[8]);
    tiers.move_to_host(key);
    let pending = tiers.pool.create_pending_anchor(key, STATE).unwrap();
    tiers.capture_prompt(&prompt, &[8]);
    assert_eq!(
        tiers
            .tier
            .begin_promotion(&mut tiers.pool, pending)
            .map(|_| ()),
        Err(KvError::StaleHandle)
    );
    tiers.abandon(pending);
    assert_eq!(tiers.drop_invalidated(), [(key, Recaptured)]);
    tiers.move_to_host(key);
    let (pending, copy) = tiers.promote(key).unwrap();
    tiers.capture_prompt(&prompt, &[8]);
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(
        tiers.publish(pending),
        PrefixPublishV1::Invalidated(Recaptured)
    );
    assert!(tiers.pool.has_anchor(key));
    assert_eq!(tiers.drop_invalidated(), [(key, Recaptured)]);
    assert_eq!(tiers.free(), 29);
}

#[test]
fn a_failed_promotion_releases_its_pending_pages_only_after_completion() {
    let prompt = counting(3, 12);
    let key = prompt.key_at(8).unwrap();
    let mut tiers = Tiers::ram(32, 16);
    tiers.capture_prompt(&prompt, &[8]);
    tiers.move_to_host(key);
    let (pending, copy) = tiers.promote(key).unwrap();
    assert_eq!(tiers.free(), 29);
    // Neither an abandon nor a publish frees pages the copy may still write.
    assert_eq!(
        tiers.pool.abandon_pending_anchor(pending),
        Err(KvError::StaleHandle)
    );
    assert_eq!(
        tiers.pool.publish_pending_anchor(pending, &prompt),
        Err(KvError::StaleHandle)
    );
    tiers.check();
    tiers.complete(copy, KvTransferStatus::FailedIdle);
    assert_eq!(
        tiers.pool.pending_anchor(pending).unwrap().copy,
        PrefixPendingCopyV1::Failed
    );
    assert_eq!(tiers.free(), 29);
    assert_eq!(tiers.publish(pending), PrefixPublishV1::NotCopied);
    assert_eq!(tiers.free(), 32);
    assert!(!tiers.pool.has_anchor(key));
    // The source entry stays, since copying in never writes host pages, and
    // a later promotion of it succeeds.
    assert!(tiers.pool.host_entry(key).unwrap().ready);
    let (pending, copy) = tiers.promote(key).unwrap();
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(tiers.publish(pending), PrefixPublishV1::Published);
}

#[test]
fn an_unconfirmed_status_quarantines_the_transfer() {
    let prompts = [counting(4, 12), counting(5, 12)];
    let keys = prompts.clone().map(|prompt| prompt.key_at(8).unwrap());
    let mut tiers = Tiers::ram(32, 16);
    let quarantine = |tiers: &mut Tiers, copy: KvTransferId| {
        let submission = tiers.copies.remove(&copy).unwrap();
        tiers
            .tier
            .complete_transfer(
                &mut tiers.pool,
                submission.completion(KvTransferStatus::Unconfirmed),
            )
            .unwrap();
        tiers.check();
        for status in [KvTransferStatus::Complete, KvTransferStatus::FailedIdle] {
            assert!(
                tiers
                    .tier
                    .complete_transfer(&mut tiers.pool, submission.completion(status))
                    .is_err()
            );
        }
        tiers.check();
    };
    // A demotion keeps its lease, its host pages and its unready entry.
    tiers.capture_prompt(&prompts[0], &[8]);
    let (_, copy) = tiers.demote(keys[0]).unwrap();
    quarantine(&mut tiers, copy);
    assert!(tiers.remove(keys[0]));
    assert_eq!(tiers.free(), 29);
    assert!(!tiers.pool.host_entry(keys[0]).unwrap().ready);
    assert_eq!(tiers.tier.used_pages(), 3);
    // A promotion keeps its pending anchor's pages and its source's host
    // pages, through the entry's drop.
    tiers.capture_prompt(&prompts[1], &[8]);
    tiers.move_to_host(keys[1]);
    let (pending, copy) = tiers.promote(keys[1]).unwrap();
    quarantine(&mut tiers, copy);
    assert!(tiers.drop_entry(keys[1]));
    assert!(matches!(
        tiers.pool.pending_anchor(pending).unwrap().copy,
        PrefixPendingCopyV1::InFlight(_)
    ));
    assert_eq!(
        tiers.pool.abandon_pending_anchor(pending),
        Err(KvError::StaleHandle)
    );
    assert_eq!(
        tiers.pool.publish_pending_anchor(pending, &prompts[1]),
        Err(KvError::StaleHandle)
    );
    tiers.check();
    assert_eq!(tiers.free(), 26);
    assert_eq!(tiers.tier.used_pages(), 6);
    assert_eq!(tiers.tier.pending_transfers(), 2);
}

#[test]
fn a_shared_tail_appended_in_place_is_copied_again_by_a_new_demotion() {
    let prompt = counting(6, 12);
    let (short, long) = (prompt.key_at(6).unwrap(), prompt.key_at(8).unwrap());
    let mut tiers = Tiers::ram(32, 16);
    let mut live = tiers.start(&prompt, 6).unwrap();
    tiers.capture(&live, 6).unwrap();
    let tail = tiers.pool.snapshot(live.handle).unwrap().blocks()[1];
    let allocation = tiers.pool.page_allocation(tail).unwrap();
    let (old, copy) = tiers.demote(short).unwrap();
    let old_pairs = tiers.copies[&copy].pages().to_vec();
    tiers.complete(copy, KvTransferStatus::Complete);
    // Once the GPU anchor drops, the sequence alone holds its tail page and
    // appends into it in place, under the same allocation.
    assert!(tiers.remove(short));
    assert_eq!(tiers.pool.block_ref_count(tail).unwrap(), 1);
    tiers.append(&mut live, 2).unwrap();
    assert_eq!(tiers.pool.snapshot(live.handle).unwrap().blocks()[1], tail);
    assert_eq!(tiers.pool.page_allocation(tail).unwrap(), allocation);
    // The host copy keeps the old bytes.
    let old_tail = tiers.tier.anchor(old).unwrap().blocks[1];
    assert_eq!(tiers.host(old_tail)[2..], [STALE; 2]);
    assert_eq!(tiers.device(tail)[2..], prompt.tokens()[6..8]);
    // A new demotion over the same allocations copies every page again,
    // into host pages of its own.
    tiers.capture(&live, 8).unwrap();
    tiers.release(live);
    let (_, copy) = tiers.demote(long).unwrap();
    let pairs = tiers.copies[&copy].pages().to_vec();
    let (_, blocks, state) = tiers.pool.anchor_pages(long).unwrap();
    assert_eq!(
        pairs.iter().map(|pair| pair.device).collect::<Vec<_>>(),
        [blocks, state].concat()
    );
    assert!(pairs[1].device == tail && old_pairs[1].device == tail);
    assert!(
        pairs
            .iter()
            .all(|pair| old_pairs.iter().all(|old| old.host != pair.host))
    );
    tiers.complete(copy, KvTransferStatus::Complete);
    // The short entry gained a resident descendant; its copy still holds
    // the old bytes until its drop.
    assert_eq!(
        tiers.pool.host_invalidation(short),
        Some(PrefixHostInvalidationV1::Descendant)
    );
    assert_eq!(tiers.host(old_tail)[2..], [STALE; 2]);
    tiers.drop_invalidated();
    // A promotion of the new entry restores the new bytes.
    assert!(tiers.remove(long));
    let (pending, copy) = tiers.promote(long).unwrap();
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(tiers.publish(pending), PrefixPublishV1::Published);
    let live = tiers.resume(long);
    tiers.release(live);
}

#[test]
fn a_retried_promotion_never_answers_to_an_earlier_attempts_id() {
    let prompt = counting(9, 12);
    let key = prompt.key_at(8).unwrap();
    let mut tiers = Tiers::ram(32, 16);
    tiers.capture_prompt(&prompt, &[8]);
    tiers.move_to_host(key);
    // An abandoned attempt, then a retry of the same entry.
    let first = tiers.pool.create_pending_anchor(key, STATE).unwrap();
    tiers.abandon(first);
    let second = tiers.pool.create_pending_anchor(key, STATE).unwrap();
    assert_ne!(first, second);
    assert_eq!(
        (first.key(), first.serial()),
        (second.key(), second.serial())
    );
    tiers.assert_retired(first);
    // A failed copy, then a retry that succeeds.
    let copy = tiers.start_promotion(second).unwrap();
    tiers.complete(copy, KvTransferStatus::FailedIdle);
    assert_eq!(tiers.publish(second), PrefixPublishV1::NotCopied);
    let (third, copy) = tiers.promote(key).unwrap();
    for retired in [first, second] {
        tiers.assert_retired(retired);
    }
    tiers.complete(copy, KvTransferStatus::Complete);
    tiers.assert_retired(second);
    assert_eq!(tiers.publish(third), PrefixPublishV1::Published);
    for retired in [first, second, third] {
        tiers.assert_retired(retired);
    }
}

#[test]
fn an_append_during_a_copy_leaves_the_copied_page_as_the_copy_found_it() {
    let prompt = counting(14, 12);
    let key = prompt.key_at(6).unwrap();
    let mut tiers = Tiers::ram(32, 16);
    let mut live = tiers.start(&prompt, 6).unwrap();
    tiers.capture(&live, 6).unwrap();
    let tail = tiers.pool.snapshot(live.handle).unwrap().blocks()[1];
    let (host, copy) = tiers.demote(key).unwrap();
    // The GPU anchor goes during the copy, and the sequence appends past
    // the rows the anchor committed. The lease still shares the tail page,
    // so the append writes a copy of it.
    assert!(tiers.remove(key));
    tiers.append(&mut live, 2).unwrap();
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_ne!(tiers.pool.snapshot(live.handle).unwrap().blocks()[1], tail);
    let copied = tiers.tier.anchor(host).unwrap().blocks[1];
    assert_eq!(tiers.host(copied)[2..], [STALE; 2]);
    tiers.release(live);
    let (pending, copy) = tiers.promote(key).unwrap();
    tiers.complete(copy, KvTransferStatus::Complete);
    assert_eq!(tiers.publish(pending), PrefixPublishV1::Published);
    let live = tiers.resume(key);
    tiers.release(live);
}

#[test]
fn followers_wait_on_one_promotion_and_then_resume_its_anchor() {
    let prompt = counting(7, 12);
    let key = prompt.key_at(8).unwrap();
    let mut tiers = Tiers::ram(32, 16);
    tiers.capture_prompt(&prompt, &[8]);
    let (_, copy) = tiers.demote(key).unwrap();
    tiers.complete(copy, KvTransferStatus::Complete);
    assert!(tiers.remove(key));
    let (pending, copy) = tiers.promote(key).unwrap();
    // One pending anchor and one transfer per entry: followers wait.
    assert_eq!(
        tiers.pool.create_pending_anchor(key, STATE),
        Err(KvError::StaleHandle)
    );
    assert_eq!(
        tiers
            .tier
            .begin_promotion(&mut tiers.pool, pending)
            .map(|_| ()),
        Err(KvError::StaleHandle)
    );
    tiers.check();
    tiers.complete(copy, KvTransferStatus::Complete);
    // The source entry stays as a shadow of the published anchor.
    assert_eq!(tiers.publish(pending), PrefixPublishV1::Published);
    assert!(tiers.pool.host_entry(key).is_some());
    assert_eq!(
        tiers.pool.create_pending_anchor(key, STATE),
        Err(KvError::StaleHandle)
    );
    // Every waiter resumes the GPU anchor; the first resume leaves the
    // shadow unusable for its drop.
    let first = tiers.resume(key);
    let second = tiers.resume(key);
    assert_eq!(
        tiers.drop_invalidated(),
        [(key, PrefixHostInvalidationV1::Resumed)]
    );
    tiers.release(first);
    tiers.release(second);
}

#[test]
fn a_host_entry_is_dropped_when_it_gains_a_resident_descendant() {
    use PrefixHostInvalidationV1::Descendant;
    let prompt = counting(8, 16);
    let key = |depth| prompt.key_at(depth).unwrap();
    for case in ["host only", "shadow", "copy in flight"] {
        let mut tiers = Tiers::ram(32, 16);
        tiers.capture_prompt(&prompt, &[8]);
        let (_, copy) = tiers.demote(key(8)).unwrap();
        let copy = if case == "copy in flight" {
            Some(copy)
        } else {
            tiers.complete(copy, KvTransferStatus::Complete);
            None
        };
        if case == "host only" {
            assert!(tiers.remove(key(8)));
        }
        // A later capture of a descendant.
        tiers.capture_prompt(&prompt, &[12]);
        assert_eq!(
            tiers.pool.host_invalidation(key(8)),
            Some(Descendant),
            "{case}"
        );
        // An unusable entry has nothing to promote.
        assert_eq!(
            tiers.pool.create_pending_anchor(key(8), STATE),
            Err(KvError::StaleHandle),
            "{case}"
        );
        assert_eq!(tiers.drop_invalidated(), [(key(8), Descendant)], "{case}");
        if let Some(copy) = copy {
            tiers.complete(copy, KvTransferStatus::Complete);
        }
        assert!(tiers.pool.host_entry(key(8)).is_none(), "{case}");
        // A descendant a demotion moved to RAM still keeps the ancestor out
        // of RAM.
        tiers.move_to_host(key(12));
        if !tiers.pool.has_anchor(key(8)) {
            tiers.capture_prompt(&prompt, &[8]);
        }
        assert!(
            tiers
                .pool
                .demotion_candidates(usize::MAX, 1)
                .iter()
                .all(|candidate| candidate.key != key(8)),
            "{case}"
        );
        assert!(tiers.demote(key(8)).is_err(), "{case}");
    }
}

#[test]
fn demotions_start_within_headroom_and_larger_allocations_wait_only_for_the_copy() {
    let prompts: Vec<_> = (0..3).map(|n| counting(10 + n, 8)).collect();
    let keys: Vec<_> = prompts.iter().map(|p| p.key_at(8).unwrap()).collect();
    for ending in [KvTransferStatus::Complete, KvTransferStatus::FailedIdle] {
        // Three anchors of three pages fill nine of twelve pages.
        let mut tiers = Tiers::ram(12, 16);
        for prompt in &prompts {
            tiers.capture_prompt(prompt, &[8]);
        }
        assert_eq!(tiers.free(), 3);
        // Free pages and the other anchors' pages must cover F plus the
        // candidate's own pages.
        assert!(tiers.pool.demotion_headroom(keys[0], 6));
        assert!(!tiers.pool.demotion_headroom(keys[0], 7));
        let (_, copy) = tiers.demote(keys[0]).unwrap();
        // Leased pages count for no candidate.
        assert!(tiers.pool.demotion_headroom(keys[1], 3));
        assert!(!tiers.pool.demotion_headroom(keys[1], 4));
        // An allocation within F evicts another anchor without waiting.
        tiers.pressure(6 * BLOCK).unwrap();
        let held: Vec<_> = keys.iter().map(|key| tiers.pool.has_anchor(*key)).collect();
        assert_eq!(held, [true, false, true]);
        // A larger one needs the copy's pages: it fails until the copy
        // completes or is cancelled, and then evicts its anchor.
        assert!(matches!(
            tiers.pressure(10 * BLOCK),
            Err(KvError::PoolExhausted { .. })
        ));
        assert!(tiers.pool.has_anchor(keys[0]));
        tiers.complete(copy, ending);
        tiers.pressure(10 * BLOCK).unwrap();
        assert!(!tiers.pool.has_anchor(keys[0]));
        let (evictions, _) = tiers.pool.drain_prefix_evictions();
        assert_eq!(
            evictions
                .last()
                .map(|eviction| (eviction.key, eviction.covered)),
            Some((keys[0], ending == KvTransferStatus::Complete))
        );
        assert_eq!(
            tiers.pool.host_entry(keys[0]).is_some(),
            ending == KvTransferStatus::Complete
        );
    }
}

/// SplitMix64, for a fixed sequence per seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<T: Clone>(&mut self, items: &[T]) -> Option<T> {
        (!items.is_empty()).then(|| items[self.below(items.len())].clone())
    }
}

/// What a copy in flight is for.
#[derive(Clone, Copy, Debug)]
enum Kind {
    Demotion(PrefixCacheKeyV1),
    Promotion(PendingAnchorId),
    Swap(HostAnchorId),
}

const SEEDS: u64 = 64;
const OPERATIONS: usize = 600;
/// Capture depths, some inside a page.
const DEPTHS: [usize; 8] = [4, 6, 8, 10, 12, 16, 20, 24];

/// Operations by weight: the tier's own operations come up most.
const OPERATION_WEIGHTS: [usize; 24] = [
    0, 0, 1, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 14, 14,
];

/// Six roots, each a trunk and two branches leaving it at 8 and 12.
fn prompts() -> Vec<PrefixChainV1> {
    let mut prompts = Vec::new();
    for root in 0..6_u32 {
        for (branch, leaves) in [(0_u32, 24), (1, 8), (2, 12)] {
            let tokens: Vec<u32> = (0..24_u32)
                .map(|i| root * 1000 + if i < leaves { i } else { branch * 100 + i })
                .collect();
            prompts.push(chain(&tokens));
        }
    }
    prompts
}

/// Random operations on both tiers: requests that resume or start cold and
/// capture, appends to live sequences, demotions, promotions and their
/// publish or abandon, copy completions of every status, host drops, GPU
/// removals, allocation pressure, owned swaps, touches, and calls with the
/// ids of earlier promotion attempts. Seeds cycle through tier order and
/// forced mode.
#[test]
fn a_seeded_model_check_of_both_tiers() {
    let prompts = prompts();
    let (mut demoted, mut promoted, mut published, mut probed) = (0usize, 0usize, 0usize, 0usize);
    for seed in 0..SEEDS {
        let mut rng = Rng(seed);
        let (ordered, forced) = (seed % 2 == 1, seed % 4 >= 2);
        let pool = KvPool::new(KvGeometry::new(BLOCK, 40).unwrap(), 8)
            .unwrap()
            .with_prefix_ancestry(ordered)
            .with_prefix_ram(true)
            .with_prefix_ram_forced(forced);
        let mut tiers = Tiers::new(pool, 24);
        let mut live: Vec<Live> = Vec::new();
        let mut copies: Vec<(KvTransferId, Kind)> = Vec::new();
        let mut settled: Vec<PendingAnchorId> = Vec::new();
        let mut swaps: Vec<HostAnchorId> = Vec::new();
        let mut quarantined = 0usize;
        let mut owner = 0u64;
        for _ in 0..OPERATIONS {
            let known: Vec<_> = tiers.keys.keys().copied().collect();
            let operation = if rng.chance(5) {
                13
            } else {
                OPERATION_WEIGHTS[rng.below(OPERATION_WEIGHTS.len())]
            };
            match operation {
                0 => {
                    // A request: resume its longest GPU match or start cold,
                    // extend, capture, and maybe stay live.
                    let prompt = rng.pick(&prompts).unwrap();
                    let target = 4 + rng.below(21);
                    let mut request = match tiers.pool.lookup_anchor(&prompt, target) {
                        Some((key, _)) => {
                            let mut resumed = tiers.resume(key);
                            resumed.prompt = prompt.clone();
                            resumed
                        }
                        None => match tiers.start(&prompt, target) {
                            Ok(started) => started,
                            Err(_) => continue,
                        },
                    };
                    let more = target - request.tokens;
                    if more > 0 && tiers.append(&mut request, more).is_err() {
                        tiers.release(request);
                        continue;
                    }
                    for _ in 0..1 + usize::from(rng.chance(30)) {
                        let depth = rng.pick(&DEPTHS).unwrap();
                        if depth <= request.tokens {
                            tiers.capture(&request, depth);
                        }
                    }
                    if live.len() < 3 && rng.chance(40) {
                        live.push(request);
                    } else {
                        tiers.release(request);
                    }
                }
                1 if !live.is_empty() => {
                    // A live sequence appends in place or past a copied tail,
                    // and may capture its new end.
                    let index = rng.below(live.len());
                    let mut request = live.swap_remove(index);
                    let room = request.prompt.tokens().len() - request.tokens;
                    let more = rng.below(room + 1);
                    if more > 0 && tiers.append(&mut request, more).is_err() {
                        tiers.release(request);
                        continue;
                    }
                    if DEPTHS.contains(&request.tokens) && rng.chance(50) {
                        tiers.capture(&request, request.tokens);
                    }
                    live.push(request);
                }
                2 if !live.is_empty() => {
                    let index = rng.below(live.len());
                    let request = live.swap_remove(index);
                    tiers.release(request);
                }
                3 => {
                    let mut keys: Vec<_> = tiers
                        .pool
                        .demotion_candidates(usize::MAX, 1)
                        .iter()
                        .map(|candidate| candidate.key)
                        .collect();
                    if forced {
                        keys.extend(
                            known
                                .iter()
                                .copied()
                                .filter(|key| tiers.pool.has_anchor(*key)),
                        );
                    }
                    if let Some(key) = rng.pick(&keys)
                        && let Ok((_, copy)) = tiers.demote(key)
                    {
                        copies.push((copy, Kind::Demotion(key)));
                    }
                }
                4 if !copies.is_empty() => {
                    let (copy, kind) = copies.swap_remove(rng.below(copies.len()));
                    let status = match rng.below(100) {
                        0..85 => KvTransferStatus::Complete,
                        85..98 => KvTransferStatus::FailedIdle,
                        _ => KvTransferStatus::Unconfirmed,
                    };
                    tiers.complete(copy, status);
                    match (kind, status) {
                        (_, KvTransferStatus::Unconfirmed) => quarantined += 1,
                        (Kind::Promotion(pending), _) => settled.push(pending),
                        (Kind::Swap(host), KvTransferStatus::Complete) => swaps.push(host),
                        // Eviction often follows a demotion soon.
                        (Kind::Demotion(key), KvTransferStatus::Complete) => {
                            demoted += 1;
                            if rng.chance(60) {
                                tiers.remove(key);
                            }
                        }
                        _ => {}
                    }
                }
                5 => {
                    // A prompt's longest host match, or a host-only key.
                    let key = if rng.chance(50) {
                        let prompt = rng.pick(&prompts).unwrap();
                        tiers
                            .pool
                            .lookup_host_anchor(&prompt, 24)
                            .map(|(key, _)| key)
                    } else {
                        let host_only: Vec<_> = known
                            .iter()
                            .copied()
                            .filter(|key| {
                                tiers.pool.host_entry(*key).is_some()
                                    && !tiers.pool.has_anchor(*key)
                            })
                            .collect();
                        rng.pick(&host_only)
                    };
                    if let Some(key) = key
                        && let Ok((pending, copy)) = tiers.promote(key)
                    {
                        promoted += 1;
                        copies.push((copy, Kind::Promotion(pending)));
                    }
                }
                6 if !settled.is_empty() => {
                    let pending = settled.swap_remove(rng.below(settled.len()));
                    if rng.chance(15) {
                        tiers.abandon(pending);
                    } else if tiers.publish(pending) == PrefixPublishV1::Published {
                        published += 1;
                    }
                    assert!(tiers.pool.pending_anchor(pending).is_none());
                }
                7 => {
                    tiers.drop_invalidated();
                }
                8 => {
                    let held: Vec<_> = known
                        .iter()
                        .copied()
                        .filter(|key| tiers.pool.host_entry(*key).is_some())
                        .collect();
                    if let Some(key) = rng.pick(&held) {
                        assert!(tiers.drop_entry(key));
                    }
                }
                9 => {
                    let held: Vec<_> = known
                        .iter()
                        .copied()
                        .filter(|key| tiers.pool.has_anchor(*key))
                        .collect();
                    if let Some(key) = rng.pick(&held) {
                        assert!(tiers.remove(key));
                    }
                }
                10 => {
                    let _ = tiers.pressure(BLOCK * (1 + rng.below(10)));
                }
                11 if !live.is_empty() => {
                    // A preempted request swaps its own boundary out, and its
                    // GPU anchor goes once the copy is submitted.
                    owner += 1;
                    let request = &live[rng.below(live.len())];
                    let mut digest = [0xee; 32];
                    digest[..8].copy_from_slice(&owner.to_le_bytes());
                    let private = PrefixCacheKeyV1::new(digest);
                    if tiers
                        .pool
                        .insert_private_anchor(private, request.handle, STATE)
                        .unwrap()
                        .is_none()
                    {
                        continue;
                    }
                    let result = tiers
                        .tier
                        .begin_offload(&mut tiers.pool, private, Some(owner));
                    tiers.check();
                    if let Ok((host, plan)) = result {
                        assert_eq!(plan.class(), KvTransferClass::SwapOut);
                        let copy = tiers.copy_out(host, plan);
                        copies.push((copy, Kind::Swap(host)));
                    }
                    assert!(tiers.remove(private));
                }
                12 if !swaps.is_empty() => {
                    let host = swaps.swap_remove(rng.below(swaps.len()));
                    tiers.tier.drop_anchor(host).unwrap();
                    tiers.check();
                }
                13 => {
                    let held: Vec<_> = known
                        .iter()
                        .copied()
                        .filter(|key| tiers.pool.has_anchor(*key))
                        .collect();
                    if let Some(key) = rng.pick(&held) {
                        let touched = tiers.resume(key);
                        tiers.release(touched);
                    }
                }
                14 if !tiers.retired.is_empty() => {
                    // An earlier attempt's id, perhaps of an entry promoted
                    // again since.
                    let retired = tiers.retired[rng.below(tiers.retired.len())];
                    tiers.assert_retired(retired);
                    probed += 1;
                }
                _ => {}
            }
            assert_eq!(
                tiers.tier.pending_transfers(),
                copies.len() + quarantined,
                "seed {seed}"
            );
        }
        // Settle everything; without a quarantine both tiers empty.
        while let Some((copy, kind)) = copies.pop() {
            tiers.complete(copy, KvTransferStatus::Complete);
            match kind {
                Kind::Promotion(pending) => settled.push(pending),
                Kind::Swap(host) => swaps.push(host),
                Kind::Demotion(_) => {}
            }
        }
        for pending in settled {
            tiers.publish(pending);
        }
        for request in live {
            tiers.release(request);
        }
        for host in swaps {
            tiers.tier.drop_anchor(host).unwrap();
        }
        tiers.drop_invalidated();
        for key in tiers.keys.keys().copied().collect::<Vec<_>>() {
            tiers.drop_entry(key);
            tiers.remove(key);
        }
        if quarantined == 0 {
            assert_eq!(tiers.tier.used_pages(), 0, "seed {seed}");
            assert_eq!(tiers.free(), 40, "seed {seed}");
        }
    }
    // The seeds reach every path, not only its refusals.
    eprintln!(
        "{demoted} demotions, {promoted} promotions, {published} publishes, \
         {probed} retired ids probed"
    );
    assert!(
        demoted > 4 * SEEDS as usize && published > 4 * SEEDS as usize,
        "{demoted} demotions, {published} publishes"
    );
}
