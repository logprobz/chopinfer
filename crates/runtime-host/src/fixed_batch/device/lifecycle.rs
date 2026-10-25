//! Exercise the same prefix, retirement and KV ownership APIs as serving.
use super::super::lifecycle::{Action, Boundary};
use super::*;
use crate::recurrent::StateSlotPoolV1;
use runtime_bundle_abi::LoadedBundleInstanceV1;
use runtime_kv::{BlockId, KvPool, PrefixCacheKeyV1, PrefixChainV1, SequenceHandle};

struct Saved {
    key: PrefixCacheKeyV1,
    tokens: Vec<u32>,
    pages: Vec<BlockId>,
}

#[derive(Default)]
pub(super) struct Lifetimes {
    saved: BTreeMap<usize, Saved>,
    pub events: Vec<Value>,
}

pub(super) struct Context<'a> {
    pub instance: &'a mut LoadedBundleInstanceV1,
    pub kv: &'a mut KvPool,
    pub slots: &'a mut StateSlotPoolV1,
    pub handles: &'a mut [SequenceHandle],
    pub history: &'a mut transition::History,
    pub quota: usize,
}

impl Context<'_> {
    fn retire(&mut self, slot: usize) -> Result<Value> {
        let request = RequestId::new(slot as u64 + 1);
        let lease = self
            .slots
            .lease(request)
            .ok_or("retire has no state lease")?;
        self.instance.retire_request(request.get())?;
        let stale = self.handles[slot];
        self.kv.release(stale)?;
        self.slots.retire(request);
        self.history.replace(slot, Vec::new())?;
        if self.kv.snapshot(stale).is_ok() || self.slots.lease(request).is_some() {
            return Err("retirement left a live handle or state lease".into());
        }
        self.kv.audit()?;
        Ok(
            json!({"action":"retire","request":request.get(),"state_lease":lease,
            "kv_slot":stale.slot(),"kv_generation":stale.generation()}),
        )
    }
}

impl Lifetimes {
    pub fn preflight(
        instance: &mut LoadedBundleInstanceV1,
        pages: usize,
        retained: usize,
    ) -> Result<()> {
        if !instance.supports_prefix_cache()
            || !instance.supports_prefix_state_slot()
            || !instance.supports_request_retirement()
        {
            return Err(
                "lifecycle replay requires prefix, state-slot restore and retirement APIs".into(),
            );
        }
        let per = usize::try_from(instance.describe_prefix_state()?)?;
        if per
            .checked_mul(retained)
            .is_none_or(|needed| needed >= pages)
        {
            return Err("declared KV pool cannot hold lifecycle snapshot payloads".into());
        }
        Ok(())
    }

    pub fn before(&mut self, boundary: &Boundary, context: &mut Context<'_>) -> Result<()> {
        for action in &boundary.actions {
            let event = match *action {
                Action::Capture { slot } => {
                    let tokens = context.history.tokens(slot)?.to_vec();
                    let snapshot = context.kv.snapshot(context.handles[slot])?;
                    if snapshot.tokens() != tokens.len() || self.saved.contains_key(&slot) {
                        return Err("capture history differs from its live sequence".into());
                    }
                    let chain = PrefixChainV1::new(
                        context.kv.geometry().block_tokens(),
                        tokens.clone().into(),
                        Vec::new(),
                    )?;
                    let key = chain.key_at(tokens.len()).ok_or("capture key is absent")?;
                    let payload = usize::try_from(context.instance.describe_prefix_state()?)?;
                    let pages = context
                        .kv
                        .insert_anchor(
                            key,
                            chain
                                .chain(chain.full_blocks())
                                .ok_or("capture chain is absent")?,
                            context.handles[slot],
                            payload,
                        )?
                        .ok_or("lifecycle prefix could not allocate its snapshot")?;
                    context.instance.capture_prefix_state(
                        slot as u64 + 1,
                        tokens.len() as u64,
                        &pages.iter().map(|p| p.index() as u64).collect::<Vec<_>>(),
                    )?;
                    for &page in &pages {
                        if context.kv.block_ref_count(page)? != 1 {
                            return Err("captured payload has unexpected owners".into());
                        }
                    }
                    let event = json!({"action":"capture","slot":slot,"tokens":tokens.len(),"payload_pages":pages.len()});
                    self.saved.insert(slot, Saved { key, tokens, pages });
                    event
                }
                Action::Retire { slot } => context.retire(slot)?,
                Action::Restore { slot, from } => {
                    let saved = self
                        .saved
                        .get(&from)
                        .ok_or("saved lifecycle prefix is absent")?;
                    let request = RequestId::new(slot as u64 + 1);
                    if context.slots.slot(request).is_some()
                        || !context.history.tokens(slot)?.is_empty()
                    {
                        return Err("restore target is still live".into());
                    }
                    let hit = context.kv.resume_anchor(saved.key, context.quota)?;
                    if hit.state() != saved.pages || hit.tokens() != saved.tokens.len() {
                        return Err("resumed prefix differs from its capture".into());
                    }
                    for &page in hit.state() {
                        if context.kv.block_ref_count(page)? != 2 {
                            return Err("restore did not pin payload pages".into());
                        }
                    }
                    let physical = context.slots.allocate(request)?;
                    // A failed callback stops the replay. No page is recycled while
                    // the bundle may still reference it; destruction drains work.
                    context.instance.restore_prefix_state_in_slot(
                        request.get(),
                        &saved.tokens,
                        &hit.state()
                            .iter()
                            .map(|p| p.index() as u64)
                            .collect::<Vec<_>>(),
                        physical,
                    )?;
                    context.kv.finish_restore(hit.handle())?;
                    for &page in hit.state() {
                        if context.kv.block_ref_count(page)? != 1 {
                            return Err("restore retained payload pins".into());
                        }
                    }
                    context.handles[slot] = hit.handle();
                    context.history.replace(slot, saved.tokens.clone())?;
                    json!({"action":"restore","slot":slot,"from":from,"tokens":saved.tokens.len(),
                        "state_lease":context.slots.lease(request),"kv_slot":hit.handle().slot(),"kv_generation":hit.handle().generation()})
                }
            };
            context.kv.audit()?;
            self.events
                .push(json!({"before_step":boundary.before_step,"event":event}));
        }
        Ok(())
    }

    pub fn prepare_prefills(
        &mut self,
        step: &super::super::trace::Step,
        context: &mut Context<'_>,
    ) -> Result<()> {
        for item in step.items.iter().filter(|item| item.prefill) {
            let request = RequestId::new(item.slot as u64 + 1);
            if context.slots.slot(request).is_none() {
                let handle = context.kv.create_sequence(context.quota)?;
                let slot = context.slots.allocate(request)?;
                context.handles[item.slot] = handle;
                self.events
                    .push(json!({"action":"restart","request":request.get(),"physical_slot":slot}));
            }
        }
        Ok(())
    }

    pub fn canceled(
        &mut self,
        step: usize,
        items: &super::super::trace::Step,
        context: &mut Context<'_>,
    ) -> Result<()> {
        for item in &items.items {
            let retired = context.retire(item.slot)?;
            self.events
                .push(json!({"after_step":step,"canceled":true,"event":retired}));
        }
        Ok(())
    }

    pub fn finish(&mut self, context: &mut Context<'_>) -> Result<()> {
        for slot in 0..context.handles.len() {
            if context
                .slots
                .slot(RequestId::new(slot as u64 + 1))
                .is_some()
            {
                context.retire(slot)?;
            }
        }
        for saved in self.saved.values() {
            if !context.kv.remove_anchor(saved.key)? {
                return Err("saved prefix was evicted during lifecycle replay".into());
            }
        }
        context.kv.audit()?;
        let stats = context.kv.stats();
        if stats.free_blocks() != stats.block_capacity() || stats.active_sequences() != 0 {
            return Err("lifecycle replay leaked KV ownership".into());
        }
        Ok(())
    }
}
