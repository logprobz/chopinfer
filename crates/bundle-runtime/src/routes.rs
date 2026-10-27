//! Optional selected-entry accounting. It performs no host readback or sync.
use kernel_core::route::{RouteClass, RouteTag};
use runtime_bundle_abi::{BundleContextHandleV1, BundleCudaSymbolHandleV1, HostServicesV1};
use std::cell::RefCell;
use std::collections::BTreeMap;

/// Reserved route symbol for bundle-facing synchronous mapped-row calls.
pub const MAPPED_GATHER_SYMBOL: &str = "host.gather_mapped_rows";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteCounts {
    pub fallback_launches: u64,
    pub families: BTreeMap<String, u64>,
    pub entries: BTreeMap<String, (RouteTag, u64)>,
    pub ranks: BTreeMap<String, RankCounts>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RankCounts {
    pub tagged: BTreeMap<String, u64>,
    pub untagged: BTreeMap<String, u64>,
}

impl RouteCounts {
    fn record(&mut self, tag: &RouteTag, launches: u64) {
        if tag.class != RouteClass::Fast {
            self.fallback_launches += launches;
            *self.families.entry(tag.family.into()).or_default() += launches;
        }
        let entry = self
            .entries
            .entry(tag.entry.clone())
            .or_insert_with(|| (tag.clone(), 0));
        assert_eq!(
            (entry.0.class, entry.0.family, entry.0.symbol.as_str()),
            (tag.class, tag.family, tag.symbol.as_str()),
            "conflicting selected route entry"
        );
        entry.1 += launches;
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"fallback_launches": self.fallback_launches, "families": self.families,
            "ranks": self.ranks.iter().map(|(rank, counts)| (rank.clone(), serde_json::json!({"tagged": counts.tagged, "untagged": counts.untagged}))).collect::<BTreeMap<_, _>>(),
            "entries": self.entries.iter().map(|(entry, (tag, launches))| (entry.clone(), serde_json::json!({"symbol": tag.symbol, "class": tag.class.name(), "family": tag.family, "wrappers": tag.wrappers, "launches": launches}))).collect::<BTreeMap<_,_>>()})
    }
}

type SymbolRegistry = BTreeMap<(usize, u64), (Option<u32>, String)>;

thread_local! {
    static ACTIVE: RefCell<Option<RouteCounts>> = const { RefCell::new(None) };
    static CONTEXTS: RefCell<BTreeMap<(usize, u64), u32>> = const { RefCell::new(BTreeMap::new()) };
    static SYMBOLS: RefCell<SymbolRegistry> = const { RefCell::new(BTreeMap::new()) };
}

pub(crate) fn register_context(
    services: &HostServicesV1,
    context: BundleContextHandleV1,
    rank: u32,
) {
    if active() {
        CONTEXTS.with(|map| {
            map.borrow_mut()
                .insert((services.diagnostic_identity(), context.raw), rank)
        });
    }
}

pub(crate) fn register_symbol(
    services: &HostServicesV1,
    context: BundleContextHandleV1,
    function: BundleCudaSymbolHandleV1,
    symbol: &str,
) {
    if active() {
        let host = services.diagnostic_identity();
        let rank = CONTEXTS.with(|map| map.borrow().get(&(host, context.raw)).copied());
        SYMBOLS.with(|map| {
            map.borrow_mut()
                .insert((host, function.raw), (rank, symbol.to_owned()))
        });
    }
}

pub(crate) fn forget_context(services: &HostServicesV1, context: BundleContextHandleV1) {
    CONTEXTS.with(|map| {
        map.borrow_mut()
            .remove(&(services.diagnostic_identity(), context.raw))
    });
}

pub(crate) fn forget_symbol(services: &HostServicesV1, function: BundleCudaSymbolHandleV1) {
    SYMBOLS.with(|map| {
        map.borrow_mut()
            .remove(&(services.diagnostic_identity(), function.raw))
    });
}

pub(crate) fn submitted_symbol(
    services: &HostServicesV1,
    function: BundleCudaSymbolHandleV1,
    tag: Option<&RouteTag>,
) {
    if !active() {
        return;
    }
    SYMBOLS.with(|map| {
        let map = map.borrow();
        let identity = map.get(&(services.diagnostic_identity(), function.raw));
        record_launch(
            identity.and_then(|(rank, _)| *rank),
            identity.map(|(_, name)| name.as_str()).unwrap_or("unknown"),
            tag,
        );
    });
}

fn record_launch(rank: Option<u32>, symbol: &str, tag: Option<&RouteTag>) {
    ACTIVE.with(|a| {
        if let Some(counts) = a.borrow_mut().as_mut() {
            if let Some(tag) = tag {
                counts.record(tag, 1);
            }
            let rank = counts
                .ranks
                .entry(rank.map_or_else(|| "unknown".into(), |rank| rank.to_string()))
                .or_default();
            let (map, key) = match tag {
                Some(tag) => (&mut rank.tagged, tag.entry.as_str()),
                None => (&mut rank.untagged, symbol),
            };
            *map.entry(key.to_owned()).or_default() += 1;
        }
    });
}

/// Nested scopes isolate graph capture from the enclosing step. Dropping a
/// failed scope restores its caller, so partial captures never count as work.
pub struct RouteCounter {
    previous: Option<RouteCounts>,
    metadata: Option<kernel_core::route::Metadata>,
    finished: bool,
}

impl RouteCounter {
    pub fn begin(enabled: bool) -> Self {
        Self {
            previous: ACTIVE.with(|a| a.replace(enabled.then(RouteCounts::default))),
            metadata: enabled.then(kernel_core::route::Metadata::enable),
            finished: false,
        }
    }
    pub fn nested() -> Self {
        Self::begin(ACTIVE.with(|a| a.borrow().is_some()))
    }
    pub fn finish(mut self) -> Option<RouteCounts> {
        let result = ACTIVE.with(|a| a.replace(self.previous.take()));
        drop(self.metadata.take());
        self.finished = true;
        result
    }
}
impl Drop for RouteCounter {
    fn drop(&mut self) {
        if !self.finished {
            ACTIVE.with(|a| {
                a.replace(self.previous.take());
            });
        }
    }
}

pub fn active() -> bool {
    ACTIVE.with(|a| a.borrow().is_some())
}

pub fn submitted(tag: Option<&RouteTag>) {
    record_launch(None, "unknown", tag);
}

/// Attribute a non-kernel route, such as a mapped-row gather, to its owner.
/// Kernel dispatches infer their rank from the acquired symbol instead.
pub fn submitted_on_rank(rank: u32, tag: &RouteTag) {
    record_launch(Some(rank), &tag.symbol, Some(tag));
}

pub fn replay(counts: &RouteCounts) {
    ACTIVE.with(|a| {
        if let Some(active) = a.borrow_mut().as_mut() {
            for (tag, count) in counts.entries.values() {
                active.record(tag, *count);
            }
            for (rank, counts) in &counts.ranks {
                let target = active.ranks.entry(rank.clone()).or_default();
                for (to, from) in [
                    (&mut target.tagged, &counts.tagged),
                    (&mut target.untagged, &counts.untagged),
                ] {
                    for (key, count) in from {
                        *to.entry(key.clone()).or_default() += count;
                    }
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::route::branch;

    #[test]
    fn census_keeps_rank_untagged_and_replay_counts() {
        let counter = RouteCounter::begin(true);
        let tag = branch("same", RouteClass::Fast, "norm", "fast", [1..=8]);
        record_launch(Some(0), "tagged", Some(&tag));
        let capture = RouteCounter::nested();
        record_launch(Some(1), "plain", None);
        let captured = capture.finish().unwrap();
        replay(&captured);
        replay(&captured);
        let counts = counter.finish().unwrap();
        assert_eq!(counts.ranks["0"].tagged[&tag.entry], 1);
        assert_eq!(counts.ranks["1"].untagged["plain"], 2);
        assert_eq!(counts.fallback_launches, 0);
        assert_eq!(counts.json()["ranks"]["1"]["untagged"]["plain"], 2);
        let disabled = RouteCounter::begin(false);
        record_launch(Some(2), "ignored", None);
        assert!(disabled.finish().is_none());
    }

    #[test]
    fn explicit_host_route_rank_preserves_aggregate_fallback_counts() {
        let counter = RouteCounter::begin(true);
        let tag = branch(
            MAPPED_GATHER_SYMBOL,
            RouteClass::Baseline,
            "gather",
            "sync",
            [1..=8192],
        );
        submitted_on_rank(2, &tag);
        let counts = counter.finish().unwrap();
        assert_eq!(counts.fallback_launches, 1);
        assert_eq!(counts.ranks["2"].tagged[&tag.entry], 1);
        assert!(!counts.ranks.contains_key("unknown"));
    }

    #[test]
    fn disabled_and_failed_nested_scopes_do_not_leak_counts() {
        let tag = branch("same", RouteClass::Baseline, "norm", "fallback", [8..=8]);
        submitted(Some(&tag));
        assert!(!kernel_core::route::Metadata::enabled());
        let outer = RouteCounter::begin(true);
        submitted(Some(&tag));
        {
            let _failed_capture = RouteCounter::nested();
            submitted(Some(&tag));
        }
        let capture = RouteCounter::nested();
        submitted(Some(&tag));
        let counts = capture.finish().unwrap();
        replay(&counts);
        assert_eq!(outer.finish().unwrap().fallback_launches, 2);
        assert!(!kernel_core::route::Metadata::enabled());
        assert!(RouteCounter::begin(false).finish().is_none());
    }
}

pub(crate) fn logging_enabled() -> bool {
    #[cfg(any(test, feature = "test-support"))]
    if let Some(enabled) = LOGGING_OVERRIDE.get() {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CHOPINFER_ROUTE_COUNTERS").as_deref() == Ok("1"))
}

#[cfg(any(test, feature = "test-support"))]
thread_local! {
    static LOGGING_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Enable optional log records on this test's thread without changing its environment.
#[cfg(any(test, feature = "test-support"))]
pub fn with_logging<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LOGGING_OVERRIDE.set(self.0);
        }
    }
    let _restore = Restore(LOGGING_OVERRIDE.replace(Some(true)));
    run()
}
