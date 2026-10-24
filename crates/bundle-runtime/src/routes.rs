//! Optional selected-entry accounting. It performs no host readback or sync.
use kernel_core::route::{RouteClass, RouteTag};
use std::cell::RefCell;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteCounts {
    pub fallback_launches: u64,
    pub families: BTreeMap<String, u64>,
    pub entries: BTreeMap<String, (RouteTag, u64)>,
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
            "entries": self.entries.iter().map(|(entry, (tag, launches))| (entry.clone(), serde_json::json!({"symbol": tag.symbol, "class": tag.class.name(), "family": tag.family, "wrappers": tag.wrappers, "launches": launches}))).collect::<BTreeMap<_,_>>()})
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<RouteCounts>> = const { RefCell::new(None) };
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
    if let Some(tag) = tag {
        ACTIVE.with(|a| {
            if let Some(counts) = a.borrow_mut().as_mut() {
                counts.record(tag, 1);
            }
        });
    }
}

pub fn replay(counts: &RouteCounts) {
    ACTIVE.with(|a| {
        if let Some(active) = a.borrow_mut().as_mut() {
            for (tag, count) in counts.entries.values() {
                active.record(tag, *count);
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::route::branch;

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
    std::env::var("CHOPINFER_ROUTE_COUNTERS").as_deref() == Ok("1")
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
