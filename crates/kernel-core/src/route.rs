//! Route quality is registration metadata, independent of numerical contracts.

use std::collections::BTreeMap;
use std::ops::RangeInclusive;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RouteClass {
    Fast,
    Baseline,
    Reference,
}

impl RouteClass {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fast => "Fast",
            Self::Baseline => "Baseline",
            Self::Reference => "Reference",
        }
    }
}

/// An untagged symbol is Fast. Admitted rows describe its fast alternatives.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RouteTag {
    /// Registration entry or selected branch identity. Different entries may
    /// reuse one CUDA symbol without sharing a route class.
    pub entry: String,
    pub symbol: String,
    pub class: RouteClass,
    pub family: &'static str,
    pub admitted: Vec<RangeInclusive<u32>>,
    /// Nested routing wrappers traversed before reaching this symbol.
    pub wrappers: Vec<&'static str>,
}

impl RouteTag {
    pub fn new(
        symbol: impl Into<String>,
        class: RouteClass,
        family: &'static str,
        admitted: impl IntoIterator<Item = RangeInclusive<u32>>,
    ) -> Self {
        let symbol = symbol.into();
        Self {
            entry: format!("{family}/{symbol}"),
            symbol,
            class,
            family,
            admitted: normalized(admitted.into_iter().collect()),
            wrappers: Vec::new(),
        }
    }

    pub fn with_entry(mut self, entry: impl Into<String>) -> Self {
        self.entry = entry.into();
        self
    }

    pub fn with_wrapper(mut self, wrapper: &'static str) -> Self {
        self.wrappers.push(wrapper);
        self
    }
}

fn normalized(mut rows: Vec<RangeInclusive<u32>>) -> Vec<RangeInclusive<u32>> {
    if rows.iter().any(|r| r.is_empty() || *r.start() == 0) {
        return rows;
    }
    rows.sort_by_key(|r| (*r.start(), *r.end()));
    let mut result: Vec<RangeInclusive<u32>> = Vec::new();
    for row in rows {
        if let Some(last) = result.last_mut()
            && *row.start() <= last.end().saturating_add(1)
        {
            *last = *last.start()..=(*last.end()).max(*row.end());
        } else {
            result.push(row);
        }
    }
    result
}

/// Name one selected branch, preserving its identity when a symbol is reused.
pub fn branch(
    symbol: &str,
    class: RouteClass,
    family: &'static str,
    wrapper: &'static str,
    admitted: impl IntoIterator<Item = RangeInclusive<u32>>,
) -> RouteTag {
    RouteTag::new(symbol, class, family, admitted)
        .with_entry(format!("{family}/{wrapper}/{symbol}"))
        .with_wrapper(wrapper)
}

/// Merge registrations deterministically. An entry has one class and family.
/// Repeated registrations may declare different admitted rows of that route.
pub fn merge(tags: impl IntoIterator<Item = RouteTag>) -> Result<Vec<RouteTag>, String> {
    let mut by_entry = BTreeMap::<String, RouteTag>::new();
    for tag in tags {
        if tag.entry.is_empty()
            || tag.symbol.is_empty()
            || tag.family.is_empty()
            || tag.admitted.iter().any(|r| r.is_empty() || *r.start() == 0)
        {
            return Err(format!("invalid route tag: {tag:?}"));
        }
        if let Some(previous) = by_entry.get_mut(&tag.entry) {
            if previous.class != tag.class
                || previous.family != tag.family
                || previous.symbol != tag.symbol
            {
                return Err(format!("conflicting route tags: {previous:?}, {tag:?}"));
            }
            for range in tag.admitted {
                if !previous.admitted.contains(&range) {
                    previous.admitted.push(range);
                }
            }
            for wrapper in tag.wrappers {
                if !previous.wrappers.contains(&wrapper) {
                    previous.wrappers.push(wrapper);
                }
            }
        } else {
            by_entry.insert(tag.entry.clone(), tag);
        }
    }
    for tag in by_entry.values_mut() {
        let mut seen = std::collections::BTreeSet::new();
        tag.wrappers.retain(|w| seen.insert(*w));
        tag.admitted.sort_by_key(|r| (*r.start(), *r.end()));
        let mut ranges: Vec<RangeInclusive<u32>> = Vec::new();
        for range in tag.admitted.drain(..) {
            if let Some(last) = ranges.last_mut()
                && *range.start() <= last.end().saturating_add(1)
            {
                *last = *last.start()..=(*last.end()).max(*range.end());
            } else {
                ranges.push(range);
            }
        }
        tag.admitted = ranges;
    }
    Ok(by_entry.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registrations_merge_ranges_and_refuse_conflicting_classes_or_families() {
        let baseline = RouteTag::new("base", RouteClass::Baseline, "projection", [2..=4]);
        let more = RouteTag::new("base", RouteClass::Baseline, "projection", [5..=8]);
        assert_eq!(
            merge([baseline.clone(), more]).unwrap()[0].admitted,
            [2..=8]
        );
        for changed in [
            RouteTag::new("base", RouteClass::Fast, "projection", []),
            RouteTag::new("base", RouteClass::Baseline, "norm", [])
                .with_entry(baseline.entry.clone()),
        ] {
            assert!(merge([baseline.clone(), changed]).is_err());
        }
    }
}

thread_local! {
    static METADATA: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Enable selected-entry metadata on this thread for a create or execute call.
pub struct Metadata(bool, std::marker::PhantomData<std::rc::Rc<()>>);
impl Metadata {
    pub fn enable() -> Self {
        Self(METADATA.replace(true), std::marker::PhantomData)
    }
    pub fn enabled() -> bool {
        METADATA.get()
    }
}
impl Drop for Metadata {
    fn drop(&mut self) {
        METADATA.set(self.0);
    }
}

pub fn selected<B>(
    op: &dyn crate::backend::PreparedOp<B>,
    counts: crate::backend::Counts,
    launch: &crate::backend::Launch,
) -> Option<RouteTag> {
    Metadata::enabled()
        .then(|| op.route(counts, launch))
        .flatten()
}

/// Attach entry metadata to an existing prepared operation without changing
/// its contract or launches. More specific wrappers may override this branch.
pub fn tagged<B: 'static>(
    inner: crate::backend::Prepared<B>,
    tags: Vec<RouteTag>,
) -> crate::backend::Prepared<B> {
    Box::new(Tagged { inner, tags })
}

struct Tagged<B> {
    inner: crate::backend::Prepared<B>,
    tags: Vec<RouteTag>,
}

impl<B> crate::backend::PreparedOp<B> for Tagged<B> {
    fn info(&self) -> &crate::backend::OpInfo {
        self.inner.info()
    }
    fn route(
        &self,
        counts: crate::backend::Counts,
        launch: &crate::backend::Launch,
    ) -> Option<RouteTag> {
        self.tags
            .iter()
            .find(|t| t.symbol == launch.symbol)
            .cloned()
            .or_else(|| self.inner.route(counts, launch))
    }
    fn initialize(
        &self,
        buffers: &B,
    ) -> Result<Vec<crate::backend::Launch>, crate::contract::Refusal> {
        self.inner.initialize(buffers)
    }
    fn defer_bf16_round(
        &self,
        buffers: &B,
        workspace: crate::backend::View,
        counts: crate::backend::Counts,
    ) -> Result<Option<crate::backend::DeferredBf16Round>, crate::contract::Refusal> {
        self.inner.defer_bf16_round(buffers, workspace, counts)
    }
    fn launches(
        &self,
        buffers: &B,
        workspace: crate::backend::View,
        counts: crate::backend::Counts,
    ) -> Result<Vec<crate::backend::Launch>, crate::contract::Refusal> {
        self.inner.launches(buffers, workspace, counts)
    }
}
