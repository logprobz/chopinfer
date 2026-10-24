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
    pub symbol: String,
    pub class: RouteClass,
    pub family: &'static str,
    pub admitted: Vec<RangeInclusive<u32>>,
}

impl RouteTag {
    pub fn new(
        symbol: impl Into<String>,
        class: RouteClass,
        family: &'static str,
        admitted: impl IntoIterator<Item = RangeInclusive<u32>>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            class,
            family,
            admitted: admitted.into_iter().collect(),
        }
    }
}

/// Merge registrations deterministically. A symbol has one class and family.
/// Repeated registrations may declare different admitted rows of that route.
pub fn merge(tags: impl IntoIterator<Item = RouteTag>) -> Result<Vec<RouteTag>, String> {
    let mut by_symbol = BTreeMap::<String, RouteTag>::new();
    for tag in tags {
        if tag.symbol.is_empty()
            || tag.family.is_empty()
            || tag.admitted.iter().any(|r| r.is_empty() || *r.start() == 0)
        {
            return Err(format!("invalid route tag: {tag:?}"));
        }
        if let Some(previous) = by_symbol.get_mut(&tag.symbol) {
            if previous.class != tag.class || previous.family != tag.family {
                return Err(format!("conflicting route tags: {previous:?}, {tag:?}"));
            }
            previous.admitted.extend(tag.admitted);
        } else {
            by_symbol.insert(tag.symbol.clone(), tag);
        }
    }
    for tag in by_symbol.values_mut() {
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
    Ok(by_symbol.into_values().collect())
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
            RouteTag::new("base", RouteClass::Baseline, "norm", []),
        ] {
            assert!(merge([baseline.clone(), changed]).is_err());
        }
    }
}
