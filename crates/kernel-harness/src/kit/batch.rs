//! Exact comparisons across changes to other sequence items. Every
//! launch checks its poisoned writes before comparing the selected bytes.

use super::{Lookup, OpCase, Tag};
use crate::conformance::Images;
use crate::{LabResult, Session};
use serde_json::{Value, json};
use std::ops::Range;

pub struct Comparison {
    pub label: String,
    pub images: Images,
    /// Selected item's written bytes, at the same offsets in both runs.
    pub compared: Vec<(String, Range<usize>)>,
}

pub(super) type Mapping<'a> = Box<dyn Fn(&Images, u32) -> LabResult<Vec<Comparison>> + 'a>;

pub(super) fn validate(
    case: &OpCase<'_>,
    mapping: &dyn Fn(&Images, u32) -> LabResult<Vec<Comparison>>,
) -> LabResult<Vec<Comparison>> {
    let images = case.images();
    let comparisons = mapping(&images, case.capacity)?;
    if comparisons.is_empty() {
        return Err("an item batch mapping must compare at least one item".into());
    }
    let first = case.oracle_bits(&images, case.capacity)?;
    for comparison in &comparisons {
        if !case
            .leaves
            .iter()
            .filter(|leaf| matches!(leaf.tag, Tag::Input | Tag::State))
            .any(|leaf| images.bytes(&leaf.path).ok() != comparison.images.bytes(&leaf.path).ok())
        {
            return Err(format!("{} changes no input or state", comparison.label));
        }
        let other = case.oracle_bits(&comparison.images, case.capacity)?;
        let mut count = 0;
        for (path, range) in &comparison.compared {
            if range.is_empty() {
                return Err("item batch compares an empty range".into());
            }
            for input in [&images, &comparison.images] {
                let written = case.case.required_writes(path, input, case.capacity)?;
                if !range
                    .clone()
                    .all(|byte| written.iter().any(|span| span.contains(&byte)))
                {
                    return Err(format!(
                        "{} compares unwritten bytes of {path}",
                        comparison.label
                    ));
                }
            }
            if let (Some(left), Some(right)) = (first.get(path), other.get(path))
                && left.get(range.clone()) != right.get(range.clone())
            {
                return Err(format!(
                    "{}: the oracle changes compared {path} bytes",
                    comparison.label
                ));
            }
            count += range.len();
        }
        if count == 0 {
            return Err("item batch compares no bytes".into());
        }
    }
    Ok(comparisons)
}

pub(super) fn check(
    case: &OpCase<'_>,
    mapping: &dyn Fn(&Images, u32) -> LabResult<Vec<Comparison>>,
    session: &Session,
    functions: &Lookup<'_>,
) -> LabResult<Value> {
    let images = case.images();
    let comparisons = validate(case, mapping)?;
    let first = case.launch(session, functions, &images, case.capacity)?;
    let mut evidence = Vec::new();
    for comparison in comparisons {
        if !case
            .leaves
            .iter()
            .filter(|leaf| matches!(leaf.tag, Tag::Input | Tag::State))
            .any(|leaf| images.bytes(&leaf.path).ok() != comparison.images.bytes(&leaf.path).ok())
        {
            return Err(format!("{} changes no input or state", comparison.label));
        }
        let other = case.launch(session, functions, &comparison.images, case.capacity)?;
        let mut bytes = 0;
        for (path, range) in &comparison.compared {
            let left = first
                .get(path)
                .and_then(|b| b.get(range.clone()))
                .ok_or("batch comparison outside the first output")?;
            let right = other
                .get(path)
                .and_then(|b| b.get(range.clone()))
                .ok_or("batch comparison outside the changed output")?;
            if left != right {
                return Err(format!(
                    "{}: {path} {range:?} differs across items",
                    comparison.label
                ));
            }
            bytes += left.len();
        }
        if bytes == 0 {
            return Err(format!("{} compares no bytes", comparison.label));
        }
        evidence.push(json!({"item": comparison.label, "compared_bytes": bytes}));
    }
    Ok(json!({"item_batch": evidence}))
}
