//! Top-k selection with an explicit tie rule and NaN policy.

use std::cmp::Ordering;

use crate::contract::Refusal;

/// How keys compare, and the order among keys that compare equal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ties {
    /// Keys compare by `f64::total_cmp`, so +0 ranks above -0; equal keys
    /// take the lower index first.
    LowerIndex,
    /// Keys compare by IEEE order, so +0 equals -0; equal keys take the lower
    /// index first. The best key is the first strict maximum.
    IeeeLowerIndex,
}

/// What top-k does with NaN keys.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NanKeys {
    Refuse,
    /// NaN keys rank by `f64::total_cmp`: a positive NaN above +inf, a
    /// negative NaN below -inf. Only under `Ties::LowerIndex`.
    TotalOrder,
}

impl Ties {
    /// The rank order of two (index, key) entries; `Less` ranks first.
    /// IEEE keys must not be NaN.
    pub fn order(self, a: (usize, f64), b: (usize, f64)) -> Ordering {
        let keys = match self {
            Ties::LowerIndex => b.1.total_cmp(&a.1),
            Ties::IeeeLowerIndex => b.1.partial_cmp(&a.1).expect("IEEE keys are not NaN"),
        };
        keys.then(a.0.cmp(&b.0))
    }
}

/// The indexes of the `k` best keys, best first, by `ties`. Keys whose
/// `mask` entry is false are not selectable, and fewer than `k` selectable
/// keys are all returned. NaN keys follow `nan`; a mask of another length is
/// refused.
pub fn topk(
    scores: &[f64],
    k: usize,
    ties: Ties,
    nan: NanKeys,
    mask: Option<&[bool]>,
) -> Result<Vec<usize>, Refusal> {
    let nan_ordered = nan == NanKeys::TotalOrder && ties == Ties::LowerIndex;
    if (!nan_ordered && scores.iter().any(|s| s.is_nan()))
        || mask.is_some_and(|m| m.len() != scores.len())
    {
        return Err(Refusal("top-k keys or mask invalid"));
    }
    let mut indices = (0..scores.len())
        .filter(|&i| mask.is_none_or(|m| m[i]))
        .collect::<Vec<_>>();
    // The order is total over indexes, so a partial selection ranks exactly
    // as a full sort.
    let order = |&a: &usize, &b: &usize| ties.order((a, scores[a]), (b, scores[b]));
    if k > 0 && k < indices.len() {
        indices.select_nth_unstable_by(k - 1, order);
    }
    indices.truncate(k);
    indices.sort_by(order);
    Ok(indices)
}
