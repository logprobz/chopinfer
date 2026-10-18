//! Each rank's checkpoint layout (`direct::checkpoint_layout`), which create
//! checks every rank index against, computed once per process for each
//! (shape, checkpoint layout, rank) and never changed after. The layout is a
//! pure function of the three, and computing it takes most of a create's
//! host time.

use std::sync::{Arc, Mutex};

use model_glm5next_contract::contract::Shape;
use model_glm5next_contract::direct::{Stored, checkpoint_layout};
use model_glm5next_contract::quantization::Layout;

type Computed = Result<Arc<Vec<Stored>>, String>;

/// Every computed layout with its key, in first-request order.
static LAYOUTS: Mutex<Vec<(Shape, Layout, u32, Computed)>> = Mutex::new(Vec::new());

/// `rank`'s layout of a checkpoint stored in `quantization`, as
/// [`checkpoint_layout`] computes it.
pub fn layout(shape: Shape, quantization: &Layout, rank: u32) -> Computed {
    // A panic while computing leaves no entry, so the list stays usable.
    let mut layouts = LAYOUTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let found = layouts
        .iter()
        .find(|(of, stored_as, at, _)| *of == shape && stored_as == quantization && *at == rank);
    if let Some((.., computed)) = found {
        return computed.clone();
    }
    let computed = checkpoint_layout(shape, quantization, rank).map(Arc::new);
    layouts.push((shape, *quantization, rank, computed.clone()));
    computed
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_glm5next_contract::quantization::representative_layouts;

    /// For every rank and representative checkpoint layout, the cached layout
    /// is the one computed fresh, and later requests return that same
    /// computation.
    #[test]
    fn cached_layouts_equal_fresh_ones() {
        for quantization in representative_layouts() {
            for rank in 0..Shape::TP4.ranks() {
                let fresh = checkpoint_layout(Shape::TP4, quantization, rank);
                let cached = layout(Shape::TP4, quantization, rank);
                assert_eq!(
                    cached.as_deref().map_err(String::as_str),
                    fresh.as_ref().map_err(String::as_str),
                    "{} rank {rank}",
                    quantization.id
                );
                let again = layout(Shape::TP4, quantization, rank).unwrap();
                assert!(Arc::ptr_eq(&cached.unwrap(), &again));
            }
        }
    }
}
