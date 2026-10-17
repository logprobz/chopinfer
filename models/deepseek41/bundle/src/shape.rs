//! What an instance runs: a generated geometry
//! (`model_deepseek41_contract::geometry`) at a rank count. Per-rank sizes
//! follow from them: attention at `heads / ranks` heads and
//! `o_groups / ranks` output groups per rank, the routed and shared experts
//! split by intermediate column, the indexer, the window and compressed
//! keys and the router replicated, the embedding and head split by
//! vocabulary row (D3).

use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::admission::{SUPPORTED_RANKS, check_ranks};
use model_deepseek41_contract::geometry::{GEOMETRIES, Geometry};

/// One instance's shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub geometry: &'static Geometry,
    pub ranks: Ranks,
}

impl Shape {
    /// `geometry` at `ranks`, or the field that refuses the split.
    pub fn new(geometry: &'static Geometry, ranks: Ranks) -> Result<Shape, String> {
        check_ranks(geometry, ranks.get())?;
        Ok(Shape { geometry, ranks })
    }

    /// The geometry's dimensions.
    pub const fn dims(&self) -> &'static model_deepseek41_contract::geometry::Shape {
        &self.geometry.shape
    }

    pub const fn ranks(&self) -> u32 {
        self.ranks.get()
    }

    /// Attention heads per rank.
    pub const fn local_heads(&self) -> u32 {
        self.dims().attention.heads / self.ranks.get()
    }

    /// `wo_a` groups per rank.
    pub const fn local_groups(&self) -> u32 {
        self.dims().attention.o_groups / self.ranks.get()
    }

    /// Routed and shared expert intermediate columns per rank.
    pub const fn local_inter(&self) -> u32 {
        self.dims().moe.intermediate / self.ranks.get()
    }

    /// Vocabulary rows per rank.
    pub const fn vocab_rows(&self) -> u32 {
        self.dims().vocab / self.ranks.get()
    }

    /// The topology ID of this shape: `deepseek41-<geometry>-tp<ranks>`.
    pub fn topology(&self) -> String {
        topology_id(self.geometry, self.ranks)
    }
}

/// The topology ID of `geometry` at `ranks`.
pub fn topology_id(geometry: &Geometry, ranks: Ranks) -> String {
    format!("deepseek41-{}-tp{}", geometry.id, ranks.get())
}

/// Every topology the build serves: each generated geometry at each rank
/// count it splits over.
pub fn topologies() -> Vec<Shape> {
    GEOMETRIES
        .iter()
        .flat_map(|geometry| {
            SUPPORTED_RANKS
                .iter()
                .filter_map(|&ranks| Shape::new(geometry, Ranks::new(ranks)?).ok())
        })
        .collect()
}

/// The shape a topology ID names.
pub fn of_topology(id: &str) -> Option<Shape> {
    topologies()
        .into_iter()
        .find(|shape| shape.topology() == id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    #[test]
    fn flash_splits_evenly_at_four_ranks() {
        let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
        assert_eq!(
            (
                shape.local_heads(),
                shape.local_groups(),
                shape.local_inter()
            ),
            (16, 2, 576)
        );
        assert_eq!(shape.vocab_rows(), 32_320);
        assert_eq!(
            shape.topology(),
            model_deepseek41_contract::FLASH_TP4_TOPOLOGY
        );
        assert_eq!(of_topology("deepseek41-flash-tp4"), Some(shape));
        assert_eq!(of_topology("deepseek41-flash-tp8"), None);
    }
}
