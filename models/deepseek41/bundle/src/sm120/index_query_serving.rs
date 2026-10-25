//! Bind the qualified wide index query without changing other phases.
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::pooled_indexer::{PositionIndexerQuery, PositionQueryBuffers};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;

impl crate::Kernels {
    pub(crate) fn wide_index_query(
        shape: &crate::shape::Shape,
        query: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
        if shape.geometry.id != FLASH.id
            || shape.ranks() != 4
            || !(33..=shape.verify_decode_rows()).contains(&need.capacity)
        {
            return Err(Refused::Unsupported);
        }
        let op = super::indexer_query::register(Sm120::new(), shape.geometry)
            .with_parallel_position_query_rows(64)?
            .position_indexer_query(query, need)?;
        Ok(super::wide_routes::tagged(op, "indexer_query", 33))
    }
}
