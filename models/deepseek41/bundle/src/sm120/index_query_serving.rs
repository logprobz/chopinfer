//! Bind the qualified wide index query without changing other phases.
use kernel_core::backend::{Backend, Need, Prepared, Refused, View};
use kernel_core::pooled_indexer::{PositionIndexerQuery, PositionQueryBuffers};
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;

impl crate::Kernels {
    pub(crate) fn wide_index_query_rows(capacity: u32, rows: u32) -> bool {
        use model_deepseek41_contract::tuning::{Family, select};
        select(
            &FLASH,
            Family::Indexer,
            "indexer.query.verify",
            "f32",
            &[("ranks", 4)],
            capacity,
            rows,
        )
        .is_some()
    }

    pub(crate) fn wide_index_query(
        shape: &crate::shape::Shape,
        query: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
        if shape.geometry.id != FLASH.id
            || shape.ranks() != 4
            || !Self::wide_index_query_rows(
                shape.verify_rows_within(model_deepseek41_contract::verify_bounds::INDEX_QUERY),
                need.capacity,
            )
        {
            return Err(Refused::Unsupported);
        }
        let op = super::indexer_query::register(Sm120::new(), shape.geometry)
            .with_parallel_position_query_rows(64)?
            .position_indexer_query(query, need)?;
        Ok(super::wide_routes::tagged(op, "indexer_query", 33))
    }

    /// Whether `indexer.query.verify` admits `rows` verify rows at the
    /// widest wide verify bucket (`verify_bounds::WIDE`).
    pub(crate) fn wide_verify_index_query_rows(rows: u32) -> bool {
        use model_deepseek41_contract::verify_bounds::WIDE;
        Self::wide_index_query_rows(WIDE[WIDE.len() - 1], rows)
    }

    /// The index query of a wide verify bucket (`Shape::wide_verify_rows`,
    /// the `wide_verify` candidate): the parallel query
    /// [`Self::wide_index_query`] binds, through the bucket's rows.
    pub(crate) fn wide_verify_index_query(
        shape: &crate::shape::Shape,
        query: &PositionIndexerQuery,
        need: Need,
    ) -> Result<Prepared<PositionQueryBuffers<View>>, Refused> {
        if !shape.wide_verify_rows().contains(&need.capacity)
            || !Self::wide_verify_index_query_rows(need.capacity)
        {
            return Err(Refused::Unsupported);
        }
        let op = super::indexer_query::register(Sm120::new(), shape.geometry)
            .with_parallel_position_query_rows(need.capacity)?
            .position_indexer_query(query, need)?;
        Ok(super::wide_routes::tagged_wide(op, "indexer_query"))
    }
}
