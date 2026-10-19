//! Dense score launch adaptation, leaving selection and its scratch unchanged.
use crate::pooled_indexer::position_simt_tile as tile;
use kernel_core::{
    backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::Refusal,
    pooled_indexer::{PositionIndexerQuery, PositionSelectBuffers},
};
pub(super) fn prepare(
    original: Result<Prepared<PositionSelectBuffers<View>>, Refused>,
    enabled: bool,
    query: &PositionIndexerQuery,
) -> Result<Prepared<PositionSelectBuffers<View>>, Refused> {
    let original = original?;
    if !enabled
        || query.heads != 32
        || query.head_dim != 128
        || query.candidates != kernel_core::pooled_indexer::IndexerCandidates::All
    {
        return Ok(original);
    }
    let mut info = original.info().clone();
    info.symbols[0].0 = tile::symbol(&info.symbols[0].0);
    Ok(Box::new(Tiled { original, info }))
}
struct Tiled {
    original: Prepared<PositionSelectBuffers<View>>,
    info: OpInfo,
}
impl PreparedOp<PositionSelectBuffers<View>> for Tiled {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &PositionSelectBuffers<View>,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let mut calls = self.original.launches(b, workspace, counts)?;
        tile::adapt(&mut calls[0], counts.rows);
        Ok(calls)
    }
}
