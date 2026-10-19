//! Add exact query repacking before native FP4 scores. Selection and candidate
//! validation retain their current entries and disjoint workspace range.
use crate::pooled_indexer::position_mma as mma;
use kernel_core::{
    backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, Refused, View},
    contract::{Argument, ArgumentList, Refusal},
    pooled_indexer::PositionIndexerQuery,
};

pub(super) fn prepare<B: 'static>(
    original: Result<Prepared<B>, Refused>,
    enabled: bool,
    query: &PositionIndexerQuery,
    rows: u32,
) -> Result<Prepared<B>, Refused> {
    let original = original?;
    if !enabled || !mma::supported(query) {
        return Ok(original);
    }
    let mut info = original.info().clone();
    let workspace = mma::Workspace::new(info.workspace, rows)?;
    let score = info.symbols[0].0.clone();
    info.symbols[0].0 = mma::score_symbol(&score);
    info.symbols.insert(0, (mma::pack_symbol(&score), 0));
    info.workspace = workspace.bytes;
    info.implementation = format!("sm120.position-native-fp4.v1({})", info.implementation);
    Ok(Box::new(Native {
        original,
        info,
        workspace,
    }))
}
struct Native<B> {
    original: Prepared<B>,
    info: OpInfo,
    workspace: mma::Workspace,
}
impl<B> PreparedOp<B> for Native<B> {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        buffers: &B,
        workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        workspace.covers(self.info.workspace)?;
        // The original planner also refuses all input/output aliases with
        // selection workspace, before we append the packed-query range.
        let mut calls = self.original.launches(buffers, workspace, counts)?;
        let score = &mut calls[0];
        let mut pack = score.clone();
        pack.symbol = self.info.symbols[0].0.clone();
        pack.plan.grid = [counts.rows, 1, 1];
        pack.plan.block = [256, 1, 1];
        pack.plan.pdl = true;
        pack.arguments = ArgumentList(vec![
            score.arguments.0[0].clone(),
            score.arguments.0[7].clone(),
        ]);
        let extra = [
            Argument::DeviceBuffer(workspace.handle),
            Argument::U64(self.workspace.codes),
            Argument::U64(self.workspace.scales),
            Argument::U64(self.workspace.status),
        ];
        pack.arguments.0.extend(extra.clone());
        score.symbol = self.info.symbols[1].0.clone();
        score.arguments.0.extend(extra);
        calls.insert(0, pack);
        Ok(calls)
    }
}
