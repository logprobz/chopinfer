//! Opt-in launch adaptation for separately compiled live-context entries.
use crate::pooled_indexer::position_live as live;
use kernel_core::backend::{Counts, Launch, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, Refusal};

pub(super) fn prepare<B: 'static>(
    original: Result<Prepared<B>, Refused>,
    enabled: bool,
    rows: u32,
    kept: u32,
) -> Result<Prepared<B>, Refused> {
    let original = original?;
    if !enabled {
        return Ok(original);
    }
    let mut info = original.info().clone();
    let offset = info.workspace.next_multiple_of(8);
    info.workspace = offset
        .checked_add(live::workspace_bytes(rows, kept))
        .ok_or(Refusal("live position workspace overflow"))?;
    info.symbols = vec![
        (live::symbol(&info.symbols[0].0), 0),
        (live::symbol(&info.symbols[1].0), 0),
        (live::merge_symbol(&info.symbols[1].0), 0),
    ];
    // The arithmetic, score/index key, and output order retain the original
    // implementation identity. Only its scheduling and scratch change.
    Ok(Box::new(Live {
        original,
        info,
        offset,
    }))
}

struct Live<B> {
    original: Prepared<B>,
    info: OpInfo,
    offset: u64,
}
impl<B> PreparedOp<B> for Live<B> {
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
        let mut calls = self.original.launches(buffers, workspace, counts)?;
        if self.offset == 0
            && calls.iter().any(|call| {
                call.arguments
                    .0
                    .contains(&Argument::DeviceBuffer(workspace.handle))
            })
        {
            return Err(Refusal(
                "live position workspace must have distinct storage",
            ));
        }
        calls[0].symbol = self.info.symbols[0].0.clone();
        calls[0].plan.grid = [live::SCORE_CTAS, counts.rows, 1];
        calls[1].symbol = self.info.symbols[1].0.clone();
        calls[1].plan.grid = [live::SHARDS, counts.rows, 1];
        calls[1].arguments.0.extend([
            Argument::DeviceBuffer(workspace.handle),
            Argument::U64(self.offset),
        ]);
        let mut merge = calls[1].clone();
        merge.symbol = self.info.symbols[2].0.clone();
        merge.plan.grid = [1, counts.rows, 1];
        calls.push(merge);
        Ok(calls)
    }
}
