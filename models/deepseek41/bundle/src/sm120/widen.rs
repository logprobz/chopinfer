//! The row widening (`own_ops::widen`) on SM120: the geometry's widen row
//! of its hidden rows, the shared entry's body (`kernels_sm120::row::
//! rounding`) that its rounding cubin compiles (`reduce.rs`).

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::Refusal;
use kernels_sm120::row::rounding;
use kernels_sm120::row::rounding::Rounding;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};

use super::reduce;

/// The widening `op` of `geometry`'s rows over up to `need.capacity` rows.
pub(super) fn prepare(
    op: &WidenRows,
    need: Need,
    geometry: &Geometry,
) -> Result<Prepared<WidenRowsBuffers<View>>, Refused> {
    op.validate(need)?;
    if op.width != geometry.shape.hidden && op.width != geometry.shape.dspark.markov_rank {
        return Err(Refused::Rule(format!(
            "sm120 compiles {}'s row widening for its {} hidden columns, not {}",
            geometry.id, geometry.shape.hidden, op.width
        )));
    }
    rounding::plan(need.capacity)?;
    let symbol = if op.width == geometry.shape.hidden {
        reduce::rounding_symbol(geometry, Rounding::Widen)
    } else {
        super::dspark_entries::rounding_symbol(geometry, Rounding::Widen)
    };
    // One block widens one row, one conversion per element, and rows at or
    // past the live count return.
    let info = OpInfo::new(
        "sm120.row-rounding.widen.v1",
        rounding::shared_invariance(),
        vec![(symbol.clone(), 0)],
    );
    need.check(&info)?;
    Ok(Box::new(Widen {
        width: op.width,
        capacity: need.capacity,
        symbol,
        info,
    }))
}

struct Widen {
    width: u32,
    capacity: u32,
    symbol: String,
    info: OpInfo,
}

impl PreparedOp<WidenRowsBuffers<View>> for Widen {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &WidenRowsBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = counts.rows;
        if rows == 0 || rows > self.capacity {
            return Err(Refusal("row widening rows outside the prepared capacity"));
        }
        let elements = u64::from(rows) * u64::from(self.width);
        buffers.input.covers(elements * 2)?;
        buffers.output.covers(elements * 4)?;
        buffers.live_rows.covers(4)?;
        Ok(vec![Launch {
            symbol: self.symbol.clone(),
            rows,
            plan: rounding::plan(rows)?,
            arguments: rounding::arguments(
                buffers.input.handle,
                buffers.output.handle,
                buffers.live_rows.handle,
            )?,
            collective: false,
            persistent: false,
        }])
    }
}
