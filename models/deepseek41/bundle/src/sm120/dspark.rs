//! Own-op preparation for the model's Markov and confidence heads.

use super::confidence_parallel;
use super::confidence_staged;
use super::dspark_entries::{MARKOV, THREADS};
use super::markov_row;
use super::markov_staged;
use super::markov_tiled;
use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::contract::{Argument, ArgumentList, LaunchPlan, Refusal};
use kernel_core::properties::Invariance;
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::dspark::{
    CONFIDENCE_CONTRACT, ConfidenceHead, ConfidenceHeadBuffers, MARKOV_CONTRACT, MarkovStep,
    MarkovStepBuffers,
};

// Both heads launch through `launch`, which uses the shared block size.
const _: () = assert!(markov_tiled::THREADS == THREADS);
const _: () = assert!(confidence_staged::THREADS == THREADS);
const _: () = assert!(confidence_staged::MAX_CHAINS <= THREADS);

/// The tiled Markov step stages each weight tile once for up to eight rows;
/// a single row takes the row step, which stages coalesced tiles before its
/// dependency wait. All three compute every logit with the per-row step's
/// chain, so the choice keeps the bits.
const TILED_MIN_ROWS: u32 = 2;

pub fn markov(
    geometry: &Geometry,
    op: &MarkovStep,
    need: Need,
) -> Result<Prepared<MarkovStepBuffers<View>>, Refused> {
    op.validate(need)?;
    if need.capacity > 65535 || geometry.shape.dspark.markov_rank != op.rank {
        return Err(Refused::Rule(
            "sm120 DSpark Markov rank or grid capacity is unsupported".into(),
        ));
    }
    if op.rank > markov_row::MAX_RANK {
        return Err(Refused::Rule(
            "sm120 DSpark Markov rank exceeds the row step's chains".into(),
        ));
    }
    let mut info = info(markov_tiled::SYMBOL, MARKOV_CONTRACT, need)?;
    info.symbols.push((MARKOV.into(), 0));
    info.symbols.push((markov_row::SYMBOL.into(), 0));
    Ok(Box::new(Markov {
        op: *op,
        capacity: need.capacity,
        info,
    }))
}

/// The staged Markov step (the `dspark_markov_staged` candidate): two or
/// more rows take `markov_staged`, which loads each thread's weight chain
/// into registers before the dependency wait and reads base rows
/// `base_stride` elements apart, and one row takes the row step. Both keep
/// every logit's chain, fold and base add, so the bits match `markov`'s.
/// Only the rank the staged entry's chains cover is served.
pub fn markov_staged(
    geometry: &Geometry,
    op: &MarkovStep,
    base_stride: u32,
    need: Need,
) -> Result<Prepared<MarkovStepBuffers<View>>, Refused> {
    op.validate(need)?;
    if need.capacity > 65535
        || geometry.shape.dspark.markov_rank != op.rank
        || op.rank != markov_staged::RANK
        || base_stride < op.vocab_rows
    {
        return Err(Refused::Rule(
            "sm120 DSpark staged Markov shape is unsupported".into(),
        ));
    }
    let mut info = info(markov_staged::SYMBOL, MARKOV_CONTRACT, need)?;
    info.symbols.push((markov_row::SYMBOL.into(), 0));
    Ok(Box::new(Staged {
        op: *op,
        base_stride,
        capacity: need.capacity,
        info,
    }))
}

pub fn confidence(
    geometry: &Geometry,
    op: &ConfidenceHead,
    need: Need,
) -> Result<Prepared<ConfidenceHeadBuffers<View>>, Refused> {
    op.validate(need)?;
    if need.capacity > 65535
        || geometry.shape.dspark.markov_rank != op.markov_rank
        || geometry.shape.hidden != op.hidden
    {
        return Err(Refused::Rule(
            "sm120 DSpark confidence shape is unsupported".into(),
        ));
    }
    let info = info(confidence_parallel::SYMBOL, CONFIDENCE_CONTRACT, need)?;
    Ok(Box::new(Confidence {
        op: *op,
        capacity: need.capacity,
        info,
        symbol: confidence_parallel::SYMBOL,
    }))
}

/// The staged confidence head (the `dspark_confidence_staged` candidate):
/// each chain's weights load into registers before the dependency wait and
/// each row's operands into shared memory, then every chain and the fold
/// run as in `confidence`, so the scores keep their bits. Only operands of
/// whole chains, at most the block's, are served.
pub fn confidence_staged(
    geometry: &Geometry,
    op: &ConfidenceHead,
    need: Need,
) -> Result<Prepared<ConfidenceHeadBuffers<View>>, Refused> {
    op.validate(need)?;
    let total = op.hidden + op.markov_rank;
    if need.capacity > 65535
        || geometry.shape.dspark.markov_rank != op.markov_rank
        || geometry.shape.hidden != op.hidden
        || !total.is_multiple_of(confidence_staged::CHAIN)
        || total / confidence_staged::CHAIN > confidence_staged::MAX_CHAINS
    {
        return Err(Refused::Rule(
            "sm120 DSpark staged confidence shape is unsupported".into(),
        ));
    }
    let info = info(confidence_staged::SYMBOL, CONFIDENCE_CONTRACT, need)?;
    Ok(Box::new(Confidence {
        op: *op,
        capacity: need.capacity,
        info,
        symbol: confidence_staged::SYMBOL,
    }))
}

fn info(symbol: &str, contract: &str, need: Need) -> Result<OpInfo, Refused> {
    let info = OpInfo::new(
        contract,
        Invariance::any_rows(need.capacity),
        vec![(symbol.into(), 0)],
    );
    need.check(&info)?;
    Ok(info)
}

fn rows(counts: Counts, capacity: u32) -> Result<u32, Refusal> {
    if counts.rows == 0 || counts.rows > capacity {
        return Err(Refusal("DSpark head rows outside the prepared capacity"));
    }
    Ok(counts.rows)
}

fn span(rows: u32, stride: u32, width: u32, bytes: u64) -> Result<u64, Refusal> {
    ((u64::from(rows) - 1) * u64::from(stride) + u64::from(width))
        .checked_mul(bytes)
        .ok_or(Refusal("DSpark head byte span overflows"))
}

// Handles identify views, not addresses. The caller must also rule out
// overlaps between distinct handles, as required by the role contracts.
fn distinct(output: View, inputs: &[View]) -> Result<(), Refusal> {
    if inputs.iter().any(|input| input.handle == output.handle) {
        return Err(Refusal("DSpark head output aliases an input role"));
    }
    Ok(())
}

fn launch(symbol: &str, rows: u32, grid: [u32; 3], arguments: Vec<Argument>) -> Vec<Launch> {
    vec![Launch {
        symbol: symbol.into(),
        rows,
        plan: LaunchPlan {
            grid,
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1; 3],
            pdl: true,
        },
        arguments: ArgumentList(arguments),
        collective: false,
        persistent: false,
    }]
}

struct Markov {
    op: MarkovStep,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<MarkovStepBuffers<View>> for Markov {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &MarkovStepBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let op = self.op;
        b.embedding.covers(span(rows, op.rank, op.rank, 2)?)?;
        b.weight.covers(span(op.vocab_rows, op.rank, op.rank, 2)?)?;
        let bytes = span(rows, op.logits_stride, op.vocab_rows, 4)?;
        b.base.covers(bytes)?;
        b.output.covers(bytes)?;
        b.live_rows.covers(4)?;
        distinct(b.output, &[b.embedding, b.weight, b.live_rows])?;
        let arguments = vec![
            Argument::DeviceBuffer(b.embedding.handle),
            Argument::DeviceBuffer(b.weight.handle),
            Argument::DeviceBuffer(b.base.handle),
            Argument::DeviceBuffer(b.output.handle),
            Argument::DeviceBuffer(b.live_rows.handle),
            Argument::U32(op.rank),
            Argument::U32(op.vocab_rows),
            Argument::U32(op.logits_stride),
        ];
        if rows >= TILED_MIN_ROWS {
            return Ok(launch(
                markov_tiled::SYMBOL,
                rows,
                [op.vocab_rows.div_ceil(THREADS), 1, 1],
                arguments,
            ));
        }
        let mut launches = launch(
            markov_row::SYMBOL,
            rows,
            [op.vocab_rows.div_ceil(markov_row::COLUMNS), 1, 1],
            arguments,
        );
        launches[0].plan.block = [markov_row::THREADS, 1, 1];
        launches[0].plan.dynamic_shared_bytes = markov_row::shared_bytes(op.rank);
        Ok(launches)
    }
}

struct Confidence {
    op: ConfidenceHead,
    capacity: u32,
    info: OpInfo,
    symbol: &'static str,
}

impl PreparedOp<ConfidenceHeadBuffers<View>> for Confidence {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &ConfidenceHeadBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let op = self.op;
        b.hidden
            .covers(span(rows, op.hidden_stride, op.hidden, 4)?)?;
        b.markov
            .covers(span(rows, op.markov_rank, op.markov_rank, 2)?)?;
        b.weight.covers(u64::from(op.hidden + op.markov_rank) * 2)?;
        b.scores.covers(span(rows, op.score_stride, 1, 4)?)?;
        b.live_rows.covers(4)?;
        distinct(b.scores, &[b.hidden, b.markov, b.weight, b.live_rows])?;
        Ok(launch(
            self.symbol,
            rows,
            [rows, 1, 1],
            vec![
                Argument::DeviceBuffer(b.hidden.handle),
                Argument::DeviceBuffer(b.markov.handle),
                Argument::DeviceBuffer(b.weight.handle),
                Argument::DeviceBuffer(b.scores.handle),
                Argument::DeviceBuffer(b.live_rows.handle),
                Argument::U32(op.hidden),
                Argument::U32(op.markov_rank),
                Argument::U32(op.hidden_stride),
                Argument::U32(op.score_stride),
            ],
        ))
    }
}

struct Staged {
    op: MarkovStep,
    base_stride: u32,
    capacity: u32,
    info: OpInfo,
}

impl PreparedOp<MarkovStepBuffers<View>> for Staged {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(
        &self,
        b: &MarkovStepBuffers<View>,
        _: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = rows(counts, self.capacity)?;
        let op = self.op;
        b.embedding.covers(span(rows, op.rank, op.rank, 2)?)?;
        b.weight.covers(span(op.vocab_rows, op.rank, op.rank, 2)?)?;
        b.base
            .covers(span(rows, self.base_stride, op.vocab_rows, 4)?)?;
        b.output
            .covers(span(rows, op.logits_stride, op.vocab_rows, 4)?)?;
        b.live_rows.covers(4)?;
        distinct(b.output, &[b.embedding, b.weight, b.live_rows])?;
        // Base and output may alias only where both rows coincide.
        if b.base.handle == b.output.handle && self.base_stride != op.logits_stride && rows > 1 {
            return Err(Refusal(
                "DSpark staged Markov output aliases base at another stride",
            ));
        }
        if rows == 1 {
            let mut launches = launch(
                markov_row::SYMBOL,
                rows,
                [op.vocab_rows.div_ceil(markov_row::COLUMNS), 1, 1],
                vec![
                    Argument::DeviceBuffer(b.embedding.handle),
                    Argument::DeviceBuffer(b.weight.handle),
                    Argument::DeviceBuffer(b.base.handle),
                    Argument::DeviceBuffer(b.output.handle),
                    Argument::DeviceBuffer(b.live_rows.handle),
                    Argument::U32(op.rank),
                    Argument::U32(op.vocab_rows),
                    Argument::U32(op.logits_stride),
                ],
            );
            launches[0].plan.block = [markov_row::THREADS, 1, 1];
            launches[0].plan.dynamic_shared_bytes = markov_row::shared_bytes(op.rank);
            return Ok(launches);
        }
        let mut launches = launch(
            markov_staged::SYMBOL,
            rows,
            [op.vocab_rows.div_ceil(markov_staged::COLUMNS), 1, 1],
            vec![
                Argument::DeviceBuffer(b.embedding.handle),
                Argument::DeviceBuffer(b.weight.handle),
                Argument::DeviceBuffer(b.base.handle),
                Argument::DeviceBuffer(b.output.handle),
                Argument::DeviceBuffer(b.live_rows.handle),
                Argument::U32(op.vocab_rows),
                Argument::U32(self.base_stride),
                Argument::U32(op.logits_stride),
            ],
        );
        launches[0].plan.block = [markov_staged::THREADS, 1, 1];
        Ok(launches)
    }
}
