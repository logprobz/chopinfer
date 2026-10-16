//! The gathers through the kernel base's comm entries: the all-gather, the
//! row gather and the row exchange. They move bits and compute nothing;
//! each keeps one state on every rank, whose flags and epoch cells start
//! zeroed and serve no other op.

use kernel_core::backend::{Counts, Launch, Need, OpInfo, Prepared, PreparedOp, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews, RowGatherBuffers, ShardedRows};
use kernel_core::contract::{
    Argument, ArgumentList, CommQuery, DType, LaunchPlan, MAX_COMM_RANKS, Refusal, Validate,
};
use kernel_core::properties::Invariance;
use kernel_core::row_exchange::{
    EXCHANGE_ROWS_EPOCH_BYTES, EXCHANGE_ROWS_FLAG_BYTES, ExchangeRowsCall, exchange_rows_arguments,
    exchange_rows_plan,
};

use super::{MAX_CAPACITY, check_plan};
use crate::comm::{
    self, GATHER_ROWS_EPOCH_BYTES, GATHER_ROWS_FLAG_BYTES, GatherRowsCall, gather_rows_arguments,
    gather_rows_plan,
};

/// The all-gather `op` that rank `rank` launches onto `target`, or onto
/// every rank.
pub(super) fn all_gather(
    op: &CommQuery,
    rank: u32,
    target: Option<u32>,
    need: Need,
) -> Result<Prepared<GatherBuffers<View>>, Refused> {
    let op = AllGatherOp::new(*op, rank, target, need.capacity)?;
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The row gather `op` that rank `rank` launches onto `target`.
pub(super) fn gather_rows(
    op: &ShardedRows,
    rank: u32,
    target: u32,
    need: Need,
) -> Result<Prepared<RowGatherBuffers<View>>, Refused> {
    let rows = RowsOp::new(*op, rank, need.capacity, "gather_rows", gather_rows_plan())?;
    if target >= op.ranks {
        return Err(Refused::Rule(
            "a row gather targets one of its ranks".into(),
        ));
    }
    let info = rows.info(GATHER_ROWS_FLAG_BYTES, GATHER_ROWS_EPOCH_BYTES);
    let op = GatherRowsOp { rows, target, info };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The row exchange `op` that rank `rank` launches.
pub(super) fn exchange_rows(
    op: &ShardedRows,
    rank: u32,
    need: Need,
) -> Result<Prepared<GatherBuffers<View>>, Refused> {
    let rows = RowsOp::new(
        *op,
        rank,
        need.capacity,
        "exchange_rows",
        exchange_rows_plan(),
    )?;
    let info = rows.info(EXCHANGE_ROWS_FLAG_BYTES, EXCHANGE_ROWS_EPOCH_BYTES);
    let op = ExchangeRowsOp { rows, info };
    need.check(&op.info)?;
    Ok(Box::new(op))
}

/// The bytes of one element of the entries' types.
fn element_bytes(dtype: DType) -> Result<u64, Refused> {
    match dtype {
        DType::F32 => Ok(4),
        DType::Bf16 => Ok(2),
        _ => Err(Refused::Rule(
            "the SM120 gathers move F32 and BF16 elements".into(),
        )),
    }
}

/// SM120's limits of every gather: the entries' pointer slots hold at most
/// `MAX_COMM_RANKS` ranks.
fn check_ranks(ranks: u32, rank: u32, capacity: u32) -> Result<(), Refused> {
    if ranks as usize > MAX_COMM_RANKS {
        return Err(Refused::Rule(format!(
            "an SM120 collective takes at most {MAX_COMM_RANKS} ranks"
        )));
    }
    if rank >= ranks || capacity == 0 {
        return Err(Refused::Rule(
            "a gather needs a rank below its ranks and rows".into(),
        ));
    }
    if capacity > MAX_CAPACITY {
        return Err(Refused::Rule(format!(
            "an SM120 gather serves at most {MAX_CAPACITY} rows"
        )));
    }
    Ok(())
}

/// The guarantees the comm declarations claim for `symbol`; none for an
/// undeclared entry.
fn declared(symbol: &str) -> Invariance {
    comm::declared_entries()
        .into_iter()
        .find(|entry| entry.symbol == symbol)
        .map_or(Invariance::NONE, |entry| entry.invariance)
}

/// Every rank's output and its flag and epoch views, after checking each
/// state view against `state`. Outputs are checked by the caller.
fn peers(ranks: &[RankViews<View>], count: u32, state: &[u64]) -> Result<Vec<[View; 3]>, Refusal> {
    if ranks.len() != count as usize {
        return Err(Refusal("a gather takes the views of every rank"));
    }
    ranks
        .iter()
        .map(|views| {
            let [flags, epoch] = views.state[..] else {
                return Err(Refusal("a gather rank lacks its state regions"));
            };
            flags.covers(state[0])?;
            epoch.covers(state[1])?;
            Ok([views.output, flags, epoch])
        })
        .collect()
}

fn checked(rows: u32, capacity: u32) -> Result<u32, Refusal> {
    if rows == 0 || rows > capacity {
        return Err(Refusal("gather rows exceed the prepared capacity"));
    }
    Ok(rows)
}

fn launch(symbol: &str, rows: u32, plan: LaunchPlan, arguments: ArgumentList) -> Launch {
    Launch {
        symbol: symbol.into(),
        rows,
        plan,
        arguments,
        collective: true,
        persistent: false,
    }
}

/// A gather of one fixed block per rank (`chop_comm_allgather_*_v1`):
/// every launch moves the whole block, whatever its rows, so the output
/// layout is the same at every step. One CTA per 4096-element tile, with a
/// flag per source and tile, an acknowledgment per destination and tile,
/// and an epoch cell per tile. It launches as the comm planner plans it
/// (`comm::plan`); the context-parallel group gather
/// (`context_parallel::group`, planned with no workspace) becomes an op of
/// the context-parallel family.
struct AllGatherOp {
    ranks: u32,
    rank: u32,
    /// The entry's target: a rank, or the rank count for every rank.
    target: u32,
    elements: u64,
    element_bytes: u64,
    capacity: u32,
    symbol: &'static str,
    plan: LaunchPlan,
    info: OpInfo,
}

impl AllGatherOp {
    fn new(op: CommQuery, rank: u32, target: Option<u32>, capacity: u32) -> Result<Self, Refused> {
        let CommQuery::AllGather {
            ranks,
            elements_per_rank: elements,
            dtype,
        } = op
        else {
            return Err(Refused::Rule(
                "an all-gather takes an all-gather query".into(),
            ));
        };
        op.validate()?;
        check_ranks(ranks, rank, capacity)?;
        let element_bytes = element_bytes(dtype)?;
        let target = match target {
            Some(target) if target >= ranks => {
                return Err(Refused::Rule(
                    "an all-gather targets one of its ranks".into(),
                ));
            }
            Some(target) => target,
            None => ranks,
        };
        // Every receiving output holds all ranks' blocks.
        u64::from(ranks)
            .checked_mul(elements)
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| Refused::Rule("all-gather sizes overflow u64".into()))?;
        let plan = comm::plan(&op, 1)?;
        check_plan(&plan)?;
        let tiles = elements.div_ceil(4096);
        let symbol = if dtype == DType::F32 {
            "chop_comm_allgather_f32_v1"
        } else {
            "chop_comm_allgather_bf16_v1"
        };
        let mut info = OpInfo::new(
            format!("sm120.comm.allgather.v1[{symbol}]"),
            declared(symbol),
            vec![(symbol.to_owned(), 0)],
        );
        info.state = vec![2 * u64::from(ranks) * tiles * 4, tiles * 4];
        Ok(Self {
            ranks,
            rank,
            target,
            elements,
            element_bytes,
            capacity,
            symbol,
            plan,
            info,
        })
    }
}

impl PreparedOp<GatherBuffers<View>> for AllGatherOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatherBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let rows = checked(counts.rows, self.capacity)?;
        let views = peers(&buffers.ranks, self.ranks, &self.info.state)?;
        buffers.input.covers(self.elements * self.element_bytes)?;
        let output_bytes = u64::from(self.ranks) * self.elements * self.element_bytes;
        for (rank, [output, ..]) in views.iter().enumerate() {
            if self.target == self.ranks || rank == self.target as usize {
                output.covers(output_bytes)?;
            }
        }
        let arguments = ArgumentList(vec![
            Argument::DeviceBuffer(buffers.input.handle),
            Argument::DevicePointerArray(views.iter().map(|v| v[0].handle).collect()),
            Argument::DevicePointerArray(views.iter().map(|v| v[1].handle).collect()),
            Argument::DeviceBuffer(views[self.rank as usize][2].handle),
            Argument::U64(self.elements),
            Argument::U32(self.rank),
            Argument::U32(self.ranks),
            Argument::U32(self.target),
        ]);
        arguments.validate()?;
        Ok(vec![launch(self.symbol, rows, self.plan, arguments)])
    }
}

/// What the row gather and exchange share: sharded rows of one dtype, an
/// entry of fixed geometry and one state of fixed size.
struct RowsOp {
    rows: ShardedRows,
    rank: u32,
    element_bytes: u64,
    capacity: u32,
    symbol: String,
    plan: LaunchPlan,
}

impl RowsOp {
    fn new(
        rows: ShardedRows,
        rank: u32,
        capacity: u32,
        entry: &str,
        plan: LaunchPlan,
    ) -> Result<Self, Refused> {
        rows.validate()?;
        check_ranks(rows.ranks, rank, capacity)?;
        let element_bytes = element_bytes(rows.dtype)?;
        // Every output holds the capacity's whole rows.
        u64::from(capacity)
            .checked_mul(u64::from(rows.ranks))
            .and_then(|rows_shards| rows_shards.checked_mul(rows.shard))
            .and_then(|elements| elements.checked_mul(element_bytes))
            .ok_or_else(|| Refused::Rule("row move sizes overflow u64".into()))?;
        check_plan(&plan)?;
        let dtype = if rows.dtype == DType::F32 {
            "f32"
        } else {
            "bf16"
        };
        Ok(Self {
            rows,
            rank,
            element_bytes,
            capacity,
            symbol: format!("chop_comm_{entry}_{dtype}_v1"),
            plan,
        })
    }

    fn info(&self, flag_bytes: u64, epoch_bytes: u64) -> OpInfo {
        let mut info = OpInfo::new(
            format!("sm120.comm.rows.v1[{}]", self.symbol),
            declared(&self.symbol),
            vec![(self.symbol.clone(), 0)],
        );
        info.state = vec![flag_bytes, epoch_bytes];
        info
    }

    /// Bytes of `rows` shard rows, and of `rows` whole rows.
    fn bytes(&self, rows: u32) -> (u64, u64) {
        let shards = u64::from(rows) * self.rows.shard * self.element_bytes;
        (shards, shards * u64::from(self.rows.ranks))
    }
}

/// `chop_comm_gather_rows_*_v1`: eight independent CTAs, each with its own
/// epoch cell, a data flag per source and an acknowledgment word.
struct GatherRowsOp {
    rows: RowsOp,
    target: u32,
    info: OpInfo,
}

impl PreparedOp<RowGatherBuffers<View>> for GatherRowsOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &RowGatherBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = &self.rows;
        let rows = checked(counts.rows, op.capacity)?;
        let views = peers(&buffers.ranks, op.rows.ranks, &self.info.state)?;
        // Indices may name any row below the capacity.
        let (shards, whole) = match buffers.indices {
            Some(indices) => {
                indices.covers(u64::from(rows) * 4)?;
                op.bytes(op.capacity)
            }
            None => op.bytes(rows),
        };
        buffers.input.covers(shards)?;
        views[self.target as usize][0].covers(whole)?;
        let arguments = gather_rows_arguments(GatherRowsCall {
            input: buffers.input.handle,
            outputs: views.iter().map(|v| v[0].handle).collect(),
            flags: views.iter().map(|v| v[1].handle).collect(),
            epoch: views[op.rank as usize][2].handle,
            row_indices: buffers.indices.map(|indices| indices.handle),
            rows,
            shard: op.rows.shard,
            rank: op.rank,
            ranks: op.rows.ranks,
            target: self.target,
        })?;
        Ok(vec![launch(&op.symbol, rows, op.plan, arguments)])
    }
}

/// `chop_comm_exchange_rows_*_v1`: one cluster of eight CTAs, each with
/// its own epoch cell, a data flag per source and an acknowledgment per
/// receiver.
struct ExchangeRowsOp {
    rows: RowsOp,
    info: OpInfo,
}

impl PreparedOp<GatherBuffers<View>> for ExchangeRowsOp {
    fn info(&self) -> &OpInfo {
        &self.info
    }

    fn launches(
        &self,
        buffers: &GatherBuffers<View>,
        _workspace: View,
        counts: Counts,
    ) -> Result<Vec<Launch>, Refusal> {
        let op = &self.rows;
        let rows = checked(counts.rows, op.capacity)?;
        let views = peers(&buffers.ranks, op.rows.ranks, &self.info.state)?;
        let (shards, whole) = op.bytes(rows);
        buffers.input.covers(shards)?;
        for [output, ..] in &views {
            output.covers(whole)?;
        }
        let arguments = exchange_rows_arguments(ExchangeRowsCall {
            input: buffers.input.handle,
            outputs: views.iter().map(|v| v[0].handle).collect(),
            flags: views.iter().map(|v| v[1].handle).collect(),
            epoch: views[op.rank as usize][2].handle,
            rows,
            shard: op.rows.shard,
            rank: op.rank,
            ranks: op.rows.ranks,
        })?;
        Ok(vec![launch(&op.symbol, rows, op.plan, arguments)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Sm120, view};
    use kernel_core::backend::Backend;

    /// Every rank's views: rank r's output 10 + r, its state regions
    /// 100 * (r + 1) + i, each of its declared size.
    fn ranks_of(info: &OpInfo, ranks: u32) -> Vec<RankViews<View>> {
        (0..ranks)
            .map(|r| RankViews {
                output: view(10 + u64::from(r), u64::MAX),
                state: info
                    .state
                    .iter()
                    .enumerate()
                    .map(|(i, &bytes)| view(100 * (u64::from(r) + 1) + i as u64, bytes))
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn all_gather_launches_the_comm_planners_launch() {
        // Pairs of a 64-row draft argmax onto every rank, and 1024-wide
        // shards onto rank 1.
        for (ranks, elements, dtype, target) in [
            (2, 128, DType::F32, None),
            (4, 128, DType::F32, None),
            (8, 4 * 1024 * 5, DType::Bf16, Some(1)),
        ] {
            let query = CommQuery::AllGather {
                ranks,
                elements_per_rank: elements,
                dtype,
            };
            let rank = ranks - 1;
            let prepared = Sm120::new()
                .all_gather(&query, rank, target, Need::rows(64))
                .unwrap();
            let info = prepared.info().clone();
            let tiles = elements.div_ceil(4096);
            assert_eq!(info.state, [2 * u64::from(ranks) * tiles * 4, tiles * 4]);
            let buffers = GatherBuffers {
                input: view(1, u64::MAX),
                ranks: ranks_of(&info, ranks),
            };
            let launches = prepared
                .launches(&buffers, View::EMPTY, Counts::rows(7))
                .unwrap();
            assert_eq!(launches.len(), 1);
            let launch = &launches[0];
            let suffix = if dtype == DType::F32 { "f32" } else { "bf16" };
            assert_eq!(launch.symbol, format!("chop_comm_allgather_{suffix}_v1"));
            assert_eq!(launch.plan, comm::plan(&query, 1).unwrap());
            assert!(launch.collective && !launch.persistent && launch.rows == 7);
            let peers = |base: u64| (0..u64::from(ranks)).map(|r| base + r).collect();
            let expected = ArgumentList(vec![
                Argument::DeviceBuffer(1),
                Argument::DevicePointerArray(peers(10)),
                Argument::DevicePointerArray(
                    (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
                ),
                Argument::DeviceBuffer(100 * u64::from(ranks) + 1),
                Argument::U64(elements),
                Argument::U32(rank),
                Argument::U32(ranks),
                Argument::U32(target.unwrap_or(ranks)),
            ]);
            assert_eq!(launch.arguments, expected);
            // A receiving output too small for every block is refused, a
            // rank that does not receive is not checked.
            let mut short = buffers.clone();
            let receiver = target.unwrap_or(0) as usize;
            short.ranks[receiver].output.bytes =
                u64::from(ranks) * elements * if dtype == DType::F32 { 4 } else { 2 } - 1;
            assert!(
                prepared
                    .launches(&short, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
            if let Some(target) = target {
                let mut other = buffers.clone();
                other.ranks[(target as usize + 1) % ranks as usize]
                    .output
                    .bytes = 0;
                assert!(
                    prepared
                        .launches(&other, View::EMPTY, Counts::rows(1))
                        .is_ok()
                );
            }
            assert!(
                prepared
                    .launches(&buffers, View::EMPTY, Counts::rows(65))
                    .is_err()
            );
            let mut stale = buffers.clone();
            stale.ranks[0].state[1].bytes -= 1;
            assert!(
                prepared
                    .launches(&stale, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
    }

    #[test]
    fn row_gather_and_exchange_launch_the_bundle_builders_launches() {
        for (ranks, shard, dtype) in [
            (2, 77_824, DType::F32),
            (4, 38_912, DType::F32),
            (8, 8, DType::Bf16),
        ] {
            let op = ShardedRows {
                ranks,
                shard,
                dtype,
            };
            let suffix = if dtype == DType::F32 { "f32" } else { "bf16" };
            let bytes = if dtype == DType::F32 { 4 } else { 2 };
            let backend = Sm120::new();
            let gather = backend.gather_rows(&op, 1, 0, Need::rows(32)).unwrap();
            let info = gather.info().clone();
            assert_eq!(
                info.state,
                [GATHER_ROWS_FLAG_BYTES, GATHER_ROWS_EPOCH_BYTES]
            );
            assert_eq!(info.guarantees, Invariance::any_rows(8192));
            let mut buffers = RowGatherBuffers {
                input: view(1, u64::MAX),
                indices: None,
                ranks: ranks_of(&info, ranks),
            };
            for indices in [None, Some(view(2, 32 * 4))] {
                buffers.indices = indices;
                let launches = gather
                    .launches(&buffers, View::EMPTY, Counts::rows(5))
                    .unwrap();
                let expected = gather_rows_arguments(GatherRowsCall {
                    input: 1,
                    outputs: (0..u64::from(ranks)).map(|r| 10 + r).collect(),
                    flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
                    epoch: 201,
                    row_indices: indices.map(|v| v.handle),
                    rows: 5,
                    shard,
                    rank: 1,
                    ranks,
                    target: 0,
                })
                .unwrap();
                assert_eq!(launches.len(), 1);
                assert_eq!(
                    launches[0].symbol,
                    format!("chop_comm_gather_rows_{suffix}_v1")
                );
                assert_eq!(launches[0].plan, gather_rows_plan());
                assert_eq!(launches[0].arguments, expected);
                assert!(launches[0].collective);
            }
            // Indices may name any row below the capacity, so the target
            // holds the capacity's whole rows.
            buffers.indices = Some(view(2, 32 * 4));
            buffers.ranks[0].output.bytes = 5 * u64::from(ranks) * shard * bytes;
            assert!(
                gather
                    .launches(&buffers, View::EMPTY, Counts::rows(5))
                    .is_err()
            );
            buffers.indices = None;
            assert!(
                gather
                    .launches(&buffers, View::EMPTY, Counts::rows(5))
                    .is_ok()
            );
            buffers.ranks[0].output.bytes -= 1;
            assert!(
                gather
                    .launches(&buffers, View::EMPTY, Counts::rows(5))
                    .is_err()
            );

            let exchange = backend
                .exchange_rows(&op, ranks - 1, Need::rows(32))
                .unwrap();
            let info = exchange.info().clone();
            assert_eq!(
                info.state,
                [EXCHANGE_ROWS_FLAG_BYTES, EXCHANGE_ROWS_EPOCH_BYTES]
            );
            let mut buffers = GatherBuffers {
                input: view(1, u64::MAX),
                ranks: ranks_of(&info, ranks),
            };
            let launches = exchange
                .launches(&buffers, View::EMPTY, Counts::rows(32))
                .unwrap();
            let expected = exchange_rows_arguments(ExchangeRowsCall {
                input: 1,
                outputs: (0..u64::from(ranks)).map(|r| 10 + r).collect(),
                flags: (0..u64::from(ranks)).map(|r| 100 * (r + 1)).collect(),
                epoch: 100 * u64::from(ranks) + 1,
                rows: 32,
                shard,
                rank: ranks - 1,
                ranks,
            })
            .unwrap();
            assert_eq!(
                launches[0].symbol,
                format!("chop_comm_exchange_rows_{suffix}_v1")
            );
            assert_eq!(launches[0].plan, exchange_rows_plan());
            assert_eq!(launches[0].arguments, expected);
            assert!(launches[0].collective);
            // Every output holds every row's offset.
            buffers.ranks[0].output.bytes = 32 * u64::from(ranks) * shard * bytes - 1;
            assert!(
                exchange
                    .launches(&buffers, View::EMPTY, Counts::rows(32))
                    .is_err()
            );
            assert!(
                exchange
                    .launches(&buffers, View::EMPTY, Counts::rows(31))
                    .is_ok()
            );
            assert!(
                exchange
                    .launches(&buffers, View::EMPTY, Counts::rows(33))
                    .is_err()
            );
            buffers.ranks.pop();
            assert!(
                exchange
                    .launches(&buffers, View::EMPTY, Counts::rows(1))
                    .is_err()
            );
        }
    }

    #[test]
    fn gathers_refuse_sm120_limits_and_unmet_exactness() {
        let backend = Sm120::new();
        let rows = |ranks, dtype| ShardedRows {
            ranks,
            shard: 64,
            dtype,
        };
        // Sixteen ranks are a valid contract past SM120's pointer slots.
        assert_eq!(rows(16, DType::F32).validate(), Ok(()));
        assert!(matches!(
            backend.exchange_rows(&rows(16, DType::F32), 0, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("at most 8 ranks")
        ));
        let gather = CommQuery::AllGather {
            ranks: 16,
            elements_per_rank: 64,
            dtype: DType::F32,
        };
        assert_eq!(gather.validate(), Ok(()));
        assert!(matches!(
            backend.all_gather(&gather, 0, None, Need::rows(4)),
            Err(Refused::Rule(rule)) if rule.contains("at most 8 ranks")
        ));
        // Element types without an entry, ranks and targets out of range.
        assert!(
            backend
                .gather_rows(&rows(2, DType::F16), 0, 0, Need::rows(4))
                .is_err()
        );
        assert!(
            backend
                .gather_rows(&rows(2, DType::F32), 0, 2, Need::rows(4))
                .is_err()
        );
        assert!(
            backend
                .exchange_rows(&rows(2, DType::F32), 2, Need::rows(4))
                .is_err()
        );
        let gather = CommQuery::AllGather {
            ranks: 2,
            elements_per_rank: 64,
            dtype: DType::F32,
        };
        assert!(
            backend
                .all_gather(&gather, 0, Some(2), Need::rows(4))
                .is_err()
        );
        assert!(
            backend
                .all_gather(&gather, 0, None, Need::rows(MAX_CAPACITY + 1))
                .is_err()
        );
        let reduce = CommQuery::AllReduce {
            ranks: 2,
            elements: 64,
            dtype: DType::F32,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        assert!(backend.all_gather(&reduce, 0, None, Need::rows(4)).is_err());
        // The row moves keep rows past the declared 8192, the BF16
        // all-gather claims nothing.
        assert!(matches!(
            backend.gather_rows(
                &rows(2, DType::F32),
                0,
                0,
                Need::rows(4).with_exact(Invariance::any_rows(16384))
            ),
            Err(Refused::Inexact { .. })
        ));
        let bf16 = CommQuery::AllGather {
            ranks: 2,
            elements_per_rank: 64,
            dtype: DType::Bf16,
        };
        assert!(matches!(
            backend.all_gather(
                &bf16,
                0,
                None,
                Need::rows(4).with_exact(Invariance::one_row(4))
            ),
            Err(Refused::Inexact { .. })
        ));
    }

    /// The F32 all-gather copies words, so it is declared a move of up to
    /// 8192 rows: it grants any claim within that and refuses wider ones.
    #[test]
    fn the_f32_all_gather_grants_its_move_declaration() {
        let backend = Sm120::new();
        let gather = CommQuery::AllGather {
            ranks: 2,
            elements_per_rank: 64,
            dtype: DType::F32,
        };
        let op = backend
            .all_gather(
                &gather,
                0,
                None,
                Need::rows(4).with_exact(Invariance::any_rows(8192)),
            )
            .unwrap();
        assert_eq!(op.info().guarantees, Invariance::any_rows(8192));
        assert!(
            backend
                .all_gather(
                    &gather,
                    1,
                    Some(0),
                    Need::rows(4).with_exact(Invariance::one_row(4)),
                )
                .is_ok()
        );
        assert!(matches!(
            backend.all_gather(
                &gather,
                0,
                None,
                Need::rows(4).with_exact(Invariance::any_rows(16384))
            ),
            Err(Refused::Inexact { .. })
        ));
    }
}
