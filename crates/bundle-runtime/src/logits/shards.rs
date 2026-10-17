//! The ops around a vocabulary-split head, prepared once at create: the
//! gather by index of the rows the head scores, and the moves of the
//! head's logits shards to full-vocabulary rows, the row exchange of
//! decode and verify rows to the rank that owns each row block and the
//! gather of prefill rows to rank 0.
//!
//! The model states every need and records every ask (`Asker`); each op
//! returns the prepared op and its role views, bound through the model's
//! binder in a fixed order, and the model launches them. Every prepared op
//! is asked under its name below, the head rows first, then each rank's
//! exchange and gather in rank order.

use kernel_core::backend::{Backend, Counts, Need, OpInfo, Prepared, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews, RowGatherBuffers, ShardedRows};
use kernel_core::contract::{DType, Refusal};
use kernel_core::row_copy::{GatherByIndex, GatherByIndexBuffers};
use runtime_bundle_abi::DeviceBufferSpanV1 as Span;

/// The model's record of what each op asked at create.
pub trait Asker {
    /// The need an op is asked with, from the need the model states.
    fn need(&self, need: Need) -> Need;
    /// Records that the op `what` was prepared for `need` and is `info`.
    fn record(&self, what: String, need: Need, info: &OpInfo);
}

/// The needs of the head's ops: the head rows' gather, each rank's
/// exchange and each rank's gather to rank 0.
#[derive(Clone, Copy, Debug)]
pub struct Needs {
    pub head_rows: Need,
    pub exchange: Need,
    pub gather: Need,
}

/// One rank's spans a logits move reaches on every rank: the full rows it
/// receives, and its flags and epoch.
#[derive(Clone, Copy, Debug)]
pub struct PeerSpans {
    pub output: Span,
    pub flags: Span,
    pub epoch: Span,
}

/// A prepared op with its role views and launch counts.
pub struct Bound<'a, B> {
    pub op: &'a Prepared<B>,
    pub buffers: B,
    pub counts: Counts,
}

/// The ops around a vocabulary-split head of an instance.
pub struct ShardedHead {
    head_rows: Prepared<GatherByIndexBuffers<View>>,
    /// Each rank's exchange of a decode or verify phase's rows.
    exchanges: Vec<Prepared<GatherBuffers<View>>>,
    /// Each rank's gather of a prefill's logits rows to rank 0.
    gathers: Vec<Prepared<RowGatherBuffers<View>>>,
}

/// The op `prepare` makes for the need `asker` asks of `need`, recorded as
/// `what`.
fn ask<B>(
    asker: &dyn Asker,
    what: String,
    need: Need,
    prepare: impl FnOnce(Need) -> Result<Prepared<B>, Refused>,
) -> Result<Prepared<B>, Refused> {
    let need = asker.need(need);
    let prepared = prepare(need)?;
    asker.record(what, need, prepared.info());
    Ok(prepared)
}

impl ShardedHead {
    /// The ops of a head over `ranks` ranks of `shard_rows` F32 vocabulary
    /// rows each, whose input rows `head_rows` gathers.
    pub fn prepare(
        backend: &dyn Backend,
        head_rows: &GatherByIndex,
        ranks: u32,
        shard_rows: u64,
        needs: Needs,
        asker: &dyn Asker,
    ) -> Result<ShardedHead, Refused> {
        let rows = ShardedRows {
            ranks,
            shard: shard_rows,
            dtype: DType::F32,
        };
        let head_rows = ask(asker, "logits.head_rows".into(), needs.head_rows, |need| {
            backend.gather_by_index(head_rows, need)
        })?;
        let mut head = ShardedHead {
            head_rows,
            exchanges: Vec::new(),
            gathers: Vec::new(),
        };
        for rank in 0..ranks {
            let exchange = ask(
                asker,
                format!("logits.exchange rank {rank}"),
                needs.exchange,
                |need| backend.exchange_rows(&rows, rank, need),
            )?;
            head.exchanges.push(exchange);
            let gather = ask(
                asker,
                format!("logits.gather rank {rank}"),
                needs.gather,
                |need| backend.gather_rows(&rows, rank, 0, need),
            )?;
            head.gathers.push(gather);
        }
        Ok(head)
    }

    /// The gather of at most `rows` head rows by index; with `grid`
    /// nonzero, gridded over `grid` rows (a step that pads to a row
    /// bucket). `view` binds the spans in role order.
    pub fn head_rows(
        &self,
        rows: u32,
        grid: u32,
        spans: GatherByIndexBuffers<Span>,
        view: &mut dyn FnMut(Span) -> View,
    ) -> Bound<'_, GatherByIndexBuffers<View>> {
        let buffers = GatherByIndexBuffers {
            source: view(spans.source),
            row_indices: view(spans.row_indices),
            count: view(spans.count),
            destination: view(spans.destination),
        };
        Bound {
            op: &self.head_rows,
            buffers,
            counts: Counts {
                padded_rows: grid,
                ..Counts::rows(rows)
            },
        }
    }

    /// Rank `rank`'s exchange of `rows` rows of its `local` shards to their
    /// owners' outputs.
    pub fn exchange(
        &self,
        rank: u32,
        rows: u32,
        local: Span,
        peers: &[PeerSpans],
        view: &mut dyn FnMut(Span) -> View,
    ) -> Result<Bound<'_, GatherBuffers<View>>, Refusal> {
        let op = of(&self.exchanges, rank)?;
        let input = view(local);
        let ranks = peer_views(rank, peers, view)?;
        Ok(Bound {
            op,
            buffers: GatherBuffers { input, ranks },
            counts: Counts::rows(rows),
        })
    }

    /// Rank `rank`'s gather of `rows` rows of its `local` shards to rank
    /// 0's output.
    pub fn gather(
        &self,
        rank: u32,
        rows: u32,
        local: Span,
        peers: &[PeerSpans],
        view: &mut dyn FnMut(Span) -> View,
    ) -> Result<Bound<'_, RowGatherBuffers<View>>, Refusal> {
        let op = of(&self.gathers, rank)?;
        let input = view(local);
        let ranks = peer_views(rank, peers, view)?;
        Ok(Bound {
            op,
            buffers: RowGatherBuffers {
                input,
                indices: None,
                ranks,
            },
            counts: Counts::rows(rows),
        })
    }
}

fn of<T>(ops: &[T], rank: u32) -> Result<&T, Refusal> {
    ops.get(rank as usize)
        .ok_or(Refusal("the head has no logits move for this rank"))
}

/// Every rank's views, bound in the order the moves pass them: the outputs,
/// the flags, then `rank`'s epoch (the one its launch takes) before the
/// others'.
fn peer_views(
    rank: u32,
    peers: &[PeerSpans],
    view: &mut dyn FnMut(Span) -> View,
) -> Result<Vec<RankViews<View>>, Refusal> {
    let own = peers
        .get(rank as usize)
        .ok_or(Refusal("the head's logits move lacks this rank's views"))?;
    let outputs: Vec<View> = peers.iter().map(|peer| view(peer.output)).collect();
    let flags: Vec<View> = peers.iter().map(|peer| view(peer.flags)).collect();
    let own_epoch = view(own.epoch);
    Ok(peers
        .iter()
        .enumerate()
        .map(|(peer, spans)| RankViews {
            output: outputs[peer],
            state: vec![
                flags[peer],
                if peer == rank as usize {
                    own_epoch
                } else {
                    view(spans.epoch)
                },
            ],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use kernel_core::backend::{Launch, PreparedOp};
    use kernel_core::properties::Invariance;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;

    /// A backend whose every op records its name and the views it binds.
    struct Echo;

    struct EchoOp(OpInfo);

    impl<B> PreparedOp<B> for EchoOp {
        fn info(&self) -> &OpInfo {
            &self.0
        }

        fn launches(&self, _: &B, _: View, _: Counts) -> Result<Vec<Launch>, Refusal> {
            Ok(Vec::new())
        }
    }

    fn echo<B>(name: String) -> Result<Prepared<B>, Refused> {
        Ok(Box::new(EchoOp(OpInfo::new(
            name,
            Invariance::NONE,
            Vec::new(),
        ))))
    }

    impl Backend for Echo {
        fn name(&self) -> &str {
            "echo"
        }

        fn gather_by_index(
            &self,
            op: &GatherByIndex,
            need: Need,
        ) -> Result<Prepared<GatherByIndexBuffers<View>>, Refused> {
            echo(format!("by index {} at {}", op.width, need.capacity))
        }

        fn exchange_rows(
            &self,
            op: &ShardedRows,
            rank: u32,
            need: Need,
        ) -> Result<Prepared<GatherBuffers<View>>, Refused> {
            echo(format!(
                "exchange {} of {} at {}",
                rank, op.shard, need.capacity
            ))
        }

        fn gather_rows(
            &self,
            op: &ShardedRows,
            rank: u32,
            target: u32,
            need: Need,
        ) -> Result<Prepared<RowGatherBuffers<View>>, Refused> {
            echo(format!(
                "gather {rank} to {target} of {} at {}",
                op.shard, need.capacity
            ))
        }
    }

    #[derive(Default)]
    struct Record(RefCell<Vec<(String, u32, String)>>);

    impl Asker for Record {
        fn need(&self, need: Need) -> Need {
            Need::rows(need.capacity + 1)
        }

        fn record(&self, what: String, need: Need, info: &OpInfo) {
            self.0
                .borrow_mut()
                .push((what, need.capacity, info.implementation.clone()));
        }
    }

    fn span(raw: u64) -> Span {
        Span {
            buffer: BundleDeviceBufferHandleV1 { raw },
            offset: 0,
            length: 16,
        }
    }

    fn head(asks: &Record) -> ShardedHead {
        let rows = GatherByIndex {
            dtype: DType::Bf16,
            width: 64,
            alignment_bytes: 16,
        };
        let needs = Needs {
            head_rows: Need::rows(8),
            exchange: Need::rows(4),
            gather: Need::rows(2),
        };
        ShardedHead::prepare(&Echo, &rows, 2, 100, needs, asks).unwrap()
    }

    #[test]
    fn every_op_is_asked_by_name_in_order_with_the_asked_need() {
        let asks = Record::default();
        head(&asks);
        let expected = [
            ("logits.head_rows", 9, "by index 64 at 9"),
            ("logits.exchange rank 0", 5, "exchange 0 of 100 at 5"),
            ("logits.gather rank 0", 3, "gather 0 to 0 of 100 at 3"),
            ("logits.exchange rank 1", 5, "exchange 1 of 100 at 5"),
            ("logits.gather rank 1", 3, "gather 1 to 0 of 100 at 3"),
        ];
        let asked = asks.0.into_inner();
        assert_eq!(asked.len(), expected.len());
        for ((what, capacity, implementation), expected) in asked.iter().zip(expected) {
            assert_eq!(
                (what.as_str(), *capacity, implementation.as_str()),
                expected
            );
        }
    }

    #[test]
    fn moves_bind_outputs_flags_then_the_own_epoch_first() {
        let head = head(&Record::default());
        let peers = [
            PeerSpans {
                output: span(10),
                flags: span(20),
                epoch: span(30),
            },
            PeerSpans {
                output: span(11),
                flags: span(21),
                epoch: span(31),
            },
        ];
        let mut bound = Vec::new();
        let mut view = |s: Span| {
            bound.push(s.buffer.raw);
            View {
                handle: bound.len() as u64,
                bytes: s.length,
            }
        };
        let exchange = head.exchange(1, 3, span(1), &peers, &mut view).unwrap();
        assert_eq!(exchange.counts, Counts::rows(3));
        let handles: Vec<_> = exchange
            .buffers
            .ranks
            .iter()
            .map(|r| (r.output.handle, r.state[0].handle, r.state[1].handle))
            .collect();
        assert_eq!(handles, [(2, 4, 7), (3, 5, 6)]);
        assert!(head.gather(2, 1, span(1), &peers, &mut view).is_err());
        assert!(head.gather(1, 1, span(1), &peers[..1], &mut view).is_err());
        assert_eq!(bound[..7], [1, 10, 11, 20, 21, 31, 30]);
    }

    #[test]
    fn the_head_rows_bind_in_role_order_and_grid_their_padding() {
        let head = head(&Record::default());
        let mut bound = Vec::new();
        let mut view = |s: Span| {
            bound.push(s.buffer.raw);
            View {
                handle: s.buffer.raw,
                bytes: s.length,
            }
        };
        let spans = GatherByIndexBuffers {
            source: span(4),
            row_indices: span(3),
            count: span(2),
            destination: span(1),
        };
        let (rows, padded, destination) = {
            let bound = head.head_rows(5, 64, spans, &mut view);
            (
                bound.counts.rows,
                bound.counts.padded_rows,
                bound.buffers.destination.handle,
            )
        };
        assert_eq!((rows, padded, destination), (5, 64, 1));
        assert_eq!(bound, [4, 3, 2, 1]);
    }
}
