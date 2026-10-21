//! The all-reduces: each sublayer's F32 rank partials summed in F32 in rank
//! order and rounded once to BF16 (`CommQuery::AllReduceRounded`, the
//! vendor's F32 all-reduce before its BF16 cast in `RowParallelLinear`,
//! `model.py:271-278`), and the embedding's BF16 shard rows widened into F32
//! partials (an own op) and reduced the same way, which gives back each
//! row's bits as one rank holds it.
//! Native Flash TP4 prefills default to the separate BF16 AllReduce
//! contract, rounding each completed local partial before transport.
//! Decode and verification retain F32 partials; the explicit zero switch
//! keeps F32 partials in every phase.
//!
//! Each rank's op keeps comm state regions (`OpInfo::state`) that its
//! peers write; every rank allocates the same regions, zeroed at create.

use kernel_core::backend::{Backend, Counts, Prepared, Refused, View};
use kernel_core::collective::{AllReduceBuffers, RankViews};
use kernel_core::contract::{CommQuery, DType, Refusal};
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::widen::{WidenRows, WidenRowsBuffers};
use model_deepseek41_contract::serving::PREFILL_TOKENS;
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::DeviceBufferSpanV1;

use super::{Asks, Bind, Lowered, exact};
use crate::program::Phase;
use crate::shape::Shape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Partials {
    Off,
    All,
    Prefill,
}

fn partials_selected(shape: &Shape, storage: KvStorage, setting: Option<&str>) -> Partials {
    if shape.ranks() != 4 || shape.geometry != &model_deepseek41_contract::geometry::FLASH {
        return Partials::Off;
    }
    match setting {
        Some("1") => Partials::All,
        Some("prefill") => Partials::Prefill,
        None if storage == KvStorage::Native => Partials::Prefill,
        _ => Partials::Off,
    }
}

fn partials_mode(shape: &Shape, storage: KvStorage) -> Partials {
    #[cfg(test)]
    return partials_selected(shape, storage, PARTIALS_IN_TESTS.get());
    #[cfg(not(test))]
    {
        let setting = std::env::var("CHOPINFER_DEEPSEEK41_BF16_PARTIALS");
        let value = match &setting {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => Some(""),
        };
        partials_selected(shape, storage, value)
    }
}

pub(crate) fn bf16_partials_requested(shape: &Shape, storage: KvStorage) -> bool {
    partials_mode(shape, storage) != Partials::Off
}

pub(crate) fn prefill_only_partials(shape: &Shape, storage: KvStorage) -> bool {
    partials_mode(shape, storage) == Partials::Prefill
}

#[cfg(test)]
thread_local! { static PARTIALS_IN_TESTS: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(Some("0")) }; }

#[cfg(test)]
pub(crate) fn with_bf16_partials<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    with_partial_setting(Some(if enabled { "1" } else { "0" }), run)
}

#[cfg(test)]
pub(crate) fn with_partial_setting<T>(setting: Option<&'static str>, run: impl FnOnce() -> T) -> T {
    struct Reset(Option<&'static str>);
    impl Drop for Reset {
        fn drop(&mut self) {
            PARTIALS_IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(PARTIALS_IN_TESTS.replace(setting));
    run()
}

#[cfg(test)]
#[test]
fn bf16_partials_default_only_to_qualified_prefill_and_keep_controls() {
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};
    for geometry in GEOMETRIES {
        for ranks in [Ranks::TP1, Ranks::TP2, Ranks::TP4] {
            let Ok(shape) = Shape::new(geometry, ranks) else {
                continue;
            };
            for setting in [Some("0"), Some(""), Some("invalid")] {
                assert_eq!(
                    partials_selected(&shape, KvStorage::Native, setting),
                    Partials::Off
                );
            }
            let selected = geometry == &FLASH && ranks == Ranks::TP4;
            assert_eq!(
                partials_selected(&shape, KvStorage::Native, None),
                if selected {
                    Partials::Prefill
                } else {
                    Partials::Off
                }
            );
            assert_eq!(
                partials_selected(&shape, KvStorage::Native, Some("1")),
                if selected {
                    Partials::All
                } else {
                    Partials::Off
                }
            );
            assert_eq!(
                partials_selected(&shape, KvStorage::Native, Some("prefill")),
                if selected {
                    Partials::Prefill
                } else {
                    Partials::Off
                }
            );
            assert_eq!(
                partials_selected(&shape, KvStorage::Bf16, None),
                Partials::Off
            );
            assert_eq!(
                partials_selected(&shape, KvStorage::Bf16, Some("prefill")),
                if selected {
                    Partials::Prefill
                } else {
                    Partials::Off
                }
            );
            assert!(matches!(
                with_partial_setting(None, || query(&shape, KvStorage::Native)),
                CommQuery::AllReduceRounded { .. }
            ));
            assert!(matches!(
                with_partial_setting(Some("prefill"), || query(&shape, KvStorage::Native)),
                CommQuery::AllReduceRounded { .. }
            ));
            assert!(matches!(
                with_bf16_partials(false, || query(&shape, KvStorage::Native)),
                CommQuery::AllReduceRounded { .. }
            ));
            assert_eq!(
                matches!(
                    with_bf16_partials(true, || query(&shape, KvStorage::Native)),
                    CommQuery::AllReduce {
                        dtype: DType::Bf16,
                        fused_residual: false,
                        fused_rmsnorm: None,
                        ..
                    }
                ),
                selected
            );
        }
    }
}

fn selected(shape: &Shape, setting: Option<&str>) -> bool {
    match setting {
        Some("1") => true,
        None => shape.geometry == &model_deepseek41_contract::geometry::FLASH,
        _ => false,
    }
}

pub(crate) fn requested(_shape: &Shape) -> bool {
    #[cfg(test)]
    return IN_TESTS.get();
    #[cfg(not(test))]
    {
        let setting = std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_BF16_GATHER");
        match setting {
            Ok(value) => selected(_shape, Some(&value)),
            Err(std::env::VarError::NotPresent) => selected(_shape, None),
            Err(std::env::VarError::NotUnicode(_)) => false,
        }
    }
}

#[cfg(test)]
#[test]
fn bf16_gather_defaults_to_the_qualified_geometry_and_keeps_the_control() {
    use bundle_runtime::geometry::Ranks;
    use model_deepseek41_contract::geometry::{FLASH, GEOMETRIES};
    for geometry in GEOMETRIES {
        let shape = Shape::new(geometry, Ranks::TP4).unwrap();
        assert_eq!(selected(&shape, None), geometry == &FLASH);
        assert!(selected(&shape, Some("1")));
        for value in ["0", "", "invalid"] {
            assert!(!selected(&shape, Some(value)));
        }
    }
}

#[cfg(test)]
thread_local! { static IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

/// `Skip` is the unset default: ordered DMA for the negotiated upper-skip
/// executor only, with the BF16 gather it was measured with. Full-schedule
/// and two-part executions keep the SM transport unless `1` selects
/// `Explicit`.
/// The element format a prefill DMA reduction copies between ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DmaPayload {
    F32,
    Bf16,
}

impl DmaPayload {
    pub(crate) fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::Bf16 => 2,
        }
    }
}

/// A row-local launch that may run on part of a reduced interval: which
/// arguments hold rows and how many bytes each row takes there. Every other
/// argument is the same for every row.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowLocal {
    pub(crate) kind: RowLocalKind,
    pub(crate) rows: Vec<(usize, u64)>,
    /// The argument reading the reduced rows, if this launch reads them.
    pub(crate) reduced: Option<usize>,
    /// The device cell holding the live row count.
    pub(crate) live: usize,
    /// A U64 argument that addresses rows from an explicit first row.
    pub(crate) row_offset: Option<usize>,
    /// Row arguments that may instead be null.
    pub(crate) optional: Vec<(usize, u64)>,
    /// The grid dimension counting rows, and the rows each step covers.
    pub(crate) grid: (usize, u32),
}

/// The consumers that follow a hidden reduction, in their served order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RowLocalKind {
    Expand,
    Mix,
    Norm,
}

/// Whether DMA runs the row-local consumers of each gathered owner
/// interval as it lands. Unset selects it; `0` keeps them after the call.
pub(crate) fn dma_consume_requested() -> bool {
    #[cfg(test)]
    return CONSUME_IN_TESTS.get();
    #[cfg(not(test))]
    {
        std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_DMA_CONSUME").as_deref() != Ok("0")
    }
}

#[cfg(test)]
thread_local! { static CONSUME_IN_TESTS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }

#[cfg(test)]
pub(crate) fn with_dma_consume<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            CONSUME_IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(CONSUME_IN_TESTS.replace(enabled));
    run()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefillDma {
    Off,
    Skip,
    Explicit,
}

impl PrefillDma {
    fn read(setting: Option<&str>) -> Self {
        match setting {
            Some("1") => Self::Explicit,
            None => Self::Skip,
            _ => Self::Off,
        }
    }

    /// The policy an instance keeps: `Skip` holds only for a create record
    /// whose prefill profile negotiates upper-skip chunks and whose BF16
    /// gather is prepared; otherwise it allocates nothing.
    pub(crate) fn admit(
        self,
        prefill: model_deepseek41_contract::prefill_profile::Profile,
        bf16_gather: bool,
    ) -> Self {
        match self {
            Self::Skip if !(prefill.skip && prefill.chunk > 0 && bf16_gather) => Self::Off,
            policy => policy,
        }
    }

    pub(crate) fn enabled(self) -> bool {
        self != Self::Off
    }

    /// Full-schedule and two-part executions require explicit selection.
    pub(crate) fn serves_full_schedule(self) -> bool {
        self == Self::Explicit
    }
}

pub(crate) fn dma_requested() -> PrefillDma {
    #[cfg(test)]
    return DMA_IN_TESTS.get();
    #[cfg(not(test))]
    {
        match std::env::var("CHOPINFER_DEEPSEEK41_PREFILL_DMA") {
            Ok(value) => PrefillDma::read(Some(&value)),
            Err(std::env::VarError::NotPresent) => PrefillDma::read(None),
            Err(std::env::VarError::NotUnicode(_)) => PrefillDma::Off,
        }
    }
}

#[cfg(test)]
thread_local! { static DMA_IN_TESTS: std::cell::Cell<PrefillDma> = const { std::cell::Cell::new(PrefillDma::Off) }; }

#[cfg(test)]
pub(crate) fn with_dma<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    with_dma_setting(Some(if enabled { "1" } else { "0" }), run)
}

#[cfg(test)]
pub(crate) fn with_dma_setting<T>(setting: Option<&str>, run: impl FnOnce() -> T) -> T {
    struct Reset(PrefillDma);
    impl Drop for Reset {
        fn drop(&mut self) {
            DMA_IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(DMA_IN_TESTS.replace(PrefillDma::read(setting)));
    run()
}

#[cfg(test)]
#[test]
fn dma_defaults_to_upper_skip_and_keeps_explicit_controls() {
    use model_deepseek41_contract::prefill_profile::Profile;
    let skip = Profile {
        chunk: 8192,
        skip: true,
    };
    let full = Profile::default();
    let chunked = Profile {
        chunk: 8192,
        skip: false,
    };
    let explicit = PrefillDma::read(Some("1"));
    assert_eq!(explicit, PrefillDma::Explicit);
    for (prefill, bf16) in [(skip, true), (skip, false), (full, true), (chunked, true)] {
        assert_eq!(explicit.admit(prefill, bf16), PrefillDma::Explicit);
    }
    assert!(explicit.enabled() && explicit.serves_full_schedule());
    let unset = PrefillDma::read(None);
    assert_eq!(unset, PrefillDma::Skip);
    assert_eq!(unset.admit(skip, true), PrefillDma::Skip);
    assert!(unset.enabled() && !unset.serves_full_schedule());
    for (prefill, bf16) in [(skip, false), (full, true), (chunked, true)] {
        assert_eq!(unset.admit(prefill, bf16), PrefillDma::Off);
    }
    for setting in [Some("0"), Some(""), Some("invalid")] {
        let off = PrefillDma::read(setting);
        assert_eq!(off, PrefillDma::Off);
        assert_eq!(off.admit(skip, true), PrefillDma::Off);
        assert!(!off.enabled() && !off.serves_full_schedule());
    }
}

#[cfg(test)]
pub(crate) fn with_bf16_gather<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            IN_TESTS.set(self.0);
        }
    }
    let _reset = Reset(IN_TESTS.replace(enabled));
    run()
}

struct Prefill {
    reduces: Vec<Prepared<AllReduceBuffers<View>>>,
    first_state: usize,
    admitted: bool,
}

/// The all-reduces of an instance, each prepared for a prefill's rows.
pub(crate) struct ReduceOps {
    bf16_partials: Partials,
    /// Each rank's rounded all-reduce of hidden rows, in rank order.
    reduces: Vec<Prepared<AllReduceBuffers<View>>>,
    /// The embedding's widening into F32 partials.
    widen: Prepared<WidenRowsBuffers<View>>,
    /// The bytes of each comm state region every rank keeps, in the ops'
    /// role order.
    state: Vec<u64>,
    width: u32,
    prefill: Option<Prefill>,
}

/// One hidden row of the all-reduce: every rank's F32 partial summed in F32
/// in rank order from zero and rounded once to BF16. The opt-in candidate
/// declares its additional input rounding through the BF16 query.
pub(crate) fn query(shape: &Shape, storage: KvStorage) -> CommQuery {
    if partials_mode(shape, storage) == Partials::All {
        return partial_query(shape);
    }
    rounded_query(shape)
}

fn rounded_query(shape: &Shape) -> CommQuery {
    CommQuery::AllReduceRounded {
        ranks: shape.ranks(),
        elements: u64::from(shape.dims().hidden),
    }
}

fn partial_query(shape: &Shape) -> CommQuery {
    CommQuery::AllReduce {
        ranks: shape.ranks(),
        elements: u64::from(shape.dims().hidden),
        dtype: DType::Bf16,
        fused_residual: false,
        fused_rmsnorm: None,
    }
}

/// One peer's spans of an all-reduce: its output rows and its comm state
/// regions in role order.
pub(crate) struct PeerViews {
    pub output: DeviceBufferSpanV1,
    pub state: Vec<DeviceBufferSpanV1>,
}

/// A step's spans of an all-reduce: this rank's F32 partials and BF16
/// output rows, the live row count and every rank's peer views in rank
/// order.
pub(crate) struct ReduceViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
    pub peers: Vec<PeerViews>,
}

/// A step's spans of the embedding's widening: the BF16 shard rows, the
/// F32 partials and the live row count.
pub(crate) struct WidenViews {
    pub input: DeviceBufferSpanV1,
    pub output: DeviceBufferSpanV1,
    pub live_rows: DeviceBufferSpanV1,
}

impl ReduceOps {
    /// Every rank's all-reduce of `shape` and the embedding's widening.
    pub(crate) fn prepare(
        backend: &dyn Backend,
        own: &dyn OwnOps,
        shape: &Shape,
        storage: KvStorage,
        asks: &Asks,
    ) -> Result<ReduceOps, Refused> {
        // Every phase reduces its sublayers.
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let query = query(shape, storage);
        let reduces = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("reduce.all_reduce rank {rank}"), need, |need| {
                    backend.all_reduce(&query, rank, need)
                })
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let state = reduces
            .first()
            .map(|op| op.info().state.clone())
            .unwrap_or_default();
        if reduces.iter().any(|op| op.info().state != state) {
            return Err(Refused::Rule(format!(
                "{}'s ranks keep different all-reduce state regions",
                super::NAME
            )));
        }
        let widen = asks.ask("reduce.widen", need, |need| {
            own.widen_rows(
                &WidenRows {
                    width: shape.dims().hidden,
                },
                need,
            )
        })?;
        Ok(ReduceOps {
            bf16_partials: partials_mode(shape, storage),
            reduces,
            widen,
            state,
            width: shape.dims().hidden,
            prefill: None,
        })
    }

    /// A separate operation and state range keep decode and verify on their
    /// original transport. Create must admit every rank before use.
    pub(crate) fn prepare_prefill(
        &mut self,
        backend: &dyn Backend,
        shape: &Shape,
        asks: &Asks,
    ) -> Result<(), Refused> {
        let need = exact::need(PREFILL_TOKENS, exact::every_phase(super::verify_rows()));
        let query = if self.bf16_partials == Partials::Prefill {
            partial_query(shape)
        } else {
            rounded_query(shape)
        };
        let reduces = (0..shape.ranks())
            .map(|rank| {
                asks.ask(format!("reduce.prefill_bf16 rank {rank}"), need, |need| {
                    backend.all_reduce(&query, rank, need)
                })
            })
            .collect::<Result<Vec<_>, Refused>>()?;
        let state = &reduces[0].info().state;
        if reduces.iter().any(|op| op.info().state != *state) {
            return Err(Refused::Rule(
                "prefill reduction ranks disagree on state".into(),
            ));
        }
        let first_state = self.state.len();
        self.state.extend_from_slice(state);
        self.prefill = Some(Prefill {
            reduces,
            first_state,
            admitted: false,
        });
        Ok(())
    }

    pub(crate) fn prefill_prepared(&self) -> bool {
        self.prefill.is_some()
    }

    pub(crate) fn bf16_partials(&self) -> bool {
        self.bf16_partials != Partials::Off
    }

    /// Decode and verify share the BF16 protocol only in all-phase mode.
    pub(crate) fn decode_partials(&self) -> bool {
        self.bf16_partials == Partials::All
    }

    pub(crate) fn prefill_only_partials(&self) -> bool {
        self.bf16_partials == Partials::Prefill
    }

    pub(crate) fn admit_prefill(&mut self, admitted: bool) {
        if let Some(prefill) = &mut self.prefill {
            prefill.admitted = admitted;
        }
    }

    pub(crate) fn prefill_state(&self) -> Option<std::ops::Range<usize>> {
        if self.bf16_partials == Partials::All {
            return Some(0..self.state.len());
        }
        self.prefill
            .as_ref()
            .filter(|p| p.admitted)
            .map(|p| p.first_state..self.state.len())
    }

    /// The bytes of each comm state region every rank allocates.
    pub(crate) fn state(&self) -> &[u64] {
        &self.state
    }

    /// The launches of rank `rank`'s all-reduce of `rows` rows.
    pub(crate) fn reduce(
        &self,
        rank: u32,
        phase: Phase,
        views: &ReduceViews,
    ) -> Result<Vec<Lowered>, Refusal> {
        let rows = phase.rows();
        let (ops, state) = match &self.prefill {
            Some(p)
                if p.admitted
                    && phase.prefill()
                    && (rows >= 512 || self.prefill_only_partials()) =>
            {
                (&p.reduces, p.first_state..self.state.len())
            }
            candidate => (
                &self.reduces,
                0..candidate
                    .as_ref()
                    .map_or(self.state.len(), |p| p.first_state),
            ),
        };
        let op = ops
            .get(rank as usize)
            .ok_or(Refusal("DeepSeek V4.1 has no all-reduce for this rank"))?;
        let mut bind = Bind::default();
        let buffers = AllReduceBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: Some(bind.view(views.live_rows)),
            ranks: views
                .peers
                .iter()
                .map(|peer| {
                    Ok(RankViews {
                        output: bind.view(peer.output),
                        state: peer
                            .state
                            .get(state.clone())
                            .ok_or(Refusal("prefill reduction state is missing"))?
                            .iter()
                            .map(|&span| bind.view(span))
                            .collect(),
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        bind.launch(op, &buffers, Counts::rows(rows))
    }

    /// Keep the exact collective prefix while a following op writes the
    /// BF16 round. The producer owns the F32 sum's state layout.
    pub(crate) fn defer_round(
        &self,
        rank: u32,
        rows: u32,
        views: &ReduceViews,
    ) -> Result<Option<(Vec<Lowered>, DeviceBufferSpanV1)>, Refusal> {
        let op = self
            .reduces
            .get(rank as usize)
            .ok_or(Refusal("no all-reduce for this rank"))?;
        // Decode keeps the original transport's state, ahead of any prefill
        // gather's range.
        let state = 0..self
            .prefill
            .as_ref()
            .map_or(self.state.len(), |p| p.first_state);
        let mut bind = Bind::default();
        let buffers = AllReduceBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: Some(bind.view(views.live_rows)),
            ranks: views
                .peers
                .iter()
                .map(|peer| {
                    Ok(RankViews {
                        output: bind.view(peer.output),
                        state: peer
                            .state
                            .get(state.clone())
                            .ok_or(Refusal("reduction state is missing"))?
                            .iter()
                            .map(|&span| bind.view(span))
                            .collect(),
                    })
                })
                .collect::<Result<_, Refusal>>()?,
        };
        let Some(round) = op.defer_bf16_round(&buffers, View::EMPTY, Counts::rows(rows))? else {
            return Ok(None);
        };
        if round.width != self.width
            || round.output != buffers.output
            || Some(round.live_rows) != buffers.live_rows
        {
            return Err(Refusal("deferred round differs from the reduction output"));
        }
        round
            .input
            .covers(u64::from(rows) * u64::from(self.width) * 4)?;
        let sum = bind
            .0
            .get(round.input.handle)
            .map_err(|_| Refusal("deferred round source is not bound"))?;
        Ok(Some((
            round
                .launches
                .into_iter()
                .map(|l| bind.lowered(l))
                .collect(),
            sum,
        )))
    }

    /// The launches of the embedding's widening of `rows` rows.
    pub(crate) fn widen(&self, rows: u32, views: &WidenViews) -> Result<Vec<Lowered>, Refusal> {
        let mut bind = Bind::default();
        let buffers = WidenRowsBuffers {
            input: bind.view(views.input),
            output: bind.view(views.output),
            live_rows: bind.view(views.live_rows),
        };
        bind.launch(&self.widen, &buffers, Counts::rows(rows))
    }
}
