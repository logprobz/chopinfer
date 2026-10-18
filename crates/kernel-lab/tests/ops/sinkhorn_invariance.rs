//! The Sinkhorn invariance adapter (`OpKind::Sinkhorn`): the served
//! mixer's entries (four streams of 4096, `sinkhorn_hc::SERVED`, the shape
//! the served model's mixers take) through the ops SM120 prepares, over the
//! kit's cases (`kit::cases::sinkhorn`).
//!
//! - The one-CTA mix-collapse, expand and head run on a backend of the
//!   one-CTA entries; the split mix-collapse on one with the split entry.
//! - A tiled mix-collapse entry runs on the split backend with that tile
//!   alone registered, at the row counts SM120's schedule gives it (144 on
//!   for four rows a CTA, 1505 on for eight); its rows alone run through
//!   the split entry, the op's one-row launch, whose bits every form
//!   declares. Chunks below its rows run the split entry.
//! - The fused mix-collapse and norm runs its tree entry.
//! - The lagged engine's lab entries (`sinkhorn_hc::lagged`) run on a
//!   backend of those entries, over the kit's lagged cases
//!   (`kit::cases::sinkhorn_lagged`).
//!
//! The negative control (`a_tiled_mutant_fails_the_rows_case_gpu`)
//! registers the tile whose warp sums run their levels in reverse
//! (`tiled::LEVELS_REVERSED_DEFINE`): its rows must differ from the split
//! entry's alone, and the served tile's must not.

use super::prepared_invariance::{Family, seed_rows};
use kernel_core::backend::{Backend, Need};
use kernel_core::sinkhorn_hyper_connection::{
    SinkhornHyperConnectionQuery as Query, StreamHeadMode,
};
use kernel_lab::kit::cases::sinkhorn as kit;
use kernel_lab::kit::cases::sinkhorn_lagged as lagged_kit;
use kernel_lab::kit::{Modules, OpCase};
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::backend::Sm120;
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::sinkhorn_hc::{self, Op, SERVED, lagged, mix_norm_tree, split, tiled};

/// Which of the family's entries a symbol is.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Form {
    OneCta(Op),
    Split,
    Tiled(u32),
    MixNorm,
    Lagged(lagged::Op, Query),
}

/// Every declared entry of the family (`kernel_lab::declared::sinkhorn`),
/// with its form.
fn entries() -> Vec<(Declared, Form)> {
    let split = split::served().symbol;
    let fused = mix_norm_tree::served().symbol;
    kernel_lab::declared::sinkhorn()
        .into_iter()
        .map(|declared| {
            let one_cta = sinkhorn_hc::catalog()
                .into_iter()
                .find(|entry| entry.symbol == declared.symbol)
                .map(|entry| Form::OneCta(entry.op));
            let tile = tiled::SERVED_TILES
                .into_iter()
                .find(|&tile| tiled::served(tile).symbol == declared.symbol)
                .map(Form::Tiled);
            let lagged = lagged::catalog()
                .into_iter()
                .find(|entry| entry.symbol == declared.symbol)
                .map(|entry| Form::Lagged(entry.op, entry.query));
            let form = one_cta
                .or(lagged)
                .or(tile)
                .or((declared.symbol == split).then_some(Form::Split))
                .or((declared.symbol == fused).then_some(Form::MixNorm))
                .expect("a served Sinkhorn entry");
            (declared, form)
        })
        .collect()
}

fn form(entry: &Declared) -> LabResult<Form> {
    entries()
        .into_iter()
        .find(|(declared, _)| declared.symbol == entry.symbol)
        .map(|(_, form)| form)
        .ok_or_else(|| format!("{}: not a served Sinkhorn entry", entry.symbol))
}

fn collapse() -> Query {
    Query::MixCollapse {
        mix: SERVED.mix,
        capacity: SERVED.capacity,
    }
}

/// The mix-collapse case of `backend` at `capacity` rows.
fn mix_case(backend: &Sm120, capacity: u32) -> LabResult<OpCase<'static>> {
    let prepared = backend
        .sinkhorn_mix_collapse(&collapse(), Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    kit::mix_collapse_case(collapse(), capacity, prepared)
}

/// The backend a tiled entry of `tile` rows runs on: the split entry and
/// that tile, or `entry` in the tile's place.
fn tiled_backend(entry: tiled::Entry) -> Sm120 {
    kit::sm120(true).with_sinkhorn_tiled([entry])
}

pub struct Sinkhorn;

impl Family for Sinkhorn {
    fn kind(&self) -> OpKind {
        OpKind::Sinkhorn
    }

    fn case(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        let need = Need::rows(capacity);
        let mix = SERVED.mix;
        let err = |e: kernel_core::backend::Refused| e.to_string();
        match form(entry)? {
            Form::OneCta(Op::MixCollapse { .. }) => mix_case(&kit::sm120(false), capacity),
            Form::OneCta(Op::Expand) => {
                let op = Query::Expand {
                    streams: mix.streams,
                    width: mix.width,
                    capacity: SERVED.capacity,
                };
                let prepared = kit::sm120(false).sinkhorn_expand(&op, need).map_err(err)?;
                kit::expand_case(op, capacity, prepared)
            }
            Form::OneCta(Op::Head) => {
                let op = Query::Head {
                    streams: mix.streams,
                    width: mix.width,
                    mode: StreamHeadMode::MeanOfStreams,
                    capacity: SERVED.capacity,
                };
                let prepared = kit::sm120(false).sinkhorn_head(&op, need).map_err(err)?;
                kit::head_case(op, capacity, prepared)
            }
            Form::Split => mix_case(&kit::sm120(true), capacity),
            Form::Tiled(tile) => mix_case(&tiled_backend(tiled::served(tile)), capacity),
            Form::MixNorm => {
                let op = Query::MixCollapseNorm {
                    mix,
                    rmsnorm_eps_bits: mix.norm_eps_bits,
                    capacity: SERVED.capacity,
                };
                let prepared = kit::sm120_mix_norm()
                    .sinkhorn_mix_collapse_norm(&op, need)
                    .map_err(err)?;
                kit::mix_collapse_norm_case(op, capacity, prepared)
            }
            Form::Lagged(op, query) => {
                let lab = lagged::LAB;
                let (streams, width) = (lab.mix.streams, lab.mix.width);
                let backend = lagged_kit::sm120();
                match op {
                    lagged::Op::MixCollapse { .. } => {
                        let op = Query::MixCollapse {
                            mix: lab.mix,
                            capacity: lab.capacity,
                        };
                        let prepared = backend
                            .sinkhorn_lagged_mix_collapse(&op, need)
                            .map_err(err)?;
                        lagged_kit::mix_collapse_case(op, capacity, prepared)
                    }
                    lagged::Op::Expand => {
                        let op = Query::ExpandF32 {
                            streams,
                            width,
                            capacity: lab.capacity,
                        };
                        let prepared = backend.sinkhorn_expand(&op, need).map_err(err)?;
                        lagged_kit::expand_case(op, capacity, prepared)
                    }
                    lagged::Op::Head => {
                        let prepared = backend.sinkhorn_weighted_head(&query, need).map_err(err)?;
                        lagged_kit::head_case(query, capacity, prepared)
                    }
                }
            }
        }
    }

    fn reference(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        match form(entry)? {
            // Every mix-collapse form declares the split (one-CTA) bits.
            Form::Tiled(_) => mix_case(&kit::sm120(true), capacity),
            _ => self.case(entry, capacity),
        }
    }

    fn source_rows(&self, entry: &Declared) -> u32 {
        match form(entry) {
            Ok(Form::Tiled(tile)) if tile > 4 => 2048,
            _ => 512,
        }
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        match form(entry) {
            Ok(Form::Tiled(4)) => vec![tiled::TILED_FROM_ROWS, tiled::TILED_FROM_ROWS + 16],
            Ok(Form::Tiled(_)) => vec![tiled::EIGHT_ROWS_FROM, tiled::EIGHT_ROWS_FROM + 31],
            _ => crate::invariance::row_counts(max),
        }
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        match form(entry) {
            Ok(Form::Tiled(4)) => tiled::TILED_FROM_ROWS as usize,
            Ok(Form::Tiled(_)) => tiled::EIGHT_ROWS_FROM as usize,
            _ => 4,
        }
    }
}

/// Every declared entry resolves to its form, the tiles to the schedule's
/// row counts, and each case prepares and launches its entry (no GPU).
#[test]
fn sinkhorn_cases_launch_their_entries() {
    assert!(entries().iter().any(|(_, form)| matches!(
        form,
        Form::Lagged(
            lagged::Op::Head,
            Query::Head {
                mode: StreamHeadMode::SigmoidWeightedF32,
                ..
            }
        )
    )));
    for (entry, form) in entries() {
        assert_ne!(
            entry.invariance,
            kernels_sm120::invariance::Invariance::NONE
        );
        let rows = Sinkhorn.row_counts(&entry, 8192)[0];
        let case = Sinkhorn.case(&entry, rows).unwrap();
        assert!(
            case.info().symbols.iter().any(|(s, _)| *s == entry.symbol),
            "{form:?}: {:?}",
            case.info().symbols
        );
        assert_eq!(case.info().guarantees, entry.invariance, "{}", entry.symbol);
        let reference = Sinkhorn.reference(&entry, 1).unwrap();
        let roles = |case: &OpCase<'_>| {
            case.row_roles(kernels_sm120::invariance::Scope::Entry)
                .unwrap()
                .outputs
        };
        assert_eq!(roles(&case), roles(&reference));
        // The reference takes rows selected from the entry's source.
        let source = Sinkhorn.case(&entry, Sinkhorn.source_rows(&entry)).unwrap();
        reference.select(&source.images(), &[3]).unwrap();
    }
}

/// The negative control's cases prepare and take their rows: the split
/// source and reference, and the served and mutant tiles at 144 rows (no
/// GPU).
#[test]
fn the_sinkhorn_control_prepares_its_cases() {
    let (suffix, _) = tiled::MUTANTS
        .into_iter()
        .find(|(_, define)| *define == tiled::LEVELS_REVERSED_DEFINE)
        .unwrap();
    let source = mix_case(&kit::sm120(true), 512).unwrap().images();
    let order = seed_rows(&[3, 10], 512).unwrap();
    mix_case(&kit::sm120(true), 1)
        .unwrap()
        .select(&source, &order[..1])
        .unwrap();
    for entry in [tiled::served(4), tiled::mutant(4, suffix)] {
        let case = mix_case(&tiled_backend(entry.clone()), tiled::TILED_FROM_ROWS).unwrap();
        assert!(
            case.info().symbols.iter().any(|(s, _)| *s == entry.symbol),
            "{:?}",
            case.info().symbols
        );
        case.select(&source, &order).unwrap();
    }
}

/// A tiled mix-collapse whose warp sums run their levels in reverse writes
/// another tree's bits: its rows at 144 differ from the split entry's
/// alone, while the served tile's equal them.
#[test]
fn a_tiled_mutant_fails_the_rows_case_gpu() {
    gpu_test(4 << 30, |session| {
        let (suffix, _) = tiled::MUTANTS
            .into_iter()
            .find(|(_, define)| *define == tiled::LEVELS_REVERSED_DEFINE)
            .ok_or("the reversed-levels mutant")?;
        let rows = tiled::TILED_FROM_ROWS;
        let source = mix_case(&kit::sm120(true), 512)?.images();
        let seeds = (0..u64::from(rows)).map(|i| i * 7 + 3).collect::<Vec<_>>();
        let order = seed_rows(&seeds, 512)?;
        let functions = kernel_lab::kit::LabModules.load(session)?;
        let reference = mix_case(&kit::sm120(true), 1)?;
        let alone = order
            .iter()
            .map(|&row| {
                let images = reference.select(&source, &[row])?;
                Ok(reference.units(session, &*functions, &images, 1)?.remove(0))
            })
            .collect::<LabResult<Vec<_>>>()?;
        for (entry, wrong) in [(tiled::served(4), false), (tiled::mutant(4, suffix), true)] {
            let case = mix_case(&tiled_backend(entry.clone()), rows)?;
            let images = case.select(&source, &order)?;
            let units = case.units(session, &*functions, &images, rows)?;
            let differing = units.iter().zip(&alone).filter(|(a, b)| a != b).count();
            eprintln!("{}: {differing} of {rows} rows differ", entry.symbol);
            if (differing > 0) != wrong {
                return Err(format!(
                    "{}: {differing} rows differ from the split entry's alone",
                    entry.symbol
                ));
            }
        }
        Ok(())
    })
    .expect("Sinkhorn negative control");
}
