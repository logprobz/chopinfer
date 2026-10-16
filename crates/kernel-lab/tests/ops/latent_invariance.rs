//! The latent attention invariance adapter (`OpKind::LatentAttention`): the
//! lab's latent entries at the served geometry (`latent_shapes`: 16 local
//! heads over a 512-wide latent, 64-token pages, 2048 selected keys and a
//! three-token pool tail) through the ops SM120 prepares, over the kit's
//! cases (`kit::cases::latent`).
//!
//! - An entry and its split merge run on a backend of the plain entries;
//!   a slice-widened or column-merge form on one that also registers the
//!   forms, so the op launches the form in the entry's place.
//! - Attention lists the served budget of keys per row (the geometry's
//!   2051, `attention_case_with`), half the rows over a sequence shorter
//!   than the budget (its positions repeat).
//! - A verify entry's rows alone run through the decode op of the same
//!   query (its one-row entry: the same kernel), so the rows case proves
//!   verify rows equal decode rows.
//! - A KV write's pool and block tables follow the case's rows, so all its
//!   cases, the reference among them, take the source's capacity.
//!
//! The negative control (`verify_rows_differ_from_one_split_rows_gpu`)
//! runs a verify launch's rows against the prefill op's one split alone:
//! they must differ (64 splits and a merge against one split), while the
//! decode op's alone equal them.

use super::prepared_invariance::{Family, seed_rows};
use kernel_core::backend::{Backend, Need};
use kernel_core::contract::AttentionPhase;
use kernel_core::latent_attention::LatentAttentionQuery;
use kernel_lab::kit::cases::latent as kit;
use kernel_lab::kit::{Modules, OpCase};
use kernel_lab::latent_shapes::{GEOMETRY, PROFILE};
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::backend::Sm120;
use kernels_sm120::invariance::{Declared, OpKind};
use kernels_sm120::latent::{self, Entry, Query, merge_columns, slices};

/// The served entries at the lab's shape.
fn entries() -> Vec<Entry> {
    latent::entries(PROFILE, GEOMETRY)
}

/// The entry a declared symbol belongs to, and whether it is a form SM120
/// runs in the entry's place.
fn owner(symbol: &str) -> LabResult<(Entry, bool)> {
    for entry in entries() {
        if entry.symbol() == symbol || entry.merge_symbol().as_deref() == Some(symbol) {
            return Ok((entry, false));
        }
        let one = std::slice::from_ref(&entry);
        let forms = slices::entries(one)
            .iter()
            .map(slices::Entry::symbol)
            .chain(
                merge_columns::entries(one)
                    .iter()
                    .map(merge_columns::Entry::symbol),
            )
            .collect::<Vec<_>>();
        if forms.iter().any(|form| form == symbol) {
            return Ok((entry, true));
        }
    }
    Err(format!("{symbol}: not a served latent entry"))
}

/// SM120 over the served entries, with their forms when `forms`.
fn backend(forms: bool) -> Sm120 {
    let entries = entries();
    if forms {
        kit::sm120_with_forms(&entries)
    } else {
        entries.into_iter().fold(Sm120::new(), Sm120::with_latent)
    }
}

/// The case of `entry`'s op on `backend` at `capacity` rows.
fn case_of(backend: &Sm120, entry: &Entry, capacity: u32) -> LabResult<OpCase<'static>> {
    let need = Need::rows(capacity);
    let err = |e: kernel_core::backend::Refused| e.to_string();
    match entry.query {
        Query::Latent(query @ LatentAttentionQuery::KvWrite { .. }) => {
            let prepared = backend.latent_kv_write(&query, need).map_err(err)?;
            kit::kv_write_case(query, capacity, prepared)
        }
        Query::Latent(query @ LatentAttentionQuery::Attention { max_keys, .. }) => {
            let prepared = backend.latent_attention(&query, need).map_err(err)?;
            kit::attention_case_with(query, capacity, max_keys, prepared)
        }
        Query::HeadGemm(query) => {
            let prepared = backend.head_batched_gemm(&query, need).map_err(err)?;
            kit::head_gemm_case(query, capacity, prepared)
        }
    }
}

/// The decode entry of a verify entry's query: the same attention but its
/// phase and capacity.
fn decode_of(entry: &Entry) -> LabResult<Entry> {
    let Query::Latent(LatentAttentionQuery::Attention {
        kv, output, phase, ..
    }) = entry.query
    else {
        return Err(format!("{}: not an attention entry", entry.symbol()));
    };
    if phase != AttentionPhase::Verify {
        return Err(format!("{}: not a verify entry", entry.symbol()));
    }
    entries()
        .into_iter()
        .find(|other| {
            matches!(other.query, Query::Latent(LatentAttentionQuery::Attention {
                kv: k, output: o, phase: AttentionPhase::Decode, ..
            }) if k == kv && o == output)
        })
        .ok_or_else(|| format!("{}: no decode entry", entry.symbol()))
}

fn is_write(entry: &Entry) -> bool {
    matches!(
        entry.query,
        Query::Latent(LatentAttentionQuery::KvWrite { .. })
    )
}

fn is_verify(entry: &Entry) -> bool {
    matches!(
        entry.query,
        Query::Latent(LatentAttentionQuery::Attention {
            phase: AttentionPhase::Verify,
            ..
        })
    )
}

pub struct Latent;

impl Family for Latent {
    fn kind(&self) -> OpKind {
        OpKind::LatentAttention
    }

    fn case(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        let (owner, form) = owner(&entry.symbol)?;
        // A KV write's fixed images follow its rows: every case takes the
        // source's.
        let capacity = if is_write(&owner) {
            self.source_rows(entry)
        } else {
            capacity
        };
        case_of(&backend(form), &owner, capacity)
    }

    fn reference(&self, entry: &Declared, capacity: u32) -> LabResult<OpCase<'static>> {
        let (owner, form) = owner(&entry.symbol)?;
        if is_verify(&owner) {
            return case_of(&backend(form), &decode_of(&owner)?, capacity);
        }
        self.case(entry, capacity)
    }

    /// Decode and verify entries hold 256 rows; the rest the default.
    fn source_rows(&self, entry: &Declared) -> u32 {
        match owner(&entry.symbol) {
            Ok((owner, _)) => owner.capacity().min(512),
            Err(_) => 512,
        }
    }

    fn reference_rows(&self, entry: &Declared) -> u32 {
        match owner(&entry.symbol) {
            Ok((owner, _)) if is_write(&owner) => self.source_rows(entry),
            _ => 1,
        }
    }
}

/// Every declared symbol resolves to its entry, and each case prepares,
/// launches the symbol and claims its declaration; a verify entry's
/// reference is the decode op over the same roles (no GPU).
#[test]
fn latent_cases_launch_their_entries() {
    let declared = kernel_lab::declared::latent();
    assert!(declared.len() > entries().len());
    for entry in &declared {
        let case = Latent.case(entry, 8).unwrap();
        assert!(
            case.info().symbols.iter().any(|(s, _)| *s == entry.symbol),
            "{}: {:?}",
            entry.symbol,
            case.info().symbols
        );
        assert_eq!(case.info().guarantees, entry.invariance, "{}", entry.symbol);
        let reference = Latent
            .reference(entry, Latent.reference_rows(entry))
            .unwrap();
        let (owner, _) = owner(&entry.symbol).unwrap();
        if is_verify(&owner) {
            assert!(
                reference.info().implementation.contains("decode"),
                "{}",
                reference.info().implementation
            );
        }
        let source = Latent.case(entry, Latent.source_rows(entry)).unwrap();
        reference.select(&source.images(), &[5]).unwrap();
        case.select(&source.images(), &[1, 200, 2]).unwrap();
    }
}

/// The negative control's cases: a BF16 splitting verify entry's case of
/// eight rows, its source of the entry's capacity, the rows it takes, and
/// the one-row cases its rows compare with: the decode op's (equal) and the
/// prefill op's one split (different).
type Control = (
    OpCase<'static>,
    kernel_lab::conformance::Images,
    Vec<usize>,
    Vec<(String, OpCase<'static>, bool)>,
);

fn control() -> LabResult<Control> {
    let plain = backend(false);
    let verify = entries()
        .into_iter()
        .find(|e| is_verify(e) && e.symbol().contains("bf16") && e.merge_symbol().is_some())
        .ok_or("a BF16 verify entry")?;
    let Query::Latent(query) = verify.query else {
        unreachable!()
    };
    let prefill = entries()
        .into_iter()
        .find(|e| {
            matches!((e.query, query), (
                Query::Latent(LatentAttentionQuery::Attention {
                    phase: AttentionPhase::Prefill, kv, output, ..
                }),
                LatentAttentionQuery::Attention { kv: k, output: o, .. },
            ) if kv == k && output == o)
        })
        .ok_or("the prefill entry of the verify query")?;
    // The source holds the verify entry's capacity of rows.
    let rows = verify.capacity();
    let source = case_of(&plain, &verify, rows)?.images();
    let order = seed_rows(&[11, 12, 13, 14, 15, 16, 17, 18], rows)?;
    let case = case_of(&plain, &verify, order.len() as u32)?;
    let references = [(decode_of(&verify)?, true), (prefill, false)]
        .into_iter()
        .map(|(reference, equal)| Ok((reference.symbol(), case_of(&plain, &reference, 1)?, equal)))
        .collect::<LabResult<Vec<_>>>()?;
    Ok((case, source, order, references))
}

/// The negative control's cases prepare and take their rows (no GPU).
#[test]
fn the_latent_control_prepares_its_cases() {
    let (case, source, order, references) = control().unwrap();
    case.select(&source, &order).unwrap();
    for (_, alone, _) in &references {
        alone.select(&source, &[order[0]]).unwrap();
    }
}

/// A verify launch's rows equal the decode op's alone and differ from the
/// prefill op's one split alone, which the rows case would catch.
#[test]
fn verify_rows_differ_from_one_split_rows_gpu() {
    gpu_test(4 << 30, |session| {
        let functions = kernel_lab::kit::LabModules.load(session)?;
        let (case, source, order, references) = control()?;
        let units = case.units(session, &*functions, &case.select(&source, &order)?, 8)?;
        for (symbol, alone, equal) in references {
            let mut differing = 0;
            for (unit, &row) in units.iter().zip(&order) {
                let images = alone.select(&source, &[row])?;
                if alone.units(session, &*functions, &images, 1)?[0] != *unit {
                    differing += 1;
                }
            }
            eprintln!("{symbol}: {differing} of 8 rows differ");
            if (differing == 0) != equal {
                return Err(format!(
                    "{symbol}: {differing} verify rows differ from it alone"
                ));
            }
        }
        Ok(())
    })
    .expect("latent negative control");
}
