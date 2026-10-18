//! Declared exactness properties of catalog entries.
//!
//! Every catalog entry carries an [`Invariance`] next to its numeric fields.
//! It states which bitwise equalities the entry guarantees, so tests and
//! program checks follow from the declaration instead of from the entry's
//! model, profile or name: kernel-lab generates one test per declared
//! property and op kind, and the bundles check their launch journals
//! against the declarations of the entries they launch. A declaration is
//! data only; it never reaches a cubin or a lock.

use crate::{Catalogs, Profile, attention, comm, gemm, linear, moe, row};
// Transitional: the property vocabulary lives in `kernel_core::properties`;
// import it from there, never from this path. This re-export goes (plan
// step 31) once no model, kernel-lab test or example imports these names
// from `kernels_sm120::invariance`, switching this crate's own
// `crate::invariance` users to `kernel_core::properties` in that change.
pub use kernel_core::properties::{
    Anchors, Batch, Chunk, ExactVerify, Invariance, Property, Rows, Scope, Ties,
};
use std::collections::BTreeMap;

/// The op kinds whose kernel-lab adapters the generated tests launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum OpKind {
    /// Dense GEMMs (fragment, tiled, skinny, cluster).
    Gemm,
    /// Grouped expert GEMMs and their routing (router, permutation,
    /// combine).
    GroupedMoe,
    /// Dense attention over paged KV or a window ring.
    Attention,
    /// Attention over selected keys and its indexer (query, key write,
    /// block scores, top-k).
    SparseAttention,
    /// Recurrent linear layers: short convolution and gated delta rule.
    Linear,
    /// Row-local kernels: norms, embeddings, RoPE, ring writes, mixes.
    Row,
    /// Cross-rank collectives.
    Comm,
    /// Media encoder kernels.
    Multimodal,
    /// Kernels a model bundle owns (state copies, stream and PLE rows,
    /// speculation rows), declared by the bundle and tested by its own
    /// kernel-lab adapter.
    Bundle,
    /// Context-parallel entries of one group rank (position localization,
    /// owner-routed writes, per-rank selections and their merges, selected
    /// partials and the partial merge), tested by running the group's
    /// ranks on one device.
    ContextParallel,
    /// Latent attention entries (`latent`): the KV write, attention over
    /// key lists with its split merge, the head-batched GEMMs, and the
    /// forms run in an entry's place.
    LatentAttention,
    /// Pooled indexer entries (`pooled_indexer`): the key write, the
    /// selections with their merges and forms, the per-rank merges and the
    /// compaction.
    PooledIndexer,
    /// Sinkhorn hyper-connection entries (`sinkhorn_hc`): the mix-collapse
    /// and its split, tiled and fused forms, the expand and the head.
    Sinkhorn,
    /// Per-channel-decay entries (`linear::channel_decay`): the short
    /// convolution, the delta rule and the gated norm.
    ChannelDecay,
    /// Chained expert entries (`moe::chained`): the local routes, the
    /// projections under each schedule and the routed sums.
    ChainedMoe,
}

/// One catalog entry's declaration, by symbol.
#[derive(Clone, Debug)]
pub struct Declared {
    pub symbol: String,
    /// The owning profile, or `None` for entries every profile shares.
    pub profile: Option<Profile>,
    pub op: OpKind,
    pub invariance: Invariance,
}

/// Every catalog entry's declaration in `catalogs`, one per symbol: the
/// kernel base's and every model's entries and extra declarations. Entries that share a
/// symbol (comm shapes, rank-invariant entries listed by several profiles)
/// must declare the same invariance.
pub fn declarations(catalogs: &Catalogs) -> Vec<Declared> {
    merged(contributions(catalogs))
        .unwrap_or_else(|symbol| panic!("{symbol} is declared twice with different properties"))
}

/// One declaration per symbol of `all`, or the first symbol two of them
/// declare with different properties. Repeated identical declarations are
/// one entry listed twice (a comm shape several profiles share, a
/// rank-invariant entry several profiles list).
pub(crate) fn merged(all: Vec<Declared>) -> Result<Vec<Declared>, String> {
    let mut by_symbol = BTreeMap::<String, Declared>::new();
    for declared in all {
        match by_symbol.get(&declared.symbol) {
            Some(seen) if (seen.op, seen.invariance) != (declared.op, declared.invariance) => {
                return Err(declared.symbol);
            }
            Some(_) => {}
            None => {
                by_symbol.insert(declared.symbol.clone(), declared);
            }
        }
    }
    Ok(by_symbol.into_values().collect())
}

/// Every declaration as `catalogs` contributes it, one per catalog listing
/// and before repeats merge: the kernel base's and every model's entries
/// and extra declarations.
pub fn contributions(catalogs: &Catalogs) -> Vec<Declared> {
    let mut all = Vec::new();
    let mut add = |symbol: &str, profile: Option<Profile>, op, invariance| {
        all.push(Declared {
            symbol: symbol.to_owned(),
            profile,
            op,
            invariance,
        })
    };
    for item in gemm::catalog(catalogs) {
        add(
            &item.symbol,
            Some(item.profile),
            OpKind::Gemm,
            item.invariance,
        );
    }
    for entry in gemm::fused::catalog(catalogs) {
        add(
            &entry.symbol,
            Some(entry.profile),
            OpKind::Gemm,
            entry.invariance,
        );
    }
    for entry in gemm::mid::catalog(catalogs) {
        add(
            &entry.kernel.symbol(),
            Some(entry.profile),
            OpKind::Gemm,
            entry.invariance,
        );
    }
    for variant in moe::catalog(catalogs) {
        add(
            &variant.symbol,
            Some(variant.profile),
            OpKind::GroupedMoe,
            variant.invariance,
        );
    }
    for entry in catalogs
        .models()
        .iter()
        .flat_map(|model| (model.moe_routes)())
    {
        add(
            &entry.symbol,
            Some(entry.profile),
            OpKind::GroupedMoe,
            entry.invariance,
        );
    }
    for variant in attention::catalog(catalogs) {
        for (symbol, invariance) in variant.declared_symbols() {
            add(
                &symbol,
                Some(variant.profile),
                OpKind::Attention,
                invariance,
            );
        }
    }
    // The gated attention rows (norm, RoPE and KV write, output gate) are
    // row-local kernels.
    for entry in attention::gated::catalog(catalogs) {
        add(
            &entry.symbol,
            Some(entry.profile),
            OpKind::Row,
            entry.invariance,
        );
    }
    for entry in attention::sparse::catalog(catalogs) {
        for (symbol, invariance) in entry.declared_symbols() {
            add(
                &symbol,
                Some(entry.profile),
                OpKind::SparseAttention,
                invariance,
            );
        }
    }
    for entry in linear::catalog(catalogs) {
        add(
            &entry.symbol,
            Some(entry.profile),
            OpKind::Linear,
            entry.invariance,
        );
    }
    // The expert permutation orders routed rows for the grouped experts;
    // the expert adapter runs it with them.
    for entry in row::catalog(catalogs) {
        let op = match entry.query {
            kernel_core::contract::RowQuery::ExpertPermute { .. } => OpKind::GroupedMoe,
            _ => OpKind::Row,
        };
        add(&entry.symbol, Some(entry.profile), op, entry.invariance);
    }
    for entry in row::hyper_connection::catalog(catalogs) {
        add(
            &entry.symbol,
            Some(entry.profile),
            OpKind::Row,
            entry.invariance,
        );
    }
    for entry in row::copy::declared_entries()
        .into_iter()
        .chain(row::gather::declared_entries())
        .chain(row::mxfp8_decode::declared_entries())
    {
        add(&entry.symbol, entry.profile, entry.op, entry.invariance);
    }
    for entry in comm::declared_entries() {
        add(&entry.symbol, None, OpKind::Comm, entry.invariance);
    }
    for entry in comm::catalog(catalogs) {
        add(
            &entry.instantiation.symbol,
            Some(entry.profile),
            OpKind::Comm,
            entry.invariance,
        );
    }
    for model in catalogs.models() {
        for declared in (model.declared)() {
            add(
                &declared.symbol,
                declared.profile,
                declared.op,
                declared.invariance,
            );
        }
    }
    all
}

/// An entry whose own output is workspace that only `follower` reads, as
/// the next launch on the same lane: a selection's per-split lists, which
/// its split merge folds. It declares nothing itself, since that workspace
/// follows the split count; a launch of it followed by `follower` keeps
/// the claim declared for the follower, the composition's.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Composed {
    pub symbol: String,
    pub follower: String,
}

/// Every catalog's composed entries: each model's context-parallel
/// head-sum selections, followed by their split merges.
pub fn compositions(catalogs: &Catalogs) -> Vec<Composed> {
    catalogs
        .models()
        .iter()
        .flat_map(|model| (model.context_parallel)())
        .filter_map(|group| {
            group.select.merge_symbol().map(|follower| Composed {
                symbol: group.select.symbol(),
                follower,
            })
        })
        .collect()
}

/// The declaration of `symbol`, if a catalog of `catalogs` lists it.
pub fn declaration(catalogs: &Catalogs, symbol: &str) -> Option<Declared> {
    declarations(catalogs)
        .into_iter()
        .find(|declared| declared.symbol == symbol)
}

/// The exactness a served program promises, from its manifest and contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Program {
    /// Prefill chunks start at multiples of this many prompt rows, and
    /// in-step anchors sit on multiples of it inside launches that start
    /// on one (or at an anchor), so an anchor lies at least this many rows
    /// after its launch's first row.
    pub prefill_granule: u32,
    /// The verify steps with exact rows.
    pub exact_verify: ExactVerify,
    /// In-step prefix anchors a step may request.
    pub anchor_items: u32,
}

/// The phase a journal launch ran in.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Launched {
    Prefill,
    /// A launch of a verify step of `rows` rows over `items` items that
    /// processes `launch_rows` of those rows (`Program::verify_launches`).
    Verify {
        rows: u32,
        items: u32,
        launch_rows: u32,
    },
    Decode,
}

impl Program {
    /// Checks one journal launch of `symbol` against its declaration:
    /// prefill entries must keep rows over the manifest's chunk granule
    /// (and so the granule must be a multiple of every entry's), and with
    /// anchors their per-slot state must support anchors at least the
    /// granule's rows after the launch's first row; a launch of a
    /// qualifying verify step must be one-row exact at the rows it
    /// processes. Decode and other verify launches promise nothing to check.
    pub fn check(
        self,
        launched: Launched,
        symbol: &str,
        declared: Option<&Declared>,
    ) -> Result<(), String> {
        let exact = matches!(
            launched,
            Launched::Verify { rows, items, .. }
                if rows <= self.exact_verify.rows && items <= self.exact_verify.items
        );
        if !(launched == Launched::Prefill || exact) {
            return Ok(());
        }
        let Some(declared) = declared else {
            return Err(format!(
                "{symbol}: {launched:?} launch of an undeclared entry"
            ));
        };
        let invariance = declared.invariance;
        match launched {
            Launched::Prefill => {
                if !invariance.chunks_at(self.prefill_granule) {
                    return Err(format!(
                        "{symbol}: prefill chunks at multiples of {} rows, entry declares {:?}",
                        self.prefill_granule, invariance.chunk
                    ));
                }
                if self.anchor_items > 0 && !invariance.anchors.supported_at(self.prefill_granule) {
                    return Err(format!(
                        "{symbol}: writes per-slot state in prefill without anchor support \
                         ({} anchor items)",
                        self.anchor_items
                    ));
                }
            }
            Launched::Verify {
                rows, launch_rows, ..
            } => {
                if !invariance.one_row_at(launch_rows) {
                    return Err(format!(
                        "{symbol}: exact verify step of {rows} rows runs {launch_rows} rows \
                         per launch, entry declares {:?}",
                        invariance.rows
                    ));
                }
            }
            Launched::Decode => {}
        }
        Ok(())
    }

    /// The launches of a verify step of `rows` rows over `items` items,
    /// each with the rows its plan says it processes (`verify`: symbol and
    /// planned rows, as `PreparedLaunch::rows` records them), checked
    /// against the launches of a one-row decode step of one item (the AR
    /// baseline, in the same form). A launch's rows are its own record;
    /// symbol counts only check the records: every baseline launch runs one
    /// row, and every symbol in the expected baseline must process at
    /// least `rows` times its baseline rows, since each verify row passes
    /// through every operation the AR row does. Returns the launches, or
    /// the records that fail these checks. Both observations must be nonempty.
    /// Callers must explicitly replace baseline symbols when verify uses a
    /// different or fused implementation, preserving their required row counts.
    pub fn verify_launches<'a>(
        rows: u32,
        items: u32,
        verify: &[(&'a str, u32)],
        baseline: &[(&str, u32)],
    ) -> Result<Vec<(Launched, &'a str)>, Vec<String>> {
        let mut errors = Vec::new();
        if verify.is_empty() || baseline.is_empty() {
            errors.push("verify and baseline observations must be nonempty".into());
        }
        let total = |launches: &[(&str, u32)], symbol: &str| {
            launches
                .iter()
                .filter(|(s, _)| *s == symbol)
                .map(|&(_, r)| u64::from(r))
                .sum::<u64>()
        };
        for &(symbol, launch_rows) in baseline {
            if launch_rows != 1 {
                errors.push(format!(
                    "{symbol}: the one-row decode step plans {launch_rows} rows"
                ));
            }
        }
        for &(symbol, launch_rows) in verify {
            if launch_rows == 0 {
                errors.push(format!("{symbol}: a verify launch plans no rows"));
            }
        }
        let symbols = baseline
            .iter()
            .map(|&(s, _)| s)
            .collect::<std::collections::BTreeSet<_>>();
        for symbol in symbols {
            let (planned, alone) = (total(verify, symbol), total(baseline, symbol));
            if planned < u64::from(rows) * alone {
                errors.push(format!(
                    "{symbol}: the verify step of {rows} rows plans {planned} rows in its \
                     launches, fewer than {rows} times the one-row step's {alone}"
                ));
            }
        }
        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(verify
            .iter()
            .map(|&(symbol, launch_rows)| {
                (
                    Launched::Verify {
                        rows,
                        items,
                        launch_rows,
                    },
                    symbol,
                )
            })
            .collect())
    }

    /// Checks every launch of a journal: (phase, symbol) of one lane (a
    /// rank's stream) in launch order. Symbols resolve to `catalogs`'
    /// declarations; `extra` declares the bundle's own kernels. A composed
    /// entry ([`compositions`]) without a declaration of its own is checked
    /// against its follower's when the follower is the next launch, in the
    /// same phase, and as undeclared otherwise. Returns each distinct
    /// violation once.
    pub fn check_journal<'a>(
        self,
        catalogs: &Catalogs,
        launches: impl IntoIterator<Item = (Launched, &'a str)>,
        extra: &[Declared],
    ) -> Vec<String> {
        let declared = declarations(catalogs)
            .into_iter()
            .chain(extra.iter().cloned())
            .map(|declared| (declared.symbol.clone(), declared))
            .collect::<BTreeMap<_, _>>();
        let composed = compositions(catalogs)
            .into_iter()
            .map(|composed| (composed.symbol, composed.follower))
            .collect::<BTreeMap<_, _>>();
        let launches = launches.into_iter().collect::<Vec<_>>();
        self.check_lane(&declared, &composed, &launches)
    }

    /// [`Program::check_journal`] over resolved declarations and composed
    /// entries (symbol to follower).
    fn check_lane(
        self,
        declared: &BTreeMap<String, Declared>,
        composed: &BTreeMap<String, String>,
        launches: &[(Launched, &str)],
    ) -> Vec<String> {
        let mut violations = std::collections::BTreeSet::new();
        for (at, &(launched, symbol)) in launches.iter().enumerate() {
            let checked = match (declared.get(symbol), composed.get(symbol)) {
                (None, Some(follower)) => match launches.get(at + 1) {
                    Some(&(next, next_symbol)) if next == launched && next_symbol == follower => {
                        self.check(launched, symbol, declared.get(follower))
                    }
                    _ => self.check(launched, symbol, None).map_err(|_| {
                        format!(
                            "{symbol}: {launched:?} launch of a composed entry not followed \
                             by {follower}"
                        )
                    }),
                },
                (own, _) => self.check(launched, symbol, own),
            };
            if let Err(violation) = checked {
                violations.insert(violation);
            }
        }
        violations.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared(invariance: Invariance) -> Declared {
        Declared {
            symbol: "chop_test_v1".into(),
            profile: None,
            op: OpKind::Row,
            invariance,
        }
    }

    #[test]
    fn verify_requires_nonempty_observations() {
        assert!(Program::verify_launches(4, 1, &[], &[("required", 1)]).is_err());
        assert!(Program::verify_launches(4, 1, &[("required", 4)], &[]).is_err());
    }

    #[test]
    fn verify_requires_every_baseline_operation() {
        let baseline = [("first", 1), ("second", 1)];
        let errors = Program::verify_launches(4, 1, &[("first", 4)], &baseline)
            .expect_err("the second operation is missing");
        assert!(errors.iter().any(|error| error.contains("second")));
    }

    #[test]
    fn verify_accepts_explicit_replacement_and_checks_its_rows() {
        let expected = [("fused", 1), ("unchanged", 1)];
        let verify = [("fused", 4), ("unchanged", 4)];
        assert!(Program::verify_launches(4, 1, &verify, &expected).is_ok());
        assert!(Program::verify_launches(4, 1, &verify, &[("original", 1)]).is_err());
        let short = [("fused", 3), ("unchanged", 4)];
        assert!(Program::verify_launches(4, 1, &short, &expected).is_err());
    }

    #[test]
    fn the_program_check_refuses_what_a_declaration_does_not_promise() {
        let program = Program {
            prefill_granule: 64,
            exact_verify: ExactVerify { rows: 8, items: 1 },
            anchor_items: 2,
        };
        let verify = |rows, launch_rows| Launched::Verify {
            rows,
            items: 1,
            launch_rows,
        };
        let check = |launched, invariance| {
            program.check(launched, "chop_test_v1", Some(&declared(invariance)))
        };
        assert!(
            program
                .check(Launched::Prefill, "chop_test_v1", None)
                .is_err()
        );
        assert!(check(Launched::Prefill, Invariance::PREFILL).is_ok());
        let granule = |n| Invariance::PREFILL.with_chunk(Chunk::Granule(n));
        assert!(check(Launched::Prefill, granule(16)).is_ok());
        assert!(check(Launched::Prefill, granule(3)).is_err());
        assert!(check(Launched::Prefill, Invariance::NONE.with_chunk(Chunk::None)).is_err());
        assert!(
            check(
                Launched::Prefill,
                Invariance::PREFILL.with_anchors(Anchors::None)
            )
            .is_err()
        );
        // Spare entries past a span: supported when the program's anchors
        // lie at least that far after their launch's start (the granule).
        let after = |rows| Invariance::PREFILL.with_anchors(Anchors::SpareAfter { rows });
        assert!(check(Launched::Prefill, after(16)).is_ok());
        assert!(check(Launched::Prefill, after(64)).is_ok());
        assert!(check(Launched::Prefill, after(65)).is_err());
        // State empty at aligned anchors: supported when the granule's
        // anchors lie on multiples of the entry's alignment.
        let aligned = |granule| Invariance::PREFILL.with_anchors(Anchors::Aligned { granule });
        assert!(check(Launched::Prefill, aligned(4)).is_ok());
        assert!(check(Launched::Prefill, aligned(64)).is_ok());
        assert!(check(Launched::Prefill, aligned(6)).is_err());
        assert!(check(Launched::Prefill, aligned(0)).is_err());
        assert!(check(verify(4, 4), Invariance::one_row(4)).is_ok());
        assert!(check(verify(8, 8), Invariance::one_row(4)).is_err());
        // A launch that processes one row of the step needs a one-row
        // contract, not the step's rows.
        assert!(check(verify(8, 1), Invariance::one_row(1)).is_ok());
        assert!(check(verify(8, 1), Invariance::NONE).is_err());
        // Verify steps past the exact rows or items and decode steps promise
        // nothing.
        assert!(check(verify(16, 16), Invariance::NONE).is_ok());
        assert!(
            check(
                Launched::Verify {
                    rows: 4,
                    items: 2,
                    launch_rows: 4
                },
                Invariance::NONE
            )
            .is_ok()
        );
        assert!(check(Launched::Decode, Invariance::NONE).is_ok());
        let without_anchors = Program {
            anchor_items: 0,
            ..program
        };
        assert!(
            without_anchors
                .check(
                    Launched::Prefill,
                    "chop_test_v1",
                    Some(&declared(Invariance::PREFILL.with_anchors(Anchors::None)))
                )
                .is_ok()
        );
    }

    /// A verify launch is checked at the rows its plan records, whatever its
    /// symbol counts: the same counts with different extents give different
    /// verdicts, and records that contradict the one-row baseline fail.
    #[test]
    fn verify_rows_come_from_the_plan_not_symbol_counts() {
        let program = Program {
            prefill_granule: 1,
            exact_verify: ExactVerify { rows: 8, items: 1 },
            anchor_items: 0,
        };
        let one_row = [Declared {
            symbol: "chop_b_v1".into(),
            profile: None,
            op: OpKind::Row,
            invariance: Invariance::one_row(1),
        }];
        let baseline = [("chop_b_v1", 1)];
        let violations = |verify: &[(&str, u32)]| {
            program.check_journal(
                &Catalogs::base(),
                Program::verify_launches(4, 1, verify, &baseline).expect("consistent records"),
                &one_row,
            )
        };
        // Four calls of a four-row step where the AR step makes one: one
        // row each passes the one-row entry...
        let per_row = [("chop_b_v1", 1); 4];
        assert_eq!(violations(&per_row), Vec::<String>::new());
        // ...while the same four calls each given all four rows (counting
        // symbols called them one row each) fail it.
        let whole = [("chop_b_v1", 4); 4];
        assert_eq!(violations(&whole).len(), 1);
        // A mixture with the same total count fails at its multirow call.
        let mixed = [
            ("chop_b_v1", 1),
            ("chop_b_v1", 1),
            ("chop_b_v1", 1),
            ("chop_b_v1", 2),
        ];
        assert_eq!(violations(&mixed).len(), 1);
        // Records that cover fewer rows than the step, a zero record and a
        // baseline that is not one row are refused before any check.
        assert!(Program::verify_launches(4, 1, &[("chop_b_v1", 1)], &baseline).is_err());
        assert!(Program::verify_launches(4, 1, &[("chop_b_v1", 0); 5], &baseline).is_err());
        assert!(Program::verify_launches(4, 1, &[("chop_b_v1", 4)], &[("chop_b_v1", 2)]).is_err());
        // A launch the baseline never makes is checked at its own rows.
        let launches =
            Program::verify_launches(4, 1, &[("chop_b_v1", 4), ("chop_c_v1", 3)], &baseline)
                .unwrap();
        assert_eq!(
            launches[1..],
            [(
                Launched::Verify {
                    rows: 4,
                    items: 1,
                    launch_rows: 3
                },
                "chop_c_v1"
            )]
        );
    }

    /// A composed entry is checked against its follower's declaration when
    /// the follower is the next launch in the same phase, and refused in a
    /// checked phase otherwise.
    #[test]
    fn composed_entries_keep_their_followers_claim() {
        let program = Program {
            prefill_granule: 64,
            exact_verify: ExactVerify { rows: 4, items: 1 },
            anchor_items: 2,
        };
        let follower = |invariance| {
            BTreeMap::from([(
                "chop_merge_v1".to_owned(),
                Declared {
                    symbol: "chop_merge_v1".into(),
                    profile: None,
                    op: OpKind::ContextParallel,
                    invariance,
                },
            )])
        };
        let claimed = follower(Invariance {
            chunk: Chunk::Any,
            rows: Rows::OneRow { max: 4 },
            ..Invariance::NONE
        });
        let composed = BTreeMap::from([("chop_select_v1".to_owned(), "chop_merge_v1".to_owned())]);
        let verify = Launched::Verify {
            rows: 4,
            items: 1,
            launch_rows: 4,
        };
        let check = |declared: &BTreeMap<String, Declared>, launches: &[(Launched, &str)]| {
            program.check_lane(declared, &composed, launches)
        };
        for phase in [Launched::Prefill, verify] {
            let pair = [(phase, "chop_select_v1"), (phase, "chop_merge_v1")];
            assert!(check(&claimed, &pair).is_empty(), "{phase:?}");
            // Alone, before another launch, or before its follower in
            // another phase: refused.
            let other = if phase == Launched::Prefill {
                verify
            } else {
                Launched::Prefill
            };
            for launches in [
                &pair[..1],
                &[(phase, "chop_select_v1"), (phase, "chop_other_v1")][..],
                &[(phase, "chop_select_v1"), (other, "chop_merge_v1")][..],
            ] {
                let violations = check(&claimed, launches);
                assert!(
                    violations.iter().any(|v| v.contains("not followed")),
                    "{phase:?} {launches:?}: {violations:?}"
                );
            }
            // The follower's claim decides.
            let weak = follower(Invariance::NONE);
            assert_eq!(check(&weak, &pair).len(), 2, "{phase:?}");
        }
        // Decode promises nothing, followed or not.
        assert!(check(&claimed, &[(Launched::Decode, "chop_select_v1")]).is_empty());
        // An entry with its own declaration is checked on its own.
        let mut own = claimed.clone();
        own.insert(
            "chop_select_v1".into(),
            Declared {
                symbol: "chop_select_v1".into(),
                profile: None,
                op: OpKind::ContextParallel,
                invariance: Invariance::PREFILL,
            },
        );
        assert!(check(&own, &[(Launched::Prefill, "chop_select_v1")]).is_empty());
    }
}
