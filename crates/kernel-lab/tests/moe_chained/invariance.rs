//! The chained MoE invariance adapter (`OpKind::ChainedMoe`): the served
//! dense chained entries (`kernel_lab::declared::chained_moe`: the
//! catalog's dense projections in every schedule the lab builds and
//! `chained::formats`' catalog) launched as this binary launches them, at
//! the served TP4 shapes.
//!
//! A unit is one token. Each token's inputs follow from its seed alone (its
//! hidden row), so a token compares with itself wherever it sits in a
//! launch. Weight banks are drawn from fixed seeds, so MLPs of every
//! capacity hold the same weights. A dense projection runs in its dense MLP
//! (a decoded-tile gate/up with the column down); a unit is the token's
//! BF16 intermediate row then its output row; its one-row launch is the
//! decode column MLP at one token.
//!
//! Every compared launch writes poisoned outputs (and intermediates) and
//! is checked afterwards: every live output element written and finite.
//!
//! The routed experts' entries, split over the ranks
//! (`kernel_lab::declared::chained_moe_split`), run through
//! `split_invariance`.

use super::*;
use crate::invariance::{Adapter, Compared, Element, Units, written};
use crate::split_invariance as split_cases;
use kernels_sm120::invariance::{Declared, OpKind};
use std::collections::HashMap;

/// The served dense projections the lab builds.
fn served_projections() -> Vec<Projection> {
    chained::catalog()
        .into_iter()
        .chain(chained::block_warp_catalog())
        .chain(formats::catalog())
        .chain(decoded_shapes::projections().to_vec())
        .collect()
}

/// The dense projection `entry` declares.
fn role(entry: &Declared) -> LabResult<Projection> {
    let symbol = entry.symbol.as_str();
    served_projections()
        .into_iter()
        .find(|p| p.symbol == symbol && p.query.gemm.grouped.is_none())
        .ok_or_else(|| format!("{symbol}: not a served dense chained entry"))
}

/// A token's hidden row.
fn token_x(seed: u64) -> Vec<u16> {
    let mut rng = Rng(seed ^ 0x0a11);
    (0..HIDDEN).map(|_| bf16_value(&mut rng, -4..=0)).collect()
}

fn u16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

/// Capacity-independent weight banks.
#[allow(clippy::too_many_arguments)]
fn bank(
    session: &Session,
    weights: Weights,
    experts: usize,
    rows: usize,
    k: usize,
    halves: usize,
    active: &[usize],
    seed: u64,
) -> LabResult<Bank> {
    Bank::build(
        session,
        weights,
        experts,
        rows,
        k,
        halves,
        active,
        &mut Rng(seed),
        false,
    )
}

/// The cases' MLPs, built once per run, and the split's cases.
#[derive(Default)]
pub struct Chained {
    split: split_cases::Split,
    dense: HashMap<String, Dense>,
}

/// The output rows of `bytes` per token of tokens `0..tokens`.
fn rows_of(bytes: &[u8], tokens: usize, row: usize) -> Units {
    (0..tokens)
        .map(|t| bytes[t * row..(t + 1) * row].to_vec())
        .collect()
}

/// Concatenates each token's units of several outputs.
fn zip_units(parts: Vec<Units>) -> Units {
    let tokens = parts.first().map_or(0, Vec::len);
    (0..tokens)
        .map(|t| parts.iter().flat_map(|units| units[t].clone()).collect())
        .collect()
}

impl Chained {
    /// The dense MLP of `projection`'s format and intermediate, its
    /// schedule's gate/up and down (a decoded-tile gate/up with the column
    /// down).
    fn dense(&mut self, session: &Session, projection: &Projection) -> LabResult<&Dense> {
        let key = projection.symbol.clone();
        if !self.dense.contains_key(&key) {
            let weights = projection.weights();
            let intermediate = if projection.gate_up() {
                projection.columns()
            } else {
                projection.query.gemm.k
            };
            let prefill = projection.row_tile == 64;
            let schedule = match projection.schedule {
                Schedule::DecodedTiles => Schedule::Columns,
                schedule => schedule,
            };
            let mut dense =
                Dense::scheduled(session, weights, intermediate, prefill, 0xde05, schedule)?;
            if projection.schedule == Schedule::DecodedTiles {
                dense.fns[0] = planned_function(Entry::of(projection).load(session)?, projection)?;
                dense.gate_up = projection.clone();
            }
            let i = intermediate as usize;
            dense.gate_up_bank = bank(session, weights, 1, 2 * i, HIDDEN, 2, &[0], 0xde0)?;
            dense.down_bank = bank(session, weights, 1, HIDDEN, i, 1, &[0], 0xde1)?;
            self.dense.insert(key.clone(), dense);
        }
        Ok(&self.dense[&key])
    }

    /// The tokens of `seeds` through `dense`: each token's output row.
    fn run_dense(session: &Session, dense: &Dense, seeds: &[u64]) -> LabResult<Units> {
        let tokens = seeds.len();
        let x = seeds.iter().flat_map(|&s| token_x(s)).collect::<Vec<_>>();
        poison::write(session, &dense.x, 0, &u16_bytes(&x))?;
        poison::write(session, &dense.live, 0, &(tokens as u32).to_ne_bytes())?;
        poison::poison(session, &[&dense.inter, &dense.out])?;
        dense.launch_projections(
            &session.stream,
            [&dense.gate_up, &dense.down],
            [&dense.fns[0], &dense.fns[1]],
            tokens as u32,
            false,
        )?;
        kernel_lab::launched(&dense.gate_up.symbol);
        kernel_lab::launched(&dense.down.symbol);
        let out = poison::read(session, &dense.out)?;
        written(
            "dense output",
            &out,
            &WriteMap::of(DType::Bf16).prefix(tokens * HIDDEN)?,
            Element::Bf16,
        )?;
        // The gate/up's own BF16 intermediate row, then the down's output.
        let i = dense.intermediate;
        let inter = poison::read(session, &dense.inter)?;
        written(
            "dense intermediate",
            &inter,
            &WriteMap::of(DType::Bf16).prefix(tokens * i)?,
            Element::Bf16,
        )?;
        Ok(zip_units(vec![
            rows_of(&inter, tokens, i * 2),
            rows_of(&out, tokens, HIDDEN * 2),
        ]))
    }

    /// The tokens of `seeds` through `entry`'s launch.
    fn launch(&mut self, session: &Session, entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        if let Some(role) = split_cases::role(&entry.symbol) {
            return self.split.launch(session, &role, seeds);
        }
        let p = role(entry)?;
        let dense = self.dense(session, &p)?;
        Self::run_dense(session, dense, seeds)
    }

    /// Each token of `seeds` alone through `entry`'s one-row launch: the
    /// decode column MLP of its format and intermediate at one token.
    fn launch_alone(
        &mut self,
        session: &Session,
        entry: &Declared,
        seeds: &[u64],
    ) -> LabResult<Units> {
        if let Some(role) = split_cases::role(&entry.symbol) {
            return self.split.alone(session, &role, seeds);
        }
        let p = role(entry)?;
        let intermediate = if p.gate_up() {
            p.columns()
        } else {
            p.query.gemm.k
        };
        let decode =
            dense_projections(p.weights(), intermediate, false)[usize::from(!p.gate_up())].clone();
        let mut units = Vec::new();
        for &seed in seeds {
            let dense = self.dense(session, &decode)?;
            units.extend(Self::run_dense(session, dense, &[seed])?);
        }
        Ok(units)
    }
}

/// Tokens a chunk case's prompt spans for `entry`: the decode capacity for
/// entries of 32 tokens, more for prefill entries.
fn prompt_tokens(entry: &Declared) -> u32 {
    if let Some(role) = split_cases::role(&entry.symbol) {
        return split_cases::prompt_rows(&role);
    }
    match role(entry) {
        Ok(p) if p.tokens_capacity <= 32 => 32,
        _ => 128,
    }
}

impl Adapter for Chained {
    fn kind(&self) -> OpKind {
        OpKind::ChainedMoe
    }

    fn prompt_rows(&self, entry: &Declared) -> u32 {
        prompt_tokens(entry)
    }

    fn chunks(
        &mut self,
        sessions: &[Session],
        entry: &Declared,
        rows: u32,
        ends: &[u32],
    ) -> LabResult<Units> {
        let mut units = Vec::new();
        let mut start = 0;
        for &end in ends {
            let seeds = (u64::from(start)..u64::from(end)).collect::<Vec<_>>();
            units.extend(self.launch(&sessions[0], entry, &seeds)?);
            start = end;
        }
        debug_assert_eq!(start, rows);
        Ok(units)
    }

    fn units(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.launch(&sessions[0], entry, seeds)
    }

    fn alone(&mut self, sessions: &[Session], entry: &Declared, seeds: &[u64]) -> LabResult<Units> {
        self.launch_alone(&sessions[0], entry, seeds)
    }

    fn row_counts(&self, entry: &Declared, max: u32) -> Vec<u32> {
        if let Some(role) = split_cases::role(&entry.symbol) {
            return split_cases::row_counts(&role, max);
        }
        crate::invariance::row_counts(max)
    }

    fn batch_units(&self, entry: &Declared) -> usize {
        if let Some(role) = split_cases::role(&entry.symbol) {
            return split_cases::batch_units(&role);
        }
        4
    }

    fn ties(&mut self, sessions: &[Session], entry: &Declared) -> LabResult<Vec<Compared>> {
        match split_cases::role(&entry.symbol) {
            Some(split_cases::Role::Route(route)) => self.split.ties(&sessions[0], route),
            _ => Err(format!("{}: no tie case", entry.symbol)),
        }
    }
}

/// Every declared dense chained entry has a role whose entry is built, and
/// the tokens' inputs follow their seeds alone (no GPU).
#[test]
fn chained_declarations_have_roles() {
    let declared = kernel_lab::declared::chained_moe();
    assert!(declared.len() > 20, "{}", declared.len());
    let mut kinds = std::collections::BTreeSet::new();
    for entry in &declared {
        let p = role(entry).unwrap();
        kinds.insert(format!("dense {:?}", p.schedule));
        let e = Entry::of(&p);
        assert!(
            kernel_lab::cubin_entries(&e.cubin)
                .unwrap()
                .contains(&e.symbol.as_str()),
            "{e:?}"
        );
    }
    for kind in ["dense Columns", "dense BlockWarps", "dense DecodedTiles"] {
        assert!(kinds.contains(kind), "{kind}: {kinds:?}");
    }
    assert_eq!(token_x(5), token_x(5));
    assert_ne!(token_x(5), token_x(6));
}
