//! The ordered sum of non-negative F32 terms, `s = 0; for t in terms { s =
//! s + t }` with one round-to-nearest-even add per term in column order (the
//! HF-ordered RMSNorm's sum of squares), computed by many threads with the
//! same bits (`cuda/row/ordered_sum.cuh`, whose header comment holds the
//! construction and the proof). This module holds the header's layout, a
//! CPU emulation of its algorithm step for step ([`Emulator`]), which the
//! tests compare with the serial sum bit for bit, the lab's test entries,
//! and the rows the CPU and GPU tests draw ([`rows`]).
//!
//! In short: while the running sum stays in one binade, the binade's ulp
//! grid holds it exactly, so each add is the running index plus the term
//! rounded to that grid, ties to the even result. A worker's chunk of terms
//! is then a pair of integer increments (one per parity of the running
//! index), consecutive chunks of one binade compose exactly, and one walker
//! applies each composed run after checking that the running sum is in the
//! binade the run was computed for and stays in it. A run that fails the
//! check, and every chunk that contains a binade crossing or a negative,
//! infinite or NaN term, is added serially with the served adds.

pub mod rows;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::Refusal;

/// The header, relative to the CUDA source root.
pub const HEADER: &str = "row/ordered_sum.cuh";
/// Lab-only mutant: a tie rounds away from zero instead of to even.
pub const TIES_AWAY_DEFINE: &str = "CHOP_ORDERED_SUM_TEST_TIES_AWAY";
/// Lab-only mutant: the walker applies every composed run without checking
/// the running sum's binade or the run's end.
pub const NO_FALLBACK_DEFINE: &str = "CHOP_ORDERED_SUM_TEST_NO_FALLBACK";

const WARP: usize = 32;
/// Every running index of a binade lies below 2^24 (in its ulps).
const LIMIT: u32 = 1 << 24;
/// The most terms a worker holds.
pub const MAX_CHUNK: usize = 128;

/// The algorithm's variants: the served one and the lab's mutants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    Served,
    /// A tie adds the larger neighbour, whatever the running parity.
    TiesAway,
    /// The walker trusts every composed run.
    NoFallback,
}

impl Mutant {
    /// The mutant's define and entry suffix; none for the served variant.
    pub fn define(self) -> Option<(&'static str, &'static str)> {
        match self {
            Mutant::Served => None,
            Mutant::TiesAway => Some((TIES_AWAY_DEFINE, "_ties_away")),
            Mutant::NoFallback => Some((NO_FALLBACK_DEFINE, "_no_fallback")),
        }
    }
}

/// `terms` terms summed by `workers` threads (whole warps). Worker `w`
/// holds terms `w * terms / workers .. (w + 1) * terms / workers`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub terms: usize,
    pub workers: usize,
}

impl Layout {
    pub fn check(self) -> Result<(), Refusal> {
        if self.workers == 0 || !self.workers.is_multiple_of(WARP) || self.workers > 1024 {
            return Err(Refusal(
                "the ordered sum's workers are whole warps of one CTA",
            ));
        }
        if self.terms < self.workers || self.terms.div_ceil(self.workers) > MAX_CHUNK {
            return Err(Refusal(
                "the ordered sum gives every worker one to 128 terms",
            ));
        }
        Ok(())
    }

    pub fn chunk(self, worker: usize) -> std::ops::Range<usize> {
        worker * self.terms / self.workers..(worker + 1) * self.terms / self.workers
    }

    pub fn warps(self) -> usize {
        self.workers / WARP
    }
}

/// The serial sum the method reproduces: from +0, one add per term in
/// order.
pub fn serial(terms: &[f32]) -> f32 {
    terms.iter().fold(0.0_f32, |s, &t| s + t)
}

/// The class of a running sum or a prediction's bits: its biased exponent,
/// with zero and the subnormals in class 1 (their grid is class 1's, 2^-149).
/// Classes 1 to 254 are finite; 255 and above are infinities, NaNs and
/// negative values.
pub fn class(bits: u32) -> u32 {
    (bits >> 23).max(1)
}

/// Term `a` (bits below 0x7f80_0000) on class `class`'s grid, in its
/// ulps: the round-to-nearest quotient `k`, whether `a` lies exactly
/// halfway (`k` is then the lower neighbour), and whether `a` reaches the
/// class's upper bound, so that any add of it leaves the class.
fn grid_term(a: u32, class: u32) -> (u32, u32, bool) {
    let exponent = a >> 23;
    let significand = (a & 0x7f_ffff) | if exponent != 0 { 0x80_0000 } else { 0 };
    let scale = exponent.max(1);
    if scale > class {
        return (0, 0, true);
    }
    let shift = (class - scale).min(25);
    if shift == 0 {
        return (significand, 0, false);
    }
    let rem = significand & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    (
        (significand >> shift) + u32::from(rem > half),
        u32::from(rem == half),
        false,
    )
}

/// One composed run's increment for each parity of the running index:
/// adding `earlier` then `later`.
fn compose(earlier: [u32; 2], later: [u32; 2]) -> [u32; 2] {
    [
        earlier[0] + later[(earlier[0] & 1) as usize],
        earlier[1] + later[((earlier[1] + 1) & 1) as usize],
    ]
}

/// What a run of the method did, besides its bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Outcome {
    pub bits: u32,
    /// Walker items: composed runs and serial chunks.
    pub items: usize,
    /// Composed runs the walker's check refused.
    pub refused: usize,
    /// Terms the walker added serially.
    pub serial_terms: usize,
    /// Ties among the composed runs' terms.
    pub ties: usize,
}

/// The device algorithm on the CPU, with its per-worker state reused
/// between rows.
pub struct Emulator {
    layout: Layout,
    predictions: Vec<[f32; 2]>,
    tables: Vec<[u32; 2]>,
    /// The run's class for a regular worker, 0 for one added serially.
    keys: Vec<u32>,
    starts: Vec<usize>,
    items: Vec<u32>,
    ties: Vec<usize>,
}

impl Emulator {
    pub fn new(layout: Layout) -> Result<Self, Refusal> {
        layout.check()?;
        let n = layout.workers;
        Ok(Self {
            layout,
            predictions: vec![[0.0; 2]; n],
            tables: vec![[0; 2]; n],
            keys: vec![0; n],
            starts: vec![0; n],
            items: vec![0; layout.warps()],
            ties: vec![0; n],
        })
    }

    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// The device's predictions of the running sum before and after each
    /// worker's chunk: its chunk added from zero in order, a warp
    /// inclusive scan of the chunk sums by doubling distances, and the
    /// warps' totals added from zero in warp order.
    pub fn predict(&mut self, terms: &[f32]) -> &[[f32; 2]] {
        let layout = self.layout;
        let mut inclusive = [0.0_f32; WARP];
        let mut base = 0.0_f32;
        for warp in 0..layout.warps() {
            for (lane, value) in inclusive.iter_mut().enumerate() {
                *value = terms[layout.chunk(warp * WARP + lane)]
                    .iter()
                    .fold(0.0_f32, |s, &t| s + t);
            }
            let mut distance = 1;
            while distance < WARP {
                for lane in (distance..WARP).rev() {
                    inclusive[lane] += inclusive[lane - distance];
                }
                distance *= 2;
            }
            for lane in 0..WARP {
                let exclusive = if lane == 0 { 0.0 } else { inclusive[lane - 1] };
                self.predictions[warp * WARP + lane] = [base + exclusive, base + inclusive[lane]];
            }
            base += inclusive[WARP - 1];
        }
        &self.predictions
    }

    /// The method's sum of `terms` with the device's own predictions.
    pub fn sum(&mut self, terms: &[f32], mutant: Mutant) -> Outcome {
        self.predict(terms);
        self.run(terms, mutant)
    }

    /// The method's sum of `terms` with `predictions` in place of the
    /// device's: the result must not depend on them.
    pub fn sum_predicted(
        &mut self,
        terms: &[f32],
        predictions: &[[f32; 2]],
        mutant: Mutant,
    ) -> Outcome {
        self.predictions.copy_from_slice(predictions);
        self.run(terms, mutant)
    }

    fn run(&mut self, terms: &[f32], mutant: Mutant) -> Outcome {
        let layout = self.layout;
        assert_eq!(terms.len(), layout.terms);
        // Each worker's increments on its predicted class's grid.
        for worker in 0..layout.workers {
            let [low, high] = self.predictions[worker].map(f32::to_bits);
            let key = class(low);
            let chunk = &terms[layout.chunk(worker)];
            let mut regular = key < 255 && key == class(high);
            let mut table = [0_u32; 2];
            let mut ties = 0;
            for &term in chunk {
                let a = term.to_bits();
                regular &= a < 0x7f80_0000;
                let (k, tie, crosses) = grid_term(a, key);
                regular &= !crosses;
                ties += tie as usize;
                if mutant == Mutant::TiesAway {
                    table = table.map(|q| q + k + tie);
                } else {
                    table[0] += k + (tie & (table[0] + k) & 1);
                    table[1] += k + (tie & (table[1] + k + 1) & 1);
                }
            }
            regular &= table[0] < LIMIT && table[1] < LIMIT;
            self.tables[worker] = table;
            self.keys[worker] = if regular { key } else { 0 };
            self.ties[worker] = ties;
        }
        // Each warp's runs: a segmented inclusive scan by doubling distances
        // over consecutive regular workers of one class.
        for warp in 0..layout.warps() {
            let at = warp * WARP;
            let keys = &self.keys[at..at + WARP];
            let mut heads = 0_u32;
            for lane in 0..WARP {
                if keys[lane] == 0 || lane == 0 || keys[lane - 1] != keys[lane] {
                    heads |= 1 << lane;
                }
            }
            for lane in 0..WARP {
                let below = heads & (u32::MAX >> (31 - lane));
                self.starts[at + lane] = 31 - below.leading_zeros() as usize;
            }
            let mut distance = 1;
            while distance < WARP {
                for lane in (distance..WARP).rev() {
                    if keys[lane] != 0 && lane - distance >= self.starts[at + lane] {
                        self.tables[at + lane] =
                            compose(self.tables[at + lane - distance], self.tables[at + lane]);
                    }
                }
                distance *= 2;
            }
            self.items[warp] = (0..WARP)
                .filter(|&lane| lane == WARP - 1 || heads & (1 << (lane + 1)) != 0)
                .fold(0, |items, lane| items | 1 << lane);
        }
        // The walker.
        let mut outcome = Outcome::default();
        let mut s = 0_u32;
        for warp in 0..layout.warps() {
            let mut items = self.items[warp];
            while items != 0 {
                let lane = items.trailing_zeros() as usize;
                items &= items - 1;
                outcome.items += 1;
                let worker = warp * WARP + lane;
                let key = self.keys[worker];
                let mut from = worker;
                if key != 0 {
                    from = warp * WARP + self.starts[worker];
                    let floor = (key - 1) << 23;
                    let index = s.wrapping_sub(floor);
                    let end = index.wrapping_add(self.tables[worker][(index & 1) as usize]);
                    if mutant == Mutant::NoFallback || (class(s) == key && end < LIMIT) {
                        s = floor.wrapping_add(end);
                        outcome.ties += self.ties[from..=worker].iter().sum::<usize>();
                        continue;
                    }
                    outcome.refused += 1;
                }
                let range = layout.chunk(from).start..layout.chunk(worker).end;
                outcome.serial_terms += range.len();
                s = terms[range]
                    .iter()
                    .fold(f32::from_bits(s), |s, &t| s + t)
                    .to_bits();
            }
        }
        outcome.bits = s;
        outcome
    }
}

/// The lab's test entry of `layout` (`mutant`'s variant): one CTA per row
/// of `terms` F32 values; warp 0's lane 0 adds the row serially while warps
/// 1 to `workers / 32` run the method; both sums' bits go to `serial` and
/// `exact` [rows].
pub fn test_entry(layout: Layout, mutant: Mutant) -> Result<EntrySpec, Refusal> {
    layout.check()?;
    // The serial thread's warp joins the workers in one CTA of at most 1024
    // threads; the method itself takes up to 1024 workers.
    if layout.workers + WARP > 1024 {
        return Err(Refusal(
            "the ordered sum's test entry runs at most 992 workers beside its serial warp",
        ));
    }
    Ok(EntrySpec::new(
        test_symbol(layout, mutant),
        vec![
            ptr("terms", "const float"),
            ptr("serial", "uint32_t"),
            ptr("exact", "uint32_t"),
            EntryParam::U32 {
                name: "rows".into(),
            },
        ],
        format!(
            "chop::row::ordered_sum_test<{},{}>(terms,serial,exact,rows);",
            layout.terms, layout.workers
        ),
        (layout.workers + WARP) as u32,
        1,
    ))
}

fn ptr(name: &str, ty: &str) -> EntryParam {
    EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: ty.into(),
    }
}

pub fn test_symbol(layout: Layout, mutant: Mutant) -> String {
    let suffix = mutant.define().map_or("", |(_, suffix)| suffix);
    format!(
        "chop_ordered_sum_w{}_t{}{suffix}_v1",
        layout.terms, layout.workers
    )
}

/// The layouts the lab tests: the fused mixer's norm (4096 terms over seven
/// warps) and the standalone norms' widths over seven and eight warps.
pub const LAB_LAYOUTS: [Layout; 6] = [
    Layout {
        terms: 4096,
        workers: 224,
    },
    Layout {
        terms: 4096,
        workers: 256,
    },
    Layout {
        terms: 1536,
        workers: 224,
    },
    Layout {
        terms: 1536,
        workers: 256,
    },
    Layout {
        terms: 512,
        workers: 224,
    },
    Layout {
        terms: 512,
        workers: 256,
    },
];

pub const MUTANTS: [Mutant; 2] = [Mutant::TiesAway, Mutant::NoFallback];

/// The lab cubin of `mutant`'s variant (`ordered_sum{suffix}_sm120`): the
/// test entry of every [`LAB_LAYOUTS`] layout.
pub fn lab_cubin_spec(mutant: Mutant) -> Result<CubinSpec, Refusal> {
    let suffix = mutant.define().map_or("", |(_, suffix)| suffix);
    Ok(CubinSpec {
        name: format!("ordered_sum{suffix}_sm120"),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: mutant
            .define()
            .map(|(define, _)| vec![(define.to_owned(), "1".to_owned())])
            .unwrap_or_default(),
        entries: LAB_LAYOUTS
            .iter()
            .map(|&layout| test_entry(layout, mutant))
            .collect::<Result<_, Refusal>>()?,
    })
}

/// The lab's cubins: the served test entries and each mutant's.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    [Mutant::Served]
        .into_iter()
        .chain(MUTANTS)
        .map(|mutant| lab_cubin_spec(mutant).expect("the ordered sum's lab layouts are valid"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_terms_round_to_the_class_grid() {
        // Class 150 has ulp 1: 2.5 is a tie between 2 and 3, 2.75 rounds up.
        assert_eq!(grid_term(2.5_f32.to_bits(), 150), (2, 1, false));
        assert_eq!(grid_term(2.75_f32.to_bits(), 150), (3, 0, false));
        assert_eq!(grid_term(2.25_f32.to_bits(), 150), (2, 0, false));
        // A term at the class's upper bound 2^24 crosses; just below does not.
        assert_eq!(grid_term(16_777_216.0_f32.to_bits(), 150), (0, 0, true));
        assert_eq!(
            grid_term(16_777_215.0_f32.to_bits(), 150),
            (16_777_215, 0, false)
        );
        // Subnormals and zero on class 1's grid; far below a grid, nothing.
        assert_eq!(grid_term(3, 1), (3, 0, false));
        assert_eq!(grid_term(0, 1), (0, 0, false));
        assert_eq!(grid_term(0, 200), (0, 0, false));
        assert_eq!(grid_term(1.0_f32.to_bits(), 254), (0, 0, false));
        assert_eq!(grid_term(0x7f7f_ffff, 254), (0x00ff_ffff, 0, false));
    }

    #[test]
    fn compose_follows_the_running_parity() {
        // A tie at m = 2: from an even index it adds 2, from an odd one 3.
        let tie = [2, 3];
        // One odd increment, then the tie.
        assert_eq!(compose([1, 1], tie), [4, 3]);
        assert_eq!(compose(tie, [5, 5]), [7, 8]);
    }

    #[test]
    fn layouts_and_lab_entries() {
        for layout in LAB_LAYOUTS {
            layout.check().unwrap();
            let covered: usize = (0..layout.workers).map(|w| layout.chunk(w).len()).sum();
            assert_eq!(covered, layout.terms);
        }
        assert!(
            Layout {
                terms: 100,
                workers: 128
            }
            .check()
            .is_err()
        );
        assert!(
            Layout {
                terms: 4096,
                workers: 16
            }
            .check()
            .is_err()
        );
        assert!(
            Layout {
                terms: 4096 * 8,
                workers: 32
            }
            .check()
            .is_err()
        );
        // The method takes 1024 workers; its test entry, which adds a serial
        // warp, at most 992.
        let widest = Layout {
            terms: 1024,
            workers: 1024,
        };
        widest.check().unwrap();
        assert!(test_entry(widest, Mutant::Served).is_err());
        let entry = test_entry(
            Layout {
                terms: 992,
                workers: 992,
            },
            Mutant::Served,
        )
        .unwrap();
        assert_eq!(entry.max_threads, 1024);
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "ordered_sum_sm120",
                "ordered_sum_ties_away_sm120",
                "ordered_sum_no_fallback_sm120"
            ]
        );
        assert_eq!(
            test_symbol(LAB_LAYOUTS[0], Mutant::TiesAway),
            "chop_ordered_sum_w4096_t224_ties_away_v1"
        );
    }
}
