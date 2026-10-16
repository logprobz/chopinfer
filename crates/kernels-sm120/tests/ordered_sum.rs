//! The ordered sum's construction (`kernels_sm120::row::ordered_sum`) on the
//! CPU: its step-for-step emulation of the device algorithm against the
//! serial F32 sum, bit for bit, on adversarial rows (with the device's
//! predictions and with predictions moved, so the result is shown not to
//! depend on them) and on random rows across exponent ranges, at every
//! width and worker count the kernels use; each mutant must give other bits.
//!
//! `CHOPINFER_ORDERED_SUM_ROWS=<rows>` sets the random rows per layout
//! (default 2,000); `CHOPINFER_ORDERED_SUM_THREADS` the threads (default
//! up to 16).

use kernels_sm120::row::ordered_sum::rows::{Rng, adversarial_rows, random_row};
use kernels_sm120::row::ordered_sum::{Emulator, Layout, MUTANTS, Mutant, Outcome, serial};

/// Every layout the kernels and the lab use, and narrower worker counts
/// whose longer chunks and fewer warps stress the runs differently.
const LAYOUTS: [Layout; 9] = [
    Layout {
        terms: 4096,
        workers: 224,
    },
    Layout {
        terms: 4096,
        workers: 256,
    },
    Layout {
        terms: 4096,
        workers: 32,
    },
    Layout {
        terms: 4096,
        workers: 128,
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
    Layout {
        terms: 512,
        workers: 64,
    },
];

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn same(label: impl Fn() -> String, terms: &[f32], outcome: Outcome) {
    let expected = serial(terms).to_bits();
    assert!(
        outcome.bits == expected,
        "{}: method {:#010x}, serial {expected:#010x} ({outcome:?})",
        label(),
        outcome.bits
    );
}

/// The device's predictions moved: every class shifted, scaled, zeroed or
/// replaced by an infinity or a NaN at random workers.
fn moved_predictions(rng: &mut Rng, predictions: &[[f32; 2]], rate: u64) -> Vec<[f32; 2]> {
    predictions
        .iter()
        .map(|&pair| {
            pair.map(|p| {
                if rng.below(rate) != 0 {
                    return p;
                }
                match rng.below(7) {
                    0 => p * 2.0,
                    1 => p * 0.5,
                    2 => 0.0,
                    3 => f32::INFINITY,
                    4 => f32::NAN,
                    5 => f32::from_bits(rng.bits() as u32 & 0x7fff_ffff),
                    _ => f32::from_bits(
                        p.to_bits()
                            .wrapping_add(rng.below(64) as u32)
                            .wrapping_sub(32),
                    ),
                }
            })
        })
        .collect()
}

#[test]
fn the_method_reproduces_the_serial_sum_on_adversarial_rows() {
    for layout in LAYOUTS {
        let mut emulator = Emulator::new(layout).unwrap();
        let mut rng = Rng::new(0xad5e);
        let mut refused = 0;
        let mut ties = 0;
        for seed in 0..16 {
            for row in adversarial_rows(layout, seed) {
                let label = format!("{layout:?} {} (seed {seed})", row.name);
                let outcome = emulator.sum(&row.terms, Mutant::Served);
                same(|| label.clone(), &row.terms, outcome);
                refused += outcome.refused;
                ties += outcome.ties;
                let predictions = emulator.predict(&row.terms).to_vec();
                for rate in [1, 3, 50] {
                    let moved = moved_predictions(&mut rng, &predictions, rate);
                    let outcome = emulator.sum_predicted(&row.terms, &moved, Mutant::Served);
                    same(
                        || format!("{label}, moved predictions"),
                        &row.terms,
                        outcome,
                    );
                }
            }
        }
        // The rows reach the check's refusal and the composed ties.
        assert!(refused > 0, "{layout:?}: no composed run was refused");
        assert!(ties > 0, "{layout:?}: no composed tie");
    }
}

#[test]
fn the_mutants_change_the_sum() {
    for layout in LAYOUTS {
        let mut emulator = Emulator::new(layout).unwrap();
        for mutant in MUTANTS {
            let mut caught = Vec::new();
            for seed in 0..4 {
                for row in adversarial_rows(layout, seed) {
                    let outcome = emulator.sum(&row.terms, mutant);
                    if outcome.bits != serial(&row.terms).to_bits() {
                        caught.push(row.name);
                    }
                }
            }
            assert!(!caught.is_empty(), "{layout:?}: {mutant:?} survived");
            caught.dedup();
            eprintln!("{layout:?}: {mutant:?} caught on {caught:?}");
        }
        // Ties to even matter on the squares of ordinary BF16 rows too.
        let mut rng = Rng::new(7);
        let mut row = vec![0.0_f32; layout.terms];
        let changed = (0..64)
            .filter(|_| {
                for t in row.iter_mut() {
                    let x = f32::from_bits(
                        (((rng.below(4) + 126) as u32) << 23) | ((rng.bits() >> 57) as u32) << 16,
                    );
                    *t = x * x;
                }
                emulator.sum(&row, Mutant::TiesAway).bits != serial(&row).to_bits()
            })
            .count();
        assert!(
            changed > 32,
            "{layout:?}: ties away moved {changed} of 64 sums"
        );
    }
}

/// `rows` random rows of `layout` from `seed`, in `threads` threads: the
/// method with the device's predictions against the serial sum, and every
/// 16th row again with moved predictions. Returns the counts of rows,
/// composed runs refused, serial terms and composed ties.
fn random_rows(layout: Layout, rows: usize, seed: u64, threads: usize) -> [usize; 4] {
    let per = rows.div_ceil(threads);
    std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || {
                    let mut emulator = Emulator::new(layout).unwrap();
                    let mut rng = Rng::new(seed ^ (thread as u64).wrapping_mul(0x51ed));
                    let mut row = vec![0.0_f32; layout.terms];
                    let mut counts = [0_usize; 4];
                    for i in 0..per.min(rows.saturating_sub(thread * per)) {
                        random_row(&mut rng, &mut row);
                        let outcome = emulator.sum(&row, Mutant::Served);
                        same(|| format!("{layout:?} random row {i}"), &row, outcome);
                        counts[0] += 1;
                        counts[1] += outcome.refused;
                        counts[2] += outcome.serial_terms;
                        counts[3] += outcome.ties;
                        if i % 16 == 0 {
                            let predictions = emulator.predict(&row).to_vec();
                            let moved = moved_predictions(&mut rng, &predictions, 8);
                            let outcome = emulator.sum_predicted(&row, &moved, Mutant::Served);
                            same(
                                || format!("{layout:?} random row {i}, moved"),
                                &row,
                                outcome,
                            );
                        }
                    }
                    counts
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|h| h.join().expect("a random-row thread"))
            .fold([0; 4], |a, b| {
                [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
            })
    })
}

#[test]
fn the_method_reproduces_the_serial_sum_on_random_rows() {
    let rows = env_usize("CHOPINFER_ORDERED_SUM_ROWS", 2_000);
    let threads = env_usize(
        "CHOPINFER_ORDERED_SUM_THREADS",
        std::thread::available_parallelism().map_or(1, |n| n.get().min(16)),
    );
    for (i, layout) in LAYOUTS.into_iter().enumerate() {
        let started = std::time::Instant::now();
        let [done, refused, serial_terms, ties] =
            random_rows(layout, rows, 0x5eed_0000 + i as u64, threads);
        assert_eq!(done, rows);
        eprintln!(
            "ordered sum {}x{}: {done} random rows bitwise equal to the serial sum in {:.1} s; \
             {:.2} serial terms and {:.1} composed ties per row, {refused} composed runs refused",
            layout.terms,
            layout.workers,
            started.elapsed().as_secs_f64(),
            serial_terms as f64 / done as f64,
            ties as f64 / done as f64,
        );
    }
}
