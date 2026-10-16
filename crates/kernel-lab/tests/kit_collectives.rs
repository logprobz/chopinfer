//! The conformance kit's collectives through its multi-rank runner
//! (`kit::collective`) on SM120: the all-reduces (the BF16-partials one
//! through its entries and, on a backend without the phased entries,
//! through its composed form), the fused all-reduce at each use, the
//! gathers, the row moves and the group transfers.
//!
//! The CPU test prepares every op at two and four ranks: SM120 prepares
//! each one, every case declares and every launched symbol is in a lab
//! cubin. The GPU test runs every check at
//! the rank count `CHOPINFER_KERNEL_GPUS` names (two or four GPUs with peer
//! access) and skips without them.

use kernel_lab::kit::cases::collectives;
use kernel_lab::kit::collective::{RequiredCollective, conform};
use kernel_lab::kit::{LabModules, OpResult, Outcome, Report};
use kernel_lab::{LabResult, gpu_test_multi};
use kernels_sm120::backend::Sm120;
use serde_json::json;
use std::collections::BTreeMap;

/// Every op is prepared and declares, every launched symbol is in a lab
/// cubin, and on a GPU every op passes its conformance run and every
/// property check it runs.
fn check(report: &Report, gpu: bool) {
    let failures = report.failures();
    assert!(failures.is_empty(), "{}: {failures:#?}", report.backend);
    assert_eq!(report.prepared().len(), report.ops.len(), "{report:#?}");
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            unreachable!("every op is prepared");
        };
        let outcome = |name: &str| {
            &checks
                .iter()
                .find(|check| check.name == name)
                .unwrap_or_else(|| panic!("{}: no {name} check", op.name))
                .outcome
        };
        let Outcome::Passed(declared) = outcome("declaration") else {
            panic!("{}: {:?}", op.name, outcome("declaration"));
        };
        for symbol in declared["launches"].as_array().expect("launches") {
            let symbol = symbol.as_str().expect("a symbol");
            assert!(
                LabModules::holds(symbol),
                "{}: no lab cubin holds {symbol}",
                op.name
            );
        }
        if gpu {
            for check in checks {
                assert!(
                    !matches!(check.outcome, Outcome::Failed(_)),
                    "{} {}: {:?}",
                    op.name,
                    check.name,
                    check.outcome
                );
            }
            assert!(
                matches!(outcome("conformance"), Outcome::Passed(_)),
                "{}: {:?}",
                op.name,
                outcome("conformance")
            );
        }
    }
}

/// Each backend of the kit's collectives and its required set: SM120 with
/// every entry the lab holds, and SM120 without the phased BF16 entries.
fn backends() -> LabResult<Vec<(Sm120, Vec<RequiredCollective<'static>>)>> {
    Ok(vec![
        (
            collectives::sm120(kernel_lab::LAB_CUBINS)?,
            collectives::required(),
        ),
        (
            collectives::sm120_composed()?,
            collectives::composed_required(),
        ),
    ])
}

#[test]
fn collective_cases_declare_on_sm120() {
    for (sm120, required) in backends().expect("SM120") {
        for ranks in [2, 4] {
            let report = conform(&sm120, &LabModules, ranks, None, &required).expect("report");
            if std::env::var_os("KIT_DUMP").is_some() {
                eprintln!("{}", report.json());
            }
            check(&report, false);
        }
    }
}

/// SM120's choice of the BF16-partials all-reduce's form at each row
/// count, as a table: the low-latency entry at one row (one CTA per 2048
/// elements), the phased one-shot to four rows and the two-shot to 32 at
/// 16 sending CTAs, the two-shot with 32 workers to 1023 rows, then the
/// turns entry; without the phased entries, the composed form (its round
/// row first) after the low-latency row. At six ranks the low-latency
/// entry does not serve.
#[test]
fn the_bf16_partials_form_follows_the_table() {
    use kernel_core::backend::{Backend, Counts, Need, View};
    use kernel_core::collective::{AllReduceBuffers, RankViews};
    use kernel_core::contract::{CommQuery, DType};
    use kernels_sm120::comm::{ll_bf16, phased_bf16, phased_turns_bf16};
    let width = 2048_u64;
    let ll = |_| (ll_bf16::SYMBOL.to_owned(), 1);
    let one_shot = |_| (phased_bf16::ONE_SHOT_SYMBOL.to_owned(), 16);
    let two_shot = |grid| move |_| (phased_bf16::SYMBOL.to_owned(), grid);
    let turns = |_| (phased_turns_bf16::SYMBOL.to_owned(), 48);
    let round = kernels_sm120::row::rounding::shared_symbol(
        kernels_sm120::row::rounding::Rounding::Round,
        width as u32,
    );
    let composed = |rows: u32| {
        (
            round.clone(),
            kernels_sm120::row::smallest_bucket(rows).unwrap(),
        )
    };
    type Form<'a> = Box<dyn Fn(u32) -> (String, u32) + 'a>;
    let phased: Vec<(std::ops::RangeInclusive<u32>, Form)> = vec![
        (1..=1, Box::new(ll)),
        (2..=4, Box::new(one_shot)),
        (5..=32, Box::new(two_shot(16))),
        (33..=1023, Box::new(two_shot(48))),
        (1024..=1100, Box::new(turns)),
    ];
    let without: Vec<(std::ops::RangeInclusive<u32>, Form)> =
        vec![(1..=1, Box::new(ll)), (2..=1100, Box::new(composed))];
    let six: Vec<(std::ops::RangeInclusive<u32>, Form)> = vec![
        (1..=4, Box::new(one_shot)),
        (5..=32, Box::new(two_shot(16))),
        (33..=1023, Box::new(two_shot(48))),
        (1024..=1100, Box::new(turns)),
    ];
    let cases = [
        (
            collectives::sm120(kernel_lab::LAB_CUBINS).unwrap(),
            2,
            &phased,
        ),
        (
            collectives::sm120(kernel_lab::LAB_CUBINS).unwrap(),
            4,
            &phased,
        ),
        (collectives::sm120(kernel_lab::LAB_CUBINS).unwrap(), 6, &six),
        (collectives::sm120_composed().unwrap(), 2, &without),
        (collectives::sm120_composed().unwrap(), 4, &without),
    ];
    for (backend, ranks, table) in cases {
        let query = CommQuery::AllReduce {
            ranks,
            elements: width,
            dtype: DType::Bf16,
            fused_residual: false,
            fused_rmsnorm: None,
        };
        let op = backend.all_reduce(&query, 0, Need::rows(1100)).unwrap();
        let view = |handle| View {
            handle,
            bytes: u64::MAX,
        };
        let buffers = AllReduceBuffers {
            input: view(1),
            output: view(2),
            live_rows: Some(view(3)),
            ranks: (0..ranks)
                .map(|rank| RankViews {
                    output: view(10 + u64::from(rank)),
                    state: (0..op.info().state.len() as u64)
                        .map(|at| view(100 + at))
                        .collect(),
                })
                .collect(),
        };
        for (rows, form) in table {
            for rows in rows.clone() {
                let launches = op
                    .launches(&buffers, View::EMPTY, Counts::rows(rows))
                    .unwrap();
                let first = &launches[0];
                assert_eq!(
                    (first.symbol.clone(), first.plan.grid[0]),
                    form(rows),
                    "{ranks} ranks, {rows} rows"
                );
            }
        }
    }
}

/// The BF16-partials all-reduce alone, through the entries and through the
/// composed form, on finite and special values, with its captured replays
/// at fewer live rows: a short run of [`sm120_passes_the_collectives_on_a_gpu`]'s
/// BF16 part, without the property checks that run spends most of its time
/// on (each row of every live count launched alone).
#[test]
fn sm120_passes_the_bf16_partials_on_a_gpu() {
    gpu_test_multi(4 << 30, |sessions| -> LabResult<()> {
        let ranks = sessions.len() as u32;
        let backends = [
            (
                collectives::sm120(kernel_lab::LAB_CUBINS)?,
                collectives::partials_required(false),
                "",
            ),
            (
                collectives::sm120_composed()?,
                collectives::composed_required(),
                "composed-",
            ),
        ];
        for (backend, required, name) in backends {
            let report = conform(&backend, &LabModules, ranks, Some(sessions), &required)?;
            sessions[0].receipt(
                &format!("conformance-kit-bf16-partials-{name}{ranks}-ranks"),
                BTreeMap::from([
                    ("report".into(), report.json()),
                    ("not_exercised".into(), json!(report.not_exercised())),
                ]),
            )?;
            check(&report, true);
        }
        Ok(())
    })
    .unwrap_or_else(|error| panic!("SM120 passes the BF16-partials all-reduce: {error}"));
}

#[test]
fn sm120_passes_the_collectives_on_a_gpu() {
    gpu_test_multi(4 << 30, |sessions| -> LabResult<()> {
        let ranks = sessions.len() as u32;
        for (at, (backend, required)) in backends()?.into_iter().enumerate() {
            let report = conform(&backend, &LabModules, ranks, Some(sessions), &required)?;
            let name = match at {
                0 => format!("conformance-kit-collectives-{ranks}-ranks"),
                _ => format!("conformance-kit-collectives-composed-{ranks}-ranks"),
            };
            sessions[0].receipt(
                &name,
                BTreeMap::from([
                    ("report".into(), report.json()),
                    ("not_exercised".into(), json!(report.not_exercised())),
                ]),
            )?;
            check(&report, true);
        }
        Ok(())
    })
    .unwrap_or_else(|error| panic!("SM120 passes the collectives: {error}"));
}

/// The special-value inputs reach every special sum the contract states,
/// at two and four ranks: canonical NaNs from NaN partials of any sign and
/// payload and from opposite infinities, infinities alone, overflow of the
/// partials, of the F32 sum and of the BF16 rounding, a subnormal and a
/// zero from negative zeros.
#[test]
fn special_value_inputs_reach_every_special_sum() {
    use kernel_core::codec::f32_to_bf16;
    use kernel_core::oracle::{BF16_PARTIALS_NAN, all_reduce_bf16_partials};
    let nan = BF16_PARTIALS_NAN;
    let expected = [
        nan,
        nan,
        nan,
        0x7f80,
        0xff80,
        nan,
        nan,
        0x7f80,
        0x7f80,
        f32_to_bf16(1.0e-40),
        0x0000,
    ];
    for ranks in [2, 4] {
        let inputs = (0..ranks)
            .map(|me| collectives::special_partials(me, ranks, 32))
            .collect::<Vec<_>>();
        let sums = all_reduce_bf16_partials(&inputs).unwrap();
        for row in [0, 16] {
            assert_eq!(sums[row..row + expected.len()], expected, "{ranks} ranks");
            assert!(
                sums[row + expected.len()..row + 16]
                    .iter()
                    .all(|&bits| f32::from_bits(u32::from(bits) << 16).is_finite())
            );
        }
    }
}
