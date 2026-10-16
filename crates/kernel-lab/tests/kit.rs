//! The conformance kit's pilot: RMSNorm and projections through SM120, over
//! one required op set, then each op family SM120 serves and the kernels the
//! host launches itself.
//!
//! The CPU tests prepare every op and check each case's declaration. The GPU tests run every check (`CHOPINFER_KERNEL_GPU`) and
//! skip without a GPU.

use kernel_core::backend::Backend;
use kernel_lab::kit::{self, LabModules, Modules, OpResult, Outcome, Report, cases};
use kernel_lab::{LabResult, Session, gpu_test};
use serde_json::json;
use std::collections::BTreeMap;

/// The SM120 catalog projections the pilot runs: the smallest BF16 ones.
const SM120_PROJECTIONS: usize = 3;

fn run(
    backend: &dyn Backend,
    modules: &dyn Modules,
    session: Option<&Session>,
) -> LabResult<Report> {
    let catalogs = kernel_lab::catalogs();
    let offers = cases::catalog_projections(catalogs, SM120_PROJECTIONS)?;
    kit::conform(backend, modules, session, &cases::pilot(catalogs, &offers)?)
}

/// SM120 as the lab builds it: its catalogs, its latent attention entries
/// and the smallest catalog projections.
fn sm120() -> LabResult<kernels_sm120::backend::Sm120> {
    let catalogs = kernel_lab::catalogs();
    cases::sm120(
        catalogs,
        &cases::latent::sm120_entries(kernel_lab::latent_shapes::catalog()),
        &cases::catalog_projections(catalogs, SM120_PROJECTIONS)?,
    )
}

/// The op names SM120 must prepare: every catalog entry the pilot lists,
/// which leaves out its narrow RMSNorm rows and row-major projections.
fn must_prepare(report: &Report) -> Vec<&str> {
    report
        .ops
        .iter()
        .map(|op| op.name.as_str())
        .filter(|name| !name.contains("row-major") && !name.contains("width 512"))
        .collect()
}

fn check_report(report: &Report, gpu: bool) {
    let failures = report.failures();
    assert!(failures.is_empty(), "{}: {failures:#?}", report.backend);
    let prepared = report.prepared();
    for name in must_prepare(report) {
        assert!(
            prepared.contains(&name),
            "{} refused {name}: {:?}",
            report.backend,
            report.ops.iter().find(|op| op.name == name)
        );
    }
    assert!(!prepared.is_empty());
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            continue;
        };
        let outcome = |name: &str| {
            &checks
                .iter()
                .find(|check| check.name == name)
                .expect("every op has the check")
                .outcome
        };
        assert!(matches!(outcome("declaration"), Outcome::Passed(_)));
        if gpu {
            assert!(
                matches!(outcome("conformance"), Outcome::Passed(_)),
                "{}: {:?}",
                op.name,
                outcome("conformance")
            );
        }
    }
}

#[test]
fn sm120_declares_every_case_it_prepares() {
    let sm120 = sm120().expect("SM120 selection");
    let report = run(&sm120, &LabModules, None).expect("SM120 report");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    check_report(&report, false);
}

fn receipt(session: &Session, report: &Report) -> LabResult<()> {
    session.receipt(
        "conformance-kit",
        BTreeMap::from([
            ("report".into(), report.json()),
            ("not_exercised".into(), json!(report.not_exercised())),
        ]),
    )?;
    Ok(())
}

#[test]
fn sm120_passes_the_pilot_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let report = run(&sm120()?, &LabModules, Some(session))?;
        receipt(session, &report)?;
        check_report(&report, true);
        Ok(())
    })
    .expect("SM120 passes the pilot");
}

/// One family's report on SM120 (its catalog selection).
fn family_report(
    sm120: &kernels_sm120::backend::Sm120,
    ops: &[kit::Required<'_>],
    session: Option<&Session>,
) -> LabResult<Report> {
    kit::conform(sm120, &LabModules, session, ops)
}

/// SM120 prepares every op of a family and passes its checks (each
/// conformance run on a GPU).
fn check_family(report: &Report, gpu: bool) {
    let failures = report.failures();
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(
        report.prepared().len(),
        report.ops.len(),
        "{:#?}",
        report.ops
    );
    check_report(report, gpu);
}

fn rope_kv() -> LabResult<(kernels_sm120::backend::Sm120, Vec<kit::Required<'static>>)> {
    let entries = cases::rope_kv::sm120_entries(kernel_lab::catalogs());
    let ops = cases::rope_kv::family(&entries)?;
    Ok((
        kernels_sm120::backend::Sm120::new().with_rope_kv(entries),
        ops,
    ))
}

#[test]
fn rope_kv_cases_declare_on_sm120() {
    let (sm120, ops) = rope_kv().expect("RoPE and KV write cases");
    let report = family_report(&sm120, &ops, None).expect("reports");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    check_family(&report, false);
    // Every form the catalogs serve: paged and ring writes in both KV
    // dtypes, a ring that does not store, the RoPE alone and ring writes.
    for form in [
        "paged p16",
        "paged p32",
        "paged p64",
        "Fp8E4M3",
        "ring 128 Bf16",
        "no store",
        "rope Rope",
        "ring write",
        "single sequence",
        "I64",
        "CosSinTable",
        "Computed",
        "GptJ",
        "capacity 512",
    ] {
        assert!(
            report.ops.iter().any(|op| op.name.contains(form)),
            "no {form} op"
        );
    }
}

#[test]
fn sm120_passes_the_rope_kv_family_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let (sm120, ops) = rope_kv()?;
        let report = family_report(&sm120, &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the RoPE and KV write family");
}

fn attention() -> (kernels_sm120::backend::Sm120, Vec<kit::Required<'static>>) {
    let variants = cases::attention::variants(kernel_lab::catalogs());
    let ops = cases::attention::family(&variants);
    let entries = cases::attention::sm120_entries(variants);
    (
        kernels_sm120::backend::Sm120::new().with_attention(entries),
        ops,
    )
}

#[test]
fn attention_cases_declare_on_sm120() {
    let (sm120, ops) = attention();
    let report = family_report(&sm120, &ops, None).expect("reports");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    check_family(&report, false);
    for form in [
        "Decode Paged",
        "Decode Ring",
        "Verify Paged",
        "Verify Ring",
        "Prefill Paged",
        "Prefill Ring",
        "Fp8E4M3",
        "fold neox",
        "CosSinTable",
        "exact rows",
        "bidirectional",
        "sink",
    ] {
        assert!(
            report.ops.iter().any(|op| op.name.contains(form)),
            "no {form} op"
        );
    }
    // Multi-row items keep their batch check (here without a GPU); only
    // the rows and chunk checks name the item gap.
    let mut item_batches = 0;
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            continue;
        };
        for check in checks {
            let Outcome::NotExercised(reason) = &check.outcome else {
                continue;
            };
            if reason == cases::attention::ITEM_ROWS {
                assert!(
                    ["property rows", "property chunk"].contains(&check.name.as_str()),
                    "{} {}",
                    op.name,
                    check.name
                );
            }
            if check.name == "property batch" && !op.name.contains("Decode") {
                assert_eq!(reason, "no GPU session", "{}", op.name);
                item_batches += 1;
            }
        }
    }
    assert!(item_batches >= 30, "{item_batches} multi-row batch checks");
}

#[test]
fn sm120_passes_the_attention_family_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let (sm120, ops) = attention();
        let report = family_report(&sm120, &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the attention family");
}

fn latent() -> (kernels_sm120::backend::Sm120, Vec<kit::Required<'static>>) {
    let entries = cases::latent::sm120_entries(kernel_lab::latent_shapes::catalog());
    let ops = cases::latent::family(&entries);
    let sm120 = entries
        .into_iter()
        .fold(kernels_sm120::backend::Sm120::new(), |sm120, entry| {
            sm120.with_latent(entry)
        });
    (sm120, ops)
}

#[test]
fn latent_cases_declare_on_sm120() {
    let (sm120, ops) = latent();
    let report = family_report(&sm120, &ops, None).expect("reports");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    check_family(&report, false);
    for form in [
        "kv write Bf16",
        "kv write Fp8E4M3Block128",
        "Prefill Normalized",
        "Decode Partial",
        "Verify Normalized",
        "head gemm",
        "F32",
    ] {
        assert!(
            report.ops.iter().any(|op| op.name.contains(form)),
            "no {form} op"
        );
    }
}

#[test]
fn pooled_indexer_cases_declare_on_sm120() {
    let entries = kernel_lab::families::pooled_indexer::shapes::lab_entries();
    let ops = cases::pooled_indexer::required(&entries).expect("pooled indexer cases");
    let report =
        family_report(&cases::pooled_indexer::sm120(&entries), &ops, None).expect("reports");
    check_family(&report, false);
    for form in [
        "key write prefill rank 0 of 1",
        "key write prefill rank 1 of 2",
        "key write verify rank 0 of 1 rows 192",
        "top-k merge of 2 ranks rows 16",
        "top-k merge of 4 ranks rows 16",
        "global rows 16 radix merge",
        "global rows 376 one-split",
        "rank 1 of 2 rows 16 tree merge",
        "rank 3 of 4 rows 16 tree merge",
        "rank 1 of 2 rows 376 one-split",
    ] {
        assert!(
            report.ops.iter().any(|op| op.name.contains(form)),
            "no {form} op"
        );
    }
}

#[test]
fn sm120_passes_the_pooled_indexer_forms_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let entries = kernel_lab::families::pooled_indexer::shapes::lab_entries();
        let ops = cases::pooled_indexer::required(&entries)?;
        let report = family_report(&cases::pooled_indexer::sm120(&entries), &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the pooled indexer forms");
}

/// SM120 with the latent forms registered, and the attention ops it runs
/// through them.
fn latent_forms() -> (kernels_sm120::backend::Sm120, Vec<kit::Required<'static>>) {
    let entries = cases::latent::sm120_entries(kernel_lab::latent_shapes::catalog());
    (
        cases::latent::sm120_with_forms(&entries),
        cases::latent::form_family(&entries),
    )
}

#[test]
fn latent_form_cases_declare_on_sm120() {
    let (sm120, ops) = latent_forms();
    let report = family_report(&sm120, &ops, None).expect("reports");
    check_family(&report, false);
    // FP8 prefill in both output modes through the slice-widened form;
    // decode and verify over both caches in both modes through the column
    // merge.
    let count = |form: &str| {
        report
            .ops
            .iter()
            .filter(|op| op.name.ends_with(form))
            .count()
    };
    assert_eq!((count(" slices"), count(" column merge")), (2, 8));
}

#[test]
fn sm120_passes_the_latent_forms_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let (sm120, ops) = latent_forms();
        let report = family_report(&sm120, &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the latent forms");
}

#[test]
fn sm120_passes_the_latent_family_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let (sm120, ops) = latent();
        let report = family_report(&sm120, &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the latent attention family");
}

fn recurrent() -> (kernels_sm120::backend::Sm120, Vec<kit::Required<'static>>) {
    let entries = cases::recurrent::sm120_entries(kernel_lab::catalogs());
    let ops = cases::recurrent::family(&entries);
    (
        kernels_sm120::backend::Sm120::new().with_linear(entries),
        ops,
    )
}

#[test]
fn recurrent_cases_declare_on_sm120() {
    let (sm120, ops) = recurrent();
    let report = family_report(&sm120, &ops, None).expect("reports");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", report.json());
    }
    check_family(&report, false);
    for form in [
        "causal conv",
        "gated delta rule",
        "gated rmsnorm",
        "Prefill",
        "Decode",
        "Verify",
    ] {
        assert!(
            report.ops.iter().any(|op| op.name.contains(form)),
            "no {form} op"
        );
    }
}

#[test]
fn sm120_passes_the_recurrent_family_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let (sm120, ops) = recurrent();
        let report = family_report(&sm120, &ops, Some(session))?;
        receipt(session, &report)?;
        check_family(&report, true);
        Ok(())
    })
    .expect("SM120 passes the recurrent family");
}

/// The kernels the host launches itself, through the kit's host adapter
/// and the cubins the host ships.
fn host_report(ops: &[kit::Required<'_>], session: Option<&Session>) -> LabResult<Report> {
    kit::conform(&kit::host::Host, &kit::host::HostModules, session, ops)
}

/// The host prepares every op and passes its checks (each conformance run
/// on a GPU).
fn check_host(report: &Report, gpu: bool) {
    assert_eq!(
        report.prepared().len(),
        report.ops.len(),
        "{:#?}",
        report.ops
    );
    check_report(report, gpu);
}

#[test]
fn host_kernel_cases_declare() {
    let sampling = host_report(&cases::sampling::family(), None).expect("sampling report");
    if std::env::var_os("KIT_DUMP").is_some() {
        eprintln!("{}", sampling.json());
    }
    check_host(&sampling, false);
    for form in [
        "simple F32",
        "simple Bf16",
        "top F32",
        "greedy F32",
        "greedy top Bf16",
        "filtered F32",
        "filtered Bf16",
        "simple Bf16 live 2600 of 4096 rows 4 of a 16-row dispatch",
        "greedy F32 live 2600 of 4096 rows 4 of a 16-row dispatch",
    ] {
        assert!(
            sampling
                .ops
                .iter()
                .any(|op| op.name.starts_with(&format!("sampling {form}"))),
            "no {form} op"
        );
    }
}

/// The served sampling paths but the filtered one, whose six stages run
/// in their own test.
#[test]
fn host_passes_the_simple_sampling_paths_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let ops = cases::sampling::family()
            .into_iter()
            .filter(|op| !op.name.contains("filtered"))
            .collect::<Vec<_>>();
        let report = host_report(&ops, Some(session))?;
        receipt(session, &report)?;
        check_host(&report, true);
        Ok(())
    })
    .expect("the host passes the simple sampling paths");
}

#[test]
fn host_passes_the_filtered_sampling_path_on_a_gpu() {
    gpu_test(4 << 30, |session| {
        let ops = cases::sampling::family()
            .into_iter()
            .filter(|op| op.name.contains("filtered"))
            .collect::<Vec<_>>();
        let report = host_report(&ops, Some(session))?;
        receipt(session, &report)?;
        check_host(&report, true);
        Ok(())
    })
    .expect("the host passes the filtered sampling path");
}
