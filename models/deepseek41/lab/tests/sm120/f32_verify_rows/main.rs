//! Flash TP4's exact F32 projections at verify row counts: every form a
//! step can take (the ordered split route, the separate-span cached
//! stages, the drafter's bounded reuse and the decode reuse), the whole-K
//! row tiles (`kernels_sm120::f32_projection::rows`), the bundle's
//! packaged row tiles and the weight-read floor. The GPU tests compare
//! every form bit for bit with the split route and time them under the
//! pair-lab rules. Each needs the lab cubin that `cpu_build_verify_rows`
//! stages in `CHOPINFER_F32_ROWS_CUBINS` (outside the repository), and runs
//! only when `CHOPINFER_KERNEL_GPU` names an idle GPU.
mod catalog;
mod fixture;
mod timing;

use catalog::{CUBIN, FAMILY, PROBE_SYMBOLS};
use fixture::{Form, Modules, Step, Weights};
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::Cubins;
use kernel_harness::kit::cases::f32_projection::Witness;
use std::path::PathBuf;
use std::sync::OnceLock;

fn root() -> PathBuf {
    let root = PathBuf::from(
        std::env::var_os("CHOPINFER_F32_ROWS_CUBINS")
            .expect("set CHOPINFER_F32_ROWS_CUBINS outside the repository"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let root = root.canonicalize().unwrap();
    let tree = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    assert!(
        !root.starts_with(tree),
        "staged cubins must stay outside the repository"
    );
    root
}

#[test]
#[ignore = "requires a staged directory named by CHOPINFER_F32_ROWS_CUBINS"]
fn cpu_build_verify_rows() {
    let spec = catalog::spec();
    let toolchain = Toolchain::detect().unwrap();
    assert!(!toolchain.skipped, "CUDA compilation is required");
    let locks = LockDir::new(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels.lock.d"))
        .family(FAMILY, [&spec.name]);
    let options = BuildOptions {
        emit_cargo_directives: false,
        ..BuildOptions::from_env()
    };
    let built =
        kernel_build::build_in_families(&toolchain, &[spec], &root(), &locks, options, None)
            .unwrap();
    assert_eq!(built.len(), 1);
}

fn lab_bytes() -> &'static [u8] {
    static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
    BYTES.get_or_init(|| {
        std::fs::read(root().join(format!("{CUBIN}.cubin"))).expect("stage the lab cubin first")
    })
}

fn cubins() -> Cubins {
    static TABLE: OnceLock<Vec<(&'static str, &'static [u8])>> = OnceLock::new();
    Cubins::new(
        TABLE.get_or_init(|| vec![("probe_sm120", lab_bytes())]),
        &[("probe_sm120", PROBE_SYMBOLS)],
    )
}

const WITNESSES: [Witness; 11] = [
    Witness::General,
    Witness::WeightBits,
    Witness::LowComponent,
    Witness::SeparateChains,
    Witness::MergeOrder,
    Witness::FoldBoundary,
    Witness::Fallback,
    Witness::Underflow,
    Witness::NonfiniteInput,
    Witness::NonfiniteWeight,
    Witness::Overflow,
];
/// Every tile boundary of 8, 16, 32 and 64 rows from one row to the
/// package's verify rows, the decode and DSpark verify widths of one to
/// eight items, the drafter's head widths and the wide buckets.
const BOUNDARY_ROWS: [u32; 42] = [
    1, 2, 4, 5, 6, 8, 9, 10, 12, 15, 16, 17, 20, 24, 25, 30, 31, 32, 33, 35, 40, 45, 47, 48, 50,
    55, 60, 63, 64, 65, 66, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192,
];
const WITNESS_ROWS: [u32; 9] = [1, 6, 16, 32, 33, 60, 64, 65, 192];

/// A named operand pattern and the row counts it runs at.
type Case = (String, (Vec<f32>, Vec<f32>), Vec<u32>);

/// Every form a row count admits, the split route first.
fn every_form(shape: catalog::Shape, rows: u32) -> Vec<Form> {
    let mut forms = vec![
        Form::Split,
        Form::Served,
        Form::Spans,
        Form::Bounded,
        Form::Reuse,
    ];
    forms.extend(
        catalog::candidates(shape)
            .into_iter()
            .map(|c| Form::Rows(Box::new(c))),
    );
    forms.extend(
        catalog::packed(shape)
            .into_iter()
            .map(|(c, served)| Form::Packed(Box::new(c), served)),
    );
    forms.retain(|f| f.admits(shape, rows));
    forms
}

#[test]
#[ignore = "requires staged cubins and an idle GPU"]
fn gpu_verify_rows_bitwise() {
    let ran = kernel_harness::gpu_test_with(cubins(), 8 << 30, |s| {
        let mut compared = 0;
        for shape in timing::selected_shapes() {
            let modules = Modules::load(s, lab_bytes(), shape)?;
            let mut cases: Vec<Case> = vec![(
                "mixed".into(),
                fixture::mixed(shape),
                timing::rows_or(&BOUNDARY_ROWS),
            )];
            for witness in WITNESSES {
                cases.push((
                    format!("{witness:?}"),
                    fixture::operands(shape, witness),
                    timing::rows_or(&WITNESS_ROWS),
                ));
            }
            for (label, (a, b), rows) in cases {
                let weights = Weights::new(s, &modules, shape, &b)?;
                drop(b);
                for rows in rows {
                    let step = Step::new(s, &weights, rows, &a, every_form(shape, rows))?;
                    let n = step.compare(s, &modules, &format!("{} {label}", shape.name))?;
                    compared += n;
                    println!(
                        "F32_ROWS_BITWISE shape={} pattern={label} rows={rows} forms={} comparisons={n} PASS",
                        shape.name,
                        step.forms.len()
                    );
                }
            }
        }
        println!("F32_ROWS_BITWISE_COMPLETE comparisons={compared}");
        Ok(compared)
    })
    .unwrap();
    assert!(ran.is_some_and(|n| n > 0), "an idle GPU is required");
}

#[test]
#[ignore = "requires staged cubins and an idle GPU; set CHOPINFER_KERNEL_BENCH=1"]
fn gpu_verify_rows_timing() {
    if std::env::var("CHOPINFER_KERNEL_BENCH").as_deref() != Ok("1") {
        eprintln!("skipping the verify-row timing: set CHOPINFER_KERNEL_BENCH=1");
        return;
    }
    let ran = kernel_harness::gpu_test_with(cubins(), 8 << 30, |s| {
        let data = timing::bench(s, lab_bytes())?;
        let path = s.receipt(timing::RECEIPT_KIND, data)?;
        println!("F32_ROWS_RECEIPT {}", path.display());
        Ok(())
    })
    .unwrap();
    assert!(ran.is_some(), "an idle GPU is required");
}
