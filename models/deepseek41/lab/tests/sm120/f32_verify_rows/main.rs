//! Flash TP4's exact F32 projections at verify row counts: every form a
//! step can take (the ordered split route, the separate-span cached
//! stages, the drafter's bounded reuse and the decode reuse), the whole-K
//! row tiles (`kernels_sm120::f32_projection::rows`), the bundle's
//! packaged row tiles and the weight-read floor. The GPU tests compare
//! every form bit for bit with the split route and time them under the
//! pair-lab rules. Each needs the lab cubin that `cpu_build_verify_rows`
//! stages in `CHOPINFER_F32_ROWS_CUBINS` (outside the repository), and runs
//! only when `CHOPINFER_KERNEL_GPU` names an idle GPU. The domain-boundary
//! witness also checks the reference against the outputs its operands fix.
//! `gpu_verify_rows_real_slices` runs the real-slice matrix (`real::SLICES`)
//! of the checkpoint `CHOPINFER_DEEPSEEK_CHECKPOINT` names under the mixed
//! activation; with `CHOPINFER_F32_ROWS_SLICE` naming one of its slices of
//! the one shape `CHOPINFER_F32_ROWS_SHAPES` selects, the timing runs that
//! slice's weights under the General activation.
mod catalog;
mod fixture;
mod real;
mod timing;

use catalog::{CUBIN, FAMILY, PROBE_SYMBOLS, Shape};
use fixture::{Form, Modules, Step, Weights};
use kernel_build::{BuildOptions, LockDir, Toolchain};
use kernel_harness::kit::cases::f32_projection::Witness;
use kernel_harness::{Cubins, LabResult, Session};
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
/// Every width from one to 32 rows (every decode, small verify and drafter
/// head width, the packed depth-three heads' 18 and 21 among them), then
/// every tile boundary of 8, 16, 32 and 64 rows to the package's verify
/// rows, the drafter's wider head widths and the wide buckets.
const BOUNDARY_ROWS: [u32; 56] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
    27, 28, 29, 30, 31, 32, 33, 35, 40, 45, 47, 48, 50, 55, 60, 63, 64, 65, 66, 95, 96, 97, 127,
    128, 129, 159, 160, 161, 191, 192,
];
const WITNESS_ROWS: [u32; 9] = [1, 6, 16, 32, 33, 60, 64, 65, 192];

/// A named operand pattern, the row counts it runs at, and what its
/// operands fix: the tiles outside the tensor domain and every output.
struct Case {
    label: String,
    a: Vec<f32>,
    b: Vec<f32>,
    rows: Vec<u32>,
    outside: Option<Vec<usize>>,
    expected: Option<Vec<f32>>,
}

impl Case {
    fn new(label: impl Into<String>, (a, b): (Vec<f32>, Vec<f32>), rows: Vec<u32>) -> Self {
        Self {
            label: label.into(),
            a,
            b,
            rows,
            outside: None,
            expected: None,
        }
    }
}

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

/// Every form of `case` against the split route at each of its row counts.
fn compare(s: &Session, modules: &Modules, shape: Shape, case: Case) -> LabResult<usize> {
    let Case {
        label,
        a,
        b,
        rows,
        outside,
        expected,
    } = case;
    let weights = Weights::new(s, modules, shape, &b)?;
    if outside.is_some_and(|o| o != weights.outside) {
        return Err(format!(
            "{} {label}: device weight domain differs",
            shape.name
        ));
    }
    drop(b);
    let mut compared = 0;
    for rows in rows {
        let step = Step::new(s, &weights, rows, &a, every_form(shape, rows))?;
        let name = format!("{} {label}", shape.name);
        let n = step.compare(s, modules, &name)?;
        if let Some(expected) = &expected {
            step.check(s, modules, expected, &name)?;
        }
        compared += n;
        println!(
            "F32_ROWS_BITWISE shape={} pattern={label} rows={rows} forms={} comparisons={n} PASS",
            shape.name,
            step.forms.len()
        );
    }
    Ok(compared)
}

#[test]
#[ignore = "requires staged cubins and an idle GPU"]
fn gpu_verify_rows_bitwise() {
    let ran = kernel_harness::gpu_test_with(cubins(), 8 << 30, |s| {
        let mut compared = 0;
        for shape in timing::selected_shapes() {
            let modules = Modules::load(s, lab_bytes(), shape)?;
            let rows = timing::rows_or(&BOUNDARY_ROWS);
            compared += compare(
                s,
                &modules,
                shape,
                Case::new("mixed", fixture::mixed(shape), rows.clone()),
            )?;
            let boundary = fixture::boundary(shape);
            compared += compare(
                s,
                &modules,
                shape,
                Case {
                    outside: Some(boundary.outside),
                    expected: Some(boundary.expected),
                    ..Case::new("Boundary", (boundary.a, boundary.b), rows)
                },
            )?;
            for witness in WITNESSES {
                compared += compare(
                    s,
                    &modules,
                    shape,
                    Case::new(
                        format!("{witness:?}"),
                        fixture::operands(shape, witness),
                        timing::rows_or(&WITNESS_ROWS),
                    ),
                )?;
            }
        }
        println!("F32_ROWS_BITWISE_COMPLETE comparisons={compared}");
        Ok(compared)
    })
    .unwrap();
    assert!(ran.is_some_and(|n| n > 0), "an idle GPU is required");
}

/// Every slice of the real-slice matrix at every boundary row, under the
/// mixed activation, whose subnormal also sends one row to the SIMT chains.
#[test]
#[ignore = "requires staged cubins, the checkpoint CHOPINFER_DEEPSEEK_CHECKPOINT names and an idle GPU"]
fn gpu_verify_rows_real_slices() {
    let root = real::checkpoint();
    let ran = kernel_harness::gpu_test_with(cubins(), 8 << 30, |s| {
        let (mut slices, mut compared) = (0, 0);
        for shape in catalog::shapes() {
            let modules = Modules::load(s, lab_bytes(), shape)?;
            for slice in real::SLICES.iter().filter(|r| r.shape() == shape) {
                println!(
                    "F32_ROWS_SLICE {} shape={} outside_tiles={:?}",
                    slice.stem, shape.name, slice.outside
                );
                compared += compare(
                    s,
                    &modules,
                    shape,
                    Case {
                        outside: Some(slice.outside.to_vec()),
                        ..Case::new(
                            format!("real:{}", slice.stem),
                            (fixture::mixed(shape).0, slice.load(&root)),
                            BOUNDARY_ROWS.to_vec(),
                        )
                    },
                )?;
                slices += 1;
            }
        }
        assert_eq!(slices, real::SLICES.len());
        println!("F32_ROWS_REAL_COMPLETE slices={slices} comparisons={compared}");
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
