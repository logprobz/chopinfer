//! The exact parallel ordered F32 sum (`cuda/row/ordered_sum.cuh`) on the
//! GPU, through its lab test entries (`row::ordered_sum::test_entry`): per
//! row, one thread adds the terms serially with `__fadd_rn` while the other
//! warps run the method, at every lab layout (4096, 1536 and 512 terms over
//! 224 and 256 workers). Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test
//! -p kernel-lab --release --test ordered_sum -- --test-threads=1`; without
//! a GPU only the CPU contract test runs.
//!
//! - The method's bits equal the serial thread's on every adversarial row
//!   (`rows::adversarial_rows`, seeds 0..16) and on 131,072 random rows per
//!   layout, written into poisoned outputs; both equal the CPU's serial
//!   sum and the CPU emulation's bits on every row whose sum is not NaN
//!   (NaN payloads differ between the GPU's and the CPU's adds).
//! - Mutants (ties away from zero, no fallback) are caught only when their
//!   method bits differ from the serial thread's on some adversarial row of
//!   every layout; the CPU emulation of the same mutant gives the GPU
//!   mutant's bits on every row whose sum is not NaN, so the emulation
//!   follows the device step for step.

use kernel_core::contract::{Argument, ArgumentList, LaunchPlan};
use kernel_lab::launch::{self, Bindings};
use kernel_lab::{LabResult, Session, poison};
use kernels_sm120::row::ordered_sum::rows::{Rng, adversarial_rows, random_row};
use kernels_sm120::row::ordered_sum::{
    self as ordered_sum, Emulator, LAB_LAYOUTS, Layout, MUTANTS, Mutant, serial,
};
use runtime_cuda::Function;
use serde_json::{Value, json};
use std::collections::BTreeMap;

const TERMS: u64 = 0;
const SERIAL: u64 = 1;
const EXACT: u64 = 2;
const RANDOM_ROWS: usize = 1 << 17;
const BATCH: usize = 1 << 13;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

fn cubin(mutant: Mutant) -> String {
    ordered_sum::lab_cubin_spec(mutant)
        .expect("the lab layouts are valid")
        .name
}

/// The serial thread's and the method's bits for `rows` (row-major, each
/// `layout.terms` long), launched into poisoned outputs.
fn launch_rows(
    session: &Session,
    function: &Function,
    layout: Layout,
    rows: &[f32],
) -> LabResult<(Vec<u32>, Vec<u32>)> {
    let count = rows.len() / layout.terms;
    let mut buffers = BTreeMap::new();
    buffers.insert(TERMS, session.upload_f32(rows)?);
    buffers.insert(SERIAL, session.allocate(count * 4)?);
    buffers.insert(EXACT, session.allocate(count * 4)?);
    poison::poison(session, &[&buffers[&SERIAL], &buffers[&EXACT]])?;
    let plan = LaunchPlan {
        grid: [count as u32, 1, 1],
        block: [(layout.workers + 32) as u32, 1, 1],
        dynamic_shared_bytes: 0,
        workspace_bytes: 0,
        cluster: [1, 1, 1],
        pdl: false,
    };
    let args = ArgumentList(vec![
        Argument::DeviceBuffer(TERMS),
        Argument::DeviceBuffer(SERIAL),
        Argument::DeviceBuffer(EXACT),
        Argument::U32(count as u32),
    ]);
    let bindings = Bindings {
        buffers: buffers.iter().map(|(&k, v)| (k, v)).collect(),
    };
    launch::launch(
        &session.context,
        function,
        &session.stream,
        plan,
        &args,
        &bindings,
    )?;
    session.stream.synchronize().map_err(err)?;
    let bits = |handle: u64| -> LabResult<Vec<u32>> {
        Ok(session
            .download_f32(&buffers[&handle], count)?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    Ok((bits(SERIAL)?, bits(EXACT)?))
}

fn nan(bits: u32) -> bool {
    f32::from_bits(bits).is_nan()
}

/// The served entry on `rows` of `layout`: the method equals the serial
/// thread on every row, and the CPU's serial sum and emulation on every row
/// whose sum is not NaN. Returns the rows checked and the NaN rows.
fn served_rows(
    session: &Session,
    function: &Function,
    emulator: &mut Emulator,
    label: &str,
    rows: &[f32],
) -> LabResult<(usize, usize)> {
    let layout = emulator.layout();
    let (gpu_serial, exact) = launch_rows(session, function, layout, rows)?;
    let mut nans = 0;
    for (i, row) in rows.chunks_exact(layout.terms).enumerate() {
        if exact[i] != gpu_serial[i] {
            return Err(format!(
                "{label} row {i}: method {:#010x}, serial thread {:#010x}",
                exact[i], gpu_serial[i]
            ));
        }
        let cpu = serial(row).to_bits();
        let emulated = emulator.sum(row, Mutant::Served).bits;
        if nan(cpu) {
            nans += 1;
            if !nan(exact[i]) || !nan(emulated) {
                return Err(format!(
                    "{label} row {i}: a NaN sum on the CPU, {:#010x} on the GPU",
                    exact[i]
                ));
            }
        } else if exact[i] != cpu || emulated != cpu {
            return Err(format!(
                "{label} row {i}: GPU {:#010x}, CPU serial {cpu:#010x}, emulation {emulated:#010x}",
                exact[i]
            ));
        }
    }
    Ok((rows.len() / layout.terms, nans))
}

fn adversarial(layout: Layout, seeds: std::ops::Range<u64>) -> (Vec<f32>, Vec<&'static str>) {
    let mut terms = Vec::new();
    let mut names = Vec::new();
    for seed in seeds {
        for row in adversarial_rows(layout, seed) {
            terms.extend_from_slice(&row.terms);
            names.push(row.name);
        }
    }
    (terms, names)
}

fn layout_receipt(session: &Session, layout: Layout) -> LabResult<Value> {
    let module = session.module(&cubin(Mutant::Served))?;
    let function = module
        .function(&ordered_sum::test_symbol(layout, Mutant::Served))
        .map_err(err)?;
    let mut emulator = Emulator::new(layout).map_err(err)?;
    let label = format!("{}x{}", layout.terms, layout.workers);
    let (rows, _) = adversarial(layout, 0..16);
    let (adversarial_rows, adversarial_nans) = served_rows(
        session,
        &function,
        &mut emulator,
        &format!("{label} adversarial"),
        &rows,
    )?;
    let mut rng = Rng::new(0x6a7e + layout.terms as u64 * 7 + layout.workers as u64);
    let mut batch = vec![0.0_f32; BATCH * layout.terms];
    let (mut random, mut random_nans) = (0, 0);
    while random < RANDOM_ROWS {
        for row in batch.chunks_exact_mut(layout.terms) {
            random_row(&mut rng, row);
        }
        let (done, nans) = served_rows(
            session,
            &function,
            &mut emulator,
            &format!("{label} random"),
            &batch,
        )?;
        random += done;
        random_nans += nans;
    }
    Ok(json!({
        "terms": layout.terms,
        "workers": layout.workers,
        "adversarial_rows": adversarial_rows,
        "adversarial_nan_rows": adversarial_nans,
        "random_rows": random,
        "random_nan_rows": random_nans,
        "bitwise_serial_thread": true,
        "bitwise_cpu_serial_and_emulation_where_not_nan": true,
    }))
}

/// `mutant` at `layout` on adversarial rows: caught when some row's method
/// bits differ from the serial thread's; the emulated mutant must give the
/// GPU mutant's bits wherever neither is NaN.
fn mutant_receipt(session: &Session, layout: Layout, mutant: Mutant) -> LabResult<Value> {
    let module = session.module(&cubin(mutant))?;
    let function = module
        .function(&ordered_sum::test_symbol(layout, mutant))
        .map_err(err)?;
    let mut emulator = Emulator::new(layout).map_err(err)?;
    let (rows, names) = adversarial(layout, 0..4);
    let (gpu_serial, exact) = launch_rows(session, &function, layout, &rows)?;
    let mut caught = Vec::new();
    for (i, row) in rows.chunks_exact(layout.terms).enumerate() {
        if exact[i] != gpu_serial[i] {
            caught.push(names[i]);
        }
        let emulated = emulator.sum(row, mutant).bits;
        if !nan(emulated) && !nan(exact[i]) && emulated != exact[i] {
            return Err(format!(
                "{}x{} {mutant:?} row {i} ({}): GPU {:#010x}, emulation {emulated:#010x}",
                layout.terms, layout.workers, names[i], exact[i]
            ));
        }
    }
    if caught.is_empty() {
        return Err(format!(
            "{}x{}: the {mutant:?} mutant reproduced the serial sum on every row",
            layout.terms, layout.workers
        ));
    }
    let rows_caught = caught.len();
    caught.dedup();
    Ok(json!({
        "terms": layout.terms,
        "workers": layout.workers,
        "mutant": format!("{mutant:?}"),
        "outcome": "caught",
        "rows_caught": rows_caught,
        "kinds": caught,
    }))
}

#[test]
fn ordered_sum_cpu_contract() {
    for mutant in [Mutant::Served].into_iter().chain(MUTANTS) {
        let entries = kernel_lab::cubin_entries(&cubin(mutant)).unwrap();
        let expected = LAB_LAYOUTS
            .iter()
            .map(|&layout| ordered_sum::test_symbol(layout, mutant))
            .collect::<Vec<_>>();
        assert_eq!(entries, expected);
        kernel_lab::cubin(&cubin(mutant)).unwrap();
    }
    for layout in LAB_LAYOUTS {
        let (rows, names) = adversarial(layout, 0..1);
        assert_eq!(rows.len(), names.len() * layout.terms);
    }
}

#[test]
fn ordered_sum_matches_the_serial_sum_bitwise() {
    kernel_lab::gpu_test(4 << 30, |session| {
        let rows = LAB_LAYOUTS
            .iter()
            .map(|&layout| layout_receipt(session, layout))
            .collect::<LabResult<Vec<_>>>()?;
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("ordered_sum_bitwise"));
        receipt.insert("layouts".into(), json!(rows));
        session.receipt("ordered_sum_bitwise", receipt)?;
        Ok(())
    })
    .expect("ordered sum bitwise comparison");
}

#[test]
fn ordered_sum_mutants_are_caught() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let mut rows = Vec::new();
        for layout in LAB_LAYOUTS {
            for mutant in MUTANTS {
                rows.push(mutant_receipt(session, layout, mutant)?);
            }
        }
        let mut receipt = BTreeMap::new();
        receipt.insert("op".into(), json!("ordered_sum_mutants"));
        receipt.insert("layouts".into(), json!(rows));
        session.receipt("ordered_sum_mutants", receipt)?;
        Ok(())
    })
    .expect("ordered sum mutants");
}
