//! The MiMo bundle's retained-row kernel writes every live row of a decode or
//! verify phase into its slot's circular drafter history, at the cell
//! the contract's oracle gives, and touches nothing else: not the
//! other columns of a history row, not other cells, and not the rows of a
//! planned bucket past the live count. Its catch-up gather reads the same
//! cells back into consecutive rows.

use std::collections::BTreeSet;

use bundle_mimo26::retain::{gather_rows_symbol, retain_rows_symbol};
use kernel_lab::launch::simple_config;
use model_mimo26_contract::{
    Tp,
    own_ops::retain::{GatherRows, History, RetainRows, gather_rows, retain_rows},
};
use runtime_cuda::{KernelArgument, LaunchConfig};

#[path = "common/bundle_launch.rs"]
mod bundle_launch;

/// One layout: input row stride, history row stride, written column offset
/// and width, all in BF16 elements, and the rows per slot.
struct Layout {
    input_stride: u32,
    history_stride: u32,
    column: u32,
    width: u32,
    capacity: u64,
}

const SLOTS: u64 = 32;

fn bytes(values: &[i32]) -> Vec<u8> {
    let mut bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    bytes.resize(4 * 64, 0);
    bytes
}

/// Items as (slot, q_len, prefix_len), packed back to back; `planned` rows
/// are launched, as a token bucket is.
fn run_case(
    session: &kernel_lab::Session,
    function: &runtime_cuda::Function,
    layout: &Layout,
    items: &[(i32, i32, i32)],
    planned: u32,
) -> kernel_lab::LabResult<()> {
    let rows = items.iter().map(|item| item.1).sum::<i32>() as usize;
    let input_elements = planned as usize * layout.input_stride as usize;
    let history_elements = (SLOTS * layout.capacity) as usize * layout.history_stride as usize;
    let input_host = (0..input_elements)
        .flat_map(|index| ((index as u32).wrapping_mul(2_654_435_761) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let history_host = (0..history_elements)
        .flat_map(|index| ((index as u32 ^ 0x5a5a) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let upload = |data: &[u8]| -> kernel_lab::LabResult<_> {
        let buffer = session.allocate(data.len())?;
        buffer
            .copy_from_bytes(data, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let input = upload(&input_host)?;
    let history = upload(&history_host)?;
    let mut q_start = Vec::new();
    let mut token_batch = Vec::new();
    for (item, &(_, q_len, _)) in items.iter().enumerate() {
        q_start.push(token_batch.len() as i32);
        token_batch.extend(std::iter::repeat_n(item as i32, q_len as usize));
    }
    let slot_ids = items.iter().map(|item| item.0).collect::<Vec<_>>();
    let prefix_len = items.iter().map(|item| item.2).collect::<Vec<_>>();
    let slots = upload(&bytes(&slot_ids))?;
    let token_batch_buffer = upload(&bytes(&token_batch))?;
    let q_start_buffer = upload(&bytes(&q_start))?;
    let prefix = upload(&bytes(&prefix_len))?;
    let live = upload(&(rows as u32).to_le_bytes())?;
    let column_bytes = layout.column as usize * 2;
    function
        .launch(
            &session.stream,
            simple_config(
                (planned, 1, 1),
                (256, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?,
            &[
                KernelArgument::Buffer(&input),
                KernelArgument::Range {
                    buffer: &history,
                    offset: column_bytes,
                    byte_len: history_host.len() - column_bytes,
                },
                KernelArgument::Buffer(&slots),
                KernelArgument::Buffer(&token_batch_buffer),
                KernelArgument::Buffer(&q_start_buffer),
                KernelArgument::Buffer(&prefix),
                KernelArgument::Buffer(&live),
                KernelArgument::U32(layout.width),
                KernelArgument::U32(layout.input_stride),
                KernelArgument::U32(layout.history_stride),
                KernelArgument::U32(layout.capacity as u32),
                KernelArgument::U32(SLOTS as u32),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut device = vec![0_u8; history_host.len()];
    history
        .copy_to_bytes(&mut device, &session.stream)
        .map_err(|e| e.to_string())?;

    let mut expected = history_host.clone();
    retain_rows(
        &RetainRows {
            width: layout.width,
            input_stride: layout.input_stride,
            history: History {
                stride: layout.history_stride,
                capacity: layout.capacity as u32,
                slots: SLOTS as u32,
            },
        },
        &input_host,
        &mut expected[column_bytes..],
        &slot_ids,
        &token_batch,
        &q_start,
        &prefix_len,
        rows as u32,
        planned,
    )
    .map_err(|e| format!("case: {}", e.0))?;
    if device != expected {
        let row_bytes = layout.history_stride as usize * 2;
        let wrong = (0..SLOTS * layout.capacity)
            .filter(|&cell| {
                let at = cell as usize * row_bytes;
                device[at..at + row_bytes] != expected[at..at + row_bytes]
            })
            .collect::<BTreeSet<_>>();
        return Err(format!("history cells differ: {wrong:?}"));
    }
    Ok(())
}

#[test]
fn retained_rows_land_in_their_history_cells_only() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = retain_rows_symbol(Tp::TP4);
        let needed = BTreeSet::from([entry.as_str()]);
        let functions = bundle_launch::functions(std::slice::from_ref(session), &needed)?;
        let function = &functions[0][entry.as_str()];
        // MTP: whole 4096-wide hidden rows, 136 rows per slot.
        let mtp = Layout {
            input_stride: 4096,
            history_stride: 4096,
            column: 0,
            width: 4096,
            capacity: 136,
        };
        // DFlash rank 1, tap 2: 2048 columns into shard columns 3072..5120,
        // 1031 rows per slot.
        let dflash = Layout {
            input_stride: 4096,
            history_stride: 5120,
            column: 3072,
            width: 2048,
            capacity: 1031,
        };
        for layout in [&mtp, &dflash] {
            let capacity = layout.capacity as i32;
            // Decode: one row per item, at and across the wrap, in a bucket
            // planned for more rows than are live.
            run_case(
                session,
                function,
                layout,
                &[
                    (0, 1, 0),
                    (5, 1, capacity - 1),
                    (31, 1, capacity),
                    (7, 1, 3 * capacity + 2),
                ],
                8,
            )?;
            // Verify: items of 2, 4 and 8 rows, one crossing the wrap.
            run_case(
                session,
                function,
                layout,
                &[(2, 2, 9), (30, 4, capacity - 2), (11, 8, 5 * capacity - 3)],
                16,
            )?;
        }
        Ok(())
    })
    .expect("MiMo retained rows");
}

/// Items as (slot, rows, first position), their rows back to back; `planned`
/// rows are launched, as a catch-up bucket is. Every live row must hold its
/// history cell and every planned row past them must keep its bytes.
fn run_gather(
    session: &kernel_lab::Session,
    function: &runtime_cuda::Function,
    capacity: u64,
    items: &[(i32, i32, i32)],
    planned: u32,
) -> kernel_lab::LabResult<()> {
    const WIDTH: usize = 5120;
    let history_host = (0..(SLOTS * capacity) as usize * WIDTH)
        .flat_map(|index| ((index as u32).wrapping_mul(2_654_435_761) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let output_host = (0..planned as usize * WIDTH)
        .flat_map(|index| ((index as u32 ^ 0x5a5a) as u16).to_le_bytes())
        .collect::<Vec<_>>();
    let upload = |data: &[u8]| -> kernel_lab::LabResult<_> {
        let buffer = session.allocate(data.len())?;
        buffer
            .copy_from_bytes(data, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(buffer)
    };
    let history = upload(&history_host)?;
    let output = upload(&output_host)?;
    let mut token_batch = Vec::new();
    let mut positions = Vec::new();
    for (item, &(_, rows, first)) in items.iter().enumerate() {
        token_batch.extend(std::iter::repeat_n(item as i32, rows as usize));
        positions.extend(first..first + rows);
    }
    let padded = |values: &[i32]| {
        let mut bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        bytes.resize(4 * planned.max(64) as usize, 0);
        upload(&bytes)
    };
    let slot_ids = items.iter().map(|item| item.0).collect::<Vec<_>>();
    let slots = padded(&slot_ids)?;
    let token_batch_buffer = padded(&token_batch)?;
    let positions_buffer = padded(&positions)?;
    let live = upload(&(positions.len() as u32).to_le_bytes())?;
    function
        .launch(
            &session.stream,
            simple_config(
                (planned, 1, 1),
                (256, 1, 1),
                0,
                LaunchConfig::NO_CLUSTER,
                true,
            )?,
            &[
                KernelArgument::Buffer(&history),
                KernelArgument::Buffer(&output),
                KernelArgument::Buffer(&slots),
                KernelArgument::Buffer(&token_batch_buffer),
                KernelArgument::Buffer(&positions_buffer),
                KernelArgument::Buffer(&live),
                KernelArgument::U32(WIDTH as u32),
                KernelArgument::U32(WIDTH as u32),
                KernelArgument::U32(WIDTH as u32),
                KernelArgument::U32(capacity as u32),
                KernelArgument::U32(SLOTS as u32),
            ],
        )
        .map_err(|e| e.to_string())?;
    let mut device = vec![0_u8; output_host.len()];
    output
        .copy_to_bytes(&mut device, &session.stream)
        .map_err(|e| e.to_string())?;
    let mut expected = output_host.clone();
    let row_bytes = WIDTH * 2;
    gather_rows(
        &GatherRows {
            width: WIDTH as u32,
            output_stride: WIDTH as u32,
            history: History {
                stride: WIDTH as u32,
                capacity: capacity as u32,
                slots: SLOTS as u32,
            },
        },
        &history_host,
        &mut expected,
        &slot_ids,
        &token_batch,
        &positions,
        positions.len() as u32,
        planned,
    )
    .map_err(|e| format!("case: {}", e.0))?;
    if device != expected {
        let wrong = (0..planned as usize)
            .filter(|&row| {
                device[row * row_bytes..(row + 1) * row_bytes]
                    != expected[row * row_bytes..(row + 1) * row_bytes]
            })
            .collect::<Vec<_>>();
        return Err(format!("gathered rows differ: {wrong:?}"));
    }
    Ok(())
}

#[test]
fn gathered_rows_read_their_history_cells() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = gather_rows_symbol(Tp::TP4);
        let needed = BTreeSet::from([entry.as_str()]);
        let functions = bundle_launch::functions(std::slice::from_ref(session), &needed)?;
        let function = &functions[0][entry.as_str()];
        let capacity = 1031_u64;
        let wrap = capacity as i32;
        // Steady drafting: 32 items of 1 to 8 kept rows, some across the
        // wrap, in the 256-row bucket.
        let steady = (0..32)
            .map(|slot| {
                (
                    slot,
                    slot % 8 + 1,
                    if slot % 3 == 0 { wrap - 2 } else { 40 * slot },
                )
            })
            .collect::<Vec<_>>();
        run_gather(session, function, capacity, &steady, 256)?;
        // A long gap after a restore reads the whole window from the wrap,
        // as one prefill-sized pass.
        run_gather(
            session,
            function,
            capacity,
            &[(9, 1023, 2045), (4, 5, 0)],
            1028,
        )?;
        Ok(())
    })
    .expect("MiMo gathered rows");
}

/// Wall time of one K=0 step's retention on one rank, as the median of 200
/// synchronized repetitions: the per-item device copies commit issued before
/// (MTP: one staging copy, then one 8 KiB copy per item), against one
/// retained-row launch. The launch is eager here; inside the decode graph it
/// is one more kernel node.
#[test]
#[ignore = "timing: run only in a benchmark window"]
fn retained_row_copy_cost() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = retain_rows_symbol(Tp::TP4);
        let needed = BTreeSet::from([entry.as_str()]);
        let functions = bundle_launch::functions(std::slice::from_ref(session), &needed)?;
        let function = &functions[0][entry.as_str()];
        const ROW: usize = 4096 * 2;
        const CAPACITY: usize = 136;
        let hidden = session.allocate(32 * ROW)?;
        let staging = session.allocate(32 * ROW)?;
        let history = session.allocate(32 * CAPACITY * ROW)?;
        let upload = |values: &[i32]| -> kernel_lab::LabResult<_> {
            let buffer = session.allocate(4 * 64)?;
            buffer
                .copy_from_bytes(&bytes(values), &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(buffer)
        };
        let median = |mut run: Box<dyn FnMut() -> kernel_lab::LabResult<()> + '_>| {
            let mut times = Vec::new();
            for repetition in 0..220 {
                let started = std::time::Instant::now();
                run()?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                if repetition >= 20 {
                    times.push(started.elapsed().as_secs_f64() * 1e6);
                }
            }
            times.sort_by(f64::total_cmp);
            Ok::<f64, String>(times[times.len() / 2])
        };
        for items in [1_usize, 4, 16, 32] {
            let positions = (0..items as i32).map(|item| 1000 + 37 * item).collect::<Vec<_>>();
            let copies = median(Box::new(|| {
                staging
                    .copy_range_from_async(0, &hidden, 0, items * ROW, &session.stream)
                    .map_err(|e| e.to_string())?;
                for (item, &position) in positions.iter().enumerate() {
                    let cell = item * CAPACITY + position as usize % CAPACITY;
                    history
                        .copy_range_from_async(cell * ROW, &staging, item * ROW, ROW, &session.stream)
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            }))?;
            let slots = upload(&(0..items as i32).collect::<Vec<_>>())?;
            let token_batch = upload(&(0..items as i32).collect::<Vec<_>>())?;
            let q_start = upload(&(0..items as i32).collect::<Vec<_>>())?;
            let prefix = upload(&positions)?;
            let live = upload(&[items as i32])?;
            let kernel = median(Box::new(|| {
                function
                    .launch(
                        &session.stream,
                        simple_config(
                            (items.next_power_of_two() as u32, 1, 1),
                            (256, 1, 1),
                            0,
                            LaunchConfig::NO_CLUSTER,
                            true,
                        )?,
                        &[
                            KernelArgument::Buffer(&hidden),
                            KernelArgument::Buffer(&history),
                            KernelArgument::Buffer(&slots),
                            KernelArgument::Buffer(&token_batch),
                            KernelArgument::Buffer(&q_start),
                            KernelArgument::Buffer(&prefix),
                            KernelArgument::Buffer(&live),
                            KernelArgument::U32(4096),
                            KernelArgument::U32(4096),
                            KernelArgument::U32(4096),
                            KernelArgument::U32(CAPACITY as u32),
                            KernelArgument::U32(SLOTS as u32),
                        ],
                    )
                    .map_err(|e| e.to_string())
            }))?;
            let empty = median(Box::new(|| Ok(())))?;
            println!(
                "retain C{items}: {} copies {copies:.1} us, one kernel {kernel:.1} us, empty sync {empty:.1} us",
                items + 1
            );
        }
        Ok(())
    })
    .expect("MiMo retained-row timing");
}
