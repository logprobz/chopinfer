//! A block-32 row slice allocates only its padded result: a narrow K window
//! of many scattered rows from a very wide source never sizes a buffer by
//! the source's K. One test, since the allocation counters are global.
// SAFETY-REVIEWED: the only unsafe code is the counting allocator below,
// which forwards to the system allocator.
#![allow(unsafe_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use kernel_core::contract::{DType, WeightFormat};
use kernel_core::precision::NibbleOrder;
use kernel_core::precision::format::{FormatId, registry};
use kernel_core::precision::packed::{MatrixSlice, MatrixView, ScaleBytes};

struct Counting;

static LARGEST: AtomicUsize = AtomicUsize::new(0);

/// Requests above this fail instead of reserving memory the test never needs.
const REFUSED: usize = 1 << 30;

// SAFETY-REVIEWED: every method forwards to the system allocator with the
// caller's pointer and layout unchanged, refuses oversized requests by
// returning null as the allocator contract allows, and updates a counter.
unsafe impl GlobalAlloc for Counting {
    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if record(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        unsafe { System.alloc(layout) }
    }

    // SAFETY-REVIEWED: forwards the caller's layout contract unchanged.
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if record(layout.size()) {
            return std::ptr::null_mut();
        }
        // SAFETY-REVIEWED: the caller guarantees a valid nonzero layout.
        unsafe { System.alloc_zeroed(layout) }
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        unsafe { System.dealloc(pointer, layout) }
    }

    // SAFETY-REVIEWED: forwards a pointer this allocator returned.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if record(size) {
            return std::ptr::null_mut();
        }
        // SAFETY-REVIEWED: the pointer came from System with this layout.
        unsafe { System.realloc(pointer, layout, size) }
    }
}

/// Records a request's size and whether it is refused.
fn record(size: usize) -> bool {
    LARGEST.fetch_max(size, Ordering::SeqCst);
    size > REFUSED
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn a_narrow_window_of_scattered_rows_allocates_only_its_result() {
    let k = 1 << 20;
    let codes: Vec<u8> = (0..2 * k).map(|i| (i % 113) as u8 & 0x7e).collect();
    let scales: Vec<u8> = (0..k / 32).map(|i| 100 + (i % 20) as u8).collect();
    let matrix = MatrixView {
        tensor: "wide",
        format: WeightFormat::Fp8Block32x32,
        rows: 2,
        k,
        row_origin: 0,
        k_origin: 0,
        nibble_order: NibbleOrder::EvenLow,
        codes: &codes,
        local: Some(ScaleBytes {
            dtype: DType::Ue8M0,
            bytes: &scales,
        }),
        global: None,
    };
    let rows: Vec<usize> = [1, 0].repeat(32_768);
    let slice = MatrixSlice {
        rows,
        k_start: 0,
        k: 32,
        padded_rows: 65_536,
        padded_k: 32,
        nibble_order: NibbleOrder::EvenLow,
    };
    LARGEST.store(0, Ordering::SeqCst);
    let packed = registry::by_id(FormatId::Fp8Block32)
        .unwrap()
        .repack_mxfp8_rows_slice(matrix, &slice)
        .unwrap();
    let largest = LARGEST.load(Ordering::SeqCst);
    assert_eq!(packed.codes.len(), 2 << 20);
    assert_eq!(packed.scales.len(), 64 << 10);
    // The codes of the result are the largest buffer.
    assert!(largest <= 2 << 20, "largest allocation {largest} bytes");
    for (out_row, &row) in slice.rows.iter().enumerate().step_by(4093) {
        assert_eq!(
            packed.codes[out_row * 32..(out_row + 1) * 32],
            codes[row * k..row * k + 32]
        );
        assert_eq!(packed.scales[out_row], scales[0]);
    }
}
