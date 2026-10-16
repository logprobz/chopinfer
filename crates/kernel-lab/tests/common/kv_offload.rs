//! Test-local CUDA batch bridge. Production ownership APIs remain unchanged.
#![allow(dead_code)]
use kernel_core::contract::DType;
use kernel_lab::{LabResult, Session};
use runtime_cuda::{DeviceBuffer, PinnedHostBuffer, Stream};
use serde_json::{Value, json};
use std::collections::BTreeMap;

pub fn enabled() -> bool {
    std::env::var("CHOPINFER_KVO_LAB").as_deref() == Ok("1")
}

pub fn record(s: &Session, measurement: &str, data: Value) -> LabResult<()> {
    let mut fields: BTreeMap<String, Value> =
        serde_json::from_value(data).map_err(|e| e.to_string())?;
    fields.insert("schema".into(), json!("chopinfer.kv-offload-lab.v1"));
    fields.insert("measurement".into(), json!(measurement));
    fields.insert(
        "run_id".into(),
        json!(std::env::var("CHOPINFER_KVO_RUN_ID").unwrap_or_default()),
    );
    fields.insert(
        "revision".into(),
        json!(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .map_err(|e| e.to_string())?
                .stdout
                .into_iter()
                .map(char::from)
                .collect::<String>()
                .trim()
        ),
    );
    let path = s.receipt("kv-offload", fields)?;
    let saved: Value = serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if saved["measurement_label"] != "exclusive-device" {
        return Err("shared-device receipt is not performance evidence".into());
    }
    eprintln!("KVO receipt {}", path.display());
    Ok(())
}

pub fn exclusive(sessions: &[Session]) -> LabResult<()> {
    for s in sessions {
        if s.metadata["measurement_label"] != "exclusive-device" {
            return Err(format!(
                "GPU {} is shared; refusing performance evidence",
                s.ordinal
            ));
        }
    }
    Ok(())
}

pub fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

// Mirrored from models/mimo26/bundle/src/contract.rs:16-31 and
// models/qwen38/bundle/src/contract.rs:23-103, like kv_offload_model.py.
// Qwen FP8 is a design-only size until its bundle contract lands.
pub const PAGES: [(&str, usize); 4] = [
    ("mimo_tp4_bf16", 9 * 64 * (192 + 128) * 2),
    ("mimo_tp4_fp8", 9 * 64 * (192 + 128)),
    ("qwen_tp4_bf16", 12 * (64 * 256 * 2 * 2 + 64 / 4 * 128 * 2)),
    (
        "qwen_tp4_fp8_design",
        12 * (64 * 256 * 2 + 64 / 4 * 128 * 2),
    ),
];

pub fn page_index(i: usize, pages: usize, scattered: bool) -> usize {
    assert!(pages.is_power_of_two());
    if scattered {
        2 * ((i * 73 + 19) % pages)
    } else {
        i
    }
}

// Unsafe code is confined to the ABI bridge, as in runtime-cuda's FFI.
// Metadata lives through the API call; DMA allocations stay alive until sync.
// SAFETY-REVIEWED: test-only batch ABI; Arena owns and bounds every DMA operand and drains its stream before freeing memory.
#[allow(unsafe_code)]
mod driver {
    use super::*;
    use std::ffi::{c_char, c_int, c_void};
    #[repr(C)]
    struct Attributes {
        order: i32,
        src: [i32; 2],
        dst: [i32; 2],
        flags: u32,
    }
    const _: () = assert!(std::mem::size_of::<Attributes>() == 24);
    const _: () = assert!(std::mem::align_of::<Attributes>() == 4);
    const _: () = assert!(std::mem::size_of::<usize>() == 8);
    // SAFETY-REVIEWED: matches CUDA 13 CUmemcpyAttributes and cuMemcpyBatchAsync_v2: two CUdeviceptr arrays, size_t counts, attributes, indices and CUstream.
    type Batch = unsafe extern "C" fn(
        *mut u64,
        *mut u64,
        *mut usize,
        usize,
        *mut Attributes,
        *mut usize,
        usize,
        *mut c_void,
    ) -> i32;
    // SAFETY-REVIEWED: these POSIX loader declarations use C int flags and opaque pointers; all input names are static NUL-terminated strings.
    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(name: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    }
    pub struct Api {
        batch: Batch,
    }
    impl Api {
        pub fn open() -> LabResult<Self> {
            // Retain libcuda for the process lifetime. No CUDA entry is called
            // until the caller has passed the lab's GPU gate.
            // SAFETY-REVIEWED: checked non-null symbol from libcuda has the Batch ABI above; retaining the library forever keeps the function pointer live.
            unsafe {
                let lib = dlopen(c"libcuda.so.1".as_ptr(), 2);
                if lib.is_null() {
                    return Err("dlopen libcuda.so.1 failed".into());
                }
                let ptr = dlsym(lib, c"cuMemcpyBatchAsync_v2".as_ptr());
                if ptr.is_null() {
                    return Err("cuMemcpyBatchAsync_v2 unavailable".into());
                }
                Ok(Self {
                    batch: std::mem::transmute::<*mut c_void, Batch>(ptr),
                })
            }
        }
        pub(super) fn enqueue(
            &self,
            stream: &Stream,
            dst: &mut [u64],
            src: &mut [u64],
            sizes: &mut [usize],
        ) -> LabResult<()> {
            assert_eq!(dst.len(), src.len());
            assert_eq!(dst.len(), sizes.len());
            let mut attrs = Attributes {
                order: 1,
                src: [0; 2],
                dst: [0; 2],
                flags: 0,
            };
            let mut index = 0;
            // SAFETY-REVIEWED: Arena supplies checked device ranges and pinned host addresses; metadata arrays live through this call, operands through stream completion, and with_raw_handle selects the live context.
            let code = stream
                .with_raw_handle(|raw| unsafe {
                    (self.batch)(
                        dst.as_mut_ptr(),
                        src.as_mut_ptr(),
                        sizes.as_mut_ptr(),
                        sizes.len(),
                        &mut attrs,
                        &mut index,
                        1,
                        raw,
                    )
                })
                .map_err(|e| e.to_string())?;
            if code != 0 {
                return Err(format!("cuMemcpyBatchAsync_v2 returned {code}"));
            }
            Ok(())
        }
    }
}
pub use driver::Api;

fn copy_pattern(page: usize, pages: usize) -> Vec<u8> {
    (0..page * pages * 2)
        .map(|i| ((i / page * 17 + i % 251) % 256) as u8)
        .collect()
}

fn check_copy(
    label: &str,
    copied: &[u8],
    expected: &[u8],
    page: usize,
    pages: usize,
    scattered: bool,
) -> LabResult<()> {
    if copied.len() != expected.len() {
        return Err(format!("{label}: allocation length differs"));
    }
    let map = kernel_lab::poison::WriteMap::of(DType::U8).byte_ranges((0..pages).map(|i| {
        let start = page_index(i, pages, scattered) * page;
        start..start + page
    }))?;
    let mut selected = vec![false; expected.len() / page];
    for range in map.ranges() {
        selected[range.start / page..range.end / page].fill(true);
    }
    for (i, (&actual, &value)) in copied.iter().zip(expected).enumerate() {
        let wanted = if selected[i / page] { value } else { !value };
        if actual != wanted {
            return Err(format!(
                "{label}: byte {i} is {actual:#04x}, expected {wanted:#04x}"
            ));
        }
    }
    Ok(())
}

#[test]
fn copy_destination_contract_cpu() {
    for scattered in [false, true] {
        let (page, pages) = (7, 4);
        let expected = copy_pattern(page, pages);
        let mut copied = expected.iter().map(|byte| !byte).collect::<Vec<_>>();
        assert!(check_copy("no-op", &copied, &expected, page, pages, scattered).is_err());
        for i in 0..pages {
            let start = page_index(i, pages, scattered) * page;
            copied[start..start + page].copy_from_slice(&expected[start..start + page]);
        }
        assert!(check_copy("copied", &copied, &expected, page, pages, scattered).is_ok());
        let dead = (0..pages * 2)
            .find(|&slot| !(0..pages).any(|i| page_index(i, pages, scattered) == slot))
            .unwrap();
        copied[dead * page] ^= 1;
        assert!(check_copy("untouched", &copied, &expected, page, pages, scattered).is_err());
    }
}

pub struct Arena {
    host: PinnedHostBuffer,
    host_address: u64,
    device: DeviceBuffer,
    pub stream: Stream,
    page: usize,
    pages: usize,
}
impl Arena {
    pub fn new(s: &Session, page: usize, pages: usize) -> LabResult<Self> {
        let bytes = page * pages * 2;
        let mut host = PinnedHostBuffer::new(&s.context, bytes).map_err(|e| e.to_string())?;
        for (i, byte) in host.as_mut_slice().iter_mut().enumerate() {
            *byte = ((i / page * 17 + i % 251) % 256) as u8;
        }
        let device = s.allocate(bytes)?;
        device
            .copy_from_pinned_at_async(0, &host, 0, bytes, &s.stream)
            .map_err(|e| e.to_string())?
            .finish();
        let stream = s.context.stream().map_err(|e| e.to_string())?;
        // Acquire the address while idle. Do not create host slice borrows
        // while later D2H batches may still be writing this allocation.
        let host_address = host.as_mut_slice().as_mut_ptr() as u64;
        Ok(Self {
            host,
            host_address,
            device,
            stream,
            page,
            pages,
        })
    }
    pub fn bytes(&self) -> usize {
        self.page * self.pages
    }
    pub fn enqueue(&mut self, api: &Api, scattered: bool, h2d: bool) -> LabResult<()> {
        let mut hosts = Vec::new();
        let mut devices = Vec::new();
        for i in 0..self.pages {
            let offset = page_index(i, self.pages, scattered) * self.page;
            hosts.push(self.host_address + offset as u64);
            devices.push(
                self.device
                    .device_address_at(offset, self.page)
                    .map_err(|e| e.to_string())?,
            );
        }
        let mut sizes = vec![self.page; self.pages];
        if h2d {
            api.enqueue(&self.stream, &mut devices, &mut hosts, &mut sizes)
        } else {
            api.enqueue(&self.stream, &mut hosts, &mut devices, &mut sizes)
        }
    }
    pub fn sync(&self) -> LabResult<()> {
        self.stream.synchronize().map_err(|e| e.to_string())
    }
    pub fn verify(&mut self, s: &Session, api: &Api, scattered: bool) -> LabResult<()> {
        // Fence both streams before preparing either DMA destination.
        self.sync()?;
        s.stream.synchronize().map_err(|e| e.to_string())?;
        let expected = copy_pattern(self.page, self.pages);
        let prepared = expected.iter().map(|byte| !byte).collect::<Vec<_>>();
        kernel_lab::poison::write(s, &self.device, 0, &prepared)?;
        self.host.as_mut_slice().copy_from_slice(&expected);
        self.enqueue(api, scattered, true)?;
        self.sync()?;
        let mut failures = Vec::new();
        let copied = kernel_lab::poison::read(s, &self.device)?;
        if let Err(error) = check_copy("H2D", &copied, &expected, self.page, self.pages, scattered)
        {
            failures.push(error);
        }
        // D2H has an independent valid source and a fresh complemented destination.
        kernel_lab::poison::write(s, &self.device, 0, &expected)?;
        self.host.as_mut_slice().copy_from_slice(&prepared);
        self.enqueue(api, scattered, false)?;
        self.sync()?;
        if let Err(error) = check_copy(
            "D2H",
            self.host.as_slice(),
            &expected,
            self.page,
            self.pages,
            scattered,
        ) {
            failures.push(error);
        }
        // Timing uses the original canonical operands, even after a miss.
        self.host.as_mut_slice().copy_from_slice(&expected);
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("; "))
        }
    }
}
impl Drop for Arena {
    fn drop(&mut self) {
        // Never release a DMA operand after unconfirmed completion.
        if self.stream.synchronize().is_err() {
            std::process::abort();
        }
    }
}

/// Queue every rank before releasing any DMA. The release streams are only
/// control traffic; payload traffic still uses one copy stream per rank.
pub struct CopyGate {
    flags: Vec<DeviceBuffer>,
    release: Vec<Stream>,
    functions: Vec<runtime_cuda::Function>,
}
impl CopyGate {
    pub fn new(sessions: &[Session]) -> LabResult<Self> {
        let flags = sessions
            .iter()
            .map(|s| s.allocate(4))
            .collect::<LabResult<Vec<_>>>()?;
        let release = sessions
            .iter()
            .map(|s| s.context.stream().map_err(|e| e.to_string()))
            .collect::<LabResult<Vec<_>>>()?;
        let functions = sessions
            .iter()
            .map(|s| {
                s.module("probe_sm120")?
                    .function("chop_probe_gate_v1")
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        Ok(Self {
            flags,
            release,
            functions,
        })
    }
    pub fn reset(&self) -> LabResult<()> {
        for (flag, stream) in self.flags.iter().zip(&self.release) {
            flag.copy_from_bytes(&0_u32.to_ne_bytes(), stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn arm(&self, rank: usize, stream: &Stream) -> LabResult<()> {
        self.functions[rank]
            .launch(
                stream,
                kernel_lab::launch::simple_config(
                    (1, 1, 1),
                    (32, 1, 1),
                    0,
                    runtime_cuda::LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[runtime_cuda::KernelArgument::Buffer(&self.flags[rank])],
            )
            .map_err(|e| e.to_string())
    }
    pub fn release(&self) -> LabResult<()> {
        for (flag, stream) in self.flags.iter().zip(&self.release) {
            flag.copy_from_bytes(&1_u32.to_ne_bytes(), stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}
