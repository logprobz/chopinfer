//! H.264 decoding on NVDEC through the driver's `libnvcuvid`, loaded at run
//! time.
//!
//! `csrc/nvdec.c` owns every NVDEC structure and compiles against the vendored
//! headers in `bindings/nvdec`. This module owns the session (the loaded
//! library and a CUDA context of its own), the Annex B input, and the
//! callbacks that choose and receive frames. A host without `libnvcuvid` gets
//! `NvdecError::LibraryAbsent` from `open` and nothing else changes.

// SAFETY-REVIEWED: this private module is the reviewed boundary to the NVDEC helper.
#![allow(unsafe_code)]

use std::error::Error;
use std::ffi::{c_int, c_uint, c_void};
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::{self, NonNull};

const OK: c_int = 0;
const LIBRARY_ABSENT: c_int = 1;
const SYMBOL_MISSING: c_int = 2;
const DRIVER: c_int = 3;
const UNSUPPORTED: c_int = 4;
const STREAM: c_int = 5;
const SINK: c_int = 6;

type SelectCallback = extern "C" fn(*mut c_void, i64) -> c_int;
type FrameCallback = extern "C" fn(*mut c_void, i64, u32, u32, *const u8) -> c_int;

// SAFETY-REVIEWED: these helpers are compiled from csrc/nvdec.c against the pinned headers.
unsafe extern "C" {
    fn runtime_cuda_nvdec_session_open(
        device_ordinal: c_int,
        session: *mut *mut c_void,
        driver_status: *mut c_uint,
    ) -> c_int;
    fn runtime_cuda_nvdec_session_close(session: *mut c_void);
    fn runtime_cuda_nvdec_h264_capacity(
        session: *mut c_void,
        max_width: *mut u32,
        max_height: *mut u32,
        driver_status: *mut c_uint,
    ) -> c_int;
    fn runtime_cuda_nvdec_decode_h264(
        session: *mut c_void,
        stream: *const u8,
        unit_ends: *const u64,
        timestamps: *const i64,
        units: u64,
        max_width: u32,
        max_height: u32,
        user: *mut c_void,
        select: SelectCallback,
        frame: FrameCallback,
        driver_status: *mut c_uint,
    ) -> c_int;
}

/// A refusal or failure from the NVDEC path.
#[derive(Debug)]
pub enum NvdecError {
    /// The driver installation has no `libnvcuvid.so.1`.
    LibraryAbsent,
    /// `libnvcuvid.so.1` lacks an entry point this helper needs.
    SymbolMissing,
    /// An NVDEC or CUDA call returned this `CUresult`.
    Driver { code: u32 },
    /// The device or the stream's format is outside what this path decodes:
    /// 8-bit 4:2:0 H.264 with even display dimensions within the limit.
    Unsupported,
    /// The parser rejected the stream, or it produced no sequence.
    Stream { code: u32 },
    /// The frame callback asked to stop.
    Stopped,
    /// A callback panicked; decoding stopped.
    CallbackPanicked,
    /// The helper refused its arguments.
    Invalid,
}

impl fmt::Display for NvdecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryAbsent => {
                formatter.write_str("the NVIDIA video decoder library libnvcuvid.so.1 is absent")
            }
            Self::SymbolMissing => {
                formatter.write_str("libnvcuvid.so.1 lacks a required NVDEC entry point")
            }
            Self::Driver { code } => write!(formatter, "NVDEC call failed with CUresult {code}"),
            Self::Unsupported => formatter.write_str(
                "video is not 8-bit 4:2:0 H.264 with even dimensions within the decoder limit",
            ),
            Self::Stream { code } => {
                write!(
                    formatter,
                    "NVDEC rejected the H.264 stream (CUresult {code})"
                )
            }
            Self::Stopped => formatter.write_str("video decoding was stopped by its consumer"),
            Self::CallbackPanicked => formatter.write_str("a video frame callback panicked"),
            Self::Invalid => formatter.write_str("the NVDEC helper refused its arguments"),
        }
    }
}

impl Error for NvdecError {}

/// An Annex B H.264 stream split into access units, each with a timestamp
/// that comes back with the frame it produces.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnnexBStream {
    bytes: Vec<u8>,
    unit_ends: Vec<u64>,
    timestamps: Vec<i64>,
}

impl AnnexBStream {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends one access unit written by `write`. A failed write leaves the
    /// stream as it was.
    pub fn push_unit<E>(
        &mut self,
        timestamp: i64,
        write: impl FnOnce(&mut Vec<u8>) -> Result<(), E>,
    ) -> Result<(), E> {
        let start = self.bytes.len();
        if let Err(error) = write(&mut self.bytes) {
            self.bytes.truncate(start);
            return Err(error);
        }
        self.unit_ends.push(self.bytes.len() as u64);
        self.timestamps.push(timestamp);
        Ok(())
    }

    #[must_use]
    pub fn units(&self) -> usize {
        self.unit_ends.len()
    }

    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// One decoded frame, borrowed for the duration of the frame callback.
#[derive(Clone, Copy, Debug)]
pub struct DecodedFrame<'a> {
    /// The timestamp of the access unit that produced this frame.
    pub timestamp: i64,
    pub width: u32,
    pub height: u32,
    /// `height` rows of `width` luma bytes, then `height / 2` rows of `width`
    /// interleaved U and V bytes.
    pub nv12: &'a [u8],
}

/// `libnvcuvid` with a CUDA context, context lock and non-blocking stream of
/// its own on one device.
///
/// The context is never one the engine serves on, so NVDEC work and the
/// engine's context-wide waits never wait on each other: an engine
/// `cuCtxSynchronize` does not include a decode, and a decode's frame copy
/// does not queue behind engine kernels on the legacy stream. Waits block
/// rather than spin. Dropping the session destroys the context and unloads
/// the library. A session stays on the thread that opened it.
pub struct NvdecSession {
    handle: NonNull<c_void>,
}

impl NvdecSession {
    /// Loads `libnvcuvid.so.1` and creates the session's context on
    /// `device_ordinal`, current on this thread afterwards.
    pub fn open(device_ordinal: i32) -> Result<Self, NvdecError> {
        let mut handle = ptr::null_mut();
        let mut driver = 0;
        // SAFETY-REVIEWED: the helper writes one handle or null and a status through valid pointers.
        let status =
            unsafe { runtime_cuda_nvdec_session_open(device_ordinal, &mut handle, &mut driver) };
        match (status, NonNull::new(handle)) {
            (OK, Some(handle)) => Ok(Self { handle }),
            (OK, None) => Err(NvdecError::Invalid),
            (other, _) => Err(status_error(other, driver, false)),
        }
    }

    /// The largest 8-bit 4:2:0 H.264 frame the session's device decodes.
    pub fn h264_capacity(&self) -> Result<(u32, u32), NvdecError> {
        let (mut width, mut height, mut driver) = (0, 0, 0);
        // SAFETY-REVIEWED: the handle is live and every out pointer names a local.
        let status = unsafe {
            runtime_cuda_nvdec_h264_capacity(
                self.handle.as_ptr(),
                &mut width,
                &mut height,
                &mut driver,
            )
        };
        match status {
            OK => Ok((width, height)),
            other => Err(status_error(other, driver, false)),
        }
    }

    /// Decodes `stream` on the session's context.
    ///
    /// Frames arrive in display order. `select` sees each displayed frame's
    /// timestamp and returns whether to copy it to the host; `frame` receives
    /// each copied frame and returns `false` to stop. Streams larger than
    /// `max_width` x `max_height` are refused before any decode surface is
    /// created.
    pub fn decode_h264(
        &self,
        stream: &AnnexBStream,
        max_width: u32,
        max_height: u32,
        select: &mut dyn FnMut(i64) -> bool,
        frame: &mut dyn FnMut(DecodedFrame<'_>) -> bool,
    ) -> Result<(), NvdecError> {
        let mut callbacks = Callbacks {
            select,
            frame,
            panicked: false,
        };
        let mut driver = 0;
        // SAFETY-REVIEWED: the stream's three buffers outlive the call and agree in
        // length; the callbacks pointer names a local that outlives the call.
        let status = unsafe {
            runtime_cuda_nvdec_decode_h264(
                self.handle.as_ptr(),
                stream.bytes.as_ptr(),
                stream.unit_ends.as_ptr(),
                stream.timestamps.as_ptr(),
                stream.unit_ends.len() as u64,
                max_width,
                max_height,
                (&raw mut callbacks).cast(),
                select_frame,
                receive_frame,
                &mut driver,
            )
        };
        match status {
            OK => Ok(()),
            other => Err(status_error(other, driver, callbacks.panicked)),
        }
    }
}

impl Drop for NvdecSession {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: the handle came from runtime_cuda_nvdec_session_open and is closed once.
        unsafe { runtime_cuda_nvdec_session_close(self.handle.as_ptr()) }
    }
}

fn status_error(status: c_int, driver: c_uint, panicked: bool) -> NvdecError {
    match status {
        LIBRARY_ABSENT => NvdecError::LibraryAbsent,
        SYMBOL_MISSING => NvdecError::SymbolMissing,
        DRIVER => NvdecError::Driver { code: driver },
        UNSUPPORTED => NvdecError::Unsupported,
        STREAM => NvdecError::Stream { code: driver },
        SINK if panicked => NvdecError::CallbackPanicked,
        SINK => NvdecError::Stopped,
        _ => NvdecError::Invalid,
    }
}

struct Callbacks<'a> {
    select: &'a mut dyn FnMut(i64) -> bool,
    frame: &'a mut dyn FnMut(DecodedFrame<'_>) -> bool,
    panicked: bool,
}

extern "C" fn select_frame(user: *mut c_void, timestamp: i64) -> c_int {
    // SAFETY-REVIEWED: user is the Callbacks decode_h264 passed, live for the call.
    let callbacks = unsafe { &mut *user.cast::<Callbacks<'_>>() };
    if callbacks.panicked {
        // Hand the frame to receive_frame, which stops decoding.
        return 1;
    }
    match catch_unwind(AssertUnwindSafe(|| (callbacks.select)(timestamp))) {
        Ok(wanted) => c_int::from(wanted),
        Err(_) => {
            callbacks.panicked = true;
            1
        }
    }
}

extern "C" fn receive_frame(
    user: *mut c_void,
    timestamp: i64,
    width: u32,
    height: u32,
    nv12: *const u8,
) -> c_int {
    // SAFETY-REVIEWED: user is the Callbacks decode_h264 passed, live for the call.
    let callbacks = unsafe { &mut *user.cast::<Callbacks<'_>>() };
    if callbacks.panicked {
        return 1;
    }
    let bytes = width as usize * height as usize * 3 / 2;
    // SAFETY-REVIEWED: the helper passes a host buffer of exactly this many NV12 bytes.
    let nv12 = unsafe { std::slice::from_raw_parts(nv12, bytes) };
    let decoded = DecodedFrame {
        timestamp,
        width,
        height,
        nv12,
    };
    match catch_unwind(AssertUnwindSafe(|| (callbacks.frame)(decoded))) {
        Ok(true) => 0,
        Ok(false) => 1,
        Err(_) => {
            callbacks.panicked = true;
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_record_their_ends_and_roll_back_failed_writes() {
        let mut stream = AnnexBStream::new();
        stream
            .push_unit(7, |out| {
                out.extend_from_slice(&[0, 0, 0, 1, 0x65]);
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(
            stream.push_unit(8, |out| {
                out.push(9);
                Err("bad sample")
            }),
            Err("bad sample")
        );
        assert_eq!(stream.units(), 1);
        assert_eq!(stream.bytes(), [0, 0, 0, 1, 0x65]);
        assert_eq!(
            (stream.unit_ends.clone(), stream.timestamps.clone()),
            (vec![5], vec![7])
        );
    }

    #[cfg(feature = "fake-driver-tests")]
    #[test]
    fn a_host_without_the_decoder_library_is_refused_cleanly() {
        assert!(matches!(
            NvdecSession::open(0),
            Err(NvdecError::LibraryAbsent)
        ));
    }
}
