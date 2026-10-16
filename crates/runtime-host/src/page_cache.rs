//! Page-cache advice for checkpoint ranges read by a direct load.

// SAFETY-REVIEWED: this module contains the one posix_fadvise call.
#![allow(unsafe_code)]

use std::fs::File;
use std::io;

/// What to tell the kernel about a byte range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CacheAdviceV1 {
    /// Start reading the range into the page cache (`POSIX_FADV_WILLNEED`).
    WillNeed,
    /// Drop the range's clean pages (`POSIX_FADV_DONTNEED`).
    DontNeed,
}

/// Advises the kernel about `length` bytes at `offset` of `file`; a zero
/// length means to the end of the file. File contents never change.
pub fn advise_v1(file: &File, offset: u64, length: u64, advice: CacheAdviceV1) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let advice = match advice {
            CacheAdviceV1::WillNeed => libc::POSIX_FADV_WILLNEED,
            CacheAdviceV1::DontNeed => libc::POSIX_FADV_DONTNEED,
        };
        let (Ok(offset), Ok(length)) = (i64::try_from(offset), i64::try_from(length)) else {
            return Err(io::Error::other(
                "advised range exceeds the file offset type",
            ));
        };
        let descriptor = file.as_raw_fd();
        // SAFETY-REVIEWED: posix_fadvise reads only integers and a descriptor `file` keeps open.
        let status = unsafe { libc::posix_fadvise(descriptor, offset, length, advice) };
        if status != 0 {
            return Err(io::Error::from_raw_os_error(status));
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (file, offset, length, advice);
    Ok(())
}

/// How many of the pages `length` bytes at `offset` of `file` touch are in
/// the page cache now, and how many it touches, from the `cachestat` system
/// call (Linux 6.5). A kernel without it, or another platform, answers
/// `Unsupported`.
pub fn cached_pages_v1(file: &File, offset: u64, length: u64) -> io::Result<(u64, u64)> {
    if length == 0 {
        return Ok((0, 0));
    }
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        /// `__NR_cachestat`, the same on every Linux architecture.
        const SYS_CACHESTAT: libc::c_long = 451;
        #[repr(C)]
        struct Range {
            offset: u64,
            length: u64,
        }
        #[repr(C)]
        #[derive(Default)]
        struct Stat {
            cache: u64,
            dirty: u64,
            writeback: u64,
            evicted: u64,
            recently_evicted: u64,
        }
        let range = Range { offset, length };
        let mut stat = Stat::default();
        let descriptor = file.as_raw_fd();
        // SAFETY-REVIEWED: cachestat reads `range` and writes `stat`, both
        // live repr(C) values of the kernel's layout, for a descriptor
        // `file` keeps open; flags must be zero.
        let status = unsafe {
            libc::syscall(
                SYS_CACHESTAT,
                descriptor,
                &raw const range,
                &raw mut stat,
                0_u32,
            )
        };
        if status != 0 {
            let error = io::Error::last_os_error();
            return Err(if error.raw_os_error() == Some(libc::ENOSYS) {
                io::Error::from(io::ErrorKind::Unsupported)
            } else {
                error
            });
        }
        // The kernel counts the pages the byte range touches. On a host with
        // larger pages this overcounts them, and ranges look less cached.
        const PAGE: u64 = 4096;
        let pages = (offset + length - 1) / PAGE - offset / PAGE + 1;
        Ok((stat.cache.min(pages), pages))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (file, offset);
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

/// The alignment direct I/O on `file` needs, from `statx` with
/// `STATX_DIOALIGN` (Linux 6.1): the larger of its memory and offset
/// alignments, or `None` when the file does not support direct I/O. A kernel
/// that does not report it, or another platform, answers `Unsupported`.
pub fn direct_io_alignment_v1(file: &File) -> io::Result<Option<u64>> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let descriptor = file.as_raw_fd();
        // SAFETY-REVIEWED: statx with AT_EMPTY_PATH reads the empty C string
        // and fills `status`, a zeroed plain-data struct of the kernel's
        // layout, for a descriptor `file` keeps open.
        let (result, status) = unsafe {
            let mut status: libc::statx = std::mem::zeroed();
            let result = libc::statx(
                descriptor,
                c"".as_ptr(),
                libc::AT_EMPTY_PATH,
                libc::STATX_DIOALIGN,
                &raw mut status,
            );
            (result, status)
        };
        if result != 0 {
            let error = io::Error::last_os_error();
            return Err(if error.raw_os_error() == Some(libc::ENOSYS) {
                io::Error::from(io::ErrorKind::Unsupported)
            } else {
                error
            });
        }
        if status.stx_mask & libc::STATX_DIOALIGN == 0 {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        let (memory, offset) = (status.stx_dio_mem_align, status.stx_dio_offset_align);
        Ok((memory != 0 && offset != 0).then(|| u64::from(memory.max(offset))))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = file;
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}
