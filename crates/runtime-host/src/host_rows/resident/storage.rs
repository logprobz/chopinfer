//! Owned anonymous memory, locked for the lifetime of a resident cache shard.
// SAFETY-REVIEWED: this module owns the mmap, mlock, munlock and munmap calls.
#![allow(unsafe_code)]

use std::io;

pub(super) struct LockedBytes {
    pointer: *mut u8,
    length: usize,
}

// SAFETY-REVIEWED: ownership may move between threads; mutation requires &mut.
unsafe impl Send for LockedBytes {}

impl LockedBytes {
    pub(super) fn new(length: usize) -> io::Result<Self> {
        // SAFETY-REVIEWED: a fresh private anonymous mapping is owned here.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        let memory = Self {
            pointer: pointer.cast(),
            length,
        };
        // SAFETY-REVIEWED: this is exactly the newly owned mapping. mlock also
        // populates its pages; failure drops the mapping without publishing it.
        if unsafe { libc::mlock(pointer, length) } != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(memory)
    }

    pub(super) fn bytes(&mut self) -> &mut [u8] {
        // SAFETY-REVIEWED: &mut self uniquely owns these writable mapped bytes.
        unsafe { std::slice::from_raw_parts_mut(self.pointer, self.length) }
    }
}

impl Drop for LockedBytes {
    fn drop(&mut self) {
        // SAFETY-REVIEWED: both operations cover exactly the owned mapping.
        unsafe {
            libc::munlock(self.pointer.cast(), self.length);
            libc::munmap(self.pointer.cast(), self.length);
        }
    }
}
