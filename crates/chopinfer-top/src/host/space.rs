//! How full this host's local filesystems are, from `statvfs`.
//!
//! Sizes are read on a thread of their own (see [`super::poll`]), so a
//! read stuck on a failing disk delays only disk space, never the CPU,
//! memory and sensor readings.

use std::io;
use std::path::Path;

use super::proc;

/// What `statvfs` says about one filesystem, in its fragments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size {
    pub blocks: u64,
    pub free: u64,
    /// Free to unprivileged users: `free` less the reserved blocks.
    pub available: u64,
    pub fragment: u64,
}

/// Reads one filesystem's size; tests stub it.
pub type Statvfs = fn(&Path) -> io::Result<Size>;

/// `statvfs(2)`, through rustix, which needs no unsafe code here.
pub fn statvfs(path: &Path) -> io::Result<Size> {
    let size = rustix::fs::statvfs(path)?;
    Ok(Size {
        blocks: size.f_blocks,
        free: size.f_bfree,
        available: size.f_bavail,
        fragment: size.f_frsize,
    })
}

/// One filesystem, in bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filesystem {
    pub mount: String,
    pub used: u64,
    pub available: u64,
    pub total: u64,
}

impl Filesystem {
    pub fn new(mount: impl Into<String>, size: Size) -> Self {
        let bytes = |count: u64| count.saturating_mul(size.fragment);
        Self {
            mount: mount.into(),
            used: bytes(size.blocks.saturating_sub(size.free)),
            available: bytes(size.available),
            total: bytes(size.blocks),
        }
    }

    /// The share in use, as `df` reports it: against what users can fill,
    /// so a filesystem whose reserved blocks remain is 100% at its end.
    pub fn share(&self) -> f64 {
        let room = self.used.saturating_add(self.available);
        if room == 0 {
            0.0
        } else {
            self.used as f64 / room as f64
        }
    }
}

/// The size of every filesystem shown, from the mounts of `proc_root`. A
/// mount whose read fails is left out until the next round.
pub fn sizes(proc_root: &Path, statvfs: Statvfs) -> Vec<Filesystem> {
    let Ok(text) = std::fs::read_to_string(proc_root.join("self/mountinfo")) else {
        return Vec::new();
    };
    let mounts = proc::mountinfo(&text);
    proc::local(&mounts, |point| Path::new(point).is_dir())
        .into_iter()
        .filter_map(|point| {
            let size = statvfs(Path::new(&point)).ok()?;
            (size.blocks > 0).then(|| Filesystem::new(point, size))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_share_is_against_what_users_can_fill() {
        // 1,000 blocks of 4 KiB: 900 free, 850 of them available.
        let filesystem = Filesystem::new(
            "/data",
            Size {
                blocks: 1000,
                free: 900,
                available: 850,
                fragment: 4096,
            },
        );
        assert_eq!(
            (filesystem.used, filesystem.available, filesystem.total),
            (100 * 4096, 850 * 4096, 1000 * 4096)
        );
        assert!((filesystem.share() - 100.0 / 950.0).abs() < 1e-12);
        let empty = Filesystem::new("/x", Size::default());
        assert_eq!(empty.share(), 0.0);
    }

    #[test]
    fn statvfs_reads_a_real_directory() {
        let size = statvfs(&std::env::temp_dir()).unwrap();
        assert!(size.blocks > 0 && size.fragment > 0, "{size:?}");
        assert!(statvfs(Path::new("/nonexistent/chopinfer-top")).is_err());
    }
}
