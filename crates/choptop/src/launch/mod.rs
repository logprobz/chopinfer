//! Starting serve processes from the loader: who holds each GPU and port,
//! where each instance goes, and the command line each one runs.
//!
//! The launches themselves, their records and stopping build on these: a
//! launch that is queued or starting is a [`Reservation`], which the
//! occupancy and the placement read.

pub mod command;
pub mod occupancy;
pub mod placement;

/// A choptop launch that holds its GPUs and port from the moment it is
/// queued until it is cancelled or its process exits, including while it
/// prepares its package and shows neither a GPU process nor a listener.
#[derive(Clone, Debug, PartialEq)]
pub struct Reservation {
    /// The row it shows as: `launch A` until it has a pid, `proc PID` after.
    pub label: String,
    /// What it serves, for the loader's placement rows.
    pub model: String,
    /// The UUIDs of its GPUs.
    pub gpus: Vec<String>,
    pub port: u16,
    /// Its process once started, by pid and start ticks; `None` while it
    /// waits for what it replaces to exit.
    pub process: Option<(u32, u64)>,
}

impl Reservation {
    /// Whether it has not started yet, so a switch cancels it rather than
    /// stopping it.
    pub fn waiting(&self) -> bool {
        self.process.is_none()
    }
}
