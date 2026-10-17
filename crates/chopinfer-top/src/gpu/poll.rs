//! Reading every GPU once a poll, on a thread of its own.
//!
//! Each poll is about twenty NVML queries per GPU, which read driver state
//! in about 0.2 ms. The fan and ECC errors, at a millisecond each, are read
//! every [`SLOW_POLLS`] polls and carried between. GPM counters are sampled
//! only while asked for, which the interface does while the GPU view is
//! open; between two samples NVML works out how busy the SMs, tensor pipes
//! and DRAM were.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread;
use std::time::{Duration, Instant};

use super::nvml::{Nvml, Sample};
use super::owners::Owners;
use super::{Activity, Device, Reading};

/// Polls between readings of the slow queries.
pub const SLOW_POLLS: u64 = 5;

/// What the GPU poller reports.
#[derive(Clone, Debug, PartialEq)]
pub enum Report {
    /// NVML is open: the driver version and the GPUs.
    Opened {
        driver: Option<String>,
        devices: Vec<Device>,
    },
    /// NVML could not be opened, and why. Nothing more follows.
    Unavailable(String),
    /// GPU `index`, whose handle did not open, described again: it opens
    /// now, or fails another way.
    Reopened { index: usize, device: Device },
    /// One reading of every GPU, in device order.
    Polled(Vec<Reading>),
}

/// Takes readings of every GPU: the slow queries every few polls, GPM
/// figures between consecutive samples, and the owner of each process.
pub struct Sampler<'a> {
    nvml: &'a Nvml,
    devices: Vec<Device>,
    polls: u64,
    /// The last slow reading of each GPU, carried to the polls between.
    slow: Vec<Reading>,
    /// Two GPM sample buffers per GPU while counters are read, and whether
    /// the first holds the previous poll's sample.
    gpm: Vec<Option<(Sample<'a>, Sample<'a>, bool)>>,
    owners: Owners,
}

impl<'a> Sampler<'a> {
    pub fn new(nvml: &'a Nvml, devices: Vec<Device>, proc_root: impl Into<PathBuf>) -> Self {
        let count = devices.len();
        Self {
            nvml,
            devices,
            polls: 0,
            slow: vec![Reading::default(); count],
            gpm: (0..count).map(|_| None).collect(),
            owners: Owners::new(proc_root),
        }
    }

    /// Every [`SLOW_POLLS`] polls, looks up again each GPU whose handle did
    /// not open, and describes again those that changed.
    pub fn reopen(&mut self) -> Vec<(usize, Device)> {
        if self.polls == 0 || self.polls % SLOW_POLLS != 0 {
            return Vec::new();
        }
        let mut reopened = Vec::new();
        for index in 0..self.devices.len() {
            if self.devices[index].unreadable.is_some() && self.nvml.reopen(index) {
                self.devices[index] = self.nvml.device(index);
                reopened.push((index, self.devices[index].clone()));
            }
        }
        reopened
    }

    /// One reading of every GPU at `at` seconds, with GPM figures when
    /// `counters` asks for them. Turning counters off frees the samples.
    pub fn poll(&mut self, at: f64, counters: bool) -> Vec<Reading> {
        let mut readings = Vec::with_capacity(self.devices.len());
        let slow = self.polls % SLOW_POLLS == 0;
        self.polls += 1;
        for index in 0..self.devices.len() {
            let device = &self.devices[index];
            let mut reading = self.nvml.read(index, device, at, slow);
            if slow {
                self.slow[index] = reading.clone();
            } else {
                let last = &self.slow[index];
                reading.fan = last.fan;
                reading.ecc_uncorrected = last.ecc_uncorrected;
                if device.temperature_limit.is_none() {
                    reading.headroom = last.headroom;
                }
            }
            if counters && self.devices[index].gpm && !reading.lost {
                reading.activity = self.activity(index);
            } else {
                self.gpm[index] = None;
            }
            for process in reading.processes.iter_mut().flatten() {
                process.owner = self.owners.resolve(process.pid);
            }
            readings.push(reading);
        }
        let pids = readings
            .iter()
            .flat_map(|reading| reading.processes.iter().flatten())
            .map(|process| process.pid)
            .collect::<Vec<_>>();
        self.owners.keep(&pids);
        readings
    }

    /// Takes a GPM sample of GPU `index` and returns the figures since the
    /// previous one.
    fn activity(&mut self, index: usize) -> Option<Activity> {
        if self.gpm[index].is_none() {
            self.gpm[index] = Some((self.nvml.sample()?, self.nvml.sample()?, false));
        }
        let (previous, current, valid) = self.gpm[index].as_mut()?;
        if !self.nvml.take(index, current) {
            *valid = false;
            return None;
        }
        let activity = if *valid {
            self.nvml.activity(previous, current)
        } else {
            None
        };
        std::mem::swap(previous, current);
        *valid = true;
        activity
    }
}

/// Opens NVML and the GPUs it lists.
pub fn open() -> Result<(Nvml, Option<String>, Vec<Device>), String> {
    let nvml = Nvml::open()?;
    let devices = (0..nvml.count()).map(|index| nvml.device(index)).collect();
    let driver = nvml.driver();
    Ok((nvml, driver, devices))
}

/// The GPU poller thread. Dropping it stops the thread.
#[derive(Debug)]
pub struct GpuPoller {
    stop: Sender<()>,
}

impl GpuPoller {
    /// Reads every GPU every `interval` seconds and sends the reports to
    /// `results`. GPM counters are sampled while `counters` is set.
    pub fn spawn(
        interval: f64,
        origin: Instant,
        counters: Arc<AtomicBool>,
        results: Sender<Report>,
    ) -> Self {
        let (stop, stopped) = mpsc::channel();
        thread::spawn(move || {
            let (nvml, driver, devices) = match open() {
                Ok(opened) => opened,
                Err(reason) => {
                    let _ = results.send(Report::Unavailable(reason));
                    return;
                }
            };
            if results
                .send(Report::Opened {
                    driver,
                    devices: devices.clone(),
                })
                .is_err()
            {
                return;
            }
            let mut sampler = Sampler::new(&nvml, devices, "/proc");
            loop {
                for (index, device) in sampler.reopen() {
                    if results.send(Report::Reopened { index, device }).is_err() {
                        return;
                    }
                }
                let at = origin.elapsed().as_secs_f64();
                let readings = sampler.poll(at, counters.load(Ordering::Relaxed));
                if results.send(Report::Polled(readings)).is_err() {
                    return;
                }
                match stopped.recv_timeout(Duration::from_secs_f64(interval)) {
                    Err(RecvTimeoutError::Timeout) => {}
                    _ => return,
                }
            }
        });
        Self { stop }
    }
}

impl Drop for GpuPoller {
    fn drop(&mut self) {
        let _ = self.stop.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads this host's GPUs; run with `--ignored` on a GPU host and
    /// compare with nvidia-smi. With `CHOPINFER_TOP_GPM=1` it also samples
    /// GPM counters.
    #[test]
    #[ignore = "needs an NVIDIA driver"]
    fn reads_this_hosts_gpus() {
        let (nvml, driver, devices) = open().unwrap();
        println!("driver {driver:?}");
        for device in &devices {
            println!("{device:?}");
        }
        let counters = std::env::var("CHOPINFER_TOP_GPM").is_ok_and(|value| value == "1");
        let mut sampler = Sampler::new(&nvml, devices, "/proc");
        let origin = Instant::now();
        for _ in 0..3 {
            for reading in sampler.poll(origin.elapsed().as_secs_f64(), counters) {
                println!("{reading:?}");
            }
            thread::sleep(Duration::from_secs(1));
        }
    }
}
