//! Shared cache preparation for cold-weight measurements.
use crate::{LabResult, Session, launch};
use runtime_cuda::{DeviceBuffer, KernelArgument, LaunchConfig, Module, Stream};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionMode {
    Write,
    Read,
}
pub const DEFAULT_EVICTION: EvictionMode = EvictionMode::Read;
impl EvictionMode {
    pub fn selected() -> LabResult<Self> {
        match std::env::var("CHOPINFER_KERNEL_EVICTION").as_deref() {
            Err(std::env::VarError::NotPresent) => Ok(DEFAULT_EVICTION),
            Ok("write") => Ok(Self::Write),
            Ok("read") => Ok(Self::Read),
            _ => Err("CHOPINFER_KERNEL_EVICTION must be write or read".into()),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Write => "write",
            Self::Read => "read",
        }
    }
}
fn sweep_bytes(l2: usize) -> LabResult<usize> {
    l2.checked_mul(2)
        .map(|bytes| bytes.max(64 << 20))
        .ok_or("L2 size overflow".into())
}
pub struct Eviction {
    module: Module,
    buffer: DeviceBuffer,
    checksum: DeviceBuffer,
    pub mode: EvictionMode,
    pub bytes: usize,
}
impl Eviction {
    pub fn new(session: &Session, mode: EvictionMode) -> LabResult<Self> {
        let bytes = match mode {
            EvictionMode::Write => 256 << 20,
            EvictionMode::Read => sweep_bytes(crate::gpu_state::l2_bytes(session.ordinal)?)?,
        };
        let result = Self {
            module: session.module("probe_sm120")?,
            buffer: session.allocate(bytes)?,
            checksum: session.allocate(4)?,
            mode,
            bytes,
        };
        // Initialize once, then retire the writes before any measured round.
        result.write(&session.stream)?;
        result
            .checksum
            .copy_from_bytes(&0u32.to_ne_bytes(), &session.stream)
            .map_err(|e| e.to_string())?;
        session.stream.synchronize().map_err(|e| e.to_string())?;
        Ok(result)
    }
    fn write(&self, stream: &Stream) -> LabResult<()> {
        self.module
            .function("chop_probe_flush_v1")
            .map_err(|e| e.to_string())?
            .launch(
                stream,
                launch::simple_config(
                    (1024, 1, 1),
                    (256, 1, 1),
                    0,
                    LaunchConfig::NO_CLUSTER,
                    false,
                )?,
                &[
                    KernelArgument::Buffer(&self.buffer),
                    KernelArgument::U64((self.bytes / 4) as u64),
                ],
            )
            .map_err(|e| e.to_string())
    }
    pub fn run(&self, stream: &Stream) -> LabResult<()> {
        match self.mode {
            EvictionMode::Write => self.write(stream),
            EvictionMode::Read => self
                .module
                .function("chop_probe_read_v1")
                .map_err(|e| e.to_string())?
                .launch(
                    stream,
                    launch::simple_config(
                        (1024, 1, 1),
                        (256, 1, 1),
                        0,
                        LaunchConfig::NO_CLUSTER,
                        false,
                    )?,
                    &[
                        KernelArgument::Buffer(&self.buffer),
                        KernelArgument::Buffer(&self.checksum),
                        KernelArgument::U64((self.bytes / 4) as u64),
                        KernelArgument::U32(1),
                    ],
                )
                .map_err(|e| e.to_string()),
        }
    }
}
/// Retain this object until the stream is idle.
pub fn evict(session: &Session) -> LabResult<Eviction> {
    let eviction = Eviction::new(session, EvictionMode::selected()?)?;
    eviction.run(&session.stream)?;
    Ok(eviction)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sweep_covers_twice_l2_with_a_floor() {
        assert_eq!(sweep_bytes(0).unwrap(), 64 << 20);
        assert_eq!(sweep_bytes(128 << 20).unwrap(), 256 << 20);
        assert!(sweep_bytes(usize::MAX).is_err());
    }
}
