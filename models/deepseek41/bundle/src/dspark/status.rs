//! Preserve status before scratch reuse and validate it at the output boundary.
use std::cell::{Cell, RefCell};

use runtime_bundle_abi::{BundleStreamHandleV1, DeviceBufferSpanV1, HostServicesV1};

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Quantized,
    Index(u32),
}

pub(crate) struct Status {
    buffer: DeviceBufferSpanV1,
    bytes: Cell<u64>,
    parts: RefCell<Vec<(u64, Kind)>>,
}

impl Status {
    pub fn new(buffer: DeviceBufferSpanV1) -> Self {
        Self {
            buffer,
            bytes: Cell::new(0),
            parts: RefCell::new(Vec::new()),
        }
    }

    pub fn save(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        source: DeviceBufferSpanV1,
        kind: Kind,
    ) -> Result<(), String> {
        let at = self.bytes.get();
        let destination = super::device::slice(self.buffer, at, source.length)?;
        services
            .copy_device_to_device(stream, destination, source)
            .map_err(|e| e.to_string())?;
        self.bytes.set(at + source.length);
        self.parts.borrow_mut().push((source.length, kind));
        Ok(())
    }

    pub fn check(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        rank: usize,
    ) -> Result<(), String> {
        let length = self.bytes.get();
        if length == 0 {
            return Ok(());
        }
        let source = super::device::slice(self.buffer, 0, length)?;
        let mut bytes = vec![0; length as usize];
        services
            .copy_device_to_host(stream, &mut bytes, source)
            .map_err(|e| e.to_string())?;
        self.validate(rank, &bytes)
    }

    pub fn pending(&self) -> Result<DeviceBufferSpanV1, String> {
        super::device::slice(self.buffer, 0, self.bytes.get())
    }

    pub fn validate(&self, rank: usize, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() as u64 != self.bytes.get() {
            return Err("DSpark saved status length differs".into());
        }
        let mut at = 0;
        for &(length, kind) in self.parts.borrow().iter() {
            let end = at + length as usize;
            if let Some(row) = bytes[at..end]
                .chunks_exact(4)
                .position(|word| word != [0; 4])
            {
                return Err(match kind {
                    Kind::Quantized => {
                        format!("DSpark quantized input on rank {rank} is invalid in row {row}")
                    }
                    Kind::Index(layer) => format!(
                        "DSpark index score on rank {rank}, layer {layer}, row {row} is invalid"
                    ),
                });
            }
            at = end;
        }
        self.parts.borrow_mut().clear();
        self.bytes.set(0);
        Ok(())
    }
}
