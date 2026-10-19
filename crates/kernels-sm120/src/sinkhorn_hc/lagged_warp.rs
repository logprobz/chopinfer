//! Typed serving access to a qualified, unchanged plain warp entry.
use super::mixer_probe;
use kernel_core::contract::{Argument, ArgumentList, Refusal};

#[derive(Clone, Debug)]
pub struct Entry {
    pub source: mixer_probe::Entry,
    pub buckets: Vec<u32>,
}
impl Entry {
    pub fn capacity(&self) -> Result<u32, Refusal> {
        self.source.validate()?;
        if self.buckets.is_empty()
            || self.buckets[0] == 0
            || self.buckets.windows(2).any(|p| p[0] >= p[1])
            || *self.buckets.last().unwrap() > self.source.reference.capacity()?
        {
            return Err(Refusal(
                "warp mixer needs sorted, unique qualified row buckets",
            ));
        }
        Ok(*self.buckets.last().unwrap())
    }
    pub fn symbol(&self) -> String {
        self.source.finalizer_symbols()[2].clone()
    }
    pub fn pack_arguments(&self, values: Vec<Argument>) -> Result<ArgumentList, Refusal> {
        self.capacity()?;
        let mut arguments = self.source.reference.pack_arguments(values)?;
        arguments.0.push(Argument::NullDevicePointer);
        Ok(arguments)
    }
}
