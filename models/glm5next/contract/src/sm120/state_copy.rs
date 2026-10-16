//! The SM120 entry of commit's copies of layered request state
//! (`crate::own_ops::state::StateCopy`, `bundle/cuda/state/copy.cuh`):
//! a verify item's kept checkpoint entry goes to its slot's entry in every
//! layer of a state buffer in one launch, rather than one host copy per
//! layer (`kda.conv` and `kda.state` over 34 layers, `indexer.tail` over the
//! sparse ones). The copy is byte for byte.

use std::path::PathBuf;

use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

/// The lock family of [`CUBIN`] (`bundle/kernels.lock.d`).
pub const FAMILY: &str = "state_copy";
/// The state copy cubin.
pub const CUBIN: &str = "state_copy_glm5next_sm120";
/// Its one entry.
pub const SYMBOL: &str = "chop_copy_layers_glm5next_v1";
/// Threads of a copy CTA.
pub const THREADS: u32 = 256;
/// Most CTAs of one layer; each thread strides over the rest.
pub const MAX_BLOCKS: u32 = 64;

/// One launch: `bytes` bytes in each of `layers` layers, the source's and
/// destination's layers `source_stride` and `destination_stride` bytes
/// apart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LayerCopy {
    pub bytes: u64,
    pub layers: u32,
    pub source_stride: u64,
    pub destination_stride: u64,
}

impl LayerCopy {
    /// One CTA row per layer, enough CTAs for one 16-byte word per thread,
    /// at most [`MAX_BLOCKS`].
    pub fn plan(&self) -> Result<LaunchPlan, Refusal> {
        if self.bytes == 0 || self.layers == 0 {
            return Err(Refusal("an empty state copy"));
        }
        let blocks = self
            .bytes
            .div_ceil(16 * u64::from(THREADS))
            .min(u64::from(MAX_BLOCKS));
        let plan = LaunchPlan {
            grid: [blocks as u32, self.layers, 1],
            block: [THREADS, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    /// The bytes from the source's first to its last layer's end.
    pub fn source_span(&self) -> u64 {
        u64::from(self.layers - 1) * self.source_stride + self.bytes
    }

    /// The bytes from the destination's first to its last layer's end.
    pub fn destination_span(&self) -> u64 {
        u64::from(self.layers - 1) * self.destination_stride + self.bytes
    }
}

/// The copy entry: source, destination, their layer strides and the bytes
/// per layer; the layers are the grid's rows.
pub fn entry() -> EntrySpec {
    let ptr = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
        name: name.into(),
        cxx_type: cxx_type.into(),
    };
    let u64 = |name: &str| EntryParam::U64 { name: name.into() };
    EntrySpec::new(
        SYMBOL.into(),
        vec![
            ptr("source", "const uint8_t"),
            ptr("destination", "uint8_t"),
            u64("source_stride"),
            u64("destination_stride"),
            u64("bytes"),
        ],
        "chop::glm5next::state::copy_layers(source, destination, source_stride, \
         destination_stride, bytes);"
            .into(),
        THREADS,
        1,
    )
}

/// [`CUBIN`]: the copy entry.
pub fn cubin_spec() -> CubinSpec {
    CubinSpec {
        name: CUBIN.into(),
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.."),
        include_dirs: vec!["models/glm5next/bundle/cuda".into()],
        headers: vec!["#include \"state/copy.cuh\"".into()],
        defines: vec![],
        entries: vec![entry()],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_plan_one_cta_row_per_layer() {
        let copy = |bytes, layers| LayerCopy {
            bytes,
            layers,
            source_stride: 1 << 20,
            destination_stride: 1 << 20,
        };
        let plan = copy(36_864, 34).plan().unwrap();
        assert_eq!(
            (plan.grid, plan.block, plan.pdl),
            ([9, 34, 1], [256, 1, 1], true)
        );
        assert_eq!(copy(1 << 20, 34).plan().unwrap().grid, [64, 34, 1]);
        assert_eq!(copy(3, 1).plan().unwrap().grid, [1, 1, 1]);
        assert!(copy(0, 1).plan().is_err());
        assert!(copy(16, 0).plan().is_err());
        assert_eq!(copy(16, 3).source_span(), 2 * (1 << 20) + 16);
    }
}
