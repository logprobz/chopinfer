//! Retained-bound-aware staged window lists.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
#[derive(Clone, Debug)]
pub struct Entry {
    pub capacity: u32,
    pub symbol: String,
}
impl Entry {
    pub fn plan(&self, rows: u32) -> Result<LaunchPlan, Refusal> {
        if self.capacity == 0
            || self.capacity > crate::backend::MAX_CAPACITY
            || rows > self.capacity
        {
            return Err(Refusal("staged window lists exceed registered rows"));
        }
        let p = LaunchPlan {
            grid: [rows.max(1), 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        p.validate()?;
        Ok(p)
    }
    pub fn spec(&self) -> EntrySpec {
        let ptr = |name: &str, ty: &str| P::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let mut args = [
            "positions",
            "row_items",
            "chunk_first",
            "chunk_last",
            "item_slots",
            "ring_last",
        ]
        .map(|n| ptr(n, "const int32_t"))
        .to_vec();
        args.extend([ptr("output", "int32_t"), ptr("live_rows", "const uint32_t")]);
        args.extend(
            ["items", "sequences", "window", "ring_slots", "staged_slots"]
                .map(|n| P::U32 { name: n.into() }),
        );
        EntrySpec::new(
            self.symbol.clone(),
            args,
            "chop::latent::staged_window_keys(positions,row_items,chunk_first,chunk_last,item_slots,ring_last,output,live_rows,items,sequences,window,ring_slots,staged_slots);".into(),
            128,
            1,
        )
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"latent/staged_window_keys.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}
