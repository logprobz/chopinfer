//! Bitwise prefix gathers and isolated numerical controls.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    WindowStart,
    PlaneStride,
    StaleSlot,
    UnclampedStart,
}
impl Mutation {
    pub const ALL: [Self; 4] = [
        Self::WindowStart,
        Self::PlaneStride,
        Self::StaleSlot,
        Self::UnclampedStart,
    ];
    fn code(self) -> u32 {
        match self {
            Self::WindowStart => 1,
            Self::PlaneStride => 2,
            Self::StaleSlot => 3,
            Self::UnclampedStart => 4,
        }
    }
    pub fn symbol(self) -> String {
        format!("chop_window_stage_mutant_{self:?}_v1")
    }
}
#[derive(Clone, Debug)]
pub struct Entry {
    pub kv: kernel_core::latent_attention::LatentKvFormat,
    pub capacity: u32,
    pub symbol: String,
    pub mutation: Option<Mutation>,
}
impl Entry {
    pub fn plan(&self, items: u32, window: u32) -> Result<LaunchPlan, Refusal> {
        if self.capacity == 0
            || self.capacity > crate::backend::MAX_CAPACITY
            || items > self.capacity
            || !(2..=1024).contains(&window)
        {
            return Err(Refusal("window stage launch exceeds registered geometry"));
        }
        let plan = LaunchPlan {
            grid: [items.max(1), window - 1, 1],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }
    pub fn spec(&self) -> EntrySpec {
        let ptr = |name: &str, ty: &str| P::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let mut params = vec![
            ptr("ring_codes", "const uint8_t"),
            ptr("ring_scales", "const uint8_t"),
            ptr("item_slots", "const int32_t"),
            ptr("first_positions", "const int32_t"),
            ptr("last_positions", "const int32_t"),
            ptr("codes", "uint8_t"),
            ptr("scales", "uint8_t"),
            ptr("live_items", "const uint32_t"),
        ];
        params.extend(
            ["items", "sequences", "window", "ring_slots"].map(|name| P::U32 { name: name.into() }),
        );
        params.extend(
            ["read_codes", "read_scales", "write_codes", "write_scales"]
                .map(|name| P::U64 { name: name.into() }),
        );
        EntrySpec::new(
            self.symbol.clone(),
            params,
            format!(
                "chop::latent::{}<{}>(ring_codes,ring_scales,item_slots,first_positions,last_positions,codes,scales,live_items,items,sequences,window,ring_slots,read_codes,read_scales,write_codes,write_scales);",
                if self.kv == kernel_core::latent_attention::LatentKvFormat::Bf16 {
                    "window_stage_bf16"
                } else {
                    "window_stage"
                },
                self.mutation.map_or(0, Mutation::code)
            ),
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
        headers: vec!["#include \"latent/window_stage.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}
