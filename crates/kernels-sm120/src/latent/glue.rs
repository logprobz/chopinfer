//! Additive latent glue entries. The legacy latent cubins are unchanged.
use kernel_build::{CubinSpec, EntryParam as P, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal, Validate};
use kernel_core::properties::Invariance;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Kernel {
    QueryRope,
    WindowRotate,
    WindowStore,
    WindowStoreBf16,
    WindowKeys,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub kernel: Kernel,
    pub capacity: u32,
    pub symbol: String,
}

impl Entry {
    pub fn plan(&self, rows: u32, heads: u32) -> Result<LaunchPlan, Refusal> {
        if self.capacity == 0
            || self.capacity > crate::backend::MAX_CAPACITY
            || rows > self.capacity
            || heads == 0
        {
            return Err(Refusal("latent glue launch exceeds registered capacity"));
        }
        let plan = LaunchPlan {
            grid: [rows.max(1), 1, 1],
            block: [
                if self.kernel == Kernel::WindowKeys {
                    128
                } else {
                    256
                },
                1,
                1,
            ],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        };
        plan.validate()?;
        Ok(plan)
    }

    pub fn invariance(&self) -> Invariance {
        Invariance::any_rows(self.capacity)
    }

    pub fn spec(&self) -> EntrySpec {
        let ptr = |name: &str, ty: &str| P::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let u32_arg = |name: &str| P::U32 { name: name.into() };
        let u64_arg = |name: &str| P::U64 { name: name.into() };
        let (mut params, mut body) = match self.kernel {
            Kernel::QueryRope => (vec![
                ptr("values", "float"), ptr("native_q", "uint16_t"), ptr("cos_sin", "const float"),
                ptr("live_rows", "const uint32_t"), u32_arg("heads"),
            ], format!("chop::latent::glue::rotate<{}>(values,native_q,cos_sin,live_rows,heads);", self.kernel == Kernel::QueryRope)),
            Kernel::WindowRotate => (vec![
                ptr("input", "const float"), ptr("weight", "const uint16_t"),
                ptr("values", "float"), ptr("cos_sin", "const float"),
                ptr("live_rows", "const uint32_t"), u32_arg("eps_bits"),
            ], "chop::latent::glue::norm_rotate(input,weight,values,cos_sin,live_rows,__uint_as_float(eps_bits));".into()),
            Kernel::WindowStore | Kernel::WindowStoreBf16 => (vec![
                ptr("input_codes", "const uint8_t"), ptr("input_scales", "const uint8_t"),
                ptr("positions", "const int32_t"), ptr("row_batch", "const int32_t"),
                ptr("first_positions", "const int32_t"), ptr("last_positions", "const int32_t"),
                ptr("codes", "uint8_t"), ptr("scales", "uint8_t"), ptr("live_rows", "const uint32_t"),
                u32_arg("sequences"), u32_arg("slots"), u32_arg("circular"),
                u64_arg("code_stride"), u64_arg("scale_stride"),
            ], "chop::latent::glue::store(input_codes,input_scales,positions,row_batch,first_positions,last_positions,codes,scales,live_rows,sequences,slots,circular,code_stride,scale_stride);".into()),
            Kernel::WindowKeys => (vec![
                ptr("positions", "const int32_t"), ptr("row_batch", "const int32_t"),
                ptr("first_positions", "const int32_t"), ptr("output", "int32_t"),
                ptr("live_rows", "const uint32_t"), u32_arg("sequences"), u32_arg("window"),
                u32_arg("slots"), u32_arg("circular"),
            ], "chop::latent::glue::keys(positions,row_batch,first_positions,output,live_rows,sequences,window,slots,circular);".into()),
        };
        if self.kernel == Kernel::WindowStoreBf16 {
            body = body.replace("::store(", "::store_bf16<>(");
        }
        if self.kernel != Kernel::WindowKeys {
            params.extend([
                ptr(
                    "status",
                    if self.kernel == Kernel::WindowStore {
                        "const uint32_t"
                    } else {
                        "uint32_t"
                    },
                ),
                u64_arg("status_row_offset"),
            ]);
            body = body.replacen(");", ",status,status_row_offset);", 1);
        }
        EntrySpec::new(
            self.symbol.clone(),
            params,
            body,
            if self.kernel == Kernel::WindowKeys {
                128
            } else {
                256
            },
            1,
        )
    }
}

pub fn cubin_spec(name: &str, entries: &[Entry]) -> CubinSpec {
    CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"latent/glue.cuh\"".into()],
        defines: vec![],
        entries: entries.iter().map(Entry::spec).collect(),
    }
}

/// Lab controls. These symbols are never registered by a serving backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mutation {
    WrongTail,
    ReversedPair,
    SplitHalf,
    DoubleRound,
    WindowEdge,
    InputRound,
    DropQueryFlag,
    DropWindowFlag,
}
impl Mutation {
    pub const ALL: [Self; 8] = [
        Self::WrongTail,
        Self::ReversedPair,
        Self::SplitHalf,
        Self::DoubleRound,
        Self::WindowEdge,
        Self::InputRound,
        Self::DropQueryFlag,
        Self::DropWindowFlag,
    ];
    pub fn symbol(self) -> String {
        format!("chop_latent_glue_mutant_{self:?}_v1")
    }
    pub fn spec(self) -> EntrySpec {
        let kernel = match self {
            Self::WindowEdge => Kernel::WindowKeys,
            Self::InputRound | Self::DropWindowFlag => Kernel::WindowRotate,
            _ => Kernel::QueryRope,
        };
        let mut spec = Entry {
            kernel,
            capacity: 32,
            symbol: self.symbol(),
        }
        .spec();
        spec.body = match self {
            Self::WrongTail => spec.body.replace("rotate<true>", "rotate<true,1>"),
            Self::ReversedPair => spec.body.replace("rotate<true>", "rotate<true,2>"),
            Self::SplitHalf => spec.body.replace("rotate<true>", "rotate_split_half"),
            Self::DoubleRound => spec.body.replace("rotate<true>", "rotate<true,3>"),
            Self::WindowEdge => spec.body.replace("::keys(", "::keys<true>("),
            Self::DropQueryFlag => spec.body.replace(
                "rotate<true>",
                "rotate<true,0,chop::latent::glue::DropFlags>",
            ),
            Self::DropWindowFlag => spec.body.replace(
                "::norm_rotate(",
                "::norm_rotate<false,chop::latent::glue::DropFlags>(",
            ),
            Self::InputRound => spec.body.replace("::norm_rotate(", "::norm_rotate<true>("),
        };
        spec
    }
}
