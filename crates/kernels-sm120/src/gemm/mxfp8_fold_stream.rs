//! Packed MMA fragments and per-op persistent counters for one-launch folds.
use super::mxfp8_fold_ticket;
use kernel_build::{CubinSpec, EntryParam, EntrySpec};
use kernel_core::contract::{LaunchPlan, Refusal};
use kernel_core::mxfp8_gemm::Mxfp8FoldGemmQuery;

#[derive(Clone, Debug)]
pub struct Entry {
    pub ticket: mxfp8_fold_ticket::Entry,
    /// Qualification probe: force output tiles to finish before admission.
    pub defer_admission: bool,
}
impl Entry {
    pub fn new(
        query: Mxfp8FoldGemmQuery,
        symbol: impl Into<String>,
        paired: bool,
    ) -> Result<Self, Refusal> {
        let result = Self {
            ticket: mxfp8_fold_ticket::Entry::new(query, symbol, paired)?,
            defer_admission: false,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<(), Refusal> {
        self.ticket.validate()?;
        if self.ticket.query.n % 8 != 0
            || self.ticket.query.k % 64 != 0
            || self.ticket.prefetch_weights
        {
            return Err(Refusal(
                "packed folds require eight-column tiles and complete 64-K groups",
            ));
        }
        Ok(())
    }
    /// The transpose retains the original byte count, including a 64-K tail.
    pub fn packed_bytes(&self) -> u64 {
        let q = self.ticket.query;
        u64::from(q.groups) * u64::from(q.n) * u64::from(q.k)
    }
    pub fn state_bytes(&self, rows: u32) -> Result<u64, Refusal> {
        self.ticket.header(rows)
    }
    pub fn symbols(&self) -> [String; 2] {
        [
            format!("{}_prepare_v1", self.ticket.symbol),
            self.ticket.symbol.clone(),
        ]
    }
    pub fn initialize_plan(&self) -> LaunchPlan {
        LaunchPlan {
            grid: [
                (self.packed_bytes() / 4).div_ceil(128).min(1024) as u32,
                1,
                1,
            ],
            block: [128, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: false,
        }
    }
    pub fn specs(&self) -> Vec<EntrySpec> {
        let q = self.ticket.query;
        let [initialize, compute] = self.symbols();
        let ptr = |name: &str, ty: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: ty.into(),
        };
        let u32p = |name: &str| EntryParam::U32 { name: name.into() };
        let namespace = "chop::gemm::mxfp8_fold_stream";
        vec![
            EntrySpec::new(
                initialize,
                vec![
                    ptr("weights", "const uint8_t"),
                    ptr("weights2", "const uint8_t"),
                    ptr("packed", "uint8_t"),
                    ptr("packed2", "uint8_t"),
                    ptr("state", "uint32_t"),
                    u32p("state_words"),
                ],
                format!(
                    "{namespace}::initialize_task<{},{},{},{}>(blockIdx.x,gridDim.x,weights,weights2,packed,packed2,state,state_words);",
                    q.n, q.k, q.groups, self.ticket.paired
                ),
                128,
                1,
            ),
            EntrySpec::new(
                compute,
                vec![
                    ptr("activations", "const uint8_t"),
                    ptr("activation_scales", "const uint8_t"),
                    ptr("packed", "const uint8_t"),
                    ptr("weight_scales", "const uint8_t"),
                    ptr("packed2", "const uint8_t"),
                    ptr("weight_scales2", "const uint8_t"),
                    ptr("output", "void"),
                    ptr("output2", "void"),
                    ptr("workspace", "uint8_t"),
                    ptr("state", "uint32_t"),
                    ptr("live_rows", "const uint32_t"),
                    ptr("status", "uint32_t"),
                    EntryParam::U64 {
                        name: "status_row_offset".into(),
                    },
                    u32p("rows"),
                    u32p("folds_per_cta"),
                ],
                format!(
                    "{namespace}::entry<{},{},{},{},{},{},{},{},{}>(activations,activation_scales,packed,weight_scales,packed2,weight_scales2,output,output2,workspace,state,live_rows,status,status_row_offset,rows,folds_per_cta);",
                    q.n,
                    q.k,
                    q.groups,
                    q.activation_row_stride,
                    q.activation_group_stride,
                    self.ticket.tile_n(),
                    q.output == kernel_core::contract::DType::F32,
                    self.ticket.paired,
                    self.defer_admission
                ),
                self.ticket.tile_n() * 4,
                1,
            ),
        ]
    }
}
pub fn cubin_spec(name: &str, entries: &[Entry]) -> Result<CubinSpec, Refusal> {
    if entries.is_empty() {
        return Err(Refusal("packed fold cubin requires entries"));
    }
    let mut symbols = std::collections::BTreeSet::new();
    for e in entries {
        e.validate()?;
        for symbol in e.symbols() {
            if !symbols.insert(symbol) {
                return Err(Refusal("duplicate packed fold symbol"));
            }
        }
    }
    Ok(CubinSpec {
        name: name.into(),
        sources_root: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec!["#include \"gemm/mxfp8_fold_stream.cuh\"".into()],
        defines: vec![],
        entries: entries
            .iter()
            .flat_map(Entry::specs)
            .chain([prefetch_spec()])
            .collect(),
    })
}

pub const PREFETCH: &str = "chop_mxfp8_stream_prefetch_v1";
fn prefetch_spec() -> EntrySpec {
    let mut parameters = Vec::new();
    let mut setup = String::from("chop::gemm::PrefetchRanges ranges{};");
    for i in 0..4 {
        parameters.push(EntryParam::DevicePointer {
            name: format!("base{i}"),
            cxx_type: "const unsigned char".into(),
        });
        parameters.push(EntryParam::U64 {
            name: format!("bytes{i}"),
        });
        setup.push_str(&format!(
            "ranges.base[{i}]=base{i};ranges.bytes[{i}]=bytes{i};"
        ));
    }
    setup.push_str("chop::ptx::griddep_launch_dependents();chop::gemm::mxfp8_fold_stream::prefetch_task(ranges,blockIdx.x,gridDim.x);chop::ptx::griddep_wait();");
    EntrySpec::new(PREFETCH.into(), parameters, setup, 32, 1)
}
/// The ranges stay immutable until their later consumer completes.
pub fn prefetch_launch(
    ranges: [kernel_core::backend::View; 4],
) -> Result<kernel_core::backend::Launch, Refusal> {
    use kernel_core::contract::{Argument, ArgumentList};
    let mut pieces = 0_u64;
    let mut arguments = vec![];
    for range in ranges {
        if range.bytes % 16 != 0 {
            return Err(Refusal(
                "fold prefetch ranges require complete 16-byte pieces",
            ));
        }
        pieces = pieces
            .checked_add(range.bytes.div_ceil(16384))
            .ok_or(Refusal("fold prefetch extent overflows"))?;
        arguments.push(if range.bytes == 0 {
            Argument::NullDevicePointer
        } else {
            Argument::DeviceBuffer(range.handle)
        });
        arguments.push(Argument::U64(range.bytes));
    }
    Ok(kernel_core::backend::Launch {
        symbol: PREFETCH.into(),
        plan: LaunchPlan {
            grid: [
                pieces.div_ceil(32).clamp(1, u64::from(super::SM_COUNT)) as u32,
                1,
                1,
            ],
            block: [32, 1, 1],
            dynamic_shared_bytes: 0,
            workspace_bytes: 0,
            cluster: [1, 1, 1],
            pdl: true,
        },
        arguments: ArgumentList(arguments),
        rows: 0,
        collective: false,
        persistent: false,
    })
}
