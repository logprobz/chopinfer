//! Model-side storage and binding for exact device hashes, shared by AR and drafts.
use super::{Bind, Lowered};
use bundle_runtime::resources::{Held, allocate_span};
use bundle_runtime::upload::DeviceInputs;
use kernel_core::backend::{Counts, Need, Prepared, View};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::OwnOps;
use model_deepseek41_contract::own_ops::engram::HashTables;
use model_deepseek41_contract::own_ops::engram_hash::{EngramHash, EngramHashBuffers};
use runtime_bundle_abi::{
    BundleAbiError, BundleMemoryPoolHandleV1, BundleStreamHandleV1, DeviceBufferSpanV1 as Span,
    HostServicesV1,
};

pub(crate) struct HashDevice {
    pub shape: EngramHash,
    prepared: Prepared<EngramHashBuffers<View>>,
    constants: Vec<u8>,
}

#[derive(Clone, Copy)]
pub(crate) struct HashMemory {
    pub constants: Span,
    pub history: Span,
    pub text: Span,
    pub ids: Span,
}

impl HashDevice {
    pub fn new(own: &dyn OwnOps, g: &Geometry, items: u32, rows: u32) -> Result<Self, String> {
        let e = g.shape.engram;
        let shape = EngramHash {
            vocab: g.shape.vocab,
            ngram: e.max_ngram,
            heads: e.heads,
            layers: e.layers.len() as u32,
            items,
            stride: rows,
        };
        let prepared = own
            .engram_hash(&shape, Need::rows(rows))
            .map_err(|e| format!("Engram device hash: {e:?}"))?;
        let constants = HashTables::of(g)?
            .device_words()
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        Ok(Self {
            shape,
            prepared,
            constants,
        })
    }

    pub fn allocate(
        &self,
        services: &HostServicesV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        held: &mut Vec<Held>,
    ) -> Result<HashMemory, BundleAbiError> {
        let mut alloc = |bytes| allocate_span(services, pool, stream, bytes, held);
        let memory = HashMemory {
            constants: alloc(self.constants.len() as u64)?,
            history: alloc(u64::from(self.shape.items) * u64::from(self.shape.ngram - 1) * 8)?,
            text: alloc(u64::from(self.shape.stride))?,
            ids: alloc(self.shape.output_bytes())?,
        };
        services.copy_host_to_device(stream, memory.constants, &self.constants)?;
        services.zero_device_buffer(stream, memory.ids)?;
        Ok(memory)
    }

    pub fn launch(
        &self,
        memory: HashMemory,
        inputs: DeviceInputs,
        status: Span,
        rows: u32,
    ) -> Result<Vec<Lowered>, String> {
        let mut bind = Bind::default();
        let buffers = EngramHashBuffers {
            tokens: bind.view(inputs.token_ids),
            token_batch: bind.view(inputs.token_batch),
            q_start: bind.view(inputs.q_start),
            history: bind.view(memory.history),
            text: bind.view(memory.text),
            constants: bind.view(memory.constants),
            ids: bind.view(memory.ids),
            status: bind.view(status),
        };
        bind.launch(&self.prepared, &buffers, Counts::rows(rows))
            .map_err(|e| e.to_string())
    }
}
