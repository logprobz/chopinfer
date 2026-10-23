//! Qualification of the native MoE prefill tiles (../../../native_moe_prefill.md):
//! the instruction probe, f64 oracle bounds, serving-entry and tile
//! identity, invariance, negative controls and paired timing, over
//! synthetic and captured operands. GPU tests are ignored by default and
//! read the staged lab cubin named by CHOPINFER_DEEPSEEK41_NATIVE_MOE_CUBINS.
#[path = "../../../sm120/native_moe.rs"]
#[allow(dead_code)]
mod family;
#[path = "../../../../bundle/src/sm120/moe.rs"]
#[allow(dead_code)]
mod registration;
#[path = "../../../sm120/native_tune.rs"]
#[allow(dead_code)]
mod tune_family;

mod captured;
mod checks;
mod oracle;
mod pipeline;
mod probe;
mod synthetic;
mod timing;
mod tune;

use kernel_core::backend::{Backend, Counts, Launch, Need, View};
use kernel_core::gated_mlp::GatedMlpProjection;
use kernel_core::moe_experts::{GatedMlpBuffers, PreparedExperts};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_harness::launch::{Bindings, launch};
use kernel_harness::{LabResult, Session};
use kernels_sm120::backend::Sm120;
use kernels_sm120::moe::native_grouped::{self as native, Shape, Tile, Variant};
use model_deepseek41_contract::{geometry::FLASH, moe};
use runtime_cuda::{DeviceBuffer, Function, Module, Stream};
use std::collections::BTreeMap;

pub const CUBINS: &str = "CHOPINFER_DEEPSEEK41_NATIVE_MOE_CUBINS";

/// Device buffers by handle, or handles alone on CPU.
pub struct Arena<'a> {
    pub session: Option<&'a Session>,
    pub next: u64,
    pub buffers: BTreeMap<u64, DeviceBuffer>,
}
impl<'a> Arena<'a> {
    pub fn new(session: Option<&'a Session>) -> Self {
        Self {
            session,
            next: 0,
            buffers: BTreeMap::new(),
        }
    }
    pub fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        self.next += 1;
        if let Some(session) = self.session {
            let b = session.allocate(bytes.len().max(1))?;
            if !bytes.is_empty() {
                b.copy_from_bytes(bytes, &session.stream)
                    .map_err(|e| e.to_string())?;
            }
            self.buffers.insert(self.next, b);
        }
        Ok(View {
            handle: self.next,
            bytes: bytes.len() as u64,
        })
    }
    pub fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        if self.session.is_none() {
            self.next += 1;
            return Ok(View {
                handle: self.next,
                bytes,
            });
        }
        self.bytes(&vec![value; bytes as usize])
    }
    pub fn read(&self, v: View) -> LabResult<Vec<u8>> {
        let mut out = vec![0; v.bytes as usize];
        self.buffers[&v.handle]
            .copy_to_bytes(&mut out, &self.session.unwrap().stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
    pub fn write(&self, v: View, bytes: &[u8]) -> LabResult<()> {
        self.buffers[&v.handle]
            .copy_from_bytes(bytes, &self.session.unwrap().stream)
            .map_err(|e| e.to_string())
    }
    pub fn poison(&self, views: &[View]) -> LabResult<()> {
        for v in views {
            self.write(*v, &vec![255; v.bytes as usize])?;
        }
        Ok(())
    }
}

/// Resolves serving symbols from the bundle's embedded cubins (what an
/// instance launches) and lab-only symbols from the staged lab cubin.
pub struct Runner<'a> {
    pub session: &'a Session,
    lab: Module,
    /// The tuning cubin, when staged beside the lab cubin.
    tune: Option<Module>,
    /// The tuning cubin's symbols, which resolve only there.
    tune_symbols: std::collections::BTreeSet<String>,
    embedded: BTreeMap<&'static str, Module>,
    functions: BTreeMap<String, Function>,
}
impl<'a> Runner<'a> {
    pub fn new(session: &'a Session) -> LabResult<Self> {
        let dir =
            std::path::PathBuf::from(std::env::var(CUBINS).map_err(|e| format!("{CUBINS}: {e}"))?);
        let bytes = std::fs::read(dir.join(format!("{}.cubin", family::CUBIN)))
            .map_err(|e| e.to_string())?;
        let tune = dir.join(format!("{}.cubin", tune_family::CUBIN));
        let tune = if tune.exists() {
            Some(session.load_cubin(&std::fs::read(tune).map_err(|e| e.to_string())?)?)
        } else {
            None
        };
        Ok(Self {
            session,
            lab: session.load_cubin(&bytes)?,
            tune,
            tune_symbols: tune_family::cubins()
                .into_iter()
                .flat_map(|s| s.entries.into_iter().map(|e| e.symbol))
                .collect(),
            embedded: BTreeMap::new(),
            functions: BTreeMap::new(),
        })
    }
    /// Whether `symbol` runs from the serving bundle rather than the lab.
    pub fn serving(symbol: &str) -> bool {
        bundle_deepseek41::sm120::fold_cubin_for_symbol(symbol).is_some()
    }
    pub fn ensure(&mut self, ops: &[Launch]) -> LabResult<()> {
        for op in ops {
            if self.functions.contains_key(&op.symbol) {
                continue;
            }
            let function = if let Some((name, bytes)) =
                bundle_deepseek41::sm120::fold_cubin_for_symbol(&op.symbol)
            {
                if !self.embedded.contains_key(name) {
                    self.embedded.insert(name, self.session.load_cubin(bytes)?);
                }
                self.embedded[name].function(&op.symbol)
            } else if self.tune_symbols.contains(&op.symbol) {
                // Looked up by name in the cubin that defines it, so no
                // driver lookup misses (the sanitizer reports those).
                self.tune
                    .as_ref()
                    .ok_or_else(|| format!("{}: the tuning cubin is not staged", op.symbol))?
                    .function(&op.symbol)
            } else {
                self.lab.function(&op.symbol)
            }
            .map_err(|e| format!("{}: {e}", op.symbol))?;
            if op.plan.dynamic_shared_bytes > 0 {
                function
                    .set_max_dynamic_shared_memory(op.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            self.functions.insert(op.symbol.clone(), function);
        }
        Ok(())
    }
    pub fn issue(&self, a: &Arena<'_>, ops: &[Launch], stream: &Stream) -> LabResult<()> {
        let bindings = Bindings {
            buffers: a.buffers.iter().map(|(&h, b)| (h, b)).collect(),
        };
        for op in ops {
            launch(
                &self.session.context,
                &self.functions[&op.symbol],
                stream,
                op.plan,
                &op.arguments,
                &bindings,
            )?;
        }
        Ok(())
    }
    pub fn run(&mut self, a: &Arena<'_>, ops: &[Launch]) -> LabResult<()> {
        self.ensure(ops)?;
        self.issue(a, ops, &self.session.stream)?;
        self.session.stream.synchronize().map_err(|e| e.to_string())
    }
}

pub fn usage(capacity: u32) -> OperationUse {
    OperationUse {
        phase: ExecutionPhase::Prefill,
        capacity,
        graph: false,
    }
}

/// The serving op of `kind` for a prefill step of `rows` at `capacity`,
/// as `ops::moe` prepares it: exact buckets take the model's measured
/// selection, and `native` adds the opt-in wrapper (target bank only).
pub fn serving_op(
    set: moe::Experts,
    kind: GatedMlpProjection,
    capacity: u32,
    rows: u32,
    native: bool,
) -> LabResult<Box<dyn PreparedExperts>> {
    let backend = registration::register(Sm120::new(), &FLASH, 4)?;
    let usage = usage(capacity);
    let query = moe::projection(&FLASH, 4, set, kind, capacity)?;
    let base = backend
        .chained_mlp_projection(&query, usage, Need::rows(capacity))
        .map_err(|e| e.to_string())?;
    let base = if capacity == rows {
        bundle_deepseek41::sm120::moe_pipeline::prepare(&FLASH, 4, set, kind, usage, base)
            .map_err(|e| e.to_string())?
    } else {
        base
    };
    if native {
        bundle_deepseek41::sm120::moe_native::wrap(kind, usage, base).map_err(|e| e.to_string())
    } else {
        Ok(base)
    }
}

/// The launches of `op` over `b`, with its workspace and any decoded
/// operand scratch carved from the workspace's end as `ops::moe` does.
pub fn op_launches(
    a: &mut Arena<'_>,
    op: &dyn PreparedExperts,
    b: &GatedMlpBuffers<View>,
    rows: u32,
) -> LabResult<(Vec<Launch>, View)> {
    let info = op.info();
    let workspace = a.filled(info.workspace.max(256), 0)?;
    let mut b = *b;
    if let Some(bytes) = info.bytes_of("scratch.partial") {
        b.scratch.partial = Some(a.filled(bytes, 0)?);
    }
    let ops = op
        .launches(&b, workspace, Counts::rows(rows))
        .map_err(|e| e.to_string())?;
    Ok((ops, workspace))
}

/// The split of a gate/up op's launches into its tiles and its ceil quantizer.
pub fn quantizer(ops: &[Launch]) -> LabResult<Vec<Launch>> {
    let split = ops
        .iter()
        .position(|l| l.symbol.contains("_moe_ceil_"))
        .ok_or("gate/up op has no ceil quantizer")?;
    Ok(ops[split..].to_vec())
}

/// A lab native projection: `tile` and `variant` of the lab cubin.
pub fn native_ops(
    shape: &Shape,
    tile: &Tile,
    rows: u32,
    buffers: &native::Buffers,
    variant: Variant,
) -> LabResult<Vec<Launch>> {
    native::launches(shape, tile, rows, buffers, variant).map_err(|e| e.to_string())
}

pub fn ints(v: &[u8]) -> Vec<i32> {
    v.chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
pub fn floats(v: &[u8]) -> Vec<f32> {
    v.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// 32 evenly spaced rows of `live`, first and last included (all below 32).
pub fn samples(live: usize) -> Vec<usize> {
    if live <= 32 {
        return (0..live).collect();
    }
    (0..32).map(|i| i * (live - 1) / 31).collect()
}

#[test]
fn samples_cover_first_and_last() {
    assert_eq!(samples(1), vec![0]);
    assert_eq!(samples(31).len(), 31);
    let s = samples(8192);
    assert_eq!((s.len(), s[0], s[31]), (32, 0, 8191));
    assert!(s.windows(2).all(|w| w[0] < w[1]));
}
