//! Guarded subspans and checked launches for the complete wide MoE pairs.
use kernel_core::backend::{Launch, View};
use kernel_core::contract::Argument;
use kernel_harness::{LabResult, Session, launch};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream};
use std::collections::BTreeMap;

const GUARD: usize = 256;
const MARK: u8 = 0xd3;

#[derive(Clone, Copy)]
struct Region {
    owner: u64,
    offset: usize,
    bytes: usize,
}

pub struct Pool<'a> {
    pub session: Option<&'a Session>,
    buffers: BTreeMap<u64, DeviceBuffer>,
    regions: BTreeMap<u64, Region>,
    roots: Vec<View>,
    next: u64,
}

impl<'a> Pool<'a> {
    pub fn new(session: Option<&'a Session>) -> Self {
        Self {
            session,
            buffers: BTreeMap::new(),
            regions: BTreeMap::new(),
            roots: Vec::new(),
            next: 0,
        }
    }
    pub fn zeros(&mut self, bytes: u64) -> LabResult<View> {
        self.next += 1;
        let view = View {
            handle: self.next,
            bytes,
        };
        let size = usize::try_from(bytes).map_err(|e| e.to_string())?;
        let allocated = size
            .checked_add(GUARD * 2)
            .ok_or("guarded allocation overflows")?;
        if let Some(s) = self.session {
            let b = s.allocate(allocated)?;
            if !b.is_aligned_to(256) {
                return Err("unaligned guarded allocation".into());
            }
            b.copy_from_bytes(&vec![MARK; allocated], &s.stream)
                .map_err(|e| e.to_string())?;
            b.zero_at_async(GUARD, size, &s.stream)
                .map_err(|e| e.to_string())?;
            self.buffers.insert(view.handle, b);
        }
        self.regions.insert(
            view.handle,
            Region {
                owner: view.handle,
                offset: GUARD,
                bytes: size,
            },
        );
        self.roots.push(view);
        Ok(view)
    }
    pub fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        let view = self.zeros(bytes.len() as u64)?;
        self.write(view, bytes)?;
        Ok(view)
    }
    pub fn subspan(&mut self, root: View, offset: u64, bytes: u64) -> LabResult<View> {
        if offset.checked_add(bytes).is_none_or(|end| end > root.bytes) {
            return Err("subspan exceeds its parent".into());
        }
        let parent = self.regions[&root.handle];
        self.next += 1;
        let view = View {
            handle: self.next,
            bytes,
        };
        self.regions.insert(
            view.handle,
            Region {
                owner: parent.owner,
                offset: parent.offset + offset as usize,
                bytes: bytes as usize,
            },
        );
        Ok(view)
    }
    pub fn write(&self, view: View, bytes: &[u8]) -> LabResult<()> {
        if bytes.len() != view.bytes as usize {
            return Err("write length differs from its span".into());
        }
        if let Some(s) = self.session {
            let r = self.regions[&view.handle];
            self.buffers[&r.owner]
                .copy_from_bytes_at(r.offset, bytes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn fill(&self, view: View, byte: u8) -> LabResult<()> {
        self.write(view, &vec![byte; view.bytes as usize])
    }
    pub fn read(&self, view: View) -> LabResult<Vec<u8>> {
        let r = self.regions[&view.handle];
        self.read_range(r.owner, r.offset, r.bytes)
    }
    fn read_range(&self, owner: u64, offset: usize, bytes: usize) -> LabResult<Vec<u8>> {
        let s = self.session.ok_or("read needs a GPU")?;
        let mut result = vec![0; bytes];
        self.buffers[&owner]
            .copy_to_bytes_at_async(offset, &mut result, &s.stream)
            .map_err(|e| e.to_string())?;
        self.sync()?;
        Ok(result)
    }
    pub fn guards(&self) -> LabResult<()> {
        for root in &self.roots {
            for offset in [0, GUARD + root.bytes as usize] {
                if self.read_range(root.handle, offset, GUARD)? != [MARK; GUARD] {
                    return Err(format!(
                        "guard changed for buffer {} at {offset}",
                        root.handle
                    ));
                }
            }
        }
        Ok(())
    }
    pub fn sync(&self) -> LabResult<()> {
        if let Some(s) = self.session {
            s.stream.synchronize().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn argument(&self, handle: u64) -> LabResult<KernelArgument<'_>> {
        let r = self.regions.get(&handle).ok_or("unbound lab view")?;
        Ok(KernelArgument::Range {
            buffer: self.buffers.get(&r.owner).ok_or("view has no allocation")?,
            offset: r.offset,
            byte_len: r.bytes,
        })
    }
}

impl Drop for Pool<'_> {
    fn drop(&mut self) {
        if let Some(s) = self.session
            && let Err(e) = s.context.synchronize()
        {
            eprintln!("wide MoE pool fence failed; retaining allocations: {e}");
            std::mem::forget(std::mem::take(&mut self.buffers));
        }
    }
}

pub struct Programs(BTreeMap<String, Function>);
impl Programs {
    pub fn load(pool: &Pool<'_>, calls: &[Launch]) -> LabResult<Self> {
        let s = pool.session.ok_or("loading needs a GPU")?;
        let mut functions = BTreeMap::new();
        for call in calls {
            if functions.contains_key(&call.symbol) {
                continue;
            }
            let (_, bytes) = bundle_deepseek41::sm120::fold_cubin_for_symbol(&call.symbol)
                .ok_or_else(|| format!("missing embedded cubin {}", call.symbol))?;
            let f = s
                .load_cubin(bytes)?
                .function(&call.symbol)
                .map_err(|e| e.to_string())?;
            functions.insert(call.symbol.clone(), f);
        }
        Ok(Self(functions))
    }
    pub fn issue(&self, pool: &Pool<'_>, calls: &[Launch], stream: &Stream) -> LabResult<()> {
        for call in calls {
            let args = call
                .arguments
                .0
                .iter()
                .map(|a| {
                    Ok(match a {
                        Argument::DeviceBuffer(h) => pool.argument(*h)?,
                        Argument::U32(v) => KernelArgument::U32(*v),
                        Argument::I32(v) => KernelArgument::I32(*v),
                        Argument::U64(v) => KernelArgument::U64(*v),
                        Argument::NullDevicePointer => KernelArgument::NullBuffer,
                        _ => return Err("wide MoE fixture needs an unsupported argument".into()),
                    })
                })
                .collect::<LabResult<Vec<_>>>()?;
            launch::launch_with(
                &self.0[&call.symbol],
                stream,
                launch::config(call.plan)?,
                &args,
            )?;
        }
        Ok(())
    }
}
