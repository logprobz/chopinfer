use super::*;
use bundle_runtime::fake_host::{Backend, FakeDevice, FakePackage, NoObserver};
use runtime_bundle_abi::*;
use std::cell::RefCell;
use std::rc::Rc;

pub struct Package {
    pub geometry: &'static Geometry,
    pub ranks: u32,
    pub device: FakeDevice,
    pub events: RefCell<Vec<String>>,
    pub memory: RefCell<BTreeMap<u32, Vec<u8>>>,
}

pub fn host(shape: crate::shape::Shape) -> Rc<Backend<Package>> {
    Rc::new(Backend::new(Package {
        geometry: shape.geometry,
        ranks: shape.ranks(),
        device: FakeDevice::default(),
        events: RefCell::default(),
        memory: RefCell::default(),
    }))
}

impl FakePackage for Package {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        &self.device
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        []
    }
    fn open_compiled_image(
        &self,
        _role: &[u8],
    ) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_compiled_image_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        _rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn open_authenticated_component(
        &self,
        _role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        Err(BundleStatusV1::NotFound)
    }
    fn component(&self, _slot: u32) -> (String, Vec<u8>) {
        unreachable!("the package has no components")
    }
    fn acquire_serving_kv(&self, _layout_id: &[u8]) -> HostServiceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_serving_kv_pool(&self, _rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        Ok(BundleRawSpeculationV1::disabled())
    }
    fn mapped_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<Vec<u8>> {
        if rank >= self.ranks || view_id.len() < 2 {
            return Err(BundleStatusV1::NotFound);
        }
        let layer = std::str::from_utf8(&view_id[..view_id.len() - 1])
            .ok()
            .and_then(|text| text.parse::<usize>().ok())
            .ok_or(BundleStatusV1::NotFound)?;
        let width = match view_id.last() {
            Some(b'c') => self.geometry.shape.engram.head_dim,
            Some(b's') => self.geometry.shape.engram.head_dim / 32,
            _ => return Err(BundleStatusV1::NotFound),
        };
        let rows = *self
            .geometry
            .shape
            .engram
            .rows
            .get(layer)
            .ok_or(BundleStatusV1::NotFound)?;
        self.events.borrow_mut().push(format!("gather rank {rank}"));
        Ok(vec![0; rows as usize * width as usize])
    }
    fn allocate_state(
        &self,
        _request: &StateRequestV1,
    ) -> HostServiceResultV1<BundleStateHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.events
            .borrow_mut()
            .push(format!("read {}", destination.len()));
        let memory = self.memory.borrow();
        let bytes = memory
            .get(&source.buffer.slot())
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let start = source.offset as usize;
        destination.copy_from_slice(&bytes[start..start + destination.len()]);
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.events.borrow_mut().push("upload".into());
        let mut memory = self.memory.borrow_mut();
        let bytes = memory.entry(destination.buffer.slot()).or_default();
        let end = destination.offset as usize + source.len();
        if bytes.len() < end {
            bytes.resize(end, 0);
        }
        bytes[destination.offset as usize..end].copy_from_slice(source);
        Ok(())
    }
}
