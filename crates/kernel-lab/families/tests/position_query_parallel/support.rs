use kernel_core::backend::{Launch, View};
use kernel_harness::{
    LabResult, Session,
    launch::{Bindings, launch},
};
use runtime_cuda::{DeviceBuffer, Module};
pub struct Arena<'a> {
    pub session: &'a Session,
    pub buffers: Vec<DeviceBuffer>,
}
impl Arena<'_> {
    pub fn input(&mut self, bytes: &[u8]) -> LabResult<View> {
        let b = self.session.allocate(bytes.len())?;
        b.copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())?;
        self.buffers.push(b);
        Ok(View {
            handle: self.buffers.len() as u64,
            bytes: bytes.len() as u64,
        })
    }
    pub fn output(&mut self, bytes: usize) -> LabResult<View> {
        self.input(&vec![0xa5; bytes])
    }
    pub fn write(&self, v: View, bytes: &[u8]) -> LabResult<()> {
        self.buffers[v.handle as usize - 1]
            .copy_from_bytes(bytes, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    pub fn read(&self, v: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session, &self.buffers[v.handle as usize - 1])
    }
    pub fn poison(&self, views: &[View]) -> LabResult<()> {
        for &v in views {
            self.write(v, &vec![0xa5; v.bytes as usize])?;
        }
        Ok(())
    }
    pub fn bindings(&self) -> Bindings<'_> {
        Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        }
    }
}
pub fn i32s(values: impl IntoIterator<Item = i32>) -> Vec<u8> {
    values.into_iter().flat_map(i32::to_le_bytes).collect()
}
pub fn fire(
    session: &Session,
    module: &Module,
    arena: &Arena<'_>,
    calls: &[Launch],
) -> LabResult<()> {
    for c in calls {
        let f = module.function(&c.symbol).map_err(|e| e.to_string())?;
        launch(
            &session.context,
            &f,
            &session.stream,
            c.plan,
            &c.arguments,
            &arena.bindings(),
        )?;
    }
    Ok(())
}

// Gated, alternating A/B rounds exclude host submission gaps. Score timings
// contain one kernel; query timings cover the whole dependency chain once.
pub fn time_pair(
    session: &Session,
    module: &Module,
    arena: &Arena<'_>,
    calls: [&[Launch]; 2],
    rounds: usize,
) -> LabResult<[f32; 2]> {
    let graphs = calls
        .iter()
        .map(|calls| session.capture(|_| fire(session, module, arena, calls)))
        .collect::<LabResult<Vec<_>>>()?;
    let mode = kernel_harness::eviction::EvictionMode::Read;
    let result = kernel_harness::time_ab(session, 1, rounds, [mode, mode], |arm, stream| {
        graphs[arm].launch(stream).map_err(|e| e.to_string())
    })?;
    Ok(result.arms.map(|x| x.median_ms))
}

pub fn gpu_test<T>(need: u64, run: impl FnOnce(&Session) -> LabResult<T>) -> LabResult<Option<T>> {
    let path =
        std::env::var_os("CHOPINFER_L4_PROBE").ok_or("the staged probe_sm120 cubin is required")?;
    let bytes: &'static [u8] = Box::leak(
        std::fs::read(path)
            .map_err(|e| e.to_string())?
            .into_boxed_slice(),
    );
    let cubins = kernel_harness::Cubins::new(
        Box::leak(vec![("probe_sm120", bytes)].into_boxed_slice()),
        &[],
    );
    kernel_harness::gpu_test_with(cubins, need, run)
}
