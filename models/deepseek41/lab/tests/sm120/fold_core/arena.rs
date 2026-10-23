use super::*;
pub struct Arena<'a> {
    pub session: Option<&'a Session>,
    next: u64,
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
    pub fn filled(&mut self, bytes: u64, value: u8) -> LabResult<View> {
        self.next += 1;
        let v = View {
            handle: self.next,
            bytes,
        };
        if let Some(s) = self.session {
            // The PDL rewrite probe copies whole u32 words. Keep the
            // logical view exact while padding its allocation, including
            // odd row counts with K=576 (18 scale bytes per row).
            let b = s.allocate((bytes as usize).div_ceil(4) * 4)?;
            kernel_harness::poison::fill_all(s, &b, &[value])?;
            self.buffers.insert(v.handle, b);
        }
        Ok(v)
    }
    /// A distinct checked role handle over the same allocation.
    pub fn alias(&mut self, source: View) -> View {
        self.next += 1;
        let view = View {
            handle: self.next,
            bytes: source.bytes,
        };
        if self.session.is_some() {
            self.buffers
                .insert(view.handle, self.buffers[&source.handle].clone());
        }
        view
    }
    pub fn bytes(&mut self, bytes: &[u8]) -> LabResult<View> {
        let v = self.filled(bytes.len() as u64, 0)?;
        self.write(v, bytes)?;
        Ok(v)
    }
    pub fn write(&self, v: View, bytes: &[u8]) -> LabResult<()> {
        if let Some(s) = self.session {
            if bytes.len() != v.bytes as usize {
                return Err("host write extent differs".into());
            }
            self.buffers[&v.handle]
                .copy_from_bytes(bytes, &s.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn fill(&self, v: View, value: u8) -> LabResult<()> {
        kernel_harness::poison::fill_all(self.session.unwrap(), &self.buffers[&v.handle], &[value])
    }
    pub fn read(&self, v: View) -> LabResult<Vec<u8>> {
        kernel_harness::poison::read(self.session.unwrap(), &self.buffers[&v.handle])
    }
}
