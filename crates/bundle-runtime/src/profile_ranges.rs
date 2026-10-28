//! Optional host annotations over the existing log callback. No device work.
use runtime_bundle_abi::HostServicesV1;
use std::rc::Rc;
use std::sync::OnceLock;

static ENABLED: OnceLock<bool> = OnceLock::new();

fn begin(services: &HostServicesV1, label: &str) -> Option<String> {
    if !ENABLED.get_or_init(|| std::env::var("CHOPINFER_NVTX_RANGES").as_deref() == Ok("1")) {
        return None;
    }
    services
        .log(format!("chopinfer.profile-range.begin/{label}").as_bytes())
        .ok()?;
    Some(format!("chopinfer.profile-range.end/{label}"))
}

pub struct Range {
    services: Rc<HostServicesV1>,
    end: String,
}

impl Range {
    pub fn new(services: &Rc<HostServicesV1>, label: &str) -> Option<Self> {
        Some(Self {
            end: begin(services, label)?,
            services: Rc::clone(services),
        })
    }
}

pub struct BorrowedRange<'a> {
    services: &'a HostServicesV1,
    end: String,
}

impl<'a> BorrowedRange<'a> {
    pub fn new(services: &'a HostServicesV1, label: &str) -> Option<Self> {
        Some(Self {
            end: begin(services, label)?,
            services,
        })
    }
}

impl Drop for BorrowedRange<'_> {
    fn drop(&mut self) {
        let _ = self.services.log(self.end.as_bytes());
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        let _ = self.services.log(self.end.as_bytes());
    }
}
