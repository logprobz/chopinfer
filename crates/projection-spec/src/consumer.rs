//! A declared dataflow consumer or a kernel that owns its arithmetic.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum Consumer {
    Declared(String),
    Kernel { kernel: String },
}
impl Consumer {
    pub fn matches_declared(&self, id: &str) -> bool {
        matches!(self,Self::Declared(name) if name==id)
    }
}
impl From<String> for Consumer {
    fn from(id: String) -> Self {
        Self::Declared(id)
    }
}
impl From<&str> for Consumer {
    fn from(id: &str) -> Self {
        Self::Declared(id.into())
    }
}
impl PartialEq<str> for Consumer {
    fn eq(&self, id: &str) -> bool {
        self.matches_declared(id)
    }
}
impl PartialEq<&str> for Consumer {
    fn eq(&self, id: &&str) -> bool {
        self.matches_declared(id)
    }
}
impl std::fmt::Display for Consumer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Declared(id) => f.write_str(id),
            Self::Kernel { kernel } => write!(f, "kernel.{kernel}"),
        }
    }
}
