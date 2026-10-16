//! Provider-owned addressing and launch controls, separate from logical ports.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use kernel_core::contract::DType;

use crate::{Error, Variant};

/// Fused execution does not request an additional standalone stage launch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StageExecution {
    Standalone { symbol: String },
    Fused { operation: String },
}

/// A runtime control binding may lie outside the stage's logical tensor ports.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlBinding {
    pub view: String,
    pub dtype: DType,
    pub interpretation: Option<crate::contracts::Interpretation>,
    pub offset_bytes: u64,
    pub stride_bytes: Option<u64>,
}

/// The provider owns field meanings. Integer values and floating-point bits
/// remain distinct, and runtime controls retain their binding and storage type.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum AdapterValue {
    U64(u64),
    I64(i64),
    F32Bits(u32),
    Bool(bool),
    Text(String),
    Binding(ControlBinding),
    List(Vec<AdapterValue>),
}

/// A registered provider's typed native adapter payload. Shared code validates
/// ownership and binding storage, and leaves addressing semantics to the provider.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageAdapter {
    pub contract: String,
    pub execution: StageExecution,
    pub facts: BTreeMap<String, AdapterValue>,
}

impl StageAdapter {
    pub(super) fn validate(&self, variant: &Variant) -> Result<(), Error> {
        let target = match &self.execution {
            StageExecution::Standalone { symbol } => symbol,
            StageExecution::Fused { operation } => operation,
        };
        if self.contract != variant.id || target.is_empty() {
            return Err(Error(
                "stage adapter changes provider identity or has no execution target".into(),
            ));
        }
        fn check_value(value: &AdapterValue, depth: u32) -> Result<(), Error> {
            if depth > 16 {
                return Err(Error("stage adapter facts exceed nesting limit".into()));
            }
            match value {
                AdapterValue::Binding(binding) => {
                    let width = kernel_core::precision::dtype_bits(binding.dtype) / 8;
                    if binding.view.is_empty()
                        || width == 0
                        || binding.offset_bytes % width != 0
                        || binding
                            .stride_bytes
                            .is_some_and(|s| s == 0 || s % width != 0)
                        || (binding.interpretation.is_some()
                            && !matches!(binding.dtype, DType::I32 | DType::I64))
                    {
                        return Err(Error("stage adapter has an invalid control binding".into()));
                    }
                }
                AdapterValue::List(values) => {
                    for item in values {
                        check_value(item, depth + 1)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        for (field, fact) in &self.facts {
            if !crate::registry::valid_name(field) {
                return Err(Error("stage adapter has an invalid fact name".into()));
            }
            check_value(fact, 0)?;
        }
        Ok(())
    }
}
