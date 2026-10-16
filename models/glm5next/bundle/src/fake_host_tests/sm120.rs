//! The SM120 facts the fake-host tests check launches against: the
//! kernels' invariance declarations and GLM's program check.

use std::collections::BTreeMap;

pub(super) use kernels_sm120::invariance::{Declared, ExactVerify, Launched, OpKind, Program};

/// Each declaration of GLM's own kernels (`sm120::own_ops::declared_kernels`)
/// and of GLM's catalogs and the kernel base's, by symbol.
pub(super) fn declarations() -> BTreeMap<String, Declared> {
    let catalogs = kernels_sm120::Catalogs::new(&[&model_glm5next_contract::sm120::CATALOG])
        .expect("GLM's catalogs are consistent");
    kernels_sm120::invariance::declarations(&catalogs)
        .into_iter()
        .chain(crate::sm120::own_ops::declared_kernels())
        .map(|declared| (declared.symbol.clone(), declared))
        .collect()
}
