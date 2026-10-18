//! The vision encoder's SM120 side (`crate::encoders`): the encoder runtime
//! created on the instance's backend, every slot resolved to the query of
//! the catalog entry generated for it, and the dynamic shared memory each
//! encoder entry loads with.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use bundle_runtime::acquire;
use bundle_runtime::encoder::{Encoder, SlotOp};
use kernels_sm120::encoder::{Entry, Shared, Slot};
use model_deepseek41_contract::encoder::declarations;
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1 as Span, HostServicesV1};

use super::encoder::kernels;

/// The declaration's slots with their catalog entries. Built once.
fn slots() -> Result<&'static [(Slot, Entry)], String> {
    static SLOTS: OnceLock<Result<Vec<(Slot, Entry)>, String>> = OnceLock::new();
    SLOTS
        .get_or_init(|| {
            let [vision] = declarations().map_err(|error| error.0.clone())?;
            kernels_sm120::encoder::slots(vision, &Shared).map_err(|error| error.0)
        })
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// The query of slot (`stage`, `part`, `block`), which the backend serves
/// from the catalog entry of its code.
fn resolve(stage: usize, part: &'static str, block: Option<u32>) -> Result<SlotOp, String> {
    let slot = Slot { stage, part, block };
    let (_, entry) = slots()?
        .iter()
        .find(|(known, _)| *known == slot)
        .ok_or_else(|| format!("the vision catalog has no slot {slot:?}"))?;
    Ok(SlotOp::Query(entry.query))
}

impl crate::Kernels {
    /// The encoder runtime over each rank's encoder image `weights`, every
    /// slot prepared on this backend, every rank's regions allocated with
    /// `allocate`.
    pub(crate) fn encoder(
        &self,
        services: &HostServicesV1,
        weights: Vec<BTreeMap<String, Span>>,
        allocate: &mut dyn FnMut(usize, u64) -> Result<Span, BundleAbiError>,
    ) -> Result<Encoder, BundleAbiError> {
        let [vision] = declarations().map_err(|_| {
            BundleAbiError::InvalidRange("the deepseek41 vision declaration is invalid")
        })?;
        Encoder::with_backend(
            services,
            &[Some(vision)],
            crate::encoders::CONFIG,
            self.backend.as_ref(),
            &|_, stage, part, block| resolve(stage, part, block),
            weights,
            allocate,
        )
    }

    /// The dynamic shared memory an encoder entry `symbol` loads with, at
    /// least the default; `None` for a symbol that is not the encoder's.
    pub(crate) fn encoder_shared_bytes(symbol: &str) -> Option<u32> {
        kernels()
            .ok()?
            .iter()
            .find(|entry| entry.symbol == symbol)
            .map(|entry| {
                entry
                    .dynamic_shared_bytes()
                    .unwrap_or(0)
                    .max(acquire::DEFAULT_SHARED_BYTES)
            })
    }
}

#[cfg(test)]
mod tests {
    use bundle_runtime::encoder::{Lowered, Shared as SharedRegistered};
    use encoder_spec::buffers::{self, Serving};
    use model_deepseek41_contract::encoder::ARENA_ROWS;
    use model_deepseek41_contract::serving::PREFILL_TOKENS;

    use super::*;

    /// The encoder cubins embed with the bundle's, holding every catalog
    /// entry, and each loads with at least its planned shared memory.
    #[test]
    fn the_encoder_cubins_embed_every_entry() {
        let embedded: Vec<&str> = crate::CUBINS
            .iter()
            .filter(|cubin| super::super::encoder::is_cubin(cubin.name))
            .flat_map(|cubin| cubin.symbols.iter().copied())
            .collect();
        let mut catalog: Vec<&str> = kernels()
            .unwrap()
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect();
        let mut sorted = embedded.clone();
        sorted.sort_unstable();
        catalog.sort_unstable();
        assert_eq!(sorted, catalog);
        for entry in kernels().unwrap() {
            let planned = entry.dynamic_shared_bytes().unwrap_or(0);
            let loaded = crate::Kernels::encoder_shared_bytes(&entry.symbol).unwrap();
            assert!(loaded >= planned);
        }
        assert_eq!(
            crate::Kernels::encoder_shared_bytes("chop_row_not_an_encoder_entry"),
            None
        );
    }

    /// Every slot of the declaration prepares on the instance's backend at
    /// TP4, the rows' and the encoder's entries registered together, and
    /// the launch list runs the catalog's entries.
    #[test]
    fn every_slot_prepares_on_the_instance_backend() {
        let shape = crate::shape::Shape::new(
            &model_deepseek41_contract::geometry::FLASH,
            bundle_runtime::geometry::Ranks::new(4).unwrap(),
        )
        .unwrap();
        let instance = crate::Kernels::new(&shape).unwrap();
        let [vision] = declarations().unwrap();
        let plan = buffers::plan(
            &[vision],
            u64::from(vision.input.capacity_rows),
            Serving {
                arena_rows: ARENA_ROWS,
                prefill_rows: u64::from(PREFILL_TOKENS),
                staging_bytes: 0,
            },
        )
        .unwrap();
        let lowered = Lowered::with_backend(
            vision,
            &plan,
            instance.backend.as_ref(),
            &resolve,
            &SharedRegistered,
            PREFILL_TOKENS,
        )
        .unwrap();
        let catalog: std::collections::BTreeSet<&str> = kernels()
            .unwrap()
            .iter()
            .map(|entry| entry.symbol.as_str())
            .collect();
        let launched: std::collections::BTreeSet<&str> = lowered.symbols().collect();
        assert!(launched.is_subset(&catalog), "{launched:?}");
        assert!(
            launched
                .iter()
                .any(|symbol| symbol.starts_with("chop_enc_merge_space_to_depth_w1024_g9_"))
        );
    }
}
