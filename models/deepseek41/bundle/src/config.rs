//! The create record of a `deepseek41.v1` instance.
//!
//! The package requires the served item count and served context; an
//! optional tap producer may follow. The topology names the geometry and the
//! rank count; the KV layout names the storage of the pages and rings.

use bundle_runtime::config::CreateConfig;
use bundle_runtime::config::Topology;
use model_deepseek41_contract::serving::{
    DeepSeek, KV_LAYOUT_BF16, KV_LAYOUT_NATIVE, PAGE_TOKENS, RANK_IMAGE_ROLE, REQUEST_STATE_ID,
    index_role,
};
use model_deepseek41_contract::state::KvStorage;
use runtime_bundle_abi::BundleStatusV1;

use crate::shape::Shape;

/// One serving KV layout the package declares: its ID and the storage of
/// its pages and rings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvLayout {
    pub layout_id: &'static [u8],
    pub storage: KvStorage,
}

/// The serving KV layouts, the native one first (the default, D4).
pub const KV_LAYOUTS: [KvLayout; 2] = [
    KvLayout {
        layout_id: KV_LAYOUT_NATIVE,
        storage: KvStorage::Native,
    },
    KvLayout {
        layout_id: KV_LAYOUT_BF16,
        storage: KvStorage::Bf16,
    },
];

/// One instance's checked configuration.
#[derive(Clone, Debug)]
pub struct Configuration {
    pub shape: Shape,
    pub create: CreateConfig,
    pub kv: KvLayout,
    /// The pages' bytes, which the layout's storage and the schedule fix.
    pub page_bytes: u64,
}

/// The per-item state bytes of one state slot for `shape` in `storage`,
/// with or without the DSpark drafter's state, whole 256-byte blocks.
pub fn slot_state_bytes(shape: &Shape, storage: KvStorage, dspark: bool) -> u64 {
    let state = model_deepseek41_contract::state::request_state(shape.geometry, storage, dspark);
    let slot = bundle_runtime::state::Extent {
        served: 1,
        bucket: 1,
        verify_rows: 0,
    };
    state
        .buffers
        .iter()
        .map(|buffer| buffer.bytes(slot))
        .sum::<u64>()
        .next_multiple_of(256)
}

/// Decodes and checks a create record against the package's declarations.
pub fn decode(bytes: &[u8], log: impl Fn(&str)) -> Result<Configuration, BundleStatusV1> {
    let shapes = crate::shape::topologies();
    let ids = shapes.iter().map(Shape::topology).collect::<Vec<_>>();
    let topologies = shapes
        .iter()
        .zip(&ids)
        .map(|(shape, id)| Topology {
            id,
            ranks: shape.ranks,
        })
        .collect::<Vec<_>>();
    let create = bundle_runtime::config::decode_with_log::<DeepSeek>(bytes, &topologies, &log)?;
    let topology = bundle_runtime::config::topology_id(bytes)?;
    let shape = crate::shape::of_topology(topology).ok_or(BundleStatusV1::Unsupported)?;
    if create
        .context_tokens
        .is_some_and(|tokens| tokens == 0 || tokens > shape.geometry.context_tokens())
    {
        log("the served context exceeds the geometry's");
        return Err(BundleStatusV1::InvalidArgument);
    }
    let kv = *KV_LAYOUTS
        .iter()
        .find(|layout| layout.layout_id == create.kv.layout_id.as_slice())
        .ok_or(BundleStatusV1::Unsupported)?;
    let page_bytes =
        model_deepseek41_contract::state::request_state(shape.geometry, kv.storage, false)
            .page_bytes();
    let has_roles = create
        .roles
        .iter()
        .any(|role| role == RANK_IMAGE_ROLE.as_bytes())
        && (0..shape.ranks()).all(|rank| {
            create
                .roles
                .iter()
                .any(|role| *role == index_role(rank).as_bytes())
        });
    let k = &create.kv;
    let s = &create.state;
    // The host allocates the per-item state of every bucket item, with or
    // without the drafter's.
    let slots = u64::from(create.items.capacity());
    let state_bytes =
        [false, true].map(|dspark| slots * slot_state_bytes(&shape, kv.storage, dspark));
    if k.page_bytes != page_bytes
        || k.page_tokens != PAGE_TOKENS
        || k.ranks != u64::from(shape.ranks())
        || k.lifetime != 0
        || k.alignment < 256
        || !k.alignment.is_power_of_two()
        || s.state_id != REQUEST_STATE_ID.as_bytes()
        || !state_bytes.contains(&s.bytes)
        || s.alignment < 256
        || !s.alignment.is_power_of_two()
        || !has_roles
    {
        log("the create record differs from the package's declarations");
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(Configuration {
        shape,
        create,
        kv,
        page_bytes,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use bundle_runtime::config::{KvLayout as Kv, StateLayout, test_support::encode};

    /// A record for `topology` over `layout`, serving `items`, with the
    /// per-item state of `dspark`.
    pub(crate) fn record(shape: &Shape, layout: KvLayout, items: u32, dspark: bool) -> Vec<u8> {
        let ranks = shape.ranks;
        let page_bytes =
            model_deepseek41_contract::state::request_state(shape.geometry, layout.storage, false)
                .page_bytes();
        let capacity = bundle_runtime::geometry::Items::new(items)
            .unwrap()
            .capacity();
        let mut roles = vec![RANK_IMAGE_ROLE.as_bytes().to_vec()];
        roles.extend((0..ranks.get()).map(|rank| index_role(rank).into_bytes()));
        let ordinals: Vec<i32> = (0..ranks.get() as i32).collect();
        encode(
            shape.topology().as_bytes(),
            &ordinals,
            &Kv {
                layout_id: layout.layout_id.to_vec(),
                page_bytes,
                alignment: 256,
                page_tokens: PAGE_TOKENS,
                ranks: u64::from(ranks.get()),
                lifetime: 0,
            },
            &StateLayout {
                state_id: REQUEST_STATE_ID.as_bytes().to_vec(),
                bytes: u64::from(capacity) * slot_state_bytes(shape, layout.storage, dspark),
                alignment: 256,
            },
            &roles,
            Some(items),
        )
    }

    #[test]
    fn a_record_of_every_topology_and_layout_decodes() {
        for shape in crate::shape::topologies() {
            for layout in KV_LAYOUTS {
                let bytes = record(&shape, layout, 4, false);
                let config = decode(&bytes, |_| {}).unwrap();
                assert_eq!(config.kv, layout);
                assert_eq!(config.shape, shape);
            }
        }
    }

    #[test]
    fn a_record_that_differs_from_the_package_is_refused() {
        let shape = crate::shape::topologies()[0];
        let mut bytes = record(&shape, KV_LAYOUTS[0], 4, false);
        // The state layout's ID, one byte off.
        let at = bytes
            .windows(REQUEST_STATE_ID.len())
            .position(|w| w == REQUEST_STATE_ID.as_bytes())
            .unwrap();
        bytes[at] ^= 1;
        assert_eq!(
            decode(&bytes, |_| {}).unwrap_err(),
            BundleStatusV1::Unsupported
        );
    }
}
