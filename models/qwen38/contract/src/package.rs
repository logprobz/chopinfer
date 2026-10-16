//! Static text autoregressive package declarations. No checkpoint or module bytes.
use crate::backend::{Kernels, Selected};
use crate::contract::*;
use kernel_core::allocation::Traversal;
use runtime_bundle_abi::*;
use serde_json::{Value, json};

pub const RANK_IMAGE_FORMAT: &str = "qwen38.rank-image.v1";
pub const RANK_INDEX_FORMAT: &str = "qwen38.rank-index.json.v1";
pub const RANK_IMAGE_ALIGNMENT: u64 = 256;

/// The most sequences an MTP package serves until wider MTP serving has
/// GPU accuracy receipts. The bundle's bounded checkpoint pool serves 256;
/// the TP4 accuracy gate covers C32 to C128 (docs/serving-profiles.md).
pub const MTP_QUALIFIED_SEQUENCES: u32 = 128;

pub use model_compiler_sdk::device::DEVICE_BYTES;
/// The device memory one rank may take on its own device.
pub fn rank_budget() -> u64 {
    DEVICE_BYTES
}

/// Sequences the working set model starts from.
const BASE_SEQUENCES: u32 = 32;
/// What a rank keeps besides its bundle buffers and weights, the host's
/// reserve (`model_compiler_sdk::device::HOST_RESERVE_BYTES`) included.
const HOST_SLACK_BYTES: u64 = 4 << 30;

/// Named buffers, persistent state, live/saved rings and the KV floor.
pub fn named_working_bytes(shape: Shape, sequences: u32) -> u64 {
    let items = sequences.next_power_of_two();
    let served = crate::buffers::Served {
        slots: sequences,
        items,
        rows: items.max(DECODE_BATCH),
    };
    let speculation = crate::buffers::Speculation {
        verify_rows: MAX_VERIFY_ROWS,
        drafter: true,
    };
    let mut buffers = 0_u64;
    crate::buffers::allocations(shape, served, speculation, Default::default(), false, 1)
        .visit(&mut |allocation| buffers += allocation.allocation_bytes());
    buffers
        + u64::from(sequences)
            * (shape.linear_layers() * (shape.conv_entry_bytes + shape.delta_entry_bytes)
                + shape.ple_entry_bytes())
        + u64::from(items) * shape.ring_state_slot_bytes(true)
        + (1 << 30)
}

fn flash_working_bytes(shape: Shape, sequences: u32) -> u64 {
    let extra = u64::from(sequences.next_power_of_two().max(BASE_SEQUENCES) - BASE_SEQUENCES);
    (4480 << 20)
        + ((73 << 29) + (232 << 20)) / u64::from(shape.ranks())
        + extra * ((10 << 20) + (215 << 20) / u64::from(shape.ranks()))
}

/// Named allocations plus the flash allowance for planner workspace,
/// communication buffers, prefix pages and allocation padding.
pub fn working_bytes(shape: Shape, sequences: u32) -> u64 {
    let capacity = sequences.next_power_of_two().max(BASE_SEQUENCES);
    let flash = Shape::flash(shape.ranks);
    let residual = flash_working_bytes(flash, capacity) - named_working_bytes(flash, capacity);
    named_working_bytes(shape, capacity) + residual
}

/// The largest rank's weights on the device: its image, less the n-gram
/// table when host memory holds it.
pub fn device_weight_bytes(shape: Shape, host_ngram: bool) -> u64 {
    (0..shape.ranks())
        .map(|rank| {
            let table = crate::views::required(rank, shape)
                .into_iter()
                .find(|view| view.layout_id == crate::views::ngram_layout(shape))
                .expect("every rank holds an n-gram table share");
            crate::views::image_bytes(rank, shape) - if host_ngram { table.bytes } else { 0 }
        })
        .max()
        .expect("a topology has ranks")
}

/// Device bytes each rank of a package with the vision encoder adds: the
/// encoder image, its buffers and arena.
#[cfg(test)]
pub fn vision_bytes() -> u64 {
    vision_bytes_for(Shape::flash(crate::contract::Ranks::TP4))
}

pub fn vision_bytes_for(shape: Shape) -> u64 {
    crate::encoder::views::image_bytes_for(shape.geometry)
        + crate::encoder::buffer_bytes(shape.geometry)
}

/// The media encoders a package plans. Qwen3.8 has one, the vision
/// encoder, which serves images; a text-only package plans none.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Encoders {
    pub vision: bool,
}

impl Encoders {
    pub const NONE: Self = Self { vision: false };
    pub const VISION: Self = Self { vision: true };
    /// Every set of encoders a package may plan.
    pub const SETS: [Self; 2] = [Self::NONE, Self::VISION];

    /// Device bytes the encoders add on each rank: [`vision_bytes_for`] for
    /// the vision encoder.
    pub fn media_bytes_flash(self) -> u64 {
        self.media_bytes_for(Shape::flash(crate::contract::Ranks::TP4))
    }

    pub fn media_bytes_for(self, shape: Shape) -> u64 {
        if self.vision {
            vision_bytes_for(shape)
        } else {
            0
        }
    }
}

/// Device bytes one rank needs before its KV pool grows when it serves
/// `sequences` sequences: its weights (and its `encoders`' media bytes),
/// its bundle's working set and the host's slack, rounded up to GiB.
fn rank_floor(shape: Shape, host_ngram: bool, sequences: u32, encoders: Encoders) -> u64 {
    (device_weight_bytes(shape, host_ngram)
        + encoders.media_bytes_for(shape)
        + working_bytes(shape, sequences)
        + HOST_SLACK_BYTES)
        .next_multiple_of(1 << 30)
}

/// Whether the n-gram table goes in host memory: when a rank with the table
/// on the device would not fit its [`rank_budget`] serving 32 sequences.
/// TP2's 24 GiB share does not fit; TP4's does.
pub fn ngram_in_host_memory(shape: Shape) -> bool {
    rank_floor(shape, false, BASE_SEQUENCES, Encoders::NONE) > rank_budget()
}

/// The most sequences a package serves: the largest graph bucket, up to 256
/// (or [`MTP_QUALIFIED_SEQUENCES`] with the MTP drafter), whose rank floor
/// fits its [`rank_budget`]. TP4 serves 256 (128 with MTP); TP2, whose ranks
/// hold twice the linear state per sequence, 64, or 32 with the encoder.
pub fn maximum_sequences(shape: Shape, mtp: bool, encoders: Encoders) -> u32 {
    let limit = if mtp {
        MTP_QUALIFIED_SEQUENCES
    } else {
        DECODE_BATCH
    };
    let host_ngram = ngram_in_host_memory(shape);
    (0..=limit.ilog2())
        .rev()
        .map(|power| 1 << power)
        .find(|&sequences| rank_floor(shape, host_ngram, sequences, encoders) <= rank_budget())
        .expect("a rank serves at least one sequence")
}

/// Per-rank device memory a package asks for: [`rank_floor`] at the
/// package's [`maximum_sequences`], with the n-gram table where
/// [`ngram_in_host_memory`] places it, so it covers every served count the
/// package admits at eight verify rows, whether the package's MTP drafter
/// or the host drafts.
pub fn minimum_device_memory(shape: Shape, mtp: bool, encoders: Encoders) -> u64 {
    rank_floor(
        shape,
        ngram_in_host_memory(shape),
        maximum_sequences(shape, mtp, encoders),
        encoders,
    )
}

fn text(bytes: &[u8]) -> &str {
    std::str::from_utf8(bytes).expect("ASCII contract")
}

pub fn declarations(
    shape: Shape,
    host_ngram: bool,
    runtime: crate::geometry::RuntimeValues,
) -> Value {
    let options = shape.geometry.model_options(runtime);
    let context_tokens = shape.context_tokens();
    let mut capabilities = vec![
        SERVING_KV_CAPABILITY_V1,
        RECURRENT_STATE_CAPABILITY_V1,
        RECURRENT_STATE_PREALLOCATED_CAPABILITY_V1,
        REQUEST_RETIREMENT_CAPABILITY_V1,
        PROMPT_ROWS_CAPABILITY_V2,
        SPECULATIVE_CAPABILITY_V1,
        PREFIX_CACHE_CAPABILITY_V1,
        KV_OFFLOAD_CAPABILITY_V1,
        PREFIX_CACHE_STATE_SLOT_CAPABILITY_V1,
        SERVED_ITEMS_CAPABILITY_V1,
        STATE_PER_ITEM_CAPABILITY_V1,
        RECURRENT_STATE_IN_PLACE_CAPABILITY_V1,
        SPECULATIVE_NARROW_ROWS_CAPABILITY_V1,
        // The module exports the declared chat frontend.
        frontend::FRONTEND_CAPABILITY_V1,
    ];
    if host_ngram {
        capabilities.push(HOST_RESIDENT_VIEWS_CAPABILITY_V1);
    }
    if !options.is_empty() {
        capabilities.push(runtime_bundle::model_options::CAPABILITY);
    }
    capabilities.sort();
    let kv = |index: usize| {
        let formats = shape.kv_formats();
        let f = &formats[index * 2];
        json!({"alignment_bytes":256,"bytes_per_token":f.page_bytes / PAGE_TOKENS,
            "id":text(f.layout_id),"maximum_sequence_tokens":context_tokens,
            "page_bytes":f.page_bytes,"page_tokens":PAGE_TOKENS,"rank_placement":"rank-local",
            "segments_per_page":1,"state_lifetime":"transaction"})
    };
    let recurrent: Vec<_> = shape
        .recurrent_state_layouts()
        .iter()
        .map(|(id, bytes)| {
            json!({
                "id":id,"bytes":bytes,"alignment_bytes":128,"lifetime":"request"
            })
        })
        .collect();
    let mut components = vec![
        json!({"id":"rank-images","role":text(RANK_IMAGE_ROLE),"format":RANK_IMAGE_FORMAT,
            "storage":{"kind":"rank-files","topology":shape.topology().id,
                "alignment_bytes":RANK_IMAGE_ALIGNMENT,"ranks":shape.ranks(),
                "views":if host_ngram { vec![json!({"id":text(NGRAM_HOST_VIEW),"alignment_bytes":RANK_IMAGE_ALIGNMENT,"placement":"host"})] } else { vec![] }}}),
        json!({"id":"frontend","role":"chop.frontend.openai.v1","format":"chop.frontend.json.v1","storage":{"kind":"whole"}}),
        json!({"id":"tokenizer-definition","role":"chop.tokenizer.definition.v1","format":"chop.tokenizer.json.v1","storage":{"kind":"whole"}}),
        json!({"id":"tokenizer-configuration","role":"chop.tokenizer.configuration.v1","format":"chop.tokenizer.json.v1","storage":{"kind":"whole"}}),
    ];
    for rank in 0..shape.ranks() {
        components.push(json!({"id":format!("rank-index{rank}"),"role":index_role(rank),"format":RANK_INDEX_FORMAT,"storage":{"kind":"whole"}}));
    }
    components.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let names: Vec<_> = components.iter().map(|c| c["id"].clone()).collect();
    let mut graphs = (0..=maximum_sequences(shape, false, Encoders::NONE).ilog2()).map(|power| json!({"id":format!("qwen38-decode-b{}",1<<power),"components":names,"operations":[]})).collect::<Vec<_>>();
    graphs.push(json!({"id":"qwen38-prefill","components":names,"operations":[]}));
    graphs.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    // The context-parallel sizes above 1 are listed only where served
    // (`context_parallel::sizes`), so a TP2 topology is unchanged.
    let topology = |shape: Shape| {
        let mut topology = json!({"id":shape.topology().id,"rank_count":shape.ranks()});
        let sizes = crate::context_parallel::sizes(shape);
        if !sizes.is_empty() {
            topology["context_parallel_sizes"] = json!(sizes);
        }
        topology
    };
    let mut declaration = json!({
        "id":BUNDLE_ID,"target":{"architectures":[Selected::ARCHITECTURE],"topologies":[topology(shape)]},
        "resource_requirements":{"minimum_device_memory_bytes":minimum_device_memory(shape, false, Encoders::NONE),"required_capabilities":capabilities},
        "kv_layout":kv(0),"kv_formats":[{"dtype":"bf16","layout":kv(0)},{"dtype":"fp8_e4m3","layout":kv(1)}],
        "state_layout":{"id":text(RING_STATE_ID),"bytes":shape.ring_state_slot_bytes(false),"alignment_bytes":256,"lifetime":"process"},
        "recurrent_state_layouts":recurrent,"components":components,"graph_buckets":graphs,
        "batch_limits":{"maximum_sequences":maximum_sequences(shape, false, Encoders::NONE),"maximum_prefill_tokens":PREFILL_TOKENS,"maximum_scored_rows":SCORED_ROWS,"prefill_granule":PREFILL_GRANULE,"anchor_items":ANCHOR_ITEMS},
        "speculation":{"maximum_draft_tokens":MAX_DRAFT_TOKENS,"bundle_drafter":false,"maximum_verify_tokens":MAX_VERIFY_TOKENS,"row_buckets":VERIFY_ROW_BUCKETS,"maximum_draft_items":DRAFT_ITEMS},
        "operation_requirements":[],"implementation_count":0,"native_symbol_sequences":[],"native_entry":"chop_bundle_get_api_v1"
    });
    if !options.is_empty() {
        declaration["model_options"] = json!(runtime_bundle::model_options::ModelOptionsV1 {
            schema: runtime_bundle::model_options::CAPABILITY.into(),
            options,
        });
    }
    declaration
}

/// Prepared packages add one authenticated description. Legacy declarations
/// remain byte-for-byte stable for the existing release path.
pub fn prepared_declarations(
    shape: Shape,
    host_ngram: bool,
    runtime: crate::geometry::RuntimeValues,
) -> Value {
    let mut value = declarations(shape, host_ngram, runtime);
    let components = value["components"].as_array_mut().expect("components");
    components.push(json!({"id":crate::prepared::component_id(shape),
        "role":crate::prepared::COMPONENT_ROLE,"format":crate::prepared::COMPONENT_FORMAT,
        "storage":{"kind":"whole"},"topologies":[shape.topology().id]}));
    components.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    let names: Vec<_> = components.iter().map(|c| c["id"].clone()).collect();
    for graph in value["graph_buckets"].as_array_mut().expect("graphs") {
        graph["components"] = json!(names);
    }
    value
}

/// The declarations of a package without rank images, which serves only by
/// loading the checkpoint directly (`serve --model DIR`, or `serve --bundle
/// DIR --checkpoint DIR --profile NAME`): the rank-image
/// component is planned (declared by role and format, with no file), the
/// rank-index components are gone, both come from the bundle's weight plan,
/// and the package requires `chop.weight-plan.v1`.
pub fn without_rank_images(mut value: Value) -> Value {
    let components = value["components"].as_array_mut().expect("components");
    components.retain(|component| {
        !component["id"]
            .as_str()
            .unwrap_or_default()
            .starts_with("rank-index")
    });
    for component in components.iter_mut() {
        if component["id"] == "rank-images" {
            component["storage"] = json!({"kind": "planned"});
        }
    }
    let names: Vec<_> = components.iter().map(|c| c["id"].clone()).collect();
    for graph in value["graph_buckets"].as_array_mut().expect("graphs") {
        graph["components"] = json!(names);
    }
    let capabilities = value["resource_requirements"]["required_capabilities"]
        .as_array_mut()
        .expect("capabilities");
    capabilities.push(json!(weight_plan::WEIGHT_PLAN_CAPABILITY_V1));
    capabilities.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime(shape: Shape) -> crate::geometry::RuntimeValues {
        crate::geometry::RuntimeValues {
            indexer_budget: shape.geometry.limits.indexer_budget as usize,
        }
    }

    /// `declaration` before the frontend export's capability, the one change
    /// since the flash package pins: it must name the capability exactly
    /// once, and loses that one entry.
    fn without_frontend_capability(mut declaration: Value, name: &str) -> Value {
        let capabilities = declaration["resource_requirements"]["required_capabilities"]
            .as_array_mut()
            .unwrap();
        let frontend = json!(frontend::FRONTEND_CAPABILITY_V1);
        let positions: Vec<_> = capabilities
            .iter()
            .enumerate()
            .filter(|(_, capability)| **capability == frontend)
            .map(|(position, _)| position)
            .collect();
        let [position] = positions[..] else {
            panic!("{name} names {} frontend capabilities", positions.len());
        };
        capabilities.remove(position);
        declaration
    }

    #[test]
    fn flash_package_declarations_change_only_by_the_frontend_capability() {
        let pins: std::collections::BTreeMap<String, String> =
            serde_json::from_str(include_str!("../tests/fixtures/flash-package-pins.json"))
                .unwrap();
        let mut checked = 0;
        for shape in Shape::ALL {
            for host in [false, true] {
                for prepared in [false, true] {
                    let declaration = if prepared {
                        prepared_declarations(shape, host, runtime(shape))
                    } else {
                        declarations(shape, host, runtime(shape))
                    };
                    for direct in [false, true] {
                        let declaration = if direct {
                            without_rank_images(declaration.clone())
                        } else {
                            declaration.clone()
                        };
                        let name = format!(
                            "tp{}-host{host}-prepared{prepared}-direct{direct}",
                            shape.ranks()
                        );
                        // Without the frontend export's capability each
                        // declaration keeps its pinned digest.
                        let declaration = without_frontend_capability(declaration, &name);
                        assert_eq!(crate::note::digest(&declaration), pins[&name], "{name}");
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, pins.len());
    }

    /// The pins admit the frontend capability once: a declaration naming it
    /// twice is not the pinned declaration plus one capability.
    #[test]
    #[should_panic(expected = "names 2 frontend capabilities")]
    fn a_declaration_naming_the_frontend_capability_twice_is_not_pinned() {
        let mut declaration = declarations(Shape::TP4, false, runtime(Shape::TP4));
        declaration["resource_requirements"]["required_capabilities"]
            .as_array_mut()
            .unwrap()
            .push(json!(frontend::FRONTEND_CAPABILITY_V1));
        without_frontend_capability(declaration, "doubled");
    }

    /// TP4 declares context-parallel size 2 (its replica sets); TP2 lists
    /// no sizes, so its declarations keep their bytes.
    #[test]
    fn only_tp4_declares_context_parallel_sizes() {
        for shape in Shape::ALL {
            let declaration = declarations(shape, false, runtime(shape));
            let topology = &declaration["target"]["topologies"][0];
            if shape.ranks() == 4 {
                assert_eq!(topology["context_parallel_sizes"], json!([2]));
            } else {
                assert!(topology.get("context_parallel_sizes").is_none());
            }
        }
    }

    #[test]
    fn named_rows_fit_every_geometry_and_count() {
        for geometry in crate::geometry::GEOMETRIES {
            for ranks in [Ranks::TP2, Ranks::TP4] {
                let shape = Shape::new(geometry, ranks);
                for sequences in 1..=DECODE_BATCH {
                    assert!(
                        working_bytes(shape, sequences) >= named_working_bytes(shape, sequences),
                        "{} TP{} C{}",
                        geometry.id,
                        shape.ranks(),
                        sequences
                    );
                }
            }
        }
    }

    #[test]
    fn flash_capacity_matches_measured_bounds() {
        for shape in Shape::ALL {
            for sequences in [1_u32, 32, 64, 128, 256] {
                let extra = u64::from(sequences.next_power_of_two().max(32) - 32);
                assert_eq!(
                    working_bytes(shape, sequences),
                    (4480 << 20)
                        + ((73 << 29) + (232 << 20)) / u64::from(shape.ranks())
                        + extra * ((10 << 20) + (215 << 20) / u64::from(shape.ranks()))
                );
            }
        }
        assert_eq!(
            crate::encoder::buffer_bytes(&crate::geometry::FLASH),
            2_101_411_856
        );
    }

    #[test]
    fn image_free_declarations_plan_the_image_drop_the_indexes_and_require_the_plan() {
        for shape in Shape::ALL {
            let full = declarations(shape, ngram_in_host_memory(shape), runtime(shape));
            let direct = without_rank_images(full.clone());
            let ids = |v: &Value| -> Vec<String> {
                v["components"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c["id"].as_str().unwrap().to_owned())
                    .collect()
            };
            assert!(ids(&direct).iter().all(|id| !id.starts_with("rank-index")));
            assert_eq!(
                ids(&direct).len() + shape.ranks() as usize,
                ids(&full).len()
            );
            let image = direct["components"]
                .as_array()
                .unwrap()
                .iter()
                .find(|c| c["id"] == "rank-images")
                .unwrap();
            assert_eq!(image["storage"], json!({"kind": "planned"}));
            for graph in direct["graph_buckets"].as_array().unwrap() {
                assert_eq!(
                    graph["components"].as_array().unwrap().len(),
                    ids(&direct).len()
                );
            }
            let capabilities = direct["resource_requirements"]["required_capabilities"]
                .as_array()
                .unwrap();
            assert!(capabilities.contains(&json!(weight_plan::WEIGHT_PLAN_CAPABILITY_V1)));
            assert!(
                capabilities
                    .windows(2)
                    .all(|pair| pair[0].as_str() < pair[1].as_str())
            );
            let prepared = without_rank_images(prepared_declarations(shape, false, runtime(shape)));
            assert!(ids(&prepared).contains(&crate::prepared::component_id(shape)));
        }
    }

    /// TP2 keeps its n-gram table in host memory; TP4 keeps its
    /// on the device, with room to spare, and every floor fits the device.
    #[test]
    fn only_tp2_needs_the_ngram_table_in_host_memory() {
        assert!(ngram_in_host_memory(Shape::TP2));
        assert!(!ngram_in_host_memory(Shape::TP4));
        let room = |shape| rank_budget() - rank_floor(shape, false, BASE_SEQUENCES, Encoders::NONE);
        assert!(room(Shape::TP4) > 24 << 30);
    }

    /// Each package serves the most sequences its floor fits: 256 (128 with
    /// MTP) at TP4; at TP2, whose floor at 128
    /// does not fit, 64 (32 with the encoder).
    #[test]
    fn maximum_sequences_are_the_largest_that_fit() {
        for mtp in [false, true] {
            let limit = if mtp {
                MTP_QUALIFIED_SEQUENCES
            } else {
                DECODE_BATCH
            };
            for encoders in Encoders::SETS {
                let media = encoders.vision;
                assert_eq!(maximum_sequences(Shape::TP4, mtp, encoders), limit);
                assert_eq!(
                    maximum_sequences(Shape::TP2, mtp, encoders),
                    if media { 32 } else { 64 }
                );
                assert!(minimum_device_memory(Shape::TP2, mtp, encoders) <= DEVICE_BYTES);
            }
        }
        assert!(rank_floor(Shape::TP2, true, 128, Encoders::NONE) > DEVICE_BYTES);
        assert!(rank_floor(Shape::TP2, true, 64, Encoders::VISION) > DEVICE_BYTES);
    }
}
