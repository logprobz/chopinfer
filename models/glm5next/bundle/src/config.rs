//! The create record of a `glm5next.v1` instance.
//!
//! The package requires the served item count and served context, so the
//! both. After them come the context-parallel size
//! (`runtime_bundle_sdk::context_parallel`, sent only above 1), the resolved
//! model options
//! (`chop.model-options.v1`, always sent: the package declares the
//! checkpoint's quantization layout and `index_topk`) and an optional tap
//! producer.

use bundle_runtime::config::{CreateConfig, MAGIC};
use bundle_runtime::kv::KvFormat;
use model_glm5next_contract::contract::{
    CONTEXT_PARALLEL_SIZES, Glm, INDEXER_TAIL_STATE_ID, PAGE_TOKENS, Shape, TOPOLOGY, has_mtp_layer,
};
use model_glm5next_contract::direct::{RANK_IMAGE_ROLE, index_role};
use model_glm5next_contract::geometry::RuntimeValues;
use model_glm5next_contract::options::{COMM_EPILOGUE_OPTION, comm_epilogue_value};
use model_glm5next_contract::quantization::{LAYOUT_OPTION, Layout};
use runtime_bundle_abi::{BundleStatusV1, MODEL_OPTIONS_TAG_V1};
use runtime_bundle_sdk::BatchReaderV1;
use runtime_bundle_sdk::context_parallel::{self, ContextParallelV1};

/// One instance's checked configuration.
#[derive(Clone, Debug)]
pub struct Configuration {
    pub shape: Shape,
    pub create: CreateConfig,
    pub format: KvFormat,
    /// The context-parallel groups, of size 1 when the record names none.
    pub context_parallel: ContextParallelV1,
    pub runtime: RuntimeValues,
    /// The checkpoint layout the package was emitted for.
    pub layout: &'static Layout,
    /// Whether the record names the vision encoder's image and every
    /// rank's index (`crate::encoders::configured`).
    pub media: bool,
    /// The resolved `runtime.sinkhorn_comm_epilogue` model option: each
    /// sublayer boundary runs the Sinkhorn mHC comm epilogue
    /// (`crate::lower`), and create loads its cubin.
    pub comm_epilogue: bool,
}

fn invalid<T>(_: T) -> BundleStatusV1 {
    BundleStatusV1::InvalidArgument
}

/// The length of the record's shared fields, through the served context.
fn shared_length(bytes: &[u8]) -> Result<usize, BundleStatusV1> {
    let mut reader = BatchReaderV1::new("bundle configuration", bytes);
    reader.expect_magic(MAGIC).map_err(invalid)?;
    let field = |reader: &mut BatchReaderV1| -> Result<(), BundleStatusV1> {
        let length = reader.count("identifier").map_err(invalid)?;
        reader.take(length).map(|_| ()).map_err(invalid)
    };
    field(&mut reader)?;
    let ranks = reader.count("rank count").map_err(invalid)?;
    reader.take(4 * ranks).map_err(invalid)?;
    // KV layout: its identifier and five u64 fields; state: its identifier
    // and two.
    field(&mut reader)?;
    reader.take(5 * 8).map_err(invalid)?;
    field(&mut reader)?;
    reader.take(2 * 8).map_err(invalid)?;
    let roles = reader.count("component roles").map_err(invalid)?;
    for _ in 0..roles {
        field(&mut reader)?;
    }
    // The served items and served context.
    reader.u32().map_err(invalid)?;
    reader.u64().map_err(invalid)?;
    Ok(reader.position())
}

/// Decodes and checks a create record against the package's declarations.
pub fn decode(bytes: &[u8], log: impl Fn(&str)) -> Result<Configuration, BundleStatusV1> {
    let shared = shared_length(bytes)?;
    let mut create =
        bundle_runtime::config::decode_with_log::<Glm>(&bytes[..shared], &[TOPOLOGY], &log)?;
    let shape = Shape::TP4;
    create.validate_context(shape.model_limits())?;
    let mut reader = BatchReaderV1::new("bundle configuration tail", &bytes[shared..]);
    let context_parallel = if context_parallel::is_next(&bytes[shared..]) {
        context_parallel::read(&mut reader, shape.ranks(), &CONTEXT_PARALLEL_SIZES)
            .inspect_err(|_| log("the context-parallel size is not one this package declares"))?
    } else {
        ContextParallelV1::NONE
    };
    if !bytes[shared + reader.position()..].starts_with(MODEL_OPTIONS_TAG_V1) {
        log("the create record carries no model options");
        return Err(BundleStatusV1::InvalidArgument);
    }
    let mut options = runtime_bundle_sdk::model_options::read(&mut reader)?;
    let layout = options
        .remove(LAYOUT_OPTION)
        .ok_or_else(|| format!("the model options name no {LAYOUT_OPTION}"))
        .and_then(|value| Layout::named(&value))
        .map_err(|error| {
            log(&error);
            BundleStatusV1::InvalidArgument
        })?;
    let comm_epilogue = comm_epilogue_value(options.remove(COMM_EPILOGUE_OPTION).as_ref())
        .map_err(|error| {
            log(&error);
            BundleStatusV1::InvalidArgument
        })?;
    let checkpoint = RuntimeValues {
        index_topk: shape.geometry.limits.index_topk,
    };
    let runtime = shape
        .geometry
        .resolve_model_options(checkpoint, &options)
        .map_err(|error| {
            log(&error);
            BundleStatusV1::InvalidArgument
        })?;
    if reader.remaining() != 0 {
        reader
            .expect_magic(runtime_bundle_sdk::taps::PRODUCER_TAG)
            .map_err(invalid)?;
        let length = reader.count("tap producer").map_err(invalid)?;
        let nonce = reader.take(length).map_err(invalid)?;
        create.tap_producer = Some(String::from_utf8(nonce.to_vec()).map_err(invalid)?);
    }
    reader.finish().map_err(invalid)?;
    let format = *shape
        .kv_formats()
        .iter()
        .find(|format| format.layout_id == create.kv.layout_id.as_slice())
        .ok_or(BundleStatusV1::Unsupported)?;
    let kv = &create.kv;
    let state = &create.state;
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
    // The encoder's image and indexes come whole or not at all.
    let Some(media) = crate::encoders::configured(&create.roles, shape.ranks()) else {
        log("the create record names part of the vision encoder's components");
        return Err(BundleStatusV1::Unsupported);
    };
    if kv.page_bytes != format.page_bytes
        || kv.page_tokens != PAGE_TOKENS
        || kv.ranks != u64::from(shape.ranks())
        || kv.lifetime != 0
        || kv.alignment < 256
        || !kv.alignment.is_power_of_two()
        || state.state_id != INDEXER_TAIL_STATE_ID.as_bytes()
        || state.bytes
            != u64::from(create.items.capacity())
                * shape.indexer_tail_slot_bytes(has_mtp_layer(&format))
        || state.alignment < 256
        || !state.alignment.is_power_of_two()
        || !has_roles
    {
        log("the create record differs from the package's declarations");
        return Err(BundleStatusV1::Unsupported);
    }
    Ok(Configuration {
        shape,
        create,
        format,
        context_parallel,
        runtime,
        layout,
        media,
        comm_epilogue,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use bundle_runtime::config::{KvLayout, StateLayout, test_support::encode};
    use model_glm5next_contract::contract::DECODE_BATCH;

    /// A record for the FP8 layout serving `items`, with `tail` appended
    /// after the served context.
    pub(crate) fn record(items: u32, tail: &[u8]) -> Vec<u8> {
        record_with(items, tail, Shape::TP4.kv_formats()[0])
    }

    /// [`record`] over the KV `format`.
    pub(crate) fn record_with(items: u32, tail: &[u8], format: KvFormat) -> Vec<u8> {
        record_naming(items, tail, format, &[])
    }

    /// The vision encoder's image role and every rank's index role.
    pub(crate) fn encoder_roles() -> Vec<Vec<u8>> {
        let mut roles = vec![crate::encoders::views::IMAGE_ROLE.to_vec()];
        roles.extend((0..4).map(|rank| crate::encoders::views::index_role(rank).into_bytes()));
        roles
    }

    /// [`record_with`] naming the components `extra` too.
    pub(crate) fn record_naming(
        items: u32,
        tail: &[u8],
        format: KvFormat,
        extra: &[Vec<u8>],
    ) -> Vec<u8> {
        let shape = Shape::TP4;
        let capacity = u64::from(items.next_power_of_two());
        let mut roles = vec![RANK_IMAGE_ROLE.as_bytes().to_vec()];
        roles.extend((0..4).map(|rank| index_role(rank).into_bytes()));
        roles.extend_from_slice(extra);
        let mut bytes = encode(
            TOPOLOGY.id.as_bytes(),
            &[0, 1, 2, 3],
            &KvLayout {
                layout_id: format.layout_id.to_vec(),
                page_bytes: format.page_bytes,
                alignment: 256,
                page_tokens: PAGE_TOKENS,
                ranks: 4,
                lifetime: 0,
            },
            &StateLayout {
                state_id: INDEXER_TAIL_STATE_ID.as_bytes().to_vec(),
                bytes: capacity
                    * shape.indexer_tail_slot_bytes(
                        model_glm5next_contract::contract::has_mtp_layer(&format),
                    ),
                alignment: 256,
            },
            &roles,
            Some(items),
        );
        bytes.extend_from_slice(&65_536_u64.to_le_bytes());
        bytes.extend_from_slice(tail);
        bytes
    }

    pub(crate) fn options(top_k: u32) -> Vec<u8> {
        options_of("fp8-block128", top_k)
    }

    /// Model options naming the layout `layout`.
    pub(crate) fn options_of(layout: &str, top_k: u32) -> Vec<u8> {
        encoded(&format!(
            "{{\"{LAYOUT_OPTION}\":\"{layout}\",\"text_config.index_topk\":{top_k}}}"
        ))
    }

    /// The block-FP8 options with the comm epilogue option set to `value`
    /// (a JSON value).
    pub(crate) fn options_comm_epilogue(top_k: u32, value: &str) -> Vec<u8> {
        encoded(&format!(
            "{{\"{LAYOUT_OPTION}\":\"fp8-block128\",\"{COMM_EPILOGUE_OPTION}\":{value},\
             \"text_config.index_topk\":{top_k}}}"
        ))
    }

    fn encoded(json: &str) -> Vec<u8> {
        let mut bytes = MODEL_OPTIONS_TAG_V1.to_vec();
        bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
        bytes.extend_from_slice(json.as_bytes());
        bytes
    }

    /// The comm epilogue option reads as a Boolean, off when the record
    /// omits it; any other value is refused.
    #[test]
    fn records_read_the_comm_epilogue_option() {
        let read = |options: &[u8]| decode(&record(8, options), |_| {});
        assert!(!read(&options(2048)).unwrap().comm_epilogue);
        assert!(
            !read(&options_comm_epilogue(2048, "false"))
                .unwrap()
                .comm_epilogue
        );
        let on = read(&options_comm_epilogue(1024, "true")).unwrap();
        assert!(on.comm_epilogue);
        assert_eq!(on.runtime.index_topk, 1024);
        for value in ["1", "\"true\"", "null"] {
            assert_eq!(
                read(&options_comm_epilogue(2048, value)).unwrap_err(),
                BundleStatusV1::InvalidArgument,
                "{value}"
            );
        }
    }

    #[test]
    fn records_read_the_context_parallel_size_and_model_options() {
        let config = decode(&record(DECODE_BATCH, &options(2048)), |_| {}).unwrap();
        assert_eq!(config.context_parallel, ContextParallelV1::NONE);
        assert_eq!(config.runtime.index_topk, 2048);
        assert_eq!(config.create.items.get(), DECODE_BATCH);
        let mut tail = context_parallel::TAG.to_vec();
        tail.extend_from_slice(&4_u32.to_le_bytes());
        tail.extend(options(1024));
        let config = decode(&record(8, &tail), |_| {}).unwrap();
        assert_eq!(
            (config.context_parallel.size(), config.runtime.index_topk),
            (4, 1024)
        );
    }

    /// The tap producer's nonce after the model options is kept; one that
    /// is not UTF-8 is refused.
    #[test]
    fn records_keep_a_utf8_tap_producer() {
        let producer = |nonce: &[u8]| {
            let mut tail = options(2048);
            tail.extend_from_slice(runtime_bundle_sdk::taps::PRODUCER_TAG);
            tail.extend_from_slice(&(nonce.len() as u32).to_le_bytes());
            tail.extend_from_slice(nonce);
            decode(&record(8, &tail), |_| {})
        };
        assert_eq!(
            producer(b"run-7").unwrap().create.tap_producer.as_deref(),
            Some("run-7")
        );
        assert_eq!(
            producer(&[0xff]).unwrap_err(),
            BundleStatusV1::InvalidArgument
        );
        let config = decode(&record(8, &options(2048)), |_| {}).unwrap();
        assert_eq!(config.create.tap_producer, None);
    }

    /// A record names the vision encoder's image and every rank's index,
    /// or none of them: a partial set is refused before create acquires
    /// anything.
    #[test]
    fn records_name_the_whole_encoder_or_none() {
        let format = Shape::TP4.kv_formats()[0];
        let config = decode(&record(8, &options(2048)), |_| {}).unwrap();
        assert!(!config.media);
        let roles = encoder_roles();
        let config = decode(&record_naming(8, &options(2048), format, &roles), |_| {}).unwrap();
        assert!(config.media);
        for missing in 0..roles.len() {
            let mut partial = roles.clone();
            partial.remove(missing);
            assert_eq!(
                decode(&record_naming(8, &options(2048), format, &partial), |_| {}).unwrap_err(),
                BundleStatusV1::Unsupported
            );
        }
    }

    #[test]
    fn bad_sizes_and_missing_or_bad_options_are_refused() {
        // Every size dividing the 4 ranks is declared; others are malformed.
        for size in [1_u32, 3, 8] {
            let mut tail = context_parallel::TAG.to_vec();
            tail.extend_from_slice(&size.to_le_bytes());
            tail.extend(options(2048));
            assert_eq!(
                decode(&record(8, &tail), |_| {}).unwrap_err(),
                BundleStatusV1::InvalidArgument
            );
        }
        assert_eq!(
            decode(&record(8, &[]), |_| {}).unwrap_err(),
            BundleStatusV1::InvalidArgument
        );
        assert_eq!(
            decode(&record(8, &options(2047)), |_| {}).unwrap_err(),
            BundleStatusV1::InvalidArgument
        );
        // The layout must be named, and be an admitted one.
        let topk_only = b"{\"text_config.index_topk\":2048}";
        let mut bytes = MODEL_OPTIONS_TAG_V1.to_vec();
        bytes.extend_from_slice(&(topk_only.len() as u32).to_le_bytes());
        bytes.extend_from_slice(topk_only);
        assert_eq!(
            decode(&record(8, &bytes), |_| {}).unwrap_err(),
            BundleStatusV1::InvalidArgument
        );
        assert_eq!(
            decode(&record(8, &options_of("int4", 2048)), |_| {}).unwrap_err(),
            BundleStatusV1::InvalidArgument
        );
    }

    #[test]
    fn records_read_the_checkpoint_layout() {
        use model_glm5next_contract::quantization::{FP8_BLOCK128, NVFP4_ROUTED};
        for layout in [FP8_BLOCK128, NVFP4_ROUTED] {
            let config = decode(&record(8, &options_of(layout.id, 2048)), |_| {}).unwrap();
            assert_eq!(*config.layout, layout);
        }
    }
}
