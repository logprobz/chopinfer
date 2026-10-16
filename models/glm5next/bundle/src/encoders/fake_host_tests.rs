//! The vision tower on a CPU fake host, over four ranks: the encoder image
//! and its rank indexes served as a compiled package would, the encoder
//! acquired and its cubins loaded, then a small image (8 x 8 patches, 16
//! tokens) and the largest (160 x 200 patches, 8000 tokens) encoded in
//! two prefill steps. Every rank runs the whole tower, the zero-padding
//! stage before the patch GEMM, and uploads the prefill's arena source
//! rows; the journals pin what reaches the device.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::acquire::{self, Log};
use bundle_runtime::fake_host::{FakeDevice, check_journal, fnv1a};
use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::pipeline::Collectives;
use bundle_runtime::recorded::UPLOAD_CHUNK;
use bundle_runtime::resources::Held;
use runtime_bundle_abi::*;
use runtime_bundle_sdk::{
    BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
    BatchRequestV1, BatchWireVersionV1,
};

use super::*;

/// The SM120 target the encoder runs on.
fn target() -> std::sync::Arc<dyn Target> {
    crate::instance::target(model_glm5next_contract::contract::Shape::TP4)
}

const RANKS: u32 = 4;
const IMAGE_SLOT: u32 = 1_000_000;
const INDEX_COMPONENT: u32 = 100;
const IDENTITY_COMPONENT: u32 = 200;
/// The image file's digest the indexes and identities name.
const SHA256: &str = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";

// The stems of the tower's entries (`Entry::stem`).
const PAD: &str = "chop_enc_pad_zero_columns_w1176_to1216_v1";
const PATCH: &str = "chop_enc_linear_pairs_n1024_k1216_bias_f32_v1";
const ATTENTION: &str = "chop_enc_attention_varlen_qk_rms_pairs_q16_kv16_d64_full_v1";

#[derive(Default)]
struct FakeHost {
    device: FakeDevice,
    /// Bytes of every buffer slot handed out, for the span checks.
    extents: RefCell<BTreeMap<u32, u64>>,
    /// The payload bytes of the record a sweep encodes, whose uploads (in
    /// windows of `UPLOAD_CHUNK` bytes) journal their destination and the
    /// FNV-1a of their first 4 KiB, not of every byte (the sweeps);
    /// zero hashes every upload whole.
    payload: Cell<u64>,
}

impl FakeHost {
    fn image_bytes() -> u64 {
        views::image_bytes().unwrap()
    }

    fn check_span(&self, span: DeviceBufferSpanV1) {
        let extent = self.extents.borrow().get(&span.buffer.slot()).copied();
        let extent = extent.unwrap_or_else(|| panic!("unknown buffer {}", span.buffer.slot()));
        assert!(span.offset + span.length <= extent, "span past its buffer");
    }

    /// The role and bytes of component handle `slot`.
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        let image = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
        if let Some(rank) = slot.checked_sub(IDENTITY_COMPONENT) {
            let identity = serde_json::json!({
                "byte_length": Self::image_bytes(),
                "sha256": SHA256,
                "direct_identity": null,
            });
            return (
                format!("chop.compiled-image-identity.v1/{image}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        let rank = slot - INDEX_COMPONENT;
        let index = views::index(
            rank,
            bundle_runtime::index::encoder::Identity::Sha256(SHA256),
        )
        .unwrap();
        (views::index_role(rank), serde_json::to_vec(&index).unwrap())
    }
}

impl HostServicesBackendV1 for FakeHost {
    fn log(&self, message: &[u8]) -> HostServiceResultV1<()> {
        self.device.log(message);
        Ok(())
    }
    fn select_context(
        &self,
        rank: u32,
        _device_ordinal: i32,
    ) -> HostServiceResultV1<BundleContextHandleV1> {
        Ok(BundleContextHandleV1::from_parts(1, rank + 1))
    }
    fn release_context(&self, _context: BundleContextHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn select_memory_pool(
        &self,
        context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleMemoryPoolHandleV1> {
        Ok(self.device.memory_pool(context))
    }
    fn release_memory_pool(&self, _pool: BundleMemoryPoolHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn allocate_device_buffer(
        &self,
        _pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        byte_length: u64,
        alignment: u64,
    ) -> HostServiceResultV1<BundleDeviceBufferHandleV1> {
        let id = self.device.id();
        self.extents.borrow_mut().insert(id, byte_length);
        Ok(self.device.allocate(id, stream, byte_length, alignment))
    }
    fn release_device_buffer(&self, buffer: BundleDeviceBufferHandleV1) -> HostServiceResultV1<()> {
        self.extents.borrow_mut().remove(&buffer.slot());
        Ok(())
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        assert_eq!(role, views::IMAGE_ROLE);
        Ok(BundleCompiledImageHandleV1::from_parts(1, self.device.id()))
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        self.device.note(format!("upload encoder arena {rank}"));
        Ok(BundleRankArenaHandleV1::from_parts(1, self.device.id()))
    }
    fn release_rank_arena(&self, _arena: BundleRankArenaHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn resolve_compiled_image_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        if !view_id.is_empty() {
            return Err(BundleStatusV1::NotFound);
        }
        let bytes = Self::image_bytes();
        self.extents.borrow_mut().insert(IMAGE_SLOT + rank, bytes);
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, IMAGE_SLOT + rank),
                0,
                bytes,
            ),
            4096,
        ))
    }
    fn release_compiled_image(
        &self,
        _image: BundleCompiledImageHandleV1,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        let image = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
        let slot = (0..RANKS)
            .find_map(|rank| {
                if role == views::index_role(rank).as_bytes() {
                    Some(INDEX_COMPONENT + rank)
                } else if role
                    == format!("chop.compiled-image-identity.v1/{image}/{rank}").as_bytes()
                {
                    Some(IDENTITY_COMPONENT + rank)
                } else {
                    None
                }
            })
            .ok_or(BundleStatusV1::NotFound)?;
        Ok(BundleAuthenticatedComponentHandleV1::from_parts(1, slot))
    }
    fn inspect_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<AuthenticatedComponentMetadataV1> {
        let (role, bytes) = self.component(component.slot());
        Ok(AuthenticatedComponentMetadataV1::new(
            role.len() as u64,
            bytes.len() as u64,
            [b'a'; 64],
        ))
    }
    fn read_authenticated_component(
        &self,
        component: BundleAuthenticatedComponentHandleV1,
        role_destination: &mut [u8],
        bytes_destination: &mut [u8],
    ) -> HostServiceResultV1<()> {
        let (role, bytes) = self.component(component.slot());
        role_destination.copy_from_slice(role.as_bytes());
        bytes_destination.copy_from_slice(&bytes);
        Ok(())
    }
    fn release_authenticated_component(
        &self,
        _component: BundleAuthenticatedComponentHandleV1,
    ) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn create_stream(
        &self,
        context: BundleContextHandleV1,
        _stream_kind: u32,
    ) -> HostServiceResultV1<BundleStreamHandleV1> {
        Ok(self.device.create_stream(self.device.id(), context))
    }
    fn synchronize_stream(&self, stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        self.device.synchronize_stream(stream);
        Ok(())
    }
    fn release_stream(&self, _stream: BundleStreamHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn create_event(
        &self,
        _context: BundleContextHandleV1,
    ) -> HostServiceResultV1<BundleEventHandleV1> {
        Ok(self.device.create_event(self.device.id()))
    }
    fn record_event(
        &self,
        event: BundleEventHandleV1,
        stream: BundleStreamHandleV1,
    ) -> HostServiceResultV1<()> {
        self.device.record_event(event, stream)
    }
    fn wait_event(
        &self,
        stream: BundleStreamHandleV1,
        event: BundleEventHandleV1,
    ) -> HostServiceResultV1<()> {
        self.device.wait_event(stream, event)
    }
    fn release_event(&self, _event: BundleEventHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn copy_host_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.check_span(destination);
        let (payload, window) = (self.payload.get(), UPLOAD_CHUNK as u64);
        let length = source.len() as u64;
        let hashed =
            match payload > 0 && (length == payload.min(window) || length == payload % window) {
                true => &source[..source.len().min(4096)],
                false => source,
            };
        self.device.note(format!(
            "upload {} {destination:?} {:016x}",
            stream.slot(),
            fnv1a(hashed)
        ));
        Ok(())
    }
    fn copy_device_to_host(
        &self,
        stream: BundleStreamHandleV1,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.check_span(source);
        self.device
            .note(format!("download {} {source:?}", stream.slot()));
        destination.fill(0);
        Ok(())
    }
    fn copy_device_to_device(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.check_span(source);
        self.check_span(destination);
        self.device
            .note(format!("copy {} {source:?} {destination:?}", stream.slot()));
        Ok(())
    }
    fn zero_device_buffer(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.check_span(destination);
        self.device.zero(stream, destination);
        Ok(())
    }
    fn load_cuda_module(
        &self,
        _context: BundleContextHandleV1,
        module_bytes: &[u8],
    ) -> HostServiceResultV1<BundleCudaModuleHandleV1> {
        Ok(self.device.load_module(
            self.device.id(),
            module_bytes,
            crate::CUBINS.iter().map(|cubin| (cubin.name, cubin.bytes)),
        ))
    }
    fn resolve_cuda_symbol(
        &self,
        module: BundleCudaModuleHandleV1,
        symbol_name: &[u8],
    ) -> HostServiceResultV1<BundleCudaSymbolHandleV1> {
        self.device
            .resolve_symbol(self.device.id(), module, symbol_name)
    }
    fn configure_cuda_symbol(
        &self,
        symbol: BundleCudaSymbolHandleV1,
        dynamic_shared_memory_bytes: u32,
    ) -> HostServiceResultV1<()> {
        assert!(dynamic_shared_memory_bytes <= 101_376);
        self.device
            .configure_symbol(symbol, dynamic_shared_memory_bytes);
        Ok(())
    }
    fn launch_cuda_symbol(
        &self,
        stream: BundleStreamHandleV1,
        symbol: BundleCudaSymbolHandleV1,
        config: LaunchConfigV1,
        arguments: &[KernelArgumentV1],
    ) -> HostServiceResultV1<()> {
        let name = self.device.symbol_name(symbol, BundleStatusV1::NotFound)?;
        for argument in arguments {
            if let KernelArgumentV1::DeviceBuffer(span) = argument {
                self.check_span(*span);
            }
        }
        self.device.launched(stream, &name, config, arguments);
        Ok(())
    }
    fn release_cuda_symbol(&self, _symbol: BundleCudaSymbolHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    fn release_cuda_module(&self, _module: BundleCudaModuleHandleV1) -> HostServiceResultV1<()> {
        Ok(())
    }
    bundle_runtime::unused_host_services! {
        set_memory_pool_peer_access(
            _pool: BundleMemoryPoolHandleV1,
            _peer_context: BundleContextHandleV1,
            _enabled: bool,
        ) -> ();
        acquire_peer_access(
            _owner_context: BundleContextHandleV1,
            _peer_context: BundleContextHandleV1,
        ) -> BundlePeerAccessHandleV1;
        release_peer_access(_peer_access: BundlePeerAccessHandleV1) -> ();
        copy_peer_to_peer(
            _stream: BundleStreamHandleV1,
            _destination_context: BundleContextHandleV1,
            _destination: DeviceBufferSpanV1,
            _source_context: BundleContextHandleV1,
            _source: DeviceBufferSpanV1,
        ) -> ();
        allocate_state(_request: &StateRequestV1) -> BundleStateHandleV1;
        resolve_state_view(_state: BundleStateHandleV1, _rank: u32) -> DeviceBufferSpanV1;
        release_state(_state: BundleStateHandleV1) -> ();
    }
    bundle_runtime::unused_host_services!();
}

/// The acquired encoder over four ranks, with its streams and symbols.
struct Served {
    table: HostServicesTableV1,
    encoder: Encoder,
    streams: Vec<BundleStreamHandleV1>,
    symbols: Vec<BTreeMap<String, BundleCudaSymbolHandleV1>>,
    _held: Vec<Held>,
}

fn serve(host: &Rc<FakeHost>) -> Served {
    let table = HostServicesTableV1::new(Rc::clone(host));
    let services = table.caller();
    let log = Log {
        services,
        bundle: LABEL,
    };
    let mut held = Vec::new();
    let ranks = acquire::open_ranks(log, &[0, 1, 2, 3], &mut held).unwrap();
    let encoder = acquire(target().as_ref(), services, &ranks, &mut held)
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.device.logs.borrow()));
    let symbols = load(target().as_ref(), log, &ranks, &mut held).unwrap();
    Served {
        streams: ranks.iter().map(|&(_, _, stream)| stream).collect(),
        table,
        encoder,
        symbols,
        _held: held,
    }
}

/// The payload bytes every record carries a prefix of: byte `i` is
/// `i % 251`, up to the payload of one merge block past the largest image.
fn pattern() -> &'static [u8] {
    static PATTERN: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| {
        let bytes = 8001 * 4 * u64::from(PATCH_COLUMNS) * 4;
        (0..bytes).map(|byte| (byte % 251) as u8).collect()
    })
}

/// One prefill item: a text row, the image's `tokens` pads and a text row,
/// with the image encoded into pages from `page` on and injected at row 1.
fn step(grid: [u32; 3], page: u64) -> (BatchRequestV1, PhaseInputs) {
    let patches = u64::from(grid[1] * grid[2]);
    let tokens = (patches / 4) as u32;
    let payload_bytes = patches * u64::from(PATCH_COLUMNS) * 4;
    let pages = u64::from(tokens).div_ceil(PAGE_ROWS);
    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [grid[0], grid[1], grid[2], 0],
            output_rows: tokens,
            pages: (page..page + pages).collect(),
            payload_offset: 0,
            payload_bytes,
        }],
        injects: (0..pages)
            .map(|at| BatchMediaInjectV3 {
                item_index: 0,
                item_row: 1 + (at * PAGE_ROWS) as u32,
                rows: (u64::from(tokens) - at * PAGE_ROWS).min(PAGE_ROWS) as u32,
                arena_row: (page + at) * PAGE_ROWS,
            })
            .collect(),
        payload: pattern()[..payload_bytes as usize].to_vec(),
    };
    let rows = tokens + 2;
    let token_ids: Vec<u32> = (0..rows).map(|row| 10 + row % 7).collect();
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(7, false, rows, true, token_ids.clone())],
        vec![BatchAppendV1::new(
            0,
            u64::from(rows),
            u64::from(rows),
            (0..u64::from(rows).div_ceil(64)).collect(),
            None,
        )],
    )
    .with_media(media);
    let prefill = PhaseInputs {
        item_order: vec![0],
        request_ids: vec![7],
        positions: (0..rows as i32).collect(),
        q_start: vec![0],
        q_len: vec![rows as i32],
        token_ids,
        ..PhaseInputs::default()
    };
    (request, prefill)
}

/// The full symbol of the entry whose symbol reads `stem`.
fn full(stem: &str) -> &'static str {
    crate::sm120::encoders::full(stem)
}

/// Runs one step's encode and returns its journal.
fn encode(host: &FakeHost, served: &Served, grid: [u32; 3], page: u64) -> Vec<String> {
    let from = host.device.journal.borrow().len();
    let (request, prefill) = step(grid, page);
    let injects = served
        .encoder
        .run(
            served.table.caller(),
            &served.streams,
            &served.symbols,
            &request.encode().unwrap(),
            &prefill,
            &Collectives::default(),
        )
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.device.logs.borrow()));
    assert!(injects);
    host.device.journal.borrow()[from..].to_vec()
}

/// Launch names per stream, in issue order.
fn by_stream(journal: &[String]) -> BTreeMap<u32, Vec<String>> {
    let mut streams = BTreeMap::<u32, Vec<String>>::new();
    for line in journal {
        let mut words = line.split(' ');
        if words.next() == Some("launch") {
            let stream = words.next().unwrap().parse().unwrap();
            let name = words.next().unwrap().to_owned();
            streams.entry(stream).or_default().push(name);
        }
    }
    streams
}

/// Every rank runs the same tower, padding the patches before the patch
/// GEMM and running 24 attention blocks, and uploads the arena row of
/// every image token, in prompt order, as the prefill's source rows.
fn check_tower(journal: &[String], grid: [u32; 3], page: u64) {
    let streams = by_stream(journal);
    assert_eq!(streams.len(), RANKS as usize);
    let names: Vec<&Vec<String>> = streams.values().collect();
    assert!(names.iter().all(|launches| *launches == names[0]));
    let launches = names[0];
    assert_eq!(launches[0], full(PAD));
    assert_eq!(launches[1], full(PATCH));
    assert_eq!(launches.iter().filter(|name| *name == full(PAD)).count(), 1);
    assert_eq!(
        launches
            .iter()
            .filter(|name| *name == full(ATTENTION))
            .count(),
        24
    );
    let tokens = u64::from(grid[1] * grid[2] / 4);
    let mut sources = vec![-1_i64];
    sources.extend((0..tokens).map(|row| (page * PAGE_ROWS + row) as i64));
    sources.push(-1);
    let hash = format!(
        "{:016x}",
        fnv1a(
            &sources
                .iter()
                .flat_map(|row| row.to_le_bytes())
                .collect::<Vec<_>>()
        )
    );
    let uploads = journal
        .iter()
        .filter(|line| line.starts_with("upload ") && line.ends_with(&hash))
        .count();
    assert_eq!(uploads, RANKS as usize);
}

/// The encoder's create (index and identity checks, the image on every
/// rank, its working memory and the arena, the two cubins), then a small
/// and the largest image, each journal pinned.
#[test]
fn the_tower_journals_a_small_and_the_largest_image() {
    let host = Rc::new(FakeHost::default());
    let served = serve(&host);
    let create = host.device.journal.borrow().clone();
    // Each rank allocates its working memory, arena and source rows once.
    let allocated: u64 = create
        .iter()
        .filter_map(|line| line.strip_prefix("allocate "))
        .map(|line| line.split(' ').next().unwrap().parse::<u64>().unwrap())
        .sum();
    assert_eq!(allocated, u64::from(RANKS) * buffer_bytes());
    check_journal(
        &create,
        (160, 0x8f56e7dd588b6be8),
        "glm5next-encoder-create.txt",
    )
    .unwrap();

    let small = encode(&host, &served, [1, 8, 8], 3);
    check_tower(&small, [1, 8, 8], 3);
    check_journal(
        &small,
        (1032, 0xed18dab7e9f7931f),
        "glm5next-encoder-small.txt",
    )
    .unwrap();

    let large = encode(&host, &served, [1, 160, 200], 0);
    check_tower(&large, [1, 160, 200], 0);
    check_journal(
        &large,
        (1668, 0xca77d11e91dd0333),
        "glm5next-encoder-large.txt",
    )
    .unwrap();
}

/// A record its declaration refuses (an odd grid) fails before any device
/// work, as does an image larger than the token budget.
#[test]
fn refused_records_reach_no_device() {
    let host = Rc::new(FakeHost::default());
    let served = serve(&host);
    for grid in [[1, 7, 8], [1, 160, 202]] {
        let from = host.device.journal.borrow().len();
        let (mut request, prefill) = step([1, 8, 8], 0);
        let mut media = request.media().clone();
        let patches = u64::from(grid[1] * grid[2]);
        media.encodes[0].geometry = [grid[0], grid[1], grid[2], 0];
        media.encodes[0].payload_bytes = patches * u64::from(PATCH_COLUMNS) * 4;
        media.payload = vec![0; media.encodes[0].payload_bytes as usize];
        request = request.with_media(media);
        let status = served.encoder.run(
            served.table.caller(),
            &served.streams,
            &served.symbols,
            &request.encode().unwrap(),
            &prefill,
            &Collectives::default(),
        );
        assert_eq!(status, Err(BundleStatusV1::InvalidArgument), "{grid:?}");
        assert_eq!(host.device.journal.borrow().len(), from, "{grid:?}");
    }
}

/// The prefill's inject reads the source rows of every prefill row.
#[test]
fn every_rank_prepares_the_prefill_inject() {
    let host = Rc::new(FakeHost::default());
    let served = serve(&host);
    let buffer = BundleDeviceBufferHandleV1::from_parts(1, 7);
    for rank in &served.encoder.ranks {
        let embed = DeviceBufferSpanV1::new(buffer, 0, u64::from(PREFILL_TOKENS) * 4096 * 2);
        let live = DeviceBufferSpanV1::new(buffer, 0, 4);
        assert!(rank.inject(embed, live, PREFILL_TOKENS).is_ok());
        assert!(rank.inject(embed, live, PREFILL_TOKENS + 1).is_err());
    }
}
