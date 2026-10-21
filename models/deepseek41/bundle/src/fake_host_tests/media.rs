//! An image prefill on the deepseek41 fake host, over the native KV layout
//! at two and four ranks: the package's fake host also serves the vision
//! encoder's image and its rank indexes, the create record names them, and
//! a prompt holding one image span (7 x 5 patches, 3 x 2 cells, 11 rows
//! with the learned delimiters) prefills, then decodes, then a text prompt
//! prefills on the same instance. The journals pin what reaches the device:
//! the encoder's run, each rank's arena source rows, the router's image
//! rows and the Engram gate's text flags, the injection after the
//! embedding's all-reduce, the masked gates, and the router's flags
//! returned to text after the prefill.

use bundle_runtime::fake_host::{check_journal, fnv1a};
use bundle_runtime::index::encoder::Identity;
use runtime_bundle_sdk::{
    BatchAppendV1, BatchItemV1, BatchMediaEncodeV3, BatchMediaInjectV3, BatchMediaSectionV3,
    BatchRequestV1, BatchWireVersionV1,
};

use super::*;
use crate::encoders::{PAGE_ROWS, PATCH_COLUMNS, views};

const ENCODER_SLOT: u32 = 4_000_000;
const ENCODER_INDEX: u32 = 300;
const ENCODER_IDENTITY: u32 = 400;
/// The encoder image's digest its indexes and identities name.
const SHA256: &str = "5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a";

/// The image's patch grid, `[t, h, w]`.
const GRID: [u32; 3] = [1, 7, 5];
/// Its rows: 3 x 2 cells, a newline after each row of cells, and the
/// start and end rows.
const IMAGE_ROWS: u32 = 3 * (2 + 1) + 2;
/// The stem of the encoder's embedding override, the layout's entry.
const INJECT: &str = "chop_enc_layout_learned_delimiters";
/// The arena page the image is encoded into.
const ARENA_PAGE: u64 = 2;

/// The package's fake host, which also serves the encoder's image, its
/// rank indexes and their image identities.
struct MediaHost {
    host: FakeHost,
    /// The handle slot of the opened encoder image.
    image: Cell<Option<u32>>,
}

impl std::ops::Deref for MediaHost {
    type Target = FakeHost;

    fn deref(&self) -> &FakeHost {
        &self.host
    }
}

impl MediaHost {
    fn new(shape: Shape, kv: KvLayout) -> Rc<Backend<MediaHost>> {
        let host = FakeHost {
            device: FakeDevice::default(),
            logs: RefCell::new(Vec::new()),
            shape,
            kv,
            speculation: Cell::new(None),
            state_bytes: Cell::new(0),
            kv_pages: Cell::new(KV_PAGES),
            drafter: Cell::new(false),
            uploads: RefCell::new(std::collections::BTreeMap::new()),
            injected: RefCell::new(None),
            injected_after: Cell::new(0),
            memory: RefCell::new(Memory::default()),
            layouts: RefCell::new(std::collections::BTreeMap::new()),
        };
        Rc::new(Backend::new(MediaHost {
            host,
            image: Cell::new(None),
        }))
    }
}

impl FakePackage for MediaHost {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        self.host.device()
    }
    fn check_launch(&self, symbol: &str, config: LaunchConfigV1) {
        self.host.check_launch(symbol, config);
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        FakeHost::cubins()
    }
    fn log(&self, message: &[u8]) {
        self.host.log(message);
    }
    fn on_acquire(&self, handle: Handle) {
        self.host.on_acquire(handle);
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.host.read_back(destination, source)
    }
    fn on_upload(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.host.on_upload(stream, destination, source)
    }
    fn on_copy(
        &self,
        stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.host.on_copy(stream, destination, source)
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        if role != views::IMAGE_ROLE {
            return self.host.open_compiled_image(role);
        }
        let slot = self.id();
        self.image.set(Some(slot));
        Ok(BundleCompiledImageHandleV1::from_parts(1, slot))
    }
    fn upload_rank_arena(
        &self,
        context: BundleContextHandleV1,
        pool: BundleMemoryPoolHandleV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        self.host
            .upload_rank_arena(context, pool, stream, image, rank)
    }
    fn resolve_compiled_image_view(
        &self,
        image: BundleCompiledImageHandleV1,
        arena: BundleRankArenaHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        if self.image.get() != Some(image.slot()) {
            return self
                .host
                .resolve_compiled_image_view(image, arena, rank, view_id);
        }
        let bytes = views::image_bytes().unwrap();
        let slot = ENCODER_SLOT + rank;
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(CompiledImageSpanV1::new(
            rank,
            DeviceBufferSpanV1::new(BundleDeviceBufferHandleV1::from_parts(1, slot), 0, bytes),
            VIEW_ALIGNMENT,
        ))
    }
    fn open_authenticated_component(
        &self,
        role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        let image = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
        for rank in 0..self.shape.ranks() {
            if role == views::index_role(rank).as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    ENCODER_INDEX + rank,
                ));
            }
            if role == format!("chop.compiled-image-identity.v1/{image}/{rank}").as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    ENCODER_IDENTITY + rank,
                ));
            }
        }
        self.host.open_authenticated_component(role)
    }
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        let image = std::str::from_utf8(views::IMAGE_ROLE).unwrap();
        if let Some(rank) = slot.checked_sub(ENCODER_IDENTITY) {
            let identity = serde_json::json!({
                "byte_length": views::image_bytes().unwrap(),
                "sha256": SHA256,
                "direct_identity": null,
            });
            return (
                format!("chop.compiled-image-identity.v1/{image}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        if let Some(rank) = slot.checked_sub(ENCODER_INDEX) {
            let index = views::index(rank, Identity::Sha256(SHA256)).unwrap();
            return (views::index_role(rank), serde_json::to_vec(&index).unwrap());
        }
        self.host.component(slot)
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        self.host.acquire_serving_kv(layout_id)
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        self.host.resolve_serving_kv_pool(rank)
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        self.host.allocate_state(request)
    }
    fn resolve_state_view(
        &self,
        state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        self.host.resolve_state_view(state, rank)
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        self.host.query_speculation()
    }
}

/// A full instance serving images and `items` items.
fn create(host: &Rc<Backend<MediaHost>>, items: u32) -> BundleInstanceV1 {
    let record = crate::config::tests::media_record(&host.shape, host.kv, items);
    api()
        .create(HostServicesTableV1::new(Rc::clone(host)), &record)
        .unwrap_or_else(|error| panic!("media create: {error:?} {:?}", host.logs.borrow()))
}

/// The prompt of `request` in `slot`: a text row, the image's span and a
/// text row, scoring its last row; with each row's arena source (-1 for
/// a text row).
fn image_prompt(request: u64, slot: u64) -> (Vec<u8>, Vec<u32>, Vec<i64>) {
    let image_token = FLASH.shape.tokens.image;
    let patches = u64::from(GRID[1] * GRID[2]);
    let payload_bytes = patches * u64::from(PATCH_COLUMNS) * 4;
    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 0,
            geometry: [GRID[0], GRID[1], GRID[2], 0],
            output_rows: IMAGE_ROWS,
            pages: vec![ARENA_PAGE],
            payload_offset: 0,
            payload_bytes,
        }],
        injects: vec![BatchMediaInjectV3 {
            item_index: 0,
            item_row: 1,
            rows: IMAGE_ROWS,
            arena_row: ARENA_PAGE * PAGE_ROWS,
        }],
        payload: (0..payload_bytes).map(|byte| (byte % 251) as u8).collect(),
    };
    let rows = IMAGE_ROWS + 2;
    let mut tokens = vec![5];
    tokens.extend(std::iter::repeat_n(image_token, IMAGE_ROWS as usize));
    tokens.push(6);
    let mut sources = vec![-1];
    sources.extend((0..u64::from(IMAGE_ROWS)).map(|row| (ARENA_PAGE * PAGE_ROWS + row) as i64));
    sources.push(-1);
    let end = u64::from(rows);
    let batch = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, rows, true, tokens.clone())],
        vec![BatchAppendV1::new(
            0,
            end,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .with_media(media)
    .encode()
    .unwrap();
    (batch, tokens, sources)
}

/// The FNV-1a an upload line ends with.
fn digest(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a(bytes))
}

/// The destinations of the uploads in `journal` whose bytes digest to
/// `hash`, as `buffer, offset` prefixes of their spans.
fn uploads(journal: &[String], hash: &str) -> Vec<String> {
    journal
        .iter()
        .filter(|line| line.starts_with("upload ") && line.ends_with(hash))
        .map(|line| span_start(line))
        .collect()
}

/// The span of a journal line up to its length: its buffer and offset.
fn span_start(line: &str) -> String {
    let from = line.find("DeviceBufferSpanV1").expect("a span");
    let to = line[from..].find(", length").expect("a span length");
    line[from..from + to].to_owned()
}

/// The launches in `journal` whose name holds `part`.
fn launches<'a>(journal: &'a [String], part: &str) -> Vec<&'a String> {
    journal
        .iter()
        .filter(|line| {
            line.starts_with("launch ") && line.split(' ').nth(2).unwrap().contains(part)
        })
        .collect()
}

/// The journal of each step `run` takes on `host`, from where it starts.
fn journal_from(host: &Backend<MediaHost>, from: usize) -> Vec<String> {
    host.device.journal.borrow()[from..].to_vec()
}

/// The pinned journal digest of the image lifecycle at each rank count.
/// Parallel ceil bindings add 35 records per rank and select the new
/// small-row symbols and thread blocks in the text decode after the image.
/// Parallel index queries and native FP4 scores include per-row query packing.
/// Image and text prefills below 128 rows use the qualified existing folds.
/// Prefill HC and ceil entries add module handles to this lifecycle.
/// Measured F32 prefill adds eight compute handles and two modules per rank,
/// with one retained tiled entry configured for its existing shared-memory limit.
/// TP4 text decode uses the direct-load MoE buckets, and the paired shared
/// gate/up keeps no single packed folds.
/// Qualified warp and input norm tiles add only their module handles; all
/// other launch arguments, allocations and stream ordering remain exact.
/// Cached norm and rounded chain producers add five resolves and seven
/// releases per rank. The finish/norm module adds two resolves and three
/// releases per rank; C1 contracts 77 pairs per rank with exact arguments,
/// unchanged allocation sizes and audited capture node reductions.
/// One-row TP4 decode fuses router reduction, selection and metadata, with
/// the cached producer unchanged and disjoint routed scratch offsets.
/// Qualified direct folds retain the image path and bind the same quantized
/// operands during text decode.
/// TP4 row reuse adds two modules, seventeen resolves and nineteen
/// releases per rank. This lifecycle keeps every launch and allocation;
/// only resource identities move. Larger-row projection graphs are audited
/// separately with the same loaded catalog in reference and serving modes.
/// The ordered head adds 24 resolves and 25 releases per rank. Twelve
/// one-row heads replace three launches with two; allocations are unchanged.
/// Full-chunk query norm resources add two resolves and three releases per
/// rank; this shorter image fixture retains every launch and operand.
/// Qualified prefill scoring adds one module, eight resolves and nine
/// releases per rank; the small image/text launches remain unchanged.
/// At four ranks the grouped 8192-row down binding adds one module, two
/// resolves, one configure and three releases per rank. The certified router
/// rows add one module, two resolves and three releases per rank, and at four
/// ranks widen only the router projection's workspace view.
/// Default native MoE prefill replaces 320 column gate/up and 320 column
/// down launches with native pack and tile pairs. It adds 16 resolves and
/// 20 releases and removes four net configures; retained launches and
/// allocations match the exact path after handle normalization.
/// The dedicated prefill arena adds two allocations and releases per rank.
/// The full journal audit preserves every other operation after handle mapping.
/// The resident small gate adds one module, two resolves and three releases
/// per TP4 rank; no launch or allocation changes. Prefill matrix tiles add
/// the same TP4 module resources as the text lifecycle. The fused candidate
/// publisher adds one module, one resolve and two releases per rank, and
/// replaces each layer-20 producer triple with one launch. At four ranks the
/// fused ceil replaces the 320 gate/up tile and intermediate quantizer
/// launches with fused tiles and fixups (8 resolves, 12 releases and four
/// configures), the dedicated arena adds two allocations and releases per
/// rank, and verify-4 grouping adds 24 resolves, three configures and 25
/// releases per rank. Full lifecycle audits preserve every other operation
/// after handle mapping.
/// The early-stop selectors add one module, six resolves and seven
/// releases per rank, and rename each prefill local selector and merge.
/// The r2 router finish module adds one module, 14 resolves and 15
/// releases per TP4 rank. At backbone Decode-1 the TP4
/// router finish becomes r2v7 and the shared expert's join moves from the
/// routed sum to directly after it: 160 renamed launches and 160 moved
/// record/wait pairs, nothing else.
/// Short dense folds preserve image operands and dependencies; staged-scale
/// module resources match the text lifecycle, with all allocations retained.
/// At four ranks the wide gate/up module adds two resolves and three
/// releases per rank and no launch, and the gate/up configure moves to it as
/// in the text lifecycle. The gate finisher loads module adds one module, 2
/// resolves and 3 releases per TP4 rank, and at backbone Decode-1 the TP4
/// gate finisher becomes the loads entry: 160 renamed launches with
/// identical config and operands, nothing else.
/// The one-launch HC boundary module adds ten resolves and eleven releases
/// per rank; the image fixture's launches and operands are unchanged.
/// The one-launch HC boundary adds ten resolves and eleven releases per
/// rank, and its C1 decode joins each of the 77 chain and finish/norm pairs
/// into one launch with the pair's operands and a zeroed ticket; graph
/// pieces keep their count and lose those nodes.
/// The cached query norm adds three resolves and four releases per rank, and
/// its one-row decode runs the 16-warp entry (grid 1 x 1, block 512) with the
/// served operands.
/// after handle mapping. The router certificate adds one allocation and its
/// release per rank, and at four ranks the router-in-norm module two
/// resolves and three releases per rank; no launch changes.
const MEDIA_JOURNALS: [(Ranks, (usize, u64)); 2] = [
    (Ranks::TP2, (22933, 0xf3b017b44088e4d5)),
    (Ranks::TP4, (56379, 0x0e9e8d56f95ae995)),
];

/// The image prefill's lifecycle at two and four ranks over the native
/// layout: create acquires the encoder (its image on every rank, its
/// cubins, the text flags); the image prompt prefills, decodes, and a
/// text prompt prefills; retire and destroy release everything.
#[test]
fn an_image_prefill_journals_at_two_and_four_ranks() {
    let mut mismatches = Vec::new();
    for (ranks, expected) in MEDIA_JOURNALS {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let count = ranks.get() as usize;
        let host = MediaHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        let mut views = Views::default();

        // The image prefill.
        let from = host.device.journal.borrow().len();
        let (batch, tokens, sources) = image_prompt(7, 0);
        step(&mut instance, &batch, &mut views);
        let prefill = journal_from(&host, from);
        let image = crate::encoders::image_rows(&sources);
        let text = crate::encoders::text_rows(&sources);
        assert_eq!(
            image.iter().filter(|&&row| row == 1).count(),
            IMAGE_ROWS as usize
        );
        let rows: Vec<u8> = sources.iter().flat_map(|row| row.to_le_bytes()).collect();
        // Each rank uploads the arena sources, the router's image rows and
        // the gate's text flags once.
        assert_eq!(uploads(&prefill, &digest(&rows)).len(), count);
        let selected = uploads(&prefill, &digest(&image));
        assert_eq!(selected.len(), count, "{ranks:?}");
        assert_eq!(uploads(&prefill, &digest(&text)).len(), count);
        // Each rank injects the image rows once through the layout's
        // embedding override, after the embedding's all-reduce and before
        // the first sublayer; every Engram gate is the masked one.
        let inject = launches(&prefill, INJECT);
        assert_eq!(inject.len(), count, "{ranks:?}");
        let position = |line: &String| prefill.iter().position(|other| other == line).unwrap();
        let reduced = position(launches(&prefill, "chop_row_rounding_round_deepseek41")[0]);
        let mixed = position(launches(&prefill, "lagged_mix_collapse")[0]);
        assert!((reduced..mixed).contains(&position(inject[0])));
        let engram_layers = (0..shape.dims().layers)
            .filter(|&layer| FLASH.is_engram_layer(layer))
            .count();
        assert!(engram_layers > 0);
        assert_eq!(
            launches(&prefill, "engram_gate").len(),
            engram_layers * count
        );
        assert!(launches(&prefill, "masked").len() == engram_layers * count);
        // The router's flags return to text after the prefill's launches.
        let last = prefill
            .iter()
            .rposition(|line| line.starts_with("launch "))
            .unwrap();
        let cleared = prefill[last..]
            .iter()
            .filter(|line| line.starts_with("zero ") && selected.contains(&span_start(line)))
            .count();
        assert_eq!(cleared, count, "{ranks:?}");
        assert_eq!(views.rows(), 1);
        assert_eq!(tokens.len(), sources.len());

        // A decode of the image prompt and a text prompt: no injection,
        // no flags, the open gates.
        let from = host.device.journal.borrow().len();
        let end = tokens.len() as u64;
        step(&mut instance, &decode(&[(7, 0, end, 3)]), &mut views);
        step(&mut instance, &prompt(9, 1, 0, 5, true), &mut views);
        let text_steps = journal_from(&host, from);
        assert!(launches(&text_steps, INJECT).is_empty());
        assert!(launches(&text_steps, "masked").is_empty());
        assert!(!launches(&text_steps, "engram_gate").is_empty());
        assert_eq!(views.rows(), 3);

        instance.retire_request(7).unwrap();
        instance.retire_request(9).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get(), "{ranks:?}");
        let lines = host.device.journal.borrow().clone();
        let topology = shape.topology();
        // The encoder's entries load with their planned dynamic shared
        // memory, as the program's do.
        assert_launches_fit_configured_shared(&lines, &topology);
        if let Some(dir) = std::env::var_os("CHOPINFER_DEEPSEEK41_GEOMETRY_JOURNAL_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                dir.join(format!("{topology}.media.journal")),
                lines.join("\n"),
            )
            .unwrap();
            super::decode_attention::export_handles(
                &host.device,
                &dir,
                &format!("{topology}.media.journal"),
            );
        }
        if let Err(observed) = check_journal(&lines, expected, &format!("{topology}.media.journal"))
        {
            mismatches.push(format!("{topology}: {observed}"));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}

/// A diagnostic bundle selecting the encoder taps writes an image prefill's
/// own step: the image's 3 x 2 cells from rank 0's arena without the
/// delimiter rows, numbered in raster order, and the 32 vision blocks'
/// rows of its 35 patches with the transformer's input, all under the
/// request's first image; a text prompt writes none.
#[cfg(feature = "diagnostic-taps")]
#[test]
fn diagnostic_image_prefills_write_the_encoder_taps() {
    let shape = Shape::new(&FLASH, Ranks::TP2).unwrap();
    let sites = [("CHOPINFER_TAPS_SITES", "enc.vision.*")];
    if std::env::var_os(super::TAPS_CHILD).is_some() {
        let host = MediaHost::new(shape, KV_LAYOUTS[0]);
        let mut instance = create(&host, 4);
        let mut views = Views::default();
        let (batch, _, _) = image_prompt(7, 0);
        step(&mut instance, &batch, &mut views);
        step(&mut instance, &prompt(9, 1, 0, 5, true), &mut views);
        instance.retire_request(7).unwrap();
        instance.retire_request(9).unwrap();
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        return;
    }
    let directory =
        std::env::temp_dir().join(format!("deepseek41-encoder-taps-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    super::in_child(
        "fake_host_tests::media::diagnostic_image_prefills_write_the_encoder_taps",
        &directory,
        &sites,
    );
    let steps: Vec<_> = std::fs::read_dir(&directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.starts_with("step"))
        .collect();
    assert_eq!(steps, ["step0000"]);
    let step = directory.join("step0000");
    let read = |name: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(step.join(name)).unwrap()).unwrap()
    };
    assert_eq!(read("completion.json")["completion"], "full");
    let index: Vec<serde_json::Value> = std::fs::read_to_string(step.join("index.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let rows = |name: &str| {
        let entry = index.iter().find(|entry| entry["name"] == name).unwrap();
        assert_eq!(entry["axis"], "encoder", "{entry}");
        (
            entry["rows"].as_u64().unwrap(),
            entry["cols"].as_u64().unwrap(),
            entry["dtype"].as_str().unwrap().to_owned(),
        )
    };
    assert_eq!(rows("enc.vision.out"), (6, 5120, "bf16".into()));
    assert_eq!(rows("enc.vision.B00.in"), (35, 1024, "f32".into()));
    for block in 0..32 {
        assert_eq!(
            rows(&format!("enc.vision.B{block:02}.out")),
            (35, 1024, "f32".into())
        );
    }
    assert_eq!(index.len(), 1 + 1 + 32);
    let items = read("items.json");
    let cells = &items["encoder"]["enc.vision.out"];
    let segments: Vec<(u64, u64, u64)> = cells
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| {
            assert_eq!(segment["request"], 7);
            assert_eq!(segment["kind"], "image");
            assert_eq!(segment["item"], 0);
            (
                segment["start"].as_u64().unwrap(),
                segment["index"].as_u64().unwrap(),
                segment["rows"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(segments, [(0, 0, 2), (2, 2, 2), (4, 4, 2)]);
    let _ = std::fs::remove_dir_all(&directory);
}
