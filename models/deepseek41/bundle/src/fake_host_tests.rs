//! The bundle on a CPU fake host: the package's components and rank images,
//! the serving KV pool and the per-item state, served through the shared
//! fake host (`bundle_runtime::fake_host`). No kernel runs: the device
//! journal pins every host call. These tests prove create, execute, commit,
//! retire and release mechanics, not numerics.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use bundle_runtime::fake_host::{Backend, FakeDevice, FakeIndex, FakePackage, Handle, NoObserver};
use bundle_runtime::geometry::Ranks;
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::serving::{INDEX_SCHEMA, RANK_IMAGE_ROLE, index_role};
use runtime_bundle_abi::*;

use crate::config::{KV_LAYOUTS, KvLayout};
use crate::shape::Shape;

const KV_PAGES: u64 = 256;
const IMAGE_SLOT: u32 = 1_000_000;
const KV_SLOT: u32 = 2_000_000;
const STATE_SLOT: u32 = 3_000_000;
const INDEX_COMPONENT: u32 = 100;
const IDENTITY_COMPONENT: u32 = 200;
const VIEW_ALIGNMENT: u64 = 4096;

/// The fake host of one package topology and KV layout.
pub(crate) struct FakeHost {
    pub device: FakeDevice,
    pub logs: RefCell<Vec<String>>,
    shape: Shape,
    kv: KvLayout,
    /// The host's speculation: (drafts, verify tokens), none when off.
    speculation: Cell<Option<(u32, u32)>>,
    /// The per-item state bytes create asked for.
    state_bytes: Cell<u64>,
    /// Every upload's bytes by its destination, which a download of the
    /// same span reads back (a snapshot's header).
    uploads: RefCell<std::collections::BTreeMap<(u64, u64), Vec<u8>>>,
}

pub(crate) type Host = Backend<FakeHost>;

impl std::ops::Deref for FakeHost {
    type Target = FakeDevice;

    fn deref(&self) -> &FakeDevice {
        &self.device
    }
}

/// The views the program binds today, with their bytes, in the order a
/// rank image lays them out. The weight plan names every view once it
/// lands; the fake host serves exactly what the lowering reads.
fn views(shape: &Shape) -> Vec<(String, u64)> {
    let d = shape.dims();
    let hidden = u64::from(d.hidden);
    let mut views = vec![(
        "embed.weight".to_owned(),
        u64::from(shape.vocab_rows()) * hidden * 2,
    )];
    for layer in 0..d.layers {
        for norm in ["attn_norm", "ffn_norm"] {
            views.push((format!("layers.{layer}.{norm}.weight"), hidden * 2));
        }
    }
    views.push(("norm.weight".into(), hidden * 2));
    views
}

/// Rank `rank`'s index and image bytes.
fn index(shape: &Shape, rank: u32) -> (Vec<u8>, u64) {
    let mut index = FakeIndex::new(INDEX_SCHEMA, rank);
    for (name, bytes) in views(shape) {
        index.push(
            name,
            bytes,
            "BF16",
            vec![bytes / 2],
            "row-major",
            VIEW_ALIGNMENT,
        );
    }
    let image_bytes = index.end(VIEW_ALIGNMENT);
    (index.bytes(image_bytes), image_bytes)
}

impl FakeHost {
    pub(crate) fn new(shape: Shape, kv: KvLayout) -> Rc<Host> {
        Rc::new(Backend::new(FakeHost {
            device: FakeDevice::default(),
            logs: RefCell::new(Vec::new()),
            shape,
            kv,
            speculation: Cell::new(None),
            state_bytes: Cell::new(0),
            uploads: RefCell::new(std::collections::BTreeMap::new()),
        }))
    }

    fn page_bytes(&self) -> u64 {
        model_deepseek41_contract::state::request_state(self.shape.geometry, self.kv.storage, false)
            .page_bytes()
    }
}

impl FakePackage for FakeHost {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        &self.device
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        crate::CUBINS.iter().map(|cubin| (cubin.name, cubin.bytes))
    }
    fn log(&self, message: &[u8]) {
        self.logs
            .borrow_mut()
            .push(String::from_utf8_lossy(message).into_owned());
    }
    fn on_acquire(&self, _handle: Handle) {}
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        match self
            .uploads
            .borrow()
            .get(&(source.buffer.raw, source.offset))
        {
            Some(bytes) if bytes.len() == destination.len() => destination.copy_from_slice(bytes),
            _ => destination.fill(0),
        }
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.uploads.borrow_mut().insert(
            (destination.buffer.raw, destination.offset),
            source.to_vec(),
        );
        Ok(())
    }
    fn open_compiled_image(&self, role: &[u8]) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        assert_eq!(role, RANK_IMAGE_ROLE.as_bytes());
        Ok(BundleCompiledImageHandleV1::from_parts(1, self.id()))
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        self.note(format!("upload arena {rank}"));
        Ok(BundleRankArenaHandleV1::from_parts(1, self.id()))
    }
    fn resolve_compiled_image_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        let (_, bytes) = index(&self.shape, rank);
        let slot = IMAGE_SLOT + rank;
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
        let ranks = self.shape.ranks();
        let identity = format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/");
        for rank in 0..ranks {
            if role == index_role(rank).as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    INDEX_COMPONENT + rank,
                ));
            }
            if role == format!("{identity}{rank}").as_bytes() {
                return Ok(BundleAuthenticatedComponentHandleV1::from_parts(
                    1,
                    IDENTITY_COMPONENT + rank,
                ));
            }
        }
        Err(BundleStatusV1::NotFound)
    }
    fn component(&self, slot: u32) -> (String, Vec<u8>) {
        if slot >= IDENTITY_COMPONENT {
            let rank = slot - IDENTITY_COMPONENT;
            let (_, bytes) = index(&self.shape, rank);
            let identity = serde_json::json!({"byte_length": bytes, "sha256": "a".repeat(64)});
            return (
                format!("chop.compiled-image-identity.v1/{RANK_IMAGE_ROLE}/{rank}"),
                serde_json::to_vec(&identity).unwrap(),
            );
        }
        let rank = slot - INDEX_COMPONENT;
        (index_role(rank), index(&self.shape, rank).0)
    }
    fn acquire_serving_kv(&self, layout_id: &[u8]) -> HostServiceResultV1<u64> {
        assert_eq!(layout_id, self.kv.layout_id);
        self.note("serving kv".into());
        Ok(KV_PAGES)
    }
    fn resolve_serving_kv_pool(&self, rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let slot = KV_SLOT + rank;
        let bytes = KV_PAGES * self.page_bytes();
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, slot),
            0,
            bytes,
        ))
    }
    fn allocate_state(&self, request: &StateRequestV1) -> HostServiceResultV1<BundleStateHandleV1> {
        assert_eq!(
            request.state_id,
            model_deepseek41_contract::serving::REQUEST_STATE_ID.as_bytes()
        );
        self.state_bytes.set(request.byte_length);
        self.note(format!("state {}", request.byte_length));
        Ok(BundleStateHandleV1::from_parts(1, self.id()))
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        let slot = STATE_SLOT + rank;
        let bytes = self.state_bytes.get();
        self.device.extents.borrow_mut().insert(slot, bytes);
        Ok(DeviceBufferSpanV1::new(
            BundleDeviceBufferHandleV1::from_parts(1, slot),
            0,
            bytes,
        ))
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        let mut speculation = BundleRawSpeculationV1::disabled();
        if let Some((drafts, rows)) = self.speculation.get() {
            speculation.maximum_draft_tokens = drafts;
            speculation.maximum_verify_tokens = rows;
        }
        Ok(speculation)
    }
}

pub(crate) fn api() -> BundleApiV1 {
    BundleApiV1::new_with_prefix_state_slot(crate::BUNDLE_DEEPSEEK41_EXTENDED_API_V1)
        .expect("bundle API")
}

/// Creates an instance serving `items` with only its lowered stages
/// (`crate::instance::PARTIAL`).
pub(crate) fn create(host: &Rc<Host>, items: u32) -> BundleInstanceV1 {
    let record = crate::config::tests::record(&host.shape, host.kv, items, false);
    crate::instance::PARTIAL.with(|partial| partial.set(true));
    let created = api().create(HostServicesTableV1::new(Rc::clone(host)), &record);
    crate::instance::PARTIAL.with(|partial| partial.set(false));
    created.unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()))
}

/// A sink that keeps the logits views.
#[derive(Default)]
pub(crate) struct Views(pub(crate) Vec<OwnedLogitsViewV1>);

impl Views {
    /// The logits rows the views hold.
    pub(crate) fn rows(&self) -> u32 {
        self.0.iter().map(|view| view.raw().row_count).sum()
    }
}

impl LogitsRowSinkV1 for Views {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(256, 1 << 20, 256 << 20)
    }

    fn emit_row(&mut self, _row: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }

    fn emit_view(&mut self, view: OwnedLogitsViewV1) -> HostServiceResultV1<()> {
        self.0.push(view);
        Ok(())
    }
}

/// A prompt chunk of `count` tokens after `previous` of `request` in
/// `slot`, scoring its last row when `last`.
pub(crate) fn prompt(request: u64, slot: u64, previous: u64, count: u64, last: bool) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let end = previous + count;
    let inputs = (previous..end).map(|token| token as u32).collect();
    BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::new(request, false, count as u32, last, inputs)],
        vec![BatchAppendV1::new(
            previous,
            count,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        )],
    )
    .with_state_slots(vec![slot])
    .encode()
    .unwrap()
}

/// One decode step of `items`: (request, slot, previous tokens, token).
pub(crate) fn decode(items: &[(u64, u64, u64, u32)]) -> Vec<u8> {
    use runtime_bundle_sdk::{BatchAppendV1, BatchItemV1, BatchRequestV1, BatchWireVersionV1};
    let mut batch = Vec::new();
    let mut appends = Vec::new();
    let mut slots = Vec::new();
    for &(request, slot, previous, token) in items {
        let end = previous + 1;
        batch.push(BatchItemV1::new(request, true, 1, true, vec![token]));
        appends.push(BatchAppendV1::new(
            previous,
            1,
            end,
            (0..end.div_ceil(64)).map(|page| slot * 8 + page).collect(),
            None,
        ));
        slots.push(slot);
    }
    BatchRequestV1::new(BatchWireVersionV1::V3, batch, appends)
        .with_state_slots(slots)
        .encode()
        .unwrap()
}

/// Every topology and KV layout creates, prefills a prompt in two chunks,
/// decodes two requests, commits each step, retires and releases
/// everything; each logits row reaches the sink.
#[test]
fn every_topology_and_layout_runs_the_lifecycle() {
    for shape in crate::shape::topologies() {
        for kv in KV_LAYOUTS {
            let host = FakeHost::new(shape, kv);
            let mut instance = create(&host, 4);
            let mut views = Views::default();
            instance
                .execute(&prompt(7, 0, 0, 64, false), &mut views)
                .unwrap();
            instance.commit().unwrap();
            assert_eq!(views.rows(), 0);
            instance
                .execute(&prompt(7, 0, 64, 6, true), &mut views)
                .unwrap();
            instance.commit().unwrap();
            assert_eq!(views.rows(), 1);
            instance
                .execute(&prompt(9, 1, 0, 3, true), &mut views)
                .unwrap();
            instance.commit().unwrap();
            instance
                .execute(&decode(&[(7, 0, 70, 1), (9, 1, 3, 2)]), &mut views)
                .unwrap();
            instance.commit().unwrap();
            assert_eq!(views.rows(), 4);
            instance.retire_request(7).unwrap();
            instance.retire_request(9).unwrap();
            instance.destroy().unwrap();
            assert_eq!(
                host.acquired.get(),
                host.released.get(),
                "{} {:?}",
                shape.topology(),
                kv.storage
            );
        }
    }
}

/// A step refuses while its stages are unlowered unless the instance runs
/// the lowered stages alone; create refuses kernels that miss a family.
#[test]
fn create_refuses_kernels_that_miss_a_family() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let record = crate::config::tests::record(&shape, KV_LAYOUTS[0], 4, false);
    let refused = api()
        .create(HostServicesTableV1::new(Rc::clone(&host)), &record)
        .err()
        .unwrap();
    assert_eq!(refused, BundleAbiError::Status(BundleStatusV1::Unsupported));
    assert!(
        host.logs
            .borrow()
            .iter()
            .any(|line| line.contains("refuses DeepSeek V4.1's row ops")),
        "{:?}",
        host.logs.borrow()
    );
    assert_eq!(host.acquired.get(), host.released.get());
}

/// A request's rings and tails reach snapshot pages after a prompt and
/// come back into another slot under a new request, which then decodes; a
/// restore that names other tokens is refused and leaves the slot free.
#[test]
fn prefix_snapshots_round_trip_through_the_pages() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let host = FakeHost::new(shape, KV_LAYOUTS[0]);
    let mut instance = create(&host, 4);
    let mut views = Views::default();
    instance
        .execute(&prompt(7, 0, 0, 70, true), &mut views)
        .unwrap();
    instance.commit().unwrap();
    let count = instance.describe_prefix_state().unwrap();
    // 40 rings of 128 cells and three tails' open rows.
    assert!(count > 0);
    let pages = (0..count).map(|page| 32 + page).collect::<Vec<_>>();
    instance.capture_prefix_state(7, 70, &pages).unwrap();
    instance.retire_request(7).unwrap();
    let tokens = (0..70).collect::<Vec<u32>>();
    assert!(
        instance
            .restore_prefix_state_in_slot(9, &tokens[..69], &pages, 1)
            .is_err()
    );
    instance
        .restore_prefix_state_in_slot(9, &tokens, &pages, 1)
        .unwrap();
    instance
        .execute(&decode(&[(9, 1, 70, 5)]), &mut views)
        .unwrap();
    instance.commit().unwrap();
    assert_eq!(views.rows(), 2);
    instance.retire_request(9).unwrap();
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
}
