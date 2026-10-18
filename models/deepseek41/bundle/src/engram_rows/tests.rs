use super::*;
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use bundle_runtime::fake_host::{Backend, FakeDevice, FakePackage, NoObserver};
use model_deepseek41_contract::geometry::FLASH;
use model_deepseek41_contract::own_ops::engram::DEAD;
use runtime_bundle_abi::{
    BundleAuthenticatedComponentHandleV1, BundleContextHandleV1, BundleMemoryPoolHandleV1,
    BundleRankArenaHandleV1, BundleRawSpeculationV1, BundleStateHandleV1, BundleStatusV1,
    CompiledImageSpanV1, HostServiceResultV1, HostServicesBackendV1, HostServicesTableV1,
    StateRequestV1,
};

fn tables() -> HashTables {
    HashTables::of(&FLASH).unwrap()
}

/// Tokens from a small deterministic generator over the vocabulary.
fn tokens(seed: u64, count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % u64::from(FLASH.shape.vocab)) as u32
        })
        .collect()
}

/// The vendor-order oracle: the ids of the last position of `sequence`
/// (text flags `text`), hashed from the sequence's start.
fn oracle(hash: &HashTables, sequence: &[u32], text: &[bool]) -> Vec<u32> {
    let compressed: Vec<i64> = sequence
        .iter()
        .zip(text)
        .map(|(&token, &text)| hash.compress(token, text).unwrap())
        .collect();
    let width = FLASH.shape.engram.layers.len() * hash.columns();
    let all = hash.sequence(&compressed).unwrap();
    all[all.len() - width..].to_vec()
}

fn row(hashed: &Hashed, index: usize) -> &[u32] {
    let width = hashed.ids().len() / hashed.rows();
    &hashed.ids()[index * width..(index + 1) * width]
}

/// A chunked prefill, then a verify step whose drafts are provisional, hash
/// every target row as the whole sequence hashed from its start would; a
/// commit keeps only the accepted tokens.
#[test]
fn target_rows_hash_their_committed_and_provisional_history() {
    let hash = tables();
    let mut rows = EngramRows::new(hash.clone());
    let prompt = tokens(7, 37);
    let mut committed: Vec<u32> = Vec::new();
    for chunk in prompt.chunks(16) {
        let step = Step {
            request: 4,
            tokens: chunk,
            text: None,
        };
        let hashed = rows.hash(&[step]).unwrap();
        for (index, _) in chunk.iter().enumerate() {
            let mut sequence = committed.clone();
            sequence.extend_from_slice(&chunk[..=index]);
            assert_eq!(
                row(&hashed, index),
                oracle(&hash, &sequence, &vec![true; sequence.len()])
            );
        }
        rows.commit(&step, chunk.len()).unwrap();
        committed.extend_from_slice(chunk);
    }
    assert_eq!(rows.history(4).len(), 3);
    // A verify step: the bonus token then five drafts, beside another
    // request's single decode row.
    let verify = tokens(11, 6);
    let other = tokens(13, 1);
    let steps = [
        Step {
            request: 4,
            tokens: &verify,
            text: None,
        },
        Step {
            request: 9,
            tokens: &other,
            text: None,
        },
    ];
    let hashed = rows.hash(&steps).unwrap();
    assert_eq!(hashed.rows(), 7);
    for index in 0..verify.len() {
        let mut sequence = committed.clone();
        sequence.extend_from_slice(&verify[..=index]);
        assert_eq!(
            row(&hashed, index),
            oracle(&hash, &sequence, &vec![true; sequence.len()])
        );
    }
    // A request with no history starts its n-grams at its own token.
    assert_eq!(row(&hashed, 6), oracle(&hash, &other, &[true]));
    // Hashing commits nothing: the same step hashes the same again.
    assert_eq!(rows.hash(&steps).unwrap(), hashed);
    // The verify accepted the bonus token and one draft: the rejected
    // drafts never enter the history.
    rows.commit(&steps[0], 2).unwrap();
    committed.extend_from_slice(&verify[..2]);
    let next = tokens(17, 3);
    let hashed = rows
        .hash(&[Step {
            request: 4,
            tokens: &next,
            text: None,
        }])
        .unwrap();
    for index in 0..next.len() {
        let mut sequence = committed.clone();
        sequence.extend_from_slice(&next[..=index]);
        assert_eq!(
            row(&hashed, index),
            oracle(&hash, &sequence, &vec![true; sequence.len()])
        );
    }
    assert_eq!(
        rows.commit(&steps[0], 7),
        Err(Error::Refused(
            "Engram commit keeps more tokens than the step ran"
        ))
    );
}

/// An image position takes no part in an n-gram, in the step and in the
/// history; a restored prefix hashes like the committed one; a forgotten
/// request starts again.
#[test]
fn image_positions_restores_and_retirement_follow_the_sequence() {
    let hash = tables();
    let mut rows = EngramRows::new(hash.clone());
    let prompt = tokens(23, 9);
    let text = [true, true, false, false, true, true, true, false, true];
    let step = Step {
        request: 1,
        tokens: &prompt,
        text: Some(&text),
    };
    let hashed = rows.hash(&[step]).unwrap();
    for index in 0..prompt.len() {
        assert_eq!(
            row(&hashed, index),
            oracle(&hash, &prompt[..=index], &text[..=index])
        );
    }
    rows.commit(&step, prompt.len()).unwrap();
    assert_eq!(rows.history(1)[1], DEAD);
    let next = tokens(29, 2);
    let after = rows
        .hash(&[Step {
            request: 1,
            tokens: &next,
            text: None,
        }])
        .unwrap();
    // A restore from the committed prefix's tail gives the same history.
    let mut restored = EngramRows::new(hash.clone());
    restored.restore(1, &prompt[4..], Some(&text[4..])).unwrap();
    assert_eq!(restored.history(1), rows.history(1));
    assert_eq!(
        restored
            .hash(&[Step {
                request: 1,
                tokens: &next,
                text: None,
            }])
            .unwrap(),
        after
    );
    // A refused restore keeps the committed history; an accepted one
    // replaces it, here with a shorter one.
    let committed = rows.history(1).to_vec();
    assert_eq!(
        rows.restore(1, &prompt, Some(&text[..3])),
        Err(Error::Refused(
            "Engram text flags differ from the step's tokens"
        ))
    );
    assert!(matches!(
        rows.restore(1, &[prompt[0], FLASH.shape.vocab], None),
        Err(Error::Refused(_))
    ));
    assert_eq!(rows.history(1), committed);
    rows.restore(1, &prompt[..1], None).unwrap();
    assert_eq!(rows.history(1), [hash.compress(prompt[0], true).unwrap()]);
    rows.forget(1);
    assert!(rows.history(1).is_empty());
    assert_eq!(
        rows.hash(&[Step {
            request: 1,
            tokens: &prompt,
            text: Some(&text[..3]),
        }]),
        Err(Error::Refused(
            "Engram text flags differ from the step's tokens"
        ))
    );
    assert!(matches!(
        rows.hash(&[Step {
            request: 1,
            tokens: &[FLASH.shape.vocab],
            text: None,
        }]),
        Err(Error::Refused(_))
    ));
}

/// However long the prefix, a request keeps storage for the n-gram window
/// only, through a restore and through a long chunk's commit.
#[test]
fn the_history_storage_is_bounded_by_the_window() {
    let hash = tables();
    let lookback = hash.engram().max_ngram as usize - 1;
    let mut rows = EngramRows::new(hash.clone());
    let prefix = tokens(37, 1 << 20);
    rows.restore(5, &prefix, None).unwrap();
    let expected: Vec<i64> = prefix[prefix.len() - lookback..]
        .iter()
        .map(|&token| hash.compress(token, true).unwrap())
        .collect();
    assert_eq!(rows.history(5), expected);
    assert_eq!(rows.history[&5].capacity(), lookback);
    let chunk = tokens(41, 16384);
    let step = Step {
        request: 5,
        tokens: &chunk,
        text: None,
    };
    rows.commit(&step, chunk.len()).unwrap();
    assert_eq!(rows.history(5).len(), lookback);
    assert_eq!(rows.history[&5].capacity(), lookback);
    // A commit of fewer tokens than the window keeps the older ids.
    rows.commit(&step, 1).unwrap();
    assert_eq!(rows.history(5)[2], hash.compress(chunk[0], true).unwrap());
    assert_eq!(
        rows.history(5)[..2],
        [
            hash.compress(chunk[16382], true).unwrap(),
            hash.compress(chunk[16383], true).unwrap()
        ]
    );
    assert_eq!(rows.history[&5].capacity(), lookback);
}

/// A package whose mapped views are two small tables per Engram layer (one
/// codes and one scales plane), whose uploads write a byte map of device
/// memory that readbacks read; it counts both.
#[derive(Default)]
struct Package {
    device: FakeDevice,
    memory: RefCell<BTreeMap<u32, Vec<u8>>>,
    uploads: Cell<usize>,
    readbacks: Cell<usize>,
}

const CODE_BYTES: u64 = 4;
const SCALE_BYTES: u64 = 1;
/// The rows of each fake table plane.
const FAKE_ROWS: u32 = 1 << 12;

/// Row `row` of view `view`: bytes that name both, each depending on every
/// bit of the row (a bijective mix of row and view), so a gather of any
/// other row gives other bytes.
fn table_row(view: &[u8], row: u32, bytes: u64) -> Vec<u8> {
    let salt = view
        .iter()
        .fold(0_u32, |salt, &byte| salt.rotate_left(8) ^ u32::from(byte));
    let mut mixed = row ^ salt;
    mixed ^= mixed >> 16;
    mixed = mixed.wrapping_mul(0x7feb_352d);
    mixed ^= mixed >> 15;
    mixed = mixed.wrapping_mul(0x846c_a68b);
    mixed ^= mixed >> 16;
    let word = mixed.to_le_bytes();
    (0..bytes as usize).map(|byte| word[byte % 4]).collect()
}

impl FakePackage for Package {
    type Observer = NoObserver;

    fn device(&self) -> &FakeDevice {
        &self.device
    }
    fn cubins() -> impl IntoIterator<Item = (&'static str, &'static [u8])> {
        []
    }
    fn open_compiled_image(
        &self,
        _role: &[u8],
    ) -> HostServiceResultV1<BundleCompiledImageHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn upload_rank_arena(
        &self,
        _context: BundleContextHandleV1,
        _pool: BundleMemoryPoolHandleV1,
        _stream: BundleStreamHandleV1,
        _image: BundleCompiledImageHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<BundleRankArenaHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_compiled_image_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        _arena: BundleRankArenaHandleV1,
        _rank: u32,
        _view_id: &[u8],
    ) -> HostServiceResultV1<CompiledImageSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn open_authenticated_component(
        &self,
        _role: &[u8],
    ) -> HostServiceResultV1<BundleAuthenticatedComponentHandleV1> {
        Err(BundleStatusV1::NotFound)
    }
    fn component(&self, _slot: u32) -> (String, Vec<u8>) {
        unreachable!("the package has no components")
    }
    fn acquire_serving_kv(&self, _layout_id: &[u8]) -> HostServiceResultV1<u64> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_serving_kv_pool(&self, _rank: u32) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn query_speculation(&self) -> HostServiceResultV1<BundleRawSpeculationV1> {
        Ok(BundleRawSpeculationV1::disabled())
    }
    fn mapped_view(
        &self,
        _image: BundleCompiledImageHandleV1,
        rank: u32,
        view_id: &[u8],
    ) -> HostServiceResultV1<Vec<u8>> {
        let bytes = match view_id.last() {
            Some(b'c') => CODE_BYTES,
            Some(b's') => SCALE_BYTES,
            _ => return Err(BundleStatusV1::NotFound),
        };
        if rank != 2 {
            return Err(BundleStatusV1::NotFound);
        }
        Ok((0..FAKE_ROWS)
            .flat_map(|row| table_row(view_id, row, bytes))
            .collect())
    }
    fn allocate_state(
        &self,
        _request: &StateRequestV1,
    ) -> HostServiceResultV1<BundleStateHandleV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn resolve_state_view(
        &self,
        _state: BundleStateHandleV1,
        _rank: u32,
    ) -> HostServiceResultV1<DeviceBufferSpanV1> {
        Err(BundleStatusV1::Unsupported)
    }
    fn read_back(
        &self,
        destination: &mut [u8],
        source: DeviceBufferSpanV1,
    ) -> HostServiceResultV1<()> {
        self.readbacks.set(self.readbacks.get() + 1);
        let memory = self.memory.borrow();
        let bytes = memory
            .get(&source.buffer.slot())
            .ok_or(BundleStatusV1::InvalidArgument)?;
        let start = source.offset as usize;
        destination.copy_from_slice(&bytes[start..start + destination.len()]);
        Ok(())
    }
    fn on_upload(
        &self,
        _stream: BundleStreamHandleV1,
        destination: DeviceBufferSpanV1,
        source: &[u8],
    ) -> HostServiceResultV1<()> {
        self.uploads.set(self.uploads.get() + 1);
        let mut memory = self.memory.borrow_mut();
        let bytes = memory.entry(destination.buffer.slot()).or_default();
        let end = destination.offset as usize + source.len();
        if bytes.len() < end {
            bytes.resize(end, 0);
        }
        bytes[destination.offset as usize..end].copy_from_slice(source);
        Ok(())
    }
}

/// The tables of the fake package: layer `l`'s planes are views `l c` and
/// `l s`, of [`FAKE_ROWS`] rows each.
fn fake_tables() -> Vec<Table> {
    (0..FLASH.shape.engram.layers.len())
        .map(|layer| Table {
            codes: Plane {
                view: format!("{layer}c").into_bytes(),
                row_bytes: CODE_BYTES,
            },
            scales: Plane {
                view: format!("{layer}s").into_bytes(),
                row_bytes: SCALE_BYTES,
            },
        })
        .collect()
}

/// The host gathers each layer's rows of both planes, in row then column
/// order, into the staging spans; the draft ids come back in one bounded
/// readback; a mis-sized span or table list is refused before the host is
/// asked.
#[test]
fn the_host_gathers_both_planes_and_reads_back_draft_ids() {
    let backend = Rc::new(Backend::new(Package::default()));
    let context = backend.select_context(0, 0).unwrap();
    let pool = backend.select_memory_pool(context).unwrap();
    let stream = backend.create_stream(context, 0).unwrap();
    let table = HostServicesTableV1::new(Rc::clone(&backend));
    let services = table.caller();
    let mapped = |rank, tables| Mapped {
        image: BundleCompiledImageHandleV1::from_parts(1, 7),
        rank,
        tables,
    };
    let hash = tables();
    let rows = EngramRows::new(hash.clone());
    let verify = tokens(31, 6);
    let mut hashed = rows
        .hash(&[Step {
            request: 3,
            tokens: &verify,
            text: None,
        }])
        .unwrap();
    // The fake tables hold FAKE_ROWS rows: reduce the ids into them,
    // keeping their order.
    hashed.ids.iter_mut().for_each(|id| *id %= FAKE_ROWS);
    let columns = hash.columns();
    let count = (hashed.rows() * columns) as u64;
    let layers = fake_tables();
    let buffer = |bytes| {
        backend
            .allocate_device_buffer(pool, stream, bytes, 64)
            .unwrap()
    };
    let destinations: Vec<Destination> = layers
        .iter()
        .map(|_| Destination {
            codes: DeviceBufferSpanV1::new(buffer(count * CODE_BYTES), 0, count * CODE_BYTES),
            scales: DeviceBufferSpanV1::new(buffer(count * SCALE_BYTES), 0, count * SCALE_BYTES),
        })
        .collect();
    // Spans checked before any copy: a short final scales span (the last
    // plane gathered) and a zero-row step with sized spans refuse with no
    // host call.
    let mut short = destinations.clone();
    let last = short.last_mut().unwrap();
    last.scales = DeviceBufferSpanV1::new(last.scales.buffer, 0, last.scales.length - 1);
    let empty = Hashed {
        rows: 0,
        ids: Vec::new(),
    };
    for (destinations, hashed) in [(&short, &hashed), (&destinations, &empty)] {
        assert_eq!(
            rows.gather(services, stream, mapped(2, &layers), destinations, hashed),
            Err(Error::Refused(
                "an Engram staging span is not rows * columns * row bytes"
            ))
        );
    }
    assert_eq!(backend.uploads.get(), 0);
    rows.gather(services, stream, mapped(2, &layers), &destinations, &hashed)
        .unwrap();
    assert_eq!(backend.uploads.get(), 2 * layers.len());
    let width = FLASH.shape.engram.layers.len() * columns;
    for (layer, (table, destination)) in layers.iter().zip(&destinations).enumerate() {
        for (plane, span) in [
            (&table.codes, destination.codes),
            (&table.scales, destination.scales),
        ] {
            let mut read = vec![0; span.length as usize];
            let before = backend.readbacks.get();
            services
                .copy_device_to_host(stream, &mut read, span)
                .unwrap();
            assert_eq!(backend.readbacks.get(), before + 1);
            let expected: Vec<u8> = hashed
                .ids()
                .chunks(width)
                .flat_map(|row| row[layer * columns..(layer + 1) * columns].to_vec())
                .flat_map(|id| table_row(&plane.view, id, plane.row_bytes))
                .collect();
            assert_eq!(read, expected, "layer {layer} plane {:?}", plane.view);
        }
    }
    // Mis-sized spans and table lists never reach the host.
    let uploads = backend.uploads.get();
    let short = Destination {
        codes: DeviceBufferSpanV1::new(destinations[0].codes.buffer, 0, CODE_BYTES),
        ..destinations[0]
    };
    assert!(matches!(
        rows.gather(
            services,
            stream,
            mapped(2, &layers),
            &[short, destinations[1]],
            &hashed
        ),
        Err(Error::Refused(_))
    ));
    assert!(matches!(
        rows.gather(
            services,
            stream,
            mapped(2, &layers[..1]),
            &destinations[..1],
            &hashed
        ),
        Err(Error::Refused(_))
    ));
    assert_eq!(backend.uploads.get(), uploads);
    // A rank the image maps nothing for is the host's refusal.
    assert_eq!(
        rows.gather(services, stream, mapped(0, &layers), &destinations, &hashed),
        Err(Error::Host(BundleAbiError::Status(
            BundleStatusV1::NotFound
        )))
    );

    // Two drafting items' five ids each, written on the device, come back
    // in one copy of exactly their bytes.
    let written: Vec<u32> = (0..10).map(|i| i * 1000 + 3).collect();
    let ids_buffer = buffer(64);
    let bytes: Vec<u8> = written.iter().flat_map(|id| id.to_le_bytes()).collect();
    services
        .copy_host_to_device(stream, DeviceBufferSpanV1::new(ids_buffer, 8, 40), &bytes)
        .unwrap();
    let mut ids = [[0_u32; 5]; 2];
    let readbacks = backend.readbacks.get();
    read_draft_ids(
        services,
        stream,
        DeviceBufferSpanV1::new(ids_buffer, 8, 40),
        &mut ids,
    )
    .unwrap();
    assert_eq!(ids.concat(), written);
    assert_eq!(backend.readbacks.get(), readbacks + 1);
    assert_eq!(
        read_draft_ids(
            services,
            stream,
            DeviceBufferSpanV1::new(ids_buffer, 8, 44),
            &mut ids
        ),
        Err(Error::Refused(
            "the draft id readback is not items * width ids"
        ))
    );
    assert_eq!(backend.readbacks.get(), readbacks + 1);
}
