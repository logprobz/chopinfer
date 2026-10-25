use super::*;
use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::codec::f32_to_bf16;
use kernel_core::latent_attention::{
    BlockTables, LatentCompressedBuffers, LatentKvCache, LatentSourcesBuffers, LatentWindowBuffers,
    LatentWindowStorage,
};
use std::collections::{BTreeMap, BTreeSet};

pub const WIDTH: usize = 512;
pub const HEADS: usize = 16;
pub const SLOTS: usize = 136;

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn value(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1_u64 << 23) as f32 - 1.0
    }
}

pub fn u32s(values: impl IntoIterator<Item = u32>) -> Vec<u8> {
    values.into_iter().flat_map(u32::to_ne_bytes).collect()
}
pub fn f32s(values: impl IntoIterator<Item = f32>) -> Vec<u8> {
    values.into_iter().flat_map(f32::to_ne_bytes).collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub context: usize,
    pub sequences: usize,
    pub verify: bool,
    pub verify_width: usize,
    /// Zero is window-only, otherwise the compression ratio.
    pub ratio: usize,
    pub holes: bool,
    pub inactive: usize,
}
impl Case {
    pub fn per_sequence(self) -> usize {
        if self.verify { self.verify_width } else { 1 }
    }
    pub fn rows(self) -> usize {
        self.sequences * self.per_sequence()
    }
    pub fn live(self) -> usize {
        self.rows() - self.inactive
    }
}

#[derive(Clone)]
pub struct Fixture {
    pub case: Case,
    /// Physical KV sequences, retained when only query rows are sliced.
    pub kv_sequences: usize,
    pub data: BTreeMap<&'static str, Vec<u8>>,
    pub budget: usize,
    pub page_stride: usize,
    pub code_stride: usize,
    pub scale_stride: usize,
    pub pages: usize,
    pub max_pages: usize,
    pub unique_kv_bytes: usize,
    pub row_kv_bytes: usize,
}

impl Fixture {
    pub fn new(case: Case, entry: &Entry) -> Self {
        Self::with_compressed_keys(case, entry, None)
    }

    /// Preserve each captured row's selected-key order while seeding its KV.
    pub fn with_compressed_keys(case: Case, entry: &Entry, captured: Option<&[u32]>) -> Self {
        Self::with_seed(case, entry, captured, 0)
    }

    /// Distinct rank fixtures keep the same layout and key-order contracts.
    pub fn with_seed(case: Case, entry: &Entry, captured: Option<&[u32]>, seed: u64) -> Self {
        assert!(case.context > 0 && case.inactive <= case.rows());
        let kernel_core::latent_attention::LatentAttentionQuery::Attention {
            heads,
            latent,
            page_size,
            max_keys,
            ..
        } = entry.query
        else {
            unreachable!()
        };
        assert_eq!((heads as usize, latent as usize), (HEADS, WIDTH));
        let rows = case.rows();
        let page = page_size as usize;
        let budget = if case.ratio == 0 {
            0
        } else {
            max_keys as usize - 128
        };
        let max_pages = (case.context + case.per_sequence())
            .div_ceil(case.ratio.max(1))
            .div_ceil(page)
            + 1;
        let page_stride = page * (WIDTH / 2 + WIDTH / 16) + usize::from(case.holes);
        let code_stride = SLOTS * WIDTH + if case.holes { 2 } else { 0 };
        let scale_stride = SLOTS * WIDTH / 32 + if case.holes { 3 } else { 0 };
        let mut rng = Rng(0x6bcd_48a2_1994_5119 ^ case.context as u64 ^ seed);
        let mut window_keys = vec![u32::MAX; rows * 128];
        if let Some(keys) = captured {
            assert_eq!(keys.len(), rows * budget);
            assert!(keys.iter().all(|&key| key == u32::MAX
                || (key as usize) < (case.context + case.per_sequence()) / case.ratio.max(1)));
        }
        let mut compressed_keys = vec![u32::MAX; rows * budget];
        let mut window_live = BTreeSet::new();
        let mut compressed_live = BTreeSet::new();
        let mut window_reads = BTreeSet::new();
        let mut compressed_reads = BTreeSet::new();
        let mut row_kv_bytes = 0;
        for sequence in 0..case.sequences {
            // Retain one sparse prefix selection across a verify block.
            let visible = case.context / case.ratio.max(1);
            let count = visible.min(budget);
            let selected = (0..count)
                .map(|j| {
                    let begin = j * visible / count;
                    let end = (j + 1) * visible / count;
                    begin + rng.next() as usize % (end - begin)
                })
                .collect::<Vec<_>>();
            for offset in 0..case.per_sequence() {
                let row = sequence * case.per_sequence() + offset;
                let position = case.context - 1 + offset;
                for (j, absolute) in (position.saturating_sub(127)..=position).enumerate() {
                    if case.holes && (j / 64 == 1 || j % 17 == 0) {
                        continue;
                    }
                    window_keys[row * 128 + j] = (absolute % SLOTS) as u32;
                    window_live.insert((sequence, absolute % SLOTS));
                    if row < case.live() {
                        row_kv_bytes += WIDTH + WIDTH / 32;
                        window_reads.insert((sequence, absolute % SLOTS));
                    }
                }
                let row_keys: Vec<_> = match captured {
                    Some(keys) => keys[row * budget..(row + 1) * budget].to_vec(),
                    None => selected.iter().map(|&position| position as u32).collect(),
                };
                for (j, &position) in row_keys.iter().enumerate() {
                    if position == u32::MAX || (case.holes && (j / 64 % 3 == 1 || j % 19 == 0)) {
                        continue;
                    }
                    let position = position as usize;
                    compressed_keys[row * budget + j] = position as u32;
                    compressed_live.insert((sequence, position));
                    if row < case.live() {
                        row_kv_bytes += WIDTH / 2 + WIDTH / 16;
                        compressed_reads.insert((sequence, position));
                    }
                }
            }
        }
        // Keep the physical cache extent at the requested context. Compacting
        // selected pages would hide the long-context address working set.
        let cached_pages = if case.ratio == 0 {
            0
        } else {
            ((case.context + case.per_sequence() - 1) / case.ratio).div_ceil(page)
        };
        let mut page_keys = (0..case.sequences)
            .flat_map(|sequence| (0..cached_pages).map(move |page| (sequence, page)))
            .collect::<Vec<_>>();
        for i in (1..page_keys.len()).rev() {
            let j = rng.next() as usize % (i + 1);
            page_keys.swap(i, j);
        }
        let physical = page_keys
            .iter()
            .enumerate()
            .map(|(i, &key)| (key, i))
            .collect::<BTreeMap<_, _>>();
        let pages = page_keys.len() + 1;
        let mut tables = vec![u32::MAX; case.sequences * max_pages];
        for (&(sequence, logical), &physical) in &physical {
            tables[sequence * max_pages + logical] = physical as u32;
        }
        // Every unselected cell and scale remains poison, including the
        // incomplete compressed page and retained-ring slots outside visibility.
        let mut pool = vec![0x7f; pages * page_stride];
        for &(sequence, position) in &compressed_live {
            let base = physical[&(sequence, position / page)] * page_stride;
            let slot = position % page;
            for code in &mut pool[base + slot * WIDTH / 2..base + (slot + 1) * WIDTH / 2] {
                *code = rng.next() as u8;
            }
            let scales = base + page * WIDTH / 2 + slot * WIDTH / 16;
            for scale in &mut pool[scales..scales + WIDTH / 16] {
                *scale = 0x20 + (rng.next() % 16) as u8;
            }
        }
        let mut window_codes = vec![0x7f; case.sequences * code_stride];
        let mut window_scales = vec![0xff; case.sequences * scale_stride];
        for &(sequence, slot) in &window_live {
            let state = case.sequences - 1 - sequence;
            let base = state * code_stride + slot * WIDTH;
            for code in &mut window_codes[base..base + WIDTH] {
                *code = (rng.next() % 127) as u8 | ((rng.next() as u8) & 0x80);
            }
            let base = state * scale_stride + slot * WIDTH / 32;
            for scale in &mut window_scales[base..base + WIDTH / 32] {
                *scale = 118 + (rng.next() % 4) as u8;
            }
        }
        let q = (0..rows * HEADS * WIDTH)
            .flat_map(|_| f32_to_bf16(rng.value() * 0.2).to_ne_bytes())
            .collect();
        let data = BTreeMap::from([
            ("q", q),
            (
                "row_batch",
                u32s((0..rows).map(|r| (r / case.per_sequence()) as u32)),
            ),
            ("live_rows", u32s([case.live() as u32])),
            ("window_codes", window_codes),
            ("window_scales", window_scales),
            (
                "window_items",
                u32s((0..case.sequences).map(|s| (case.sequences - 1 - s) as u32)),
            ),
            ("window_keys", u32s(window_keys)),
            ("pool", pool),
            ("tables", u32s(tables)),
            ("compressed_keys", u32s(compressed_keys)),
            ("sinks", f32s([3.0; HEADS])),
            (
                "cos_sin",
                f32s((0..rows * 64).map(|i| if i % 2 == 0 { 0.8 } else { 0.6 })),
            ),
            ("out", vec![0xa5; rows * HEADS * WIDTH * 2]),
            (
                "workspace",
                vec![0xa5; entry.workspace_bytes(rows as u32) as usize],
            ),
        ]);
        let unique_kv_bytes = window_reads.len() * (WIDTH + WIDTH / 32)
            + compressed_reads.len() * (WIDTH / 2 + WIDTH / 16);
        Self {
            case,
            kv_sequences: case.sequences,
            data,
            budget,
            page_stride,
            code_stride,
            scale_stride,
            pages,
            max_pages,
            unique_kv_bytes,
            row_kv_bytes,
        }
    }

    pub fn view(&self, name: &str) -> View {
        let index = self.data.keys().position(|&key| key == name).unwrap();
        View {
            handle: index as u64 + 1,
            bytes: self.data[name].len() as u64,
        }
    }

    pub fn views(&self) -> LatentSourcesBuffers<View> {
        let window = LatentWindowBuffers {
            codes: self.view("window_codes"),
            scales: self.view("window_scales"),
            item_slots: self.view("window_items"),
            items: self.kv_sequences as u32,
            sequences: self.kv_sequences as u32,
            code_stride: self.code_stride as u64,
            scale_stride: self.scale_stride as u64,
            storage: LatentWindowStorage::Staged {
                slots: SLOTS as u32,
            },
            key_lists: self.view("window_keys"),
            key_stride: 128,
            budget: 128,
        };
        let compressed = (self.case.ratio != 0).then(|| LatentCompressedBuffers {
            cache: LatentKvCache {
                pool: self.view("pool"),
                pages: self.pages as u32,
                page_stride: self.page_stride as u64,
            },
            block_tables: BlockTables {
                tables: self.view("tables"),
                sequences: self.kv_sequences as u32,
                max_pages: self.max_pages as u32,
            },
            key_lists: self.view("compressed_keys"),
            key_stride: self.budget as u32,
            budget: self.budget as u32,
        });
        LatentSourcesBuffers {
            q: self.view("q"),
            row_batch: self.view("row_batch"),
            live_rows: self.view("live_rows"),
            window,
            compressed,
            sinks: Some(self.view("sinks")),
            cos_sin: Some(self.view("cos_sin")),
            output: Some(self.view("out")),
            partial: None,
            workspace: self.view("workspace"),
        }
    }

    pub fn launches(&self, entry: &Entry) -> LabResult<Vec<kernel_core::backend::Launch>> {
        let rows = self.case.rows() as u32;
        kernels_sm120::backend::Sm120::new()
            .with_latent_sources(entry.clone())
            .latent_attention_sources(&entry.query, Need::rows(rows))
            .map_err(|e| e.to_string())?
            .launches(&self.views(), View::default(), Counts::rows(rows))
            .map_err(|e| e.to_string())
    }
}

pub struct DeviceFixture<'a> {
    pub host: &'a Fixture,
    pub buffers: BTreeMap<&'static str, runtime_cuda::DeviceBuffer>,
}
impl<'a> DeviceFixture<'a> {
    pub fn new(session: &Session, host: &'a Fixture) -> LabResult<Self> {
        let mut buffers = BTreeMap::new();
        for (&name, bytes) in &host.data {
            let buffer = session.allocate(bytes.len().max(16))?;
            buffer
                .copy_from_bytes(bytes, &session.stream)
                .map_err(|e| e.to_string())?;
            buffers.insert(name, buffer);
        }
        Ok(Self { host, buffers })
    }
    pub fn reset(&self, session: &Session) -> LabResult<()> {
        for name in ["workspace", "out"] {
            self.buffers[name]
                .copy_from_bytes(&self.host.data[name], &session.stream)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }
    pub fn download(&self, session: &Session, name: &str) -> LabResult<Vec<u8>> {
        let mut bytes = vec![0; self.host.data[name].len()];
        self.buffers[name]
            .copy_to_bytes(&mut bytes, &session.stream)
            .map_err(|e| e.to_string())?;
        Ok(bytes)
    }
    pub fn run(
        &self,
        session: &Session,
        modules: &Modules,
        calls: &[kernel_core::backend::Launch],
        stream: &runtime_cuda::Stream,
    ) -> LabResult<()> {
        let bindings = kernel_harness::launch::Bindings {
            buffers: self
                .buffers
                .iter()
                .map(|(&name, buffer)| (self.host.view(name).handle, buffer))
                .collect(),
        };
        for call in calls {
            modules.launch(session, stream, call, &bindings)?;
        }
        Ok(())
    }
}
