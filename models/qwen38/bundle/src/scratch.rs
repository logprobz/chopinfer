//! Per-rank device buffers of one Qwen3.8 rank program, sized for the
//! largest prefill chunk. Decode binds the first rows of the same buffers.

use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};

use bundle_runtime::geometry::Items;

use crate::contract::*;
use kernel_core::allocation::{Allocation, Traversal};

const P: u64 = PREFILL_TOKENS as u64;
const BF16: u64 = BF16_BYTES;

/// What an instance's buffers hold for the items it serves (`Items`): slot
/// tables and per-slot state for `slots` sequences, per-item arrays of a
/// phase for `items` (the largest decode bucket), and per-row arrays of a
/// decode or verify graph for `rows` (its largest token bucket).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Served {
    pub slots: u32,
    pub items: u32,
    pub rows: u32,
}

impl Served {
    pub fn new(items: Items) -> Self {
        Self {
            slots: items.get(),
            items: items.capacity(),
            rows: bundle_runtime::graphs::verify_tokens::<Qwen>(items),
        }
    }
}

/// Rows one QSA score row holds: every complete block of the context,
/// rounded to the scoring stage.
#[cfg(test)]
pub const SCORE_STRIDE: u32 = (CONTEXT_TOKENS as u32 / QSA_BLOCK).next_multiple_of(64);

pub const fn score_stride(shape: Shape) -> u32 {
    (shape.context_tokens() as u32 / shape.qsa_block()).next_multiple_of(64)
}

pub const fn wide_row_bytes(shape: Shape) -> u64 {
    shape.wide() as u64 * BF16
}

pub const fn anchor_ring(shape: Shape) -> Ring {
    Ring {
        shape,
        layers: shape.full_layers_mtp(),
        slots: ANCHOR_ITEMS as u64,
    }
}

/// What an instance serves beyond autoregressive steps: verify items of up
/// to `verify_rows` rows (zero without speculation), and the MTP drafter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Speculation {
    pub verify_rows: u32,
    pub drafter: bool,
}

impl Speculation {
    /// Entries of each linear-state pool plane: the committed entries, the
    /// step checkpoints, then `ANCHOR_ITEMS` anchor entries.
    pub const fn pool_entries(self, slots: u32) -> u64 {
        self.checkpoint_end(slots) + ANCHOR_ITEMS as u64
    }

    /// The pool entry that keeps anchor item `anchor`'s linear state.
    pub const fn anchor_entry(self, slots: u32, anchor: u32) -> u64 {
        self.checkpoint_end(slots) + anchor as u64
    }

    /// The committed entries and the step checkpoints after them.
    const fn checkpoint_end(self, slots: u32) -> u64 {
        slots as u64
            + if self.verify_rows > 1 {
                (if slots < DRAFT_ITEMS {
                    slots
                } else {
                    DRAFT_ITEMS
                }) as u64
                    * self.verify_rows as u64
            } else {
                0
            }
    }
}

/// X rows of each slot the drafter has not consumed yet, by position modulo
/// this many: a verify item's accepted rows, or a decode item's rows until
/// the drafter catches up.
pub const RETAINED_ROWS: u32 = MAX_VERIFY_ROWS;
/// Rows one drafter run takes: a whole prefill chunk, or the retained rows
/// of every item.
pub const DRAFTER_ROWS: u64 = P;

/// The logits gather's flag set, after one set per n-gram all-gather target
/// (see [`Shape::gather_sets`]).
pub const fn logits_gather(shape: Shape) -> usize {
    shape.ranks.count()
}

pub type DrafterScratch = model_qwen38_contract::buffers::DrafterScratch<DeviceBufferSpanV1>;
pub type PleStaging = model_qwen38_contract::buffers::PleStaging<DeviceBufferSpanV1>;

#[derive(Clone, Copy, Debug)]
pub struct GatherState {
    pub flags: DeviceBufferSpanV1,
    pub epoch: DeviceBufferSpanV1,
}

#[derive(Clone, Debug)]
pub struct RankScratch {
    buffers: model_qwen38_contract::buffers::Scratch<DeviceBufferSpanV1>,
    pub gathers: Vec<GatherState>,
}

impl std::ops::Deref for RankScratch {
    type Target = model_qwen38_contract::buffers::Scratch<DeviceBufferSpanV1>;
    fn deref(&self) -> &Self::Target {
        &self.buffers
    }
}

impl std::ops::DerefMut for RankScratch {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffers
    }
}

/// The bytes of the comm state each rank keeps, which the instance's target
/// states (`Target::comm_bytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommBytes {
    /// The hyper-connection all-reduce's inbox, flags and epoch.
    pub hc_inbox: u64,
    pub hc_flags: u64,
    pub hc_epoch: u64,
    /// The decode all-reduce's inbox, flags and epochs.
    pub decode_inbox: u64,
    pub decode_flags: u64,
    pub decode_epochs: u64,
    /// The decode all-reduce inbox's region that must start at zero.
    pub decode_zeroed_inbox: u64,
    /// One row gather set's flags and epoch.
    pub gather_flags: u64,
    pub gather_epoch: u64,
}

/// Bytes of every buffer, in allocation order, with a name for logs.
/// `page_bytes` is the serving pool's page, which sizes the prefix page table;
/// `context_parallel` is the instance's group size (1 without groups); `comm`
/// sizes the comm state.
#[allow(clippy::too_many_arguments)]
fn layout(
    comm: &CommBytes,
    workspace: u64,
    speculation: Speculation,
    page_bytes: u64,
    shape: Shape,
    served: Served,
    overlap: bool,
    context_parallel: u32,
) -> model_qwen38_contract::buffers::Scratch<Allocation> {
    use model_qwen38_contract::buffers;
    buffers::allocations(
        shape,
        buffers::Served {
            slots: served.slots,
            items: served.items,
            rows: served.rows,
        },
        buffers::Speculation {
            verify_rows: speculation.verify_rows,
            drafter: speculation.drafter,
        },
        buffers::Overhead {
            workspace,
            hc_inbox: comm.hc_inbox,
            hc_flags: comm.hc_flags,
            hc_epoch: comm.hc_epoch,
            hc_decode_inbox: comm.decode_inbox,
            hc_decode_flags: comm.decode_flags,
            hc_decode_epochs: comm.decode_epochs,
            gathers: shape.gather_sets() as u64 * (comm.gather_flags + comm.gather_epoch),
            prefix_pages: crate::prefix::page_table_entries(page_bytes, shape) * 8,
        },
        overlap,
        context_parallel,
    )
}

pub fn sizes(
    comm: &CommBytes,
    workspace: u64,
    speculation: Speculation,
    page_bytes: u64,
    shape: Shape,
    served: Served,
) -> Vec<(String, u64)> {
    let mut rows = Vec::new();
    layout(
        comm,
        workspace,
        speculation,
        page_bytes,
        shape,
        served,
        false,
        1,
    )
    .visit_named(&mut |name, allocation| {
        rows.push((
            model_qwen38_contract::buffers::allocation_name(name).to_owned(),
            allocation.allocation_bytes(),
        ));
    });
    rows
}

impl RankScratch {
    /// Offset comm-visible rows and the final head input into doubled backing. Compute-only
    /// packed expert workspaces and collective epochs stay shared. PLE
    /// histories retain whole-item indices and hand off through a spare input.
    pub fn overlap_part(
        &self,
        start: u32,
        first_item: usize,
        history: Option<DeviceBufferSpanV1>,
        shape: Shape,
    ) -> Self {
        let mut part = self.clone();
        let row = |span: DeviceBufferSpanV1| {
            let skip = u64::from(start) * (span.length / (2 * P));
            DeviceBufferSpanV1::new(span.buffer, span.offset + skip, span.length - skip)
        };
        part.x = self.x.map(row);
        part.xn = row(self.xn);
        part.hc_inject = row(self.hc_inject);
        part.u = row(self.u);
        part.partial = row(self.partial);
        let item = |span: DeviceBufferSpanV1| {
            let skip = first_item as u64 * shape.ple_entry_bytes();
            DeviceBufferSpanV1::new(span.buffer, span.offset + skip, span.length - skip)
        };
        part.ple_staging[0] = PleStaging {
            input: history.unwrap_or_else(|| item(self.ple_staging[0].input)),
            output: item(self.ple_staging[0].output),
        };
        part
    }

    /// Allocates every buffer through `allocate`, in `sizes` order.
    pub fn allocate(
        comm: &CommBytes,
        workspace: u64,
        speculation: Speculation,
        page_bytes: u64,
        shape: Shape,
        served: Served,
        allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        Self::allocate_with_overlap(
            comm,
            workspace,
            speculation,
            page_bytes,
            shape,
            served,
            false,
            1,
            allocate,
        )
    }

    /// Allocates every buffer through `allocate` for an instance with
    /// prefill overlap `overlap` at context-parallel size `context_parallel`.
    #[allow(clippy::too_many_arguments)]
    pub fn allocate_with_overlap(
        comm: &CommBytes,
        workspace: u64,
        speculation: Speculation,
        page_bytes: u64,
        shape: Shape,
        served: Served,
        overlap: bool,
        context_parallel: u32,
        allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
    ) -> Result<Self, BundleAbiError> {
        let buffers = bundle_runtime::scratch::bind(
            layout(
                comm,
                workspace,
                speculation,
                page_bytes,
                shape,
                served,
                overlap,
                context_parallel,
            ),
            allocate,
        )?;
        let gathers_span = buffers.gathers;
        let set = comm.gather_flags + comm.gather_epoch;
        let gathers = (0..shape.gather_sets())
            .map(|index| {
                let base = gathers_span.offset + index as u64 * set;
                GatherState {
                    flags: DeviceBufferSpanV1::new(gathers_span.buffer, base, comm.gather_flags),
                    epoch: DeviceBufferSpanV1::new(
                        gathers_span.buffer,
                        base + comm.gather_flags,
                        comm.gather_epoch,
                    ),
                }
            })
            .collect();
        Ok(Self { buffers, gathers })
    }

    /// Buffers that must start at zero: collective flags, epochs and the
    /// decode all-reduce inbox's LL region of `comm`, GEMM semaphores and the
    /// PLE pool.
    pub fn zeroed(&self, comm: &CommBytes) -> Vec<DeviceBufferSpanV1> {
        let ll = comm.decode_zeroed_inbox;
        let mut spans = vec![
            self.route_ticket,
            self.gemm_semaphores,
            self.hc_flags,
            self.hc_epoch,
            DeviceBufferSpanV1::new(self.hc_decode_inbox.buffer, self.hc_decode_inbox.offset, ll),
            self.hc_decode_flags,
            self.hc_decode_epochs,
            self.ple_pool,
        ];
        for gather in &self.gathers {
            spans.extend([gather.flags, gather.epoch]);
        }
        spans
    }
}

/// The F32 cos/sin table every rotary kernel reads: `[position][pair]` of
/// (cos, sin) using the geometry's HF rotary rule, cached per geometry.
#[cfg(test)]
pub fn cos_sin_table() -> &'static [u8] {
    cos_sin_table_for(Shape::TP4)
}

/// Cache one immutable table per registered geometry, shared by its ranks.
pub fn cos_sin_table_for(shape: Shape) -> &'static [u8] {
    static TABLES: std::sync::OnceLock<Vec<std::sync::OnceLock<Vec<u8>>>> =
        std::sync::OnceLock::new();
    crate::contract::per_geometry::<Vec<u8>>(&TABLES, shape, |g| {
        build_cos_sin(Shape::new(g, Ranks::TP4))
    })
}

fn build_cos_sin(shape: Shape) -> Vec<u8> {
    use kernel_core::rotary::{AngleRule, Axes, ComputedTable};
    let pairs = shape.rotary() as usize / 2;
    let table = ComputedTable {
        theta: f64::from(shape.rope_theta()),
        rule: AngleRule::HfF32AngleBf16CosSin,
        axes: Axes::Linear,
    };
    let mut bytes = Vec::with_capacity(shape.context_tokens() as usize * pairs * 8);
    for position in 0..shape.context_tokens() as usize {
        for (cos, sin) in table
            .row([position; 3], pairs)
            .expect("validated rotary geometry")
        {
            bytes.extend_from_slice(&(cos as f32).to_le_bytes());
            bytes.extend_from_slice(&(sin as f32).to_le_bytes());
        }
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_rope_table_matches_hf_cpu_fixture_bits() {
        let fixture = include_bytes!("../../../../fixtures/qwen_rope_cpu.bin");
        let table = cos_sin_table();
        for row in fixture[128..].chunks_exact(17) {
            let pair = row[0] as usize;
            let position = u32::from_le_bytes(row[1..5].try_into().unwrap()) as usize;
            for component in 0..2 {
                let offset = 9 + 2 * component;
                let expected =
                    u16::from_le_bytes(row[offset..offset + 2].try_into().unwrap()) as u32;
                let offset = (position * 64 + pair * 2 + component) * 4;
                let actual = u32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                assert_eq!(actual, expected << 16, "position {position} pair {pair}");
            }
        }
    }

    #[test]
    fn rotary_32_and_unpinned_theta_follow_the_hf_rule() {
        static GEOMETRY: model_qwen38_contract::geometry::Geometry = {
            let mut geometry = model_qwen38_contract::geometry::FLASH;
            geometry.shape.context = 128;
            geometry.shape.attention.rotary = 32;
            geometry.shape.attention.theta = 5e6;
            geometry
        };
        let table = build_cos_sin(Shape::new(&GEOMETRY, Ranks::TP4));
        assert_eq!(table.len(), 128 * 16 * 8);
        let bf16_bits = |value: f32| {
            let bits = value.to_bits();
            (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
        };
        for position in [0, 1, 17, 127] {
            for pair in 0..16 {
                let frequency = 1.0_f32 / 5e6_f32.powf(2.0 * pair as f32 / 32.0);
                let angle = f64::from(position as f32 * frequency);
                for (component, expected) in [angle.cos(), angle.sin()].into_iter().enumerate() {
                    let offset = (position * 16 * 2 + pair * 2 + component) * 4;
                    let actual = u32::from_le_bytes(table[offset..offset + 4].try_into().unwrap());
                    assert_eq!(
                        actual,
                        bf16_bits(expected as f32),
                        "position {position} pair {pair}"
                    );
                }
            }
        }
    }

    /// The decode all-reduce's state holds the wide entry's regions only
    /// when the instance serves graph buckets above 32 rows.
    #[test]
    fn the_decode_comm_state_grows_for_wide_buckets() {
        use model_qwen38_contract::sm120::comm::decode::{buffers, wide_buffers};
        let inbox = |items| {
            sizes(
                &crate::instance::target(Shape::TP4, false)
                    .comm_bytes(Served::new(Items::new(items).unwrap())),
                0,
                Speculation::default(),
                PAGE_BYTES_BF16,
                Shape::TP4,
                Served::new(Items::new(items).unwrap()),
            )
            .into_iter()
            .find(|(name, _)| *name == "hc_decode_inbox")
            .unwrap()
            .1
        };
        let (decode, wide) = (buffers(4).unwrap(), wide_buffers(4).unwrap());
        for items in [1, 8, 32] {
            assert_eq!(inbox(items), decode.inbox_bytes);
            assert_eq!(Served::new(Items::new(items).unwrap()).rows, 32);
        }
        for items in [33, 128, 256] {
            assert_eq!(inbox(items), wide.inbox_bytes);
            assert_eq!(Served::new(Items::new(items).unwrap()).rows, 256);
        }
        assert_eq!(wide.zeroed_inbox_bytes, decode.zeroed_inbox_bytes);
    }

    #[test]
    fn scratch_fits_beside_the_weights() {
        let served = Served::new(Items::new(32).unwrap());
        let workspace =
            crate::sm120::catalog::linear_prefill_workspace(Shape::TP4, served).unwrap();
        let tp4 = crate::instance::target(Shape::TP4, false);
        let total = |speculation| {
            sizes(
                &tp4.comm_bytes(served),
                workspace,
                speculation,
                PAGE_BYTES_BF16,
                Shape::TP4,
                served,
            )
            .iter()
            .map(|(_, bytes)| bytes)
            .sum::<u64>()
        };
        // About 49 GB of weights per rank leave room for this and the KV
        // pool; the scored prompt logits are bounded independently of prefill.
        let autoregressive = total(Speculation::default());
        assert!(autoregressive < 8 << 30, "{autoregressive}");
        // MTP at three drafts: four checkpoints per slot add 2.8 GB of
        // linear state; the PLE checkpoints and the drafter's own buffers
        // add 54 MB.
        let drafted = total(Speculation {
            verify_rows: 4,
            drafter: true,
        });
        assert_eq!(
            Speculation {
                verify_rows: 4,
                drafter: true
            }
            .pool_entries(128),
            256 + ANCHOR_ITEMS as u64
        );
        assert!(drafted - autoregressive < 3_800_000_000, "{drafted}");
        // Per-rank buffers double at TP2; the full
        // vocabulary buffers do not change.
        let at = |shape: Shape| {
            let target = crate::instance::target(shape, false);
            let workspace = target.workspace_bytes(served).unwrap();
            sizes(
                &target.comm_bytes(served),
                workspace,
                Speculation::default(),
                PAGE_BYTES_BF16,
                shape,
                served,
            )
            .iter()
            .map(|(_, bytes)| bytes)
            .sum::<u64>()
        };
        let (tp2, tp4) = (at(Shape::TP2), at(Shape::TP4));
        eprintln!("autoregressive scratch per rank: TP2 {tp2}, TP4 {tp4}");
        assert!(tp4 < tp2 && tp2 < 12 << 30, "{tp2} {tp4}");
        let gathers = |shape: Shape| {
            sizes(
                &crate::instance::target(shape, false).comm_bytes(served),
                0,
                Speculation::default(),
                PAGE_BYTES_BF16,
                shape,
                served,
            )
            .into_iter()
            .find(|(name, _)| *name == "gathers")
            .unwrap()
            .1
        };
        assert_eq!(gathers(Shape::TP2), 3 * gathers(Shape::TP4) / 5);
        assert_eq!(SCORE_STRIDE, 65_536);
        let table = cos_sin_table();
        assert_eq!(table.len() as u64, CONTEXT_TOKENS * 32 * 8);
        let value = |index: usize| f32::from_le_bytes(table[4 * index..][..4].try_into().unwrap());
        // Position 0 is the identity; position 1, pair 0 rotates by one radian.
        assert_eq!((value(0), value(1)), (1.0, 0.0));
        assert_eq!(value(64).to_bits(), 0x3f0a0000);
        assert_eq!(value(65).to_bits(), 0x3f570000);
    }
}
