//! Write maps for the fixture's prefill stage units, with launch-only callbacks.
use super::execution;
pub(super) use checks::Failures;
use kernel_build::EntryParam;
use kernel_core::contract::{Argument, DType};
use kernel_lab::poison as checks;
use kernel_lab::poison::{self, Phase, Replay, WriteMap};
use kernel_lab::{LabResult, Session};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

pub(super) struct Unit<'a> {
    fixed: Replay<'a>,
    routing: checks::Checks<'a>,
}
impl Unit<'_> {
    fn prepare(&self, phase: Phase) -> LabResult<()> {
        self.fixed.prepare(phase)?;
        self.routing.prepare()
    }
    fn finish(&mut self, phase: Phase, pdl: bool) -> LabResult<()> {
        let mut failures = Failures::default();
        failures.check("fixed outputs", self.fixed.finish(phase, pdl));
        failures.check("routing outputs", self.routing.finish(phase, pdl));
        failures.finish()
    }
    pub(super) fn phase(
        &mut self,
        phase: Phase,
        pdl: bool,
        launch: impl FnOnce() -> LabResult<()>,
    ) -> LabResult<()> {
        self.prepare(phase)?;
        launch()?;
        self.finish(phase, pdl)
    }
    pub(super) fn evidence(&self) -> Value {
        let mut fixed = BTreeMap::new();
        self.fixed.record(&mut fixed);
        let mut routing = BTreeMap::new();
        self.routing.record(&mut routing);
        json!({"fixed": fixed, "routing": routing})
    }
}

pub(super) fn collective(
    sessions: &[Session],
    units: &mut [Unit<'_>],
    phase: Phase,
    pdl: bool,
    launch: impl FnOnce() -> LabResult<()>,
) -> LabResult<()> {
    poison::synchronize(sessions)?;
    for unit in units.iter() {
        unit.prepare(phase)?;
    }
    launch()?;
    poison::synchronize(sessions)?;
    let mut failures = Failures::default();
    for (rank, unit) in units.iter_mut().enumerate() {
        failures.check(&format!("rank {rank}"), unit.finish(phase, pdl));
    }
    failures.finish()
}

fn words(s: &Session, rank: &execution::Rank, id: u64) -> LabResult<Vec<i32>> {
    Ok(poison::read(s, &rank.buffers[&id])?
        .chunks_exact(4)
        .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
        .collect())
}

fn attention_map(rows: usize, global: bool) -> LabResult<WriteMap> {
    // attention/prefill.cuh:78-90,692-716. One-piece tiles write ATTN
    // directly; other tiles write all 130 F32 terms of every actual piece.
    WriteMap::of(DType::F32).byte_ranges((0..rows).flat_map(|token| {
        let end = token / 8 * 8 + 8;
        let length = 256 * end.div_ceil(1024);
        let pieces = end.div_ceil(length);
        (0..if global && pieces > 1 { pieces } else { 0 }).flat_map(move |piece| {
            (0..16).map(move |head| {
                let start = ((token * 4 + piece) * 16 + head) * 130 * 4;
                start..start + 130 * 4
            })
        })
    }))
}

fn width(id: u64, global: bool) -> LabResult<(usize, DType)> {
    // Physical IDs and widths: the fixture execution API (buffer IDs and intermediate widths).
    Ok(match id {
        21 | 30..=33 | 43..=46 => (4096, DType::Bf16),
        22 => (if global { 3392 } else { 3712 }, DType::Bf16),
        23 => (3072, DType::Bf16),
        24 => (if global { 192 } else { 384 }, DType::Bf16),
        25 => (if global { 128 } else { 256 }, DType::Bf16),
        28 => (2048, DType::Bf16),
        34 => (256, DType::F32),
        35 => (8, DType::I32),
        36 => (8, DType::F32),
        41 => (8 * 512, DType::Bf16),
        42 => (8 * 4096, DType::Bf16),
        _ => return Err(format!("unknown prefill output {id}")),
    })
}

fn row_writes(symbol: &str, name: &str) -> bool {
    ["y", "output", "q", "k_new", "v_new", "residual_out"].contains(&name)
        || (symbol.starts_with("chop_row_router_") && ["ids", "weights"].contains(&name))
}

pub(super) fn unit<'a>(
    s: &'a Session,
    rank: &'a execution::Rank,
    rows: usize,
    range: Range<usize>,
    local: bool,
    effective_ids: Option<&[i32]>,
) -> LabResult<Unit<'a>> {
    if rank.batch != 1
        || rank.layer == 0
        || rank.bundle_mode
        || rows == 0
        || rows > rank.rows_capacity
        || words(s, rank, 59)?[0] != 0
    {
        return Err(
            "prefill poison maps require the selected one-sequence, zero-prefix unfused fixtures"
                .into(),
        );
    }
    let (mut outputs, mut partials) = (BTreeSet::new(), BTreeSet::new());
    let (mut route, mut attention, mut cache, mut tiles) = (false, false, false, None);
    for stage in &rank.prefill[range] {
        let name = stage.name.as_str();
        if name.starts_with("chop_comm_") {
            if !local {
                for arg in &stage.args.0 {
                    if let Argument::DeviceBuffer(id @ (31 | 44)) = arg {
                        outputs.insert(*id);
                    }
                }
            }
        } else if name.starts_with("chop_gemm_") {
            let Some(Argument::DeviceBuffer(out)) = stage.args.0.get(2) else {
                return Err("unexpected GEMM output ABI".into());
            };
            let Some(Argument::DeviceBuffer(partial)) = stage.args.0.get(3) else {
                return Err("unexpected GEMM partial ABI".into());
            };
            outputs.insert(*out);
            partials.insert(*partial);
        } else if name.starts_with("chop_attn_") {
            outputs.insert(28);
            attention = true;
        } else if name.starts_with("chop_moe_") && name.contains("gateup_prefill_") {
            outputs.insert(41);
        } else if name.starts_with("chop_moe_") && name.contains("down_prefill_") {
            outputs.insert(42);
        } else if name.starts_with("chop_row_permute_") {
            route = true;
            if name.contains("tile_count") {
                tiles = Some(stage.plan.grid[0] as usize);
            }
        } else if name.starts_with("chop_row_ring_write_") {
            cache = true;
        } else if name.starts_with("chop_row_") {
            let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
                .into_iter()
                .find(|e| e.symbol == name)
                .ok_or_else(|| format!("missing row ABI {name}"))?;
            for (param, arg) in entry.params.iter().zip(&stage.args.0) {
                if let (EntryParam::DevicePointer { name, .. }, Argument::DeviceBuffer(id)) =
                    (param, arg)
                    && row_writes(stage.name.as_str(), name.as_str())
                    && (21..=46).contains(id)
                    && ![26, 27, 29, 37, 38, 39, 40].contains(id)
                {
                    outputs.insert(*id);
                }
            }
            if name.starts_with("chop_row_rope_") && rank.global {
                cache = true;
            }
        } else {
            return Err(format!("unmapped prefill stage {name}"));
        }
    }
    let mut fixed = Replay::new(s);
    for id in outputs {
        let (n, element) = width(id, rank.global)?;
        let live = if !local && [31, 44].contains(&id) {
            rank.rows_capacity
        } else {
            rows
        };
        fixed = fixed.output(
            format!("prefill buffer {id}"),
            &rank.buffers[&id],
            WriteMap::of(element).prefix(live * n)?,
        );
    }
    if attention {
        fixed = fixed.output(
            "attention partials",
            &rank.buffers[&29],
            attention_map(rows, rank.global)?,
        );
    }
    for id in partials {
        // fragment_prefill.cuh:342,531-556. Arithmetic scratch has a
        // schedule-dependent required subset; separate semaphores stay intact.
        let buffer = &rank.buffers[&id];
        fixed = fixed.scratch_within(
            format!("dense partials {id}"),
            buffer,
            std::iter::once(0..buffer.byte_len()).collect(),
        );
    }
    if cache {
        let positions = words(s, rank, 51)?;
        let batches = words(s, rank, 52)?;
        let table = words(s, rank, 54)?;
        let slots = words(s, rank, 55)?;
        let last = poison::read(s, &rank.buffers[&53])?
            .chunks_exact(8)
            .map(|b| i64::from_ne_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        for (id, dim) in [(26, 192), (27, 128)] {
            let mut ranges = Vec::new();
            for token in 0..rows {
                let (pos, batch) = (positions[token] as usize, batches[token] as usize);
                // row/rope_kv.cuh:63-99 and ring_write:110-145.
                let (cell, heads) = if rank.global {
                    let page = table[batch * 16384 + pos / 64];
                    if page < 0 {
                        return Err("negative live cache page".into());
                    }
                    (page as usize * 64 + pos % 64, 1)
                } else {
                    if (pos as i64) + 128 <= last[batch] {
                        continue;
                    }
                    (slots[batch] as usize * 128 + pos % 128, 2)
                };
                let start = cell * heads * dim * 2;
                ranges.push(start..start + heads * dim * 2);
            }
            fixed = fixed.state(
                format!("cache {id}"),
                &rank.buffers[&id],
                poison::read(s, &rank.buffers[&id])?,
                WriteMap::of(DType::Bf16).byte_ranges(ranges)?,
                Vec::new(),
            );
        }
    }
    let mut routing = checks::Checks::new(s);
    if route {
        // router.cuh:148-213,236-337. Count maps include dead prepared
        // tiles; prefix offsets include only live tiles. Frozen IDs stay inputs.
        for (id, n) in [(37, 256), (38, 257), (40, 1)] {
            routing = routing.output(
                &format!("route {id}"),
                &rank.buffers[&id],
                WriteMap::of(DType::I32).prefix(n)?,
            );
        }
        if let Some(tiles) = tiles {
            routing = routing
                .output(
                    "tile counts",
                    &rank.buffers[&47],
                    WriteMap::of(DType::I32).prefix(tiles * 256)?,
                )
                .output(
                    "tile offsets",
                    &rank.buffers[&48],
                    WriteMap::of(DType::I32).prefix((rows * 8).div_ceil(256) * 256)?,
                );
        }
        let frozen = effective_ids.map(<[i32]>::to_vec);
        routing = routing.dynamic(
            "sorted routes",
            &rank.buffers[&39],
            move |session| {
                let ids = match &frozen {
                    Some(ids) => ids.clone(),
                    None => checks::ids(session, &rank.buffers[&35], rows * 8)?,
                };
                if ids.len() != rows * 8 || ids.iter().any(|id| !(0..256).contains(id)) {
                    return Err("invalid live prefill route ID".into());
                }
                let counts = checks::counts(&ids, 0..256)?;
                checks::sorted_map(&counts, 64, 0)
            },
            &[0x80],
        );
    }
    Ok(Unit { fixed, routing })
}

#[test]
fn prefill_partial_contract_cpu() {
    let map = attention_map(264, true).unwrap();
    assert_eq!(map.written_bytes(), 8 * 2 * 16 * 130 * 4);
    assert_eq!(attention_map(256, true).unwrap().written_bytes(), 0);
    assert_eq!(attention_map(2048, false).unwrap().written_bytes(), 0);
    let mut bytes = vec![0xff; 264 * 4 * 16 * 130 * 4];
    for range in map.ranges() {
        bytes[range.clone()].fill(0);
    }
    assert!(poison::check("pieces", &bytes, &map, poison::Fill::Pattern(&[0xff])).is_ok());
    bytes[map.ranges()[0].start..map.ranges()[0].start + 4].fill(0xff);
    assert!(poison::check("missing term", &bytes, &map, poison::Fill::Pattern(&[0xff])).is_err());
}

#[test]
fn stage_input_contract_cpu() {
    assert!(!row_writes("chop_row_combine_v1", "weights"));
    assert!(!row_writes("chop_row_rms_residual_v1", "residual"));
    assert!(row_writes("chop_row_rms_residual_v1", "residual_out"));
    assert!(row_writes("chop_row_router_v1", "weights"));
    assert!(row_writes("chop_row_router_v1", "ids"));
    for id in [50, 61, 62, 71, 73, 75, 80, 81, 90, 91, 92] {
        assert!(width(id, true).is_err());
    }
}
