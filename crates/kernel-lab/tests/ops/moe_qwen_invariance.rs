//! The Qwen3.8 chains of the grouped expert invariance adapter
//! (`moe_invariance.rs`), over tokens whose hidden input, router logits and
//! shared-expert output are given, on one rank's random experts of a
//! profile:
//!
//! - the routed prefill chain: router, tiled permutation, this rank's
//!   gate/up and down experts, and the shared-expert gate combine;
//! - the decode chain of a token bucket: decode route, gate/up, down and
//!   combine, as a decode or K3 verify step launches them;
//! - the combine alone, over given routed rows and expert IDs.
//!
//! Every launch starts with its outputs poisoned and is checked afterwards
//! against the write maps of qwen.cuh (`qwen_layout`, and the shared checks
//! of the adapter): the routes' IDs and weights, the permutation, the gate/up
//! and down rows of the rank's routes only and the live tokens' combined
//! rows written; padding, other ranks' routes and rows past the live ones
//! untouched.

use super::super::invariance::{
    Experts, Layout, Route, chain_image, experts_written, ids_written, route_written, rows_written,
    with,
};
use super::*;
use crate::invariance::Units;
use kernel_lab::poison;
use std::ops::Range;

/// What a decode unit holds for its token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Part {
    /// The decode route's expert IDs and route weights.
    Route,
    /// The routed intermediate rows of the token's local routes, each after
    /// its route slot.
    GateUp,
    /// The weighted down rows of the token's local routes, likewise.
    Down,
}

/// The chains of one Qwen3.8 profile, sharing its expert banks.
pub(crate) struct Chains {
    profile: Profile,
    /// Experts per rank.
    local: usize,
    banks: Banks,
    /// The prefill chain, holding up to `PREFILL_ROWS` rows.
    prefill: Option<Case>,
    /// Decode chains by token bucket.
    decode: Vec<Case>,
}

/// Rows the prefill chain holds: the largest prompt or batch a case runs.
const PREFILL_ROWS: usize = 512;
/// Rows each local expert's range of a decode route is padded to (qwen.cuh
/// `kDecodeBlock`).
const DECODE_BLOCK: usize = 16;

/// Where a Qwen3.8 chain over `tokens` tokens on the rank running global
/// experts `experts` writes (qwen.cuh): ten routes per token and the active
/// list; a decode route pads each expert's range to `kDecodeBlock` rows; the
/// prefill permutation pads to `ROUTE_BLOCK` rows and its count stage writes
/// the counts of the live 256-route tiles only (later tiles return first).
pub(crate) fn qwen_layout(experts: Range<usize>, tokens: usize, prefill: bool) -> Layout {
    Layout {
        top_k: QK,
        experts,
        block: if prefill {
            qwen::ROUTE_BLOCK as usize
        } else {
            DECODE_BLOCK
        },
        tiles: prefill.then(|| (tokens * QK).div_ceil(256)),
        active: true,
        inter: I,
        hidden: H,
    }
}

impl Case {
    /// The global IDs of the experts of the rank the case runs.
    fn rank_experts(&self) -> Range<usize> {
        let first = self.rank as usize * self.local;
        first..first + self.local
    }

    /// Poisons what a launch of the chain writes before reading it: the
    /// route's IDs and weights, its expert counts, offsets, sorted rows,
    /// padded total and active list, with `tiled` the prefill permutation's
    /// per-tile counts and offsets, and the gate/up, down and combined
    /// rows. Inputs stay.
    fn poison_writes(&self, session: &Session, tiled: bool) -> LabResult<()> {
        let mut buffers = vec![
            &self.ids,
            &self.weights,
            &self.counts,
            &self.offsets,
            &self.padded,
            &self.active,
            &self.inter,
            &self.routed,
            &self.output,
        ];
        if tiled {
            buffers.extend([&self.tile_counts, &self.tile_offsets]);
        }
        poison::poison(session, &buffers)?;
        poison::fill_all(session, &self.sorted, &[0x80])
    }

    /// Every byte of the buffers `poison_writes` poisons, after the launch.
    fn image(&self, session: &Session, tiled: bool) -> LabResult<(Route, Experts)> {
        session.stream.synchronize().map_err(err)?;
        let read = |buffer: &DeviceBuffer| poison::read(session, buffer);
        let route = Route {
            ids: read(&self.ids)?,
            weights: read(&self.weights)?,
            counts: read(&self.counts)?,
            offsets: read(&self.offsets)?,
            sorted: read(&self.sorted)?,
            padded: read(&self.padded)?,
            active: Some(read(&self.active)?),
            tiles: if tiled {
                Some((read(&self.tile_counts)?, read(&self.tile_offsets)?))
            } else {
                None
            },
        };
        let experts = Experts {
            inter: read(&self.inter)?,
            routed: read(&self.routed)?,
            output: read(&self.output)?,
        };
        Ok((route, experts))
    }
}

/// `values` padded with zeros to `rows` rows of `width`.
fn padded<T: Copy + Default>(values: &[T], rows: usize, width: usize) -> Vec<T> {
    let mut out = values.to_vec();
    out.resize(rows * width, T::default());
    out
}

/// The first `rows` rows of `width` BF16 elements of `bytes`, one unit each.
fn bf16_units(bytes: &[u8], rows: usize, width: usize) -> Units {
    bytes[..rows * width * 2]
        .chunks(width * 2)
        .map(<[u8]>::to_vec)
        .collect()
}

impl Chains {
    pub(crate) const HIDDEN: usize = H;
    pub(crate) const EXPERTS: usize = GLOBAL;
    pub(crate) const TOP_K: usize = QK;

    pub(crate) fn new(session: &Session, profile: Profile) -> LabResult<Self> {
        let local = qwen::local_experts(profile).map_err(err)? as usize;
        Ok(Self {
            profile,
            local,
            banks: Banks::new(session, local)?,
            prefill: None,
            decode: Vec::new(),
        })
    }

    pub(crate) fn profile(&self) -> Profile {
        self.profile
    }

    /// The global IDs of the experts of the rank the chains run.
    pub(crate) fn rank_experts(&self) -> std::ops::Range<usize> {
        let first = rank(self.local) as usize * self.local;
        first..first + self.local
    }

    fn prefill(&mut self, session: &Session) -> LabResult<&mut Case> {
        if self.prefill.is_none() {
            self.prefill = Some(Case::for_profile(
                session,
                self.profile,
                Phase::Prefill,
                PREFILL_ROWS,
            )?);
        }
        Ok(self.prefill.as_mut().expect("the prefill chain"))
    }

    /// Uploads `rows` rows of hidden input and router logits to `case`.
    fn load(session: &Session, case: &mut Case, x: &[u16], logits: &[f32]) -> LabResult<usize> {
        let rows = x.len() / H;
        if rows == 0 || rows > case.prepared_tokens || logits.len() != rows * GLOBAL {
            return Err(format!("{rows} rows past the chain's capacity"));
        }
        case.x = session.upload_bf16(&padded(x, case.prepared_tokens, H))?;
        case.logits = session.upload_f32(&padded(logits, case.prepared_tokens, GLOBAL))?;
        case.live = upload_bytes(session, &i32_bytes(&[rows as i32]))?;
        Ok(rows)
    }

    /// Each row's combined output of the prefill chain, on poisoned
    /// outputs checked after the launch.
    pub(crate) fn rows(
        &mut self,
        session: &Session,
        x: &[u16],
        logits: &[f32],
        shared: &[u16],
    ) -> LabResult<Units> {
        let case = self.prefill(session)?;
        let rows = Self::load(session, case, x, logits)?;
        let capacity = case.prepared_tokens;
        case.shared = session.upload_bf16(&padded(shared, capacity, H))?;
        case.poison_writes(session, true)?;
        case.tokens = rows;
        case.prepared_tokens = rows;
        let case = self.prefill.as_mut().expect("the prefill chain");
        let launched = case.launch(session, &self.banks, true);
        case.tokens = capacity;
        case.prepared_tokens = capacity;
        launched?;
        let (route, experts) = case.image(session, true)?;
        let layout = qwen_layout(case.rank_experts(), rows, true);
        route_written(&route, &layout, rows)
            .and_then(|ids| experts_written(&experts, &layout, &ids))
            .map_err(|e| format!("Qwen3.8 prefill chain of {rows} rows: {e}"))?;
        Ok(bf16_units(&experts.output, rows, H))
    }

    /// The prefill router's top-k expert IDs of each logits row, on
    /// poisoned IDs and weights checked after the launch.
    pub(crate) fn route_ids(&mut self, session: &Session, logits: &[f32]) -> LabResult<Vec<i32>> {
        let rows = logits.len() / GLOBAL;
        let case = self.prefill(session)?;
        Self::load(session, case, &vec![0; rows * H], logits)?;
        poison::poison(session, &[&case.ids, &case.weights])?;
        let capacity = case.prepared_tokens;
        case.tokens = rows;
        let launched = case.launch_entry(
            session,
            "router",
            &[&case.logits, &case.ids, &case.weights, &case.live],
            false,
            false,
        );
        case.tokens = capacity;
        launched?;
        session.stream.synchronize().map_err(err)?;
        ids_written(
            &poison::read(session, &case.ids)?,
            &poison::read(session, &case.weights)?,
            rows * QK,
        )
        .map_err(|e| format!("Qwen3.8 router of {rows} rows: {e}"))
    }

    /// Each token's `part` of the decode chain of `bucket` tokens over the
    /// given tokens (at most `bucket`), on poisoned outputs checked after
    /// the launch: a gate/up or down row is written for the rank's routes
    /// only.
    pub(crate) fn decode(
        &mut self,
        session: &Session,
        bucket: u32,
        part: Part,
        x: &[u16],
        logits: &[f32],
        shared: &[u16],
    ) -> LabResult<Units> {
        let tokens = x.len() / H;
        if tokens == 0 || tokens > bucket as usize || shared.len() != x.len() {
            return Err(format!("{tokens} tokens in a decode bucket of {bucket}"));
        }
        let index = match self
            .decode
            .iter()
            .position(|case| case.gate.tokens_capacity == bucket)
        {
            Some(index) => index,
            None => {
                let case =
                    Case::for_profile(session, self.profile, Phase::Decode, bucket as usize)?;
                self.decode.push(case);
                self.decode.len() - 1
            }
        };
        let case = &mut self.decode[index];
        let capacity = case.tokens;
        Self::load(session, case, x, logits)?;
        case.shared = session.upload_bf16(&padded(shared, capacity, H))?;
        case.poison_writes(session, false)?;
        case.tokens = tokens;
        let launched = case.launch(session, &self.banks, true);
        case.tokens = capacity;
        launched?;
        let (route, experts) = case.image(session, false)?;
        let layout = qwen_layout(case.rank_experts(), tokens, false);
        let ids = route_written(&route, &layout, tokens)
            .and_then(|ids| experts_written(&experts, &layout, &ids).map(|()| ids))
            .map_err(|e| format!("Qwen3.8 decode chain t{bucket} of {tokens} tokens: {e}"))?;
        // A token's local route rows, each after its slot.
        let routed_rows = |bytes: &[u8], width: usize| -> Units {
            (0..tokens)
                .map(|token| {
                    let mut unit = Vec::new();
                    for route in token * QK..(token + 1) * QK {
                        if layout.local(ids[route]).is_some() {
                            unit.extend(((route - token * QK) as u32).to_ne_bytes());
                            unit.extend_from_slice(
                                &bytes[route * width * 2..(route + 1) * width * 2],
                            );
                        }
                    }
                    unit
                })
                .collect()
        };
        Ok(match part {
            Part::Route => (0..tokens)
                .map(|token| {
                    let slots = token * QK * 4..(token + 1) * QK * 4;
                    let mut unit = route.ids[slots.clone()].to_vec();
                    unit.extend_from_slice(&route.weights[slots]);
                    unit
                })
                .collect(),
            Part::GateUp => routed_rows(&experts.inter, I),
            Part::Down => routed_rows(&experts.routed, H),
        })
    }

    /// Each row's output of the combine alone (on the prefill chain's
    /// buffers) over rows of hidden input `x`, shared-expert output
    /// `shared`, global expert IDs `ids` (top-k per row) and routed rows
    /// `routed` (top-k rows of `HIDDEN` per row), on a poisoned output: each
    /// live row must be written, the rows past them untouched.
    pub(crate) fn combine(
        &mut self,
        session: &Session,
        x: &[u16],
        shared: &[u16],
        ids: &[i32],
        routed: &[u16],
    ) -> LabResult<Units> {
        let rows = x.len() / H;
        let case = self.prefill(session)?;
        let capacity = case.prepared_tokens;
        if rows == 0
            || rows > capacity
            || shared.len() != x.len()
            || ids.len() != rows * QK
            || routed.len() != rows * QK * H
        {
            return Err(format!("a combine of {rows} rows"));
        }
        // Into the chain's buffers, which hold `capacity` rows.
        let bf16 =
            |values: &[u16]| -> Vec<u8> { values.iter().flat_map(|v| v.to_ne_bytes()).collect() };
        for (buffer, bytes) in [
            (&case.x, bf16(x)),
            (&case.shared, bf16(shared)),
            (&case.ids, i32_bytes(ids)),
            (&case.routed, bf16(routed)),
            (&case.live, i32_bytes(&[rows as i32])),
        ] {
            buffer
                .copy_from_bytes(&bytes, &session.stream)
                .map_err(err)?;
        }
        poison::poison(session, &[&case.output])?;
        case.tokens = rows;
        let launched = case.launch_combine(session, false);
        case.tokens = capacity;
        launched?;
        session.stream.synchronize().map_err(err)?;
        let output = poison::read(session, &case.output)?;
        rows_written("Qwen3.8 combine rows", &output, rows, H)?;
        Ok(bf16_units(&output, rows, H))
    }
}

#[test]
fn moe_qwen_write_checks_reject_unwritten_outputs() {
    use kernel_lab::poison::POISON;
    let one = 0x3f80_u16.to_ne_bytes();
    // Two tokens of a decode bucket of four on TP4 rank 1 (experts 128 to
    // 255): token 0 routes 0, 2 and 4 and token 1 routes 0 and 1 are local,
    // two of them to expert 130.
    let rank = 128..256;
    let ids = [
        130, 5, 140, 300, 129, 7, 8, 9, 10, 11, 130, 200, 1, 2, 3, 4, 6, 12, 13, 14,
    ];
    let layout = qwen_layout(rank.clone(), 2, false);
    let (route, experts) = chain_image(&layout, &ids, 4);
    assert_eq!(route_written(&route, &layout, 2), Ok(ids.to_vec()));
    assert!(experts_written(&experts, &layout, &ids).is_ok());
    // Nothing written.
    let poisoned = |bytes: &Vec<u8>| vec![POISON; bytes.len()];
    assert!(
        experts_written(
            &Experts {
                inter: poisoned(&experts.inter),
                routed: poisoned(&experts.routed),
                output: poisoned(&experts.output),
            },
            &layout,
            &ids
        )
        .is_err()
    );
    assert!(
        route_written(
            &Route {
                ids: poisoned(&route.ids),
                ..route.clone()
            },
            &layout,
            2
        )
        .is_err()
    );
    // Expert 130's second route slot left poisoned; its padding written.
    // Expert 129's range comes first.
    let first = DECODE_BLOCK;
    for sorted in [
        with(&route.sorted, first + 1, &[POISON; 4]),
        with(&route.sorted, first + 2, &0_i32.to_ne_bytes()),
        with(&route.sorted, first + 2, &[0x80; 4]),
    ] {
        let check = Route {
            sorted,
            ..route.clone()
        };
        assert!(route_written(&check, &layout, 2).is_err());
    }
    // The decode route lists its four active experts (129, 130, 140 and
    // 200) and nothing after.
    let active = route.active.clone().unwrap();
    for active in [
        with(&active, 3, &[POISON; 4]),
        with(&active, 5, &0_i32.to_ne_bytes()),
    ] {
        let check = Route {
            active: Some(active),
            ..route.clone()
        };
        assert!(route_written(&check, &layout, 2).is_err());
    }
    // Expert rows: a hole in a local route's gate/up row, a NaN down row,
    // a write to a route of another rank's expert (route 1, slot 1 of token
    // 0), a write to a route past the live tokens, and a combined row past
    // them.
    let nan = 0x7fc0_u16.to_ne_bytes();
    for (inter, routed, output) in [
        (
            with(&experts.inter, 4 * I + 9, &[POISON; 2]),
            experts.routed.clone(),
            experts.output.clone(),
        ),
        (
            experts.inter.clone(),
            with(&experts.routed, 10 * H + 1, &nan),
            experts.output.clone(),
        ),
        (
            experts.inter.clone(),
            with(&experts.routed, H, &one),
            experts.output.clone(),
        ),
        (
            with(&experts.inter, I, &one),
            experts.routed.clone(),
            experts.output.clone(),
        ),
        (
            with(&experts.inter, 20 * I, &one),
            experts.routed.clone(),
            experts.output.clone(),
        ),
        (
            experts.inter.clone(),
            experts.routed.clone(),
            with(&experts.output, 2 * H, &one),
        ),
    ] {
        let check = Experts {
            inter,
            routed,
            output,
        };
        assert!(experts_written(&check, &layout, &ids).is_err());
    }
    // A token without a local route leaves every expert row poisoned.
    let remote = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let (route, experts) = chain_image(&layout, &remote, 4);
    assert_eq!(route_written(&route, &layout, 1), Ok(remote.to_vec()));
    assert!(experts_written(&experts, &layout, &remote).is_ok());
    let check = Experts {
        routed: with(&experts.routed, 0, &one),
        ..experts
    };
    assert!(experts_written(&check, &layout, &remote).is_err());

    // The prefill permutation counts the live tiles only: 30 routes fill
    // one tile.
    let layout = qwen_layout(rank, 3, true);
    assert_eq!(layout.tiles, Some(1));
    let ids = (0..30).map(|route| route * 17 % 512).collect::<Vec<_>>();
    let (route, _) = chain_image(&layout, &ids, 4);
    assert_eq!(route_written(&route, &layout, 3), Ok(ids.clone()));
    let (tile_counts, tile_offsets) = route.tiles.clone().unwrap();
    for tiles in [
        (
            with(&tile_counts, 128, &0_i32.to_ne_bytes()),
            tile_offsets.clone(),
        ),
        (tile_counts.clone(), with(&tile_offsets, 5, &[POISON; 4])),
    ] {
        let check = Route {
            tiles: Some(tiles),
            ..route.clone()
        };
        assert!(route_written(&check, &layout, 3).is_err());
    }
    // The router alone: IDs and weights of the live routes.
    assert_eq!(ids_written(&route.ids, &route.weights, 30), Ok(ids.clone()));
    assert!(ids_written(&route.ids, &route.weights, 31).is_err());
    assert!(ids_written(&route.ids, &route.weights, 29).is_err());
}
