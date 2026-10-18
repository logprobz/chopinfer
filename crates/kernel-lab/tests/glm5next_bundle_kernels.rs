//! The glm5next bundle's latent attention kernels, run from the cubin the
//! bundle embeds (`bundle_glm5next::embedded_cubin`) under its own symbol
//! names, so the schedule nvcc gave that compilation unit is the one
//! qualified. Its TP4 geometry and entries are the lab's served ones (a CPU
//! test), so the lab's latent checks (`common/latent_checks.rs`) apply
//! unchanged: the oracle cases, the attention 128-K test, both projections,
//! the poisoned replays and the split merge order, which together launch
//! every one of its entries and merges. The lab-only entries (F32
//! projections, other KV write widths, the PDL producer) stay with the lab
//! cubin in `tests/latent_attention.rs`.
//!
//! The bundle's own ops as `Sm120Own` prepares them (the rounding of an
//! all-reduced F32 row, the widening of a BF16 row, the gather of the rows
//! that need logits, the drafter's residual add, its input halves and its
//! row-input gather, the stacked KDA input projections and commit's state
//! copy) run from the
//! cubins the bundle embeds through the conformance kit, each against its
//! contract oracle (`model_glm5next_contract::own_ops`) at every live count
//! of its bucket, poisoned through eager, relaunch, graph replay and
//! PDL-producer runs. The rounding, widening and residual add read NaNs of
//! both signs, quiet and signaling, with varied payloads, whose results the
//! contract states as a class (`own_ops::same_bf16`, `same_f32`); the runs
//! print the NaN bits the kernels wrote. Commit's state copy also runs
//! within one pool, as serving copies a checkpoint entry into a committed
//! entry of the same allocation.
//!
//! The stream expansion and the score compaction launch the shared copies
//! (`kernels_sm120::row::copy`) from the bundle's embedded
//! `row_copy_sm120`, which the conformance kit runs the same way against
//! the shared copy's oracle (`bundle_copies_match_their_oracles_poisoned`).
//!
//! The bundle's payload admission (`sm120_payload::CUBIN`, run at create over
//! each placed scale view) runs from its embedded cubin too: each check
//! reports no element of a valid payload and the lowest of two planted
//! invalid codes of every kind its rule refuses, as the contract's oracle
//! (`own_ops::state::admission`) does, over more elements than one grid
//! covers, poisoned through eager, relaunch, graph replay and a PDL
//! producer that writes the checked view.
//!
//! The drafter's argmax as the bundle prepares it on the shared sharded
//! argmax (the contract's `draft_argmax`) runs as a kit collective case on
//! four GPUs over captured head logits and special rows
//! (`the_shared_draft_argmax_agrees_on_four_ranks`).
//!
//! On four GPUs, the bucket-sized dense GEMMs and F32 all-reduce keep
//! their inactive rows out of the live ones (`bucket_rows_stay_out_of_live_rows`).
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test glm5next_bundle_kernels`, and the four-rank tests with
//! `CHOPINFER_KERNEL_GPUS=0,1,2,3`; without their GPUs the GPU tests skip.

#[path = "common/latent_checks.rs"]
mod latent_checks;

use bundle_glm5next::{
    cubin_of, embedded_cubin,
    sm120::{backend as sm120_backend, engines::Engines as BundleEngines, own_ops::Sm120Own},
};
use kernel_build::EntryParam;
use kernel_core::backend::{Backend, Counts, Need, Prepared, Refused, View};
use kernel_core::collective::{GatherBuffers, RankViews};
use kernel_core::contract::{Argument, DType, WeightFormat};
use kernel_core::precision::{ExecutionPhase, OperationUse};
use kernel_core::properties::Property;
use kernel_core::row_copy::{GatherByIndexBuffers, RowCopyBuffers};
use kernel_lab::conformance::{Expected, Holes, Images, Probes, Rows, Write};
use kernel_lab::kit::collective::{CollectiveCase, RequiredCollective, Slot};
use kernel_lab::kit::modules::{Functions, Modules};
use kernel_lab::kit::{self, OpCase, OpResult, Outcome, Spec};
use kernel_lab::latent_shapes::{self, GEOMETRY};
use kernel_lab::poison::{self, CopyProducer, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, gpu_test_multi, launch};
use kernels_sm120::latent::Entry;
use latent_checks::{
    Kernels, Report, attention_accumulation, oracle_cases, poisoned_replay, projections,
    split_merge_order,
};
use model_glm5next_contract::{
    contract::Shape,
    copies,
    draft_argmax::{self, DraftArgmax, DraftArgmaxBuffers},
    own_ops::{
        OwnOps,
        draft::{self as draft_ops, DraftGather, DraftGatherBuffers},
        rows::{
            self as row_ops, EhInput, EhInputBuffers, ResidualAdd, ResidualAddBuffers, RoundRows,
            RoundRowsBuffers, WidenRows, WidenRowsBuffers,
        },
        same_bf16, same_f32,
        stacked::{self as stacked_ops, StackedPart, StackedProjection, StackedProjectionBuffers},
        state::{self as state_ops, Admission, StateCopy, StateCopyBuffers},
    },
    payload::{self, Check},
    precision, quantization,
    sm120::{
        context_parallel as cp_entries,
        engines::{self, ModelRow},
        payload as sm120_payload, precision as sm120_precision, projection_formats,
        stacked as kda_stacked, state_copy,
    },
};
use runtime_cuda::{DeviceBuffer, KernelArgument, Stream};
use serde_json::json;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

/// The bundle's latent entries at TP4, refused unless its geometry is the
/// lab's, so the lab's oracle shapes apply to them.
fn bundle_entries() -> LabResult<Vec<Entry>> {
    let geometry = engines::latent_geometry(Shape::TP4);
    if geometry != GEOMETRY {
        return Err(format!(
            "the bundle's latent geometry {geometry:?} is not the lab's {GEOMETRY:?}"
        ));
    }
    Ok(engines::latent_entries(Shape::TP4))
}

/// The one embedded cubin holding every symbol of `entries`.
fn bundle_cubin(entries: &[Entry]) -> LabResult<&'static str> {
    let mut cubins = BTreeSet::new();
    for entry in entries {
        for symbol in std::iter::once(entry.symbol()).chain(entry.merge_symbol()) {
            cubins.insert(cubin_of(&symbol).ok_or(format!("no embedded cubin holds {symbol}"))?);
        }
    }
    match Vec::from_iter(cubins)[..] {
        [cubin] => Ok(cubin),
        ref other => Err(format!("the bundle's latent entries span cubins {other:?}")),
    }
}

/// The bundle's latent entries are the lab's served entries: the same
/// geometry, the same queries in the same order and the same kernel names
/// under each profile's tag, so every latent check applies to them; one
/// embedded cubin holds all their kernels and merges.
#[test]
fn bundle_latent_entries_are_the_lab_served_entries() {
    let bundle = bundle_entries().expect("the bundle's latent geometry is the lab's");
    let lab = kernels_sm120::latent::entries(latent_shapes::PROFILE, GEOMETRY);
    let queries = |entries: &[Entry]| entries.iter().map(|e| e.query).collect::<Vec<_>>();
    assert_eq!(queries(&bundle), queries(&lab));
    let kernel_names = |entries: &[Entry]| {
        entries
            .iter()
            .map(|e| {
                let tag = format!("{}_", e.profile.tag());
                e.name.strip_prefix(&tag).map(str::to_owned)
            })
            .collect::<Option<Vec<_>>>()
            .expect("every entry name starts with its profile tag")
    };
    assert_eq!(kernel_names(&bundle), kernel_names(&lab));
    bundle_cubin(&bundle).expect("one embedded cubin holds the latent entries");
}

/// The bundle's embedded latent cubin passes every check of the served
/// entries, and the checks launch each of its kernels and merges.
#[test]
fn bundle_latent_cubin_passes_the_served_checks() {
    gpu_test(8 << 30, |session| {
        let entries = bundle_entries()?;
        let name = bundle_cubin(&entries)?;
        let cubin = embedded_cubin(name).ok_or(format!("no embedded cubin {name}"))?;
        let kernels = &Kernels::embedded(session, name, cubin, entries)?;
        let mut failures = Vec::new();
        oracle_cases(session, kernels).unwrap_or_else(|e| failures.push(e));
        let mut report = Report::new(kernels, "accumulation");
        attention_accumulation(session, kernels, &mut report)?;
        report
            .finish(session, BTreeMap::new())
            .unwrap_or_else(|e| failures.push(e));
        projections(session, kernels).unwrap_or_else(|e| failures.push(e));
        poisoned_replay(session, kernels).unwrap_or_else(|e| failures.push(e));
        split_merge_order(session, kernels).unwrap_or_else(|e| failures.push(e));
        let unlaunched = kernels.unlaunched();
        if !unlaunched.is_empty() {
            failures.push(format!("no check launched {unlaunched:?}"));
        }
        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures.join("\n"))
        }
    })
    .expect("bundle latent cubin");
}

// ------------------------------------------------------------- own ops ----

/// The bucket every own-row case plans for; its live counts are
/// `poison::live_counts(CAPACITY)`.
const CAPACITY: u32 = 32;
/// Rows the gather reads from: more than the bucket, so the indices reach
/// past it.
const GATHER_SOURCE_ROWS: usize = 101;
/// Retained rows the drafter's row gather reads from.
const RETAINED_ROWS: usize = 40;
/// Verify input tokens the draft argmax cases scatter into.
const VERIFY_TOKENS: usize = 64;

/// The one embedded cubin holding every own row.
fn own_row_cubin() -> LabResult<&'static str> {
    let cubins = ModelRow::ALL
        .iter()
        .map(|row| {
            cubin_of(&row.symbol()).ok_or(format!("no embedded cubin holds {}", row.symbol()))
        })
        .collect::<LabResult<BTreeSet<_>>>()?;
    match Vec::from_iter(cubins)[..] {
        [cubin] => Ok(cubin),
        ref other => Err(format!("the own rows span cubins {other:?}")),
    }
}

/// The bundle's embedded cubin `name`, loaded on `session`.
fn embedded(session: &Session, name: &str) -> LabResult<runtime_cuda::Module> {
    session.load_cubin(embedded_cubin(name).ok_or(format!("no embedded cubin {name}"))?)
}

/// The embedded own-row cubin, loaded on `session`.
fn own_row_module(session: &Session) -> LabResult<runtime_cuda::Module> {
    embedded(session, own_row_cubin()?)
}

/// The bundle's embedded cubins, which hold every symbol a prepared own op
/// launches, and SM120's PDL probes (the live-count producer and the latent
/// lab module's copy producer).
struct Embedded;

impl Modules for Embedded {
    fn load<'s>(&'s self, session: &'s Session) -> LabResult<Functions<'s>> {
        let loaded = RefCell::new(BTreeMap::<&'static str, runtime_cuda::Module>::new());
        Ok(Box::new(move |symbol| {
            let name = cubin_of(symbol).ok_or(format!("no embedded cubin holds {symbol}"))?;
            let mut loaded = loaded.borrow_mut();
            if !loaded.contains_key(name) {
                loaded.insert(name, embedded(session, name)?);
            }
            loaded[name].function(symbol).map_err(|e| e.to_string())
        }))
    }

    fn probes(&self, session: &Session) -> LabResult<Option<Probes>> {
        let lab = session.module(&latent_shapes::cubin_name())?;
        Probes::sm120(session, Some(&lab)).map(Some)
    }
}

/// Deterministic values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// Finite BF16 values of both signs, exponents -16 through 15.
    fn bf16(&mut self, count: usize) -> Vec<u16> {
        (0..count)
            .map(|_| {
                let bits = self.next();
                (bits as u16 & 0x807f) | ((111 + (bits >> 16) % 32) as u16) << 7
            })
            .collect()
    }

    /// Finite F32 values, a third of them on a BF16 rounding tie or one ULP
    /// either side of it, so the rounding's ties-to-even shows.
    fn f32(&mut self, count: usize) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let bits = self.next();
                let high = (bits & 0x807f_0000) | ((111 + (bits >> 8) % 32) << 23);
                let low = match index % 9 {
                    0 => 0x8000,
                    1 => 0x7fff,
                    2 => 0x8001,
                    _ => self.next() & 0xffff,
                };
                f32::from_bits(high | low)
            })
            .collect()
    }
}

fn halves(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn words(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bits(values: &[f32]) -> Vec<u8> {
    words(
        &values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    )
}

/// F32 NaNs of both signs, quiet and signaling, with varied payloads; no
/// top half is all ones, so no rounding of them reads as the poison.
const F32_NANS: [u32; 6] = [
    0x7fc0_0000,
    0xffc0_0000,
    0x7f80_0001,
    0xff80_4321,
    0x7fa5_a5a5,
    0xffe0_1234,
];
/// BF16 NaNs of both signs, quiet and signaling, with varied payloads.
const BF16_NANS: [u16; 6] = [0x7fc0, 0xffc0, 0x7f81, 0xff85, 0x7fa5, 0xffd3];

/// A judge of a BF16 buffer against an oracle's bits: its first `classed`
/// elements by `same_bf16` (an own row's NaN results are NaN of any sign
/// and payload), the elements up to `exact` bit for bit. At the bucket's
/// largest live count it prints the NaN bits the kernel wrote.
fn bf16_judge(name: &'static str, expected: Vec<u16>, classed: usize, exact: usize) -> Expected {
    Expected::Judge(Box::new(move |bytes| {
        let mut nans = BTreeSet::new();
        for (index, &expected) in expected.iter().enumerate().take(classed.max(exact)) {
            let actual = u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
            let same = if index < classed {
                same_bf16(expected, actual)
            } else {
                expected == actual
            };
            if !same {
                return Err(format!(
                    "element {index} is {actual:#06x}, the oracle {expected:#06x}"
                ));
            }
            if index < classed && actual & 0x7f80 == 0x7f80 && actual & 0x7f != 0 {
                nans.insert(actual);
            }
        }
        if classed == CAPACITY as usize * Shape::TP4.profile.hidden as usize {
            eprintln!("{name} NaN results: {nans:#06x?}");
        }
        Ok(())
    }))
}

fn upload(session: &Session, bytes: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(bytes.len())?;
    poison::write(session, &buffer, 0, bytes)?;
    Ok(buffer)
}

/// An oracle's refusal as a lab error.
fn refused(refusal: kernel_core::contract::Refusal) -> String {
    refusal.0.to_string()
}

/// The u32 words of a case image.
fn image_words(images: &Images, name: &str) -> LabResult<Vec<u32>> {
    Ok(images.i32s(name)?.into_iter().map(|w| w as u32).collect())
}

/// An op `Sm120Own` prepared, or its refusal as a lab error.
fn own_op<B>(result: Result<Prepared<B>, Refused>) -> LabResult<Prepared<B>> {
    result.map_err(|refused| format!("Sm120Own refuses an own op: {refused}"))
}

/// One own op's case: its name, the need it was prepared for, the case.
type OwnCase = (&'static str, Need, OpCase<'static>);

/// The own rows as `Sm120Own` prepares them over the bundle's TP4 shape,
/// each at a 32-row bucket against its contract oracle (`own_ops::rows`,
/// `own_ops::draft`): every live count of the bucket, poisoned through
/// eager, relaunch, graph and PDL-producer runs by the conformance kit.
fn own_row_cases(rng: &mut Rng) -> LabResult<Vec<OwnCase>> {
    let shape = Shape::TP4;
    let own = Sm120Own::new(shape);
    let need = Need::rows(CAPACITY);
    let p = shape.profile;
    let (width, rows) = (p.hidden as usize, CAPACITY as usize);
    let mut cases = Vec::new();

    // Round: F32 rows to BF16, ties to even; NaNs of both signs, quiet and
    // signaling, with varied payloads round to a NaN.
    let op = RoundRows::of(shape);
    let mut values = rng.f32(rows * width);
    for (index, value) in values.iter_mut().enumerate().skip(5).step_by(37) {
        *value = f32::from_bits(F32_NANS[index % F32_NANS.len()]);
    }
    let roles = RoundRowsBuffers {
        input: Spec::input(DType::F32, (Rows::Live, width), bits(&values)),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.round_rows(&op, need))?, CAPACITY, roles)?.oracle(
        move |images, live| {
            let mut output = vec![0; rows * width];
            row_ops::round_rows(&op, &images.f32s("input")?, &mut output, live, CAPACITY)
                .map_err(refused)?;
            let judge = bf16_judge("round", output, live as usize * width, 0);
            Ok(vec![("output".into(), judge)])
        },
    );
    cases.push(("round", need, case));

    // Widen: BF16 rows to F32, exact for numbers; NaNs of both signs,
    // quiet and signaling, with varied payloads widen to a NaN.
    let op = WidenRows::of(shape);
    let mut values = rng.bf16(rows * width);
    for (index, value) in values.iter_mut().enumerate().skip(3).step_by(41) {
        *value = BF16_NANS[index % BF16_NANS.len()];
    }
    let roles = WidenRowsBuffers {
        input: Spec::input(DType::Bf16, (Rows::Live, width), halves(&values)),
        output: Spec::output(DType::F32, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.widen_rows(&op, need))?, CAPACITY, roles)?.oracle(
        move |images, live| {
            let mut output = vec![0; rows * width];
            row_ops::widen_rows(&op, &images.u16s("input")?, &mut output, live, CAPACITY)
                .map_err(refused)?;
            let compared = live as usize * width;
            let judge = move |bytes: &[u8]| {
                let mut nans = BTreeSet::new();
                for (index, &expected) in output.iter().enumerate().take(compared) {
                    let actual =
                        u32::from_le_bytes(bytes[4 * index..4 * index + 4].try_into().unwrap());
                    if !same_f32(expected, actual) {
                        return Err(format!(
                            "element {index} is {actual:#010x}, the oracle {expected:#010x}"
                        ));
                    }
                    if f32::from_bits(actual).is_nan() {
                        nans.insert(actual);
                    }
                }
                if live == CAPACITY {
                    eprintln!("widen NaN results: {nans:#010x?}");
                }
                Ok(())
            };
            Ok(vec![("output".into(), Expected::Judge(Box::new(judge)))])
        },
    );
    cases.push(("widen", need, case));

    // Residual add in place: x = bf16(x + y) for the live rows, which the
    // kernel reads and rewrites, the rest of x unchanged; NaNs in x and y
    // of both signs, quiet and signaling, with varied payloads add to a NaN.
    let op = ResidualAdd::of(shape);
    let mut x = rng.bf16(rows * width);
    let mut y = rng.bf16(rows * width);
    for (index, value) in x.iter_mut().enumerate().skip(7).step_by(43) {
        *value = BF16_NANS[index % BF16_NANS.len()];
    }
    for (index, value) in y.iter_mut().enumerate().skip(11).step_by(47) {
        *value = BF16_NANS[(index + 3) % BF16_NANS.len()];
    }
    let roles = ResidualAddBuffers {
        x: Spec::state_updating(
            DType::Bf16,
            (Rows::Live, width),
            halves(&x),
            Write::Map(Box::new(|_, _| WriteMap::of(DType::Bf16).none())),
            move |_, live| Ok(std::iter::once(0..live as usize * width * 2).collect()),
        ),
        y: Spec::input(DType::Bf16, (Rows::Live, width), halves(&y)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.residual_add(&op, need))?, CAPACITY, roles)?.oracle(
        move |images, live| {
            let mut x = images.u16s("x")?;
            row_ops::residual_add(&op, &mut x, &images.u16s("y")?, live, CAPACITY)
                .map_err(refused)?;
            let judge = bf16_judge("residual add", x, live as usize * width, rows * width);
            Ok(vec![("x".into(), judge)])
        },
    );
    cases.push(("residual_add", need, case));

    // The drafter's input halves at a 32-row decode use: the normalized
    // embedding rows at column 0, the hidden rows at the hidden width.
    let op = EhInput::of(shape);
    let usage = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: CAPACITY,
        graph: true,
    };
    let roles = EhInputBuffers {
        embed: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            halves(&rng.bf16(rows * width)),
        ),
        hidden: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            halves(&rng.bf16(rows * width)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, 2 * width)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.eh_input(&op, usage, need))?, CAPACITY, roles)?.oracle(
        move |images, live| {
            let mut output = vec![0; rows * 2 * width];
            row_ops::eh_input(
                &op,
                &images.u16s("embed")?,
                &images.u16s("hidden")?,
                &mut output,
                live,
                CAPACITY,
            )
            .map_err(refused)?;
            Ok(vec![("output".into(), Expected::Bits(halves(&output)))])
        },
    );
    cases.push(("eh_input", need, case));

    // The drafter's row inputs: hidden rows from 40 retained rows (negative
    // codes, -1 among them) or the previous program's rows in reverse,
    // tokens kept (a negative code) or taken from a drafting item's draft.
    let op = DraftGather::of(shape);
    let codes = (0..rows as i32)
        .flat_map(|row| {
            let hidden = match row % 3 {
                0 => -1 - (row * 7) % RETAINED_ROWS as i32,
                _ => rows as i32 - 1 - row,
            };
            let token = if row % 4 == 0 {
                (row * 5 + 3) % rows as i32
            } else {
                -1 - row % 2
            };
            [hidden, token]
        })
        .collect::<Vec<_>>();
    let drafted = {
        let codes = codes.clone();
        move |live: u32| {
            (0..live as usize)
                .filter(|&row| codes[2 * row + 1] >= 0)
                .collect::<Vec<_>>()
        }
    };
    let draft_values = (0..rows as u32)
        .map(|item| 1000 + 37 * item)
        .collect::<Vec<_>>();
    let roles = DraftGatherBuffers {
        retained: Spec::input(
            DType::Bf16,
            (Rows::Fixed(RETAINED_ROWS), width),
            halves(&rng.bf16(RETAINED_ROWS * width)),
        ),
        last: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            halves(&rng.bf16(rows * width)),
        ),
        codes: Spec::input(
            DType::I32,
            (Rows::Live, 2),
            words(&codes.iter().map(|&code| code as u32).collect::<Vec<_>>()),
        ),
        drafts: Spec::input(DType::I32, (Rows::Fixed(rows), 1), words(&draft_values)),
        live_rows: Spec::live(),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        tokens: Spec::output_with(
            DType::I32,
            (Rows::Live, 1),
            Write::Map(Box::new(move |_, live| {
                WriteMap::of(DType::I32).rows(1, drafted(live))
            })),
        ),
    };
    let case = OpCase::new(own_op(own.draft_gather(&op, need))?, CAPACITY, roles)?.oracle(
        move |images, live| {
            let (mut output, mut tokens) = (vec![0; rows * width], vec![0; rows]);
            rows_of_draft_gather(&op, images, live, &mut output, &mut tokens)?;
            Ok(vec![
                ("output".into(), Expected::Bits(halves(&output))),
                ("tokens".into(), Expected::Bits(words(&tokens))),
            ])
        },
    );
    cases.push(("draft_gather", need, case));

    Ok(cases)
}

/// The draft gather's oracle over a case's images.
fn rows_of_draft_gather(
    op: &DraftGather,
    images: &Images,
    live: u32,
    output: &mut [u16],
    tokens: &mut [u32],
) -> LabResult<()> {
    draft_ops::draft_gather(
        op,
        &images.u16s("retained")?,
        &images.u16s("last")?,
        &images.i32s("codes")?,
        &image_words(images, "drafts")?,
        live,
        output,
        tokens,
        CAPACITY,
    )
    .map_err(refused)
}

/// The stacked KDA input projections at the 8-row decode use, over random
/// BF16 activations and weights, each part's output judged within the
/// contract's bound (`own_ops::stacked`) at its row stride.
fn stacked_case(rng: &mut Rng) -> LabResult<OwnCase> {
    let shape = Shape::TP4;
    let usage = OperationUse {
        phase: ExecutionPhase::Decode,
        capacity: 8,
        graph: true,
    };
    let entry = kda_stacked::part(kda_stacked::PROJECTIONS[0], WeightFormat::Bf16, usage)?
        .ok_or("no stacked KDA input projections at an 8-row decode use")?
        .entry;
    let op = StackedProjection {
        k: entry.k,
        parts: entry
            .parts
            .iter()
            .map(|part| StackedPart {
                rows: part.rows,
                width: part.width,
                row_stride: part.row_stride,
            })
            .collect(),
    };
    let need = Need::rows(usage.capacity);
    let prepared = Sm120Own::new(shape)
        .stacked_projection(&op, usage, need)
        .map_err(|refused| format!("Sm120Own refuses the stacked projections: {refused}"))?;
    let (k, rows) = (op.k as usize, usage.capacity as usize);
    let weights = op
        .parts
        .iter()
        .map(|part| rng.bf16(part.rows as usize * k))
        .collect::<Vec<_>>();
    let roles = StackedProjectionBuffers {
        x: Spec::input(DType::Bf16, (Rows::Live, k), halves(&rng.bf16(rows * k))),
        weights: weights.iter().map(|w| Spec::weight(halves(w))).collect(),
        outputs: op
            .parts
            .iter()
            .map(|&part| {
                let (stride, width) = (part.row_stride as usize, part.width as usize);
                Spec::output_with(
                    DType::Bf16,
                    (Rows::Live, stride),
                    Write::Map(Box::new(move |_, live| {
                        WriteMap::of(DType::Bf16).ranges(
                            (0..live as usize).map(|row| row * stride..row * stride + width),
                        )
                    })),
                )
            })
            .collect(),
        tickets: Spec::untouched(16),
    };
    // Rows are independent: the capacity's reference serves every live
    // count, whose judge reads its live rows.
    let reference = RefCell::new(None);
    let case = OpCase::new(prepared, usage.capacity, roles)?.oracle(move |images, live| {
        let mut cached = reference.borrow_mut();
        if cached.is_none() {
            let x = images.u16s("x")?;
            let parts = weights.iter().map(Vec::as_slice).collect::<Vec<_>>();
            *cached = Some(
                stacked_ops::stacked_projection(&op, &x, &parts, usage.capacity)
                    .map_err(refused)?,
            );
        }
        let references = cached.as_ref().expect("computed above");
        Ok(op
            .parts
            .iter()
            .zip(references)
            .enumerate()
            .map(|(index, (&part, reference))| {
                let reference = reference.clone();
                let judge = move |bytes: &[u8]| {
                    let output = bytes
                        .chunks_exact(2)
                        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                        .collect::<Vec<_>>();
                    let width = part.width as usize;
                    let live_reference = kernel_core::oracle::OracleOutput {
                        values: reference.values[..live as usize * width].to_vec(),
                        sum_abs_products: reference.sum_abs_products[..live as usize * width]
                            .to_vec(),
                        reduction_lengths: reference.reduction_lengths[..live as usize * width]
                            .to_vec(),
                    };
                    stacked_ops::check_part(&part, &live_reference, &output, live)
                        .map_err(|failure| format!("part {index}: {failure:?}"))
                };
                (
                    format!("outputs[{index}]"),
                    Expected::Judge(Box::new(judge)),
                )
            })
            .collect())
    });
    Ok(("stacked_projection", need, case))
}

/// The bucket of the wide own-row cases: past the decode buckets, so the
/// kit's property checks compare a launch of the 256-row bucket against
/// launches of smaller buckets.
const WIDE: u32 = 256;

/// What the bundle asks of its own target rows, which run in every phase
/// (its `ops::exact::every_phase`): rows kept over chunk
/// splits on the 64-row granule and the one-row bits of the widest verify
/// launch, 32 rows. The gather of the rows that need logits runs in
/// prefill alone and asks the chunk splits only. None writes per-slot
/// state, so none asks anchors.
fn served_ask(capacity: u32, verify: bool) -> Need {
    use kernel_core::properties::{Chunk, Invariance};
    let prefill = Invariance {
        chunk: Chunk::Granule(64),
        ..Invariance::NONE
    };
    let exact = match verify {
        true => Invariance {
            rows: kernel_core::properties::Rows::OneRow { max: 32 },
            ..prefill
        },
        false => prefill,
    };
    Need::rows(capacity).with_exact(exact)
}

/// The bundle's declared own rows (`Sm120Own`'s `declared_kernels`: rounding
/// and widening)
/// at the [`WIDE`] bucket, each prepared for what the bundle asks of it and
/// checked against its contract oracle: the kit then checks the rows,
/// chunk and batch claims at 256 rows, whose launches grid buckets from
/// one to 256 rows.
fn wide_row_cases(rng: &mut Rng) -> LabResult<Vec<OwnCase>> {
    let shape = Shape::TP4;
    let own = Sm120Own::new(shape);
    let p = shape.profile;
    let (width, rows) = (p.hidden as usize, WIDE as usize);
    let mut cases = Vec::new();

    let need = served_ask(WIDE, true);
    let op = RoundRows::of(shape);
    let roles = RoundRowsBuffers {
        input: Spec::input(
            DType::F32,
            (Rows::Live, width),
            bits(&rng.f32(rows * width)),
        ),
        output: Spec::output(DType::Bf16, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.round_rows(&op, need))?, WIDE, roles)?.oracle(
        move |images, live| {
            let mut output = vec![0; rows * width];
            row_ops::round_rows(&op, &images.f32s("input")?, &mut output, live, WIDE)
                .map_err(refused)?;
            Ok(vec![("output".into(), Expected::Bits(halves(&output)))])
        },
    );
    cases.push(("round wide", need, case));

    let op = WidenRows::of(shape);
    let roles = WidenRowsBuffers {
        input: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            halves(&rng.bf16(rows * width)),
        ),
        output: Spec::output(DType::F32, (Rows::Live, width)),
        live_rows: Spec::live(),
    };
    let case = OpCase::new(own_op(own.widen_rows(&op, need))?, WIDE, roles)?.oracle(
        move |images, live| {
            let mut output = vec![0; rows * width];
            row_ops::widen_rows(&op, &images.u16s("input")?, &mut output, live, WIDE)
                .map_err(refused)?;
            Ok(vec![("output".into(), Expected::Bits(words(&output)))])
        },
    );
    cases.push(("widen wide", need, case));

    Ok(cases)
}

/// Every own op's case.
fn own_cases() -> LabResult<Vec<OwnCase>> {
    let mut rng = Rng(0x0c1a_55e5);
    let mut cases = own_row_cases(&mut rng)?;
    cases.push(stacked_case(&mut rng)?);
    cases.extend(state_copy_cases(&mut rng)?);
    Ok(cases)
}

/// Every own op's case declares on the CPU: each role bound, images sized,
/// the oracle covering every output and state it writes, the launches at
/// the capacity within the op's info, and the op's guarantees meeting its
/// need; so do the wide own-row cases, whose ops the bundle's asks prepare.
#[test]
fn bundle_own_op_cases_are_declared_whole() {
    let wide = wide_row_cases(&mut Rng(0x0c1a_55e5)).expect("wide own-row cases");
    assert_eq!(wide.len(), 2);
    for (name, need, case) in &wide {
        for check in case.checks(*need, None, None, None) {
            match (check.name.as_str(), &check.outcome) {
                ("declaration" | "need", Outcome::Passed(_)) => {}
                (_, Outcome::NotExercised(_)) => {}
                (_, outcome) => panic!("{name} {}: {outcome:?}", check.name),
            }
        }
    }
    let cases = own_cases().expect("own op cases");
    assert_eq!(
        cases.len(),
        ModelRow::ALL.len() + 2 + state_copy_layouts().len()
    );
    for (name, need, case) in &cases {
        for check in case.checks(*need, None, None, None) {
            match (check.name.as_str(), &check.outcome) {
                ("declaration" | "need", Outcome::Passed(_)) => {}
                (_, Outcome::NotExercised(_)) => {}
                (_, outcome) => panic!("{name} {}: {outcome:?}", check.name),
            }
        }
    }
}

/// Every own op `Sm120Own` prepares (each own row, the drafter's input
/// halves, the stacked KDA input projections and commit's state copy)
/// matches its contract oracle through the conformance kit at every live
/// count, writes only what its write map names, and reproduces its eager
/// bits poisoned through relaunch, graph replay and a PDL producer that
/// writes an input. Every check must pass; a check may go unexercised only
/// where it does not apply (an op that waits on no PDL producer, or reads
/// no device live count).
#[test]
fn bundle_own_ops_match_their_oracles_poisoned() {
    gpu_test(1 << 30, |session| {
        let functions = Embedded.load(session)?;
        let probes = Embedded.probes(session)?;
        let mut evidence = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, need, case) in own_cases()? {
            let checks = case.checks(need, Some(session), Some(&*functions), probes.as_ref());
            let mut receipt = BTreeMap::new();
            for check in checks {
                match &check.outcome {
                    Outcome::Passed(value) => {
                        receipt.insert(check.name.clone(), value.clone());
                    }
                    Outcome::NotExercised(reason)
                        if check.inapplicable() || check.name == "device live count" =>
                    {
                        receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                    }
                    outcome => failures.push(format!("{name} {}: {outcome:?}", check.name)),
                }
            }
            evidence.insert(name.to_owned(), json!(receipt));
        }
        if !failures.is_empty() {
            return Err(failures.join("\n"));
        }
        session.receipt(
            "bundle_own_ops_poisoned",
            BTreeMap::from([("ops".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle own ops");
}

/// The bundle's declared own rows at the 256-row bucket (`wide_row_cases`),
/// prepared for its asks, match their contract oracles through the
/// conformance kit, poisoned through relaunch, graph replay and a PDL
/// producer, and keep the claims they declare: each row's bits at 256 rows
/// equal its bits alone and beside copies of itself, and splits at 1, 128
/// and 255 rows give the bits of the whole launch. Every check must pass,
/// the rows, chunk and batch checks among them.
#[test]
fn bundle_own_rows_keep_their_claims_at_a_wide_bucket() {
    gpu_test(1 << 30, |session| {
        let functions = Embedded.load(session)?;
        let probes = Embedded.probes(session)?;
        let mut evidence = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, need, case) in wide_row_cases(&mut Rng(0x0c1a_55e5))? {
            let checks = case.checks(need, Some(session), Some(&*functions), probes.as_ref());
            let mut receipt = BTreeMap::new();
            let mut properties = 0;
            for check in checks {
                properties += usize::from(
                    ["rows", "chunk", "batch"]
                        .iter()
                        .any(|p| check.name.contains(p))
                        && matches!(check.outcome, Outcome::Passed(_)),
                );
                match &check.outcome {
                    Outcome::Passed(value) => {
                        receipt.insert(check.name.clone(), value.clone());
                    }
                    Outcome::NotExercised(reason) if check.inapplicable() => {
                        receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                    }
                    outcome => failures.push(format!("{name} {}: {outcome:?}", check.name)),
                }
            }
            if properties < 3 {
                failures.push(format!(
                    "{name}: {properties} property checks passed, not 3"
                ));
            }
            evidence.insert(name.to_owned(), json!(receipt));
        }
        if !failures.is_empty() {
            return Err(failures.join("\n"));
        }
        session.receipt(
            "bundle_own_rows_wide",
            BTreeMap::from([("ops".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle own rows at a wide bucket");
}

/// The shared row copies' entries (`kernels_sm120::row::copy`) and the
/// shared gather by index (`kernels_sm120::row::gather`) on SM120, as the
/// bundle registers them.
fn shared_copies() -> kernels_sm120::backend::Sm120 {
    kernels_sm120::row::copy::entries()
        .into_iter()
        .fold(kernels_sm120::backend::Sm120::new(), |backend, entry| {
            backend.with_row_copy(entry)
        })
        .with_row_gather(kernels_sm120::row::gather::SYMBOL)
}

/// The head rows' gather by index (`copies::head_rows`) at `capacity` rows
/// for `need`, from a source of `source` rows by device indices that run
/// backwards, stride past the bucket and repeat; the kit reruns them moved
/// to the top of the source.
fn gather_case(
    rng: &mut Rng,
    shared: &kernels_sm120::backend::Sm120,
    capacity: u32,
    need: Need,
    source: usize,
) -> LabResult<OpCase<'static>> {
    let op = copies::head_rows(Shape::TP4);
    let (width, rows) = (op.width as usize, capacity as usize);
    let mut indices = (0..rows as i32)
        .map(|i| (i * 37 + 11) % source as i32)
        .collect::<Vec<_>>();
    indices[0] = source as i32 - 1;
    indices[1] = 0;
    indices[5] = indices[3];
    let roles = GatherByIndexBuffers {
        source: Spec::input(
            DType::Bf16,
            (Rows::Fixed(source), width),
            halves(&rng.bf16(source * width)),
        ),
        row_indices: Spec::ids(
            (Rows::Live, 1),
            indices,
            &[("source", width * 2)],
            Holes::Never,
        ),
        count: Spec::live(),
        destination: Spec::output(DType::Bf16, (Rows::Live, width)),
    };
    let prepared = shared
        .gather_by_index(&op, need)
        .map_err(|refused| format!("SM120 refuses the bundle's head rows: {refused}"))?;
    Ok(
        OpCase::new(prepared, capacity, roles)?.oracle(move |images, live| {
            let mut output = vec![0; rows * width * 2];
            let indices = image_words(images, "row_indices")?;
            kernel_core::row_copy::gather_by_index(
                &op,
                images.bytes("source")?,
                &indices,
                live,
                &mut output,
            )
            .map_err(refused)?;
            Ok(vec![("destination".into(), Expected::Bits(output))])
        }),
    )
}

/// The bundle's stream expansion, router score compaction and head rows'
/// gather at its TP4 shape (the contract's `copies`), as SM120 prepares the
/// shared copies the bundle launches, at a 32-row bucket (and the gather at
/// the wide bucket too) against the shared copy's oracle
/// (`kernel_core::row_copy`): every live count, poisoned through eager,
/// relaunch, graph and PDL-producer runs, from the bundle's embedded
/// `row_copy_sm120` and `row_gather_sm120`.
fn copy_cases(rng: &mut Rng) -> LabResult<Vec<OwnCase>> {
    let shape = Shape::TP4;
    let shared = shared_copies();
    let need = Need::rows(CAPACITY);
    let rows = CAPACITY as usize;
    let mut cases = Vec::new();

    let broadcast = copies::expand_streams(shape);
    let (width, streams) = (broadcast.width as usize, broadcast.streams as usize);
    let roles = RowCopyBuffers {
        source: Spec::input(
            DType::Bf16,
            (Rows::Live, width),
            halves(&rng.bf16(rows * width)),
        ),
        destination: Spec::output(DType::Bf16, (Rows::Live, streams * width)),
        live_rows: Spec::live(),
    };
    let prepared = shared
        .broadcast_streams(&broadcast, need)
        .map_err(|refused| format!("SM120 refuses the bundle's broadcast: {refused}"))?;
    let case = OpCase::new(prepared, CAPACITY, roles)?.oracle(move |images, live| {
        let mut output = vec![0; rows * streams * width * 2];
        kernel_core::row_copy::broadcast_streams(
            &broadcast,
            images.bytes("source")?,
            &mut output,
            live,
        )
        .map_err(refused)?;
        Ok(vec![("destination".into(), Expected::Bits(output))])
    });
    cases.push(("expand_streams", need, case));

    let slice = copies::router_scores(shape);
    let (columns, stride) = (slice.width as usize, slice.stride as usize);
    let roles = RowCopyBuffers {
        source: Spec::input(
            DType::F32,
            (Rows::Live, stride),
            bits(&rng.f32(rows * stride)),
        ),
        destination: Spec::output(DType::F32, (Rows::Live, columns)),
        live_rows: Spec::live(),
    };
    let prepared = shared
        .column_slice(&slice, need)
        .map_err(|refused| format!("SM120 refuses the bundle's slice: {refused}"))?;
    let case = OpCase::new(prepared, CAPACITY, roles)?.oracle(move |images, live| {
        let mut output = vec![0; rows * columns * 4];
        kernel_core::row_copy::column_slice(&slice, images.bytes("source")?, &mut output, live)
            .map_err(refused)?;
        Ok(vec![("destination".into(), Expected::Bits(output))])
    });
    cases.push(("compact_scores", need, case));

    let case = gather_case(rng, &shared, CAPACITY, need, GATHER_SOURCE_ROWS)?;
    cases.push(("head_rows", need, case));
    // At the wide bucket, asked what the bundle asks of it: the kit checks
    // the rows, chunk and batch claims at 256 rows.
    let wide = served_ask(WIDE, false);
    let case = gather_case(rng, &shared, WIDE, wide, 2 * WIDE as usize + 13)?;
    cases.push(("head_rows wide", wide, case));
    Ok(cases)
}

/// The bundle embeds the shared copies in its `row_copy_sm120` and the
/// shared gather in its `row_gather_sm120`, the lab's cubins of them byte
/// for byte, and no other embedded cubin holds them; each copy case
/// declares on the CPU.
#[test]
fn the_bundle_embeds_the_shared_copies_and_their_cases_declare() {
    let cubin = engines::ROW_COPY_CUBIN;
    for entry in kernels_sm120::row::copy::entries() {
        assert_eq!(cubin_of(&entry.symbol), Some(cubin), "{}", entry.symbol);
    }
    assert_eq!(
        embedded_cubin(cubin),
        Some(kernel_lab::cubin(kernels_sm120::row::copy::LAB_CUBIN).unwrap())
    );
    let gather = kernels_sm120::row::gather::CUBIN;
    assert_eq!(cubin_of(kernels_sm120::row::gather::SYMBOL), Some(gather));
    assert_eq!(
        embedded_cubin(gather),
        Some(kernel_lab::cubin(gather).unwrap())
    );
    let cases = copy_cases(&mut Rng(0x0c09_1e55)).expect("copy cases");
    assert_eq!(cases.len(), 4);
    for (name, need, case) in &cases {
        for check in case.checks(*need, None, None, None) {
            match (check.name.as_str(), &check.outcome) {
                ("declaration" | "need", Outcome::Passed(_)) => {}
                (_, Outcome::NotExercised(_)) => {}
                (_, outcome) => panic!("{name} {}: {outcome:?}", check.name),
            }
        }
    }
}

/// The bundle's stream expansion and score compaction, the shared copies
/// from its embedded cubin, match the shared copy's oracle at every live
/// count of the bucket, write only their live rows and reproduce their
/// eager bits poisoned through relaunch, graph replay and a PDL producer.
/// Every check must pass, as for the own ops.
#[test]
fn bundle_copies_match_their_oracles_poisoned() {
    gpu_test(1 << 30, |session| {
        let functions = Embedded.load(session)?;
        let probes = Embedded.probes(session)?;
        let mut evidence = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, need, case) in copy_cases(&mut Rng(0x0c09_1e55))? {
            let checks = case.checks(need, Some(session), Some(&*functions), probes.as_ref());
            let mut receipt = BTreeMap::new();
            for check in checks {
                match &check.outcome {
                    Outcome::Passed(value) => {
                        receipt.insert(check.name.clone(), value.clone());
                    }
                    Outcome::NotExercised(reason)
                        if check.inapplicable() || check.name == "device live count" =>
                    {
                        receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                    }
                    outcome => failures.push(format!("{name} {}: {outcome:?}", check.name)),
                }
            }
            evidence.insert(name.to_owned(), json!(receipt));
        }
        if !failures.is_empty() {
            return Err(failures.join("\n"));
        }
        session.receipt(
            "bundle_copies_poisoned",
            BTreeMap::from([("copies".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle copies");
}

// ------------------------------------------------ shared draft argmax ----

/// The reference runner's F32 head logits (refcache entries in F32 mode,
/// named by key prefix and row): each row's 20 largest logits by global id
/// and F32 bits, the eight with the smallest margins between two leaders on
/// different ranks first. The drafter scores with the target's head, and no
/// drafter logits are captured, so these rows stand for them.
const HEAD_LOGITS: &str = include_str!("data/draft_head_logits.json");

/// The special rows after the captured ones, with the global id each
/// drafts (`draft_argmax_rows_win_where_they_say`).
const SPECIAL_ROWS: usize = 16;

/// The rows of the draft argmax cases over the whole vocabulary of
/// `shards` ranks of `shard` ids, [`CAPACITY`] in all: the captured rows,
/// each row's 20 logits at their ids over values below the 20th, then
/// [`SPECIAL_ROWS`] rows over finite values of both signs: the largest
/// value at two ranks and twice on one; adjacent F32 values on two ranks,
/// the larger at the higher id; -0 and +0 winners in both orders; +inf on
/// two ranks; only -inf; only NaN of both signs, quiet and signaling;
/// NaN with one -inf; NaN at id 0 and every third id beside a finite
/// winner; NaN with two -inf; only equal values; the winner at the last
/// id; the winner at rank 1's first id after NaN and -0; NaN at id 0 with
/// +inf at rank 3's first; the smallest subnormal over zeros; large
/// negative values.
fn draft_argmax_rows(shards: usize, shard: usize) -> LabResult<Vec<Vec<f32>>> {
    let vocab = shards * shard;
    let captured: serde_json::Value =
        serde_json::from_str(HEAD_LOGITS).map_err(|e| e.to_string())?;
    let captured = captured["rows"]
        .as_array()
        .ok_or("the captured logits hold no rows")?;
    if captured.len() + SPECIAL_ROWS != CAPACITY as usize {
        return Err(format!("{} captured rows", captured.len()));
    }
    let mut rng = Rng(0xd7a5_a7a6_0001);
    let mut rows = Vec::with_capacity(CAPACITY as usize);
    for row in captured {
        let numbers = |name: &str| -> LabResult<Vec<u64>> {
            row[name]
                .as_array()
                .ok_or(format!("a captured row lacks {name}"))?
                .iter()
                .map(|value| value.as_u64().ok_or(format!("a captured {name} entry")))
                .collect()
        };
        let (ids, logits) = (numbers("ids")?, numbers("bits")?);
        let logits = logits
            .into_iter()
            .map(|bits| f32::from_bits(bits as u32))
            .collect::<Vec<_>>();
        let floor = logits.iter().copied().fold(f32::INFINITY, f32::min) - 1.0;
        // Below the 20th logit, as the head's other logits are.
        let mut values = (0..vocab)
            .map(|_| floor - (rng.next() % 4096) as f32 / 256.0)
            .collect::<Vec<_>>();
        for (&id, &logit) in ids.iter().zip(&logits) {
            *values
                .get_mut(id as usize)
                .ok_or("a captured id past the vocabulary")? = logit;
        }
        rows.push(values);
    }
    let nan = |at: usize| f32::from_bits(F32_NANS[at % F32_NANS.len()]);
    for kind in 0..SPECIAL_ROWS {
        let mut values = rng
            .f32(vocab)
            .into_iter()
            .map(|value| value.clamp(-8.0, 8.0))
            .collect::<Vec<_>>();
        match kind {
            0 => [shard + 5, shard + 70, 3 * shard + 2]
                .into_iter()
                .for_each(|id| values[id] = 9.0),
            1 => {
                values[100] = 9.0;
                values[2 * shard + 7] = f32::from_bits(9.0_f32.to_bits() + 1);
            }
            2 | 3 => {
                values.iter_mut().for_each(|v| *v = -v.abs() - 1.0);
                let (low, high) = if kind == 2 { (-0.0, 0.0) } else { (0.0, -0.0) };
                values[shard + 9] = low;
                values[3 * shard + 1] = high;
            }
            4 => {
                values[2 * shard + 3] = f32::INFINITY;
                values[900] = f32::INFINITY;
            }
            5 => values.fill(f32::NEG_INFINITY),
            6 => values
                .iter_mut()
                .enumerate()
                .for_each(|(id, v)| *v = nan(id)),
            7 => {
                values
                    .iter_mut()
                    .enumerate()
                    .for_each(|(id, v)| *v = nan(id));
                values[3 * shard + 11] = f32::NEG_INFINITY;
            }
            8 => {
                (0..vocab).step_by(3).for_each(|id| values[id] = nan(id));
                values[2 * shard + 400] = 9.0;
            }
            9 => {
                values
                    .iter_mut()
                    .enumerate()
                    .for_each(|(id, v)| *v = nan(id));
                values[2 * shard] = f32::NEG_INFINITY;
                values[shard + 2] = f32::NEG_INFINITY;
            }
            10 => values.fill(1.0),
            11 => values[vocab - 1] = 9.0,
            12 => {
                values[0] = nan(3);
                values[1] = -0.0;
                values[shard] = 9.0;
            }
            13 => {
                values[0] = nan(1);
                values[3 * shard] = f32::INFINITY;
            }
            14 => {
                values.fill(0.0);
                values[2 * shard + 17] = f32::from_bits(1);
            }
            _ => {
                values.iter_mut().for_each(|v| *v = *v * 1e29 - 1e30);
                values[2 * shard + 31] = -1e29;
            }
        }
        rows.push(values);
    }
    Ok(rows)
}

/// Rank `rank`'s logits shard of `rows`, `[rows, shard]`.
fn shard_of(rows: &[Vec<f32>], rank: usize, shard: usize) -> Vec<f32> {
    rows.iter()
        .flat_map(|row| row[rank * shard..(rank + 1) * shard].iter().copied())
        .collect()
}

/// The draft rows table the drafter's argmax reads: `[CAPACITY]` run rows,
/// then each draft's drafting item (a permutation of the drafts) and its
/// verify row (distinct rows of the verify tokens).
fn draft_rows_table() -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let capacity = CAPACITY as i32;
    let run = (0..capacity).collect::<Vec<_>>();
    let items = (0..capacity).map(|d| (d * 7 + 3) % capacity).collect();
    let verify = (0..capacity)
        .map(|d| (d * 3 + 1) % VERIFY_TOKENS as i32)
        .collect();
    (run, items, verify)
}

/// Each row's token by the contract's oracle over `shards` ranks.
fn draft_tokens(rows: &[Vec<f32>], shards: usize, shard: usize) -> LabResult<Vec<u32>> {
    rows.iter()
        .map(|row| {
            let ranks = (0..shards)
                .map(|rank| &row[rank * shard..(rank + 1) * shard])
                .collect::<Vec<_>>();
            draft_argmax::token(&ranks).map_err(refused)
        })
        .collect()
}

/// The special rows draft the ids their kinds name.
#[test]
fn draft_argmax_rows_win_where_they_say() {
    let shape = Shape::TP4;
    let (shards, shard) = (shape.ranks() as usize, shape.profile.vocab_rows as usize);
    let rows = draft_argmax_rows(shards, shard).unwrap();
    let tokens = draft_tokens(&rows, shards, shard).unwrap();
    let vocab = (shards * shard) as u32;
    let s = shard as u32;
    assert_eq!(
        tokens[CAPACITY as usize - SPECIAL_ROWS..],
        [
            s + 5,
            2 * s + 7,
            s + 9,
            s + 9,
            900,
            0,
            0,
            3 * s + 11,
            2 * s + 400,
            s + 2,
            0,
            vocab - 1,
            s,
            3 * s,
            2 * s + 17,
            2 * s + 31,
        ]
    );
}

/// The bundle's engines of `shape`.
fn bundle_engines(shape: Shape) -> BundleEngines {
    BundleEngines::new(shape).expect("the bundle's engines")
}

/// The gathered pairs a rank's gather output holds: [`CAPACITY`] per rank,
/// of which each rank's live rows are written (the rest copy the partial's
/// unwritten fill).
fn gathered_pairs<'a>(ranks: u32, dtype: DType) -> Spec<'a> {
    let written = move |_: &Images, live: u32| -> LabResult<WriteMap> {
        WriteMap::of(dtype).ranges((0..ranks as usize).map(|rank| {
            let start = rank * CAPACITY as usize;
            start..start + live as usize
        }))
    };
    Spec::output_with(
        dtype,
        (Rows::Fixed(ranks as usize * CAPACITY as usize), 1),
        Write::Map(Box::new(written)),
    )
}

/// Rank `me`'s record of the gather of the pairs' `dtype` word in a group
/// of `ranks`, its state regions `state` bytes.
fn pair_gather<'a>(
    ranks: u32,
    me: u32,
    gather: &str,
    dtype: DType,
    state: &[u64],
) -> GatherBuffers<Slot<'a>> {
    GatherBuffers {
        input: Slot::Own(Spec::output(dtype, (Rows::Live, 1))),
        ranks: (0..ranks)
            .map(|rank| RankViews {
                output: if rank == me {
                    Slot::Own(gathered_pairs(ranks, dtype))
                } else {
                    Slot::peer(rank, format!("{gather}.ranks[{rank}].output"))
                },
                state: (0..state.len())
                    .map(|at| match rank == me {
                        true => Slot::State(state[at]),
                        false => Slot::peer(rank, format!("{gather}.ranks[{rank}].state[{at}]")),
                    })
                    .collect(),
            })
            .collect(),
    }
}

/// A scattered output of the draft argmax: `elements` I32, the live
/// drafts' entries of `path`'s indexes written and every other element
/// left as filled. An output, not in-place state: a collective case reads
/// back only its outputs and scratch after a step.
fn scattered_tokens<'a>(path: &'static str, elements: usize) -> Spec<'a> {
    let written = move |images: &Images, live: u32| -> LabResult<WriteMap> {
        let rows = images.i32s(path)?;
        WriteMap::of(DType::I32).ranges(
            rows[..live as usize]
                .iter()
                .filter_map(|&at| usize::try_from(at).ok().filter(|&at| at < elements))
                .map(|at| at..at + 1),
        )
    };
    Spec::output_with(
        DType::I32,
        (Rows::Fixed(elements), 1),
        Write::Map(Box::new(written)),
    )
}

/// The draft argmax of `shape` on each rank (`ops`) over the draft argmax
/// rows: every rank's pairs, gathered pairs, drafts and verify tokens as
/// the contract oracle's bits, the drafts and verify tokens alike on every
/// rank.
fn draft_argmax_case<'a>(
    shape: Shape,
    ops: Vec<Prepared<DraftArgmaxBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    let (ranks, shard) = (shape.ranks(), shape.profile.vocab_rows as usize);
    let rows = draft_argmax_rows(ranks as usize, shard)?;
    let state = ops
        .first()
        .map(|op| op.info().state.clone())
        .unwrap_or_default();
    let half = state.len() / 2;
    let (_, items, verify_rows) = draft_rows_table();
    let i32_image = |values: &[i32]| values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let records = (0..ranks)
        .map(|me| DraftArgmaxBuffers {
            logits: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Live, shard),
                bits(&shard_of(&rows, me as usize, shard)),
            )),
            live_width: Slot::Own(Spec::input(
                DType::I32,
                (Rows::Fixed(1), 1),
                (shard as u32).to_le_bytes().to_vec(),
            )),
            scores: pair_gather(ranks, me, "scores", DType::F32, &state[..half]),
            ids: pair_gather(ranks, me, "ids", DType::I32, &state[half..]),
            items: Slot::Own(Spec::input(DType::I32, (Rows::Live, 1), i32_image(&items))),
            drafts: Slot::Own(scattered_tokens("items", CAPACITY as usize)),
            verify_rows: Slot::Own(Spec::input(
                DType::I32,
                (Rows::Live, 1),
                i32_image(&verify_rows),
            )),
            verify_tokens: Some(Slot::Own(scattered_tokens("verify_rows", VERIFY_TOKENS))),
            live_drafts: Slot::Own(Spec::live()),
        })
        .collect();
    let oracle = move |images: &[Images], live: u32| -> LabResult<Vec<Vec<(String, Expected)>>> {
        let shards = images
            .iter()
            .map(|images| images.f32s("logits"))
            .collect::<LabResult<Vec<_>>>()?;
        let pairs = shards
            .iter()
            .zip(0..)
            .map(|(logits, rank)| {
                logits
                    .chunks_exact(shard)
                    .map(|row| {
                        let [(score, id)] = kernel_core::oracle::sharded_argmax_partials(&[row])
                            .map_err(refused)?[..]
                        else {
                            unreachable!("one shard, one pair");
                        };
                        Ok((score, (rank * shard + id) as i32))
                    })
                    .collect::<LabResult<Vec<_>>>()
            })
            .collect::<LabResult<Vec<_>>>()?;
        let by_rank = shards.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let first = &images[0];
        // Only the scattered entries are compared (the outputs' write maps).
        let mut drafts = vec![0; CAPACITY as usize];
        let mut verify = vec![0; VERIFY_TOKENS];
        draft_argmax::draft_argmax(
            &by_rank,
            live,
            CAPACITY,
            &first.i32s("items")?,
            &mut drafts,
            &first.i32s("verify_rows")?,
            Some(&mut verify),
        )
        .map_err(refused)?;
        let scores = |pairs: &[(f32, i32)]| bits(&pairs.iter().map(|p| p.0).collect::<Vec<_>>());
        let ids =
            |pairs: &[(f32, i32)]| words(&pairs.iter().map(|p| p.1 as u32).collect::<Vec<_>>());
        let gathered_scores = pairs
            .iter()
            .flat_map(|rank| scores(rank))
            .collect::<Vec<_>>();
        let gathered_ids = pairs.iter().flat_map(|rank| ids(rank)).collect::<Vec<_>>();
        Ok((0..images.len())
            .map(|rank| {
                vec![
                    ("scores.input".into(), Expected::Bits(scores(&pairs[rank]))),
                    ("ids.input".into(), Expected::Bits(ids(&pairs[rank]))),
                    (
                        format!("scores.ranks[{rank}].output"),
                        Expected::Bits(gathered_scores.clone()),
                    ),
                    (
                        format!("ids.ranks[{rank}].output"),
                        Expected::Bits(gathered_ids.clone()),
                    ),
                    ("drafts".into(), Expected::Bits(words(&drafts))),
                    ("verify_tokens".into(), Expected::Bits(words(&verify))),
                ]
            })
            .collect())
    };
    Ok(CollectiveCase::new(ops, CAPACITY, records)?
        .oracle(oracle)
        .same_on_every_rank(&["drafts", "verify_tokens"])
        .without_properties(
            &[Property::Chunk, Property::Rows, Property::Batch],
            "the gathered pairs are blocks of every row, not rows of the case",
        ))
}

/// The draft argmax of `shape` as a required collective at its ranks.
fn draft_argmax_required<'a>(shape: Shape) -> RequiredCollective<'a> {
    RequiredCollective::new(
        "draft argmax",
        Need::rows(CAPACITY),
        move |backend, _, rank, _| {
            DraftArgmax::prepare(backend, shape, rank)
                .map(|op| Box::new(op) as Prepared<DraftArgmaxBuffers<View>>)
        },
        move |_, ops| draft_argmax_case(shape, ops),
    )
    .required()
}

/// The backend the draft argmax runs on: the one an instance of `shape`
/// prepares its ops on, its entries from the bundle's embedded cubins
/// ([`Embedded`]).
fn draft_argmax_backend(shape: Shape) -> Box<dyn kernel_core::backend::Backend> {
    let engines = bundle_engines(shape);
    sm120_backend(&engines, shape).0
}

/// Whether `report` prepared and declared the draft argmax, every launched
/// symbol in an embedded cubin, and with a GPU passed conformance.
fn check_draft_argmax(report: &kernel_lab::kit::Report, gpu: bool) -> LabResult<()> {
    let failures = report.failures();
    if !failures.is_empty() {
        return Err(format!("{failures:#?}"));
    }
    let [op] = &report.ops[..] else {
        return Err("one op".into());
    };
    let OpResult::Prepared { checks, .. } = &op.result else {
        return Err(format!("{:?}", op.result));
    };
    let outcome = |name: &str| {
        checks
            .iter()
            .find(|check| check.name == name)
            .map(|check| &check.outcome)
            .ok_or(format!("no {name} check"))
    };
    let Outcome::Passed(declared) = outcome("declaration")? else {
        return Err(format!("{:?}", outcome("declaration")?));
    };
    for symbol in declared["launches"].as_array().ok_or("no launches")? {
        let symbol = symbol.as_str().ok_or("a symbol")?;
        if cubin_of(symbol).is_none() {
            return Err(format!("no embedded cubin holds {symbol}"));
        }
    }
    if gpu && !matches!(outcome("conformance")?, Outcome::Passed(_)) {
        return Err(format!("{:?}", outcome("conformance")?));
    }
    Ok(())
}

/// The draft argmax prepares on every rank on the bundle's backend and its
/// case declares: its launches in embedded cubins, its oracle covering
/// every output.
#[test]
fn the_shared_draft_argmax_declares_on_every_rank() {
    let shape = Shape::TP4;
    let backend = draft_argmax_backend(shape);
    let report = kit::collective::conform(
        backend.as_ref(),
        &Embedded,
        shape.ranks(),
        None,
        &[draft_argmax_required(shape)],
    )
    .expect("report");
    check_draft_argmax(&report, false).unwrap();
}

/// On four GPUs the draft argmax as the bundle prepares and embeds it
/// (partial, the gathers of the scores and ids, the final scattered into
/// the drafts and verify tokens) writes the
/// contract oracle's bits on every rank over the draft argmax rows at
/// every live count the kit steps through, poisoned through eager,
/// relaunch and graph replay runs.
#[test]
fn the_shared_draft_argmax_agrees_on_four_ranks() {
    gpu_test_multi(1 << 30, |sessions| {
        let shape = Shape::TP4;
        if sessions.len() != shape.ranks() as usize {
            return kernel_lab::skip("the draft argmax runs on four GPUs");
        }
        let backend = draft_argmax_backend(shape);
        let report = kit::collective::conform(
            backend.as_ref(),
            &Embedded,
            shape.ranks(),
            Some(sessions),
            &[draft_argmax_required(shape)],
        )?;
        sessions[0].receipt(
            "conformance-kit-draft-argmax",
            BTreeMap::from([
                ("report".into(), report.json()),
                ("not_exercised".into(), json!(report.not_exercised())),
            ]),
        )?;
        check_draft_argmax(&report, true)
    })
    .expect("the shared draft argmax on four ranks");
}

// ------------------------------------------------- bucket-sized rows ----

/// The decode bucket the dense GEMMs plan for, and the live counts its
/// steps take in turn.
const DENSE_BUCKET: u32 = 8;
const DENSE_LIVE: [u32; 3] = [5, 7, 5];
/// A byte whose BF16 and F32 repetitions are finite.
const FINITE: u8 = 0x11;

/// The dense GEMMs and the F32 all-reduce work on every row of their
/// bucket, the live count being a scalar fixed per bucket under the shared
/// GEMM ABI; inactive rows never reach live ones. On four ranks, at a
/// decode bucket of 8 rows and live counts 5, 7 and 5 in turn (eagerly and
/// by replaying one graph captured at the first), with every inactive input
/// row NaN:
///
/// - the KDA query projection's tiled GEMM writes all 8 rows of its
///   temporary, NaN past the live ones, and its layout copy, by the device
///   count, writes only the live rows of its third of the convolution's
///   input, bitwise the clean run's, leaving the rest and the other thirds
///   untouched;
/// - the KDA output projection's F32 partials are all-reduced over all 8
///   rows (element by element, so the sum's inactive rows are NaN and its
///   live rows bitwise the clean run's), and the rounding, by the device
///   count, writes only the live rows of the sublayer's output.
#[test]
fn bucket_rows_stay_out_of_live_rows() {
    use kernel_core::contract::WeightFormat;
    use kernel_core::precision::{ExecutionPhase, OperationUse};
    use kernels_sm120::precision::Compute;
    gpu_test_multi(1 << 30, |sessions| {
        let shape = Shape::TP4;
        let ranks = shape.ranks() as usize;
        if sessions.len() != ranks {
            return kernel_lab::skip("the bucket-sized rows run on four GPUs");
        }
        let bucket = DENSE_BUCKET as usize;
        let hidden = shape.profile.hidden as usize;
        let usage = OperationUse {
            phase: ExecutionPhase::Decode,
            capacity: DENSE_BUCKET,
            graph: true,
        };
        // Each projection's tiled GEMM entry at the bucket, its geometry
        // (n, k) and its plan over the bucket's rows.
        let dense = |id: &str| -> LabResult<_> {
            let generated = sm120_precision::offer(id, WeightFormat::Bf16, usage)?;
            let Compute::Dense(spec) = &generated.offer.handle().compute else {
                return Err(format!("{id} is not a dense GEMM"));
            };
            let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                return Err(format!("{id} has no GEMM query"));
            };
            if query.m_capacity != DENSE_BUCKET {
                return Err(format!("{id} compiles {} rows", query.m_capacity));
            }
            let counts = kernels_sm120::gemm::row_counts(&spec.key, DENSE_BUCKET);
            let plan = kernels_sm120::gemm::plan(&spec.key, counts).map_err(|r| r.to_string())?;
            Ok((
                spec.clone(),
                counts,
                plan,
                query.n as usize,
                query.k as usize,
            ))
        };
        let (q_spec, q_counts, q_plan, width, q_k) = dense("linear.q")?;
        let (o_spec, o_counts, o_plan, o_n, o_k) = dense("linear.output")?;
        assert_eq!((q_k, o_n, o_k), (hidden, hidden, width));
        let copy = sm120_precision::copy("linear.q", WeightFormat::Bf16, usage)?
            .ok_or("linear.q copies its rows")?;
        let copy_plan = copy.stage().plan(DENSE_BUCKET).map_err(|r| r.to_string())?;
        assert!(engines::one_shot(shape, DENSE_BUCKET)?);
        let comm = engines::one_shot_bytes(shape, DENSE_BUCKET);
        let elements = (bucket * hidden) as u64;
        let workspace = (q_plan.workspace_bytes.max(o_plan.workspace_bytes) as usize).max(256);
        struct Rank {
            gemm: runtime_cuda::Module,
            copies: runtime_cuda::Module,
            rows: runtime_cuda::Module,
            q_weight: DeviceBuffer,
            o_weight: DeviceBuffer,
            q_input: DeviceBuffer,
            o_input: DeviceBuffer,
            temporary: DeviceBuffer,
            qkv: DeviceBuffer,
            partial: DeviceBuffer,
            summed: DeviceBuffer,
            reduced: DeviceBuffer,
            live: DeviceBuffer,
            workspace: DeviceBuffer,
            semaphores: DeviceBuffer,
            inbox: DeviceBuffer,
            flags: DeviceBuffer,
            epoch: DeviceBuffer,
        }
        let mut rng = Rng(0xb0c4_e7ed);
        let mut state_rank = |session: &Session| -> LabResult<Rank> {
            Ok(Rank {
                gemm: embedded(session, sm120_precision::CUBIN)?,
                copies: embedded(session, sm120_precision::COPIES_CUBIN)?,
                rows: own_row_module(session)?,
                q_weight: upload(session, &halves(&rng.bf16(width * hidden)))?,
                o_weight: upload(session, &halves(&rng.bf16(hidden * width)))?,
                q_input: session.allocate(bucket * hidden * 2)?,
                o_input: session.allocate(bucket * width * 2)?,
                temporary: session.allocate(bucket * width * 2)?,
                qkv: session.allocate(bucket * 3 * width * 2)?,
                partial: session.allocate(bucket * hidden * 4)?,
                summed: session.allocate(bucket * hidden * 4)?,
                reduced: session.allocate(bucket * hidden * 2)?,
                live: upload(session, &[0; 4])?,
                workspace: session.allocate(workspace)?,
                semaphores: upload(session, &[0; 256])?,
                inbox: session.allocate(comm.inbox as usize)?,
                flags: upload(session, &vec![0; comm.flags as usize])?,
                epoch: upload(session, &vec![0; comm.epoch as usize])?,
            })
        };
        let states = sessions
            .iter()
            .map(&mut state_rank)
            .collect::<LabResult<Vec<_>>>()?;
        let function = |module: &runtime_cuda::Module, symbol: &str, shared: u32| {
            let function = module.function(symbol).map_err(|e| e.to_string())?;
            if shared >= 48 * 1024 {
                function
                    .set_max_dynamic_shared_memory(shared)
                    .map_err(|e| e.to_string())?;
            }
            Ok::<_, String>(function)
        };
        let functions = states
            .iter()
            .map(|rank| {
                Ok((
                    function(&rank.gemm, &q_spec.symbol, q_plan.dynamic_shared_bytes)?,
                    function(&rank.gemm, &o_spec.symbol, o_plan.dynamic_shared_bytes)?,
                    function(&rank.copies, &copy.stage().entry().symbol, 0)?,
                    function(&rank.rows, &ModelRow::Round.symbol(), 0)?,
                ))
            })
            .collect::<LabResult<Vec<_>>>()?;
        // Each rank's GEMMs, copy and rounding (the all-reduce between).
        let gemms = |rank: usize, session: &Session| -> LabResult<()> {
            let s = &states[rank];
            for (function, spec, counts, plan, weight, input, output) in [
                (
                    &functions[rank].0,
                    &q_spec,
                    q_counts,
                    q_plan,
                    &s.q_weight,
                    &s.q_input,
                    &s.temporary,
                ),
                (
                    &functions[rank].1,
                    &o_spec,
                    o_counts,
                    o_plan,
                    &s.o_weight,
                    &s.o_input,
                    &s.partial,
                ),
            ] {
                let maps =
                    kernels_sm120::gemm::tensor_maps(&spec.key, 1, 2).map_err(|r| r.to_string())?;
                let args = kernels_sm120::gemm::arguments(
                    &spec.key,
                    counts,
                    maps,
                    kernels_sm120::gemm::GemmBuffers {
                        output: 3,
                        partial: 4,
                        semaphores: 5,
                        weight_scales: None,
                        activation_scales: None,
                        residual: None,
                        weight_global_scale: None,
                        activation_global_scale: None,
                    },
                )
                .map_err(|r| r.to_string())?;
                let buffers = BTreeMap::from([
                    (1, weight),
                    (2, input),
                    (3, output),
                    (4, &s.workspace),
                    (5, &s.semaphores),
                ]);
                launch::launch(
                    &session.context,
                    function,
                    &session.stream,
                    plan,
                    &args,
                    &launch::Bindings { buffers },
                )?;
            }
            let args = copy.arguments(1, 2, 3).map_err(|r| r.to_string())?;
            let buffers = BTreeMap::from([(1, &s.temporary), (2, &s.qkv), (3, &s.live)]);
            launch::launch(
                &session.context,
                &functions[rank].2,
                &session.stream,
                copy_plan,
                &args,
                &launch::Bindings { buffers },
            )
        };
        let round = |rank: usize, session: &Session| -> LabResult<()> {
            let s = &states[rank];
            functions[rank]
                .3
                .launch(
                    &session.stream,
                    launch::config(ModelRow::Round.plan(DENSE_BUCKET)?)?,
                    &[
                        KernelArgument::Buffer(&s.summed),
                        KernelArgument::Buffer(&s.reduced),
                        KernelArgument::Buffer(&s.live),
                    ],
                )
                .map_err(|e| e.to_string())
        };
        let reduce_plan = kernels_sm120::comm::plan(&engines::reduce_query(shape, DENSE_BUCKET), 1)
            .map_err(|r| r.to_string())?;
        let modules = launch::CollectiveModules::load(sessions, "comm_sm120")?;
        let handles: Vec<u64> = (0..ranks as u64).map(|rank| rank + 1).collect();
        let reduce = |only: Option<usize>| {
            launch::launch_collective(
                sessions,
                &modules,
                "chop_comm_allreduce_oneshot_f32_v1",
                reduce_plan,
                only,
                |rank| {
                    let mut buffers: BTreeMap<u64, &DeviceBuffer> = BTreeMap::new();
                    buffers.insert(100, &states[rank].partial);
                    buffers.insert(101, &states[rank].summed);
                    buffers.insert(102, &states[rank].epoch);
                    for (peer, other) in states.iter().enumerate() {
                        buffers.insert(peer as u64 + 1, &other.inbox);
                        buffers.insert(peer as u64 + 11, &other.flags);
                    }
                    let args = kernels_sm120::comm::arguments(kernels_sm120::comm::OneShotCall {
                        input: 100,
                        output: 101,
                        peer_inboxes: handles.clone(),
                        peer_flags: handles.iter().map(|handle| handle + 10).collect(),
                        epoch: 102,
                        elements,
                        rank: rank as u32,
                        ranks: ranks as u32,
                    })
                    .map_err(|e| e.to_string())?;
                    Ok(launch::RankCall {
                        args,
                        bindings: launch::Bindings { buffers },
                    })
                },
            )
        };
        let eager = || -> LabResult<()> {
            for (rank, session) in sessions.iter().enumerate() {
                gemms(rank, session)?;
            }
            reduce(None)?;
            for (rank, session) in sessions.iter().enumerate() {
                round(rank, session)?;
            }
            Ok(())
        };
        // The inputs: every row clean, then the live rows clean and the rest
        // NaN; the outputs poisoned before each run.
        let q_rows = (0..ranks)
            .map(|_| rng.bf16(bucket * hidden))
            .collect::<Vec<_>>();
        let o_rows = (0..ranks)
            .map(|_| rng.bf16(bucket * width))
            .collect::<Vec<_>>();
        let prepare = |live: u32, poisoned: bool| -> LabResult<()> {
            poison::synchronize(sessions)?;
            for (rank, session) in sessions.iter().enumerate() {
                let s = &states[rank];
                let mut q = halves(&q_rows[rank]);
                let mut o = halves(&o_rows[rank]);
                if poisoned {
                    q[live as usize * hidden * 2..].fill(poison::POISON);
                    o[live as usize * width * 2..].fill(poison::POISON);
                }
                poison::write(session, &s.q_input, 0, &q)?;
                poison::write(session, &s.o_input, 0, &o)?;
                // The consumers' outputs poisoned; the bucket-sized outputs
                // filled with finite values, so NaN there shows a write.
                for (buffer, fill) in [
                    (&s.temporary, FINITE),
                    (&s.partial, FINITE),
                    (&s.summed, FINITE),
                    (&s.qkv, poison::POISON),
                    (&s.reduced, poison::POISON),
                ] {
                    poison::write(session, buffer, 0, &vec![fill; buffer.byte_len()])?;
                }
                poison::write(session, &s.live, 0, &live.to_le_bytes())?;
            }
            Ok(())
        };
        let read = |rank: usize, buffer: fn(&Rank) -> &DeviceBuffer| {
            poison::read(&sessions[rank], buffer(&states[rank]))
        };
        // The clean run over every row of the bucket.
        prepare(DENSE_BUCKET, false)?;
        eager()?;
        poison::synchronize(sessions)?;
        let clean = (0..ranks)
            .map(|rank| {
                Ok((
                    read(rank, |s| &s.qkv)?,
                    read(rank, |s| &s.summed)?,
                    read(rank, |s| &s.reduced)?,
                ))
            })
            .collect::<LabResult<Vec<_>>>()?;
        let graphs = kernel_lab::topology::RankReplay::new(sessions, |rank, session| {
            gemms(rank, session)?;
            reduce(Some(rank))?;
            round(rank, session)
        })?;
        let mut evidence = Vec::new();
        for (call, live) in DENSE_LIVE.into_iter().enumerate() {
            for graph in [false, true] {
                prepare(live, true)?;
                if graph {
                    graphs.launch(sessions)?;
                } else {
                    eager()?;
                }
                poison::synchronize(sessions)?;
                let live = live as usize;
                for (rank, clean) in clean.iter().enumerate() {
                    let at = |what: &str| {
                        format!("call {call} (graph {graph}, live {live}): rank {rank}'s {what}")
                    };
                    // The GEMM wrote every bucket row: NaN past the live ones.
                    let temporary = read(rank, |s| &s.temporary)?;
                    let nan = |bits: u16| bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0;
                    let inactive = temporary[live * width * 2..]
                        .chunks(2)
                        .all(|pair| nan(u16::from_le_bytes([pair[0], pair[1]])));
                    if !inactive {
                        return Err(at(
                            "temporary's inactive rows are not the NaN rows the GEMM computed",
                        ));
                    }
                    // The copy wrote the live rows of the first third alone.
                    let qkv = read(rank, |s| &s.qkv)?;
                    for row in 0..bucket {
                        let line = row * 3 * width * 2;
                        let (third, rest) = qkv[line..line + 3 * width * 2].split_at(width * 2);
                        let expected = if row < live {
                            &clean.0[line..line + width * 2]
                        } else {
                            &vec![poison::POISON; width * 2][..]
                        };
                        if third != expected || rest.iter().any(|&b| b != poison::POISON) {
                            return Err(at(&format!("convolution input row {row}")));
                        }
                    }
                    // The all-reduce summed every bucket row element by
                    // element; the rounding read the live rows alone.
                    let summed = read(rank, |s| &s.summed)?;
                    if summed[..live * hidden * 4] != clean.1[..live * hidden * 4] {
                        return Err(at("sum's live rows"));
                    }
                    let inactive = summed[live * hidden * 4..].chunks(4).all(|word| {
                        f32::from_le_bytes([word[0], word[1], word[2], word[3]]).is_nan()
                    });
                    if !inactive {
                        return Err(at("sum's inactive rows are not NaN"));
                    }
                    let reduced = read(rank, |s| &s.reduced)?;
                    if reduced[..live * hidden * 2] != clean.2[..live * hidden * 2]
                        || reduced[live * hidden * 2..]
                            .iter()
                            .any(|&b| b != poison::POISON)
                    {
                        return Err(at("rounded rows"));
                    }
                }
                evidence.push(json!({ "call": call, "graph": graph, "live": live }));
            }
        }
        sessions[0].receipt(
            "bucket_rows",
            BTreeMap::from([("calls".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bucket-sized rows on four ranks");
}

// ---------------------------------------------------- payload admission ----

/// Elements of every admission case: more than one grid of
/// `sm120_payload::MAX_BLOCKS` CTAs covers, so each thread strides, and whole
/// MXFP8 blocks.
const ADMITTED: usize = (sm120_payload::MAX_BLOCKS * sm120_payload::THREADS) as usize + 129 * 32;
/// The lower planted element, past the first grid, and the higher one;
/// both start an MXFP8 block.
const PLANTED: [usize; 2] = [
    (sm120_payload::MAX_BLOCKS * sm120_payload::THREADS) as usize + 32,
    ADMITTED - 32,
];

/// One admission case: the checked view's bytes (which the producer
/// writes), MXFP8's scales, and the report the check must give.
struct Admitted {
    name: String,
    codes: Vec<u8>,
    scales: Vec<u8>,
    report: u32,
}

/// A valid payload for `check` of [`ADMITTED`] elements, with every kind
/// of admitted edge: zeros of both signs, F32 subnormals and the largest
/// finite scale, every UE4M3 code below 0x7f, and MXFP8 codes at every
/// scale, those a scale under 3 or over 246 cannot hold exactly replaced
/// by a signed zero.
fn admitted_payload(check: Check, rng: &mut Rng) -> (Vec<u8>, Vec<u8>) {
    match check {
        Check::F32Scales => {
            let values = (0..ADMITTED)
                .map(|index| match index % 7 {
                    0 => 0,
                    1 => 0x8000_0000,
                    2 => rng.next() & 0x007f_ffff,
                    3 => 0x7f7f_ffff,
                    _ => rng.next() % 0x7f80_0000,
                })
                .collect::<Vec<_>>();
            (words(&values), Vec::new())
        }
        Check::Ue4m3Scales => (
            (0..ADMITTED).map(|_| (rng.next() % 0x7f) as u8).collect(),
            Vec::new(),
        ),
        Check::Mxfp8 => {
            let scales = (0..ADMITTED / 32)
                .map(|_| match rng.next() as u8 {
                    0xff => 127,
                    scale => scale,
                })
                .collect::<Vec<_>>();
            let codes = (0..ADMITTED)
                .map(|index| {
                    let code = rng.next() as u8;
                    match check.first_invalid(&[code], &[scales[index / 32]]) {
                        Some(_) => code & 0x80,
                        None => code,
                    }
                })
                .collect();
            (codes, scales)
        }
    }
}

/// Writes an invalid code (and its scale) at an element of a payload.
type Plant<'a> = dyn Fn(&mut [u8], &mut [u8], usize) + 'a;

/// Plants each invalid kind `check` refuses at both [`PLANTED`] elements of
/// a valid payload: non-finite and negative F32 scales; UE4M3 NaN and
/// signed codes; MXFP8 NaN codes, scale 255, and a code its scale makes
/// too small or too large for a BF16 (its block's other codes zeroed, so
/// the planted one is the block's first invalid element).
fn admission_cases(check: Check, rng: &mut Rng) -> Vec<Admitted> {
    let (codes, scales) = admitted_payload(check, rng);
    let valid = Admitted {
        name: "valid".into(),
        codes: codes.clone(),
        scales: scales.clone(),
        report: payload::NONE,
    };
    let planted = |name: String, plant: &Plant<'_>| {
        let (mut codes, mut scales) = (codes.clone(), scales.clone());
        for at in PLANTED {
            plant(&mut codes, &mut scales, at);
        }
        Admitted {
            name,
            codes,
            scales,
            report: PLANTED[0] as u32,
        }
    };
    let mut cases = vec![valid];
    match check {
        Check::F32Scales => {
            for bits in [
                0x7f80_0000_u32,
                0xff80_0000,
                0x7fc0_0000,
                0xffc0_0001,
                0xbf80_0000,
                0x8000_0001,
            ] {
                cases.push(planted(format!("{bits:#010x}"), &|codes, _, at| {
                    codes[4 * at..4 * at + 4].copy_from_slice(&bits.to_le_bytes())
                }));
            }
        }
        Check::Ue4m3Scales => {
            for code in [0x7f_u8, 0x80, 0xc3, 0xff] {
                cases.push(planted(format!("{code:#04x}"), &|codes, _, at| {
                    codes[at] = code
                }));
            }
        }
        Check::Mxfp8 => {
            for code in [0x7f_u8, 0xff] {
                cases.push(planted(format!("code {code:#04x}"), &|codes, _, at| {
                    codes[at] = code
                }));
            }
            // A zero code at scale 255 too, which no product admits.
            cases.push(planted("scale 0xff".into(), &|codes, scales, at| {
                codes[at] = 0;
                scales[at / 32] = 0xff
            }));
            // 2^-9 * 2^-127 is below BF16's smallest subnormal; 448 * 2^127
            // is past its largest finite value.
            for (code, scale) in [(0x01_u8, 0_u8), (0x7e, 254)] {
                cases.push(planted(
                    format!("code {code:#04x} at scale {scale}"),
                    &|codes, scales, at| {
                        codes[at..at + 32].fill(0);
                        codes[at] = code;
                        scales[at / 32] = scale
                    },
                ));
            }
        }
    }
    cases
}

/// Each admission check runs from the bundle's embedded cubin over a valid
/// payload and payloads with each planted invalid kind, and reports what
/// the contract's oracle reports: no element, or the lower planted one. Each
/// case is poisoned through eager, relaunch, graph replay and a PDL
/// producer that writes the checked view, so a check that read it before
/// its PDL wait would see poison, which every rule refuses.
#[test]
fn bundle_payload_admission_reports_planted_codes_poisoned() {
    gpu_test(1 << 30, |session| {
        let module = embedded(session, sm120_payload::CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let producer = CopyProducer::new(&lab)?;
        let pdl = std::env::var("CHOPINFER_LATENT_NO_PDL").as_deref() != Ok("1");
        let stream = &session.stream;
        let mut rng = Rng(0xad31_5510);
        let mut evidence = BTreeMap::new();
        for check in Check::ALL {
            let symbol = sm120_payload::symbol(check);
            let function = module.function(symbol).map_err(|e| e.to_string())?;
            let cases = admission_cases(check, &mut rng);
            let bytes = cases[0].codes.len();
            let (staged, codes) = (session.allocate(bytes)?, session.allocate(bytes)?);
            let scales = session.allocate(cases[0].scales.len().max(4))?;
            let first = session.allocate(4)?;
            let admission = payload::Admission {
                check,
                view: symbol.into(),
                scales: (check == Check::Mxfp8).then(|| "scales".into()),
                count: ADMITTED as u64,
            };
            let plan = sm120_payload::plan(admission.count).map_err(|r| r.0.to_string())?;
            let mut arguments = vec![KernelArgument::Buffer(&codes)];
            if admission.scales.is_some() {
                arguments.push(KernelArgument::Buffer(&scales));
            }
            arguments.push(KernelArgument::U64(admission.count));
            arguments.push(KernelArgument::Buffer(&first));
            let run = |stream: &Stream, pdl: bool| {
                let config = launch::config(kernel_core::contract::LaunchPlan { pdl, ..plan })?;
                function
                    .launch(stream, config, &arguments)
                    .map_err(|e| format!("{symbol}: {e}"))
            };
            let chain = |stream: &Stream, pdl: bool| {
                producer.copy(stream, &staged, &codes)?;
                run(stream, pdl)
            };
            let graph = session.capture(|stream| chain(stream, pdl))?;
            let mut receipts = BTreeMap::new();
            for case in cases {
                let label = format!("{symbol} {}", case.name);
                let mut mirror = payload::NONE;
                let op = Admission {
                    rule: check,
                    count: ADMITTED as u64,
                };
                state_ops::admission(&op, &case.codes, &case.scales, &mut mirror)
                    .map_err(refused)?;
                if mirror != case.report {
                    return Err(format!(
                        "{label}: the oracle reports {mirror:#x}, not {:#x}",
                        case.report
                    ));
                }
                // The producer's writes are checked per word, so no staged
                // word may be all poison.
                if case.codes.chunks(4).any(|word| word == [poison::POISON; 4]) {
                    return Err(format!("{label}: a staged word is the poison pattern"));
                }
                poison::write(session, &staged, 0, &case.codes)?;
                if !case.scales.is_empty() {
                    poison::write(session, &scales, 0, &case.scales)?;
                }
                let mut runs = Replay::new(session)
                    .state(
                        "codes",
                        &codes,
                        case.codes.clone(),
                        WriteMap::of(DType::I32).byte_prefix(bytes)?,
                        vec![],
                    )
                    .state(
                        "first",
                        &first,
                        payload::NONE.to_le_bytes().to_vec(),
                        WriteMap::of(DType::I32).none()?,
                        std::iter::once(0..4).collect(),
                    );
                runs.eager(|| chain(stream, false))?;
                let report = runs
                    .expected("first")
                    .and_then(|bytes| bytes.try_into().ok())
                    .map(u32::from_le_bytes)
                    .ok_or(format!("{label}: no eager report"))?;
                if report != case.report {
                    return Err(format!(
                        "{label}: reported {report:#x}, the oracle {:#x}",
                        case.report
                    ));
                }
                runs.relaunch(|| chain(stream, false))
                    .map_err(|e| format!("{label}: {e}"))?;
                runs.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
                runs.pdl(
                    || producer.copy(stream, &staged, &codes),
                    || run(stream, pdl),
                    pdl,
                )
                .map_err(|e| format!("{label}: {e}"))?;
                let mut receipt = BTreeMap::new();
                runs.record(&mut receipt);
                receipt.insert("report".into(), json!(report));
                receipts.insert(case.name, json!(receipt));
            }
            evidence.insert(symbol.to_owned(), json!(receipts));
        }
        session.receipt(
            "bundle_payload_admission_poisoned",
            BTreeMap::from([("checks".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle payload admission");
}

/// The planted cases are what the GPU test expects of them on the host:
/// the valid payload admits, and each planted kind's lowest invalid
/// element is the lower planted one.
#[test]
fn admission_cases_plant_their_codes_at_the_lower_element() {
    let mut rng = Rng(0xad31_5510);
    for check in Check::ALL {
        let cases = admission_cases(check, &mut rng);
        assert!(cases.len() > 2, "{check:?}");
        for case in cases {
            let mut report = payload::NONE;
            let op = Admission {
                rule: check,
                count: ADMITTED as u64,
            };
            state_ops::admission(&op, &case.codes, &case.scales, &mut report).unwrap();
            assert_eq!(report, case.report, "{check:?} {}", case.name);
            assert!(!case.codes.chunks(4).any(|word| word == [poison::POISON; 4]));
        }
    }
}

// ------------------------------------------------------------ state copy ----

/// Each checkpointing state buffer's entry bytes and layer count, as commit
/// copies them, plus a length of more 16-byte words than one layer's grid
/// has threads and a length off the 16-byte words (the byte path).
fn state_copy_layouts() -> Vec<(&'static str, u64, u32)> {
    let shape = Shape::TP4;
    let grid_words = u64::from(state_copy::MAX_BLOCKS) * u64::from(state_copy::THREADS) * 16;
    vec![
        (
            "state_copy kda.conv",
            shape.kda_conv_bytes(),
            shape.linear_layers() as u32,
        ),
        (
            "state_copy kda.state",
            shape.kda_state_bytes(),
            shape.linear_layers() as u32,
        ),
        (
            "state_copy indexer.tail",
            shape.indexer_tail_bytes(),
            shape.paged_layers(true) as u32,
        ),
        ("state_copy past one grid", grid_words + 48, 3),
        ("state_copy bytes", 1001, 5),
    ]
}

/// Commit's state copy (`state_copy::CUBIN`) as `Sm120Own` prepares it, one
/// case per layout: an entry of every layer of a pool of three entries per
/// layer goes to the same entry of another pool, byte for byte as one host
/// copy per layer (`own_ops::state::state_copy`), and every other byte of
/// the destination keeps its value. The kit's PDL producer rewrites the
/// source.
fn state_copy_cases(rng: &mut Rng) -> LabResult<Vec<OwnCase>> {
    let own = Sm120Own::new(Shape::TP4);
    let need = Need::rows(1);
    state_copy_layouts()
        .into_iter()
        .map(|(name, bytes, layers)| {
            let op = StateCopy {
                bytes,
                layers,
                source_stride: 3 * bytes,
                destination_stride: 3 * bytes,
            };
            // No byte is the poison pattern, so no produced word is; the
            // source fills whole words.
            let mut image = |bytes: u64| {
                (0..bytes.next_multiple_of(4))
                    .map(|_| (rng.next() & 0x7f) as u8)
                    .collect::<Vec<_>>()
            };
            let source = image(op.source_span());
            let destination = image(op.destination_span());
            let written = (0..u64::from(layers))
                .map(|layer| {
                    let start = (layer * op.destination_stride) as usize;
                    start..start + bytes as usize
                })
                .collect::<Vec<_>>();
            let roles = StateCopyBuffers {
                source: Spec::input(DType::U8, (Rows::Fixed(1), source.len()), source),
                destination: Spec::state(
                    DType::U8,
                    (Rows::Fixed(1), destination.len()),
                    destination,
                    Write::Map(Box::new(move |_, live| {
                        let map = WriteMap::of(DType::U8);
                        if live == 0 {
                            map.none()
                        } else {
                            map.byte_ranges(written.clone())
                        }
                    })),
                ),
            };
            let case = OpCase::new(own_op(own.state_copy(&op, need))?, 1, roles)?.oracle(
                move |images, _| {
                    let mut destination = images.bytes("destination")?.to_vec();
                    state_ops::state_copy(&op, images.bytes("source")?, &mut destination)
                        .map_err(refused)?;
                    Ok(vec![("destination".into(), Expected::Bits(destination))])
                },
            );
            Ok((name, need, case))
        })
        .collect()
}

/// Commit's state copy within one state pool, as serving runs it
/// (`instance/execute.rs`, `copy_layers`, takes both entries from the same
/// pool): entry 2 of every layer of a pool of three entries per layer goes
/// to entry 0, launched as `Sm120Own` plans it, byte for byte as the
/// contract's oracle (one host copy per layer), and every other byte of the
/// pool keeps its value. The pool's sources start stale (complemented) and
/// only the producer's copy publishes them. Eager, relaunched, replayed
/// from a graph and behind a PDL producer that writes the pool; every phase
/// compares the whole pool.
#[test]
fn bundle_state_copy_within_one_pool_matches_its_oracle_poisoned() {
    gpu_test(1 << 30, |session| {
        let module = embedded(session, state_copy::CUBIN)?;
        let function = module
            .function(state_copy::SYMBOL)
            .map_err(|e| e.to_string())?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let producer = CopyProducer::new(&lab)?;
        let pdl = std::env::var("CHOPINFER_LATENT_NO_PDL").as_deref() != Ok("1");
        let stream = &session.stream;
        let own = Sm120Own::new(Shape::TP4);
        let mut rng = Rng(0x5747_e0c0);
        let mut evidence = BTreeMap::new();
        for (name, bytes, layers) in state_copy_layouts() {
            let label = format!("{} {name} in one pool", state_copy::SYMBOL);
            let stride = 3 * bytes;
            let op = StateCopy {
                bytes,
                layers,
                source_stride: stride,
                destination_stride: stride,
            };
            let image_bytes = (u64::from(layers) * stride).next_multiple_of(4) as usize;
            // No byte is the poison pattern, so no staged word is.
            let image = (0..image_bytes)
                .map(|_| (rng.next() & 0x7f) as u8)
                .collect::<Vec<_>>();
            let entry = |layer: u32, entry: u64| {
                let start = (u64::from(layer) * stride + entry * bytes) as usize;
                start..start + bytes as usize
            };
            let mut expected = image.clone();
            state_ops::state_copy(&op, &image[2 * bytes as usize..], &mut expected)
                .map_err(refused)?;
            let sources = (0..layers).map(|l| entry(l, 2)).collect::<Vec<_>>();
            let destinations = (0..layers).map(|l| entry(l, 0)).collect::<Vec<_>>();
            let mut stale = image.clone();
            for range in &sources {
                for byte in &mut stale[range.clone()] {
                    *byte = !*byte;
                }
            }
            let staged = upload(session, &image)?;
            let pool = upload(session, &stale)?;
            // The op's launch over views 1 (entry 2's start) and 2 (entry
            // 0's), each its span, bound into the one pool.
            let prepared = own_op(own.state_copy(&op, Need::rows(1)))?;
            let views = StateCopyBuffers {
                source: View {
                    handle: 1,
                    bytes: op.source_span(),
                },
                destination: View {
                    handle: 2,
                    bytes: op.destination_span(),
                },
            };
            let mut launches = prepared
                .launches(&views, View::EMPTY, Counts::rows(1))
                .map_err(refused)?;
            let launch = launches.remove(0);
            let arguments = launch
                .arguments
                .0
                .iter()
                .map(|argument| match argument {
                    Argument::DeviceBuffer(1) => Ok(KernelArgument::Range {
                        buffer: &pool,
                        offset: 2 * bytes as usize,
                        byte_len: op.source_span() as usize,
                    }),
                    Argument::DeviceBuffer(2) => Ok(KernelArgument::Range {
                        buffer: &pool,
                        offset: 0,
                        byte_len: op.destination_span() as usize,
                    }),
                    Argument::U64(value) => Ok(KernelArgument::U64(*value)),
                    other => Err(format!("{label}: an unexpected argument {other:?}")),
                })
                .collect::<LabResult<Vec<_>>>()?;
            let plan = launch.plan;
            let run = |stream: &Stream, pdl: bool| {
                let config = launch::config(kernel_core::contract::LaunchPlan { pdl, ..plan })?;
                function
                    .launch(stream, config, &arguments)
                    .map_err(|e| format!("{label}: {e}"))
            };
            let chain = |stream: &Stream, pdl: bool| {
                producer.copy(stream, &staged, &pool)?;
                run(stream, pdl)
            };
            let graph = session.capture(|stream| chain(stream, pdl))?;
            let written = WriteMap::of(DType::U8).byte_ranges(destinations)?;
            let mut runs =
                Replay::new(session).state("pool", &pool, stale.clone(), written, sources.clone());
            runs.eager(|| chain(stream, false))?;
            let eager = runs
                .expected("pool")
                .ok_or(format!("{label}: no eager pool"))?;
            if let Some(at) = eager.iter().zip(&expected).position(|(a, b)| a != b) {
                return Err(format!(
                    "{label}: byte {at} is {:#04x}, the oracle's {:#04x}",
                    eager[at], expected[at]
                ));
            }
            runs.relaunch(|| chain(stream, false))
                .map_err(|e| format!("{label}: {e}"))?;
            runs.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
            runs.pdl(
                || producer.copy(stream, &staged, &pool),
                || run(stream, pdl),
                pdl,
            )
            .map_err(|e| format!("{label}: {e}"))?;
            let mut receipt = BTreeMap::new();
            runs.record(&mut receipt);
            receipt.insert("bytes".into(), json!(bytes));
            receipt.insert("layers".into(), json!(layers));
            receipt.insert("grid".into(), json!(plan.grid));
            evidence.insert(name.to_owned(), json!(receipt));
        }
        session.receipt(
            "bundle_state_copy_in_one_pool_poisoned",
            BTreeMap::from([("cases".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle state copy within one pool");
}

/// Finite FP8 E4M3 codes (no NaN code) of both signs.
fn e4m3_codes(rng: &mut Rng, count: usize) -> Vec<u8> {
    (0..count)
        .map(|_| {
            let code = rng.next() as u8;
            if code & 0x7f == 0x7f { code ^ 1 } else { code }
        })
        .collect()
}

/// F32 rows whose values are finite over wide exponents, -0 and +0, and in
/// one row of every five infinities and NaNs of either sign.
fn special_rows(rng: &mut Rng, rows: usize, width: usize) -> Vec<f32> {
    let mut values = rng.f32(rows * width);
    for (index, value) in values.iter_mut().enumerate() {
        let bits = rng.next();
        *value = match (index / width % 5, bits % 16) {
            (_, 0) => 0.0,
            (_, 1) => -0.0,
            (4, 2) => f32::INFINITY,
            (4, 3) => f32::NEG_INFINITY,
            (4, 4) => f32::from_bits(0x7fc0_0123),
            (4, 5) => f32::from_bits(0xffc0_0456),
            _ => *value,
        };
    }
    values
}

/// The decode and verify graphs' shared expert beside the routed experts
/// (2026-10-09), from the bundle's embedded cubins: at every graph use of the
/// shared down projection (the decode and verify buckets of 1 to 32 rows
/// and the 64-, 128- and 192-row verify buckets compiled at the prefill
/// capacity), on every rank and at every live count up to 32 (edge counts
/// above), the partial of the projection over zero rows (`ZeroRows`)
/// followed by the routed sum that adds it (`MOE_SUM_ADD_CUBIN`) equals,
/// bit for bit and NaN payloads included, the partial of the routed sum
/// (the split route cubin, F32) followed by the projection that adds it as
/// its residual, the prefill's order and the graphs' before. Every rank
/// runs every expert's routes (one shard). Inputs hold -0, infinities and
/// NaNs in the routed rows and infinities in the activation (some products
/// overflow to infinity), tokens without a route and with cancelling
/// routes. The new pair is poisoned through
/// eager, relaunch, graph replay and a PDL producer that publishes the
/// activation late: the projection must write every bucket row of the MLP
/// output and the sum exactly the live rows of the partial.
#[test]
fn shared_down_beside_the_routed_sum_keeps_the_partial_poisoned() {
    use kernel_core::contract::{RuntimeCounts, ScaleMode, WeightFormat};
    use kernel_core::precision::ExecutionPhase;
    use kernels_sm120::gemm;
    use kernels_sm120::moe::chained;
    use kernels_sm120::precision::Compute;
    gpu_test(2 << 30, |session| {
        let shape = Shape::TP4;
        let p = shape.profile;
        let (n, k) = (p.hidden as usize, p.shared_intermediate as usize);
        let (top_k, experts) = (p.top_k as usize, p.local_experts);
        let fp8 = WeightFormat::Fp8Block128x128(ScaleMode::ExactF32);
        let gemm_module = embedded(session, sm120_precision::CUBIN)?;
        let prefix = chained::split::PREFIX;
        let route_module = embedded(session, &format!("{prefix}_route"))?;
        let add_module = embedded(session, engines::MOE_SUM_ADD_CUBIN)?;
        let sum = engines::routed_sum(shape)?;
        let sum_fn = route_module
            .function(&sum.symbol(prefix))
            .map_err(|e| e.to_string())?;
        let add_fn = add_module
            .function(&sum.add_symbol(prefix))
            .map_err(|e| e.to_string())?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let producer = CopyProducer::new(&lab)?;
        let pdl = std::env::var("CHOPINFER_LATENT_NO_PDL").as_deref() != Ok("1");
        let stream = &session.stream;
        let mut rng = Rng(0x5ba2_ed0d);
        // The rank's shared down weights: FP8 codes [n, k], F32 block scales
        // [n / 128, k / 128] (one block zero).
        let codes = upload(session, &e4m3_codes(&mut rng, n * k))?;
        let mut scales = rng.f32((n / 128) * (k / 128));
        for scale in &mut scales {
            *scale = scale.abs() * 2f32.powi(-6);
        }
        scales[5] = 0.0;
        let scales = upload(session, &bytes_f32(&scales))?;
        let mut evidence = Vec::new();
        let mut compared = 0_u64;
        let mut uses = precision::uses("moe.shared.down")?
            .into_iter()
            .map(|(usage, _)| usage)
            .filter(|usage| usage.phase != ExecutionPhase::Prefill)
            .collect::<Vec<_>>();
        uses.sort_by_key(|usage| (usage.capacity, format!("{:?}", usage.phase)));
        for usage in uses {
            let generated = sm120_precision::offer("moe.shared.down", fp8, usage)?;
            let Compute::Dense(spec) = &generated.offer.handle().compute else {
                return Err("the shared down projection is not a dense GEMM".into());
            };
            let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                return Err("the shared down projection has no GEMM query".into());
            };
            assert!(query.epilogues.residual_add && query.output == DType::F32);
            let rows = usage.capacity as usize;
            let compiled = query.m_capacity as usize;
            let counts: RuntimeCounts = gemm::row_counts(&spec.key, usage.capacity);
            let plan = gemm::plan(&spec.key, counts).map_err(|r| r.to_string())?;
            let gemm_fn = gemm_module
                .function(&spec.symbol)
                .map_err(|e| e.to_string())?;
            if plan.dynamic_shared_bytes >= 48 * 1024 {
                gemm_fn
                    .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            // Activation rows at the compiled capacity (its tensor map's),
            // a quarter of the rows' values scaled up so some products and
            // sums overflow to infinity.
            let mut activation = rng.bf16(compiled * k);
            for (index, value) in activation.iter_mut().enumerate().take(rows * k) {
                if index / k % 4 == 3 && index % 7 == 0 {
                    *value = (*value & 0x807f) | (0xfe << 7);
                }
            }
            let staged = upload(session, &halves(&activation))?;
            let stale = halves(&activation).iter().map(|b| !b).collect::<Vec<_>>();
            let act = upload(session, &stale)?;
            let routed = upload(session, &bytes_f32(&special_rows(&mut rng, rows * top_k, n)))?;
            let zeros = upload(session, &vec![0; rows * n * 4])?;
            let workspace = upload(session, &vec![0; (plan.workspace_bytes as usize).max(256)])?;
            let semaphores = upload(session, &[0; 4096])?;
            let [old_mlp, old_partial, new_mlp, new_partial] =
                [(); 4].map(|()| session.allocate(rows * n * 4));
            let (old_mlp, old_partial, new_mlp, new_partial) =
                (old_mlp?, old_partial?, new_mlp?, new_partial?);
            let gemm_run = |stream: &Stream, residual: &DeviceBuffer, output: &DeviceBuffer, pdl: bool| {
                const W: u64 = 1;
                const A: u64 = 2;
                let args = gemm::arguments(
                    &spec.key,
                    counts,
                    gemm::tensor_maps(&spec.key, W, A).map_err(|r| r.to_string())?,
                    gemm::GemmBuffers {
                        output: 3,
                        partial: 4,
                        semaphores: 5,
                        weight_scales: Some(6),
                        activation_scales: None,
                        residual: Some(7),
                        weight_global_scale: None,
                        activation_global_scale: None,
                    },
                )
                .map_err(|r| r.to_string())?;
                let buffers = BTreeMap::from([
                    (W, &codes),
                    (A, &act),
                    (3, output),
                    (4, &workspace),
                    (5, &semaphores),
                    (6, &scales),
                    (7, residual),
                ]);
                launch::launch(
                    &session.context,
                    &gemm_fn,
                    stream,
                    kernel_core::contract::LaunchPlan { pdl, ..plan },
                    &args,
                    &launch::Bindings { buffers },
                )
            };
            let sum_plan = sum.plan(usage.capacity).map_err(|r| r.to_string())?;
            let lives: Vec<u32> = if rows <= 32 {
                (0..=usage.capacity).collect()
            } else {
                let c = usage.capacity;
                vec![0, 1, 31, c / 2, c - 1, c]
            };
            {
                // Every rank holds a slice of every expert and runs every
                // route: the routed sum's rank argument is 0.
                let rank = 0;
                // Routes: tokens of class 0 have no route, class 1 pads its
                // last route, the rest draws over every expert.
                let ids = (0..rows * top_k)
                    .map(|route| match (route / top_k % 4, route % top_k) {
                        (0, _) => -1,
                        (1, slot) if slot == top_k - 1 => -1,
                        _ => (rng.next() % experts) as i32,
                    })
                    .collect::<Vec<_>>();
                let ids = upload(session, &words(&ids.iter().map(|&id| id as u32).collect::<Vec<_>>()))?;
                for &live in &lives {
                    let label = format!("{} {:?} rank {rank} live {live}", spec.symbol, usage);
                    let live_buffer = upload(session, &live.to_le_bytes())?;
                    let sum_args = |output, addend| {
                        sum_arguments([&routed, &ids, &live_buffer], rank, addend, output)
                    };
                    // Today's order on one stream, from poisoned outputs.
                    poison::write(session, &act, 0, &halves(&activation))?;
                    poison::poison(session, &[&old_mlp, &old_partial])?;
                    sum_fn
                        .launch(stream, launch::config(sum_plan)?, &sum_args(&old_mlp, None))
                        .map_err(|e| format!("{label}: {e}"))?;
                    gemm_run(stream, &old_mlp, &old_partial, false)?;
                    stream.synchronize().map_err(|e| e.to_string())?;
                    let old = poison::read(session, &old_partial)?;
                    // The graphs' order: the projection over zeros, then the
                    // sum that adds its output.
                    let chain = |stream: &Stream, pdl: bool| {
                        producer.copy(stream, &staged, &act)?;
                        gemm_run(stream, &zeros, &new_mlp, pdl)?;
                        add_fn
                            .launch(
                                stream,
                                launch::config(kernel_core::contract::LaunchPlan { pdl, ..sum_plan })?,
                                &sum_args(&new_partial, Some(&new_mlp)),
                            )
                            .map_err(|e| e.to_string())
                    };
                    let graph = session.capture(|stream| chain(stream, pdl))?;
                    let mut runs = Replay::new(session)
                        .output(
                            "mlp_out",
                            &new_mlp,
                            WriteMap::of(DType::F32).prefix(rows * n)?,
                        )
                        .output(
                            "partial",
                            &new_partial,
                            WriteMap::of(DType::F32).prefix(live as usize * n)?,
                        )
                        .state(
                            "activation",
                            &act,
                            stale.clone(),
                            WriteMap::of(DType::U8).none()?,
                            std::iter::once(0..stale.len()).collect(),
                        );
                    runs.eager(|| chain(stream, false))
                        .map_err(|e| format!("{label}: {e}"))?;
                    let new = runs.expected("partial").ok_or("no eager partial")?;
                    let live_bytes = live as usize * n * 4;
                    if let Some(at) = old[..live_bytes]
                        .chunks_exact(4)
                        .zip(new[..live_bytes].chunks_exact(4))
                        .position(|(a, b)| a != b)
                    {
                        return Err(format!(
                            "{label}: partial element {at} is {:#010x}, today's order gives {:#010x}",
                            u32::from_le_bytes(new[4 * at..4 * at + 4].try_into().unwrap()),
                            u32::from_le_bytes(old[4 * at..4 * at + 4].try_into().unwrap()),
                        ));
                    }
                    compared += live as u64 * n as u64;
                    runs.relaunch(|| chain(stream, false))
                        .map_err(|e| format!("{label}: {e}"))?;
                    runs.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
                    runs.pdl(
                        || producer.copy(stream, &staged, &act),
                        || {
                            gemm_run(stream, &zeros, &new_mlp, pdl)?;
                            add_fn
                                .launch(
                                    stream,
                                    launch::config(kernel_core::contract::LaunchPlan {
                                        pdl,
                                        ..sum_plan
                                    })?,
                                    &sum_args(&new_partial, Some(&new_mlp)),
                                )
                                .map_err(|e| e.to_string())
                        },
                        pdl,
                    )
                    .map_err(|e| format!("{label}: {e}"))?;
                    if live == usage.capacity {
                        let mut receipt = BTreeMap::new();
                        runs.record(&mut receipt);
                        receipt.insert("use".into(), json!(format!("{usage:?}")));
                        receipt.insert("symbol".into(), json!(spec.symbol));
                        receipt.insert("rank".into(), json!(rank));
                        evidence.push(json!(receipt));
                    }
                }
            }
        }
        session.receipt(
            "bundle_shared_down_beside_routed_sum",
            BTreeMap::from([
                ("cases".to_owned(), json!(evidence)),
                ("compared_elements".to_owned(), json!(compared)),
            ]),
        )?;
        Ok(())
    })
    .expect("shared down beside the routed sum");
}

/// The routed sum's arguments over its routed rows, ids and live count,
/// with the addend of its form that adds one.
fn sum_arguments<'a>(
    [routed, ids, live]: [&'a DeviceBuffer; 3],
    rank: u32,
    addend: Option<&'a DeviceBuffer>,
    output: &'a DeviceBuffer,
) -> Vec<KernelArgument<'a>> {
    let mut args = vec![
        KernelArgument::Buffer(routed),
        KernelArgument::Buffer(ids),
        KernelArgument::Buffer(live),
        KernelArgument::U32(rank),
    ];
    args.extend(addend.map(KernelArgument::Buffer));
    args.push(KernelArgument::Buffer(output));
    args
}

fn bytes_f32(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The shared down projection's entry over device pointers, for the
/// declarative case: its weight and activation tensor maps, split-K
/// workspace, semaphores and residual come from the bound handles, in
/// parameter order. It runs at its bucket's rows and writes every one.
struct ResidualProjection {
    key: kernel_core::contract::CompileKey,
    symbol: String,
    rows: u32,
}

impl kernel_lab::lab_entry::LabEntry for ResidualProjection {
    fn symbol(&self) -> String {
        self.symbol.clone()
    }

    fn params(&self) -> Vec<EntryParam> {
        let pointer = |name: &str, cxx_type: &str| EntryParam::DevicePointer {
            name: name.into(),
            cxx_type: cxx_type.into(),
        };
        vec![
            pointer("weights", "const uint16_t"),
            pointer("activation", "const uint16_t"),
            pointer("zeros", "const float"),
            pointer("mlp_out", "float"),
            pointer("workspace", "float"),
            pointer("semaphores", "uint32_t"),
        ]
    }

    fn plan(
        &self,
        rows: u32,
    ) -> Result<kernel_core::contract::LaunchPlan, kernel_core::contract::Refusal> {
        if rows != self.rows {
            return Err(kernel_core::contract::Refusal(
                "the projection runs at its bucket's rows",
            ));
        }
        let plan = kernels_sm120::gemm::plan(
            &self.key,
            kernels_sm120::gemm::row_counts(&self.key, self.rows),
        )?;
        Ok(kernel_core::contract::LaunchPlan { pdl: true, ..plan })
    }

    fn pack_arguments(
        &self,
        values: Vec<kernel_core::contract::Argument>,
    ) -> Result<kernel_core::contract::ArgumentList, kernel_core::contract::Refusal> {
        use kernel_core::contract::{Argument, Refusal};
        use kernels_sm120::gemm;
        let handles = values
            .iter()
            .map(|value| match value {
                Argument::DeviceBuffer(handle) => Ok(*handle),
                _ => Err(Refusal("the projection takes device buffers alone")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let &[weights, activation, zeros, output, workspace, semaphores] = handles.as_slice()
        else {
            return Err(Refusal("the projection takes six buffers"));
        };
        gemm::arguments(
            &self.key,
            gemm::row_counts(&self.key, self.rows),
            gemm::tensor_maps(&self.key, weights, activation)?,
            gemm::GemmBuffers {
                output,
                partial: workspace,
                semaphores,
                weight_scales: None,
                activation_scales: None,
                residual: Some(zeros),
                weight_global_scale: None,
                activation_global_scale: None,
            },
        )
    }
}

/// The routed sum that adds a row, its addend named `mlp_out` and its
/// output `partial`, planned at its bucket's tokens.
struct SumAdd {
    spec: kernel_build::EntrySpec,
    plan: kernel_core::contract::LaunchPlan,
}

impl kernel_lab::lab_entry::LabEntry for SumAdd {
    fn symbol(&self) -> String {
        self.spec.symbol.clone()
    }

    fn params(&self) -> Vec<EntryParam> {
        self.spec
            .params
            .iter()
            .map(|param| match param {
                EntryParam::DevicePointer { name, cxx_type } => EntryParam::DevicePointer {
                    name: match name.as_str() {
                        "addend" => "mlp_out".into(),
                        "output" => "partial".into(),
                        other => other.into(),
                    },
                    cxx_type: cxx_type.clone(),
                },
                other => other.clone(),
            })
            .collect()
    }

    fn plan(
        &self,
        _rows: u32,
    ) -> Result<kernel_core::contract::LaunchPlan, kernel_core::contract::Refusal> {
        Ok(kernel_core::contract::LaunchPlan {
            pdl: true,
            ..self.plan
        })
    }
}

/// Shared down entries by symbol and bucket rows: the offer's spec and the
/// uses sharing it.
type SharedDownEntries =
    BTreeMap<(String, u32), (kernel_core::contract::InstantiationSpec, Vec<String>)>;

/// The BF16 shared down entries of the layouts that store the shared
/// experts in BF16, at each graph use (target and drafter), once per symbol
/// and bucket rows, with the uses sharing one.
fn bf16_shared_down_entries(shape: Shape) -> LabResult<SharedDownEntries> {
    use kernel_core::contract::WeightFormat;
    use kernel_core::precision::ExecutionPhase;
    use kernels_sm120::precision::Compute;
    let mut entries = BTreeMap::new();
    for &layout in quantization::covering_layouts() {
        let formats = precision::layout_formats(shape, layout)?;
        for id in ["moe.shared.down", "mtp.moe.shared.down"] {
            let &(_, weight) = formats
                .iter()
                .find(|(name, _)| *name == id)
                .ok_or(format!("{}: no format for {id}", layout.id))?;
            if weight != WeightFormat::Bf16 {
                continue;
            }
            for (usage, _) in precision::uses(id)? {
                if usage.phase == ExecutionPhase::Prefill {
                    continue;
                }
                let generated = sm120_precision::offer(id, weight, usage)?;
                let Compute::Dense(spec) = &generated.offer.handle().compute else {
                    return Err(format!("{id} is not a dense GEMM"));
                };
                entries
                    .entry((spec.symbol.clone(), usage.capacity))
                    .or_insert_with(|| (spec.clone(), Vec::new()))
                    .1
                    .push(format!("{} {id} {usage:?}", layout.id));
            }
        }
    }
    Ok(entries)
}

/// One declarative case of the graphs' order for a BF16 shared down
/// entry at `rows` on `rank`: the projection over zero rows into `mlp_out`
/// (scratch: it writes every bucket row, live or not, and the sum reads the
/// live ones), then the routed sum that adds it into `partial`, expected
/// bit for bit as `today` gives at each live count. Inputs hold -0,
/// infinities and NaNs of both signs and other payloads in the routed rows
/// and the activation, tokens without a route and with routes over every
/// expert.
#[allow(clippy::too_many_arguments)]
fn shared_down_case<'a>(
    projection: &'a ResidualProjection,
    sum_add: &'a SumAdd,
    images: &BTreeMap<&'static str, Vec<u8>>,
    shape: Shape,
    compiled: usize,
    rank: u32,
    workspace: usize,
    today: impl Fn(u32) -> LabResult<Vec<u8>> + 'a,
) -> kernel_lab::conformance::Case<'a> {
    use kernel_core::contract::Argument;
    use kernel_lab::conformance::{Case, Expected, Rows, Write};
    let p = shape.profile;
    let (n, k, top_k) = (
        p.hidden as usize,
        p.shared_intermediate as usize,
        p.top_k as usize,
    );
    let rows = projection.rows as usize;
    let mlp_bytes = rows * n * 4;
    let case = Case::new(projection, projection.rows)
        .then(sum_add)
        .live("live_tokens")
        .input(
            "weights",
            DType::Bf16,
            (Rows::Fixed(n), k),
            images["weights"].clone(),
        )
        .input(
            "activation",
            DType::Bf16,
            (Rows::Fixed(compiled), k),
            images["activation"].clone(),
        )
        .produced("activation")
        .input(
            "zeros",
            DType::F32,
            (Rows::Fixed(rows), n),
            vec![0; mlp_bytes],
        )
        .scratch("mlp_out", mlp_bytes, move |_, _| {
            Ok(std::iter::once(0..mlp_bytes).collect())
        })
        .scratch("workspace", workspace, move |_, _| {
            Ok(std::iter::once(0..workspace).collect())
        })
        .state(
            "semaphores",
            DType::U8,
            (Rows::Fixed(1), 4096),
            vec![0; 4096],
            Write::Map(Box::new(|_, _| WriteMap::of(DType::U8).none())),
            |_, _| Ok(Vec::new()),
        )
        .input(
            "routed",
            DType::F32,
            (Rows::Fixed(rows * top_k), n),
            images["routed"].clone(),
        )
        .input(
            "ids",
            DType::I32,
            (Rows::Fixed(rows), top_k),
            images[&*format!("ids{rank}")].clone(),
        )
        .scalar("rank", Argument::U32(rank))
        .output("partial", DType::F32, (Rows::Live, n), Write::LiveRows)
        .oracle(move |_, live| {
            Ok(if live == 0 {
                Vec::new()
            } else {
                vec![("partial".to_owned(), Expected::Bits(today(live)?))]
            })
        });
    // The split-K tickets end as they start: each tile's last split resets
    // its ticket.
    poison::live_counts(projection.rows)
        .into_iter()
        .filter(|&live| live > 0)
        .fold(case, |case, live| {
            case.no_write("semaphores", live, "split-K tickets are reset to zero")
        })
}

/// The case inputs at `rows` of `compiled` activation rows.
fn shared_down_images(
    rng: &mut Rng,
    shape: Shape,
    rows: usize,
    compiled: usize,
    edge: bool,
) -> BTreeMap<&'static str, Vec<u8>> {
    let p = shape.profile;
    let (n, k, top_k, experts) = (
        p.hidden as usize,
        p.shared_intermediate as usize,
        p.top_k as usize,
        p.local_experts as usize,
    );
    // Activation rows. The mixed inputs give every row, the first included,
    // one +2^120 and one -2^120 element, so the projection's output columns
    // mix finite values, both infinities (one product overflows) and NaNs
    // (both do, Inf - Inf), a few -0 elements, and in rows 1 mod 4 quiet
    // NaNs of either sign and other payloads, against routed rows holding
    // their own NaN payloads. The edge inputs are all -0.
    let mut activation = rng.bf16(compiled * k);
    for (index, value) in activation.iter_mut().enumerate().take(rows * k) {
        let (row, column) = (index / k, index % k);
        *value = if edge {
            0x8000
        } else if column == row * 7 % k {
            0x7b80
        } else if column == (row * 7 + 3) % k {
            0xfb80
        } else if row % 4 == 1 && column % 97 == 3 {
            if column % 2 == 0 { 0x7fc1 } else { 0xffa3 }
        } else if column % 61 == 5 {
            0x8000
        } else {
            *value
        };
    }
    let mut images = BTreeMap::from([
        ("weights", halves(&rng.bf16(n * k))),
        ("activation", halves(&activation)),
        ("routed", bytes_f32(&special_rows(rng, rows * top_k, n))),
    ]);
    // Routes of the one expert shard every rank runs: tokens of class 0
    // have no route, class 1 pads its last route, the rest draws over every
    // expert. The mixed inputs start at class 2, so the first token has all
    // its routes; the edge inputs are class 0.
    let ids = (0..rows * top_k)
        .map(|route| {
            let class = if edge { 0 } else { (route / top_k + 2) % 4 };
            match (class, route % top_k) {
                (0, _) => -1,
                (1, slot) if slot == top_k - 1 => -1,
                _ => (rng.next() as usize % experts) as i32,
            }
        })
        .collect::<Vec<_>>();
    images.insert("ids0", ids.iter().flat_map(|id| id.to_le_bytes()).collect());
    images
}

/// The NVFP4 exports' BF16 shared down projection beside the routed sum
/// through the declarative case runner (2026-10-09, with review cases): two
/// cases per distinct BF16 entry and bucket rows of the target's and the
/// drafter's graph uses, mixed inputs (NaN payloads, infinities and finite
/// values in every row, routes) and edge inputs (no route, -0 activation),
/// from the bundle's embedded cubins (projection formats and the split
/// routed sum add), on the one expert shard every rank runs. Each case runs the graphs' order
/// poisoned, eager, relaunched, graph-replayed and behind PDL producers,
/// and compares the live rows of the partial byte for byte (signed zeros,
/// infinities and NaN payloads included) with today's order, the routed
/// sum then the projection adding it as its residual, run once per live
/// count from the same inputs.
#[test]
fn bf16_shared_down_beside_the_routed_sum_cases() {
    use kernels_sm120::moe::chained;
    gpu_test(2 << 30, |session| {
        let shape = Shape::TP4;
        let p = shape.profile;
        let n = p.hidden as usize;
        let projections = embedded(session, projection_formats::CUBIN)?;
        let prefix = chained::split::PREFIX;
        let route_module = embedded(session, &format!("{prefix}_route"))?;
        let add_module = embedded(session, engines::MOE_SUM_ADD_CUBIN)?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let sum = engines::routed_sum(shape)?;
        let sum_fn = route_module
            .function(&sum.symbol(prefix))
            .map_err(|e| e.to_string())?;
        let entries = bf16_shared_down_entries(shape)?;
        let mut rng = Rng(0x5ba2_ed0e);
        let mut cases = Vec::new();
        for (((symbol, rows), (spec, aliases)), edge) in entries
            .iter()
            .flat_map(|entry| [(entry, false), (entry, true)])
        {
            let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
                return Err("the shared down projection has no GEMM query".into());
            };
            let compiled = query.m_capacity as usize;
            // Every rank holds a slice of every expert and runs every route.
            let rank = 0;
            let projection = ResidualProjection {
                key: spec.key.clone(),
                symbol: symbol.clone(),
                rows: *rows,
            };
            let plan = kernel_lab::lab_entry::LabEntry::plan(&projection, *rows)
                .map_err(|r| r.to_string())?;
            let sum_add = SumAdd {
                spec: sum.add_entry(prefix).map_err(|r| r.to_string())?,
                plan: sum.plan(*rows).map_err(|r| r.to_string())?,
            };
            let workspace = (plan.workspace_bytes as usize).max(256);
            let images = shared_down_images(&mut rng, shape, *rows as usize, compiled, edge);
            // Today's order once per live count, eager on one stream: the
            // routed sum into the MLP output, then the projection adding it.
            let gemm_fn = projections.function(symbol).map_err(|e| e.to_string())?;
            if plan.dynamic_shared_bytes >= 48 * 1024 {
                gemm_fn
                    .set_max_dynamic_shared_memory(plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            let inputs =
                ["weights", "activation", "routed"].map(|name| upload(session, &images[name]));
            let [weights, activation, routed] = inputs;
            let (weights, activation, routed) = (weights?, activation?, routed?);
            let ids = upload(session, &images[&*format!("ids{rank}")])?;
            let workspace_buffer = upload(session, &vec![0; workspace])?;
            let semaphores = upload(session, &[0; 4096])?;
            let mlp_out = upload(session, &vec![0; *rows as usize * n * 4])?;
            let partial = upload(session, &vec![0; *rows as usize * n * 4])?;
            let args = kernel_lab::lab_entry::LabEntry::pack_arguments(
                &projection,
                (1..=6)
                    .map(kernel_core::contract::Argument::DeviceBuffer)
                    .collect(),
            )
            .map_err(|r| r.to_string())?;
            let mut today = BTreeMap::new();
            for live in poison::live_counts(*rows)
                .into_iter()
                .filter(|&live| live > 0)
            {
                let live_buffer = upload(session, &live.to_le_bytes())?;
                sum_fn
                    .launch(
                        &session.stream,
                        launch::config(sum_add.plan)?,
                        &sum_arguments([&routed, &ids, &live_buffer], rank, None, &mlp_out),
                    )
                    .map_err(|e| e.to_string())?;
                launch::launch(
                    &session.context,
                    &gemm_fn,
                    &session.stream,
                    kernel_core::contract::LaunchPlan { pdl: false, ..plan },
                    &args,
                    &launch::Bindings {
                        buffers: BTreeMap::from([
                            (1, &weights),
                            (2, &activation),
                            (3, &mlp_out),
                            (4, &partial),
                            (5, &workspace_buffer),
                            (6, &semaphores),
                        ]),
                    },
                )?;
                session.stream.synchronize().map_err(|e| e.to_string())?;
                today.insert(live, poison::read(session, &partial)?);
            }
            // What the expected partial's live rows hold at the capacity: the
            // mixed inputs must reach the final add with NaNs, infinities and
            // finite values in every bucket, the first row's included. The
            // NaN payloads the device produces are recorded, not assumed:
            // the bytes are compared either way.
            let full = &today[rows][..*rows as usize * n * 4];
            let words = full
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().expect("4 bytes")))
                .collect::<Vec<_>>();
            let first = &words[..n];
            let nan = |w: &&u32| f32::from_bits(**w).is_nan();
            let inf = |w: &&u32| f32::from_bits(**w).is_infinite();
            let finite = |w: &&u32| f32::from_bits(**w).is_finite() && f32::from_bits(**w) != 0.0;
            let payloads = words.iter().filter(nan).collect::<BTreeSet<_>>().len();
            let row0 = [
                first.iter().filter(nan).count(),
                first.iter().filter(inf).count(),
                first.iter().filter(finite).count(),
            ];
            if !edge && row0.contains(&0) {
                return Err(format!(
                    "{symbol} rows {rows}: row 0 of the mixed inputs' partial holds \
                     {row0:?} NaN, infinite and finite nonzero values"
                ));
            }
            let classes = json!({
                "nan": words.iter().filter(nan).count(),
                "nan_payloads": payloads,
                "infinite": words.iter().filter(inf).count(),
                "negative_zero": words.iter().filter(|w| **w == 0x8000_0000).count(),
                "positive_zero": words.iter().filter(|w| **w == 0).count(),
                "finite_nonzero": words.iter().filter(finite).count(),
            });
            let case = shared_down_case(
                &projection,
                &sum_add,
                &images,
                shape,
                compiled,
                rank,
                workspace,
                move |live| {
                    today
                        .get(&live)
                        .cloned()
                        .ok_or(format!("no partial of today's order at live {live}"))
                },
            );
            let evidence = case
                .run_modules(session, &[&projections, &add_module], Some(&lab))
                .map_err(|e| format!("{symbol} rows {rows} rank {rank}: {e}"))?;
            cases.push(json!({
                "symbol": symbol,
                "rows": rows,
                "rank": rank,
                "inputs": if edge { "edge: no route, -0 activation" } else { "mixed" },
                "expected_at_capacity": classes,
                "aliases": aliases,
                "case": evidence,
            }));
        }
        if cases.is_empty() {
            return Err("no BF16 shared down entry".into());
        }
        session.receipt(
            "bundle_bf16_shared_down_beside_routed_sum",
            BTreeMap::from([("cases".to_owned(), json!(cases))]),
        )?;
        Ok(())
    })
    .expect("BF16 shared down beside the routed sum");
}

/// The BF16 shared down cases' declarations hold on the host: every
/// admitted BF16 entry and bucket, its plan, and the chain's links, write
/// maps and oracle coverage (the oracle stands in for today's order).
#[test]
fn bf16_shared_down_cases_are_declared() {
    use kernels_sm120::moe::chained;
    let shape = Shape::TP4;
    let entries = bf16_shared_down_entries(shape).expect("BF16 entries");
    let symbols = entries
        .keys()
        .map(|(symbol, _)| symbol.as_str())
        .collect::<BTreeSet<_>>();
    for wanted in ["28cdc02fe9fb30d7", "2763bff4741b9b46", "91b4d3748cf69f0a"] {
        assert!(
            symbols.iter().any(|symbol| symbol.contains(wanted)),
            "{wanted} is not among {symbols:?}"
        );
    }
    let sum = engines::routed_sum(shape).expect("routed sum");
    let mut rng = Rng(1);
    for (((symbol, rows), (spec, _)), edge) in entries
        .iter()
        .flat_map(|entry| [(entry, false), (entry, true)])
    {
        let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
            panic!("{symbol} has no GEMM query");
        };
        let projection = ResidualProjection {
            key: spec.key.clone(),
            symbol: symbol.clone(),
            rows: *rows,
        };
        let plan = kernel_lab::lab_entry::LabEntry::plan(&projection, *rows).expect("plan");
        let sum_add = SumAdd {
            spec: sum
                .add_entry(chained::split::PREFIX)
                .expect("sum add entry"),
            plan: sum.plan(*rows).expect("sum plan"),
        };
        let compiled = query.m_capacity as usize;
        let images = shared_down_images(&mut rng, shape, *rows as usize, compiled, edge);
        let n = shape.profile.hidden as usize;
        shared_down_case(
            &projection,
            &sum_add,
            &images,
            shape,
            compiled,
            0,
            (plan.workspace_bytes as usize).max(256),
            move |_| Ok(vec![0; *rows as usize * n * 4]),
        )
        .validate()
        .unwrap_or_else(|e| panic!("{symbol} rows {rows}: {e}"));
    }
}

// ---------------------------------------------------- dense projections ----

/// The verify bucket the decode-capacity dense cases run at.
const DENSE_VERIFY_ROWS: u32 = 8;
/// The most rows of a case whose declaration a CPU test checks: the f64
/// GEMM oracle of a prefill entry's whole capacity (8,192 rows) is a GPU
/// run's work, in release.
const DENSE_CPU_ROWS: u32 = 64;

/// One dense case: its name, the need the bundle asks of its use, and the
/// kit cases that check its claims together: the first maps its rows to
/// the capacity-1 offer's one-row launches; a prefill entry's second
/// checks its chunk claim over splits of its own launch.
type DenseCase = (String, Need, Vec<OpCase<'static>>);

/// The use of `id` at `phase` and `capacity` its declaration lists.
fn dense_use(id: &str, phase: ExecutionPhase, capacity: u32) -> Option<OperationUse> {
    precision::uses(id)
        .ok()?
        .into_iter()
        .map(|(usage, _)| usage)
        .find(|usage| usage.phase == phase && usage.capacity == capacity)
}

/// The bundle's offer of `id` over `weight` at `usage` as its backend binds
/// it (the tiled entry, its tensor-map form), when the use is a plain
/// dense projection the kit's case covers: one weight, no epilogue and no
/// layout copy.
fn plain_offer(
    id: &str,
    weight: WeightFormat,
    usage: OperationUse,
) -> Option<kernels_sm120::backend::ProjectionOffer> {
    if sm120_precision::copy(id, weight, usage).ok()?.is_some() {
        return None;
    }
    let operation = sm120_precision::domain(id, weight, usage).ok()?;
    if operation.weights.len() != 1 || operation.epilogues != Default::default() {
        return None;
    }
    Some(kernels_sm120::backend::ProjectionOffer {
        offer: sm120_precision::offer(id, weight, usage).ok()?.offer,
        copy: None,
        dense: kernels_sm120::backend::DenseForm::Tensor,
        live_output: false,
    })
}

/// The dense form an offer launches, without its counts: its kernel, its
/// weight, whether it is compiled at a decode capacity, and there whether
/// it splits its K chains over CTAs, at a prefill capacity its knobs (the
/// warp, staged or decoded tiles).
fn dense_form(offer: &kernels_sm120::backend::ProjectionOffer) -> Option<String> {
    let kernels_sm120::precision::Compute::Dense(spec) = &offer.offer.handle().compute else {
        return None;
    };
    let kernel_core::contract::StaticQuery::Gemm(query) = spec.key.query else {
        return None;
    };
    let kernel = format!("{:?}", spec.key.dense?);
    let kernel = kernel.split(' ').next().unwrap_or_default().to_owned();
    let detail = match query.m_capacity <= 32 {
        true if spec.key.knobs.split_k_capacity > 1 => "decode split".to_owned(),
        true => "decode".to_owned(),
        false => format!("prefill {:?}", spec.key.knobs),
    };
    Some(format!("{kernel} {:?} {detail}", query.weight))
}

/// Every plain dense use of the bundle whose offer claims something, with
/// its offer and dense form: (form, id, weight, use, offer).
#[allow(clippy::type_complexity)]
fn claimed_dense_uses() -> Vec<(
    String,
    &'static str,
    WeightFormat,
    OperationUse,
    kernels_sm120::backend::ProjectionOffer,
)> {
    let mut claimed = Vec::new();
    for &layout in quantization::covering_layouts() {
        for (id, weight) in precision::layout_formats(Shape::TP4, layout).unwrap() {
            for (usage, _) in precision::uses(id).unwrap_or_default() {
                let Some(offer) = plain_offer(id, weight, usage) else {
                    continue;
                };
                if offer.offer.invariance() == kernel_core::properties::Invariance::NONE {
                    continue;
                }
                if let Some(form) = dense_form(&offer) {
                    claimed.push((form, id, weight, usage, offer));
                }
            }
        }
    }
    claimed
}

/// The bundle's plain dense projections, one per dense form it serves
/// (the smallest weight of each), each with its use's offer and its
/// capacity-1 decode offer: (form, id, weight, use, offer, decode offer).
/// A decode-capacity form runs at its verify bucket of
/// [`DENSE_VERIFY_ROWS`]; a prefill-capacity form (prefill uses, and the
/// verify buckets above 32 rows compiled at the prefill capacity, such as
/// FP8 warp tiles) at any use that launches it.
#[allow(clippy::type_complexity)]
fn dense_representatives() -> Vec<(
    String,
    &'static str,
    WeightFormat,
    OperationUse,
    kernels_sm120::backend::ProjectionOffer,
    kernels_sm120::backend::ProjectionOffer,
)> {
    let mut chosen = BTreeMap::new();
    for &layout in quantization::covering_layouts() {
        for (id, weight) in precision::layout_formats(Shape::TP4, layout).unwrap() {
            let Some(decode) = dense_use(id, ExecutionPhase::Decode, 1)
                .and_then(|usage| plain_offer(id, weight, usage))
            else {
                continue;
            };
            let uses = precision::uses(id).unwrap_or_default();
            for (usage, _) in uses {
                let Some(offer) = plain_offer(id, weight, usage) else {
                    continue;
                };
                let verify_bucket =
                    usage.phase == ExecutionPhase::Verify && usage.capacity == DENSE_VERIFY_ROWS;
                if !verify_bucket && compiled_rows(&offer).is_none_or(|rows| rows <= 32) {
                    continue;
                }
                let Some(form) = dense_form(&offer) else {
                    continue;
                };
                // A use below its entry's compiled capacity runs through
                // the entry's offer at that capacity; a use at it is
                // preferred.
                let widened = offer.offer.usage().capacity < compiled_rows(&offer).unwrap_or(0);
                let offer = if widened {
                    at_compiled_capacity(&offer)
                } else {
                    offer
                };
                let size = sm120_precision::domain(id, weight, usage)
                    .ok()
                    .and_then(|operation| operation.matrix_geometry().ok())
                    .map_or(u64::MAX, |geometry| geometry.rows * geometry.k);
                let key = (widened, size);
                let better = chosen
                    .get(&form)
                    .is_none_or(|(held, ..): &((bool, u64), _, _, _, _, _)| key < *held);
                if better {
                    chosen.insert(
                        form.clone(),
                        (key, id, weight, usage, offer, decode.clone()),
                    );
                }
            }
        }
    }
    chosen
        .into_iter()
        .map(|(form, (_, id, weight, usage, offer, decode))| {
            (form, id, weight, usage, offer, decode)
        })
        .collect()
}

/// The rows `offer`'s GEMM entry is compiled for.
fn compiled_rows(offer: &kernels_sm120::backend::ProjectionOffer) -> Option<u32> {
    let kernels_sm120::precision::Compute::Dense(spec) = &offer.offer.handle().compute else {
        return None;
    };
    match spec.key.query {
        kernel_core::contract::StaticQuery::Gemm(query) => Some(query.m_capacity),
        _ => None,
    }
}

/// `offer`'s entry offered at its compiled capacity, keeping its claim: a
/// verify bucket above 32 rows launches an entry compiled at the prefill
/// capacity, whose tensor maps cover that capacity's rows, so its kit case
/// allocates and launches them all (the bucket's rows among them) through
/// the same compiled entry (a catalog offer of it, whose handle equals the
/// offer's: BF16 activations bind no quantizer).
fn at_compiled_capacity(
    offer: &kernels_sm120::backend::ProjectionOffer,
) -> kernels_sm120::backend::ProjectionOffer {
    let kernels_sm120::precision::Compute::Dense(spec) = &offer.offer.handle().compute else {
        panic!("a dense projection offer has a dense entry");
    };
    let compiled = compiled_rows(offer).expect("a dense GEMM entry");
    let usage = OperationUse {
        capacity: compiled,
        ..offer.offer.usage()
    };
    let widened = kernels_sm120::precision::KernelOffer::catalog_dense(spec.clone(), usage)
        .expect("the entry's catalog offer")
        .with_invariance(offer.offer.invariance());
    assert_eq!(
        format!("{:?}", widened.handle()),
        format!("{:?}", offer.offer.handle()),
        "the catalog offer launches the offer's entry"
    );
    kernels_sm120::backend::ProjectionOffer {
        offer: widened,
        ..offer.clone()
    }
}

/// SM120 over `offers`, bound through their capability inventory as the
/// kit's pilot binds offers.
fn dense_backend(
    offers: &[kernels_sm120::backend::ProjectionOffer],
) -> LabResult<kernels_sm120::backend::Sm120> {
    let kernels = offers.iter().map(|o| o.offer.clone()).collect::<Vec<_>>();
    let available = kernels_sm120::precision::capabilities(&kernels)
        .map_err(|e| e.0.to_string())?
        .entries();
    let list = kernels_sm120::backend::OfferList::new(offers.to_vec(), available, "0".repeat(64))
        .map_err(|e| e.0.to_string())?;
    Ok(kernels_sm120::backend::Sm120::new().with_projections(Box::new(list)))
}

/// The operation `offer`'s capability states, as the kit's pilot derives it.
fn dense_operation(
    offer: &kernels_sm120::backend::ProjectionOffer,
) -> LabResult<kernel_core::precision::OperationSpec> {
    let capability = offer.offer.capability().map_err(|e| e.0.to_string())?;
    kernels_sm120::precision::generated::comparison::operation(&capability)
}

/// The rows of a launch of `rows` rows the rows check maps to one-row
/// launches: every row of a decode bucket; of a prefill entry's launch,
/// every row below `every` (the verify bucket a use launches it at), then
/// rows at both ends, around the 64-row granule and in the middle.
fn mapped_rows(rows: u32, every: u32) -> Vec<usize> {
    let rows = rows as usize;
    if rows <= 32 {
        return (0..rows).collect();
    }
    let mut mapped = BTreeSet::from([0, 1, 2, 63, 64, 65, rows / 2, rows - 2, rows - 1]);
    mapped.extend(0..every as usize);
    mapped.retain(|&row| row < rows);
    mapped.into_iter().collect()
}

/// The kit case of `offer`'s op over `rows` rows (a prefill entry's whole
/// compiled capacity, which its tensor maps cover), asked `need`.
fn dense_op_case(
    offer: &kernels_sm120::backend::ProjectionOffer,
    rows: u32,
    need: Need,
) -> LabResult<OpCase<'static>> {
    let backend = dense_backend(std::slice::from_ref(offer))?;
    let spec = dense_operation(offer)?;
    let op = backend
        .projection(&spec, offer.offer.usage(), need)
        .map_err(|refused| format!("{refused}"))?
        .handle;
    kit::cases::projection_case(&spec, rows, op)
}

/// The rows a prefill entry's chunk case leaves to its mapped case: a
/// projection serves the capacity-1 offer at one row, so the rows check
/// of the entry's launch against its own one-row launches would not be
/// the bits the bundle relies on.
const DENSE_ROWS_ELSEWHERE: &str =
    "the rows check runs against the capacity-1 offer in the entry's mapped case";

/// The kit case of `offer`'s op over `rows` rows asked `need`, whose rows
/// map to one-row launches of `decode`'s op ([`mapped_rows`] below
/// `every`): each such row's activation as the decode case's one row, its
/// output row against the decode output's. Its rows are rows, not items, but the mapping
/// reports chunk not exercised; [`dense_op_case`] checks it.
fn dense_case(
    offer: &kernels_sm120::backend::ProjectionOffer,
    decode: &kernels_sm120::backend::ProjectionOffer,
    rows: u32,
    every: u32,
    need: Need,
) -> LabResult<OpCase<'static>> {
    use kernel_lab::kit::decode::{Compared, DecodeLaunch};
    let backend = dense_backend(&[offer.clone(), decode.clone()])?;
    let spec = dense_operation(offer)?;
    let decode_spec = dense_operation(decode)?;
    let usage = offer.offer.usage();
    let op = backend
        .projection(&spec, usage, need)
        .map_err(|refused| format!("{refused}"))?
        .handle;
    let one = backend
        .projection(&decode_spec, decode.offer.usage(), Need::rows(1))
        .map_err(|refused| format!("{refused}"))?
        .handle;
    let case = kit::cases::projection_case(&spec, rows, op)?;
    let decode_case = kit::cases::projection_case(&decode_spec, 1, one)?;
    let geometry = spec.matrix_geometry().map_err(|e| e.to_string())?;
    let (k, n) = (geometry.k as usize, geometry.rows as usize);
    let output = match spec.arithmetic.output {
        DType::F32 => 4,
        _ => 2,
    } * n;
    let decoded = kit::Decoded::new(
        decode_case,
        |images, _| Ok(images.clone()),
        move |verified, images| {
            let activation = verified.images.bytes("activation")?;
            Ok(mapped_rows(verified.live, every)
                .into_iter()
                .map(|row| DecodeLaunch {
                    label: format!("row {row}"),
                    images: images.with(
                        "activation",
                        activation[row * k * 2..(row + 1) * k * 2].to_vec(),
                    ),
                    compared: vec![Compared {
                        verify: ("output".into(), row * output..(row + 1) * output),
                        decode: ("output".into(), 0..output),
                    }],
                })
                .collect())
        },
    );
    Ok(case
        .item_rows("a row's one-row reference is the operation's capacity-1 offer")
        .decoded(Ok(decoded)))
}

/// Every dense case: each representative's verify use asked one-row bits
/// up to its bucket, or its prefill use asked the bundle's chunk granule,
/// over the bucket's rows at a decode capacity and over the entry's
/// compiled capacity at a prefill one.
fn dense_cases() -> LabResult<Vec<DenseCase>> {
    use kernel_core::properties::{Chunk, Invariance};
    let mut cases = Vec::new();
    for (form, id, weight, usage, offer, decode) in dense_representatives() {
        let compiled = compiled_rows(&offer).ok_or_else(|| format!("{form}: no GEMM entry"))?;
        let prefill_capacity = compiled > 32;
        // A prefill-capacity entry runs over its whole compiled capacity,
        // which its tensor maps cover; a verify use of it compares its
        // bucket's every row.
        let rows = if prefill_capacity {
            compiled
        } else {
            usage.capacity
        };
        let (exact, every) = match usage.phase {
            ExecutionPhase::Prefill => (
                Invariance {
                    chunk: Chunk::Granule(64),
                    ..Invariance::NONE
                },
                0,
            ),
            // Every row of the largest verify bucket that launches the form.
            _ => {
                let largest = precision::uses(id)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(other, _)| other)
                    .filter(|other| other.phase == ExecutionPhase::Verify)
                    .filter(|&other| {
                        plain_offer(id, weight, other).and_then(|o| dense_form(&o))
                            == Some(form.clone())
                    })
                    .map(|other| other.capacity)
                    .max()
                    .unwrap_or(usage.capacity);
                (Invariance::one_row(usage.capacity), largest)
            }
        };
        let need = Need::rows(rows).with_exact(exact);
        let name = format!(
            "{form} ({id} {weight:?} {:?} {})",
            usage.phase, usage.capacity
        );
        let mut checked = vec![dense_case(&offer, &decode, rows, every, need)?];
        if prefill_capacity {
            checked.push(
                dense_op_case(&offer, rows, need)?
                    .without_properties(&[Property::Rows], DENSE_ROWS_ELSEWHERE),
            );
        }
        cases.push((name, need, checked));
    }
    Ok(cases)
}

/// The dense forms the declaration claims are the bundle's: the cases
/// cover Tiled over BF16 and NVFP4, TiledChains over BF16 and StripChains
/// over BF16 and FP8 at decode capacities, and Tiled over NVFP4 and
/// TiledChains over BF16 and FP8 at prefill capacities, and every plain
/// dense use whose offer claims something launches one of the cases'
/// forms (the knobs of a prefill-capacity form included, so the FP8 warp
/// tiles of the verify buckets above 32 rows are a case of their own);
/// each declares on the CPU (the decode cases; a prefill-capacity case's
/// on the GPU run) and its op claims what its use asks.
#[test]
fn dense_rows_cases_cover_every_claimed_form_and_declare() {
    let cases = dense_cases().expect("dense cases");
    let forms = cases
        .iter()
        .map(|(name, ..)| {
            let mut words = name.split(' ');
            let kernel = words.next().unwrap_or_default();
            let weight = words.next().unwrap_or_default();
            let weight = weight.split('(').next().unwrap_or_default();
            let range = words.find(|w| *w == "decode" || *w == "prefill");
            format!("{kernel} {weight} {}", range.unwrap_or_default())
        })
        .collect::<BTreeSet<_>>();
    for form in [
        "Tiled Bf16 decode",
        "Tiled Nvfp4 decode",
        "TiledChains Bf16 decode",
        "StripChains Bf16 decode",
        "StripChains Fp8Block128x128 decode",
        "Tiled Nvfp4 prefill",
        "TiledChains Bf16 prefill",
        "TiledChains Fp8Block128x128 prefill",
    ] {
        assert!(forms.contains(form), "no case of {form}: {forms:?}");
    }
    let cased = cases
        .iter()
        .map(|(name, ..)| name.split(" (").next().unwrap_or_default())
        .collect::<BTreeSet<_>>();
    let claimed = claimed_dense_uses();
    assert!(!claimed.is_empty());
    for (form, id, weight, usage, _) in &claimed {
        assert!(
            cased.contains(form.as_str()),
            "{id} {weight:?} {usage:?} launches {form}, which no case covers"
        );
    }
    for (name, _, checked) in &cases {
        let prefill = name.contains(" prefill ");
        assert_eq!(checked.len(), 1 + usize::from(prefill), "{name}");
    }
    let mut declared = 0;
    for (name, need, checked) in &cases {
        if need.capacity > DENSE_CPU_ROWS {
            continue;
        }
        declared += 1;
        for case in checked {
            for check in case.checks(*need, None, None, None) {
                match (check.name.as_str(), &check.outcome) {
                    ("declaration" | "need", Outcome::Passed(_)) => {}
                    (_, Outcome::NotExercised(_)) => {}
                    (_, outcome) => panic!("{name} {}: {outcome:?}", check.name),
                }
            }
        }
    }
    assert!(declared >= 5, "{declared} decode cases declared");
}

/// Every dense case passes on a GPU from the bundle's embedded cubins: the
/// conformance checks against the GEMM oracle, poisoned through relaunch,
/// graph replay and a PDL producer, and each claimed property: every row
/// of the verify bucket, or the mapped rows of a prefill launch, equals
/// that row launched alone through the capacity-1 offer (rows), keeps its
/// bits beside copies of itself (batch), and a prefill launch split at 1,
/// half and all but one of its rows gives the whole launch's bits (chunk,
/// in the entry's second case). A check one case does not exercise must
/// pass in another case of the same use; a check may otherwise go
/// unexercised only where it does not apply (an op that waits on no PDL
/// producer, or reads no device live count: the projection kernels launch
/// the rows the host passes, which every run's live counts cover).
#[test]
fn dense_rows_keep_their_claims_on_a_gpu() {
    gpu_test(1 << 30, |session| {
        let functions = Embedded.load(session)?;
        let probes = Embedded.probes(session)?;
        let mut evidence = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, need, checked) in dense_cases()? {
            let claims = checked[0].info().guarantees.claims();
            let mut receipts = Vec::new();
            let mut passed = BTreeSet::new();
            let mut elsewhere = Vec::new();
            for case in &checked {
                let checks = case.checks(need, Some(session), Some(&*functions), probes.as_ref());
                let mut receipt = BTreeMap::new();
                for check in checks {
                    match &check.outcome {
                        Outcome::Passed(value) => {
                            passed.insert(check.name.clone());
                            receipt.insert(check.name.clone(), value.clone());
                        }
                        Outcome::NotExercised(reason)
                            if check.inapplicable() || check.name == "device live count" =>
                        {
                            receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                        }
                        Outcome::NotExercised(reason) => {
                            receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                            elsewhere.push((check.name.clone(), reason.clone()));
                        }
                        Outcome::Failed(error) => {
                            failures.push(format!("{name} {}: {error}", check.name));
                        }
                    }
                }
                receipts.push(receipt);
            }
            for (check, reason) in elsewhere {
                if !passed.contains(&check) {
                    failures.push(format!(
                        "{name} {check}: not exercised in any case: {reason}"
                    ));
                }
            }
            for property in claims {
                let check = kit::properties::name(property);
                if !passed.contains(&check) {
                    failures.push(format!("{name}: {check} did not pass"));
                }
            }
            evidence.insert(name, json!(receipts));
        }
        if !failures.is_empty() {
            return Err(failures.join("\n"));
        }
        session.receipt(
            "bundle_dense_rows",
            BTreeMap::from([("cases".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle dense rows");
}

// ------------------------------------------------- context parallelism ----

/// The rows of every context-parallel case: a whole 64-row granule, so a
/// chunk split on it falls at the case's ends.
const CP_ROWS: u32 = 64;

/// The context-parallel merge and localization cases: (name, need, case).
type CpCase = (String, Need, OpCase<'static>);

/// What the bundle asks of its localizations and partial merges: what every
/// phase needs (the 64-row chunk granule, one-row bits of 32 verify rows),
/// over `rows` rows.
fn cp_need(rows: u32) -> Need {
    use kernel_core::properties::{Chunk, Invariance};
    Need::rows(rows).with_exact(Invariance {
        chunk: Chunk::Granule(64),
        ..Invariance::one_row(32.min(rows))
    })
}

/// The bundle's backend, with its context-parallel entries and what it
/// declares for them.
fn cp_backend() -> Box<dyn Backend> {
    let shape = Shape::TP4;
    sm120_backend(&bundle_engines(shape), shape).0
}

/// A partial merge case over `rows` rows of the bundle's entry `op`, whose
/// rows map to one-row launches of the same op: each row's pairs, on
/// every rank, as a one-row case's packed pairs, its merged heads against
/// the one-row output.
fn partial_merge_cp_case(
    backend: &dyn Backend,
    op: kernel_core::context_parallel::ContextParallelQuery,
    rows: u32,
) -> LabResult<OpCase<'static>> {
    use kernel_core::context_parallel::ContextParallelQuery;
    use kernel_lab::kit::cases::context_parallel::{MergeLayout, partial_merge_case};
    use kernel_lab::kit::decode::{Compared, DecodeLaunch};
    let ContextParallelQuery::PartialMerge {
        ranks,
        heads,
        width,
        ..
    } = op
    else {
        return Err("a partial merge case needs a partial merge".into());
    };
    let (ranks, heads, width) = (ranks as usize, heads as usize, width as usize);
    let prepared = |rows| {
        backend
            .partial_merge(&op, cp_need(rows))
            .map_err(|refused| format!("{refused}"))
    };
    let layout = MergeLayout::packed(rows as usize, heads);
    let case = partial_merge_case(op, layout, rows, prepared(rows)?)?;
    let one = partial_merge_case(op, MergeLayout::packed(1, heads), 1, prepared(1)?)?;
    let row_bytes = heads * width * 2;
    let decoded = kit::Decoded::new(
        one,
        |images, _| Ok(images.clone()),
        move |verified, images| {
            let partials = verified.images.f32s("partials")?;
            let (pairs, one_pairs) = (verified.live as usize * heads, heads);
            let (segment, one_segment) = (pairs * (width + 2), one_pairs * (width + 2));
            (0..verified.live as usize)
                .map(|row| {
                    let mut own = vec![0.0f32; ranks * one_segment];
                    for rank in 0..ranks {
                        let (at, to) = (rank * segment, rank * one_segment);
                        let pair = row * heads;
                        own[to..to + heads]
                            .copy_from_slice(&partials[at + pair..at + pair + heads]);
                        own[to + one_pairs..to + one_pairs + heads].copy_from_slice(
                            &partials[at + pairs + pair..at + pairs + pair + heads],
                        );
                        let (o, one_o) = (at + 2 * pairs + pair * width, to + 2 * one_pairs);
                        own[one_o..one_o + heads * width]
                            .copy_from_slice(&partials[o..o + heads * width]);
                    }
                    Ok(DecodeLaunch {
                        label: format!("row {row}"),
                        images: images.with(
                            "partials",
                            own.iter().flat_map(|v| v.to_le_bytes()).collect(),
                        ),
                        compared: vec![Compared {
                            verify: ("output".into(), row * row_bytes..(row + 1) * row_bytes),
                            decode: ("output".into(), 0..row_bytes),
                        }],
                    })
                })
                .collect()
        },
    );
    Ok(case
        .item_rows("a row's pairs sit in every rank's segment of the gathered partials")
        .decoded(Ok(decoded)))
}

/// Every context-parallel case at each group size: the key and position
/// localizations to the group's last rank over [`CP_ROWS`], and the
/// partial merge over [`CP_ROWS`] and over 33 rows, each asked what the
/// bundle asks.
fn cp_cases() -> LabResult<Vec<CpCase>> {
    use kernel_core::context_parallel::ContextParallelQuery;
    use kernel_lab::kit::cases::context_parallel::localize_keys_case;
    let backend = cp_backend();
    let mut cases = Vec::new();
    for entry in cp_entries::entries(Shape::TP4) {
        let op = entry.query();
        let name = entry.symbol();
        match op {
            ContextParallelQuery::LocalizeKeys { ranks, .. } => {
                let rank = ranks - 1;
                let need = cp_need(CP_ROWS);
                let prepared = backend
                    .localize_keys(&op, rank, need)
                    .map_err(|refused| format!("{refused}"))?;
                let case = localize_keys_case(op, rank, CP_ROWS, prepared)?;
                cases.push((format!("{name} rank {rank}"), need, case));
            }
            ContextParallelQuery::PartialMerge { .. } => {
                for rows in [CP_ROWS, 33] {
                    let case = partial_merge_cp_case(backend.as_ref(), op, rows)?;
                    cases.push((format!("{name} rows {rows}"), cp_need(rows), case));
                }
            }
            _ => {}
        }
    }
    Ok(cases)
}

/// The bundle's localizations and partial merges at both group sizes
/// declare on the CPU, and their ops grant what the bundle asks.
#[test]
fn context_parallel_cases_declare_and_grant_the_asks() {
    let cases = cp_cases().expect("context-parallel cases");
    assert_eq!(cases.len(), 2 * 4);
    for (name, need, case) in &cases {
        for check in case.checks(*need, None, None, None) {
            match (check.name.as_str(), &check.outcome) {
                ("declaration" | "need", Outcome::Passed(_)) => {}
                (_, Outcome::NotExercised(_)) => {}
                (_, outcome) => panic!("{name} {}: {outcome:?}", check.name),
            }
        }
    }
}

/// Every context-parallel case passes on a GPU from the bundle's embedded
/// cubins: conformance against the contract's oracle, poisoned replays,
/// and each claimed property. A localization is a row op: its rows, batch
/// and chunk checks must pass. A partial merge reads every rank's pairs of
/// a row from one gathered buffer, so its rows check maps each row to a
/// one-row merge of its own pairs, at two row counts; its batch and chunk
/// claims follow from one-row bits at any row count and are not run
/// otherwise.
#[test]
fn context_parallel_cases_keep_their_claims_on_a_gpu() {
    gpu_test(1 << 30, |session| {
        let functions = Embedded.load(session)?;
        let probes = Embedded.probes(session)?;
        let mut evidence = BTreeMap::new();
        let mut failures = Vec::new();
        for (name, need, case) in cp_cases()? {
            let merge = name.contains("partial_merge");
            let claims = case.info().guarantees.claims();
            let checks = case.checks(need, Some(session), Some(&*functions), probes.as_ref());
            let mut receipt = BTreeMap::new();
            let mut passed = BTreeSet::new();
            for check in checks {
                let property = ["rows", "batch", "chunk"]
                    .iter()
                    .any(|p| check.name.contains(p));
                match &check.outcome {
                    Outcome::Passed(value) => {
                        passed.insert(check.name.clone());
                        receipt.insert(check.name.clone(), value.clone());
                    }
                    Outcome::NotExercised(reason)
                        if check.inapplicable() || (merge && property) =>
                    {
                        receipt.insert(check.name.clone(), json!({ "not exercised": reason }));
                    }
                    outcome => failures.push(format!("{name} {}: {outcome:?}", check.name)),
                }
            }
            for property in claims {
                let check = kit::properties::name(property);
                let required = !merge || property == Property::Rows;
                if required && !passed.contains(&check) {
                    failures.push(format!("{name}: {check} did not pass"));
                }
            }
            evidence.insert(name, json!(receipt));
        }
        if !failures.is_empty() {
            return Err(failures.join("\n"));
        }
        session.receipt(
            "bundle_context_parallel_claims",
            BTreeMap::from([("cases".to_owned(), json!(evidence))]),
        )?;
        Ok(())
    })
    .expect("bundle context-parallel claims");
}

// ------------------------------- all-reduces of BF16 partials, on 2 ranks ----

/// The ranks the bundle's all-reduce cases run on: a pair, which the
/// entries serve as they serve four.
const PAIR: u32 = 2;
/// The rows of the composed all-reduce case: past the BF16-partials
/// table's turns rows, so every row count the table has is a composed one.
const COMPOSED_ROWS: u32 = kernels_sm120::backend::BF16_TURNS_ROWS + 8;
/// The rows of the comm epilogue cases: past every comm protocol
/// threshold, and a whole 64-row granule.
const EPILOGUE_ROWS: u32 = 64;

/// What the bundle asks of its all-reduces and comm epilogues over `rows`
/// rows: the 64-row chunk granule and one-row bits of 32 verify rows.
fn every_phase_need(rows: u32) -> Need {
    use kernel_core::properties::{Chunk, Invariance};
    Need::rows(rows).with_exact(Invariance {
        chunk: Chunk::Granule(64),
        ..Invariance::one_row(32)
    })
}

/// The bundle's comm backend of the admission without the low-latency
/// entry and with (`phased`) or without the phased entries: its rounding
/// rows, with what they declare, its comm epilogue entry, and the phased
/// entries where admitted.
fn reduce_backend(phased: bool) -> std::sync::Arc<dyn Backend> {
    let shape = Shape::TP4;
    let (_, served) = sm120_backend(&bundle_engines(shape), shape);
    let (_, backend) = served
        .1
        .0
        .iter()
        .find(|(admission, _)| !admission.low_latency && admission.phased == phased)
        .expect("the admission's backend");
    backend.clone()
}

/// Every rank's views of an all-reduce for rank `me`'s record under
/// `prefix`: its own output and state regions of `state` bytes, every
/// other rank's by path.
fn reduce_rank_views<'a>(
    ranks: u32,
    me: u32,
    prefix: &str,
    state: &[u64],
) -> Vec<RankViews<Slot<'a>>> {
    (0..ranks)
        .map(|rank| RankViews {
            output: Slot::peer(rank, format!("{prefix}output")),
            state: state
                .iter()
                .enumerate()
                .map(|(at, &bytes)| match rank == me {
                    true => Slot::State(bytes),
                    false => Slot::peer(rank, format!("{prefix}ranks[{rank}].state[{at}]")),
                })
                .collect(),
        })
        .collect()
}

/// The rounded sums of every rank's live F32 partials `rows` rows of
/// `width`, as the all-reduce's contract gives them, padded to `capacity`
/// rows.
fn rounded_sums(
    images: &[Images],
    path: &str,
    live: u32,
    width: usize,
    capacity: u32,
) -> LabResult<Vec<u16>> {
    let elements = live as usize * width;
    let inputs = images
        .iter()
        .map(|images| Ok(images.f32s(path)?[..elements].to_vec()))
        .collect::<LabResult<Vec<_>>>()?;
    let mut sums = match elements {
        0 => Vec::new(),
        _ => kernel_core::oracle::all_reduce_bf16_partials(&inputs).map_err(|e| e.0.to_string())?,
    };
    sums.resize(capacity as usize * width, 0);
    Ok(sums)
}

/// The composed BF16-partials all-reduce of the bundle's comm backend
/// without the phased entries over [`COMPOSED_ROWS`] rows of the hidden
/// width: the kit's finite partials, the contract's rounded sums as bits
/// on every rank, at the table's bounds' row counts.
fn composed_case<'a>(
    ranks: u32,
    ops: Vec<Prepared<kernel_core::collective::AllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    all_reduce_case(ranks, ops, false)
}

/// The MoE sublayers' rounded all-reduce of the bundle's comm backend
/// (its phased F32 entries where admitted, else the F32 pair and the bundle's
/// round row) over [`COMPOSED_ROWS`] rows of the hidden width: the kit's
/// finite partials, the F32 rank-order sums rounded once as bits on every
/// rank.
fn rounded_case<'a>(
    ranks: u32,
    ops: Vec<Prepared<kernel_core::collective::AllReduceBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    all_reduce_case(ranks, ops, true)
}

/// [`composed_case`], or with `rounded` [`rounded_case`].
fn all_reduce_case<'a>(
    ranks: u32,
    ops: Vec<Prepared<kernel_core::collective::AllReduceBuffers<View>>>,
    rounded: bool,
) -> LabResult<CollectiveCase<'a>> {
    use kernel_core::collective::AllReduceBuffers;
    use kernel_lab::kit::cases::collectives::{PARTIALS_COUNTS, finite_partials};
    let width = Shape::TP4.profile.hidden as usize;
    let state = ops
        .first()
        .map(|op| op.info().state.clone())
        .unwrap_or_default();
    let rows = COMPOSED_ROWS as usize;
    let records = (0..ranks)
        .map(|me| AllReduceBuffers {
            input: Slot::Own(Spec::input(
                DType::F32,
                (Rows::Live, width),
                finite_partials(me, ranks, rows * width)
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect(),
            )),
            output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, width))),
            live_rows: Some(Slot::Own(Spec::live())),
            ranks: reduce_rank_views(ranks, me, "", &state),
        })
        .collect();
    Ok(CollectiveCase::new(ops, COMPOSED_ROWS, records)?
        .oracle(move |images, live| {
            let sums = match rounded {
                true => f32_rounded_sums(images, "input", live, width, COMPOSED_ROWS)?,
                false => rounded_sums(images, "input", live, width, COMPOSED_ROWS)?,
            };
            let bits = kit::bf16_bytes(&sums);
            Ok((0..images.len())
                .map(|_| vec![("output".to_owned(), Expected::Bits(bits.clone()))])
                .collect())
        })
        .same_on_every_rank(&["output"])
        .counts(&PARTIALS_COUNTS))
}

/// The sums of the rounded all-reduce over every rank's `live` rows of
/// `path` (`oracle::all_reduce_rounded`), zero-padded to `capacity` rows.
fn f32_rounded_sums(
    images: &[Images],
    path: &str,
    live: u32,
    width: usize,
    capacity: u32,
) -> LabResult<Vec<u16>> {
    let elements = live as usize * width;
    let inputs = images
        .iter()
        .map(|images| Ok(images.f32s(path)?[..elements].to_vec()))
        .collect::<LabResult<Vec<_>>>()?;
    let mut sums = match elements {
        0 => Vec::new(),
        _ => kernel_core::oracle::all_reduce_rounded(&inputs).map_err(|e| e.0.to_string())?,
    };
    sums.resize(capacity as usize * width, 0);
    Ok(sums)
}

/// The MoE sublayers' rounded all-reduce, asked what the bundle asks,
/// required: through the phased F32 entries where the admission has them,
/// else composed.
fn rounded_required<'a>(phased: bool) -> RequiredCollective<'a> {
    let width = u64::from(Shape::TP4.profile.hidden);
    let name = match phased {
        true => "rounded all-reduce through the phased F32 entries",
        false => "composed rounded all-reduce",
    };
    RequiredCollective::new(
        name,
        every_phase_need(COMPOSED_ROWS),
        move |backend, ranks, rank, need| {
            let query = kernel_core::contract::CommQuery::AllReduceRounded {
                ranks,
                elements: width,
            };
            backend.all_reduce(&query, rank, need)
        },
        rounded_case,
    )
    .required()
}

/// The composed all-reduce, asked what the bundle asks, required.
fn composed_required<'a>() -> RequiredCollective<'a> {
    let width = u64::from(Shape::TP4.profile.hidden);
    RequiredCollective::new(
        "composed BF16-partials all-reduce",
        every_phase_need(COMPOSED_ROWS),
        move |backend, ranks, rank, need| {
            let query = kernel_core::contract::CommQuery::AllReduce {
                ranks,
                elements: width,
                dtype: DType::Bf16,
                fused_residual: false,
                fused_rmsnorm: None,
            };
            backend.all_reduce(&query, rank, need)
        },
        composed_case,
    )
    .required()
}

/// The comm epilogue's ending sublayer's `post` and `comb`, which the op
/// reads and then overwrites with the next mixer's: every element 0.25,
/// the fill each launch's buffers start from.
const ENDING_MIX: [u8; 2] = [0x80, 0x3e];

/// The comm epilogue of the bundle's comm backend (with or without the
/// phased entries) over [`EPILOGUE_ROWS`] rows on every rank: each rank's
/// finite partials, the same streams on every rank, the next mixer with
/// saturated collapse weights (`scale[0]` zero, `base` 64 on even streams
/// and -200 on odd ones, as the kit's mix-collapse and norm case) and a
/// BF16 norm weight. The oracle gives the rounded sums, the reduced rows,
/// the expanded streams and the collapsed and normed rows as bits (the
/// contract's sums, `expand`, `collapse` and `epilogue_norm` of the query
/// the case prepares, the HF norm with its squares summed in the order the
/// query names) and the next `post` and `comb` within the mix-collapse's
/// gate.
fn epilogue_case<'a>(
    ranks: u32,
    ops: Vec<Prepared<kernel_core::sinkhorn_hyper_connection::ReduceExpandMixNormBuffers<View>>>,
) -> LabResult<CollectiveCase<'a>> {
    use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
    use kernel_core::collective::AllReduceBuffers;
    use kernel_core::sinkhorn_hyper_connection::{
        MixScratch, MixerWeights, ReduceExpandMixNormBuffers, SinkhornHyperConnectionQuery as Query,
    };
    use kernel_core::sinkhorn_hyper_connection_oracle as hc;
    use kernel_lab::kit::cases::collectives::finite_partials;
    let entry = bundle_engines(Shape::TP4).comm_epilogue;
    let fused = epilogue_query(ranks);
    let mix = entry.mix;
    let (n, h, mixes) = (
        mix.streams as usize,
        mix.width as usize,
        mix.mixes() as usize,
    );
    let rows = EPILOGUE_ROWS as usize;
    let info = ops.first().ok_or("no rank's op")?.info().clone();
    let state = info.state.clone();
    let streams = kit::bf16_values(32, rows * n * h);
    let projection = kit::bf16_values(31, mixes * n * h)
        .into_iter()
        .map(|b| f32_to_bf16(bf16_to_f32(b) / 128.0))
        .collect::<Vec<_>>();
    let base = (0..mixes)
        .map(|m| match m {
            m if m < n && m % 2 == 0 => 64.0_f32,
            m if m < n => -200.0,
            m => (m % 5) as f32 * 0.25 - 0.5,
        })
        .collect::<Vec<_>>();
    let scale = [0.0_f32, 0.25, 1.0];
    let weight = kit::bf16_values(38, h);
    let f32s = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>()
    };
    let scratch = |path: &str, zeroed: bool| {
        info.bytes_of(path).map(|bytes| {
            Slot::Own(match zeroed {
                true => Spec::zeroed(bytes as usize),
                false => Spec::scratch(bytes as usize),
            })
        })
    };
    let records = (0..ranks)
        .map(|me| ReduceExpandMixNormBuffers {
            reduce: AllReduceBuffers {
                input: Slot::Own(Spec::input(
                    DType::F32,
                    (Rows::Live, h),
                    f32s(&finite_partials(me, ranks, rows * h)),
                )),
                output: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, h))),
                live_rows: Some(Slot::peer(me, "live_rows")),
                ranks: reduce_rank_views(ranks, me, "reduce.", &state),
            },
            streams: Slot::Own(Spec::input(
                DType::Bf16,
                (Rows::Live, n * h),
                kit::bf16_bytes(&streams),
            )),
            reduced: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, h))),
            expanded: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, n * h))),
            mixer: MixerWeights {
                fn_: Slot::Own(Spec::input(
                    DType::Bf16,
                    (Rows::Fixed(mixes), n * h),
                    kit::bf16_bytes(&projection),
                )),
                base: Slot::Own(Spec::input(
                    DType::F32,
                    (Rows::Fixed(1), mixes),
                    f32s(&base),
                )),
                scale: Slot::Own(Spec::input(DType::F32, (Rows::Fixed(1), 3), f32s(&scale))),
            },
            post: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, n)).filled_with(&ENDING_MIX)),
            comb: Slot::Own(
                Spec::output(DType::Bf16, (Rows::Live, n * n)).filled_with(&ENDING_MIX),
            ),
            collapsed: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, h))),
            norm_weight: Slot::Own(Spec::input(
                DType::Bf16,
                (Rows::Fixed(1), h),
                kit::bf16_bytes(&weight),
            )),
            normed: Slot::Own(Spec::output(DType::Bf16, (Rows::Live, h))),
            scratch: MixScratch {
                partials: scratch("scratch.partials", false),
                tickets: scratch("scratch.tickets", true),
            },
            live_rows: Slot::Own(Spec::live()),
        })
        .collect();
    let eps = f32::from_bits(mix.eps_bits);
    let ending = u16::from_le_bytes(ENDING_MIX);
    let pad = move |mut bits: Vec<u16>, width: usize| {
        bits.resize(rows * width, 0);
        kit::bf16_bytes(&bits)
    };
    let padded = move |output: kernel_core::oracle::OracleOutput, width: usize| {
        let mut output = output;
        let elements = rows * width;
        output.values.resize(elements, 0.0);
        output.sum_abs_products.resize(elements, 0.0);
        output.reduction_lengths.resize(elements, 1);
        output
    };
    Ok(CollectiveCase::new(ops, EPILOGUE_ROWS, records)?
        .oracle(move |images, live| {
            let l = live as usize;
            let sums = rounded_sums(images, "reduce.input", live, h, EPILOGUE_ROWS)?;
            let reduced = sums[..l * h].to_vec();
            let x = images[0].u16s("streams")?[..l * n * h].to_vec();
            let expand = Query::Expand {
                streams: mix.streams,
                width: mix.width,
                capacity: EPILOGUE_ROWS,
            };
            let expanded = match l {
                0 => Vec::new(),
                _ => hc::expand(
                    expand,
                    &x,
                    &reduced,
                    &vec![ending; l * n],
                    &vec![ending; l * n * n],
                )
                .map_err(|e| e.0.to_string())?,
            };
            let query = Query::MixCollapse {
                mix,
                capacity: EPILOGUE_ROWS,
            };
            let (collapsed, post, comb) = match l {
                0 => (Vec::new(), None, None),
                _ => {
                    let widened = expanded
                        .iter()
                        .map(|&b| f64::from(bf16_to_f32(b)))
                        .collect::<Vec<_>>();
                    let projection64 = projection
                        .iter()
                        .map(|&b| f64::from(bf16_to_f32(b)))
                        .collect::<Vec<_>>();
                    let base64 = base.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
                    let scale64 = scale.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
                    let output = hc::mix_collapse(
                        query,
                        &widened,
                        hc::MixWeights {
                            projection: &projection64,
                            base: &base64,
                            scale: &scale64,
                        },
                    )
                    .map_err(|e| e.0.to_string())?;
                    let pre = (0..l * n)
                        .map(|i| if i % n % 2 == 0 { 1.0 + eps } else { eps })
                        .collect::<Vec<_>>();
                    let collapsed =
                        hc::collapse(query, &expanded, &pre).map_err(|e| e.0.to_string())?;
                    (collapsed, Some(output.post), Some(output.comb))
                }
            };
            let normed = match l {
                0 => Vec::new(),
                _ => hc::epilogue_norm(fused, &collapsed, &weight).map_err(|e| e.0.to_string())?,
            };
            let none = || kernel_core::oracle::OracleOutput {
                values: Vec::new(),
                sum_abs_products: Vec::new(),
                reduction_lengths: Vec::new(),
            };
            let post = padded(post.unwrap_or_else(none), n);
            let comb = padded(comb.unwrap_or_else(none), n * n);
            let expected = [
                ("reduce.output", kit::bf16_bytes(&sums)),
                ("reduced", pad(reduced, h)),
                ("expanded", pad(expanded, n * h)),
                ("collapsed", pad(collapsed, h)),
                ("normed", pad(normed, h)),
            ];
            Ok((0..images.len())
                .map(|_| {
                    let mut rank = expected
                        .iter()
                        .map(|(name, bits)| (name.to_string(), Expected::Bits(bits.clone())))
                        .collect::<Vec<_>>();
                    rank.push(("post".to_owned(), Expected::Values(post.clone())));
                    rank.push(("comb".to_owned(), Expected::Values(comb.clone())));
                    rank
                })
                .collect())
        })
        .same_on_every_rank(&[
            "reduce.output",
            "reduced",
            "expanded",
            "post",
            "comb",
            "collapsed",
            "normed",
        ]))
}

/// The comm epilogue with or without the phased entries, asked what the
/// bundle asks, required.
/// The comm epilogue query the bundle asks over `ranks` ranks: its entry's
/// mixer and eps, its input norm summed in the order the bundle's
/// hidden-width norms sum (`Shape::rmsnorm_requirement`, the 256-lane
/// tree).
fn epilogue_query(
    ranks: u32,
) -> kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery {
    let shape = Shape::TP4;
    let entry = bundle_engines(shape).comm_epilogue;
    let norm = shape
        .rmsnorm_requirement(shape.profile.hidden)
        .numerics
        .expect("the hidden-width norm pins its numerics")
        .reduction;
    kernel_core::sinkhorn_hyper_connection::SinkhornHyperConnectionQuery::AllReduceExpandMixNorm {
        ranks,
        dtype: DType::Bf16,
        mix: entry.mix,
        rmsnorm_eps_bits: entry.rmsnorm_eps_bits,
        norm,
        capacity: entry.capacity,
    }
}

fn epilogue_required<'a>(phased: bool) -> RequiredCollective<'a> {
    let name = match phased {
        true => "comm epilogue after the phased all-reduce",
        false => "comm epilogue after the composed all-reduce",
    };
    RequiredCollective::new(
        name,
        every_phase_need(EPILOGUE_ROWS),
        move |backend, ranks, rank, need| {
            backend.sinkhorn_reduce_expand_mix_norm(&epilogue_query(ranks), rank, need)
        },
        epilogue_case,
    )
    .required()
}

/// The all-reduce cases on `backend`'s admission: the composed all-reduce
/// without the phased entries, the comm epilogue and the MoE sublayers'
/// rounded all-reduce either way.
fn reduce_required<'a>(phased: bool) -> Vec<RequiredCollective<'a>> {
    match phased {
        true => vec![epilogue_required(true), rounded_required(true)],
        false => vec![
            composed_required(),
            epilogue_required(false),
            rounded_required(false),
        ],
    }
}

/// Whether `report` prepared every op, declared it and, with a GPU, passed
/// conformance and every property its op claims.
fn check_reduces(report: &kit::Report, gpu: bool) -> LabResult<()> {
    let failures = report.failures();
    if !failures.is_empty() {
        return Err(format!("{failures:#?}"));
    }
    for op in &report.ops {
        let OpResult::Prepared { checks, .. } = &op.result else {
            return Err(format!("{}: {:?}", op.name, op.result));
        };
        let passed = |name: &str| {
            checks
                .iter()
                .any(|check| check.name == name && matches!(check.outcome, Outcome::Passed(_)))
        };
        let mut wanted = vec!["declaration".to_owned(), "need".to_owned()];
        if gpu {
            wanted.push("conformance".into());
            wanted.extend(
                [Property::Rows, Property::Chunk, Property::Batch].map(kit::properties::name),
            );
        }
        for name in wanted {
            if !passed(&name) {
                return Err(format!("{}: {name} did not pass: {checks:?}", op.name));
            }
        }
    }
    Ok(())
}

/// The bundle's composed all-reduce, comm epilogues and rounded
/// all-reduces prepare on two ranks for what the bundle asks and declare.
#[test]
fn bundle_reduces_declare_on_two_ranks() {
    for phased in [false, true] {
        let backend = reduce_backend(phased);
        let report = kit::collective::conform(
            backend.as_ref(),
            &Embedded,
            PAIR,
            None,
            &reduce_required(phased),
        )
        .expect("the reduces' cases");
        assert_eq!(report.ops.len(), 3 - usize::from(phased));
        check_reduces(&report, false).expect("declared");
    }
}

/// On two GPUs, the bundle's composed all-reduce (every row count of the
/// BF16-partials table its own form), its comm epilogue after the composed
/// and after the phased all-reduce, and the MoE sublayers' rounded
/// all-reduce composed and through the phased F32 entries pass
/// conformance, poisoned and replayed, with equal bits on both ranks, and
/// keep every claimed property over both ranks' rows: rows, chunk and
/// batch.
#[test]
fn bundle_reduces_keep_their_claims_on_two_gpus() {
    gpu_test_multi(1 << 30, |sessions| {
        if sessions.len() != PAIR as usize {
            return kernel_lab::skip("the reduce cases run on two GPUs");
        }
        let mut reports = BTreeMap::new();
        for phased in [false, true] {
            let backend = reduce_backend(phased);
            let report = kit::collective::conform(
                backend.as_ref(),
                &Embedded,
                PAIR,
                Some(sessions),
                &reduce_required(phased),
            )?;
            check_reduces(&report, true)?;
            reports.insert(format!("phased {phased}"), report.json());
        }
        sessions[0].receipt(
            "conformance-kit-bundle-reduces",
            BTreeMap::from([("reports".to_owned(), json!(reports))]),
        )?;
        Ok(())
    })
    .expect("the bundle's reduces on two ranks");
}
