//! The F32 routed sum that adds a row (`chained::RoutedSum::add_entry`,
//! routed_sum_add.cuh) at the split's route sum (`chained::split::sums`,
//! every expert the rank's), from the lab's `moe_chained_split_sum_add`
//! cubin.
//!
//! Each case's output must equal, bit for bit, the host's F32 sum of the
//! token's local routes in top-k order from +0, then plus the addend: the
//! routed sum's adds followed by the add a shared expert's residual
//! epilogue makes (one add of the same two values, which commutes). Cases
//! cover every rank (the split's one), every live count of the decode
//! buckets and edge live counts of the wide verify buckets, tokens with no
//! route to sum (every route a hole), negative
//! zeros, wide exponent ranges, and tokens whose local routes are all -0 or
//! cancel in pairs (`x + -x`) under a -0 addend: the routed sum starts from
//! +0 and `+0 + -0` and `x + -x` are +0 under round-to-nearest, so it is
//! never -0 and those rows are +0 (which the shared expert's zero residual
//! relies on: `acc + 0` differs from `acc` only for `acc = -0`, and then
//! only a -0 routed sum could tell). Tokens with infinities and NaNs (of
//! either sign and other payloads) in their routes and addend must give
//! NaN where the host does and the host's bits elsewhere. Every output is poisoned through eager,
//! relaunch, graph replay and a PDL producer that publishes the addend late
//! (the buffer holds the addend's complement until then), and rows past the
//! live count must keep their poison.
//!
//! Each deliberate defect of `SumAddMutation` must change some output: the
//! addend dropped, the addend starting the sum, and no dependency wait
//! (behind the producer).
//!
//! Run with `CHOPINFER_KERNEL_GPU=<ordinal> cargo test -p kernel-lab
//! --release --test routed_sum_add`; without a GPU the test skips.

use kernel_core::contract::DType;
use kernel_lab::families::moe_chained_split_sum_add as shapes;
use kernel_lab::latent_shapes;
use kernel_lab::poison::{self, CopyProducer, Replay, WriteMap};
use kernel_lab::{LabResult, Session, gpu_test, launch};
use runtime_cuda::{DeviceBuffer, Function, KernelArgument, Stream};
use serde_json::json;
use std::collections::BTreeMap;

/// The ranks of the split's shape: its rank holds every expert.
const RANKS: u32 = 1;

/// Bucket capacities and the live counts run at each: every count of the
/// decode buckets, edges of the wide verify buckets.
fn cases() -> Vec<(u32, Vec<u32>)> {
    let mut cases: Vec<(u32, Vec<u32>)> = [1, 2, 4, 8, 16, 32]
        .into_iter()
        .map(|capacity| (capacity, (0..=capacity).collect()))
        .collect();
    for capacity in [64, 96, 128, 192] {
        cases.push((capacity, vec![0, 1, capacity / 2, capacity - 1, capacity]));
    }
    cases
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }

    /// Finite F32 values of both signs over exponents -24 through 7, one in
    /// eight a signed zero.
    fn value(&mut self) -> f32 {
        let bits = self.next();
        if bits % 8 == 0 {
            return if bits & 8 != 0 { -0.0 } else { 0.0 };
        }
        let exponent = 103 + (bits >> 3) % 32;
        f32::from_bits((bits & 0x8000_0000) | (exponent << 23) | (self.next() & 0x007f_ffff))
    }

    /// One in four an infinity or a NaN of either sign (quiet, with a
    /// payload), else [`Self::value`].
    fn special(&mut self) -> f32 {
        let bits = self.next();
        match bits % 16 {
            0 => f32::INFINITY,
            1 => f32::NEG_INFINITY,
            2 => f32::from_bits(0x7fc0_0123),
            3 => f32::from_bits(0xffc0_0456),
            _ => self.value(),
        }
    }
}

/// What a token's routes and addend hold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Class {
    /// No route to sum: every expert is the rank's, so every route is a
    /// hole (-1).
    NoLocal,
    /// The last route padded with -1.
    Padded,
    /// Every route local and every routed value -0; the addend -0 in every
    /// other column.
    NegativeZeros,
    /// Every route local, in pairs that cancel; the addend -0 in every
    /// other column.
    Cancelling,
    /// Infinities and NaNs among finite routed values and addends.
    Special,
    Random,
}

impl Class {
    fn of(token: usize) -> Self {
        [
            Self::NoLocal,
            Self::Padded,
            Self::NegativeZeros,
            Self::Cancelling,
            Self::Special,
            Self::Random,
        ][token % 6]
    }
}

/// One case's host inputs.
struct Inputs {
    capacity: u32,
    rank: u32,
    ids: Vec<i32>,
    routed: Vec<f32>,
    addend: Vec<f32>,
}

impl Inputs {
    /// Routes over every rank's experts and routed rows and addends of
    /// [`Rng::value`], by token class ([`Class::of`]).
    fn draw(capacity: u32, rank: u32, rng: &mut Rng) -> Self {
        let sum = shapes::sum();
        let (top_k, width) = (sum.top_k as usize, sum.width as usize);
        let tokens = capacity as usize;
        let experts = sum.local_experts * RANKS;
        let local =
            |rng: &mut Rng| (rank * sum.local_experts + rng.next() % sum.local_experts) as i32;
        let mut ids = Vec::with_capacity(tokens * top_k);
        let mut routed: Vec<f32> = Vec::with_capacity(tokens * top_k * width);
        let mut addend: Vec<f32> = Vec::with_capacity(tokens * width);
        for token in 0..tokens {
            let class = Class::of(token);
            for k in 0..top_k {
                ids.push(match class {
                    Class::NoLocal => -1,
                    Class::Padded if k == top_k - 1 => -1,
                    Class::NegativeZeros | Class::Cancelling => local(rng),
                    _ => (rng.next() % experts) as i32,
                });
            }
            for k in 0..top_k {
                for _ in 0..width {
                    routed.push(match class {
                        Class::NegativeZeros => -0.0,
                        // Route 2j adds x, route 2j + 1 adds -x: the sum
                        // returns to +0 after every pair.
                        Class::Cancelling if k % 2 == 1 => -routed[routed.len() - width],
                        Class::Cancelling => rng.value(),
                        Class::Special => rng.special(),
                        _ => rng.value(),
                    });
                }
            }
            for column in 0..width {
                addend.push(match class {
                    Class::NegativeZeros | Class::Cancelling if column % 2 == 0 => -0.0,
                    Class::Special => rng.special(),
                    _ => rng.value(),
                });
            }
        }
        Self {
            capacity,
            rank,
            ids,
            routed,
            addend,
        }
    }

    /// The host's rows for the first `live` tokens: the local routes summed
    /// in top-k order from +0, then plus the addend (`addend_first` starts
    /// from the addend instead and adds nothing after).
    fn expected(&self, live: u32, with_addend: bool, addend_first: bool) -> Vec<f32> {
        let sum = shapes::sum();
        let (top_k, width) = (sum.top_k as usize, sum.width as usize);
        let mut out = Vec::with_capacity(live as usize * width);
        for token in 0..live as usize {
            for column in 0..width {
                let addend = self.addend[token * width + column];
                let mut acc = if addend_first { addend } else { 0.0_f32 };
                for k in 0..top_k {
                    let id = self.ids[token * top_k + k];
                    if id >= 0 && id as u32 / sum.local_experts == self.rank {
                        acc += self.routed[(token * top_k + k) * width + column];
                    }
                }
                if with_addend && !addend_first {
                    acc += addend;
                }
                out.push(acc);
            }
        }
        out
    }
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_ne_bytes()).collect()
}

fn upload(session: &Session, data: &[u8]) -> LabResult<DeviceBuffer> {
    let buffer = session.allocate(data.len().max(4))?;
    buffer
        .copy_from_bytes(data, &session.stream)
        .map_err(|e| e.to_string())?;
    session.stream.synchronize().map_err(|e| e.to_string())?;
    Ok(buffer)
}

/// The first output word against `expected` that differs in its bits, or
/// is a NaN where the host's is not or the other way round (NaN payloads
/// follow the device's rules, not the host's), if any.
fn first_difference(actual: &[u8], expected: &[f32]) -> Option<(usize, u32, u32)> {
    actual
        .chunks_exact(4)
        .zip(expected)
        .position(|(a, e)| {
            let a = f32::from_ne_bytes(a.try_into().unwrap());
            if a.is_nan() || e.is_nan() {
                a.is_nan() != e.is_nan()
            } else {
                a.to_bits() != e.to_bits()
            }
        })
        .map(|at| {
            (
                at,
                u32::from_ne_bytes(actual[4 * at..4 * at + 4].try_into().unwrap()),
                expected[at].to_bits(),
            )
        })
}

/// One case's device buffers.
struct Device {
    routed: DeviceBuffer,
    ids: DeviceBuffer,
    live: DeviceBuffer,
    staged: DeviceBuffer,
    addend: DeviceBuffer,
    stale: Vec<u8>,
    output: DeviceBuffer,
}

impl Device {
    fn new(session: &Session, inputs: &Inputs, live: u32) -> LabResult<Self> {
        let width = shapes::sum().width as usize;
        let ids = inputs
            .ids
            .iter()
            .flat_map(|id| id.to_ne_bytes())
            .collect::<Vec<_>>();
        let addend = bytes(&inputs.addend);
        // The addend buffer holds the complement until a producer copies the
        // staged addend in, so a sum that reads it early differs.
        let stale = addend.iter().map(|b| !b).collect::<Vec<_>>();
        Ok(Self {
            routed: upload(session, &bytes(&inputs.routed))?,
            ids: upload(session, &ids)?,
            live: upload(session, &live.to_ne_bytes())?,
            staged: upload(session, &addend)?,
            addend: upload(session, &stale)?,
            stale,
            output: session.allocate(inputs.capacity as usize * width * 4)?,
        })
    }

    fn run(
        &self,
        function: &Function,
        stream: &Stream,
        inputs: &Inputs,
        pdl: bool,
    ) -> LabResult<()> {
        let plan = shapes::sum()
            .plan(inputs.capacity)
            .map_err(|r| r.to_string())?;
        function
            .launch(
                stream,
                launch::config(kernel_core::contract::LaunchPlan { pdl, ..plan })?,
                &[
                    KernelArgument::Buffer(&self.routed),
                    KernelArgument::Buffer(&self.ids),
                    KernelArgument::Buffer(&self.live),
                    KernelArgument::U32(inputs.rank),
                    KernelArgument::Buffer(&self.addend),
                    KernelArgument::Buffer(&self.output),
                ],
            )
            .map_err(|e| e.to_string())
    }
}

#[test]
fn routed_sum_add_equals_the_sum_then_the_add_poisoned() {
    gpu_test(1 << 30, |session| {
        let module = session.module(shapes::CUBIN)?;
        let functions = shapes::MUTATIONS
            .iter()
            .map(|mutation| {
                module
                    .function(&shapes::symbol(mutation))
                    .map_err(|e| e.to_string())
            })
            .collect::<LabResult<Vec<_>>>()?;
        let lab = session.module(&latent_shapes::cubin_name())?;
        let producer = CopyProducer::new(&lab)?;
        let pdl = std::env::var("CHOPINFER_LATENT_NO_PDL").as_deref() != Ok("1");
        let stream = &session.stream;
        let width = shapes::sum().width as usize;
        let mut rng = Rng(0x5a3d_d0e1);
        let mut evidence = Vec::new();
        // Elements checked +0 from a zero routed sum and a -0 addend.
        let mut zero_rows = 0_u64;
        // Per mutation: cases run and cases whose output differed.
        let mut caught = BTreeMap::<&str, (u32, u32)>::new();
        for (capacity, lives) in cases() {
            for rank in 0..RANKS {
                let inputs = Inputs::draw(capacity, rank, &mut rng);
                for &live in &lives {
                    let label = format!("capacity {capacity} rank {rank} live {live}");
                    let device = Device::new(session, &inputs, live)?;
                    let served = &functions[0];
                    let chain = |stream: &Stream, function: &Function, pdl: bool| {
                        producer.copy(stream, &device.staged, &device.addend)?;
                        device.run(function, stream, &inputs, pdl)
                    };
                    let graph = session.capture(|stream| chain(stream, served, pdl))?;
                    let map = WriteMap::of(DType::F32).prefix(live as usize * width)?;
                    // The addend is the producer's: every run starts from
                    // its complement.
                    let stale = device.stale.clone();
                    let addend_bytes = 0..stale.len();
                    let mut runs = Replay::new(session)
                        .output("output", &device.output, map)
                        .state(
                            "addend",
                            &device.addend,
                            stale,
                            WriteMap::of(DType::U8).none()?,
                            vec![addend_bytes],
                        );
                    runs.eager(|| chain(stream, served, false))
                        .map_err(|e| format!("{label}: {e}"))?;
                    let expected = inputs.expected(live, true, false);
                    let eager = runs.expected("output").ok_or("no eager output")?.to_vec();
                    if let Some((at, got, want)) = first_difference(&eager, &expected) {
                        return Err(format!(
                            "{label}: element {at} is {got:#010x}, the host's sum then add \
                             {want:#010x}"
                        ));
                    }
                    // The routed sum of the -0 and cancelling tokens is +0,
                    // never -0: under a -0 addend their rows are +0.
                    for token in (0..live as usize).filter(|&token| {
                        matches!(Class::of(token), Class::NegativeZeros | Class::Cancelling)
                    }) {
                        for column in (0..width).step_by(2) {
                            let at = 4 * (token * width + column);
                            let bits = u32::from_ne_bytes(eager[at..at + 4].try_into().unwrap());
                            if bits != 0 {
                                return Err(format!(
                                    "{label}: token {token} column {column} is {bits:#010x}, \
                                     not +0"
                                ));
                            }
                            zero_rows += 1;
                        }
                    }
                    runs.relaunch(|| chain(stream, served, false))
                        .map_err(|e| format!("{label}: {e}"))?;
                    runs.graph(&graph).map_err(|e| format!("{label}: {e}"))?;
                    runs.pdl(
                        || producer.copy(stream, &device.staged, &device.addend),
                        || device.run(served, stream, &inputs, pdl),
                        pdl,
                    )
                    .map_err(|e| format!("{label}: {e}"))?;
                    // The mutants, each behind the producer from the stale
                    // addend and a poisoned output: any byte that differs
                    // from the served eager output catches it; a launch
                    // error fails the test.
                    for (mutation, function) in shapes::MUTATIONS.iter().zip(&functions).skip(1) {
                        poison::write(session, &device.addend, 0, &device.stale)?;
                        poison::poison(session, &[&device.output])?;
                        producer.copy(stream, &device.staged, &device.addend)?;
                        device
                            .run(function, stream, &inputs, pdl)
                            .map_err(|e| format!("{label} {mutation}: {e}"))?;
                        stream
                            .synchronize()
                            .map_err(|e| format!("{label} {mutation}: {e}"))?;
                        let differs = poison::read(session, &device.output)? != eager;
                        let entry = caught.entry(mutation).or_default();
                        entry.0 += 1;
                        entry.1 += u32::from(differs);
                    }
                    if live == capacity || live == 0 {
                        let mut receipt = BTreeMap::new();
                        runs.record(&mut receipt);
                        receipt.insert("capacity".into(), json!(capacity));
                        receipt.insert("rank".into(), json!(rank));
                        receipt.insert("live".into(), json!(live));
                        evidence.push(json!(receipt));
                    }
                }
            }
        }
        if zero_rows == 0 {
            return Err("no case checked a zero routed sum".into());
        }
        for mutation in &shapes::MUTATIONS[1..] {
            let (run, differed) = caught.get(mutation).copied().unwrap_or_default();
            if differed == 0 {
                return Err(format!(
                    "mutant {mutation} matched the served rows in all {run} cases"
                ));
            }
        }
        session.receipt(
            "routed_sum_add_poisoned",
            BTreeMap::from([
                ("cases".to_owned(), json!(evidence)),
                ("positive_zero_elements".to_owned(), json!(zero_rows)),
                (
                    "mutants".to_owned(),
                    json!(
                        caught
                            .iter()
                            .map(|(m, (run, differed))| (
                                m.to_string(),
                                json!({"run": run, "differed": differed})
                            ))
                            .collect::<BTreeMap<_, _>>()
                    ),
                ),
            ]),
        )?;
        Ok(())
    })
    .expect("routed sum add");
}

/// The host reference's order matters: on these inputs, starting from the
/// addend or dropping it changes some row, so the mutants are observable
/// without a GPU too.
#[test]
fn the_reference_separates_the_mutants() {
    let mut rng = Rng(0x5a3d_d0e1);
    let inputs = Inputs::draw(8, 0, &mut rng);
    let served = inputs.expected(8, true, false);
    for (with_addend, addend_first) in [(false, false), (true, true)] {
        let other = inputs.expected(8, with_addend, addend_first);
        assert!(
            served
                .iter()
                .zip(&other)
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "with_addend {with_addend} addend_first {addend_first}"
        );
    }
}

/// The zero residual's argument on the host: adding +0 to the shared
/// expert's sum and then the routed sum gives today's bits (the routed sum
/// added to the shared sum) for every pair of values the routed sum can
/// take, specials included; only a -0 routed sum after a -0 shared sum
/// would differ, and a routed sum from +0 is never -0.
#[test]
fn a_zero_residual_then_the_routed_sum_keeps_the_bits() {
    let mut rng = Rng(0x0dd5_0e70);
    let mut values = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    values.extend((0..64).map(|_| rng.special()));
    let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    for &shared in &values {
        for &routed in &values {
            let today = shared + routed;
            let served = routed + (shared + 0.0);
            assert_eq!(
                same(today, served),
                !(shared.to_bits() == 0x8000_0000 && routed.to_bits() == 0x8000_0000),
                "shared {shared:e} routed {routed:e}"
            );
        }
    }
    // The routed sum from +0 over local routes of -0 values or cancelling
    // pairs is +0.
    for routes in [
        vec![-0.0_f32; 8],
        vec![1.5, -1.5, -0.0, 0.0, f32::MAX, -f32::MAX, 2e-45, -2e-45],
    ] {
        let sum = routes.iter().fold(0.0_f32, |acc, &v| acc + v);
        assert_eq!(sum.to_bits(), 0, "{routes:?}");
    }
    let inputs = Inputs::draw(12, 0, &mut rng);
    let sums = inputs.expected(12, false, false);
    let width = shapes::sum().width as usize;
    for token in 0..12 {
        if matches!(Class::of(token), Class::NegativeZeros | Class::Cancelling) {
            assert!(
                sums[token * width..(token + 1) * width]
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
    }
}
