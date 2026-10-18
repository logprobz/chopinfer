//! One real ring survives both prefill steps; only metadata is uploaded again.
use kernel_core::backend::{Backend, Counts, Need, PreparedOp, View};
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::DType;
use kernel_core::latent_attention::{
    LatentSourcesBuffers, LatentWindowBuffers, LatentWindowStorage,
};
use kernel_core::latent_attention_oracle::{
    EncodedLatent, LatentAttentionResult, LatentKeyLists, LatentSourcesInput, LatentWindowInput,
    latent_attention_with_sources,
};
use kernel_core::latent_glue::{
    LatentWindowKeyBuffers, LatentWindowKeys, LatentWindowWrite, LatentWindowWriteBuffers,
    WindowAddressing,
};
use kernel_core::latent_staged_window_keys::{
    self as bounded, LatentStagedWindowKeyBuffers, LatentStagedWindowKeys,
};
use kernel_core::latent_window_stage::{LatentWindowStage, LatentWindowStageBuffers};
use kernel_harness::{
    LabResult, Session,
    kit::{Modules, modules::Lookup},
    launch::{self, Bindings},
};
use runtime_cuda::DeviceBuffer;
use std::collections::BTreeMap;
const D: usize = 512;
const H: usize = 16;
const N: usize = 32;
const RC: usize = 136 * D + 66;
const RS: usize = 136 * 16 + 23;
const SC: usize = 256 * D + 34;
const SS: usize = 256 * 16 + 11;
const KQ: LatentStagedWindowKeys = LatentStagedWindowKeys {
    window: 128,
    ring_slots: 136,
    staged_slots: 256,
};
fn ints(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn halves(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn input(first: i32, rows: usize) -> Vec<f32> {
    (first..first + rows as i32)
        .flat_map(|p| {
            (0..D).map(move |col| {
                ((p as usize * 17 + col * 7) % 61) as f32 / 16.0 - 1.875 + (p % 7) as f32 / 32.0
            })
        })
        .collect()
}
fn weights() -> Vec<u16> {
    (0..D)
        .map(|i| f32_to_bf16(1.0 + (i % 9) as f32 / 16.0))
        .collect()
}
fn query(addressing: WindowAddressing) -> LatentWindowWrite {
    LatentWindowWrite {
        latent: 512,
        rotary: 64,
        epsilon_bits: 1e-20_f32.to_bits(),
        addressing,
    }
}
fn history(last: i32) -> Vec<EncodedLatent> {
    kernel_core::latent_glue_oracle::window_rows(
        query(WindowAddressing::Circular { slots: 136 }),
        &input(0, last as usize + 1),
        &weights(),
        &[1.0, 0.0].repeat((last as usize + 1) * 32),
    )
    .unwrap()
    .1
}
fn copy_row(c: &mut [u8], s: &mut [u8], slot: usize, row: &EncodedLatent) {
    c[slot * D..(slot + 1) * D].copy_from_slice(&row.values);
    s[slot * 16..(slot + 1) * 16].copy_from_slice(&row.scale_bytes);
}
struct Device<'a> {
    session: &'a Session,
    lookup: &'a Lookup<'a>,
    buffers: Vec<DeviceBuffer>,
}
impl<'a> Device<'a> {
    fn add(&mut self, data: &[u8]) -> LabResult<View> {
        let b = self.session.allocate(data.len().max(1))?;
        b.copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())?;
        let v = View {
            handle: self.buffers.len() as u64 + 1,
            bytes: data.len() as u64,
        };
        self.buffers.push(b);
        Ok(v)
    }
    fn fill(&mut self, n: usize, byte: u8) -> LabResult<View> {
        self.add(&vec![byte; n.max(1)])
    }
    fn upload(&self, v: View, data: &[u8]) -> LabResult<()> {
        self.buffers[v.handle as usize - 1]
            .copy_from_bytes(data, &self.session.stream)
            .map_err(|e| e.to_string())
    }
    fn read(&self, v: View) -> LabResult<Vec<u8>> {
        let mut out = vec![0; v.bytes as usize];
        self.buffers[v.handle as usize - 1]
            .copy_to_bytes(&mut out, &self.session.stream)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }
    fn poison(&self, v: View) -> LabResult<()> {
        self.upload(v, &vec![0xff; v.bytes as usize])
    }
    fn call<B>(&self, op: &dyn PreparedOp<B>, b: &B, workspace: View, rows: u32) -> LabResult<()> {
        let bindings = Bindings {
            buffers: self
                .buffers
                .iter()
                .enumerate()
                .map(|(i, b)| (i as u64 + 1, b))
                .collect(),
        };
        for l in op
            .launches(b, workspace, Counts::rows(rows))
            .map_err(|e| e.to_string())?
        {
            let f = (self.lookup)(&l.symbol)?;
            if l.plan.dynamic_shared_bytes >= 48 * 1024 {
                f.set_max_dynamic_shared_memory(l.plan.dynamic_shared_bytes)
                    .map_err(|e| e.to_string())?;
            }
            launch::launch(
                &self.session.context,
                &f,
                &self.session.stream,
                l.plan,
                &l.arguments,
                &bindings,
            )?;
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
enum Fault {
    None,
    SkipFirstCircularWrite,
    UnboundedLists,
}
fn equal(label: &str, got: &[u8], expected: &[u8]) -> LabResult<()> {
    if got != expected {
        return Err(format!("{label}: bytes differ"));
    }
    Ok(())
}
fn execute(
    session: &Session,
    modules: &super::FamilyModules,
    hole_case: bool,
    fault: Fault,
) -> LabResult<()> {
    let lookup = modules.load(session)?;
    let mut d = Device {
        session,
        lookup: &*lookup,
        buffers: Vec::new(),
    };
    let backend = super::backend();
    let gather = backend
        .latent_window_stage(
            &LatentWindowStage {
                latent: 512,
                window: 128,
                ring_slots: 136,
                staged_slots: 256,
            },
            Need::rows(1),
        )
        .map_err(|e| e.to_string())?;
    let staged_write = backend
        .latent_window_write(
            &query(WindowAddressing::Staged { slots: 256 }),
            Need::rows(N as u32),
        )
        .map_err(|e| e.to_string())?;
    let circular_write = backend
        .latent_window_write(
            &query(WindowAddressing::Circular { slots: 136 }),
            Need::rows(N as u32),
        )
        .map_err(|e| e.to_string())?;
    let keys = backend
        .latent_staged_window_keys(&KQ, Need::rows(N as u32))
        .map_err(|e| e.to_string())?;
    let unbounded = backend
        .latent_window_keys(
            &LatentWindowKeys {
                window: 128,
                addressing: WindowAddressing::Staged { slots: 256 },
            },
            Need::rows(N as u32),
        )
        .map_err(|e| e.to_string())?;
    let aq = kernel_lab_families::latent_window_chain::query();
    let attention = backend
        .latent_attention_sources(&aq, Need::rows(N as u32))
        .map_err(|e| e.to_string())?;
    let reference = history(200);
    let initial_last = if hole_case { 200 } else { 139 };
    let (mut expected_ring_codes, mut expected_ring_scales) = (vec![0xff; RC], vec![0xff; RS]);
    for p in initial_last - 135..=initial_last {
        copy_row(
            &mut expected_ring_codes,
            &mut expected_ring_scales,
            p as usize % 136,
            &reference[p as usize],
        );
    }
    // These are the only uploads of ring payloads in this run.
    let ring_codes = d.add(&expected_ring_codes)?;
    let ring_scales = d.add(&expected_ring_scales)?;
    let stage_codes = d.fill(SC, 0xff)?;
    let stage_scales = d.fill(SS, 0xff)?;
    let item_slots = d.add(&ints(&[0]))?;
    let positions = d.fill(N * 4, 0x80)?;
    let row_items = d.add(&ints(&[0; N]))?;
    let first = d.add(&ints(&[0]))?;
    let chunk_last = d.add(&ints(&[0]))?;
    let ring_last = d.add(&ints(&[initial_last]))?;
    let stage_first = d.add(&ints(&[0]))?;
    let live = d.add(&(0_u32).to_le_bytes())?;
    let status = d.fill(N * 4, 0)?;
    let live_items = d.add(&(1_u32).to_le_bytes())?;
    let input_view = d.fill(N * D * 4, 0)?;
    let weight = d.add(&halves(&weights()))?;
    let rope = d.add(&floats(&[1.0, 0.0].repeat(N * 32)))?;
    let latent = d.fill(N * D * 4, 0xff)?;
    let quant_codes = d.fill(N * D, 0xff)?;
    let quant_scales = d.fill(N * 16, 0xff)?;
    let lists = d.fill(N * 128 * 4, 0x80)?;
    let q = d.fill(N * H * D * 2, 0)?;
    let sinks = d.fill(H * 4, 0)?;
    let out = d.fill(N * H * D * 2, 0xff)?;
    let scratch = d.fill(
        attention
            .info()
            .bytes
            .iter()
            .find(|(name, _)| name == "workspace")
            .ok_or("attention workspace absent")?
            .1 as usize,
        0xff,
    )?;
    let workspace = d.fill(
        staged_write
            .info()
            .workspace
            .max(circular_write.info().workspace) as usize,
        0xff,
    )?;
    let plane = |codes, scales, code_stride, scale_stride, slots| LatentWindowBuffers {
        codes,
        scales,
        item_slots,
        items: 1,
        sequences: 1,
        code_stride,
        scale_stride,
        storage: LatentWindowStorage::Staged { slots },
        key_lists: lists,
        key_stride: 128,
        budget: 128,
    };
    let ring = plane(ring_codes, ring_scales, RC as u64, RS as u64, 136);
    let staged = plane(stage_codes, stage_scales, SC as u64, SS as u64, 256);
    let steps = if hole_case {
        vec![(8, 1)]
    } else {
        vec![(140, 32), (172, 16)]
    };
    let mut retained = initial_last;
    for (step, (begin, rows)) in steps.into_iter().enumerate() {
        let end = begin + rows as i32 - 1;
        let base = (begin - 127).max(0);
        d.upload(status, &[0; N * 4])?;
        d.upload(first, &ints(&[begin]))?;
        d.upload(chunk_last, &ints(&[end]))?;
        d.upload(ring_last, &ints(&[retained]))?;
        d.upload(stage_first, &ints(&[base]))?;
        let mut pos = vec![-2; N];
        for (i, p) in pos.iter_mut().take(rows).enumerate() {
            *p = begin + i as i32;
        }
        d.upload(positions, &ints(&pos))?;
        d.upload(live, &(rows as u32).to_le_bytes())?;
        let mut x = vec![0.0; N * D];
        x[..rows * D].copy_from_slice(&input(begin, rows));
        d.upload(input_view, &floats(&x))?;
        for v in [
            stage_codes,
            stage_scales,
            latent,
            quant_codes,
            quant_scales,
            out,
            scratch,
        ] {
            d.poison(v)?;
        }
        d.upload(lists, &vec![0x80; N * 128 * 4])?;
        d.call(
            gather.as_ref(),
            &LatentWindowStageBuffers {
                ring,
                staged,
                first_positions: first,
                last_positions: ring_last,
                live_items,
            },
            View::EMPTY,
            1,
        )?;
        let (mut expected_codes, mut expected_scales) = (vec![0xff; SC], vec![0xff; SS]);
        for p in base..begin {
            if p <= retained && p + 136 > retained {
                copy_row(
                    &mut expected_codes,
                    &mut expected_scales,
                    (p - base) as usize,
                    &reference[p as usize],
                );
            }
        }
        let label = if step == 1 {
            "second prefix 45..171"
        } else {
            "first prefix"
        };
        equal(label, &d.read(stage_codes)?, &expected_codes)?;
        equal(label, &d.read(stage_scales)?, &expected_scales)?;
        let write = LatentWindowWriteBuffers {
            status,
            status_row_offset: 0,
            input: input_view,
            weight,
            latent,
            cos_sin: rope,
            positions,
            row_batch: row_items,
            first_positions: Some(stage_first),
            last_positions: chunk_last,
            window: staged,
            quantized_codes: quant_codes,
            quantized_scales: quant_scales,
            live_rows: live,
        };
        d.call(staged_write.as_ref(), &write, workspace, rows as u32)?;
        for p in begin..=end {
            copy_row(
                &mut expected_codes,
                &mut expected_scales,
                (p - base) as usize,
                &reference[p as usize],
            );
        }
        equal("staged write codes", &d.read(stage_codes)?, &expected_codes)?;
        equal(
            "staged write scales",
            &d.read(stage_scales)?,
            &expected_scales,
        )?;
        if fault == Fault::UnboundedLists {
            d.call(
                unbounded.as_ref(),
                &LatentWindowKeyBuffers {
                    positions,
                    row_batch: Some(row_items),
                    first_positions: Some(stage_first),
                    sequences: 1,
                    key_lists: lists,
                    live_rows: live,
                },
                View::EMPTY,
                rows as u32,
            )?;
        } else {
            d.call(
                keys.as_ref(),
                &LatentStagedWindowKeyBuffers {
                    positions,
                    row_items,
                    chunk_first: first,
                    chunk_last,
                    item_slots,
                    ring_last,
                    items: 1,
                    sequences: 1,
                    key_lists: lists,
                    live_rows: live,
                },
                View::EMPTY,
                rows as u32,
            )?;
        }
        let wanted = bounded::keys(
            KQ,
            &pos[..rows],
            &vec![0; rows],
            &[begin],
            &[end],
            &[0],
            &[retained],
        )
        .map_err(|e| e.to_string())?;
        if fault != Fault::UnboundedLists {
            equal(
                "bounded key lists",
                &d.read(lists)?[..rows * 128 * 4],
                &ints(&wanted),
            )?;
        }
        if hole_case {
            assert!(d.read(stage_codes)?[..8 * D].iter().all(|&v| v == 0xff));
            assert!(d.read(stage_scales)?[..8 * 16].iter().all(|&v| v == 0xff));
        }
        d.call(
            attention.as_ref(),
            &LatentSourcesBuffers {
                q,
                row_batch: row_items,
                window: staged,
                compressed: None,
                sinks: Some(sinks),
                cos_sin: Some(rope),
                output: Some(out),
                partial: None,
                workspace: scratch,
                live_rows: live,
            },
            View::EMPTY,
            rows as u32,
        )?;
        let expected = latent_attention_with_sources(
            aq,
            LatentSourcesInput {
                q: &vec![0.0; rows * H * D],
                row_batch: &vec![0; rows],
                window: LatentWindowInput {
                    codes: &expected_codes,
                    scales: &expected_scales,
                    item_slots: &[0],
                    sequences: 1,
                    code_stride: SC,
                    scale_stride: SS,
                    storage: LatentWindowStorage::Staged { slots: 256 },
                    lists: LatentKeyLists {
                        indices: &wanted,
                        stride: 128,
                        budget: 128,
                    },
                },
                compressed: None,
                sinks: Some(&[0.0; H]),
            },
        )
        .map_err(|e| e.to_string())?;
        let LatentAttentionResult::Normalized(reference_attention) = expected else {
            unreachable!()
        };
        let actual = d.read(out)?[..rows * H * D * 2]
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes(b.try_into().unwrap())))
            .collect::<Vec<_>>();
        let mut receipt = BTreeMap::new();
        kernel_harness::check_output(&actual, &reference_attention, DType::Bf16, &mut receipt)
            .map_err(|e| format!("attention consumed an unavailable cell: {e}"))?;
        session.receipt(
            &format!("window-chain-attention-{hole_case}-{step}"),
            receipt,
        )?;
        if !(fault == Fault::SkipFirstCircularWrite && step == 0) {
            let ring_write = LatentWindowWriteBuffers {
                first_positions: None,
                window: ring,
                ..write
            };
            d.call(circular_write.as_ref(), &ring_write, workspace, rows as u32)?;
        }
        for p in begin..=end {
            copy_row(
                &mut expected_ring_codes,
                &mut expected_ring_scales,
                p as usize % 136,
                &reference[p as usize],
            );
        }
        equal("step status", &d.read(status)?, &[0; N * 4])?;
        retained = end;
    }
    equal(
        "final ring codes",
        &d.read(ring_codes)?,
        &expected_ring_codes,
    )?;
    equal(
        "final ring scales",
        &d.read(ring_scales)?,
        &expected_ring_scales,
    )?;
    Ok(())
}
pub(super) fn run(session: &Session, modules: &super::FamilyModules) -> LabResult<()> {
    execute(session, modules, true, Fault::None)?;
    let bad = execute(session, modules, true, Fault::UnboundedLists)
        .expect_err("unbounded lists read poisoned prefix cells");
    assert!(
        bad.contains("attention consumed an unavailable cell")
            && !bad.contains("CUDA")
            && !bad.contains("trap"),
        "{bad}"
    );
    execute(session, modules, false, Fault::None)?;
    let omitted = execute(session, modules, false, Fault::SkipFirstCircularWrite)
        .expect_err("circular write omission must break the next gather");
    assert!(omitted.contains("second prefix 45..171"), "{omitted}");
    session.receipt(
        "window-stage-real-prefill-chain",
        BTreeMap::from([
            ("same_ring_allocation".into(), serde_json::json!(true)),
            ("second_prefix".into(), serde_json::json!([45, 171])),
            (
                "missing_circular_write_rejected".into(),
                serde_json::json!(omitted),
            ),
            ("unbounded_lists_rejected".into(), serde_json::json!(bad)),
        ]),
    )?;
    Ok(())
}

#[test]
fn trained_payloads_distinguish_every_uncommitted_circular_cell() {
    let encoded = history(187);
    for p in 140..172 {
        assert_ne!(
            encoded[p].values,
            encoded[p - 136].values,
            "ring write omission is invisible at position {p}"
        );
    }
    let mask = bounded::keys(KQ, &[8], &[0], &[8], &[8], &[0], &[200]).unwrap();
    assert_eq!(
        mask.iter()
            .filter(|&&slot| slot >= 0)
            .copied()
            .collect::<Vec<_>>(),
        vec![8]
    );
}
