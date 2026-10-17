//! Ordered merging with exact weights, including empty splits and heads.

use kernel_build::EntryParam;
use kernel_core::codec::f32_to_bf16;
use kernel_core::contract::{Argument, ArgumentList, DType};
use kernel_lab::families::latent_sources_controls as family;
use kernel_lab::kit::{LabModules, Modules};
use kernel_lab::launch::{Bindings, launch};
use kernel_lab::poison::{Replay, WriteMap};
use std::collections::BTreeMap;

#[test]
fn native_merge_folds_splits_in_order_with_poison_graph_and_pdl() {
    kernel_lab::gpu_test(1 << 30, |session| {
        let entry = family::catalog()
            .into_iter()
            .find(|e| e.splits > 1)
            .unwrap();
        let (heads, width, splits) = (4, 512, entry.splits as usize);
        let mut workspace = vec![0.0_f32; splits * heads * (width + 2)];
        for split in 0..splits {
            for head in 0..heads {
                let at = (split * heads + head) * (width + 2);
                let empty = head == 2
                    || (head == 0 && split >= 4)
                    || (head == 1 && split != 0 && split != splits - 1);
                workspace[at] = if empty { f32::NEG_INFINITY } else { 0.0 };
                if empty {
                    continue;
                }
                workspace[at + 1] = 1.0;
                for d in 0..width {
                    workspace[at + 2 + d] = ((split + d) % 11) as f32 * 0.5 - 2.5;
                }
                if head == 0 {
                    workspace[at + 2] = [2_f32.powi(25), -2_f32.powi(25), 1.0, 0.0][split];
                }
                if head == 1 {
                    workspace[at + 3] = if split == 0 { 2.0 } else { 6.0 };
                }
            }
        }
        let mut expected = Vec::new();
        for head in 0..heads {
            let den = (0..splits)
                .map(|s| workspace[(s * heads + head) * (width + 2) + 1])
                .sum::<f32>();
            for d in 0..width {
                let mut value = 0.0_f32;
                for split in 0..splits {
                    let at = (split * heads + head) * (width + 2);
                    let weight = if workspace[at] == f32::NEG_INFINITY {
                        0.0_f32
                    } else {
                        1.0
                    };
                    value = weight.mul_add(workspace[at + 2 + d], value);
                }
                expected.push(f32_to_bf16(if den == 0.0 { 0.0 } else { value / den }));
            }
        }
        assert_eq!(expected[0], f32_to_bf16(0.25));
        assert_eq!(expected[width + 1], f32_to_bf16(4.0));
        let reversed = (0..splits).rev().fold(0.0_f32, |sum, s| {
            sum + workspace[s * heads * (width + 2) + 2]
        });
        assert_eq!(reversed, 0.0);
        let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>();
        let upload = |data: &[u8]| -> kernel_lab::LabResult<runtime_cuda::DeviceBuffer> {
            let b = session.allocate(data.len())?;
            b.copy_from_bytes(data, &session.stream)
                .map_err(|e| e.to_string())?;
            Ok(b)
        };
        let buffers = BTreeMap::from([
            ("workspace", upload(&bytes(&workspace))?),
            ("live_rows", upload(&1_u32.to_le_bytes())?),
            ("out", session.allocate(heads * width * 2)?),
            ("sinks", upload(&bytes(&vec![-1000.0; heads]))?),
            ("cos_sin", upload(&bytes(&[1.0, 0.0].repeat(32)))?),
        ]);
        let handles = buffers
            .keys()
            .enumerate()
            .map(|(i, &name)| (name, i as u64 + 1))
            .collect::<BTreeMap<_, _>>();
        let bindings = Bindings {
            buffers: buffers
                .iter()
                .map(|(&name, buffer)| (handles[name], buffer))
                .collect(),
        };
        let arguments = ArgumentList(
            entry
                .merge_params()
                .unwrap()
                .iter()
                .map(|p| {
                    let EntryParam::DevicePointer { name, .. } = p else {
                        panic!("merge pointer ABI")
                    };
                    Argument::DeviceBuffer(handles[name.as_str()])
                })
                .collect(),
        );
        let functions = LabModules.load(session)?;
        let function = functions(&entry.merge_symbol())?;
        let plan = entry.merge_plan(1).map_err(|e| e.to_string())?;
        let run = |stream: &runtime_cuda::Stream| {
            launch(
                &session.context,
                &function,
                stream,
                plan,
                &arguments,
                &bindings,
            )
        };
        let mut replay = Replay::new(session).output(
            "output",
            &buffers["out"],
            WriteMap::of(DType::Bf16).prefix(heads * width)?,
        );
        replay.eager(|| run(&session.stream))?;
        assert_eq!(
            replay.expected("output").unwrap(),
            kernel_lab::kit::bf16_bytes(&expected)
        );
        replay.relaunch(|| run(&session.stream))?;
        let graph = session.capture(run)?;
        replay.graph(&graph)?;
        let staged = upload(&bytes(&workspace))?;
        let probes = LabModules.probes(session)?.ok_or("missing PDL probes")?;
        let copy = probes.copy.ok_or("missing PDL copy producer")?;
        replay.pdl(
            || {
                kernel_lab::poison::fill_all(session, &buffers["workspace"], &[0xff])?;
                copy.copy(&session.stream, &staged, &buffers["workspace"])
            },
            || run(&session.stream),
            true,
        )?;
        let mut receipt = BTreeMap::new();
        replay.record(&mut receipt);
        receipt.insert("splits".into(), serde_json::json!(splits));
        receipt.insert("reversed_fold_rejected".into(), serde_json::json!(true));
        session.receipt("native-latent-merge-order", receipt)?;
        Ok(())
    })
    .expect("native ordered merge");
}
