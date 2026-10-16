use kernel_build::EntryParam;
use kernel_core::codec::{bf16_to_f32, f32_to_bf16};
use kernel_core::contract::{Argument, DType, RuntimeCounts};
use kernel_lab::{LabResult, gpu_test_multi, launch};
use std::collections::BTreeMap;
#[path = "common/concurrency.rs"]
mod conc;
use kernel_lab::poison::{Replay, WriteMap};

#[test]
fn m2_mimo_combine() {
    if !conc::enabled("m2") {
        return;
    }
    gpu_test_multi(128 << 20, |sessions| {
        assert_eq!(sessions.len(), 4);
        let entry = kernels_sm120::row::catalog(kernel_lab::catalogs())
            .into_iter()
            .find(|e| e.symbol.contains("combine_mimo4096"))
            .expect("combine entry");
        let mut failures = conc::replay::Failures::default();
        for s in sessions {
            conc::exclusive(s)?;
            let module = s
                .context
                .load_module(include_bytes!(concat!(
                    env!("OUT_DIR"),
                    "/row_engine.cubin"
                )))
                .map_err(|e| e.to_string())?;
            let f = module.function(&entry.symbol).map_err(|e| e.to_string())?;
            for rows in conc::ROWS {
                let result = (|| -> LabResult<()> {
                    let routed: Vec<_> = (0..rows * 8 * 4096)
                        .map(|i| f32_to_bf16(((i * 31 % 509) as f32 - 254.0) / 1024.0))
                        .collect();
                    let expected: Vec<_> = (0..rows * 4096)
                        .map(|i| {
                            let (r, c) = (i / 4096, i % 4096);
                            f32_to_bf16(
                                (0..8)
                                    .map(|k| bf16_to_f32(routed[(r * 8 + k) * 4096 + c]))
                                    .sum(),
                            )
                        })
                        .collect();
                    let input = s.upload_bf16(&routed)?;
                    let weights = s.upload_f32(&vec![1.0; rows * 8])?;
                    let output = s.allocate(rows * 4096 * 2)?;
                    let live = s.allocate(4)?;
                    live.copy_from_bytes(&(rows as u32).to_ne_bytes(), &s.stream)
                        .map_err(|e| e.to_string())?;
                    let args = entry
                        .pack_arguments(
                            entry
                                .params
                                .iter()
                                .map(|p| match p {
                                    EntryParam::DevicePointer { name, .. } => {
                                        Ok(Argument::DeviceBuffer(match name.as_str() {
                                            "routed" => 1,
                                            "weights" => 2,
                                            "output" => 3,
                                            "live_rows" | "live_tokens" => 4,
                                            _ => {
                                                return Err(format!(
                                                    "unexpected combine pointer {name}"
                                                ));
                                            }
                                        }))
                                    }
                                    _ => Err("unexpected combine scalar".to_string()),
                                })
                                .collect::<LabResult<Vec<_>>>()?,
                        )
                        .map_err(|e| e.to_string())?;
                    let plan = entry
                        .plan(
                            rows as u32,
                            RuntimeCounts {
                                m: rows as u32,
                                batch: rows as u32,
                                context: 0,
                                routed_rows: rows as u32 * 8,
                                split_k: 1,
                            },
                        )
                        .map_err(|e| e.to_string())?;
                    let bindings = launch::Bindings {
                        buffers: BTreeMap::from([
                            (1, &input),
                            (2, &weights),
                            (3, &output),
                            (4, &live),
                        ]),
                    };
                    let run = || launch::launch(&s.context, &f, &s.stream, plan, &args, &bindings);
                    let mut replay = Replay::new(s).output(
                        "combine",
                        &output,
                        WriteMap::of(DType::Bf16).prefix(rows * 4096)?,
                    );
                    replay.eager(run)?;
                    assert_eq!(expected, s.download_bf16(&output, rows * 4096)?);
                    conc::measure(
                        s,
                        "M2",
                        "mimo_combine",
                        rows,
                        "row_engine",
                        run,
                        || {
                            assert_eq!(expected, s.download_bf16(&output, rows * 4096)?);
                            Ok(())
                        },
                        |graph| {
                            replay.graph(graph)?;
                            let mut fields = BTreeMap::new();
                            replay.record(&mut fields);
                            Ok(fields)
                        },
                    )?;
                    assert_eq!(expected, s.download_bf16(&output, rows * 4096)?);

                    Ok(())
                })();
                failures.check(&format!("rank={} rows={rows}", s.ordinal), result);
            }
        }
        failures.finish()?;
        Ok(())
    })
    .expect("M2 combine")
    .expect("GPU required");
}
