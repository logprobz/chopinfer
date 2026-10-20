//! Dispatch the bundle's prepared op, including its argument and grid choices.
use super::device::{Input, Kernels, Output};
use bundle_deepseek41::sm120 as b;
use kernel_core::{
    backend::{Backend, Counts, Launch, Need, View},
    contract::Argument,
    properties::Invariance,
    rmsnorm_quantize::RmsNormQuantizeBuffers as Norm,
    sinkhorn_expand_mix::{
        SinkhornRoundExpandMixBuffers as Hc, SinkhornRoundNormBuffers as Buffers,
        SinkhornRoundNormQuery,
    },
};
use kernel_harness::LabResult;
use kernels_sm120::backend::Sm120;
use model_deepseek41_contract::geometry::FLASH;
use runtime_cuda::{KernelArgument as A, Stream};

fn plans(rows: u32, views: &[View], widen: bool) -> LabResult<Vec<Launch>> {
    let backend = b::norm_quantize::register(Sm120::new(), &FLASH)?;
    let backend = b::norm_quantize_cached::register(backend, &FLASH);
    let backend = b::hc_round_expand_mix::register(backend, &FLASH);
    let backend = b::hc_round_chain::register(backend, &FLASH);
    let backend = b::hc_finish_norm::register(backend, &FLASH);
    let entry = b::hc_finish_norm::entry(&FLASH);
    let op = backend
        .sinkhorn_round_expand_mix_norm(
            &SinkhornRoundNormQuery {
                mixer: entry.mixer.query,
                norm: entry.norm.reference.query,
            },
            Need::rows(32).with_exact(Invariance::any_rows(32)),
        )
        .map_err(|e| format!("{e:?}"))?;
    let v = |index| views[index];
    op.launches(
        &Buffers {
            hc: Hc {
                x: v(0),
                sum: v(1),
                rounded: v(2),
                post_in: v(3),
                comb_in: v(4),
                expanded: v(5),
                fn_: v(6),
                base: v(7),
                scale: v(8),
                pre_in: v(9),
                collapsed: v(10),
                pre: v(11),
                post: v(12),
                comb: v(13),
                live_rows: v(14),
            },
            norm: Norm {
                input: v(10),
                weight: v(15),
                normed: v(16),
                widened: widen.then(|| v(17)),
                codes: v(18),
                scales: v(19),
                live_rows: v(14),
                status: v(20),
                status_row_offset: 0,
            },
        },
        v(21),
        Counts::rows(rows),
    )
    .map_err(|e| e.to_string())
}

pub fn launch(s: &Stream, k: &Kernels, i: &Input, o: &Output, widen: bool) -> LabResult<()> {
    let d = &i.data;
    let v = &o.data;
    let buffers = [
        &d["streams"],
        &d["sum"],
        &v["rounded"],
        &d["post_in"],
        &d["comb_in"],
        &v["expanded"],
        &d["fn"],
        &d["base"],
        &d["scale"],
        &d["pre_in"],
        &v["collapsed"],
        &v["pre"],
        &v["post"],
        &v["comb"],
        &i.live,
        &d["norm_weight"],
        &v["normed"],
        &v["widened"],
        &v["codes"],
        &v["scales"],
        &v["status"],
        &o.scratch,
    ];
    let views = buffers
        .iter()
        .enumerate()
        .map(|(n, buffer)| View {
            handle: n as u64 + 1,
            bytes: buffer.byte_len() as u64,
        })
        .collect::<Vec<_>>();
    for launch in plans(i.rows as u32, &views, widen)? {
        let args = launch
            .arguments
            .0
            .iter()
            .map(|arg| match *arg {
                Argument::DeviceBuffer(id) => Ok(A::Buffer(buffers[id as usize - 1])),
                Argument::NullDevicePointer => Ok(A::NullBuffer),
                Argument::U64(value) => Ok(A::U64(value)),
                Argument::U32(value) if value == i.eps => Ok(A::U32(value)),
                _ => Err("unqualified served argument or norm epsilon".to_owned()),
            })
            .collect::<LabResult<Vec<_>>>()?;
        k.call(s, &launch.symbol, launch.plan, &args)?;
    }
    Ok(())
}

#[test]
fn served_plans_match_the_executed_bitwise_recipes() {
    let views = (1..=22)
        .map(|handle| View {
            handle,
            bytes: 1 << 30,
        })
        .collect::<Vec<_>>();
    for rows in [8, 32] {
        for widen in [false, true] {
            let actual = plans(rows, &views, widen).unwrap();
            assert_eq!(actual.len(), 2);
            let (symbol, plan) = if rows == 8 {
                let e = b::hc_round_chain::entry(&FLASH);
                (e.symbols()[0].clone(), e.plan(rows, rows, 4).unwrap())
            } else {
                let e = b::hc_round_rows::entry(&FLASH);
                (e.symbols()[0].clone(), e.plan(rows, rows, 2).unwrap())
            };
            assert_eq!(actual[0].symbol, symbol);
            assert_eq!(actual[0].plan, plan);
            let d = |index: u64| Argument::DeviceBuffer(index + 1);
            assert_eq!(actual[0].arguments.0, [0, 1, 2, 3, 4, 5, 6, 21, 14].map(d));
            let tail = b::hc_finish_norm::entry(&FLASH);
            assert_eq!(actual[1].symbol, tail.symbols()[1]);
            assert_eq!(actual[1].plan, tail.plan(rows, rows, 32).unwrap());
            let mut args = [5, 7, 8, 9, 10, 11, 12, 13, 21, 14, 15, 16].map(d).to_vec();
            args.push(if widen {
                d(17)
            } else {
                Argument::NullDevicePointer
            });
            args.extend([d(18), d(19), d(20), Argument::U64(0)]);
            args.push(Argument::U32(1e-20_f32.to_bits()));
            assert_eq!(actual[1].arguments.0, args);
            for launch in actual {
                assert!(!launch.collective && !launch.persistent);
                assert_eq!(launch.rows, rows);
                assert!(b::fold_cubin_for_symbol(&launch.symbol).is_some());
            }
        }
    }
}
