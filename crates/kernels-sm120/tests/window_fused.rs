use kernel_core::backend::{Backend, Counts, Need, View};
use kernel_core::contract::{Argument, DType};
use kernel_core::latent_attention::{LatentKvFormat, LatentWindowBuffers, LatentWindowStorage};
use kernel_core::latent_glue::{LatentWindowWrite, LatentWindowWriteBuffers, WindowAddressing};
use kernel_core::precision::quantizer::{BlockQuantization, BlockQuantizeQuery, QuantizeOutput};
use kernels_sm120::{
    backend::Sm120,
    latent::{glue, window_fused},
    row::block_quantize,
};

fn entry(kv: LatentKvFormat) -> window_fused::Entry {
    window_fused::Entry {
        kv,
        capacity: 32,
        symbol: format!(
            "chop_window_fused_{}_v1",
            if kv == LatentKvFormat::Bf16 {
                "bf16"
            } else {
                "native"
            }
        ),
    }
}
#[test]
fn window_fusion_retains_validation_and_independent_planes() {
    for kv in [LatentKvFormat::Fp8E4M3Ue8M0Block32, LatentKvFormat::Bf16] {
        let mut backend = Sm120::new().with_latent_window_fused(entry(kv));
        for kernel in [
            glue::Kernel::WindowRotate,
            glue::Kernel::WindowStore,
            glue::Kernel::WindowStoreBf16,
        ] {
            backend = backend.with_latent_glue(glue::Entry {
                kernel,
                capacity: 32,
                symbol: format!("test_{kernel:?}"),
            });
        }
        backend = backend.with_block_quantize(
            block_quantize::Entry::new(
                BlockQuantizeQuery {
                    width: 512,
                    input: DType::F32,
                    format: BlockQuantization::E4M3_CEIL32,
                    output: QuantizeOutput::Codes,
                },
                32,
                "chop_quant_v1",
            )
            .unwrap()
            .parallel()
            .unwrap(),
        );
        let q = LatentWindowWrite {
            kv,
            latent: 512,
            rotary: 64,
            epsilon_bits: 1e-20_f32.to_bits(),
            addressing: WindowAddressing::Circular { slots: 136 },
        };
        let old = backend.latent_window_write(&q, Need::rows(32)).unwrap();
        let new = backend
            .latent_window_write_fused(&q, Need::rows(32))
            .unwrap();
        assert_eq!(old.info().bytes, new.info().bytes);
        assert_eq!(new.info().workspace, 0);
        assert_eq!(new.info().symbols.len(), 1);
        let v = |handle| View {
            handle,
            bytes: 1 << 40,
        };
        let b = LatentWindowWriteBuffers {
            input: v(1),
            weight: v(2),
            latent: v(3),
            cos_sin: v(4),
            positions: v(5),
            row_batch: v(6),
            first_positions: None,
            last_positions: v(7),
            window: LatentWindowBuffers {
                codes: v(8),
                scales: if kv == LatentKvFormat::Bf16 {
                    View::EMPTY
                } else {
                    v(9)
                },
                sequences: 2,
                code_stride: (1 << 32) + 136 * 1024 + 66,
                scale_stride: 136 * 16 + 7,
                storage: LatentWindowStorage::Staged { slots: 136 },
                item_slots: View::EMPTY,
                items: 0,
                key_lists: View::EMPTY,
                key_stride: 0,
                budget: 0,
            },
            quantized_codes: v(10),
            quantized_scales: v(11),
            live_rows: v(12),
            status: v(13),
            status_row_offset: (1 << 32) + 3,
        };
        for rows in [1, 2, 4, 8, 16, 32] {
            let baseline = old.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            let fused = new.launches(&b, View::EMPTY, Counts::rows(rows)).unwrap();
            assert_eq!((baseline.len(), fused.len()), (3, 1));
            let args = &fused[0].arguments.0;
            assert_eq!(&args[..8], &baseline[0].arguments.0);
            assert_eq!(&args[10..16], &baseline[2].arguments.0[2..8]);
            assert_eq!(&args[16..21], &baseline[2].arguments.0[9..14]);
            assert_eq!(args[7], Argument::U64((1 << 32) + 3));
            assert_eq!(fused[0].plan.workspace_bytes, 0);
            assert!(fused[0].plan.pdl);
        }
        assert!(
            backend
                .latent_window_write_fused(&q, Need::rows(33))
                .is_err()
        );
        for variant in 0..5 {
            let mut bad = b;
            match variant {
                0 => bad.status_row_offset = u64::MAX,
                1 => bad.latent.handle = bad.input.handle,
                2 => bad.window.codes.bytes = 1,
                3 => bad.window.code_stride += 1,
                _ => bad.quantized_scales.bytes = 1,
            }
            assert!(new.launches(&bad, View::EMPTY, Counts::rows(2)).is_err());
        }
    }
}

#[test]
fn window_fusion_compiles_and_audits_both_storage_modes() {
    let toolchain = kernel_build::Toolchain::detect().unwrap();
    assert!(!toolchain.skipped);
    let root = std::env::temp_dir().join(format!("window-fused-build-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let spec = window_fused::cubin_spec(
        "window_fused",
        &[
            entry(LatentKvFormat::Fp8E4M3Ue8M0Block32),
            entry(LatentKvFormat::Bf16),
        ],
    )
    .unwrap();
    let built = kernel_build::build(
        &toolchain,
        &[spec],
        &root.join("out"),
        &root.join("lock"),
        kernel_build::BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(built[0].entries.len(), 2);
    for e in &built[0].entries {
        assert_eq!((e.local_bytes, e.stack_bytes), (0, 0));
    }
    std::fs::remove_dir_all(root).unwrap();
}
