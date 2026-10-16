//! Build and audit sampling cubins without opening a GPU.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use kernel_build::{BuildOptions, Toolchain, build};
use kernel_core::contract::{Argument, DType, RuntimeCounts};
use kernels_sm120::sampling::{
    FILTER_ROW_BYTES, FilteredSamplingBuffers, SamplingBuffers, SamplingQuery,
    SimpleSamplingBuffers, cubin_spec, filtered_histogram_groups, instantiation, plan,
    plan_filtered, plan_simple,
};

#[test]
fn sampling_planner_checks_capacity_and_workspace() {
    let query = SamplingQuery {
        row_capacity: 64,
        width_capacity: 152_576,
        dtype: DType::F32,
    };
    let buffers = SamplingBuffers {
        logits: 1,
        metadata: 2,
        counts: 3,
        biases: 4,
        eos: 5,
        adjusted: 6,
        order: 7,
        output: 8,
        masks: None,
    };
    let counts = RuntimeCounts {
        m: 0,
        batch: 32,
        context: 0,
        routed_rows: 0,
        split_k: 1,
    };
    let planned = plan(query, counts, 152_576, 154_112, buffers).unwrap();
    assert_eq!(planned.launch.grid, [64, 1, 1]);
    assert_eq!(planned.launch.workspace_bytes, 64 * 152_576 * 12);
    assert!(planned.launch.pdl);
    assert_eq!(
        planned.arguments.0[11..],
        [Argument::NullDevicePointer, Argument::U32(0)]
    );
    let masked = SamplingBuffers {
        masks: Some(9),
        ..buffers
    };
    let planned = plan(query, counts, 152_576, 154_112, masked).unwrap();
    assert_eq!(
        planned.arguments.0[11..],
        [Argument::DeviceBuffer(9), Argument::U32(4768)]
    );
    assert!(plan(query, counts, 152_577, 154_112, buffers).is_err());
    assert!(plan(query, counts, 152_576, 152_575, buffers).is_err());
    assert_eq!(instantiation(query).unwrap().symbol, "chop_sampling_f32_v1");
    let simple_buffers = SimpleSamplingBuffers {
        logits: 1,
        metadata: 2,
        partials: 3,
        output: 4,
    };
    let simple = plan_simple(query, counts, 152_576, 154_112, simple_buffers, false).unwrap();
    assert_eq!(simple.partials.grid, [149, 32, 1]);
    assert_eq!(simple.partials.block, [64, 1, 1]);
    // The top-logprob tile sort owns four keys per thread of 1024.
    let top = plan_simple(query, counts, 152_576, 154_112, simple_buffers, true).unwrap();
    assert_eq!(top.partials.block, [256, 1, 1]);
    assert_eq!(
        (top.partial_args, top.finish),
        (simple.partial_args.clone(), simple.finish)
    );
    assert_eq!(simple.finish.block, [32, 1, 1]);
    assert!(simple.partials.pdl && simple.finish.pdl);
    let filtered = plan_filtered(
        query,
        counts,
        152_576,
        154_112,
        FilteredSamplingBuffers {
            logits: 1,
            metadata: 2,
            counts: 3,
            biases: 4,
            eos: 5,
            partials: 6,
            filter: 7,
            adjusted: 8,
            order: 9,
            output: 10,
            masks: None,
        },
    )
    .unwrap();
    for launch in [filtered.stats, filtered.collect, filtered.gumbel] {
        assert_eq!(launch.grid, [149, 32, 1]);
        assert_eq!(launch.block, [256, 1, 1]);
        assert!(launch.pdl);
    }
    assert_eq!(filtered.histogram.grid, [38, 32, 1]);
    assert_eq!(filtered.resolve.grid, [32, 1, 1]);
    assert_eq!(filtered.resolve.block, [1024, 1, 1]);
    assert_eq!(filtered.finish.grid, [32, 1, 1]);
    assert_eq!(filtered_histogram_groups(149, 1), 149);
    assert_eq!(filtered_histogram_groups(149, 4), 96);
    assert_eq!(filtered_histogram_groups(149, 64), 38);
    assert_eq!(filtered_histogram_groups(2, 64), 2);
    assert_eq!(
        filtered.stats.workspace_bytes,
        64 * (149 * 112 + FILTER_ROW_BYTES + 152_576 * 12)
    );
    assert_eq!(filtered.tile_args.0.len(), 14);
    assert_eq!(filtered.finish_args.0.len(), 7);
    assert!(
        plan_filtered(
            query,
            counts,
            152_577,
            154_112,
            FilteredSamplingBuffers {
                logits: 1,
                metadata: 2,
                counts: 3,
                biases: 4,
                eos: 5,
                partials: 6,
                filter: 7,
                adjusted: 8,
                order: 9,
                output: 10,
                masks: Some(11),
            }
        )
        .is_err()
    );
}

#[test]
fn sampling_entries_compile_and_have_no_spills() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 is available");
    assert!(!toolchain.skipped, "SM120 compilation is required on Linux");
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = PathBuf::from(std::env::var("TMPDIR").unwrap_or_else(|_| "/tmp".into()))
        .join(format!("sampling-cubin-{}-{stamp}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let cubins = build(
        &toolchain,
        &[cubin_spec()],
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .expect("sampling cubin builds and audits");
    assert_eq!(cubins.len(), 1);
    assert_eq!(cubins[0].entries.len(), 22);
    assert!(cubins[0].entries.iter().all(|entry| entry.local_bytes == 0));
    let sass = Command::new("/usr/local/cuda/bin/cuobjdump")
        .arg("-sass")
        .arg(&cubins[0].path)
        .output()
        .expect("inspect sampling SASS");
    assert!(sass.status.success());
    let text = String::from_utf8_lossy(&sass.stdout);
    assert!(text.contains("PREEXIT") && text.contains("ACQBULK"));
    fs::remove_dir_all(root).unwrap();
}
