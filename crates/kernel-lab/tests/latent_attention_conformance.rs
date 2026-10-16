//! Latent attention (`latent_shapes.rs`) through the declarative
//! conformance runner (`kernel_lab::conformance`): the BF16 prefill entry
//! and its partial form over gathered key lists of shuffled pages of a
//! larger pool. Prefill takes one split, so the launch is one kernel; the
//! decode and verify entries cut every row into 64 fixed splits
//! (`latent/fixed_splits.cuh`) and fold them with a second (merge) kernel,
//! which `latent_split_merge_conformance.rs` runs as a chain. The fixture
//! is `common/latent_case.rs`. The block
//! tables rerun with their pages moved to the top of the pool and the last
//! live key list reruns all holes. The keys are one-hot
//! (`common/hot_keys.rs`), so the oracle gives every output's exact bits: the
//! output rows witness the high-id rerun (the moved pages give the first
//! run's bits, the old IDs over the moved pool other tokens' latents) and
//! the hole row writes zeros, or -inf, +0 and +0 partials. The prefill
//! entry reruns over a pool of [`hot_keys::WIDE_PAGES`] too, whose rerun
//! moves its pages to IDs past 16 bits and byte offsets past 2^32; the
//! oracle shows that the IDs narrowed to 16 bits change its outputs
//! (`narrowing_checked`).
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! latent_attention_conformance -- --test-threads=1 --nocapture`

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/latent_case.rs"]
mod latent_case;

use std::collections::BTreeMap;
use std::sync::atomic::Ordering::Relaxed;

use hot_keys::{HIGHEST, NARROWED, WIDE_PAGES, narrowing_checked};
use kernel_core::contract::DType;
use kernel_lab::conformance::{Case, Rows, Write};
use kernel_lab::gpu_test;
use kernel_lab::latent_shapes::cubin_name;
use kernels_sm120::latent::Entry;
use latent_case::{CAPACITY, HEADS, LATENT, POOL_PAGES, STRIDE, inputs, outputs};
use serde_json::json;

fn entry(partial: bool) -> Entry {
    latent_case::entry(if partial {
        "prefill_partial"
    } else {
        "prefill"
    })
}

/// The case over a pool of `pages` pages.
fn case(entry: &Entry, partial: bool, pages: usize) -> Case<'_> {
    let mut case = inputs(Case::new(entry, CAPACITY as u32), entry.splits(), pages)
        // One split writes no split partials.
        .scratch("workspace", 16, |_, _| Ok(Vec::new()))
        .oracle(move |images, live| {
            narrowing_checked(images, live, "block_tables", |images, live, read| {
                outputs(images, live, partial, read)
            })
        });
    case = if partial {
        case.output("out_max", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output("out_den", DType::F32, (Rows::Live, HEADS), Write::LiveRows)
            .output(
                "out_num",
                DType::F32,
                (Rows::Live, HEADS * LATENT),
                Write::LiveRows,
            )
    } else {
        case.output(
            "out",
            DType::Bf16,
            (Rows::Live, HEADS * LATENT),
            Write::LiveRows,
        )
    };
    case
}

#[test]
fn latent_attention_cases_validate() {
    for partial in [false, true] {
        let entry = entry(partial);
        assert_eq!(entry.splits(), 1, "prefill takes one split");
        case(&entry, partial, POOL_PAGES)
            .validate()
            .unwrap_or_else(|e| panic!("partial {partial}: {e}"));
    }
}

/// The wide prefill case's high-id rerun moves its pages to the top of the
/// pool, IDs 65632 to 65635, and the IDs narrowed to 16 bits change its
/// outputs.
#[test]
fn wide_latent_attention_case_validates() {
    let entry = entry(false);
    case(&entry, false, WIDE_PAGES)
        .validate()
        .expect("wide latent prefill");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

#[test]
fn latent_attention_conforms_gpu() {
    // The wide case's pool, and 1 GiB for the rest.
    let need = WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(&cubin_name())?;
        for (partial, pages) in [(false, POOL_PAGES), (true, POOL_PAGES), (false, WIDE_PAGES)] {
            let entry = entry(partial);
            let evidence = case(&entry, partial, pages)
                .run(session, &module, Some(&module))
                .map_err(|e| format!("partial {partial}, {pages} pages: {e}"))?;
            session.receipt(
                "latent_attention_conformance",
                BTreeMap::from([
                    ("partial".into(), json!(partial)),
                    ("pool_pages".into(), json!(pages)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("latent attention conforms");
}
