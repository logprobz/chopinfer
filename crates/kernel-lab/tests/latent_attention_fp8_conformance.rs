//! Latent attention (`latent_shapes.rs`) over an FP8 E4M3 block-128 pool
//! through the declarative conformance runner (`kernel_lab::conformance`):
//! the FP8 prefill entry and its partial form over the BF16 cases' rows,
//! key lists and tables (`latent_attention_conformance.rs`), with each
//! page's codes `[page][latent]` then its F32 block scales `[page][latent /
//! 128]` (`common/latent_fp8_case.rs`). The keys are FP8 one-hot
//! (`common/hot_fp8.rs`), so the oracle gives every output's exact bits:
//! the output rows witness the high-id rerun (the moved pages give the
//! first run's bits, the old IDs over the moved pool other tokens' codes
//! and scales) and the hole row writes zeros, or -inf, +0 and +0 partials.
//! The block scales past block 0 vary by token, so the outputs also show
//! that a launch reads each key's scales from its own page and slot. The
//! prefill entry reruns over a pool of [`hot_keys::WIDE_PAGES`] too, whose
//! rerun moves its pages to IDs past 16 bits and byte offsets past 2^32;
//! the oracle shows that the IDs narrowed to 16 bits change its outputs
//! (`narrowing_checked`), and a host model of a launch that computes page
//! byte offsets in unsigned 32 bits fails that rerun on the CPU.
//!
//! GPU: `CHOPINFER_KERNEL_GPU=<ordinal> CHOPINFER_KERNEL_REQUIRE=1 cargo test
//! -p kernel-lab --release --locked --offline --test
//! latent_attention_fp8_conformance -- --test-threads=1 --nocapture`

#[path = "common/hot_keys.rs"]
mod hot_keys;

#[path = "common/hot_fp8.rs"]
mod hot_fp8;

#[path = "common/latent_case.rs"]
mod latent_case;

#[path = "common/latent_fp8_case.rs"]
mod latent_fp8_case;

use std::collections::BTreeMap;
use std::f32::consts::LOG2_E;
use std::ops::Range;
use std::sync::atomic::Ordering::Relaxed;

use hot_fp8::BLOCK;
use hot_keys::{HIGHEST, NARROWED, Partials, WIDE_PAGES, narrowing_checked};
use kernel_core::contract::DType;
use kernel_lab::conformance::{Case, Expected, Images, Rows, Write};
use kernel_lab::latent_shapes::GEOMETRY;
use kernel_lab::latent_shapes::cubin_name;
use kernel_lab::{LabResult, gpu_test};
use kernels_sm120::latent::Entry;
use latent_case::{
    CAPACITY, HEADS, LATENT, MAX_KEYS, MAX_PAGES, PAGE, POOL_PAGES, RowPartials, STRIDE,
};
use latent_fp8_case::{inputs, outputs};
use serde_json::json;

fn entry(partial: bool) -> Entry {
    latent_case::entry_of(
        "fp8",
        if partial {
            "prefill_partial"
        } else {
            "prefill"
        },
    )
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
fn fp8_latent_attention_cases_validate() {
    for partial in [false, true] {
        let entry = entry(partial);
        assert_eq!(entry.splits(), 1, "prefill takes one split");
        case(&entry, partial, POOL_PAGES)
            .validate()
            .unwrap_or_else(|e| panic!("partial {partial}: {e}"));
    }
}

/// The wide prefill case's high-id rerun moves its pages to the top of the
/// pool, IDs 65632 to 65635 at byte offsets past 2^32, and the IDs narrowed
/// to 16 bits change its outputs.
#[test]
fn wide_fp8_latent_attention_case_validates() {
    let entry = entry(false);
    assert!(((WIDE_PAGES - 4) * STRIDE) as u64 > 1 << 32);
    case(&entry, false, WIDE_PAGES)
        .validate()
        .expect("wide fp8 latent prefill");
    assert!(
        NARROWED.load(Relaxed) > 0,
        "no ID past 16 bits was narrowed"
    );
    assert_eq!(HIGHEST.load(Relaxed), WIDE_PAGES as i32 - 1);
}

/// Page `id`'s byte offset as a launch that computes it in 64 bits reads
/// it, and as one that computes it in unsigned 32 bits does.
fn offset(id: usize) -> usize {
    id * STRIDE
}

fn u32_offset(id: usize) -> usize {
    (id * STRIDE) as u32 as usize
}

/// A host model of the kernel's arithmetic on row `row`'s keys, without
/// the oracle's domain checks, reading page `id` at byte offset `base(id)`:
/// F32 block chains times their block scales, then the softmax scale;
/// weights `exp2((s - m) log2 e)` from the row's maximum (NaN scores do not
/// raise it, as with `fmaxf`); F32 sums of the weights and of `weight *
/// block scale * code` per dim. On one-hot operands it gives the oracle's
/// bits; on other bytes, what such arithmetic makes of them.
fn modelled_partials(
    images: &Images,
    row: usize,
    entries: Range<usize>,
    read: fn(i32) -> i32,
    base: fn(usize) -> usize,
) -> LabResult<Partials> {
    let q = hot_keys::bf16_values(images.bytes("q")?);
    let pool = images.bytes("pool")?;
    let tables = images.i32s("block_tables")?;
    let item = usize::try_from(images.i32s("row_batch")?[row]).map_err(|e| e.to_string())?;
    let tokens = images.i32s("key_lists")?[row * MAX_KEYS..][entries]
        .iter()
        .filter_map(|&token| usize::try_from(token).ok())
        .map(|token| {
            let id = read(tables[item * MAX_PAGES + token / PAGE]);
            let id = usize::try_from(id).map_err(|e| e.to_string())?;
            hot_fp8::latent_token_at(pool, base(id), PAGE, LATENT, token % PAGE)
        })
        .collect::<LabResult<Vec<_>>>()?;
    let mut partials = Partials {
        max: vec![f32::NEG_INFINITY; HEADS],
        den: vec![0.0; HEADS],
        num: vec![0.0; HEADS * LATENT],
    };
    for head in 0..HEADS {
        let query = &q[(row * HEADS + head) * LATENT..][..LATENT];
        let scores = tokens
            .iter()
            .map(|(codes, scales)| {
                let sum = (0..LATENT / BLOCK).fold(0.0_f32, |sum, b| {
                    let chain = (b * BLOCK..(b + 1) * BLOCK).fold(0.0_f32, |chain, d| {
                        chain + query[d] as f32 * codes[d] as f32
                    });
                    sum + chain * scales[b]
                });
                sum * GEOMETRY.softmax_scale
            })
            .collect::<Vec<_>>();
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if top == f32::NEG_INFINITY {
            continue;
        }
        let weights = scores
            .iter()
            .map(|&s| ((s - top) * LOG2_E).exp2())
            .collect::<Vec<_>>();
        partials.max[head] = top;
        partials.den[head] = weights.iter().sum();
        for (d, out) in partials.num[head * LATENT..][..LATENT]
            .iter_mut()
            .enumerate()
        {
            *out = tokens
                .iter()
                .zip(&weights)
                .map(|((codes, scales), &p)| p * scales[d / BLOCK] * codes[d] as f32)
                .sum();
        }
    }
    Ok(partials)
}

fn exact_partials(
    images: &Images,
    row: usize,
    entries: Range<usize>,
    read: fn(i32) -> i32,
) -> LabResult<Partials> {
    modelled_partials(images, row, entries, read, offset)
}

fn u32_partials(
    images: &Images,
    row: usize,
    entries: Range<usize>,
    read: fn(i32) -> i32,
) -> LabResult<Partials> {
    modelled_partials(images, row, entries, read, u32_offset)
}

/// The runs that refuse the model's outputs at `live`.
fn refusing(case: Case<'_>, partial: bool, model: RowPartials) -> LabResult<Vec<String>> {
    case.refusing_runs(|images, live| {
        latent_case::outputs_of(images, live, partial, hot_keys::whole, model)?
            .into_iter()
            .map(|(name, expected)| match expected {
                Expected::Bits(bits) => Ok((name, bits)),
                _ => Err(format!("{name} has no exact bits")),
            })
            .collect()
    })
}

/// A launch that computes page byte offsets in unsigned 32 bits reads the
/// small pools' pages correctly, so only the wide case's high-id rerun,
/// whose moved pages lie past 2^32 bytes, refuses it. The model at 64-bit
/// offsets gives the oracle's bits in every run.
#[test]
fn a_launch_with_u32_byte_offsets_fails_the_wide_high_id_rerun() {
    for (partial, pages) in [(false, POOL_PAGES), (true, POOL_PAGES), (false, WIDE_PAGES)] {
        let entry = entry(partial);
        let label = format!("partial {partial}, {pages} pages");
        let exact = refusing(case(&entry, partial, pages), partial, exact_partials)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(exact.is_empty(), "{label}: {exact:?}");
        let wrapped = refusing(case(&entry, partial, pages), partial, u32_partials)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let wanted: &[&str] = if pages == WIDE_PAGES {
            &["high block_tables"]
        } else {
            &[]
        };
        assert_eq!(wrapped, wanted, "{label}");
    }
}

#[test]
fn fp8_latent_attention_conforms_gpu() {
    // The wide case's pool, and 1 GiB for the rest.
    let need = WIDE_PAGES * STRIDE + (1 << 30);
    gpu_test(need as u64, |session| {
        let module = session.module(&cubin_name())?;
        for (partial, pages) in [(false, POOL_PAGES), (true, POOL_PAGES), (false, WIDE_PAGES)] {
            let entry = entry(partial);
            let evidence = case(&entry, partial, pages)
                .run(session, &module, Some(&module))
                .map_err(|e| format!("fp8 partial {partial}, {pages} pages: {e}"))?;
            session.receipt(
                "latent_attention_fp8_conformance",
                BTreeMap::from([
                    ("partial".into(), json!(partial)),
                    ("pool_pages".into(), json!(pages)),
                    ("evidence".into(), evidence),
                ]),
            )?;
        }
        Ok(())
    })
    .expect("fp8 latent attention conforms");
}
