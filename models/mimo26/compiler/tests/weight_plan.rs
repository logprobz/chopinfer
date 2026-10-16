//! The bundle's weight plans. The text, MTP and DFlash images of the
//! synthetic served headers lay out, at every rank of TP2 and TP4, the
//! views, offsets, padding, arena length and pieces the pre-port planner
//! recorded (`fixtures/mimo26/image-layout-goldens.json`). Every planned
//! encoder record, executed by `kernel-core`'s transform set over synthetic
//! source bytes, produces its piece's length inside its view, the pieces of
//! an image never overlap, and the encoder views hold the bytes the encoder
//! kernels read.

use std::collections::BTreeMap;

use kernel_core::weight_transform::{SourceTensor, WeightRecord, WeightTransform};
use model_compiler_sdk::quantization::import::ImportedOperand;
use model_mimo26_contract::Tp;
use model_mimo26_contract::catalog::TensorRecord;
use model_mimo26_contract::drafters::MASK_BYTES;
use model_mimo26_contract::precision;
use model_mimo26_contract::weight_plan::{
    DFLASH_FILE, Drafter, MTP_FILE, PlannedView, drafter_views, encoder_views_flash,
    prepared_views, sequential_offsets, text_views,
};
use runtime_weights::{DType, sha256_bytes};
use serde_json::Value;

const GOLDENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../fixtures/mimo26/image-layout-goldens.json"
);

/// Tensors laid end to end in one in-memory file.
struct Synthetic {
    file: String,
    data: Vec<u8>,
    tensors: BTreeMap<String, TensorRecord>,
}

impl Synthetic {
    fn new(file: &str) -> Self {
        Self {
            file: file.into(),
            data: Vec::new(),
            tensors: BTreeMap::new(),
        }
    }

    fn add(&mut self, name: &str, dtype: DType, shape: &[u64]) {
        let element = match dtype {
            DType::Bf16 => 2,
            DType::F32 => 4,
            _ => 1,
        };
        let bytes = shape.iter().product::<u64>() * element;
        let seed = self.tensors.len() as u64;
        let start = self.data.len() as u64;
        self.data
            .extend((0..bytes).map(|n| (n * 37 + (n >> 9) + seed * 11) as u8));
        self.tensors.insert(
            name.into(),
            TensorRecord {
                name: name.into(),
                dtype,
                shape: shape.to_vec(),
                file: self.file.clone(),
                file_start: start,
                bytes,
            },
        );
    }

    /// Executes every planned piece of one rank's image and returns the
    /// image, with zero where no piece writes (alignment padding).
    fn image(&self, views: &[PlannedView]) -> Vec<u8> {
        let (offsets, image_bytes) = sequential_offsets(views);
        let mut image = vec![0_u8; image_bytes as usize];
        let mut covered = vec![false; image.len()];
        let mut end = 0;
        for (view, offset) in views.iter().zip(&offsets) {
            assert!(*offset >= end, "{} overlaps the view before it", view.name);
            end = offset + view.bytes;
            assert!(end <= image_bytes, "{} ends past the image", view.name);
            for piece in &view.pieces {
                let sources: Vec<SourceTensor> = piece
                    .sources
                    .iter()
                    .map(|name| {
                        let t = &self.tensors[name];
                        SourceTensor {
                            dtype: t.dtype.safetensors_name().into(),
                            shape: t.shape.clone(),
                            byte_length: t.bytes,
                        }
                    })
                    .collect();
                let record = WeightRecord {
                    transform: piece.transform.clone(),
                    sources,
                    axis: piece.axis,
                    ranges: piece.ranges.clone(),
                    stored_shape: piece.stored_shape.clone(),
                }
                .prepare(piece.bytes)
                .unwrap_or_else(|e| panic!("{}: {e}", view.name));
                let bytes = record
                    .produce_whole(&mut |index, at, buffer| {
                        let t = &self.tensors[&piece.sources[index]];
                        let start = (t.file_start + at) as usize;
                        buffer.copy_from_slice(&self.data[start..start + buffer.len()]);
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(bytes.len() as u64, piece.bytes, "{}", view.name);
                assert!(
                    piece.offset + piece.bytes <= view.bytes,
                    "{} piece at {} ends past its view",
                    view.name,
                    piece.offset
                );
                let at = (offset + piece.offset) as usize;
                assert!(
                    covered[at..at + bytes.len()].iter().all(|c| !c),
                    "{} piece at {} overlaps another",
                    view.name,
                    piece.offset
                );
                covered[at..at + bytes.len()].fill(true);
                image[at..at + bytes.len()].copy_from_slice(&bytes);
            }
        }
        image
    }
}

/// Admitted operands by name.
type Admitted = BTreeMap<String, ImportedOperand>;

/// The images the pin covers.
const IMAGES: [&str; 3] = ["text", "mtp", "dflash"];

/// The synthetic served headers of each image, admitted as the planner
/// admits a checkpoint.
struct Headers {
    text: BTreeMap<String, TensorRecord>,
    mtp: BTreeMap<String, TensorRecord>,
    dflash: BTreeMap<String, TensorRecord>,
    text_admitted: Admitted,
    mtp_admitted: Admitted,
    dflash_admitted: Admitted,
    mask: [u8; MASK_BYTES],
}

impl Headers {
    fn new() -> Self {
        let geometry = &model_mimo26_contract::geometry::V26_FLASH;
        let text_roles = precision::text_roles(geometry, |layer| geometry.is_global(layer as u32));
        let mtp_roles = precision::mtp_roles(geometry);
        let text = placed(
            "model.safetensors",
            precision::expected_source_tensors(&text_roles).unwrap(),
        );
        let mtp = placed(
            MTP_FILE,
            precision::expected_source_tensors(&mtp_roles).unwrap(),
        );
        let dflash = placed(
            DFLASH_FILE,
            model_mimo26::catalog::expected_dflash_tensors(geometry),
        );
        let metadata = precision::export_metadata(
            Some(&model_mimo26_contract::served_quantization_config(geometry)),
            None,
        )
        .unwrap();
        Self {
            text_admitted: precision::admit(geometry, &metadata, &text, &text_roles).unwrap(),
            mtp_admitted: precision::admit(geometry, &metadata, &mtp, &mtp_roles).unwrap(),
            dflash_admitted: precision::admit_dflash(geometry, &dflash).unwrap(),
            text,
            mtp,
            dflash,
            mask: mask(),
        }
    }

    fn views(&self, image: &str, rank: usize, tp: Tp) -> Vec<PlannedView> {
        match image {
            "text" => text_views(&self.text, &self.text_admitted, None, rank, tp),
            "mtp" => drafter_views(&self.mtp, &self.mtp_admitted, Drafter::Mtp, None, rank, tp),
            _ => drafter_views(
                &self.dflash,
                &self.dflash_admitted,
                Drafter::Dflash,
                Some(&self.mask),
                rank,
                tp,
            ),
        }
        .unwrap()
    }
}

/// The served headers with synthetic placement: each file's tensors laid
/// end to end by decoder layer, then name, so file order is not name order
/// (`model.layers.10` follows `model.layers.9`).
fn placed(
    file: &str,
    headers: BTreeMap<String, (DType, Vec<u64>)>,
) -> BTreeMap<String, TensorRecord> {
    let layer = |name: &str| {
        let mut parts = name.split('.');
        parts
            .by_ref()
            .find(|part| *part == "layers")
            .and_then(|_| parts.next()?.parse::<u64>().ok())
            .map_or(0, |layer| layer + 1)
    };
    let mut ordered: Vec<_> = headers.into_iter().collect();
    ordered.sort_by_key(|(name, _)| (layer(name), name.clone()));
    let mut start = 0;
    ordered
        .into_iter()
        .map(|(name, (dtype, shape))| {
            let width = match dtype {
                DType::Bf16 => 2,
                DType::F32 => 4,
                _ => 1,
            };
            let bytes = shape.iter().product::<u64>() * width;
            let record = TensorRecord {
                name: name.clone(),
                dtype,
                shape,
                file: file.into(),
                file_start: start,
                bytes,
            };
            start += bytes;
            (name, record)
        })
        .collect()
}

/// DFlash's mask embedding, a fixed byte pattern.
fn mask() -> [u8; MASK_BYTES] {
    std::array::from_fn(|n| (n * 7 + 3) as u8)
}

/// One image's canonical lines: its length, then each view in arena order
/// with its offset, size and the padding after it, then the view's pieces
/// at their arena offsets. Returns the lines and the image's summary.
fn canonical(views: &[PlannedView], offsets: &[u64], image_bytes: u64) -> (String, Value) {
    let mut placed: Vec<(u64, &PlannedView)> = offsets.iter().copied().zip(views).collect();
    placed.sort_by_key(|(offset, _)| *offset);
    let mut lines = format!("image bytes={image_bytes} views={}\n", views.len());
    let (mut pieces, mut padding) = (0, 0);
    for (index, (offset, view)) in placed.iter().enumerate() {
        let end = offset + view.bytes;
        let next = placed.get(index + 1).map_or(image_bytes, |(next, _)| *next);
        padding += next - end;
        lines += &format!(
            "view {} offset={offset} bytes={} padding={} dtype={} shape={:?} layout={}\n",
            view.name,
            view.bytes,
            next - end,
            view.dtype,
            view.shape,
            view.layout_id
        );
        let mut ordered: Vec<_> = view.pieces.iter().collect();
        ordered.sort_by_key(|piece| piece.offset);
        for piece in ordered {
            pieces += 1;
            // The pre-port planner copied scales unchecked; a checked copy
            // stores the same bytes.
            let transform = match piece.transform {
                WeightTransform::CopyScales => WeightTransform::Copy,
                ref transform => transform.clone(),
            };
            lines += &format!(
                "piece offset={} bytes={} transform={} params={} sources={:?} axis={:?} ranges={:?} stored={:?}\n",
                offset + piece.offset,
                piece.bytes,
                transform.id(),
                sha256_bytes(&transform.encode_params().unwrap()),
                piece.sources,
                piece.axis,
                piece.ranges,
                piece.stored_shape
            );
        }
    }
    let first = lines.lines().nth(1).unwrap_or_default().to_owned();
    let last = lines
        .lines()
        .rev()
        .find(|line| line.starts_with("view "))
        .unwrap_or_default()
        .to_owned();
    let summary = serde_json::json!({
        "arena_bytes": image_bytes,
        "views": views.len(),
        "pieces": pieces,
        "padding_bytes": padding,
        "first_view": first,
        "last_view": last,
        "sha256": sha256_bytes(lines.as_bytes()),
    });
    (lines, summary)
}

fn goldens() -> Value {
    serde_json::from_slice(&std::fs::read(GOLDENS).unwrap()).unwrap()
}

/// Every field in which one rank's image differs from the recorded one.
fn differences(image: &str, tp: Tp, rank: usize, summary: &Value) -> Vec<String> {
    let goldens = goldens();
    let recorded = &goldens["images"][image][tp.count().to_string()][rank];
    let fields = summary.as_object().unwrap();
    assert_eq!(recorded.as_object().map(|r| r.len()), Some(fields.len()));
    fields
        .iter()
        .filter(|(field, value)| recorded[field.as_str()] != **value)
        .map(|(field, value)| {
            format!(
                "{image} tp{} rank {rank} {field}: {value}, recorded {}",
                tp.count(),
                recorded[field.as_str()]
            )
        })
        .collect()
}

/// Every rank's text, MTP and DFlash image at TP2 and TP4 holds the views,
/// offsets, padding, arena length and pieces the pre-port planner laid out.
#[test]
fn images_equal_the_pre_port_planner_at_every_rank() {
    let headers = Headers::new();
    let goldens = goldens();
    let jobs: Vec<(&str, Tp, usize)> = IMAGES
        .into_iter()
        .flat_map(|image| {
            Tp::SUPPORTED
                .into_iter()
                .flat_map(move |tp| (0..tp.count()).map(move |rank| (image, tp, rank)))
        })
        .collect();
    for image in IMAGES {
        for tp in Tp::SUPPORTED {
            assert_eq!(
                goldens["images"][image][tp.count().to_string()]
                    .as_array()
                    .map(Vec::len),
                Some(tp.count())
            );
        }
    }
    let found: Vec<String> = std::thread::scope(|threads| {
        let headers = &headers;
        let running: Vec<_> = jobs
            .into_iter()
            .map(|(image, tp, rank)| {
                threads.spawn(move || {
                    let views = headers.views(image, rank, tp);
                    let (offsets, image_bytes) = sequential_offsets(&views);
                    let (_, summary) = canonical(&views, &offsets, image_bytes);
                    differences(image, tp, rank, &summary)
                })
            })
            .collect();
        running
            .into_iter()
            .flat_map(|job| job.join().unwrap())
            .collect()
    });
    assert!(found.is_empty(), "{found:#?}");
}

/// Every block-FP8 scale piece of the text and MTP images checks its scales
/// as it streams; no other piece does.
#[test]
fn block_scale_pieces_check_their_scales() {
    let headers = Headers::new();
    for image in ["text", "mtp"] {
        for tp in Tp::SUPPORTED {
            let mut checked = 0;
            for view in headers.views(image, 0, tp) {
                for piece in &view.pieces {
                    let scales = piece.sources[0].ends_with(".weight_scale_inv");
                    assert_eq!(
                        piece.transform == WeightTransform::CopyScales,
                        scales,
                        "{image} {}: {:?}",
                        view.name,
                        piece.transform
                    );
                    checked += usize::from(scales);
                }
            }
            assert!(checked > 0, "{image} tp{} checks no scales", tp.count());
        }
    }
}

/// Two views that trade places, a view moved by one alignment step (its
/// predecessor's padding grows) and a longer arena are each not the
/// recorded image.
#[test]
fn a_swapped_view_shifted_offset_or_longer_arena_is_detected() {
    let headers = Headers::new();
    let tp = Tp::SUPPORTED[1];
    let mut views = headers.views("dflash", 1, tp);
    let (offsets, image_bytes) = sequential_offsets(&views);
    let check = |views: &[PlannedView], offsets: &[u64], image_bytes: u64| {
        differences("dflash", tp, 1, &canonical(views, offsets, image_bytes).1)
    };
    assert!(check(&views, &offsets, image_bytes).is_empty());
    let alignment = model_mimo26_contract::weight_plan::VIEW_ALIGNMENT;
    let mut shifted = offsets.clone();
    for offset in &mut shifted[1..] {
        *offset += alignment;
    }
    let found = check(&views, &shifted, image_bytes + alignment);
    assert!(
        found.iter().any(|f| f.contains(" padding_bytes:"))
            && found.iter().any(|f| f.contains(" sha256:")),
        "{found:#?}"
    );
    let found = check(&views, &offsets, image_bytes + alignment);
    assert!(
        found.iter().any(|f| f.contains(" arena_bytes:"))
            && found.iter().any(|f| f.contains(" sha256:")),
        "{found:#?}"
    );
    views.swap(0, 1);
    let (offsets, image_bytes) = sequential_offsets(&views);
    let found = check(&views, &offsets, image_bytes);
    assert!(
        found.iter().any(|f| f.contains(" first_view:"))
            && found.iter().any(|f| f.contains(" sha256:")),
        "{found:#?}"
    );
}

/// An encoder image planned from its tensor list holds, at each view, the
/// bytes the encoder kernels read: GEMM weights stacked as stored, their
/// biases stacked with zeros for a weight without one, a convolution weight
/// with its taps before its channels, sinks widened to F32 and codebooks
/// preserved bit for bit in F32.
#[test]
fn encoder_plans_lay_out_the_tensors_the_kernels_read() {
    use model_mimo26_contract::encoders::{Part, Prepared};
    let mut main = Synthetic::new("renamed-encoder.safetensors");
    main.add("a.weight", DType::Bf16, &[32, 48]);
    main.add("a.bias", DType::Bf16, &[32]);
    main.add("b.weight", DType::Bf16, &[16, 48]);
    main.add("c.weight", DType::Bf16, &[4, 3, 2]);
    main.add("d.sinks", DType::Bf16, &[8]);
    let codebook = "encoder.quantizer.vq.layers.0._codebook.embed";
    main.add(codebook, DType::F32, &[8, 8]);
    // Values whose low bits narrowing would lose, including NaN payloads
    // and a negative subnormal, must survive the F32 codebook copy.
    let cases = [
        0x3f80_8000_u32,
        0x3f81_8000,
        0x7f7f_ffff,
        0x7f80_0001,
        0xffc1_2345,
        0x8000_0001,
    ];
    let start = main.tensors[codebook].file_start as usize;
    for (index, bits) in cases.into_iter().enumerate() {
        main.data[start + index * 4..start + index * 4 + 4].copy_from_slice(&bits.to_le_bytes());
    }
    let tensor = |name: &str, dtype, shape: &[u64], parts: Vec<Part>| Prepared {
        name: name.into(),
        dtype,
        shape: shape.to_vec(),
        parts,
    };
    let prepared = vec![
        tensor(
            "a.weight",
            "BF16",
            &[48, 48],
            vec![Part::Copy("a.weight".into()), Part::Copy("b.weight".into())],
        ),
        tensor(
            "a.bias",
            "BF16",
            &[48],
            vec![Part::Copy("a.bias".into()), Part::Zeros(32)],
        ),
        tensor(
            "c.weight",
            "BF16",
            &[4, 2, 3],
            vec![Part::Taps("c.weight".into())],
        ),
        tensor(
            "d.sinks",
            "F32",
            &[8],
            vec![Part::Widened("d.sinks".into())],
        ),
        tensor(
            "@codebooks",
            "F32",
            &[8, 8],
            vec![Part::Copy(codebook.into())],
        ),
    ];
    let views = prepared_views(prepared, &main.tensors).unwrap();
    let image = main.image(&views);
    let (offsets, _) = sequential_offsets(&views);
    let stored = |name: &str| {
        let t = &main.tensors[name];
        &main.data[t.file_start as usize..(t.file_start + t.bytes) as usize]
    };
    let view = |name: &str| {
        let at = views.iter().position(|view| view.name == name).unwrap();
        &image[offsets[at] as usize..(offsets[at] + views[at].bytes) as usize]
    };
    assert_eq!(
        view("a.weight"),
        [stored("a.weight"), stored("b.weight")].concat()
    );
    assert_eq!(view("a.bias"), [stored("a.bias"), &[0; 32]].concat());
    let conv = stored("c.weight");
    let mut taps = vec![0; conv.len()];
    for (out, channel, tap) in
        (0..4).flat_map(|o| (0..3).flat_map(move |c| (0..2).map(move |t| (o, c, t))))
    {
        let from = ((out * 3 + channel) * 2 + tap) * 2;
        let to = ((out * 2 + tap) * 3 + channel) * 2;
        taps[to..to + 2].copy_from_slice(&conv[from..from + 2]);
    }
    assert_eq!(view("c.weight"), taps);
    let widened: Vec<u8> = stored("d.sinks")
        .chunks_exact(2)
        .flat_map(|x| (u32::from(u16::from_le_bytes([x[0], x[1]])) << 16).to_le_bytes())
        .collect();
    assert_eq!(view("d.sinks"), widened);
    assert_eq!(view("@codebooks"), stored(codebook));
    // The contract's own tensor list plans only from every encoder tensor.
    assert!(encoder_views_flash(&main.tensors).is_err());
}
