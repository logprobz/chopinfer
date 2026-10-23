use super::*;
use crate::weight_transform::{MXFP8_FOLD_PACKED, SourceTensor, WeightTransform};

fn codes(bytes: usize) -> Vec<u8> {
    (0..bytes)
        .map(|i| {
            let byte = (i.wrapping_mul(2_654_435_761) >> 17) as u8;
            if byte & 0x7f == 0x7f { byte ^ 1 } else { byte }
        })
        .collect()
}

fn record(layout: Mxfp8FoldLayout) -> WeightRecord {
    let shape = vec![layout.groups as u64, layout.n as u64, layout.k as u64];
    WeightRecord {
        transform: WeightTransform::Mxfp8FoldPacked,
        sources: vec![SourceTensor {
            dtype: "F8_E4M3".into(),
            shape: shape.clone(),
            byte_length: layout.packed_bytes().unwrap() as u64,
        }],
        axis: None,
        ranges: vec![],
        stored_shape: shape,
    }
}

/// Transcribe the prepare kernel's output-word index mapping from
/// kernels-sm120/cuda/gemm/mxfp8_fold_stream.cuh:32-48. Lines 37-44 derive
/// within, fold, words_per_lane, word, lane, tile, row and K in this order.
fn cuda_prepare_model(layout: Mxfp8FoldLayout, source: &[u8]) -> Vec<u8> {
    let k = layout.k;
    let folds = k.div_ceil(128);
    let words = layout.groups * layout.n * k / 4;
    let mut out = vec![0xa5; words * 4];
    let mut seen = vec![false; words];
    for i in 0..words {
        let within = i % (2 * k);
        let fold = within / 256;
        let words_per_lane = if k % 128 != 0 && fold == folds - 1 {
            4
        } else {
            8
        };
        let word = (within % 256) % words_per_lane;
        let lane = (within % 256) / words_per_lane;
        let tile = i / (2 * k);
        let row = tile * 8 + lane / 4;
        let column = fold * 128 + (word / 2) * 32 + (word % 2) * 16 + (lane % 4) * 4;
        assert!(column < k);
        let at = row * k + column;
        assert!(!std::mem::replace(&mut seen[at / 4], true));
        out[i * 4..i * 4 + 4].copy_from_slice(&source[at..at + 4]);
    }
    assert!(seen.into_iter().all(|visited| visited));
    out
}

#[test]
fn pack_and_oracle_equal_cuda_prepare_index_mapping() {
    for groups in [1, 2, 3] {
        for n in [8, 16, 24] {
            for k in [64, 128, 192, 256, 320, 512, 576, 1280, 5120, 6144] {
                let layout = Mxfp8FoldLayout { groups, n, k };
                // Unique four-byte words also witness the exact permutation,
                // independently of the finite E4M3 payload admission checks.
                let source: Vec<_> = (0..layout.packed_bytes().unwrap() / 4)
                    .flat_map(|word| (word as u32).to_le_bytes())
                    .collect();
                let expected = cuda_prepare_model(layout, &source);
                assert_eq!(layout.pack(&source).unwrap(), expected, "{layout:?}");
                assert_eq!(layout.oracle(&source).unwrap(), expected, "{layout:?}");
                assert_ne!(source, expected);
            }
        }
    }
}

#[test]
fn word_offset_finds_every_row_major_word_in_the_packed_codes() {
    for groups in [1, 3] {
        for n in [8, 24] {
            for k in [64, 128, 192, 576, 1280] {
                let layout = Mxfp8FoldLayout { groups, n, k };
                let source: Vec<_> = (0..layout.packed_bytes().unwrap() / 4)
                    .flat_map(|word| (word as u32).to_le_bytes())
                    .collect();
                let packed = layout.pack(&source).unwrap();
                let mut seen = vec![false; source.len() / 4];
                for row in 0..groups * n {
                    for column in (0..k).step_by(4) {
                        let at = layout.word_offset(row, column).unwrap();
                        assert_eq!(at % 4, 0);
                        assert!(!std::mem::replace(&mut seen[at / 4], true));
                        let from = row * k + column;
                        assert_eq!(packed[at..at + 4], source[from..from + 4], "{layout:?}");
                    }
                }
                assert!(seen.into_iter().all(|visited| visited));
            }
        }
    }
    let layout = Mxfp8FoldLayout {
        groups: 2,
        n: 8,
        k: 192,
    };
    for (row, column) in [(0, 2), (0, 192), (16, 0), (usize::MAX, 0)] {
        assert!(layout.word_offset(row, column).is_err());
    }
    assert!(
        Mxfp8FoldLayout {
            groups: 1,
            n: 7,
            k: 64
        }
        .word_offset(0, 0)
        .is_err()
    );
}

#[test]
fn closed_set_identity_and_shape_refusals() {
    let transform = WeightTransform::Mxfp8FoldPacked;
    assert_eq!(transform.id(), 10);
    assert_eq!(MXFP8_FOLD_PACKED, 10);
    assert!(transform.encode_params().unwrap().is_empty());
    assert_eq!(WeightTransform::decode(10, &[]).unwrap(), transform);
    assert!(WeightTransform::decode(10, &[0; 4]).is_err());
    assert!(WeightTransform::decode(11, &[]).is_err());
    assert_eq!(
        Mxfp8FoldLayout::weight_layout().id,
        "mxfp8-fold-packed-8x128-v1"
    );
    assert_eq!(Mxfp8FoldLayout::weight_layout().version, 1);
    let valid = Mxfp8FoldLayout {
        groups: 2,
        n: 8,
        k: 64,
    };
    for (groups, n, k) in [
        (0, 8, 64),
        (1, 0, 64),
        (1, 8, 0),
        (1, 7, 64),
        (1, 8, 96),
        (usize::MAX, 8, 64),
    ] {
        assert!(Mxfp8FoldLayout { groups, n, k }.packed_bytes().is_err());
    }
    for shape in [
        vec![512],
        vec![2, 4, 64],
        vec![1, 8, 96],
        vec![0, 8, 64],
        vec![u64::MAX, 8, 64],
    ] {
        let mut r = record(valid);
        r.stored_shape = shape;
        assert!(r.output_bytes().is_err());
        assert!(r.prepare(1024).is_err());
    }
    for dtype in ["BF16", "F32", "F8_E5M2"] {
        let mut r = record(valid);
        r.sources[0].dtype = dtype.into();
        assert!(r.prepare(1024).is_err());
    }
    for bad_source in [vec![u64::MAX, 8, 64], vec![0, 8, 64]] {
        let mut r = record(valid);
        r.sources[0].shape = bad_source;
        assert!(r.prepare(1024).is_err());
    }
    assert!(record(valid).prepare(1023).is_err());
    assert!(valid.pack(&[0; 1023]).is_err());
    assert!(valid.oracle(&[0; 1023]).is_err());
    assert!(valid.pack_strips(&[0; 512], 2, &mut [0; 512]).is_err());
    assert!(valid.pack_strips(&[0; 511], 0, &mut [0; 511]).is_err());
}

#[test]
fn streamed_ranges_and_gathers_equal_the_oracle() {
    for k in [64, 128, 192, 576] {
        let layout = Mxfp8FoldLayout {
            groups: 2,
            n: 24,
            k,
        };
        for axis in [None, Some(1), Some(2)] {
            let mut r = record(layout);
            if let Some(axis) = axis {
                r.axis = Some(axis);
                let width = r.stored_shape[axis as usize];
                r.sources[0].shape[axis as usize] += width;
                r.sources[0].byte_length *= 2;
                r.ranges = std::iter::once(width / 2..width / 2 + width).collect();
            }
            let source = codes(r.sources[0].byte_length as usize);
            let mut read = |index, offset, out: &mut [u8]| {
                assert_eq!(index, 0);
                let offset = offset as usize;
                out.copy_from_slice(&source[offset..offset + out.len()]);
                Ok(())
            };
            let mut copy = r.clone();
            copy.transform = WeightTransform::Copy;
            let bytes = r.output_bytes().unwrap();
            let gathered = copy
                .prepare(bytes)
                .unwrap()
                .produce_whole(&mut read)
                .unwrap();
            let expected = layout.oracle(&gathered).unwrap();
            let prepared = r.prepare(bytes).unwrap();
            assert!(!prepared.is_whole());
            assert_eq!(prepared.unit_bytes(), (8 * k) as u64);
            assert_eq!(prepared.produce_whole(&mut read).unwrap(), expected);
            for chunk in [1, 3, 257, 8 * k - 1, 8 * k, 8 * k + 3, 16 * k] {
                let mut actual = vec![0xa5; bytes as usize];
                let mut scratch = Vec::new();
                for (i, part) in actual.chunks_mut(chunk).enumerate() {
                    let start = (i * chunk) as u64;
                    let range = start..start + part.len() as u64;
                    for (source_id, offset, length) in
                        prepared.source_ranges(range.clone()).unwrap()
                    {
                        assert_eq!(source_id, 0);
                        assert!(offset + length <= source.len() as u64);
                    }
                    prepared
                        .produce(range, part, &mut read, &mut scratch)
                        .unwrap();
                }
                assert_eq!(actual, expected, "K={k}, axis={axis:?}, chunk={chunk}");
            }
            prepared
                .produce(bytes..bytes, &mut [], &mut read, &mut Vec::new())
                .unwrap();
            assert!(
                prepared
                    .produce(bytes..bytes + 1, &mut [0], &mut read, &mut Vec::new())
                    .is_err()
            );
        }
    }
}

#[test]
fn fold_record_refuses_nan_codes_in_every_supported_dtype() {
    let layout = Mxfp8FoldLayout {
        groups: 1,
        n: 8,
        k: 192,
    };
    for dtype in ["F8_E4M3", "U8", "I8"] {
        for nan in [0x7f, 0xff] {
            let mut r = record(layout);
            r.sources[0].dtype = dtype.into();
            let mut source = codes(layout.packed_bytes().unwrap());
            *source.last_mut().unwrap() = nan;
            let prepared = r.prepare(source.len() as u64).unwrap();
            assert!(
                prepared
                    .produce_whole(&mut |_, offset, out| {
                        let offset = offset as usize;
                        out.copy_from_slice(&source[offset..offset + out.len()]);
                        Ok(())
                    })
                    .is_err()
            );
        }
    }
}
