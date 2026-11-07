use super::*;
use runtime_bundle_abi::weight_plan::{CENSUS_SCALE_ROW_OFFSET_V1, CensusItemV1, WeightRecordV1};

fn measured() -> MeasuredV1 {
    MeasuredV1 {
        answer: vec![
            CensusRankV1 {
                rank: 0,
                document_sha256: [1; 32],
                counts: vec![[3, 2, 1], [0, 0, 0]],
            },
            CensusRankV1 {
                rank: 1,
                document_sha256: [2; 32],
                counts: vec![],
            },
        ],
        unique: 2,
        bytes_read: 4096,
    }
}

fn documents() -> Vec<(u32, WeightCensusV1, Vec<u8>)> {
    let item = CensusItemV1 {
        kind: CENSUS_SCALE_ROW_OFFSET_V1,
        rows: 2,
        groups: 2,
        record: WeightRecordV1 {
            arena_offset: 0,
            byte_length: 4,
            transform: 1,
            params: vec![],
            sources: vec![],
            axis: None,
            ranges: vec![],
            stored_shape: vec![],
        },
    };
    (0..2)
        .map(|rank| {
            let census = WeightCensusV1 {
                ranks: 2,
                rank,
                items: vec![item.clone(); 2 - rank as usize],
            };
            let bytes = census.encode();
            (rank, census, bytes)
        })
        .collect()
}

fn digests(documents: &DocumentsV1) -> Vec<(u32, [u8; 32])> {
    documents
        .iter()
        .map(|(rank, _, bytes)| (*rank, digest_v1(bytes)))
        .collect()
}

#[test]
fn entries_round_trip_and_refuse_any_other_length() {
    let measured = measured();
    let bytes = encode_v1(&measured);
    assert_eq!(decode_v1(&bytes), Some(measured));
    for length in 0..bytes.len() {
        assert_eq!(decode_v1(&bytes[..length]), None, "{length} bytes");
    }
    let mut longer = bytes.clone();
    longer.push(0);
    assert_eq!(decode_v1(&longer), None);
    // A count past the entry's end is refused, not allocated.
    let mut huge = bytes;
    huge[20 + 36..20 + 44].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(decode_v1(&huge), None);
}

/// The checkpoint identity, the implementation and every stamp name the
/// files; neither key's tag is the other's.
#[test]
fn every_file_input_changes_the_prefix() {
    let identity = [7; 32];
    let shared = [8; 32];
    let stamps = vec!["a.safetensors 1 2 3 4.000000005 6.000000007".to_owned()];
    let base = files_v1(DOCUMENTS_TAG, &identity, &shared, &stamps);
    let mut other_stamps = stamps.clone();
    other_stamps[0].push('0');
    let mut more_stamps = stamps.clone();
    more_stamps.push(String::new());
    for prefix in [
        files_v1(REQUESTS_TAG, &identity, &shared, &stamps),
        files_v1(DOCUMENTS_TAG, &[6; 32], &shared, &stamps),
        files_v1(DOCUMENTS_TAG, &identity, &[0; 32], &stamps),
        files_v1(DOCUMENTS_TAG, &identity, &shared, &other_stamps),
        files_v1(DOCUMENTS_TAG, &identity, &shared, &more_stamps),
    ] {
        assert_ne!(prefix, base);
    }
    assert!(!DOCUMENTS_TAG.starts_with(REQUESTS_TAG) && !REQUESTS_TAG.starts_with(DOCUMENTS_TAG));
}

#[test]
fn stored_counts_answer_only_their_own_documents() {
    let documents = documents();
    let digests = digests(&documents);
    let answer = |documents: &DocumentsV1| MeasuredV1 {
        answer: documents
            .iter()
            .map(|(rank, census, bytes)| CensusRankV1 {
                rank: *rank,
                document_sha256: digest_v1(bytes),
                counts: vec![[4, 1, 0]; census.items.len()],
            })
            .collect(),
        unique: 1,
        bytes_read: 4,
    };
    let stored = answer(&documents);
    assert!(answers_v1(&stored, &documents, &digests));
    let changes: Vec<fn(&mut MeasuredV1)> = vec![
        |stored| stored.answer[0].rank = 1,
        |stored| stored.answer[1].document_sha256[0] ^= 1,
        |stored| {
            stored.answer[0].counts.pop();
        },
        |stored| stored.answer[1].counts.push([0; 3]),
        // More exceptions than the item's 2 x 2 scales.
        |stored| stored.answer[0].counts[1][2] = 5,
        |stored| {
            stored.answer.pop();
        },
        |stored| stored.answer.swap(0, 1),
    ];
    for change in changes {
        let mut wrong = stored.clone();
        change(&mut wrong);
        assert!(!answers_v1(&wrong, &documents, &digests), "{wrong:?}");
    }
}

#[test]
fn counts_stored_for_requests_name_the_load_s_ranks_in_order() {
    let stored = measured();
    assert!(ranked_v1(&stored, 2));
    assert!(!ranked_v1(&stored, 1));
    let mut swapped = stored.clone();
    swapped.answer.swap(0, 1);
    assert!(!ranked_v1(&swapped, 2));
    let mut repeated = stored.clone();
    repeated.answer[1].rank = 0;
    assert!(!ranked_v1(&repeated, 2));
    let mut sparse = stored;
    sparse.answer.remove(0);
    assert!(ranked_v1(&sparse, 2));
}
