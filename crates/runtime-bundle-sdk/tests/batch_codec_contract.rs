use runtime_bundle_sdk::{
    BATCH_PHASE_PREFILL_V1, BATCH_REQUEST_MAGIC_V1, BATCH_REQUEST_MAGIC_V2, BATCH_REQUEST_MAGIC_V3,
    BatchAppendCopyV2, BatchAppendV1, BatchCodecV1, BatchItemV1, BatchMediaEncodeV3,
    BatchMediaInjectV3, BatchMediaSectionV3, BatchReadErrorV1, BatchRequestV1, BatchWireVersionV1,
    BatchWriteErrorV1, MAXIMUM_BATCH_WIRE_ELEMENTS_V1,
};

fn request(version: BatchWireVersionV1, copy: Option<BatchAppendCopyV2>) -> BatchRequestV1 {
    BatchRequestV1::new(
        version,
        vec![BatchItemV1::new(9, false, 2, true, vec![5, 6])],
        vec![BatchAppendV1::new(0, 2, 2, vec![4], copy)],
    )
}

fn v1_bytes() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(BATCH_REQUEST_MAGIC_V1);
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&9_u64.to_le_bytes());
    bytes.push(BATCH_PHASE_PREFILL_V1);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.push(1);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&5_u32.to_le_bytes());
    bytes.extend_from_slice(&6_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&4_u64.to_le_bytes());
    bytes
}

#[test]
fn v1_decoder_and_encoder_preserve_the_original_bytes_exactly() {
    let expected = v1_bytes();
    let decoded = BatchCodecV1::decode_v1(&expected).expect("the V1 fixture decodes");

    assert_eq!(decoded, request(BatchWireVersionV1::V1, None));
    assert_eq!(decoded.encode().expect("the V1 fixture encodes"), expected);
    assert_eq!(BatchCodecV1::decode(&expected), Ok(decoded));
}

#[test]
fn v2_extends_each_append_with_optional_copy_metadata() {
    let copy = BatchAppendCopyV2::new(4, 7, 3);
    let encoded = request(BatchWireVersionV1::V2, Some(copy))
        .encode()
        .expect("the V2 fixture encodes");
    let mut expected = v1_bytes();
    expected[..BATCH_REQUEST_MAGIC_V1.len()].copy_from_slice(BATCH_REQUEST_MAGIC_V2);
    expected.push(1);
    expected.extend_from_slice(&4_u64.to_le_bytes());
    expected.extend_from_slice(&7_u64.to_le_bytes());
    expected.extend_from_slice(&3_u64.to_le_bytes());

    assert_eq!(encoded, expected);
    assert_eq!(
        BatchCodecV1::decode(&encoded),
        Ok(request(BatchWireVersionV1::V2, Some(copy)))
    );

    let without_copy = request(BatchWireVersionV1::V2, None)
        .encode()
        .expect("the V2 append without a copy encodes");
    assert_eq!(without_copy.last(), Some(&0));
    assert_eq!(
        BatchCodecV1::decode(&without_copy),
        Ok(request(BatchWireVersionV1::V2, None))
    );
}

#[test]
fn prompt_rows_use_a_v2_item_flag_and_v1_refuses_it() {
    let item = BatchItemV1::new(9, false, 2, true, vec![5, 6]).with_prompt_rows(true);
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V2,
        vec![item.clone()],
        vec![BatchAppendV1::new(0, 2, 2, vec![4], None)],
    );
    let encoded = request.encode().unwrap();
    assert_eq!(BatchCodecV1::decode(&encoded), Ok(request));
    assert!(BatchCodecV1::decode_v1(&encoded).is_err());
    assert_eq!(
        BatchRequestV1::new(
            BatchWireVersionV1::V1,
            vec![item],
            vec![BatchAppendV1::new(0, 2, 2, vec![4], None)]
        )
        .encode(),
        Err(BatchWriteErrorV1::PromptRowsRequireV2)
    );
}

#[test]
fn v1_refuses_copy_metadata_and_versioned_decode_refuses_bad_extensions() {
    assert_eq!(
        request(
            BatchWireVersionV1::V1,
            Some(BatchAppendCopyV2::new(4, 7, 3)),
        )
        .encode(),
        Err(BatchWriteErrorV1::CopyRequiresV2)
    );

    let mut bad_flag = request(BatchWireVersionV1::V2, None)
        .encode()
        .expect("the V2 fixture encodes");
    *bad_flag.last_mut().expect("the copy flag is present") = 2;
    assert_eq!(
        BatchCodecV1::decode(&bad_flag),
        Err(BatchReadErrorV1::UnknownFlag {
            surface: "batch append copy",
            actual: 2,
        })
    );

    let truncated = &bad_flag[..bad_flag.len() - 1];
    assert!(matches!(
        BatchCodecV1::decode(truncated),
        Err(BatchReadErrorV1::Truncated { .. })
    ));
}

#[test]
fn decoder_bounds_collection_lengths_before_allocation() {
    let mut bytes = Vec::from(BATCH_REQUEST_MAGIC_V1.as_slice());
    bytes.extend_from_slice(&(MAXIMUM_BATCH_WIRE_ELEMENTS_V1 + 1).to_le_bytes());

    assert_eq!(
        BatchCodecV1::decode(&bytes),
        Err(BatchReadErrorV1::CountOutOfRange {
            surface: "batch item",
            actual: u64::from(MAXIMUM_BATCH_WIRE_ELEMENTS_V1 + 1),
        })
    );
}

#[test]
fn v3_round_trips_bundle_and_host_drafts_with_the_shared_media_section() {
    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 7,
            geometry: [2, 3, 4, 5],
            output_rows: 6,
            pages: vec![10, 11],
            payload_offset: 1,
            payload_bytes: 3,
        }],
        injects: vec![BatchMediaInjectV3 {
            item_index: 2,
            item_row: 4,
            rows: 2,
            arena_row: 64,
        }],
        payload: vec![9, 8, 7, 6],
    };
    let request = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![
            BatchItemV1::verify(1, 3, true, vec![12]),
            BatchItemV1::verify(2, 2, false, vec![20, 21]),
            BatchItemV1::new(3, false, 6, true, vec![1, 2, 3, 4, 5, 6]),
        ],
        vec![
            BatchAppendV1::new(8, 3, 11, vec![1], None),
            BatchAppendV1::new(16, 2, 18, vec![2], Some(BatchAppendCopyV2::new(2, 3, 16))),
            BatchAppendV1::new(0, 6, 6, vec![4], None),
        ],
    )
    .with_media(media);
    let bytes = request.encode().unwrap();
    assert!(bytes.starts_with(BATCH_REQUEST_MAGIC_V3));
    assert_eq!(BatchCodecV1::decode(&bytes), Ok(request.clone()));
    assert_eq!(
        BatchCodecV1::decode(&bytes).unwrap().encode().unwrap(),
        bytes
    );
    assert!(BatchCodecV1::decode_v1(&bytes).is_err());

    let mut truncated = bytes;
    truncated.pop();
    assert!(matches!(
        BatchCodecV1::decode(&truncated),
        Err(BatchReadErrorV1::Truncated { .. })
    ));
}

#[test]
fn v2_refuses_verify_phase_and_bundle_draft_flag() {
    let item = BatchItemV1::verify(4, 2, true, vec![17]);
    let append = BatchAppendV1::new(4, 2, 6, vec![1], None);
    assert_eq!(
        BatchRequestV1::new(
            BatchWireVersionV1::V2,
            vec![item.clone()],
            vec![append.clone()]
        )
        .encode(),
        Err(BatchWriteErrorV1::VerifyRequiresV3)
    );
    let mut encoded = BatchRequestV1::new(BatchWireVersionV1::V3, vec![item], vec![append])
        .encode()
        .unwrap();
    encoded[..BATCH_REQUEST_MAGIC_V3.len()].copy_from_slice(BATCH_REQUEST_MAGIC_V2);
    assert_eq!(
        BatchCodecV1::decode(&encoded),
        Err(BatchReadErrorV1::UnknownPhase { actual: 2 })
    );
    encoded[24] = 1;
    assert!(matches!(
        BatchCodecV1::decode(&encoded),
        Err(BatchReadErrorV1::UnknownFlag { .. })
    ));
}

#[test]
fn verify_item_refuses_inconsistent_rows_and_media_offsets() {
    let append = BatchAppendV1::new(0, 3, 3, vec![1], None);
    let bad = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(9, 3, false, vec![1, 2])],
        vec![append.clone()],
    );
    assert_eq!(bad.encode(), Err(BatchWriteErrorV1::InvalidVerify));

    let media = BatchMediaSectionV3 {
        encodes: vec![BatchMediaEncodeV3 {
            encoder: 1,
            geometry: [1; 4],
            output_rows: 1,
            pages: vec![1],
            payload_offset: u64::MAX,
            payload_bytes: 2,
        }],
        injects: vec![],
        payload: vec![1, 2],
    };
    let bad = BatchRequestV1::new(
        BatchWireVersionV1::V3,
        vec![BatchItemV1::verify(9, 3, true, vec![1])],
        vec![append],
    )
    .with_media(media);
    assert_eq!(bad.encode(), Err(BatchWriteErrorV1::InvalidMedia));
}

#[test]
fn an_anchor_marks_a_v3_prompt_ending_prefill_item() {
    let item = BatchItemV1::new(9, false, 70, true, (0..70).collect()).with_anchor(true);
    let append = || vec![BatchAppendV1::new(0, 70, 70, vec![4, 5], None)];
    let request = BatchRequestV1::new(BatchWireVersionV1::V3, vec![item.clone()], append());
    let encoded = request.encode().unwrap();
    let decoded = BatchCodecV1::decode(&encoded).unwrap();
    assert!(decoded.items()[0].anchor());
    assert_eq!(decoded, request);
    // V2 has no anchor flag, and a chunk without the prompt's logits row or
    // a decode item cannot carry one.
    assert_eq!(
        BatchRequestV1::new(BatchWireVersionV1::V2, vec![item], append()).encode(),
        Err(BatchWriteErrorV1::InvalidAnchor)
    );
    for item in [
        BatchItemV1::new(9, false, 70, false, (0..70).collect()).with_anchor(true),
        BatchItemV1::new(9, true, 1, true, vec![3]).with_anchor(true),
    ] {
        let rows = u64::from(item.phase_tokens());
        assert_eq!(
            BatchRequestV1::new(
                BatchWireVersionV1::V3,
                vec![item],
                vec![BatchAppendV1::new(0, rows, rows, vec![4, 5], None)]
            )
            .encode(),
            Err(BatchWriteErrorV1::InvalidAnchor)
        );
    }
    let mut refused = encoded.clone();
    // Clear the logits flag the anchor needs: item flags follow the magic,
    // the item count, the request ID, the phase and its tokens.
    let flags = 12 + 4 + 8 + 1 + 4;
    assert_eq!(refused[flags], 1 | 8);
    refused[flags] = 8;
    assert_eq!(
        BatchCodecV1::decode(&refused),
        Err(BatchReadErrorV1::InvalidAnchor)
    );
}
