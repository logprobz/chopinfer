#![deny(unsafe_code)]

use std::sync::Mutex;

use runtime_bundle_abi::*;

const HANDLE: BundleHandleV1 = BundleHandleV1::from_raw(0x1_0000_0001);
static ACCEPTED: Mutex<Vec<BundleKeptRowsV1>> = Mutex::new(Vec::new());

fn create(_: HostServicesV1, _: &[u8]) -> BundleCreateResultV1 {
    BundleCreateResultV1::success(HANDLE)
}

fn execute(_: BundleHandleV1, _: &[u8], _: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn commit(_: BundleHandleV1) -> BundleStatusV1 {
    BundleStatusV1::InvalidState
}

fn success(_: BundleHandleV1) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn retire(_: BundleHandleV1, _: u64) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn commit_accepted(_: BundleHandleV1, records: &[BundleKeptRowsV1]) -> BundleStatusV1 {
    ACCEPTED.lock().unwrap().extend_from_slice(records);
    BundleStatusV1::Success
}

export_bundle_api_v1!(
    with_speculation BASE_API, EXTENDED_API,
    create, execute, commit, success, success, retire, commit_accepted,
);

struct Sink;

impl LogitsRowSinkV1 for Sink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        ResultSinkLimitsV1::new(8, 8, 64)
    }

    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Ok(())
    }
}

#[test]
fn speculation_export_preserves_prefix_and_forwards_kept_rows() {
    assert_eq!(chop_bundle_get_api_v1(), &raw const EXTENDED_API.base.base);
    assert_eq!(
        BASE_API.struct_size as usize,
        std::mem::size_of::<BundleApiWithSpeculationV1>()
    );
    assert!(!BundleApiV1::new(BASE_API).unwrap().supports_speculation());

    let api = BundleApiV1::new_with_speculation(EXTENDED_API).unwrap();
    assert!(api.supports_speculation());
    assert!(api.supports_request_retirement());
    let host = HostServicesTableV1::new(std::rc::Rc::new(UnsupportedHostServicesBackendV1));
    let mut instance = api.create(host, &[]).unwrap();
    instance.execute(&[], &mut Sink).unwrap();
    let kept = BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id: 41,
        kept_rows: 2,
        reserved_tail: 0,
    };
    instance.commit_accepted(&[kept]).unwrap();
    assert_eq!(*ACCEPTED.lock().unwrap(), vec![kept]);
    instance.retire_request(41).unwrap();
    instance.destroy().unwrap();
}

#[test]
fn speculation_tail_needs_its_full_size_and_valid_records() {
    let mut short = EXTENDED_API;
    short.base.base.struct_size =
        std::mem::offset_of!(BundleApiWithSpeculationV1, commit_accepted_v1) as u32;
    assert!(
        !BundleApiV1::new_with_speculation(short)
            .unwrap()
            .supports_speculation()
    );
    let mut absent = EXTENDED_API;
    absent.commit_accepted_v1 = None;
    assert!(
        !BundleApiV1::new_with_speculation(absent)
            .unwrap()
            .supports_speculation()
    );

    let mut kept = BundleKeptRowsV1 {
        struct_size: std::mem::size_of::<BundleKeptRowsV1>() as u32,
        reserved: 0,
        request_id: 41,
        kept_rows: 1,
        reserved_tail: 0,
    };
    validate_kept_rows_v1(kept).unwrap();
    kept.request_id = 0;
    validate_kept_rows_v1(kept).unwrap();
    kept.struct_size += 8;
    assert!(matches!(
        validate_kept_rows_v1(kept),
        Err(BundleAbiError::StructSizeExact { .. })
    ));
    kept.struct_size -= 8;
    kept.kept_rows = 0;
    assert_eq!(
        validate_kept_rows_v1(kept),
        Err(BundleAbiError::InvalidKeptRows)
    );
}

#[test]
fn draft_record_accepts_a_forward_tail_and_refuses_short_or_wrong_bytes() {
    let bytes = 7_u32.to_le_bytes();
    let mut draft = BundleDraftTokensV1 {
        struct_size: std::mem::size_of::<BundleDraftTokensV1>() as u32,
        reserved: 0,
        request_id: 41,
        token_count: 1,
        token_bytes: BundleByteViewV1::from_bytes(&bytes),
    };
    validate_draft_tokens_v1(draft).unwrap();
    draft.struct_size += 8;
    validate_draft_tokens_v1(draft).unwrap();
    draft.struct_size = 3;
    assert!(matches!(
        validate_draft_tokens_v1(draft),
        Err(BundleAbiError::StructSize { .. })
    ));
    draft.struct_size = std::mem::size_of::<BundleDraftTokensV1>() as u32;
    draft.token_count = 2;
    assert_eq!(
        validate_draft_tokens_v1(draft),
        Err(BundleAbiError::InvalidDraftTokens)
    );
}
