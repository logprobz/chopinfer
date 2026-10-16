#![deny(unsafe_code)]

use std::sync::Mutex;

use runtime_bundle_abi::*;

const HANDLE: BundleHandleV1 = BundleHandleV1::from_raw(0x1_0000_0001);
static CALLS: Mutex<Vec<(&str, PrefixStateRequestV1)>> = Mutex::new(Vec::new());

fn create(_: HostServicesV1, _: &[u8]) -> BundleCreateResultV1 {
    BundleCreateResultV1::success(HANDLE)
}

fn execute(_: BundleHandleV1, _: &[u8], _: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn success(_: BundleHandleV1) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn retire(_: BundleHandleV1, _: u64) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn commit_accepted(_: BundleHandleV1, _: &[BundleKeptRowsV1]) -> BundleStatusV1 {
    BundleStatusV1::Success
}

fn describe(_: BundleHandleV1) -> BundlePrefixStateInfoV1 {
    BundlePrefixStateInfoV1::success(3)
}

fn capture(_: BundleHandleV1, request: PrefixStateRequestV1) -> BundleStatusV1 {
    let refused = request.request_id == 99;
    CALLS.lock().unwrap().push(("capture", request));
    if refused {
        BundleStatusV1::ResourceExhausted
    } else {
        BundleStatusV1::Success
    }
}

fn restore(_: BundleHandleV1, request: PrefixStateRequestV1) -> BundleStatusV1 {
    CALLS.lock().unwrap().push(("restore", request));
    BundleStatusV1::Success
}

export_bundle_api_v1!(
    with_prefix_cache BASE_API, EXTENDED_API,
    create, execute, success, success, success, retire, commit_accepted,
    describe, capture, restore,
);

#[test]
fn prefix_cache_export_decodes_records_and_refusals_do_not_poison() {
    assert_eq!(
        chop_bundle_get_api_v1(),
        &raw const EXTENDED_API.base.base.base
    );
    assert!(
        !BundleApiV1::new_with_speculation(EXTENDED_API.base)
            .unwrap()
            .supports_prefix_cache()
    );
    let api = BundleApiV1::new_with_prefix_cache(EXTENDED_API).unwrap();
    assert!(api.supports_prefix_cache());
    let host = HostServicesTableV1::new(std::rc::Rc::new(UnsupportedHostServicesBackendV1));
    let mut instance = api.create(host, &[]).unwrap();
    assert_eq!(instance.describe_prefix_state().unwrap(), 3);
    instance.capture_prefix_state(41, 70, &[5, 6, 7]).unwrap();
    instance
        .restore_prefix_state(42, &[1, 2, 3], &[5, 6, 7])
        .unwrap();
    assert_eq!(
        instance.capture_prefix_state(99, 1, &[8]),
        Err(BundleAbiError::Status(BundleStatusV1::ResourceExhausted))
    );
    assert!(!instance.is_poisoned(), "a refused capture changes nothing");
    assert_eq!(
        *CALLS.lock().unwrap(),
        vec![
            (
                "capture",
                PrefixStateRequestV1 {
                    request_id: 41,
                    tokens: 70,
                    token_ids: Vec::new(),
                    pages: vec![5, 6, 7],
                },
            ),
            (
                "restore",
                PrefixStateRequestV1 {
                    request_id: 42,
                    tokens: 3,
                    token_ids: vec![1, 2, 3],
                    pages: vec![5, 6, 7],
                },
            ),
            (
                "capture",
                PrefixStateRequestV1 {
                    request_id: 99,
                    tokens: 1,
                    token_ids: Vec::new(),
                    pages: vec![8],
                },
            ),
        ]
    );
    instance.destroy().unwrap();
}

#[test]
fn prefix_cache_tail_needs_its_full_size_and_every_callback() {
    let mut short = EXTENDED_API;
    short.base.base.base.struct_size =
        std::mem::offset_of!(BundleApiWithPrefixCacheV1, restore_prefix_state_v1) as u32;
    assert!(
        !BundleApiV1::new_with_prefix_cache(short)
            .unwrap()
            .supports_prefix_cache()
    );
    let mut absent = EXTENDED_API;
    absent.capture_prefix_state_v1 = None;
    assert!(
        !BundleApiV1::new_with_prefix_cache(absent)
            .unwrap()
            .supports_prefix_cache()
    );
}
