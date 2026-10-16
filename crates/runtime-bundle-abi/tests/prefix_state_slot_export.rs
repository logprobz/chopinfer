#![deny(unsafe_code)]

use std::sync::Mutex;

use runtime_bundle_abi::*;

const HANDLE: BundleHandleV1 = BundleHandleV1::from_raw(0x1_0000_0001);
static CALLS: Mutex<Vec<(&str, PrefixStateRequestV1, Option<u64>)>> = Mutex::new(Vec::new());

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

fn describe(_: BundleHandleV1) -> BundlePrefixStateInfoV1 {
    BundlePrefixStateInfoV1::success(35)
}

fn capture(_: BundleHandleV1, request: PrefixStateRequestV1) -> BundleStatusV1 {
    CALLS.lock().unwrap().push(("capture", request, None));
    BundleStatusV1::Success
}

fn restore(_: BundleHandleV1, request: PrefixStateRequestV1) -> BundleStatusV1 {
    CALLS.lock().unwrap().push(("restore", request, None));
    BundleStatusV1::Unsupported
}

fn restore_in_slot(_: BundleHandleV1, request: PrefixStateRequestV1, slot: u64) -> BundleStatusV1 {
    let status = match slot {
        0..32 => BundleStatusV1::Success,
        32 => BundleStatusV1::InvalidArgument,
        // A copy whose completion the bundle could not confirm.
        _ => BundleStatusV1::InternalError,
    };
    CALLS
        .lock()
        .unwrap()
        .push(("restore_in_slot", request, Some(slot)));
    status
}

// A bundle without speculation: empty brackets export a null commit_accepted_v1.
export_bundle_api_v1!(
    with_prefix_state_slot BASE_API, EXTENDED_API,
    create, execute, success, success, success, retire, [],
    describe, capture, restore, restore_in_slot,
);

#[test]
fn state_slot_export_passes_the_slot_and_refusals_do_not_poison() {
    assert_eq!(
        chop_bundle_get_api_v1(),
        &raw const EXTENDED_API.base.base.base.base
    );
    assert!(EXTENDED_API.base.base.commit_accepted_v1.is_none());
    assert!(
        !BundleApiV1::new_with_prefix_cache(EXTENDED_API.base)
            .unwrap()
            .supports_prefix_state_slot()
    );
    let api = BundleApiV1::new_with_prefix_state_slot(EXTENDED_API).unwrap();
    assert!(api.supports_prefix_cache());
    assert!(api.supports_prefix_state_slot());
    assert!(!api.supports_speculation());
    let host = HostServicesTableV1::new(std::rc::Rc::new(UnsupportedHostServicesBackendV1));
    let mut instance = api.create(host, &[]).unwrap();
    assert_eq!(instance.describe_prefix_state().unwrap(), 35);
    instance
        .restore_prefix_state_in_slot(42, &[1, 2, 3], &[5, 6], 7)
        .unwrap();
    assert_eq!(
        instance.restore_prefix_state_in_slot(43, &[4], &[8], 32),
        Err(BundleAbiError::Status(BundleStatusV1::InvalidArgument))
    );
    assert!(!instance.is_poisoned(), "a refused restore changes nothing");
    // INTERNAL_ERROR means the copy may still touch the pages.
    assert_eq!(
        instance.restore_prefix_state_in_slot(44, &[5], &[9], 33),
        Err(BundleAbiError::Status(BundleStatusV1::InternalError))
    );
    assert!(instance.is_poisoned());
    assert_eq!(
        *CALLS.lock().unwrap(),
        vec![
            (
                "restore_in_slot",
                PrefixStateRequestV1 {
                    request_id: 42,
                    tokens: 3,
                    token_ids: vec![1, 2, 3],
                    pages: vec![5, 6],
                },
                Some(7),
            ),
            (
                "restore_in_slot",
                PrefixStateRequestV1 {
                    request_id: 43,
                    tokens: 1,
                    token_ids: vec![4],
                    pages: vec![8],
                },
                Some(32),
            ),
            (
                "restore_in_slot",
                PrefixStateRequestV1 {
                    request_id: 44,
                    tokens: 1,
                    token_ids: vec![5],
                    pages: vec![9],
                },
                Some(33),
            ),
        ]
    );
}

#[test]
fn state_slot_tail_needs_its_full_size_and_the_prefix_callbacks() {
    let mut short = EXTENDED_API;
    short.base.base.base.base.struct_size = std::mem::offset_of!(
        BundleApiWithPrefixStateSlotV1,
        restore_prefix_state_in_slot_v1
    ) as u32;
    let api = BundleApiV1::new_with_prefix_state_slot(short).unwrap();
    assert!(api.supports_prefix_cache());
    assert!(!api.supports_prefix_state_slot());
    let mut absent = EXTENDED_API;
    absent.base.capture_prefix_state_v1 = None;
    assert!(
        !BundleApiV1::new_with_prefix_state_slot(absent)
            .unwrap()
            .supports_prefix_state_slot()
    );
    let host = HostServicesTableV1::new(std::rc::Rc::new(UnsupportedHostServicesBackendV1));
    let mut instance = BundleApiV1::new_with_prefix_cache(EXTENDED_API.base)
        .unwrap()
        .create(host, &[])
        .unwrap();
    assert_eq!(
        instance.restore_prefix_state_in_slot(44, &[1], &[5], 0),
        Err(BundleAbiError::MissingApiCallback(
            "restore_prefix_state_in_slot"
        ))
    );
    instance.destroy().unwrap();
}
