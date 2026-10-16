#![deny(unsafe_code)]

use std::cell::Cell;
use std::mem::size_of;
use std::rc::Rc;

use runtime_bundle_abi::*;

thread_local! {
    static RETIRED: Cell<Option<(BundleHandleV1, u64)>> = const { Cell::new(None) };
}

fn panic_create(_: HostServicesV1, _: &[u8]) -> BundleCreateResultV1 {
    panic!("create panic")
}

fn panic_execute(_: BundleHandleV1, _: &[u8], _: &mut ExecuteResultSinkV1) -> BundleStatusV1 {
    panic!("execute panic")
}

fn panic_status(_: BundleHandleV1) -> BundleStatusV1 {
    panic!("finalization panic")
}

fn retire(handle: BundleHandleV1, request_id: u64) -> BundleStatusV1 {
    if request_id == u64::MAX {
        panic!("retirement panic");
    }
    RETIRED.set(Some((handle, request_id)));
    BundleStatusV1::InvalidState
}

export_bundle_api_v1!(
    with_request_retirement BASE_API,
    EXTENDED_API,
    panic_create,
    panic_execute,
    panic_status,
    panic_status,
    panic_status,
    retire,
);

fn sink() -> BundleExecuteResultSinkV1 {
    extern "C" fn discard(_: *mut std::ffi::c_void, _: BundleLogitsRowV1) -> BundleStatusV1 {
        BundleStatusV1::Success
    }
    let mut sink = BundleExecuteResultSinkV1::empty();
    sink.maximum_rows = 1;
    sink.maximum_logits_per_row = 1;
    sink.maximum_total_logits = 1;
    sink.emit_logits_row = Some(discard);
    sink
}

#[test]
fn export_returns_extended_storage_and_keeps_copied_prefix_callers_compatible() {
    assert_eq!(chop_bundle_get_api_v1(), &raw const EXTENDED_API.base);
    assert_eq!(
        EXTENDED_API.base.struct_size as usize,
        size_of::<BundleApiWithRequestRetirementV1>()
    );
    assert_eq!(BASE_API.struct_size, EXTENDED_API.base.struct_size);
    assert_eq!(BASE_API.abi_version, BUNDLE_ABI_VERSION_V1);
    assert!(
        !BundleApiV1::new(BASE_API)
            .unwrap()
            .supports_request_retirement()
    );
    assert!(
        BundleApiV1::new_with_request_retirement(EXTENDED_API)
            .unwrap()
            .supports_request_retirement()
    );
}

#[test]
fn retirement_forwards_handle_id_and_status_and_contains_panics() {
    let callback = EXTENDED_API.retire_request_v1.unwrap();
    let handle = BundleHandleV1::from_parts(7, 9);
    for id in [0, 42, u64::MAX - 1] {
        assert_eq!(callback(handle, id), BundleStatusV1::InvalidState);
        assert_eq!(RETIRED.get(), Some((handle, id)));
    }
    assert_eq!(callback(handle, u64::MAX), BundleStatusV1::Panic);
    assert_eq!(RETIRED.get(), Some((handle, u64::MAX - 1)));
}

#[test]
// SAFETY-REVIEWED: the test calls typed ABI callbacks with pinned or deliberately invalid inputs
#[allow(unsafe_code)]
fn extended_base_callbacks_keep_pointer_validation_and_panic_containment() {
    let host = HostServicesTableV1::new(Rc::new(UnsupportedHostServicesBackendV1));
    let handle = BundleHandleV1::from_parts(1, 1);
    for base in [BASE_API, EXTENDED_API.base] {
        let create = base.create.unwrap();
        let execute = base.execute.unwrap();
        // SAFETY: the host pins its table for the call and the empty view has no data.
        // SAFETY-REVIEWED: the test calls typed ABI callbacks with pinned or deliberately invalid inputs
        let created = unsafe { create(host.services(), BundleByteViewV1::empty()) };
        assert_eq!(created.status, BundleStatusV1::Panic);
        // SAFETY: the deliberately null table must be rejected before decoding.
        // SAFETY-REVIEWED: the test calls typed ABI callbacks with pinned or deliberately invalid inputs
        let invalid = unsafe { create(std::ptr::null(), BundleByteViewV1::empty()) };
        assert_eq!(invalid.status, BundleStatusV1::InvalidArgument);
        // SAFETY: the empty view and callback-only sink are live for the call.
        // SAFETY-REVIEWED: the test calls typed ABI callbacks with pinned or deliberately invalid inputs
        let status = unsafe { execute(handle, BundleByteViewV1::empty(), sink()) };
        assert_eq!(status, BundleStatusV1::Panic);
        let mut invalid_request = BundleByteViewV1::empty();
        invalid_request.length = 1;
        // SAFETY: a null view with nonzero length must be rejected before decoding.
        // SAFETY-REVIEWED: the test calls typed ABI callbacks with pinned or deliberately invalid inputs
        let status = unsafe { execute(handle, invalid_request, sink()) };
        assert_eq!(status, BundleStatusV1::InvalidArgument);
        for callback in [base.commit, base.cancel, base.destroy] {
            assert_eq!(callback.unwrap()(handle), BundleStatusV1::Panic);
        }
    }
}
