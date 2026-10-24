//! A serving host-memory device keeps a bounded record of what it was asked,
//! while a test's device keeps every entry.

use runtime_host::HostDeviceBackendV1;
use runtime_host::fake::{FAKE_SERVED_HISTORY_ENTRIES_V1, FakeHostDeviceV1};

#[test]
fn a_history_limit_keeps_the_oldest_entries_and_no_more() {
    let mut test = FakeHostDeviceV1::new();
    let mut served = FakeHostDeviceV1::new().with_history_limit(FAKE_SERVED_HISTORY_ENTRIES_V1);
    let lines = FAKE_SERVED_HISTORY_ENTRIES_V1 + 100;
    for line in 0..lines {
        let message = format!("line {line}");
        test.log(message.as_bytes()).unwrap();
        served.log(message.as_bytes()).unwrap();
    }
    assert_eq!(test.logs().len(), lines);
    assert_eq!(served.logs().len(), FAKE_SERVED_HISTORY_ENTRIES_V1);
    assert_eq!(served.logs()[0], b"line 0");
    assert_eq!(
        served.logs().last().unwrap(),
        format!("line {}", FAKE_SERVED_HISTORY_ENTRIES_V1 - 1).as_bytes()
    );
}
