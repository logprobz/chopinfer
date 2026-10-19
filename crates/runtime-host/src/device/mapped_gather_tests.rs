use super::*;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct Probe {
    clock: AtomicU64,
    step: u64,
    asked: Mutex<Vec<Vec<u32>>>,
}

impl Probe {
    fn new(step: u64) -> Self {
        Self {
            clock: AtomicU64::new(0),
            step,
            asked: Mutex::new(Vec::new()),
        }
    }
}

impl GatherProbe for Probe {
    fn now(&self) -> Duration {
        Duration::from_micros(self.clock.fetch_add(self.step, Ordering::Relaxed))
    }
    fn read_ahead(&self, rows: &[u32]) {
        self.asked.lock().unwrap().push(rows.to_vec());
    }
}

fn plane(label: &str) -> MappedGatherPlaneV1 {
    // The mapping starts inside a page; rows cross pages and carry arbitrary bytes.
    let bytes: Vec<u8> = (0..(257 * 97 + 19))
        .map(|i| (i * 31 + i / 251) as u8)
        .collect();
    let path = std::env::temp_dir().join(format!("async-gather-{label}-{}", std::process::id()));
    std::fs::write(&path, &bytes).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mapping = crate::host_rows::map_rows_v1(&file, 19, (257 * 97) as u64).unwrap();
    std::fs::remove_file(path).unwrap();
    MappedGatherPlaneV1 {
        mapping,
        row_bytes: 97,
        index_offset: 8,
        stream: DeviceObjectV1::new(1),
        destination: DeviceRangeV1 {
            object: DeviceObjectV1::new(2),
            offset: 0,
            length: 97 * 80,
        },
    }
}

fn input(ids: &[u32]) -> Vec<u8> {
    [u32::MAX; 2]
        .into_iter()
        .chain(ids.iter().copied())
        .chain([u32::MAX; 2])
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn expected(plane: &MappedGatherPlaneV1, ids: &[u32]) -> Vec<u8> {
    ids.iter()
        .flat_map(|&id| {
            plane.mapping.bytes()[id as usize * plane.row_bytes..][..plane.row_bytes].to_vec()
        })
        .collect()
}

#[test]
fn cold_async_plane_advises_only_the_unread_live_rows_once() {
    let plane = plane("cold");
    let ids: Vec<u32> = (0..53).map(|i| (i * 37 % 257) as u32).collect();
    let probe = Probe::new(50);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
    assert_eq!(output, expected(&plane, &ids));
    assert_eq!(*probe.asked.lock().unwrap(), [ids[16..].to_vec()]);
}

#[test]
fn warm_and_short_async_planes_keep_bytes_without_advice() {
    let plane = plane("warm");
    for (count, step) in [(0, 100), (1, 100), (16, 100), (53, 0), (53, 40)] {
        // Duplicate and reversed indices must retain their exact byte order.
        let ids: Vec<u32> = (0..count).map(|i| 256 - (i / 2) as u32).collect();
        let probe = Probe::new(step);
        let mut output = vec![0xa5; count * plane.row_bytes];
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
        assert_eq!(output, expected(&plane, &ids));
        assert!(probe.asked.lock().unwrap().is_empty(), "{count} {step}");
    }
}

#[test]
fn invalid_async_ids_are_refused_before_copy_or_advice() {
    let plane = plane("invalid");
    let mut ids = vec![1; 53];
    ids[52] = 257;
    let probe = Probe::new(100);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    assert_eq!(
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true),
        Err("mapped gather device index is outside the table".into())
    );
    assert!(output.iter().all(|&v| v == 0xa5));
    assert!(probe.asked.lock().unwrap().is_empty());
    assert_eq!(probe.clock.load(Ordering::Relaxed), 0);
    assert_eq!(
        gather_plane_with_probe(&plane, &[0; 8], &mut output, &probe, true),
        Err("mapped gather input is truncated".into())
    );
    assert!(output.iter().all(|&v| v == 0xa5));
}

#[test]
fn async_plane_resets_read_ahead_for_each_reused_gather() {
    let plane = plane("reuse");
    let probe = Probe::new(50);
    for ids in [vec![2; 33], vec![256; 17]] {
        let mut output = vec![0; ids.len() * plane.row_bytes];
        gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, true).unwrap();
        assert_eq!(output, expected(&plane, &ids));
    }
    assert_eq!(*probe.asked.lock().unwrap(), [vec![2; 17], vec![256; 1]]);
}

#[test]
fn disabled_async_read_ahead_copies_without_advice() {
    let plane = plane("disabled");
    let ids: Vec<u32> = (0..53).map(|i| (i * 37 % 257) as u32).collect();
    let probe = Probe::new(50);
    let mut output = vec![0xa5; ids.len() * plane.row_bytes];
    gather_plane_with_probe(&plane, &input(&ids), &mut output, &probe, false).unwrap();
    assert_eq!(output, expected(&plane, &ids));
    assert!(probe.asked.lock().unwrap().is_empty());
}
