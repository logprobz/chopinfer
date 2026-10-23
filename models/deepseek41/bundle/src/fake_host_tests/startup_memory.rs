//! Create logs one `startup memory` line per rank that attributes every
//! byte it allocated on that rank to a named category.

use super::*;
use crate::buffers::Scratch;
use kernel_core::allocation::{Allocation, Traversal};

/// Each rank's `startup memory` record that `host` logged.
fn records(host: &FakeHost) -> Vec<serde_json::Value> {
    host.logs
        .borrow()
        .iter()
        .filter_map(|line| line.strip_prefix("startup memory: "))
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

#[test]
fn startup_memory_attributes_every_allocated_byte() {
    for (ranks, speculation) in [
        (Ranks::TP1, None),
        (Ranks::TP4, None),
        (Ranks::TP4, Some((3, 16))),
    ] {
        let shape = Shape::new(&FLASH, ranks).unwrap();
        let host = FakeHost::new(shape, KV_LAYOUTS[0]);
        host.speculation.set(speculation);
        let mut instance = create(&host, 4);
        let mut allocated = vec![0_u64; shape.ranks() as usize];
        for &(rank, bytes) in host.device.allocation_ranks.borrow().values() {
            allocated[rank as usize] += bytes;
        }
        let records = records(&host);
        assert_eq!(records.len(), allocated.len());
        let declared = Scratch::of_storage(&shape, KV_LAYOUTS[0].storage);
        for (rank, record) in records.iter().enumerate() {
            assert_eq!(record["rank"], rank, "{record}");
            let bytes = record["bytes"].as_object().unwrap();
            let total = record["total"].as_u64().unwrap();
            assert_eq!(
                bytes.values().map(|v| v.as_u64().unwrap()).sum::<u64>(),
                total,
                "{record}"
            );
            assert_eq!(total, allocated[rank], "{record}");
            assert!(!bytes.contains_key("other"), "{record}");
            for category in [
                "inputs",
                "scratch",
                "workspace",
                "sorted_rows",
                "rope_table",
            ] {
                assert!(
                    bytes[category].as_u64().unwrap() > 0,
                    "{category}: {record}"
                );
            }
            if speculation.is_none() {
                let mut scratch = 0;
                declared
                    .visit(&mut |allocation: &Allocation| scratch += allocation.allocation_bytes());
                assert_eq!(bytes["scratch"].as_u64().unwrap(), scratch, "{record}");
            } else {
                assert!(bytes["dspark"].as_u64().unwrap() > 0, "{record}");
            }
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
    }
}
