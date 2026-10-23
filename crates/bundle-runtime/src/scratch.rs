//! Bind ordered scratch declarations through the caller's allocator.

use kernel_core::allocation::{Allocation, AllocationError, Traversal};
use runtime_bundle_abi::{BundleAbiError, DeviceBufferSpanV1};

/// Convert portable size validation failures into the bundle's range error.
pub fn allocation_error(error: AllocationError) -> BundleAbiError {
    BundleAbiError::InvalidRange(match error {
        AllocationError::Overflow => "scratch allocation size overflow",
        AllocationError::SpanExceedsAllocation => "scratch span exceeds allocation",
    })
}

/// A bound declaration and the device bytes its binding allocated.
#[derive(Clone, Debug)]
pub struct Bound<M> {
    pub spans: M,
    pub bytes: u64,
}

/// The allocator records acquisitions in the caller's `Held` list. On any
/// failure those acquisitions remain there for the caller's cleanup.
/// Returned handles and offsets are retained; no memory is initialized.
pub fn bind<R: Traversal<Allocation>>(
    declaration: R,
    mut allocate: impl FnMut(u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
) -> Result<R::Mapped<DeviceBufferSpanV1>, BundleAbiError> {
    bind_named(declaration, |_, bytes| allocate(bytes)).map(|bound| bound.spans)
}

/// [`bind`], passing `allocate` each allocation's declaration path (such as
/// `rows[1].flags`) and returning the bytes it allocated.
pub fn bind_named<R: Traversal<Allocation>>(
    declaration: R,
    mut allocate: impl FnMut(&str, u64) -> Result<DeviceBufferSpanV1, BundleAbiError>,
) -> Result<Bound<R::Mapped<DeviceBufferSpanV1>>, BundleAbiError> {
    let mut names = Vec::new();
    declaration.visit_named(&mut |name, _| names.push(name.to_owned()));
    let mut names = names.into_iter();
    let mut bytes = 0_u64;
    let spans =
        declaration.try_map(&mut |layout| {
            let name = names.next().ok_or(BundleAbiError::InvalidRange(
                "scratch traversal orders differ",
            ))?;
            let span = allocate(&name, layout.allocation_bytes())?;
            if span.length < layout.allocation_bytes()
                || span.offset.checked_add(layout.allocation_bytes()).is_none()
            {
                return Err(BundleAbiError::InvalidRange("scratch allocation range"));
            }
            bytes = bytes.checked_add(layout.allocation_bytes()).ok_or(
                BundleAbiError::InvalidRange("scratch allocation size overflow"),
            )?;
            Ok(DeviceBufferSpanV1::new(
                span.buffer,
                span.offset,
                layout.span_bytes(),
            ))
        })?;
    Ok(Bound { spans, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::Held;
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    kernel_core::scratch_record! { struct Group { flags: value, epoch: value } }
    kernel_core::scratch_record! { struct Scratch {
        plain: value, rows: array(2), comm: record(Group),
        optional: optional(Group), groups: records(Group, 2), list: list(Group), temporary: option, tail: value,
    } }
    fn declaration(present: bool, listed: bool) -> Scratch<Allocation> {
        Scratch {
            plain: Allocation::new(20, 12).unwrap(),
            rows: [Allocation::bytes(4), Allocation::bytes(8)],
            comm: Group {
                flags: Allocation::bytes(16),
                epoch: Allocation::bytes(24),
            },
            optional: present.then_some(Group {
                flags: Allocation::bytes(32),
                epoch: Allocation::bytes(40),
            }),
            groups: [
                Group {
                    flags: Allocation::bytes(50),
                    epoch: Allocation::bytes(51),
                },
                Group {
                    flags: Allocation::bytes(52),
                    epoch: Allocation::bytes(53),
                },
            ],
            list: if listed {
                vec![
                    Group {
                        flags: Allocation::bytes(54),
                        epoch: Allocation::bytes(55),
                    },
                    Group {
                        flags: Allocation::bytes(56),
                        epoch: Allocation::bytes(57),
                    },
                ]
            } else {
                vec![]
            },
            temporary: present.then_some(Allocation::bytes(58)),
            tail: Allocation::bytes(48),
        }
    }
    fn handle(index: usize) -> BundleDeviceBufferHandleV1 {
        BundleDeviceBufferHandleV1::from_parts(7, index as u32 + 1)
    }
    fn held_handles(held: &[Held]) -> Vec<BundleDeviceBufferHandleV1> {
        held.iter()
            .map(|item| match item {
                Held::Buffer(buffer) => *buffer,
                _ => panic!("unexpected resource"),
            })
            .collect()
    }
    #[test]
    fn scratch_binding_records_every_field_and_failure_cutoff() {
        for (present, listed) in [(false, false), (false, true), (true, false), (true, true)] {
            // Independent allocation order, physical bytes and visible bytes.
            let mut expected = if present {
                vec![
                    (20, 12),
                    (4, 4),
                    (8, 8),
                    (16, 16),
                    (24, 24),
                    (32, 32),
                    (40, 40),
                ]
            } else {
                vec![(20, 12), (4, 4), (8, 8), (16, 16), (24, 24)]
            };
            expected.extend([(50, 50), (51, 51), (52, 52), (53, 53)]);
            if listed {
                expected.extend([(54, 54), (55, 55), (56, 56), (57, 57)]);
            }
            if present {
                expected.push((58, 58));
            }
            expected.push((48, 48));
            let mut held = Vec::new();
            let mut physical = Vec::new();
            let bound = bind(declaration(present, listed), |bytes| {
                let index = physical.len();
                physical.push(bytes);
                held.push(Held::Buffer(handle(index)));
                Ok(DeviceBufferSpanV1::new(
                    handle(index),
                    100 + index as u64 * 13,
                    bytes,
                ))
            })
            .unwrap();
            let mut names = Vec::new();
            bound.visit_named(&mut |name, _| names.push(name.to_owned()));
            let mut expected_names =
                vec!["plain", "rows[0]", "rows[1]", "comm.flags", "comm.epoch"];
            if present {
                expected_names.extend(["optional.flags", "optional.epoch"]);
            }
            expected_names.extend([
                "groups[0].flags",
                "groups[0].epoch",
                "groups[1].flags",
                "groups[1].epoch",
            ]);
            if listed {
                expected_names.extend([
                    "list[0].flags",
                    "list[0].epoch",
                    "list[1].flags",
                    "list[1].epoch",
                ]);
            }
            if present {
                expected_names.push("temporary");
            }
            expected_names.push("tail");
            assert_eq!(names, expected_names);
            let mut actual = Vec::new();
            bound.visit(&mut |span| actual.push((span.buffer, span.offset, span.length)));
            let table = expected
                .iter()
                .enumerate()
                .map(|(index, &(physical, visible))| {
                    (handle(index), 100 + index as u64 * 13, visible, physical)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual
                    .into_iter()
                    .zip(physical)
                    .map(|((h, o, v), p)| (h, o, v, p))
                    .collect::<Vec<_>>(),
                table
            );
            assert_eq!(
                held_handles(&held),
                (0..expected.len()).map(handle).collect::<Vec<_>>()
            );
            for cutoff in 0..expected.len() {
                let mut held = Vec::new();
                let mut requests = Vec::new();
                let result = bind(declaration(present, listed), |bytes| {
                    let index = requests.len();
                    requests.push(bytes);
                    if index == cutoff {
                        return Err(BundleAbiError::InvalidRange("injected"));
                    }
                    held.push(Held::Buffer(handle(index)));
                    Ok(DeviceBufferSpanV1::new(
                        handle(index),
                        100 + index as u64 * 13,
                        bytes,
                    ))
                });
                assert!(matches!(
                    result,
                    Err(BundleAbiError::InvalidRange("injected"))
                ));
                assert_eq!(
                    requests,
                    expected[..=cutoff].iter().map(|v| v.0).collect::<Vec<_>>()
                );
                assert_eq!(
                    held_handles(&held),
                    (0..cutoff).map(handle).collect::<Vec<_>>()
                );
            }
        }
    }
    #[test]
    fn named_binding_passes_paths_and_totals_its_allocations() {
        for (present, listed) in [(false, false), (true, true)] {
            let mut names = Vec::new();
            let mut requested = 0;
            let bound = bind_named(declaration(present, listed), |name, bytes| {
                names.push(name.to_owned());
                requested += bytes;
                Ok(DeviceBufferSpanV1::new(handle(names.len()), 0, bytes))
            })
            .unwrap();
            let mut expected = Vec::new();
            bound
                .spans
                .visit_named(&mut |name, _| expected.push(name.to_owned()));
            assert_eq!(names, expected);
            assert_eq!(bound.bytes, requested);
            // Physical bytes: `plain`'s 20, not its 12 visible ones.
            let optional = if present { 72 + 58 } else { 0 };
            let list = if listed { 222 } else { 0 };
            assert_eq!(bound.bytes, 20 + 12 + 40 + 206 + 48 + optional + list);
        }
    }
    #[test]
    fn scratch_accepts_exact_offset_boundary() {
        kernel_core::scratch_record! { struct One { value: value } }
        let bound = bind(
            One {
                value: Allocation::new(20, 12).unwrap(),
            },
            |bytes| {
                assert_eq!(bytes, 20);
                Ok(DeviceBufferSpanV1::new(handle(0), u64::MAX - 20, bytes))
            },
        )
        .unwrap();
        assert_eq!(bound.value.buffer, handle(0));
        assert_eq!(bound.value.offset, u64::MAX - 20);
        assert_eq!(bound.value.length, 12);
    }
    #[test]
    fn scratch_converts_allocation_errors() {
        assert!(matches!(
            allocation_error(AllocationError::Overflow),
            BundleAbiError::InvalidRange("scratch allocation size overflow")
        ));
        assert!(matches!(
            allocation_error(AllocationError::SpanExceedsAllocation),
            BundleAbiError::InvalidRange("scratch span exceeds allocation")
        ));
    }
    #[test]
    fn scratch_refuses_short_backing_and_offset_overflow() {
        for (offset, length) in [(1, 19), (u64::MAX - 19, 20)] {
            let mut held = Vec::new();
            let result = bind(declaration(false, false), |_| {
                held.push(Held::Buffer(handle(0)));
                Ok(DeviceBufferSpanV1::new(handle(0), offset, length))
            });
            assert!(matches!(
                result,
                Err(BundleAbiError::InvalidRange("scratch allocation range"))
            ));
            assert_eq!(held_handles(&held), vec![handle(0)]);
        }
    }
}
