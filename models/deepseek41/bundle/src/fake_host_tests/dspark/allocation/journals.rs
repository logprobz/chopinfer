//! Complete allocated lifetimes pin drafting, computed target rows, sampling and commit.
use super::*;
use runtime_bundle_abi::device_results::*;

struct ProofSink {
    host: Rc<Host>,
    metadata: DeviceCycleMetadataV1,
    samples: u32,
}
impl LogitsRowSinkV1 for ProofSink {
    fn limits(&self) -> ResultSinkLimitsV1 {
        Sink::default().limits()
    }
    fn emit_row(&mut self, _: OwnedLogitsRowV1) -> HostServiceResultV1<()> {
        Err(BundleStatusV1::Unsupported)
    }
    fn device_results_enabled(&self) -> bool {
        true
    }
    fn mapped_sampling_enabled(&self) -> bool {
        true
    }
    fn device_cycle_metadata(&self) -> HostServiceResultV1<DeviceCycleMetadataV1> {
        Ok(self.metadata.clone())
    }
    fn declare_input_dependency(&mut self, value: u32) -> HostServiceResultV1<()> {
        assert_eq!(value, DEVICE_RESULTS_HOST_VALUES_V1);
        Ok(())
    }
    fn emit_device_drafts(&mut self, value: BundleDeviceDraftsV1) -> HostServiceResultV1<()> {
        value.validate()?;
        Ok(())
    }
    fn sample_device_logits(
        &mut self,
        value: BundleDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        value.validate().unwrap();
        self.samples += value.logits.logits.row_count;
        let stream = BundleStreamHandleV1::from_parts(
            1,
            self.host.rank_streams.borrow()[&value.logits.logits.rank],
        );
        self.host.record_event(value.done, stream)
    }
    fn sample_mapped_device_logits(
        &mut self,
        value: BundleMappedDeviceSamplingRequestV1,
    ) -> HostServiceResultV1<()> {
        value.validate().unwrap();
        self.samples += value.sample_rows;
        let stream = BundleStreamHandleV1::from_parts(
            1,
            self.host.rank_streams.borrow()[&value.source.logits.rank],
        );
        self.host.record_event(value.done, stream)
    }
    fn emit_device_completion(
        &mut self,
        value: BundleDeviceCompletionV1,
    ) -> HostServiceResultV1<()> {
        value.validate()?;
        assert_eq!(value.sample_rows, self.samples);
        Ok(())
    }
}

/// The `-wide` lifetimes run the two widest with the `wide_verify` option,
/// the `-phased` ones the first and a 65-row one with the
/// `phased_decode_reduce` option, and the `-wide-draft` ones the two widest
/// with `wide_verify` and `wide_draft`: their 16 and 32 items draft in one
/// forward each.
#[test]
fn allocated_lifetimes_journal() {
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let mut mismatches = Vec::new();
    for (items, width, depth, suffix, expected) in [
        (4, 16, 5, "", (69334, 0x94b43e6c45e6f09b)),
        (8, 32, 5, "", (106926, 0x8d92d278eff57bef)),
        (4, 12, 3, "-d3", (69502, 0xdedf204e99d57b9b)),
        (16, 65, 5, "-odd", (190434, 0xa53d7eeecf401235)),
        (32, 192, 5, "-maximum", (343172, 0x423f402926f10c51)),
        (16, 65, 5, "-odd-wide", (191802, 0x64bd42c3caef55c0)),
        (32, 192, 5, "-maximum-wide", (344540, 0x07785bc19d29a6ff)),
        (4, 16, 5, "-phased", (69358, 0xfdbd662fbe69bdf6)),
        (16, 65, 5, "-odd-phased", (189194, 0xb6fd29e113df566c)),
        (16, 65, 5, "-odd-wide-draft", (191702, 0x7ec0e494350dc4a8)),
        (
            32,
            192,
            5,
            "-maximum-wide-draft",
            (343532, 0xf9ee11d6af367dda),
        ),
    ] {
        use model_deepseek41_contract::execution_options::{
            PHASED_DECODE_REDUCE, WIDE_DRAFT, WIDE_VERIFY,
        };
        let options: &[&str] = if suffix.ends_with("-wide") {
            &[WIDE_VERIFY]
        } else if suffix.ends_with("-wide-draft") {
            &[WIDE_VERIFY, WIDE_DRAFT]
        } else if suffix.ends_with("-phased") {
            &[PHASED_DECODE_REDUCE]
        } else {
            &[]
        };
        let (host, mut instance) = if options.is_empty() {
            create_with_limits(shape, 32, 192)
        } else {
            let options = options
                .iter()
                .map(|&option| (option, true.into()))
                .collect::<Vec<_>>();
            create_with_options(shape, 32, 192, &options)
        };
        for slot in 0..items {
            execute(
                &mut instance,
                &host,
                &super::super::super::prompt(u64::from(slot + 1), u64::from(slot), 0, 3, true),
            );
            instance.commit().unwrap();
        }
        for cycle in 1..=2 {
            let mut sink = ProofSink {
                host: host.clone(),
                metadata: metadata(items, depth, cycle),
                samples: 0,
            };
            instance
                .execute(
                    &allocated(items, cycle + 2, depth, width).encode().unwrap(),
                    &mut sink,
                )
                .unwrap();
            assert_eq!(sink.samples, items * (depth + 1));
            instance
                .commit_accepted(
                    &(1..=items)
                        .map(|id| super::super::kept(u64::from(id), 1))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }
        instance.destroy().unwrap();
        assert_eq!(host.acquired.get(), host.released.get());
        if let Err(observed) = bundle_runtime::fake_host::check_journal(
            &host.journal.borrow(),
            expected,
            &format!("deepseek41-allocation-c{items}{suffix}.journal"),
        ) {
            mismatches.push(format!("C{items}{suffix}: {observed}"));
        }
    }
    assert!(mismatches.is_empty(), "{mismatches:#?}");
}
