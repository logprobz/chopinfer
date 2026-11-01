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
/// `phased_decode_reduce` option, the `-wide-draft` ones the two widest
/// with `wide_verify` and `wide_draft` (their 16 and 32 items draft in one
/// forward each), and the drafter ones the C8 lifetime with each drafter
/// candidate option, the C4 one with `dspark_hc_fusion` and
/// `dspark_hc_draft_rows`, and the C4 one with all five.
#[test]
fn allocated_lifetimes_journal() {
    use model_deepseek41_contract::execution_options::{
        DSPARK_CONFIDENCE_STAGED, DSPARK_HC_DRAFT_ROWS, DSPARK_HC_FUSION, DSPARK_MARKOV_STAGED,
        DSPARK_MOE_DIRECT, PHASED_DECODE_REDUCE, WIDE_DRAFT, WIDE_VERIFY,
    };
    let drafter = [
        DSPARK_MARKOV_STAGED,
        DSPARK_HC_FUSION,
        DSPARK_CONFIDENCE_STAGED,
        DSPARK_MOE_DIRECT,
        DSPARK_HC_DRAFT_ROWS,
    ];
    let shape = Shape::new(&FLASH, Ranks::TP4).unwrap();
    let mut mismatches = Vec::new();
    for (items, width, depth, suffix, options, expected) in [
        (4, 16, 5, "", &[][..], (69334, 0x94b43e6c45e6f09b)),
        (8, 32, 5, "", &[], (106926, 0x8d92d278eff57bef)),
        (4, 12, 3, "-d3", &[], (69502, 0xdedf204e99d57b9b)),
        (16, 65, 5, "-odd", &[], (190434, 0xa53d7eeecf401235)),
        (32, 192, 5, "-maximum", &[], (343172, 0x423f402926f10c51)),
        (
            16,
            65,
            5,
            "-odd-wide",
            &[WIDE_VERIFY],
            (191806, 0x3fc23092499e338a),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide",
            &[WIDE_VERIFY],
            (344544, 0x4a0b9dbbab2a004d),
        ),
        (
            4,
            16,
            5,
            "-phased",
            &[PHASED_DECODE_REDUCE],
            (69358, 0xfdbd662fbe69bdf6),
        ),
        (
            16,
            65,
            5,
            "-odd-phased",
            &[PHASED_DECODE_REDUCE],
            (189194, 0xb6fd29e113df566c),
        ),
        (
            16,
            65,
            5,
            "-odd-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (191706, 0x0c1f4b48329803ea),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (343536, 0xa83e54b9d1652410),
        ),
        (
            8,
            32,
            5,
            "-markov-staged",
            &drafter[..1],
            (106918, 0xec2cda8f7c64a88f),
        ),
        (
            8,
            32,
            5,
            "-hc-fusion",
            &drafter[1..2],
            (106854, 0x85c7e7bd95972561),
        ),
        (
            8,
            32,
            5,
            "-confidence-staged",
            &drafter[2..3],
            (106938, 0x0156530484d50f83),
        ),
        (
            8,
            32,
            5,
            "-moe-direct",
            &drafter[3..4],
            (106866, 0x39dbc028f5685113),
        ),
        (
            4,
            16,
            5,
            "-hc-draft-rows",
            &[DSPARK_HC_FUSION, DSPARK_HC_DRAFT_ROWS],
            (69262, 0x4db4a4e157875315),
        ),
        (4, 16, 5, "-drafter", &drafter, (69206, 0xf7b1328155f83920)),
    ] {
        let (host, mut instance) = if options.is_empty() {
            create_with_limits(shape, 32, 192)
        } else {
            let options: Vec<_> = options.iter().map(|&o| (o, true.into())).collect();
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
