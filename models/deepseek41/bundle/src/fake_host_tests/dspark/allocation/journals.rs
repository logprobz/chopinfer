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
        (4, 16, 5, "", &[][..], (69458, 0x31e8feae163fcdae)),
        (8, 32, 5, "", &[], (107050, 0x2c1031ee4933ca42)),
        (4, 12, 3, "-d3", &[], (69626, 0x18ef6fee16402a52)),
        (16, 65, 5, "-odd", &[], (190558, 0x759dd31d2c863d27)),
        (32, 192, 5, "-maximum", &[], (343296, 0x1a96fa62a770e259)),
        (
            16,
            65,
            5,
            "-odd-wide",
            &[WIDE_VERIFY],
            (191926, 0x49511b69ebd37509),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide",
            &[WIDE_VERIFY],
            (344664, 0x8df4cd20aa5fc564),
        ),
        (
            4,
            16,
            5,
            "-phased",
            &[PHASED_DECODE_REDUCE],
            (69482, 0xc36f79e0d5d3d8ae),
        ),
        (
            16,
            65,
            5,
            "-odd-phased",
            &[PHASED_DECODE_REDUCE],
            (189318, 0x2162072561a7ef58),
        ),
        (
            16,
            65,
            5,
            "-odd-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (191826, 0xb23641665a5e27f7),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (343656, 0x9f646bd1031a80ff),
        ),
        (
            8,
            32,
            5,
            "-markov-staged",
            &drafter[..1],
            (107042, 0xe90fed7432635aca),
        ),
        (
            8,
            32,
            5,
            "-hc-fusion",
            &drafter[1..2],
            (106978, 0x1ce1fc7778fc43a4),
        ),
        (
            8,
            32,
            5,
            "-confidence-staged",
            &drafter[2..3],
            (107062, 0xe888efa9eb5e76e7),
        ),
        (
            8,
            32,
            5,
            "-moe-direct",
            &drafter[3..4],
            (106990, 0x63361922e9fdd91a),
        ),
        (
            4,
            16,
            5,
            "-hc-draft-rows",
            &[DSPARK_HC_FUSION, DSPARK_HC_DRAFT_ROWS],
            (69386, 0xef446c1cc9c37708),
        ),
        (4, 16, 5, "-drafter", &drafter, (69330, 0xd5210fed492f9516)),
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
