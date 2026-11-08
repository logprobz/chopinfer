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
        (4, 16, 5, "", &[][..], (69598, 0x76ea98292927d86f)),
        (8, 32, 5, "", &[], (107194, 0x14000fb89453950e)),
        (4, 12, 3, "-d3", &[], (69766, 0xe45aa2411e3fdea5)),
        (16, 65, 5, "-odd", &[], (190702, 0x3de9a05d344a0bab)),
        (32, 192, 5, "-maximum", &[], (343436, 0x14258f786ebbd325)),
        (
            16,
            65,
            5,
            "-odd-wide",
            &[WIDE_VERIFY],
            (191990, 0x3c2ddfbfd9beed66),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide",
            &[WIDE_VERIFY],
            (344724, 0xf1ba47dae14fbe86),
        ),
        (
            4,
            16,
            5,
            "-phased",
            &[PHASED_DECODE_REDUCE],
            (69622, 0xc2e6f056b556b60c),
        ),
        (
            16,
            65,
            5,
            "-odd-phased",
            &[PHASED_DECODE_REDUCE],
            (189462, 0x59d2cfe1cee3eeda),
        ),
        (
            16,
            65,
            5,
            "-odd-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (191894, 0xb1b3cde27f8285e2),
        ),
        (
            32,
            192,
            5,
            "-maximum-wide-draft",
            &[WIDE_VERIFY, WIDE_DRAFT],
            (343724, 0x6b0db6ce72758454),
        ),
        (
            8,
            32,
            5,
            "-markov-staged",
            &drafter[..1],
            (107186, 0x9222c41b135c27bf),
        ),
        (
            8,
            32,
            5,
            "-hc-fusion",
            &drafter[1..2],
            (107122, 0x22a46fad440a4f4e),
        ),
        (
            8,
            32,
            5,
            "-confidence-staged",
            &drafter[2..3],
            (107206, 0xcce5662d7ec4aba5),
        ),
        (
            8,
            32,
            5,
            "-moe-direct",
            &drafter[3..4],
            (107134, 0xffb5d1e97e20baa0),
        ),
        (
            4,
            16,
            5,
            "-hc-draft-rows",
            &[DSPARK_HC_FUSION, DSPARK_HC_DRAFT_ROWS],
            (69526, 0x9b44c6b6960b5345),
        ),
        (4, 16, 5, "-drafter", &drafter, (69470, 0xb9d87a22ad27e6e3)),
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
