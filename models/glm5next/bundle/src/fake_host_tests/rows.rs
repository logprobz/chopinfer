//! The rows recorder (`crate::submit` in test builds) sees every launch the
//! host receives, eager or captured, the bundle's own and the encoder's, in
//! the order the host receives them.

use super::*;

/// Create, an image prompt's prefill with the MTP drafter's extend pass,
/// two drafted verify steps (their graphs captured, then replayed), each
/// with a commit of kept rows (the state copies), a third and a cancel,
/// with the shared gate/up overlap `overlap`: every launch the host received
/// is recorded, by stream and symbol, in order. The records' lines.
fn recorded_scenario(overlap: crate::shared_overlap::Settings) -> Vec<String> {
    let format = Shape::TP4.kv_formats()[1];
    assert!(has_mtp_layer(&format));
    let host = FakeHost::with_format((3, 32), format);
    *host.launch_args.borrow_mut() = Some(Vec::new());
    crate::submit::start_recording();
    let roles = crate::config::tests::encoder_roles();
    let mut instance = crate::shared_overlap::with_settings_in_tests(overlap, || {
        api().create(
            HostServicesTableV1::new(Rc::clone(&host)),
            &crate::config::tests::record_naming(DECODE_BATCH, &options(2048), format, &roles),
        )
    })
    .unwrap_or_else(|error| panic!("fake-host create: {error:?} {:?}", host.logs.borrow()));
    crate::instance::run_lowered_stages_alone(instance.handle());
    // An image of 16 tokens between two text rows.
    instance
        .execute(&image_prompt(7, [1, 8, 8]), &mut DraftSink::default())
        .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
    instance.commit().unwrap();
    for (previous, bonus) in [(18, 21), (20, 22)] {
        instance
            .execute(&drafted(7, previous, 4, bonus), &mut DraftSink::default())
            .unwrap_or_else(|error| panic!("{error:?} {:?}", host.logs.borrow()));
        instance.commit_accepted(&[kept(7, 2)]).unwrap();
    }
    instance
        .execute(&drafted(7, 22, 4, 23), &mut DraftSink::default())
        .unwrap();
    instance.cancel().unwrap();
    let recorded = crate::submit::take_recorded();
    let launched = host.launch_args.borrow_mut().take().unwrap();
    let seen = recorded
        .iter()
        .map(|launch| (launch.stream, launch.symbol.as_str()))
        .collect::<Vec<_>>();
    let received = launched
        .iter()
        .map(|(stream, symbol, _)| (*stream, symbol.as_str()))
        .collect::<Vec<_>>();
    if let Some(index) =
        (0..seen.len().max(received.len())).find(|&i| seen.get(i) != received.get(i))
    {
        panic!(
            "recorded launch {index} is {:?}, the host's is {:?} ({} and {})",
            seen.get(index),
            received.get(index),
            seen.len(),
            received.len()
        );
    }
    instance.destroy().unwrap();
    assert_eq!(host.acquired.get(), host.released.get());
    recorded.into_iter().map(|launch| launch.line).collect()
}

/// [`recorded_scenario`] with the shared gate/up overlap on every bucket (the default)
/// and off: each kind of submission is among the records.
#[test]
fn every_launch_is_recorded_with_its_rows() {
    let lines = [
        crate::shared_overlap::Settings::ALL,
        crate::shared_overlap::Settings::OFF,
    ]
    .into_iter()
    .flat_map(recorded_scenario)
    .collect::<Vec<_>>();
    // Lowered rank programs, launch lists on a stream, single launches
    // without rows (commit's state copies) and the encoder's programs.
    for (kind, found) in [
        ("rank", lines.iter().any(|line| line.starts_with("rank "))),
        (
            "stream",
            lines
                .iter()
                .any(|line| line.starts_with("stream ") && !line.ends_with("\t-")),
        ),
        (
            "single",
            lines.iter().any(|line| {
                line.ends_with("\t-")
                    && line.contains(model_glm5next_contract::sm120::state_copy::SYMBOL)
            }),
        ),
        (
            "encoder",
            lines.iter().any(|line| line.starts_with("encoder stream ")),
        ),
    ] {
        assert!(found, "no {kind} launch recorded");
    }
}
