use runtime_kv::KvGeometry;
use runtime_scheduler::*;

fn run(lookahead: usize) -> usize {
    // 2 requests, block 4, 6 pages = 24 tokens.
    let config = SchedulerConfig::new(2, 8, 32, 8, 1)
        .unwrap()
        .with_admission(AdmissionPolicy::Optimistic(
            OptimisticAdmission::new(0, 3).unwrap(),
        ));
    let mut s = Scheduler::new(config, KvGeometry::new(4, 6).unwrap()).unwrap();
    for id in 1..=2 {
        s.submit(RequestSpec::new(RequestId::new(id), 8, 8).unwrap())
            .unwrap();
    }
    let mut preempted = 0;
    for _ in 0..3 {
        preempted += s
            .preempt_for_next_step(lookahead, |_, _| None)
            .unwrap()
            .len();
        for c in s.take_prefix_captures() {
            s.begin_prefix_capture(&c).unwrap();
            s.end_prefix_capture(&c).unwrap();
        }
        // Plan no drafts at all: the policy decided drafting does not pay.
        let limits = if lookahead == 0 {
            DraftLimits::NONE
        } else {
            DraftLimits::new(lookahead, 64)
        };
        let Some(b) = s
            .next_batch_with_drafts(limits, |c, _| vec![0; c.len()])
            .unwrap()
        else {
            break;
        };
        let outcomes: Vec<_> = b
            .items()
            .iter()
            .map(|i| {
                BatchOutcome::success(
                    i.request_id(),
                    i.phase(),
                    usize::from(i.expects_output()),
                    false,
                )
            })
            .collect();
        s.commit_batch(b.id(), &outcomes).unwrap();
    }
    preempted
}

#[test]
fn optional_drafts_do_not_force_preemption() {
    let a = run(0);
    let b = run(7);
    eprintln!("preemptions without lookahead: {a}, with lookahead 7 and zero drafts planned: {b}");
    assert_eq!(a, 0);
    assert_eq!(b, 0);
}
