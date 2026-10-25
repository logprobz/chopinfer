use super::*;

struct Producer {
    info: OpInfo,
    launches: Vec<Launch>,
    route_calls: std::sync::Arc<std::sync::atomic::AtomicU32>,
}
impl PreparedOp<()> for Producer {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, _: &(), _: View, _: Counts) -> Result<Vec<Launch>, Refusal> {
        Ok(self.launches.clone())
    }
    fn route(&self, _: Counts, launch: &Launch) -> Option<kernel_core::route::RouteTag> {
        self.route_calls
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Some(kernel_core::route::branch(
            &launch.symbol,
            kernel_core::route::RouteClass::Fast,
            "producer",
            "independent",
            [1..=32],
        ))
    }
}

fn producer() -> Producer {
    Producer {
        route_calls: Default::default(),
        info: OpInfo::new(
            "independent",
            Invariance::any_rows(32),
            vec![("producer".into(), 0)],
        ),
        launches: vec![Launch {
            symbol: "producer".into(),
            rows: 16,
            plan: kernel_core::contract::LaunchPlan {
                grid: [64, 1, 1],
                block: [128, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: kernel_core::contract::ArgumentList(vec![]),
            collective: false,
            persistent: false,
        }],
    }
}

#[test]
fn disabled_routes_skip_selection_in_both_lowering_paths() {
    let producer = producer();
    let calls = producer.route_calls.clone();
    let op: Prepared<()> = Box::new(producer);
    for enabled in [false, true] {
        let _counter = bundle_runtime::routes::RouteCounter::begin(enabled);
        for independent in [false, true] {
            let bind = Bind::default();
            let lowered = if independent {
                bind.launch_independent(&op, &(), Counts::rows(16))
            } else {
                bind.launch(&op, &(), Counts::rows(16))
            }
            .unwrap();
            assert_eq!(lowered.len(), 1);
            assert_eq!(lowered[0].launch.bindings.route().is_some(), enabled);
        }
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            if enabled { 2 } else { 0 }
        );
    }
}

#[test]
fn independent_producer_guard_refuses_spinning_workspace_and_inexact_forms() {
    let accepted = |p: Producer| {
        let op: Prepared<()> = Box::new(p);
        Bind::default()
            .launch_independent(&op, &(), Counts::rows(16))
            .is_ok()
    };
    assert!(accepted(producer()));
    let mut tiled = producer();
    tiled.launches.push(tiled.launches[0].clone());
    assert!(accepted(tiled));
    for change in [
        |p: &mut Producer| p.info.guarantees = Invariance::NONE,
        |p: &mut Producer| p.info.workspace = 4,
        |p: &mut Producer| p.launches[0].persistent = true,
        |p: &mut Producer| p.launches[0].collective = true,
        |p: &mut Producer| p.launches[0].plan.workspace_bytes = 4,
        |p: &mut Producer| p.launches[0].plan.cluster = [2, 1, 1],
        |p: &mut Producer| p.launches.clear(),
    ] {
        let mut p = producer();
        change(&mut p);
        assert!(!accepted(p));
    }
}
