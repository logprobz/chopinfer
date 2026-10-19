use super::*;

struct Producer {
    info: OpInfo,
    launches: Vec<Launch>,
}
impl PreparedOp<()> for Producer {
    fn info(&self) -> &OpInfo {
        &self.info
    }
    fn launches(&self, _: &(), _: View, _: Counts) -> Result<Vec<Launch>, Refusal> {
        Ok(self.launches.clone())
    }
}

fn producer() -> Producer {
    Producer {
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
