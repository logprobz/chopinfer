//! Capture the lowerer's fork/join units and a deliberately late shared producer.
use super::device::{Pool, Programs};
use super::recorded::{Arm, Unit};
use kernel_harness::{LabResult, Session};
use runtime_cuda::{Event, GraphExec, Stream};

pub struct Graphs<'a> {
    session: &'a Session,
    graphs: Vec<GraphExec>,
    events: Vec<Event>,
    side: Option<Stream>,
}

pub fn serial(p: &Pool<'_>, programs: &Programs, arm: &Arm, stream: &Stream) -> LabResult<()> {
    programs.issue(p, &arm.prefix, stream)?;
    programs.issue(p, &arm.calls, stream)
}

impl<'a> Graphs<'a> {
    pub fn build(p: &Pool<'a>, programs: &Programs, arms: &[Arm; 2]) -> LabResult<Self> {
        let session = p.session.ok_or("graphs need a GPU")?;
        let mut result = Self {
            session,
            graphs: Vec::new(),
            events: Vec::new(),
            side: Some(session.context.stream().map_err(|e| e.to_string())?),
        };
        for mode in 0..5 {
            let arm = &arms[if mode == 4 { 1 } else { mode % 2 }];
            let first = result.events.len();
            for _ in 0..3 {
                result
                    .events
                    .push(session.context.event().map_err(|e| e.to_string())?);
            }
            let [fork, join, consumed] = [
                &result.events[first],
                &result.events[first + 1],
                &result.events[first + 2],
            ];
            let side = result.side.as_ref().unwrap();
            let graph = session.capture(|main| {
                if mode < 2 {
                    return serial(p, programs, arm, main);
                }
                programs.issue(p, &arm.prefix, main)?;
                fork.record(main).map_err(|e| e.to_string())?;
                side.wait_event(fork).map_err(|e| e.to_string())?;
                if mode == 4 {
                    // Consume the poisoned shared output before allowing its
                    // producer to run. Rejoin before releasing either stream.
                    for unit in &arm.units {
                        if let Unit::Compute(range) = unit {
                            programs.issue(p, &arm.calls[range.clone()], main)?;
                        }
                    }
                    consumed.record(main).map_err(|e| e.to_string())?;
                    side.wait_event(consumed).map_err(|e| e.to_string())?;
                    for unit in &arm.units {
                        if let Unit::Side(range) = unit {
                            programs.issue(p, &arm.calls[range.clone()], side)?;
                        }
                    }
                    join.record(side).map_err(|e| e.to_string())?;
                    main.wait_event(join).map_err(|e| e.to_string())?;
                    return Ok(());
                }
                let mut joined = false;
                for unit in &arm.units {
                    match unit {
                        Unit::Fork => {}
                        Unit::Side(range) => programs.issue(p, &arm.calls[range.clone()], side)?,
                        Unit::Join => {
                            join.record(side).map_err(|e| e.to_string())?;
                            main.wait_event(join).map_err(|e| e.to_string())?;
                            joined = true;
                        }
                        Unit::Compute(range) => {
                            let mut calls = arm.calls[range.clone()].to_vec();
                            if joined {
                                // Match launch_after_join: a full dependency
                                // on the shared producer precedes this range.
                                calls[0].plan.pdl = false;
                            }
                            programs.issue(p, &calls, main)?;
                        }
                    }
                }
                Ok(())
            })?;
            result.graphs.push(graph);
        }
        Ok(result)
    }
    pub fn run(&self, index: usize) -> LabResult<()> {
        self.graphs[index]
            .launch(&self.session.stream)
            .map_err(|e| e.to_string())?;
        self.session.stream.synchronize().map_err(|e| e.to_string())
    }
    pub fn close(mut self) -> LabResult<usize> {
        self.session
            .context
            .synchronize()
            .map_err(|e| e.to_string())?;
        let count = self.graphs.len();
        for graph in self.graphs.drain(..) {
            graph.release().map_err(|e| e.to_string())?;
        }
        for event in self.events.drain(..) {
            event.release().map_err(|e| e.to_string())?;
        }
        self.side
            .take()
            .unwrap()
            .release()
            .map_err(|e| e.to_string())?;
        Ok(count)
    }
}
impl Drop for Graphs<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.session.context.synchronize() {
            eprintln!("wide graph fence failed; retaining graph dependencies: {e}");
            std::mem::forget(std::mem::take(&mut self.graphs));
            std::mem::forget(std::mem::take(&mut self.events));
            std::mem::forget(self.side.take());
        }
    }
}
