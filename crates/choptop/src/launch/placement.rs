//! Where each instance of a launch goes: the GPUs picked, split into groups
//! of the tensor parallel size in index order, one instance per group,
//! lettered A, B and so on; and a port for each, from serve's default.
//!
//! A GPU held by a serve process of this user stops that process first; a
//! GPU or port a pending choptop launch holds replaces that launch. A GPU
//! held by anything else, or not known, blocks the start with the reason.

use std::collections::{BTreeMap, BTreeSet};
use std::net::TcpListener;

use super::Reservation;
use super::occupancy::{Gpu, Holder, Occupancy, PortHolder, Serve};
use crate::discover::DEFAULT_LISTEN;

/// What must be gone before an instance starts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Dependency {
    /// A serve process this switch stops.
    Serve(Serve),
    /// A pending launch this switch cancels or stops, by its position
    /// among the reservations.
    Launch(usize),
}

/// One instance of a launch.
#[derive(Clone, Debug, PartialEq)]
pub struct Planned {
    /// `A`, `B` and so on.
    pub letter: char,
    /// NVML indices, in order.
    pub gpus: Vec<u32>,
    /// Their UUIDs, for `CUDA_VISIBLE_DEVICES`.
    pub uuids: Vec<String>,
    pub host: String,
    pub port: u16,
    pub after: Vec<Dependency>,
}

impl Planned {
    /// `127.0.0.1:8310`.
    pub fn listen(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// A switch: the instances to start, what stops or is replaced first, and
/// what blocks it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Placement {
    pub instances: Vec<Planned>,
    /// Serve processes stopped first, in GPU order.
    pub stops: Vec<Serve>,
    /// Pending launches replaced, by position among the reservations.
    pub replaces: Vec<usize>,
    /// Why the GPUs picked make no instances: none, or not a whole number
    /// of groups.
    pub grouping: Option<String>,
    /// What else blocks the start.
    pub problems: Vec<String>,
}

/// `HOST:PORT`, the host kept as written (`[::1]` with its brackets).
pub fn parse_listen(text: &str) -> Option<(String, u16)> {
    let (host, port) = text.trim().rsplit_once(':')?;
    let port = port.parse::<u16>().ok().filter(|port| *port > 0)?;
    (!host.is_empty()).then(|| (host.to_owned(), port))
}

/// Whether nothing listens on `port` of `host` now.
pub fn bind_probe(host: &str, port: u16) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    TcpListener::bind((host, port)).is_ok()
}

/// What occupies the GPUs and ports, and the reservations they name.
pub struct Field<'a> {
    pub gpus: &'a [Gpu],
    pub ports: &'a BTreeMap<u16, PortHolder>,
    pub reservations: &'a [Reservation],
}

impl Field<'_> {
    fn port_holder(&self, holder: &PortHolder) -> String {
        match holder {
            PortHolder::Serve(serve) => serve.label(),
            PortHolder::Launch(position) => self
                .reservations
                .get(*position)
                .map_or_else(|| "a launch".to_owned(), |launch| launch.label.clone()),
            PortHolder::Other(what) => what.clone(),
        }
    }
}

/// Places `picked` GPUs at `tensor_parallel` per instance. Ports start at
/// the form's `listen`, else serve's default, and skip ports that
/// instances which keep running or pending launches hold, and ports
/// `probe` finds taken. A port held by a process this switch stops is
/// reused, which makes that process a dependency.
pub fn place(
    field: &Field,
    picked: &BTreeSet<u32>,
    tensor_parallel: u32,
    listen: Option<&str>,
    probe: &mut dyn FnMut(&str, u16) -> bool,
) -> Placement {
    let mut placement = Placement::default();
    let size = tensor_parallel.max(1) as usize;
    if picked.is_empty() {
        placement.grouping = Some("pick GPUs with Space".to_owned());
    } else if picked.len() % size != 0 {
        placement.grouping = Some(format!(
            "pick GPUs in groups of {size} for TP{tensor_parallel}"
        ));
    }
    let mut owners = BTreeMap::new();
    for index in picked {
        let Some(gpu) = field.gpus.iter().find(|gpu| gpu.index == *index) else {
            placement.problems.push(format!("GPU {index} is gone"));
            continue;
        };
        match &gpu.occupancy {
            Occupancy::Free => {}
            Occupancy::Unknown(reason) => placement
                .problems
                .push(format!("GPU {index} is unknown: {reason}")),
            Occupancy::Blocked(what) => placement
                .problems
                .push(format!("GPU {index} is held by {what}")),
            Occupancy::Held(holders) => {
                let mut after = Vec::new();
                for holder in holders {
                    match holder {
                        Holder::Serve(serve) => {
                            if !placement.stops.contains(serve) {
                                placement.stops.push(serve.clone());
                            }
                            after.push(Dependency::Serve(serve.clone()));
                        }
                        Holder::Launch(position) => {
                            if !placement.replaces.contains(position) {
                                placement.replaces.push(*position);
                            }
                            after.push(Dependency::Launch(*position));
                        }
                    }
                }
                owners.insert(*index, after);
            }
        }
    }
    if placement.grouping.is_some() {
        return placement;
    }
    let default = parse_listen(DEFAULT_LISTEN).unwrap_or(("127.0.0.1".to_owned(), 8310));
    let (host, start, explicit) = match listen.map(parse_listen) {
        Some(Some((host, port))) => (host, port, true),
        Some(None) => {
            placement
                .problems
                .push("listen must be HOST:PORT".to_owned());
            (default.0, default.1, false)
        }
        None => (default.0, default.1, false),
    };
    let ordered = picked.iter().copied().collect::<Vec<_>>();
    let mut next = u32::from(start);
    for (number, group) in ordered.chunks(size).enumerate() {
        let letter = char::from(b'A' + (number % 26) as u8);
        let mut after: Vec<Dependency> = Vec::new();
        for index in group {
            for dependency in owners.get(index).into_iter().flatten() {
                if !after.contains(dependency) {
                    after.push(dependency.clone());
                }
            }
        }
        // The form's address is the first instance's own; the others take
        // the next free port.
        let named = explicit && number == 0;
        let port = loop {
            let Ok(port) = u16::try_from(next) else {
                placement
                    .problems
                    .push(format!("no free port from {start}"));
                break None;
            };
            next += 1;
            match field.ports.get(&port) {
                Some(PortHolder::Serve(serve)) if placement.stops.contains(serve) => {
                    let dependency = Dependency::Serve(serve.clone());
                    if !after.contains(&dependency) {
                        after.push(dependency);
                    }
                    break Some(port);
                }
                Some(PortHolder::Launch(position))
                    if named || placement.replaces.contains(position) =>
                {
                    if !placement.replaces.contains(position) {
                        placement.replaces.push(*position);
                    }
                    let dependency = Dependency::Launch(*position);
                    if !after.contains(&dependency) {
                        after.push(dependency);
                    }
                    break Some(port);
                }
                Some(holder) if named => {
                    placement.problems.push(format!(
                        "port {port} is held by {}, which keeps running",
                        field.port_holder(holder)
                    ));
                    break Some(port);
                }
                Some(_) => {}
                None if probe(&host, port) => break Some(port),
                None if named => {
                    placement.problems.push(format!("port {port} is in use"));
                    break Some(port);
                }
                None => {}
            }
        };
        let Some(port) = port else {
            break;
        };
        placement.instances.push(Planned {
            letter,
            gpus: group.to_vec(),
            uuids: group
                .iter()
                .filter_map(|index| {
                    field
                        .gpus
                        .iter()
                        .find(|gpu| gpu.index == *index)
                        .and_then(|gpu| gpu.uuid.clone())
                })
                .collect(),
            host: host.clone(),
            port,
            after,
        });
    }
    placement
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::launch::occupancy::tests::uuid;

    fn gpu(index: u32, occupancy: Occupancy) -> Gpu {
        Gpu {
            index,
            uuid: Some(uuid(index)),
            memory_total: Some(96 << 30),
            occupancy,
            graphics: Vec::new(),
        }
    }

    fn serve(pid: u32, ports: &[u16]) -> Serve {
        Serve {
            pid,
            start_ticks: u64::from(pid),
            name: Some("MiMo-V2.6-Flash".into()),
            ports: ports.to_vec(),
            starting: false,
        }
    }

    fn picked(indices: &[u32]) -> BTreeSet<u32> {
        indices.iter().copied().collect()
    }

    #[test]
    fn picked_gpus_split_into_lettered_groups_in_index_order() {
        let gpus = (0..4)
            .map(|index| gpu(index, Occupancy::Free))
            .collect::<Vec<_>>();
        let ports = BTreeMap::new();
        let field = Field {
            gpus: &gpus,
            ports: &ports,
            reservations: &[],
        };
        let mut free = |_: &str, _: u16| true;
        let placement = place(&field, &picked(&[3, 0, 2, 1]), 2, None, &mut free);
        let instances = placement
            .instances
            .iter()
            .map(|planned| (planned.letter, planned.gpus.clone(), planned.listen()))
            .collect::<Vec<_>>();
        assert_eq!(
            instances,
            [
                ('A', vec![0, 1], "127.0.0.1:8310".to_owned()),
                ('B', vec![2, 3], "127.0.0.1:8311".to_owned()),
            ]
        );
        assert_eq!(placement.instances[1].uuids, [uuid(2), uuid(3)]);
        assert_eq!(placement.grouping, None);
        assert!(placement.problems.is_empty());
        let three = place(&field, &picked(&[0, 1, 2]), 2, None, &mut free);
        assert_eq!(
            three.grouping.as_deref(),
            Some("pick GPUs in groups of 2 for TP2")
        );
        assert!(three.instances.is_empty());
        let none = place(&field, &picked(&[]), 4, None, &mut free);
        assert_eq!(none.grouping.as_deref(), Some("pick GPUs with Space"));
    }

    #[test]
    fn a_reused_port_makes_the_stopping_instance_a_dependency() {
        let mimo = serve(48211, &[8310]);
        let other = serve(50000, &[8311]);
        let gpus = vec![
            gpu(0, Occupancy::Held(vec![Holder::Serve(mimo.clone())])),
            gpu(1, Occupancy::Held(vec![Holder::Serve(mimo.clone())])),
            gpu(2, Occupancy::Free),
            gpu(3, Occupancy::Free),
        ];
        let mut ports = BTreeMap::new();
        ports.insert(8310, PortHolder::Serve(mimo.clone()));
        ports.insert(8311, PortHolder::Serve(other.clone()));
        let field = Field {
            gpus: &gpus,
            ports: &ports,
            reservations: &[],
        };
        // 8312 is taken by something discovery does not know.
        let mut probe = |_: &str, port: u16| port != 8312;
        let placement = place(&field, &picked(&[2, 3, 0, 1]), 2, None, &mut probe);
        assert_eq!(placement.stops, std::slice::from_ref(&mimo));
        let a = &placement.instances[0];
        assert_eq!((a.gpus.clone(), a.port), (vec![0, 1], 8310));
        assert_eq!(a.after, [Dependency::Serve(mimo.clone())]);
        // B skips 8311, which an instance that keeps running holds, and
        // 8312, which the probe finds taken.
        let b = &placement.instances[1];
        assert_eq!((b.gpus.clone(), b.port), (vec![2, 3], 8313));
        assert!(b.after.is_empty());
        // An instance on free GPUs that reuses the stopping one's port
        // waits for it all the same.
        let gpus = vec![
            gpu(0, Occupancy::Free),
            gpu(1, Occupancy::Free),
            gpu(2, Occupancy::Held(vec![Holder::Serve(mimo.clone())])),
        ];
        let field = Field {
            gpus: &gpus,
            ports: &ports,
            reservations: &[],
        };
        let mut free = |_: &str, _: u16| true;
        let placement = place(&field, &picked(&[0, 2]), 1, None, &mut free);
        assert_eq!(placement.instances[0].port, 8310);
        assert_eq!(
            placement.instances[0].after,
            [Dependency::Serve(mimo.clone())]
        );
        assert_eq!(placement.instances[1].port, 8312);
        // A port named in the form that a running instance holds blocks.
        let named = place(&field, &picked(&[0]), 1, Some("127.0.0.1:8311"), &mut free);
        assert_eq!(
            named.problems,
            ["port 8311 is held by proc 50000, which keeps running"]
        );
        let busy = place(
            &field,
            &picked(&[0]),
            1,
            Some("0.0.0.0:9000"),
            &mut |_, _| false,
        );
        assert_eq!(busy.problems, ["port 9000 is in use"]);
        assert_eq!(busy.instances[0].listen(), "0.0.0.0:9000");
        let bad = place(&field, &picked(&[0]), 1, Some("9000"), &mut free);
        assert_eq!(bad.problems, ["listen must be HOST:PORT"]);
    }

    #[test]
    fn a_pending_launch_on_the_picked_gpus_or_port_is_replaced() {
        let reservations = vec![
            Reservation {
                label: "launch A".into(),
                model: "Qwen3.8-Flash".into(),
                gpus: vec![uuid(0), uuid(1)],
                port: 8310,
                process: None,
                launch: 1,
            },
            Reservation {
                label: "proc 61204".into(),
                model: "Qwen3.8-Flash".into(),
                gpus: vec![uuid(2), uuid(3)],
                port: 8311,
                process: Some((61204, 7)),
                launch: 2,
            },
        ];
        let gpus = vec![
            gpu(0, Occupancy::Held(vec![Holder::Launch(0)])),
            gpu(1, Occupancy::Held(vec![Holder::Launch(0)])),
            gpu(2, Occupancy::Held(vec![Holder::Launch(1)])),
            gpu(3, Occupancy::Held(vec![Holder::Launch(1)])),
        ];
        let mut ports = BTreeMap::new();
        ports.insert(8310, PortHolder::Launch(0));
        ports.insert(8311, PortHolder::Launch(1));
        let field = Field {
            gpus: &gpus,
            ports: &ports,
            reservations: &reservations,
        };
        let mut free = |_: &str, _: u16| true;
        // A second start on the waiting launch's GPUs replaces it, and
        // takes its port; the starting one's port is skipped.
        let placement = place(&field, &picked(&[0, 1]), 2, None, &mut free);
        assert_eq!(placement.replaces, [0]);
        assert_eq!(placement.instances[0].port, 8310);
        assert_eq!(placement.instances[0].after, [Dependency::Launch(0)]);
        // Naming the starting launch's port replaces it too.
        let named = place(
            &field,
            &picked(&[0, 1]),
            2,
            Some("127.0.0.1:8311"),
            &mut free,
        );
        assert_eq!(named.replaces, [0, 1]);
        assert!(named.problems.is_empty());
        // Both launches, on all four GPUs.
        let both = place(&field, &picked(&[0, 1, 2, 3]), 4, None, &mut free);
        assert_eq!(both.replaces, [0, 1]);
        assert_eq!(
            both.instances[0].after,
            [Dependency::Launch(0), Dependency::Launch(1)]
        );
    }

    #[test]
    fn unknown_and_foreign_gpus_block_with_the_reason() {
        let gpus = vec![
            gpu(
                0,
                Occupancy::Unknown("its processes have not been read".into()),
            ),
            gpu(1, Occupancy::Blocked("docker glm-tp4".into())),
        ];
        let ports = BTreeMap::new();
        let field = Field {
            gpus: &gpus,
            ports: &ports,
            reservations: &[],
        };
        let placement = place(&field, &picked(&[0, 1, 7]), 3, None, &mut |_, _| true);
        assert_eq!(
            placement.problems,
            [
                "GPU 0 is unknown: its processes have not been read",
                "GPU 1 is held by docker glm-tp4",
                "GPU 7 is gone",
            ]
        );
        assert_eq!(parse_listen("[::1]:8310"), Some(("[::1]".to_owned(), 8310)));
        assert_eq!(parse_listen(":8310"), None);
        assert_eq!(parse_listen("host:0"), None);
    }
}
