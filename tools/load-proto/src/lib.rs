//! Shared helpers of the direct-loading CPU checks (`qwen38-load-direct`,
//! `mimo26-load-direct`): argument parsing, process and page-cache
//! measurements, page-cache eviction, the bundle metadata a package passes
//! a plan ([`package_components`]), and the digest ledger of a load
//! ([`ledger`]).
pub mod ledger;

use runtime_bundle_abi::weight_plan::NamedBytesV1;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub const MIB: usize = 1 << 20;
pub const GIB: f64 = (1u64 << 30) as f64;

/// The bundle-metadata components (`chop.bundle-metadata.` roles) the host
/// passes a plan of the package at `root` loaded for `ranks` ranks, sorted
/// by role: the package is verified and narrowed to its one topology of that
/// rank count (`BundlePackageV1::for_topology`, as serving selects it), so
/// a component another topology declares under the same role is left out.
pub fn package_components(root: &Path, ranks: u32) -> Result<Vec<NamedBytesV1>> {
    let manifest = runtime_bundle::BundleManifestV1::from_canonical_json(&std::fs::read(
        root.join("manifest.json"),
    )?)?;
    let topology = match manifest
        .target()
        .topologies()
        .iter()
        .filter(|topology| topology.rank_count() == ranks)
        .collect::<Vec<_>>()[..]
    {
        [topology] => topology.id().as_str().to_owned(),
        [] => return Err(format!("the package declares no topology of {ranks} ranks").into()),
        _ => {
            return Err(format!("the package declares several topologies of {ranks} ranks").into());
        }
    };
    let architecture = manifest
        .target()
        .architectures()
        .first()
        .ok_or("the package names no architecture")?;
    let host = runtime_bundle::BundleHostCompatibilityV1::new(
        architecture.as_str(),
        manifest.abi_version(),
    )?
    .with_capabilities(manifest.capabilities_of_every_topology().cloned());
    let package =
        runtime_bundle::BundlePackageV1::verify_directory(root, &host)?.for_topology(&topology)?;
    let mut components = Vec::new();
    for component in package.manifest().components() {
        if component
            .role()
            .as_str()
            .starts_with("chop.bundle-metadata.")
        {
            components.push(NamedBytesV1 {
                name: component.role().as_str().into(),
                bytes: package.read_component(component)?,
            });
        }
    }
    components.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(components)
}

/// Drops the page cache of `paths` with POSIX_FADV_DONTNEED over the whole
/// file (GNU dd's `iflag=nocache count=0`), without privileges.
pub fn evict(paths: &[PathBuf]) -> Result<()> {
    for path in paths {
        let status = std::process::Command::new("dd")
            .arg(format!("if={}", path.display()))
            .args(["iflag=nocache", "count=0", "status=none"])
            .status()?;
        if !status.success() {
            return Err(format!("evict {}: dd failed", path.display()).into());
        }
    }
    Ok(())
}

/// Peak resident set of this process, from /proc.
pub fn peak_rss_bytes() -> u64 {
    status_kib("VmHWM:") * 1024
}

fn status_kib(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1)?.parse().ok())
        })
        .unwrap_or(0)
}

/// The CPUs this process may run on.
pub fn allowed_cpus() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Cpus_allowed_list:"))
                .map(|l| l.split_whitespace().nth(1).unwrap_or("").to_string())
        })
        .unwrap_or_default()
}

/// User and system CPU seconds of this process (clock ticks at 100 Hz).
pub fn cpu_seconds() -> (f64, f64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let tail = stat.rsplit_once(')').map_or("", |(_, t)| t);
    let fields: Vec<_> = tail.split_whitespace().collect();
    let tick = |i: usize| {
        fields
            .get(i)
            .and_then(|f| f.parse::<f64>().ok())
            .unwrap_or(0.0)
            / 100.0
    };
    // After the command name: state is field 0, utime 11 and stime 12.
    (tick(11), tick(12))
}

/// Page-cache bytes on the host, from /proc/meminfo.
pub fn cached_bytes() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("Cached:"))
                .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
        })
        .unwrap_or(0)
        * 1024
}

/// Parses `--key value` pairs and bare `--flag`s.
pub fn parse_args(args: impl Iterator<Item = String>) -> Result<BTreeMap<String, String>> {
    let args: Vec<_> = args.collect();
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        let key = args[i]
            .strip_prefix("--")
            .ok_or_else(|| format!("unexpected {}", args[i]))?;
        if i + 1 < args.len() && !args[i + 1].starts_with("--") {
            map.insert(key.to_string(), args[i + 1].clone());
            i += 2;
        } else {
            map.insert(key.to_string(), String::new());
            i += 1;
        }
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    //! The bundle metadata `--package-index` passes a plan: a package
    //! serving two topologies declares a component of the same role for
    //! each, and a load of either rank count gets its own topology's alone.
    use super::package_components;
    use model_compiler_sdk::{
        BundleDraftV1, ComponentContentV1, ComponentDraftV1, StagedContentV1,
        emit_bundle_directory_v1,
    };
    use runtime_bundle::{
        BundleGraphBucketV1, BundleKvLayoutV1, BundleOpaqueId, BundleRankPlacementV1,
        BundleRelativePathV1, BundleResourceRequirementsV1, BundleStateLayoutV1,
        BundleStateLifetimeV1, BundleTargetV1, BundleTopologyV1,
    };

    const ROLE: &str = "chop.bundle-metadata.fixture.precision.v1";

    fn id(value: &str) -> BundleOpaqueId {
        BundleOpaqueId::new(value).unwrap()
    }

    fn path(value: &str) -> BundleRelativePathV1 {
        BundleRelativePathV1::new(value).unwrap()
    }

    /// One topology's draft: a component every topology shares and this
    /// topology's own precision document under the shared role.
    fn draft(topology: &str, ranks: u32) -> BundleDraftV1 {
        let shared = ComponentDraftV1::new(
            id("fixture.weights"),
            id("fixture.weights.v1"),
            id("fixture-weight-bytes-v1"),
            path("components/weights.bin"),
            ComponentContentV1::Whole(StagedContentV1::bytes(vec![7; 64])),
        );
        let precision = ComponentDraftV1::new(
            id(&format!("fixture.precision.{topology}")),
            id(ROLE),
            id("json"),
            path(&format!("components/precision-{topology}.json")),
            ComponentContentV1::Whole(StagedContentV1::bytes(
                format!(r#"{{"topology":"{topology}"}}"#).into_bytes(),
            )),
        )
        .with_topologies(vec![id(topology)]);
        let bucket = BundleGraphBucketV1::new(
            id("fixture-bucket"),
            vec![shared.id().clone(), precision.id().clone()],
            Vec::new(),
        )
        .unwrap();
        BundleDraftV1::new(
            id("fixture.package-components.v1"),
            BundleTargetV1::new(
                vec![id("sm_120f")],
                vec![BundleTopologyV1::new(id(topology), ranks).unwrap()],
            )
            .unwrap(),
            BundleResourceRequirementsV1::new(1, Vec::new()).unwrap(),
            BundleKvLayoutV1::new(
                id("fixture.kv.v1"),
                64,
                64,
                1,
                64,
                1,
                1_024,
                BundleRankPlacementV1::RankLocal,
                BundleStateLifetimeV1::Request,
            )
            .unwrap(),
            BundleStateLayoutV1::new(
                id("fixture.state.v1"),
                64,
                64,
                BundleStateLifetimeV1::Request,
            )
            .unwrap(),
        )
        .with_components(vec![shared, precision])
        .with_graph_buckets(vec![bucket])
    }

    #[test]
    fn each_rank_count_gets_its_own_topology_s_metadata() {
        let root =
            std::env::temp_dir().join(format!("load-proto-components-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let package = root.join("package");
        let joined = BundleDraftV1::for_topologies(vec![draft("tp2", 2), draft("tp4", 4)]).unwrap();
        emit_bundle_directory_v1(&joined, &package).unwrap();
        for (ranks, topology) in [(2, "tp2"), (4, "tp4")] {
            let components = package_components(&package, ranks).unwrap();
            let names: Vec<_> = components.iter().map(|c| c.name.as_str()).collect();
            assert_eq!(names, [ROLE], "{ranks} ranks");
            assert_eq!(
                components[0].bytes,
                format!(r#"{{"topology":"{topology}"}}"#).into_bytes()
            );
        }
        let error = package_components(&package, 8).unwrap_err().to_string();
        assert!(error.contains("no topology of 8 ranks"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }
}
