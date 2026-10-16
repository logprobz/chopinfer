//! Compile-only audit of the shared cluster decode template entries: the lab
//! instances (`gemm::cluster::lab_cubin_specs`) beside the served entries
//! they reproduce, and served entries that run the template.

use kernel_build::{BuildOptions, Toolchain};
use kernels_sm120::gemm::{self, cluster, fused};
use std::collections::BTreeMap;

#[test]
fn cluster_template_instances_build_like_their_served_entries() {
    let toolchain = Toolchain::detect().expect("CUDA 13.2 toolchain");
    assert!(!toolchain.skipped, "the template audit requires CUDA");
    let instances = cluster::lab_instances(kernel_lab::catalogs());
    let lab = cluster::lab_cubin_specs(kernel_lab::catalogs());
    let served_names: Vec<_> = {
        let mut names: Vec<_> = instances.iter().map(|i| i.served_cubin.clone()).collect();
        names.sort();
        names.dedup();
        names
    };
    let served = fused::cubin_specs(kernel_lab::catalogs(), None)
        .into_iter()
        .chain(gemm::cubin_specs(kernel_lab::catalogs(), None))
        .filter(|spec| served_names.contains(&spec.name));
    let specs: Vec<_> = lab.iter().cloned().chain(served).collect();
    let root = std::env::temp_dir().join(format!("cluster-template-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &toolchain,
        &specs,
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    let resources: BTreeMap<_, _> = reports
        .iter()
        .flat_map(|cubin| &cubin.entries)
        .map(|entry| (entry.symbol.clone(), entry))
        .collect();
    let mut differ = Vec::new();
    for item in &instances {
        let template = resources[item.symbol()];
        assert_eq!(template.local_bytes, 0, "{} spills", item.symbol());
        assert!(
            template.shared_static_bytes + item.shape.dynamic_shared_bytes() <= 101_376,
            "{} exceeds shared memory",
            item.symbol()
        );
        if item.serves_template() {
            continue;
        }
        let served = resources[&item.served_symbol];
        assert_eq!(
            template.shared_static_bytes,
            served.shared_static_bytes,
            "{}",
            item.symbol()
        );
        if template.registers != served.registers {
            differ.push(format!(
                "{} registers {} served {}",
                item.symbol(),
                template.registers,
                served.registers
            ));
        }
    }
    println!(
        "{} instances, {} with other register counts:\n{}",
        instances.len(),
        differ.len(),
        differ.join("\n")
    );
    std::fs::remove_dir_all(root).unwrap();
}
