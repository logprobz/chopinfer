//! Whether ABI V1 may be called stable, and what it would take.
//!
//! Two separately compiled bundle fixtures running through one unchanged host
//! binary is a necessary result, not a sufficient one. The unchanged-host test
//! passes on macOS, where every device service is host memory and no CUDA is
//! compiled at all, so a macOS pass says the boundary holds without a driver
//! and says nothing about SM120.
//!
//! Freezing V1 takes two artifacts this repository does not have: an authentic
//! Linux unchanged-host result and an SM120 smoke receipt. This test is what
//! keeps the header, the architecture document, and those two files saying the same
//! thing, in both directions. While either artifact is absent, the header and
//! the architecture document must call V1 a draft. Once both exist, they must stop.

use std::fs;
use std::path::{Path, PathBuf};

/// The header the ABI is declared in.
const HEADER: &str = "include/chop_bundle_v1.h";

/// The architecture document that governs it.
const ARCHITECTURE: &str = "BUNDLE_ARCHITECTURE.md";

/// The Linux unchanged-host result, which the Linux fixture gate produces.
const LINUX_UNCHANGED_HOST_RESULT: &str = "fixtures/bundle-abi-v1/linux-unchanged-host.json";

/// The SM120 smoke receipt, which only a run on that device produces.
const SM120_SMOKE_RECEIPT: &str = "fixtures/bundle-abi-v1/sm120-smoke.json";

/// What the header says while V1 is not frozen.
const HEADER_DRAFT_NOTICE: &str = "DRAFT: this V1 header is not stable.";

/// What the architecture document says while V1 is not frozen.
const ARCHITECTURE_DRAFT_NOTICE: &str = "The V1 header remains a draft.";

#[test]
fn abi_v1_is_draft_until_both_prerequisites_exist_in_this_repository() {
    let header = read(HEADER);
    let architecture = read(ARCHITECTURE);

    // Both documents name both prerequisites, so a reader of either one learns
    // what is missing without having to find this test.
    for prerequisite in [LINUX_UNCHANGED_HOST_RESULT, SM120_SMOKE_RECEIPT] {
        assert!(
            header.contains(prerequisite),
            "{HEADER} must name {prerequisite} as a condition of stability"
        );
        assert!(
            architecture.contains(prerequisite),
            "{ARCHITECTURE} must name {prerequisite} as a condition of stability"
        );
    }

    let missing = [LINUX_UNCHANGED_HOST_RESULT, SM120_SMOKE_RECEIPT]
        .into_iter()
        .filter(|prerequisite| !root().join(prerequisite).is_file())
        .collect::<Vec<_>>();

    if missing.is_empty() {
        assert!(
            !header.contains(HEADER_DRAFT_NOTICE),
            "both prerequisites exist, so {HEADER} must no longer call V1 a draft"
        );
        assert!(
            !architecture.contains(ARCHITECTURE_DRAFT_NOTICE),
            "both prerequisites exist, so {ARCHITECTURE} must no longer call V1 a draft"
        );
        return;
    }

    assert!(
        header.contains(HEADER_DRAFT_NOTICE),
        "{missing:?} are absent, so {HEADER} must still say '{HEADER_DRAFT_NOTICE}'"
    );
    assert!(
        architecture.contains(ARCHITECTURE_DRAFT_NOTICE),
        "{missing:?} are absent, so {ARCHITECTURE} must still say '{ARCHITECTURE_DRAFT_NOTICE}'"
    );
}

/// A prerequisite that exists has to say what produced it.
///
/// The unchanged-host gate passes on macOS over host-memory device services,
/// so a green local run is not the evidence either file stands for. Neither
/// file may therefore be a placeholder: the Linux result must name the platform
/// it ran on, and the smoke receipt must name the architecture it ran against,
/// or it is not the thing the header is waiting for.
#[test]
fn a_prerequisite_that_exists_names_what_produced_it() {
    for (prerequisite, claim) in [
        (LINUX_UNCHANGED_HOST_RESULT, "linux"),
        (SM120_SMOKE_RECEIPT, "sm_120"),
    ] {
        let path = root().join(prerequisite);
        if !path.is_file() {
            continue;
        }
        let recorded = read(prerequisite);
        assert!(
            recorded.contains(claim),
            "{prerequisite} must record '{claim}', or it is not the result V1 is waiting for"
        );
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read(relative: &str) -> String {
    let path: &Path = &root().join(relative);
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("{} is readable: {error}", path.display());
    })
}
