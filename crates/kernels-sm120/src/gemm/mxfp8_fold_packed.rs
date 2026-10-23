//! Entries that read E4M3 weight codes stored as `mxfp8-fold-packed-8x128-v1`
//! (`cuda/gemm/mxfp8_fold_packed.cuh`) instead of row-major.
//!
//! A packed entry is its row-major entry with only the weight-code address
//! changed: the same parameters, launch plan, dynamic shared memory and
//! arithmetic, so its outputs equal the row-major entry's bit for bit. It is
//! named by the row-major symbol plus [`SUFFIX`], and a prepared op moves to
//! packed codes by renaming its launches ([`launches`]) and binding the
//! packed view where it bound the row-major one.

use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::backend::Launch;
use kernel_core::contract::Refusal;
use kernel_core::precision::WeightLayout;
use kernel_core::weight_transform::mxfp8_fold::Mxfp8FoldLayout;

pub const SUFFIX: &str = "_packed_v1";

/// The packed entry of row-major entry `row_major`.
pub fn symbol(row_major: &str) -> String {
    format!("{row_major}{SUFFIX}")
}

/// The layout a packed entry's weight view holds.
pub fn layout() -> WeightLayout {
    Mxfp8FoldLayout::weight_layout()
}

/// Refuses weights the packed layout cannot hold: `groups` matrices of `n`
/// rows by `k` codes need N divisible by 8 and K divisible by 64.
pub fn check(groups: u32, n: u32, k: u32) -> Result<(), Refusal> {
    let layout = Mxfp8FoldLayout {
        groups: groups as usize,
        n: n as usize,
        k: k as usize,
    };
    layout
        .packed_bytes()
        .map(|_| ())
        .map_err(|_| Refusal("packed fold codes need N divisible by 8 and K by 64"))
}

/// `spec` with each row-major entry that `twins` names (by its packed symbol)
/// replaced by that packed twin: a cubin of a fold family whose weights are
/// stored packed. Each twin must replace exactly one entry; entries that read
/// no weight codes stay.
pub fn replace(
    mut spec: CubinSpec,
    twins: impl IntoIterator<Item = EntrySpec>,
) -> Result<CubinSpec, Refusal> {
    for twin in twins {
        let row_major = twin
            .symbol
            .strip_suffix(SUFFIX)
            .ok_or(Refusal("a packed twin is not named by the packed suffix"))?;
        let mut found = spec.entries.iter_mut().filter(|e| e.symbol == row_major);
        let (Some(entry), None) = (found.next(), found.next()) else {
            return Err(Refusal("a packed twin replaces no single row-major entry"));
        };
        *entry = twin;
    }
    Ok(spec)
}

/// `launches` with every symbol that `packed` accepts renamed to its packed
/// entry; companion launches that read no weight codes keep their symbol.
pub fn launches(mut launches: Vec<Launch>, packed: impl Fn(&str) -> bool) -> Vec<Launch> {
    for launch in &mut launches {
        if packed(&launch.symbol) {
            launch.symbol = symbol(&launch.symbol);
        }
    }
    launches
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{ArgumentList, LaunchPlan};

    #[test]
    fn packed_entries_rename_only_weight_readers_and_check_the_shape() {
        assert_eq!(symbol("chop_x_v2"), "chop_x_v2_packed_v1");
        assert_eq!(layout().id, "mxfp8-fold-packed-8x128-v1");
        assert!(check(2, 8, 64).is_ok() && check(1, 5120, 576).is_ok());
        for (groups, n, k) in [(1, 7, 64), (1, 8, 96), (0, 8, 64), (1, 8, 0)] {
            assert!(check(groups, n, k).is_err());
        }
        let launch = |symbol: &str| Launch {
            symbol: symbol.into(),
            rows: 1,
            plan: LaunchPlan {
                grid: [1, 1, 1],
                block: [32, 1, 1],
                dynamic_shared_bytes: 0,
                workspace_bytes: 0,
                cluster: [1, 1, 1],
                pdl: true,
            },
            arguments: ArgumentList(vec![]),
            collective: false,
            persistent: false,
        };
        let renamed = launches(vec![launch("fold"), launch("finish")], |s| s == "fold");
        assert_eq!(renamed[0].symbol, "fold_packed_v1");
        assert_eq!(renamed[1].symbol, "finish");
    }

    #[test]
    fn a_packed_cubin_replaces_each_weight_reader_once() {
        let entry = |symbol: &str| EntrySpec::new(symbol.into(), vec![], "".into(), 32, 1);
        let spec = CubinSpec {
            name: "c".into(),
            sources_root: std::path::PathBuf::new(),
            include_dirs: vec![],
            headers: vec![],
            defines: vec![],
            entries: vec![entry("admit_v1"), entry("fold_v1")],
        };
        let packed = replace(spec.clone(), [entry("fold_v1_packed_v1")]).unwrap();
        let symbols: Vec<_> = packed.entries.iter().map(|e| e.symbol.as_str()).collect();
        assert_eq!(symbols, ["admit_v1", "fold_v1_packed_v1"]);
        assert!(replace(spec.clone(), [entry("other_v1_packed_v1")]).is_err());
        assert!(replace(spec, [entry("fold_v1")]).is_err());
    }
}
