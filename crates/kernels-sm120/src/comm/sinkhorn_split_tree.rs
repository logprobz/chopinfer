//! The split Sinkhorn mHC comm epilogue with the input RMSNorm's sum of
//! squares in the 256-lane tree of
//! [`kernel_core::rmsnorm::RmsNormNumerics::HF_TREE_256`]
//! (`cuda/comm/epilogue/sinkhorn_split_tree.cuh`): [`super::sinkhorn_split`]'s
//! entry (the same [`Entry`] type, grid, plan and ABI), whose outputs are
//! the unfused launches' with the row engine's tree norm
//! ([`crate::row::rmsnorm_tree`]) bit for bit. Build its cubins with
//! [`cubin_spec`], never [`sinkhorn_split::cubin_spec`], which compiles the
//! ordered norm's body.

use std::path::PathBuf;

use super::sinkhorn_split::{self, Entry};
use crate::Profile;
use crate::row::rmsnorm_tree;
use kernel_build::{CubinSpec, EntrySpec};
use kernel_core::contract::{DType, Refusal};
use kernel_core::rmsnorm::RmsNormReduction;
use kernel_core::sinkhorn_hyper_connection::SinkhornMix;

/// The entry's header, relative to the CUDA source root.
pub const HEADER: &str = "comm/epilogue/sinkhorn_split_tree.cuh";
/// The lab's late-wait mutant (`sinkhorn_split.cuh`'s).
pub const LATE_WAIT_DEFINE: &str = "CHOP_COMM_TEST_LATE_WAIT";

/// The tree epilogue's rows entry of `mix` under `profile`,
/// `chop_sinkhorn_epilogue_{tag}_rows_split_tree_v1`: the split entry's
/// grid and ABI under its own symbol, its contract the split's with the
/// input norm's squares summed in the 256-lane tree.
pub fn entry(
    profile: Profile,
    mix: SinkhornMix,
    rmsnorm_eps_bits: u32,
    capacity: u32,
) -> Result<Entry, Refusal> {
    let mut entry = sinkhorn_split::entry(profile, mix, rmsnorm_eps_bits, capacity)?;
    entry.symbol = format!(
        "chop_sinkhorn_epilogue_{}_rows_split_tree_v1",
        profile.tag()
    );
    entry.norm = RmsNormReduction::WarpTree { threads: 256 };
    entry.contract()?;
    Ok(entry)
}

/// The tree epilogue's body for `entry`.
pub fn body(entry: &Entry) -> String {
    let mix = entry.mix;
    let mixer = format!(
        "chop::sinkhorn_hc::Mixer<{},{},{},{:#x}u,{:#x}u,{}>",
        mix.streams,
        mix.width,
        mix.iterations,
        mix.eps_bits,
        mix.norm_eps_bits,
        mix.fn_dtype == DType::F32
    );
    format!(
        "const chop::comm::epilogue::SinkhornSplitExpandMixTreeNorm<{mixer}> epilogue{{{{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)}}}};chop::comm::epilogue::sinkhorn_split_rows(summed,epilogue);"
    )
}

/// A cubin of the tree epilogue's `entries` named `name`, compiled with
/// each of `defines` set to 1.
pub fn cubin_spec(name: String, entries: &[Entry], defines: &[&str]) -> CubinSpec {
    CubinSpec {
        name,
        sources_root: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda"),
        include_dirs: vec![],
        headers: vec![format!("#include \"{HEADER}\"")],
        defines: defines
            .iter()
            .map(|define| ((*define).to_owned(), "1".to_owned()))
            .collect(),
        entries: entries
            .iter()
            .map(|entry| {
                EntrySpec::new(
                    entry.symbol.clone(),
                    entry.params.clone(),
                    body(entry),
                    sinkhorn_split::THREADS,
                    1,
                )
            })
            .collect(),
    }
}

/// The served mixer's tree epilogue under the Sinkhorn engine's served
/// profile, with the input norm's eps 1e-5.
pub fn catalog() -> Entry {
    let served = crate::sinkhorn_hc::SERVED;
    entry(
        crate::sinkhorn_hc::SERVED_PROFILE,
        served.mix,
        1e-5_f32.to_bits(),
        served.capacity,
    )
    .expect("the served tree Sinkhorn comm epilogue")
}

/// The lab's mutants of [`catalog`]: (suffix, define).
pub const MUTANTS: [(&str, &str); 2] = [
    ("_late_wait", LATE_WAIT_DEFINE),
    (
        "_distances_reversed",
        rmsnorm_tree::DISTANCES_REVERSED_DEFINE,
    ),
];

/// [`catalog`] with `suffix` before its version, as a mutant cubin names
/// it.
pub fn mutant(suffix: &str) -> Entry {
    let mut entry = catalog();
    entry.symbol = entry.symbol.replace("_v1", &format!("{suffix}_v1"));
    entry
}

/// The lab's cubins: `sinkhorn_epilogue_split_tree_{tag}_sm120` of
/// [`catalog`], and each mutant in
/// `sinkhorn_epilogue_split_tree{suffix}_{tag}_sm120`.
pub fn lab_cubin_specs() -> Vec<CubinSpec> {
    let tag = crate::sinkhorn_hc::SERVED_PROFILE.tag();
    let mut specs = vec![cubin_spec(
        format!("sinkhorn_epilogue_split_tree_{tag}_sm120"),
        &[catalog()],
        &[],
    )];
    for (suffix, define) in MUTANTS {
        specs.push(cubin_spec(
            format!("sinkhorn_epilogue_split_tree{suffix}_{tag}_sm120"),
            &[mutant(suffix)],
            &[define],
        ));
    }
    specs
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel_core::contract::{MAX_COMM_RANKS, NumericalContract};

    #[test]
    fn the_tree_epilogue_is_the_split_entry_with_its_own_body() {
        let tree = catalog();
        let split = sinkhorn_split::catalog();
        assert_eq!(
            tree.symbol,
            "chop_sinkhorn_epilogue_hc4x4096_rows_split_tree_v1"
        );
        assert_eq!(tree.params, split.params);
        // The contracts differ in the norm's order alone.
        assert_eq!(split.norm, RmsNormReduction::Ordered);
        assert_eq!(tree.norm, RmsNormReduction::WarpTree { threads: 256 });
        assert_ne!(tree.query(4), split.query(4));
        assert_eq!(
            tree.query(4),
            Entry {
                norm: tree.norm,
                ..split.clone()
            }
            .query(4)
        );
        assert_ne!(
            tree.query(4).numerical_contract_id().unwrap(),
            split.query(4).numerical_contract_id().unwrap()
        );
        for ranks in 1..=MAX_COMM_RANKS as u32 {
            for rows in [1, 2, 7, 32, 6 * 32, sinkhorn_split::MAX_ROWS] {
                assert_eq!(
                    tree.plan(rows, ranks).unwrap(),
                    split.plan(rows, ranks).unwrap()
                );
            }
        }
        assert_eq!(
            body(&tree),
            "const chop::comm::epilogue::SinkhornSplitExpandMixTreeNorm<chop::sinkhorn_hc::Mixer<4,4096,20,0x358637bdu,0x3727c5acu,false>> epilogue{{streams,post,comb,reduced,expanded,fn,base,scale,collapsed,weight,normed,partials,tickets,live_rows,__uint_as_float(eps_bits)}};chop::comm::epilogue::sinkhorn_split_rows(summed,epilogue);"
        );
        let specs = lab_cubin_specs();
        assert_eq!(
            specs.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            [
                "sinkhorn_epilogue_split_tree_hc4x4096_sm120",
                "sinkhorn_epilogue_split_tree_late_wait_hc4x4096_sm120",
                "sinkhorn_epilogue_split_tree_distances_reversed_hc4x4096_sm120",
            ]
        );
        assert!(specs[0].sources_root.join(HEADER).is_file());
        assert_eq!(specs[0].entries[0].body, body(&tree));
        assert_eq!(
            specs[1].entries[0].symbol,
            "chop_sinkhorn_epilogue_hc4x4096_rows_split_tree_late_wait_v1"
        );
        assert_eq!(specs[1].entries[0].body, specs[0].entries[0].body);
        assert_eq!(
            specs[2].defines,
            [(rmsnorm_tree::DISTANCES_REVERSED_DEFINE.into(), "1".into())]
        );
        assert!(
            entry(
                crate::sinkhorn_hc::SERVED_PROFILE,
                SinkhornMix {
                    width: 5120,
                    ..crate::sinkhorn_hc::SERVED.mix
                },
                1e-5_f32.to_bits(),
                8
            )
            .is_err()
        );
    }
}
