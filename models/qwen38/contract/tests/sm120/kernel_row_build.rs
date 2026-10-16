//! Qwen3.8's hyper-connection, n-gram and PLE rows (`sm120::row`): their
//! plans and argument checks, the n-gram bodies of each rank count, and a
//! ptxas build of every profile's cubin. CPU only.

use kernel_build::{BuildOptions, Toolchain};
use kernel_core::contract::{Argument, StaticQuery};
use kernels_sm120::row::hyper_connection::{self, ngram_decode_entry};

use model_qwen38_contract::sm120::{catalogs, row};

#[test]
fn qwen_rows_compile_and_refuse_invalid_launches() {
    // TP4's rows, then the n-gram rows of TP2.
    assert_eq!(
        row::hyper_connection_rows().len(),
        model_qwen38_contract::sm120::geometry::MODELS.len() * (10 + 2)
    );
    for owner in model_qwen38_contract::sm120::geometry::MODELS {
        let profile = owner.model.at(4);
        let entries = row::hyper_connection_rows()
            .into_iter()
            .filter(|entry| entry.profile == profile)
            .collect::<Vec<_>>();
        assert_eq!(entries.len(), 10);
        // The per-sequence PLE convolution takes one decode row; the item entry
        // takes a row per bucket item.
        let conv = |symbol: &str| {
            entries
                .iter()
                .find(|e| e.symbol == symbol.replacen("qwen38", profile.tag(), 1))
                .unwrap()
        };
        assert!(
            conv("chop_row_qwen38_ple_conv_decode_v1")
                .plan(32, 2)
                .is_err()
        );
        assert!(
            conv("chop_row_qwen38_ple_conv_items_decode_v1")
                .plan(32, 32)
                .is_ok()
        );
        assert!(
            hyper_connection::cubin_specs(
                catalogs(),
                Some(
                    kernels_sm120::Model::new(
                        kernels_sm120::profile_names!("other-model", "other", 4, 2),
                        model_qwen38_contract::sm120::config::CONFIG
                    )
                    .at(4)
                )
            )
            .is_empty()
        );
        for entry in entries.iter() {
            assert_eq!(entry.profile, profile);
            assert!(matches!(
                entry.query,
                StaticQuery::HyperConnection(_) | StaticQuery::Ngram(_) | StaticQuery::Recurrent(_)
            ));
            entry.contract().unwrap();
            assert!(entry.plan(0, 0).is_err());
            assert!(entry.plan(8193, 1).is_err());
            assert!(entry.plan(32, 33).is_err());
            assert!(entry.pack_arguments(vec![]).is_err());
            assert!(
                entry
                    .pack_arguments(vec![Argument::U32(1); entry.params.len()])
                    .is_err()
            );
            for capacity in [1, 2, 8, 16, 32, 512, 8192] {
                let plan = entry.plan(capacity, 1).unwrap();
                assert_eq!(plan.grid[0], capacity);
                assert!(plan.pdl);
                assert_eq!(plan.workspace_bytes, 0);
            }
        }
    }
    let root = std::env::temp_dir().join(format!("qwen38-rows-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let reports = kernel_build::build(
        &Toolchain::detect().unwrap(),
        &hyper_connection::cubin_specs(catalogs(), None),
        &root.join("out"),
        &root.join("kernels.lock"),
        BuildOptions {
            lock_update: true,
            emit_cargo_directives: false,
            profile_line_info: false,
        },
    )
    .unwrap();
    assert_eq!(reports.len(), model_qwen38_contract::sm120::PROFILES.len());
    for report in reports.iter().flat_map(|cubin| &cubin.entries) {
        assert_eq!(report.local_bytes, 0, "{} spills", report.symbol);
    }
    std::fs::remove_dir_all(root).unwrap();
}

/// The derived n-gram bodies keep the spellings the TP4 cubins were locked
/// with; TP2 names its own.
#[test]
fn derived_rank_values_keep_the_locked_spellings() {
    for (tp, local, template) in [(2, 8, "<8>"), (4, 4, "")] {
        let profile = model_qwen38_contract::sm120::config::MODEL.at(tp);
        assert_eq!(
            model_qwen38_contract::sm120::config::profile_shape(profile)
                .unwrap()
                .ngram_heads(),
            local
        );
        let ngram = row::ngram(profile);
        let decode = ngram_decode_entry(
            &ngram,
            model_qwen38_contract::sm120::config::NGRAM_HEADER_LOCAL,
        )
        .unwrap();
        assert!(
            ngram
                .body
                .starts_with(&format!("chop::row::qwen38::ngram{template}("))
        );
        assert!(
            decode
                .body
                .starts_with(&format!("chop::row::qwen38::ngram_decode{template}("))
        );
    }
}
