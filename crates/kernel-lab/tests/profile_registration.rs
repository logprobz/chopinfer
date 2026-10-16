//! Contributed entries must register their profile before cubins are built.
use kernel_core::contract::Refusal;
use kernels_sm120::{Catalogs, ModelCatalog, Profile, ProfileNames};

const THIRD: Profile = Profile::synthetic(ProfileNames {
    name: "third",
    tag: "third",
});

fn check(contribution: ModelCatalog) {
    let unregistered = Box::leak(Box::new(contribution));
    let registered = Box::leak(Box::new(ModelCatalog {
        profiles: &[THIRD],
        ..*unregistered
    }));
    let catalogs = Catalogs::new(&[registered]).expect("explicitly registered contribution");
    let cubins = catalogs.cubins();
    assert!(
        cubins
            .iter()
            .any(|cubin| cubin.name.contains("third") && !cubin.entries.is_empty())
    );
    let refusal = Some(Refusal("a model contribution uses an unregistered profile"));
    assert_eq!(Catalogs::new(&[unregistered]).err(), refusal);
    static OTHER_MODEL: ModelCatalog = ModelCatalog {
        name: "other-model",
        profiles: &[THIRD],
        ..ModelCatalog::EMPTY
    };
    assert_eq!(Catalogs::new(&[&OTHER_MODEL, unregistered]).err(), refusal);
}

#[test]
fn attention_requires_its_own_registered_profile() {
    check(ModelCatalog {
        name: "third",
        attention: || {
            let mut entry = kernels_sm120::attention::own_catalog().remove(0);
            entry.profile = THIRD;
            entry.name = "third_attention";
            vec![entry]
        },
        ..ModelCatalog::EMPTY
    });
}

#[test]
fn gated_requires_its_own_registered_profile() {
    check(ModelCatalog {
        name: "third",
        gated: || {
            let mut entry = model_qwen38_contract::sm120::attention::gated_entries().remove(0);
            entry.profile = THIRD;
            vec![entry]
        },
        ..ModelCatalog::EMPTY
    });
}

#[test]
fn sparse_requires_its_own_registered_profile() {
    check(ModelCatalog {
        name: "third",
        sparse: || {
            let mut entry = model_qwen38_contract::sm120::attention::sparse_catalog().remove(0);
            entry.profile = THIRD;
            vec![entry]
        },
        ..ModelCatalog::EMPTY
    });
}

#[test]
fn linear_requires_its_own_registered_profile() {
    check(ModelCatalog {
        name: "third",
        linear: || {
            let mut entry = model_qwen38_contract::sm120::linear::catalog().remove(0);
            entry.profile = THIRD;
            vec![entry]
        },
        ..ModelCatalog::EMPTY
    });
}

#[test]
fn hyper_connection_requires_its_own_registered_profile() {
    check(ModelCatalog {
        name: "third",
        hyper_connection_rows: || {
            kernels_sm120::row::hyper_connection::hc_entries(
                THIRD,
                model_qwen38_contract::sm120::row::HC,
            )
            .unwrap()
        },
        ..ModelCatalog::EMPTY
    });
}

#[test]
fn the_base_standard_profile_is_registered_implicitly() {
    static MODEL: ModelCatalog = ModelCatalog {
        name: "base-profile-contributor",
        rows: || {
            vec![kernels_sm120::row::embedding(
                Profile::StdDenseMoe,
                "registered_base",
                128,
                64,
                false,
            )]
        },
        ..ModelCatalog::EMPTY
    };
    let catalogs = Catalogs::new(&[&MODEL]).unwrap();
    assert!(
        catalogs
            .cubins()
            .iter()
            .flat_map(|cubin| &cubin.entries)
            .any(|entry| entry.symbol == "chop_row_registered_base_v1")
    );
}
