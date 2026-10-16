//! The compiled MiMo dense and row entries each topology selects from:
//! TP4's profile, which holds the rank-invariant entries, and its own.

pub mod dense {
    use kernel_core::contract::InstantiationSpec;
    use kernels_sm120::gemm::GemmVariant;
    /// The compiled MiMo dense entries of every topology with their
    /// profiles and declared claims, built once per process; their names
    /// are distinct.
    fn profiled() -> &'static [GemmVariant] {
        static ENTRIES: std::sync::OnceLock<Vec<GemmVariant>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| {
            crate::sm120::gemm::catalog()
                .into_iter()
                .filter(|entry| crate::sm120::PROFILES.contains(&entry.profile))
                .collect()
        })
    }

    /// The compiled MiMo dense entries of every topology.
    pub fn mimo_entries() -> impl Iterator<Item = &'static InstantiationSpec> {
        profiled().iter().map(|variant| &variant.instantiation)
    }

    /// The dense entries a `tp` instance can select: TP4's profile, which
    /// holds the router, and `tp`'s own.
    pub fn entries_for(
        tp: crate::topology::Tp,
    ) -> impl Iterator<Item = &'static InstantiationSpec> {
        variants_for(tp).map(|variant| &variant.instantiation)
    }

    /// [`entries_for`] with each entry's declared claims.
    pub fn variants_for(tp: crate::topology::Tp) -> impl Iterator<Item = &'static GemmVariant> {
        let own = crate::sm120::profile(tp);
        let base =
            crate::sm120::profile(crate::Tp::of_geometry(tp.geometry, 4).expect("listed geometry"));
        profiled()
            .iter()
            .filter(move |variant| variant.profile == base || variant.profile == own)
    }
}

pub mod row {
    use kernels_sm120::row::RowInstantiation;
    /// The compiled MiMo row entries of every topology, built once per
    /// process; their names are distinct.
    pub fn mimo_entries() -> &'static [RowInstantiation] {
        static ENTRIES: std::sync::OnceLock<Vec<RowInstantiation>> = std::sync::OnceLock::new();
        ENTRIES.get_or_init(|| {
            kernels_sm120::row::catalog(crate::sm120::catalogs())
                .into_iter()
                .filter(|entry| crate::sm120::PROFILES.contains(&entry.profile))
                .collect()
        })
    }

    /// The row entries a `tp` instance can select: TP4's profile, which
    /// holds the rank-invariant rows, and `tp`'s own.
    pub fn entries_for(tp: crate::topology::Tp) -> impl Iterator<Item = &'static RowInstantiation> {
        let own = crate::sm120::profile(tp);
        let base =
            crate::sm120::profile(crate::Tp::of_geometry(tp.geometry, 4).expect("listed geometry"));
        mimo_entries()
            .iter()
            .filter(move |entry| entry.profile == base || entry.profile == own)
    }
}
