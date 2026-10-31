//! The HC boundary of Flash TP4 verify steps above the 64-row decode route,
//! with the `wide_verify` candidate (`verify_bounds::WIDE`): the all-reduce
//! round deferred into the column-block producer and split finish that
//! serve 33 to 64 rows (`hc_cols`), which write the round, the expansion,
//! the next mixer and its input norm. Each entry is the served one holding
//! the row bucket of the widest wide verify bucket the instance prepares
//! (the input norm takes row-bucket capacities, [`capacity`]), with its
//! symbol, cubin and arguments unchanged: every launch grids the step's
//! rows and indexes them in 64 bits, and the ops' capacity sizes only the
//! workspace.
//!
//! One backend holds one finish norm per mixer, so the wide boundary takes
//! a backend of its own and the decode route's ops keep their
//! registrations.
use kernel_core::backend::Backend;
use kernels_sm120::backend::Sm120;
use kernels_sm120::row::rmsnorm_quantize_cached;
use kernels_sm120::sinkhorn_hc::{finish_norm, finish_split, round_chain_partials, round_cols};
use model_deepseek41_contract::geometry::{FLASH, Geometry};
use model_deepseek41_contract::verify_bounds::WIDE;

/// The tuning bucket of the wide sites: the widest wide verify bucket.
pub(crate) const BUCKET: u32 = WIDE[WIDE.len() - 1];

/// Whether the wide HC site `op` admits `rows` verify rows.
pub(crate) fn admitted(op: &str, rows: u32) -> bool {
    super::hc_finish_norm::choice(&FLASH, op, BUCKET, rows).is_some()
}

/// The rows of the wide finish norm (`hc.finish_pair.verify`).
pub(crate) fn pair_buckets() -> Vec<u32> {
    super::hc_finish_norm::selected_rows("hc.finish_pair.verify", BUCKET)
}

/// The wide column-block pair's rows, tiles and warps
/// (`hc.finish_cols.verify`).
pub(crate) fn cols_buckets() -> Vec<finish_split::Bucket> {
    (1..=BUCKET)
        .filter_map(|rows| {
            let row = super::hc_finish_norm::choice(&FLASH, "hc.finish_cols.verify", BUCKET, rows)?;
            Some(finish_split::Bucket {
                rows,
                tile: row.knobs["tile"].count().expect("HC column tile"),
                transposed: row.knobs["transposed"].flag().expect("HC transpose"),
                warps: row.knobs["warps"].count().expect("HC finish warps"),
            })
        })
        .collect()
}

/// The capacity of the wide boundary's entries for verify steps of up to
/// `widest` rows: the least row bucket that holds them
/// (`kernels_sm120::row::smallest_bucket`), 256 at the package's 192.
pub(crate) fn capacity(widest: u32) -> u32 {
    kernels_sm120::row::smallest_bucket(widest).expect("a row bucket holds the wide verify rows")
}

/// `g`'s finish norm (`hc_finish_norm::entry`) holding `capacity` rows:
/// its rounded mixer and cached input norm at that capacity.
pub(crate) fn finish_entry(g: &Geometry, capacity: u32) -> finish_norm::Entry {
    let served = super::hc_finish_norm::entry(g);
    let norm = rmsnorm_quantize_cached::Entry {
        reference: kernels_sm120::row::rmsnorm_quantize::Entry {
            capacity,
            ..served.norm.reference.clone()
        },
        ..served.norm.clone()
    };
    finish_norm::Entry {
        mixer: kernels_sm120::sinkhorn_hc::expand_mix::Entry {
            capacity,
            ..served.mixer.clone()
        },
        norm,
        ..served
    }
}

/// The backend of the wide boundary of verify steps of up to `widest`
/// rows: `g`'s rounded mixer, input norm, its cached form, finish norm and
/// column-block pair (`hc_cols`), each holding [`capacity`] rows, at the
/// wide sites' buckets.
pub(crate) fn finish_backend(g: &Geometry, widest: u32) -> Sm120 {
    let finish = finish_entry(g, capacity(widest));
    let served_producer = super::hc_cols::producer();
    let producer = round_cols::Entry {
        reference: round_chain_partials::Entry {
            reference: finish.mixer.clone(),
            ..served_producer.reference.clone()
        },
        ..served_producer
    };
    let split = finish_split::Entry {
        reference: finish.clone(),
        ..super::hc_cols::finish()
    };
    Sm120::new()
        .with_sinkhorn_round_expand_mix(finish.mixer.clone())
        .with_rmsnorm_quantize(finish.norm.reference.clone())
        .with_rmsnorm_quantize_cached(finish.norm.clone(), super::norm_quantize_cached::buckets())
        .with_sinkhorn_finish_norm(finish, pair_buckets())
        .with_sinkhorn_finish_norm_cols(producer, split, cols_buckets())
}

impl crate::Kernels {
    /// The backend of the HC boundary above the decode route up to
    /// `widest` rows, the widest wide verify bucket ([`finish_backend`]).
    pub(crate) fn wide_hc_backend(g: &Geometry, widest: u32) -> Box<dyn Backend> {
        Box::new(finish_backend(g, widest))
    }

    /// Whether the wide HC site `op` admits `rows` verify rows
    /// ([`admitted`]).
    pub(crate) fn wide_hc_admitted(op: &str, rows: u32) -> bool {
        admitted(op, rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wide sites cover every verify row above the decode route with
    /// the column pair of 33 to 64 rows, and no row it serves.
    #[test]
    fn wide_sites_take_the_served_column_pair_above_the_decode_route() {
        let rows: Vec<u32> = (65..=BUCKET).collect();
        assert_eq!(pair_buckets(), rows);
        let served = super::super::hc_cols::buckets();
        let at_64 = served.iter().find(|b| b.rows == 64).unwrap();
        let wide = cols_buckets();
        assert_eq!(wide.iter().map(|b| b.rows).collect::<Vec<_>>(), rows);
        for b in &wide {
            assert_eq!(
                (b.tile, b.transposed, b.warps),
                (at_64.tile, at_64.transposed, at_64.warps)
            );
        }
        for rows in 1..=BUCKET {
            assert_eq!(admitted("hc.finish_admit.verify", rows), rows > 64);
            assert!(!admitted("hc.wide_mix.verify", rows));
        }
    }

    /// The wide entries change only their capacity: every cubin spec, and
    /// so every symbol and lock record, is the served one's.
    #[test]
    fn wide_entries_keep_the_served_specs() {
        let served = super::super::hc_finish_norm::entry(&FLASH);
        assert_eq!(
            (capacity(65), capacity(128), capacity(BUCKET)),
            (128, 128, 256)
        );
        let wide = finish_entry(&FLASH, capacity(BUCKET));
        assert_eq!(wide.mixer.capacity, 256);
        assert_eq!(wide.norm.reference.capacity, 256);
        wide.validate().unwrap();
        assert_eq!(
            format!("{:?}", served.cubin_spec("tail").unwrap()),
            format!("{:?}", wide.cubin_spec("tail").unwrap())
        );
        assert_eq!(
            format!("{:?}", served.norm.cubin_spec("norm").unwrap()),
            format!("{:?}", wide.norm.cubin_spec("norm").unwrap())
        );
        assert_eq!(
            format!(
                "{:?}",
                kernels_sm120::sinkhorn_hc::round_expand_mix::cubin_spec("m", &[served.mixer])
                    .unwrap()
            ),
            format!(
                "{:?}",
                kernels_sm120::sinkhorn_hc::round_expand_mix::cubin_spec("m", &[wide.mixer])
                    .unwrap()
            )
        );
    }
}
