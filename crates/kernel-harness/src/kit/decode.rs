//! Verify rows against decode rows, for cases whose rows are items of
//! several rows (attention queries, recurrent sequences).
//!
//! `Rows::OneRow { max }` says a launch of up to `max` rows gives every row
//! the bits of that row launched alone through the operation's one-row
//! entry. For a row op that entry is the op itself at one row, which the
//! rows property checks ([`super::properties`]). For an op over items of
//! several rows (a verify step) it is another op, the decode op of the
//! same query, over items of one row each: a case row then moves a whole
//! item, so the rows property cannot split it. A case of such items
//! declares so ([`super::OpCase::item_rows`]) and gives a [`Decoded`]: the
//! decode op's case and how the verify step's items map to its decode
//! launches. The rows property then runs the verify step over every item
//! of up to `max` rows and each mapped decode launch of one row alone, and
//! requires the bytes each mapping names to be equal. A case of such items
//! whose op claims or needs one-row bits without a mapping does not
//! declare ([`super::OpCase::rows_mapping`]): the check would have nothing
//! to compare, so it is never passed vacuously.

use super::OpCase;
use crate::conformance::Images;
use crate::{LabResult, Session};
use runtime_cuda::Function;
use serde_json::json;
use std::collections::BTreeMap;
use std::ops::Range;

/// One output's bytes of a launch's results, by role path.
pub type Bytes = BTreeMap<String, Vec<u8>>;

/// Bytes of the verify step that a decode launch must reproduce.
#[derive(Clone, Debug, PartialEq)]
pub struct Compared {
    /// The verify step's role path and byte range.
    pub verify: (String, Range<usize>),
    /// The decode launch's role path and byte range, of equal length.
    pub decode: (String, Range<usize>),
}

/// One decode launch of one row: the decode case's images, and what it
/// compares with the verify step.
#[derive(Clone, Debug)]
pub struct DecodeLaunch {
    /// Which verify row it is, for reports.
    pub label: String,
    pub images: Images,
    pub compared: Vec<Compared>,
}

/// The verify step a check runs: its images and the results it gave, at
/// `live` items of up to `max` rows each.
pub struct Verified<'v> {
    pub images: &'v Images,
    pub results: &'v Bytes,
    pub live: u32,
    pub max: u32,
}

/// The verify step's images over the case's own: its items cut to at most
/// `max` rows.
type VerifyImages<'a> = Box<dyn Fn(&Images, u32) -> LabResult<Images> + 'a>;

/// The decode launches of a verify step over the decode case's own images.
type Launches<'a> = Box<dyn Fn(&Verified<'_>, &Images) -> LabResult<Vec<DecodeLaunch>> + 'a>;

/// How a verify step's items map to one-row launches of a decode op.
pub struct Decoded<'a> {
    /// The decode op's case, whose buffers hold any decode launch the
    /// mapping gives. Its declaration is checked with the verify case's.
    pub(crate) case: OpCase<'a>,
    verify: VerifyImages<'a>,
    launches: Launches<'a>,
}

impl<'a> Decoded<'a> {
    /// The decode op's `case`; `verify` gives the verify step's images from
    /// the case's images and a row limit (items cut to at most that many
    /// rows), and `launches` the decode launches of a verify step (one row
    /// each, from the decode case's images), each naming the bytes it must
    /// reproduce.
    pub fn new(
        case: OpCase<'a>,
        verify: impl Fn(&Images, u32) -> LabResult<Images> + 'a,
        launches: impl Fn(&Verified<'_>, &Images) -> LabResult<Vec<DecodeLaunch>> + 'a,
    ) -> Self {
        Self {
            case,
            verify: Box::new(verify),
            launches: Box::new(launches),
        }
    }

    /// The verify step's images over the case's `images` at a claim of
    /// `max` rows.
    pub(crate) fn verify_images(&self, images: &Images, max: u32) -> LabResult<Images> {
        (self.verify)(images, max)
    }

    /// The decode launches of the verify step over `images` at `live` items
    /// of up to `max` rows, given the verify step's launch: what
    /// [`Decoded::check`] runs, without a GPU.
    pub(crate) fn plan(
        &self,
        images: &Images,
        live: u32,
        max: u32,
        execute: &dyn Fn(&Images) -> LabResult<Bytes>,
    ) -> LabResult<(Images, Bytes, Vec<DecodeLaunch>)> {
        let images = self.verify_images(images, max)?;
        let results = execute(&images)?;
        let verified = Verified {
            images: &images,
            results: &results,
            live,
            max,
        };
        let launches = (self.launches)(&verified, &self.case.case.images())?;
        Ok((images, results, launches))
    }

    /// Runs the verify step over the case's `images` at `live` items of up
    /// to `max` rows with `execute`, then each decode launch alone, and
    /// compares the bytes each names. Fails when nothing is compared.
    pub(crate) fn check(
        &self,
        session: &Session,
        functions: &dyn Fn(&str) -> LabResult<Function>,
        images: &Images,
        live: u32,
        max: u32,
        execute: &dyn Fn(&Images) -> LabResult<Bytes>,
    ) -> LabResult<serde_json::Value> {
        let (_, verify, launches) = self.plan(images, live, max, execute)?;
        let mut bytes = 0;
        for launch in &launches {
            let decode = self
                .case
                .case
                .execute(session, functions, &launch.images, 1, 1)?;
            bytes += compare(&launch.label, &launch.compared, &verify, &decode)?;
        }
        if bytes == 0 {
            return Err(
                "the mapping compares no byte of the verify step with a decode launch".into(),
            );
        }
        Ok(json!({
            "items": live,
            "max_rows": max,
            "decode_launches": launches.len(),
            "decode_op": self.case.info().implementation,
            "compared_bytes": bytes,
        }))
    }
}

/// Compares the ranges `compared` names between the verify step's and a
/// decode launch's results, naming the first byte that differs; returns
/// the bytes compared.
pub(crate) fn compare(
    label: &str,
    compared: &[Compared],
    verify: &Bytes,
    decode: &Bytes,
) -> LabResult<usize> {
    let mut bytes = 0;
    for Compared {
        verify: (verify_path, verify_range),
        decode: (decode_path, decode_range),
    } in compared
    {
        if verify_range.len() != decode_range.len() {
            return Err(format!(
                "{label}: {verify_path} and {decode_path} compare ranges of unequal length"
            ));
        }
        let get = |results: &Bytes, path: &str, range: &Range<usize>| -> LabResult<Vec<u8>> {
            results
                .get(path)
                .and_then(|bytes| bytes.get(range.clone()))
                .map(<[u8]>::to_vec)
                .ok_or_else(|| format!("{label}: the launch gave no bytes {range:?} of {path}"))
        };
        let left = get(verify, verify_path, verify_range)?;
        let right = get(decode, decode_path, decode_range)?;
        if let Some(at) = left.iter().zip(&right).position(|(a, b)| a != b) {
            return Err(format!(
                "{label}: verify {verify_path} byte {} is {:#04x}, decode {decode_path} byte {} \
                 is {:#04x}",
                verify_range.start + at,
                left[at],
                decode_range.start + at,
                right[at]
            ));
        }
        bytes += left.len();
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn results(path: &str, bytes: Vec<u8>) -> Bytes {
        Bytes::from([(path.to_owned(), bytes)])
    }

    fn compared(verify: Range<usize>, decode: Range<usize>) -> Vec<Compared> {
        vec![Compared {
            verify: ("output".into(), verify),
            decode: ("output".into(), decode),
        }]
    }

    /// The comparison names the first differing byte by both launches'
    /// offsets, and refuses ranges of unequal length or past the results.
    #[test]
    fn decode_launches_compare_the_ranges_they_name() {
        let verify = results("output", vec![1, 2, 3, 4, 5, 6]);
        let decode = results("output", vec![3, 4, 0xff]);
        assert_eq!(
            compare("row", &compared(2..4, 0..2), &verify, &decode),
            Ok(2)
        );
        let error = compare("row 2", &compared(3..5, 1..3), &verify, &decode).unwrap_err();
        assert!(
            error.starts_with("row 2: verify output byte 4 is 0x05, decode output byte 2 is 0xff"),
            "{error}"
        );
        assert!(compare("row", &compared(0..2, 0..1), &verify, &decode).is_err());
        assert!(compare("row", &compared(4..8, 0..4), &verify, &decode).is_err());
    }
}
