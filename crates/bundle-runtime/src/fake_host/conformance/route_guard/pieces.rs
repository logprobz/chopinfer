//! Persist only compact route counts between bounded full-tier jobs.

use super::{Tier, audit};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

const MAX_RECEIPT: u64 = 256 * 1024 * 1024;
const MAX_TOTAL: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug)]
pub(super) struct Pieces {
    directory: PathBuf,
    index: Option<usize>,
    count: usize,
    identity: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Receipt {
    identity: String,
    index: usize,
    count: usize,
    samples: Vec<audit::Sample>,
    tags: Vec<audit::Tag>,
}

impl Pieces {
    pub fn configured(
        tier: Tier,
        bundle: &str,
        profile: &str,
        seed: u64,
        cases: usize,
    ) -> Option<Self> {
        if !matches!(tier, Tier::Full | Tier::Exhaustive) {
            return None;
        }
        let directory = std::env::var_os("CHOPINFER_ROUTE_PARTS_DIR")
            .expect("full sweeps run in bounded pieces; use tools/route_guard/sweep.py");
        let directory = PathBuf::from(directory)
            .canonicalize()
            .expect("existing route parts directory");
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        assert!(
            !directory.starts_with(root),
            "route receipts must be outside the repository"
        );
        let count = std::env::var("CHOPINFER_ROUTE_PARTS")
            .expect("route part count")
            .parse::<usize>()
            .expect("integer route part count");
        assert!(
            (1..=4096.min(cases)).contains(&count),
            "invalid route part count"
        );
        let action = std::env::var("CHOPINFER_ROUTE_PART").expect("route part index or audit");
        let index = if action == "audit" {
            None
        } else {
            let index = action.parse::<usize>().expect("integer route part index");
            assert!(index < count);
            Some(index)
        };
        // Bind every piece to the exact executable and deterministic case plan.
        // The runner supplies its SHA256 and never reuses an output directory.
        let build = std::env::var("CHOPINFER_ROUTE_BUILD").expect("route executable identity");
        assert!(build.len() == 64 && build.bytes().all(|c| c.is_ascii_hexdigit()));
        let identity = format!("{build}:{bundle}:{profile}:{tier:?}:{seed}:{cases}");
        Some(Self {
            directory,
            index,
            count,
            identity,
        })
    }

    pub fn audit(&self) -> bool {
        self.index.is_none()
    }
    pub fn includes(&self, index: usize) -> bool {
        index % self.count == self.index.unwrap()
    }
    fn path(&self, index: usize) -> PathBuf {
        self.directory.join(format!("part-{index}.json"))
    }

    pub fn save(&self, samples: Vec<audit::Sample>, tags: Vec<audit::Tag>) {
        let index = self.index.unwrap();
        assert!(!samples.is_empty(), "empty route piece");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.path(index))
            .expect("new route piece receipt");
        let mut writer = std::io::BufWriter::new(file);
        serde_json::to_writer(
            &mut writer,
            &Receipt {
                identity: self.identity.clone(),
                index,
                count: self.count,
                samples,
                tags,
            },
        )
        .expect("write route piece");
        writer.flush().expect("flush route piece");
        println!(
            "route piece {index}/{} saved; coverage verdict requires the final audit",
            self.count
        );
    }

    pub fn load(&self) -> (Vec<audit::Sample>, Vec<audit::Tag>) {
        assert!(self.audit());
        let mut budget = super::budget::Budget::begin("route receipt audit", true);
        let mut samples = Vec::new();
        let mut tags = std::collections::BTreeMap::new();
        let mut total = 0;
        for index in 0..self.count {
            let file =
                std::fs::File::open(self.path(index)).expect("every route piece must be present");
            let bytes = file.metadata().unwrap().len();
            total += bytes;
            assert!(
                bytes <= MAX_RECEIPT && total <= MAX_TOTAL,
                "route receipts exceed audit memory budget"
            );
            let receipt: Receipt =
                serde_json::from_reader(std::io::BufReader::new(file)).expect("valid route piece");
            assert_eq!(
                receipt.identity, self.identity,
                "mixed route plans or executables"
            );
            assert_eq!((receipt.index, receipt.count), (index, self.count));
            assert!(!receipt.samples.is_empty());
            samples.extend(receipt.samples);
            budget.check();
            for tag in receipt.tags {
                audit::remember(&mut tags, tag);
            }
        }
        (samples, tags.into_values().collect())
    }
}
