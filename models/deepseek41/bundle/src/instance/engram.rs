//! The instance's Engram rows (`crate::engram_rows`): each phase's hash
//! before the step opens, the host's gather of the tables' rows into every
//! rank's staging buffers as the phase uploads, and each request's history
//! through the step's end.
//!
//! The plan maps each Engram layer's table (`layers.{l}.engram.embed`, its
//! E4M3 codes and UE8M0 scales) as host-mapped views
//! (`chop.host-mapped-gather.v1`) under the names below. An image whose
//! rank index names none of them maps no table, and nothing is gathered;
//! one that names only some is refused.
//!
//! A step's rows are provisional: the history takes the rows a commit
//! keeps, a cancel that frees a request's slot forgets it, and so does a
//! retire. A prefix restore names the restored tokens.
//!
//! An instance serving images marks every image position as no text: the
//! vendor writes each position of an image span with `image_token_id`
//! (`prepare_vl_inputs`) and caches it as DEAD (`engram.py:160-173`), so
//! no n-gram spans an image. Its steps, commits and restores read the
//! text flags from the tokens alike; text-only tokens take none.

use bundle_runtime::inputs::PhaseInputs;
use bundle_runtime::ledger::{Committed, Ledger};
use model_deepseek41_contract::geometry::Geometry;
use model_deepseek41_contract::own_ops::engram::HashTables;
use runtime_bundle_abi::{BundleCompiledImageHandleV1, BundleStreamHandleV1, HostServicesV1};
use runtime_bundle_sdk::rank_index::RankIndex;

use crate::engram_rows::{self, Destination, EngramRows, Hashed, Mapped, Plane, Step, Table};

/// Engram layer `layer`'s table views: its codes, then its scales.
fn table_views(layer: u32) -> [String; 2] {
    ["weight", "scale"].map(|part| format!("layers.{layer}.engram.embed.{part}"))
}

/// The Engram rows of an instance of a geometry with Engram layers.
pub(super) struct Engram {
    rows: EngramRows,
    /// Each Engram layer's mapped table, in layer order, when the image
    /// maps them.
    tables: Option<Vec<Table>>,
    /// The open step's tokens, per request in item order.
    open: Vec<(u64, Vec<u32>)>,
    /// The image token of an instance serving images.
    image: Option<u32>,
}

/// Each token's text flag when any of `tokens` is the image token `image`:
/// `None` (every token text) otherwise.
fn text_flags(tokens: &[u32], image: Option<u32>) -> Option<Vec<bool>> {
    let image = image?;
    tokens
        .contains(&image)
        .then(|| tokens.iter().map(|&token| token != image).collect())
}

impl Engram {
    pub(super) fn register_resident(
        &self,
        services: &runtime_bundle_abi::HostServicesV1,
        image: runtime_bundle_abi::BundleCompiledImageHandleV1,
        rank: u32,
    ) -> Result<(), runtime_bundle_abi::BundleAbiError> {
        for table in self.tables.iter().flatten() {
            let planes =
                [&table.codes, &table.scales].map(|plane| runtime_bundle_abi::MappedRowPlaneV1 {
                    image,
                    rank,
                    view_id: plane.view.clone(),
                    row_bytes: plane.row_bytes,
                });
            match services.register_mapped_row_set(&planes) {
                Ok(())
                | Err(runtime_bundle_abi::BundleAbiError::MissingHostService(_))
                | Err(runtime_bundle_abi::BundleAbiError::Status(
                    runtime_bundle_abi::BundleStatusV1::Unsupported,
                )) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    pub(super) fn tables(&self) -> Option<&[Table]> {
        self.tables.as_deref()
    }
    /// The Engram rows of `geometry`, `None` without Engram layers; with
    /// `images`, image positions take no part in an n-gram.
    pub(super) fn new(geometry: &Geometry, images: bool) -> Result<Option<Engram>, String> {
        if geometry.engram_layers() == 0 {
            return Ok(None);
        }
        Ok(Some(Engram {
            rows: EngramRows::new(HashTables::of(geometry)?),
            tables: None,
            open: Vec::new(),
            image: images.then_some(geometry.shape.tokens.image),
        }))
    }

    /// The mapped views `index` names: every Engram layer's table views,
    /// or none. Records the tables when the image maps them.
    pub(super) fn mapped(
        &mut self,
        geometry: &Geometry,
        index: &RankIndex,
    ) -> Result<Vec<String>, String> {
        let names: Vec<String> = geometry
            .shape
            .engram
            .layers
            .iter()
            .flat_map(|&layer| table_views(layer))
            .collect();
        let named = names
            .iter()
            .filter(|name| index.views.contains_key(name.as_str()))
            .count();
        if named == 0 {
            return Ok(Vec::new());
        }
        if named != names.len() {
            return Err(format!(
                "rank {} maps {named} of the {} Engram table views",
                index.rank,
                names.len()
            ));
        }
        let e = &geometry.shape.engram;
        let plane = |view: &str, row_bytes: u32| Plane {
            view: view.as_bytes().to_vec(),
            row_bytes: u64::from(row_bytes),
        };
        self.tables = Some(
            names
                .chunks_exact(2)
                .map(|views| Table {
                    codes: plane(&views[0], e.head_dim),
                    scales: plane(&views[1], e.head_dim / 32),
                })
                .collect(),
        );
        Ok(names)
    }

    /// The hash ids of `inputs`' rows, each item's rows in turn.
    pub(super) fn hash(&self, inputs: &PhaseInputs) -> Result<Hashed, String> {
        let text = text_flags(&inputs.token_ids, self.image);
        let steps = steps(inputs, text.as_deref())?;
        let hashed = self.rows.hash(&steps).map_err(refused)?;
        if hashed.rows() != inputs.token_ids.len() {
            return Err("the Engram rows do not cover the phase's tokens".into());
        }
        Ok(hashed)
    }

    /// Has the host gather `hashed`'s table rows of every Engram layer into
    /// rank `rank`'s staging buffers on `stream`; nothing when the image
    /// maps no table or the phase has no rows.
    pub(super) fn gather(
        &self,
        services: &HostServicesV1,
        stream: BundleStreamHandleV1,
        image: BundleCompiledImageHandleV1,
        rank: u32,
        destinations: &[Destination],
        hashed: &Hashed,
    ) -> Result<(), engram_rows::Error> {
        let Some(tables) = &self.tables else {
            return Ok(());
        };
        let mapped = Mapped {
            image,
            rank,
            tables,
        };
        self.rows
            .gather(services, stream, mapped, destinations, hashed)
    }

    /// Holds the step's tokens of every phase until the step ends.
    pub(super) fn begin<'a>(&mut self, phases: impl IntoIterator<Item = &'a PhaseInputs>) {
        self.open.clear();
        for inputs in phases {
            for (item, &request) in inputs.request_ids.iter().enumerate() {
                let start = inputs.q_start[item].max(0) as usize;
                let len = inputs.q_len[item].max(0) as usize;
                self.open
                    .push((request, inputs.token_ids[start..start + len].to_vec()));
            }
        }
    }

    /// Moves each committed item's kept tokens into its request's history.
    pub(super) fn commit(&mut self, committed: &[Committed]) -> Result<(), String> {
        let open = std::mem::take(&mut self.open);
        for item in committed {
            let (_, tokens) = open
                .iter()
                .find(|(request, _)| *request == item.request)
                .ok_or("a committed item has no Engram rows")?;
            let text = text_flags(tokens, self.image);
            let step = Step {
                request: item.request,
                tokens,
                text: text.as_deref(),
            };
            self.rows
                .commit(&step, item.kept as usize)
                .map_err(refused)?;
        }
        Ok(())
    }

    /// Forgets each request of the ended step that `ledger` no longer
    /// holds.
    pub(super) fn cancelled(&mut self, ledger: &Ledger) {
        for (request, _) in std::mem::take(&mut self.open) {
            if ledger.slot_of(request).is_none() {
                self.rows.forget(request);
            }
        }
    }

    /// Sets a restored request's history from its prefix `tokens`.
    pub(super) fn restore(&mut self, request: u64, tokens: &[u32]) -> Result<(), String> {
        let text = text_flags(tokens, self.image);
        self.rows
            .restore(request, tokens, text.as_deref())
            .map_err(refused)
    }

    pub(super) fn forget(&mut self, request: u64) {
        self.rows.forget(request);
    }
}

fn refused(error: engram_rows::Error) -> String {
    match error {
        engram_rows::Error::Refused(reason) => reason.to_owned(),
        engram_rows::Error::Host(error) => format!("{error}"),
    }
}

/// Each item's rows of `inputs`: its request, its tokens, which must
/// follow the previous item's, and their flags of `text` (the phase's).
fn steps<'a>(inputs: &'a PhaseInputs, text: Option<&'a [bool]>) -> Result<Vec<Step<'a>>, String> {
    let mut next = 0;
    inputs
        .request_ids
        .iter()
        .enumerate()
        .map(|(item, &request)| {
            let start = usize::try_from(inputs.q_start[item]).ok();
            let len = usize::try_from(inputs.q_len[item]).ok();
            let (Some(start), Some(len)) = (start, len) else {
                return Err("an item's rows are negative".to_owned());
            };
            if start != next || start + len > inputs.token_ids.len() {
                return Err("the phase's items do not lay their rows end to end".to_owned());
            }
            next = start + len;
            Ok(Step {
                request,
                tokens: &inputs.token_ids[start..next],
                text: text.map(|text| &text[start..next]),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_deepseek41_contract::geometry::FLASH;

    fn inputs(requests: &[(u64, &[u32])]) -> PhaseInputs {
        let mut inputs = PhaseInputs::default();
        for &(request, tokens) in requests {
            inputs.request_ids.push(request);
            inputs.q_start.push(inputs.token_ids.len() as i32);
            inputs.q_len.push(tokens.len() as i32);
            inputs.token_ids.extend(tokens);
        }
        inputs
    }

    /// The tables are mapped when the index names every Engram layer's
    /// codes and scales, absent when it names none, and refused when it
    /// names some.
    #[test]
    fn the_tables_are_the_views_the_index_names() {
        let view = || runtime_bundle_sdk::rank_index::View {
            offset: 0,
            bytes: 256,
            dtype: "U8".into(),
            shape: vec![256],
            layout_id: "row-major".into(),
        };
        let mut index = RankIndex {
            rank: 1,
            image: "image".into(),
            image_bytes: 256,
            sha256: None,
            direct_identity: None,
            views: Default::default(),
        };
        let mut engram = Engram::new(&FLASH, false).unwrap().unwrap();
        assert!(engram.mapped(&FLASH, &index).unwrap().is_empty());
        assert!(engram.tables.is_none());
        let names: Vec<String> = FLASH
            .shape
            .engram
            .layers
            .iter()
            .flat_map(|&layer| table_views(layer))
            .collect();
        assert_eq!(names[0], "layers.1.engram.embed.weight");
        index.views.insert(names[0].clone(), view());
        assert!(engram.mapped(&FLASH, &index).is_err());
        for name in &names {
            index.views.insert(name.clone(), view());
        }
        assert_eq!(engram.mapped(&FLASH, &index).unwrap(), names);
        let tables = engram.tables.as_ref().unwrap();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[1].scales.view, names[3].as_bytes());
        assert_eq!(
            (tables[0].codes.row_bytes, tables[0].scales.row_bytes),
            (256, 8)
        );
    }

    /// A committed prefill's tokens hash the next step's rows as one
    /// sequence; a verify keeps only the rows its commit names, and a
    /// cancel forgets a request the ledger let go.
    #[test]
    fn the_history_follows_what_each_step_commits() {
        let mut engram = Engram::new(&FLASH, false).unwrap().unwrap();
        let prompt = [11, 12, 13, 14, 15];
        let whole = engram.hash(&inputs(&[(7, &prompt)])).unwrap();
        // The prompt in two chunks, the first committed.
        engram.begin([&inputs(&[(7, &prompt[..3])])]);
        let committed = |request, rows, kept| Committed {
            request,
            slot: 0,
            start: 0,
            rows,
            kept,
        };
        engram.commit(&[committed(7, 3, 3)]).unwrap();
        let rest = engram.hash(&inputs(&[(7, &prompt[3..])])).unwrap();
        let width = whole.ids().len() / 5;
        assert_eq!(rest.ids(), &whole.ids()[3 * width..]);
        // A verify of three rows keeps one: only token 14 joins.
        engram.begin([&inputs(&[(7, &[14, 99, 98])])]);
        engram.commit(&[committed(7, 3, 1)]).unwrap();
        let next = engram.hash(&inputs(&[(7, &[15])])).unwrap();
        assert_eq!(next.ids(), &whole.ids()[4 * width..]);
        // A step of an unknown item's request refuses to commit.
        engram.begin([&inputs(&[(8, &[1])])]);
        assert!(engram.commit(&[committed(9, 1, 1)]).is_err());
        // Items must lay their rows end to end.
        let mut gap = inputs(&[(7, &[1]), (8, &[2])]);
        gap.q_start[1] = 2;
        gap.token_ids.push(3);
        assert!(engram.hash(&gap).is_err());
    }

    /// An instance serving images hashes image positions as DEAD, as the
    /// vendor does: a prompt with an image span hashes as its explicit text
    /// flags do, the text after the span as a sequence starting there, and
    /// the history a commit or restore keeps holds the DEAD ids. A
    /// text-only prompt hashes as on an instance without images.
    #[test]
    fn image_positions_take_no_part_in_an_n_gram() {
        let image = FLASH.shape.tokens.image;
        let with = Engram::new(&FLASH, true).unwrap().unwrap();
        let without = Engram::new(&FLASH, false).unwrap().unwrap();
        let text = [11, 12, 13, 14, 15];
        assert_eq!(
            with.hash(&inputs(&[(7, &text)])).unwrap(),
            without.hash(&inputs(&[(7, &text)])).unwrap()
        );
        let prompt = [11, 12, image, image, image, 13, 14, 15];
        let hashed = with.hash(&inputs(&[(7, &prompt)])).unwrap();
        let flags: Vec<bool> = prompt.iter().map(|&token| token != image).collect();
        let explicit = with
            .rows
            .hash(&[Step {
                request: 7,
                tokens: &prompt,
                text: Some(&flags),
            }])
            .unwrap();
        assert_eq!(hashed, explicit);
        // The text after the span hashes as a sequence that starts there.
        let width = hashed.ids().len() / prompt.len();
        let after = without.hash(&inputs(&[(8, &prompt[5..])])).unwrap();
        assert_eq!(&hashed.ids()[5 * width..], after.ids());
        assert_ne!(
            without.hash(&inputs(&[(7, &prompt)])).unwrap().ids()[5 * width..],
            after.ids()[..]
        );
        // A committed chunk ending in the span keeps DEAD ids: the rest
        // hashes as in the whole prompt.
        let mut chunked = Engram::new(&FLASH, true).unwrap().unwrap();
        chunked.begin([&inputs(&[(7, &prompt[..4])])]);
        chunked
            .commit(&[Committed {
                request: 7,
                slot: 0,
                start: 0,
                rows: 4,
                kept: 4,
            }])
            .unwrap();
        let rest = chunked.hash(&inputs(&[(7, &prompt[4..])])).unwrap();
        assert_eq!(rest.ids(), &hashed.ids()[4 * width..]);
        // So does a restored prefix.
        let mut restored = Engram::new(&FLASH, true).unwrap().unwrap();
        restored.restore(7, &prompt[..4]).unwrap();
        let rest = restored.hash(&inputs(&[(7, &prompt[4..])])).unwrap();
        assert_eq!(rest.ids(), &hashed.ids()[4 * width..]);
    }

    /// A cancel that invalidates the slots it touched forgets their
    /// requests and keeps the others; a retire forgets.
    #[test]
    fn a_cancel_forgets_the_requests_whose_slots_it_freed() {
        use bundle_runtime::ledger::{Item, Placement, Rules};
        let mut engram = Engram::new(&FLASH, false).unwrap().unwrap();
        let mut ledger = Ledger::new(
            4,
            Placement::Ledger,
            Rules {
                invalidate_on_cancel: true,
                tail_window: None,
            },
            64,
        );
        let item = |request, start, rows| Item {
            request,
            slot: None,
            start,
            rows,
            verify: false,
        };
        // Requests 7 and 8 commit two tokens each; the next step of 7
        // alone is cancelled.
        ledger
            .begin(&[item(7, 0, 2), item(8, 0, 2)], Vec::new())
            .unwrap();
        engram.begin([&inputs(&[(7, &[11, 12]), (8, &[21, 22])])]);
        let committed = ledger.commit(&[]).unwrap();
        engram.commit(&committed).unwrap();
        ledger.begin(&[item(7, 2, 1)], Vec::new()).unwrap();
        engram.begin([&inputs(&[(7, &[13])])]);
        ledger.cancel().unwrap();
        engram.cancelled(&ledger);
        assert!(engram.rows.history(7).is_empty());
        assert_eq!(engram.rows.history(8).len(), 2);
        engram.forget(8);
        assert!(engram.rows.history(8).is_empty());
    }
}
