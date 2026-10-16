//! The launches each encoder records, as `rank`, symbol and rows lines, for
//! comparing the launch lists of two builds. Debug builds only: with
//! `ROWS_ENV` naming a directory, read once when an encoder is created, its
//! lines go to `<dir>/<label>-<n>.tsv`, `n` counting the encoders the
//! process created. Release builds compile a probe that is never on and read
//! no variable.

use std::sync::Arc;

#[cfg(debug_assertions)]
pub use on::{Probe, ROWS_ENV};

#[cfg(not(debug_assertions))]
pub use off::Probe;

#[cfg(debug_assertions)]
mod on {
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    /// The directory the probe writes to.
    pub const ROWS_ENV: &str = "CHOPINFER_ENCODER_ROWS_DIR";

    /// Encoders created so far, which number the files.
    static CREATED: AtomicUsize = AtomicUsize::new(0);

    pub struct Probe {
        file: Mutex<File>,
        /// The launches of the record being built, written once per rank
        /// that runs them.
        pending: Mutex<Vec<(String, u32)>>,
    }

    impl Probe {
        /// The probe of an encoder labelled `label`, when `ROWS_ENV` is set.
        pub fn configured(label: &str) -> Option<Arc<Probe>> {
            let dir = PathBuf::from(std::env::var_os(ROWS_ENV)?);
            let n = CREATED.fetch_add(1, Ordering::Relaxed);
            std::fs::create_dir_all(&dir).expect("encoder rows directory");
            let file = File::create(dir.join(format!("{label}-{n}.tsv"))).expect("encoder rows");
            Some(Arc::new(Probe {
                file: Mutex::new(file),
                pending: Mutex::default(),
            }))
        }

        /// A launch of the record being built.
        pub fn launch(&self, symbol: &str, rows: u32) {
            self.pending
                .lock()
                .expect("probe")
                .push((symbol.to_owned(), rows));
        }

        /// Writes the record's launches once for each of `ranks`.
        pub fn flush(&self, ranks: impl IntoIterator<Item = usize>) {
            let pending = std::mem::take(&mut *self.pending.lock().expect("probe"));
            for rank in ranks {
                for (symbol, rows) in &pending {
                    self.line(rank, symbol, *rows);
                }
            }
        }

        /// One launch of `rank`.
        pub fn line(&self, rank: usize, symbol: &str, rows: u32) {
            let mut file = self.file.lock().expect("probe");
            writeln!(file, "{rank}\t{symbol}\t{rows}").expect("encoder rows");
        }
    }
}

#[cfg(not(debug_assertions))]
mod off {
    use std::sync::Arc;

    pub struct Probe;

    impl Probe {
        pub fn configured(_: &str) -> Option<Arc<Probe>> {
            None
        }

        pub fn launch(&self, _: &str, _: u32) {}

        pub fn flush(&self, _: impl IntoIterator<Item = usize>) {}

        pub fn line(&self, _: usize, _: &str, _: u32) {}
    }
}

/// `probe`'s launch of `symbol`, if on.
pub(super) fn launch(probe: &Option<Arc<Probe>>, symbol: &str, rows: u32) {
    if let Some(probe) = probe {
        probe.launch(symbol, rows);
    }
}
