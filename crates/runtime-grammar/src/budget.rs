//! Conservative cumulative allocation and work limits for one compilation.
use std::cell::Cell;
use std::rc::Rc;

use crate::GrammarError;

pub(crate) const MAX_ALTERNATIVES: usize = 16_384;
const MAX_WORK: usize = 512 * 1024 * 1024;
// Includes conservative capacity and refinement overhead for existing schemas.
const MAX_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_EXPRESSION_BYTES: usize = 64 * 1024 * 1024;
const MAX_RULE_WORK: usize = 1_000_000;
const MAX_LITERAL_COMPILATIONS: usize = 1024;
const MAX_DFA_GROWTH: usize = 32 * 1024;

/// Peak/cumulative costs charged while constructing or compiling a grammar.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompileUsage {
    pub work: usize,
    pub peak_bytes: usize,
    pub expression_bytes: usize,
    pub rule_work: usize,
    pub literal_compilations: usize,
    /// Largest DFA-state/NFA-state ratio, rounded up in units of 1/1024.
    pub dfa_growth: usize,
    pub schema_depth: usize,
    pub narrowing_depth: usize,
    pub nfa_states: usize,
    pub table_entries: usize,
}

impl CompileUsage {
    /// Smallest limit/cost ratio across the compilation budgets (infinity at zero cost).
    #[must_use]
    pub fn headroom(&self) -> f64 {
        self.costs()
            .iter()
            .map(|(_, used, limit)| {
                if *used == 0 {
                    f64::INFINITY
                } else {
                    *limit as f64 / *used as f64
                }
            })
            .fold(f64::INFINITY, f64::min)
    }

    fn costs(&self) -> [(&'static str, usize, usize); 10] {
        [
            ("work", self.work, MAX_WORK),
            ("peak_bytes", self.peak_bytes, MAX_BYTES),
            (
                "expression_bytes",
                self.expression_bytes,
                MAX_EXPRESSION_BYTES,
            ),
            ("rule_work", self.rule_work, MAX_RULE_WORK),
            (
                "literal_compilations",
                self.literal_compilations,
                MAX_LITERAL_COMPILATIONS,
            ),
            ("dfa_growth", self.dfa_growth, MAX_DFA_GROWTH),
            ("schema_depth", self.schema_depth, 128),
            ("narrowing_depth", self.narrowing_depth, 32),
            (
                "nfa_states",
                self.nfa_states,
                crate::automaton::MAX_NFA_STATES,
            ),
            (
                "table_entries",
                self.table_entries,
                crate::automaton::MAX_TABLE_ENTRIES,
            ),
        ]
    }
}

#[derive(Default)]
struct Metrics {
    peak: Cell<usize>,
    expression_bytes: Cell<usize>,
    rule_work: Cell<usize>,
    literal_compilations: Cell<usize>,
    dfa_growth: Cell<usize>,
    schema_depth: Cell<usize>,
    narrowing_depth: Cell<usize>,
    nfa_states: Cell<usize>,
    table_entries: Cell<usize>,
}

#[derive(Clone, Default)]
pub(crate) struct Budget {
    used: Rc<Cell<(usize, usize)>>,
    metrics: Rc<Metrics>,
    narrowing_depth: usize,
    schema_depth: Rc<Cell<usize>>,
}

impl Budget {
    pub(crate) fn usage(&self) -> CompileUsage {
        CompileUsage {
            work: self.used.get().0,
            peak_bytes: self.metrics.peak.get(),
            expression_bytes: self.metrics.expression_bytes.get(),
            rule_work: self.metrics.rule_work.get(),
            literal_compilations: self.metrics.literal_compilations.get(),
            dfa_growth: self.metrics.dfa_growth.get(),
            schema_depth: self.metrics.schema_depth.get(),
            narrowing_depth: self.metrics.narrowing_depth.get(),
            nfa_states: self.metrics.nfa_states.get(),
            table_entries: self.metrics.table_entries.get(),
        }
    }

    pub(crate) fn literal_compilation(&self) -> Result<(), GrammarError> {
        let used = self.metrics.literal_compilations.get() + 1;
        if used > MAX_LITERAL_COMPILATIONS {
            return Err(GrammarError::too_large(
                "grammar literal compilation budget exceeded (limit: 1024 compilations)",
            ));
        }
        self.metrics.literal_compilations.set(used);
        Ok(())
    }

    pub(crate) fn rule_work(&self, work: usize) -> Result<(), GrammarError> {
        let used = self.metrics.rule_work.get().saturating_add(work);
        if used > MAX_RULE_WORK {
            return Err(GrammarError::too_large(
                "grammar reference-analysis budget exceeded (limit: 1000000 visits)",
            ));
        }
        self.metrics.rule_work.set(used);
        Ok(())
    }

    pub(crate) fn dfa_growth(&self, states: usize, nfa: usize) -> Result<(), GrammarError> {
        let growth = states.saturating_mul(1024).div_ceil(nfa.max(1));
        if growth > MAX_DFA_GROWTH {
            return Err(GrammarError::too_large(
                "grammar determinization growth budget exceeded (limit: 32 DFA states per reachable NFA state)",
            ));
        }
        self.metrics
            .dfa_growth
            .set(self.metrics.dfa_growth.get().max(growth));
        Ok(())
    }

    pub(crate) fn automaton(&self, nfa: usize, entries: usize) {
        self.metrics
            .nfa_states
            .set(self.metrics.nfa_states.get().max(nfa));
        self.metrics
            .table_entries
            .set(self.metrics.table_entries.get().max(entries));
    }

    /// Optional audit output contains costs and a fingerprint, never schema text.
    pub(crate) fn report(&self, phase: &str, value: &impl std::fmt::Debug) {
        use std::fmt::Write as _;
        use std::hash::{DefaultHasher, Hasher};
        use std::io::Write as _;
        let Ok(path) = std::env::var("CHOPINFER_GRAMMAR_BUDGET_REPORT") else {
            return;
        };
        struct Fingerprint(DefaultHasher);
        impl std::fmt::Write for Fingerprint {
            fn write_str(&mut self, text: &str) -> std::fmt::Result {
                self.0.write(text.as_bytes());
                Ok(())
            }
        }
        let mut key = Fingerprint(DefaultHasher::new());
        let _ = write!(&mut key, "{value:?}");
        let usage = self.usage();
        let costs: serde_json::Map<_, _> = usage
            .costs()
            .into_iter()
            .map(|(name, used, limit)| {
                (
                    name.to_owned(),
                    serde_json::json!({"used":used,"limit":limit}),
                )
            })
            .collect();
        let mut row = serde_json::to_vec(&serde_json::json!({"phase":phase,"key":key.0.finish(),"test":std::thread::current().name(),"costs":costs,"headroom":usage.headroom()})).expect("budget audit serializes");
        row.push(b'\n');
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = file.write_all(&row);
        }
    }

    pub(crate) fn enter_schema(&self) -> Result<SchemaDepth, GrammarError> {
        if self.schema_depth.get() == 128 {
            return Err(GrammarError::too_large(
                "schema nesting depth budget exceeded",
            ));
        }
        self.schema_depth.set(self.schema_depth.get() + 1);
        self.metrics
            .schema_depth
            .set(self.metrics.schema_depth.get().max(self.schema_depth.get()));
        Ok(SchemaDepth(Rc::clone(&self.schema_depth)))
    }

    pub(crate) fn nested(&self) -> Result<Self, GrammarError> {
        // Narrowing creates a new schema compiler, so its recursion limit
        // must cross compiler boundaries, including recursive references.
        if self.narrowing_depth == 32 {
            return Err(GrammarError::too_large(
                "enum narrowing depth budget exceeded",
            ));
        }
        self.metrics.narrowing_depth.set(
            self.metrics
                .narrowing_depth
                .get()
                .max(self.narrowing_depth + 1),
        );
        Ok(Self {
            metrics: Rc::clone(&self.metrics),
            used: Rc::clone(&self.used),
            narrowing_depth: self.narrowing_depth + 1,
            schema_depth: Rc::clone(&self.schema_depth),
        })
    }

    pub(crate) fn charge(&self, work: usize, bytes: usize) -> Result<(), GrammarError> {
        let (old_work, old_bytes) = self.used.get();
        let used = (
            old_work.saturating_add(work),
            old_bytes.saturating_add(bytes),
        );
        if used.0 > MAX_WORK || used.1 > MAX_BYTES {
            return Err(GrammarError::too_large(
                "grammar compilation work or memory budget exceeded",
            ));
        }
        self.used.set(used);
        if bytes != 0 {
            self.metrics.peak.set(self.metrics.peak.get().max(used.1));
        }
        Ok(())
    }

    pub(crate) fn scratch(&self, bytes: usize) -> Result<(), GrammarError> {
        let (_, used) = self.used.get();
        self.metrics
            .peak
            .set(self.metrics.peak.get().max(used.saturating_add(bytes)));
        if used.saturating_add(bytes) > MAX_BYTES {
            return Err(GrammarError::too_large(
                "grammar compilation temporary memory budget exceeded",
            ));
        }
        Ok(())
    }

    pub(crate) fn expression_charge(&self, work: usize, bytes: usize) -> Result<(), GrammarError> {
        let used = self.metrics.expression_bytes.get().saturating_add(bytes);
        if used > MAX_EXPRESSION_BYTES {
            return Err(GrammarError::too_large(
                "grammar expression expansion budget exceeded (limit: 67108864 bytes)",
            ));
        }
        self.metrics.expression_bytes.set(used);
        self.charge(work, bytes)
    }

    /// Reserve schema expression copies before arrays duplicate their items.
    pub(crate) fn expr_copies(
        &self,
        expr: &crate::Expr,
        copies: usize,
    ) -> Result<(), GrammarError> {
        use crate::Expr;
        let mut pending = vec![expr];
        while let Some(expr) = pending.pop() {
            self.expression_charge(copies, 128 * copies)?;
            match expr {
                Expr::Empty | Expr::Nothing | Expr::Rule(_) => {}
                Expr::Literal(bytes) => {
                    self.expression_charge(bytes.len() * copies, bytes.len() * 2 * copies)?
                }
                Expr::Class(class) => self.expression_charge(
                    class.ranges().len() * copies,
                    class.ranges().len() * 16 * copies,
                )?,
                Expr::Seq(items) | Expr::Alt(items) => pending.extend(items),
                Expr::Repeat { expr, .. } => pending.push(expr),
                Expr::OrderedSubset {
                    items,
                    tail,
                    separator,
                } => {
                    pending.extend(items.iter().map(|(expr, _)| expr));
                    if let Some(tail) = tail {
                        pending.push(tail);
                    }
                    pending.push(separator);
                }
                Expr::TextExcluding(markers) | Expr::TextUntil { avoid: markers, .. } => {
                    for marker in markers {
                        self.expression_charge(
                            marker.len() * copies,
                            (64 + marker.len() * 2) * copies,
                        )?;
                    }
                    if let Expr::TextUntil { end, .. } = expr {
                        self.expression_charge(end.len() * copies, end.len() * 2 * copies)?;
                    }
                }
            }
        }
        Ok(())
    }

    // Walk before cloning schema fragments or serializing enum literals.
    pub(crate) fn schema(&self, value: &serde_json::Value) -> Result<(), GrammarError> {
        let mut pending = vec![value];
        while let Some(value) = pending.pop() {
            self.charge(1, 128)?;
            match value {
                serde_json::Value::String(text) => {
                    self.charge(text.len(), text.len().saturating_mul(16))?
                }
                serde_json::Value::Array(items) => {
                    if items.len() > MAX_ALTERNATIVES {
                        return Err(GrammarError::too_large("too many grammar alternatives"));
                    }
                    self.charge(items.len(), items.len() * 16)?;
                    pending.extend(items);
                }
                serde_json::Value::Object(fields) => {
                    self.charge(fields.len(), fields.len().saturating_mul(128))?;
                    for (key, value) in fields {
                        self.charge(key.len(), key.len().saturating_mul(16))?;
                        pending.push(value);
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}

/// Counts active schema calls across sibling-narrowing compiler instances.
pub(crate) struct SchemaDepth(Rc<Cell<usize>>);

impl Drop for SchemaDepth {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}
