//! A versioned byte encoding of a [`Grammar`], so a grammar built in one
//! binary (a bundle's frontend) can be compiled in another (the host).
//!
//! The document is little-endian: the 8-byte magic `CHOPGRAM`, a `u32`
//! layout ([`GRAMMAR_WIRE_LAYOUT_V1`]), a `u32` root rule index, a list of
//! rules (one expression each), and a list of special token texts. A list is
//! a `u32` count and its entries; "bytes" is a `u64` length and that many
//! bytes; a flag is one byte, 0 or 1. An expression is a one-byte kind, then:
//!
//! | kind | expression | fields |
//! |---|---|---|
//! | 0 | `Empty` | |
//! | 1 | `Nothing` | |
//! | 2 | `Literal` | bytes |
//! | 3 | `Class` | list of `u32` low, `u32` high (inclusive) |
//! | 4 | `Seq` | list of expressions |
//! | 5 | `Alt` | list of expressions |
//! | 6 | `Repeat` | `u32` min, flag bounded, `u32` max (0 when unbounded), expression |
//! | 7 | `Rule` | `u32` rule index |
//! | 8 | `TextExcluding` | list of bytes |
//! | 9 | `TextUntil` | bytes end, list of bytes avoid |
//! | 10 | `OrderedSubset` | list of {flag required, expression}, flag has tail, the tail expression when present, separator expression |
//!
//! Decoding is closed: it refuses another magic or layout before reading a
//! field, an unknown kind or flag, truncation, trailing bytes, a root or rule
//! reference outside the rules, a class that is not sorted disjoint
//! non-adjacent scalar ranges, a repetition whose max is below its min or an
//! unbounded one that states a max, and a repeated special token text. The
//! depth compilation will recurse to, inlined rules included, is limited to
//! [`GRAMMAR_WIRE_NESTING_MAX_V1`] so the grammar compiles on a 2 MiB thread
//! stack.
//!
//! Decoding charges no construction budget. Its work and memory are linear
//! in the document's length, which the caller already holds: every list
//! count is bounded by the bytes left, each node is read once, and the depth
//! analysis visits each node and distinct reference a bounded number of
//! times. Compilation then applies its usual budgets, as it does to a locally
//! built grammar, so a decoded grammar that is too large fails where and how
//! the same grammar built locally fails: at compilation, with compilation's
//! message. (Charging the construction budget here refused such a grammar
//! earlier, with another message.)
use crate::expr::{CharClass, Expr, Grammar, GrammarError, RuleId};

const MAGIC: &[u8; 8] = b"CHOPGRAM";
/// The layout this build writes and reads.
pub const GRAMMAR_WIRE_LAYOUT_V1: u32 = 1;
/// Deepest compile depth a document may reach (`compile_depth`: the
/// compiler's recursion through expressions and inlined rules, an ordered
/// subset counting twice), which also bounds each expression's nesting.
///
/// The bound is the reach of the local builders, so a decoded grammar
/// admits what local building produces: the deepest grammar the schema,
/// regex and choice builders produce has depth 511 (a 127-level object
/// schema around a pattern of 64 nested starred groups), and it compiles
/// locally.
///
/// Measured as the smallest thread stack that compiles each shape at the
/// bound (nested alternatives, a rule reference chain, nested ordered
/// subsets and that schema): at most about 1365 KiB in a debug build, a 1.5x
/// margin on a 2 MiB stack, the same exposure locally built grammars have,
/// and about 160 KiB optimized, a 12.8x margin.
pub const GRAMMAR_WIRE_NESTING_MAX_V1: usize = 512;

/// Counts the depth analysis's steps, so a test can bound them; it refuses
/// nothing.
#[derive(Default)]
struct WorkV1(std::cell::Cell<usize>);

impl WorkV1 {
    fn add(&self, units: usize) {
        self.0.set(self.0.get().saturating_add(units));
    }
}

const EMPTY: u8 = 0;
const NOTHING: u8 = 1;
const LITERAL: u8 = 2;
const CLASS: u8 = 3;
const SEQ: u8 = 4;
const ALT: u8 = 5;
const REPEAT: u8 = 6;
const RULE: u8 = 7;
const TEXT_EXCLUDING: u8 = 8;
const TEXT_UNTIL: u8 = 9;
const ORDERED_SUBSET: u8 = 10;

impl Grammar {
    /// This grammar as a layout-1 wire document.
    #[must_use]
    pub fn encode_wire(&self) -> Vec<u8> {
        let mut writer = Writer(MAGIC.to_vec());
        writer.u32(GRAMMAR_WIRE_LAYOUT_V1);
        writer.u32(self.root.0);
        writer.count(self.rules.len());
        for rule in &self.rules {
            writer.expr(rule);
        }
        writer.texts(&self.special_tokens);
        writer.0
    }

    /// Decodes a wire document. See the module documentation for what is
    /// refused, and why no construction budget is.
    pub fn decode_wire(bytes: &[u8]) -> Result<Self, GrammarError> {
        let mut reader = Reader { bytes, rules: 0 };
        if reader.take(8)? != MAGIC {
            return Err(GrammarError::invalid("not a grammar wire document"));
        }
        let layout = reader.u32()?;
        if layout != GRAMMAR_WIRE_LAYOUT_V1 {
            return Err(GrammarError::unsupported(format!(
                "grammar wire layout {layout}; this build reads layout {GRAMMAR_WIRE_LAYOUT_V1}"
            )));
        }
        let root = reader.u32()?;
        let count = reader.count()?;
        reader.rules = count;
        let mut rules = Vec::new();
        for _ in 0..count {
            rules.push(reader.expr()?);
        }
        let special_tokens = reader.texts()?;
        if !reader.bytes.is_empty() {
            return Err(GrammarError::invalid("grammar wire: trailing bytes"));
        }
        if root as usize >= count {
            return Err(GrammarError::invalid(
                "grammar wire: the root is outside the rules",
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(special_tokens.len());
        if !special_tokens
            .iter()
            .all(|text| seen.insert(text.as_slice()))
        {
            return Err(GrammarError::invalid(
                "grammar wire: a special token text repeats",
            ));
        }
        rules.shrink_to_fit();
        let grammar = Self {
            rules,
            root: RuleId(root),
            special_tokens,
        };
        let depth = compile_depth(&grammar, &WorkV1::default());
        if depth > GRAMMAR_WIRE_NESTING_MAX_V1 {
            return Err(GrammarError::too_large(format!(
                "grammar wire compiles {depth} levels deep, past {GRAMMAR_WIRE_NESTING_MAX_V1}"
            )));
        }
        Ok(grammar)
    }

    /// How deep compiling this grammar recurses, as decoding measures it: a
    /// wire document deeper than [`GRAMMAR_WIRE_NESTING_MAX_V1`] is refused.
    #[must_use]
    pub fn compile_depth(&self) -> usize {
        compile_depth(self, &WorkV1::default())
    }
}

/// How deep compiling `grammar` recurses, in sequence levels: the deepest
/// path through each separately compiled rule (the root and every recursive
/// rule), where a reference to any other rule inlines that rule's
/// expression and an ordered subset counts twice, since it builds each item
/// inside a sequence.
///
/// The analysis is linear in the grammar's size, as `work` counts: each
/// expression node and each distinct reference is visited a bounded number
/// of times, and each rule is resolved once.
fn compile_depth(grammar: &Grammar, work: &WorkV1) -> usize {
    let profiles = grammar
        .rules
        .iter()
        .map(|rule| RuleProfile::of(rule, work))
        .collect::<Vec<_>>();
    let recursive = recursive_rules(&profiles, work);
    let called = |id: u32| id == grammar.root.0 || recursive[id as usize];
    // Inlined rules form an acyclic graph: a cycle would make them
    // recursive. Resolve each once, in post order.
    let mut inlined: Vec<Option<usize>> = vec![None; profiles.len()];
    let inline = |inlined: &[Option<usize>], reference: u32| {
        if called(reference) {
            Some(0)
        } else {
            inlined[reference as usize]
        }
    };
    let mut deepest = 0;
    for unit in (0..profiles.len() as u32).filter(|&id| called(id)) {
        let mut pending = vec![(unit, false)];
        while let Some((id, expanded)) = pending.pop() {
            work.add(1);
            if id != unit && inlined[id as usize].is_some() {
                continue;
            }
            let profile = &profiles[id as usize];
            work.add(profile.references.len());
            if !expanded {
                pending.push((id, true));
                for &(reference, _) in &profile.references {
                    if inline(&inlined, reference).is_none() {
                        pending.push((reference, false));
                    }
                }
                continue;
            }
            let depth = profile
                .references
                .iter()
                .map(|&(reference, prefix)| prefix + inline(&inlined, reference).unwrap_or(0))
                .fold(profile.depth, usize::max);
            if id == unit {
                deepest = deepest.max(depth);
            } else {
                inlined[id as usize] = Some(depth);
            }
        }
    }
    deepest
}

/// One rule's own compile depth and, for each rule it refers to, the
/// deepest depth at which it does, counting the reference itself.
struct RuleProfile {
    depth: usize,
    references: Vec<(u32, usize)>,
}

impl RuleProfile {
    fn of(expr: &Expr, work: &WorkV1) -> Self {
        let mut depth_of_reference = std::collections::HashMap::new();
        let mut own = 0;
        let mut pending = vec![(expr, 0_usize)];
        while let Some((expr, above)) = pending.pop() {
            work.add(1);
            let depth = above
                + if matches!(expr, Expr::OrderedSubset { .. }) {
                    2
                } else {
                    1
                };
            own = own.max(depth);
            match expr {
                Expr::Seq(items) | Expr::Alt(items) => {
                    pending.extend(items.iter().map(|item| (item, depth)));
                }
                Expr::Repeat { expr, .. } => pending.push((expr, depth)),
                Expr::OrderedSubset {
                    items,
                    tail,
                    separator,
                } => {
                    pending.extend(items.iter().map(|(item, _)| (item, depth)));
                    pending.extend(tail.as_deref().map(|tail| (tail, depth)));
                    pending.push((separator, depth));
                }
                Expr::Rule(id) => match depth_of_reference.entry(id.0) {
                    std::collections::hash_map::Entry::Occupied(mut deepest) => {
                        *deepest.get_mut() = depth.max(*deepest.get());
                    }
                    std::collections::hash_map::Entry::Vacant(entry) => {
                        entry.insert(depth);
                    }
                },
                Expr::Empty
                | Expr::Nothing
                | Expr::Literal(_)
                | Expr::Class(_)
                | Expr::TextExcluding(_)
                | Expr::TextUntil { .. } => {}
            }
        }
        Self {
            depth: own,
            references: depth_of_reference.into_iter().collect(),
        }
    }
}

/// Which rules lie on a reference cycle, by Tarjan's strongly connected
/// components, without recursion.
fn recursive_rules(profiles: &[RuleProfile], work: &WorkV1) -> Vec<bool> {
    const UNVISITED: usize = usize::MAX;
    let count = profiles.len();
    let mut index = vec![UNVISITED; count];
    let mut low = vec![0; count];
    let mut on_stack = vec![false; count];
    let mut stack = Vec::new();
    let mut recursive = vec![false; count];
    let mut next = 0;
    for start in 0..count {
        if index[start] != UNVISITED {
            continue;
        }
        // (rule, the next of its references to visit)
        let mut walk = vec![(start, 0_usize)];
        index[start] = next;
        low[start] = next;
        next += 1;
        stack.push(start);
        on_stack[start] = true;
        while let Some(&mut (rule, ref mut position)) = walk.last_mut() {
            work.add(1);
            if let Some(&(reference, _)) = profiles[rule].references.get(*position) {
                *position += 1;
                let reference = reference as usize;
                if reference == rule {
                    recursive[rule] = true;
                }
                if index[reference] == UNVISITED {
                    index[reference] = next;
                    low[reference] = next;
                    next += 1;
                    stack.push(reference);
                    on_stack[reference] = true;
                    walk.push((reference, 0));
                } else if on_stack[reference] {
                    low[rule] = low[rule].min(index[reference]);
                }
                continue;
            }
            walk.pop();
            if let Some(&(parent, _)) = walk.last() {
                low[parent] = low[parent].min(low[rule]);
            }
            if low[rule] == index[rule] {
                let mut component = Vec::new();
                loop {
                    let member = stack.pop().expect("the component is on the stack");
                    on_stack[member] = false;
                    component.push(member);
                    if member == rule {
                        break;
                    }
                }
                if component.len() > 1 {
                    for member in component {
                        recursive[member] = true;
                    }
                }
            }
        }
    }
    recursive
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn count(&mut self, count: usize) {
        self.u32(u32::try_from(count).expect("grammar list length fits u32"));
    }

    fn flag(&mut self, flag: bool) {
        self.0.push(u8::from(flag));
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.0
            .extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        self.0.extend_from_slice(bytes);
    }

    fn texts(&mut self, texts: &[Vec<u8>]) {
        self.count(texts.len());
        for text in texts {
            self.bytes(text);
        }
    }

    fn exprs(&mut self, items: &[Expr]) {
        self.count(items.len());
        for item in items {
            self.expr(item);
        }
    }

    fn expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Empty => self.0.push(EMPTY),
            Expr::Nothing => self.0.push(NOTHING),
            Expr::Literal(bytes) => {
                self.0.push(LITERAL);
                self.bytes(bytes);
            }
            Expr::Class(class) => {
                self.0.push(CLASS);
                self.count(class.ranges().len());
                for &(lo, hi) in class.ranges() {
                    self.u32(lo);
                    self.u32(hi);
                }
            }
            Expr::Seq(items) => {
                self.0.push(SEQ);
                self.exprs(items);
            }
            Expr::Alt(items) => {
                self.0.push(ALT);
                self.exprs(items);
            }
            Expr::Repeat { expr, min, max } => {
                self.0.push(REPEAT);
                self.u32(*min);
                self.flag(max.is_some());
                self.u32(max.unwrap_or(0));
                self.expr(expr);
            }
            Expr::Rule(id) => {
                self.0.push(RULE);
                self.u32(id.0);
            }
            Expr::TextExcluding(markers) => {
                self.0.push(TEXT_EXCLUDING);
                self.texts(markers);
            }
            Expr::TextUntil { end, avoid } => {
                self.0.push(TEXT_UNTIL);
                self.bytes(end);
                self.texts(avoid);
            }
            Expr::OrderedSubset {
                items,
                tail,
                separator,
            } => {
                self.0.push(ORDERED_SUBSET);
                self.count(items.len());
                for (item, required) in items {
                    self.flag(*required);
                    self.expr(item);
                }
                self.flag(tail.is_some());
                if let Some(tail) = tail {
                    self.expr(tail);
                }
                self.expr(separator);
            }
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    /// The rule count, which every reference must stay below.
    rules: usize,
}

fn truncated() -> GrammarError {
    GrammarError::invalid("grammar wire: document ends early")
}

impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8], GrammarError> {
        if count > self.bytes.len() {
            return Err(truncated());
        }
        let (head, tail) = self.bytes.split_at(count);
        self.bytes = tail;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, GrammarError> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, GrammarError> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    /// A list count. Every entry takes at least one byte, so a count past
    /// the remaining bytes is truncation.
    fn count(&mut self) -> Result<usize, GrammarError> {
        let count = self.u32()? as usize;
        if count > self.bytes.len() {
            return Err(truncated());
        }
        Ok(count)
    }

    fn flag(&mut self) -> Result<bool, GrammarError> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(GrammarError::invalid("grammar wire: a flag is not 0 or 1")),
        }
    }

    fn bytes(&mut self) -> Result<&'a [u8], GrammarError> {
        let length = u64::from_le_bytes(self.take(8)?.try_into().expect("8 bytes"));
        let length = usize::try_from(length).map_err(|_| truncated())?;
        self.take(length)
    }

    /// A text a grammar keeps.
    fn text(&mut self) -> Result<Vec<u8>, GrammarError> {
        Ok(self.bytes()?.to_vec())
    }

    fn texts(&mut self) -> Result<Vec<Vec<u8>>, GrammarError> {
        let count = self.count()?;
        let mut texts = Vec::new();
        for _ in 0..count {
            texts.push(self.text()?);
        }
        texts.shrink_to_fit();
        Ok(texts)
    }

    /// One expression, decoded with an explicit stack so a deep document
    /// cannot exhaust the thread's stack. Each node is read once and
    /// allocates no more than its own bytes and a node.
    fn expr(&mut self) -> Result<Expr, GrammarError> {
        let mut open: Vec<Open> = Vec::new();
        loop {
            if let Some(Open::Subset {
                phase: Phase::Items,
                required,
                ..
            }) = open.last_mut()
            {
                *required = self.flag()?;
            }
            if open.len() == GRAMMAR_WIRE_NESTING_MAX_V1 {
                return Err(GrammarError::too_large(format!(
                    "grammar wire nests past {GRAMMAR_WIRE_NESTING_MAX_V1} expressions"
                )));
            }
            let mut value = match self.u8()? {
                EMPTY => Expr::Empty,
                NOTHING => Expr::Nothing,
                LITERAL => Expr::Literal(self.bytes()?.to_vec()),
                CLASS => self.class()?,
                kind @ (SEQ | ALT) => {
                    let remaining = self.count()?;
                    let items = Vec::new();
                    if remaining == 0 {
                        list_expr(kind, items)
                    } else {
                        open.push(Open::List {
                            kind,
                            items,
                            remaining,
                        });
                        continue;
                    }
                }
                REPEAT => {
                    let min = self.u32()?;
                    let bounded = self.flag()?;
                    let max = self.u32()?;
                    let max = match (bounded, max) {
                        (true, max) if max < min => {
                            return Err(GrammarError::invalid(
                                "grammar wire: a repetition has max below min",
                            ));
                        }
                        (true, max) => Some(max),
                        (false, 0) => None,
                        (false, _) => {
                            return Err(GrammarError::invalid(
                                "grammar wire: an unbounded repetition states a max",
                            ));
                        }
                    };
                    open.push(Open::Repeat { min, max });
                    continue;
                }
                RULE => {
                    let id = self.u32()?;
                    if id as usize >= self.rules {
                        return Err(GrammarError::invalid(
                            "grammar wire: a rule reference is outside the rules",
                        ));
                    }
                    Expr::Rule(RuleId(id))
                }
                TEXT_EXCLUDING => Expr::TextExcluding(self.texts()?),
                TEXT_UNTIL => {
                    let end = self.bytes()?;
                    Expr::TextUntil {
                        end: end.to_vec(),
                        avoid: self.texts()?,
                    }
                }
                ORDERED_SUBSET => {
                    let remaining = self.count()?;
                    let phase = if remaining == 0 {
                        self.after_items()?
                    } else {
                        Phase::Items
                    };
                    open.push(Open::Subset {
                        items: Vec::new(),
                        remaining,
                        required: false,
                        phase,
                        tail: None,
                    });
                    continue;
                }
                kind => {
                    return Err(GrammarError::invalid(format!(
                        "grammar wire: unknown expression kind {kind}"
                    )));
                }
            };
            // Hand the finished value to its parent, closing every parent it
            // finishes in turn.
            loop {
                let Some(parent) = open.last_mut() else {
                    return Ok(value);
                };
                match parent {
                    Open::List {
                        items, remaining, ..
                    } => {
                        items.push(value);
                        *remaining -= 1;
                        if *remaining > 0 {
                            break;
                        }
                        let Some(Open::List {
                            kind, mut items, ..
                        }) = open.pop()
                        else {
                            unreachable!("the parent is a list");
                        };
                        items.shrink_to_fit();
                        value = list_expr(kind, items);
                    }
                    &mut Open::Repeat { min, max } => {
                        open.pop();
                        value = Expr::Repeat {
                            expr: Box::new(value),
                            min,
                            max,
                        };
                    }
                    Open::Subset {
                        items,
                        remaining,
                        required,
                        phase,
                        tail,
                    } => match phase {
                        Phase::Items => {
                            items.push((value, *required));
                            *remaining -= 1;
                            if *remaining == 0 {
                                *phase = self.after_items()?;
                            }
                            break;
                        }
                        Phase::Tail => {
                            *tail = Some(Box::new(value));
                            *phase = Phase::Separator;
                            break;
                        }
                        Phase::Separator => {
                            let Some(Open::Subset {
                                mut items, tail, ..
                            }) = open.pop()
                            else {
                                unreachable!("the parent is an ordered subset");
                            };
                            items.shrink_to_fit();
                            value = Expr::OrderedSubset {
                                items,
                                tail,
                                separator: Box::new(value),
                            };
                        }
                    },
                }
            }
        }
    }

    /// What follows an ordered subset's items: its tail when the flag says
    /// it has one, then its separator.
    fn after_items(&mut self) -> Result<Phase, GrammarError> {
        Ok(if self.flag()? {
            Phase::Tail
        } else {
            Phase::Separator
        })
    }

    fn class(&mut self) -> Result<Expr, GrammarError> {
        let count = self.u32()? as usize;
        if count > self.bytes.len() / 8 {
            return Err(truncated());
        }
        let mut ranges = Vec::with_capacity(count);
        for _ in 0..count {
            ranges.push((self.u32()?, self.u32()?));
        }
        // The canonical form drops or clips ranges past MAX_SCALAR and
        // merges unsorted, overlapping or adjacent ones, so any of those
        // differs from it.
        let class = CharClass::from_ranges(ranges.iter().copied());
        if class.ranges() != ranges {
            return Err(GrammarError::invalid(
                "grammar wire: a class is not sorted disjoint scalar ranges",
            ));
        }
        Ok(Expr::Class(class))
    }
}

/// An expression whose children are still being decoded.
enum Open {
    List {
        kind: u8,
        items: Vec<Expr>,
        remaining: usize,
    },
    Repeat {
        min: u32,
        max: Option<u32>,
    },
    Subset {
        items: Vec<(Expr, bool)>,
        remaining: usize,
        /// The flag read before the item being decoded.
        required: bool,
        phase: Phase,
        tail: Option<Box<Expr>>,
    },
}

/// Which child of an ordered subset is being decoded.
#[derive(Clone, Copy)]
enum Phase {
    Items,
    Tail,
    Separator,
}

fn list_expr(kind: u8, items: Vec<Expr>) -> Expr {
    if kind == SEQ {
        Expr::Seq(items)
    } else {
        Expr::Alt(items)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrammarBuilder;

    /// Root refers to a middle rule `width` times and the middle rule to
    /// an empty leaf `width` times: the analysis works linearly in the
    /// grammar's size, never per repeated reference.
    #[test]
    fn compile_depth_works_linearly_in_the_references() {
        let width = 32_000;
        let mut builder = GrammarBuilder::new();
        let root = builder.declare();
        let middle = builder.declare();
        let leaf = builder.rule(Expr::Empty);
        builder.define(root, Expr::Seq(vec![Expr::Rule(middle); width]));
        builder.define(middle, Expr::Seq(vec![Expr::Rule(leaf); width]));
        let grammar = builder.finish(root).unwrap();
        let counted = WorkV1::default();
        // Sequence, reference, sequence, reference, empty.
        assert_eq!(compile_depth(&grammar, &counted), 5);
        let nodes = 2 * width + 3;
        let work = counted.0.get();
        assert!(work <= 8 * nodes, "{work} units for {nodes} nodes");
    }
}
