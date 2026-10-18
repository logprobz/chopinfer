//! Completion parsing for the declared XML tool-call grammars.

use std::ops::Range;
use std::sync::Arc;

use serde_json::{Map, Value};

use crate::descriptor::{
    FormatDescriptorV1, FrontendJsonEncodingV1, FrontendParserDeclarationV1, FrontendToolGrammarV1,
};
use crate::format_decode::completion_calls_v1;
use crate::request::{FrontendJsonV1, FrontendToolV1};
use crate::toolkit::{json_to_string_v1, resource_v1, validate_name_v1};

/// The strings a completion is parsed by: the parser declaration's
/// delimiters without the whitespace the template writes around them.
///
/// Rendering writes the declared strings exactly, but the reference parser
/// matches only the bare wrappers and skips what lies between them, so a call
/// parses whether or not the model reproduced the template's newlines.
#[derive(Clone, Copy)]
pub(crate) struct ChatMlTerminalsV1<'a> {
    pub(crate) reasoning_open: &'a str,
    pub(crate) reasoning_close: &'a str,
    pub(crate) tool_open: &'a str,
    pub(crate) tool_close: &'a str,
    pub(crate) function_open: &'a str,
    function_name_close: &'a str,
    function_close: &'a str,
    parameter_open: &'a str,
    parameter_name_close: &'a str,
    /// The typed grammar's close of a JSON value's name; empty otherwise.
    parameter_json_name_close: &'a str,
    parameter_value_open: &'a str,
    parameter_close: &'a str,
    /// The whitespace the declared parameter name close writes after its
    /// bare marker, which precedes a function parameter's value.
    value_lead: &'a str,
    /// The whitespace the declared parameter close writes before its bare
    /// marker, which follows a function parameter's value.
    value_trail: &'a str,
    grammar: FrontendToolGrammarV1,
    /// How parsed arguments are written.
    json_encoding: FrontendJsonEncodingV1,
}

impl<'a> ChatMlTerminalsV1<'a> {
    pub(crate) fn new(parser: &'a FrontendParserDeclarationV1) -> Self {
        Self {
            reasoning_open: parser.reasoning_open.trim(),
            reasoning_close: parser.reasoning_close.trim(),
            tool_open: call_open_v1(parser).0,
            tool_close: parser.tool_call_close.trim(),
            function_open: parser.function_open.trim(),
            function_name_close: parser.function_name_close.trim(),
            function_close: parser.function_close.trim(),
            parameter_open: parser.parameter_open.trim(),
            parameter_name_close: parser.parameter_name_close.trim(),
            parameter_json_name_close: parser.parameter_json_name_close.trim(),
            parameter_value_open: parser.parameter_value_open.trim(),
            parameter_close: parser.parameter_close.trim(),
            value_lead: &parser.parameter_name_close
                [parser.parameter_name_close.trim_end().len()..],
            value_trail: &parser.parameter_close
                [..parser.parameter_close.len() - parser.parameter_close.trim_start().len()],
            grammar: parser.tool_grammar,
            json_encoding: parser.json_encoding,
        }
    }

    /// The length of the longest marker a tool region is read by.
    fn longest(&self) -> usize {
        [
            self.tool_open,
            self.tool_close,
            self.function_open,
            self.function_name_close,
            self.function_close,
            self.parameter_open,
            self.parameter_name_close,
            self.parameter_json_name_close,
            self.parameter_close,
        ]
        .iter()
        .map(|marker| marker.len())
        .max()
        .unwrap_or_default()
        .max(1)
    }
}

/// The call open as a completion is read by, and the whitespace the template
/// writes after it: the declared marker trimmed, except that the typed
/// grammar keeps the whitespace written before it.
pub(crate) fn call_open_v1(parser: &FrontendParserDeclarationV1) -> (&str, &str) {
    let marker = parser.tool_call_open.as_str();
    let read = if parser.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
        marker.trim_end()
    } else {
        marker.trim()
    };
    let end = marker.find(read).unwrap_or_default() + read.len();
    (read, &marker[end..])
}

/// One call a tool region named.
pub(crate) struct RegionCallV1 {
    pub(crate) name: String,
    /// Where the raw text between the name and the function close lies in
    /// the region's text.
    arguments: Range<usize>,
    /// Whether the function close arrived.
    pub(crate) closed: bool,
}

/// What one tool region holds so far.
#[derive(Default)]
pub(crate) struct ToolRegionV1 {
    /// Every call whose name is complete, in order.
    pub(crate) calls: Vec<RegionCallV1>,
    /// The bytes the region used, once the wrapper that returns to text arrived.
    pub(crate) end: Option<usize>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum RegionStateV1 {
    Preamble,
    Name,
    Arguments,
    Between,
}

/// Where a region scan stands.
#[derive(Clone, Copy)]
struct RegionCursorV1 {
    state: RegionStateV1,
    /// Where the text of this state began.
    start: usize,
    /// Where this state's search resumes: nothing it seeks begins between
    /// `start` and here.
    from: usize,
    /// How many calls the region had named.
    calls: usize,
    /// The open call's parameter walk.
    walk: ParameterWalkV1,
}

/// Which keys of one call hold JSON values: the call's index, its name and
/// [`json_arguments_v1`].
type JsonRuleV1 = (usize, String, Box<dyn Fn(&str) -> bool + Send>);

/// One tool region, scanned as its text grows.
///
/// A scan of a longer text that begins with the one scanned before resumes
/// where every earlier decision is final: before the last bytes a marker could
/// still begin in. Each scan reads the same as [`scan_tool_region`] over the
/// whole text, and costs the new bytes plus the length of the longest marker.
#[derive(Default)]
pub(crate) struct RegionScanV1 {
    region: ToolRegionV1,
    /// The cursor that every longer text shares, once the region's opener
    /// is whole.
    settled: Option<RegionCursorV1>,
    /// The JSON-value rule of the call at this index and with this name.
    json: Option<JsonRuleV1>,
}

/// Walks one tool region the way the reference parser's state machine does.
///
/// `text` starts at the tool or function opener that began the region. Text
/// around the wrappers is ignored, a function may open without its tool
/// wrapper, and the arguments run to the function close whatever they hold,
/// except a function close inside a JSON value's string literal (see
/// [`ParameterWalkV1`]).
pub(crate) fn scan_tool_region(
    terminals: ChatMlTerminalsV1<'_>,
    text: &str,
    tools: &[FrontendToolV1],
) -> ToolRegionV1 {
    let mut scan = RegionScanV1::default();
    scan.scan(terminals, text, tools);
    scan.region
}

impl RegionScanV1 {
    /// Scans `text`, which begins with the text this scan read before.
    pub(crate) fn scan(
        &mut self,
        terminals: ChatMlTerminalsV1<'_>,
        text: &str,
        tools: &[FrontendToolV1],
    ) -> &ToolRegionV1 {
        if terminals.grammar == FrontendToolGrammarV1::XmlKeyValueArgumentsV1 {
            self.region = scan_key_value_call(terminals, text, tools);
            return &self.region;
        }
        let opener = terminals.tool_open.len().max(terminals.function_open.len());
        let mut cursor = self.settled.unwrap_or_else(|| {
            let (state, start) = if text.starts_with(terminals.tool_open) {
                (RegionStateV1::Preamble, terminals.tool_open.len())
            } else {
                (RegionStateV1::Name, terminals.function_open.len())
            };
            RegionCursorV1 {
                state,
                start,
                from: start,
                calls: 0,
                walk: ParameterWalkV1::default(),
            }
        });
        self.region.calls.truncate(cursor.calls);
        self.region.end = None;
        if cursor.state == RegionStateV1::Arguments {
            let call = self
                .region
                .calls
                .last_mut()
                .expect("arguments follow a name");
            call.closed = false;
        }
        // Every marker that begins before here fits in `text`, so what the
        // scan decides before here no longer depends on what follows.
        let settles =
            floor_char_boundary_v1(text, text.len().saturating_sub(terminals.longest() - 1));
        let mut settling = text.len() >= opener;
        // Keeps the cursor before the earliest sought marker, `found`, or
        // where the search reached the bytes that are not yet final.
        let mut settle =
            |settled: &mut Option<RegionCursorV1>, cursor: RegionCursorV1, found: Option<usize>| {
                if settling {
                    let from = cursor.from.max(settles).min(found.unwrap_or(usize::MAX));
                    *settled = Some(RegionCursorV1 { from, ..cursor });
                    settling = found.is_some_and(|at| at < settles);
                }
            };
        loop {
            let rest = text.get(cursor.from..).unwrap_or_default();
            match cursor.state {
                RegionStateV1::Preamble | RegionStateV1::Between => {
                    let mut wrappers = vec![
                        (terminals.tool_close, RegionStateV1::Preamble),
                        (terminals.function_open, RegionStateV1::Name),
                    ];
                    if cursor.state == RegionStateV1::Between {
                        wrappers.push((terminals.tool_open, RegionStateV1::Between));
                    }
                    let Some((at, length, next)) = earliest_v1(rest, &wrappers) else {
                        settle(&mut self.settled, cursor, None);
                        return &self.region;
                    };
                    let at = cursor.from + at;
                    settle(&mut self.settled, cursor, Some(at));
                    let position = at + length;
                    match next {
                        // The tool close returns to text.
                        RegionStateV1::Preamble => {
                            self.region.end = Some(position);
                            return &self.region;
                        }
                        RegionStateV1::Between => cursor.state = RegionStateV1::Preamble,
                        _ => cursor.state = next,
                    }
                    (cursor.start, cursor.from) = (position, position);
                }
                RegionStateV1::Name => {
                    let closes = [
                        (terminals.function_name_close, RegionStateV1::Arguments),
                        (terminals.function_close, RegionStateV1::Between),
                    ];
                    let Some((at, length, next)) = earliest_v1(rest, &closes) else {
                        settle(&mut self.settled, cursor, None);
                        return &self.region;
                    };
                    let at = cursor.from + at;
                    settle(&mut self.settled, cursor, Some(at));
                    let position = at + length;
                    self.region.calls.push(RegionCallV1 {
                        name: text[cursor.start..at].trim().to_owned(),
                        arguments: position..position,
                        closed: next == RegionStateV1::Between,
                    });
                    cursor = RegionCursorV1 {
                        state: next,
                        start: position,
                        from: position,
                        calls: self.region.calls.len(),
                        walk: ParameterWalkV1::default(),
                    };
                }
                RegionStateV1::Arguments => {
                    let index = self.region.calls.len() - 1;
                    let name = &self.region.calls[index].name;
                    if self
                        .json
                        .as_ref()
                        .is_none_or(|(call, called, _)| (*call, called) != (index, name))
                    {
                        let json = json_arguments_v1(tools, name);
                        self.json = Some((index, name.clone(), Box::new(json)));
                    }
                    let json = &*self.json.as_ref().expect("the open call's rule").2;
                    let arguments = &text[cursor.start..];
                    let close = loop {
                        let from = cursor.from - cursor.start;
                        let Some(at) = arguments[from..].find(terminals.function_close) else {
                            break None;
                        };
                        let at = from + at;
                        settle(&mut self.settled, cursor, Some(cursor.start + at));
                        // A close inside a JSON value's string literal is
                        // content; a raw value ends at the first close.
                        if !cursor
                            .walk
                            .walk(terminals, &arguments[..at], json, &mut |_, _, _| ())
                        {
                            break Some(at);
                        }
                        cursor.from = cursor.start + at + utf8_width_v1(&arguments[at..]);
                    };
                    let call = self
                        .region
                        .calls
                        .last_mut()
                        .expect("arguments follow a name");
                    let Some(at) = close else {
                        call.arguments = cursor.start..text.len();
                        settle(&mut self.settled, cursor, None);
                        return &self.region;
                    };
                    call.arguments = cursor.start..cursor.start + at;
                    call.closed = true;
                    let position = cursor.start + at + terminals.function_close.len();
                    cursor.state = RegionStateV1::Between;
                    (cursor.start, cursor.from) = (position, position);
                }
            }
        }
    }

    /// The region as the last scan read it.
    pub(crate) const fn region(&self) -> &ToolRegionV1 {
        &self.region
    }

    /// How many of the region's first calls are closed and stay as they are
    /// whatever text follows.
    pub(crate) fn settled_calls(&self) -> usize {
        self.settled.map_or(0, |cursor| {
            cursor.calls - usize::from(cursor.state == RegionStateV1::Arguments)
        })
    }
}

/// The largest character boundary of `text` at or before `index`.
fn floor_char_boundary_v1(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// The width of the first character of `text`, or 1 when it is empty.
fn utf8_width_v1(text: &str) -> usize {
    text.chars().next().map_or(1, char::len_utf8)
}

/// Walks one key-value call, which is a whole tool region.
///
/// `text` starts at the call open. The name runs to the first argument or the
/// call close, and the arguments run to the call close that
/// [`walk_key_value_arguments_v1`] finds.
fn scan_key_value_call(
    terminals: ChatMlTerminalsV1<'_>,
    text: &str,
    tools: &[FrontendToolV1],
) -> ToolRegionV1 {
    let open = terminals.tool_open.len();
    let rest = text.get(open..).unwrap_or_default();
    let closes = [
        (terminals.parameter_open, false),
        (terminals.tool_close, true),
    ];
    let Some((at, length, closed)) = earliest_v1(rest, &closes) else {
        return ToolRegionV1 {
            calls: Vec::new(),
            end: None,
        };
    };
    let name = rest[..at].trim().to_owned();
    if closed {
        return ToolRegionV1 {
            calls: vec![RegionCallV1 {
                name,
                arguments: open + at..open + at,
                closed: true,
            }],
            end: Some(open + at + length),
        };
    }
    let arguments = &rest[at..];
    let json = json_arguments_v1(tools, &name);
    let (_, end) = walk_key_value_arguments_v1(terminals, arguments, &json);
    let kept = end.map_or(arguments.len(), |end| end - terminals.tool_close.len());
    ToolRegionV1 {
        calls: vec![RegionCallV1 {
            name,
            arguments: open + at..open + at + kept,
            closed: end.is_some(),
        }],
        end: end.map(|end| open + at + end),
    }
}

/// Walks a key-value call's arguments, from its first argument open.
///
/// Returns every complete key and value, and the end of the call close once
/// it arrived. Each argument is a key element, optional whitespace, and a
/// value element; an argument without a value element is skipped. A raw
/// value ends at the first value close, and at the first call close, which
/// ends the call and drops the value. A JSON value ends the same way, except
/// that a marker inside one of its string literals is content: an
/// unterminated string keeps the value, and the call, open.
fn walk_key_value_arguments_v1<'a>(
    terminals: ChatMlTerminalsV1<'_>,
    text: &'a str,
    json: &dyn Fn(&str) -> bool,
) -> (Vec<(&'a str, &'a str)>, Option<usize>) {
    let mut pairs = Vec::new();
    let mut position = 0;
    loop {
        let opens = [
            (terminals.parameter_open, false),
            (terminals.tool_close, true),
        ];
        let Some((at, length, closes)) = earliest_v1(&text[position..], &opens) else {
            return (pairs, None);
        };
        if closes {
            return (pairs, Some(position + at + length));
        }
        let key_start = position + at + terminals.parameter_open.len();
        let key_closes = [
            (terminals.parameter_name_close, false),
            (terminals.tool_close, true),
        ];
        let Some((at, length, closes)) = earliest_v1(&text[key_start..], &key_closes) else {
            return (pairs, None);
        };
        if closes {
            return (pairs, Some(key_start + at + length));
        }
        let key = text[key_start..key_start + at].trim();
        let after_key = key_start + at + length;
        let Some(value_start) = text[after_key..]
            .trim_start()
            .strip_prefix(terminals.parameter_value_open)
            .map(|value| text.len() - value.len())
        else {
            position = after_key;
            continue;
        };
        let value = &text[value_start..];
        let value_closes = [
            (terminals.parameter_close, false),
            (terminals.tool_close, true),
        ];
        let found = if json(key) {
            earliest_outside_strings_v1(value, &value_closes)
        } else {
            earliest_v1(value, &value_closes)
        };
        match found {
            None => return (pairs, None),
            Some((at, length, true)) => return (pairs, Some(value_start + at + length)),
            Some((at, length, false)) => {
                pairs.push((key, &value[..at]));
                position = value_start + at + length;
            }
        }
    }
}

/// The schema of one parameter of a call, as the tool's parameters resolve
/// its name. The decoder's value scan, its coercion and the tool-choice
/// grammar all resolve a parameter by [`parameter_schema_v1`].
#[derive(Clone, Copy)]
pub(crate) enum ParameterSchemaV1<'a> {
    /// The value has this schema.
    Typed(&'a Value),
    /// No schema types the value: it is free text, read as a string.
    Free,
    /// The parameters admit no parameter of this name.
    Refused,
}

/// The schema of the parameter `name` under a tool's `parameters`: its
/// declared property, else the additional-property schema
/// ([`additional_schema_v1`]).
pub(crate) fn parameter_schema_v1<'a>(parameters: &'a Value, name: &str) -> ParameterSchemaV1<'a> {
    let Value::Object(fields) = parameters else {
        return ParameterSchemaV1::Free;
    };
    match fields
        .get("properties")
        .and_then(|properties| properties.get(name))
    {
        Some(schema) => ParameterSchemaV1::Typed(schema),
        None => additional_schema_v1(fields),
    }
}

/// The schema of a parameter that `fields`, a parameters schema object, does
/// not declare as a property: free text when pattern properties may type it,
/// none when `additionalProperties` is false, the `additionalProperties`
/// schema when it is one, and otherwise free text.
pub(crate) fn additional_schema_v1(fields: &Map<String, Value>) -> ParameterSchemaV1<'_> {
    if fields.contains_key("patternProperties") {
        return ParameterSchemaV1::Free;
    }
    match fields.get("additionalProperties") {
        Some(Value::Bool(false)) => ParameterSchemaV1::Refused,
        Some(schema @ Value::Object(_)) => ParameterSchemaV1::Typed(schema),
        _ => ParameterSchemaV1::Free,
    }
}

/// The parameters schema of the offered tool `name`, whatever its `strict`.
fn tool_parameters_v1(tools: &[FrontendToolV1], name: &str) -> Option<Value> {
    tools
        .iter()
        .find(|tool| tool.name == name)
        .map(|tool| tool.parameters.to_json_value())
}

/// Which keys of a call to `name` hold JSON values: exactly those the
/// tool-choice grammar writes as JSON. The tool constrains its arguments
/// (`strict` is not false and it declares parameters) and the key's schema
/// ([`parameter_schema_v1`]) does not admit a string. Every other value,
/// including any value of an unknown function, is raw text.
fn json_arguments_v1(tools: &[FrontendToolV1], name: &str) -> impl Fn(&str) -> bool + use<> {
    let parameters = tools
        .iter()
        .find(|tool| tool.name == name)
        .filter(|tool| tool.strict != Some(false) && tool.parameters_present)
        .map(|tool| tool.parameters.to_json_value());
    move |key: &str| {
        parameters.as_ref().is_some_and(|parameters| {
            matches!(
                parameter_schema_v1(parameters, key),
                ParameterSchemaV1::Typed(schema) if schema.is_object() && !admits_string_v1(schema)
            )
        })
    }
}

/// Whether a value of this schema could be a string, so its text is raw.
pub(crate) fn admits_string_v1(schema: &Value) -> bool {
    let Value::Object(fields) = schema else {
        return true;
    };
    // A type the coercion reads as a string, or reads as no JSON type at all
    // and so keeps as text, admits a string.
    match fields.get("type") {
        Some(Value::String(kind)) => return text_type_v1(kind),
        Some(Value::Array(kinds)) => {
            return kinds
                .iter()
                .any(|kind| kind.as_str().is_some_and(text_type_v1));
        }
        _ => {}
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(Value::Array(branches)) = fields.get(keyword) {
            return branches.iter().any(admits_string_v1);
        }
    }
    if let Some(Value::Array(values)) = fields.get("enum") {
        return values.iter().any(Value::is_string);
    }
    if let Some(value) = fields.get("const") {
        return value.is_string();
    }
    // With no type information the decoder keeps the text as a string.
    !["properties", "items", "minimum", "maximum"]
        .iter()
        .any(|keyword| fields.contains_key(*keyword))
}

/// [`earliest_v1`] over the JSON text `text`, skipping string literals and
/// the escapes inside them.
fn earliest_outside_strings_v1<T: Copy>(
    text: &str,
    terminals: &[(&str, T)],
) -> Option<(usize, usize, T)> {
    let mut strings = JsonStringsV1::default();
    for (index, character) in text.char_indices() {
        if strings.read(character) {
            continue;
        }
        if let Some((terminal, value)) = longest_terminal_v1(&text[index..], terminals) {
            return Some((index, terminal.len(), value));
        }
    }
    None
}

/// The longest terminal `text` starts with; the last listed at a tie.
fn longest_terminal_v1<'t, T: Copy>(
    text: &str,
    terminals: &[(&'t str, T)],
) -> Option<(&'t str, T)> {
    terminals
        .iter()
        .filter(|(terminal, _)| text.starts_with(terminal))
        .max_by_key(|(terminal, _)| terminal.len())
        .copied()
}

/// Where a reading of JSON text stands: inside a string literal, and after
/// a backslash there.
#[derive(Clone, Copy, Default)]
struct JsonStringsV1 {
    in_string: bool,
    escaped: bool,
}

impl JsonStringsV1 {
    /// Reads one character, and returns whether it belongs to a string
    /// literal, its quotes included.
    fn read(&mut self, character: char) -> bool {
        if self.in_string {
            if self.escaped {
                self.escaped = false;
            } else if character == '\\' {
                self.escaped = true;
            } else if character == '"' {
                self.in_string = false;
            }
            return true;
        }
        if character == '"' {
            self.in_string = true;
            return true;
        }
        false
    }
}

/// The first of several terminals in `text`, preferring the longer at a tie.
fn earliest_v1<T: Copy>(text: &str, terminals: &[(&str, T)]) -> Option<(usize, usize, T)> {
    terminals
        .iter()
        .filter_map(|(terminal, value)| text.find(terminal).map(|at| (at, terminal.len(), *value)))
        .min_by_key(|(at, length, _)| (*at, usize::MAX - *length))
}

/// What a parameter walk is reading.
#[derive(Clone, Copy)]
enum ParameterStateV1 {
    /// Text before the next parameter open.
    Seek,
    /// A name that began here.
    Name(usize),
    /// A value that began at `value`, of the parameter named at `name`. A
    /// JSON value tracks its string literals.
    Value {
        name: (usize, usize),
        value: usize,
        json: Option<JsonStringsV1>,
    },
}

/// A walk of a function's arguments, one character at a time.
///
/// A parameter ends at its close or where the next one opens, and only the
/// last one can lack a value. A raw value ends at the first close or open
/// whatever it holds, as the reference parser reads it. A JSON value ends the
/// same way, except that a marker inside one of its string literals is
/// content. The reference parser also ends a JSON value there; this decoder
/// diverges so that a value the tool-choice grammar writes, whose strings may
/// spell any marker, parses back whole. The typed grammar's raw value ends
/// only at its close, as that grammar's reference parser reads it, so it
/// holds parameter opens and function closes as text.
///
/// A walk of a longer text that begins with the one walked before resumes
/// where every earlier decision is final, so the walk to each function close
/// in turn costs the arguments once, not once per close.
#[derive(Clone, Copy)]
struct ParameterWalkV1 {
    state: ParameterStateV1,
    /// The next byte to read.
    at: usize,
}

impl Default for ParameterWalkV1 {
    fn default() -> Self {
        Self {
            state: ParameterStateV1::Seek,
            at: 0,
        }
    }
}

impl ParameterWalkV1 {
    /// Walks `raw`, a function's arguments up to a function close, from
    /// where this walk stands; reports each parameter whose name is complete,
    /// with its value once that ended; and returns whether `raw` ends inside a
    /// string literal of a JSON value, or inside a typed grammar's raw value:
    /// where a function close is text.
    ///
    /// The walk then stands before the last bytes a marker could still begin
    /// in, so it can go on over any longer text that begins with `raw`.
    fn walk<'a>(
        &mut self,
        terminals: ChatMlTerminalsV1<'_>,
        raw: &'a str,
        json: &dyn Fn(&str) -> bool,
        found: &mut dyn FnMut(&'a str, Option<&'a str>, bool),
    ) -> bool {
        let settles =
            floor_char_boundary_v1(raw, raw.len().saturating_sub(terminals.longest() - 1));
        let ends = [
            (terminals.parameter_close, true),
            (terminals.parameter_open, false),
        ];
        let mut walk = *self;
        let mut settling = true;
        loop {
            if settling && walk.at >= settles {
                *self = walk;
                settling = false;
            }
            let here = &raw[walk.at..];
            let Some(character) = here.chars().next() else {
                break;
            };
            match walk.state {
                ParameterStateV1::Seek => {
                    if here.starts_with(terminals.parameter_open) {
                        walk.at += terminals.parameter_open.len();
                        walk.state = ParameterStateV1::Name(walk.at);
                        continue;
                    }
                }
                ParameterStateV1::Name(name) => {
                    // The typed grammar's close says whether the value is
                    // JSON; otherwise the parameter's schema does.
                    let typed = terminals.grammar == FrontendToolGrammarV1::XmlTypedParametersV1;
                    let close = if here.starts_with(terminals.parameter_name_close) {
                        Some((terminals.parameter_name_close, false))
                    } else if typed && here.starts_with(terminals.parameter_json_name_close) {
                        Some((terminals.parameter_json_name_close, true))
                    } else {
                        None
                    };
                    if let Some((close, flagged)) = close {
                        let parameter = raw[name..walk.at].trim();
                        let value = walk.at + close.len();
                        let scanned = if typed { flagged } else { json(parameter) };
                        walk.state = ParameterStateV1::Value {
                            name: (name, walk.at),
                            value,
                            json: scanned.then(JsonStringsV1::default),
                        };
                        walk.at = value;
                        continue;
                    }
                }
                ParameterStateV1::Value {
                    name,
                    value,
                    json: None,
                } => {
                    // An open ends the value before a close at the same byte,
                    // except in the typed grammar, whose raw value runs to its
                    // close whatever it holds, as the reference parser reads
                    // it.
                    let typed = terminals.grammar == FrontendToolGrammarV1::XmlTypedParametersV1;
                    let end = if !typed && here.starts_with(terminals.parameter_open) {
                        Some(0)
                    } else if here.starts_with(terminals.parameter_close) {
                        Some(terminals.parameter_close.len())
                    } else {
                        None
                    };
                    if let Some(length) = end {
                        found(
                            raw[name.0..name.1].trim(),
                            Some(&raw[value..walk.at]),
                            false,
                        );
                        walk.state = ParameterStateV1::Seek;
                        walk.at += length;
                        continue;
                    }
                }
                ParameterStateV1::Value {
                    name,
                    value,
                    json: Some(mut strings),
                } => {
                    if strings.read(character) {
                        walk.state = ParameterStateV1::Value {
                            name,
                            value,
                            json: Some(strings),
                        };
                    } else if let Some((terminal, closes)) = longest_terminal_v1(here, &ends) {
                        found(raw[name.0..name.1].trim(), Some(&raw[value..walk.at]), true);
                        walk.state = ParameterStateV1::Seek;
                        walk.at += if closes { terminal.len() } else { 0 };
                        continue;
                    }
                }
            }
            walk.at += character.len_utf8();
        }
        match walk.state {
            ParameterStateV1::Seek | ParameterStateV1::Name(_) => false,
            ParameterStateV1::Value { name, json, .. } => {
                found(raw[name.0..name.1].trim(), None, json.is_some());
                // A typed grammar's raw value holds a function close as text.
                match json {
                    Some(strings) => strings.in_string,
                    None => terminals.grammar == FrontendToolGrammarV1::XmlTypedParametersV1,
                }
            }
        }
    }
}

/// One call's arguments in the declared JSON encoding.
///
/// The parameters are walked as the region scan walked them. The whitespace
/// the declared markers write around a value is markup ([`framed_value_v1`]);
/// every other byte between the markers is the value. A closed call keeps only its
/// complete parameters, as the reference does when it finishes a call. Open
/// calls withhold arguments because later parameters can replace earlier ones.
pub(crate) fn call_arguments_v1(
    descriptor: &FormatDescriptorV1,
    terminals: ChatMlTerminalsV1<'_>,
    call: &RegionCallV1,
    text: &str,
    tools: &[FrontendToolV1],
    open: bool,
) -> Result<String, String> {
    if open {
        return Ok(String::new());
    }
    let raw = &text[call.arguments.clone()];
    if terminals.grammar == FrontendToolGrammarV1::XmlKeyValueArgumentsV1 {
        return key_value_arguments_v1(descriptor, terminals, &call.name, raw, tools);
    }
    let json = json_arguments_v1(tools, &call.name);
    let mut parameters = Vec::new();
    ParameterWalkV1::default().walk(terminals, raw, &json, &mut |parameter, value, flagged| {
        parameters.push((parameter, value, flagged));
    });
    let schema = tool_parameters_v1(tools, &call.name);
    let mut arguments: Vec<(String, FrontendJsonV1)> = Vec::new();
    for (parameter, value, flagged) in parameters {
        validate_name_v1(descriptor, parameter, "parsed parameter")?;
        let Some(value) = value else {
            break;
        };
        let value = framed_value_v1(terminals, value);
        let value = if terminals.grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
            typed_value_v1(value, flagged)
        } else {
            coerce_parameter(schema.as_ref(), parameter, value)
        };
        insert_argument_v1(descriptor, &mut arguments, parameter, value)?;
    }
    json_to_string_v1(
        &FrontendJsonV1::Object(arguments),
        terminals.json_encoding,
        "tool arguments",
    )
}

/// A typed grammar's value as its close typed it: raw text is the string it
/// spells, and JSON the value it spells, a repeated key taking its last
/// value. JSON that does not read, or holds a number outside the range a
/// request's JSON reader accepts (a fraction or exponent beyond a finite
/// binary64; integers keep every digit), is kept as its text, as the other
/// grammars' coercion keeps it.
fn typed_value_v1(value: &str, json: bool) -> FrontendJsonV1 {
    if !json {
        return FrontendJsonV1::String(value.to_owned());
    }
    match serde_json::from_str::<Value>(value) {
        Ok(parsed) if numbers_in_range_v1(&parsed) => json_value_v1(parsed),
        _ => FrontendJsonV1::String(value.to_owned()),
    }
}

/// Whether every number in `value` is one a request's JSON reader accepts.
fn numbers_in_range_v1(value: &Value) -> bool {
    match value {
        Value::Number(number) => {
            !number.to_string().contains(['.', 'e', 'E'])
                || number.as_f64().is_some_and(f64::is_finite)
        }
        Value::Array(values) => values.iter().all(numbers_in_range_v1),
        Value::Object(entries) => entries.values().all(numbers_in_range_v1),
        _ => true,
    }
}

/// A `serde_json` value as a request's JSON value, keys in map order.
fn json_value_v1(value: Value) -> FrontendJsonV1 {
    match value {
        Value::Null => FrontendJsonV1::Null,
        Value::Bool(value) => FrontendJsonV1::Bool(value),
        Value::Number(number) => FrontendJsonV1::Number(number),
        Value::String(text) => FrontendJsonV1::String(text),
        Value::Array(values) => {
            FrontendJsonV1::Array(values.into_iter().map(json_value_v1).collect())
        }
        Value::Object(entries) => FrontendJsonV1::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key, json_value_v1(value)))
                .collect(),
        ),
    }
}

/// One key-value call's arguments in the declared JSON encoding.
///
/// The arguments are walked as the region scan walked them, and each value is
/// kept raw, between its markers, before the schema coerces it.
fn key_value_arguments_v1(
    descriptor: &FormatDescriptorV1,
    terminals: ChatMlTerminalsV1<'_>,
    name: &str,
    raw: &str,
    tools: &[FrontendToolV1],
) -> Result<String, String> {
    let json = json_arguments_v1(tools, name);
    let (pairs, _) = walk_key_value_arguments_v1(terminals, raw, &json);
    let schema = tool_parameters_v1(tools, name);
    let mut arguments: Vec<(String, FrontendJsonV1)> = Vec::new();
    for (parameter, value) in pairs {
        validate_name_v1(descriptor, parameter, "parsed parameter")?;
        let value = coerce_parameter(schema.as_ref(), parameter, value);
        insert_argument_v1(descriptor, &mut arguments, parameter, value)?;
    }
    json_to_string_v1(
        &FrontendJsonV1::Object(arguments),
        terminals.json_encoding,
        "tool arguments",
    )
}

/// Adds one parsed argument; a repeated one replaces the earlier value in
/// place.
fn insert_argument_v1(
    descriptor: &FormatDescriptorV1,
    arguments: &mut Vec<(String, FrontendJsonV1)>,
    parameter: &str,
    value: FrontendJsonV1,
) -> Result<(), String> {
    if let Some(existing) = arguments.iter_mut().find(|(key, _)| key == parameter) {
        existing.1 = value;
    } else {
        if arguments.len() >= descriptor.limits.maximum_tool_parameters {
            return Err(resource_v1("too many completion tool parameters"));
        }
        arguments.push((parameter.to_owned(), value));
    }
    Ok(())
}

/// Whether the raw value text `value` of the parameter `key`, written as the
/// tool-choice grammar writes it in a call to `tool` (alone, between the
/// declared markers and wrappers, whitespace included), reads back as
/// exactly that text. The call is read by the served parser itself, as a
/// whole completion that stops after it (`completion_calls_v1`), so the
/// check reads wrappers, framing and the key's scan as a completion does: it
/// holds when that completion yields this one call, whose argument is what
/// the coercion makes of the whole text. A literal the grammar offers, such
/// as a string enumeration value, may spell a marker that ends a raw value
/// early, or open a string literal that a JSON value's scan never closes.
pub(crate) fn literal_value_reads_back_v1(
    descriptor: &Arc<FormatDescriptorV1>,
    tool: &FrontendToolV1,
    key: &str,
    value: &str,
) -> bool {
    let parser = &descriptor.parser;
    let text = [
        parser.tool_call_open.as_str(),
        &parser.function_open,
        &tool.name,
        &parser.function_name_close,
        &parser.parameter_open,
        key,
        &parser.parameter_name_close,
        &parser.parameter_value_open,
        value,
        &parser.parameter_close,
        &parser.function_close,
        &parser.tool_call_close,
    ]
    .concat();
    let tools = std::slice::from_ref(tool);
    let Ok(calls) = completion_calls_v1(descriptor, &text, tools) else {
        return false;
    };
    let schema = tool_parameters_v1(tools, &tool.name);
    let argument = if parser.tool_grammar == FrontendToolGrammarV1::XmlTypedParametersV1 {
        FrontendJsonV1::String(value.to_owned())
    } else {
        coerce_parameter(schema.as_ref(), key, value)
    };
    let Ok(arguments) = json_to_string_v1(
        &FrontendJsonV1::Object(vec![(key.to_owned(), argument)]),
        parser.json_encoding,
        "tool arguments",
    ) else {
        return false;
    };
    matches!(calls.as_slice(), [call] if call.name == tool.name && call.arguments == arguments)
}

/// A function parameter's value without the framing its declared markers
/// write: the whitespace after the bare name close, once, and the whitespace
/// before the bare parameter close, once. A template that writes
/// `>\n` and `\n</parameter>` frames each value in one newline; one that
/// writes `>` and `</parameter>` frames nothing, so every byte is the value.
fn framed_value_v1<'a>(terminals: ChatMlTerminalsV1<'_>, value: &'a str) -> &'a str {
    let value = value.strip_prefix(terminals.value_lead).unwrap_or(value);
    value.strip_suffix(terminals.value_trail).unwrap_or(value)
}

/// A parameter's raw value as its schema reads it ([`parameter_schema_v1`]):
/// coerced to a typed schema's JSON type, and otherwise kept as a string.
/// Coercion follows the schema whatever the tool's `strict`, as the reference
/// parsers coerce.
fn coerce_parameter(parameters: Option<&Value>, parameter: &str, value: &str) -> FrontendJsonV1 {
    match parameters.map(|parameters| parameter_schema_v1(parameters, parameter)) {
        Some(ParameterSchemaV1::Typed(schema)) => coerce_raw(schema, value),
        _ => FrontendJsonV1::String(value.to_owned()),
    }
}

fn schema_types(schema: &Value) -> Vec<String> {
    let mut types = Vec::new();
    collect_schema_types(schema, &mut types);
    if types.is_empty() {
        types.push("string".to_owned());
    }
    types
}

fn collect_schema_types(schema: &Value, types: &mut Vec<String>) {
    match schema.get("type") {
        Some(Value::String(kind)) => push_type(types, kind),
        Some(Value::Array(values)) => {
            for value in values {
                if let Value::String(kind) = value {
                    push_type(types, kind);
                }
            }
        }
        _ => {}
    }
    if let Some(Value::Array(values)) = schema.get("enum") {
        for value in values {
            let kind = match value {
                Value::Null => "null",
                Value::Bool(_) => "boolean",
                Value::Number(number) if !number.to_string().contains(['.', 'e', 'E']) => "integer",
                Value::Number(_) => "number",
                Value::String(_) => "string",
                Value::Array(_) => "array",
                Value::Object(_) => "object",
            };
            push_type(types, kind);
        }
    }
    for choice in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(values)) = schema.get(choice) {
            for value in values {
                collect_schema_types(value, types);
            }
        }
    }
}

/// The JSON type a schema's type name means to the coercion: the name
/// trimmed and lowercased, with the aliases the reference parsers read
/// (`str`, `int`, `float`, `bool`, `dict`, `list` and the like) mapped to
/// the JSON type.
fn json_type_name_v1(kind: &str) -> String {
    let normalized = kind.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "str" | "text" | "varchar" | "char" | "enum" => "string".to_owned(),
        "int" | "int32" | "int64" | "uint" | "uint32" | "uint64" | "long" | "short"
        | "unsigned" => "integer".to_owned(),
        "float" | "float32" | "float64" | "double" => "number".to_owned(),
        "bool" => "boolean".to_owned(),
        "dict" => "object".to_owned(),
        "arr" | "list" | "sequence" => "array".to_owned(),
        _ => normalized,
    }
}

/// Whether the coercion keeps a value of the type `kind` as raw text: a
/// string type or alias, or a name it reads as no other JSON type.
fn text_type_v1(kind: &str) -> bool {
    !["integer", "number", "boolean", "object", "array", "null"]
        .contains(&json_type_name_v1(kind).as_str())
}

fn push_type(types: &mut Vec<String>, kind: &str) {
    let kind = json_type_name_v1(kind);
    if !types.contains(&kind) {
        types.push(kind);
    }
}

fn coerce_nested(value: &mut FrontendJsonV1, schema: &Value) {
    match value {
        FrontendJsonV1::Object(fields) => {
            if let Some(properties) = schema.get("properties") {
                for (name, value) in fields {
                    if let Some(property) = properties.get(name.as_str()) {
                        coerce_value(value, property);
                    }
                }
            }
        }
        FrontendJsonV1::Array(values) => {
            if let Some(items) = schema.get("items") {
                for value in values {
                    coerce_value(value, items);
                }
            }
        }
        _ => {}
    }
}

fn coerce_value(value: &mut FrontendJsonV1, schema: &Value) {
    if let FrontendJsonV1::String(raw) = value {
        *value = coerce_raw(schema, raw);
    } else {
        coerce_nested(value, schema);
    }
}

fn decimal_integer(text: &str) -> Option<serde_json::Number> {
    let negative = text.starts_with('-');
    let digits = text.strip_prefix(['-', '+']).unwrap_or(text);
    if digits.is_empty() || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Some(0.into());
    }
    format!("{}{digits}", if negative { "-" } else { "" })
        .parse()
        .ok()
}

fn coerce_raw(schema: &Value, value: &str) -> FrontendJsonV1 {
    let types = schema_types(schema);
    if types.iter().any(|kind| kind == "null") && value.eq_ignore_ascii_case("null") {
        return FrontendJsonV1::Null;
    }
    if types
        .iter()
        .any(|kind| kind == "integer" || kind == "number")
        && let Some(number) = decimal_integer(value)
    {
        return FrontendJsonV1::Number(number);
    }
    if types.iter().any(|kind| kind == "number")
        && let Ok(number) = value.parse::<f64>()
        && number.is_finite()
        && let Some(number) = serde_json::Number::from_f64(number)
    {
        return FrontendJsonV1::Number(number);
    }
    if types.iter().any(|kind| kind == "boolean") {
        match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" => return FrontendJsonV1::Bool(true),
            "false" | "0" => return FrontendJsonV1::Bool(false),
            _ => {}
        }
    }
    if types.iter().any(|kind| kind == "object" || kind == "array")
        && let Ok(mut parsed) = serde_json::from_str::<FrontendJsonV1>(value)
    {
        coerce_nested(&mut parsed, schema);
        return parsed;
    }
    FrontendJsonV1::String(value.to_owned())
}

#[cfg(test)]
mod number_tests {
    use super::*;

    #[test]
    fn typed_integer_parameters_preserve_every_decimal_digit() {
        for kind in ["integer", "number"] {
            let schema = serde_json::json!({"type":kind});
            for text in [
                "18446744073709551617",
                "-9223372036854775809",
                "1234567890123456789012345678901234567890",
            ] {
                let value = coerce_raw(&schema, text);
                assert_eq!(value.to_json_value().to_string(), text);
            }
        }
    }
}
