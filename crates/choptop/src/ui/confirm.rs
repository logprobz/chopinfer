//! The confirm dialogs: what a switch starts and stops before it runs, and
//! what `X` stops.

use super::buffer::{Buffer, Color, Style};
use super::widgets::{boxed, sanitize, wrap};
use crate::app::{App, Dialog, StopConfirm};
use crate::launch::switch::Confirm;
use crate::launch::{DRAIN_SECONDS, indices};

/// The start dialog's width, and the stop dialog's.
const START_WIDTH: u16 = 96;
const STOP_WIDTH: u16 = 92;

/// A line of styled pieces.
type Line = Vec<(String, Style)>;

/// Spaces after the widest entry of an instance row's column.
const GAP: usize = 4;

/// Most rows the profile's qualification takes in the start dialog.
const QUALIFICATION_ROWS: usize = 4;

/// The indent of a line a narrow dialog breaks off a longer one.
const FLOW_INDENT: usize = 5;

fn cyan() -> Style {
    Style::fg(Color::Cyan).bold()
}

/// `text` wrapped to `width`, as lines indented by three cells, then five.
fn wrapped(lines: &mut Vec<Line>, text: &str, width: usize, style: Style) {
    for (index, part) in wrap(&sanitize(text), width).into_iter().enumerate() {
        let indent = if index == 0 { "   " } else { "     " };
        lines.push(vec![(format!("{indent}{part}"), style)]);
    }
}

/// `path` in lines of at most `width` characters, each broken after a
/// `/` where one fits.
fn path_lines(path: &str, width: usize) -> Vec<String> {
    let mut rest = path.chars().collect::<Vec<_>>();
    let mut lines = Vec::new();
    while rest.len() > width && width > 0 {
        let cut = rest[..width]
            .iter()
            .rposition(|character| *character == '/')
            .filter(|slash| *slash > 0)
            .map_or(width, |slash| slash + 1);
        lines.push(rest.drain(..cut).collect());
    }
    lines.push(rest.into_iter().collect());
    lines
}

/// `line` broken at spaces into lines of at most `width` characters, a
/// word longer than that broken anywhere, each line after the first
/// indented by [`FLOW_INDENT`] cells. A line that fits stays as it is.
fn flow(line: Line, width: usize) -> Vec<Line> {
    if length(&line) <= width {
        return vec![line];
    }
    let width = width.max(FLOW_INDENT + 1);
    let characters = line
        .iter()
        .flat_map(|(text, style)| text.chars().map(move |character| (character, *style)))
        .collect::<Vec<_>>();
    let indent = vec![(' ', Style::PLAIN); FLOW_INDENT];
    let mut lines = Vec::new();
    let mut current = Vec::new();
    // Whether the current line has no word yet.
    let mut fresh = true;
    let mut index = 0;
    while index < characters.len() {
        let from = index;
        let start = from
            + characters[from..]
                .iter()
                .take_while(|(character, _)| *character == ' ')
                .count();
        let letters = characters[start..]
            .iter()
            .take_while(|(character, _)| *character != ' ')
            .count();
        let mut word = &characters[start..start + letters];
        index = start + letters;
        if word.is_empty() {
            break;
        }
        if !fresh && current.len() + (start - from) + word.len() > width {
            lines.push(std::mem::replace(&mut current, indent.clone()));
            fresh = true;
        }
        // The spaces before a word stay, but for the first word of a line
        // broken off.
        if !fresh || lines.is_empty() {
            current.extend_from_slice(&characters[from..start]);
        }
        while current.len() + word.len() > width {
            let take = width.saturating_sub(current.len());
            current.extend_from_slice(&word[..take]);
            word = &word[take..];
            lines.push(std::mem::replace(&mut current, indent.clone()));
            fresh = true;
        }
        if !word.is_empty() {
            current.extend_from_slice(word);
            fresh = false;
        }
    }
    if !fresh || lines.is_empty() {
        lines.push(current);
    }
    lines.into_iter().map(pieces).collect()
}

/// Every line of `lines` flowed to `width`.
fn flowed(lines: Vec<Line>, width: usize) -> Vec<Line> {
    lines
        .into_iter()
        .flat_map(|line| flow(line, width))
        .collect()
}

/// `characters` as pieces, one for each run of a style.
fn pieces(characters: Vec<(char, Style)>) -> Line {
    let mut line: Line = Vec::new();
    for (character, style) in characters {
        match line.last_mut() {
            Some((text, last)) if *last == style => text.push(character),
            _ => line.push((character.to_string(), style)),
        }
    }
    line
}

/// The characters `line` shows, without trailing spaces.
fn length(line: &Line) -> usize {
    line.iter()
        .map(|(text, _)| text.as_str())
        .collect::<String>()
        .trim_end()
        .chars()
        .count()
}

/// A dialog as a terminal draws it.
struct Layout {
    /// The box's width.
    width: u16,
    title: String,
    lines: Vec<Line>,
    /// Whether every line the dialog must show is drawn, uncut.
    whole: bool,
}

/// `dialog` on a terminal `width` by `height`: in a box as wide as the
/// terminal allows, its lines broken to fit, and when its rows do not fit,
/// how many it needs instead.
fn layout(app: &App, dialog: &Dialog, width: u16, height: u16) -> Layout {
    let room = room(height);
    let (width, title, lines, whole) = match dialog {
        Dialog::Start(confirm) => {
            let width = START_WIDTH.min(width.saturating_sub(2));
            let inner = usize::from(width.saturating_sub(2));
            let lines = start_lines(app, confirm, inner);
            let rows = lines.rows();
            let (lines, whole) = match lines.fit(room) {
                Some(lines) => (lines, true),
                None => {
                    let back = " back to the loader";
                    let what = ("this start stops and checks", "starts", back);
                    (
                        too_short(model_line(confirm), what, rows, height, inner),
                        false,
                    )
                }
            };
            (width, confirm.title(), lines, whole)
        }
        Dialog::Stop(stop) => {
            let width = STOP_WIDTH.min(width.saturating_sub(2));
            let inner = usize::from(width.saturating_sub(2));
            let lines = flowed(stop_lines(stop), inner);
            let rows = lines.len() + 4;
            let (lines, whole) = if lines.len() <= room {
                (lines, true)
            } else {
                let (enter, back) = if stop.waiting {
                    ("cancels it", " keep it queued")
                } else {
                    ("stops it", " keep it running")
                };
                let what = ("about this stop", enter, back);
                (
                    too_short(stop_label(stop), what, rows, height, inner),
                    false,
                )
            };
            let title = format!("Stop {}", sanitize(&stop.label));
            (width, title, lines, whole)
        }
    };
    let inner = usize::from(width.saturating_sub(2));
    let whole = whole && lines.iter().all(|line| length(line) <= inner);
    Layout {
        width,
        title,
        lines,
        whole,
    }
}

pub fn draw(buffer: &mut Buffer, app: &App, dialog: &Dialog) {
    let layout = layout(app, dialog, buffer.width, buffer.height);
    overlay(buffer, layout.width, &layout.title, layout.lines);
}

/// Whether a terminal `width` by `height` draws `dialog` whole, every line
/// it must show uncut. When it does not, the dialog says so instead, and
/// Enter does not confirm it.
pub fn whole(app: &App, dialog: &Dialog, width: u16, height: u16) -> bool {
    layout(app, dialog, width, height).whole
}

/// The rows a dialog's lines can take on a terminal `height` rows tall.
fn room(height: u16) -> usize {
    usize::from(height.saturating_sub(4))
}

/// Draws `lines` in a box `width` wide, centered, titled `title`. When the
/// terminal is too short, lines before the last two (a blank and the keys)
/// are left out from the end. A line too long for the box ends with
/// `...`, in the style of the piece it is cut in.
fn overlay(buffer: &mut Buffer, width: u16, title: &str, mut lines: Vec<Line>) {
    let width = width.min(buffer.width.saturating_sub(2));
    let room = room(buffer.height);
    if lines.len() > room && room >= 2 {
        let keep = lines.split_off(lines.len() - 2);
        lines.truncate(room - 2);
        lines.extend(keep);
    }
    let height = (lines.len() as u16 + 2).min(buffer.height.saturating_sub(2));
    let x = (buffer.width - width) / 2;
    let y = (buffer.height - height) / 2;
    buffer.clear(x, y, width, height, None);
    boxed(buffer, (x, y, width, height), &[(title, cyan())], true);
    let room = usize::from(width.saturating_sub(2));
    for (index, line) in lines.iter().take(usize::from(height - 2)).enumerate() {
        let row = y + 1 + index as u16;
        let length = length(line);
        let mut left = if length > room {
            room.saturating_sub(3)
        } else {
            room
        };
        let mut column = x + 1;
        let mut cut = None;
        for (text, style) in line {
            let shown = text.chars().take(left).collect::<String>();
            left -= shown.chars().count();
            column = buffer.put(column, row, &shown, *style);
            if left == 0 && cut.is_none() {
                cut = Some(*style);
            }
        }
        if length > room {
            buffer.put(
                column,
                row,
                &".".repeat(room.min(3)),
                cut.unwrap_or(Style::PLAIN),
            );
        }
    }
}

/// The start dialog's lines. Every line but the qualification's must
/// show; the qualification takes the rows the rest leaves, up to
/// [`QUALIFICATION_ROWS`].
struct StartLines {
    /// From the top to the Check heading.
    head: Vec<Line>,
    qualification: Vec<Line>,
    /// The warnings, host memory, the note on GPU failures and the keys.
    tail: Vec<Line>,
}

impl StartLines {
    /// The terminal rows that show every line that must show.
    fn rows(&self) -> usize {
        self.head.len() + self.tail.len() + 4
    }

    /// The lines in `room` rows, the qualification cut to what is left
    /// and ending with `...` when cut; `None` when the lines that must
    /// show do not fit.
    fn fit(self, room: usize) -> Option<Vec<Line>> {
        let left = room
            .checked_sub(self.head.len() + self.tail.len())?
            .min(QUALIFICATION_ROWS);
        let cut = self.qualification.len() > left;
        let mut lines = self.head;
        lines.extend(self.qualification.into_iter().take(left));
        if cut
            && left > 0
            && let Some((text, _)) = lines.last_mut().and_then(|line| line.last_mut())
        {
            text.push_str("...");
        }
        lines.extend(self.tail);
        Some(lines)
    }
}

/// A dialog on a terminal too short to show everything `what` names,
/// under its `head` line where there is room: how many `rows` it needs
/// before Enter does what `what` says, and only Esc, which `what` says the
/// meaning of. Its lines are flowed to `inner`.
fn too_short(
    head: Line,
    (what, enter, back): (&str, &str, &str),
    rows: usize,
    height: u16,
    inner: usize,
) -> Vec<Line> {
    let mut lines = vec![
        Vec::new(),
        head,
        Vec::new(),
        vec![(
            format!("   The terminal is too short to show everything {what}."),
            Style::fg(Color::Yellow),
        )],
        vec![(
            format!("   Enter {enter} once it is {rows} rows tall; it is {height}."),
            Style::PLAIN,
        )],
        Vec::new(),
        vec![(" Esc".to_owned(), cyan()), (back.to_owned(), Style::PLAIN)],
    ];
    if flowed(lines.clone(), inner).len() > room(height) {
        lines.drain(1..3);
    }
    flowed(lines, inner)
}

/// The served model name, the profile and the instances.
fn model_line(confirm: &Confirm) -> Line {
    let count = confirm.start.placement.instances.len();
    vec![
        (
            format!(" {}", sanitize(&confirm.served)),
            Style::fg(Color::White).bold(),
        ),
        (
            format!("  {}  ", sanitize(&confirm.start.profile)),
            Style::PLAIN,
        ),
        (
            format!(
                "{count} instance{} at TP{}",
                if count == 1 { "" } else { "s" },
                confirm.tensor_parallel
            ),
            Style::fg(Color::Dim),
        ),
    ]
}

/// The start dialog's lines for a box `inner` cells wide inside.
fn start_lines(app: &App, confirm: &Confirm, inner: usize) -> StartLines {
    let dim = Style::fg(Color::Dim);
    let yellow = Style::fg(Color::Yellow);
    let start = &confirm.start;
    let mut lines: Vec<Line> = vec![Vec::new(), model_line(confirm), Vec::new()];
    // Each column as wide as its widest entry and the gap, so a TP4 or TP8
    // list stays apart from the address.
    let rows = start
        .placement
        .instances
        .iter()
        .map(|planned| {
            (
                planned.letter,
                format!("GPUs {}", indices(&planned.gpus)),
                sanitize(&planned.listen()),
            )
        })
        .collect::<Vec<_>>();
    let width = |column: fn(&(char, String, String)) -> &String| {
        rows.iter()
            .map(|row| column(row).chars().count())
            .max()
            .unwrap_or(0)
            + GAP
    };
    let (gpus, listen) = (width(|row| &row.1), width(|row| &row.2));
    for ((letter, on, at), log) in rows.iter().zip(&confirm.logs) {
        let mut row = vec![
            (format!(" {letter}  "), cyan()),
            (format!("{on:<gpus$}"), Style::PLAIN),
            (format!("{at:<listen$}"), Style::PLAIN),
        ];
        let log = format!("log {}", sanitize(log));
        // On a narrow terminal the log goes below.
        if length(&row) + log.chars().count() <= inner {
            row.push((log, dim));
            lines.push(row);
        } else {
            lines.push(row);
            lines.push(vec![(format!("     {log}"), dim)]);
        }
    }
    let state = sanitize(app.logs.strip_suffix("/logs").unwrap_or(&app.logs));
    let stay = "logs and launch records stay in";
    if stay.len() + 1 + state.chars().count() <= inner.saturating_sub(4) {
        lines.push(vec![(format!("    {stay} {state}"), dim)]);
    } else {
        lines.push(vec![(format!("    {stay}"), dim)]);
        for part in path_lines(&state, inner.saturating_sub(6)) {
            lines.push(vec![(format!("      {part}"), dim)]);
        }
    }
    if !confirm.stops.is_empty() {
        lines.push(Vec::new());
        lines.push(vec![(" Stops first".to_owned(), cyan())]);
        for stop in &confirm.stops {
            let on = if stop.gpus.is_empty() {
                String::new()
            } else {
                format!(" on GPUs {}", indices(&stop.gpus))
            };
            let what = format!(
                "   {}  {}{on}: ",
                sanitize(&stop.label),
                sanitize(&stop.model)
            );
            if what.chars().count() + stop.how.chars().count() <= inner {
                lines.push(vec![(what, yellow), (stop.how.clone(), Style::PLAIN)]);
            } else {
                lines.push(vec![(what, yellow)]);
                lines.push(vec![(format!("     {}", stop.how), Style::PLAIN)]);
            }
        }
        lines.push(vec![(
            format!(
                "   serve drains admitted requests for up to {DRAIN_SECONDS} s (--drain-timeout-ms) before it exits"
            ),
            dim,
        )]);
    }
    lines.push(Vec::new());
    lines.push(vec![(" Check".to_owned(), cyan())]);
    let width = inner.saturating_sub(8);
    let mut qualification = Vec::new();
    if let Some(text) = &confirm.qualification {
        wrapped(&mut qualification, text, width, yellow);
    }
    let mut tail = Vec::new();
    for warning in &confirm.warnings {
        wrapped(&mut tail, warning, width, yellow);
    }
    if let Some(memory) = &confirm.memory {
        wrapped(&mut tail, memory, width, dim);
    }
    wrapped(
        &mut tail,
        "GPU failures such as memory fit or peer access can still happen after the stop",
        width,
        dim,
    );
    tail.push(Vec::new());
    tail.push(vec![
        (" Enter".to_owned(), cyan()),
        (" start   ".to_owned(), Style::PLAIN),
        ("Esc".to_owned(), cyan()),
        (" back to the loader".to_owned(), Style::PLAIN),
    ]);
    StartLines {
        head: flowed(lines, inner),
        qualification,
        tail: flowed(tail, inner),
    }
}

/// What the stop dialog stops.
fn stop_label(stop: &StopConfirm) -> Line {
    vec![
        (
            format!(" {}  ", sanitize(&stop.label)),
            Style::fg(Color::White).bold(),
        ),
        (sanitize(&stop.what), Style::PLAIN),
    ]
}

fn stop_lines(stop: &StopConfirm) -> Vec<Line> {
    let dim = Style::fg(Color::Dim);
    let (how, after, verb, keep) = if stop.waiting {
        (
            "   It has not started: Enter cancels it, and its GPUs and port are free at once."
                .to_owned(),
            "   Its row leaves the list.",
            " cancel it   ",
            " keep it queued",
        )
    } else {
        (
            format!(
                "   SIGTERM: serve stops admitting, drains admitted requests for up to {DRAIN_SECONDS} s, then exits."
            ),
            "   Its row leaves the list once the process is gone.",
            " stop   ",
            " keep it running",
        )
    };
    vec![
        Vec::new(),
        stop_label(stop),
        Vec::new(),
        vec![(how, Style::PLAIN)],
        vec![(after.to_owned(), dim)],
        Vec::new(),
        vec![
            (" Enter".to_owned(), cyan()),
            (verb.to_owned(), Style::PLAIN),
            ("Esc".to_owned(), cyan()),
            (keep.to_owned(), Style::PLAIN),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_too_long_for_its_dialog_ends_with_dots() {
        let mut buffer = Buffer::new(40, 8);
        let dim = Style::fg(Color::Dim);
        overlay(
            &mut buffer,
            30,
            "Start",
            vec![
                vec![("x".repeat(20), Style::PLAIN), ("y".repeat(20), dim)],
                vec![("fits".to_owned(), Style::PLAIN)],
            ],
        );
        let text = buffer.text();
        let row = text.lines().nth(3).unwrap();
        assert_eq!(
            row,
            format!("     │{}{}...│", "x".repeat(20), "y".repeat(5))
        );
        assert_eq!(buffer.cell(31, 3).unwrap().style, dim);
        assert!(text.lines().nth(4).unwrap().starts_with("     │fits "));
    }

    #[test]
    fn a_line_too_wide_breaks_at_spaces_then_anywhere() {
        let dim = Style::fg(Color::Dim);
        let line = vec![
            ("   abc def".to_owned(), Style::PLAIN),
            (" ghijklmnop".to_owned(), dim),
        ];
        assert_eq!(flow(line.clone(), 21), vec![line.clone()]);
        assert_eq!(
            flow(line, 12),
            [
                vec![("   abc def".to_owned(), Style::PLAIN)],
                vec![
                    ("     ".to_owned(), Style::PLAIN),
                    ("ghijklm".to_owned(), dim)
                ],
                vec![("     ".to_owned(), Style::PLAIN), ("nop".to_owned(), dim)],
            ]
        );
    }

    #[test]
    fn a_long_path_breaks_after_a_slash() {
        assert_eq!(
            path_lines("/var/lib/a-long-name/choptop", 22),
            ["/var/lib/a-long-name/", "choptop"]
        );
        assert_eq!(path_lines("/state", 22), ["/state"]);
        assert_eq!(path_lines("/abcdefgh", 4), ["/abc", "defg", "h"]);
    }
}
