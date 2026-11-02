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

pub fn draw(buffer: &mut Buffer, app: &App, dialog: &Dialog) {
    let (width, title, lines) = match dialog {
        Dialog::Start(confirm) => (START_WIDTH, confirm.title(), start_lines(app, confirm)),
        Dialog::Stop(stop) => (
            STOP_WIDTH,
            format!("Stop {}", sanitize(&stop.label)),
            stop_lines(stop),
        ),
    };
    overlay(buffer, width, &title, lines);
}

/// Draws `lines` in a box `width` wide, centered, titled `title`. When the
/// terminal is too short, lines before the last two (a blank and the keys)
/// are left out from the end.
fn overlay(buffer: &mut Buffer, width: u16, title: &str, mut lines: Vec<Line>) {
    let width = width.min(buffer.width.saturating_sub(2));
    let room = usize::from(buffer.height.saturating_sub(4));
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
    let end = x + width - 1;
    for (index, line) in lines.iter().take(usize::from(height - 2)).enumerate() {
        let mut column = x + 1;
        for (text, style) in line {
            let room = usize::from(end.saturating_sub(column));
            column = buffer.put(
                column,
                y + 1 + index as u16,
                &text.chars().take(room).collect::<String>(),
                *style,
            );
        }
    }
}

fn start_lines(app: &App, confirm: &Confirm) -> Vec<Line> {
    let dim = Style::fg(Color::Dim);
    let yellow = Style::fg(Color::Yellow);
    let inner = usize::from(START_WIDTH - 2);
    let start = &confirm.start;
    let count = start.placement.instances.len();
    let mut lines: Vec<Line> = vec![
        Vec::new(),
        vec![
            (
                format!(" {}", sanitize(&confirm.served)),
                Style::fg(Color::White).bold(),
            ),
            (format!("  {}  ", sanitize(&start.profile)), Style::PLAIN),
            (
                format!(
                    "{count} instance{} at TP{}",
                    if count == 1 { "" } else { "s" },
                    confirm.tensor_parallel
                ),
                dim,
            ),
        ],
        Vec::new(),
    ];
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
        lines.push(vec![
            (format!(" {letter}  "), cyan()),
            (format!("{on:<gpus$}"), Style::PLAIN),
            (format!("{at:<listen$}"), Style::PLAIN),
            (format!("log {}", sanitize(log)), dim),
        ]);
    }
    let state = app.logs.strip_suffix("/logs").unwrap_or(&app.logs);
    lines.push(vec![(
        format!("    logs and launch records stay in {}", sanitize(state)),
        dim,
    )]);
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
    let width = inner - 8;
    if let Some(qualification) = &confirm.qualification {
        wrapped(&mut lines, qualification, width, yellow);
    }
    for warning in &confirm.warnings {
        wrapped(&mut lines, warning, width, yellow);
    }
    if let Some(memory) = &confirm.memory {
        wrapped(&mut lines, memory, width, dim);
    }
    wrapped(
        &mut lines,
        "GPU failures such as memory fit or peer access can still happen after the stop",
        width,
        dim,
    );
    lines.push(Vec::new());
    lines.push(vec![
        (" Enter".to_owned(), cyan()),
        (" start   ".to_owned(), Style::PLAIN),
        ("Esc".to_owned(), cyan()),
        (" back to the loader".to_owned(), Style::PLAIN),
    ]);
    lines
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
        vec![
            (
                format!(" {}  ", sanitize(&stop.label)),
                Style::fg(Color::White).bold(),
            ),
            (sanitize(&stop.what), Style::PLAIN),
        ],
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
