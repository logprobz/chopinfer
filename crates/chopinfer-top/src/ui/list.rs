//! The list view: every instance, a summary of the selected one, and the
//! events.

use super::buffer::{Buffer, Color, Style};
use super::widgets::{
    age, boxed, buckets, ceiling_label, gauge, integer, latency, nice_ceiling, pad, percent, right,
    sanitize, sparkline,
};
use super::{Figures, INTER_TOKEN, TTFT, footer, header, shown_name, state_group, tone_color};
use crate::app::App;
use crate::instances::{Instance, Source, State};

/// The row of the instances table's top border.
pub const TABLE_TOP: u16 = 2;

/// A table column: its title, width, alignment, the gap after it, and the
/// order in which it is dropped when the terminal is narrow (higher first;
/// zero is never dropped).
pub(super) struct Column {
    pub(super) title: &'static str,
    pub(super) width: usize,
    pub(super) right: bool,
    pub(super) gap: usize,
    pub(super) drop: u8,
}

const COLUMNS: [Column; 11] = [
    Column {
        title: "SOURCE",
        width: 14,
        right: false,
        gap: 2,
        drop: 0,
    },
    Column {
        title: "ADDRESS",
        width: 15,
        right: false,
        gap: 2,
        drop: 0,
    },
    Column {
        title: "MODEL",
        width: 15,
        right: false,
        gap: 2,
        drop: 0,
    },
    Column {
        title: "STATE",
        width: 13,
        right: false,
        gap: 1,
        drop: 0,
    },
    Column {
        title: "DECODE",
        width: 7,
        right: true,
        gap: 1,
        drop: 1,
    },
    Column {
        title: "PREFILL",
        width: 7,
        right: true,
        gap: 1,
        drop: 4,
    },
    Column {
        title: "RUN/WAIT",
        width: 8,
        right: true,
        gap: 1,
        drop: 2,
    },
    Column {
        title: "KV",
        width: 4,
        right: true,
        gap: 1,
        drop: 3,
    },
    Column {
        title: "PREFIX",
        width: 6,
        right: true,
        gap: 1,
        drop: 5,
    },
    Column {
        title: "SPEC",
        width: 4,
        right: true,
        gap: 1,
        drop: 6,
    },
    Column {
        title: "TTFT50",
        width: 6,
        right: true,
        gap: 2,
        drop: 5,
    },
];
/// The narrowest decode sparkline worth showing.
const SPARK_MINIMUM: usize = 8;

/// The instances table's height on a terminal `height` rows tall: one line
/// per row, with the borders and titles, and room left for the events.
fn table_height(rows: usize, height: u16) -> u16 {
    u16::try_from(rows.saturating_add(3))
        .unwrap_or(u16::MAX)
        .clamp(4, height.saturating_sub(8).max(4))
}

/// How many rows the table shows.
pub fn table_rows(rows: usize, height: u16) -> usize {
    usize::from(table_height(rows, height) - 3)
}

/// The first row shown: `scroll`, moved just enough that `selected` is one
/// of the `room` rows shown.
pub fn first_row(scroll: usize, selected: usize, room: usize) -> usize {
    scroll
        .min(selected)
        .max((selected + 1).saturating_sub(room.max(1)))
}

/// Which of `columns` fit `room` cells, and the sparkline width left over.
pub(super) fn layout(columns: &[Column], room: usize) -> (Vec<usize>, usize) {
    let mut shown = (0..columns.len()).collect::<Vec<_>>();
    let used = |shown: &[usize]| {
        shown
            .iter()
            .map(|index| columns[*index].width + columns[*index].gap)
            .sum::<usize>()
    };
    let mut level = columns.iter().map(|column| column.drop).max().unwrap_or(0);
    while used(&shown) > room && level > 0 {
        shown.retain(|index| columns[*index].drop != level);
        level -= 1;
    }
    let spark = room.saturating_sub(used(&shown));
    (shown, if spark >= SPARK_MINIMUM { spark } else { 0 })
}

pub fn draw(buffer: &mut Buffer, app: &App, now: f64) {
    let rows = app.rows();
    let mut counts: Vec<(&'static str, usize, Color)> = Vec::new();
    for row in &rows {
        let state = row.state(now);
        let group = state_group(&state);
        match counts.iter_mut().find(|(seen, _, _)| *seen == group) {
            Some((_, count, _)) => *count += 1,
            None => counts.push((group, 1, tone_color(state.tone()))),
        }
    }
    let mut crumbs = vec![(
        format!(
            "   {} instance{}: ",
            rows.len(),
            if rows.len() == 1 { "" } else { "s" }
        ),
        Style::PLAIN,
    )];
    for (group, count, color) in &counts {
        crumbs.push((format!("{count} {group}  "), Style::fg(*color)));
    }
    crumbs.push((
        format!(" sources: {}", app.sources.join(", ")),
        Style::fg(Color::Default),
    ));
    header(buffer, app, &crumbs);

    let width = buffer.width;
    let height = buffer.height;
    let table_height = table_height(rows.len(), height);
    let visible = table_rows(rows.len(), height);
    let selected = app.selected_index();
    let first = first_row(app.scroll, selected, visible);
    let room = usize::from(width) - 4;
    let (shown, spark) = layout(&COLUMNS, room);
    let position = if rows.len() > visible {
        format!(
            "  rows {}-{} of {}",
            first + 1,
            (first + visible).min(rows.len()),
            rows.len()
        )
    } else {
        String::new()
    };
    boxed(
        buffer,
        (0, TABLE_TOP, width, table_height),
        &[
            ("Instances", Style::fg(Color::Cyan).bold()),
            (&position, Style::fg(Color::Yellow)),
            (
                &format!(
                    "  tok/s over the last 3 polls; ratios and TTFT over {}",
                    app.window.label()
                ),
                Style::fg(Color::Dim),
            ),
        ],
        true,
    );
    let mut column = 2;
    for index in &shown {
        let spec = &COLUMNS[*index];
        let title = if spec.right {
            right(spec.title, spec.width)
        } else {
            pad(spec.title, spec.width)
        };
        column = buffer.put(column, TABLE_TOP + 1, &title, Style::fg(Color::Dim).bold())
            + spec.gap as u16;
    }
    if spark > 0 {
        buffer.put(
            column,
            TABLE_TOP + 1,
            &pad("DECODE 5m", spark),
            Style::fg(Color::Dim).bold(),
        );
    }
    if rows.is_empty() {
        buffer.put(
            2,
            TABLE_TOP + 2,
            "No instances yet. Discovery looks for chopinfer serve processes and containers; --target adds a server.",
            Style::fg(Color::Dim),
        );
    }
    for (line, (index, row)) in rows
        .iter()
        .enumerate()
        .skip(first)
        .take(visible)
        .enumerate()
    {
        draw_row(
            buffer,
            row,
            (index == selected, TABLE_TOP + 2 + line as u16),
            (&shown, spark),
            app,
            now,
        );
    }

    let summary_top = TABLE_TOP + table_height;
    let left = height - 1 - summary_top;
    let summary_height = if left >= 18 { 14 } else { 0 };
    if summary_height > 0
        && let Some(row) = rows.get(selected)
    {
        summary(
            buffer,
            app,
            row,
            (0, summary_top, width, summary_height),
            now,
        );
    }
    let events_top = summary_top + summary_height;
    events(
        buffer,
        app,
        (0, events_top, width, height - 1 - events_top),
        now,
    );
    footer(
        buffer,
        &[
            ("↑↓ jk", "select", 0),
            ("Enter", "details", 0),
            ("g", "gpus", 0),
            ("w", "window", 2),
            ("r", "rescan", 3),
            ("p", "pause", 1),
            ("?", "help", 0),
            ("q", "quit", 0),
        ],
    );
}

/// Draws `row` on line `y`, highlighted when `selected`, with the `shown`
/// columns and a sparkline `spark` cells wide.
fn draw_row(
    buffer: &mut Buffer,
    row: &Instance,
    (selected, y): (bool, u16),
    (shown, spark): (&[usize], usize),
    app: &App,
    now: f64,
) {
    let figures = Figures::new(row, app.window);
    let state = row.state(now);
    if selected {
        buffer.background(1, y, buffer.width - 2, Color::Selection);
        buffer.put(1, y, "▌", Style::fg(Color::Cyan));
    }
    let dim = Style::fg(Color::Dim);
    let plain = Style::PLAIN;
    let number = |value: Option<String>| match value {
        Some(text) => (text, plain),
        None => ("-".to_owned(), dim),
    };
    let (running, waiting) = (figures.running(), figures.waiting());
    let cells = [
        (sanitize(&row.source.label()), plain),
        match row.address() {
            Some(address) => (sanitize(&address), plain),
            None => ("-".to_owned(), dim),
        },
        match shown_name(row) {
            Some(name) if state.is_live() => (name, Style::fg(Color::White).bold()),
            Some(name) => (name, dim),
            None => ("-".to_owned(), dim),
        },
        (
            state.label().to_owned(),
            Style::fg(tone_color(state.tone())),
        ),
        number(figures.decode().map(integer)),
        number(figures.prefill().map(integer)),
        match (running, waiting) {
            (Some(running), Some(waiting)) => (
                format!("{}/{}", integer(running), integer(waiting)),
                if waiting > 0.0 {
                    Style::fg(Color::Yellow)
                } else {
                    plain
                },
            ),
            _ => ("-".to_owned(), dim),
        },
        match figures.kv() {
            Some(kv) => (
                percent(Some(kv)),
                if kv > 0.9 {
                    Style::fg(Color::Red)
                } else {
                    plain
                },
            ),
            None => ("-".to_owned(), dim),
        },
        number(figures.token_hits().map(|ratio| percent(Some(ratio)))),
        number(figures.acceptance().map(|ratio| percent(Some(ratio)))),
        number(
            figures
                .quantile(TTFT, 0.5)
                .map(|seconds| latency(Some(seconds))),
        ),
    ];
    let mut column = 2;
    for index in shown {
        let spec = &COLUMNS[*index];
        let (text, style) = &cells[*index];
        let text = if spec.right {
            right(text, spec.width)
        } else {
            pad(text, spec.width)
        };
        column = buffer.put(column, y, &text, *style) + spec.gap as u16;
    }
    if spark == 0 {
        return;
    }
    match &state {
        State::Starting | State::NotAnswering => {
            let note = row
                .uptime(now)
                .map(|seconds| format!("up {}", age(seconds)));
            buffer.put(column, y, &pad(&note.unwrap_or_default(), spark), dim);
        }
        State::Unreachable(_) | State::Connecting | State::NotChopinfer => {
            if matches!(row.source, Source::Target(_)) {
                buffer.put(column, y, &pad("retrying", spark), dim);
            }
        }
        _ => {
            let series = row.stats.rate_series(|s| s.counters.generation_tokens);
            let columns = buckets(
                &series,
                row.stats.oldest_poll(),
                row.stats.capacity().saturating_sub(1),
                spark,
            );
            sparkline(
                buffer,
                (column, y, spark as u16, 1),
                &columns,
                None,
                Color::Cyan,
            );
        }
    }
}

fn summary(buffer: &mut Buffer, app: &App, row: &Instance, area: (u16, u16, u16, u16), now: f64) {
    let (x, y, width, height) = area;
    let name = shown_name(row).unwrap_or_else(|| row.address().unwrap_or_default());
    let state = row.state(now);
    let status = match &state {
        State::NotReady(reason) => format!("  not ready: {}", sanitize(reason)),
        _ => String::new(),
    };
    boxed(
        buffer,
        area,
        &[
            (
                &sanitize(&row.source.label()),
                Style::fg(Color::Cyan).bold(),
            ),
            (&format!("  {name}"), Style::fg(Color::White).bold()),
            (&status, Style::fg(Color::Yellow)),
            ("  Enter for details", Style::fg(Color::Dim)),
        ],
        false,
    );
    let figures = Figures::new(row, app.window);
    let dim = Style::fg(Color::Dim);
    if !figures.live() {
        status_lines(buffer, row, &state, (x + 2, y + 2), now);
        return;
    }
    let half = (width - 6) / 2;
    let plot_height = height.saturating_sub(8).max(1);
    for (offset, label, rate, series, color) in [
        (
            0,
            "DECODE ",
            figures.decode(),
            row.stats.rate_series(|s| s.counters.generation_tokens),
            Color::Cyan,
        ),
        (
            half + 2,
            "PREFILL",
            figures.prefill(),
            row.stats.rate_series(|s| s.counters.prefill_tokens),
            Color::Magenta,
        ),
    ] {
        let column = x + 2 + offset;
        let recent = &series[series.len().saturating_sub(usize::from(half))..];
        let average =
            (!recent.is_empty()).then(|| recent.iter().sum::<f64>() / recent.len() as f64);
        let maximum = recent.iter().copied().fold(0.0_f64, f64::max);
        let rate_text = right(&rate.map(integer).unwrap_or_else(|| "-".into()), 6);
        let average_text = right(&average.map(integer).unwrap_or_else(|| "-".into()), 6);
        let maximum_text = right(&integer(maximum), 6);
        let groups: [&[(&str, Style)]; 4] = [
            &[
                (label, dim.bold()),
                ("  now ", dim),
                (&rate_text, Style::fg(Color::White).bold()),
            ],
            &[("  avg ", dim), (&average_text, Style::PLAIN)],
            &[("  max ", dim), (&maximum_text, Style::PLAIN)],
            &[("  tok/s", dim)],
        ];
        // Whole groups while they fit, so none runs into the other plot.
        let mut end = column;
        for group in groups {
            let length: u16 = group
                .iter()
                .map(|(text, _)| text.chars().count() as u16)
                .sum();
            if end - column + length > half {
                break;
            }
            end = buffer.pieces(end, y + 1, group);
        }
        // A blank row above the plot, but for the scale at its right.
        let scale = nice_ceiling(maximum);
        if scale > 0.0 {
            buffer.put(
                column,
                y + 2,
                &right(
                    &format!("{} tok/s", ceiling_label(scale)),
                    usize::from(half),
                ),
                dim,
            );
        }
        sparkline(
            buffer,
            (column, y + 3, half, plot_height),
            recent,
            Some(scale),
            color,
        );
        // The bars stand on a rule between the span's ends.
        let since = format!("-{} ", age(f64::from(half) * app.interval));
        let rule = usize::from(half).saturating_sub(since.chars().count() + 4);
        buffer.put(
            column,
            y + 3 + plot_height,
            &format!("{since}{} now", "▔".repeat(rule)),
            dim,
        );
    }
    let line = y + 4 + plot_height;
    let left = x + 2;
    let middle = x + 2 + half + 2;
    let gauge_width = half.saturating_sub(12);
    // Room for a value on the left and on the right of a line.
    let left_room = usize::from(half.saturating_sub(12));
    let right_room = usize::from((x + width - 2).saturating_sub(middle + 12));
    let running = figures.running().unwrap_or(0.0);
    let capacity = figures
        .max_sequences()
        .map_or(0.0, |maximum| maximum as f64);
    buffer.put(left, line, "sequences", dim);
    gauge(
        buffer,
        left + 12,
        line,
        gauge_width,
        if capacity > 0.0 {
            running / capacity
        } else {
            0.0
        },
        Color::Blue,
        &format!(
            "{:>5} run",
            format!(
                "{}/{}",
                integer(running),
                if capacity > 0.0 {
                    integer(capacity)
                } else {
                    "-".into()
                }
            )
        ),
    );
    buffer.put(middle, line, "KV cache", dim);
    let kv = figures.kv().unwrap_or(0.0);
    gauge(
        buffer,
        middle + 12,
        line,
        gauge_width,
        kv,
        if kv > 0.9 { Color::Red } else { Color::Green },
        &format!("{:>4} used", percent(Some(kv))),
    );
    let waiting = figures.waiting().unwrap_or(0.0);
    buffer.put(left, line + 1, "waiting", dim);
    buffer.put(
        left + 12,
        line + 1,
        &integer(waiting),
        if waiting > 0.0 {
            Style::fg(Color::Yellow)
        } else {
            Style::PLAIN
        },
    );
    buffer.put(middle, line + 1, "prefix", dim);
    match (figures.token_hits(), figures.lookup_hits()) {
        (Some(tokens), lookups) => buffer.put(
            middle + 12,
            line + 1,
            &fit(
                &[
                    format!(
                        "{} of prompt tokens reused, {} of lookups hit",
                        percent(Some(tokens)),
                        percent(lookups)
                    ),
                    format!("{} of prompt tokens reused", percent(Some(tokens))),
                ],
                right_room,
            ),
            Style::PLAIN,
        ),
        (None, _) => buffer.put(
            middle + 12,
            line + 1,
            &fit(&["no lookups in the window".to_owned()], right_room),
            dim,
        ),
    };
    buffer.put(left, line + 2, "latency", dim);
    let ttft = figures.quantile(TTFT, 0.5);
    if ttft.is_some() {
        let itl = figures.quantile(INTER_TOKEN, 0.5);
        buffer.put(
            left + 12,
            line + 2,
            &fit(
                &[
                    format!(
                        "TTFT p50 {}  p99 {}   ITL p50 {}  p99 {}",
                        latency(ttft),
                        latency(figures.quantile(TTFT, 0.99)),
                        latency(itl),
                        latency(figures.quantile(INTER_TOKEN, 0.99)),
                    ),
                    format!("TTFT p50 {}   ITL p50 {}", latency(ttft), latency(itl)),
                ],
                left_room,
            ),
            Style::PLAIN,
        );
    } else {
        buffer.put(
            left + 12,
            line + 2,
            &fit(&["no requests in the window".to_owned()], left_room),
            dim,
        );
    }
    buffer.put(middle, line + 2, "spec", dim);
    let speculation = row
        .health
        .as_ref()
        .and_then(|health| health.serving.as_ref())
        .and_then(|serving| serving.profile.speculation.clone());
    let forms = match speculation {
        Some(speculation) if speculation.mode != "off" => {
            let drafter = speculation
                .drafter
                .map(|drafter| format!(" {}", sanitize(&drafter)))
                .unwrap_or_default();
            let declared = format!(
                "{} K<={}{drafter}",
                sanitize(&speculation.mode),
                speculation.tokens
            );
            match figures.acceptance() {
                Some(acceptance) => vec![
                    format!(
                        "{declared}, {} of drafts accepted",
                        percent(Some(acceptance))
                    ),
                    format!("{declared}, {} accepted", percent(Some(acceptance))),
                    declared,
                ],
                None => vec![format!("{declared}, no drafts in the window"), declared],
            }
        }
        Some(_) => vec!["off".to_owned()],
        None => vec!["-".to_owned()],
    };
    buffer.put(
        middle + 12,
        line + 2,
        &fit(&forms, right_room),
        Style::PLAIN,
    );
}

/// The first of `forms`, longest first, that fits `room` cells; else the
/// last cut to fit.
fn fit(forms: &[String], room: usize) -> String {
    forms
        .iter()
        .find(|form| form.chars().count() <= room)
        .cloned()
        .unwrap_or_else(|| {
            forms
                .last()
                .map(|form| form.chars().take(room).collect())
                .unwrap_or_default()
        })
}

/// What to say about a row that is not answering.
pub fn status_lines(buffer: &mut Buffer, row: &Instance, state: &State, at: (u16, u16), now: f64) {
    let (x, y) = at;
    let dim = Style::fg(Color::Dim);
    let up = row
        .uptime(now)
        .map(|seconds| format!(" (up {})", age(seconds)))
        .unwrap_or_default();
    let error = |row: &Instance| sanitize(row.error.as_deref().unwrap_or("no answer yet"));
    let (first, second, color) = match (state, &row.source) {
        (State::Starting, Source::Container { .. }) => (
            format!("The container runs serve{up} but does not answer yet: {}.", error(row)),
            "serve opens its listener after loading and warmup; this row fills in when /health answers.".to_owned(),
            Color::Yellow,
        ),
        (State::Starting | State::NotAnswering, _) if row.endpoints.is_empty() => (
            format!("The serve process is running{up} but has no listening socket yet."),
            "It is loading weights or warming up; metrics appear once /health answers.".to_owned(),
            Color::Yellow,
        ),
        (State::Starting, _) => (
            format!("The serve process is listening{up}; waiting for /health."),
            String::new(),
            Color::Yellow,
        ),
        (State::NotAnswering, _) => (
            format!("The serve process has listened for over 30 s without a valid answer: {}.", error(row)),
            String::new(),
            Color::Yellow,
        ),
        (State::Connecting, _) => ("Not polled yet.".to_owned(), String::new(), Color::Yellow),
        (State::NotChopinfer, _) => (
            format!("{} answers without the chopinfer startup nonce header.", sanitize(&row.address().unwrap_or_default())),
            "It is not a chopinfer server, or a proxy strips the x-chopinfer-startup-nonce header.".to_owned(),
            Color::Red,
        ),
        _ => (
            format!("GET http://{}/health failed: {}.", sanitize(&row.address().unwrap_or_default()), error(row)),
            match row.source {
                Source::Target(_) => "Explicit targets stay listed and are retried every 5 s.".to_owned(),
                _ => "Discovered rows are dropped when their process or container goes away.".to_owned(),
            },
            Color::Red,
        ),
    };
    // The boxes these lines sit in end one column before the right edge.
    let room = usize::from(buffer.width.saturating_sub(x + 2));
    buffer.put(x, y, &pad(&first, room), Style::fg(color));
    buffer.put(x, y + 1, &pad(&second, room), dim);
    if !row.arguments.is_empty() {
        let command = sanitize(&row.arguments.join(" "));
        buffer.put(
            x,
            y + 3,
            &pad(&format!("Command line:  {command}"), room),
            dim,
        );
    }
}

fn events(buffer: &mut Buffer, app: &App, area: (u16, u16, u16, u16), now: f64) {
    let (x, y, width, height) = area;
    if height < 3 {
        return;
    }
    boxed(
        buffer,
        area,
        &[("Events", Style::fg(Color::Cyan).bold())],
        false,
    );
    let room = usize::from(height - 2);
    let who_width = usize::from(width / 4).min(26);
    let events = &app.registry.events;
    for (index, event) in events
        .iter()
        .skip(events.len().saturating_sub(room))
        .enumerate()
    {
        let line = y + 1 + index as u16;
        let when = format!("{} ago", age(now - event.at));
        let column = buffer.put(x + 2, line, &right(&when, 10), Style::fg(Color::Dim));
        let column = buffer.put(
            column + 2,
            line,
            &pad(&sanitize(&event.who), who_width),
            Style::PLAIN,
        );
        let room = usize::from(width).saturating_sub(usize::from(column) + 2);
        buffer.put(
            column,
            line,
            &pad(&sanitize(&event.what), room),
            Style::fg(tone_color(event.tone)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_terminals_drop_columns_from_the_right() {
        let (all, spark) = layout(&COLUMNS, 128);
        assert_eq!(all.len(), COLUMNS.len());
        assert!(spark >= SPARK_MINIMUM, "{spark}");
        let titles = |shown: Vec<usize>| {
            shown
                .into_iter()
                .map(|index| COLUMNS[index].title)
                .collect::<Vec<_>>()
        };
        // An 80-column terminal keeps decode throughput.
        let (shown, spark) = layout(&COLUMNS, 76);
        assert_eq!(spark, 0);
        assert_eq!(
            titles(shown),
            ["SOURCE", "ADDRESS", "MODEL", "STATE", "DECODE"]
        );
        let (shown, _) = layout(&COLUMNS, 70);
        assert_eq!(titles(shown), ["SOURCE", "ADDRESS", "MODEL", "STATE"]);
        let (shown, spark) = layout(&COLUMNS, 1020);
        assert_eq!((shown.len(), spark), (COLUMNS.len(), 1020 - 114));
        let (shown, spark) = layout(&COLUMNS, 100);
        assert_eq!(spark, 0);
        assert_eq!(
            titles(shown),
            [
                "SOURCE", "ADDRESS", "MODEL", "STATE", "DECODE", "PREFILL", "RUN/WAIT", "KV"
            ]
        );
    }

    #[test]
    fn the_viewport_follows_the_selection() {
        // 80x24 shows 13 of 30 rows.
        assert_eq!(table_rows(30, 24), 13);
        assert_eq!(table_rows(3, 24), 3);
        assert_eq!(first_row(0, 5, 13), 0);
        // Moving below the last shown row scrolls by one.
        assert_eq!(first_row(0, 13, 13), 1);
        assert_eq!(first_row(1, 20, 13), 8);
        // Moving back up keeps the viewport until the selection leaves it.
        assert_eq!(first_row(8, 12, 13), 8);
        assert_eq!(first_row(8, 3, 13), 3);
    }
}
