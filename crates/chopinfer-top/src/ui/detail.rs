//! The detail view of one instance.

use super::buffer::{Buffer, Color, Style};
use super::list::status_lines;
use super::widgets::{
    age, bar, boxed, buckets, bytes, count, gauge, integer, latency, pad, percent, rate, right,
    sanitize, sparkline,
};
use super::{Figures, LATENCIES, footer, header, shown_name, tone_color};
use crate::app::App;
use crate::gpu::ranges;
use crate::instances::{Instance, Source, State};
use crate::stats::{HistogramOf, Snapshot};

/// A panel of the detail view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Panel {
    Throughput,
    Load,
    Latency,
    Requests,
    Health,
    Prefix,
    Speculation,
    Steps,
    Identity,
}

/// The fewest lines the throughput row keeps as other rows are added: its
/// height at 100x32, the smallest terminal that shows every panel.
const THROUGHPUT_MINIMUM: u16 = 7;

pub fn draw(buffer: &mut Buffer, app: &App, now: f64) {
    let rows = app.rows();
    let Some(row) = app.current() else {
        header(buffer, app, &[]);
        buffer.put(
            2,
            2,
            "No instance to show; Esc returns to the list.",
            Style::fg(Color::Dim),
        );
        return;
    };
    let state = row.state(now);
    let name = shown_name(row).unwrap_or_else(|| row.address().unwrap_or_default());
    let mut crumbs = vec![
        ("  >  ".to_owned(), Style::fg(Color::Dim)),
        (name, Style::fg(Color::White).bold()),
        (
            format!("  [{}/{}]  ", app.selected_index() + 1, rows.len()),
            Style::fg(Color::Dim),
        ),
        (format!("{}  ", sanitize(&row.source.label())), Style::PLAIN),
    ];
    if let Some(address) = row.address() {
        crumbs.push((format!("{}  ", sanitize(&address)), Style::PLAIN));
    }
    crumbs.push((
        match &state {
            State::NotReady(reason) => format!("not ready: {}", sanitize(reason)),
            state => state.label().to_owned(),
        },
        Style::fg(tone_color(state.tone())).bold(),
    ));
    if let Some(seconds) = row.uptime(now) {
        crumbs.push((format!("  up {}", age(seconds)), Style::fg(Color::Dim)));
    }
    header(buffer, app, &crumbs);
    footer(
        buffer,
        &[
            ("Esc", "back", 0),
            ("← → Tab", "instance", 0),
            ("g", "gpus", 0),
            ("w", "window", 2),
            ("r", "rescan", 3),
            ("p", "pause", 1),
            ("?", "help", 0),
            ("q", "quit", 0),
        ],
    );
    let (width, height) = (buffer.width, buffer.height);
    let figures = Figures::new(row, app.window);
    let gpus = app.gpus.of_instance(&app.registry.instances, row.id);
    if !figures.live() {
        boxed(
            buffer,
            (0, 1, width, 8),
            &[("Status", Style::fg(Color::Cyan).bold())],
            true,
        );
        status_lines(buffer, row, &state, (2, 3), now);
        identity(
            buffer,
            row,
            &gpus,
            (0, 9, width, height.saturating_sub(10).min(6)),
        );
        return;
    }
    let mut y = 1;
    for (row_height, panels) in layout(width, height - 2) {
        let mut x = 0;
        for (panel, panel_width) in panels {
            let area = (x, y, panel_width, row_height);
            match panel {
                Panel::Throughput => throughput(buffer, row, area),
                Panel::Load => load(buffer, row, &figures, area),
                Panel::Latency => latencies(buffer, &figures, area),
                Panel::Requests => requests(buffer, row, area),
                Panel::Health => health(buffer, row, &state, area, now),
                Panel::Prefix => prefix(buffer, row, &figures, area),
                Panel::Speculation => speculation(buffer, row, &figures, area),
                Panel::Steps => steps(buffer, row, &figures, area),
                Panel::Identity => identity(buffer, row, &gpus, area),
            }
            x += panel_width;
        }
        y += row_height;
    }
}

/// The rows of panels that fit `width` x `height` cells: each row's height
/// and its panels' widths. Width picks the arrangement. Rows are added in
/// order, the least needed last, while the throughput row keeps
/// [`THROUGHPUT_MINIMUM`] lines, and it takes the height left.
fn layout(width: u16, height: u16) -> Vec<(u16, Vec<(Panel, u16)>)> {
    // The last panel of a row takes the rest of its width (0 here).
    let rows: Vec<(u16, Vec<(Panel, u16)>)> = if width >= 100 {
        vec![
            (
                0,
                vec![(Panel::Throughput, share(width, 64)), (Panel::Load, 0)],
            ),
            (
                9,
                vec![
                    (Panel::Latency, share(width, 48)),
                    (Panel::Requests, share(width, 26)),
                    (Panel::Health, 0),
                ],
            ),
            (
                9,
                vec![
                    (Panel::Prefix, width / 3),
                    (Panel::Speculation, width / 3),
                    (Panel::Steps, 0),
                ],
            ),
            (5, vec![(Panel::Identity, 0)]),
        ]
    } else if width >= 80 {
        vec![
            (
                0,
                vec![(Panel::Throughput, share(width, 64)), (Panel::Load, 0)],
            ),
            (
                9,
                vec![(Panel::Latency, share(width, 58)), (Panel::Health, 0)],
            ),
            (9, vec![(Panel::Requests, width / 2), (Panel::Steps, 0)]),
            (9, vec![(Panel::Prefix, width / 2), (Panel::Speculation, 0)]),
            (5, vec![(Panel::Identity, 0)]),
        ]
    } else {
        [
            (0, Panel::Throughput),
            (7, Panel::Load),
            (9, Panel::Latency),
            (9, Panel::Health),
            (9, Panel::Requests),
            (9, Panel::Steps),
            (9, Panel::Prefix),
            (9, Panel::Speculation),
            (5, Panel::Identity),
        ]
        .into_iter()
        .map(|(row_height, panel)| (row_height, vec![(panel, 0)]))
        .collect()
    };
    let mut used = THROUGHPUT_MINIMUM;
    let mut shown = Vec::new();
    for (row_height, mut panels) in rows {
        if used + row_height > height {
            break;
        }
        used += row_height;
        let given: u16 = panels[..panels.len() - 1]
            .iter()
            .map(|(_, cells)| cells)
            .sum();
        if let Some(last) = panels.last_mut() {
            last.1 = width - given;
        }
        shown.push((row_height, panels));
    }
    if let Some(first) = shown.first_mut() {
        first.0 = height - (used - THROUGHPUT_MINIMUM);
    }
    shown
}

/// `percent` of `total` cells, computed wide so a large terminal does not
/// overflow.
fn share(total: u16, percent: u16) -> u16 {
    u16::try_from(u32::from(total) * u32::from(percent) / 100).unwrap_or(total)
}

fn dim() -> Style {
    Style::fg(Color::Dim)
}

fn title(text: &str) -> (&str, Style) {
    (text, Style::fg(Color::Cyan).bold())
}

/// A `label  value` line with the value right-aligned to `width`.
fn pair(buffer: &mut Buffer, x: u16, y: u16, width: u16, label: &str, value: &str, style: Style) {
    let label_width = usize::from(width).saturating_sub(value.chars().count() + 1);
    buffer.put(x, y, &pad(label, label_width), dim());
    buffer.put(
        x + label_width as u16,
        y,
        &right(value, usize::from(width) - label_width),
        style,
    );
}

fn throughput(buffer: &mut Buffer, row: &Instance, area: (u16, u16, u16, u16)) {
    let (x, y, width, height) = area;
    boxed(
        buffer,
        area,
        &[title("Throughput"), ("  tok/s, last 5m", dim())],
        true,
    );
    let plot = (height.saturating_sub(4) / 2).max(1);
    let plot_width = width - 4;
    for (index, (label, value)) in [
        (
            "DECODE",
            (|s: &Snapshot| s.counters.generation_tokens) as fn(&Snapshot) -> f64,
        ),
        ("PREFILL", |s: &Snapshot| s.counters.prefill_tokens),
    ]
    .into_iter()
    .enumerate()
    {
        let top = y + 1 + index as u16 * (plot + 1);
        let series = row.stats.rate_series(value);
        let now = row.stats.rate(value);
        let average =
            (!series.is_empty()).then(|| series.iter().sum::<f64>() / series.len() as f64);
        let maximum = series.iter().copied().fold(0.0_f64, f64::max);
        buffer.pieces(
            x + 2,
            top,
            &[
                (&pad(label, 8), dim().bold()),
                ("now ", dim()),
                (
                    &right(&now.map(integer).unwrap_or_else(|| "-".into()), 6),
                    Style::fg(Color::White).bold(),
                ),
                ("  avg ", dim()),
                (
                    &right(&average.map(integer).unwrap_or_else(|| "-".into()), 6),
                    Style::PLAIN,
                ),
                ("  max ", dim()),
                (&right(&integer(maximum), 6), Style::PLAIN),
            ],
        );
        let color = if index == 0 {
            Color::Cyan
        } else {
            Color::Magenta
        };
        let columns = buckets(
            &series,
            row.stats.oldest_poll(),
            row.stats.capacity().saturating_sub(1),
            usize::from(plot_width),
        );
        sparkline(
            buffer,
            (x + 2, top + 1, plot_width, plot),
            &columns,
            None,
            color,
        );
    }
}

fn load(buffer: &mut Buffer, row: &Instance, figures: &Figures, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    boxed(buffer, area, &[title("Load")], false);
    let column = x + 13;
    let gauge_width = width.saturating_sub(15);
    let running = figures.running().unwrap_or(0.0);
    let capacity = figures
        .max_sequences()
        .map_or(0.0, |maximum| maximum as f64);
    buffer.put(x + 2, y + 1, "sequences", dim());
    gauge(
        buffer,
        column,
        y + 1,
        gauge_width,
        if capacity > 0.0 {
            running / capacity
        } else {
            0.0
        },
        Color::Blue,
        &format!(
            "{:>5}",
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
    let kv = figures.kv().unwrap_or(0.0);
    buffer.put(x + 2, y + 2, "KV cache", dim());
    gauge(
        buffer,
        column,
        y + 2,
        gauge_width,
        kv,
        if kv > 0.9 { Color::Red } else { Color::Green },
        &format!("{:>5}", percent(Some(kv))),
    );
    let gauges = figures.latest().map(|s| &s.gauges);
    let intake = gauges.and_then(|g| Some((g.intake_used?, g.intake_budget?)));
    buffer.put(x + 2, y + 3, "intake", dim());
    match intake {
        Some((used, budget)) if budget > 0.0 => gauge(
            buffer,
            column,
            y + 3,
            gauge_width,
            used / budget,
            Color::Blue,
            &format!("{:>5}", percent(Some(used / budget))),
        ),
        _ => {
            buffer.put(column, y + 3, "-", dim());
        }
    }
    let waiting = figures.waiting().unwrap_or(0.0);
    buffer.put(x + 2, y + 5, "waiting", dim());
    buffer.put(
        column,
        y + 5,
        &integer(waiting),
        if waiting > 0.0 {
            Style::fg(Color::Yellow).bold()
        } else {
            Style::PLAIN
        },
    );
    // Stage lines end above requests/s, or above the bottom border.
    let requests_shown = area.3 > 10;
    let last = if requests_shown {
        y + 8
    } else {
        (y + area.3).saturating_sub(2)
    };
    if let Some(health) = &row.health
        && !health.waiting_stages.is_empty()
        && y + 6 <= last
    {
        buffer.put(x + 2, y + 6, "  by stage", dim());
        let stages = health
            .waiting_stages
            .iter()
            .map(|(stage, count)| format!("{} {count}", sanitize(stage)))
            .collect::<Vec<_>>();
        let room = usize::from(width.saturating_sub(15));
        let mut lines = vec![String::new()];
        for stage in stages {
            let text = lines.last_mut().expect("one line");
            if !text.is_empty() && text.len() + 2 + stage.len() > room {
                lines.push(stage);
                continue;
            }
            if !text.is_empty() {
                text.push_str("  ");
            }
            text.push_str(&stage);
        }
        for (line, text) in (y + 6..=last).zip(&lines) {
            buffer.put(column, line, &pad(text, room), Style::PLAIN);
        }
    }
    let finished = row
        .stats
        .rate(|s| s.counters.finished.iter().map(|(_, value)| value).sum());
    if requests_shown {
        buffer.put(x + 2, y + 9, "requests/s", dim());
        buffer.put(
            column,
            y + 9,
            &finished.map(rate).unwrap_or_else(|| "-".into()),
            Style::PLAIN,
        );
    }
}

/// The latency table's label width and value column width inside `inner`
/// cells: a 15-cell label and four 9-cell columns when they fit, narrower
/// columns down to 6 cells, then a shorter label.
fn latency_columns(inner: usize) -> (usize, usize) {
    let column = (inner.saturating_sub(14) / 4).clamp(6, 9);
    (inner.saturating_sub(column * 4).min(15), column)
}

fn latencies(buffer: &mut Buffer, figures: &Figures, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    let label = figures.window_label();
    boxed(
        buffer,
        area,
        &[title("Latency"), (&format!("  {label}"), dim())],
        false,
    );
    let (label_width, column) = latency_columns(usize::from(width.saturating_sub(4)));
    buffer.put(
        x + 2,
        y + 1,
        &format!(
            "{}{}{}{}{}",
            pad("", label_width),
            right("p50", column),
            right("p90", column),
            right("p99", column),
            right("count", column)
        ),
        dim().bold(),
    );
    for (index, (label, histogram)) in LATENCIES.iter().enumerate() {
        let line = y + 2 + index as u16;
        let window = figures.instance.stats.histogram(figures.window, *histogram);
        let total = window.as_ref().map_or(0.0, |histogram| {
            histogram.buckets.last().map_or(0.0, |bucket| bucket.1)
        });
        let values = [0.5, 0.9, 0.99].map(|q| figures.quantile(*histogram, q));
        let style = if values[0].is_some() {
            Style::PLAIN
        } else {
            dim()
        };
        let text = values
            .iter()
            .map(|value| right(&latency(*value), column))
            .collect::<String>();
        buffer.pieces(
            x + 2,
            line,
            &[
                (&pad(label, label_width), dim()),
                (&text, style),
                (&right(&count(total, column - 1), column), dim()),
            ],
        );
    }
}

fn requests(buffer: &mut Buffer, row: &Instance, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    boxed(
        buffer,
        area,
        &[title("Requests"), ("  since start", dim())],
        false,
    );
    let Some(latest) = row.stats.latest() else {
        return;
    };
    let counters = &latest.counters;
    let finished = |reason: &str| {
        counters
            .finished
            .iter()
            .find(|(seen, _)| seen == reason)
            .map_or(0.0, |(_, value)| *value)
    };
    let refused: f64 = counters.refusals.iter().map(|(_, value)| value).sum();
    let reasons = counters
        .refusals
        .iter()
        .filter(|(_, value)| *value > 0.0)
        .map(|(reason, value)| format!("{} {}", sanitize(reason), integer(*value)))
        .collect::<Vec<_>>()
        .join(", ");
    let inner = width - 4;
    let lines = [
        ("finished stop", finished("stop"), Color::Default),
        ("finished length", finished("length"), Color::Default),
        ("aborted", finished("abort"), Color::Default),
        ("cancellations", counters.cancellations, Color::Default),
        ("preemptions", counters.preemptions, Color::Default),
        (
            "engine errors",
            counters.errors,
            if counters.errors > 0.0 {
                Color::Red
            } else {
                Color::Default
            },
        ),
        (
            "refused",
            refused,
            if refused > 0.0 {
                Color::Yellow
            } else {
                Color::Default
            },
        ),
    ];
    for (index, (label, value, color)) in lines.into_iter().enumerate() {
        pair(
            buffer,
            x + 2,
            y + 1 + index as u16,
            inner,
            label,
            &integer(value),
            Style::fg(color),
        );
    }
    if !reasons.is_empty() && area.3 > 9 {
        buffer.put(
            x + 2,
            y + 8,
            &pad(&format!("  {reasons}"), usize::from(inner)),
            dim(),
        );
    }
}

fn health(
    buffer: &mut Buffer,
    row: &Instance,
    state: &State,
    area: (u16, u16, u16, u16),
    now: f64,
) {
    let (x, y, width, _) = area;
    let status = match state {
        State::NotReady(reason) => format!("  not ready: {}", sanitize(reason)),
        _ => String::new(),
    };
    boxed(
        buffer,
        area,
        &[title("Health"), (&status, Style::fg(Color::Yellow))],
        false,
    );
    let inner = width - 4;
    let gauges = row.stats.latest().map(|s| &s.gauges);
    let counters = row.stats.latest().map(|s| &s.counters);
    let worker = gauges.and_then(|g| g.worker).unwrap_or("-");
    let poisoned = gauges.and_then(|g| g.poisoned);
    let transport = gauges.and_then(|g| Some((g.transport_reserved?, g.transport_budget?)));
    let version = row
        .health
        .as_ref()
        .and_then(|health| health.serving.as_ref())
        .and_then(|serving| serving.version.clone())
        .map_or_else(|| "-".to_owned(), |version| sanitize(&version));
    let restarts = match row.last_restart {
        Some(at) => format!("{} ({} ago)", row.restarts, age(now - at)),
        None => "0".to_owned(),
    };
    let lines = [
        (
            "worker",
            worker.to_owned(),
            match worker {
                "ready" => Color::Green,
                "failed" => Color::Red,
                _ => Color::Yellow,
            },
        ),
        (
            "poisoned",
            match poisoned {
                Some(true) => "yes".to_owned(),
                Some(false) => "no".to_owned(),
                None => "-".to_owned(),
            },
            if poisoned == Some(true) {
                Color::Red
            } else {
                Color::Green
            },
        ),
        (
            "batch refusals",
            counters.map_or_else(|| "-".to_owned(), |c| integer(c.batch_refusals)),
            Color::Default,
        ),
        (
            "restarts seen",
            restarts,
            if row.restarts > 0 {
                Color::Yellow
            } else {
                Color::Default
            },
        ),
        (
            "transport buf",
            transport.map_or_else(
                || "-".to_owned(),
                |(used, budget)| format!("{}/{}", bytes(used), bytes(budget)),
            ),
            Color::Default,
        ),
        ("version", version, Color::Default),
        (
            "metrics",
            row.problem
                .as_deref()
                .map_or_else(|| "read".to_owned(), sanitize),
            if row.problem.is_some() {
                Color::Yellow
            } else {
                Color::Default
            },
        ),
    ];
    for (index, (label, value, color)) in lines.into_iter().enumerate() {
        let line = y + 1 + index as u16;
        buffer.put(x + 2, line, &pad(label, 15), dim());
        buffer.put(
            x + 17,
            line,
            &pad(&value, usize::from(inner.saturating_sub(15))),
            Style::fg(color),
        );
    }
}

fn prefix(buffer: &mut Buffer, row: &Instance, figures: &Figures, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    boxed(
        buffer,
        area,
        &[
            title("Prefix cache"),
            (&format!("  {}", figures.window.label()), dim()),
        ],
        false,
    );
    let inner = width - 4;
    let latest = figures.latest();
    let gauge_pair = latest.and_then(|s| Some((s.gauges.prefix_anchors?, s.gauges.prefix_blocks?)));
    let failures = latest.map(|s| {
        (
            s.counters.prefix_capture_failures,
            s.counters.prefix_restore_failures,
        )
    });
    let lines = [
        ("reused / queried tok", percent(figures.token_hits())),
        ("lookups that hit", percent(figures.lookup_hits())),
        (
            "reused tok/s",
            row.stats
                .rate(|s| s.counters.prefix_hit_tokens)
                .map_or_else(|| "-".into(), integer),
        ),
        (
            "recompute seen tok/s",
            row.stats
                .rate(|s| s.counters.prefix_recomputed_seen_tokens)
                .map_or_else(|| "-".into(), integer),
        ),
        (
            "anchors / pages",
            gauge_pair.map_or_else(
                || "-".into(),
                |(anchors, pages)| format!("{} / {}", integer(anchors), integer(pages)),
            ),
        ),
        (
            "evictions/s",
            row.stats
                .rate(|s| s.counters.prefix_evictions)
                .map_or_else(|| "-".into(), rate),
        ),
        (
            "capture/restore fail",
            failures.map_or_else(
                || "-".into(),
                |(capture, restore)| format!("{} / {}", integer(capture), integer(restore)),
            ),
        ),
    ];
    for (index, (label, value)) in lines.into_iter().enumerate() {
        pair(
            buffer,
            x + 2,
            y + 1 + index as u16,
            inner,
            label,
            &value,
            Style::PLAIN,
        );
    }
}

fn speculation(buffer: &mut Buffer, row: &Instance, figures: &Figures, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    let declared = row
        .health
        .as_ref()
        .and_then(|health| health.serving.as_ref())
        .and_then(|serving| serving.profile.speculation.clone());
    let subtitle = match &declared {
        Some(speculation) if speculation.mode != "off" => format!(
            "  {} K<={}{}",
            sanitize(&speculation.mode),
            speculation.tokens,
            speculation
                .drafter
                .as_deref()
                .map(|drafter| format!(" {}", sanitize(drafter)))
                .unwrap_or_default()
        ),
        _ => String::new(),
    };
    boxed(
        buffer,
        area,
        &[title("Speculation"), (&subtitle, dim())],
        false,
    );
    if declared
        .as_ref()
        .is_none_or(|speculation| speculation.mode == "off")
    {
        buffer.put(x + 2, y + 2, "off (autoregressive)", dim());
        return;
    }
    buffer.pieces(
        x + 2,
        y + 1,
        &[
            ("accepted drafts ", dim()),
            (
                &percent(figures.acceptance()),
                Style::fg(Color::White).bold(),
            ),
        ],
    );
    buffer.put(x + 2, y + 2, "by position", dim());
    let positions = row.stats.acceptance_by_position(figures.window);
    let column_width = (width - 4) / 2;
    let bar_width = usize::from(column_width.saturating_sub(9));
    for (index, value) in positions.iter().enumerate().take(8) {
        let line = y + 3 + (index % 4) as u16;
        let column = x + 2 + (index / 4) as u16 * column_width;
        buffer.pieces(
            column,
            line,
            &[
                (&format!("{index} "), dim()),
                (
                    &pad(&bar(value.unwrap_or(0.0), bar_width), bar_width),
                    Style::fg(Color::Cyan),
                ),
                (&right(&percent(*value), 5), Style::PLAIN),
            ],
        );
    }
    let shares = row
        .stats
        .speculation_k(figures.window)
        .into_iter()
        .map(|(k, share)| format!("{} {}", sanitize(&k), percent(Some(share))))
        .collect::<Vec<_>>()
        .join("  ");
    if !shares.is_empty() {
        buffer.put(
            x + 2,
            y + 7,
            &pad(&format!("K chosen: {shares}"), usize::from(width - 4)),
            dim(),
        );
    }
}

fn steps(buffer: &mut Buffer, row: &Instance, figures: &Figures, area: (u16, u16, u16, u16)) {
    let (x, y, width, _) = area;
    boxed(
        buffer,
        area,
        &[
            title("Steps"),
            (&format!("  {}", figures.window.label()), dim()),
        ],
        false,
    );
    buffer.pieces(
        x + 2,
        y + 1,
        &[
            ("batches/s ", dim()),
            (
                &row.stats
                    .rate(|s| s.counters.batches)
                    .map_or_else(|| "-".into(), rate),
                Style::PLAIN,
            ),
        ],
    );
    let window = figures.window;
    let shares = [
        (
            "decode",
            row.stats.delta(window, |s| s.counters.decode_only_batches),
            Color::Cyan,
        ),
        (
            "mixed",
            row.stats.delta(window, |s| s.counters.mixed_batches),
            Color::Blue,
        ),
        (
            "prefill",
            row.stats.delta(window, |s| s.counters.prefill_only_batches),
            Color::Magenta,
        ),
    ];
    let total: f64 = shares.iter().filter_map(|(_, value, _)| *value).sum();
    buffer.put(x + 2, y + 2, "mix", dim());
    if total > 0.0 {
        let room = f64::from(width - 10);
        let mut column = x + 6;
        for (_, value, color) in &shares {
            let cells = (value.unwrap_or(0.0) / total * room).round() as usize;
            column = buffer.put(column, y + 2, &"█".repeat(cells), Style::fg(*color));
        }
        // As many labels as fit inside the box.
        let room = usize::from(width.saturating_sub(8));
        let mut labels = String::new();
        for (name, value, _) in &shares {
            let label = format!("{name} {}", percent(Some(value.unwrap_or(0.0) / total)));
            let gap = if labels.is_empty() { "" } else { "  " };
            if labels.len() + gap.len() + label.len() > room {
                break;
            }
            labels.push_str(gap);
            labels.push_str(&label);
        }
        buffer.put(x + 6, y + 3, &labels, dim());
    } else {
        buffer.put(x + 6, y + 2, "no batches in the window", dim());
    }
    buffer.put(
        x + 2,
        y + 4,
        &format!(
            "{}{}{}",
            pad("step time", 12),
            right("p50", 9),
            right("p99", 9)
        ),
        dim().bold(),
    );
    let phases: [(&str, HistogramOf); 3] = [
        ("decode", |s| s.histograms.step_decode.as_ref()),
        ("mixed", |s| s.histograms.step_mixed.as_ref()),
        ("prefill", |s| s.histograms.step_prefill.as_ref()),
    ];
    for (index, (phase, histogram)) in phases.into_iter().enumerate() {
        let line = y + 5 + index as u16;
        buffer.put(x + 2, line, &pad(phase, 12), dim());
        let p50 = figures.quantile(histogram, 0.5);
        let p99 = figures.quantile(histogram, 0.99);
        buffer.put(
            x + 14,
            line,
            &format!("{}{}", right(&latency(p50), 9), right(&latency(p99), 9)),
            if p50.is_some() { Style::PLAIN } else { dim() },
        );
    }
}

/// The instance's identity; `gpus` are the GPUs of this host it holds.
fn identity(buffer: &mut Buffer, row: &Instance, gpus: &[u32], area: (u16, u16, u16, u16)) {
    let (x, y, width, height) = area;
    if height < 3 {
        return;
    }
    boxed(buffer, area, &[title("Identity")], false);
    let serving = row
        .health
        .as_ref()
        .and_then(|health| health.serving.as_ref());
    let text = |value: Option<&str>| value.map_or_else(|| "-".to_owned(), sanitize);
    let short = |value: Option<&str>| {
        value.map_or_else(
            || "-".to_owned(),
            |value| {
                format!(
                    "{}...",
                    sanitize(&value.chars().take(8).collect::<String>())
                )
            },
        )
    };
    let mut lines: Vec<Vec<(String, String)>> = Vec::new();
    if let Some(serving) = serving {
        let profile = &serving.profile;
        let ranks = match (profile.tensor_parallel_size, profile.context_parallel_size) {
            (Some(tp), Some(cp)) => format!("TP{tp} CP{cp}"),
            (Some(tp), None) => format!("TP{tp}"),
            _ => "-".to_owned(),
        };
        lines.push(vec![
            ("bundle".into(), text(serving.bundle_id.as_deref())),
            ("profile".into(), text(profile.name.as_deref())),
            ("ranks".into(), ranks),
            ("kv".into(), text(profile.kv_cache_dtype.as_deref())),
            (
                "max seq".into(),
                profile
                    .max_sequences
                    .map_or_else(|| "-".into(), |value| value.to_string()),
            ),
            (
                "page".into(),
                serving
                    .kv_page_tokens
                    .map_or_else(|| "-".into(), |value| format!("{value} tok")),
            ),
            (
                "prefix cache".into(),
                match profile.prefix_cache {
                    Some(true) => "on".into(),
                    Some(false) => "off".into(),
                    None => "-".into(),
                },
            ),
        ]);
        lines.push(vec![
            (
                "served as".into(),
                sanitize(&serving.served_model_names.join(", ")),
            ),
            ("version".into(), text(serving.version.as_deref())),
            ("source".into(), short(serving.source_commit.as_deref())),
            (
                "checkpoint".into(),
                short(serving.checkpoint_identity.as_deref()),
            ),
            (
                "startup nonce".into(),
                short(serving.startup_nonce.as_deref()),
            ),
        ]);
    }
    let at = row.address().map_or_else(
        || ", not listening yet".to_owned(),
        |address| format!(", at {}", sanitize(&address)),
    );
    let found = match &row.source {
        Source::Target(target) => format!("--target {}", sanitize(&target.display())),
        Source::Process { pid, .. } => format!("serve process {pid} of this user{at}"),
        Source::Container { name, .. } => format!("docker container {}{at}", sanitize(name)),
    };
    let mut via = vec![("found via".to_owned(), found)];
    if !gpus.is_empty() {
        via.push(("on GPUs".to_owned(), ranges(gpus)));
    }
    if !row.aliases.is_empty() {
        via.push(("also seen as".to_owned(), sanitize(&row.aliases.join(", "))));
    }
    lines.push(via);
    // Each line holds the pairs that fit whole; a first pair too long for
    // the line is cut at the box.
    let end = usize::from(x + width - 2);
    for (index, line) in lines.iter().enumerate().take(usize::from(height - 2)) {
        let mut column = usize::from(x + 2);
        for (position, (label, value)) in line.iter().enumerate() {
            let length = label.chars().count() + 1 + value.chars().count();
            if position > 0 && column + length > end {
                break;
            }
            let room = end.saturating_sub(column);
            let value = pad(value, room.saturating_sub(label.chars().count() + 1));
            buffer.pieces(
                column as u16,
                y + 1 + index as u16,
                &[
                    (&format!("{label} "), dim()),
                    (value.trim_end(), Style::PLAIN),
                ],
            );
            column += length + 3;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smaller_terminals_show_fewer_panels() {
        let panels = |width, height| {
            layout(width, height)
                .into_iter()
                .map(|(row_height, panels)| {
                    let cells: u16 = panels.iter().map(|(_, cells)| cells).sum();
                    assert_eq!(cells, width, "{width}x{height}");
                    (
                        row_height,
                        panels.into_iter().map(|(panel, _)| panel).collect(),
                    )
                })
                .collect::<Vec<(u16, Vec<Panel>)>>()
        };
        // 132x38: today's layout, every panel.
        assert_eq!(
            layout(132, 36),
            [
                (13, vec![(Panel::Throughput, 84), (Panel::Load, 48)]),
                (
                    9,
                    vec![
                        (Panel::Latency, 63),
                        (Panel::Requests, 34),
                        (Panel::Health, 35)
                    ]
                ),
                (
                    9,
                    vec![
                        (Panel::Prefix, 44),
                        (Panel::Speculation, 44),
                        (Panel::Steps, 44)
                    ]
                ),
                (5, vec![(Panel::Identity, 132)]),
            ]
        );
        // 80x24: two columns, the most needed rows.
        assert_eq!(
            panels(80, 22),
            [
                (13, vec![Panel::Throughput, Panel::Load]),
                (9, vec![Panel::Latency, Panel::Health]),
            ]
        );
        // 80x32 stops before Prefix and Speculation; the shorter Identity
        // row after them is not added either.
        assert_eq!(
            panels(80, 30),
            [
                (12, vec![Panel::Throughput, Panel::Load]),
                (9, vec![Panel::Latency, Panel::Health]),
                (9, vec![Panel::Requests, Panel::Steps]),
            ]
        );
        // 60x24: one column.
        assert_eq!(
            panels(60, 22),
            [(15, vec![Panel::Throughput]), (7, vec![Panel::Load]),]
        );
        // 60x12, the smallest terminal drawn: throughput only.
        assert_eq!(panels(60, 10), [(10, vec![Panel::Throughput])]);
    }
}
