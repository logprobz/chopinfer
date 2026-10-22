//! Drawing the interface into a [`buffer::Buffer`].

pub mod buffer;
pub mod detail;
pub mod gpus;
pub mod list;
pub mod widgets;

use buffer::{Buffer, Color, Style};
use widgets::{boxed, pad};

use crate::app::{App, View};
use crate::instances::{Instance, State, Tone};
use crate::stats::{HistogramOf, Snapshot, Window};

/// The smallest terminal anything is drawn in.
const MINIMUM: (u16, u16) = (60, 12);

/// Draws a whole frame at time `now`.
pub fn draw(app: &App, width: u16, height: u16, now: f64) -> Buffer {
    let mut buffer = Buffer::new(width, height);
    if width < MINIMUM.0 || height < MINIMUM.1 {
        buffer.put(
            0,
            0,
            &format!(
                "needs {}x{}; terminal is {width}x{height}",
                MINIMUM.0, MINIMUM.1
            ),
            Style::PLAIN,
        );
        return buffer;
    }
    match app.view {
        View::List => list::draw(&mut buffer, app, now),
        View::Detail => detail::draw(&mut buffer, app, now),
        View::Gpus => gpus::draw(&mut buffer, app, now),
    }
    if app.help {
        help(&mut buffer, app.view);
    }
    buffer
}

pub fn tone_color(tone: Tone) -> Color {
    match tone {
        Tone::Normal => Color::Default,
        Tone::Good => Color::Green,
        Tone::Warning => Color::Yellow,
        Tone::Bad => Color::Red,
    }
}

/// The header bar: `crumbs` after the program name, the window and poll
/// interval on the right.
pub fn header(buffer: &mut Buffer, app: &App, crumbs: &[(String, Style)]) {
    let width = buffer.width;
    buffer.background(0, 0, width, Color::Bar);
    // Dim text is the bar's own color, so on the bar it is plain.
    let bar = |style: Style| Style {
        fg: if style.fg == Color::Dim {
            Color::Default
        } else {
            style.fg
        },
        bg: Some(Color::Bar),
        ..style
    };
    let mut column = buffer.put(1, 0, "chopinfer-top", bar(Style::fg(Color::White).bold()));
    for (text, style) in crumbs {
        column = buffer.put(column, 0, text, bar(*style));
    }
    let right = header_right(app);
    let start = width.saturating_sub(right.chars().count() as u16);
    if start > column.saturating_add(1) {
        let style = if app.paused {
            Style::fg(Color::Yellow).bold()
        } else {
            Style::fg(Color::Default)
        };
        buffer.put(start, 0, &right, bar(style));
    }
}

/// The right of the header bar: pause, window, poll and rescan.
pub fn header_right(app: &App) -> String {
    format!(
        "{}window {}  poll {}s  rescan {}s ",
        if app.paused { "PAUSED  " } else { "" },
        app.window.label(),
        widgets::rate(app.interval)
            .trim_end_matches('0')
            .trim_end_matches('.'),
        app.rescan
    )
}

/// The footer bar of `(key, meaning, rank)` triples. Only whole pairs are
/// drawn: when they do not all fit, the pairs of the highest rank are left
/// out first, and those of rank 0 never are.
pub fn footer(buffer: &mut Buffer, keys: &[(&str, &str, u8)]) {
    let y = buffer.height - 1;
    let width = buffer.width;
    buffer.background(0, y, width, Color::Bar);
    let length = |keys: &[&(&str, &str, u8)]| {
        keys.iter()
            .map(|(key, meaning, _)| key.chars().count() + meaning.chars().count() + 4)
            .sum::<usize>()
            .saturating_sub(2)
    };
    let mut shown = keys.iter().collect::<Vec<_>>();
    while length(&shown) > usize::from(width) {
        let Some(rank) = shown
            .iter()
            .map(|(_, _, rank)| *rank)
            .max()
            .filter(|rank| *rank > 0)
        else {
            break;
        };
        shown.retain(|(_, _, seen)| *seen != rank);
    }
    let mut column = 1;
    for (key, meaning, _) in shown {
        column = buffer.put(
            column,
            y,
            key,
            Style {
                bg: Some(Color::Bar),
                ..Style::fg(Color::Cyan).bold()
            },
        );
        column = buffer.put(
            column,
            y,
            &format!(" {meaning}   "),
            Style {
                bg: Some(Color::Bar),
                ..Style::PLAIN
            },
        );
    }
}

fn help(buffer: &mut Buffer, view: View) {
    const INSTANCES: &[(&str, &str)] = &[
        ("Keys", ""),
        ("  ↑ ↓  j k", "select an instance (list)"),
        ("  Enter", "open its details"),
        ("  Esc  h", "back to the list"),
        ("  ← →  Tab", "previous or next instance (details)"),
        ("  g", "this host's GPUs: heat, clocks, memory and activity"),
        ("  w", "latency and ratio window: 60s, 5m, since start"),
        ("  r", "rescan for processes and containers now"),
        ("  p", "pause the screen; polls keep their latest result"),
        ("  q  Ctrl-C", "quit"),
        ("", ""),
        ("Sources", ""),
        (
            "  proc",
            "chopinfer serve processes of this user in this network namespace",
        ),
        (
            "  docker",
            "running containers whose process list shows chopinfer serve",
        ),
        (
            "  target",
            "--target, --targets-file and CHOPINFER_TOP_TARGETS",
        ),
        ("", ""),
        (
            "",
            "Reads only /health and /metrics. Rates come from counter deltas",
        ),
        ("", "between polls; a new startup nonce re-baselines them."),
        ("", ""),
        ("", "Press any key to close."),
    ];
    const GPUS: &[(&str, &str)] = &[
        ("Keys", ""),
        ("  ↑ ↓  j k", "select a GPU"),
        ("  Enter", "open the details of the instance on the GPU"),
        ("  Esc  h  g", "back to where g was pressed"),
        ("  w", "latency and ratio window: 60s, 5m, since start"),
        ("  r", "rescan for processes and containers now"),
        ("  p", "pause the screen; polls keep their latest result"),
        ("  q  Ctrl-C", "quit"),
        ("", ""),
        ("GPU time", ""),
        (
            "  kernels",
            "share of time a kernel ran on the GPU, from any process",
        ),
        ("  no kernel", "the rest: waiting for the host or for work"),
        ("  SM busy", "share of SMs with work (GPM)"),
        (
            "  occupancy",
            "active warps against the most the SMs hold (GPM)",
        ),
        ("  tensor", "share of time the tensor pipes ran (GPM)"),
        ("  DRAM", "share of peak memory bandwidth in use (GPM)"),
        ("", ""),
        (
            "",
            "NVML is read once a poll, GPM counters only while this view is",
        ),
        ("", "open. GPU numbers are nvidia-smi's, not CUDA ordinals."),
        ("", ""),
        ("", "Press any key to close."),
    ];
    let lines = if view == View::Gpus { GPUS } else { INSTANCES };
    let width = 92.min(buffer.width - 2);
    let height = (lines.len() as u16 + 2).min(buffer.height - 2);
    let x = (buffer.width - width) / 2;
    let y = (buffer.height - height) / 2;
    buffer.clear(x, y, width, height, None);
    boxed(
        buffer,
        (x, y, width, height),
        &[("Help", Style::fg(Color::Cyan).bold())],
        true,
    );
    for (row, (left, right)) in lines.iter().enumerate().take(usize::from(height - 2)) {
        let y = y + 1 + row as u16;
        let room = usize::from(width - 4);
        if right.is_empty() {
            buffer.put(x + 2, y, &pad(left, room), Style::fg(Color::Cyan).bold());
        } else if left.is_empty() {
            buffer.put(x + 2, y, &pad(right, room), Style::fg(Color::Dim));
        } else {
            let column = buffer.put(x + 2, y, &pad(left, 16), Style::fg(Color::Cyan));
            buffer.put(
                column,
                y,
                &pad(right, room.saturating_sub(16)),
                Style::PLAIN,
            );
        }
    }
}

/// How a row's state is shown, grouped for the header counts.
pub fn state_group(state: &State) -> &'static str {
    match state {
        State::Ready | State::Idle => "ready",
        State::NotReady(_) => "not ready",
        State::Connecting | State::Starting => "starting",
        State::NotAnswering => "not answering",
        State::Unreachable(_) | State::NotChopinfer => "unreachable",
        State::Failed => "failed",
        State::Poisoned => "poisoned",
    }
}

/// Figures of one instance over one window; `None` where there is no data.
pub struct Figures<'a> {
    pub instance: &'a Instance,
    pub window: Window,
}

impl<'a> Figures<'a> {
    pub fn new(instance: &'a Instance, window: Window) -> Self {
        Self { instance, window }
    }

    pub fn live(&self) -> bool {
        self.instance.reachable && !self.instance.stats.is_empty()
    }

    pub fn latest(&self) -> Option<&'a Snapshot> {
        self.instance
            .stats
            .latest()
            .filter(|_| self.instance.reachable)
    }

    pub fn decode(&self) -> Option<f64> {
        self.live()
            .then(|| self.instance.stats.rate(|s| s.counters.generation_tokens))
            .flatten()
    }

    pub fn prefill(&self) -> Option<f64> {
        self.live()
            .then(|| self.instance.stats.rate(|s| s.counters.prefill_tokens))
            .flatten()
    }

    pub fn token_hits(&self) -> Option<f64> {
        self.instance.stats.ratio(
            self.window,
            |s| s.counters.prefix_hit_tokens,
            |s| s.counters.prefix_query_tokens,
        )
    }

    pub fn lookup_hits(&self) -> Option<f64> {
        self.instance.stats.ratio(
            self.window,
            |s| s.counters.prefix_lookup_hits,
            |s| s.counters.prefix_lookups,
        )
    }

    pub fn acceptance(&self) -> Option<f64> {
        self.instance.stats.ratio(
            self.window,
            |s| s.counters.spec_accepted_tokens,
            |s| s.counters.spec_draft_tokens,
        )
    }

    pub fn quantile(&self, histogram: HistogramOf, q: f64) -> Option<f64> {
        self.instance.stats.quantile(self.window, histogram, q)
    }

    /// The mean observation of a histogram within the window.
    pub fn mean(&self, histogram: HistogramOf) -> Option<f64> {
        let window = self.instance.stats.histogram(self.window, histogram)?;
        (window.count > 0.0).then(|| window.sum / window.count)
    }

    pub fn running(&self) -> Option<f64> {
        self.latest().and_then(|s| s.gauges.running)
    }

    pub fn waiting(&self) -> Option<f64> {
        self.latest().and_then(|s| s.gauges.waiting)
    }

    pub fn kv(&self) -> Option<f64> {
        self.latest().and_then(|s| s.gauges.kv_usage)
    }

    pub fn max_sequences(&self) -> Option<u64> {
        self.instance
            .health
            .as_ref()?
            .serving
            .as_ref()?
            .profile
            .max_sequences
    }

    /// The window label with how much of it has data, when not all of it.
    pub fn window_label(&self) -> String {
        match self.instance.stats.coverage(self.window) {
            Some(coverage) if !coverage.complete && coverage.seconds.is_finite() => format!(
                "window {} ({} of data)",
                self.window.label(),
                widgets::age(coverage.seconds)
            ),
            _ => format!("window {}", self.window.label()),
        }
    }
}

/// The shown name of a row, sanitized.
pub fn shown_name(instance: &Instance) -> Option<String> {
    instance.name().map(widgets::sanitize)
}

/// The latency histograms, as `(label, selector)`.
pub const LATENCIES: [(&str, HistogramOf); 6] = [
    ("TTFT", |s| s.histograms.ttft.as_ref()),
    ("inter-token", |s| s.histograms.inter_token.as_ref()),
    ("end to end", |s| s.histograms.end_to_end.as_ref()),
    ("queue", |s| s.histograms.queue.as_ref()),
    ("prefill", |s| s.histograms.prefill.as_ref()),
    ("per output tok", |s| s.histograms.per_output_token.as_ref()),
];

pub const TTFT: HistogramOf = |s| s.histograms.ttft.as_ref();
pub const INTER_TOKEN: HistogramOf = |s| s.histograms.inter_token.as_ref();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_footer_drops_whole_pairs_by_rank() {
        let keys = [
            ("Esc", "back", 0),
            ("g", "gpus", 0),
            ("w", "window", 2),
            ("r", "rescan", 3),
            ("p", "pause", 1),
            ("q", "quit", 0),
        ];
        let line = |width| {
            let mut buffer = Buffer::new(width, 2);
            footer(&mut buffer, &keys);
            buffer
                .text()
                .lines()
                .nth(1)
                .unwrap_or_default()
                .trim_end()
                .to_owned()
        };
        assert_eq!(
            line(80),
            " Esc back   g gpus   w window   r rescan   p pause   q quit"
        );
        // Everything takes 59 columns.
        assert_eq!(line(59), line(80));
        assert_eq!(line(58), " Esc back   g gpus   w window   p pause   q quit");
        assert_eq!(line(36), " Esc back   g gpus   q quit");
        // Rank 0 is never dropped, so a tiny footer is cut instead.
        assert_eq!(line(10), " Esc back");
    }
}
