//! The monitor: this host's CPU, memory, I/O, disk space and sensors above
//! its GPUs. Panels and lines are given up in a fixed order as the terminal
//! shrinks: the Sensors panel runs down the side from 160 columns, sits
//! beside Host from 100, and folds into Host's lines below that.

use super::buffer::{Buffer, Color, Style};
use super::gpus;
use super::list::TABLE_TOP;
use super::widgets::{
    age, boxed, buckets, bytes, gauge, groups, integer, pad, percent, right, sanitize, sparkline,
};
use super::{footer, header, header_right};
use crate::app::App;
use crate::gpu::{Reasons, Status as GpuStatus};
use crate::host::{
    self, Chip, Field, Filesystem, Host, Kind, Level, Quantity, Rapl, Reading, Sensor,
    Status as HostStatus,
};

const GIB: f64 = 1_073_741_824.0;
const TIB: f64 = 1024.0 * GIB;
const SWAP_SHOWN: u64 = 64 << 20;
/// Eighths of a cell, for the cores line.
const CELLS: [char; 9] = [' ', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

pub type Area = (u16, u16, u16, u16);
type Pieces = Vec<(String, Style)>;

/// A line of the Host panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    Cpu,
    Memory,
    Cores,
    Load,
    Io,
    Space,
    Heat,
}

impl Line {
    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Memory => "memory",
            Self::Cores => "cores",
            Self::Load => "load",
            Self::Io => "I/O",
            Self::Space => "space",
            Self::Heat => "heat",
        }
    }
}

/// Where the monitor's panels go on a terminal of one size.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    pub host: Area,
    /// The Host lines shown, in order.
    pub lines: Vec<Line>,
    /// One line: CPU and memory side by side.
    pub compact: bool,
    /// The Sensors box beside Host, from 100 to 159 columns.
    pub sensors: Option<Area>,
    /// The full-height Sensors panel, from 160 columns.
    pub side: Option<Area>,
    /// Below Host: the GPU table and panels, or why there are none.
    pub gpus: Area,
    /// The GPU table, when there are GPUs to show.
    pub table: Option<Area>,
}

/// The most Host lines a terminal `height` rows tall shows; with no GPUs
/// to show, the most from 24 rows.
fn most_lines(height: u16, gpus_shown: bool) -> usize {
    match height {
        24.. if !gpus_shown => 7,
        52.. => 7,
        46.. => 6,
        36.. => 5,
        28.. => 3,
        20.. => 2,
        _ => 1,
    }
}

/// Where everything goes on a terminal `width` by `height`.
pub fn layout(width: u16, height: u16, app: &App) -> Layout {
    let gpus_shown = app.gpus.status == GpuStatus::Open && !app.gpus.devices.is_empty();
    let share = |percent: u32| (u32::from(width) * percent / 100) as u16;
    let side_width = if width >= 160 {
        share(24).clamp(36, 48)
    } else {
        0
    };
    let left = width - side_width;
    let box_width = if side_width == 0 && width >= 100 {
        ((u32::from(width) * 30 + 50) / 100).clamp(32, 44) as u16
    } else {
        0
    };
    // The lines and the order they are given up in, the highest first.
    // Disk space moves to the side panel when there is one; without a
    // Sensors panel, heat comes right after CPU and memory.
    let mut lines: Vec<(u8, Line)> = if side_width > 0 {
        vec![
            (0, Line::Cpu),
            (0, Line::Memory),
            (1, Line::Cores),
            (2, Line::Load),
            (3, Line::Io),
            (4, Line::Heat),
        ]
    } else if box_width > 0 {
        vec![
            (0, Line::Cpu),
            (0, Line::Memory),
            (1, Line::Cores),
            (2, Line::Load),
            (3, Line::Io),
            (4, Line::Space),
            (5, Line::Heat),
        ]
    } else {
        vec![
            (0, Line::Cpu),
            (0, Line::Memory),
            (1, Line::Heat),
            (2, Line::Cores),
            (3, Line::Load),
            (4, Line::Io),
            (5, Line::Space),
        ]
    };
    let most = most_lines(height, gpus_shown);
    while lines.len() > most {
        let Some(highest) = lines.iter().map(|(priority, _)| *priority).max() else {
            break;
        };
        let Some(position) = lines.iter().rposition(|(priority, _)| *priority == highest) else {
            break;
        };
        lines.remove(position);
    }
    let compact = most == 1 || app.host.latest().is_none();
    let host_height = if compact { 3 } else { lines.len() as u16 + 2 };
    let below = TABLE_TOP + host_height;
    let gpus = (0, below, left, height.saturating_sub(1 + below));
    let table = gpus_shown.then(|| {
        (
            0,
            below,
            left,
            gpus::table_height(app.gpus.devices.len(), gpus.3),
        )
    });
    Layout {
        host: (0, TABLE_TOP, left - box_width, host_height),
        lines: lines.into_iter().map(|(_, line)| line).collect(),
        compact,
        sensors: (box_width > 0).then_some((left - box_width, TABLE_TOP, box_width, host_height)),
        side: (side_width > 0).then(|| {
            (
                left,
                TABLE_TOP,
                side_width,
                height.saturating_sub(1 + TABLE_TOP),
            )
        }),
        gpus,
        table,
    }
}

fn dim() -> Style {
    Style::fg(Color::Dim)
}

fn title() -> Style {
    Style::fg(Color::Cyan).bold()
}

fn piece(text: impl Into<String>, style: Style) -> (String, Style) {
    (text.into(), style)
}

fn level_style(level: Level) -> Style {
    match level {
        Level::Hot => Style::fg(Color::Red),
        Level::Warm => Style::fg(Color::Yellow),
        Level::Normal => Style::PLAIN,
    }
}

fn level_color(level: Level) -> Color {
    match level {
        Level::Hot => Color::Red,
        Level::Warm => Color::Yellow,
        Level::Normal => Color::Green,
    }
}

fn temperature_level(sensor: &Sensor) -> Level {
    sensor
        .value
        .map_or(Level::Normal, |celsius| host::level(celsius, sensor.limit))
}

fn celsius(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |celsius| format!("{celsius:.0}C"))
}

fn rate(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_owned(), |value| format!("{}/s", bytes(value)))
}

/// `2.9/3.6 TiB`, `0.4/1.9 GiB`, `3/126 GiB`.
fn space(used: u64, total: u64) -> String {
    let (used, total) = (used as f64, total as f64);
    if total >= TIB {
        format!("{:.1}/{:.1} TiB", used / TIB, total / TIB)
    } else {
        format!("{} GiB", gpus::memory(used as u64, total as u64))
    }
}

fn temperatures(chip: &Chip) -> impl Iterator<Item = &Sensor> {
    chip.sensors
        .iter()
        .filter(|sensor| sensor.quantity == Quantity::Celsius)
}

/// The hottest main temperature of the CPU chips.
fn cpu_temperature(host: &Host) -> Option<&Sensor> {
    hottest(
        host.chips
            .iter()
            .filter(|chip| chip.kind == Kind::Cpu)
            .flat_map(temperatures)
            .filter(|sensor| sensor.primary),
    )
}

/// The sensor of `sensors` nearest its limit, else the hottest.
fn hottest<'a>(sensors: impl Iterator<Item = &'a Sensor>) -> Option<&'a Sensor> {
    sensors
        .filter(|sensor| sensor.value.is_some())
        .max_by(|a, b| {
            (temperature_level(a), a.value)
                .partial_cmp(&(temperature_level(b), b.value))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn hottest_of(host: &Host, kind: Kind) -> Option<&Sensor> {
    hottest(
        host.chips
            .iter()
            .filter(|chip| chip.kind == kind)
            .flat_map(temperatures),
    )
}

pub fn draw(buffer: &mut Buffer, app: &App, now: f64) {
    let crumbs = crumbs(app, buffer.width);
    header(buffer, app, &crumbs);
    footer(
        buffer,
        &[
            ("Esc", "back", 0),
            ("↑↓ jk", "select", 0),
            ("Enter", "instance", 0),
            ("w", "window", 2),
            ("r", "rescan", 3),
            ("p", "pause", 1),
            ("?", "help", 0),
            ("q", "quit", 0),
        ],
    );
    let layout = layout(buffer.width, buffer.height, app);
    host_panel(buffer, app, &layout, now);
    if let Some(area) = layout.sensors {
        sensors_box(buffer, &app.host, area);
    }
    if let Some(area) = layout.side {
        side_panel(buffer, &app.host, area, now);
    }
    gpus::panels(buffer, app, layout.gpus, now);
}

/// The header after the program name: CPUs and GPUs, what needs a look
/// (GPUs slowing down or lost, a hot CPU, memory or a filesystem nearly
/// full), then the driver and GPM state when they leave room for the
/// window.
fn crumbs(app: &App, width: u16) -> Vec<(String, Style)> {
    let (host, gpus) = (&app.host, &app.gpus);
    let mut crumbs = vec![
        piece("  >  ", dim()),
        piece("Monitor", Style::fg(Color::White).bold()),
    ];
    if host.threads > 0 {
        crumbs.push(piece(format!("   {} CPUs", host.threads), Style::PLAIN));
    }
    let shown = gpus.status == GpuStatus::Open;
    if shown {
        crumbs.push(piece(
            format!("  {} GPUs", gpus.devices.len()),
            Style::PLAIN,
        ));
        let slow = (0..gpus.devices.len())
            .filter_map(|index| gpus.latest(index))
            .filter(|reading| reading.reasons.is_some_and(Reasons::slowdown))
            .count();
        if slow > 0 {
            crumbs.push(piece(
                format!("  {slow} slowing down"),
                Style::fg(Color::Yellow).bold(),
            ));
        }
        let lost = (0..gpus.devices.len())
            .filter(|index| gpus.lost(*index))
            .count();
        if lost > 0 {
            crumbs.push(piece(
                format!("  {lost} lost"),
                Style::fg(Color::Red).bold(),
            ));
        }
    }
    if let Some(sensor) = cpu_temperature(host)
        && temperature_level(sensor) > Level::Normal
    {
        crumbs.push(piece(
            format!("  CPU {}", celsius(sensor.value)),
            level_style(temperature_level(sensor)).bold(),
        ));
    }
    if let Some(share) = host
        .latest()
        .and_then(|reading| reading.memory?.share())
        .filter(|share| *share > 0.95)
    {
        crumbs.push(piece(
            format!("  memory {}", percent(Some(share))),
            Style::fg(Color::Red).bold(),
        ));
    }
    for filesystem in &host.filesystems {
        let level = host::fullness(filesystem.share());
        if level > Level::Normal {
            crumbs.push(piece(
                format!(
                    "  {} {}% full",
                    sanitize(&filesystem.mount),
                    filesystem.percent()
                ),
                level_style(level).bold(),
            ));
        }
    }
    let used = 14
        + crumbs
            .iter()
            .map(|(text, _)| text.chars().count())
            .sum::<usize>();
    let room = usize::from(width).saturating_sub(header_right(app).chars().count() + 2);
    let extras = match &gpus.status {
        GpuStatus::Open => {
            let gpm = if !gpus.counters {
                "off"
            } else if gpus.gpm() {
                "on"
            } else {
                "unsupported"
            };
            let driver = gpus
                .driver
                .as_deref()
                .map_or_else(|| "-".to_owned(), sanitize);
            vec![
                format!("   driver {driver}   GPM {gpm}"),
                format!("   GPM {gpm}"),
            ]
        }
        GpuStatus::Unavailable(_) => vec!["   no GPU telemetry".to_owned()],
        GpuStatus::Off => vec!["   GPUs not read".to_owned()],
        GpuStatus::Opening => Vec::new(),
    };
    for extra in extras {
        if used + extra.chars().count() <= room {
            crumbs.push(piece(extra, dim()));
            break;
        }
    }
    crumbs
}

/// A sparkline of `field` over the history, after a `5m` label.
fn trend(
    buffer: &mut Buffer,
    host: &Host,
    (x, y, width): (u16, u16, u16),
    field: Field,
    color: Color,
) {
    let history = &host.history;
    buffer.put(x.saturating_sub(3), y, "5m", dim());
    let columns = buckets(
        &history.series(field),
        history.oldest_poll(),
        history.capacity(),
        usize::from(width),
    );
    sparkline(buffer, (x, y, width, 1), &columns, Some(1.0), color);
}

/// CPU, memory and the lines the layout leaves room for.
fn host_panel(buffer: &mut Buffer, app: &App, layout: &Layout, now: f64) {
    let host = &app.host;
    let (x, y, width, height) = layout.host;
    let model = host
        .model
        .as_deref()
        .map_or_else(String::new, |model| format!("  {}", sanitize(model)));
    let uptime = host
        .latest()
        .and_then(|reading| reading.uptime)
        .map_or_else(String::new, |seconds| format!("  up {}", age(seconds)));
    boxed(
        buffer,
        layout.host,
        &[
            ("Host", title()),
            (&model, Style::fg(Color::White).bold()),
            (&uptime, dim()),
        ],
        false,
    );
    if height < 3 || width < 20 {
        return;
    }
    let Some(reading) = host.latest() else {
        let (text, style) = match &host.status {
            HostStatus::Unavailable(reason) => (
                format!("This host is not read: {}.", sanitize(reason)),
                Style::fg(Color::Yellow),
            ),
            _ => ("waiting for the first reading".to_owned(), dim()),
        };
        buffer.put(x + 2, y + 1, &pad(&text, usize::from(width - 4)), style);
        return;
    };
    if layout.compact {
        compact(buffer, host, reading, (x, y + 1, width));
        return;
    }
    let value = x + 12;
    let end = x + width - 2;
    let room = end.saturating_sub(value);
    let spark = if room >= 64 {
        (room * 3 / 10).clamp(12, 48)
    } else {
        0
    };
    let spark_x = end - spark;
    let text_end = if spark > 0 { spark_x - 4 } else { end };
    let gauge_width = (text_end.saturating_sub(value) / 2).clamp(14, 44);
    for (offset, line) in layout.lines.iter().enumerate() {
        let line_y = y + 1 + offset as u16;
        buffer.put(x + 2, line_y, line.label(), dim());
        let at = (value, line_y);
        match line {
            Line::Cpu => {
                cpu_line(buffer, host, layout, at, (gauge_width, text_end));
                if spark > 0 {
                    trend(
                        buffer,
                        host,
                        (spark_x, line_y, spark),
                        |reading| reading.cpu.map(|cpu| cpu.busy),
                        Color::Cyan,
                    );
                }
            }
            Line::Memory => {
                memory_line(buffer, reading, at, (gauge_width, text_end));
                if spark > 0 {
                    trend(
                        buffer,
                        host,
                        (spark_x, line_y, spark),
                        |reading| reading.memory?.share(),
                        Color::Green,
                    );
                }
            }
            Line::Cores => cores_line(buffer, host, at, end),
            Line::Load => load_line(buffer, host, reading, at, end),
            Line::Io => io_line(buffer, host, reading, at, end),
            Line::Space => space_line(buffer, host, at, end, now),
            Line::Heat => heat_line(buffer, host, reading, at, end),
        }
    }
}

fn dash(buffer: &mut Buffer, (x, y): (u16, u16)) {
    buffer.put(x, y, "-", dim());
}

fn cpu_line(
    buffer: &mut Buffer,
    host: &Host,
    layout: &Layout,
    (x, y): (u16, u16),
    (gauge_width, end): (u16, u16),
) {
    let busy = host.recent(|reading| reading.cpu.map(|cpu| cpu.busy));
    gauge(
        buffer,
        x,
        y,
        gauge_width,
        busy.unwrap_or(0.0),
        Color::Cyan,
        &right(&percent(busy), 4),
    );
    let mut pieces: Vec<Pieces> = Vec::new();
    // Without the heat line, the CPU's temperature rides here.
    if !layout.lines.contains(&Line::Heat)
        && let Some(sensor) = cpu_temperature(host)
    {
        pieces.push(vec![
            piece("   ", dim()),
            piece(
                celsius(sensor.value),
                level_style(temperature_level(sensor)).bold(),
            ),
        ]);
    }
    let iowait = host.recent(|reading| reading.cpu.map(|cpu| cpu.iowait));
    pieces.extend([
        vec![
            piece("   user ", dim()),
            piece(
                percent(host.recent(|reading| reading.cpu.map(|cpu| cpu.user))),
                Style::PLAIN,
            ),
        ],
        vec![
            piece("  system ", dim()),
            piece(
                percent(host.recent(|reading| reading.cpu.map(|cpu| cpu.system))),
                Style::PLAIN,
            ),
        ],
        vec![
            piece("  iowait ", dim()),
            piece(
                percent(iowait),
                if iowait.is_some_and(|iowait| iowait >= 0.1) {
                    Style::fg(Color::Yellow)
                } else {
                    Style::PLAIN
                },
            ),
        ],
    ]);
    groups(buffer, x + gauge_width, y, &pieces, end);
}

fn memory_line(
    buffer: &mut Buffer,
    reading: &Reading,
    (x, y): (u16, u16),
    (gauge_width, end): (u16, u16),
) {
    let Some(memory) = reading.memory.filter(|memory| memory.total > 0) else {
        dash(buffer, (x, y));
        return;
    };
    let share = memory.share().unwrap_or(0.0);
    let color = if share > 0.95 {
        Color::Red
    } else if share > 0.9 {
        Color::Yellow
    } else {
        Color::Green
    };
    gauge(
        buffer,
        x,
        y,
        gauge_width,
        share,
        color,
        &format!("{} GiB", gpus::memory(memory.used(), memory.total)),
    );
    let swap = if memory.swap_total == 0 {
        piece("none", dim())
    } else {
        piece(
            format!(
                "{:.1}/{:.1} GiB",
                memory.swap_used() as f64 / GIB,
                memory.swap_total as f64 / GIB
            ),
            if memory.swap_used() > SWAP_SHOWN {
                Style::fg(Color::Yellow)
            } else {
                Style::PLAIN
            },
        )
    };
    groups(
        buffer,
        x + gauge_width,
        y,
        &[
            vec![piece("   swap ", dim()), swap],
            vec![
                piece("   cache ", dim()),
                piece(bytes(memory.cached as f64), Style::PLAIN),
            ],
        ],
        end,
    );
}

/// One cell per CPU, or per few when they do not fit, then how many are
/// nearly saturated.
fn cores_line(buffer: &mut Buffer, host: &Host, (x, y): (u16, u16), end: u16) {
    let cores = host.history.cores();
    if cores.is_empty() {
        dash(buffer, (x, y));
        return;
    }
    let room = usize::from(end.saturating_sub(x)).saturating_sub(16).max(8);
    let per = cores.len().div_ceil(room).max(1);
    let mut column = x;
    for chunk in cores.chunks(per) {
        let busy = chunk.iter().sum::<f64>() / chunk.len() as f64;
        let eighths = ((busy * 8.0).round() as usize).clamp(1, 8);
        column = buffer.put(
            column,
            y,
            &CELLS[eighths].to_string(),
            Style::fg(if busy < 0.05 { Color::Dim } else { Color::Cyan }),
        );
    }
    let saturated = cores.iter().filter(|busy| **busy >= 0.9).count();
    let mut pieces = vec![vec![
        piece("   ", dim()),
        piece(saturated.to_string(), Style::PLAIN),
        piece(" above 90%", dim()),
    ]];
    if per > 1 {
        pieces.push(vec![piece(format!("   {per} CPUs a cell"), dim())]);
    }
    groups(buffer, column, y, &pieces, end);
}

fn load_line(buffer: &mut Buffer, host: &Host, reading: &Reading, (x, y): (u16, u16), end: u16) {
    let Some(load) = reading.load else {
        dash(buffer, (x, y));
        return;
    };
    let over = |load: f64| {
        if host.threads > 0 && load > host.threads as f64 {
            Style::fg(Color::Yellow)
        } else {
            Style::PLAIN
        }
    };
    let mut pieces = vec![
        vec![piece(format!("{:.2}", load.one), over(load.one))],
        vec![piece(format!("  {:.2}", load.five), over(load.five))],
        vec![piece(format!("  {:.2}", load.fifteen), over(load.fifteen))],
        vec![piece(format!(" of {}", host.threads), dim())],
    ];
    let stalled = |share: f64| {
        if share >= 0.4 {
            Style::fg(Color::Red)
        } else if share >= 0.1 {
            Style::fg(Color::Yellow)
        } else {
            Style::PLAIN
        }
    };
    let pressure = reading.pressure;
    let mut first = true;
    for (name, share) in [
        ("cpu", pressure.cpu),
        ("memory", pressure.memory),
        ("io", pressure.io),
    ] {
        let Some(share) = share else {
            continue;
        };
        let lead = if first { "   pressure " } else { "  " };
        first = false;
        pieces.push(vec![
            piece(format!("{lead}{name} "), dim()),
            piece(percent(Some(share)), stalled(share)),
        ]);
    }
    if let Some(tasks) = load.tasks {
        pieces.push(vec![
            piece(format!("   {}", integer(tasks as f64)), Style::PLAIN),
            piece(" tasks", dim()),
        ]);
    }
    groups(buffer, x, y, &pieces, end);
}

fn io_line(buffer: &mut Buffer, host: &Host, reading: &Reading, (x, y): (u16, u16), end: u16) {
    let mut pieces = Vec::new();
    if reading.network_visible {
        pieces.push(vec![
            piece("net  rx ", dim()),
            piece(
                rate(host.recent(|reading| reading.network.map(|rates| rates.incoming))),
                Style::PLAIN,
            ),
        ]);
        pieces.push(vec![
            piece("  tx ", dim()),
            piece(
                rate(host.recent(|reading| reading.network.map(|rates| rates.outgoing))),
                Style::PLAIN,
            ),
        ]);
    } else {
        pieces.push(vec![piece(
            "net not visible in this network namespace",
            dim(),
        )]);
    }
    pieces.push(vec![
        piece("   disk  read ", dim()),
        piece(
            rate(host.recent(|reading| reading.disk.map(|rates| rates.incoming))),
            Style::PLAIN,
        ),
    ]);
    pieces.push(vec![
        piece("  write ", dim()),
        piece(
            rate(host.recent(|reading| reading.disk.map(|rates| rates.outgoing))),
            Style::PLAIN,
        ),
    ]);
    groups(buffer, x, y, &pieces, end);
}

/// "as read 1m05s ago" when the sizes are older than three rounds.
fn stale(host: &Host, now: f64) -> Option<String> {
    host.space_age(now)
        .map(|seconds| format!("as read {} ago", age(seconds)))
}

/// Each filesystem as `/data 82% of 3.6 TiB`, in mount order. When they do
/// not all fit, the fullest are kept and the line ends in how many are not.
fn space_line(buffer: &mut Buffer, host: &Host, (x, y): (u16, u16), end: u16, now: f64) {
    if host.space_at.is_none() {
        buffer.put(x, y, "not read yet", dim());
        return;
    }
    let filesystems = &host.filesystems;
    let mut lead: Vec<Pieces> = Vec::new();
    if let Some(stale) = stale(host, now) {
        lead.push(vec![piece(format!("{stale}   "), Style::fg(Color::Yellow))]);
    }
    if filesystems.is_empty() {
        lead.push(vec![piece("no local filesystems", dim())]);
        groups(buffer, x, y, &lead, end);
        return;
    }
    let item = |filesystem: &Filesystem, first: bool| {
        let level = host::fullness(filesystem.share());
        let mut style = level_style(level);
        style.bold = level > Level::Normal;
        vec![
            piece(if first { "" } else { "   " }, dim()),
            piece(format!("{} ", sanitize(&filesystem.mount)), Style::PLAIN),
            piece(format!("{}%", filesystem.percent()), style),
            piece(format!(" of {}", bytes(filesystem.total as f64)), dim()),
        ]
    };
    let mut fullest = (0..filesystems.len()).collect::<Vec<_>>();
    fullest.sort_by(|a, b| {
        filesystems[*b]
            .share()
            .partial_cmp(&filesystems[*a].share())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let pick = |count: usize| {
        let mut pieces = lead.clone();
        let mut kept = fullest[..count].to_vec();
        kept.sort_unstable();
        for (position, index) in kept.iter().enumerate() {
            pieces.push(item(&filesystems[*index], position == 0));
        }
        if count < filesystems.len() {
            pieces.push(vec![piece(
                format!(
                    "{}+{} more",
                    if count > 0 { "   " } else { "" },
                    filesystems.len() - count
                ),
                dim(),
            )]);
        }
        pieces
    };
    let length = |pieces: &[Pieces]| {
        pieces
            .iter()
            .flatten()
            .map(|(text, _)| text.chars().count())
            .sum::<usize>()
    };
    let room = usize::from(end.saturating_sub(x));
    let mut count = filesystems.len();
    while count > 0 && length(&pick(count)) > room {
        count -= 1;
    }
    groups(buffer, x, y, &pick(count), end);
}

/// The fans shown, each its speed, or why there are none.
fn fan_pieces(host: &Host) -> Pieces {
    let fans = host.fans();
    if fans.is_empty() {
        let why = if host.fan_channels == 0 {
            "none exposed"
        } else {
            "none reporting"
        };
        return vec![piece(why, dim())];
    }
    let mut pieces = Vec::new();
    for (index, (_, sensor)) in fans.iter().enumerate() {
        let rpm = sensor.value.unwrap_or(0.0);
        pieces.push(piece(if index == 0 { "" } else { " " }, dim()));
        pieces.push(piece(
            integer(rpm),
            if rpm > 0.0 {
                Style::PLAIN
            } else {
                Style::fg(Color::Yellow)
            },
        ));
    }
    pieces.push(piece(" rpm", dim()));
    pieces
}

fn heat_line(buffer: &mut Buffer, host: &Host, reading: &Reading, (x, y): (u16, u16), end: u16) {
    let mut pieces: Vec<Pieces> = Vec::new();
    match cpu_temperature(host) {
        Some(sensor) => pieces.push(vec![
            piece("CPU ", dim()),
            piece(
                celsius(sensor.value),
                level_style(temperature_level(sensor)).bold(),
            ),
            piece(format!(" {}", sanitize(&sensor.label)), dim()),
        ]),
        None => pieces.push(vec![piece("CPU -", dim())]),
    }
    for (kind, label) in [
        (Kind::Nvme, "NVMe"),
        (Kind::Dimm, "DIMM"),
        (Kind::Nic, "NIC"),
    ] {
        if let Some(sensor) = hottest_of(host, kind) {
            pieces.push(vec![
                piece(format!("   {label} "), dim()),
                piece(
                    celsius(sensor.value),
                    level_style(temperature_level(sensor)),
                ),
            ]);
        }
    }
    let mut fans = vec![piece("   fans ", dim())];
    fans.extend(fan_pieces(host));
    pieces.push(fans);
    if host.rapl == Rapl::Readable
        && let Some(watts) = reading.package
    {
        pieces.push(vec![
            piece("   CPU ", dim()),
            piece(format!("{watts:.0} W"), Style::PLAIN),
        ]);
    }
    groups(buffer, x, y, &pieces, end);
}

/// CPU and memory side by side, for a Host panel of one line.
fn compact(buffer: &mut Buffer, host: &Host, reading: &Reading, (x, y, width): (u16, u16, u16)) {
    let end = x + width - 2;
    let half = (end - (x + 2)) / 2;
    buffer.put(x + 2, y, "CPU", dim());
    let busy = host.recent(|reading| reading.cpu.map(|cpu| cpu.busy));
    gauge(
        buffer,
        x + 6,
        y,
        half.saturating_sub(6),
        busy.unwrap_or(0.0),
        Color::Cyan,
        &right(&percent(busy), 4),
    );
    let column = x + 2 + half + 1;
    buffer.put(column, y, "memory", dim());
    if let Some(memory) = reading.memory.filter(|memory| memory.total > 0) {
        let share = memory.share().unwrap_or(0.0);
        gauge(
            buffer,
            column + 7,
            y,
            end.saturating_sub(column + 7),
            share,
            if share > 0.95 {
                Color::Red
            } else {
                Color::Green
            },
            &format!("{} GiB", gpus::memory(memory.used(), memory.total)),
        );
    }
}

/// One line of the Sensors box: a kind of chip and what its sensors read.
struct Summary<'a> {
    label: String,
    pieces: Vec<Pieces>,
    hottest: Option<&'a Sensor>,
}

/// Temperatures one after another, each in the colour of its level.
fn temperature_pieces<'a>(sensors: impl Iterator<Item = &'a Sensor>, bold: bool) -> Vec<Pieces> {
    sensors
        .enumerate()
        .map(|(index, sensor)| {
            let mut style = level_style(temperature_level(sensor));
            style.bold = bold;
            vec![
                piece(if index == 0 { "" } else { "  " }, dim()),
                piece(celsius(sensor.value), style),
            ]
        })
        .collect()
}

/// The main temperature of each of `chips`.
fn primaries<'a>(chips: &[&'a Chip]) -> Vec<&'a Sensor> {
    chips
        .iter()
        .flat_map(|chip| temperatures(chip))
        .filter(|sensor| sensor.primary)
        .collect()
}

/// The Sensors box's lines: CPU, fans, NVMe, DIMMs, NIC, board, then other
/// chips.
fn summaries(host: &Host) -> Vec<Summary<'_>> {
    let of = |kind: Kind| {
        host.chips
            .iter()
            .filter(move |chip| chip.kind == kind)
            .collect::<Vec<_>>()
    };
    let mut lines = Vec::new();
    let cpus = of(Kind::Cpu);
    if !cpus.is_empty() {
        let main = cpus
            .iter()
            .flat_map(|chip| temperatures(chip))
            .filter(|sensor| sensor.primary)
            .collect::<Vec<_>>();
        let rest = cpus
            .iter()
            .flat_map(|chip| temperatures(chip))
            .filter(|sensor| !sensor.primary)
            .collect::<Vec<_>>();
        let mut pieces = temperature_pieces(main.iter().copied(), true);
        if !rest.is_empty() {
            let what = if rest.iter().all(|sensor| sensor.label.starts_with("Tccd")) {
                "ccd"
            } else if rest.iter().all(|sensor| sensor.label.starts_with("Core")) {
                "cores"
            } else {
                "more"
            };
            if rest.len() <= 8 {
                let values = rest
                    .iter()
                    .map(|sensor| {
                        sensor
                            .value
                            .map_or_else(|| "-".to_owned(), |value| format!("{value:.0}"))
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                let level = rest.iter().map(|sensor| temperature_level(sensor)).max();
                pieces.push(vec![
                    piece(format!("   {what} "), dim()),
                    piece(values, level_style(level.unwrap_or(Level::Normal))),
                    piece("C", dim()),
                ]);
            } else if let Some(top) = hottest(rest.iter().copied()) {
                pieces.push(vec![
                    piece(format!("   hottest of {} {what} ", rest.len()), dim()),
                    piece(celsius(top.value), level_style(temperature_level(top))),
                ]);
            }
        }
        lines.push(Summary {
            label: "CPU".to_owned(),
            pieces,
            hottest: hottest(main.iter().chain(&rest).copied()),
        });
    }
    lines.push(Summary {
        label: "fans".to_owned(),
        pieces: vec![fan_pieces(host)],
        hottest: None,
    });
    let mut drives = of(Kind::Nvme);
    drives.sort_by(|a, b| a.name.cmp(&b.name));
    let drive_sensors = primaries(&drives);
    if !drive_sensors.is_empty() {
        lines.push(Summary {
            label: "NVMe".to_owned(),
            pieces: temperature_pieces(drive_sensors.iter().copied(), false),
            hottest: hottest(drive_sensors.iter().copied()),
        });
    }
    let dimms = of(Kind::Dimm)
        .into_iter()
        .flat_map(temperatures)
        .collect::<Vec<_>>();
    if let Some(top) = hottest(dimms.iter().copied()) {
        let values = dimms
            .iter()
            .filter_map(|sensor| sensor.value)
            .collect::<Vec<_>>();
        let low = values.iter().copied().fold(f64::INFINITY, f64::min);
        let pieces = if dimms.len() == 1 {
            temperature_pieces(dimms.iter().copied(), false)
        } else {
            vec![vec![
                piece(format!("{} at ", dimms.len()), dim()),
                piece(
                    format!("{low:.0}-{}", celsius(top.value)),
                    level_style(temperature_level(top)),
                ),
            ]]
        };
        lines.push(Summary {
            label: "DIMMs".to_owned(),
            pieces,
            hottest: Some(top),
        });
    }
    let nics = primaries(&of(Kind::Nic));
    if !nics.is_empty() {
        lines.push(Summary {
            label: "NIC".to_owned(),
            pieces: temperature_pieces(nics.iter().copied(), false),
            hottest: hottest(nics.iter().copied()),
        });
    }
    let boards = of(Kind::Board)
        .into_iter()
        .flat_map(temperatures)
        .collect::<Vec<_>>();
    if !boards.is_empty() {
        lines.push(Summary {
            label: "board".to_owned(),
            pieces: temperature_pieces(boards.iter().copied(), false),
            hottest: hottest(boards.iter().copied()),
        });
    }
    for chip in of(Kind::Other) {
        let sensors = temperatures(chip).collect::<Vec<_>>();
        if sensors.is_empty() {
            continue;
        }
        lines.push(Summary {
            label: sanitize(&chip.name),
            pieces: temperature_pieces(sensors.iter().copied(), false),
            hottest: hottest(sensors.iter().copied()),
        });
    }
    lines
}

/// The Sensors box beside Host: a line per kind of chip, and how many more
/// when they do not all fit.
fn sensors_box(buffer: &mut Buffer, host: &Host, area: Area) {
    let (x, y, width, height) = area;
    let count = format!("  {} on {} chips", host.sensors(), host.chips.len());
    boxed(
        buffer,
        area,
        &[("Sensors", title()), (&count, dim())],
        false,
    );
    let lines = usize::from(height.saturating_sub(2));
    if lines == 0 || width < 12 {
        return;
    }
    let end = x + width - 2;
    if host.chips.is_empty() {
        let what = match host.status {
            HostStatus::Open => "no sensors in /sys/class/hwmon",
            _ => "-",
        };
        buffer.put(x + 2, y + 1, &pad(what, usize::from(width - 4)), dim());
        return;
    }
    let summaries = summaries(host);
    let (shown, rest) = if summaries.len() > lines {
        summaries.split_at(lines - 1)
    } else {
        (&summaries[..], &[][..])
    };
    for (offset, summary) in shown.iter().enumerate() {
        let line_y = y + 1 + offset as u16;
        buffer.put(x + 2, line_y, &pad(&summary.label, 7), dim());
        groups(buffer, x + 10, line_y, &summary.pieces, end);
    }
    if !rest.is_empty() {
        let mut pieces = vec![vec![piece(format!("+{} more", rest.len()), dim())]];
        if let Some(top) = hottest(rest.iter().filter_map(|summary| summary.hottest)) {
            pieces.push(vec![
                piece("   hottest ", dim()),
                piece(celsius(top.value), level_style(temperature_level(top))),
            ]);
        }
        let names = rest
            .iter()
            .map(|summary| summary.label.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        pieces.push(vec![piece(format!("  {names}"), dim())]);
        groups(buffer, x + 2, y + lines as u16, &pieces, end);
    }
}

/// A line of the side panel.
enum Row<'a> {
    Heading(String, String, Style),
    Sensor(&'a Chip, &'a Sensor, String),
    Disk(&'a Filesystem),
    Package(Option<f64>),
    Note(&'static str),
}

/// A row with the group it belongs to and how late it is given up: the
/// highest priority goes first.
struct Entry<'a> {
    group: usize,
    priority: u8,
    row: Row<'a>,
}

/// The side panel's rows, grouped: CPU, fans, each NVMe drive, memory,
/// NICs, boards, other chips, disks and power.
fn side_rows<'a>(host: &'a Host, now: f64) -> Vec<Entry<'a>> {
    let mut rows = Vec::new();
    let mut group = 0;
    let mut heading = |rows: &mut Vec<Entry<'a>>, kind: &str, detail: String, style: Style| {
        group += 1;
        rows.push(Entry {
            group,
            priority: 0,
            row: Row::Heading(kind.to_owned(), detail, style),
        });
        group
    };
    let of = |kind: Kind| {
        host.chips
            .iter()
            .filter(move |chip| chip.kind == kind)
            .collect::<Vec<_>>()
    };
    let readings = |chip: &'a Chip| {
        chip.sensors.iter().filter(|sensor| {
            matches!(
                sensor.quantity,
                Quantity::Celsius | Quantity::Watts | Quantity::Volts
            )
        })
    };
    for chip in of(Kind::Cpu) {
        let group = heading(
            &mut rows,
            &chip.name,
            format!("  {}", sanitize(&chip.driver)),
            dim(),
        );
        for sensor in readings(chip) {
            rows.push(Entry {
                group,
                priority: if sensor.primary { 0 } else { 2 },
                row: Row::Sensor(chip, sensor, sanitize(&sensor.label)),
            });
        }
    }
    let fan_drivers = {
        let mut drivers = host
            .chips
            .iter()
            .filter(|chip| {
                chip.sensors
                    .iter()
                    .any(|sensor| sensor.quantity == Quantity::Rpm)
            })
            .map(|chip| sanitize(&chip.driver))
            .collect::<Vec<_>>();
        drivers.dedup();
        drivers
    };
    let group_of_fans = heading(
        &mut rows,
        "Fans",
        if fan_drivers.is_empty() {
            String::new()
        } else {
            format!("  {}", fan_drivers.join(", "))
        },
        dim(),
    );
    let fans = host.fans();
    if fans.is_empty() {
        rows.push(Entry {
            group: group_of_fans,
            priority: 0,
            row: Row::Note(if host.fan_channels == 0 {
                "none exposed by the kernel"
            } else {
                "none reporting"
            }),
        });
    }
    for (chip, sensor) in fans {
        let label = if fan_drivers.len() > 1 {
            format!("{} {}", sanitize(&chip.name), sanitize(&sensor.label))
        } else {
            sanitize(&sensor.label)
        };
        rows.push(Entry {
            group: group_of_fans,
            priority: 0,
            row: Row::Sensor(chip, sensor, label),
        });
    }
    let mut drives = of(Kind::Nvme);
    drives.sort_by(|a, b| a.name.cmp(&b.name));
    for chip in drives {
        let model = chip
            .model
            .as_deref()
            .map_or_else(String::new, |model| format!("  {}", sanitize(model)));
        let group = heading(&mut rows, &chip.name, model, dim());
        for sensor in readings(chip) {
            rows.push(Entry {
                group,
                priority: if sensor.primary { 0 } else { 3 },
                row: Row::Sensor(chip, sensor, sanitize(&sensor.label)),
            });
        }
    }
    for (kind, name) in [(Kind::Dimm, "Memory"), (Kind::Nic, "NIC")] {
        let chips = of(kind);
        let Some(first) = chips.first() else {
            continue;
        };
        let mut drivers = chips
            .iter()
            .map(|chip| sanitize(&chip.driver))
            .collect::<Vec<_>>();
        drivers.dedup();
        let detail = if kind == Kind::Dimm && chips.len() > 1 && drivers.len() == 1 {
            format!("  {} x{}", sanitize(&first.driver), chips.len())
        } else {
            format!("  {}", drivers.join(", "))
        };
        let group = heading(&mut rows, name, detail, dim());
        for chip in chips {
            for sensor in readings(chip) {
                rows.push(Entry {
                    group,
                    priority: 1,
                    row: Row::Sensor(chip, sensor, sanitize(&chip.sensor_name(sensor))),
                });
            }
        }
    }
    for chip in of(Kind::Board).into_iter().chain(of(Kind::Other)) {
        if readings(chip).next().is_none() {
            continue;
        }
        let (name, detail) = if chip.kind == Kind::Board {
            ("Board", format!("  {}", sanitize(&chip.driver)))
        } else {
            (chip.name.as_str(), String::new())
        };
        let group = heading(&mut rows, name, detail, dim());
        for sensor in readings(chip) {
            let priority = match (chip.kind, sensor.primary) {
                (Kind::Other, true) => 1,
                _ => 3,
            };
            rows.push(Entry {
                group,
                priority,
                row: Row::Sensor(chip, sensor, sanitize(&sensor.label)),
            });
        }
    }
    if host.space_at.is_some() {
        let (detail, style) = match stale(host, now) {
            Some(stale) => (format!("  {stale}"), Style::fg(Color::Yellow)),
            None => ("  used of each filesystem".to_owned(), dim()),
        };
        let group = heading(&mut rows, "Disks", detail, style);
        for filesystem in &host.filesystems {
            rows.push(Entry {
                group,
                priority: 2,
                row: Row::Disk(filesystem),
            });
        }
    }
    match host.rapl {
        Rapl::Absent => {}
        Rapl::NeedsRoot => {
            let group = heading(&mut rows, "Power", String::new(), dim());
            rows.push(Entry {
                group,
                priority: 4,
                row: Row::Note("CPU package: needs root (RAPL)"),
            });
        }
        Rapl::Readable => {
            let group = heading(&mut rows, "Power", "  RAPL".to_owned(), dim());
            rows.push(Entry {
                group,
                priority: 1,
                row: Row::Package(host.latest().and_then(|reading| reading.package)),
            });
        }
    }
    // A heading whose group has nothing to show goes too.
    let filled = rows
        .iter()
        .filter(|entry| !matches!(entry.row, Row::Heading(..)))
        .map(|entry| entry.group)
        .collect::<std::collections::HashSet<_>>();
    rows.retain(|entry| filled.contains(&entry.group));
    rows
}

/// Every sensor by chip down the right of the screen, then disk space and
/// power. Rows are given up by priority as the panel shortens, a heading
/// with its last row; past that the panel is cut.
fn side_panel(buffer: &mut Buffer, host: &Host, area: Area, now: f64) {
    let (x, y, width, height) = area;
    let count = format!("  {} on {} chips", host.sensors(), host.chips.len());
    boxed(
        buffer,
        area,
        &[("Sensors", title()), (&count, dim())],
        false,
    );
    let room = usize::from(height.saturating_sub(2));
    if room == 0 || width < 20 {
        return;
    }
    let end = x + width - 2;
    if host.chips.is_empty() && host.space_at.is_none() {
        let what = match host.status {
            HostStatus::Open => "no sensors in /sys/class/hwmon",
            _ => "-",
        };
        buffer.put(x + 2, y + 1, &pad(what, usize::from(width - 4)), dim());
        return;
    }
    let mut rows = side_rows(host, now);
    while rows.len() > room {
        let Some(highest) = rows
            .iter()
            .filter(|entry| !matches!(entry.row, Row::Heading(..)))
            .map(|entry| entry.priority)
            .max()
            .filter(|highest| *highest > 0)
        else {
            break;
        };
        let Some(position) = rows
            .iter()
            .rposition(|entry| !matches!(entry.row, Row::Heading(..)) && entry.priority == highest)
        else {
            break;
        };
        let group = rows[position].group;
        rows.remove(position);
        if !rows
            .iter()
            .any(|entry| entry.group == group && !matches!(entry.row, Row::Heading(..)))
        {
            rows.retain(|entry| entry.group != group);
        }
    }
    let cut = rows.len().saturating_sub(room);
    if cut > 0 {
        rows.truncate(room - 1);
    }
    let value = x + 15;
    for (offset, entry) in rows.iter().enumerate() {
        let line_y = y + 1 + offset as u16;
        match &entry.row {
            Row::Heading(kind, detail, style) => {
                let column = buffer.put(x + 2, line_y, &sanitize(kind), Style::fg(Color::Cyan));
                groups(
                    buffer,
                    column,
                    line_y,
                    &[vec![piece(detail.clone(), *style)]],
                    end,
                );
            }
            Row::Note(text) => {
                buffer.put(x + 4, line_y, &pad(text, usize::from(end - (x + 4))), dim());
            }
            Row::Sensor(chip, sensor, label) => {
                buffer.put(x + 4, line_y, &pad(label, 11), dim());
                sensor_value(buffer, host, (chip, sensor), (value, line_y, end));
            }
            Row::Disk(filesystem) => {
                buffer.put(x + 4, line_y, &pad(&sanitize(&filesystem.mount), 11), dim());
                gauge(
                    buffer,
                    value,
                    line_y,
                    end - value,
                    filesystem.share(),
                    level_color(host::fullness(filesystem.share())),
                    &space(filesystem.used, filesystem.total),
                );
            }
            Row::Package(watts) => {
                buffer.put(x + 4, line_y, &pad("package", 11), dim());
                let text = watts.map_or_else(|| "-".to_owned(), |watts| format!("{watts:.0} W"));
                buffer.put(value, line_y, &text, Style::PLAIN);
            }
        }
    }
    if cut > 0 {
        buffer.put(x + 2, y + room as u16, &format!("+{} more", cut + 1), dim());
    }
}

/// A sensor's value on the side panel: a 0-100C gauge for a temperature,
/// a speed for a fan, watts or volts.
fn sensor_value(
    buffer: &mut Buffer,
    host: &Host,
    (chip, sensor): (&Chip, &Sensor),
    (x, y, end): (u16, u16, u16),
) {
    let Some(value) = sensor.value else {
        buffer.put(x, y, "-", dim());
        return;
    };
    match sensor.quantity {
        Quantity::Celsius => {
            let level = temperature_level(sensor);
            gauge(
                buffer,
                x,
                y,
                end - x,
                value / 100.0,
                level_color(level),
                &right(&celsius(Some(value)), 4),
            );
        }
        Quantity::Rpm => {
            let stopped = value <= 0.0 && host.fan_shown(chip, sensor);
            groups(
                buffer,
                x,
                y,
                &[vec![
                    piece(
                        right(&integer(value), 6),
                        if stopped {
                            Style::fg(Color::Yellow)
                        } else {
                            Style::PLAIN
                        },
                    ),
                    piece(" rpm", dim()),
                ]],
                end,
            );
        }
        Quantity::Watts => {
            buffer.put(x, y, &format!("{value:.1} W"), Style::PLAIN);
        }
        Quantity::Volts => {
            buffer.put(x, y, &format!("{value:.2} V"), Style::PLAIN);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host::poll::Report;
    use crate::instances::Registry;

    fn app() -> App {
        App::new(Registry::new(1.0), 1.0, 15.0, vec!["--target"])
    }

    #[test]
    fn host_lines_follow_the_height_and_sensors_the_width() {
        let mut app = app();
        let mut registry = Registry::new(1.0);
        app.host.apply(
            Report::Polled {
                reading: Box::default(),
                chips: Vec::new(),
            },
            &mut registry,
            0.0,
        );
        // No GPUs: the tallest Host from 24 rows.
        assert_eq!(layout(132, 24, &app).lines.len(), 7);
        app.gpus.status = GpuStatus::Open;
        app.gpus.devices = vec![crate::gpu::Device::default(); 4];
        let wide = layout(200, 50, &app);
        assert_eq!(
            wide.lines,
            [
                Line::Cpu,
                Line::Memory,
                Line::Cores,
                Line::Load,
                Line::Io,
                Line::Heat
            ]
        );
        assert_eq!(wide.side, Some((152, 2, 48, 47)));
        assert_eq!(wide.sensors, None);
        let middle = layout(132, 38, &app);
        assert_eq!(
            middle.lines,
            [Line::Cpu, Line::Memory, Line::Cores, Line::Load, Line::Io]
        );
        assert_eq!(middle.sensors, Some((92, 2, 40, 7)));
        assert!(layout(132, 52, &app).lines.contains(&Line::Space));
        let narrow = layout(80, 24, &app);
        assert_eq!(narrow.lines, [Line::Cpu, Line::Memory]);
        assert_eq!((narrow.sensors, narrow.side), (None, None));
        assert_eq!(
            layout(90, 30, &app).lines,
            [Line::Cpu, Line::Memory, Line::Heat]
        );
        assert!(layout(60, 16, &app).compact);
        // The table follows Host, and takes the rows no panel can use.
        assert_eq!(middle.table, Some((0, 9, 132, 7)));
        assert_eq!(layout(60, 16, &app).table, Some((0, 5, 60, 7)));
    }
}
