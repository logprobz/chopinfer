//! The GPU view: this host's GPUs, then the selected one's history, where
//! its time goes, its processes and the GPU events.

use super::buffer::{Buffer, Color, Style};
use super::list::{Column, TABLE_TOP, first_row, layout, table_rows};
use super::widgets::{
    age, boxed, buckets, bytes, gauge, groups, integer, nice_ceiling, pad, percent, right,
    sanitize, sparkline,
};
use super::{Figures, footer, header, header_right, shown_name, tone_color};
use crate::app::App;
use crate::gpu::{Field, Reading, Reasons, Status};
use crate::instances::Instance;

const GIB: f64 = 1_073_741_824.0;

const COLUMNS: [Column; 12] = [
    Column {
        title: "GPU",
        width: 3,
        right: false,
        gap: 2,
        drop: 0,
    },
    Column {
        title: "NAME",
        width: 18,
        right: false,
        gap: 2,
        drop: 6,
    },
    Column {
        title: "TEMP",
        width: 4,
        right: true,
        gap: 1,
        drop: 0,
    },
    Column {
        title: "FAN",
        width: 4,
        right: true,
        gap: 1,
        drop: 5,
    },
    Column {
        title: "POWER",
        width: 8,
        right: true,
        gap: 1,
        drop: 2,
    },
    Column {
        title: "SM MHZ",
        width: 6,
        right: true,
        gap: 1,
        drop: 3,
    },
    Column {
        title: "MEM MHZ",
        width: 7,
        right: true,
        gap: 1,
        drop: 7,
    },
    Column {
        title: "VRAM",
        width: 10,
        right: true,
        gap: 1,
        drop: 0,
    },
    Column {
        title: "UTIL",
        width: 4,
        right: true,
        gap: 1,
        drop: 0,
    },
    Column {
        title: "TENSOR",
        width: 6,
        right: true,
        gap: 1,
        drop: 4,
    },
    Column {
        title: "DRAM",
        width: 4,
        right: true,
        gap: 2,
        drop: 4,
    },
    Column {
        title: "INSTANCE",
        width: 18,
        right: false,
        gap: 2,
        drop: 1,
    },
];

type Area = (u16, u16, u16, u16);
type Pieces = Vec<(String, Style)>;

fn dim() -> Style {
    Style::fg(Color::Dim)
}

fn title() -> Style {
    Style::fg(Color::Cyan).bold()
}

fn piece(text: impl Into<String>, style: Style) -> (String, Style) {
    (text.into(), style)
}

fn share(value: Option<f64>) -> String {
    percent(value)
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
    let gpus = &app.gpus;
    if gpus.status != Status::Open || gpus.devices.is_empty() {
        message(buffer, &gpus.status);
        return;
    }
    let (width, height) = (buffer.width, buffer.height);
    let top = TABLE_TOP + table(buffer, app);
    let left = height.saturating_sub(1 + top);
    if left < 6 {
        return;
    }
    let mut upper = if left >= 25 {
        (left - 12).min(16)
    } else if left >= 20 {
        13
    } else {
        left
    };
    if left - upper < 6 {
        upper = left;
    }
    let index = app.gpu.min(gpus.devices.len() - 1);
    if width >= 100 {
        let split = (u32::from(width) * 55 / 100) as u16;
        summary(buffer, app, index, (0, top, split, upper));
        time(buffer, app, index, (split, top, width - split, upper), now);
    } else {
        time(buffer, app, index, (0, top, width, upper), now);
    }
    let lower = left - upper;
    if lower >= 6 {
        let area = (0, top + upper, width, lower);
        if width >= 100 {
            let split = (u32::from(width) * 45 / 100) as u16;
            processes(buffer, app, index, (0, top + upper, split, lower));
            events(buffer, app, (split, top + upper, width - split, lower), now);
        } else {
            processes(buffer, app, index, area);
        }
    }
}

/// The header after the program name: the count, slowdowns and lost GPUs,
/// then the driver and GPM state when they leave room for the window.
fn crumbs(app: &App, width: u16) -> Vec<(String, Style)> {
    let gpus = &app.gpus;
    let mut crumbs = vec![
        piece("  >  ", dim()),
        piece("GPUs", Style::fg(Color::White).bold()),
    ];
    if gpus.status != Status::Open {
        return crumbs;
    }
    let latest = (0..gpus.devices.len())
        .filter_map(|index| gpus.latest(index))
        .collect::<Vec<_>>();
    crumbs.push(piece(
        format!("   {} on this host", gpus.devices.len()),
        Style::PLAIN,
    ));
    let slow = latest
        .iter()
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
    let gpm = if !gpus.counters {
        "off"
    } else if gpus.gpm() {
        "on"
    } else {
        "unsupported"
    };
    let used = 14
        + crumbs
            .iter()
            .map(|(text, _)| text.chars().count())
            .sum::<usize>();
    let room = usize::from(width).saturating_sub(header_right(app).chars().count() + 2);
    let driver = gpus
        .driver
        .as_deref()
        .map_or_else(|| "-".to_owned(), sanitize);
    for extra in [
        format!("   driver {driver}   GPM {gpm}"),
        format!("   GPM {gpm}"),
    ] {
        if used + extra.chars().count() <= room {
            crumbs.push(piece(extra, dim()));
            break;
        }
    }
    crumbs
}

fn message(buffer: &mut Buffer, status: &Status) {
    let width = buffer.width;
    boxed(buffer, (0, TABLE_TOP, width, 8), &[("GPUs", title())], true);
    let lines = match status {
        Status::Off => vec![piece(
            "GPUs are not read: chopinfer-top was started with --no-gpus.",
            dim(),
        )],
        Status::Opening => vec![piece("Opening NVML.", dim())],
        Status::Unavailable(reason) => vec![
            piece(
                format!("No GPU telemetry: {}.", sanitize(reason)),
                Style::fg(Color::Yellow),
            ),
            piece(
                "chopinfer-top shows the GPUs of the host it runs on, through the NVIDIA driver.",
                dim(),
            ),
            piece(
                "In a container, the utility driver capability provides it; the image sets it.",
                dim(),
            ),
            piece(
                "Instances on other hosts are still watched; their GPUs are not shown.",
                dim(),
            ),
        ],
        Status::Open => vec![piece("NVML lists no GPUs on this host.", dim())],
    };
    for (line, (text, style)) in lines.iter().enumerate() {
        buffer.put(
            2,
            TABLE_TOP + 2 + line as u16,
            &pad(text, usize::from(width) - 4),
            *style,
        );
    }
}

/// The instance on GPU `index`: the row of its first process that is one.
fn instance_on(app: &App, index: usize) -> Option<&Instance> {
    app.gpus
        .processes(index, &app.registry.instances)
        .into_iter()
        .find_map(|(_, row)| row)
        .and_then(|id| app.registry.get(id))
}

fn instance_name(instance: &Instance) -> String {
    shown_name(instance).unwrap_or_else(|| sanitize(&instance.label()))
}

fn temperature_style(reading: &Reading) -> Style {
    let thermal = reading.reasons.is_some_and(Reasons::thermal);
    match reading.headroom {
        _ if thermal => Style::fg(Color::Red),
        Some(headroom) if headroom <= 0 => Style::fg(Color::Red),
        Some(headroom) if headroom < 10 => Style::fg(Color::Yellow),
        _ => Style::PLAIN,
    }
}

fn clock_style(reading: &Reading) -> Style {
    if reading.reasons.is_some_and(Reasons::slowdown) {
        Style::fg(Color::Red)
    } else {
        Style::PLAIN
    }
}

/// `87.3/95.6 GiB` without the unit's suffix past `G`, whole GiB from 100.
fn memory(used: u64, total: u64) -> String {
    let (used, total) = (used as f64 / GIB, total as f64 / GIB);
    if total >= 100.0 {
        format!("{used:.0}/{total:.0}")
    } else {
        format!("{used:.1}/{total:.1}")
    }
}

/// Draws the GPU table and returns its height.
fn table(buffer: &mut Buffer, app: &App) -> u16 {
    let gpus = &app.gpus;
    let (width, height) = (buffer.width, buffer.height);
    let count = gpus.devices.len();
    let visible = table_rows(count, height);
    let first = first_row(app.gpu_scroll, app.gpu, visible);
    let (shown, spark) = layout(&COLUMNS, usize::from(width) - 4);
    let position = if count > visible {
        format!(
            "  rows {}-{} of {count}",
            first + 1,
            (first + visible).min(count)
        )
    } else {
        String::new()
    };
    let table_height = visible as u16 + 3;
    boxed(
        buffer,
        (0, TABLE_TOP, width, table_height),
        &[
            ("GPUs", title()),
            (&position, Style::fg(Color::Yellow)),
            ("  rates and shares over the last 3 polls", dim()),
        ],
        true,
    );
    let mut column = 2;
    for index in &shown {
        let spec = &COLUMNS[*index];
        let text = if spec.right {
            right(spec.title, spec.width)
        } else {
            pad(spec.title, spec.width)
        };
        column = buffer.put(column, TABLE_TOP + 1, &text, dim().bold()) + spec.gap as u16;
    }
    if spark > 0 {
        buffer.put(column, TABLE_TOP + 1, &pad("UTIL 5m", spark), dim().bold());
    }
    for (line, index) in (first..count).take(visible).enumerate() {
        row(
            buffer,
            app,
            index,
            TABLE_TOP + 2 + line as u16,
            (&shown, spark),
        );
    }
    table_height
}

fn row(buffer: &mut Buffer, app: &App, index: usize, y: u16, (shown, spark): (&[usize], usize)) {
    let gpus = &app.gpus;
    let device = &gpus.devices[index];
    let history = &gpus.histories[index];
    if index == app.gpu {
        buffer.background(1, y, buffer.width - 2, Color::Selection);
        buffer.put(1, y, "▌", Style::fg(Color::Cyan));
    }
    let plain = Style::PLAIN;
    let latest = history.latest();
    // A lost GPU's averages would still show the polls before it was lost.
    let lost = gpus.lost(index);
    let recent = |field: Field| history.recent(field).filter(|_| !lost);
    let number = |value: Option<String>, style: Style| {
        value.map_or_else(|| piece("-", dim()), |text| (text, style))
    };
    let shares = |field: Field| number(recent(field).map(|value| share(Some(value))), plain);
    let processes = gpus.processes(index, &app.registry.instances);
    let instance = match instance_on(app, index) {
        Some(instance) => {
            let others = processes.len().saturating_sub(1);
            let name = instance_name(instance);
            let text = if others > 0 {
                format!("{name} +{others}")
            } else {
                name
            };
            (text, Style::fg(Color::White).bold())
        }
        None if lost => piece("lost: no answer", Style::fg(Color::Red)),
        // Why NVML gave no handle: no permission, not found.
        None if device.unreadable.is_some() => piece(
            device.unreadable.as_deref().unwrap_or_default(),
            Style::fg(Color::Yellow),
        ),
        None if !processes.is_empty() => piece(
            format!(
                "{} process{}",
                processes.len(),
                if processes.len() == 1 { "" } else { "es" }
            ),
            dim(),
        ),
        None => piece("-", dim()),
    };
    let cells = [
        (device.index.to_string(), plain),
        (sanitize(&device.short_name()), plain),
        number(
            latest
                .and_then(|reading| reading.temperature)
                .map(|celsius| format!("{celsius}C")),
            latest.map_or(plain, temperature_style),
        ),
        number(
            latest
                .and_then(|reading| reading.fan)
                .map(|fan| format!("{fan}%")),
            plain,
        ),
        number(
            recent(|reading| reading.power).map(|watts| {
                match latest.and_then(|reading| reading.power_limit) {
                    Some(limit) => format!("{watts:.0}/{limit:.0}W"),
                    None => format!("{watts:.0}W"),
                }
            }),
            plain,
        ),
        number(
            latest
                .and_then(|reading| reading.sm_mhz)
                .map(|mhz| integer(f64::from(mhz))),
            latest.map_or(plain, clock_style),
        ),
        number(
            latest
                .and_then(|reading| reading.memory_mhz)
                .map(|mhz| integer(f64::from(mhz))),
            plain,
        ),
        match (
            latest.and_then(|reading| reading.memory_used),
            device.memory_total,
        ) {
            (Some(used), Some(total)) if total > 0 => (
                format!("{}G", memory(used, total)),
                if used as f64 > 0.95 * total as f64 {
                    Style::fg(Color::Red)
                } else {
                    plain
                },
            ),
            _ => piece("-", dim()),
        },
        shares(Reading::kernels),
        shares(|reading| reading.activity.and_then(|activity| activity.tensor)),
        shares(|reading| reading.activity.and_then(|activity| activity.dram)),
        instance,
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
    if spark > 0 && latest.is_some() {
        let columns = buckets(
            &history.series(Reading::kernels),
            history.oldest_poll(),
            history.capacity(),
            spark,
        );
        sparkline(
            buffer,
            (column, y, spark as u16, 1),
            &columns,
            Some(1.0),
            Color::Cyan,
        );
    }
}

/// The selected GPU: its activity and power over the last minutes, then
/// memory, temperature, clocks and their limit, and the PCIe link.
fn summary(buffer: &mut Buffer, app: &App, index: usize, area: Area) {
    let (x, y, width, height) = area;
    let device = &app.gpus.devices[index];
    let history = &app.gpus.histories[index];
    let instance = instance_on(app, index);
    let name = instance.map_or_else(String::new, |instance| {
        format!("  {}", instance_name(instance))
    });
    let label = format!("GPU {}", device.index);
    let short = format!("  {}", sanitize(&device.short_name()));
    let hint = "  Enter for details";
    let length = [&label, &short, &name, hint]
        .iter()
        .map(|text| text.chars().count())
        .sum::<usize>();
    boxed(
        buffer,
        area,
        &[
            (&label, title()),
            (&short, Style::fg(Color::White).bold()),
            (&name, Style::PLAIN),
            (
                if instance.is_some() && length + 6 <= usize::from(width) {
                    hint
                } else {
                    ""
                },
                dim(),
            ),
        ],
        false,
    );
    let Some(reading) = history.latest() else {
        buffer.put(x + 2, y + 1, "waiting for the first reading", dim());
        return;
    };
    if height < 6 {
        return;
    }
    let inner = height - 2;
    let info = inner.saturating_sub(4).min(5);
    let plot = inner.saturating_sub(3 + info).max(1);
    let half = (width - 6) / 2;
    let power_series = history.series(|reading| reading.power);
    let power_top = reading
        .power_limit
        .unwrap_or_else(|| nice_ceiling(power_series.iter().copied().fold(0.0, f64::max)));
    let plots = [
        Plot {
            offset: 0,
            label: "ACTIVITY",
            field: Reading::kernels,
            top: 1.0,
            scale: "100%".to_owned(),
            color: Color::Cyan,
            unit: "",
            format: |value| share(Some(value)),
        },
        Plot {
            offset: half + 2,
            label: "POWER",
            field: |reading| reading.power,
            top: power_top,
            scale: format!("{} W", integer(power_top)),
            color: Color::Magenta,
            unit: "  W",
            format: integer,
        },
    ];
    for Plot {
        offset,
        label,
        field,
        top,
        scale,
        color,
        unit,
        format,
    } in plots
    {
        let column = x + 2 + offset;
        let series = history.series(field);
        let recent = &series[series.len().saturating_sub(usize::from(half))..];
        let average =
            (!recent.is_empty()).then(|| recent.iter().sum::<f64>() / recent.len() as f64);
        let maximum = recent.iter().copied().fold(0.0_f64, f64::max);
        let shown = |value: Option<f64>| right(&value.map_or_else(|| "-".to_owned(), format), 4);
        groups(
            buffer,
            column,
            y + 1,
            &[
                vec![
                    piece(label, dim().bold()),
                    piece("  now ", dim()),
                    piece(shown(history.recent(field)), Style::fg(Color::White).bold()),
                ],
                vec![piece("  avg ", dim()), piece(shown(average), Style::PLAIN)],
                vec![
                    piece("  max ", dim()),
                    piece(shown(Some(maximum)), Style::PLAIN),
                ],
                vec![piece(unit, dim())],
            ],
            column + half,
        );
        if top > 0.0 {
            buffer.put(column, y + 2, &right(&scale, usize::from(half)), dim());
        }
        sparkline(
            buffer,
            (column, y + 3, half, plot),
            recent,
            Some(top),
            color,
        );
        let since = format!("-{} ", age(f64::from(half) * app.interval));
        let rule = usize::from(half).saturating_sub(since.chars().count() + 4);
        buffer.put(
            column,
            y + 3 + plot,
            &format!("{since}{} now", "▔".repeat(rule)),
            dim(),
        );
    }
    let (label, value) = (x + 2, x + 2 + 13);
    let end = x + width - 2;
    let room = width.saturating_sub(4 + 13);
    let lines = info_lines(app, index, reading);
    for (line, (name, pieces)) in lines.into_iter().take(usize::from(info)).enumerate() {
        let line = y + 4 + plot + line as u16;
        buffer.put(label, line, name, dim());
        match pieces {
            Info::Gauge(ratio, color, text) => {
                gauge(buffer, value, line, room, ratio, color, &text);
            }
            Info::Groups(pieces) => {
                groups(buffer, value, line, &pieces, end);
            }
        }
    }
}

/// One of the summary's two plots: where it is, what it shows and to what
/// scale.
struct Plot {
    offset: u16,
    label: &'static str,
    field: Field,
    top: f64,
    scale: String,
    color: Color,
    unit: &'static str,
    format: fn(f64) -> String,
}

enum Info {
    Gauge(f64, Color, String),
    Groups(Vec<Pieces>),
}

fn info_lines(app: &App, index: usize, reading: &Reading) -> Vec<(&'static str, Info)> {
    let device = &app.gpus.devices[index];
    let history = &app.gpus.histories[index];
    let dash = || Info::Groups(vec![vec![piece("-", dim())]]);
    let mut lines = Vec::new();
    lines.push((
        "VRAM",
        match (reading.memory_used, device.memory_total) {
            (Some(used), Some(total)) if total > 0 => {
                let ratio = used as f64 / total as f64;
                Info::Gauge(
                    ratio,
                    if ratio > 0.95 {
                        Color::Red
                    } else {
                        Color::Green
                    },
                    format!("{} GiB", memory(used, total)),
                )
            }
            _ => dash(),
        },
    ));
    let temperature_style = temperature_style(reading);
    let mut temperature = vec![vec![piece(
        reading
            .temperature
            .map_or_else(|| "-".to_owned(), |celsius| format!("{celsius}C")),
        temperature_style.bold(),
    )]];
    if let Some(headroom) = reading.headroom {
        temperature.push(vec![piece(
            format!("  {}C under the limit", headroom.max(0)),
            if headroom < 10 {
                temperature_style
            } else {
                dim()
            },
        )]);
    }
    if let Some(fan) = reading.fan {
        temperature.push(vec![
            piece("   fan ", dim()),
            piece(format!("{fan}%"), Style::PLAIN),
        ]);
    }
    lines.push(("temperature", Info::Groups(temperature)));
    let mut clocks = vec![vec![
        piece("SM ", dim()),
        piece(
            reading
                .sm_mhz
                .map_or_else(|| "-".to_owned(), |mhz| integer(f64::from(mhz))),
            clock_style(reading),
        ),
        piece(
            device.sm_max_mhz.map_or_else(
                || " MHz".to_owned(),
                |max| format!(" of {} MHz", integer(f64::from(max))),
            ),
            dim(),
        ),
    ]];
    if let Some(mhz) = reading.memory_mhz {
        clocks.push(vec![
            piece("   memory ", dim()),
            piece(integer(f64::from(mhz)), Style::PLAIN),
            piece(" MHz", dim()),
        ]);
    }
    if let Some(pstate) = reading.pstate {
        clocks.push(vec![piece(format!("   P{pstate}"), Style::PLAIN)]);
    }
    lines.push(("clocks", Info::Groups(clocks)));
    let reasons = reading.reasons.unwrap_or_default();
    let limited = match reasons.label() {
        _ if reading.reasons.is_none() => vec![vec![piece("-", dim())]],
        Some(label) if reasons.slowdown() => vec![vec![piece(label, Style::fg(Color::Red).bold())]],
        Some("power cap") => {
            let mut pieces = vec![vec![piece("power cap", Style::PLAIN)]];
            if let Some(limit) = reading.power_limit {
                pieces.push(vec![piece(format!("  at {limit:.0} W"), dim())]);
            }
            pieces
        }
        Some(label) => vec![vec![piece(label, dim())]],
        None => vec![vec![piece("nothing", dim())]],
    };
    lines.push(("limited by", Info::Groups(limited)));
    let mut link = vec![vec![piece(
        reading.pcie.map_or_else(
            || "-".to_owned(),
            |(generation, width)| format!("gen {generation} x{width}"),
        ),
        Style::PLAIN,
    )]];
    if let (Some(current), Some((generation, width))) = (reading.pcie, device.pcie_max)
        && current < (generation, width)
    {
        let busy = history
            .recent(Reading::kernels)
            .is_some_and(|kernels| kernels > 0.5);
        link.push(vec![if busy {
            piece(
                format!("  below gen {generation} x{width}"),
                Style::fg(Color::Yellow),
            )
        } else {
            piece(format!("  (gen {generation} x{width} under load)"), dim())
        }]);
    }
    link.push(vec![
        piece("   ECC ", dim()),
        match (device.ecc, reading.ecc_uncorrected) {
            (Some(true), Some(errors)) if errors > 0 => piece(
                format!("on, {errors} uncorrected"),
                Style::fg(Color::Red).bold(),
            ),
            (Some(true), _) => piece("on", Style::PLAIN),
            (Some(false), _) => piece("off", Style::PLAIN),
            (None, _) => piece("-", dim()),
        },
    ]);
    lines.push(("PCIe", Info::Groups(link)));
    lines
}

/// A line of the Time panel, and the order lines are given up when the
/// panel is short: the highest first.
enum Line {
    Bar,
    Legend,
    Load,
    Blank,
    Gauge(&'static str, Field, Color),
    Pcie,
    Note(&'static str),
}

/// Where the selected GPU's time goes: kernels against no kernel, the load
/// of its instance, and with GPM how busy its SMs, tensor pipes and DRAM
/// were.
fn time(buffer: &mut Buffer, app: &App, index: usize, area: Area, now: f64) {
    let (x, y, width, height) = area;
    if height < 3 {
        return;
    }
    let device = &app.gpus.devices[index];
    let history = &app.gpus.histories[index];
    boxed(
        buffer,
        area,
        &[
            ("Time", title()),
            (&format!("  GPU {}, last 3 polls", device.index), dim()),
        ],
        false,
    );
    let mut lines = vec![
        (0, Line::Bar),
        (0, Line::Legend),
        (1, Line::Load),
        (5, Line::Blank),
    ];
    if app.gpus.counters && device.gpm {
        lines.extend([
            (
                2,
                Line::Gauge(
                    "SM busy",
                    |reading| reading.activity.and_then(|activity| activity.sm),
                    Color::Cyan,
                ),
            ),
            (
                4,
                Line::Gauge(
                    "SM occupancy",
                    |reading| reading.activity.and_then(|activity| activity.occupancy),
                    Color::Blue,
                ),
            ),
            (
                2,
                Line::Gauge(
                    "tensor pipes",
                    |reading| reading.activity.and_then(|activity| activity.tensor),
                    Color::Magenta,
                ),
            ),
            (
                2,
                Line::Gauge(
                    "DRAM bandwidth",
                    |reading| reading.activity.and_then(|activity| activity.dram),
                    Color::Green,
                ),
            ),
            (3, Line::Pcie),
        ]);
    } else {
        lines.push((
            2,
            Line::Gauge(
                "memory busy",
                |reading| reading.utilization.map(|(_, memory)| memory),
                Color::Green,
            ),
        ));
        lines.push((
            3,
            Line::Note(if app.gpus.counters {
                "SM, tensor and DRAM figures need GPM: Hopper or newer."
            } else {
                "GPM counters are off (--no-gpu-counters)."
            }),
        ));
    }
    let room = usize::from(height - 2);
    while lines.len() > room {
        let Some(highest) = lines.iter().map(|(priority, _)| *priority).max() else {
            break;
        };
        let Some(position) = lines.iter().rposition(|(priority, _)| *priority == highest) else {
            break;
        };
        lines.remove(position);
    }
    let (label, value) = (x + 2, x + 2 + 15);
    let end = x + width - 2;
    let room = width.saturating_sub(4 + 15);
    let kernels = history.recent(Reading::kernels);
    for (offset, (_, line)) in lines.iter().enumerate() {
        let line_y = y + 1 + offset as u16;
        match line {
            Line::Bar => {
                buffer.put(label, line_y, "wall time", dim());
                let filled = kernels.map_or(0, |kernels| {
                    (kernels.clamp(0.0, 1.0) * f64::from(room)).round() as usize
                });
                let column = buffer.put(value, line_y, &"█".repeat(filled), Style::fg(Color::Cyan));
                buffer.put(
                    column,
                    line_y,
                    &"─".repeat(usize::from(room).saturating_sub(filled)),
                    dim(),
                );
            }
            Line::Legend => {
                groups(
                    buffer,
                    value,
                    line_y,
                    &[
                        vec![piece("kernels ", dim()), piece(share(kernels), title())],
                        vec![
                            piece("  no kernel ", dim()),
                            piece(share(kernels.map(|kernels| 1.0 - kernels)), Style::PLAIN),
                        ],
                        vec![piece("  all processes", dim())],
                    ],
                    end,
                );
            }
            Line::Load => {
                buffer.put(label, line_y, "instance load", dim());
                load(buffer, app, index, (value, line_y), end, now);
            }
            Line::Blank => {}
            Line::Gauge(name, field, color) => {
                buffer.put(label, line_y, name, dim());
                let level = history.recent(*field);
                gauge(
                    buffer,
                    value,
                    line_y,
                    room,
                    level.unwrap_or(0.0),
                    *color,
                    &right(&share(level), 4),
                );
            }
            Line::Pcie => {
                buffer.put(label, line_y, "PCIe", dim());
                let rate = |field: Field| {
                    history
                        .recent(field)
                        .map_or_else(|| "-".to_owned(), |rate| format!("{}/s", bytes(rate)))
                };
                groups(
                    buffer,
                    value,
                    line_y,
                    &[
                        vec![
                            piece("rx ", dim()),
                            piece(
                                rate(|reading| {
                                    reading.activity.and_then(|activity| activity.pcie_rx)
                                }),
                                Style::PLAIN,
                            ),
                        ],
                        vec![
                            piece("   tx ", dim()),
                            piece(
                                rate(|reading| {
                                    reading.activity.and_then(|activity| activity.pcie_tx)
                                }),
                                Style::PLAIN,
                            ),
                        ],
                    ],
                    end,
                );
            }
            Line::Note(text) => {
                buffer.put(
                    label,
                    line_y,
                    &pad(text, usize::from(end.saturating_sub(label))),
                    dim(),
                );
            }
        }
    }
}

/// The running and waiting requests of the instance on GPU `index`.
fn load(buffer: &mut Buffer, app: &App, index: usize, (x, y): (u16, u16), end: u16, now: f64) {
    let Some(instance) = instance_on(app, index) else {
        buffer.put(
            x,
            y,
            &pad(
                "no chopinfer instance on this GPU",
                usize::from(end.saturating_sub(x)),
            ),
            dim(),
        );
        return;
    };
    let figures = Figures::new(instance, app.window);
    if !figures.live() {
        let name = vec![piece(
            instance_name(instance),
            Style::fg(Color::White).bold(),
        )];
        let state = instance.state(now);
        groups(
            buffer,
            x,
            y,
            &[
                name,
                vec![piece(
                    format!("  {}", state.label()),
                    Style::fg(tone_color(state.tone())),
                )],
            ],
            end,
        );
        return;
    }
    let count = |value: Option<f64>| value.map_or_else(|| "-".to_owned(), integer);
    let waiting = figures.waiting().unwrap_or(0.0);
    groups(
        buffer,
        x,
        y,
        &[
            vec![
                piece(count(figures.running()), Style::PLAIN),
                piece(" running", dim()),
            ],
            vec![
                piece("  ", dim()),
                piece(
                    count(figures.waiting()),
                    if waiting > 0.0 {
                        Style::fg(Color::Yellow)
                    } else {
                        Style::PLAIN
                    },
                ),
                piece(" waiting", dim()),
            ],
            vec![piece(format!("  {}", instance_name(instance)), dim())],
        ],
        end,
    );
}

/// The processes on GPU `index` and the instance each one is.
fn processes(buffer: &mut Buffer, app: &App, index: usize, area: Area) {
    let (x, y, width, height) = area;
    if height < 4 {
        return;
    }
    let device = &app.gpus.devices[index];
    boxed(
        buffer,
        area,
        &[
            ("Processes", title()),
            (&format!("  on GPU {}", device.index), dim()),
        ],
        false,
    );
    let end = x + width - 2;
    buffer.put(
        x + 2,
        y + 1,
        &pad(
            &format!(
                "{}{}{}  PROCESS",
                pad("PID", 8),
                pad("TYPE", 5),
                right("VRAM", 9)
            ),
            usize::from(end.saturating_sub(x + 2)),
        ),
        dim().bold(),
    );
    let processes = app.gpus.processes(index, &app.registry.instances);
    if processes.is_empty() {
        buffer.put(x + 2, y + 2, "no processes", dim());
        return;
    }
    let serving = processes.iter().any(|(_, row)| row.is_some());
    let room = usize::from(height - 3);
    let shown = if processes.len() > room {
        room - 1
    } else {
        processes.len()
    };
    for (line, (process, row)) in processes.iter().take(shown).enumerate() {
        let line_y = y + 2 + line as u16;
        let column = buffer.put(
            x + 2,
            line_y,
            &pad(&process.pid.to_string(), 8),
            Style::PLAIN,
        );
        let column = buffer.put(column, line_y, &pad(process.kind.label(), 5), dim());
        let memory = process.memory.map_or_else(
            || "-".to_owned(),
            |bytes| format!("{:.1} GiB", bytes as f64 / GIB),
        );
        let column = buffer.put(
            column,
            line_y,
            &format!("{}  ", right(&memory, 9)),
            Style::PLAIN,
        );
        let room = usize::from(end.saturating_sub(column));
        // The instance a process is, else its command, marked when it
        // shares a GPU with an instance.
        let (name, style, note) = match row.and_then(|id| app.registry.get(id)) {
            Some(instance) => (
                instance_name(instance),
                Style::fg(Color::White).bold(),
                format!("  {}", sanitize(&instance.label())),
            ),
            None => (
                process.owner.name.clone().unwrap_or_else(|| "-".to_owned()),
                if serving {
                    Style::fg(Color::Yellow)
                } else {
                    Style::PLAIN
                },
                "  not an instance".to_owned(),
            ),
        };
        let column = buffer.put(
            column,
            line_y,
            &name.chars().take(room).collect::<String>(),
            style,
        );
        groups(buffer, column, line_y, &[vec![piece(note, dim())]], end);
    }
    if processes.len() > shown {
        buffer.put(
            x + 2,
            y + 2 + shown as u16,
            &format!("+{} more", processes.len() - shown),
            dim(),
        );
    }
}

/// The events about GPUs, the latest last.
fn events(buffer: &mut Buffer, app: &App, area: Area, now: f64) {
    let (x, y, width, height) = area;
    if height < 3 {
        return;
    }
    boxed(buffer, area, &[("GPU events", title())], false);
    let events = app
        .registry
        .events
        .iter()
        .filter(|event| event.who.starts_with("GPU"))
        .collect::<Vec<_>>();
    if events.is_empty() {
        buffer.put(x + 2, y + 1, "none", dim());
        return;
    }
    let room = usize::from(height - 2);
    let end = x + width - 2;
    for (line, event) in events[events.len().saturating_sub(room)..]
        .iter()
        .enumerate()
    {
        let line_y = y + 1 + line as u16;
        let when = format!("{} ago", age(now - event.at));
        let column = buffer.put(x + 1, line_y, &right(&when, 10), dim());
        let column = buffer.put(
            column + 2,
            line_y,
            &pad(&sanitize(&event.who), 7),
            Style::PLAIN,
        );
        buffer.put(
            column,
            line_y,
            &pad(
                &sanitize(&event.what),
                usize::from(end.saturating_sub(column)),
            ),
            Style::fg(tone_color(event.tone)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrow_terminals_drop_gpu_columns_in_order() {
        let titles = |room| {
            let (shown, spark) = layout(&COLUMNS, room);
            let titles = shown
                .iter()
                .map(|index| COLUMNS[*index].title)
                .collect::<Vec<_>>();
            (titles, spark)
        };
        let (all, spark) = titles(128);
        assert_eq!((all.len(), spark), (COLUMNS.len(), 20));
        let (shown, _) = titles(96);
        assert!(!shown.contains(&"MEM MHZ") && !shown.contains(&"NAME"));
        let (shown, _) = titles(76);
        assert_eq!(
            shown,
            [
                "GPU", "TEMP", "POWER", "SM MHZ", "VRAM", "UTIL", "TENSOR", "DRAM", "INSTANCE"
            ]
        );
        // Temperature, memory and use stay at any width.
        let (shown, _) = titles(20);
        assert_eq!(shown, ["GPU", "TEMP", "VRAM", "UTIL"]);
    }

    #[test]
    fn memory_drops_decimals_from_100_gib() {
        assert_eq!(memory(87 << 30, 97_887 << 20), "87.0/95.6");
        assert_eq!(memory(130 << 30, 141 << 30), "130/141");
    }
}
