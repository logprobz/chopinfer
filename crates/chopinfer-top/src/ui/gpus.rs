//! The monitor's GPU panels: this host's GPUs, then the selected one's
//! history, where its time goes, its processes and the host and GPU events.

use super::buffer::{Buffer, Color, Style};
use super::list::{Column, SPARK_MINIMUM, first_row, layout};
use super::widgets::{
    age, boxed, buckets, bytes, gauge, groups, integer, nice_ceiling, pad, percent, right,
    sanitize, sparkline,
};
use super::{Figures, shown_name, tone_color};
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
/// The INSTANCE column, which widens into the sparkline's spare cells.
const INSTANCE: usize = 11;

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

/// The GPU table's height in an area `height` rows tall: a line a GPU with
/// its borders and titles, leaving room for the panels below, or taking
/// the rows they cannot use when none fits.
pub fn table_height(rows: usize, height: u16) -> u16 {
    let wanted = u16::try_from(rows.saturating_add(3)).unwrap_or(u16::MAX);
    let table = wanted.clamp(4, height.saturating_sub(5).max(4));
    if height.saturating_sub(table) < 6 {
        wanted.clamp(4, height.max(4))
    } else {
        table
    }
}

/// The GPU table in `area`, then the selected GPU's history and where its
/// time goes, then its processes and the events, as room allows.
pub fn panels(buffer: &mut Buffer, app: &App, area: Area, now: f64) {
    let gpus = &app.gpus;
    let (x, y, width, height) = area;
    if gpus.status != Status::Open || gpus.devices.is_empty() {
        message(buffer, &gpus.status, area);
        if height >= 12 {
            events(buffer, app, (x, y + 8, width, height - 8), now);
        }
        return;
    }
    let top = y + table(buffer, app, area);
    let left = (y + height).saturating_sub(top);
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
        summary(buffer, app, index, (x, top, split, upper));
        time(
            buffer,
            app,
            index,
            (x + split, top, width - split, upper),
            now,
        );
    } else {
        time(buffer, app, index, (x, top, width, upper), now);
    }
    let lower = left - upper;
    if lower >= 6 {
        let area = (x, top + upper, width, lower);
        if width >= 100 {
            let split = (u32::from(width) * 45 / 100) as u16;
            processes(buffer, app, index, (x, top + upper, split, lower), now);
            events(
                buffer,
                app,
                (x + split, top + upper, width - split, lower),
                now,
            );
        } else {
            processes(buffer, app, index, area, now);
        }
    }
}

/// Why there are no GPUs to show, in a box at the top of `area`.
fn message(buffer: &mut Buffer, status: &Status, area: Area) {
    let (x, y, width, _) = area;
    boxed(buffer, (x, y, width, 8), &[("GPUs", title())], true);
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
            x + 2,
            y + 2 + line as u16,
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

/// The INSTANCE cell of GPU `index`: its text, how many other processes
/// share the GPU (` +1`, else empty), and its style.
fn instance_cell(app: &App, index: usize) -> (String, String, Style) {
    let gpus = &app.gpus;
    let device = &gpus.devices[index];
    let processes = gpus.processes(index, &app.registry.instances);
    // A lost GPU's processes are those it had before.
    match instance_on(app, index) {
        _ if gpus.lost(index) => (
            "lost: no answer".to_owned(),
            String::new(),
            Style::fg(Color::Red),
        ),
        Some(instance) => {
            let others = processes.len().saturating_sub(1);
            (
                instance_name(instance),
                if others > 0 {
                    format!(" +{others}")
                } else {
                    String::new()
                },
                Style::fg(Color::White).bold(),
            )
        }
        // Why NVML gave no handle: no permission, not found.
        None if device.unreadable.is_some() => (
            device.unreadable.clone().unwrap_or_default(),
            String::new(),
            Style::fg(Color::Yellow),
        ),
        None if !processes.is_empty() => (
            format!(
                "{} process{}",
                processes.len(),
                if processes.len() == 1 { "" } else { "es" }
            ),
            String::new(),
            dim(),
        ),
        None => ("-".to_owned(), String::new(), dim()),
    }
}

/// `text` then `suffix` in `width` cells, `text` cut so `suffix` stays.
fn fit(text: &str, suffix: &str, width: usize) -> String {
    let room = width.saturating_sub(suffix.chars().count());
    let text = text.chars().take(room).collect::<String>();
    pad(&format!("{text}{suffix}"), width)
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
pub(super) fn memory(used: u64, total: u64) -> String {
    let (used, total) = (used as f64 / GIB, total as f64 / GIB);
    if total >= 100.0 {
        format!("{used:.0}/{total:.0}")
    } else {
        format!("{used:.1}/{total:.1}")
    }
}

/// Draws the GPU table at the top of `area` and returns its height.
fn table(buffer: &mut Buffer, app: &App, area: Area) -> u16 {
    let gpus = &app.gpus;
    let (x, y, width, height) = area;
    let count = gpus.devices.len();
    let visible = usize::from(table_height(count, height) - 3);
    let first = first_row(app.gpu_scroll, app.gpu, visible);
    let (shown, mut spark) = layout(&COLUMNS, usize::from(width) - 4);
    // INSTANCE takes what the sparkline can spare to show its longest cell.
    let mut instance_width = COLUMNS[INSTANCE].width;
    if spark > 0 && shown.contains(&INSTANCE) {
        let widest = (0..count)
            .map(|index| {
                let (name, others, _) = instance_cell(app, index);
                name.chars().count() + others.chars().count()
            })
            .max()
            .unwrap_or(0);
        let grow = widest
            .saturating_sub(instance_width)
            .min(spark.saturating_sub(SPARK_MINIMUM));
        instance_width += grow;
        spark -= grow;
    }
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
        (x, y, width, table_height),
        &[
            ("GPUs", title()),
            (&position, Style::fg(Color::Yellow)),
            ("  rates and shares over the last 3 polls", dim()),
        ],
        true,
    );
    let mut column = x + 2;
    for index in &shown {
        let spec = &COLUMNS[*index];
        let column_width = if *index == INSTANCE {
            instance_width
        } else {
            spec.width
        };
        let text = if spec.right {
            right(spec.title, column_width)
        } else {
            pad(spec.title, column_width)
        };
        column = buffer.put(column, y + 1, &text, dim().bold()) + spec.gap as u16;
    }
    if spark > 0 {
        buffer.put(column, y + 1, &pad("UTIL 5m", spark), dim().bold());
    }
    for (line, index) in (first..count).take(visible).enumerate() {
        row(
            buffer,
            app,
            index,
            (x, y + 2 + line as u16, width),
            (&shown, spark, instance_width),
        );
    }
    table_height
}

/// GPU `index` on line `y` of a table `x` to `x + width`.
fn row(
    buffer: &mut Buffer,
    app: &App,
    index: usize,
    (x, y, width): (u16, u16, u16),
    (shown, spark, instance_width): (&[usize], usize, usize),
) {
    let gpus = &app.gpus;
    let device = &gpus.devices[index];
    let history = &gpus.histories[index];
    if index == app.gpu {
        buffer.background(x + 1, y, width - 2, Color::Selection);
        buffer.put(x + 1, y, "▌", Style::fg(Color::Cyan));
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
    let (name, others, style) = instance_cell(app, index);
    let instance = (fit(&name, &others, instance_width), style);
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
    let mut column = x + 2;
    for index in shown {
        let spec = &COLUMNS[*index];
        let (text, style) = &cells[*index];
        let column_width = if *index == INSTANCE {
            instance_width
        } else {
            spec.width
        };
        let text = if spec.right {
            right(text, column_width)
        } else {
            pad(text, column_width)
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

/// The processes on GPU `index` and the instance each one is, from an
/// earlier poll, and marked so, when this one's process lists failed.
fn processes(buffer: &mut Buffer, app: &App, index: usize, area: Area, now: f64) {
    let (x, y, width, height) = area;
    if height < 4 {
        return;
    }
    let device = &app.gpus.devices[index];
    let read = app.gpus.processes_read(index);
    let earlier = read
        .filter(|at| {
            app.gpus
                .latest(index)
                .is_some_and(|reading| reading.at > *at)
        })
        .map_or_else(String::new, |at| format!("  as read {} ago", age(now - at)));
    boxed(
        buffer,
        area,
        &[
            ("Processes", title()),
            (&format!("  on GPU {}", device.index), dim()),
            (&earlier, Style::fg(Color::Yellow)),
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
        let what = if read.is_some() {
            "no processes"
        } else {
            "processes not read"
        };
        buffer.put(x + 2, y + 2, what, dim());
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

/// The events about this host and its GPUs, the latest last.
fn events(buffer: &mut Buffer, app: &App, area: Area, now: f64) {
    let (x, y, width, height) = area;
    if height < 3 {
        return;
    }
    boxed(
        buffer,
        area,
        &[("Events", title()), ("  host and GPUs", dim())],
        false,
    );
    let events = app
        .registry
        .events
        .iter()
        .filter(|event| event.who.starts_with("GPU") || event.who == "host")
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
        assert_eq!(COLUMNS[INSTANCE].title, "INSTANCE");
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
