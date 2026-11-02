//! The model loader: Models, Profiles and Placement down the left, the
//! Options pane on the right, and the command line below. Below 110
//! columns the focused pane fills the space and Tab moves between them.

use super::buffer::{Buffer, Color, Style};
use super::widgets::{boxed, pad, right, sanitize, wrap};
use super::{footer, header};
use crate::app::App;
use crate::catalog::{Availability, Checkpoint, Fetch, Pick, Profile};
use crate::launch::command::{self, API_KEY, Origin};
use crate::launch::occupancy::{self, Holder, Occupancy};
use crate::loader::options::{From, Group, Line, Row, Tone};
use crate::loader::{FLASH_SECONDS, Loader, Pane, Target};

/// The narrowest terminal that shows every pane at once.
pub const WIDE: u16 = 110;
/// The left column's width.
const LEFT: u16 = 44;
/// The command box's height.
const COMMAND: u16 = 7;
/// Most recent launches shown.
const RECENT: usize = 2;

type Area = (u16, u16, u16, u16);

/// Where each pane goes; `None` for a pane not shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub models: Option<Area>,
    pub profiles: Option<Area>,
    pub placement: Option<Area>,
    pub options: Option<Area>,
    pub command: Area,
}

/// The layout on a `width` by `height` terminal for `loader` and `app`.
pub fn layout(width: u16, height: u16, app: &App) -> Layout {
    let loader = &app.loader;
    let top = 2;
    let room = height.saturating_sub(top + COMMAND + 1);
    let command = (
        0,
        top + room,
        width,
        COMMAND.min(height.saturating_sub(top + room + 1)),
    );
    // The recent launches' rows are kept even before there are any, so the
    // panes below stay where they are.
    let models = app.catalog.models().len().max(1) as u16 + 1 + RECENT as u16 + 2;
    let placement = loader.plan.gpus.len().max(1) as u16 + 4;
    let profiles = loader.profiles(&app.catalog).len().max(1) as u16 + 2;
    let wide = width >= WIDE && models + profiles + placement <= room;
    if wide {
        let profiles = room - models - placement;
        return Layout {
            models: Some((0, top, LEFT, models)),
            profiles: Some((0, top + models, LEFT, profiles)),
            placement: Some((0, top + models + profiles, LEFT, placement)),
            options: Some((LEFT, top, width - LEFT, room)),
            command,
        };
    }
    let whole = Some((0, top, width, room));
    let mut layout = Layout {
        models: None,
        profiles: None,
        placement: None,
        options: None,
        command,
    };
    match loader.pane {
        Pane::Models => layout.models = whole,
        Pane::Profiles => layout.profiles = whole,
        Pane::Placement => layout.placement = whole,
        Pane::Options => layout.options = whole,
    }
    layout
}

/// How many option rows an Options box `height` rows high shows.
pub fn option_rows(height: u16) -> usize {
    usize::from(if height >= 9 {
        height - 6
    } else {
        height.saturating_sub(3)
    })
}

/// The columns of the Options box at `x`, `width` wide: name, value,
/// FROM and NOTE, as at 132 columns, narrower in a narrow box.
fn columns(x: u16, width: u16) -> (u16, u16, u16, u16) {
    let total = width.saturating_sub(6);
    let (name, value) = if total >= 82 {
        (26, 29)
    } else {
        let name = 26u16.saturating_sub((82 - total) / 2).max(14);
        (name, total.saturating_sub(name + 9 + 18).max(8))
    };
    let start = x + 4;
    (
        start,
        start + name,
        start + name + value,
        start + name + value + 9,
    )
}

fn selected(buffer: &mut Buffer, x: u16, y: u16, width: u16) {
    buffer.background(x + 1, y, width.saturating_sub(2), Color::Selection);
    buffer.put(x + 1, y, "▌", Style::fg(Color::Cyan));
}

fn tone(tone: Tone) -> Color {
    match tone {
        Tone::Hint => Color::Dim,
        Tone::Warning => Color::Yellow,
        Tone::Problem => Color::Red,
    }
}

pub fn draw(buffer: &mut Buffer, app: &App, now: f64) {
    let loader = &app.loader;
    let plan = &loader.plan;
    let (width, height) = (buffer.width, buffer.height);
    let mut crumbs = vec![
        ("  >  ".to_owned(), Style::fg(Color::Dim)),
        ("Load model".to_owned(), Style::fg(Color::White).bold()),
    ];
    let replacing = replacing(app);
    if !replacing.is_empty() {
        crumbs.push((
            format!("   replacing {}", replacing.join(", ")),
            Style::fg(Color::Yellow),
        ));
    }
    let models = app.catalog.models().len();
    let gpus = plan.gpus.len();
    crumbs.push((
        format!(
            "   {models} model{}  {gpus} GPU{}",
            if models == 1 { "" } else { "s" },
            if gpus == 1 { "" } else { "s" }
        ),
        Style::fg(Color::Dim),
    ));
    header(buffer, app, &crumbs);
    let layout = layout(width, height, app);
    if let Some(area) = layout.models {
        models_pane(buffer, app, area, layout.options.is_none());
    }
    if let Some(area) = layout.profiles {
        profiles_pane(buffer, app, area, layout.options.is_none());
    }
    if let Some(area) = layout.placement {
        placement_pane(buffer, app, area, layout.options.is_none());
    }
    if let Some(area) = layout.options {
        options_pane(buffer, app, area, layout.models.is_none());
    }
    command_box(buffer, app, layout.command, now);
    if loader.edit.is_some() {
        footer(
            buffer,
            &[
                ("Enter", "keep", 0),
                ("Esc", "cancel", 0),
                ("Backspace", "delete", 0),
            ],
        );
    } else if loader.finding {
        footer(
            buffer,
            &[
                ("type", "to find", 0),
                ("Enter", "keep the filter", 0),
                ("Esc", "clear", 0),
            ],
        );
    } else {
        footer(
            buffer,
            &[
                ("Tab", "pane", 0),
                ("↑↓ jk", "move", 0),
                ("←→", "change", 1),
                ("Enter", "edit", 0),
                ("Space", "pick GPU", 2),
                ("/", "find", 1),
                ("a", "add", 2),
                ("x", "reset", 2),
                ("S", "start", 0),
                ("Esc", "back", 0),
                ("?", "help", 3),
            ],
        );
    }
}

/// What the switch stops or replaces: `proc 48211`, `launch A`.
fn replacing(app: &App) -> Vec<String> {
    let placement = &app.loader.plan.placement;
    placement
        .stops
        .iter()
        .map(|serve| serve.label())
        .chain(placement.replaces.iter().filter_map(|position| {
            app.reservations
                .get(*position)
                .map(|reservation| sanitize(&reservation.label))
        }))
        .collect()
}

/// The bottom border's note of a pane drawn alone: where Tab goes.
fn alone(buffer: &mut Buffer, area: Area, pane: Pane) {
    let (x, y, width, height) = area;
    let label = format!(" Tab: {} ", pane.next().title());
    let length = label.chars().count() as u16;
    if height >= 2 && width > length + 4 {
        buffer.put(x + 2, y + height - 1, &label, Style::fg(Color::Dim));
    }
}

/// The first of `count` rows shown in `room` lines so `cursor` is one.
fn first(cursor: usize, room: usize) -> usize {
    (cursor + 1).saturating_sub(room.max(1))
}

fn models_pane(buffer: &mut Buffer, app: &App, area: Area, alone_: bool) {
    let loader = &app.loader;
    let (x, y, width, height) = area;
    let focus = loader.pane == Pane::Models;
    boxed(
        buffer,
        area,
        &[
            ("Models", Style::fg(Color::Cyan).bold()),
            ("  installed beside chopinfer", Style::fg(Color::Dim)),
        ],
        focus,
    );
    if alone_ {
        alone(buffer, area, Pane::Models);
    }
    let inner = width.saturating_sub(4);
    let room = usize::from(height.saturating_sub(2));
    let mut lines: Vec<Vec<(u16, String, Style)>> = Vec::new();
    let catalog = &app.catalog;
    let models = catalog.models();
    let message = match (&catalog.binary, &catalog.models) {
        (Err(error), _) => Some((error.clone(), Color::Yellow)),
        (_, Some(Fetch::Waiting) | None) => {
            Some(("reading chopinfer models --json".to_owned(), Color::Dim))
        }
        (_, Some(Fetch::Failed(error))) => Some((error.clone(), Color::Red)),
        _ if models.is_empty() => Some((
            "no chopinfer-model-* helper is installed beside chopinfer".to_owned(),
            Color::Yellow,
        )),
        _ => None,
    };
    if let Some((text, color)) = message {
        for line in wrap(&sanitize(&text), usize::from(inner)) {
            lines.push(vec![(2, line, Style::fg(color))]);
        }
    }
    let mut cursor = None;
    for (index, model) in models.iter().enumerate() {
        let chosen = loader.model.as_deref() == Some(model.model.as_str());
        let (state, color) = match catalog.checkpoint(&model.model) {
            Checkpoint::Found { .. } => ("found", Color::Dim),
            Checkpoint::Checking => ("checking", Color::Dim),
            Checkpoint::Missing => ("missing", Color::Yellow),
        };
        if index == loader.model_cursor {
            cursor = Some(lines.len());
        }
        let mut line = vec![
            (
                2,
                pad(&sanitize(&model.served_model_name), 21),
                if chosen {
                    Style::fg(Color::White).bold()
                } else {
                    Style::PLAIN
                },
            ),
            (23, pad(&sanitize(&model.model), 11), Style::fg(Color::Dim)),
            (34, state.to_owned(), Style::fg(color)),
        ];
        if chosen {
            line.push((42, "●".to_owned(), Style::fg(Color::Cyan)));
        }
        lines.push(line);
    }
    if !models.is_empty() {
        let mut label = vec![(2, "recent launches".to_owned(), Style::fg(Color::Dim))];
        if loader.recent.is_empty() {
            label.push((17, "  none yet".to_owned(), Style::fg(Color::Dim)));
        }
        lines.push(label);
        for (index, recent) in loader.recent.iter().take(RECENT).enumerate() {
            if models.len() + index == loader.model_cursor {
                cursor = Some(lines.len());
            }
            let set = recent.choice.edits.len() + recent.choice.custom.len();
            lines.push(vec![
                (2, pad(&sanitize(&recent.label), 20), Style::PLAIN),
                (
                    23,
                    pad(
                        &if set > 0 {
                            format!("{set} set")
                        } else {
                            String::new()
                        },
                        8,
                    ),
                    Style::fg(Color::Cyan),
                ),
                (
                    31,
                    right(&sanitize(&recent.when), 10),
                    Style::fg(Color::Dim),
                ),
            ]);
        }
    }
    let skip = cursor.map_or(0, |cursor| first(cursor, room));
    for (offset, line) in lines.iter().enumerate().skip(skip).take(room) {
        let row = y + 1 + (offset - skip) as u16;
        if focus && Some(offset) == cursor {
            selected(buffer, x, row, width);
        }
        for (column, text, style) in line {
            if *column + 1 < width {
                let room = usize::from(width - 1 - column);
                buffer.put(
                    x + column,
                    row,
                    &pad(text, text.chars().count().min(room)),
                    *style,
                );
            }
        }
    }
}

/// `AR`, `MTP 3`, `DSPARK 5`, `AR DCP2`.
fn speculation(profile: &Profile) -> String {
    // An unlisted profile's drafter is not known.
    if profile.drafter.is_empty() {
        return String::new();
    }
    let mut label = if profile.drafter == "none" || profile.speculative_tokens == 0 {
        "AR".to_owned()
    } else {
        format!(
            "{} {}",
            profile.drafter.to_uppercase(),
            profile.speculative_tokens
        )
    };
    if let Some(size) = profile
        .settings
        .get("context-parallel-size")
        .and_then(|size| size.parse::<u32>().ok())
        .filter(|size| *size > 1)
    {
        label.push_str(&format!(" DCP{size}"));
    }
    sanitize(&label)
}

/// Why `profile` cannot run on the checkpoint `pick` describes.
fn unavailable<'a>(pick: Option<&'a Pick>, profile: &Profile) -> Option<&'a str> {
    match &pick?
        .size(profile.tensor_parallel)?
        .profile(&profile.name)?
        .availability
    {
        Availability::Unavailable(reason) => Some(reason),
        Availability::Available { .. } => None,
    }
}

/// Whether `serve --model` picks `profile` at its size.
fn picked(pick: Option<&Pick>, profile: &Profile) -> bool {
    pick.and_then(|pick| pick.size(profile.tensor_parallel))
        .and_then(|size| size.pick.as_deref())
        == Some(profile.name.as_str())
}

fn profiles_pane(buffer: &mut Buffer, app: &App, area: Area, alone_: bool) {
    let loader = &app.loader;
    let (x, y, width, height) = area;
    let focus = loader.pane == Pane::Profiles;
    boxed(
        buffer,
        area,
        &[
            ("Profiles", Style::fg(Color::Cyan).bold()),
            ("  ● what serve --model picks", Style::fg(Color::Dim)),
        ],
        focus,
    );
    if alone_ {
        alone(buffer, area, Pane::Profiles);
    }
    if loader.model(&app.catalog).is_none() {
        return;
    }
    let profiles = loader.profiles(&app.catalog);
    let pick = loader.pick(&app.catalog);
    let room = usize::from(height.saturating_sub(2));
    let chosen = loader.profile.as_deref();
    let cursor = profiles
        .iter()
        .position(|profile| Some(profile.name.as_str()) == chosen)
        .unwrap_or(0);
    let skip = first(cursor, room);
    let mut row = y + 1;
    for (index, profile) in profiles.iter().enumerate().skip(skip).take(room) {
        let is_chosen = index == cursor && chosen.is_some();
        if focus && is_chosen {
            selected(buffer, x, row, width);
        }
        let not_here = unavailable(pick, profile).is_some();
        let name_style = if not_here {
            Style::fg(Color::Dim)
        } else if is_chosen {
            Style::fg(Color::White).bold()
        } else {
            Style::PLAIN
        };
        let (status, color) = if not_here {
            ("not here".to_owned(), Color::Dim)
        } else if let Some(qualified) = profile.qualified_max_sequences {
            (format!("evidence C{qualified}"), Color::Green)
        } else {
            (sanitize(&profile.status), Color::Yellow)
        };
        let pieces = [
            (2, pad(&sanitize(&profile.name), 13), name_style),
            (
                15,
                if picked(pick, profile) { "●" } else { " " }.to_owned(),
                Style::fg(Color::Cyan),
            ),
            (
                17,
                pad(&speculation(profile), 11),
                if not_here {
                    Style::fg(Color::Dim)
                } else {
                    Style::PLAIN
                },
            ),
            (29, pad(&status, 13), Style::fg(color)),
        ];
        for (column, text, style) in pieces {
            if column + 1 < width {
                buffer.put(
                    x + column,
                    row,
                    &pad(
                        &text,
                        text.chars().count().min(usize::from(width - 1 - column)),
                    ),
                    style,
                );
            }
        }
        row += 1;
    }
    // The chosen profile's evidence, or why it cannot run here.
    let Some(profile) = loader.profile(&app.catalog) else {
        return;
    };
    let (text, color) = match unavailable(pick, profile) {
        Some(reason) => (
            format!("{} cannot run here: {reason}", profile.name),
            Color::Red,
        ),
        None => (profile.qualification.clone(), Color::Dim),
    };
    let top = row + 1;
    let bottom = y + height.saturating_sub(1);
    if top >= bottom {
        return;
    }
    let inner = usize::from(width.saturating_sub(4));
    let lines = wrap(&sanitize(&text), inner);
    let room = usize::from(bottom - top);
    for (index, line) in lines.iter().take(room).enumerate() {
        let cut = index + 1 == room && lines.len() > room;
        let shown = if cut {
            pad(line, inner.saturating_sub(3)).trim_end().to_owned() + "..."
        } else {
            line.clone()
        };
        buffer.put(x + 2, top + index as u16, &shown, Style::fg(color));
    }
}

fn placement_pane(buffer: &mut Buffer, app: &App, area: Area, alone_: bool) {
    let loader = &app.loader;
    let plan = &loader.plan;
    let placement = &plan.placement;
    let (x, y, width, height) = area;
    let focus = loader.pane == Pane::Placement;
    let size = plan
        .tensor_parallel
        .map_or_else(String::new, |size| format!("  TP{size}"));
    let title = match &placement.grouping {
        Some(problem) => (format!("{size}  {problem}"), Style::fg(Color::Red)),
        None => {
            let count = placement.instances.len();
            (
                format!(
                    "{size}  {count} instance{}",
                    if count == 1 { "" } else { "s" }
                ),
                Style::fg(Color::Dim),
            )
        }
    };
    boxed(
        buffer,
        area,
        &[
            ("Placement", Style::fg(Color::Cyan).bold()),
            (&title.0, title.1),
        ],
        focus,
    );
    if alone_ {
        alone(buffer, area, Pane::Placement);
    }
    let inner = width.saturating_sub(4);
    if plan.gpus.is_empty() {
        let text = match &app.live.unavailable {
            Some(reason) => format!("GPUs not read: {}", sanitize(reason)),
            None => "reading this host's GPUs".to_owned(),
        };
        buffer.put(
            x + 2,
            y + 1,
            &pad(&text, usize::from(inner)),
            Style::fg(Color::Dim),
        );
        return;
    }
    let room = usize::from(height.saturating_sub(4)).max(1);
    let skip = first(loader.gpu_cursor, room);
    let owner_width = usize::from(width.saturating_sub(15));
    let mut row = y + 1;
    for (index, gpu) in plan.gpus.iter().enumerate().skip(skip).take(room) {
        if row + 1 >= y + height {
            break;
        }
        if focus && index == loader.gpu_cursor {
            selected(buffer, x, row, width);
        }
        let on = loader.picked.contains(&gpu.index);
        buffer.put(x + 2, row, &format!("GPU {}", gpu.index), Style::PLAIN);
        buffer.put(
            x + 8,
            row,
            if on { "■" } else { "□" },
            Style::fg(if on { Color::Cyan } else { Color::Dim }),
        );
        if let Some(planned) = placement
            .instances
            .iter()
            .find(|planned| planned.gpus.contains(&gpu.index))
        {
            buffer.put(
                x + 10,
                row,
                &planned.letter.to_string(),
                Style::fg(Color::Cyan).bold(),
            );
        }
        let (text, color) = match &gpu.occupancy {
            Occupancy::Free => (
                gpu.memory_total.map_or_else(
                    || "free".to_owned(),
                    |bytes| format!("free  {:.1} GiB", bytes as f64 / f64::from(1 << 30)),
                ),
                Color::Dim,
            ),
            Occupancy::Unknown(_) => ("unknown".to_owned(), Color::Yellow),
            Occupancy::Blocked(what) => (sanitize(what), if on { Color::Red } else { Color::Dim }),
            // Yellow when the switch stops what holds it, on every GPU it
            // holds, picked or not.
            Occupancy::Held(holders) => (
                holders
                    .iter()
                    .map(|holder| match holder {
                        Holder::Serve(serve) => serve.described(),
                        Holder::Launch(position) => app
                            .reservations
                            .get(*position)
                            .map_or_else(String::new, |launch| {
                                sanitize(&format!("{} {}", launch.label, launch.model))
                            }),
                    })
                    .collect::<Vec<_>>()
                    .join(", "),
                if holders.iter().any(|holder| match holder {
                    Holder::Serve(serve) => placement.stops.contains(serve),
                    Holder::Launch(position) => placement.replaces.contains(position),
                }) {
                    Color::Yellow
                } else {
                    Color::Dim
                },
            ),
        };
        // A graphics process keeps running beside a launch; it is named on
        // the row, a free GPU's too.
        let graphics = match gpu.graphics.as_slice() {
            [] => String::new(),
            [one] => format!(
                " + {}",
                sanitize(one.owner.name.as_deref().unwrap_or("graphics"))
            ),
            many => format!(" + {} graphics", many.len()),
        };
        if width > 15 {
            let room = owner_width.saturating_sub(graphics.chars().count());
            let text = pad(&text, text.chars().count().min(room));
            let column = buffer.put(x + 13, row, &text, Style::fg(color));
            buffer.put(
                column,
                row,
                &pad(
                    &graphics,
                    graphics
                        .chars()
                        .count()
                        .min(owner_width - text.chars().count()),
                ),
                Style::fg(if on { Color::Yellow } else { Color::Dim }),
            );
        }
        row += 1;
    }
    if row + 1 < y + height {
        let mut column = x + 2;
        for planned in placement.instances.iter().take(2) {
            column = buffer.put(
                column,
                row,
                &format!("{} ", planned.letter),
                Style::fg(Color::Cyan).bold(),
            );
            column = buffer.put(
                column,
                row,
                &format!("{}   ", planned.listen()),
                Style::PLAIN,
            );
        }
        row += 1;
    }
    if row + 1 < y + height {
        let replacing = replacing(app);
        if !replacing.is_empty() {
            buffer.put(
                x + 2,
                row,
                &pad(
                    &format!("stops {} first", replacing.join(", ")),
                    usize::from(inner),
                ),
                Style::fg(Color::Yellow),
            );
        } else if !placement.instances.is_empty() {
            buffer.put(x + 2, row, "on free GPUs", Style::fg(Color::Dim));
        }
    }
}

/// Draws an inline edit field at `x`, `width` cells wide.
fn field(buffer: &mut Buffer, x: u16, y: u16, width: u16, text: &str) {
    let width = usize::from(width.max(2));
    let shown = text
        .chars()
        .rev()
        .take(width - 1)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<String>();
    buffer.put(
        x,
        y,
        &pad(&format!("{shown}▏"), width),
        Style {
            bg: Some(Color::Bar),
            ..Style::fg(Color::White).bold()
        },
    );
}

fn options_pane(buffer: &mut Buffer, app: &App, area: Area, alone_: bool) {
    let loader = &app.loader;
    let (x, y, width, height) = area;
    let focus = loader.pane == Pane::Options;
    let context = loader.context(&app.catalog);
    let lines = loader.lines(&app.catalog);
    let set = loader.edits.len() + loader.custom.len();
    let mut title = vec![("Options".to_owned(), Style::fg(Color::Cyan).bold())];
    if loader.finding || !loader.find.is_empty() {
        title.push(("  find ".to_owned(), Style::fg(Color::Dim)));
        title.push((
            format!(
                "{}{}",
                sanitize(&loader.find),
                if loader.finding { "▏" } else { "" }
            ),
            Style::fg(Color::White).bold(),
        ));
        title.push(("  Esc clears".to_owned(), Style::fg(Color::Dim)));
    } else {
        title.push((
            format!("  {} serve options", context.option_count()),
            Style::fg(Color::Dim),
        ));
        if set > 0 {
            title.push((format!("  {set} set"), Style::fg(Color::Cyan)));
        }
    }
    let pieces = title
        .iter()
        .map(|(text, style)| (text.as_str(), *style))
        .collect::<Vec<_>>();
    boxed(buffer, area, &pieces, focus);
    if alone_ {
        alone(buffer, area, Pane::Options);
    }
    if height < 4 || width < 20 {
        return;
    }
    let (name, value, from, note) = columns(x, width);
    let end = x + width - 2;
    let heading = Style::fg(Color::Dim).bold();
    buffer.put(
        name,
        y + 1,
        &pad("OPTION", usize::from(value - name)),
        heading,
    );
    buffer.put(
        value,
        y + 1,
        &pad("VALUE", usize::from(from - value)),
        heading,
    );
    buffer.put(from, y + 1, &pad("FROM", usize::from(note - from)), heading);
    if note < end {
        buffer.put(note, y + 1, "NOTE", heading);
    }
    let shown = option_rows(height);
    let scroll = loader.option_scroll.min(lines.len().saturating_sub(1));
    for (offset, (row, line)) in lines.iter().enumerate().skip(scroll).take(shown) {
        let line_y = y + 2 + (offset - scroll) as u16;
        let focused = focus && offset == loader.option_cursor;
        if focused {
            selected(buffer, x, line_y, width);
        }
        option_line(
            buffer,
            loader,
            (row, line),
            line_y,
            (x, name, value, from, note, end),
            focused,
        );
    }
    if lines.len() > shown && shown > 0 {
        let label = format!(
            " {}-{} of {} ",
            scroll + 1,
            (scroll + shown).min(lines.len()),
            lines.len()
        );
        let length = label.chars().count() as u16;
        if width > length + 4 {
            buffer.put(
                x + width - 2 - length,
                y + height - 1,
                &label,
                Style::fg(Color::Dim),
            );
        }
    }
    if height < 9 {
        return;
    }
    let (flag, help, color) = pane_help(app, &lines);
    let room = usize::from(width.saturating_sub(4));
    let first_room = room.saturating_sub(flag.chars().count() + 2);
    let wrapped = wrap(&help, first_room.max(1));
    let help_y = y + height - 3;
    let column = buffer.put(
        x + 2,
        help_y,
        &format!("{flag}  "),
        Style::fg(Color::Cyan).bold(),
    );
    if let Some(first) = wrapped.first() {
        buffer.put(column, help_y, first, Style::fg(color));
    }
    if wrapped.len() > 1 {
        let rest = wrapped[1..].join(" ");
        let rest = if rest.chars().count() > room {
            pad(&rest, room.saturating_sub(3)).trim_end().to_owned() + "..."
        } else {
            rest
        };
        buffer.put(
            x + 2,
            help_y + 1,
            &rest,
            Style::fg(if color == Color::Default {
                Color::Dim
            } else {
                color
            }),
        );
    }
}

/// The two help lines: the focused row's flag and help, or the focused
/// pane's hint.
fn pane_help(app: &App, lines: &[(Row, Line)]) -> (String, String, Color) {
    let loader = &app.loader;
    let catalog = &app.catalog;
    match loader.pane {
        Pane::Models => {
            let troubles = catalog.troubles();
            if troubles.is_empty() {
                (
                    "Models".to_owned(),
                    "chopinfer-model-* helpers installed beside chopinfer. found: models pick names it for a checkpoint directory. Enter on a recent launch restores it.".to_owned(),
                    Color::Default,
                )
            } else {
                ("Models".to_owned(), troubles.join("; "), Color::Yellow)
            }
        }
        Pane::Profiles => {
            let Some(profile) = loader.profile(catalog) else {
                return ("Profiles".to_owned(), String::new(), Color::Default);
            };
            let pick = loader.pick(catalog);
            if let Some(reason) = unavailable(pick, profile) {
                return (
                    sanitize(&profile.name),
                    format!("{} cannot run here: {}", profile.name, sanitize(reason)),
                    Color::Red,
                );
            }
            let base = pick
                .and_then(|pick| pick.size(profile.tensor_parallel))
                .and_then(|size| size.pick.clone());
            let how = match base {
                Some(base) if base == profile.name => {
                    "serve --model picks this profile itself. ".to_owned()
                }
                Some(base) => format!(
                    "Runs as serve --model with the values where {} differs from {base} passed as flags. ",
                    profile.name
                ),
                None => String::new(),
            };
            (
                sanitize(&profile.name),
                sanitize(&format!("{how}Status: {}.", profile.status)),
                Color::Default,
            )
        }
        Pane::Placement => {
            let size = loader.plan.tensor_parallel.unwrap_or(1);
            let graphics = loader
                .plan
                .gpus
                .get(loader.gpu_cursor)
                .map(|gpu| {
                    gpu.graphics
                        .iter()
                        .map(occupancy::described)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            match loader
                .plan
                .gpus
                .get(loader.gpu_cursor)
                .map(|gpu| (gpu.index, &gpu.occupancy))
            {
                Some((index, Occupancy::Unknown(reason))) => (
                    format!("GPU {index}"),
                    sanitize(&format!(
                        "unknown: {reason}. An unknown GPU cannot be picked."
                    )),
                    Color::Yellow,
                ),
                Some((index, Occupancy::Blocked(what))) => (
                    format!("GPU {index}"),
                    sanitize(&format!(
                        "held by {what}, which choptop does not stop, so a launch cannot use it."
                    )),
                    Color::Yellow,
                ),
                Some((index, _)) if !graphics.is_empty() => (
                    format!("GPU {index}"),
                    sanitize(&format!(
                        "{} keeps running beside a launch here and holds its memory; a desktop's display often runs on GPU 0.",
                        graphics.join(", ")
                    )),
                    Color::Yellow,
                ),
                _ => (
                    "Space".to_owned(),
                    format!(
                        "picks a GPU. Each group of {size} picked GPUs is one instance on its own port; GPUs another instance holds stop it first."
                    ),
                    Color::Default,
                ),
            }
        }
        Pane::Options => {
            let Some((_, line)) = lines.get(loader.option_cursor) else {
                return (String::new(), String::new(), Color::Default);
            };
            match &line.note {
                Some((note, Tone::Problem)) => (
                    sanitize(&line.flag),
                    sanitize(&noted(note, &line.help)),
                    Color::Red,
                ),
                Some((note, Tone::Warning)) => (
                    sanitize(&line.flag),
                    sanitize(&noted(note, &line.help)),
                    Color::Yellow,
                ),
                _ => (sanitize(&line.flag), sanitize(&line.help), Color::Default),
            }
        }
    }
}

/// A row's note before its help, which is a lowercase phrase from
/// chopinfer or a sentence of choptop's own.
fn noted(note: &str, help: &str) -> String {
    match help.chars().next() {
        None => note.to_owned(),
        Some(first) if first.is_lowercase() => format!("{note}; {help}"),
        Some(_) => format!("{note}. {help}"),
    }
}

fn option_line(
    buffer: &mut Buffer,
    loader: &Loader,
    (row, line): (&Row, &Line),
    y: u16,
    (x, name, value, from, note, end): (u16, u16, u16, u16, u16, u16),
    focused: bool,
) {
    let editing = |target: &Target| loader.edit.as_ref().filter(|edit| edit.target == *target);
    match row {
        Row::Group {
            group,
            count,
            set,
            open,
        } => {
            let mut column = buffer.put(
                x + 2,
                y,
                if *open { "▾ " } else { "▸ " },
                Style::fg(Color::Cyan),
            );
            column = buffer.put(column, y, group.title(), Style::fg(Color::Cyan).bold());
            if *group == Group::Package {
                column = buffer.put(column, y, "  --model-option", Style::fg(Color::Dim));
            }
            if !open {
                column = buffer.put(column, y, &format!("  {count}"), Style::fg(Color::Dim));
            }
            if *set > 0 {
                buffer.put(column, y, &format!("  {set} set"), Style::fg(Color::Cyan));
            }
        }
        Row::Info(text, kind) => {
            buffer.put(
                name,
                y,
                &pad(&sanitize(text), usize::from(end.saturating_sub(name))),
                Style::fg(tone(*kind)),
            );
        }
        Row::Add => {
            buffer.put(
                name,
                y,
                "+ add an argument or VAR=value",
                Style::fg(if focused { Color::Default } else { Color::Dim }),
            );
            if note < end {
                buffer.put(note, y, "a", Style::fg(Color::Dim));
            }
        }
        Row::NewCustom | Row::Custom(_) => {
            let target = match row {
                Row::Custom(index) => Target::Custom(*index),
                _ => Target::NewCustom,
            };
            if let Some(edit) = editing(&target) {
                field(buffer, name, y, note.saturating_sub(name + 2), &edit.text);
                return;
            }
            buffer.put(
                name,
                y,
                &pad(
                    &sanitize(&line.name),
                    usize::from(from.saturating_sub(name + 1)),
                ),
                Style::fg(Color::Magenta),
            );
            buffer.put(from, y, "custom", Style::fg(Color::Cyan));
            if let Some((text, kind)) = &line.note
                && note < end
            {
                buffer.put(
                    note,
                    y,
                    &pad(text, usize::from(end - note)),
                    Style::fg(tone(*kind)),
                );
            }
        }
        Row::Option(_) | Row::Package(_) => {
            let set = line.from == From::Set;
            buffer.put(
                name,
                y,
                &pad(
                    &sanitize(&line.name),
                    usize::from(value.saturating_sub(name + 1)),
                ),
                if set {
                    Style::fg(Color::White).bold()
                } else {
                    Style::PLAIN
                },
            );
            let key = row.key().map(Target::Value);
            let value_width = from.saturating_sub(value + 1);
            if let Some(edit) = key.as_ref().and_then(editing) {
                field(buffer, value, y, value_width, &edit.text);
            } else {
                let style = if line.wrong {
                    Style::fg(Color::Red).bold()
                } else if set {
                    Style::fg(Color::White).bold()
                } else if line.from == From::Missing {
                    Style::fg(Color::Red)
                } else if line.from == From::Profile {
                    Style::PLAIN
                } else {
                    Style::fg(Color::Dim)
                };
                let mut text = sanitize(&line.value);
                if line.choice && focused {
                    text.push_str("  ◂▸");
                }
                buffer.put(value, y, &pad(&text, usize::from(value_width)), style);
            }
            buffer.put(
                from,
                y,
                &pad(line.from.label(), 8),
                match line.from {
                    From::Set => Style::fg(Color::Cyan).bold(),
                    From::Missing => Style::fg(Color::Red),
                    _ => Style::fg(Color::Dim),
                },
            );
            if let Some((text, kind)) = &line.note
                && note < end
            {
                buffer.put(
                    note,
                    y,
                    &pad(&sanitize(text), usize::from(end - note)),
                    Style::fg(tone(*kind)),
                );
            }
        }
    }
}

/// The local guess at the command line before `models command` answers:
/// the base, the flags that make the pick serve the profile, and the
/// form's options.
fn draft(app: &App) -> Vec<String> {
    let loader = &app.loader;
    let plan = &loader.plan;
    let checkpoint = match loader
        .model(&app.catalog)
        .map(|model| app.catalog.checkpoint(&model.model))
    {
        Some(Checkpoint::Found { directory, .. }) => directory.display().to_string(),
        _ => "<checkpoint>".to_owned(),
    };
    let mut arguments = vec![
        "serve".to_owned(),
        "--model".to_owned(),
        checkpoint,
        "--tensor-parallel-size".to_owned(),
        plan.tensor_parallel.unwrap_or(1).to_string(),
    ];
    arguments.extend(loader.profile_arguments(&app.catalog));
    arguments.extend(plan.explicit.arguments());
    arguments
}

fn origin_style(origin: Origin) -> Style {
    match origin {
        Origin::Environment => Style::fg(Color::Dim),
        Origin::Base => Style::PLAIN,
        Origin::Profile => Style::fg(Color::Cyan),
        Origin::Set => Style::fg(Color::White).bold(),
        Origin::Custom => Style::fg(Color::Magenta),
    }
}

/// A row of the command box: its pieces and their styles.
type CommandLine = Vec<(String, Style)>;

/// One instance's block: `A  ` and its tokens wrapped to `inner` columns,
/// each row after the first indented by five. A token longer than a row
/// goes on across rows, so nothing of it is cut.
fn block(letter: char, tokens: Vec<(String, Origin)>, inner: usize) -> Vec<CommandLine> {
    const INDENT: usize = 5;
    let mut lines = Vec::new();
    let mut line = vec![(format!("{letter}  "), Style::fg(Color::Cyan).bold())];
    let mut used = 3;
    for (token, origin) in tokens {
        let token = sanitize(&token).chars().collect::<Vec<_>>();
        let style = origin_style(origin);
        if line.len() > 1 {
            if used + 1 + token.len() > inner {
                lines.push(std::mem::take(&mut line));
                line.push((" ".repeat(INDENT), Style::PLAIN));
                used = INDENT;
            } else {
                line.push((" ".to_owned(), Style::PLAIN));
                used += 1;
            }
        }
        let mut rest = &token[..];
        loop {
            let room = inner.saturating_sub(used).max(1);
            if rest.len() <= room {
                line.push((rest.iter().collect(), style));
                used += rest.len();
                break;
            }
            line.push((rest[..room].iter().collect(), style));
            lines.push(std::mem::take(&mut line));
            line.push((" ".repeat(INDENT), Style::PLAIN));
            used = INDENT;
            rest = &rest[room..];
        }
    }
    lines.push(line);
    lines
}

/// The command box's rows from each instance's block in `rows` rows: all
/// of them when they fit, else the first and `same` for the rest. A block
/// cut short ends in a row that says how many rows are not shown.
fn fit(blocks: Vec<Vec<CommandLine>>, same: Option<String>, rows: usize) -> Vec<CommandLine> {
    if blocks.iter().map(Vec::len).sum::<usize>() <= rows {
        return blocks.into_iter().flatten().collect();
    }
    let Some(mut lines) = blocks.into_iter().next() else {
        return Vec::new();
    };
    let room = rows.saturating_sub(usize::from(same.is_some()));
    if lines.len() > room {
        let kept = room.saturating_sub(1);
        let more = lines.len() - kept;
        lines.truncate(kept);
        if room > 0 {
            lines.push(vec![(
                format!(
                    "     ... {more} more row{} of this command",
                    if more == 1 { "" } else { "s" }
                ),
                Style::fg(Color::Dim),
            )]);
        }
    }
    if let Some(same) = same {
        lines.push(vec![
            ("   ".to_owned(), Style::PLAIN),
            (same, Style::fg(Color::Dim)),
        ]);
    }
    lines
}

fn command_box(buffer: &mut Buffer, app: &App, area: Area, now: f64) {
    let loader = &app.loader;
    let plan = &loader.plan;
    let (x, y, width, height) = area;
    let problems = plan.problems.len();
    let instances = &plan.placement.instances;
    let title = if problems > 0 {
        (
            format!(
                "  fix {problems} problem{} before starting",
                if problems == 1 { "" } else { "s" }
            ),
            Style::fg(Color::Red),
        )
    } else {
        (
            format!(
                "  {} process{} of this user",
                instances.len(),
                if instances.len() == 1 { "" } else { "es" }
            ),
            Style::fg(Color::Dim),
        )
    };
    boxed(
        buffer,
        area,
        &[
            ("Command", Style::fg(Color::Cyan).bold()),
            (&title.0, title.1),
        ],
        false,
    );
    if height < 3 || width < 8 {
        return;
    }
    let inner = usize::from(width - 4);
    let rows = usize::from(height.saturating_sub(3)).min(4);
    let program = app
        .catalog
        .binary
        .as_ref()
        .ok()
        .and_then(|path| path.file_name())
        .map_or_else(
            || "chopinfer".to_owned(),
            |name| sanitize(&name.to_string_lossy()),
        );
    let arguments = plan.command.clone().unwrap_or_else(|| draft(app));
    let mut lines: Vec<Vec<(String, Style)>> = Vec::new();
    if instances.is_empty() {
        lines.push(vec![(
            "   no GPUs picked; Space picks them in the placement pane".to_owned(),
            Style::fg(Color::Dim),
        )]);
    } else {
        let blocks = instances
            .iter()
            .map(|planned| {
                block(
                    planned.letter,
                    command::tokens(
                        &program,
                        &arguments,
                        &plan.explicit,
                        &planned.gpus,
                        &planned.listen(),
                        &[API_KEY],
                    ),
                    inner,
                )
            })
            .collect::<Vec<_>>();
        let others = instances[1..]
            .iter()
            .map(|planned| {
                format!(
                    "{} on GPUs {} at {}",
                    planned.letter,
                    planned
                        .gpus
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    planned.listen()
                )
            })
            .collect::<Vec<_>>();
        let mut same = format!("the same for {}", others.join(", "));
        if same.chars().count() + 3 > inner {
            same = format!(
                "the same for {}, each on its own GPUs and port",
                instances[1..]
                    .iter()
                    .map(|planned| planned.letter.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        lines = fit(blocks, (!others.is_empty()).then_some(same), rows);
    }
    for (index, line) in lines.iter().take(rows).enumerate() {
        let mut column = x + 2;
        for (text, style) in line {
            let room = usize::from((x + width - 2).saturating_sub(column));
            column = buffer.put(
                column,
                y + 1 + index as u16,
                &text.chars().take(room).collect::<String>(),
                *style,
            );
        }
    }
    let summary_y = y + height - 2;
    let end = x + width - 2;
    if let Some(flash) = loader
        .flash
        .as_ref()
        .filter(|flash| now - flash.at < FLASH_SECONDS)
    {
        buffer.put(
            x + 2,
            summary_y,
            &pad(&sanitize(&flash.text), inner),
            Style::fg(if flash.good { Color::Green } else { Color::Red }),
        );
        return;
    }
    let Some(profile) = loader.profile(&app.catalog) else {
        return;
    };
    let pick = loader
        .pick(&app.catalog)
        .and_then(|pick| pick.size(profile.tensor_parallel))
        .and_then(|size| size.pick.clone());
    let tokens = command::tokens(&program, &arguments, &plan.explicit, &[], "", &[]);
    let from_profile = tokens
        .iter()
        .filter(|(_, origin)| *origin == Origin::Profile)
        .count();
    let set_here = plan
        .explicit
        .options
        .iter()
        .filter(|(_, origin)| *origin == Origin::Set)
        .count()
        + plan
            .explicit
            .environment
            .iter()
            .filter(|(_, _, origin)| *origin == Origin::Set)
            .count()
        + usize::from(plan.explicit.listen.is_some());
    let what = match pick {
        Some(pick) if pick == profile.name => format!(
            "{} is what serve --model picks at TP{}",
            profile.name, profile.tensor_parallel
        ),
        Some(pick) => format!(
            "{} is serve --model's {pick} with {from_profile} profile option{} (cyan)",
            profile.name,
            if from_profile == 1 { "" } else { "s" }
        ),
        None => profile.name.clone(),
    };
    let mut pieces = vec![
        (
            sanitize(&format!("{what}; {set_here} set here (white).")),
            Style::fg(Color::Dim),
        ),
        (format!("  Logs go to {}.", app.logs), Style::fg(Color::Dim)),
    ];
    let replacing = replacing(app);
    if !replacing.is_empty() {
        pieces.push((
            format!("  Stops {} first.", replacing.join(", ")),
            Style::fg(Color::Yellow),
        ));
    }
    if problems == 0 && !plan.waits.is_empty() {
        pieces.push((
            format!("  Waiting for {}.", plan.waits.join(", ")),
            Style::fg(Color::Dim),
        ));
    }
    // The logs note, then the profile's description, give way to what
    // stops and what is awaited.
    let length = |pieces: &[(String, Style)]| {
        pieces
            .iter()
            .map(|(text, _)| text.chars().count())
            .sum::<usize>()
    };
    if length(&pieces) > inner {
        pieces.remove(1);
    }
    if length(&pieces) > inner {
        pieces[0].0 = sanitize(&format!("{}; {set_here} set here (white).", profile.name));
    }
    let mut column = x + 2;
    for (text, style) in pieces {
        let room = usize::from(end.saturating_sub(column));
        column = buffer.put(
            column,
            summary_y,
            &text.chars().take(room).collect::<String>(),
            style,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(lines: &[CommandLine]) -> Vec<String> {
        lines
            .iter()
            .map(|line| line.iter().map(|(text, _)| text.as_str()).collect())
            .collect()
    }

    #[test]
    fn a_token_longer_than_a_row_goes_on_across_rows() {
        let path = format!("/models/{}", "q".repeat(70));
        let tokens = vec![
            ("CUDA_VISIBLE_DEVICES=0,1".to_owned(), Origin::Environment),
            ("chopinfer serve".to_owned(), Origin::Base),
            (format!("--model {path}"), Origin::Base),
            ("--served-model-name Qwen".to_owned(), Origin::Set),
        ];
        let shown = texts(&block('A', tokens, 40));
        assert_eq!(
            shown,
            [
                "A  CUDA_VISIBLE_DEVICES=0,1".to_owned(),
                "     chopinfer serve".to_owned(),
                format!("     --model /models/{}", "q".repeat(19)),
                format!("     {}", "q".repeat(35)),
                format!("     {}", "q".repeat(16)),
                "     --served-model-name Qwen".to_owned(),
            ]
        );
    }

    #[test]
    fn a_command_longer_than_the_box_says_how_much_is_not_shown() {
        let row = |text: &str| vec![(text.to_owned(), Style::PLAIN)];
        let a = (0..5)
            .map(|number| row(&format!("a{number}")))
            .collect::<Vec<_>>();
        let b = vec![row("b0")];
        let same = || Some("the same for B".to_owned());
        assert_eq!(
            texts(&fit(vec![a.clone(), b.clone()], same(), 4)),
            [
                "a0",
                "a1",
                "     ... 3 more rows of this command",
                "   the same for B"
            ]
        );
        assert_eq!(
            texts(&fit(vec![a.clone()], None, 4)),
            ["a0", "a1", "a2", "     ... 2 more rows of this command"]
        );
        // What fits is shown whole.
        assert_eq!(
            texts(&fit(vec![a[..2].to_vec(), b.clone()], same(), 4)),
            ["a0", "a1", "b0"]
        );
        // A first block that fits is shown whole before the others' line.
        assert_eq!(
            texts(&fit(vec![a[..3].to_vec(), b.clone(), b], same(), 4)),
            ["a0", "a1", "a2", "   the same for B"]
        );
    }
}
