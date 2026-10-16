//! MiMo's encoder kernels on SM120: the `kernels_sm120::encoder` catalog
//! its declarations (`crate::encoders::spec`) run, the kernels of its
//! registered variants for that catalog, MiMo's own encoder row kernels
//! ([`rows`]), and which geometries an encoder cubin serves.

pub mod rows;

use crate::encoders::spec::{declarations_flash, declarations_for};
use crate::geometry::V26_FLASH as GEOMETRY;

use encoder_spec::Error;
use encoder_spec::components::rotary::RotaryComponent;
use encoder_spec::components::{Attention, Boundary, Component};
use kernels_sm120::encoder::{self, Entry, Kernel};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// The kernels of MiMo's registered variants, for the encoder catalog
/// (`kernels_sm120::encoder`).
pub struct Kernels;

impl encoder::Registered for Kernels {
    fn stage(
        &self,
        component: &dyn Component,
        _: &Boundary,
        _: u32,
    ) -> Result<Vec<Kernel>, String> {
        let id = component.id();
        match id.as_str() {
            // Model-owned rows, outside the shared catalog.
            "mimo26:quantize.rvq20-f32.v1" | "mimo26:embed.sum20-f32.v1" => Ok(Vec::new()),
            _ => Err(format!(
                "`{id}` has no MiMo kernel; BF16-legacy variants are retired"
            )),
        }
    }
    fn rotary(
        &self,
        rotary: &dyn RotaryComponent,
        _: &Attention,
        _: u32,
    ) -> Result<Vec<Kernel>, String> {
        Err(format!(
            "`{}` has no MiMo rotary kernel; legacy variants are retired",
            rotary.id()
        ))
    }
}

/// The entry of `kernels()` whose symbol reads `stem`: the hand-written
/// launch sequences name entries by their stems (`Entry::stem`).
pub fn kernel_flash(stem: &str) -> Result<&'static Entry, Error> {
    encoder::by_stem(kernels_flash()?, stem).map_err(Error)
}

pub fn kernels_for(
    geometry: &'static crate::geometry::Geometry,
) -> Result<&'static [Entry], Error> {
    if geometry.id == GEOMETRY.id {
        return kernels_flash();
    }
    static KERNELS: OnceLock<Vec<Result<Vec<Entry>, Error>>> = OnceLock::new();
    let index = crate::geometry::GEOMETRIES
        .iter()
        .position(|g| g.id == geometry.id)
        .ok_or_else(|| Error(format!("unlisted encoder geometry {}", geometry.id)))?;
    KERNELS.get_or_init(|| {
        crate::geometry::GEOMETRIES
            .iter()
            .map(|g| encoder::model_catalog(declarations_for(g)?, &Kernels))
            .collect()
    })[index]
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// MiMo's encoder kernel catalog: the entries its declarations run,
/// vision's first. Built once.
pub fn kernels_flash() -> Result<&'static [Entry], Error> {
    static KERNELS: OnceLock<Result<Vec<Entry>, Error>> = OnceLock::new();
    KERNELS
        .get_or_init(|| encoder::model_catalog(declarations_flash()?, &Kernels))
        .as_ref()
        .map(Vec::as_slice)
        .map_err(Clone::clone)
}

/// Keep the first declaration of each encoder symbol in catalog order.
pub fn deduplicate<T: std::fmt::Debug>(
    entries: &mut Vec<T>,
    known: &mut BTreeMap<String, String>,
    symbol: impl Fn(&T) -> &str,
) {
    entries.retain(|entry| {
        let name = symbol(entry);
        let declaration = format!("{entry:?}");
        if let Some(previous) = known.get(name) {
            assert_eq!(
                previous, &declaration,
                "shared encoder declarations must be identical"
            );
            false
        } else {
            known.insert(name.to_owned(), declaration);
            true
        }
    });
}

/// Every geometry whose encoder declarations use any of these symbols.
pub fn cubin_members(symbols: &[&str]) -> Result<Vec<&'static str>, encoder_spec::Error> {
    let mut members = Vec::new();
    for g in crate::geometry::GEOMETRIES.iter().filter(|g| g.encoders) {
        if kernels_for(g)?
            .iter()
            .any(|entry| symbols.contains(&entry.symbol.as_str()))
            || rows::kernels_for(g)?
                .iter()
                .any(|entry| symbols.contains(&entry.symbol.as_str()))
        {
            members.push(g.id);
        }
    }
    Ok(members)
}
