//! MiMo's encoders on the backend path against the kernels their slots
//! resolve to and the media injection they replace, which stays here as
//! the reference.

use std::collections::BTreeSet;

use bundle_runtime::encoder::Lowered;
use kernel_core::contract::Argument;
use model_mimo26_contract::geometry::{GEOMETRIES, Geometry, V26_FLASH};
use model_mimo26_contract::sm120::encoders::rows;
use runtime_bundle_abi::BundleDeviceBufferHandleV1;

use crate::dispatch::BufferBindings;
use crate::pipeline::PreparedLaunch;

use super::*;

/// The symbols a declaration's slots resolve to: the catalog entries of
/// its shared slots, and MiMo's own row kernels of its registered stages.
fn resolved_symbols(
    geometry: &'static Geometry,
    encoder: usize,
) -> Result<BTreeSet<String>, String> {
    let declaration = &spec::declarations_for(geometry).map_err(|error| error.0)?[encoder];
    let catalog = encoder_kernels::kernels_for(geometry).map_err(|error| error.0)?;
    let slots =
        kernels_sm120::encoder::slots(declaration, &encoder_kernels::Kernels).map_err(|e| e.0)?;
    let mut symbols = BTreeSet::new();
    for (slot, entry) in &slots {
        if slot.part == "layout" {
            continue;
        }
        if !catalog.iter().any(|known| known.symbol == entry.symbol) {
            return Err(format!("the MiMo catalog has no {}", entry.symbol));
        }
        symbols.insert(entry.symbol.clone());
    }
    for stage in 0..declaration.stages.len() {
        if let Some(own) = rows::stage_kernel(declaration, stage).map_err(|error| error.0)? {
            symbols.insert(own.symbol);
        }
    }
    Ok(symbols)
}

/// The media injection's launch before it ran through `Rank::inject_part`:
/// the catalog's override entry over `rows` rows of `activation` and
/// `source_rows`, both starting at the part's first row.
fn prepare_override_rows_for(
    geometry: &'static Geometry,
    activation: Span,
    arena: Span,
    source_rows: Span,
    live_rows: Span,
    rows: u32,
) -> Result<PreparedLaunch, Refusal> {
    if rows == 0
        || u64::from(rows) > CHUNK_ROWS
        || activation.length < u64::from(rows) * geometry.hidden() as u64 * 2
        || arena.length < arena::ROWS * geometry.hidden() as u64 * 2
        || source_rows.length < u64::from(rows) * 8
        || live_rows.length < 4
    {
        return Err(Refusal("MiMo media injection buffers are too small"));
    }
    let symbol = format!("chop_enc_layout_tokens_w{}_v1", geometry.hidden());
    let catalog = encoder_kernels::kernels_for(geometry)
        .map_err(|_| Refusal("MiMo media injection catalog entry is missing"))?;
    let entry = kernels_sm120::encoder::by_stem(catalog, &symbol)
        .map_err(|_| Refusal("MiMo media injection catalog entry is missing"))?;
    let mut bindings = BufferBindings::default();
    let arguments = entry.pack_arguments(vec![
        Argument::DeviceBuffer(bindings.bind(activation)),
        Argument::DeviceBuffer(bindings.bind(arena)),
        Argument::DeviceBuffer(bindings.bind(source_rows)),
        Argument::DeviceBuffer(bindings.bind(live_rows)),
        Argument::U64(arena::ROWS),
    ])?;
    Ok(PreparedLaunch {
        symbol: entry.symbol.clone(),
        rows,
        plan: entry.plan(CHUNK_ROWS as u32)?,
        arguments,
        bindings,
    })
}

/// Each encoder's launch order before the transitional resolver was
/// deleted: (geometry, encoder, launches, FNV-1a of the launched symbols
/// each followed by a newline), recorded from that resolver's lowering.
const LAUNCH_ORDER: [(&str, usize, usize, u64); 2] = [
    ("v26-flash", 0, 280, 0xB497_229E_0603_245F),
    ("v26-flash", 1, 260, 0x985B_8FD5_A867_A32F),
];

/// The FNV-1a hash of `symbols`, each followed by a newline.
fn order_hash(symbols: &[&str]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in symbols
        .iter()
        .flat_map(|symbol| symbol.bytes().chain([b'\n']))
    {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Every geometry that serves encoders.
fn geometries() -> impl Iterator<Item = &'static Geometry> {
    GEOMETRIES.iter().filter(|geometry| geometry.encoders)
}

/// Every declaration of every geometry, lowered over the buffer plan of
/// both, launches exactly the kernels its slots resolve to: shared slots on
/// the catalog entries, registered ones on MiMo's own row kernels.
#[test]
fn every_slot_launches_the_kernel_it_resolved_to() {
    let mut seen = 0;
    for geometry in geometries() {
        let declarations = spec::declarations_for(geometry).unwrap();
        let plan =
            model_mimo26_contract::encoders::plan_declarations(&declarations.each_ref()).unwrap();
        let own = Sm120Own::new(Tp::of_geometry(geometry, 2).unwrap());
        let slots = Slots::new(declarations, &own);
        let backend = backend(geometry).unwrap();
        for (encoder, declaration) in declarations.iter().enumerate() {
            let prepared = Lowered::with_backend(
                declaration,
                &plan,
                &backend,
                &|stage, part, block| slots.op(encoder, stage, part, block),
                &MimoVariants,
                CHUNK_ROWS as u32,
            )
            .unwrap();
            let symbols = prepared.symbols().collect::<Vec<_>>();
            assert_eq!(
                symbols
                    .iter()
                    .map(|symbol| symbol.to_string())
                    .collect::<BTreeSet<_>>(),
                resolved_symbols(geometry, encoder).unwrap(),
                "{} {}",
                geometry.id,
                declaration.name
            );
            // The launches keep their order and count.
            let pinned = LAUNCH_ORDER
                .iter()
                .find(|&&(id, at, ..)| id == geometry.id && at == encoder)
                .map(|&(.., launches, hash)| (launches, hash));
            assert_eq!(
                Some((symbols.len(), order_hash(&symbols))),
                pinned,
                "{} {}: {symbols:?}",
                geometry.id,
                declaration.name
            );
            assert!(
                symbols
                    .iter()
                    .any(|symbol| symbol.starts_with("chop_mimo_enc_"))
                    == (encoder == 1)
            );
            seen += 1;
        }
    }
    assert!(seen >= 2);
}

/// A registered slot binds exactly its op's roles, each once.
#[test]
fn own_ops_bind_their_roles_by_name() {
    let view = |handle| View {
        handle,
        bytes: u64::MAX,
    };
    let named = |names: &[&'static str]| {
        names
            .iter()
            .enumerate()
            .map(|(index, &name)| (name, view(index as u64 + 1)))
            .collect::<Named<View>>()
    };
    let roles = ["tables", "codes", "output", "live_rows"];
    assert_eq!(
        embedding_sum_roles(&named(&["live_rows", "tables", "output", "codes"])).unwrap(),
        EmbeddingSumBuffers {
            tables: view(2),
            codes: view(4),
            output: view(3),
            live_rows: view(1),
        }
    );
    assert!(embedding_sum_roles(&named(&roles[..3])).is_err());
    assert!(embedding_sum_roles(&named(&["tables", "codes", "output", "output"])).is_err());
    assert!(
        embedding_sum_roles(&named(&["tables", "codes", "output", "live_rows", "input"])).is_err()
    );
    let rvq = [
        "input",
        "codebooks",
        "codebook_offsets",
        "codes",
        "live_rows",
    ];
    assert_eq!(rvq_roles(&named(&rvq)).unwrap().codebook_offsets, view(3));
    assert!(rvq_roles(&named(&roles)).is_err());
}

/// The override through `Rank::inject_part` launches what the media
/// injection launched, over the same views, for parts at several rows and
/// first rows; both refuse empty and oversized parts.
#[test]
fn the_override_launches_the_media_injection() {
    crate::fake_host_tests::with_services(|services| {
        let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let (encoder, _) = crate::fake_host_tests::shared_encoder(services, &[handle]);
        let rank = &encoder.ranks[0];
        let arena = rank.region("arena").unwrap();
        let source = rank.region("source-rows").unwrap();
        let live = Span::new(BundleDeviceBufferHandleV1::from_parts(2, 7), 64, 4);
        let hidden = V26_FLASH.hidden() as u64 * 2;
        let chunk = CHUNK_ROWS as u32;
        for (first, rows) in [
            (0, 1),
            (0, 37),
            (5, 37),
            (0, chunk),
            (chunk - 64, 64),
            (chunk - 1, 1),
        ] {
            let base = Span::new(
                BundleDeviceBufferHandleV1::from_parts(3, 1),
                4096,
                (CHUNK_ROWS + 8) * hidden,
            );
            let skip = u64::from(first) * hidden;
            let activation = Span::new(base.buffer, base.offset + skip, base.length - skip);
            let launch = rank.inject_part(activation, live, first, rows).unwrap();
            let skip = u64::from(first) * 8;
            let shifted = Span::new(source.buffer, source.offset + skip, source.length - skip);
            let reference =
                prepare_override_rows_for(&V26_FLASH, activation, arena, shifted, live, rows)
                    .unwrap();
            assert_eq!(launch.symbol, reference.symbol);
            assert_eq!(launch.rows, reference.rows);
            assert_eq!(launch.plan, reference.plan);
            assert_eq!(launch.arguments, reference.arguments);
            for argument in &launch.arguments.0 {
                if let Argument::DeviceBuffer(id) = argument {
                    assert_eq!(
                        launch.bindings.get(*id).unwrap(),
                        reference.bindings.get(*id).unwrap()
                    );
                }
            }
        }
        let activation = Span::new(
            BundleDeviceBufferHandleV1::from_parts(3, 1),
            0,
            (CHUNK_ROWS + 8) * hidden,
        );
        for (first, rows) in [(0, 0), (0, chunk + 1), (1, chunk)] {
            assert!(
                rank.inject_part(activation, live, first, rows).is_err(),
                "{first} {rows}"
            );
        }
        let short = Span::new(activation.buffer, 0, 37 * hidden - 2);
        assert!(rank.inject_part(short, live, 0, 37).is_err());
        assert!(prepare_override_rows_for(&V26_FLASH, short, arena, source, live, 37).is_err());
    });
}

/// Every geometry's encoders, for every media set, create over image
/// views of exactly the bytes the encoder image lays out for them, so the
/// create-time check of every launch passes on the real layout; a quantizer
/// codebook view four bytes short refuses create.
#[test]
fn encoders_create_over_the_image_views() {
    crate::fake_host_tests::with_services(|services| {
        let handle = BundleDeviceBufferHandleV1::from_parts(1, 1);
        let create = |geometry: &'static Geometry, media: Encoders, short: Option<&str>| {
            let mut next = 0;
            let views = model_mimo26_contract::encoders::views_for(geometry, media)
                .into_iter()
                .map(|(name, view)| {
                    let bytes = view.bytes - if short == Some(name.as_str()) { 4 } else { 0 };
                    let span = Span::new(handle, next, bytes);
                    next += view.bytes.next_multiple_of(256);
                    (name, span)
                })
                .collect::<BTreeMap<_, _>>();
            let mut region = 1 << 40;
            create_for(
                Tp::of_geometry(geometry, 2).unwrap(),
                services,
                media,
                vec![views],
                |_, bytes| {
                    let span = Span::new(handle, region, bytes);
                    region += bytes.next_multiple_of(256);
                    Ok(span)
                },
            )
        };
        let mut created = 0;
        for geometry in geometries() {
            for media in Encoders::SETS.into_iter().filter(|media| media.any()) {
                if let Err(error) = create(geometry, media, None) {
                    panic!("{} {media:?}: {error:?}", geometry.id);
                }
                created += 1;
            }
            let audio = Encoders::of(false, false, true);
            assert!(
                create(geometry, audio, Some("@codebooks")).is_err(),
                "{}",
                geometry.id
            );
        }
        assert!(created >= 7);
    });
}

/// The instance's MediaInject stage launches on every rank what the media
/// injection launched for the same prefill part: the activation and the
/// arena source rows both from the part's first row (a pipelined prefill's
/// second half starts at a nonzero row, as a 1,024-token step's at 512).
/// Parts past the source rows or the prefill chunk refuse on both paths.
#[test]
fn the_instance_injects_each_part_as_the_media_injection_did() {
    crate::fake_host_tests::with_encoder_instance(each_part_injects_as_before);
}

fn each_part_injects_as_before(instance: runtime_bundle_abi::BundleHandleV1) {
    let hidden = V26_FLASH.hidden() as u64 * 2;
    let chunk = CHUNK_ROWS as u32;
    let mut launched = 0;
    for (row0, rows) in [
        (0, 1),
        (0, 37),
        (5, 37),
        (512, 512),
        (100, 4000),
        (u64::from(chunk) - 64, 64),
        (u64::from(chunk) - 1, 1),
        (0, 0),
        (0, chunk + 1),
        (1, chunk),
        (u64::from(chunk), 1),
        (1 << 33, 1),
    ] {
        for rank in 0..crate::contract::RANKS as usize {
            let (launches, [activation, live, arena, source]) =
                crate::instance::media_inject_probe(instance, rank, row0, rows);
            let skip = |span: Span, bytes: u64| {
                (bytes <= span.length)
                    .then(|| Span::new(span.buffer, span.offset + bytes, span.length - bytes))
            };
            let reference = match (skip(activation, row0 * hidden), skip(source, row0 * 8)) {
                (Some(activation), Some(source)) => {
                    prepare_override_rows_for(&V26_FLASH, activation, arena, source, live, rows)
                }
                _ => Err(Refusal("outside the buffers")),
            };
            let at = format!("rank {rank} rows {row0}+{rows}");
            let (launches, reference) = match (launches, reference) {
                (Ok(launches), Ok(reference)) => (launches, reference),
                (Err(_), Err(_)) => continue,
                (launches, reference) => panic!(
                    "{at}: {:?} against {:?}",
                    launches.map(|l| l.len()),
                    reference.map(|r| r.symbol)
                ),
            };
            let [launch] = <[PreparedLaunch; 1]>::try_from(launches)
                .unwrap_or_else(|_| panic!("{at}: one launch"));
            assert_eq!(launch.symbol, reference.symbol, "{at}");
            assert_eq!(launch.rows, reference.rows, "{at}");
            assert_eq!(launch.plan, reference.plan, "{at}");
            assert_eq!(launch.arguments, reference.arguments, "{at}");
            for argument in &launch.arguments.0 {
                if let Argument::DeviceBuffer(id) = argument {
                    assert_eq!(
                        launch.bindings.get(*id).unwrap(),
                        reference.bindings.get(*id).unwrap(),
                        "{at}"
                    );
                }
            }
            launched += 1;
        }
    }
    // Seven parts launch on every rank, the 512-row second half among them.
    assert_eq!(launched, 7 * crate::contract::RANKS);
}
