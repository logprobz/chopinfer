//! The steps a bundle's create takes to acquire its host resources.
//!
//! Each step holds what it acquires in the caller's [`Held`] list, newest
//! last, so a failed create releases it. The bundle composes the steps in
//! its own order (which images, which cubins, which buffers) and keeps the
//! meaning of every view; the host calls and their checks are here.
//!
//! A rank image's views are placed by class ([`Placement`]): in the rank's
//! device arena, in page-locked device-mapped host memory the host reads a
//! view into (`chop.host-resident-views.v1`), and either of those holding
//! only this rank's rows of a table split by rows across the ranks. A model
//! whose weights do not fit its devices declares host or row-sharded views
//! rather than a different layout.

use std::collections::BTreeMap;

use runtime_bundle_abi::{
    BundleAbiError, BundleCompiledImageHandleV1, BundleContextHandleV1, BundleCudaSymbolHandleV1,
    BundleMemoryPoolHandleV1, BundleRankArenaHandleV1, BundleStatusV1, BundleStreamHandleV1,
    DeviceBufferSpanV1, HostServicesV1, RankArenaUploadV1, StateRequestV1,
};

use crate::index::RankIndex;
use crate::resources::Held;

/// One rank's context, memory pool and first stream.
pub type RankHandles = (
    BundleContextHandleV1,
    BundleMemoryPoolHandleV1,
    BundleStreamHandleV1,
);

/// The host services of one bundle's create, logging each failed service
/// as `<bundle> create <service>: <error>`.
#[derive(Clone, Copy)]
pub struct Log<'a> {
    pub services: &'a HostServicesV1,
    pub bundle: &'static str,
}

impl Log<'_> {
    /// `result` of the service `name`, logged when it failed.
    pub fn named<T>(
        &self,
        name: &str,
        result: Result<T, BundleAbiError>,
    ) -> Result<T, BundleAbiError> {
        result.map_err(|error| {
            let _ = self
                .services
                .log(format!("{} create {name}: {error}", self.bundle).as_bytes());
            error
        })
    }
}

/// The bytes of the authenticated component `role`, held open for the
/// instance's life. Refuses a component past `max_bytes` or under another
/// role.
pub fn read_component(
    log: Log,
    role: &str,
    max_bytes: u64,
    held: &mut Vec<Held>,
) -> Result<Vec<u8>, BundleAbiError> {
    let services = log.services;
    let component = log.named(
        &format!("open_authenticated_component {role}"),
        services.open_authenticated_component(role.as_bytes()),
    )?;
    held.push(Held::Component(component));
    let metadata = log.named(
        &format!("inspect_authenticated_component {role}"),
        services.inspect_authenticated_component(component),
    )?;
    if metadata.role_length != role.len() as u64 || metadata.byte_length > max_bytes {
        return Err(BundleAbiError::InvalidRange(
            "authenticated component metadata",
        ));
    }
    let mut role_bytes = vec![0; role.len()];
    let mut bytes = vec![0; metadata.byte_length as usize];
    log.named(
        &format!("read_authenticated_component {role}"),
        services.read_authenticated_component(component, &mut role_bytes, &mut bytes),
    )?;
    if role_bytes != role.as_bytes() {
        return Err(BundleAbiError::InvalidRange("authenticated component role"));
    }
    Ok(bytes)
}

/// Selects each rank's context on its device ordinal, its memory pool, and
/// creates its first stream.
pub fn open_ranks(
    log: Log,
    ordinals: &[i32],
    held: &mut Vec<Held>,
) -> Result<Vec<RankHandles>, BundleAbiError> {
    let services = log.services;
    let mut ranks = Vec::with_capacity(ordinals.len());
    for (rank, ordinal) in ordinals.iter().enumerate() {
        let rank = rank as u32;
        let context = log.named(
            &format!("rank {rank} select_context ordinal {ordinal}"),
            services.select_context(rank, *ordinal),
        )?;
        held.push(Held::Context(context));
        let pool = services.select_memory_pool(context)?;
        held.push(Held::Pool(pool));
        let stream = services.create_stream(context, 0)?;
        held.push(Held::Stream(stream));
        ranks.push((context, pool, stream));
    }
    Ok(ranks)
}

/// Opens the compiled image `role`.
pub fn open_image(
    log: Log,
    role: &[u8],
    held: &mut Vec<Held>,
) -> Result<BundleCompiledImageHandleV1, BundleAbiError> {
    let image = log.named(
        &format!("open_compiled_image {}", String::from_utf8_lossy(role)),
        log.services.open_compiled_image(role),
    )?;
    held.push(Held::Image(image));
    Ok(image)
}

/// What a rank index binds its rank's image to: the image's length and
/// either its file's SHA-256 or its plan's identity
/// ([`crate::index::authenticate_rank_identity`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Identity<'a> {
    pub image_bytes: u64,
    pub sha256: Option<&'a str>,
    pub direct_identity: Option<&'a str>,
}

/// The identity a shared-schema rank index names.
pub fn identity(index: &RankIndex) -> Identity<'_> {
    Identity {
        image_bytes: index.image_bytes,
        sha256: index.sha256.as_deref(),
        direct_identity: index.direct_identity.as_deref(),
    }
}

/// A compiled image split into one rank image per rank, each described by
/// a rank index the package supplies as an authenticated component.
#[derive(Clone, Copy)]
pub struct RankImages<'a> {
    /// The compiled image's role.
    pub role: &'a [u8],
    /// The component role of rank `r`'s index.
    pub index_role: &'a dyn Fn(u32) -> String,
    /// The most bytes one rank's index may hold.
    pub max_index_bytes: u64,
}

/// Binds rank `rank`'s index, naming `identity`, to the image the host
/// serves for that rank of the image `image_role`
/// (`chop.compiled-image-identity.v1/<image role>/<rank>`). A mismatch is
/// logged as `<bundle> rank <rank> index names another image`.
pub fn authenticate(
    log: Log,
    image_role: &[u8],
    rank: u32,
    identity: Identity,
) -> Result<(), BundleAbiError> {
    let services = log.services;
    let image_role = std::str::from_utf8(image_role)
        .map_err(|_| BundleAbiError::InvalidRange("compiled image role"))?;
    crate::index::authenticate_rank_identity(
        services,
        image_role,
        rank,
        identity.image_bytes,
        identity.sha256,
        identity.direct_identity,
    )
    .inspect_err(|_| {
        let _ = services
            .log(format!("{} rank {rank} index names another image", log.bundle).as_bytes());
    })
}

/// Reads the index of each of `ranks` ranks of `image`, admits it with
/// `admit` (the bundle's parse and layout checks) and authenticates the
/// identity it names ([`authenticate`]), rank by rank; only then opens the
/// image. No rank image is opened or uploaded while any rank's index is
/// malformed, inadmissible or names another image. Returns the image and
/// the admitted indexes in rank order.
pub fn rank_images<I>(
    log: Log,
    image: RankImages,
    ranks: u32,
    mut admit: impl FnMut(u32, &[u8]) -> Result<I, BundleAbiError>,
    identity: impl Fn(&I) -> Identity<'_>,
    held: &mut Vec<Held>,
) -> Result<(BundleCompiledImageHandleV1, Vec<I>), BundleAbiError> {
    let mut indexes = Vec::with_capacity(ranks as usize);
    for rank in 0..ranks {
        let bytes = read_component(log, &(image.index_role)(rank), image.max_index_bytes, held)?;
        let index = admit(rank, &bytes)?;
        authenticate(log, image.role, rank, identity(&index))?;
        indexes.push(index);
    }
    let handle = open_image(log, image.role, held)?;
    Ok((handle, indexes))
}

/// Uploads `image`'s rank images to every rank's arena at once.
pub fn upload_arenas(
    log: Log,
    image: BundleCompiledImageHandleV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<BundleRankArenaHandleV1>, BundleAbiError> {
    let uploads = ranks
        .iter()
        .enumerate()
        .map(|(rank, &(context, pool, stream))| RankArenaUploadV1 {
            context,
            pool,
            stream,
            image,
            rank: rank as u32,
        })
        .collect::<Vec<_>>();
    let arenas = log.named(
        "upload_rank_arenas",
        log.services.upload_rank_arenas(&uploads),
    )?;
    held.extend(arenas.iter().map(|&arena| Held::Arena(arena)));
    Ok(arenas)
}

/// Rows `[start, start + rows)` of a table split by rows across the ranks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RowShard {
    pub start: u64,
    pub rows: u64,
}

/// How a package may place one view of a rank image. A view without one
/// is in the device arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Placement<'a> {
    /// The view's name in the rank index.
    pub name: &'a str,
    /// The view id the host resolves the view under when the package
    /// placed it in pinned host memory; the view is in the arena when the
    /// host does not know the id.
    pub host_view: Option<&'a [u8]>,
    /// This rank's rows of a table split by rows across the ranks: the
    /// view holds exactly these rows.
    pub rows: Option<RowShard>,
}

/// Where a placed view's bytes are.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Residency {
    DeviceArena,
    /// Page-locked host memory the device reads over PCIe.
    PinnedHost,
}

/// One view of a rank image, placed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Placed {
    pub span: DeviceBufferSpanV1,
    pub residency: Residency,
    pub rows: Option<RowShard>,
}

/// Why a rank image's views could not be placed.
#[derive(Debug)]
pub enum PlaceError {
    Host(BundleAbiError),
    /// The views, the arena and the host views do not agree.
    Layout(String),
}

impl From<BundleAbiError> for PlaceError {
    fn from(error: BundleAbiError) -> Self {
        Self::Host(error)
    }
}

/// Places every view `index` names on its rank, whose image arena is
/// `arena`. A view `placements` lets the package put in host memory is
/// there when the host resolves its view id: the host cut it out of the
/// arena, its end rounded up to the image alignment and clipped to the
/// image, and the views after it moved down by that length
/// (`chop.host-resident-views.v1`). Every other view is
/// at its offset in the arena, which must hold the image less its cuts.
pub fn place_views(
    services: &HostServicesV1,
    image: BundleCompiledImageHandleV1,
    arena: BundleRankArenaHandleV1,
    index: &RankIndex,
    placements: &[Placement],
) -> Result<BTreeMap<String, Placed>, PlaceError> {
    place_views_mapped(services, image, arena, index, placements, &[])
}

/// [`place_views`] for an image whose plan also maps the views `mapped`
/// (their names in the rank index) from the checkpoint
/// (`chop.host-mapped-gather.v1`). The host cut each out of the arena as it
/// cuts a host view, so the views after it moved down by its rounded length.
/// A mapped view has no device span and is not placed: the bundle copies
/// rows of it with `HostServicesV1::gather_mapped_rows`.
pub fn place_views_mapped(
    services: &HostServicesV1,
    image: BundleCompiledImageHandleV1,
    arena: BundleRankArenaHandleV1,
    index: &RankIndex,
    placements: &[Placement],
    mapped: &[&str],
) -> Result<BTreeMap<String, Placed>, PlaceError> {
    let rank = index.rank;
    let whole = services.resolve_compiled_image_view(image, arena, rank, b"")?;
    let mut hosted = BTreeMap::new();
    for placement in placements {
        let Some(view_id) = placement.host_view else {
            continue;
        };
        match services.resolve_compiled_image_view(image, arena, rank, view_id) {
            Ok(host) => {
                hosted.insert(placement.name, host.buffer);
            }
            Err(BundleAbiError::Status(BundleStatusV1::NotFound)) => {}
            Err(error) => return Err(error.into()),
        }
    }
    place_mapped(
        index,
        whole.buffer,
        whole.alignment,
        &hosted,
        placements,
        mapped,
    )
    .map_err(PlaceError::Layout)
}

/// [`place_views`] once the host resolved the arena (`arena`, the image's
/// `alignment`) and the views it holds in host memory (`hosted`).
pub fn place(
    index: &RankIndex,
    arena: DeviceBufferSpanV1,
    alignment: u64,
    hosted: &BTreeMap<&str, DeviceBufferSpanV1>,
    placements: &[Placement],
) -> Result<BTreeMap<String, Placed>, String> {
    place_mapped(index, arena, alignment, hosted, placements, &[])
}

/// [`place`] with the views `mapped` cut out of the arena and left unplaced
/// ([`place_views_mapped`]).
pub fn place_mapped(
    index: &RankIndex,
    arena: DeviceBufferSpanV1,
    alignment: u64,
    hosted: &BTreeMap<&str, DeviceBufferSpanV1>,
    placements: &[Placement],
    mapped: &[&str],
) -> Result<BTreeMap<String, Placed>, String> {
    let rank = index.rank;
    let view = |name: &str| {
        index
            .views
            .get(name)
            .ok_or_else(|| format!("rank {rank} lacks view {name}"))
    };
    for placement in placements {
        let view = view(placement.name)?;
        if let Some(rows) = placement.rows
            && (rows.rows == 0 || view.bytes % rows.rows != 0)
        {
            return Err(format!(
                "rank {rank} view {} of {} bytes does not hold {} whole rows",
                placement.name, view.bytes, rows.rows
            ));
        }
    }
    // The image ranges the host cut out, in image order.
    let mut cuts = Vec::with_capacity(hosted.len() + mapped.len());
    for name in hosted.keys().chain(mapped) {
        let view = view(name)?;
        if let Some(host) = hosted.get(name) {
            if mapped.contains(name) {
                return Err(format!("rank {rank} view {name} is both host and mapped"));
            }
            if host.length != view.bytes {
                return Err(format!(
                    "rank {rank} host view {name} holds {} bytes, the index {}",
                    host.length, view.bytes
                ));
            }
        }
        // As the host cuts it: the view's end rounded up to the image
        // alignment, clipped to the image.
        let end = view
            .offset
            .checked_add(view.bytes)
            .and_then(|end| end.checked_next_multiple_of(alignment))
            .ok_or_else(|| format!("rank {rank} host view {name} has no aligned end"))?;
        cuts.push((view.offset, end.min(index.image_bytes)));
    }
    cuts.sort_unstable();
    if cuts.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(format!("rank {rank} host and mapped views overlap"));
    }
    let cut_bytes = cuts.iter().map(|(start, end)| end - start).sum::<u64>();
    if index.image_bytes.checked_sub(cut_bytes) != Some(arena.length) {
        return Err(format!(
            "rank {rank} arena holds {} bytes, the image {} less {cut_bytes} host bytes",
            arena.length, index.image_bytes
        ));
    }
    let rows = |name: &str| {
        placements
            .iter()
            .find(|placement| placement.name == name)
            .and_then(|placement| placement.rows)
    };
    index
        .views
        .iter()
        .filter(|(name, _)| !mapped.contains(&name.as_str()))
        .map(|(name, view)| {
            if let Some(&span) = hosted.get(name.as_str()) {
                let placed = Placed {
                    span,
                    residency: Residency::PinnedHost,
                    rows: rows(name),
                };
                return Ok((name.clone(), placed));
            }
            let end = view
                .offset
                .checked_add(view.bytes)
                .ok_or_else(|| format!("rank {rank} view {name} end"))?;
            let mut removed = 0;
            for &(start, cut_end) in &cuts {
                if end <= start {
                    break;
                }
                if view.offset < cut_end {
                    return Err(format!(
                        "rank {rank} view {name} overlaps a host view's cut"
                    ));
                }
                removed += cut_end - start;
            }
            let offset = arena
                .offset
                .checked_add(view.offset - removed)
                .ok_or_else(|| format!("rank {rank} view {name} offset"))?;
            let placed = Placed {
                span: DeviceBufferSpanV1::new(arena.buffer, offset, view.bytes),
                residency: Residency::DeviceArena,
                rows: rows(name),
            };
            Ok((name.clone(), placed))
        })
        .collect()
}

/// The spans of placed views, by name.
pub fn spans(placed: BTreeMap<String, Placed>) -> BTreeMap<String, DeviceBufferSpanV1> {
    placed
        .into_iter()
        .map(|(name, placed)| (name, placed.span))
        .collect()
}

/// Grants every rank's context access to every other rank's memory, and
/// every rank's pool to every other rank.
pub fn grant_peers(
    services: &HostServicesV1,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<(), BundleAbiError> {
    for (owner, pool, _) in ranks {
        for (peer, _, _) in ranks {
            if owner != peer {
                let lease = services.acquire_peer_access(*owner, *peer)?;
                held.push(Held::Peer(lease));
                services.set_memory_pool_peer_access(*pool, *peer, true)?;
                held.push(Held::PoolPeer(*pool, *peer));
            }
        }
    }
    Ok(())
}

/// Dynamic shared memory a symbol may take before it must be configured
/// for more.
pub const DEFAULT_SHARED_BYTES: u32 = 48 * 1024;

/// Configures a resolved symbol only when it needs more dynamic shared
/// memory than the device's default.
pub fn configure_symbol(
    log: Log,
    symbol: &str,
    function: BundleCudaSymbolHandleV1,
    shared: u32,
) -> Result<(), BundleAbiError> {
    if shared > DEFAULT_SHARED_BYTES {
        log.named(
            &format!("configure_cuda_symbol {symbol}"),
            log.services.configure_cuda_symbol(function, shared),
        )?;
    }
    Ok(())
}

/// Loads the cubin `(name, bytes)` into `context` and resolves each of
/// `symbols` into `resolved`, configuring each that takes more than the
/// default dynamic shared memory with `shared_bytes(symbol)`.
pub fn load_module<'a>(
    log: Log,
    context: BundleContextHandleV1,
    (name, bytes): (&str, &[u8]),
    symbols: impl IntoIterator<Item = &'a str>,
    mut shared_bytes: impl FnMut(&str) -> Result<u32, BundleAbiError>,
    resolved: &mut BTreeMap<String, BundleCudaSymbolHandleV1>,
    held: &mut Vec<Held>,
) -> Result<(), BundleAbiError> {
    let services = log.services;
    let module = log.named(
        &format!("load_cuda_module {name}"),
        services.load_cuda_module(context, bytes),
    )?;
    held.push(Held::Module(module));
    for symbol in symbols {
        let function = log.named(
            &format!("resolve_cuda_symbol {symbol}"),
            services.resolve_cuda_symbol(module, symbol.as_bytes()),
        )?;
        held.push(Held::Symbol(function));
        if resolved.insert(symbol.to_owned(), function).is_some() {
            return Err(BundleAbiError::InvalidRange("duplicate CUDA symbol"));
        }
        configure_symbol(log, symbol, function, shared_bytes(symbol)?)?;
    }
    Ok(())
}

/// Allocates the host-owned request state `state_id` of `bytes` per rank
/// and zeroes each rank's view on its stream.
pub fn zeroed_state(
    services: &HostServicesV1,
    state_id: &[u8],
    bytes: u64,
    ranks: &[RankHandles],
    held: &mut Vec<Held>,
) -> Result<Vec<DeviceBufferSpanV1>, BundleAbiError> {
    let state = services.allocate_state(&StateRequestV1 {
        flags: 0,
        byte_length: bytes,
        alignment: 256,
        state_id: state_id.to_vec(),
    })?;
    held.push(Held::State(state));
    let mut views = Vec::with_capacity(ranks.len());
    for (rank, (_, _, stream)) in ranks.iter().enumerate() {
        let span = services.resolve_state_view(state, rank as u32)?;
        if span.length != bytes {
            return Err(BundleAbiError::InvalidRange("request state length"));
        }
        services.zero_device_buffer(*stream, span)?;
        views.push(span);
    }
    Ok(views)
}

/// The host's serving KV pool of layout `layout_id`: its page count and
/// each of `ranks` ranks' view, `page_bytes` per page.
pub fn serving_kv(
    log: Log,
    layout_id: &[u8],
    page_bytes: u64,
    ranks: u32,
) -> Result<(u64, Vec<DeviceBufferSpanV1>), BundleAbiError> {
    let services = log.services;
    let pages = log.named("acquire_serving_kv", services.acquire_serving_kv(layout_id))?;
    if pages == 0 {
        return Err(BundleAbiError::InvalidRange("empty serving KV pool"));
    }
    let bytes = pages
        .checked_mul(page_bytes)
        .ok_or(BundleAbiError::InvalidRange(
            "serving KV pool length overflow",
        ))?;
    let mut views = Vec::with_capacity(ranks as usize);
    for rank in 0..ranks {
        let view = log.named(
            &format!("rank {rank} resolve_serving_kv_pool"),
            services.resolve_serving_kv_pool(rank),
        )?;
        if view.length != bytes {
            return Err(BundleAbiError::InvalidRange("serving KV pool length"));
        }
        views.push(view);
    }
    Ok((pages, views))
}

/// Waits for every rank's stream: each rank's collective state is zero
/// before any peer can post to it.
pub fn synchronize(log: Log, ranks: &[RankHandles]) -> Result<(), BundleAbiError> {
    for (rank, (_, _, stream)) in ranks.iter().enumerate() {
        log.named(
            &format!("rank {rank} create synchronize_stream"),
            log.services.synchronize_stream(*stream),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use runtime_bundle_abi::BundleDeviceBufferHandleV1;

    use super::*;
    use crate::index::View;

    fn view(offset: u64, bytes: u64) -> View {
        View {
            offset,
            bytes,
            dtype: "U8".into(),
            shape: vec![bytes],
            layout_id: "rows".into(),
        }
    }

    /// The host rounds a host view's end up to the image alignment and
    /// clips it to the image (`host_view_excision_v1`): a table that ends
    /// the image leaves no padding past it to cut.
    #[test]
    fn a_terminal_host_view_is_cut_only_to_the_image_end() {
        let mut index = RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: 12288,
            sha256: Some("0".repeat(64)),
            direct_identity: None,
            views: BTreeMap::new(),
        };
        index.views.insert("head".into(), view(0, 8192));
        index.views.insert("table".into(), view(8192, 4000));
        let span = |buffer, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, buffer),
                offset,
                length,
            )
        };
        let placements = [Placement {
            name: "table",
            host_view: Some(b"table.v1"),
            rows: None,
        }];
        let host = BTreeMap::from([("table", span(2, 0, 4000))]);
        let placed = place(&index, span(1, 0, 8192), 8192, &host, &placements).unwrap();
        assert_eq!(placed["head"].span, span(1, 0, 8192));
        assert_eq!(placed["table"].residency, Residency::PinnedHost);
        // An arena missing the clipped-off padding as well is refused.
        assert!(place(&index, span(1, 0, 4096), 8192, &host, &placements).is_err());
        // So is an image alignment the host cannot round to.
        assert!(place(&index, span(1, 0, 8192), 0, &host, &placements).is_err());
    }

    #[test]
    fn a_host_view_moves_later_views_down_by_its_rounded_cut() {
        let mut index = RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: 1536,
            sha256: Some("0".repeat(64)),
            direct_identity: None,
            views: BTreeMap::new(),
        };
        index.views.insert("head".into(), view(0, 256));
        index.views.insert("table".into(), view(256, 1000));
        index.views.insert("tail".into(), view(1280, 256));
        let span = |buffer, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, buffer),
                offset,
                length,
            )
        };
        let shard = RowShard { start: 40, rows: 8 };
        let placements = [Placement {
            name: "table",
            host_view: Some(b"table.v1"),
            rows: Some(shard),
        }];
        let device = place(
            &index,
            span(1, 64, 1536),
            256,
            &BTreeMap::new(),
            &placements,
        )
        .unwrap();
        assert_eq!(device["tail"].span, span(1, 64 + 1280, 256));
        assert_eq!(device["table"].residency, Residency::DeviceArena);
        assert_eq!(device["table"].rows, Some(shard));
        assert!(place(&index, span(1, 0, 512), 256, &BTreeMap::new(), &placements).is_err());
        let host = BTreeMap::from([("table", span(2, 0, 1000))]);
        let placed = place(&index, span(1, 0, 512), 256, &host, &placements).unwrap();
        assert_eq!(placed["head"].span, span(1, 0, 256));
        assert_eq!(placed["table"].span, span(2, 0, 1000));
        assert_eq!(placed["table"].residency, Residency::PinnedHost);
        assert_eq!(placed["tail"].span, span(1, 256, 256));
        // The host view must be the table, and the arena must have lost it.
        let short = BTreeMap::from([("table", span(2, 0, 999))]);
        assert!(place(&index, span(1, 0, 512), 256, &short, &placements).is_err());
        assert!(place(&index, span(1, 0, 1536), 256, &host, &placements).is_err());
        // A view inside the cut has no arena address; rows must divide the
        // view.
        let mut torn = index.clone();
        torn.views.insert("torn".into(), view(1260, 20));
        assert!(place(&torn, span(1, 0, 512), 256, &host, &placements).is_err());
        let uneven = [Placement {
            rows: Some(RowShard { start: 0, rows: 3 }),
            ..placements[0]
        }];
        assert!(place(&index, span(1, 0, 1536), 256, &BTreeMap::new(), &uneven).is_err());
    }

    /// A mapped view leaves the arena like a host view and is not placed;
    /// the views after both cuts move down by both rounded lengths.
    #[test]
    fn mapped_views_are_cut_out_and_left_unplaced() {
        let mut index = RankIndex {
            rank: 0,
            image: "rank0.img".into(),
            image_bytes: 2048,
            sha256: Some("0".repeat(64)),
            direct_identity: None,
            views: BTreeMap::new(),
        };
        index.views.insert("head".into(), view(0, 256));
        index.views.insert("codes".into(), view(256, 700));
        index.views.insert("table".into(), view(1024, 256));
        index.views.insert("tail".into(), view(1280, 256));
        let span = |buffer, offset, length| {
            DeviceBufferSpanV1::new(
                BundleDeviceBufferHandleV1::from_parts(1, buffer),
                offset,
                length,
            )
        };
        let placements = [Placement {
            name: "table",
            host_view: Some(b"table.v1"),
            rows: None,
        }];
        let host = BTreeMap::from([("table", span(2, 0, 256))]);
        // 2048 less the codes rounded to 768 and the table's 256.
        let placed = place_mapped(
            &index,
            span(1, 0, 1024),
            256,
            &host,
            &placements,
            &["codes"],
        )
        .unwrap();
        assert!(!placed.contains_key("codes"));
        assert_eq!(placed["head"].span, span(1, 0, 256));
        assert_eq!(placed["table"].residency, Residency::PinnedHost);
        assert_eq!(placed["tail"].span, span(1, 256, 256));
        // An arena that still holds the mapped bytes, a view the index does
        // not name, and a view both host and mapped are refused.
        assert!(place(&index, span(1, 0, 1024), 256, &host, &placements).is_err());
        assert!(
            place_mapped(
                &index,
                span(1, 0, 1024),
                256,
                &host,
                &placements,
                &["absent"]
            )
            .is_err()
        );
        assert!(
            place_mapped(
                &index,
                span(1, 0, 1024),
                256,
                &host,
                &placements,
                &["table"]
            )
            .is_err()
        );
    }
}
