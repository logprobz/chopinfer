//! `--kv-cache-dtype`: which of a bundle's serving KV layouts to run.
//!
//! A package lists every KV layout its bundle can run, one per element, and
//! makes one of them its preferred `kv_layout`. `auto`, the default, runs that
//! preference; `bf16` and `fp8` run the named element and are refused when the
//! package does not declare it. A manifest that predates the list declares one
//! layout, FP8 when it requires the FP8 KV capability and BF16 otherwise.

use runtime_bundle::{BundleKvDtypeV1, BundleKvLayoutV1, BundleManifestV1};

/// The KV cache dtype an operator asked for.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvCacheDtypeV1 {
    /// The package's preferred layout.
    #[default]
    Auto,
    Bf16,
    Fp8,
}

impl KvCacheDtypeV1 {
    /// Parses `auto`, `bf16` (or `bfloat16`) and `fp8` (or `fp8_e4m3`).
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "auto" => Some(Self::Auto),
            "bf16" | "bfloat16" => Some(Self::Bf16),
            "fp8" | "fp8_e4m3" => Some(Self::Fp8),
            _ => None,
        }
    }
}

/// The serving KV layout resolved for one manifest at startup.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvCacheChoiceV1 {
    pub dtype: BundleKvDtypeV1,
    /// Whether `auto` chose it.
    pub automatic: bool,
}

impl KvCacheChoiceV1 {
    /// The resolved element: `bf16` or `fp8`.
    #[must_use]
    pub fn label(self) -> &'static str {
        label(self.dtype)
    }

    /// The startup line that records the choice.
    #[must_use]
    pub fn describe(self, manifest: &BundleManifestV1) -> String {
        let supported = declared_kv_formats_v1(manifest)
            .iter()
            .map(|(dtype, _)| label(*dtype))
            .collect::<Vec<_>>()
            .join(", ");
        let how = if self.automatic {
            "auto, the package's preference"
        } else {
            "requested"
        };
        format!(
            "serving KV cache dtype: {} ({how}; the package supports {supported})",
            label(self.dtype)
        )
    }
}

fn label(dtype: BundleKvDtypeV1) -> &'static str {
    match dtype {
        BundleKvDtypeV1::Bf16 => "bf16",
        BundleKvDtypeV1::Fp8E4m3 => "fp8",
    }
}

/// Every KV layout the manifest declares, by element, preferred one included.
#[must_use]
pub fn declared_kv_formats_v1(
    manifest: &BundleManifestV1,
) -> Vec<(BundleKvDtypeV1, &BundleKvLayoutV1)> {
    if manifest.kv_formats().is_empty() {
        return vec![(preferred_kv_dtype_v1(manifest), manifest.kv_layout())];
    }
    manifest
        .kv_formats()
        .iter()
        .map(|format| (format.dtype(), format.layout()))
        .collect()
}

/// The element of the manifest's preferred `kv_layout`.
#[must_use]
pub fn preferred_kv_dtype_v1(manifest: &BundleManifestV1) -> BundleKvDtypeV1 {
    if let Some(format) = manifest
        .kv_formats()
        .iter()
        .find(|format| format.layout() == manifest.kv_layout())
    {
        return format.dtype();
    }
    let legacy_fp8 = manifest
        .resource_requirements()
        .required_capabilities()
        .iter()
        .any(|capability| capability.as_str() == runtime_bundle_abi::FP8_KV_CAPABILITY_V1);
    if legacy_fp8 {
        BundleKvDtypeV1::Fp8E4m3
    } else {
        BundleKvDtypeV1::Bf16
    }
}

/// The declared layout of `dtype`, if the manifest has one.
#[must_use]
pub fn kv_layout_of_v1(
    manifest: &BundleManifestV1,
    dtype: BundleKvDtypeV1,
) -> Option<&BundleKvLayoutV1> {
    declared_kv_formats_v1(manifest)
        .into_iter()
        .find_map(|(declared, layout)| (declared == dtype).then_some(layout))
}

/// Resolves `requested` against what the manifest declares, refusing an
/// explicit dtype the package does not support.
pub fn resolve_kv_cache_dtype_v1(
    manifest: &BundleManifestV1,
    requested: KvCacheDtypeV1,
) -> Result<KvCacheChoiceV1, String> {
    let dtype = match requested {
        KvCacheDtypeV1::Auto => {
            return Ok(KvCacheChoiceV1 {
                dtype: preferred_kv_dtype_v1(manifest),
                automatic: true,
            });
        }
        KvCacheDtypeV1::Bf16 => BundleKvDtypeV1::Bf16,
        KvCacheDtypeV1::Fp8 => BundleKvDtypeV1::Fp8E4m3,
    };
    if kv_layout_of_v1(manifest, dtype).is_none() {
        let supported = declared_kv_formats_v1(manifest)
            .iter()
            .map(|(dtype, _)| label(*dtype))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "--kv-cache-dtype {}: bundle {} declares no {} KV layout; it supports {supported}",
            label(dtype),
            manifest.id(),
            label(dtype)
        ));
    }
    Ok(KvCacheChoiceV1 {
        dtype,
        automatic: false,
    })
}
