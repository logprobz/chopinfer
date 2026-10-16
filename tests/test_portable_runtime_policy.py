"""Tests for the portable-crate and ahead-of-time policy verifier.

The verifier replaces two case-insensitive text scans that could not tell a
policy declaration from a dependency or an executable route. These tests pin
the distinction: a payload-free enum variant that records a rejected or
declared capability is allowed, while a manifest dependency, an import, or a
callable route naming the same capability is not.
"""

import dataclasses
import importlib.util
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
VERIFIER_PATH = TESTS_DIR / "portable-runtime-policy.py"


def _load_verifier():
    specification = importlib.util.spec_from_file_location(
        "portable_runtime_policy", VERIFIER_PATH
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {VERIFIER_PATH}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


policy_module = _load_verifier()


WORKSPACE_MANIFEST = """\
[workspace]
members = ["crates/portable-a", "crates/native-b"]
resolver = "3"
"""

PORTABLE_MANIFEST = """\
[package]
name = "portable-a"
version = "0.1.0"
edition = "2024"

[dependencies]
serde = "1"
"""

NATIVE_MANIFEST = """\
[package]
name = "native-b"
version = "0.1.0"
edition = "2024"
"""

LOCKFILE = """\
version = 4

[[package]]
name = "portable-a"
version = "0.1.0"
dependencies = ["serde"]

[[package]]
name = "native-b"
version = "0.1.0"

[[package]]
name = "serde"
version = "1.0.0"
"""

DECLARATION_SOURCE = """\
//! Declared and rejected capability records.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RejectedGraphCapability {
    RuntimeJit,
    RuntimeAutotune,
    BackendFallback,
    DynamicKernelLoader,
    GenericDispatch,
}

const REJECTED_CAPABILITIES: [RejectedGraphCapability; 5] = [
    RejectedGraphCapability::RuntimeJit,
    RejectedGraphCapability::RuntimeAutotune,
    RejectedGraphCapability::BackendFallback,
    RejectedGraphCapability::DynamicKernelLoader,
    RejectedGraphCapability::GenericDispatch,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CollectiveArithmeticDeclarationId {
    Nccl23004LargeMessageSumV1,
}

impl CollectiveArithmeticDeclarationId {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nccl23004LargeMessageSumV1 => "declared large-message sum",
        }
    }
}

pub fn rejected() -> &'static [RejectedGraphCapability] {
    &REJECTED_CAPABILITIES
}

pub fn declared_collective() -> CollectiveArithmeticDeclarationId {
    CollectiveArithmeticDeclarationId::Nccl23004LargeMessageSumV1
}
"""


class PolicyFixture(unittest.TestCase):
    """Base class that materializes a small workspace under a temporary root."""

    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="portable-policy-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.policy = dataclasses.replace(
            policy_module.Policy(), portable_crates=("portable-a",)
        )
        self.write(
            {
                "Cargo.toml": WORKSPACE_MANIFEST,
                "Cargo.lock": LOCKFILE,
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST,
                "crates/portable-a/src/lib.rs": DECLARATION_SOURCE,
                "crates/native-b/Cargo.toml": NATIVE_MANIFEST,
                "crates/native-b/src/lib.rs": "pub fn native() {}\n",
            }
        )

    def write(self, files):
        for relative, payload in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                path.write_text(payload, encoding="utf-8")

    def verify(self):
        return policy_module.verify(self.root, self.policy)

    def assertClean(self):
        violations = self.verify()
        self.assertEqual([policy_module.render(item) for item in violations], [])

    def assertRejects(self, needle):
        violations = self.verify()
        rendered = "\n".join(policy_module.render(item) for item in violations)
        self.assertTrue(violations, f"expected a violation mentioning {needle!r}")
        self.assertIn(needle, rendered)
        return rendered


class ModelTreeCoverage(PolicyFixture):
    def test_model_source_is_scanned(self):
        self.write({"models/example/bundle/src/lib.rs": "pub fn RuntimeJit() {}\n"})
        self.assertRejects("models/example/bundle/src/lib.rs")

    def test_model_manifest_is_scanned(self):
        self.write({
            "models/example/bundle/src/lib.rs": "pub fn example() {}\n",
            "models/example/bundle/Cargo.toml": NATIVE_MANIFEST + '\n[dependencies]\nRuntimeJit = "1"\n',
        })
        self.assertRejects("models/example/bundle/Cargo.toml")

    def test_missing_declared_models_fails(self):
        self.write({"Cargo.toml": WORKSPACE_MANIFEST.replace("crates/native-b", "models/example/bundle")})
        with self.assertRaises(policy_module.PolicyError):
            self.verify()


class DeclarationsPass(PolicyFixture):
    def test_rejection_and_declaration_enums_pass(self):
        self.assertClean()

    def test_comments_and_strings_do_not_create_violations(self):
        self.write(
            {
                "crates/portable-a/src/prose.rs": (
                    "// This route never links NCCL or cudarc.\n"
                    "/* RuntimeJit and DynamicKernelLoader stay out of the build. */\n"
                    "/// Documents that NVRTC is absent.\n"
                    "pub const NOTES: &str = \"nccl, cudarc, NVRTC, RuntimeJit\";\n"
                    "pub const RAW: &str = r#\"cuda-sys BackendFallback\"#;\n"
                    "pub const LETTER: char = 'n';\n"
                ),
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + "\n# no cudarc, cuda-sys, or nccl dependency is permitted here\n",
            }
        )
        self.assertClean()

    def test_repository_tree_satisfies_the_shipped_policy(self):
        violations = policy_module.verify(REPO_ROOT)
        self.assertEqual([policy_module.render(item) for item in violations], [])


class ManifestDependenciesFail(PolicyFixture):
    def test_direct_forbidden_dependency_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + 'cudarc = { version = "0.11", default-features = false }\n'
            }
        )
        self.assertRejects("cudarc")

    def test_renamed_forbidden_dependency_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + 'accelerator = { version = "0.1", package = "cuda-sys" }\n'
            }
        )
        self.assertRejects("cuda-sys")

    def test_forbidden_dev_dependency_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + '\n[dev-dependencies]\nnccl-sys = "0.3"\n'
            }
        )
        self.assertRejects("nccl-sys")

    def test_forbidden_target_dependency_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + '\n[target."cfg(target_os = \\"linux\\")".dependencies]\n'
                + 'cudarc = "0.11"\n'
            }
        )
        self.assertRejects("cudarc")

    def test_forbidden_feature_reference_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + '\n[features]\naccel = ["dep:cudarc"]\n'
            }
        )
        self.assertRejects("cudarc")

    def test_forbidden_native_link_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST.replace(
                    'edition = "2024"\n', 'edition = "2024"\nlinks = "nccl"\n'
                )
            }
        )
        self.assertRejects("nccl")

    def test_forbidden_transitive_lock_dependency_fails(self):
        self.write(
            {
                "Cargo.lock": LOCKFILE.replace(
                    'dependencies = ["serde"]', 'dependencies = ["serde", "accel-glue"]'
                )
                + '\n[[package]]\nname = "accel-glue"\nversion = "0.1.0"\n'
                + 'dependencies = ["cudarc"]\n'
                + '\n[[package]]\nname = "cudarc"\nversion = "0.11.0"\n'
            }
        )
        self.assertRejects("cudarc")

    def test_forbidden_name_in_an_unmodelled_table_fails(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + '\n[lints]\ncudarc = { version = "0.11" }\n'
            }
        )
        self.assertRejects("cudarc")

    def test_forbidden_inherited_workspace_dependency_fails(self):
        self.write(
            {
                "Cargo.toml": WORKSPACE_MANIFEST
                + '\n[workspace.dependencies]\naccelerator = { package = "cudarc" }\n',
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST
                + "accelerator = { workspace = true }\n",
            }
        )
        self.assertRejects("cudarc")

    def test_manifest_prose_does_not_create_violations(self):
        self.write(
            {
                "crates/portable-a/Cargo.toml": PORTABLE_MANIFEST.replace(
                    'edition = "2024"\n',
                    'edition = "2024"\n'
                    'description = "Portable types that never link nccl or cudarc"\n'
                    'keywords = ["nccl-free", "no-nvrtc"]\n',
                )
            }
        )
        self.assertClean()

    def test_ahead_of_time_dependency_fails_in_any_crate(self):
        self.write(
            {
                "crates/native-b/Cargo.toml": NATIVE_MANIFEST
                + '\n[dependencies]\nnvrtc-sys = "0.1"\n'
            }
        )
        self.assertRejects("nvrtc-sys")


class SourceRoutesFail(PolicyFixture):
    def test_forbidden_import_fails(self):
        self.write(
            {
                "crates/portable-a/src/route.rs": (
                    "use cudarc::driver::CudaDevice;\n"
                    "pub fn device() -> Option<CudaDevice> {\n    None\n}\n"
                )
            }
        )
        self.assertRejects("cudarc")

    def test_forbidden_extern_crate_fails(self):
        self.write({"crates/portable-a/src/route.rs": "extern crate nccl_sys;\n"})
        self.assertRejects("nccl_sys")

    def test_forbidden_call_fails(self):
        self.write(
            {
                "crates/portable-a/src/route.rs": (
                    "pub fn reduce(handle: usize) -> i32 {\n"
                    "    unsafe { ncclAllReduce(handle) }\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("ncclAllReduce")

    def test_forbidden_extern_declaration_fails(self):
        self.write(
            {
                "crates/portable-a/src/route.rs": (
                    'unsafe extern "C" {\n'
                    "    pub fn nvrtcCompileProgram(program: usize) -> i32;\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("nvrtcCompileProgram")

    def test_forbidden_link_attribute_literal_fails(self):
        self.write(
            {
                "crates/portable-a/src/route.rs": (
                    '#[link(name = "nccl", kind = "dylib")]\n'
                    'unsafe extern "C" {\n'
                    "    pub fn reduce(handle: usize) -> i32;\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("nccl")

    def test_ahead_of_time_route_type_fails(self):
        self.write(
            {
                "crates/native-b/src/jit.rs": (
                    "pub struct RuntimeJit;\n"
                    "impl RuntimeJit {\n"
                    "    pub fn compile(&self, source: &str) -> usize {\n"
                    "        source.len()\n"
                    "    }\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("RuntimeJit")

    def test_ahead_of_time_route_call_fails(self):
        self.write(
            {
                "crates/native-b/src/jit.rs": (
                    "pub fn dispatch(source: &str) -> usize {\n"
                    "    crate::loader::DynamicKernelLoader::open(source)\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("DynamicKernelLoader")

    def test_ahead_of_time_route_in_cuda_source_fails(self):
        self.write(
            {
                "crates/native-b/csrc/kernel.cu": (
                    "#include <stdint.h>\n"
                    "int32_t launch(void) { return nvrtcVersion(); }\n"
                )
            }
        )
        self.assertRejects("nvrtcVersion")

    def test_variant_with_payload_is_not_a_declaration(self):
        self.write(
            {
                "crates/native-b/src/jit.rs": (
                    "pub struct Compiler;\n"
                    "pub enum Capability {\n"
                    "    RuntimeJit(Compiler),\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("RuntimeJit")

    def test_reference_through_an_undeclared_type_fails(self):
        self.write(
            {
                "crates/native-b/src/jit.rs": (
                    "pub fn route() {\n"
                    "    external_crate::BackendFallback::install();\n"
                    "}\n"
                )
            }
        )
        self.assertRejects("BackendFallback")

    def test_enum_named_for_a_forbidden_route_fails(self):
        self.write(
            {
                "crates/native-b/src/jit.rs": (
                    "pub enum BackendFallback {\n    Disabled,\n}\n"
                )
            }
        )
        self.assertRejects("BackendFallback")


class ScannerErrorsFailClosed(PolicyFixture):
    def test_undecodable_source_raises(self):
        self.write({"crates/portable-a/src/broken.rs": b"pub fn a() {}\n\xff\xfe\n"})
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_unterminated_block_comment_raises(self):
        self.write({"crates/portable-a/src/broken.rs": "/* unterminated\npub fn a() {}\n"})
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_unbalanced_delimiters_raise(self):
        self.write({"crates/portable-a/src/broken.rs": "pub fn a() { let b = 1; )\n"})
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_unreadable_manifest_raises(self):
        self.write({"crates/portable-a/Cargo.toml": "[package\nname = 'portable-a'\n"})
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_missing_scoped_crate_raises(self):
        shutil.rmtree(self.root / "crates/portable-a")
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_missing_lockfile_raises(self):
        (self.root / "Cargo.lock").unlink()
        with self.assertRaises(policy_module.PolicyError):
            self.verify()

    def test_scanner_error_exits_non_zero(self):
        self.write({"crates/portable-a/src/broken.rs": b"\xff\xfe\n"})
        completed = subprocess.run(
            [sys.executable, "-B", str(VERIFIER_PATH), "--root", str(self.root)],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("policy scan failed", completed.stderr)


class CommandLine(unittest.TestCase):
    """Drive the command line against a tree shaped like the shipped policy."""

    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp(prefix="portable-policy-cli-"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        crates = policy_module.PORTABLE_CRATES
        members = ", ".join(f'"crates/{crate}"' for crate in crates)
        files = {
            "Cargo.toml": f"[workspace]\nmembers = [{members}]\nresolver = \"3\"\n",
            "Cargo.lock": "version = 4\n"
            + "".join(
                f'\n[[package]]\nname = "{crate}"\nversion = "0.1.0"\n' for crate in crates
            ),
        }
        for crate in crates:
            files[f"crates/{crate}/Cargo.toml"] = (
                f'[package]\nname = "{crate}"\nversion = "0.1.0"\nedition = "2024"\n'
            )
            files[f"crates/{crate}/src/lib.rs"] = "pub fn placeholder() {}\n"
        files[f"crates/{crates[0]}/src/lib.rs"] = DECLARATION_SOURCE
        for relative, payload in files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")

    def run_verifier(self):
        return subprocess.run(
            [sys.executable, "-B", str(VERIFIER_PATH), "--root", str(self.root)],
            capture_output=True,
            text=True,
        )

    def test_clean_tree_exits_zero(self):
        completed = self.run_verifier()
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_violation_exits_one(self):
        target = self.root / "crates" / policy_module.PORTABLE_CRATES[0] / "src" / "route.rs"
        target.write_text("use cudarc::driver::Cuda;\n", encoding="utf-8")
        completed = self.run_verifier()
        self.assertEqual(completed.returncode, 1)
        self.assertIn("cudarc", completed.stderr)

    def test_repository_default_root_exits_zero(self):
        completed = subprocess.run(
            [sys.executable, "-B", "tests/portable-runtime-policy.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
