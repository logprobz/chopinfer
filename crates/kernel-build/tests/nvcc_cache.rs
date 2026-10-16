//! Real builds through the cubin cache wrapper (tools/nvcc_cache). A change
//! that alters the cubin outside the source closure must not be served from
//! the cache: the cached build fails the lock exactly as an uncached one does.
//! Each build runs in a child process, since the wrapper, toolkit and
//! preprocessor settings come from the environment.

use kernel_build::{BuildOptions, CubinSpec, EntryParam, EntrySpec, Toolchain, build};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const JOB: &str = "KERNEL_BUILD_CACHE_JOB";
const FMA_BODY: &str = "out[threadIdx.x] = input[0] * input[1] + input[2];";

fn wrapper() -> Option<PathBuf> {
    let wrapper =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/nvcc_cache/nvcc_cache.py");
    let available = env::consts::OS == "linux"
        && Path::new("/usr/local/cuda/bin/nvcc").exists()
        && wrapper.exists();
    if !available {
        eprintln!("SKIP: cubin cache builds require Linux, nvcc and tools/nvcc_cache");
        return None;
    }
    Some(fs::canonicalize(wrapper).unwrap())
}

/// One build, run when `Scenario::build` re-executes this binary.
#[test]
fn cache_job() {
    let Ok(job) = env::var(JOB) else {
        return;
    };
    let [root, out, lock, mode, define, body] = job.split('\n').collect::<Vec<_>>()[..] else {
        panic!("invalid {JOB}");
    };
    let spec = CubinSpec {
        name: "cache_probe".into(),
        sources_root: PathBuf::from(root),
        include_dirs: vec![PathBuf::from(".")],
        headers: vec![],
        defines: define
            .split_once('=')
            .map(|(name, value)| (name.into(), value.into()))
            .into_iter()
            .collect(),
        entries: vec![EntrySpec::new(
            "cache_probe_v1".into(),
            vec![
                EntryParam::DevicePointer {
                    name: "out".into(),
                    cxx_type: "float".into(),
                },
                EntryParam::DevicePointer {
                    name: "input".into(),
                    cxx_type: "const float".into(),
                },
            ],
            body.into(),
            32,
            1,
        )],
    };
    let options = BuildOptions {
        lock_update: mode == "update",
        emit_cargo_directives: false,
        profile_line_info: false,
    };
    match build(
        &Toolchain::detect().unwrap(),
        &[spec],
        Path::new(out),
        Path::new(lock),
        options,
    ) {
        Ok(built) => println!("cubin={}", built[0].sha256),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

struct Scenario {
    dir: PathBuf,
    wrapper: PathBuf,
    body: &'static str,
    /// One spec define, `NAME=VALUE`, or empty.
    define: &'static str,
    env: Vec<(&'static str, String)>,
}
impl Scenario {
    fn new(name: &str, wrapper: PathBuf, body: &'static str) -> Self {
        let dir = env::temp_dir().join(format!("kernel-build-cache-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        Self {
            dir,
            wrapper,
            body,
            define: "",
            env: Vec::new(),
        }
    }
    /// Build into a fresh output directory; Err holds the failure message.
    fn build(&self, label: &str, update: bool, cached: bool) -> Result<String, String> {
        let job = [
            self.dir.join("src"),
            self.dir.join(label),
            self.dir.join("kernels.lock"),
        ]
        .map(|path| path.display().to_string())
        .join("\n");
        let mode = if update { "update" } else { "verify" };
        let mut command = Command::new(env::current_exe().unwrap());
        command
            .args(["cache_job", "--exact", "--nocapture", "--test-threads=1"])
            .env(
                JOB,
                format!("{job}\n{mode}\n{}\n{}", self.define, self.body),
            )
            .env("CHOPINFER_NVCC_CACHE_DIR", self.dir.join("cache"))
            .env("CHOPINFER_KERNEL_BUILD_JOBS", "1");
        for name in [
            "CHOPINFER_NVCC_WRAPPER",
            "CHOPINFER_NVCC_CACHE_LOG",
            "CHOPINFER_CUDA_HOME",
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
            "SOURCE_DATE_EPOCH",
            "PTXAS_FLAGS",
        ] {
            command.env_remove(name);
        }
        if cached {
            command.env("CHOPINFER_NVCC_WRAPPER", &self.wrapper);
        }
        command.envs(self.env.iter().map(|(name, value)| (name, value)));
        let output = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() {
            // libtest prints the test name on the same line.
            let cubin = stdout
                .split("cubin=")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next());
            Ok(cubin.expect("child build printed its cubin").into())
        } else {
            Err(stderr.into_owned())
        }
    }
    fn set(&mut self, name: &'static str, value: impl Into<String>) {
        self.env.retain(|(existing, _)| *existing != name);
        self.env.push((name, value.into()));
    }
    fn events(&self) -> String {
        fs::read_to_string(self.dir.join("cache/v1/events.log")).unwrap_or_default()
    }
    /// After `change`, the cached build must fail the lock like an uncached
    /// build, and a lock rewrite must record a different cubin.
    fn assert_change_is_seen(&mut self, change: impl FnOnce(&mut Self)) {
        let cold = self.build("cold", true, true).unwrap();
        change(self);
        let cached = self.build("cached", false, true);
        let uncached = self.build("uncached", false, false);
        for (label, result) in [("cached", &cached), ("uncached", &uncached)] {
            let error = result.as_ref().expect_err(label);
            assert!(error.contains("kernels.lock mismatch"), "{label}: {error}");
        }
        assert_ne!(self.build("relocked", true, false).unwrap(), cold);
    }
}
impl Drop for Scenario {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn an_unchanged_build_hits() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    let scenario = Scenario::new("hit", wrapper, FMA_BODY);
    let cold = scenario.build("cold", true, true).unwrap();
    assert_eq!(scenario.build("warm", false, true).unwrap(), cold);
    let events = scenario.events();
    assert!(
        events.contains(" refresh ") && events.contains(" hit "),
        "{events}"
    );
}

#[test]
fn an_options_file_change_is_not_served_from_the_cache() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    let mut scenario = Scenario::new("options-file", wrapper, FMA_BODY);
    let options = scenario.dir.join("options");
    fs::write(&options, "--fmad=true\n").unwrap();
    scenario.set("NVCC_APPEND_FLAGS", format!("-optf {}", options.display()));
    scenario.assert_change_is_seen(|_| fs::write(&options, "--fmad=false\n").unwrap());
    let events = scenario.events();
    assert!(
        events.contains(" bypass ") && !events.contains(" hit "),
        "{events}"
    );
}

#[test]
fn an_nvcc_profile_change_is_not_served_from_the_cache() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    let mut scenario = Scenario::new("nvcc-profile", wrapper, FMA_BODY);
    // A private toolkit: its own nvcc and nvcc.profile, the real components.
    let real = fs::canonicalize("/usr/local/cuda").unwrap();
    let bin = scenario.dir.join("cuda/bin");
    fs::create_dir_all(&bin).unwrap();
    fs::copy(real.join("bin/nvcc"), bin.join("nvcc")).unwrap();
    for item in fs::read_dir(real.join("bin")).unwrap() {
        let item = item.unwrap();
        if !["nvcc", "nvcc.profile"].contains(&item.file_name().to_str().unwrap()) {
            std::os::unix::fs::symlink(item.path(), bin.join(item.file_name())).unwrap();
        }
    }
    let profile = fs::read_to_string(real.join("bin/nvcc.profile"))
        .unwrap()
        .replace("$(_HERE_)/..", &real.display().to_string());
    fs::write(bin.join("nvcc.profile"), &profile).unwrap();
    scenario.set(
        "CHOPINFER_CUDA_HOME",
        scenario.dir.join("cuda").display().to_string(),
    );
    scenario.assert_change_is_seen(|scenario| {
        fs::write(
            scenario.dir.join("cuda/bin/nvcc.profile"),
            format!("{profile}\nPTXAS_FLAGS += -O0\n"),
        )
        .unwrap()
    });
    // The fixture toolkit was keyed: its first build stored, nothing hit.
    let events = scenario.events();
    assert!(
        events.contains(" refresh ") && !events.contains(" hit "),
        "{events}"
    );
}

#[test]
fn a_date_dependent_cubin_is_not_cached() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    let mut scenario = Scenario::new(
        "date",
        wrapper,
        "out[threadIdx.x] = __DATE__[4] * 100 + __DATE__[5];",
    );
    scenario.set("SOURCE_DATE_EPOCH", "0");
    scenario.assert_change_is_seen(|scenario| scenario.set("SOURCE_DATE_EPOCH", "864000"));
    assert_eq!(scenario.events(), "");
}

#[test]
fn an_environment_compiler_setting_is_not_served_from_the_cache() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    let mut scenario = Scenario::new("ptxas-flags", wrapper, FMA_BODY);
    scenario.assert_change_is_seen(|scenario| scenario.set("PTXAS_FLAGS", "-O0"));
    let events = scenario.events();
    assert!(
        events.contains(" refresh ") && !events.contains(" hit "),
        "{events}"
    );
}

#[test]
fn a_time_macro_in_a_define_is_not_cached() {
    let Some(wrapper) = wrapper() else {
        return;
    };
    const BODY: &str = "out[threadIdx.x] = BUILD_CLOCK[6] * 10 + BUILD_CLOCK[7];";
    let later = |_: &mut Scenario| std::thread::sleep(std::time::Duration::from_millis(1100));
    let mut spec = Scenario::new("time-define", wrapper.clone(), BODY);
    spec.define = "BUILD_CLOCK=__TIME__";
    spec.assert_change_is_seen(later);
    assert_eq!(spec.events(), "");
    let mut environment = Scenario::new("time-environment", wrapper, BODY);
    environment.set("NVCC_APPEND_FLAGS", "-DBUILD_CLOCK=__TIME__");
    environment.assert_change_is_seen(later);
    assert_eq!(environment.events(), "");
}
