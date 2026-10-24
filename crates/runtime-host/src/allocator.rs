//! Process-wide allocator policy for the serving CLI, before it starts threads.

#[cfg(any(all(target_os = "linux", target_env = "gnu"), test))]
const VARIABLE: &str = "CHOPINFER_MALLOC_ARENA_MAX";

#[cfg(any(all(target_os = "linux", target_env = "gnu"), test))]
fn arena_limit(value: Option<&std::ffi::OsStr>) -> Result<Option<i32>, String> {
    let Some(value) = value else {
        return Ok(Some(8));
    };
    let limit = value
        .to_str()
        .filter(|text| !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|text| text.parse::<i32>().ok())
        .ok_or_else(|| format!("{VARIABLE} must be 0 or a positive 32-bit integer"))?;
    Ok((limit != 0).then_some(limit))
}

/// Called by the CLI before command parsing, model loading or thread startup.
/// Zero leaves glibc's startup configuration intact. Library embedders retain
/// ownership of their process-wide allocator policy.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
pub(crate) fn configure_serving_process() -> Result<(), String> {
    let limit = arena_limit(std::env::var_os(VARIABLE).as_deref())?;
    if let Some(limit) = limit {
        // SAFETY-REVIEWED: mallopt changes allocator policy without moving or
        // invalidating live allocations. The value is a positive c_int, and
        // the serving CLI calls this before it starts worker or pool threads.
        #[allow(unsafe_code)]
        let applied = unsafe { libc::mallopt(libc::M_ARENA_MAX, limit) };
        if applied == 0 {
            return Err(format!("glibc refused {VARIABLE}={limit}"));
        }
        eprintln!("host allocator: arena_max={limit}");
    } else {
        eprintln!("host allocator: arena_max=default");
    }
    Ok(())
}

#[cfg(not(all(target_os = "linux", target_env = "gnu")))]
pub(crate) fn configure_serving_process() -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn limits_include_the_serving_default_and_an_allocator_opt_out() {
        assert_eq!(arena_limit(None), Ok(Some(8)));
        for (text, expected) in [("0", None), ("1", Some(1)), ("16", Some(16))] {
            assert_eq!(arena_limit(Some(OsStr::new(text))), Ok(expected));
        }
        for text in ["", "-1", "+8", " 8", "8 ", "1.5", "default", "2147483648"] {
            assert!(arena_limit(Some(OsStr::new(text))).is_err(), "{text:?}");
        }
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    #[test]
    fn the_early_limit_bounds_thread_arenas_and_preserves_live_bytes() {
        const CHILD: &str = "CHOPINFER_TEST_ALLOCATOR_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "allocator::tests::the_early_limit_bounds_thread_arenas_and_preserves_live_bytes",
                    "--exact",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env(VARIABLE, "2")
                .env_remove("MALLOC_ARENA_MAX")
                .env_remove("GLIBC_TUNABLES")
                .env_remove("LD_PRELOAD")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
            return;
        }
        configure_serving_process().unwrap();
        let ready = std::sync::Arc::new(std::sync::Barrier::new(33));
        let release = std::sync::Arc::new(std::sync::Barrier::new(33));
        let threads: Vec<_> = (0..32)
            .map(|index| {
                let ready = ready.clone();
                let release = release.clone();
                std::thread::spawn(move || {
                    let allocations: Vec<_> = (0..64).map(|_| vec![index as u8; 16384]).collect();
                    ready.wait();
                    release.wait();
                    assert!(
                        allocations
                            .iter()
                            .flatten()
                            .all(|&byte| byte == index as u8)
                    );
                })
            })
            .collect();
        ready.wait();
        let count = arena_count();
        release.wait();
        for thread in threads {
            thread.join().unwrap();
        }
        assert!(
            count <= 2,
            "allocator retained {count} arenas with a limit of 2"
        );
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn arena_count() -> usize {
        // SAFETY-REVIEWED: open_memstream initializes the buffer and length;
        // fclose finalizes them before the slice is read. malloc_info owns no
        // borrowed pointers after returning, and the buffer is freed once.
        #[allow(unsafe_code)]
        unsafe {
            let mut bytes = std::ptr::null_mut();
            let mut length = 0;
            let stream = libc::open_memstream(&mut bytes, &mut length);
            assert!(!stream.is_null());
            let result = libc::malloc_info(0, stream);
            assert_eq!(libc::fclose(stream), 0);
            assert_eq!(result, 0);
            let xml = std::slice::from_raw_parts(bytes.cast::<u8>(), length);
            let count = std::str::from_utf8(xml)
                .unwrap()
                .matches("<heap nr=")
                .count();
            libc::free(bytes.cast());
            assert!(count > 0);
            count
        }
    }
}
