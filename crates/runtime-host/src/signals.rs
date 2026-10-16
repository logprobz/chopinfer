//! Process-scoped POSIX shutdown handlers. No allocation or locks in the handler.
// SAFETY-REVIEWED: this module confines POSIX signal registration and immediate process exit.
#![allow(unsafe_code)]

use std::{
    io,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

static STOPPING: AtomicBool = AtomicBool::new(false);

extern "C" fn shutdown_signal(_: libc::c_int) {
    if STOPPING.swap(true, Ordering::SeqCst) {
        // Queued trace lines may be lost on a second signal; do not wait for stderr.
        // SAFETY-REVIEWED: _exit is async-signal-safe and deliberately skips cleanup on a second signal.
        unsafe { libc::_exit(0) }
    }
}

pub(crate) fn install() -> io::Result<()> {
    // SAFETY-REVIEWED: a zeroed sigaction is initialized below before being passed to libc.
    let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
    action.sa_sigaction = shutdown_signal as usize;
    action.sa_flags = libc::SA_RESTART;
    // SAFETY-REVIEWED: action owns valid writable storage for its signal mask.
    unsafe { libc::sigemptyset(&mut action.sa_mask) };
    for signal in [libc::SIGTERM, libc::SIGINT] {
        // SAFETY-REVIEWED: the handler has C ABI, uses only a lock-free atomic and async-signal-safe _exit.
        if unsafe { libc::sigaction(signal, &action, std::ptr::null_mut()) } != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub(crate) async fn shutdown() {
    while !STOPPING.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// Ends the process now with `code`. Exit handlers and buffered output are
/// skipped, since a stalled process may block in either.
pub(crate) fn exit_now(code: i32) -> ! {
    // SAFETY-REVIEWED: _exit takes a plain integer and never returns.
    unsafe { libc::_exit(code) }
}

/// Writes `message` to standard error only when that cannot block: without
/// the standard library's stderr lock, which a blocked writer may hold, and
/// only when the descriptor has room for one atomic pipe write. Anything else
/// (a full pipe, a failing descriptor) drops the message.
pub(crate) fn try_write_stderr(message: &str) {
    let bytes = message.as_bytes();
    if bytes.len() > libc::PIPE_BUF {
        return;
    }
    let mut descriptor = libc::pollfd {
        fd: libc::STDERR_FILENO,
        events: libc::POLLOUT,
        revents: 0,
    };
    // SAFETY-REVIEWED: poll reads one valid pollfd for this call only, with no wait.
    let ready = unsafe { libc::poll(&mut descriptor, 1, 0) };
    if ready != 1 || descriptor.revents & libc::POLLOUT == 0 {
        return;
    }
    // SAFETY-REVIEWED: the pointer and length name the message's live bytes.
    let _ = unsafe { libc::write(libc::STDERR_FILENO, bytes.as_ptr().cast(), bytes.len()) };
}
