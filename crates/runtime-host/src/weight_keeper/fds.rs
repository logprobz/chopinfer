//! Descriptor passing for the weight keeper: descriptors sent with a message
//! over a Unix stream socket (`SCM_RIGHTS`), the peer's credentials, and
//! anonymous files. Linux only; elsewhere every call is unsupported.

// SAFETY-REVIEWED: this module contains the sendmsg, recvmsg, getsockopt and memfd_create calls.
#![allow(unsafe_code)]

use std::io;
use std::os::fd::BorrowedFd;

/// Most descriptors one message carries.
pub(crate) const MAX_FDS: usize = 64;

/// The process on the other end of a connected socket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PeerV1 {
    pub(crate) pid: i32,
    pub(crate) uid: u32,
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{MAX_FDS, PeerV1};
    use std::io::{self, Write};
    use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd, RawFd};
    use std::os::unix::fs::FileExt;
    use std::os::unix::net::UnixStream;

    /// Control buffer words for [`MAX_FDS`] descriptors; `u64` keeps the
    /// buffer aligned for `cmsghdr`.
    const CONTROL_WORDS: usize = (MAX_FDS * size_of::<RawFd>() + 64).div_ceil(8);

    fn control_space(count: usize) -> usize {
        // SAFETY-REVIEWED: CMSG_SPACE is arithmetic on its argument.
        unsafe { libc::CMSG_SPACE((count * size_of::<RawFd>()) as u32) as usize }
    }

    pub(crate) fn send_with_fds(
        stream: &UnixStream,
        bytes: &[u8],
        fds: &[BorrowedFd<'_>],
    ) -> io::Result<()> {
        if bytes.is_empty() || fds.len() > MAX_FDS {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        let raw: Vec<RawFd> = fds.iter().map(AsRawFd::as_raw_fd).collect();
        let mut control = [0_u64; CONTROL_WORDS];
        let mut vector = libc::iovec {
            iov_base: bytes.as_ptr().cast_mut().cast(),
            iov_len: bytes.len(),
        };
        // SAFETY-REVIEWED: an all-zero msghdr is a valid empty message.
        let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
        message.msg_iov = &raw mut vector;
        message.msg_iovlen = 1;
        if !raw.is_empty() {
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = control_space(raw.len()) as _;
            // SAFETY-REVIEWED: the control buffer is aligned, zeroed and holds
            // CMSG_SPACE bytes for these descriptors, so the first header and
            // its data lie inside it.
            unsafe {
                let header = libc::CMSG_FIRSTHDR(&raw const message);
                (*header).cmsg_level = libc::SOL_SOCKET;
                (*header).cmsg_type = libc::SCM_RIGHTS;
                (*header).cmsg_len = libc::CMSG_LEN((raw.len() * size_of::<RawFd>()) as u32) as _;
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    libc::CMSG_DATA(header).cast::<RawFd>(),
                    raw.len(),
                );
            }
        }
        let sent = loop {
            // SAFETY-REVIEWED: the message names live buffers for the call.
            let sent = unsafe {
                libc::sendmsg(stream.as_raw_fd(), &raw const message, libc::MSG_NOSIGNAL)
            };
            if sent >= 0 {
                break sent as usize;
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        };
        // The descriptors went with the first byte; any rest follows plainly.
        (&*stream).write_all(&bytes[sent..])
    }

    pub(crate) fn receive_line(
        stream: &UnixStream,
        limit: usize,
    ) -> io::Result<(Vec<u8>, Vec<OwnedFd>)> {
        let mut line = Vec::new();
        let mut fds = Vec::new();
        loop {
            let mut chunk = [0_u8; 4096];
            let mut control = [0_u64; CONTROL_WORDS];
            let mut vector = libc::iovec {
                iov_base: chunk.as_mut_ptr().cast(),
                iov_len: chunk.len(),
            };
            // SAFETY-REVIEWED: an all-zero msghdr is a valid empty message.
            let mut message: libc::msghdr = unsafe { std::mem::zeroed() };
            message.msg_iov = &raw mut vector;
            message.msg_iovlen = 1;
            message.msg_control = control.as_mut_ptr().cast();
            message.msg_controllen = size_of_val(&control) as _;
            // SAFETY-REVIEWED: the message names live buffers for the call.
            let received = unsafe {
                libc::recvmsg(stream.as_raw_fd(), &raw mut message, libc::MSG_CMSG_CLOEXEC)
            };
            if received < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            // Every descriptor received is owned at once, so an error below
            // closes it rather than leaking it.
            // SAFETY-REVIEWED: recvmsg filled msg_controllen bytes of the
            // aligned control buffer; the CMSG macros walk only those.
            unsafe {
                let mut header = libc::CMSG_FIRSTHDR(&raw const message);
                while !header.is_null() {
                    if (*header).cmsg_level == libc::SOL_SOCKET
                        && (*header).cmsg_type == libc::SCM_RIGHTS
                    {
                        let data = (*header).cmsg_len as usize - libc::CMSG_LEN(0) as usize;
                        let pointer = libc::CMSG_DATA(header).cast::<RawFd>();
                        for index in 0..data / size_of::<RawFd>() {
                            fds.push(OwnedFd::from_raw_fd(pointer.add(index).read_unaligned()));
                        }
                    }
                    header = libc::CMSG_NXTHDR(&raw const message, header);
                }
            }
            if message.msg_flags & libc::MSG_CTRUNC != 0 {
                return Err(io::Error::other("descriptors were truncated"));
            }
            if received == 0 {
                return Err(io::Error::from(io::ErrorKind::UnexpectedEof));
            }
            line.extend_from_slice(&chunk[..received as usize]);
            if let Some(end) = line.iter().position(|&byte| byte == b'\n') {
                if end + 1 != line.len() {
                    return Err(io::Error::other("bytes followed the message"));
                }
                line.truncate(end);
                return Ok((line, fds));
            }
            if line.len() > limit {
                return Err(io::Error::other("message too long"));
            }
        }
    }

    pub(crate) fn peer(stream: &UnixStream) -> io::Result<PeerV1> {
        let mut credentials = libc::ucred {
            pid: 0,
            uid: 0,
            gid: 0,
        };
        let mut length = size_of::<libc::ucred>() as libc::socklen_t;
        // SAFETY-REVIEWED: getsockopt writes at most `length` bytes into the
        // ucred it is given.
        let status = unsafe {
            libc::getsockopt(
                stream.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_PEERCRED,
                (&raw mut credentials).cast(),
                &raw mut length,
            )
        };
        if status != 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(PeerV1 {
            pid: credentials.pid,
            uid: credentials.uid,
        })
    }

    pub(crate) fn effective_uid() -> u32 {
        // SAFETY-REVIEWED: geteuid cannot fail and touches no memory.
        unsafe { libc::geteuid() }
    }

    pub(crate) fn anonymous_file(bytes: &[u8], length: u64) -> io::Result<OwnedFd> {
        if (bytes.len() as u64) > length {
            return Err(io::Error::from(io::ErrorKind::InvalidInput));
        }
        // SAFETY-REVIEWED: the name is a NUL-terminated literal.
        let fd = unsafe { libc::memfd_create(c"chopinfer-lent".as_ptr(), libc::MFD_CLOEXEC) };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY-REVIEWED: memfd_create returned a new descriptor nothing
        // else owns.
        let file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(fd) });
        file.set_len(length)?;
        file.write_all_at(bytes, 0)?;
        Ok(file.into())
    }
}

#[cfg(target_os = "linux")]
pub(crate) use linux::{anonymous_file, effective_uid, peer, receive_line, send_with_fds};

#[cfg(not(target_os = "linux"))]
fn unsupported<T>() -> io::Result<T> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Sends `bytes` with `fds` attached to its first byte.
#[cfg(not(target_os = "linux"))]
pub(crate) fn send_with_fds(
    _: &std::os::unix::net::UnixStream,
    _: &[u8],
    _: &[BorrowedFd<'_>],
) -> io::Result<()> {
    unsupported()
}

/// Reads one newline-terminated message of at most `limit` bytes, and the
/// descriptors sent with it, without the newline. Bytes after the newline
/// are an error: each side sends one message and waits for the answer.
#[cfg(not(target_os = "linux"))]
pub(crate) fn receive_line(
    _: &std::os::unix::net::UnixStream,
    _: usize,
) -> io::Result<(Vec<u8>, Vec<std::os::fd::OwnedFd>)> {
    unsupported()
}

/// The credentials of the process on the other end of `stream`.
#[cfg(not(target_os = "linux"))]
pub(crate) fn peer(_: &std::os::unix::net::UnixStream) -> io::Result<PeerV1> {
    unsupported()
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn effective_uid() -> u32 {
    u32::MAX
}

/// A new anonymous file of `length` bytes that begins with `bytes`.
#[cfg(not(target_os = "linux"))]
pub(crate) fn anonymous_file(_: &[u8], _: u64) -> io::Result<std::os::fd::OwnedFd> {
    unsupported()
}

/// The first `byte_length` bytes of a file of exactly `length` bytes.
pub(crate) fn read_anonymous_file(
    fd: BorrowedFd<'_>,
    length: u64,
    byte_length: u64,
) -> io::Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::from(fd.try_clone_to_owned()?);
    if file.metadata()?.len() != length || byte_length > length {
        return Err(io::Error::other("lent file length differs"));
    }
    let mut bytes = vec![0; usize::try_from(byte_length).map_err(io::Error::other)?];
    file.read_exact_at(&mut bytes, 0)?;
    Ok(bytes)
}
