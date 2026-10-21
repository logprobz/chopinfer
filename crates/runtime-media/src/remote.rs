//! Opt-in HTTP media reads. Call only from the bounded preparation blocking pool.
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use hyper::Uri;

use crate::{MediaError, MediaLimits, MediaResult, MediaSource, Probe, probe};

const POLL: Duration = Duration::from_millis(10);
const HEADER_LIMIT: usize = 16 * 1024;
// URL request bodies are small. Bound network readers independently of intake bytes.
static FETCH_SLOTS: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(8);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoteMediaPolicy {
    /// Exact ASCII host names, without ports or wildcards. Empty disables fetching.
    pub allowed_domains: Vec<String>,
    pub allow_private_addresses: bool,
    pub connect_timeout: Duration,
    /// Covers DNS, all redirects, and the complete response body.
    pub total_timeout: Duration,
    /// Wall-clock budget shared by all remote items in one preparation.
    pub request_timeout: Duration,
    pub maximum_redirects: usize,
}

impl Default for RemoteMediaPolicy {
    fn default() -> Self {
        Self {
            allowed_domains: Vec::new(),
            allow_private_addresses: false,
            connect_timeout: Duration::from_secs(3),
            total_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            maximum_redirects: 3,
        }
    }
}

impl RemoteMediaPolicy {
    pub fn validate(&self) -> MediaResult<()> {
        if self.connect_timeout.is_zero()
            || self.request_timeout.is_zero()
            || self.request_timeout > Duration::from_secs(300)
            || self.total_timeout.is_zero()
            || self.total_timeout > Duration::from_secs(300)
            || self.connect_timeout > Duration::from_secs(300)
            || self.maximum_redirects > 10
        {
            return Err(MediaError::new(
                "invalid remote media timeout or redirect limit",
            ));
        }
        for domain in &self.allowed_domains {
            validate_host(domain)?;
        }
        Ok(())
    }

    pub fn validate_url(&self, url: &str) -> MediaResult<()> {
        self.destination(url).map(|_| ())
    }

    fn destination(&self, url: &str) -> MediaResult<(Uri, String, u16)> {
        // Reject syntax that curl and an RFC URI parser might interpret differently.
        if !url.is_ascii()
            || url.bytes().any(|b| b <= b' ' || b == 127)
            || url.contains(['\\', '#'])
        {
            return Err(MediaError::new("invalid remote media URL"));
        }
        let uri: Uri = url
            .parse()
            .map_err(|_| MediaError::new("invalid remote media URL"))?;
        let scheme = uri.scheme_str().unwrap_or_default();
        if !matches!(scheme, "http" | "https") {
            return Err(MediaError::new("remote media requires http or https"));
        }
        let authority = uri
            .authority()
            .ok_or_else(|| MediaError::new("media URL lacks host"))?;
        if authority.as_str().contains(['@', '%']) {
            return Err(MediaError::new(
                "media URL credentials and host escapes are refused",
            ));
        }
        let host = uri
            .host()
            .unwrap_or_default()
            .trim_start_matches('[')
            .trim_end_matches(']')
            .to_ascii_lowercase();
        validate_host(&host)?;
        if !self
            .allowed_domains
            .iter()
            .any(|allowed| allowed.eq_ignore_ascii_case(&host))
        {
            return Err(MediaError::new(
                "remote media host is not in --allowed-media-domains",
            ));
        }
        let port = uri
            .port_u16()
            .unwrap_or(if scheme == "https" { 443 } else { 80 });
        // A malformed/out-of-range port must not silently become the default port.
        let expected = if host.contains(':') {
            format!("[{host}]")
        } else {
            host.clone()
        };
        if !authority.as_str().eq_ignore_ascii_case(&expected)
            && !authority
                .as_str()
                .eq_ignore_ascii_case(&format!("{expected}:{port}"))
        {
            return Err(MediaError::new("invalid media URL port"));
        }
        Ok((uri, host, port))
    }
}

fn validate_host(host: &str) -> MediaResult<()> {
    if host.parse::<IpAddr>().is_ok() {
        return Ok(());
    }
    if host.is_empty()
        || host.len() > 253
        || host.ends_with('.')
        || host.split('.').any(|label| {
            label.is_empty()
                || label.len() > 63
                || label.starts_with('-')
                || label.ends_with('-')
                || !label
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'-')
        })
        || host.split('.').all(|label| {
            label.bytes().all(|b| b.is_ascii_digit())
                || label.to_ascii_lowercase().starts_with("0x")
        })
    {
        return Err(MediaError::new(
            "invalid media host (use an exact ASCII domain or canonical IP)",
        ));
    }
    Ok(())
}

fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(matches!(a, 0 | 10 | 127)
        || a >= 224
        || (a == 100 && (64..=127).contains(&b))
        || (a == 169 && b == 254)
        || (a == 172 && (16..=31).contains(&b))
        || (a == 192 && (b == 168 || b == 0 || (b == 88 && c == 99)))
        || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
        || (a == 203 && b == 0 && c == 113))
}

fn public_address(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => {
            let s = ip.segments();
            // Global unicast only, excluding transition, documentation and special-use space.
            (s[0] & 0xe000) == 0x2000
                && s[0] != 0x2002
                && !(s[0] == 0x2001 && (s[1] < 0x0200 || s[1] == 0x0db8))
                && !(s[0] == 0x3fff && s[1] < 0x1000)
        }
    }
}

fn check(deadline: Instant, cancelled: &impl Fn() -> bool) -> MediaResult<()> {
    if cancelled() {
        return Err(MediaError::new("remote media fetch cancelled"));
    }
    if Instant::now() >= deadline {
        return Err(MediaError::new("remote media fetch timeout"));
    }
    Ok(())
}

/// Uses the current Tokio handle for cancellable DNS, but performs process I/O
/// on the caller's blocking preparation worker. No shell or new TLS dependency.
pub fn fetch(
    url: &str,
    policy: &RemoteMediaPolicy,
    limits: MediaLimits,
    cancelled: impl Fn() -> bool,
) -> MediaResult<MediaSource> {
    policy.validate()?;
    fetch_with_deadline(
        url,
        policy,
        limits,
        Instant::now() + policy.total_timeout,
        cancelled,
    )
}

/// Fetches one item within the deadline shared by all media in a request.
pub fn fetch_with_deadline(
    url: &str,
    policy: &RemoteMediaPolicy,
    limits: MediaLimits,
    request_deadline: Instant,
    cancelled: impl Fn() -> bool,
) -> MediaResult<MediaSource> {
    policy.validate()?;
    let deadline = request_deadline.min(Instant::now() + policy.total_timeout);
    let _slot = loop {
        check(deadline, &cancelled)?;
        if let Ok(slot) = FETCH_SLOTS.try_acquire() {
            break slot;
        }
        std::thread::sleep(POLL.min(deadline.saturating_duration_since(Instant::now())));
    };
    let mut url = url.to_owned();
    let mut remaining = limits.maximum_bytes;
    for redirects in 0..=policy.maximum_redirects {
        check(deadline, &cancelled)?;
        let (uri, host, port) = policy.destination(&url)?;
        let addresses = if let Ok(ip) = host.parse::<IpAddr>() {
            vec![ip]
        } else {
            let handle = tokio::runtime::Handle::try_current()
                .map_err(|_| MediaError::new("remote media DNS requires the serving runtime"))?;
            handle.block_on(async {
                let lookup = tokio::net::lookup_host((host.as_str(), port));
                tokio::pin!(lookup);
                loop {
                    check(deadline, &cancelled)?;
                    tokio::select! {
                        result = &mut lookup => return result.map(|items| items.map(|a| a.ip()).collect::<Vec<_>>())
                            .map_err(|_| MediaError::new("media host DNS lookup failed")),
                        _ = tokio::time::sleep(POLL) => {}
                    }
                }
            })?
        };
        if addresses.is_empty()
            || (!policy.allow_private_addresses && addresses.iter().any(|ip| !public_address(*ip)))
        {
            return Err(MediaError::new(
                "remote media private or special-use address is refused",
            ));
        }
        check(deadline, &cancelled)?;
        let response = transfer(
            &url,
            &host,
            port,
            addresses[0],
            policy,
            remaining,
            deadline,
            &cancelled,
        )?;
        remaining -= response.body.len();
        if matches!(response.status, 301 | 302 | 303 | 307 | 308) {
            if redirects == policy.maximum_redirects {
                return Err(MediaError::new("remote media redirect limit exceeded"));
            }
            let location = response
                .location
                .ok_or_else(|| MediaError::new("media redirect lacks Location"))?;
            url = redirect_url(&uri, &location)?;
            continue;
        }
        if !(200..300).contains(&response.status) {
            return Err(MediaError::new(
                "remote media HTTP status is not successful",
            ));
        }
        let bytes = response.body;
        let mime = if bytes.get(4..8) == Some(b"ftyp".as_slice()) {
            "video/mp4"
        } else {
            match probe(&bytes, limits)? {
                Probe::Image { format, .. } => format.mime(),
                Probe::Wav { .. } => "audio/wav",
            }
        };
        return Ok(MediaSource {
            mime: mime.into(),
            bytes,
        });
    }
    unreachable!()
}

fn redirect_url(base: &Uri, location: &str) -> MediaResult<String> {
    if location.is_empty() {
        return Err(MediaError::new("invalid media redirect URL"));
    }
    let location = location.split('#').next().unwrap();
    if location.is_empty() {
        return Ok(base.to_string());
    }
    if base.scheme_str() == Some("https") && location.starts_with("http://") {
        return Err(MediaError::new("remote media HTTPS downgrade is refused"));
    }
    if location.starts_with("http://") || location.starts_with("https://") {
        return Ok(location.into());
    }
    if location.starts_with("//") {
        return Ok(format!("{}:{location}", base.scheme_str().unwrap()));
    }
    if !location.starts_with(['/', '?']) && location.split(['/', '?']).next().unwrap().contains(':')
    {
        return Err(MediaError::new("invalid media redirect URL"));
    }
    let origin = format!(
        "{}://{}",
        base.scheme_str().unwrap(),
        base.authority().unwrap()
    );
    if location.starts_with('/') {
        return Ok(format!("{origin}{location}"));
    }
    if location.starts_with('?') {
        return Ok(format!("{origin}{}{location}", base.path()));
    }
    let directory = base.path().rsplit_once('/').map_or("", |(dir, _)| dir);
    Ok(format!("{origin}{directory}/{location}"))
}

struct Response {
    status: u16,
    location: Option<String>,
    body: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn transfer(
    url: &str,
    host: &str,
    port: u16,
    ip: IpAddr,
    policy: &RemoteMediaPolicy,
    maximum: usize,
    deadline: Instant,
    cancelled: &impl Fn() -> bool,
) -> MediaResult<Response> {
    let address = match ip {
        IpAddr::V4(ip) => ip.to_string(),
        IpAddr::V6(ip) => format!("[{ip}]"),
    };
    let host = if host.contains(':') {
        format!("[{host}]")
    } else {
        host.into()
    };
    let mut child = Command::new("/usr/bin/curl")
        .env_clear()
        .args([
            "-q",
            "--silent",
            "--globoff",
            "--proto",
            "=http,https",
            "--proto-redir",
            "=http,https",
            "--proxy",
            "",
            "--noproxy",
            "*",
            "--max-redirs",
            "0",
            "--include",
            "--http1.1",
        ])
        .args([
            "--connect-timeout",
            &policy.connect_timeout.as_secs_f64().to_string(),
            "--max-time",
            &deadline
                .saturating_duration_since(Instant::now())
                .as_secs_f64()
                .to_string(),
            "--resolve",
            &format!("{host}:{port}:{address}"),
            "--url",
            url,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|_| MediaError::new("cannot start /usr/bin/curl for remote media"))?;
    let mut stdout = child.stdout.take().unwrap();
    let (send, receive) = mpsc::sync_channel(1);
    let reader = std::thread::Builder::new()
        .name("media-curl-reader".into())
        .spawn(move || {
            loop {
                let mut chunk = vec![0; 8192];
                match stdout.read(&mut chunk) {
                    Ok(0) => break,
                    Ok(count) => {
                        chunk.truncate(count);
                        if send.send(Ok(chunk)).is_err() {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = send.send(Err(error));
                        break;
                    }
                }
            }
        });
    let reader = match reader {
        Ok(reader) => reader,
        Err(_) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(MediaError::new("cannot start media reader"));
        }
    };
    let result = (|| {
        let mut header = Vec::new();
        let mut parsed = None;
        let mut body = Vec::new();
        loop {
            check(deadline, cancelled)?;
            match receive.recv_timeout(POLL) {
                Ok(Ok(chunk)) => {
                    let payload = if parsed.is_none() {
                        header.extend_from_slice(&chunk);
                        if let Some(end) = header.windows(4).position(|w| w == b"\r\n\r\n") {
                            if end + 4 > HEADER_LIMIT {
                                return Err(MediaError::new("media HTTP header limit exceeded"));
                            }
                            parsed = Some(parse_header(&header[..end])?);
                            &header[end + 4..]
                        } else {
                            if header.len() > HEADER_LIMIT {
                                return Err(MediaError::new("media HTTP header limit exceeded"));
                            }
                            continue;
                        }
                    } else {
                        &chunk
                    };
                    if payload.len() > maximum.saturating_sub(body.len()) {
                        return Err(MediaError::new("compressed media byte limit exceeded"));
                    }
                    body.extend_from_slice(payload);
                }
                Ok(Err(_)) => return Err(MediaError::new("media response read failed")),
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        check(deadline, cancelled)?;
        // EOF can precede process exit. Keep the same cancellation/deadline bound.
        loop {
            check(deadline, cancelled)?;
            if let Some(status) = child
                .try_wait()
                .map_err(|_| MediaError::new("media fetch wait failed"))?
            {
                if !status.success() {
                    return Err(MediaError::new("media fetch failed or timed out"));
                }
                break;
            }
            std::thread::sleep(POLL);
        }
        let (status, location) =
            parsed.ok_or_else(|| MediaError::new("missing media HTTP header"))?;
        Ok(Response {
            status,
            location,
            body,
        })
    })();
    // Close the channel before joining, so a producer cannot remain blocked on send.
    drop(receive);
    let _ = child.kill();
    let _ = child.wait();
    let _ = reader.join();
    result
}

fn parse_header(bytes: &[u8]) -> MediaResult<(u16, Option<String>)> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| MediaError::new("invalid media HTTP header"))?;
    let mut lines = text.split("\r\n");
    let status = lines
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|status| status.parse::<u16>().ok())
        .ok_or_else(|| MediaError::new("invalid media HTTP status"))?;
    if status < 200 {
        return Err(MediaError::new("interim media HTTP responses are refused"));
    }
    let mut location = None;
    for line in lines {
        if let Some((name, value)) = line.split_once(':')
            && name.eq_ignore_ascii_case("location")
        {
            if location.is_some() {
                return Err(MediaError::new("duplicate media redirect Location"));
            }
            location = Some(value.trim().to_owned());
        }
    }
    Ok((status, location))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirects_preserve_https_and_accept_path_colons_and_fragments() {
        let base: Uri = "https://example.com/dir/a".parse().unwrap();
        assert!(redirect_url(&base, "http://example.com/plain").is_err());
        assert_eq!(
            redirect_url(&base, "/a:b#section").unwrap(),
            "https://example.com/a:b"
        );
        assert_eq!(
            redirect_url(&base, "path/a:b#section").unwrap(),
            "https://example.com/dir/path/a:b"
        );
        assert_eq!(redirect_url(&base, "#section").unwrap(), base.to_string());
        assert_eq!(
            redirect_url(&base, "//example.com/a#section").unwrap(),
            "https://example.com/a"
        );
        assert!(redirect_url(&base, "file:/etc/passwd").is_err());
    }

    #[test]
    fn excludes_private_and_special_use_addresses_in_both_families() {
        for address in [
            "0.0.0.0",
            "10.1.2.3",
            "100.64.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "172.31.0.1",
            "192.168.1.1",
            "192.0.2.1",
            "198.19.0.1",
            "198.51.100.1",
            "203.0.113.1",
            "224.1.1.1",
            "255.255.255.255",
            "::",
            "::1",
            "::ffff:127.0.0.1",
            "64:ff9b::a00:1",
            "fc00::1",
            "fe80::1",
            "ff02::1",
            "2001:db8::1",
            "2002:7f00:1::",
            "3fff::1",
        ] {
            assert!(!public_address(address.parse().unwrap()), "{address}");
        }
        for address in [
            "8.8.8.8",
            "1.1.1.1",
            "2606:4700:4700::1111",
            "2001:4860:4860::8888",
        ] {
            assert!(public_address(address.parse().unwrap()), "{address}");
        }
    }

    #[test]
    fn waits_for_fetch_capacity_with_deadline_and_cancellation() {
        let all = FETCH_SLOTS.try_acquire_many(8).unwrap();
        let policy = RemoteMediaPolicy {
            total_timeout: Duration::from_millis(60),
            ..Default::default()
        };
        let limits = MediaLimits {
            maximum_bytes: 1,
            maximum_pixels: 1,
            maximum_audio_frames: 1,
            maximum_frames: 1,
            maximum_tokens: 1,
            maximum_aspect_ratio: 1.0,
        };
        let started = Instant::now();
        let error = fetch("http://example.com/a", &policy, limits, || false).unwrap_err();
        assert!(error.0.contains("timeout"), "{error}");
        assert!(started.elapsed() >= policy.total_timeout);
        assert!(started.elapsed() < Duration::from_secs(1));
        let error = fetch("http://example.com/a", &policy, limits, || true).unwrap_err();
        assert!(error.0.contains("cancelled"), "{error}");
        std::thread::scope(|scope| {
            scope.spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                drop(all);
            });
            let error = fetch("http://example.com/a", &policy, limits, || false).unwrap_err();
            assert!(error.0.contains("allowed-media-domains"), "{error}");
        });
        assert_eq!(FETCH_SLOTS.available_permits(), 8);
    }

    #[test]
    fn rejects_ambiguous_numeric_hosts() {
        for host in [
            "127.1",
            "2130706433",
            "0177.0.0.1",
            "0x7f000001",
            "0X7F000001",
            "127.0.0.1.",
        ] {
            assert!(validate_host(host).is_err(), "{host}");
        }
    }
}
