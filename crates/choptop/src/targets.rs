//! Explicit targets: `--target`, `--targets-file` and `CHOPINFER_TOP_TARGETS`.

use crate::http::Endpoint;

/// A server named by the operator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Target {
    pub host: String,
    pub port: u16,
}

impl Target {
    /// Parses `http://host:port`, `host:port` or `:port` (meaning
    /// `127.0.0.1:port`). An IPv6 host goes in brackets.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.starts_with("https://") {
            return Err(format!(
                "{text}: https is not supported; reach a remote server through an SSH tunnel to its loopback listener"
            ));
        }
        let authority = text.strip_prefix("http://").unwrap_or(text);
        let authority = authority.strip_suffix('/').unwrap_or(authority);
        if authority.contains('/') || authority.contains("://") {
            return Err(format!(
                "{text}: name a server as host:port; paths are not supported"
            ));
        }
        let invalid_port = |port: &str| format!("{text}: invalid port {port:?}");
        let (host, port) = if let Some(bracketed) = authority.strip_prefix('[') {
            let (host, rest) = bracketed
                .split_once(']')
                .ok_or_else(|| format!("{text}: unterminated IPv6 address"))?;
            let port = rest
                .strip_prefix(':')
                .ok_or_else(|| format!("{text}: name a port"))?;
            (host, port)
        } else {
            authority
                .rsplit_once(':')
                .ok_or_else(|| format!("{text}: name a port, as host:port"))?
        };
        let port = port
            .parse::<u16>()
            .ok()
            .filter(|port| *port != 0)
            .ok_or_else(|| invalid_port(port))?;
        let host = if host.is_empty() { "127.0.0.1" } else { host };
        if host.chars().any(|character| character.is_whitespace()) {
            return Err(format!("{text}: invalid host"));
        }
        Ok(Self {
            host: host.to_owned(),
            port,
        })
    }

    pub fn endpoint(&self) -> Endpoint {
        Endpoint::Tcp {
            host: self.host.clone(),
            port: self.port,
        }
    }

    /// `host:port`, as shown.
    pub fn display(&self) -> String {
        self.endpoint().authority()
    }
}

/// Every explicit target, in the order given and without duplicates:
/// `--target` flags, then the lines of a targets file (blank lines and `#`
/// comments skipped), then the comma-separated environment value.
pub fn collect(
    flags: &[String],
    file: Option<&str>,
    environment: Option<&str>,
) -> Result<Vec<Target>, String> {
    let file_lines = file
        .into_iter()
        .flat_map(str::lines)
        .map(|line| line.split('#').next().unwrap_or_default().trim())
        .filter(|line| !line.is_empty());
    let environment = environment
        .into_iter()
        .flat_map(|value| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let mut targets: Vec<Target> = Vec::new();
    for text in flags
        .iter()
        .map(String::as_str)
        .chain(file_lines)
        .chain(environment)
    {
        let target = Target::parse(text)?;
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    Ok(targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(host: &str, port: u16) -> Target {
        Target {
            host: host.to_owned(),
            port,
        }
    }

    #[test]
    fn accepts_the_documented_forms() {
        assert_eq!(
            Target::parse("http://10.0.0.12:8310/").unwrap(),
            target("10.0.0.12", 8310)
        );
        assert_eq!(
            Target::parse("gpu-box:8311").unwrap(),
            target("gpu-box", 8311)
        );
        assert_eq!(Target::parse(":8431").unwrap(), target("127.0.0.1", 8431));
        assert_eq!(Target::parse("[::1]:8310").unwrap(), target("::1", 8310));
        assert_eq!(Target::parse("[::1]:8310").unwrap().display(), "[::1]:8310");
    }

    #[test]
    fn refuses_https_paths_and_missing_ports() {
        for (text, message) in [
            ("https://host:8310", "SSH tunnel"),
            ("http://host:8310/v1", "paths are not supported"),
            ("host", "name a port"),
            ("host:0", "invalid port"),
            ("host:http", "invalid port"),
            ("[::1]", "name a port"),
            ("[::1:8310", "unterminated"),
        ] {
            let error = Target::parse(text).unwrap_err();
            assert!(error.contains(message), "{text}: {error}");
        }
    }

    #[test]
    fn collects_flags_file_and_environment_without_duplicates() {
        let flags = vec![":8311".to_owned(), "127.0.0.1:8311".to_owned()];
        let file = "# servers\n\n10.0.0.12:8310  # remote\n:8431\n";
        let targets = collect(&flags, Some(file), Some(" :8431 , http://[::1]:9000,")).unwrap();
        assert_eq!(
            targets,
            [
                target("127.0.0.1", 8311),
                target("10.0.0.12", 8310),
                target("127.0.0.1", 8431),
                target("::1", 9000),
            ]
        );
        assert!(collect(&[], Some("bad"), None).is_err());
        assert_eq!(collect(&[], None, None).unwrap(), []);
    }
}
