//! Codex ACP - An Agent Client Protocol implementation for Codex.
#![recursion_limit = "256"]
#![deny(clippy::print_stdout, clippy::print_stderr)]

use agent_client_protocol::ByteStreams;
use codex_core::config::{Config, ConfigOverrides};
use codex_utils_cli::CliConfigOverrides;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};
use tracing_subscriber::EnvFilter;

mod codex_agent;
mod thread;

/// Run the Codex ACP agent.
///
/// This sets up an ACP agent that communicates over stdio, bridging
/// the ACP protocol with the existing codex-rs infrastructure.
///
/// # Errors
///
/// If unable to parse the config or start the program.
pub async fn run_main(
    codex_linux_sandbox_exe: Option<PathBuf>,
    cli_config_overrides: CliConfigOverrides,
) -> std::io::Result<()> {
    // Manually parse .env from the current working directory or fallback to `/root/Desktop/codex-acp/.env`
    let env_path = if std::path::Path::new(".env").exists() {
        std::path::PathBuf::from(".env")
    } else {
        std::path::PathBuf::from("/root/Desktop/codex-acp/.env")
    };
    if let Ok(content) = std::fs::read_to_string(&env_path) {
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_matches('"').trim_matches('\'');
                if !key.to_ascii_uppercase().starts_with("CODEX_") {
                    // It is safe to call set_var() because our process is single-threaded at this point.
                    unsafe { std::env::set_var(key, val) };
                }
            }
        }
    }

    // Install a simple subscriber so `tracing` output is visible.
    // Users can control the log level with `RUST_LOG`.
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(env_filter)
        .init();

    // Parse CLI overrides and load configuration
    let cli_kv_overrides = cli_config_overrides.parse_overrides().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("error parsing -c overrides: {e}"),
        )
    })?;

    let config_overrides = ConfigOverrides {
        codex_linux_sandbox_exe: codex_linux_sandbox_exe.clone(),
        ..ConfigOverrides::default()
    };

    let config =
        Config::load_with_cli_overrides_and_harness_overrides(cli_kv_overrides, config_overrides)
            .await
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("error loading config: {e}"),
                )
            })?;
    // Apply residency requirement so the HTTP client sends the
    // x-openai-internal-codex-residency header on all requests.
    codex_login::default_client::set_default_client_residency_requirement(
        config.enforce_residency.value(),
    );

    let agent = Arc::new(codex_agent::CodexAgent::new(config, codex_linux_sandbox_exe).await?);

    let file_log_enabled = std::env::var("DCODEX_FILE_LOG")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    let stdin: std::pin::Pin<Box<dyn futures::io::AsyncRead + Send + Unpin>> = if file_log_enabled {
        let log_dir = std::env::var("DCODEX_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/root/Desktop/tmp"));

        std::fs::create_dir_all(&log_dir)?;

        let stdin_log = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_dir.join("codex_acp_stdin.log"))?;

        Box::pin(
            LoggingReader {
                inner: tokio::io::stdin(),
                file: stdin_log,
            }
            .compat(),
        )
    } else {
        Box::pin(tokio::io::stdin().compat())
    };

    let stdout: std::pin::Pin<Box<dyn futures::io::AsyncWrite + Send + Unpin>> = if file_log_enabled
    {
        let log_dir = std::env::var("DCODEX_LOG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/root/Desktop/tmp"));

        std::fs::create_dir_all(&log_dir)?;

        let stdout_log = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(log_dir.join("codex_acp_stdout.log"))?;

        Box::pin(
            LoggingWriter {
                inner: tokio::io::stdout(),
                file: stdout_log,
            }
            .compat_write(),
        )
    } else {
        Box::pin(tokio::io::stdout().compat_write())
    };

    agent
        .serve(ByteStreams::new(stdout, stdin))
        .await
        .map_err(|e| std::io::Error::other(format!("ACP error: {e}")))?;

    Ok(())
}

// Re-export the MCP server types for compatibility
pub use codex_mcp_server::{
    CodexToolCallParam, CodexToolCallReplyParam, ExecApprovalElicitRequestParams,
    ExecApprovalResponse, PatchApprovalElicitRequestParams, PatchApprovalResponse,
};

struct LoggingReader<R> {
    inner: R,
    file: std::fs::File,
}

impl<R: AsyncRead + Unpin> AsyncRead for LoggingReader<R> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let filled_before = buf.filled().len();
        let res = Pin::new(&mut self.inner).poll_read(cx, buf);
        if let Poll::Ready(Ok(())) = &res {
            let filled_after = buf.filled().len();
            if filled_after > filled_before {
                let bytes = &buf.filled()[filled_before..filled_after];
                let _unused = self.file.write_all(bytes);
                let _unused = self.file.flush();
            }
        }
        res
    }
}

struct LoggingWriter<W> {
    inner: W,
    file: std::fs::File,
}

impl<W: AsyncWrite + Unpin> AsyncWrite for LoggingWriter<W> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<Result<usize, std::io::Error>> {
        let res = Pin::new(&mut self.inner).poll_write(cx, buf);
        if let Poll::Ready(Ok(n)) = &res {
            if *n > 0 {
                let bytes = &buf[..*n];
                let _unused = self.file.write_all(bytes);
                let _unused = self.file.flush();
            }
        }
        res
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), std::io::Error>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
}
