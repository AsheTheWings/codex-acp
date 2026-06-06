use crate::thread::{RevertPreviewResponse, RevertStep, Thread};
use acp::schema::{
    AgentAuthCapabilities, AgentCapabilities, AuthEnvVar, AuthMethod, AuthMethodAgent,
    AuthMethodEnvVar, AuthMethodId, AuthenticateRequest, AuthenticateResponse, CancelNotification,
    ClientCapabilities, CloseSessionRequest, CloseSessionResponse, Implementation,
    InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
    LoadSessionRequest, LoadSessionResponse, LogoutCapabilities, LogoutRequest, LogoutResponse,
    McpCapabilities, McpServer, McpServerHttp, McpServerStdio, NewSessionRequest,
    NewSessionResponse, PromptCapabilities, PromptRequest, PromptResponse, ProtocolVersion,
    SessionAdditionalDirectoriesCapabilities, SessionCapabilities, SessionCloseCapabilities,
    SessionId, SessionInfo, SessionListCapabilities, SetSessionConfigOptionRequest,
    SetSessionConfigOptionResponse, SetSessionModeRequest, SetSessionModeResponse,
    SetSessionModelRequest, SetSessionModelResponse,
};
use acp::{Agent, Client, ConnectTo, ConnectionTo, Error};
use agent_client_protocol as acp;
use agent_client_protocol::{JsonRpcRequest, JsonRpcResponse};
use codex_config::{McpServerConfig, McpServerTransportConfig};
use codex_core::{
    NewThread, RolloutRecorder, SortDirection, StateDbHandle, ThreadManager, ThreadSortKey,
    config::Config, find_thread_path_by_id_str, init_state_db, parse_cursor,
    resolve_installation_id, thread_store_from_config,
};
use codex_exec_server::{EnvironmentManager, ExecServerRuntimePaths};
use codex_extension_api::empty_extension_registry;
use codex_login::{
    CODEX_API_KEY_ENV_VAR, OPENAI_API_KEY_ENV_VAR,
    auth::{AuthManager, CodexAuth, read_codex_api_key_from_env, read_openai_api_key_from_env},
};
use codex_protocol::{
    ThreadId,
    protocol::{InitialHistory, Op, SessionSource},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use tracing::{debug, info};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/revert/listSteps", response = RevertListStepsResponse)]
#[serde(rename_all = "camelCase")]
pub struct RevertListStepsRequest {
    #[serde(rename = "sessionId")]
    pub session_id: SessionId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct RevertListStepsResponse {
    pub steps: Vec<RevertStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/revert/preview", response = RevertPreviewResponse)]
#[serde(rename_all = "camelCase")]
pub struct RevertPreviewRequest {
    #[serde(rename = "sessionId")]
    pub session_id: SessionId,
    #[serde(rename = "targetNodeId")]
    pub target_node_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/revert/execute", response = RevertExecuteResponse)]
#[serde(rename_all = "camelCase")]
pub struct RevertExecuteRequest {
    #[serde(rename = "sessionId")]
    pub session_id: SessionId,
    #[serde(rename = "targetNodeId")]
    pub target_node_id: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct RevertExecuteResponse {
    #[serde(rename = "forkedSessionId")]
    pub forked_session_id: SessionId,
    pub outcomes: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/mcp/listServers", response = McpListServersResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpListServersRequest {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpListServersResponse {
    pub servers: Vec<McpServerStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpServerStatus {
    pub server_id: String,
    pub disabled: bool,
    pub disabled_tools: Vec<String>,
    pub connection_status: String,
    pub source_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/mcp/connectServer", response = McpConnectServerResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpConnectServerRequest {
    pub server_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpConnectServerResponse {
    pub connection_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/mcp/listTools", response = McpListToolsResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpListToolsRequest {
    pub server_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpListToolsResponse {
    pub server_id: String,
    pub tools: Vec<McpToolDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolDetail {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcRequest)]
#[request(method = "_cognition.ai/mcp/toggleTool", response = McpToggleToolResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpToggleToolRequest {
    pub server_id: String,
    pub tool_name: String,
    pub source_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonRpcResponse)]
#[serde(rename_all = "camelCase")]
pub struct McpToggleToolResponse {}

/// The Codex implementation of the ACP Agent.
///
/// This bridges the ACP protocol with the existing codex-rs infrastructure,
/// allowing codex to be used as an ACP agent.
pub struct CodexAgent {
    /// Handle to the current authentication
    auth_manager: Arc<AuthManager>,
    /// Capabilities of the connected client
    client_capabilities: Arc<Mutex<ClientCapabilities>>,
    /// The underlying codex configuration
    config: Config,
    /// Thread manager for handling sessions
    thread_manager: ThreadManager,
    /// SQLite-backed Codex state index, when initialization succeeds
    state_db: Option<StateDbHandle>,
    /// Active sessions mapped by `SessionId`
    sessions: Arc<Mutex<HashMap<SessionId, Arc<Thread>>>>,
    /// Session working directories for filesystem sandboxing
    session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>>,
}

const SESSION_LIST_PAGE_SIZE: usize = 25;
const SESSION_TITLE_MAX_GRAPHEMES: usize = 120;

impl CodexAgent {
    /// Create a new `CodexAgent` with the given configuration
    pub async fn new(
        config: Config,
        codex_linux_sandbox_exe: Option<PathBuf>,
    ) -> std::io::Result<Self> {
        let auth_manager = AuthManager::shared(
            config.codex_home.to_path_buf(),
            false,
            config.cli_auth_credentials_store_mode,
            Some(config.chatgpt_base_url.clone()),
        )
        .await;

        let client_capabilities: Arc<Mutex<ClientCapabilities>> = Arc::default();
        let session_roots: Arc<Mutex<HashMap<SessionId, PathBuf>>> = Arc::default();
        let state_db = init_state_db(&config).await;
        let local_runtime_paths =
            ExecServerRuntimePaths::new(std::env::current_exe()?, codex_linux_sandbox_exe)?;
        let environment_manager = Arc::new(
            EnvironmentManager::from_codex_home(&config.codex_home, Some(local_runtime_paths))
                .await
                .map_err(std::io::Error::other)?,
        );
        let thread_store = thread_store_from_config(&config, state_db.clone());
        let installation_id = resolve_installation_id(&config.codex_home).await?;
        let thread_manager = ThreadManager::new(
            &config,
            auth_manager.clone(),
            SessionSource::Unknown,
            environment_manager,
            empty_extension_registry(),
            None,
            thread_store,
            state_db.clone(),
            installation_id,
            None,
        );
        Ok(Self {
            auth_manager,
            client_capabilities,
            config,
            thread_manager,
            state_db,
            sessions: Arc::default(),
            session_roots,
        })
    }

    /// Build and run the ACP agent, serving requests over the given transport.
    pub async fn serve(
        self: Arc<Self>,
        transport: impl ConnectTo<Agent> + 'static,
    ) -> acp::Result<()> {
        let agent = self;
        Agent
            .builder()
            .name("codex-acp")
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: InitializeRequest, responder, _cx| {
                        responder.respond_with_result(agent.initialize(request).await)
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: AuthenticateRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.authenticate(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: LogoutRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.logout(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: NewSessionRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let session_cx = cx.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.new_session(request, session_cx).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: LoadSessionRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        let session_cx = cx.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.load_session(request, session_cx).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: ListSessionsRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.list_sessions(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: CloseSessionRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.close_session(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: PromptRequest, responder, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.prompt(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_notification(
                {
                    let agent = agent.clone();
                    async move |notification: CancelNotification, cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            if let Err(e) = agent.cancel(notification).await {
                                tracing::error!("Error handling cancel: {:?}", e);
                            }
                            Ok(())
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_notification!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionModeRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.set_session_mode(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionModelRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.set_session_model(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: SetSessionConfigOptionRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder
                                .respond_with_result(agent.set_session_config_option(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: RevertListStepsRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.revert_list_steps(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: RevertPreviewRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.revert_preview(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: RevertExecuteRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.revert_execute(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: McpListServersRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.mcp_list_servers(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: McpConnectServerRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.mcp_connect_server(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: McpListToolsRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.mcp_list_tools(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .on_receive_request(
                {
                    let agent = agent.clone();
                    async move |request: McpToggleToolRequest,
                                responder,
                                cx: ConnectionTo<Client>| {
                        let agent = agent.clone();
                        cx.spawn(async move {
                            responder.respond_with_result(agent.mcp_toggle_tool(request).await)
                        })?;
                        Ok(())
                    }
                },
                acp::on_receive_request!(),
            )
            .connect_to(transport)
            .await
    }

    fn session_id_from_thread_id(thread_id: ThreadId) -> SessionId {
        SessionId::new(thread_id.to_string())
    }

    fn get_thread(&self, session_id: &SessionId) -> Result<Arc<Thread>, Error> {
        Ok(self
            .sessions
            .lock()
            .unwrap()
            .get(session_id)
            .ok_or_else(|| Error::resource_not_found(None))?
            .clone())
    }

    fn get_any_active_thread(&self) -> Result<Arc<Thread>, Error> {
        let sessions = self.sessions.lock().unwrap();
        if let Some(thread) = sessions.values().next() {
            Ok(Arc::clone(thread))
        } else {
            Err(Error::resource_not_found(Some(
                "No active sessions".to_string(),
            )))
        }
    }

    async fn check_auth(&self) -> Result<(), Error> {
        if self.config.model_provider_id == "openai"
            && self.auth_manager.auth().await.is_none()
            // Check if anything changed on disk since the last reload
            && !self.auth_manager.reload().await
        {
            return Err(Error::auth_required());
        }
        Ok(())
    }

    /// Build a session config from base config, working directory, and MCP servers.
    /// This is shared between `new_session` and `load_session`.
    fn build_session_config(
        &self,
        cwd: &Path,
        mcp_servers: Vec<McpServer>,
    ) -> Result<Config, Error> {
        let mut config = self.config.clone();
        config.cwd = cwd.try_into().map_err(Error::into_internal_error)?;
        let cwd = config.cwd.clone();

        // Propagate any client-provided MCP servers that codex-rs supports.
        let mut new_mcp_servers = config.mcp_servers.get().clone();
        for mcp_server in mcp_servers {
            match mcp_server {
                // Not supported in codex
                McpServer::Sse(..) => {}
                McpServer::Http(McpServerHttp {
                    name, url, headers, ..
                }) => {
                    // Codex does not allow whitespace in MCP server names; replace with underscores.
                    let name = name.replace(|c: char| c.is_whitespace(), "_");
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::StreamableHttp {
                                url,
                                bearer_token_env_var: None,
                                http_headers: if headers.is_empty() {
                                    None
                                } else {
                                    Some(headers.into_iter().map(|h| (h.name, h.value)).collect())
                                },
                                env_http_headers: None,
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth: None,
                            oauth_resource: None,
                            tools: Default::default(),
                            environment_id: codex_config::DEFAULT_MCP_SERVER_ENVIRONMENT_ID
                                .to_string(),
                            supports_parallel_tool_calls: false,
                            default_tools_approval_mode: None,
                        },
                    );
                }
                McpServer::Stdio(McpServerStdio {
                    name,
                    command,
                    args,
                    env,
                    ..
                }) => {
                    // Codex does not allow whitespace in MCP server names; replace with underscores.
                    let name = name.replace(|c: char| c.is_whitespace(), "_");
                    new_mcp_servers.insert(
                        name,
                        McpServerConfig {
                            transport: McpServerTransportConfig::Stdio {
                                command: command.display().to_string(),
                                args,
                                env: if env.is_empty() {
                                    None
                                } else {
                                    Some(env.into_iter().map(|env| (env.name, env.value)).collect())
                                },
                                env_vars: vec![],
                                cwd: Some(cwd.to_path_buf()),
                            },
                            required: false,
                            enabled: true,
                            startup_timeout_sec: None,
                            tool_timeout_sec: None,
                            disabled_tools: None,
                            enabled_tools: None,
                            disabled_reason: None,
                            scopes: None,
                            oauth: None,
                            oauth_resource: None,
                            tools: Default::default(),
                            environment_id: codex_config::DEFAULT_MCP_SERVER_ENVIRONMENT_ID
                                .to_string(),
                            supports_parallel_tool_calls: false,
                            default_tools_approval_mode: None,
                        },
                    );
                }
                _ => {}
            }
        }

        config
            .mcp_servers
            .set(new_mcp_servers)
            .map_err(|e| anyhow::anyhow!(e))?;

        Ok(config)
    }
}

impl CodexAgent {
    async fn initialize(&self, request: InitializeRequest) -> Result<InitializeResponse, Error> {
        let InitializeRequest {
            protocol_version,
            client_capabilities,
            client_info,
            meta,
            ..
        } = request;
        debug!("Received initialize request with protocol version {protocol_version:?}",);
        let protocol_version = ProtocolVersion::V1;

        if let Some(ref info) = client_info {
            info!(
                "Connected Client Info: name={}, version={}",
                info.name, info.version
            );
        }
        if let Ok(json_caps) = serde_json::to_string_pretty(&client_capabilities) {
            info!("Client Capabilities: {}", json_caps);
        }

        *self.client_capabilities.lock().unwrap() = client_capabilities.clone();

        let mut agent_capabilities = AgentCapabilities::new()
            .prompt_capabilities(PromptCapabilities::new().embedded_context(true).image(true))
            .mcp_capabilities(McpCapabilities::new().http(false).sse(false))
            .load_session(true)
            .auth(AgentAuthCapabilities::new().logout(LogoutCapabilities::new()));

        agent_capabilities.session_capabilities = SessionCapabilities::new()
            .close(SessionCloseCapabilities::new())
            .list(SessionListCapabilities::new())
            .additional_directories(SessionAdditionalDirectoriesCapabilities::new());

        let mut agent_meta = client_capabilities.meta.unwrap_or_default();
        agent_meta.insert(
            "cognition.ai/mcp".to_string(),
            serde_json::Value::Bool(true),
        );
        agent_meta.insert(
            "cognition.ai/canManageMcpServers".to_string(),
            serde_json::Value::Bool(true),
        );
        agent_meta.insert(
            "cognition.ai/sessionRename".to_string(),
            serde_json::Value::Bool(true),
        );
        agent_meta.insert(
            "cognition.ai/documentLifecycle".to_string(),
            serde_json::Value::Bool(true),
        );
        agent_capabilities.meta = Some(agent_meta);

        let mut auth_methods = vec![
            CodexAuthMethod::WindsurfApiKey.into(),
            CodexAuthMethod::ChatGpt.into(),
            CodexAuthMethod::CodexApiKey.into(),
            CodexAuthMethod::OpenAiApiKey.into(),
        ];
        // Until codex device code auth works, we can't use this in remote ssh projects
        if std::env::var("NO_BROWSER").is_ok() {
            auth_methods.remove(1);
        }

        let mut response_meta = serde_json::Map::new();
        if let Some(meta) = meta {
            response_meta.extend(meta);
        }

        let mcp_config_path = self.config.codex_home.join(codex_config::CONFIG_TOML_FILE);

        response_meta.insert(
            "mcpConfigPath".to_string(),
            serde_json::json!(mcp_config_path.to_string_lossy()),
        );

        response_meta.insert(
            "cognition.ai/canManageMcpServers".to_string(),
            serde_json::json!(true),
        );

        let response = InitializeResponse::new(protocol_version)
            .agent_capabilities(agent_capabilities)
            .agent_info(Implementation::new("codex-acp", env!("CARGO_PKG_VERSION")).title("Codex"))
            .auth_methods(auth_methods)
            .meta(response_meta);

        Ok(response)
    }

    async fn authenticate(
        &self,
        request: AuthenticateRequest,
    ) -> Result<AuthenticateResponse, Error> {
        let auth_method = CodexAuthMethod::try_from(request.method_id)?;

        // Check before starting login flow if already authenticated with the same method
        if let Some(auth) = self.auth_manager.auth().await {
            match (auth, auth_method) {
                (
                    CodexAuth::ApiKey(..),
                    CodexAuthMethod::CodexApiKey | CodexAuthMethod::OpenAiApiKey,
                )
                | (CodexAuth::Chatgpt(..), CodexAuthMethod::ChatGpt) => {
                    return Ok(AuthenticateResponse::new());
                }
                _ => {}
            }
        }

        match auth_method {
            CodexAuthMethod::ChatGpt => {
                // Perform browser/device login via codex-rs, then report success/failure to the client.
                let opts = codex_login::ServerOptions::new(
                    self.config.codex_home.to_path_buf(),
                    codex_login::auth::CLIENT_ID.to_string(),
                    None,
                    self.config.cli_auth_credentials_store_mode,
                );

                let server =
                    codex_login::run_login_server(opts).map_err(Error::into_internal_error)?;

                server
                    .block_until_done()
                    .await
                    .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::CodexApiKey => {
                let api_key = read_codex_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{CODEX_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::OpenAiApiKey => {
                let api_key = read_openai_api_key_from_env().ok_or_else(|| {
                    Error::internal_error().data(format!("{OPENAI_API_KEY_ENV_VAR} is not set"))
                })?;
                codex_login::login_with_api_key(
                    &self.config.codex_home,
                    &api_key,
                    self.config.cli_auth_credentials_store_mode,
                )
                .map_err(Error::into_internal_error)?;
            }
            CodexAuthMethod::WindsurfApiKey => {}
        }

        self.auth_manager.reload().await;

        Ok(AuthenticateResponse::new())
    }

    async fn logout(&self, _request: LogoutRequest) -> Result<LogoutResponse, Error> {
        self.auth_manager
            .logout()
            .await
            .map_err(Error::into_internal_error)?;
        Ok(LogoutResponse::new())
    }

    async fn new_session(
        &self,
        request: NewSessionRequest,
        cx: ConnectionTo<Client>,
    ) -> Result<NewSessionResponse, Error> {
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let NewSessionRequest {
            cwd, mcp_servers, ..
        } = request;
        info!("Creating new session with cwd: {}", cwd.display());

        let config = self.build_session_config(&cwd, mcp_servers)?;
        let num_mcp_servers = config.mcp_servers.len();

        let NewThread {
            thread_id,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.start_thread(config.clone()))
            .await
            .map_err(|_e| Error::internal_error())?;

        let session_id = Self::session_id_from_thread_id(thread_id);
        // Record the session root for filesystem sandboxing.
        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd.to_path_buf());
        let thread = Arc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            Arc::new(self.thread_manager.get_models_manager()),
            self.client_capabilities.clone(),
            config.clone(),
            cx,
        ));
        let load = thread.load().await?;

        self.sessions
            .lock()
            .unwrap()
            .insert(session_id.clone(), thread);

        debug!("Created new session with {} MCP servers", num_mcp_servers);

        Ok(NewSessionResponse::new(session_id)
            .modes(load.modes)
            .models(load.models)
            .config_options(load.config_options))
    }

    async fn load_session(
        &self,
        request: LoadSessionRequest,
        cx: ConnectionTo<Client>,
    ) -> Result<LoadSessionResponse, Error> {
        info!("Loading session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let LoadSessionRequest {
            session_id,
            cwd,
            mcp_servers,
            ..
        } = request;

        let rollout_path = find_thread_path_by_id_str(
            &self.config.codex_home,
            session_id.0.as_ref(),
            self.state_db.as_deref(),
        )
        .await
        .map_err(|e| Error::internal_error().data(e.to_string()))?
        .ok_or_else(|| Error::resource_not_found(None))?;

        let history = RolloutRecorder::get_rollout_history(&rollout_path)
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        let rollout_items = match &history {
            InitialHistory::Resumed(resumed) => resumed.history.clone(),
            InitialHistory::Forked(items) => items.clone(),
            InitialHistory::Cleared | InitialHistory::New => Vec::new(),
        };

        let config = self.build_session_config(&cwd, mcp_servers)?;

        let NewThread {
            thread_id: _,
            thread,
            session_configured: _,
        } = Box::pin(self.thread_manager.resume_thread_from_rollout(
            config.clone(),
            rollout_path,
            self.auth_manager.clone(),
            None,
        ))
        .await
        .map_err(|e| Error::internal_error().data(e.to_string()))?;

        let thread = Arc::new(Thread::new(
            session_id.clone(),
            thread,
            self.auth_manager.clone(),
            Arc::new(self.thread_manager.get_models_manager()),
            self.client_capabilities.clone(),
            config.clone(),
            cx,
        ));

        thread.replay_history(rollout_items).await?;

        let load = thread.load().await?;

        self.session_roots
            .lock()
            .unwrap()
            .insert(session_id.clone(), config.cwd.to_path_buf());
        self.sessions.lock().unwrap().insert(session_id, thread);

        Ok(LoadSessionResponse::new()
            .modes(load.modes)
            .models(load.models)
            .config_options(load.config_options))
    }

    async fn list_sessions(
        &self,
        request: ListSessionsRequest,
    ) -> Result<ListSessionsResponse, Error> {
        self.check_auth().await?;

        let ListSessionsRequest { cwd, cursor, .. } = request;
        let cursor_obj = cursor.as_deref().and_then(parse_cursor);

        let page = RolloutRecorder::list_threads(
            self.state_db.clone(),
            &self.config,
            SESSION_LIST_PAGE_SIZE,
            cursor_obj.as_ref(),
            ThreadSortKey::UpdatedAt,
            SortDirection::Desc,
            &[
                SessionSource::Cli,
                SessionSource::VSCode,
                SessionSource::Unknown,
            ],
            None,
            None,
            self.config.model_provider_id.as_str(),
            None,
        )
        .await
        .map_err(|err| Error::internal_error().data(format!("failed to list sessions: {err}")))?;

        let sessions = page
            .items
            .into_iter()
            .filter_map(|item| {
                let thread_id = item.thread_id?;
                let item_cwd = item.cwd?;

                if let Some(filter_cwd) = cwd.as_ref()
                    && item_cwd != *filter_cwd
                {
                    return None;
                }

                let title = item
                    .first_user_message
                    .as_deref()
                    .and_then(format_session_title);
                let updated_at = item.updated_at.or(item.created_at);

                Some(
                    SessionInfo::new(SessionId::new(thread_id.to_string()), item_cwd)
                        .title(title)
                        .updated_at(updated_at),
                )
            })
            .collect::<Vec<_>>();

        let next_cursor = page
            .next_cursor
            .as_ref()
            .and_then(|next_cursor| serde_json::to_value(next_cursor).ok())
            .and_then(|value| value.as_str().map(str::to_owned));

        Ok(ListSessionsResponse::new(sessions).next_cursor(next_cursor))
    }

    async fn close_session(
        &self,
        request: CloseSessionRequest,
    ) -> Result<CloseSessionResponse, Error> {
        self.get_thread(&request.session_id)?.shutdown().await?;
        self.thread_manager
            .remove_thread(
                &ThreadId::from_string(&request.session_id.0)
                    .map_err(Error::into_internal_error)?,
            )
            .await;
        self.sessions.lock().unwrap().remove(&request.session_id);
        self.session_roots
            .lock()
            .unwrap()
            .remove(&request.session_id);
        Ok(CloseSessionResponse::new())
    }

    async fn prompt(&self, request: PromptRequest) -> Result<PromptResponse, Error> {
        info!("Processing prompt for session: {}", request.session_id);
        // Check before sending if authentication was successful or not
        self.check_auth().await?;

        let user_message_id = request.message_id.clone();

        // Get the session state
        let thread = self.get_thread(&request.session_id)?;
        let stop_reason = thread.prompt(request).await?;

        let mut response = PromptResponse::new(stop_reason);
        response.user_message_id = user_message_id;
        Ok(response)
    }

    async fn cancel(&self, args: CancelNotification) -> Result<(), Error> {
        info!("Cancelling operations for session: {}", args.session_id);
        self.get_thread(&args.session_id)?.cancel().await?;
        Ok(())
    }

    async fn set_session_mode(
        &self,
        args: SetSessionModeRequest,
    ) -> Result<SetSessionModeResponse, Error> {
        info!("Setting session mode for session: {}", args.session_id);
        self.get_thread(&args.session_id)?
            .set_mode(args.mode_id)
            .await?;
        Ok(SetSessionModeResponse::default())
    }

    async fn set_session_model(
        &self,
        args: SetSessionModelRequest,
    ) -> Result<SetSessionModelResponse, Error> {
        info!("Setting session model for session: {}", args.session_id);

        self.get_thread(&args.session_id)?
            .set_model(args.model_id)
            .await?;

        Ok(SetSessionModelResponse::default())
    }

    async fn set_session_config_option(
        &self,
        args: SetSessionConfigOptionRequest,
    ) -> Result<SetSessionConfigOptionResponse, Error> {
        info!(
            "Setting session config option for session: {} (config_id: {}, value: {:?})",
            args.session_id, args.config_id.0, args.value
        );

        let thread = self.get_thread(&args.session_id)?;

        thread.set_config_option(args.config_id, args.value).await?;

        let config_options = thread.config_options().await?;

        Ok(SetSessionConfigOptionResponse::new(config_options))
    }

    async fn revert_list_steps(
        &self,
        request: RevertListStepsRequest,
    ) -> Result<RevertListStepsResponse, Error> {
        info!("Listing revert steps for session: {}", request.session_id);
        let thread = self.get_thread(&request.session_id)?;
        let steps = thread.list_steps().await?;
        Ok(RevertListStepsResponse { steps })
    }

    async fn revert_preview(
        &self,
        request: RevertPreviewRequest,
    ) -> Result<RevertPreviewResponse, Error> {
        info!(
            "Previewing revert for session: {} target_node_id: {}",
            request.session_id, request.target_node_id
        );
        let thread = self.get_thread(&request.session_id)?;
        let response = thread.preview(request.target_node_id).await?;
        Ok(response)
    }

    async fn revert_execute(
        &self,
        request: RevertExecuteRequest,
    ) -> Result<RevertExecuteResponse, Error> {
        info!(
            "Executing revert for session: {} target_node_id: {}",
            request.session_id, request.target_node_id
        );
        let thread = self.get_thread(&request.session_id)?;

        thread.revert(request.target_node_id).await?;

        let old_thread = self.sessions.lock().unwrap().remove(&request.session_id);
        if let Some(old_thread) = old_thread {
            let _unused = old_thread.shutdown().await;
        }

        Ok(RevertExecuteResponse {
            forked_session_id: request.session_id.clone(),
            outcomes: Vec::new(),
        })
    }

    fn get_active_session_id_and_cwd(&self) -> Result<(SessionId, PathBuf), Error> {
        let sessions = self.sessions.lock().unwrap();
        if let Some((session_id, _)) = sessions.iter().next() {
            let roots = self.session_roots.lock().unwrap();
            if let Some(cwd) = roots.get(session_id) {
                return Ok((session_id.clone(), cwd.clone()));
            }
        }
        Err(Error::resource_not_found(Some(
            "No active sessions".to_string(),
        )))
    }

    async fn mcp_list_servers(
        &self,
        _request: McpListServersRequest,
    ) -> Result<McpListServersResponse, Error> {
        info!("Listing MCP servers");
        let thread = self.get_any_active_thread()?;
        let config = thread.config().await;

        let mcp_manager = self.thread_manager.mcp_manager();
        let mcp_servers = mcp_manager.configured_servers(&config).await;

        let manager = thread.mcp_connection_manager();
        let manager_read = manager.read().await;

        let mut servers = Vec::new();
        let source_path = self.config.codex_home.join(codex_config::CONFIG_TOML_FILE).to_string_lossy().to_string();

        for (name, cfg) in mcp_servers {
            let connection_status = manager_read.get_server_connection_status(&name).await;

            servers.push(McpServerStatus {
                server_id: name,
                disabled: !cfg.enabled,
                disabled_tools: cfg.disabled_tools.clone().unwrap_or_default(),
                connection_status,
                source_path: source_path.clone(),
            });
        }

        Ok(McpListServersResponse { servers })
    }

    async fn mcp_connect_server(
        &self,
        request: McpConnectServerRequest,
    ) -> Result<McpConnectServerResponse, Error> {
        info!("Connecting to MCP server: {}", request.server_id);
        let thread = self.get_any_active_thread()?;
        let manager = thread.mcp_connection_manager();

        let has_client = {
            let manager_read = manager.read().await;
            manager_read
                .get_server_connection_status(&request.server_id)
                .await
                != "not_started"
        };

        if !has_client {
            if let Ok((_session_id, cwd)) = self.get_active_session_id_and_cwd() {
                let config = self.build_session_config(&cwd, vec![])?;

                let mcp_servers = config.mcp_servers.get().clone();
                let refresh_config = codex_protocol::protocol::McpServerRefreshConfig {
                    mcp_servers: serde_json::to_value(mcp_servers)
                        .map_err(|e| Error::internal_error().data(e.to_string()))?,
                    mcp_oauth_credentials_store_mode: serde_json::to_value(
                        config.mcp_oauth_credentials_store_mode,
                    )
                    .map_err(|e| Error::internal_error().data(e.to_string()))?,
                };

                thread
                    .submit(Op::RefreshMcpServers {
                        config: refresh_config,
                    })
                    .await
                    .map_err(|e| Error::internal_error().data(e.to_string()))?;
            }
        }

        let is_ready = {
            let manager_read = manager.read().await;
            manager_read
                .wait_for_server_ready(&request.server_id, std::time::Duration::from_secs(10))
                .await
        };

        let status = if is_ready {
            "connected".to_string()
        } else {
            let manager_read = manager.read().await;
            manager_read
                .get_server_connection_status(&request.server_id)
                .await
        };

        Ok(McpConnectServerResponse {
            connection_status: status,
        })
    }

    async fn mcp_list_tools(
        &self,
        request: McpListToolsRequest,
    ) -> Result<McpListToolsResponse, Error> {
        info!("Listing tools for MCP server: {}", request.server_id);
        let thread = self.get_any_active_thread()?;
        let manager = thread.mcp_connection_manager();
        let manager_read = manager.read().await;

        let mut tools = Vec::new();
        if let Some(server_tools) = manager_read.list_tools_for_server(&request.server_id).await {
            for t in server_tools {
                let mut name = t.tool.name.to_string();
                let prefix_double = format!("{}__", request.server_id);
                let prefix_single = format!("{}_", request.server_id);
                if name.starts_with(&prefix_double) {
                    name = name.split_off(prefix_double.len());
                } else if name.starts_with(&prefix_single) {
                    name = name.split_off(prefix_single.len());
                }

                tools.push(McpToolDetail {
                    name,
                    description: t.tool.description.as_ref().map(|s| s.to_string()),
                    input_schema: serde_json::Value::Object((*t.tool.input_schema).clone()),
                });
            }
        }

        Ok(McpListToolsResponse {
            server_id: request.server_id,
            tools,
        })
    }

    async fn mcp_toggle_tool(
        &self,
        request: McpToggleToolRequest,
    ) -> Result<McpToggleToolResponse, Error> {
        info!(
            "Toggling tool: {} on server: {}",
            request.tool_name, request.server_id
        );

        let thread = self.get_any_active_thread()?;
        let config = thread.config().await;

        let mcp_manager = self.thread_manager.mcp_manager();
        let mcp_servers = mcp_manager.configured_servers(&config).await;

        let mut disabled_tools = if let Some(server_cfg) = mcp_servers.get(&request.server_id) {
            server_cfg.disabled_tools.clone().unwrap_or_default()
        } else {
            Vec::new()
        };

        let tool_pos = disabled_tools.iter().position(|v| v == &request.tool_name);
        if let Some(pos) = tool_pos {
            disabled_tools.remove(pos);
        } else {
            disabled_tools.push(request.tool_name.clone());
        }

        let codex_home = self.config.codex_home.clone();

        codex_config::ConfigEditsBuilder::new(&codex_home)
            .with_edits([codex_config::ConfigEdit::SetPath {
                segments: vec![
                    "mcp_servers".to_string(),
                    request.server_id.clone(),
                    "disabled_tools".to_string(),
                ],
                value: codex_config::TomlValue::Array(
                    disabled_tools
                        .into_iter()
                        .map(codex_config::TomlValue::String)
                        .collect(),
                ),
            }])
            .apply()
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        // Refresh the runtime config in the active thread by reloading from disk
        let next_config = codex_core::config::Config::load_with_cli_overrides(vec![])
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        thread.refresh_runtime_config(next_config).await;

        // Trigger a refresh of the session configuration with the updated configs
        let reloaded_config = thread.config().await;
        let refreshed_mcp_servers = mcp_manager.configured_servers(&reloaded_config).await;
        let refresh_config = codex_protocol::protocol::McpServerRefreshConfig {
            mcp_servers: serde_json::to_value(refreshed_mcp_servers)
                .map_err(|e| Error::internal_error().data(e.to_string()))?,
            mcp_oauth_credentials_store_mode: serde_json::to_value(
                reloaded_config.mcp_oauth_credentials_store_mode,
            )
            .map_err(|e| Error::internal_error().data(e.to_string()))?,
        };

        thread
            .submit(Op::RefreshMcpServers {
                config: refresh_config,
            })
            .await
            .map_err(|e| Error::internal_error().data(e.to_string()))?;

        Ok(McpToggleToolResponse {})
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CodexAuthMethod {
    ChatGpt,
    CodexApiKey,
    OpenAiApiKey,
    WindsurfApiKey,
}

impl From<CodexAuthMethod> for AuthMethodId {
    fn from(method: CodexAuthMethod) -> Self {
        Self::new(match method {
            CodexAuthMethod::ChatGpt => "chatgpt",
            CodexAuthMethod::CodexApiKey => "codex-api-key",
            CodexAuthMethod::OpenAiApiKey => "openai-api-key",
            CodexAuthMethod::WindsurfApiKey => "windsurf-api-key",
        })
    }
}

impl From<CodexAuthMethod> for AuthMethod {
    fn from(method: CodexAuthMethod) -> Self {
        match method {
            CodexAuthMethod::ChatGpt => Self::Agent(
                AuthMethodAgent::new(method, "Login with ChatGPT").description(
                    "Use your ChatGPT login with Codex CLI (requires a paid ChatGPT subscription)",
                ),
            ),
            CodexAuthMethod::CodexApiKey => Self::EnvVar(
                AuthMethodEnvVar::new(
                    method,
                    format!("Use {CODEX_API_KEY_ENV_VAR}"),
                    vec![AuthEnvVar::new(CODEX_API_KEY_ENV_VAR)],
                )
                .description(format!(
                    "Requires setting the `{CODEX_API_KEY_ENV_VAR}` environment variable."
                )),
            ),
            CodexAuthMethod::OpenAiApiKey => Self::EnvVar(
                AuthMethodEnvVar::new(
                    method,
                    format!("Use {OPENAI_API_KEY_ENV_VAR}"),
                    vec![AuthEnvVar::new(OPENAI_API_KEY_ENV_VAR)],
                )
                .description(format!(
                    "Requires setting the `{OPENAI_API_KEY_ENV_VAR}` environment variable."
                )),
            ),
            CodexAuthMethod::WindsurfApiKey => Self::Agent(
                AuthMethodAgent::new(method, "API Key")
                    .description("Authenticate with your API key"),
            ),
        }
    }
}

impl TryFrom<AuthMethodId> for CodexAuthMethod {
    type Error = Error;

    fn try_from(value: AuthMethodId) -> Result<Self, Self::Error> {
        match value.0.as_ref() {
            "chatgpt" => Ok(CodexAuthMethod::ChatGpt),
            "codex-api-key" => Ok(CodexAuthMethod::CodexApiKey),
            "openai-api-key" => Ok(CodexAuthMethod::OpenAiApiKey),
            "windsurf-api-key" => Ok(CodexAuthMethod::WindsurfApiKey),
            _ => Err(Error::invalid_params().data("unsupported authentication method")),
        }
    }
}

fn truncate_graphemes(text: &str, max_graphemes: usize) -> String {
    let mut graphemes = text.grapheme_indices(true);

    if let Some((byte_index, _)) = graphemes.nth(max_graphemes) {
        if max_graphemes >= 3 {
            let mut truncate_graphemes = text.grapheme_indices(true);
            if let Some((truncate_byte_index, _)) = truncate_graphemes.nth(max_graphemes - 3) {
                let truncated = &text[..truncate_byte_index];
                format!("{truncated}...")
            } else {
                text.to_string()
            }
        } else {
            let truncated = &text[..byte_index];
            truncated.to_string()
        }
    } else {
        text.to_string()
    }
}

fn format_session_title(message: &str) -> Option<String> {
    let normalized = message.replace(['\r', '\n'], " ");
    let trimmed = normalized.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(truncate_graphemes(trimmed, SESSION_TITLE_MAX_GRAPHEMES))
    }
}
