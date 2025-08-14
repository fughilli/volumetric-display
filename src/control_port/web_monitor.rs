use crate::control_port::{ControlPortManager, ControlPortStats, LogEntry};
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{Html, Json},
    routing::get,
    Router,
};
use serde_json::json;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub struct WebMonitor {
    control_port_manager: Arc<ControlPortManager>,
    log_buffer_size: usize,
    bind_address: String,
}

impl WebMonitor {
    pub fn new(control_port_manager: Arc<ControlPortManager>) -> Self {
        Self {
            control_port_manager,
            log_buffer_size: 1000,               // Default log buffer size
            bind_address: "0.0.0.0".to_string(), // Default bind address
        }
    }

    pub fn with_log_buffer_size(mut self, size: usize) -> Self {
        self.log_buffer_size = size;
        self
    }

    pub fn with_bind_address(mut self, bind_address: String) -> Self {
        self.bind_address = bind_address;
        self
    }

    pub fn create_router(&self) -> Router {
        Router::new()
            .route("/", get(dashboard_html))
            .route("/api/control_ports", get(get_control_ports))
            .route("/api/control_ports/:dip/logs", get(get_control_port_logs))
            .route("/api/control_ports/:dip/stats", get(get_control_port_stats))
            .with_state(self.control_port_manager.clone())
            .layer(CorsLayer::permissive())
    }

    pub async fn start_server(
        &self,
        port: u16,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let app = self.create_router();

        let bind_addr = format!("{}:{}", self.bind_address, port);
        let listener = tokio::net::TcpListener::bind(&bind_addr).await?;

        // Show both localhost and the actual bind address for convenience
        if self.bind_address == "0.0.0.0" {
            println!("Web monitor server running on:");
            println!("  Local: http://localhost:{}", port);
            println!("  Network: http://0.0.0.0:{}", port);
        } else {
            println!(
                "Web monitor server running on http://{}:{}",
                self.bind_address, port
            );
        }

        axum::serve(listener, app).await?;
        Ok(())
    }
}

async fn dashboard_html() -> Html<&'static str> {
    Html(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Control Port Monitor Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .control-port-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .control-port-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-connected { color: green; font-weight: bold; }
        .status-disconnected { color: red; font-weight: bold; }
        .logs { background: #f8f9fa; padding: 10px; border-radius: 4px; max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 12px; }
        .log-entry { padding: 2px 0; border-bottom: 1px solid #eee; }
        .log-incoming { color: blue; }
        .log-outgoing { color: green; }
        .log-error { color: red; }
        .log-info { color: #666; }
        .refresh-btn { background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        .heartbeat-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #ccc;
            margin-left: 8px;
            transition: background-color 0.3s ease;
        }
        .heartbeat-active { background-color: #4CAF50; animation: pulse 1s infinite; }
        .noop-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background-color: #ccc;
            margin-left: 8px;
            transition: background-color 0.3s ease;
        }
        .noop-active { background-color: #2196F3; animation: pulse 1s infinite; }
        .heartbeat-container {
            display: flex;
            align-items: center;
            margin: 5px 0;
            font-size: 14px;
        }
        .heartbeat-label {
            min-width: 120px;
            font-weight: bold;
        }
        .heartbeat-time {
            margin-left: 10px;
            color: #666;
            font-size: 12px;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .status-line { display: flex; align-items: center; justify-content: space-between; }
        .status-left { display: flex; align-items: center; }
        .logs-container {
            position: relative;
            max-height: 200px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 12px;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .logs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 5px;
        }
        .scroll-indicator {
            font-size: 10px;
            color: #666;
            cursor: help;
        }
        .scroll-indicator.auto { color: #4CAF50; }
        .scroll-indicator.manual { color: #FF9800; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Control Port Monitor Dashboard</h1>
        <p>Real-time monitoring of control port connections and communication</p>
    </div>
    <div id="control_ports" class="control-port-grid">
        <div style="text-align: center; padding: 40px;">Loading control port data...</div>
    </div>
    <script>
        async function fetchControlPorts() {
            const response = await fetch('/api/control_ports');
            return response.ok ? (await response.json()).control_ports : [];
        }
        async function fetchLogs(dip) {
            const response = await fetch(`/api/control_ports/${dip}/logs`);
            return response.ok ? await response.json() : [];
        }
        async function fetchHeartbeat(dip) {
            const response = await fetch(`/api/control_ports/${dip}/stats`);
            return response.ok ? await response.json() : {
                heartbeat_received_active: false,
                noop_sent_active: false,
                last_heartbeat_received: null,
                last_noop_sent: null
            };
        }
        function formatTime(timestamp) { return new Date(timestamp).toLocaleTimeString(); }
        function formatTimeElapsed(timestamp) {
            if (!timestamp) return 'Never';
            const now = new Date();
            const elapsed = Math.floor((now - new Date(timestamp)) / 1000);
            if (elapsed < 60) return `${elapsed}s ago`;
            if (elapsed < 3600) return `${Math.floor(elapsed / 60)}m ago`;
            return `${Math.floor(elapsed / 3600)}h ago`;
        }
        function formatBytes(bytes) { return bytes < 1024 ? bytes + ' B' : Math.round(bytes/1024) + ' KB'; }
        function formatThroughput(bps) {
            if (bps < 1024) return bps.toFixed(1) + ' B/s';
            if (bps < 1024*1024) return (bps/1024).toFixed(1) + ' KB/s';
            return (bps/(1024*1024)).toFixed(1) + ' MB/s';
        }

        // Smart scrolling state management
        const scrollStates = new Map();

        function isAtBottom(element) {
            return Math.abs(element.scrollHeight - element.scrollTop - element.clientHeight) < 5;
        }

        function saveScrollState(dip, element) {
            const wasAtBottom = isAtBottom(element);
            scrollStates.set(dip, {
                wasAtBottom,
                scrollTop: element.scrollTop,
                scrollHeight: element.scrollHeight,
                clientHeight: element.clientHeight
            });
        }

        function restoreScrollState(dip, element) {
            const state = scrollStates.get(dip);
            if (!state) return;

            // If user was at bottom, auto-scroll to new bottom
            if (state.wasAtBottom) {
                element.scrollTop = element.scrollHeight;
            } else {
                // Calculate the relative position and maintain it
                const oldScrollRatio = state.scrollTop / (state.scrollHeight - state.clientHeight);
                const newScrollHeight = element.scrollHeight;
                const newClientHeight = element.clientHeight;
                const newScrollTop = oldScrollRatio * (newScrollHeight - newClientHeight);
                element.scrollTop = Math.max(0, newScrollTop);
            }
        }
        async function updateDashboard() {
            const controlPorts = await fetchControlPorts();
            const container = document.getElementById('control_ports');
            if (controlPorts.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 40px;">No control ports found</div>';
                return;
            }
            const cards = await Promise.all(controlPorts.map(async controlPort => {
                const logs = (await fetchLogs(controlPort.dip)).slice(-10); // Show last 10 filtered messages
                const heartbeat = await fetchHeartbeat(controlPort.dip);
                const statusClass = controlPort.connected ? 'status-connected' : 'status-disconnected';
                const statusText = controlPort.connected ? 'Connected' : 'Disconnected';
                const heartbeatReceivedClass = heartbeat.heartbeat_received_active ? 'heartbeat-active' : '';
                const noopSentClass = heartbeat.noop_sent_active ? 'noop-active' : '';

                return `
                    <div class="control-port-card">
                        <h3>Control Port ${controlPort.dip}</h3>
                        <div class="status-line">
                            <div class="status-left">
                                <div class="${statusClass}">${statusText}</div>
                            </div>
                        </div>
                        <div class="heartbeat-container">
                            <span class="heartbeat-label">Heartbeat Received:</span>
                            <div class="heartbeat-indicator ${heartbeatReceivedClass}" title="Heartbeat Received Status"></div>
                            <span class="heartbeat-time">${formatTimeElapsed(heartbeat.last_heartbeat_received)}</span>
                        </div>
                        <div class="heartbeat-container">
                            <span class="heartbeat-label">Noop Sent:</span>
                            <div class="noop-indicator ${noopSentClass}" title="Noop Sent Status"></div>
                            <span class="heartbeat-time">${formatTimeElapsed(heartbeat.last_noop_sent)}</span>
                        </div>
                        <p><strong>Address:</strong> ${controlPort.ip}:${controlPort.port}</p>
                        <p><strong>Messages:</strong> ↑${controlPort.messages_sent} ↓${controlPort.messages_received}</p>
                        <p><strong>Data:</strong> ↑${formatBytes(controlPort.bytes_sent)} ↓${formatBytes(controlPort.bytes_received)}</p>
                        <p><strong>Throughput:</strong> ↑${formatThroughput(controlPort.throughput_sent_bps || 0)} ↓${formatThroughput(controlPort.throughput_received_bps || 0)}</p>
                        <div class="logs-container" id="logs-${controlPort.dip}" onscroll="saveScrollState('${controlPort.dip}', this)">
                            <div class="logs-header">
                                <strong>Recent Messages (heartbeats filtered)</strong>
                                <span class="scroll-indicator" id="scroll-indicator-${controlPort.dip}" title="Auto-scroll status">●</span>
                            </div>
                            ${logs.length > 0 ? logs.map(log => `<div class="log-entry log-${log.direction}">${formatTime(log.timestamp)} ${log.direction}: ${log.message}</div>`).join('') : '<div class="log-entry log-info">No recent activity</div>'}
                        </div>
                    </div>
                `;
            }));
            container.innerHTML = cards.join('');

            // Restore scroll states after DOM update
            controlPorts.forEach(controlPort => {
                const logsElement = document.getElementById(`logs-${controlPort.dip}`);
                const indicatorElement = document.getElementById(`scroll-indicator-${controlPort.dip}`);
                if (logsElement) {
                    restoreScrollState(controlPort.dip, logsElement);

                    // Update scroll indicator
                    if (indicatorElement) {
                        const state = scrollStates.get(controlPort.dip);
                        if (state && state.wasAtBottom) {
                            indicatorElement.className = 'scroll-indicator auto';
                            indicatorElement.title = 'Auto-scroll enabled (at bottom)';
                        } else {
                            indicatorElement.className = 'scroll-indicator manual';
                            indicatorElement.title = 'Manual scroll (not at bottom)';
                        }
                    }
                }
            });
        }

        // Handle window resize to maintain scroll positions
        window.addEventListener('resize', () => {
            // Small delay to let the resize complete
            setTimeout(() => {
                const controlPorts = document.querySelectorAll('.logs-container');
                controlPorts.forEach(element => {
                    const dip = element.id.replace('logs-', '');
                    restoreScrollState(dip, element);
                });
            }, 100);
        });

        // Initialize scroll states for new elements
        function initializeScrollStates() {
            const controlPorts = document.querySelectorAll('.logs-container');
            controlPorts.forEach(element => {
                const dip = element.id.replace('logs-', '');
                if (!scrollStates.has(dip)) {
                    // Initially assume user wants to see the latest messages
                    scrollStates.set(dip, {
                        wasAtBottom: true,
                        scrollTop: 0,
                        scrollHeight: element.scrollHeight,
                        clientHeight: element.clientHeight
                    });
                }
            });
        }

        updateDashboard();
        setInterval(updateDashboard, 2000);

        // Initialize scroll states after first load
        setTimeout(initializeScrollStates, 100);
    </script>
</body>
</html>"#,
    )
}

async fn get_control_ports(
    State(manager): State<Arc<ControlPortManager>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = manager.get_all_stats().await;
    Ok(Json(json!({ "control_ports": stats })))
}

async fn get_control_port_logs(
    Path(dip): Path<String>,
    State(manager): State<Arc<ControlPortManager>>,
) -> Result<Json<Vec<LogEntry>>, StatusCode> {
    if let Some(control_port) = manager.get_control_port(&dip) {
        let logs = control_port.logs.read().await;

        // Filter out heartbeat and noop messages and limit buffer size
        let filtered_logs: Vec<LogEntry> = logs
            .iter()
            .filter(|log| {
                // Filter out heartbeat and noop messages
                !log.message.contains("noop")
                    && !log.message.contains("Noop")
                    && !log.message.contains("heartbeat")
                    && !log.message.contains("Heartbeat")
            })
            .cloned()
            .collect();

        Ok(Json(filtered_logs))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

async fn get_control_port_stats(
    Path(dip): Path<String>,
    State(manager): State<Arc<ControlPortManager>>,
) -> Result<Json<ControlPortStats>, StatusCode> {
    if let Some(control_port) = manager.get_control_port(&dip) {
        let stats = control_port.get_stats().await;
        Ok(Json(stats))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}
