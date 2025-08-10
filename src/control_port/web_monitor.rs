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
}

impl WebMonitor {
    pub fn new(control_port_manager: Arc<ControlPortManager>) -> Self {
        Self {
            control_port_manager,
        }
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

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
        println!("Web monitor server running on http://localhost:{}", port);

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
        .refresh-btn { background: #667eea; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
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
        function formatTime(timestamp) { return new Date(timestamp).toLocaleTimeString(); }
        function formatBytes(bytes) { return bytes < 1024 ? bytes + ' B' : Math.round(bytes/1024) + ' KB'; }
        async function updateDashboard() {
            const controlPorts = await fetchControlPorts();
            const container = document.getElementById('control_ports');
            if (controlPorts.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 40px;">No control ports found</div>';
                return;
            }
            const cards = await Promise.all(controlPorts.map(async controlPort => {
                const logs = (await fetchLogs(controlPort.dip)).slice(-5);
                const statusClass = controlPort.connected ? 'status-connected' : 'status-disconnected';
                const statusText = controlPort.connected ? 'Connected' : 'Disconnected';
                return `
                    <div class="control-port-card">
                        <h3>Control Port ${controlPort.dip}</h3>
                        <div class="${statusClass}">${statusText}</div>
                        <p><strong>Address:</strong> ${controlPort.ip}:${controlPort.port}</p>
                        <p><strong>Messages:</strong> ↑${controlPort.messages_sent} ↓${controlPort.messages_received}</p>
                        <p><strong>Data:</strong> ↑${formatBytes(controlPort.bytes_sent)} ↓${formatBytes(controlPort.bytes_received)}</p>
                        <div class="logs">
                            <strong>Recent Messages:</strong><br>
                            ${logs.map(log => `<div class="log-entry log-${log.direction}">${formatTime(log.timestamp)} ${log.direction}: ${log.message}</div>`).join('') || 'No recent messages'}
                        </div>
                    </div>
                `;
            }));
            container.innerHTML = cards.join('');
        }
        updateDashboard();
        setInterval(updateDashboard, 2000);
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
        Ok(Json(logs.iter().cloned().collect()))
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
