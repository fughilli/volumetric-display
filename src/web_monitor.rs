use crate::control_port::{ControllerManager, ControllerStats, LogEntry};
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
    controller_manager: Arc<ControllerManager>,
}

impl WebMonitor {
    pub fn new(controller_manager: Arc<ControllerManager>) -> Self {
        Self { controller_manager }
    }

    pub fn create_router(&self) -> Router {
        Router::new()
            .route("/", get(dashboard_html))
            .route("/api/controllers", get(get_controllers))
            .route("/api/controllers/:dip/logs", get(get_controller_logs))
            .route("/api/controllers/:dip/stats", get(get_controller_stats))
            .with_state(self.controller_manager.clone())
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
    <title>Controller Monitor Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .controller-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .controller-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
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
        <h1>Controller Monitor Dashboard</h1>
        <p>Real-time monitoring of controller connections and communication</p>
    </div>
    <div id="controllers" class="controller-grid">
        <div style="text-align: center; padding: 40px;">Loading controller data...</div>
    </div>
    <script>
        async function fetchControllers() {
            const response = await fetch('/api/controllers');
            return response.ok ? (await response.json()).controllers : [];
        }
        async function fetchLogs(dip) {
            const response = await fetch(`/api/controllers/${dip}/logs`);
            return response.ok ? await response.json() : [];
        }
        function formatTime(timestamp) { return new Date(timestamp).toLocaleTimeString(); }
        function formatBytes(bytes) { return bytes < 1024 ? bytes + ' B' : Math.round(bytes/1024) + ' KB'; }
        async function updateDashboard() {
            const controllers = await fetchControllers();
            const container = document.getElementById('controllers');
            if (controllers.length === 0) {
                container.innerHTML = '<div style="text-align: center; padding: 40px;">No controllers found</div>';
                return;
            }
            const cards = await Promise.all(controllers.map(async controller => {
                const logs = (await fetchLogs(controller.dip)).slice(-5);
                const statusClass = controller.connected ? 'status-connected' : 'status-disconnected';
                const statusText = controller.connected ? 'Connected' : 'Disconnected';
                return `
                    <div class="controller-card">
                        <h3>Controller ${controller.dip}</h3>
                        <div class="${statusClass}">${statusText}</div>
                        <p><strong>Address:</strong> ${controller.ip}:${controller.port}</p>
                        <p><strong>Messages:</strong> ↑${controller.messages_sent} ↓${controller.messages_received}</p>
                        <p><strong>Data:</strong> ↑${formatBytes(controller.bytes_sent)} ↓${formatBytes(controller.bytes_received)}</p>
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

async fn get_controllers(
    State(manager): State<Arc<ControllerManager>>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let stats = manager.get_controller_stats().await;
    Ok(Json(json!({ "controllers": stats })))
}

async fn get_controller_logs(
    Path(dip): Path<String>,
    State(manager): State<Arc<ControllerManager>>,
) -> Result<Json<Vec<LogEntry>>, StatusCode> {
    match manager.get_controller_logs(&dip).await {
        Some(logs) => Ok(Json(logs)),
        None => Err(StatusCode::NOT_FOUND),
    }
}

async fn get_controller_stats(
    Path(dip): Path<String>,
    State(manager): State<Arc<ControllerManager>>,
) -> Result<Json<ControllerStats>, StatusCode> {
    if let Some(controller) = manager.get_controller(&dip) {
        controller.update_stats().await;
        let stats = controller.stats.read().await.clone();
        Ok(Json(stats))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}
