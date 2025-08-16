use crate::sender_monitor::SenderMonitor;
use axum::{
    extract::State,
    response::{Html, Json},
    routing::get,
    Router,
};
use serde_json::json;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

pub struct WebMonitor {
    sender_monitor: Arc<SenderMonitor>,
    bind_address: String,
}

impl WebMonitor {
    pub fn new(sender_monitor: Arc<SenderMonitor>) -> Self {
        Self {
            sender_monitor,
            bind_address: "0.0.0.0".to_string(),
        }
    }

    pub fn with_bind_address(mut self, bind_address: String) -> Self {
        self.bind_address = bind_address;
        self
    }

    pub fn create_router(&self) -> Router {
        Router::new()
            .route("/", get(dashboard_html))
            .route("/api/stats", get(get_stats))
            .route("/api/controllers", get(get_controllers))
            .route("/api/system", get(get_system_stats))
            .with_state(self.sender_monitor.clone())
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
            println!("Sender monitor server running on:");
            println!("  Local: http://localhost:{}", port);
            println!("  Network: http://0.0.0.0:{}", port);
        } else {
            println!(
                "Sender monitor server running on http://{}:{}",
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
    <title>ArtNet Sender Monitor Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 5px;
        }
        .controller-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }
        .controller-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status-connected {
            color: green;
            font-weight: bold;
        }
        .status-disconnected {
            color: red;
            font-weight: bold;
        }
        .refresh-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .refresh-btn:hover {
            background: #5a6fd8;
        }
        .error-details {
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎬 ArtNet Sender Monitor</h1>
        <p>Real-time monitoring of ArtNet controller status and system performance</p>
    </div>

    <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Data</button>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-value" id="fps">--</div>
            <div class="stat-label">Current FPS</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="uptime">--</div>
            <div class="stat-label">Uptime</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="total-frames">--</div>
            <div class="stat-label">Total Frames</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" id="routable-controllers">--</div>
            <div class="stat-label">Routable Controllers</div>
        </div>
    </div>

    <h2>🎛️ Controller Status</h2>
    <div class="controller-grid" id="controller-grid">
        <div class="controller-card">
            <p>Loading controller data...</p>
        </div>
    </div>

    <script>
        function formatUptime(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            return `${hours}h ${minutes}m ${secs}s`;
        }

        function formatDateTime(dateString) {
            if (!dateString) return 'Never';
            const date = new Date(dateString);
            return date.toLocaleString();
        }

        function updateStats(data) {
            document.getElementById('fps').textContent = data.system.fps.toFixed(1);
            document.getElementById('uptime').textContent = formatUptime(data.system.uptime_seconds);
            document.getElementById('total-frames').textContent = data.system.total_frames.toLocaleString();
            document.getElementById('routable-controllers').textContent =
                data.controllers.filter(c => c.is_routable).length + ' / ' + data.controllers.length;
        }

        function updateControllers(data) {
            const grid = document.getElementById('controller-grid');
            grid.innerHTML = '';

            data.controllers.forEach(controller => {
                const card = document.createElement('div');
                card.className = 'controller-card';

                const statusClass = controller.is_routable ? 'status-connected' : 'status-disconnected';
                const statusText = controller.is_routable ? '🟢 Connected' : '🔴 Disconnected';

                card.innerHTML = `
                    <h3>${controller.ip}:${controller.port}</h3>
                    <p><span class="${statusClass}">${statusText}</span></p>
                    <p><strong>Last Success:</strong> ${formatDateTime(controller.last_success)}</p>
                    <p><strong>Last Failure:</strong> ${formatDateTime(controller.last_failure)}</p>
                    <p><strong>Failure Count:</strong> ${controller.failure_count}</p>
                    ${controller.last_error ? `<div class="error-details"><strong>Last Error:</strong> ${controller.last_error}</div>` : ''}
                `;

                grid.appendChild(card);
            });
        }

        async function refreshData() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                updateStats(data);
                updateControllers(data);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        // Refresh data every 2 seconds
        setInterval(refreshData, 2000);

        // Initial load
        refreshData();
    </script>
</body>
</html>"#,
    )
}

async fn get_stats(State(sender_monitor): State<Arc<SenderMonitor>>) -> Json<serde_json::Value> {
    let stats = sender_monitor.get_stats().await;
    Json(serde_json::to_value(stats).unwrap_or(json!({"error": "Failed to serialize stats"})))
}

async fn get_controllers(
    State(sender_monitor): State<Arc<SenderMonitor>>,
) -> Json<serde_json::Value> {
    let stats = sender_monitor.get_stats().await;
    Json(json!({
        "controllers": stats.controllers,
        "total": stats.controllers.len(),
        "routable": stats.controllers.iter().filter(|c| c.is_routable).count()
    }))
}

async fn get_system_stats(
    State(sender_monitor): State<Arc<SenderMonitor>>,
) -> Json<serde_json::Value> {
    let stats = sender_monitor.get_stats().await;
    Json(json!({
        "system": stats.system,
        "controller_count": sender_monitor.get_controller_count(),
        "routable_controller_count": sender_monitor.get_routable_controller_count()
    }))
}
