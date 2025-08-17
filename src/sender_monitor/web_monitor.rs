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
        .cooldown-info {
            background: #fffbf0;
            border: 1px solid #f6e05e;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
            font-family: monospace;
            font-size: 12px;
        }
        .status-cooldown {
            color: orange;
            font-weight: bold;
        }
        .status-connecting {
            color: orange;
            font-weight: bold;
        }
        .compact-view {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .compact-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .compact-item:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .compact-ip {
            font-family: monospace;
            font-size: 14px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        .compact-status {
            font-size: 12px;
            font-weight: bold;
        }
        .view-toggle {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .view-toggle:hover {
            background: #5a6fd8;
        }
        .controller-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }
        .controller-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        .controller-card.collapsed {
            padding: 15px;
        }
        .controller-card.collapsed .details {
            display: none;
        }
        .controller-card.expanded .details {
            display: block;
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

    <div style="display: flex; gap: 10px; margin-bottom: 20px;">
        <button class="view-toggle" onclick="toggleView('compact')">📱 Compact View</button>
        <button class="view-toggle" onclick="toggleView('detailed')">📋 Detailed View</button>
    </div>

    <div id="compact-view" class="compact-view" style="display: none;">
        <!-- Compact view will be populated here -->
    </div>

    <div id="detailed-view">
        <h2>🎛️ Controller Status</h2>
        <div class="controller-grid" id="controller-grid">
            <div class="controller-card">
                <p>Loading controller data...</p>
            </div>
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
                card.className = 'controller-card collapsed';
                card.setAttribute('data-ip', controller.ip);

                // Determine status and styling based on the new logic
                let statusClass, statusText, statusIcon;
                if (controller.is_connecting) {
                    statusClass = 'status-connecting';
                    statusText = '🟡 Connecting...';
                    statusIcon = '🟡';
                } else if (controller.is_routable) {
                    statusClass = 'status-connected';
                    statusText = '🟢 Connected';
                    statusIcon = '🟢';
                } else {
                    statusClass = 'status-disconnected';
                    statusText = '🔴 Disconnected';
                    statusIcon = '🔴';
                }

                // Calculate time remaining in cooldown
                let cooldownInfo = '';
                if (controller.cooldown_until && controller.is_connecting) {
                    const cooldownTime = new Date(controller.cooldown_until);
                    const now = new Date();
                    if (cooldownTime > now) {
                        const remainingMs = cooldownTime - now;
                        const remainingSeconds = Math.ceil(remainingMs / 1000);
                        cooldownInfo = `<div class="cooldown-info">
                            <strong>⏰ Cooldown Active:</strong> ${remainingSeconds}s remaining before connection attempt
                        </div>`;
                    } else {
                        // Cooldown expired, show transition message
                        cooldownInfo = `<div class="cooldown-info">
                            <strong>✅ Cooldown Complete:</strong> Controller will transition to Connected on next successful transmission
                        </div>`;
                    }
                }

                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;" onclick="toggleCard(this.parentElement)">
                        <h3>${controller.ip}:${controller.port}</h3>
                        <span class="${statusClass}">${statusText}</span>
                        <span style="font-size: 20px;">📋</span>
                    </div>
                    <div class="details">
                        <p><strong>Last Success:</strong> ${formatDateTime(controller.last_success)}</p>
                        <p><strong>Last Failure:</strong> ${formatDateTime(controller.last_failure)}</p>
                        <p><strong>Failure Count:</strong> ${controller.failure_count}</p>
                        ${controller.last_error ? `<div class="error-details"><strong>Last Error:</strong> ${controller.last_error}</div>` : ''}
                        ${cooldownInfo}
                    </div>
                `;

                // Restore expanded state if this card was previously expanded
                if (expandedCards.has(controller.ip)) {
                    card.classList.remove('collapsed');
                    card.classList.add('expanded');
                }

                grid.appendChild(card);
            });
        }

        function toggleCard(card) {
            const ip = card.getAttribute('data-ip');

            if (card.classList.contains('collapsed')) {
                card.classList.remove('collapsed');
                card.classList.add('expanded');
                expandedCards.add(ip); // Remember this card is expanded
            } else {
                card.classList.remove('expanded');
                card.classList.add('collapsed');
                expandedCards.delete(ip); // Remember this card is collapsed
            }
        }

        async function refreshData() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                updateStats(data);
                updateControllers(data);
                window.lastData = data; // Store data for compact view update
                updateCompactView(); // Update compact view after stats and controllers are loaded
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        }

        // Refresh data every 2 seconds
        setInterval(refreshData, 2000);

        // Initial load
        refreshData();

        // View management
        let currentView = 'detailed';
        let expandedCards = new Set(); // Track which cards are expanded

        function toggleView(view) {
            currentView = view;
            const compactView = document.getElementById('compact-view');
            const detailedView = document.getElementById('detailed-view');

            if (view === 'compact') {
                compactView.style.display = 'grid';
                detailedView.style.display = 'none';
                updateCompactView();
            } else {
                compactView.style.display = 'none';
                detailedView.style.display = 'block';
            }
        }

        function updateCompactView() {
            const compactView = document.getElementById('compact-view');
            if (!window.lastData) return;

            compactView.innerHTML = '';

            window.lastData.controllers.forEach(controller => {
                const item = document.createElement('div');
                item.className = 'compact-item';

                // Determine status and styling
                let statusClass, statusText, statusIcon;
                if (controller.is_connecting) {
                    statusClass = 'status-connecting';
                    statusText = '🟡 Connecting...';
                } else if (controller.is_routable) {
                    statusClass = 'status-connected';
                    statusText = '🟢 Connected';
                } else {
                    statusClass = 'status-disconnected';
                    statusText = '🔴 Disconnected';
                }

                item.innerHTML = `
                    <div class="compact-ip">${controller.ip}</div>
                    <div class="compact-status ${statusClass}">${statusText}</div>
                `;

                // Make compact items clickable to expand details
                item.onclick = () => {
                    toggleView('detailed');
                    // Scroll to the specific controller
                    const controllerCard = document.querySelector(`[data-ip="${controller.ip}"]`);
                    if (controllerCard) {
                        controllerCard.scrollIntoView({ behavior: 'smooth' });
                        // Highlight the controller briefly
                        controllerCard.style.background = '#fff3cd';
                        setTimeout(() => {
                            controllerCard.style.background = 'white';
                        }, 2000);
                    }
                };

                compactView.appendChild(item);
            });
        }
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
