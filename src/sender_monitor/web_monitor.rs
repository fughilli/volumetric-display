use crate::sender_monitor::{
    DebugCommand, MappingTesterCommand, PowerDrawTesterCommand, SenderMonitor,
};
use axum::{
    extract::{Json, State},
    response::{Html, Json as JsonResponse},
    routing::{get, post},
    Router,
};
use runfiles::Runfiles;
use serde_json::json;
use std::fs;
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
            .route("/api/debug/state", get(get_debug_state))
            .route("/api/debug/world-dimensions", get(get_world_dimensions))
            .route("/api/debug/mode", post(set_debug_mode))
            .route("/api/debug/pause", post(set_debug_pause))
            .route("/api/debug/mapping-tester", post(set_mapping_tester))
            .route("/api/debug/power-draw-tester", post(set_power_draw_tester))
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

async fn dashboard_html() -> Html<String> {
    // Use runfiles to locate the HTML file
    let r = Runfiles::create().expect("Failed to create runfiles");

    // Try to read the HTML file from runfiles
    // The path should be relative to the workspace root
    match r.rlocation("_main/static/debug_dashboard.html") {
        Some(path) => {
            match fs::read_to_string(path) {
                Ok(content) => Html(content),
                Err(e) => {
                    eprintln!("Failed to read debug dashboard HTML: {}", e);
                    // Fallback to a simple HTML if file not found
                    Html(
                        r#"<!DOCTYPE html>
<html>
<head><title>ArtNet Sender Monitor</title></head>
<body>
    <h1>ArtNet Sender Monitor</h1>
    <p>Debug dashboard HTML file not found. Please ensure static/debug_dashboard.html exists.</p>
    <p>Error: Failed to read file from runfiles</p>
</body>
</html>"#
                            .to_string(),
                    )
                }
            }
        }
        None => {
            eprintln!("Could not locate debug dashboard HTML in runfiles");
            // Fallback to a simple HTML if file not found
            Html(r#"<!DOCTYPE html>
<html>
<head><title>ArtNet Sender Monitor</title></head>
<body>
    <h1>ArtNet Sender Monitor</h1>
    <p>Debug dashboard HTML file not found in runfiles. Please ensure static/debug_dashboard.html exists.</p>
</body>
</html>"#.to_string())
        }
    }
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

async fn get_debug_state(
    State(sender_monitor): State<Arc<SenderMonitor>>,
) -> JsonResponse<serde_json::Value> {
    let debug_state = sender_monitor.get_debug_state().await;
    JsonResponse(json!(debug_state))
}

async fn get_world_dimensions(
    State(sender_monitor): State<Arc<SenderMonitor>>,
) -> JsonResponse<serde_json::Value> {
    if let Some((width, height, length)) = sender_monitor.get_world_dimensions().await {
        JsonResponse(json!({
            "width": width,
            "height": height,
            "length": length
        }))
    } else {
        JsonResponse(json!({
            "error": "World dimensions not set"
        }))
    }
}

async fn set_debug_mode(
    State(sender_monitor): State<Arc<SenderMonitor>>,
    Json(payload): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    if let Some(enabled) = payload.get("enabled").and_then(|v| v.as_bool()) {
        sender_monitor.set_debug_mode(enabled).await;
        JsonResponse(json!({"success": true, "debug_mode": enabled}))
    } else {
        JsonResponse(json!({"success": false, "error": "Missing 'enabled' field"}))
    }
}

async fn set_debug_pause(
    State(sender_monitor): State<Arc<SenderMonitor>>,
    Json(payload): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    if let Some(paused) = payload.get("paused").and_then(|v| v.as_bool()) {
        sender_monitor.set_debug_pause(paused).await;
        JsonResponse(json!({"success": true, "paused": paused}))
    } else {
        JsonResponse(json!({"success": false, "error": "Missing 'paused' field"}))
    }
}

async fn set_mapping_tester(
    State(sender_monitor): State<Arc<SenderMonitor>>,
    Json(payload): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    // Check if this is a clear command
    if payload
        .get("clear")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        // Clear all debug commands
        let command = DebugCommand {
            command_type: "clear".to_string(),
            mapping_tester: None,
            power_draw_tester: None,
        };

        sender_monitor.set_debug_command(command).await;
        JsonResponse(json!({"success": true, "command": "clear"}))
    } else {
        // Normal mapping tester command
        if let (Some(orientation), Some(layer), Some(color)) = (
            payload.get("orientation").and_then(|v| v.as_str()),
            payload.get("layer").and_then(|v| v.as_u64()),
            payload.get("color").and_then(|v| v.as_str()),
        ) {
            let command = DebugCommand {
                command_type: "mapping_tester".to_string(),
                mapping_tester: Some(MappingTesterCommand {
                    orientation: orientation.to_string(),
                    layer: layer as usize,
                    color: color.to_string(),
                }),
                power_draw_tester: None,
            };

            sender_monitor.set_debug_command(command).await;
            JsonResponse(json!({"success": true, "command": "mapping_tester"}))
        } else {
            JsonResponse(
                json!({"success": false, "error": "Missing required fields: orientation, layer, color"}),
            )
        }
    }
}

async fn set_power_draw_tester(
    State(sender_monitor): State<Arc<SenderMonitor>>,
    Json(payload): Json<serde_json::Value>,
) -> JsonResponse<serde_json::Value> {
    if let (
        Some(color),
        Some(modulation_type),
        Some(frequency),
        Some(amplitude),
        Some(offset),
        Some(global_brightness),
    ) = (
        payload.get("color").and_then(|v| v.as_str()),
        payload.get("modulation_type").and_then(|v| v.as_str()),
        payload.get("frequency").and_then(|v| v.as_f64()),
        payload.get("amplitude").and_then(|v| v.as_f64()),
        payload.get("offset").and_then(|v| v.as_f64()),
        payload.get("global_brightness").and_then(|v| v.as_f64()),
    ) {
        let command = DebugCommand {
            command_type: "power_draw_tester".to_string(),
            mapping_tester: None,
            power_draw_tester: Some(PowerDrawTesterCommand {
                color: color.to_string(),
                modulation_type: modulation_type.to_string(),
                frequency,
                amplitude,
                offset,
                global_brightness,
            }),
        };

        sender_monitor.set_debug_command(command).await;
        JsonResponse(json!({"success": true, "command": "power_draw_tester"}))
    } else {
        JsonResponse(
            json!({"success": false, "error": "Missing required fields: color, modulation_type, frequency, amplitude, offset, global_brightness"}),
        )
    }
}
