use chrono::{DateTime, Duration, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

// Helper function to parse IP address for proper sorting
fn parse_ip_for_sorting(ip: &str) -> Vec<u8> {
    ip.split('.')
        .filter_map(|octet| octet.parse::<u8>().ok())
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerStatus {
    pub ip: String,
    pub port: u16,
    pub is_routable: bool,
    pub is_connecting: bool, // True when in cooldown period
    pub last_success: Option<DateTime<Utc>>,
    pub last_failure: Option<DateTime<Utc>>,
    pub failure_count: u64,
    pub last_error: Option<String>,
    pub cooldown_until: Option<DateTime<Utc>>, // Cooldown period after failure
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub fps: f64,
    pub uptime_seconds: f64,
    pub total_frames: u64,
    pub last_update: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenderMonitorStats {
    pub controllers: Vec<ControllerStatus>,
    pub system: SystemStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugState {
    pub is_debug_mode: bool,
    pub is_paused: bool,
    pub current_debug_command: Option<String>,
    pub debug_data: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappingTesterCommand {
    pub orientation: String, // "xy", "xz", "yz"
    pub layer: usize,
    pub color: String,  // hex color like "#FF0000"
    pub target: String, // "world" or "cube_0", "cube_1", etc.
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerDrawTesterCommand {
    pub color: String,           // hex color like "#FF0000"
    pub modulation_type: String, // "sin" or "square"
    pub frequency: f64,
    pub amplitude: f64,
    pub offset: f64,
    pub global_brightness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CubeInfo {
    pub id: String,
    pub position: (usize, usize, usize),
    pub dimensions: (usize, usize, usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugCommand {
    pub command_type: String, // "mapping_tester" or "power_draw_tester"
    pub mapping_tester: Option<MappingTesterCommand>,
    pub power_draw_tester: Option<PowerDrawTesterCommand>,
}

pub struct SenderMonitor {
    controllers: DashMap<String, ControllerStatus>,
    system_stats: Arc<RwLock<SystemStats>>,
    start_time: DateTime<Utc>,
    frame_counter: AtomicU64,
    cooldown_duration: Arc<RwLock<Duration>>, // Duration of cooldown period
    debug_state: Arc<RwLock<DebugState>>,
    debug_command: Arc<RwLock<Option<DebugCommand>>>,
    world_dimensions: Arc<RwLock<Option<(usize, usize, usize)>>>, // (width, height, length)
    cube_list: Arc<RwLock<Vec<CubeInfo>>>,                        // List of available cubes
}

impl SenderMonitor {
    pub fn new() -> Self {
        Self {
            controllers: DashMap::new(),
            system_stats: Arc::new(RwLock::new(SystemStats {
                fps: 0.0,
                uptime_seconds: 0.0,
                total_frames: 0,
                last_update: Utc::now(),
            })),
            start_time: Utc::now(),
            frame_counter: AtomicU64::new(0),
            cooldown_duration: Arc::new(RwLock::new(Duration::seconds(30))), // 30 second cooldown by default
            debug_state: Arc::new(RwLock::new(DebugState {
                is_debug_mode: false,
                is_paused: false,
                current_debug_command: None,
                debug_data: serde_json::json!({}),
            })),
            debug_command: Arc::new(RwLock::new(None)),
            world_dimensions: Arc::new(RwLock::new(None)),
            cube_list: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn with_cooldown_duration(mut self, cooldown_seconds: i64) -> Self {
        self.cooldown_duration = Arc::new(RwLock::new(Duration::seconds(cooldown_seconds)));
        self
    }

    pub async fn set_cooldown_duration(&self, cooldown_seconds: i64) {
        let mut duration = self.cooldown_duration.write().await;
        *duration = Duration::seconds(cooldown_seconds);
    }

    pub fn register_controller(&self, ip: String, port: u16) {
        let status = ControllerStatus {
            ip: ip.clone(),
            port,
            is_routable: true,
            is_connecting: false,
            last_success: Some(Utc::now()),
            last_failure: None,
            failure_count: 0,
            last_error: None,
            cooldown_until: None,
        };
        // Use composite key of IP:port to uniquely identify controllers
        let key = format!("{}:{}", ip, port);
        self.controllers.insert(key, status);
    }

    pub async fn report_controller_success(&self, ip: &str, port: u16) {
        let key = format!("{}:{}", ip, port);
        if let Some(mut status) = self.controllers.get_mut(&key) {
            let now = Utc::now();
            status.last_success = Some(now);

            // Check if we're still in cooldown period
            if let Some(cooldown_until) = status.cooldown_until {
                if now < cooldown_until {
                    // Still in cooldown - record success but don't change status
                    // Controller remains "Connecting..." until cooldown expires
                    return;
                }
            }

            // Out of cooldown and no recent failures - mark as connected
            status.is_routable = true;
            status.is_connecting = false;
            status.last_error = None;
            status.cooldown_until = None; // Clear cooldown
        }
    }

    pub async fn report_controller_failure(&self, ip: &str, port: u16, error: &str) {
        let key = format!("{}:{}", ip, port);
        if let Some(mut status) = self.controllers.get_mut(&key) {
            let now = Utc::now();
            status.is_routable = false;
            status.is_connecting = true; // Enter connecting state
            status.last_failure = Some(now);
            status.failure_count += 1;
            status.last_error = Some(error.to_string());

            // Set cooldown period - controller must be error-free for this duration
            let cooldown_duration = self.cooldown_duration.read().await;
            status.cooldown_until = Some(now + *cooldown_duration);
        }
    }

    pub fn report_frame(&self) {
        self.frame_counter.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn update_system_stats(&self) {
        let total_frames = self.frame_counter.load(Ordering::Relaxed);
        let now = Utc::now();
        let uptime = (now - self.start_time).num_milliseconds() as f64 / 1000.0;

        // Calculate FPS over the last second
        let fps = if uptime > 0.0 {
            total_frames as f64 / uptime
        } else {
            0.0
        };

        let mut stats = self.system_stats.write().await;
        stats.fps = fps;
        stats.uptime_seconds = uptime;
        stats.total_frames = total_frames;
        stats.last_update = now;
    }

    pub async fn update_controller_statuses(&self) {
        // Check for controllers that have completed their cooldown period
        let now = Utc::now();

        for mut status in self.controllers.iter_mut() {
            if let Some(cooldown_until) = status.cooldown_until {
                if now >= cooldown_until && status.is_connecting {
                    // Cooldown expired and no failures occurred during cooldown
                    // Transition from "Connecting..." to "Connected"
                    status.is_routable = true;
                    status.is_connecting = false;
                    status.cooldown_until = None;
                }
            }
        }
    }

    pub async fn get_stats(&self) -> SenderMonitorStats {
        self.update_system_stats().await;
        self.update_controller_statuses().await; // Update controller statuses

        let mut controllers: Vec<ControllerStatus> = self
            .controllers
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        // Sort controllers by IP address for consistent ordering
        controllers.sort_by(|a, b| parse_ip_for_sorting(&a.ip).cmp(&parse_ip_for_sorting(&b.ip)));

        let system = self.system_stats.read().await.clone();

        SenderMonitorStats {
            controllers,
            system,
        }
    }

    pub fn get_controller_count(&self) -> usize {
        self.controllers.len()
    }

    pub fn get_routable_controller_count(&self) -> usize {
        self.controllers
            .iter()
            .filter(|entry| entry.value().is_routable)
            .count()
    }

    pub async fn set_debug_mode(&self, enabled: bool) {
        let mut debug_state = self.debug_state.write().await;
        debug_state.is_debug_mode = enabled;
        if !enabled {
            debug_state.is_paused = false;
            debug_state.current_debug_command = None;
            debug_state.debug_data = serde_json::json!({});
        }
    }

    pub async fn set_debug_pause(&self, paused: bool) {
        let mut debug_state = self.debug_state.write().await;
        debug_state.is_paused = paused;
    }

    pub async fn set_debug_command(&self, command: DebugCommand) {
        let mut debug_cmd = self.debug_command.write().await;
        *debug_cmd = Some(command.clone());

        let mut debug_state = self.debug_state.write().await;
        debug_state.current_debug_command = Some(command.command_type.clone());

        match command.command_type.as_str() {
            "mapping_tester" => {
                if let Some(mt) = &command.mapping_tester {
                    debug_state.debug_data = serde_json::json!({
                        "orientation": mt.orientation.clone(),
                        "layer": mt.layer,
                        "color": mt.color.clone()
                    });
                }
            }
            "power_draw_tester" => {
                if let Some(pdt) = &command.power_draw_tester {
                    debug_state.debug_data = serde_json::json!({
                        "color": pdt.color.clone(),
                        "modulation_type": pdt.modulation_type.clone(),
                        "frequency": pdt.frequency,
                        "amplitude": pdt.amplitude,
                        "offset": pdt.offset,
                        "global_brightness": pdt.global_brightness
                    });
                }
            }
            _ => {}
        }
    }

    pub async fn get_debug_state(&self) -> DebugState {
        self.debug_state.read().await.clone()
    }

    pub async fn get_debug_command(&self) -> Option<DebugCommand> {
        self.debug_command.read().await.clone()
    }

    pub async fn is_debug_mode(&self) -> bool {
        self.debug_state.read().await.is_debug_mode
    }

    pub async fn is_paused(&self) -> bool {
        self.debug_state.read().await.is_paused
    }

    pub async fn set_world_dimensions(&self, width: usize, height: usize, length: usize) {
        let mut dimensions = self.world_dimensions.write().await;
        *dimensions = Some((width, height, length));
    }

    pub async fn get_world_dimensions(&self) -> Option<(usize, usize, usize)> {
        self.world_dimensions.read().await.clone()
    }

    pub async fn set_cube_list(&self, cubes: Vec<CubeInfo>) {
        let mut cube_list = self.cube_list.write().await;
        *cube_list = cubes;
    }

    pub async fn get_cube_list(&self) -> Vec<CubeInfo> {
        self.cube_list.read().await.clone()
    }
}

impl Default for SenderMonitor {
    fn default() -> Self {
        Self::new()
    }
}
