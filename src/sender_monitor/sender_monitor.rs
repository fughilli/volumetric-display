use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerStatus {
    pub ip: String,
    pub port: u16,
    pub is_routable: bool,
    pub last_success: Option<DateTime<Utc>>,
    pub last_failure: Option<DateTime<Utc>>,
    pub failure_count: u64,
    pub last_error: Option<String>,
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

pub struct SenderMonitor {
    controllers: DashMap<String, ControllerStatus>,
    system_stats: Arc<RwLock<SystemStats>>,
    start_time: DateTime<Utc>,
    frame_counter: AtomicU64,
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
        }
    }

    pub fn register_controller(&self, ip: String, port: u16) {
        let status = ControllerStatus {
            ip: ip.clone(),
            port,
            is_routable: true,
            last_success: Some(Utc::now()),
            last_failure: None,
            failure_count: 0,
            last_error: None,
        };
        self.controllers.insert(ip, status);
    }

    pub fn report_controller_success(&self, ip: &str) {
        if let Some(mut status) = self.controllers.get_mut(ip) {
            status.is_routable = true;
            status.last_success = Some(Utc::now());
            status.last_error = None;
        }
    }

    pub fn report_controller_failure(&self, ip: &str, error: &str) {
        if let Some(mut status) = self.controllers.get_mut(ip) {
            status.is_routable = false;
            status.last_failure = Some(Utc::now());
            status.failure_count += 1;
            status.last_error = Some(error.to_string());
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

    pub async fn get_stats(&self) -> SenderMonitorStats {
        self.update_system_stats().await;

        let controllers: Vec<ControllerStatus> = self
            .controllers
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

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
}

impl Default for SenderMonitor {
    fn default() -> Self {
        Self::new()
    }
}
