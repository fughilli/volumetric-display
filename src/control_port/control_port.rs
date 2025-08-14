use crate::WebMonitor;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::time::{interval, timeout};
// use uuid::Uuid;

// Configuration structures
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ControllerConfig {
    pub ip: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub controller_addresses: std::collections::HashMap<String, ControllerConfig>,
}

// Message types for communication with controllers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IncomingMessage {
    Heartbeat,
    Controller { dip: String },
    Button { buttons: Vec<bool> },
}

impl IncomingMessage {
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        // Try to parse as a button message first (most common)
        if let Ok(button_msg) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(buttons) = button_msg.get("buttons") {
                if let Ok(buttons_vec) = serde_json::from_value::<Vec<bool>>(buttons.clone()) {
                    return Ok(IncomingMessage::Button {
                        buttons: buttons_vec,
                    });
                }
            }
        }

        // Try to parse as other message types
        serde_json::from_str(json_str)
    }
}

#[derive(Debug, Clone)]
pub enum OutgoingMessage {
    Noop,
    LcdClear,
    LcdWrite { x: u16, y: u16, text: String },
    Backlight { states: Vec<bool> },
    Led { rgb_values: Vec<(u8, u8, u8)> },
}

impl OutgoingMessage {
    pub fn to_bytes(&self) -> Bytes {
        match self {
            OutgoingMessage::Noop => Bytes::from("noop\n"),
            OutgoingMessage::LcdClear => Bytes::from("lcd:clear\n"),
            OutgoingMessage::LcdWrite { x, y, text } => {
                Bytes::from(format!("lcd:{}:{}:{}\n", x, y, text))
            }
            OutgoingMessage::Backlight { states } => {
                let payload = states
                    .iter()
                    .map(|s| if *s { "1" } else { "0" })
                    .collect::<Vec<_>>()
                    .join(":");
                Bytes::from(format!("backlight:{}\n", payload))
            }
            OutgoingMessage::Led { rgb_values } => {
                let num_leds = rgb_values.len() as u16;
                let mut payload = vec![num_leds as u8, (num_leds >> 8) as u8];
                for (r, g, b) in rgb_values {
                    payload.extend_from_slice(&[*r, *g, *b]);
                }
                let encoded = general_purpose::STANDARD.encode(&payload);
                Bytes::from(format!("led:{}\n", encoded))
            }
        }
    }
}

// Log entry for tracking communication
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: DateTime<Utc>,
    pub direction: LogDirection,
    pub message: String,
    pub raw_data: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LogDirection {
    Incoming,
    Outgoing,
    Error,
    Info,
}

// Controller statistics
#[derive(Debug, Clone, Serialize)]
pub struct ControllerStats {
    pub dip: String,
    pub ip: String,
    pub port: u16,
    pub connected: bool,
    pub last_message_time: Option<DateTime<Utc>>,
    pub connection_time: Option<DateTime<Utc>>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connection_attempts: u64,
    pub last_error: Option<String>,
}

// Controller state management
#[derive(Debug)]
pub struct ControllerState {
    pub dip: String,
    pub config: ControllerConfig,
    pub connected: Arc<RwLock<bool>>,
    pub stats: Arc<RwLock<ControllerStats>>,
    pub log: Arc<RwLock<VecDeque<LogEntry>>>,
    pub bytes_sent: AtomicU64,
    pub bytes_received: AtomicU64,
    pub messages_sent: AtomicU64,
    pub messages_received: AtomicU64,
    pub connection_attempts: AtomicU64,

    // Display buffer management
    pub display_width: u16,
    pub display_height: u16,
    pub front_buffer: Arc<RwLock<Vec<Vec<char>>>>,
    pub back_buffer: Arc<RwLock<Vec<Vec<char>>>>,

    // Communication channels
    pub message_tx: mpsc::UnboundedSender<OutgoingMessage>,
    pub message_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<OutgoingMessage>>>>,
    pub button_broadcast: broadcast::Sender<Vec<bool>>,

    // Internal task handles
    pub connection_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl ControllerState {
    pub fn new(dip: String, config: ControllerConfig) -> Self {
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (button_broadcast, _) = broadcast::channel(100);

        let stats = ControllerStats {
            dip: dip.clone(),
            ip: config.ip.clone(),
            port: config.port,
            connected: false,
            last_message_time: None,
            connection_time: None,
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            connection_attempts: 0,
            last_error: None,
        };

        let width = 20;
        let height = 4;
        let front_buffer = vec![vec![' '; width]; height];
        let back_buffer = vec![vec![' '; width]; height];

        Self {
            dip,
            config,
            connected: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(stats)),
            log: Arc::new(RwLock::new(VecDeque::new())),
            bytes_sent: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            connection_attempts: AtomicU64::new(0),
            display_width: width as u16,
            display_height: height as u16,
            front_buffer: Arc::new(RwLock::new(front_buffer)),
            back_buffer: Arc::new(RwLock::new(back_buffer)),
            message_tx,
            message_rx: Arc::new(RwLock::new(Some(message_rx))),
            button_broadcast,
            connection_task: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn add_log(
        &self,
        direction: LogDirection,
        message: String,
        raw_data: Option<String>,
    ) {
        let entry = LogEntry {
            timestamp: Utc::now(),
            direction,
            message,
            raw_data,
        };

        let mut log = self.log.write().await;
        log.push_back(entry);

        // Keep only last 1000 entries
        while log.len() > 1000 {
            log.pop_front();
        }
    }

    pub async fn update_stats(&self) {
        let mut stats = self.stats.write().await;
        stats.bytes_sent = self.bytes_sent.load(Ordering::Relaxed);
        stats.bytes_received = self.bytes_received.load(Ordering::Relaxed);
        stats.messages_sent = self.messages_sent.load(Ordering::Relaxed);
        stats.messages_received = self.messages_received.load(Ordering::Relaxed);
        stats.connection_attempts = self.connection_attempts.load(Ordering::Relaxed);
        stats.connected = *self.connected.read().await;
        if stats.connected {
            stats.last_message_time = Some(Utc::now());
        }
    }

    pub async fn clear_display(&self) {
        let mut back_buffer = self.back_buffer.write().await;
        for y in 0..self.display_height as usize {
            for x in 0..self.display_width as usize {
                back_buffer[y][x] = ' ';
            }
        }
    }

    pub async fn write_display(&self, x: u16, y: u16, text: &str) {
        if y >= self.display_height || x >= self.display_width {
            return;
        }

        let mut back_buffer = self.back_buffer.write().await;
        let chars: Vec<char> = text.chars().collect();
        let y = y as usize;
        let mut x = x as usize;

        for ch in chars {
            if x >= self.display_width as usize {
                break;
            }
            back_buffer[y][x] = ch;
            x += 1;
        }
    }

    pub async fn commit_display(&self) -> Result<Vec<OutgoingMessage>> {
        let mut messages = Vec::new();
        let front_buffer = self.front_buffer.read().await;
        let back_buffer = self.back_buffer.read().await;

        // Check if back buffer is all spaces - if so, send clear
        let all_spaces = back_buffer
            .iter()
            .all(|row| row.iter().all(|&ch| ch == ' '));

        if all_spaces {
            messages.push(OutgoingMessage::LcdClear);
            drop(front_buffer);
            let mut front_buffer = self.front_buffer.write().await;
            for y in 0..self.display_height as usize {
                for x in 0..self.display_width as usize {
                    front_buffer[y][x] = ' ';
                }
            }
            return Ok(messages);
        }

        // Find differences and send updates
        for y in 0..self.display_height as usize {
            let changes = self.find_contiguous_changes(&front_buffer, &back_buffer, y);
            for (start, end) in changes {
                let text: String = back_buffer[y][start..end].iter().collect();
                messages.push(OutgoingMessage::LcdWrite {
                    x: start as u16,
                    y: y as u16,
                    text,
                });
            }
        }

        // Update front buffer
        drop(front_buffer);
        let mut front_buffer = self.front_buffer.write().await;
        for y in 0..self.display_height as usize {
            for x in 0..self.display_width as usize {
                front_buffer[y][x] = back_buffer[y][x];
            }
        }

        Ok(messages)
    }

    fn find_contiguous_changes(
        &self,
        front_buffer: &[Vec<char>],
        back_buffer: &[Vec<char>],
        y: usize,
    ) -> Vec<(usize, usize)> {
        let mut changes: Vec<(usize, usize)> = Vec::new();
        let mut start = None;
        let mut last_change_end = None;

        for x in 0..self.display_width as usize {
            if front_buffer[y][x] != back_buffer[y][x] {
                if start.is_none() {
                    // If within 3 chars of previous change, extend previous change
                    if let Some(end) = last_change_end {
                        let distance = x - end;
                        if distance <= 3 && !changes.is_empty() {
                            // Ensure we don't go beyond buffer bounds
                            let new_end = (x + 1).min(self.display_width as usize);
                            changes.last_mut().unwrap().1 = new_end;
                            last_change_end = Some(new_end);
                            continue;
                        }
                    }
                    start = Some(x);
                }
            } else if let Some(s) = start {
                changes.push((s, x));
                last_change_end = Some(x);
                start = None;
            }
        }

        if let Some(s) = start {
            changes.push((s, self.display_width as usize));
        }

        changes
    }

    pub async fn send_message(&self, message: OutgoingMessage) -> Result<()> {
        self.message_tx
            .send(message)
            .map_err(|e| anyhow!("Failed to send message: {}", e))?;
        Ok(())
    }
}

// New ControlPortManager that manages multiple ControlPorts
pub struct ControlPortManager {
    pub control_ports: DashMap<String, Arc<ControlPort>>,
    pub config: Config,
    pub web_monitor: Arc<Mutex<Option<Arc<WebMonitor>>>>,
    shutdown_tx: broadcast::Sender<()>,
}

impl ControlPortManager {
    pub fn new(config: Config) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        Self {
            control_ports: DashMap::new(),
            config,
            web_monitor: Arc::new(Mutex::new(None)),
            shutdown_tx,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        for (dip, config) in &self.config.controller_addresses {
            // Use the existing shutdown_tx to create a receiver for this ControlPort
            let shutdown_rx = self.shutdown_tx.subscribe();
            let control_port = Arc::new(ControlPort::new(dip.clone(), config.clone(), shutdown_rx));

            if let Err(e) = control_port.start().await {
                return Err(anyhow!(
                    "Failed to start control port for DIP {}: {}",
                    dip,
                    e
                ));
            }
            self.control_ports.insert(dip.clone(), control_port);
        }
        Ok(())
    }

    pub async fn start_web_monitor(&self, port: u16) -> Result<()> {
        let web_monitor = Arc::new(WebMonitor::new(Arc::new(self.clone())));
        let web_monitor_clone = web_monitor.clone();

        // Start web monitor in background task
        tokio::spawn(async move {
            if let Err(e) = web_monitor_clone.start_server(port).await {
                eprintln!("Web monitor error: {}", e);
            }
        });

        // Use interior mutability to update the web_monitor
        let mut guard = self.web_monitor.lock().await;
        *guard = Some(web_monitor);
        Ok(())
    }

    pub fn get_control_port(&self, dip: &str) -> Option<Arc<ControlPort>> {
        self.control_ports.get(dip).map(|cp| cp.clone())
    }

    pub async fn get_all_stats(&self) -> Vec<ControlPortStats> {
        let mut all_stats = Vec::new();

        for control_port in self.control_ports.iter() {
            let stats = control_port.get_stats().await;
            all_stats.push(stats);
        }

        all_stats
    }

    pub async fn shutdown(&self) {
        // Send shutdown signal to all control ports
        let _ = self.shutdown_tx.send(());

        // Wait for all control ports to shut down
        for control_port in self.control_ports.iter() {
            control_port.shutdown().await;
        }

        // Clear the collection
        self.control_ports.clear();
    }
}

impl Clone for ControlPortManager {
    fn clone(&self) -> Self {
        Self {
            control_ports: self.control_ports.clone(),
            config: self.config.clone(),
            web_monitor: self.web_monitor.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
        }
    }
}

// New ControlPort struct that represents a single controller connection
pub struct ControlPort {
    pub dip: String,
    pub config: ControllerConfig,
    pub state: Arc<RwLock<ControlPortState>>,
    pub stats: Arc<RwLock<ControlPortStats>>,
    pub logs: Arc<RwLock<VecDeque<LogEntry>>>,

    // Communication channels
    pub message_tx: mpsc::UnboundedSender<OutgoingMessage>,
    pub button_broadcast: broadcast::Sender<Vec<bool>>,

    // Internal task handles
    connection_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    button_forward_task: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
    shutdown_rx: broadcast::Receiver<()>,

    // Store reference to the underlying ControllerState
    controller_state: Arc<RwLock<Option<Arc<ControllerState>>>>,
}

#[derive(Debug, Clone)]
pub struct ControlPortState {
    pub connected: bool,
    pub last_message_time: Option<DateTime<Utc>>,
    pub connection_time: Option<DateTime<Utc>>,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPortStats {
    pub dip: String,
    pub ip: String,
    pub port: u16,
    pub connected: bool,
    pub last_message_time: Option<DateTime<Utc>>,
    pub connection_time: Option<DateTime<Utc>>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub connection_attempts: u64,
    pub last_error: Option<String>,
}

impl ControlPort {
    pub fn new(
        dip: String,
        config: ControllerConfig,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> Self {
        let (message_tx, _message_rx) = mpsc::unbounded_channel();
        let (button_broadcast, _) = broadcast::channel(100);

        let state = Arc::new(RwLock::new(ControlPortState {
            connected: false,
            last_message_time: None,
            connection_time: None,
            last_error: None,
        }));

        let stats = Arc::new(RwLock::new(ControlPortStats {
            dip: dip.clone(),
            ip: config.ip.clone(),
            port: config.port,
            connected: false,
            last_message_time: None,
            connection_time: None,
            bytes_sent: 0,
            bytes_received: 0,
            messages_sent: 0,
            messages_received: 0,
            connection_attempts: 0,
            last_error: None,
        }));

        let logs = Arc::new(RwLock::new(VecDeque::new()));

        Self {
            dip,
            config,
            state,
            stats,
            logs,
            message_tx,
            button_broadcast,
            connection_task: Arc::new(RwLock::new(None)),
            button_forward_task: Arc::new(RwLock::new(None)),
            shutdown_rx,
            controller_state: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        // Create a new controller state
        let controller = Arc::new(ControllerState::new(self.dip.clone(), self.config.clone()));

        // Store the controller directly in this ControlPort

        *self.controller_state.write().await = Some(controller.clone());

        // Start the button forwarding task to connect ControllerState button events to ControlPort button broadcast
        let controller_clone = controller.clone();
        let button_broadcast_tx = self.button_broadcast.clone();
        let mut shutdown_rx = self.shutdown_rx.resubscribe();
        let button_forward_task = tokio::spawn(async move {
            // Subscribe to the controller's button broadcast
            let mut button_rx = controller_clone.button_broadcast.subscribe();

            loop {
                tokio::select! {
                    button_event = button_rx.recv() => {
                        match button_event {
                            Ok(buttons) => {
                                // Forward the button event to the ControlPort's button broadcast
                                if let Err(e) = button_broadcast_tx.send(buttons) {
                                    println!(
                                        "[RUST-DEBUG] Failed to forward button event for DIP {}: {:?}",
                                        controller_clone.dip, e
                                    );
                                }
                            }
                            Err(broadcast::error::RecvError::Closed) => {
                                println!(
                                    "[RUST-DEBUG] Controller button broadcast channel closed for DIP {}, stopping forwarding task",
                                    controller_clone.dip
                                );
                                break;
                            }
                            Err(broadcast::error::RecvError::Lagged(n)) => {
                                println!(
                                    "[RUST-DEBUG] Button forwarding task lagged by {} messages for DIP {}, continuing",
                                    n, controller_clone.dip
                                );
                                continue;
                            }
                        }
                    }
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                }
            }
        });

        // Store the button forwarding task handle
        *self.button_forward_task.write().await = Some(button_forward_task);

        // Start the controller task
        let controller_clone = controller.clone();
        let shutdown_rx = self.shutdown_rx.resubscribe();
        let task_handle = tokio::spawn(async move {
            let dip = controller_clone.dip.clone();

            // Add panic handler to see if there are any panics
            std::panic::set_hook(Box::new(|panic_info| {
                println!("[RUST-DEBUG] PANIC in controller task: {:?}", panic_info);
            }));

            Self::run_controller_task(controller_clone, shutdown_rx).await;
        });

        // Store the task handle
        *self.connection_task.write().await = Some(task_handle);

        Ok(())
    }

    async fn run_controller_task(
        controller: Arc<ControllerState>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut reconnect_interval = interval(Duration::from_secs(2));
        let mut heartbeat_interval = interval(Duration::from_secs(1));

        // Attempt initial connection immediately instead of waiting for first tick
        match Self::attempt_connection(&controller).await {
            Ok(_) => {}
            Err(e) => {
                controller
                    .add_log(
                        LogDirection::Error,
                        format!("Initial connection failed: {}", e),
                        None,
                    )
                    .await;
            }
        }

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                _ = reconnect_interval.tick() => {
                    let connected = *controller.connected.read().await;
                    if !connected {
                        match Self::attempt_connection(&controller).await {
                            Ok(_) => {
                            }
                            Err(e) => {
                                controller.add_log(
                                    LogDirection::Error,
                                    format!("Connection failed: {}", e),
                                    None,
                                ).await;
                            }
                        }
                    }
                }
                _ = heartbeat_interval.tick() => {
                    let connected = *controller.connected.read().await;
                    if connected {
                        if let Err(e) = controller.send_message(OutgoingMessage::Noop).await {
                            controller.add_log(
                                LogDirection::Error,
                                format!("Heartbeat failed: {}", e),
                                None,
                            ).await;
                        }
                    }
                }
            }

            // Note: I/O tasks are spawned by attempt_connection, not here
            // This task just manages reconnection attempts and heartbeats
        }
    }

    async fn attempt_connection(controller: &Arc<ControllerState>) -> Result<()> {
        controller
            .connection_attempts
            .fetch_add(1, Ordering::Relaxed);

        let addr = format!("{}:{}", controller.config.ip, controller.config.port);
        let socket_addr: SocketAddr = addr.parse()?;

        controller
            .add_log(
                LogDirection::Info,
                format!("Attempting connection to {}", addr),
                None,
            )
            .await;

        let stream = timeout(Duration::from_secs(2), TcpStream::connect(socket_addr)).await??;

        // TCP connection success is sufficient validation

        // Set connected = true immediately to prevent multiple connection attempts
        *controller.connected.write().await = true;
        let mut stats = controller.stats.write().await;
        stats.last_error = None;
        stats.connection_time = Some(Utc::now());
        drop(stats);

        controller
            .add_log(
                LogDirection::Info,
                "Connection established and validated, spawning I/O task".to_string(),
                None,
            )
            .await;

        // Spawn the I/O handling task with the established connection
        let controller_clone = controller.clone();
        tokio::spawn(Self::handle_connection(controller_clone, stream));

        Ok(())
    }

    async fn handle_connection(controller: Arc<ControllerState>, stream: TcpStream) {
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);

        // Take the message receiver from the controller
        let message_rx = {
            let mut rx_guard = controller.message_rx.write().await;
            rx_guard.take()
        };

        if message_rx.is_none() {
            controller
                .add_log(
                    LogDirection::Error,
                    "Message receiver already taken".to_string(),
                    None,
                )
                .await;
            return;
        }

        let mut message_rx = message_rx.unwrap();

        // Controller is already marked as connected from attempt_connection

        controller
            .add_log(
                LogDirection::Info,
                "I/O task started successfully - controller connected".to_string(),
                None,
            )
            .await;

        loop {
            let mut line = String::new();
            tokio::select! {
                // Handle incoming messages
                result = buf_reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            // Connection closed
                            break;
                        }
                        Ok(_) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                if let Err(e) = Self::process_incoming_message(&controller, line.as_bytes()).await {
                                    controller.add_log(
                                        LogDirection::Error,
                                        format!("Error processing message: {}", e),
                                        Some(line.clone()),
                                    ).await;
                                }
                            }
                        }
                        Err(e) => {
                            println!("[RUST-DEBUG] handle_connection: Read error from DIP {}: {}", controller.dip, e);
                            controller.add_log(
                                LogDirection::Error,
                                format!("Read error: {}", e),
                                None,
                            ).await;
                            break;
                        }
                    }
                }
                // Handle outgoing messages
                Some(message) = message_rx.recv() => {
                    let data = message.to_bytes();

                    if let Err(e) = writer.write_all(&data).await {
                        controller.add_log(
                            LogDirection::Error,
                            format!("Write error: {}", e),
                            None,
                        ).await;
                        break;
                    }

                    controller.bytes_sent.fetch_add(data.len() as u64, Ordering::Relaxed);
                    controller.messages_sent.fetch_add(1, Ordering::Relaxed);

                    controller.add_log(
                        LogDirection::Outgoing,
                        format!("Sent: {:?}", message),
                        Some(String::from_utf8_lossy(&data).to_string()),
                    ).await;
                }
            }
        }

        // Mark as disconnected
        *controller.connected.write().await = false;
        controller
            .add_log(LogDirection::Info, "Connection closed".to_string(), None)
            .await;
    }

    async fn process_incoming_message(
        controller: &Arc<ControllerState>,
        data: &[u8],
    ) -> Result<()> {
        let line = String::from_utf8_lossy(data).trim().to_string();
        if line.is_empty() {
            return Ok(());
        }

        controller
            .bytes_received
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        controller.messages_received.fetch_add(1, Ordering::Relaxed);

        match IncomingMessage::from_json(&line) {
            Ok(message) => {
                controller
                    .add_log(
                        LogDirection::Incoming,
                        format!("Received: {:?}", message),
                        Some(line.clone()),
                    )
                    .await;

                match message {
                    IncomingMessage::Heartbeat => {
                        // Respond with noop
                        controller.send_message(OutgoingMessage::Noop).await?;
                    }
                    IncomingMessage::Controller { dip } => {
                        controller
                            .add_log(
                                LogDirection::Info,
                                format!("Controller identified with DIP: {}", dip),
                                None,
                            )
                            .await;
                        // Update DIP if different
                        if controller.dip != dip {
                            controller
                                .add_log(
                                    LogDirection::Info,
                                    format!(
                                        "DIP mismatch: expected {}, got {}",
                                        controller.dip, dip
                                    ),
                                    None,
                                )
                                .await;
                        }
                    }
                    IncomingMessage::Button { buttons } => {
                        // Broadcast button state
                        if let Err(e) = controller.button_broadcast.send(buttons) {
                            println!(
                                "[RUST-DEBUG] Button broadcast failed for DIP {}: {:?}",
                                controller.dip, e
                            )
                        }
                    }
                }
            }
            Err(e) => {
                println!(
                    "[RUST-DEBUG] Failed to parse message from DIP {}: '{}' -> error: {}",
                    controller.dip, line, e
                );
                controller
                    .add_log(
                        LogDirection::Error,
                        format!("Failed to parse message: {}", e),
                        Some(line.clone()),
                    )
                    .await;
            }
        }

        Ok(())
    }

    pub async fn get_stats(&self) -> ControlPortStats {
        self.stats.read().await.clone()
    }

    pub async fn shutdown(&self) {
        // Cancel connection task
        if let Some(task) = self.connection_task.write().await.take() {
            task.abort();
        }

        // Cancel button forwarding task
        if let Some(task) = self.button_forward_task.write().await.take() {
            task.abort();
        }

        // Update state
        let mut state = self.state.write().await;
        state.connected = false;
        state.last_error = Some("Shutdown".to_string());
    }

    // Delegate methods to the underlying ControllerState
    pub async fn clear_display(&self) {
        if let Some(controller) = self.get_controller_state().await {
            controller.clear_display().await;
        }
    }

    pub async fn write_display(&self, x: u16, y: u16, text: &str) {
        if let Some(controller) = self.get_controller_state().await {
            controller.write_display(x, y, text).await;
        }
    }

    pub async fn commit_display(&self) -> Result<(), String> {
        if let Some(controller) = self.get_controller_state().await {
            match controller.commit_display().await {
                Ok(messages) => {
                    for message in messages {
                        if let Err(e) = self.send_message(message).await {
                            println!("[RUST-DEBUG] ControlPort::commit_display: Failed to send message for DIP {}: {}", self.dip, e);
                        }
                    }
                    Ok(())
                }
                Err(e) => Err(format!(
                    "[RUST-DEBUG] commit_display: Error committing display for DIP {}: {}",
                    controller.dip, e
                )),
            }
        } else {
            Err(format!(
                "[RUST-DEBUG] commit_display: No controller state found"
            ))
        }
    }

    pub async fn set_leds(&self, rgb_values: Vec<(u8, u8, u8)>) {
        if let Some(controller) = self.get_controller_state().await {
            let _ = controller
                .send_message(OutgoingMessage::Led { rgb_values })
                .await;
        }
    }

    pub async fn set_backlights(&self, states: Vec<bool>) {
        if let Some(controller) = self.get_controller_state().await {
            let _ = controller
                .send_message(OutgoingMessage::Backlight { states })
                .await;
        }
    }

    pub async fn get_controller_state(&self) -> Option<Arc<ControllerState>> {
        self.controller_state.read().await.as_ref().cloned()
    }

    pub async fn send_message(&self, message: OutgoingMessage) -> Result<()> {
        if let Some(controller) = self.get_controller_state().await {
            controller.send_message(message).await
        } else {
            Err(anyhow!("No controller state available"))
        }
    }

    // Update ControlPortState to match ControllerState
    async fn update_connection_state(&self, connected: bool) {
        let mut state = self.state.write().await;
        state.connected = connected;
        if connected {
            state.connection_time = Some(Utc::now());
            state.last_error = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    fn create_test_controller_state() -> ControllerState {
        let config = ControllerConfig {
            ip: "127.0.0.1".to_string(),
            port: 1234,
        };
        ControllerState::new("test_dip".to_string(), config)
    }

    #[tokio::test]
    async fn test_lcd_commit_on_empty_buffer_causes_clear_command() {
        let controller = create_test_controller_state();

        // Clear the display (this sets back buffer to all spaces)
        controller.clear_display().await;

        // Commit should send clear command
        let messages = controller.commit_display().await.unwrap();

        assert_eq!(messages.len(), 1);
        match &messages[0] {
            OutgoingMessage::LcdClear => {
                // Expected
            }
            _ => panic!("Expected LcdClear message, got {:?}", messages[0]),
        }
    }

    #[tokio::test]
    async fn test_lcd_commit_with_changes_causes_lcd_command_to_be_sent() {
        let controller = create_test_controller_state();

        // Write text to display
        controller.write_display(0, 0, "Hello, world!").await;

        // Commit should send LCD write command
        let messages = controller.commit_display().await.unwrap();

        assert_eq!(messages.len(), 1);
        match &messages[0] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages[0]),
        }
    }

    #[tokio::test]
    async fn test_lcd_commit_with_minimal_change_causes_correct_command_sequence() {
        let controller = create_test_controller_state();

        // First commit: write "Hello, world!"
        controller.write_display(0, 0, "Hello, world!").await;
        let messages1 = controller.commit_display().await.unwrap();

        assert_eq!(messages1.len(), 1);
        match &messages1[0] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(text, "Hello, world!");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages1[0]),
        }

        // Second commit: change to "Hello, there!"
        controller.write_display(0, 0, "Hello, there!").await;
        let messages2 = controller.commit_display().await.unwrap();

        // Should only send the changed part: "there" starting at position 7
        assert_eq!(messages2.len(), 1);
        match &messages2[0] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 7);
                assert_eq!(*y, 0);
                assert_eq!(text, "there");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages2[0]),
        }
    }

    #[tokio::test]
    async fn test_lcd_commit_with_multiple_changes_causes_correct_command_sequence() {
        let controller = create_test_controller_state();

        // First commit: write two lines
        controller.write_display(0, 0, "ABCDEFGH").await;
        controller.write_display(0, 1, "IJKLMNOP").await;
        let messages1 = controller.commit_display().await.unwrap();

        assert_eq!(messages1.len(), 2);

        // Check first line
        match &messages1[0] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 0);
                assert_eq!(text, "ABCDEFGH");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages1[0]),
        }

        // Check second line
        match &messages1[1] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 1);
                assert_eq!(text, "IJKLMNOP");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages1[1]),
        }

        // Second commit: make minimal changes
        controller.write_display(0, 0, "ABCDEFGG").await; // Change H to G at position 7
        controller.write_display(0, 1, "JJKLMNOP").await; // Change I to J at position 0
        let messages2 = controller.commit_display().await.unwrap();

        // Should send only the changed parts
        assert_eq!(messages2.len(), 2);

        // Check first change: G at position 7
        match &messages2[0] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 7);
                assert_eq!(*y, 0);
                assert_eq!(text, "G");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages2[0]),
        }

        // Check second change: J at position 0
        match &messages2[1] {
            OutgoingMessage::LcdWrite { x, y, text } => {
                assert_eq!(*x, 0);
                assert_eq!(*y, 1);
                assert_eq!(text, "J");
            }
            _ => panic!("Expected LcdWrite message, got {:?}", messages2[1]),
        }
    }

    #[tokio::test]
    async fn test_find_contiguous_changes_logic() {
        let controller = create_test_controller_state();

        // Test the find_contiguous_changes method directly
        // Create buffers with the correct size (20x4 as per controller default)
        let mut front_buffer = vec![vec![' '; 20]; 4];
        let mut back_buffer = vec![vec![' '; 20]; 4];

        // Set up test data in first two rows
        front_buffer[0][0..8].copy_from_slice(&['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
        front_buffer[1][0..8].copy_from_slice(&['I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']);

        back_buffer[0][0..8].copy_from_slice(&['A', 'B', 'X', 'Y', 'E', 'F', 'G', 'H']); // Changes at positions 2-3
        back_buffer[1][0..8].copy_from_slice(&['I', 'J', 'K', 'L', 'M', 'N', 'O', 'Q']); // Change at position 7

        // Test first row changes
        let changes_row0 = controller.find_contiguous_changes(&front_buffer, &back_buffer, 0);
        assert_eq!(changes_row0, vec![(2, 4)]); // Positions 2-3 (exclusive end)

        // Test second row changes
        let changes_row1 = controller.find_contiguous_changes(&front_buffer, &back_buffer, 1);
        assert_eq!(changes_row1, vec![(7, 8)]); // Position 7 only
    }

    #[tokio::test]
    async fn test_contiguous_changes_with_gaps() {
        let controller = create_test_controller_state();

        // Test changes that are close enough to be merged (within 3 chars)
        // Create buffers with the correct size (20x4 as per controller default)
        let mut front_buffer = vec![vec![' '; 20]; 4];
        let mut back_buffer = vec![vec![' '; 20]; 4];

        // Set up test data in first row
        front_buffer[0][0..8].copy_from_slice(&['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']);
        back_buffer[0][0..8].copy_from_slice(&['A', 'X', 'C', 'D', 'E', 'Y', 'G', 'H']); // Changes at positions 1 and 5

        // Since positions 1 and 5 are 4 chars apart, but the algorithm merges changes
        // that are within 3 chars of the end of the previous change
        // End of change 1: position 2, start of change 2: position 5
        // Distance: 5 - 2 = 3, which is <= 3, so they get merged
        let changes = controller.find_contiguous_changes(&front_buffer, &back_buffer, 0);
        assert_eq!(changes, vec![(1, 6)]); // Merged into one change

        // Test changes that are close enough to merge (within 3 chars)
        let mut back_buffer2 = vec![vec![' '; 20]; 4];
        back_buffer2[0][0..8].copy_from_slice(&['A', 'X', 'C', 'Y', 'E', 'F', 'G', 'H']); // Changes at positions 1 and 3

        // Since positions 1 and 3 are 2 chars apart (≤ 3), they should be merged
        let changes2 = controller.find_contiguous_changes(&front_buffer, &back_buffer2, 0);
        assert_eq!(changes2, vec![(1, 4)]); // Merged into one change
    }

    #[tokio::test]
    async fn test_display_buffer_management() {
        let controller = create_test_controller_state();

        // Test that writing to display updates back buffer correctly
        controller.write_display(5, 2, "TEST").await;

        let back_buffer = controller.back_buffer.read().await;
        assert_eq!(back_buffer[2][5], 'T');
        assert_eq!(back_buffer[2][6], 'E');
        assert_eq!(back_buffer[2][7], 'S');
        assert_eq!(back_buffer[2][8], 'T');

        // Test that front buffer is unchanged until commit
        let front_buffer = controller.front_buffer.read().await;
        assert_eq!(front_buffer[2][5], ' '); // Still space

        drop(back_buffer);
        drop(front_buffer);

        // Commit should update front buffer
        let _ = controller.commit_display().await.unwrap();

        let front_buffer = controller.front_buffer.read().await;
        assert_eq!(front_buffer[2][5], 'T');
        assert_eq!(front_buffer[2][6], 'E');
        assert_eq!(front_buffer[2][7], 'S');
        assert_eq!(front_buffer[2][8], 'T');
    }

    #[tokio::test]
    async fn test_outgoing_message_serialization() {
        // Test that OutgoingMessage serializes to the expected format
        let clear_msg = OutgoingMessage::LcdClear;
        assert_eq!(clear_msg.to_bytes(), Bytes::from("lcd:clear\n"));

        let write_msg = OutgoingMessage::LcdWrite {
            x: 5,
            y: 2,
            text: "TEST".to_string(),
        };
        assert_eq!(write_msg.to_bytes(), Bytes::from("lcd:5:2:TEST\n"));

        let noop_msg = OutgoingMessage::Noop;
        assert_eq!(noop_msg.to_bytes(), Bytes::from("noop\n"));
    }
}
