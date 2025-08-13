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
#[serde(tag = "type")]
pub enum IncomingMessage {
    #[serde(rename = "heartbeat")]
    Heartbeat,
    #[serde(rename = "controller")]
    Controller { dip: String },
    #[serde(rename = "button")]
    Button { buttons: Vec<bool> },
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
                        if x - end <= 3 && !changes.is_empty() {
                            changes.last_mut().unwrap().1 = x + 1;
                            last_change_end = Some(x + 1);
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
        println!("[RUST-DEBUG] Initializing ControlPortManager");

        for (dip, config) in &self.config.controller_addresses {
            println!(
                "[RUST-DEBUG] Creating ControlPort for DIP {} at {}:{}",
                dip, config.ip, config.port
            );

            // Use the existing shutdown_tx to create a receiver for this ControlPort
            let shutdown_rx = self.shutdown_tx.subscribe();
            let control_port = Arc::new(ControlPort::new(dip.clone(), config.clone(), shutdown_rx));

            println!("[RUST-DEBUG] Starting ControlPort for DIP {}", dip);
            if let Err(e) = control_port.start().await {
                println!(
                    "[RUST-DEBUG] Failed to start ControlPort for DIP {}: {}",
                    dip, e
                );
                return Err(anyhow!(
                    "Failed to start control port for DIP {}: {}",
                    dip,
                    e
                ));
            }

            println!(
                "[RUST-DEBUG] ControlPort started successfully for DIP {}",
                dip
            );
            self.control_ports.insert(dip.clone(), control_port);
        }

        println!(
            "[RUST-DEBUG] ControlPortManager initialized successfully with {} controllers",
            self.control_ports.len()
        );
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
            shutdown_rx,
            controller_state: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn start(&self) -> Result<()> {
        println!(
            "[RUST-DEBUG] ControlPort::start: Starting ControlPort for DIP {}",
            self.dip
        );

        // Create a new controller state
        let controller = Arc::new(ControllerState::new(self.dip.clone(), self.config.clone()));
        println!(
            "[RUST-DEBUG] ControlPort::start: Created ControllerState for DIP {}",
            self.dip
        );

        // Store the controller directly in this ControlPort
        println!(
            "[RUST-DEBUG] ControlPort::start: Storing controller for DIP {}",
            self.dip
        );
        *self.controller_state.write().await = Some(controller.clone());

        // Start the controller task
        println!(
            "[RUST-DEBUG] ControlPort::start: About to spawn controller task for DIP {}",
            self.dip
        );
        let controller_clone = controller.clone();
        let shutdown_rx = self.shutdown_rx.resubscribe();
        let task_handle = tokio::spawn(async move {
            println!(
                "[RUST-DEBUG] TASK STARTED: Controller task beginning execution for DIP {}",
                controller_clone.dip
            );
            let dip = controller_clone.dip.clone();
            println!(
                "[RUST-DEBUG] ControlPort::start: Controller task spawned for DIP {}",
                dip
            );
            println!("[RUST-DEBUG] ControlPort::start: Got shutdown receiver, calling run_controller_task for DIP {}", dip);

            // Add panic handler to see if there are any panics
            std::panic::set_hook(Box::new(|panic_info| {
                println!("[RUST-DEBUG] PANIC in controller task: {:?}", panic_info);
            }));

            Self::run_controller_task(controller_clone, shutdown_rx).await;
            println!(
                "[RUST-DEBUG] ControlPort::start: run_controller_task completed for DIP {}",
                dip
            );
        });

        println!(
            "[RUST-DEBUG] ControlPort::start: Task spawned successfully for DIP {}, handle: {:?}",
            self.dip,
            task_handle.id()
        );

        println!(
            "[RUST-DEBUG] ControlPort::start: ControlPort started successfully for DIP {}",
            self.dip
        );
        Ok(())
    }

    async fn run_controller_task(
        controller: Arc<ControllerState>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        println!(
            "[RUST-DEBUG] Controller task started for DIP {}",
            controller.dip
        );

        let mut reconnect_interval = interval(Duration::from_secs(2));
        let mut heartbeat_interval = interval(Duration::from_secs(1));

        // Attempt initial connection immediately instead of waiting for first tick
        println!(
            "[RUST-DEBUG] run_controller_task: Attempting initial connection for DIP {}",
            controller.dip
        );
        match Self::attempt_connection(&controller).await {
            Ok(_) => {
                println!("[RUST-DEBUG] run_controller_task: Initial connection attempt succeeded for DIP {}", controller.dip);
            }
            Err(e) => {
                println!(
                    "[RUST-DEBUG] run_controller_task: Initial connection failed for DIP {}: {}",
                    controller.dip, e
                );
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
                    println!("[RUST-DEBUG] Controller task shutting down for DIP {}", controller.dip);
                    break;
                }
                _ = reconnect_interval.tick() => {
                    let connected = *controller.connected.read().await;
                    println!("[RUST-DEBUG] run_controller_task: Reconnect tick for DIP {}: connected={}", controller.dip, connected);
                    if !connected {
                        println!("[RUST-DEBUG] run_controller_task: Attempting connection for DIP {}", controller.dip);
                        match Self::attempt_connection(&controller).await {
                            Ok(_) => {
                                println!("[RUST-DEBUG] run_controller_task: Reconnection attempt succeeded for DIP {}", controller.dip);
                            }
                            Err(e) => {
                                println!("[RUST-DEBUG] run_controller_task: Connection failed for DIP {}: {}", controller.dip, e);
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
                        println!("[RUST-DEBUG] Sending heartbeat to DIP {}", controller.dip);
                        if let Err(e) = controller.send_message(OutgoingMessage::Noop).await {
                            println!("[RUST-DEBUG] Heartbeat failed for DIP {}: {}", controller.dip, e);
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
        println!(
            "[RUST-DEBUG] attempt_connection: Starting connection attempt for DIP {}",
            controller.dip
        );

        controller
            .connection_attempts
            .fetch_add(1, Ordering::Relaxed);

        let addr = format!("{}:{}", controller.config.ip, controller.config.port);
        let socket_addr: SocketAddr = addr.parse()?;

        println!(
            "[RUST-DEBUG] attempt_connection: Attempting connection to {} for DIP {}",
            addr, controller.dip
        );

        controller
            .add_log(
                LogDirection::Info,
                format!("Attempting connection to {}", addr),
                None,
            )
            .await;

        println!(
            "[RUST-DEBUG] attempt_connection: About to call TcpStream::connect to {}",
            addr
        );
        let stream = timeout(Duration::from_secs(2), TcpStream::connect(socket_addr)).await??;

        println!(
            "[RUST-DEBUG] attempt_connection: TCP connection established to {} for DIP {}",
            addr, controller.dip
        );

        // Validate the connection by sending a test message and waiting for a response
        println!(
            "[RUST-DEBUG] attempt_connection: About to validate connection for DIP {}",
            controller.dip
        );
        if let Err(e) = Self::validate_connection(&stream).await {
            println!(
                "[RUST-DEBUG] attempt_connection: Connection validation failed for DIP {}: {}",
                controller.dip, e
            );
            controller
                .add_log(
                    LogDirection::Error,
                    format!("Connection validation failed: {}", e),
                    None,
                )
                .await;
            return Err(e);
        }

        println!(
            "[RUST-DEBUG] attempt_connection: Connection validated successfully for DIP {}",
            controller.dip
        );

        // Don't set connected = true here - let the I/O task set it when it actually starts
        // This ensures we only report as connected when there's actually a working connection
        let mut stats = controller.stats.write().await;
        stats.last_error = None;
        drop(stats);

        println!(
            "[RUST-DEBUG] attempt_connection: Connection established and validated for DIP {}, spawning I/O task",
            controller.dip
        );

        controller
            .add_log(
                LogDirection::Info,
                "Connection established and validated, spawning I/O task".to_string(),
                None,
            )
            .await;

        // Spawn the I/O handling task with the established connection
        let controller_clone = controller.clone();
        println!(
            "[RUST-DEBUG] attempt_connection: Spawning I/O task for DIP {} with established connection",
            controller.dip
        );
        tokio::spawn(Self::handle_connection(controller_clone, stream));

        println!(
            "[RUST-DEBUG] attempt_connection: I/O task spawned successfully for DIP {}",
            controller.dip
        );
        Ok(())
    }

    async fn validate_connection(stream: &TcpStream) -> Result<()> {
        println!("[RUST-DEBUG] validate_connection: Starting validation");

        // Set a short timeout for validation
        stream.set_nodelay(true)?;
        println!("[RUST-DEBUG] validate_connection: Set nodelay successfully");

        // Actually test the connection by trying to send a small amount of data
        // and then checking if we can read a response (or at least if the socket is readable)
        // This will fail if the connection is not actually working or if there's no server
        println!("[RUST-DEBUG] validate_connection: About to check if stream is writable");
        match stream.writable().await {
            Ok(_) => {
                println!(
                    "[RUST-DEBUG] validate_connection: Stream is writable, attempting write test"
                );
                // Try to send a single byte to test the connection
                // This is a more reliable way to test if the connection is actually working
                let test_data = [0u8; 1];
                match stream.try_write(&test_data) {
                    Ok(_) => {
                        println!("[RUST-DEBUG] validate_connection: Write test succeeded, checking if readable");
                        // Now check if the socket is readable - this will fail if there's no server
                        // or if the connection is broken
                        match stream.readable().await {
                            Ok(_) => {
                                println!("[RUST-DEBUG] validate_connection: Stream is readable, validation successful");
                                // Connection appears to be working
                                Ok(())
                            }
                            Err(e) => {
                                println!(
                                    "[RUST-DEBUG] validate_connection: Stream not readable: {}",
                                    e
                                );
                                // Socket is not readable, connection is broken
                                Err(anyhow::anyhow!(
                                    "Connection validation failed - socket not readable: {}",
                                    e
                                ))
                            }
                        }
                    }
                    Err(e) => {
                        println!("[RUST-DEBUG] validate_connection: Write test failed: {}", e);
                        // Connection failed the write test
                        Err(anyhow::anyhow!(
                            "Connection validation failed - write test failed: {}",
                            e
                        ))
                    }
                }
            }
            Err(e) => {
                println!(
                    "[RUST-DEBUG] validate_connection: Stream not writable: {}",
                    e
                );
                // Stream is not writable
                Err(anyhow::anyhow!(
                    "Connection validation failed - stream not writable: {}",
                    e
                ))
            }
        }
    }

    async fn handle_connection(controller: Arc<ControllerState>, stream: TcpStream) {
        println!(
            "[RUST-DEBUG] handle_connection: I/O task started for DIP {}",
            controller.dip
        );

        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        println!(
            "[RUST-DEBUG] handle_connection: Stream split into reader/writer for DIP {}",
            controller.dip
        );

        // Take the message receiver from the controller
        let message_rx = {
            let mut rx_guard = controller.message_rx.write().await;
            rx_guard.take()
        };

        if message_rx.is_none() {
            println!(
                "[RUST-DEBUG] handle_connection: ERROR: Message receiver already taken for DIP {}",
                controller.dip
            );
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
        println!(
            "[RUST-DEBUG] handle_connection: Message receiver acquired for DIP {}",
            controller.dip
        );

        // Now that we have successfully started the I/O task and can communicate,
        // mark the controller as connected
        println!(
            "[RUST-DEBUG] handle_connection: Setting connected = true for DIP {}",
            controller.dip
        );
        *controller.connected.write().await = true;
        let mut stats = controller.stats.write().await;
        stats.connection_time = Some(Utc::now());
        stats.last_error = None;
        drop(stats);

        println!(
            "[RUST-DEBUG] handle_connection: Controller DIP {} marked as connected - I/O task running",
            controller.dip
        );

        controller
            .add_log(
                LogDirection::Info,
                "I/O task started successfully - controller connected".to_string(),
                None,
            )
            .await;

        println!(
            "[RUST-DEBUG] handle_connection: Starting main I/O loop for DIP {}",
            controller.dip
        );
        loop {
            let mut line = String::new();
            tokio::select! {
                // Handle incoming messages
                result = buf_reader.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            // Connection closed
                            println!("[RUST-DEBUG] handle_connection: Connection closed by peer for DIP {}", controller.dip);
                            break;
                        }
                        Ok(_) => {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                println!("[RUST-DEBUG] handle_connection: Received raw data from DIP {}: '{}'", controller.dip, trimmed);
                                if let Err(e) = Self::process_incoming_message(&controller, line.as_bytes()).await {
                                    println!("[RUST-DEBUG] handle_connection: Error processing message from DIP {}: {}", controller.dip, e);
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
                    println!("[RUST-DEBUG] handle_connection: Sending message to DIP {}: {:?} -> '{}'",
                           controller.dip, message, String::from_utf8_lossy(&data));

                    if let Err(e) = writer.write_all(&data).await {
                        println!("[RUST-DEBUG] handle_connection: Write error to DIP {}: {}", controller.dip, e);
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
        println!(
            "[RUST-DEBUG] handle_connection: I/O loop ended, marking DIP {} as disconnected",
            controller.dip
        );
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

        println!(
            "[RUST-DEBUG] Processing incoming message from DIP {}: '{}'",
            controller.dip, line
        );

        controller
            .bytes_received
            .fetch_add(data.len() as u64, Ordering::Relaxed);
        controller.messages_received.fetch_add(1, Ordering::Relaxed);

        match serde_json::from_str::<IncomingMessage>(&line) {
            Ok(message) => {
                println!(
                    "[RUST-DEBUG] Successfully parsed message from DIP {}: {:?}",
                    controller.dip, message
                );

                controller
                    .add_log(
                        LogDirection::Incoming,
                        format!("Received: {:?}", message),
                        Some(line.clone()),
                    )
                    .await;

                match message {
                    IncomingMessage::Heartbeat => {
                        println!(
                            "[RUST-DEBUG] Received heartbeat from DIP {}, responding with noop",
                            controller.dip
                        );
                        // Respond with noop
                        controller.send_message(OutgoingMessage::Noop).await?;
                    }
                    IncomingMessage::Controller { dip } => {
                        println!(
                            "[RUST-DEBUG] Controller identified with DIP: {} (expected: {})",
                            dip, controller.dip
                        );
                        controller
                            .add_log(
                                LogDirection::Info,
                                format!("Controller identified with DIP: {}", dip),
                                None,
                            )
                            .await;
                        // Update DIP if different
                        if controller.dip != dip {
                            println!(
                                "[RUST-DEBUG] DIP mismatch for controller: expected {}, got {}",
                                controller.dip, dip
                            );
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
                        println!(
                            "[RUST-DEBUG] Received button press from DIP {}: {:?}",
                            controller.dip, buttons
                        );
                        // Broadcast button state
                        let result = controller.button_broadcast.send(buttons);
                        match result {
                            Ok(_) => println!(
                                "[RUST-DEBUG] Button broadcast sent successfully for DIP {}",
                                controller.dip
                            ),
                            Err(e) => println!(
                                "[RUST-DEBUG] Button broadcast failed for DIP {}: {:?}",
                                controller.dip, e
                            ),
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

    pub async fn commit_display(&self) -> Result<Vec<OutgoingMessage>> {
        if let Some(controller) = self.get_controller_state().await {
            controller.commit_display().await
        } else {
            Ok(vec![])
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

    async fn get_controller_state(&self) -> Option<Arc<ControllerState>> {
        self.controller_state.read().await.as_ref().cloned()
    }

    pub async fn send_message(&self, message: OutgoingMessage) -> Result<()> {
        let _ = self.message_tx.send(message);
        Ok(())
    }
}
