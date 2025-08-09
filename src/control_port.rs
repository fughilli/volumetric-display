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
use tokio::sync::{broadcast, mpsc, RwLock};
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

// Main controller manager
pub struct ControllerManager {
    pub controllers: DashMap<String, Arc<ControllerState>>,
    pub config: Config,
    shutdown_tx: broadcast::Sender<()>,
}

impl ControllerManager {
    pub fn new(config: Config) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);

        Self {
            controllers: DashMap::new(),
            config,
            shutdown_tx,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        for (dip, controller_config) in &self.config.controller_addresses {
            let controller_state =
                Arc::new(ControllerState::new(dip.clone(), controller_config.clone()));
            self.controllers
                .insert(dip.clone(), controller_state.clone());

            // Start connection task for each controller
            let task = tokio::spawn(Self::controller_task(
                controller_state.clone(),
                self.shutdown_tx.subscribe(),
            ));

            *controller_state.connection_task.write().await = Some(task);
        }

        Ok(())
    }

    async fn controller_task(
        controller: Arc<ControllerState>,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) {
        let mut reconnect_interval = interval(Duration::from_secs(5));
        let mut heartbeat_interval = interval(Duration::from_secs(1));

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                _ = reconnect_interval.tick() => {
                    if !*controller.connected.read().await {
                        if let Err(e) = Self::attempt_connection(&controller).await {
                            controller.add_log(
                                LogDirection::Error,
                                format!("Connection failed: {}", e),
                                None,
                            ).await;
                        }
                    }
                }
                _ = heartbeat_interval.tick() => {
                    if *controller.connected.read().await {
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

        let stream = timeout(Duration::from_secs(5), TcpStream::connect(socket_addr)).await??;

        *controller.connected.write().await = true;
        let mut stats = controller.stats.write().await;
        stats.connection_time = Some(Utc::now());
        stats.last_error = None;
        drop(stats);

        controller
            .add_log(
                LogDirection::Info,
                "Connected successfully".to_string(),
                None,
            )
            .await;

        // Start communication task
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
                            if let Err(e) = Self::process_incoming_message(&controller, line.as_bytes()).await {
                                controller.add_log(
                                    LogDirection::Error,
                                    format!("Error processing message: {}", e),
                                    Some(line.clone()),
                                ).await;
                            }
                        }
                        Err(e) => {
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

        match serde_json::from_str::<IncomingMessage>(&line) {
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
                        let _ = controller.button_broadcast.send(buttons);
                    }
                }
            }
            Err(_) => {
                controller
                    .add_log(
                        LogDirection::Incoming,
                        "Received non-JSON message".to_string(),
                        Some(line),
                    )
                    .await;
            }
        }

        controller.update_stats().await;
        Ok(())
    }

    pub async fn get_controller_stats(&self) -> Vec<ControllerStats> {
        let mut stats = Vec::new();
        for entry in self.controllers.iter() {
            let controller = entry.value();
            controller.update_stats().await;
            stats.push(controller.stats.read().await.clone());
        }
        stats
    }

    pub async fn get_controller_logs(&self, dip: &str) -> Option<Vec<LogEntry>> {
        if let Some(controller) = self.controllers.get(dip) {
            Some(controller.log.read().await.iter().cloned().collect())
        } else {
            None
        }
    }

    pub fn get_controller(&self, dip: &str) -> Option<Arc<ControllerState>> {
        self.controllers.get(dip).map(|entry| entry.value().clone())
    }

    pub async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(());

        // Wait for all tasks to complete
        for entry in self.controllers.iter() {
            let controller = entry.value();
            if let Some(task) = controller.connection_task.read().await.as_ref() {
                task.abort();
            }
        }
    }
}
