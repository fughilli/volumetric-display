use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::RwLock;
use std::net::{SocketAddr, UdpSocket};
use std::time::Duration;
use tokio::time::interval;
use tokio::sync::mpsc; // For channels between MIDI callback and MIDI processing task

use midir::{MidiInput, MidiOutput, Ignore, MidiOutputConnection, MidiInputPort, MidiOutputPort};
use rosc::{OscPacket, OscMessage, OscType, encoder, decoder::decode_udp};
use tracing::{info, warn, error, Level, debug};
use tracing_subscriber::FmtSubscriber;
use clap::Parser; // For argument parsing

// --- Constants from Python script ---
const NUM_ROWS: usize = 8;
const NUM_COLS: usize = 8;
const NUM_LFO_BANKS: usize = 4;
const NUM_EFFECT_BANKS: usize = 4;
const TOTAL_ROWS: usize = NUM_ROWS * NUM_LFO_BANKS;
const TOTAL_COLS: usize = NUM_COLS * NUM_EFFECT_BANKS;

// APC MINI LED Velocities
const LED_OFF: u8 = 0;
const LED_GREEN: u8 = 1;
const LED_RED: u8 = 3;
const LED_ORANGE: u8 = 5;
const LED_BLUE_ISH: u8 = 6;

// OSC Buffer Size
const OSC_BUF_SIZE: usize = 1536; // A common buffer size for OSC over UDP

lazy_static::lazy_static! {
    static ref NOTE_GRID: [[u8; NUM_COLS]; NUM_ROWS] = {
        let mut grid = [[0u8; NUM_COLS]; NUM_ROWS];
        for r in 0..NUM_ROWS {
            for c in 0..NUM_COLS {
                grid[r][c] = ((NUM_ROWS - 1 - r) * 8 + c) as u8;
            }
        }
        grid
    };
}

// --- LedState for diffing MIDI messages ---
struct LedState {
    grid: [[u8; NUM_COLS]; NUM_ROWS], // Velocities for the 8x8 visible grid
    lfo_banks: [u8; NUM_LFO_BANKS],   // Velocities for LFO bank LEDs (notes 82-85)
    effect_banks: [u8; NUM_EFFECT_BANKS], // Velocities for Effect bank LEDs (notes 86-89)
}

impl LedState {
    fn new() -> Self {
        Self {
            grid: [[LED_OFF; NUM_COLS]; NUM_ROWS],
            lfo_banks: [LED_OFF; NUM_LFO_BANKS],
            effect_banks: [LED_OFF; NUM_EFFECT_BANKS],
        }
    }

    // Sends a MIDI note for a grid LED if its state has changed.
    // r_vis and c_vis are 0-indexed for the visible 8x8 grid.
    fn send_grid_note_if_changed(&mut self, conn: &mut MidiOutputConnection, r_vis: usize, c_vis: usize, desired_velocity: u8) {
        if r_vis < NUM_ROWS && c_vis < NUM_COLS { // Bounds check for safety
            let note = NOTE_GRID[r_vis][c_vis];
            if self.grid[r_vis][c_vis] != desired_velocity {
                debug!("GRID LED CHANGE: Note {}, Vis ({},{}), From {}, To {}", note, r_vis, c_vis, self.grid[r_vis][c_vis], desired_velocity);
                if let Err(e) = conn.send(&[0x90, note, desired_velocity]) {
                    warn!("Failed to send MIDI note {} (grid): {}", note, e);
                }
                self.grid[r_vis][c_vis] = desired_velocity;
            } else {
                // debug!("Grid LED no change: Note {}, Vis ({},{}), Vel {}", note, r_vis, c_vis, desired_velocity);
            }
        } else {
            warn!("Attempted to send grid note out of bounds: r_vis={}, c_vis={}", r_vis, c_vis);
        }
    }

    // Sends a MIDI note for an LFO bank LED if its state has changed.
    // bank_idx is 0-indexed (0-3).
    fn send_lfo_bank_note_if_changed(&mut self, conn: &mut MidiOutputConnection, bank_idx: usize, desired_velocity: u8) {
        if bank_idx < NUM_LFO_BANKS { // Bounds check
            let note = (82 + bank_idx) as u8;
            if self.lfo_banks[bank_idx] != desired_velocity {
                debug!("LFO BANK LED CHANGE: Note {}, Bank Idx {}, From {}, To {}", note, bank_idx, self.lfo_banks[bank_idx], desired_velocity);
                if let Err(e) = conn.send(&[0x90, note, desired_velocity]) {
                    warn!("Failed to send MIDI note {} (LFO bank): {}", note, e);
                }
                self.lfo_banks[bank_idx] = desired_velocity;
            } else {
                // debug!("LFO Bank LED no change: Note {}, Bank Idx {}, Vel {}", note, bank_idx, desired_velocity);
            }
        } else {
            warn!("Attempted to send LFO bank note out of bounds: bank_idx={}", bank_idx);
        }
    }

    // Sends a MIDI note for an Effect bank LED if its state has changed.
    // bank_idx is 0-indexed (0-3).
    fn send_effect_bank_note_if_changed(&mut self, conn: &mut MidiOutputConnection, bank_idx: usize, desired_velocity: u8) {
        if bank_idx < NUM_EFFECT_BANKS { // Bounds check
            let note = (86 + bank_idx) as u8;
            if self.effect_banks[bank_idx] != desired_velocity {
                debug!("EFFECT BANK LED CHANGE: Note {}, Bank Idx {}, From {}, To {}", note, bank_idx, self.effect_banks[bank_idx], desired_velocity);
                if let Err(e) = conn.send(&[0x90, note, desired_velocity]) {
                    warn!("Failed to send MIDI note {} (Effect bank): {}", note, e);
                }
                self.effect_banks[bank_idx] = desired_velocity;
            } else {
                // debug!("Effect Bank LED no change: Note {}, Bank Idx {}, Vel {}", note, bank_idx, desired_velocity);
            }
        } else {
             warn!("Attempted to send Effect bank note out of bounds: bank_idx={}", bank_idx);
        }
    }
}

// --- Command Line Arguments ---
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct CliArgs {
    #[clap(long, default_value = "127.0.0.1")]
    in_host: String,
    #[clap(long, default_value_t = 9000)]
    in_port: u16,
    #[clap(long, default_value = "127.0.0.1")]
    out_host: String,
    #[clap(long, default_value_t = 9001)]
    out_port: u16,
}

// --- Shared Application State (Refactored for Granular Locking & Atomics) ---
struct AppStateBanks {
    current_lfo_bank: AtomicUsize,     // Now Atomic
    current_effect_bank: AtomicUsize,  // Now Atomic
}

// Placed LedUpdateRequest at the module level for wider scope
#[derive(Debug)]
enum LedUpdateRequest {
    FullRefresh,
    BothRefresh, 
    FaderColumnRefresh { actual_effect_idx: usize },
}

struct AppState {
    banks: Arc<AppStateBanks>, // No longer Mutex wrapped
    mapping: Arc<RwLock<Vec<Vec<bool>>>>, 
    fader_override_active: Arc<RwLock<Vec<Vec<bool>>>>, 
    fader_override_value: Arc<RwLock<Vec<Vec<f32>>>>,  
    latest_lfo_values: Arc<RwLock<Vec<f32>>>,
}

impl AppState {
    fn new() -> Self {
        AppState {
            banks: Arc::new(AppStateBanks {
                current_lfo_bank: AtomicUsize::new(0),
                current_effect_bank: AtomicUsize::new(0),
            }),
            mapping: Arc::new(RwLock::new(vec![vec![false; TOTAL_COLS]; TOTAL_ROWS])),
            fader_override_active: Arc::new(RwLock::new(vec![vec![false; TOTAL_COLS]; NUM_LFO_BANKS])),
            fader_override_value: Arc::new(RwLock::new(vec![vec![0.0; TOTAL_COLS]; NUM_LFO_BANKS])),
            latest_lfo_values: Arc::new(RwLock::new(vec![0.0; TOTAL_ROWS])),
        }
    }
}

// Define a common error type for the application
type AppError = Box<dyn std::error::Error + Send + Sync>;

// --- Main Application ---
#[tokio::main]
async fn main() -> Result<(), AppError> { // Use AppError
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::DEBUG)
        .with_thread_ids(true)
        .with_thread_names(true)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("Setting default subscriber failed");
    
    let args = CliArgs::parse();
    info!("Starting ArtNet Mapper in Rust with args: {:?}", args);

    let app_state = Arc::new(AppState::new()); // Now Arc<AppState>

    let osc_in_addr_str = format!("{}:{}", args.in_host, args.in_port);
    let osc_out_addr_str = format!("{}:{}", args.out_host, args.out_port);
    
    let osc_out_addr: SocketAddr = osc_out_addr_str.parse().map_err(AppError::from)?;
    let osc_in_addr: SocketAddr = osc_in_addr_str.parse().map_err(AppError::from)?;

    // Restore MIDI Output and LED update channel
    let midi_out_conn_arc = match setup_midi_output() { 
        Ok(conn) => Arc::new(Mutex::new(conn)),
        Err(e) => {
            error!("Failed to setup MIDI output: {}. LED feedback will be disabled.", e);
            // Optionally, allow the app to continue without LED feedback
            // For now, we return the error to be consistent with previous behavior.
            return Err(e.into()); 
        }
    };
    let (led_tx, led_rx) = mpsc::channel::<LedUpdateRequest>(8); 
    
    {
        let mut initial_midi_out = midi_out_conn_arc.lock().unwrap();
        clear_all_leds(&mut initial_midi_out); 
        // Initial _update_bank_select_leds and _refresh_grid_leds calls are removed from here.
        // The led_update_loop will handle initial setup via a BothRefresh request.
        info!("Hardware LEDs cleared. Initial state will be set by LED update task.");
    }

    let osc_input_task = tokio::spawn(handle_osc_input(Arc::clone(&app_state), osc_in_addr));
    
    let (midi_event_tx, midi_event_rx) = mpsc::channel(64); 
    let midi_input_setup_task = tokio::spawn(keep_midi_input_alive(midi_event_tx)); 
    
    let led_update_task_handle = tokio::spawn(led_update_loop(led_rx, Arc::clone(&midi_out_conn_arc), Arc::clone(&app_state)));

    // Send initial refresh request to the LED update task
    if let Err(e) = led_tx.try_send(LedUpdateRequest::BothRefresh) {
        warn!("Failed to send initial BothRefresh LED update request: {}", e);
    }

    let midi_processing_task = tokio::spawn(process_midi_messages(Arc::clone(&app_state), midi_event_rx, led_tx.clone()));
    let osc_sender_task = tokio::spawn(osc_sender_loop(Arc::clone(&app_state), osc_out_addr));

    info!("OSC Input: {}", osc_in_addr);
    info!("OSC Output: {}", osc_out_addr);
    info!("Control mapper running...");

    match tokio::try_join!(
        osc_input_task,
        midi_input_setup_task,
        midi_processing_task,
        osc_sender_task,
        led_update_task_handle // Add LED task to try_join!
    ) {
        Ok((res1, res2, res3, res4, _res_led)) => { // Add result for LED task, mark _res_led as unused
            res1?; 
            res2.map_err(|s| AppError::from(Box::new(std::io::Error::new(std::io::ErrorKind::Other,s))))?;
            res3?; 
            res4?; 
            // res_led.map_err(|join_err| AppError::from(Box::new(join_err)))?; // Removed: JoinError handled by try_join!
        }
        Err(e) => return Err(AppError::from(e)), 
    }

    Ok(())
}

fn process_osc_message(msg: OscMessage, app_state: &Arc<AppState>) {
    if msg.addr.starts_with("/lfo/") {
        if let Some(row_str) = msg.addr.split('/').last() {
            if let Ok(lfo_source_on_grid) = row_str.parse::<usize>() {
                // LFOs are by row, so lfo_source_on_grid (1-8) corresponds to a row.
                if lfo_source_on_grid >= 1 && lfo_source_on_grid <= NUM_ROWS { // Check against NUM_ROWS
                    if let Some(OscType::Float(value)) = msg.args.get(0) {
                        let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
                        let actual_lfo_idx = current_lfo_bank * NUM_ROWS + (lfo_source_on_grid - 1); // Use NUM_ROWS for LFO from row
                        
                        let mut latest_lfo_values_guard = app_state.latest_lfo_values.write().unwrap();
                        if actual_lfo_idx < latest_lfo_values_guard.len() { 
                            latest_lfo_values_guard[actual_lfo_idx] = *value;
                        } else {
                            warn!("actual_lfo_idx {} out of bounds for latest_lfo_values (len {}). OSC lfo_source_on_grid: {}", actual_lfo_idx, latest_lfo_values_guard.len(), lfo_source_on_grid);
                        }
                    } else {
                        warn!("LFO message did not contain a float argument: {:?}", msg.args);
                    }
                } else {
                    warn!("LFO source on grid out of range: {}", lfo_source_on_grid);
                }
            } else {
                warn!("Could not parse LFO row from address: {}", msg.addr);
            }
        }
    } else if msg.addr == "/_samplerate" {
        // known message, can ignore if not used
    } else {
        warn!("Received unhandled OSC message: {:?}", msg);
    }
}

// --- OSC Input Handling ---
async fn handle_osc_input(app_state: Arc<AppState>, addr: SocketAddr) -> Result<(), AppError> {
    info!("Starting OSC input listener on {}", addr);
    let socket = UdpSocket::bind(addr).map_err(AppError::from)?;
    socket.set_nonblocking(true).map_err(AppError::from)?;
    let mut buf = [0u8; OSC_BUF_SIZE]; 
    loop {
        match socket.recv_from(&mut buf) {
            Ok((size, _src_addr)) => { 
                match decode_udp(&buf[..size]) { 
                    Ok((_remaining_buf, OscPacket::Message(msg))) => {
                        process_osc_message(msg, &app_state);
                    }
                    Ok((_remaining_buf, OscPacket::Bundle(bundle))) => {
                        // warn!("Received OSC Bundle, processing contents...");
                        for packet in bundle.content {
                            match packet {
                                OscPacket::Message(msg) => {
                                    process_osc_message(msg, &app_state);
                                }
                                OscPacket::Bundle(inner_bundle) => {
                                    warn!("Received nested OSC Bundle, not yet handled: {:?}", inner_bundle);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Error decoding OSC packet: {}", e);
                    }
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                tokio::time::sleep(Duration::from_millis(1)).await;
                continue;
            }
            Err(e) => {
                error!("Error receiving from OSC socket: {}", e);
                return Err(e.into());
            }
        }
    }
}

// --- MIDI Input Handling (Corrected Lifetime Management) ---
async fn keep_midi_input_alive(midi_tx: mpsc::Sender<Vec<u8>>) -> Result<(), String> {
    let mut midi_in = MidiInput::new("ArtNetMapperRust_Input")
        .map_err(|e| format!("Failed to create MidiInput: {}", e))?;
    midi_in.ignore(Ignore::None);

    let ports = midi_in.ports();
    let apc_port_info: Option<(MidiInputPort, String)> = ports.iter().find_map(|p| {
        let port_name = midi_in.port_name(p).unwrap_or_default();
        if port_name.to_uppercase().contains("APC MINI") {
            Some((p.clone(), port_name))
        } else {
            info!("Available MIDI In Port: {}", port_name);
            None
        }
    });

    if let Some((port, name)) = apc_port_info {
        info!("Connecting to MIDI Input: {}", name);
        let _conn_in = midi_in.connect(&port, "apc-mini-in", move |_timestamp, message, _| {
            debug!("MIDI Input: {:?}", message);
            if midi_tx.try_send(message.to_vec()).is_err() {
                // warn!("MIDI input channel full or closed, message dropped.");
            }
        }, ()).map_err(|e| format!("Failed to connect to MIDI input: {}", e))?;
        
        loop {
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    } else {
        Err("APC MINI MIDI input not found".to_string())
    }
}

// --- MIDI Output Setup (Commented out as LED feedback is removed) ---
fn setup_midi_output() -> Result<MidiOutputConnection, String> {
    let midi_out = MidiOutput::new("ArtNetMapperRust_Output")
        .map_err(|e| format!("Failed to create MidiOutput: {}", e))?;
    
    let ports = midi_out.ports();
    let apc_port_info: Option<(MidiOutputPort, String)> = ports.iter().find_map(|p| {
        let port_name = midi_out.port_name(p).unwrap_or_default();
        if port_name.to_uppercase().contains("APC MINI") {
            Some((p.clone(), port_name))
        } else {
            info!("Available MIDI Out Port: {}", port_name);
            None
        }
    });

    if let Some((port, name)) = apc_port_info {
        info!("Connecting to MIDI Output: {}", name);
        midi_out.connect(&port, "apc-mini-out")
            .map_err(|e| format!("Failed to connect to MIDI output: {}", e))
    } else {
        Err("APC MINI MIDI output not found".to_string())
    }
}

// --- LED Utility Functions (Commented out as LED feedback is removed) ---
// fn send_midi_note(conn: &mut MidiOutputConnection, note: u8, velocity: u8) { // REMOVE THIS FUNCTION
//     if let Err(e) = conn.send(&[0x90, note, velocity]) {
//         warn!("Failed to send MIDI note: {}", e);
//     }
// }

fn clear_all_leds(midi_out_conn: &mut MidiOutputConnection) {
    info!("Clearing all LEDs (Notes 0-95).");
    for note_to_clear in 0..96 {
        // Directly send MIDI message to clear, as this is a startup hardware reset
        if let Err(e) = midi_out_conn.send(&[0x90, note_to_clear, LED_OFF]) {
            warn!("Failed to send MIDI note {} during clear_all_leds: {}", note_to_clear, e);
        }
    }
}

fn _update_bank_select_leds(midi_out_conn: &mut MidiOutputConnection, app_state: &Arc<AppState>, led_state: &mut LedState) {
    let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
    let current_effect_bank = app_state.banks.current_effect_bank.load(Ordering::SeqCst);

    for i in 0..NUM_LFO_BANKS {
        let velocity = if i == current_lfo_bank { LED_ORANGE } else { LED_OFF };
        led_state.send_lfo_bank_note_if_changed(midi_out_conn, i, velocity);
    }
    for i in 0..NUM_EFFECT_BANKS {
        let velocity = if i == current_effect_bank { LED_BLUE_ISH } else { LED_OFF };
        led_state.send_effect_bank_note_if_changed(midi_out_conn, i, velocity);
    }
}

fn _refresh_grid_leds(midi_out_conn: &mut MidiOutputConnection, app_state: &Arc<AppState>, led_state: &mut LedState) {
    let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
    let current_effect_bank = app_state.banks.current_effect_bank.load(Ordering::SeqCst);
    debug!("Refreshing grid LEDs. LFO Bank: {}, Effect Bank: {}", current_lfo_bank, current_effect_bank);
    let mapping_guard = app_state.mapping.read().unwrap();
    let fader_override_active_guard = app_state.fader_override_active.read().unwrap();

    for r_vis in 0..NUM_ROWS { // Visual rows on APC (LFOs)
        for c_vis in 0..NUM_COLS { // Visual columns on APC (Effects)
            let actual_r_lfo_idx = current_lfo_bank * NUM_ROWS + r_vis; 
            let actual_c_effect_idx = current_effect_bank * NUM_COLS + c_vis; 
            
            let mut led_velocity = LED_OFF;

            if actual_r_lfo_idx < TOTAL_ROWS && actual_c_effect_idx < TOTAL_COLS { 
                let is_fader_override = fader_override_active_guard[current_lfo_bank][actual_c_effect_idx];
                let is_mapped = mapping_guard[actual_r_lfo_idx][actual_c_effect_idx];
                if r_vis == 0 { // Log for top row only to reduce spam
                    // debug!("LED Refresh (r_vis={}, c_vis={}): LFO_idx={}, Effect_idx={}, Mapped={}, Override={}",
                    //    r_vis, c_vis, actual_r_lfo_idx, actual_c_effect_idx, is_mapped, is_fader_override);
                }

                if is_fader_override && is_mapped {
                    led_velocity = LED_RED;
                } else if is_mapped {
                    led_velocity = LED_GREEN;
                }
            } 
            led_state.send_grid_note_if_changed(midi_out_conn, r_vis, c_vis, led_velocity);
        }
    }
}

// --- MIDI Message Processing (Simplified: No LED Updates) --- -> Restoring LED logic
async fn process_midi_messages(app_state: Arc<AppState>, mut midi_rx: mpsc::Receiver<Vec<u8>>, led_tx: mpsc::Sender<LedUpdateRequest>) -> Result<(), AppError> {
    info!("Starting MIDI message processing task.");
    while let Some(message_data) = midi_rx.recv().await {
        if message_data.is_empty() { continue; }
        let status = message_data[0];
        let data1 = if message_data.len() > 1 { message_data[1] } else { 0 };
        let data2 = if message_data.len() > 2 { message_data[2] } else { 0 };

        if status & 0xF0 == 0x90 { // Note-on
            let note = data1;
            let velocity = data2;
            if velocity > 0 { // True note-on
                if (82..=85).contains(&note) { // LFO Bank
                    let new_lfo_bank = (note - 82) as usize;
                    app_state.banks.current_lfo_bank.store(new_lfo_bank, Ordering::SeqCst);
                    info!("Switched to LFO Bank {}", new_lfo_bank);
                    if let Err(e) = led_tx.try_send(LedUpdateRequest::BothRefresh) {
                        warn!("Failed to send BothRefresh LED update request for LFO bank switch: {}", e);
                    }
                } else if (86..=89).contains(&note) { // Effect Bank
                    let new_effect_bank = (note - 86) as usize;
                    app_state.banks.current_effect_bank.store(new_effect_bank, Ordering::SeqCst);
                    info!("Switched to Effect Bank {}", new_effect_bank);
                    if let Err(e) = led_tx.try_send(LedUpdateRequest::BothRefresh) {
                        warn!("Failed to send BothRefresh LED update request for effect bank switch: {}", e);
                    }
                } else { // Grid button
                    let mut r_pressed_vis: Option<usize> = None;
                    let mut c_pressed_vis: Option<usize> = None;
                    for r_vis in 0..NUM_ROWS {
                        for c_vis_inner in 0..NUM_COLS { 
                            if NOTE_GRID[r_vis][c_vis_inner] == note {
                                r_pressed_vis = Some(r_vis);
                                c_pressed_vis = Some(c_vis_inner);
                                break;
                            }
                        }
                        if r_pressed_vis.is_some() { break; }
                    }

                    if let (Some(r_pv), Some(c_pv)) = (r_pressed_vis, c_pressed_vis) {
                        let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
                        let current_effect_bank = app_state.banks.current_effect_bank.load(Ordering::SeqCst);
                        
                        // --- REVERTING TO: LFO from visual Row, Effect from visual Column ---
                        let actual_r_lfo_idx = current_lfo_bank * NUM_ROWS + r_pv;    // LFO index from visual row r_pv
                        let actual_c_effect_idx = current_effect_bank * NUM_COLS + c_pv; // Effect index from visual column c_pv
                        // --- END REVERT ---

                        if actual_c_effect_idx < TOTAL_COLS && actual_r_lfo_idx < TOTAL_ROWS { // Bounds check with new var names
                            let mut mapping_guard = app_state.mapping.write().unwrap(); 
                            let mut fader_override_active_guard = app_state.fader_override_active.write().unwrap();
                            
                            // When a grid button is pressed, deactivate fader override for that column ONLY in the context of the CURRENT LFO bank.
                            if fader_override_active_guard[current_lfo_bank][actual_c_effect_idx] {
                                fader_override_active_guard[current_lfo_bank][actual_c_effect_idx] = false;
                                info!("Fader override on actual col {} for LFO bank {} deactivated by button press.", actual_c_effect_idx, current_lfo_bank);
                            }

                            if mapping_guard[actual_r_lfo_idx][actual_c_effect_idx] {
                                mapping_guard[actual_r_lfo_idx][actual_c_effect_idx] = false;
                                debug!("Toggled OFF mapping: LFO {} to Effect {}", actual_r_lfo_idx, actual_c_effect_idx);
                            } else {
                                debug!("Attempting to map LFO {} to Effect {}. Applying mutual exclusivity...", actual_r_lfo_idx, actual_c_effect_idx);
                                // Mutual exclusivity: An Effect (from visual column) can only be driven by one LFO (from visual row).
                                // Unmap other LFOs (from different visual rows) from this specific Effect (actual_c_effect_idx).
                                for r_iter_vis in 0..NUM_ROWS { // Iterate through visual rows (LFOs in current bank)
                                    let iter_lfo_idx = current_lfo_bank * NUM_ROWS + r_iter_vis;
                                    // If this iter_lfo_idx is different from the LFO we are currently processing (actual_r_lfo_idx)
                                    if iter_lfo_idx != actual_r_lfo_idx && iter_lfo_idx < TOTAL_ROWS { // Check iter_lfo_idx bounds
                                         if mapping_guard[iter_lfo_idx][actual_c_effect_idx] { // Check if this other LFO is mapped to the current Effect
                                            debug!("MUTEX: Unmapping LFO {} from Effect {}", iter_lfo_idx, actual_c_effect_idx);
                                            mapping_guard[iter_lfo_idx][actual_c_effect_idx] = false;
                                         }
                                    }
                                }
                                mapping_guard[actual_r_lfo_idx][actual_c_effect_idx] = true;
                                debug!("Toggled ON mapping: LFO {} to Effect {}", actual_r_lfo_idx, actual_c_effect_idx);
                            }
                            // Grid button presses should always trigger a full refresh of the grid LEDs for the current view
                            if let Err(e) = led_tx.try_send(LedUpdateRequest::FullRefresh) {
                                warn!("Failed to send FullRefresh LED update request for grid button: {}", e);
                            }
                        } else { warn!("Calculated actual pressed note out of bounds!"); }
                    }
                }
            } 
        } else if status & 0xF0 == 0xB0 { // Control Change (Faders)
            let cc_number = data1;
            let cc_value = data2;
            debug!("MIDI CC Rcvd: Num={}, Val={}", cc_number, cc_value);

            if (48..=55).contains(&cc_number) { 
                let col_index_on_grid = (cc_number - 48) as usize;
                let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
                let current_effect_bank = app_state.banks.current_effect_bank.load(Ordering::SeqCst);
                let actual_col_idx_fader = current_effect_bank * NUM_COLS + col_index_on_grid;

                if actual_col_idx_fader < TOTAL_COLS {
                    let mut fader_override_active_guard = app_state.fader_override_active.write().unwrap();
                    let mut fader_override_value_guard = app_state.fader_override_value.write().unwrap();

                    if !fader_override_active_guard[current_lfo_bank][actual_col_idx_fader] {
                        info!("Fader CC {} taking control of actual col {} for LFO Bank {}", cc_number, actual_col_idx_fader, current_lfo_bank);
                    }
                    fader_override_active_guard[current_lfo_bank][actual_col_idx_fader] = true;
                    let fader_float_val = cc_value as f32 / 127.0;
                    fader_override_value_guard[current_lfo_bank][actual_col_idx_fader] = fader_float_val;
                    debug!("Fader CC: Set val={} for actual_col(Effect)={}, lfo_bank={}", fader_float_val, actual_col_idx_fader, current_lfo_bank);
                    
                    if let Err(e) = led_tx.try_send(LedUpdateRequest::FaderColumnRefresh { actual_effect_idx: actual_col_idx_fader }) {
                        warn!("Failed to send FaderColumnRefresh LED update request: {}", e);
                    }
                } else { 
                    warn!("Calculated actual fader column out of bounds: {}", actual_col_idx_fader); 
                }
            }
        }
    }
    Ok(())
}

// --- Dedicated LED Update Loop (Commented out) --- -> Restoring
async fn led_update_loop(mut led_rx: mpsc::Receiver<LedUpdateRequest>, midi_out_conn_arc: Arc<Mutex<MidiOutputConnection>>, app_state: Arc<AppState>) {
    info!("Starting LED update loop with diffing.");
    let mut led_state = LedState::new(); // Initialize LedState

    while let Some(request) = led_rx.recv().await {
        debug!("LED Update Task: Received {:?}", request); 
        let mut midi_out_guard = midi_out_conn_arc.lock().unwrap();
        match request {
            LedUpdateRequest::FullRefresh => {
                _refresh_grid_leds(&mut midi_out_guard, &app_state, &mut led_state);
            }
            LedUpdateRequest::BothRefresh => {
                _update_bank_select_leds(&mut midi_out_guard, &app_state, &mut led_state);
                _refresh_grid_leds(&mut midi_out_guard, &app_state, &mut led_state);
            }
            LedUpdateRequest::FaderColumnRefresh { actual_effect_idx } => {
                _refresh_fader_column_leds(&mut midi_out_guard, &app_state, actual_effect_idx, &mut led_state);
            }
        }
    }
    info!("LED update loop ended.");
}

// Helper function to refresh LEDs for a single fader's column (which is an Effect column)
fn _refresh_fader_column_leds(midi_out_conn: &mut MidiOutputConnection, app_state: &Arc<AppState>, actual_effect_idx_of_fader: usize, led_state: &mut LedState) {
    let current_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);
    let current_effect_bank = app_state.banks.current_effect_bank.load(Ordering::SeqCst);
    let mapping_guard = app_state.mapping.read().unwrap();
    let fader_override_active_guard = app_state.fader_override_active.read().unwrap();

    // Determine the visual column index (c_vis) for this actual_effect_idx_of_fader
    // This actual_effect_idx_of_fader is what the fader controls.
    // We need to update the LEDs in the hardware column that corresponds to this effect.
    let c_vis_of_effect = if actual_effect_idx_of_fader >= current_effect_bank * NUM_COLS && actual_effect_idx_of_fader < (current_effect_bank + 1) * NUM_COLS {
        Some(actual_effect_idx_of_fader % NUM_COLS)
    } else {
        None // This effect is not in the current_effect_bank's view
    };

    if let Some(c_vis) = c_vis_of_effect { // c_vis is the visual column of the effect/fader
        for r_vis in 0..NUM_ROWS { // Iterate through visual rows (LFOs)
            let actual_r_lfo_idx = current_lfo_bank * NUM_ROWS + r_vis;
            // actual_c_effect_idx is fixed for this fader's column: it's actual_effect_idx_of_fader
            
            let mut led_velocity = LED_OFF;

            if actual_r_lfo_idx < TOTAL_ROWS { // actual_effect_idx_of_fader is already known to be < TOTAL_COLS if c_vis_of_effect is Some
                let is_fader_override = fader_override_active_guard[current_lfo_bank][actual_effect_idx_of_fader];
                let is_mapped = mapping_guard[actual_r_lfo_idx][actual_effect_idx_of_fader];

                if is_fader_override && is_mapped {
                    led_velocity = LED_RED;
                } else if is_mapped {
                    led_velocity = LED_GREEN;
                }
            }
            led_state.send_grid_note_if_changed(midi_out_conn, r_vis, c_vis, led_velocity);
        }
    }
}

// --- OSC Sender Loop ---
async fn osc_sender_loop(app_state: Arc<AppState>, target_addr: SocketAddr) -> Result<(), AppError> {
    info!("Starting OSC sender loop for {}", target_addr);
    let socket = UdpSocket::bind("0.0.0.0:0").map_err(AppError::from)?;
    let mut interval = interval(Duration::from_millis(16)); // 60 Hz
    let mut osc_sent_values = vec![-1.0f32; TOTAL_COLS];
    loop {
        interval.tick().await;
        let mut next_osc_values_to_send = osc_sent_values.clone(); 
        
        {
            // Acquire all necessary read locks at the beginning of the scope
            let mapping_guard = app_state.mapping.read().unwrap();
            let fader_override_active_guard = app_state.fader_override_active.read().unwrap();
            let fader_override_value_guard = app_state.fader_override_value.read().unwrap();
            let latest_lfo_values_guard = app_state.latest_lfo_values.read().unwrap();
            // Read current LFO bank for context-sensitive LFO mapping search
            let active_lfo_bank = app_state.banks.current_lfo_bank.load(Ordering::SeqCst);

            for actual_col_idx_effect in 0..TOTAL_COLS { 
                let mut found_active_driver_for_col = false;
                
                for lfo_bank_idx_for_fader_check in 0..NUM_LFO_BANKS {
                    if fader_override_active_guard[lfo_bank_idx_for_fader_check][actual_col_idx_effect] {
                        next_osc_values_to_send[actual_col_idx_effect] = fader_override_value_guard[lfo_bank_idx_for_fader_check][actual_col_idx_effect];
                        found_active_driver_for_col = true;
                        break;
                    }
                }
                if found_active_driver_for_col { continue; }

                // PRIORITY 2: LFO Mappings (if no fader override for this actual_col_idx_effect)
                // Search LFOs only within the currently active LFO bank.
                // Iterate visual LFO rows (0 to NUM_ROWS-1) in the active bank, from highest visual row to lowest.
                for visual_row_idx_lfo in (0..NUM_ROWS).rev() { 
                    let actual_row_idx_lfo = active_lfo_bank * NUM_ROWS + visual_row_idx_lfo;
                    
                    if actual_row_idx_lfo < TOTAL_ROWS { // Ensure global LFO index is within bounds of mapping array
                        if mapping_guard[actual_row_idx_lfo][actual_col_idx_effect] { 
                            if actual_row_idx_lfo < latest_lfo_values_guard.len() {
                                let lfo_val = latest_lfo_values_guard[actual_row_idx_lfo];
                                next_osc_values_to_send[actual_col_idx_effect] = lfo_val;
                            }
                            break; 
                        }
                    }
                }
            }
        } // All read locks are released here

        let mut messages_for_bundle: Vec<OscPacket> = Vec::new();
        let mut indices_updated_in_bundle: Vec<usize> = Vec::new();

        for i in 0..TOTAL_COLS {
            if (next_osc_values_to_send[i] - osc_sent_values[i]).abs() > f32::EPSILON {
                let msg_addr = format!("/effect/{}", i + 1);
                let msg_args = vec![OscType::Float(next_osc_values_to_send[i])];
                messages_for_bundle.push(OscPacket::Message(OscMessage { addr: msg_addr, args: msg_args }));
                indices_updated_in_bundle.push(i);
            }
        }

        if !messages_for_bundle.is_empty() {
            let bundle = OscPacket::Bundle(rosc::OscBundle {
                timetag: rosc::OscTime { seconds: 0, fractional: 1 }, // Represents "immediately"
                content: messages_for_bundle,
            });
            match encoder::encode(&bundle) {
                Ok(encoded_bundle) => {
                    if let Err(e) = socket.send_to(&encoded_bundle, target_addr) {
                        error!("Failed to send OSC bundle: {}", e);
                    } else {
                        // If send was successful (or at least, no immediate error),
                        // update the sent values for the included messages.
                        for &idx in &indices_updated_in_bundle {
                            osc_sent_values[idx] = next_osc_values_to_send[idx];
                        }
                        // tracing::debug!("Sent OSC bundle with {} messages", indices_updated_in_bundle.len());
                    }
                }
                Err(e) => {
                    error!("Failed to encode OSC bundle: {}", e);
                }
            }
        }
    }
} 
