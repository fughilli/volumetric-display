use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::runtime::Runtime;
use tokio::sync::{broadcast, RwLock};
use tokio::time::{interval, timeout};

// Re-export the control_port module
pub mod control_port;
pub mod web_monitor;

use control_port::{Config, ControlPort, ControlPortManager};
use web_monitor::WebMonitor;

#[pymodule]
mod control_port_rs {
    use super::*;

    #[pyclass(name = "ControlPortManager")]
    struct ControlPortManagerPy {
        runtime: Runtime,
        manager: Arc<ControlPortManager>,
        web_monitor: Option<Arc<WebMonitor>>,
    }

    #[pymethods]
    impl ControlPortManagerPy {
        #[new]
        fn new(config_json: &str) -> PyResult<Self> {
            let config: Config = serde_json::from_str(config_json)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

            let runtime = Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let manager = Arc::new(ControlPortManager::new(config));

            Ok(ControlPortManagerPy {
                runtime,
                manager,
                web_monitor: None,
            })
        }

        fn initialize(&mut self) -> PyResult<()> {
            self.runtime
                .block_on(async { self.manager.initialize().await })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(())
        }

        fn start_web_monitor(&mut self, port: u16) -> PyResult<()> {
            let manager = self.manager.clone();
            self.runtime.spawn(async move {
                if let Err(e) = manager.start_web_monitor(port).await {
                    eprintln!("Web monitor error: {}", e);
                }
            });
            Ok(())
        }

        fn get_control_port(&self, dip: &str) -> Option<ControlPortPy> {
            self.manager
                .get_control_port(dip)
                .map(|control_port| ControlPortPy {
                    runtime_handle: self.runtime.handle().clone(),
                    control_port,
                })
        }

        fn get_all_stats(&self) -> PyResult<Vec<PyObject>> {
            let stats = self
                .runtime
                .block_on(async { self.manager.get_all_stats().await });

            Python::with_gil(|py| {
                let result: Result<Vec<PyObject>, PyErr> = stats
                    .iter()
                    .map(|stat| {
                        // Manual serialization to avoid pythonize dependency
                        let dict = PyDict::new(py);
                        dict.set_item("dip", stat.dip.clone())?;
                        dict.set_item("ip", stat.ip.clone())?;
                        dict.set_item("port", stat.port)?;
                        dict.set_item("connected", stat.connected)?;
                        dict.set_item(
                            "last_message_time",
                            stat.last_message_time.map(|dt| dt.to_rfc3339()),
                        )?;
                        dict.set_item(
                            "connection_time",
                            stat.connection_time.map(|dt| dt.to_rfc3339()),
                        )?;
                        dict.set_item("bytes_sent", stat.bytes_sent)?;
                        dict.set_item("bytes_received", stat.bytes_received)?;
                        dict.set_item("messages_sent", stat.messages_sent)?;
                        dict.set_item("messages_received", stat.messages_received)?;
                        dict.set_item("connection_attempts", stat.connection_attempts)?;
                        dict.set_item("last_error", stat.last_error.as_deref())?;
                        Ok(dict.into())
                    })
                    .collect();
                result
            })
        }

        fn shutdown(&self) -> PyResult<()> {
            self.runtime.block_on(async {
                self.manager.shutdown().await;
            });
            Ok(())
        }
    }

    #[pyclass(name = "ControlPort")]
    struct ControlPortPy {
        runtime_handle: tokio::runtime::Handle,
        control_port: Arc<ControlPort>,
    }

    #[pymethods]
    impl ControlPortPy {
        fn clear_display(&self) -> PyResult<()> {
            self.runtime_handle.block_on(async {
                self.control_port.clear_display().await;
            });
            Ok(())
        }

        fn write_display(&self, x: u16, y: u16, text: &str) -> PyResult<()> {
            self.runtime_handle.block_on(async {
                self.control_port.write_display(x, y, text).await;
            });
            Ok(())
        }

        fn commit_display(&self) -> PyResult<()> {
            self.runtime_handle
                .block_on(async { self.control_port.commit_display().await })
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to commit display: {}",
                        e
                    ))
                })
        }

        fn set_leds(&self, rgb_values: Vec<(u8, u8, u8)>) -> PyResult<()> {
            self.runtime_handle.block_on(async {
                self.control_port.set_leds(rgb_values).await;
            });
            Ok(())
        }

        fn set_backlights(&self, states: Vec<bool>) -> PyResult<()> {
            self.runtime_handle.block_on(async {
                self.control_port.set_backlights(states).await;
            });
            Ok(())
        }

        fn register_button_callback(&self, callback: PyObject) -> PyResult<ButtonEventReceiver> {
            println!(
                "[RUST-DEBUG] Registering button callback for DIP {}",
                self.control_port.dip
            );

            let receiver = self.control_port.button_broadcast.subscribe();
            let receiver = Arc::new(tokio::sync::Mutex::new(receiver));
            let callback = Arc::new(callback);

            println!(
                "[RUST-DEBUG] Button callback registered successfully for DIP {}",
                self.control_port.dip
            );

            let button_receiver = ButtonEventReceiver {
                runtime_handle: self.runtime_handle.clone(),
                receiver,
                callback,
            };

            println!(
                "[RUST-DEBUG] Created ButtonEventReceiver for DIP {}",
                self.control_port.dip
            );

            Ok(button_receiver)
        }

        fn dip(&self) -> String {
            self.control_port.dip.clone()
        }

        fn connected(&self) -> PyResult<bool> {
            self.runtime_handle.block_on(async {
                // Check the controller state instead of the control port state
                if let Some(controller) = self.control_port.get_controller_state().await {
                    let connected = *controller.connected.read().await;
                    Ok(connected)
                } else {
                    Ok(false)
                }
            })
        }
    }

    #[pyclass(name = "ButtonEventReceiver")]
    struct ButtonEventReceiver {
        runtime_handle: tokio::runtime::Handle,
        receiver: Arc<tokio::sync::Mutex<tokio::sync::broadcast::Receiver<Vec<bool>>>>,
        callback: Arc<PyObject>,
    }

    #[pymethods]
    impl ButtonEventReceiver {
        fn start_listening(&self) -> PyResult<()> {
            println!("[RUST-DEBUG] Starting button event listener");

            let receiver = self.receiver.clone();
            let callback = self.callback.clone();
            let runtime_handle = self.runtime_handle.clone();

            self.runtime_handle.spawn(async move {
                println!("[RUST-DEBUG] Button event listener task started");
                loop {
                    let mut receiver_guard = receiver.lock().await;
                    match receiver_guard.recv().await {
                        Ok(buttons) => {
                            println!("[RUST-DEBUG] Received button event: {:?}", buttons);
                            let callback = callback.clone();
                            runtime_handle.spawn_blocking(move || {
                                Python::with_gil(|py| {
                                    println!(
                                        "[RUST-DEBUG] Executing button callback with buttons: {:?}",
                                        buttons
                                    );
                                    if let Err(e) = callback.call1(py, (buttons,)) {
                                        println!("[RUST-DEBUG] Button callback error: {}", e);
                                    } else {
                                        println!(
                                            "[RUST-DEBUG] Button callback executed successfully"
                                        );
                                    }
                                });
                            });
                        }
                        Err(e) => {
                            println!("[RUST-DEBUG] Button event receiver error: {:?}", e);
                            break;
                        }
                    }
                }
                println!("[RUST-DEBUG] Button event listener task ended");
            });
            Ok(())
        }
    }
}
