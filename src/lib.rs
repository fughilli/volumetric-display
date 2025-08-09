use pyo3::prelude::*;
use pyo3::types::PyList;
use std::net::UdpSocket;
use std::sync::Arc;
use tokio::runtime::Runtime;

mod control_port;
mod web_monitor;

use control_port::{Config, ControllerManager, ControllerState, OutgoingMessage};
use web_monitor::WebMonitor;

fn saturate_u8(value: f32) -> u8 {
    value.max(0.0).min(255.0) as u8
}

#[pymodule]
mod artnet_rs {
    use super::*;

    #[pyclass(name = "ArtNetController")]
    struct ArtNetControllerRs {
        socket: UdpSocket,
        target_addr: String,
    }

    impl ArtNetControllerRs {
        fn create_dmx_packet(&self, universe: u16, data: &[u8]) -> Vec<u8> {
            let mut packet = Vec::with_capacity(18 + data.len());
            packet.extend_from_slice(b"Art-Net\x00");
            packet.extend_from_slice(&0x5000u16.to_le_bytes()); // OpDmx
            packet.extend_from_slice(&14u16.to_be_bytes()); // ProtVer
            packet.push(0); // Sequence
            packet.push(0); // Physical
            packet.extend_from_slice(&universe.to_le_bytes());
            packet.extend_from_slice(&(data.len() as u16).to_be_bytes());
            packet.extend_from_slice(data);
            packet
        }

        fn create_sync_packet(&self) -> Vec<u8> {
            let mut packet = Vec::with_capacity(14);
            packet.extend_from_slice(b"Art-Net\x00");
            packet.extend_from_slice(&0x5200u16.to_le_bytes()); // OpSync
            packet.extend_from_slice(&14u16.to_be_bytes()); // ProtVer
            packet.push(0); // Aux1
            packet.push(0); // Aux2
            packet
        }
    }

    #[pymethods]
    impl ArtNetControllerRs {
        #[new]
        fn new(ip: String, port: u16) -> PyResult<Self> {
            let socket = UdpSocket::bind("0.0.0.0:0")?;
            socket.set_broadcast(true)?;
            let target_addr = format!("{}:{}", ip, port);
            Ok(ArtNetControllerRs {
                socket,
                target_addr,
            })
        }

        #[pyo3(signature = (base_universe, raster, channels_per_universe=510, universes_per_layer=3, channel_span=1, z_indices=None))]
        fn send_dmx(
            &self,
            base_universe: u16,
            raster: &Bound<'_, PyAny>,
            channels_per_universe: usize,
            universes_per_layer: u16,
            channel_span: usize,
            z_indices: Option<Vec<usize>>,
        ) -> PyResult<()> {
            let width: usize = raster.getattr("width")?.extract()?;
            let height: usize = raster.getattr("height")?.extract()?;
            let length: usize = raster.getattr("length")?.extract()?;
            let brightness: f32 = raster.getattr("brightness")?.extract()?;
            let raster_data_attr = raster.getattr("data")?;
            let raster_data: &Bound<'_, PyList> = raster_data_attr.downcast()?;

            let z_indices_vec: Vec<usize>;
            let z_indices_ref: &[usize] = match z_indices {
                Some(ref v) => v,
                None => {
                    z_indices_vec = (0..length).step_by(channel_span).collect();
                    &z_indices_vec
                }
            };

            let mut data_bytes = Vec::with_capacity(width * height * 3);

            for (out_z, &z) in z_indices_ref.iter().enumerate() {
                let mut universe =
                    (out_z / channel_span) as u16 * universes_per_layer + base_universe;

                let start = z * width * height;
                let end = (z + 1) * width * height;

                if end > raster_data.len() {
                    // This is a safeguard, in case of inconsistent raster data.
                    // You might want to return an error instead.
                    continue;
                }

                for i in start..end {
                    let rgb_obj = raster_data.get_item(i)?;
                    let r: f32 = rgb_obj.getattr("red")?.extract()?;
                    let g: f32 = rgb_obj.getattr("green")?.extract()?;
                    let b: f32 = rgb_obj.getattr("blue")?.extract()?;

                    data_bytes.push(saturate_u8(r * brightness));
                    data_bytes.push(saturate_u8(g * brightness));
                    data_bytes.push(saturate_u8(b * brightness));
                }

                let mut data_to_send = &data_bytes[..];
                while !data_to_send.is_empty() {
                    let chunk_size = std::cmp::min(data_to_send.len(), channels_per_universe);
                    let chunk = &data_to_send[..chunk_size];
                    let dmx_packet = self.create_dmx_packet(universe, chunk);
                    self.socket.send_to(&dmx_packet, &self.target_addr)?;

                    data_to_send = &data_to_send[chunk_size..];
                    universe += 1;
                }
                data_bytes.clear();
            }

            let sync_packet = self.create_sync_packet();
            self.socket.send_to(&sync_packet, &self.target_addr)?;

            Ok(())
        }
    }

    // Control Port Python bindings
    #[pyclass(name = "ControllerManager")]
    struct ControllerManagerPy {
        runtime: Runtime,
        manager: Arc<ControllerManager>,
        web_monitor: Option<Arc<WebMonitor>>,
    }

    #[pymethods]
    impl ControllerManagerPy {
        #[new]
        fn new(config_json: &str) -> PyResult<Self> {
            let config: Config = serde_json::from_str(config_json).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid config JSON: {}",
                    e
                ))
            })?;

            let runtime = Runtime::new().map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to create runtime: {}",
                    e
                ))
            })?;

            let manager = Arc::new(ControllerManager::new(config));

            Ok(Self {
                runtime,
                manager,
                web_monitor: None,
            })
        }

        fn initialize(&mut self) -> PyResult<()> {
            self.runtime
                .block_on(async { self.manager.initialize().await })
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to initialize: {}",
                        e
                    ))
                })
        }

        fn start_web_monitor(&mut self, port: u16) -> PyResult<()> {
            let web_monitor = Arc::new(WebMonitor::new(self.manager.clone()));
            let web_monitor_clone = web_monitor.clone();

            self.runtime.spawn(async move {
                if let Err(e) = web_monitor_clone.as_ref().start_server(port).await {
                    eprintln!("Web monitor error: {}", e);
                }
            });

            self.web_monitor = Some(web_monitor);
            Ok(())
        }

        fn get_controller(&self, dip: &str) -> Option<ControllerStatePy> {
            self.manager
                .get_controller(dip)
                .map(|controller| ControllerStatePy {
                    runtime: self.runtime.handle().clone(),
                    controller,
                })
        }

        fn get_controller_stats(&self) -> PyResult<Vec<PyObject>> {
            let stats = self
                .runtime
                .block_on(async { self.manager.get_controller_stats().await });

            Python::with_gil(|py| {
                let result: Result<Vec<PyObject>, PyErr> = stats
                    .iter()
                    .map(|stat| {
                        pythonize::pythonize(py, stat)
                            .map(|bound| bound.into())
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                                    "Serialization error: {}",
                                    e
                                ))
                            })
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

    #[pyclass(name = "ControllerState")]
    struct ControllerStatePy {
        runtime: tokio::runtime::Handle,
        controller: Arc<ControllerState>,
    }

    #[pymethods]
    impl ControllerStatePy {
        fn clear_display(&self) -> PyResult<()> {
            self.runtime.block_on(async {
                self.controller.clear_display().await;
            });
            Ok(())
        }

        fn write_display(&self, x: u16, y: u16, text: &str) -> PyResult<()> {
            self.runtime.block_on(async {
                self.controller.write_display(x, y, text).await;
            });
            Ok(())
        }

        fn commit_display(&self) -> PyResult<()> {
            let messages = self
                .runtime
                .block_on(async { self.controller.commit_display().await })
                .map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                        "Failed to commit display: {}",
                        e
                    ))
                })?;

            self.runtime.block_on(async {
                for message in messages {
                    if let Err(e) = self.controller.send_message(message).await {
                        eprintln!("Failed to send message: {}", e);
                    }
                }
            });
            Ok(())
        }

        fn set_leds(&self, rgb_values: Vec<(u8, u8, u8)>) -> PyResult<()> {
            self.runtime.block_on(async {
                if let Err(e) = self
                    .controller
                    .send_message(OutgoingMessage::Led { rgb_values })
                    .await
                {
                    eprintln!("Failed to set LEDs: {}", e);
                }
            });
            Ok(())
        }

        fn set_backlights(&self, states: Vec<bool>) -> PyResult<()> {
            self.runtime.block_on(async {
                if let Err(e) = self
                    .controller
                    .send_message(OutgoingMessage::Backlight { states })
                    .await
                {
                    eprintln!("Failed to set backlights: {}", e);
                }
            });
            Ok(())
        }

        fn register_button_callback(&self, callback: PyObject) -> PyResult<ButtonEventReceiver> {
            let button_rx = self.controller.button_broadcast.subscribe();
            let handle = self.runtime.clone();

            let receiver = ButtonEventReceiver {
                runtime: handle,
                receiver: Arc::new(tokio::sync::Mutex::new(button_rx)),
                callback: Arc::new(callback),
            };

            Ok(receiver)
        }

        #[getter]
        fn dip(&self) -> String {
            self.controller.dip.clone()
        }

        #[getter]
        fn connected(&self) -> PyResult<bool> {
            Ok(self
                .runtime
                .block_on(async { *self.controller.connected.read().await }))
        }
    }

    #[pyclass(name = "ButtonEventReceiver")]
    struct ButtonEventReceiver {
        runtime: tokio::runtime::Handle,
        receiver: Arc<tokio::sync::Mutex<tokio::sync::broadcast::Receiver<Vec<bool>>>>,
        callback: Arc<PyObject>,
    }

    #[pymethods]
    impl ButtonEventReceiver {
        fn start_listening(&self) -> PyResult<()> {
            let receiver = self.receiver.clone();
            let callback = self.callback.clone();
            let _runtime = self.runtime.clone();

            self.runtime.spawn(async move {
                let mut rx = receiver.lock().await;
                while let Ok(buttons) = rx.recv().await {
                    Python::with_gil(|py| {
                        if let Err(e) = callback.call1(py, (buttons,)) {
                            eprintln!("Button callback error: {}", e);
                        }
                    });
                }
            });

            Ok(())
        }
    }
}
