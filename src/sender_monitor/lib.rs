use pyo3::prelude::*;
use std::sync::Arc;
use tokio::runtime::Runtime;

// Re-export the sender_monitor module
pub mod sender_monitor;
pub mod web_monitor;

use sender_monitor::SenderMonitor;
use web_monitor::WebMonitor;

#[pymodule]
mod sender_monitor_rs {
    use super::*;

    #[pyclass(name = "SenderMonitorManager")]
    struct SenderMonitorManagerPy {
        runtime: Arc<Runtime>,
        sender_monitor: Arc<SenderMonitor>,
        web_monitor: Option<Arc<WebMonitor>>,
    }

    #[pymethods]
    impl SenderMonitorManagerPy {
        #[new]
        fn new() -> PyResult<Self> {
            let runtime =
                Arc::new(Runtime::new().map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?);

            let sender_monitor = Arc::new(SenderMonitor::new());

            Ok(SenderMonitorManagerPy {
                runtime,
                sender_monitor,
                web_monitor: None,
            })
        }

        fn register_controller(&self, ip: String, port: u16) -> PyResult<()> {
            self.sender_monitor.register_controller(ip, port);
            Ok(())
        }

        fn set_cooldown_duration(&self, cooldown_seconds: i64) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor.set_cooldown_duration(cooldown_seconds).await;
            });
            Ok(())
        }

        fn report_controller_success(&self, ip: String, port: u16) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor.report_controller_success(&ip, port).await;
            });
            Ok(())
        }

        fn report_controller_failure(&self, ip: String, port: u16, error: String) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor
                    .report_controller_failure(&ip, port, &error)
                    .await;
            });
            Ok(())
        }

        fn report_frame(&self) -> PyResult<()> {
            self.sender_monitor.report_frame();
            Ok(())
        }

        fn set_debug_mode(&self, enabled: bool) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor.set_debug_mode(enabled).await;
            });
            Ok(())
        }

        fn set_debug_pause(&self, paused: bool) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor.set_debug_pause(paused).await;
            });
            Ok(())
        }

        fn is_debug_mode(&self) -> PyResult<bool> {
            let sender_monitor = self.sender_monitor.clone();
            let runtime = self.runtime.clone();

            // Use block_on for synchronous access
            let result = runtime.block_on(async { sender_monitor.is_debug_mode().await });
            Ok(result)
        }

        fn is_paused(&self) -> PyResult<bool> {
            let sender_monitor = self.sender_monitor.clone();
            let runtime = self.runtime.clone();

            // Use block_on for synchronous access
            let result = runtime.block_on(async { sender_monitor.is_paused().await });
            Ok(result)
        }

        fn get_debug_command(&self) -> PyResult<Option<pyo3::PyObject>> {
            let sender_monitor = self.sender_monitor.clone();
            let runtime = self.runtime.clone();

            // Use block_on for synchronous access
            let result = runtime.block_on(async { sender_monitor.get_debug_command().await });

            // Convert to Python object if present
            match result {
                Some(cmd) => {
                    let py_cmd = pyo3::Python::with_gil(|py| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("command_type", cmd.command_type).unwrap();

                        if let Some(mt) = cmd.mapping_tester {
                            let mt_dict = pyo3::types::PyDict::new(py);
                            mt_dict.set_item("orientation", mt.orientation).unwrap();
                            mt_dict.set_item("layer", mt.layer).unwrap();
                            mt_dict.set_item("color", mt.color).unwrap();
                            dict.set_item("mapping_tester", mt_dict).unwrap();
                        }

                        if let Some(pdt) = cmd.power_draw_tester {
                            let pdt_dict = pyo3::types::PyDict::new(py);
                            pdt_dict.set_item("color", pdt.color).unwrap();
                            pdt_dict
                                .set_item("modulation_type", pdt.modulation_type)
                                .unwrap();
                            pdt_dict.set_item("frequency", pdt.frequency).unwrap();
                            pdt_dict.set_item("amplitude", pdt.amplitude).unwrap();
                            pdt_dict.set_item("offset", pdt.offset).unwrap();
                            pdt_dict
                                .set_item("global_brightness", pdt.global_brightness)
                                .unwrap();
                            dict.set_item("power_draw_tester", pdt_dict).unwrap();
                        }

                        dict.into()
                    });
                    Ok(Some(py_cmd))
                }
                None => Ok(None),
            }
        }

        fn start_web_monitor(&mut self, port: u16) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            let web_monitor = Arc::new(WebMonitor::new(sender_monitor));

            let web_monitor_clone = web_monitor.clone();
            self.runtime.spawn(async move {
                if let Err(e) = web_monitor_clone.start_server(port).await {
                    eprintln!("Sender monitor web server error: {}", e);
                }
            });

            self.web_monitor = Some(web_monitor);
            Ok(())
        }

        fn start_web_monitor_with_bind_address(
            &mut self,
            port: u16,
            bind_address: String,
        ) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            let web_monitor =
                Arc::new(WebMonitor::new(sender_monitor).with_bind_address(bind_address));

            let web_monitor_clone = web_monitor.clone();
            self.runtime.spawn(async move {
                if let Err(e) = web_monitor_clone.start_server(port).await {
                    eprintln!("Sender monitor web server error: {}", e);
                }
            });

            self.web_monitor = Some(web_monitor);
            Ok(())
        }

        fn get_controller_count(&self) -> PyResult<usize> {
            Ok(self.sender_monitor.get_controller_count())
        }

        fn get_routable_controller_count(&self) -> PyResult<usize> {
            Ok(self.sender_monitor.get_routable_controller_count())
        }

        fn set_world_dimensions(&self, width: usize, height: usize, length: usize) -> PyResult<()> {
            let sender_monitor = self.sender_monitor.clone();
            self.runtime.spawn(async move {
                sender_monitor
                    .set_world_dimensions(width, height, length)
                    .await;
            });
            Ok(())
        }

        fn shutdown(&self) -> PyResult<()> {
            // The runtime will be dropped when this object is dropped
            Ok(())
        }
    }
}
