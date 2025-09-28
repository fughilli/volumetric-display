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
        runtime: Runtime,
        sender_monitor: Arc<SenderMonitor>,
        web_monitor: Option<Arc<WebMonitor>>,
    }

    #[pymethods]
    impl SenderMonitorManagerPy {
        #[new]
        fn new() -> PyResult<Self> {
            let runtime = Runtime::new()
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

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

        fn shutdown(&self) -> PyResult<()> {
            // The runtime will be dropped when this object is dropped
            Ok(())
        }
    }
}
