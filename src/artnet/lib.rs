use pyo3::prelude::*;
use pyo3::types::PyList;
use std::net::UdpSocket;

fn saturate_u8(value: f32) -> u8 {
    value.max(0.0).min(255.0) as u8
}

#[pymodule]
mod artnet_rs {
    use super::*;

    #[pyclass(name = "RGB")]
    #[derive(Clone, Debug)]
    struct RGB {
        red: u8,
        green: u8,
        blue: u8,
    }

    #[pymethods]
    impl RGB {
        #[new]
        fn new(red: u8, green: u8, blue: u8) -> Self {
            RGB { red, green, blue }
        }

        #[staticmethod]
        fn from_hsv(hsv: &HSV) -> Self {
            let h = hsv.hue as f32 / (256.0 / 6.0);
            let s = hsv.saturation as f32 / 255.0;
            let v = hsv.value as f32 / 255.0;

            let c = v * s;
            let x = c * (1.0 - (h % 2.0 - 1.0).abs());
            let m = v - c;

            let (r, g, b) = if h < 1.0 {
                (c, x, 0.0)
            } else if h < 2.0 {
                (x, c, 0.0)
            } else if h < 3.0 {
                (0.0, c, x)
            } else if h < 4.0 {
                (0.0, x, c)
            } else if h < 5.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            RGB {
                red: saturate_u8((r + m) * 255.0),
                green: saturate_u8((g + m) * 255.0),
                blue: saturate_u8((b + m) * 255.0),
            }
        }
    }

    #[pyclass(name = "HSV")]
    #[derive(Clone, Debug)]
    struct HSV {
        hue: u8,
        saturation: u8,
        value: u8,
    }

    #[pymethods]
    impl HSV {
        #[new]
        fn new(hue: u8, saturation: u8, value: u8) -> Self {
            HSV {
                hue,
                saturation,
                value,
            }
        }
    }

    #[pyclass(name = "Raster")]
    #[derive(Clone)]
    struct Raster {
        width: usize,
        height: usize,
        length: usize,
        brightness: f32,
        data: Vec<RGB>,
        orientation: Vec<String>,
        transform: Vec<(usize, i32)>, // (axis, sign)
    }

    #[pymethods]
    impl Raster {
        #[new]
        fn new(
            width: usize,
            height: usize,
            length: usize,
            orientation: Option<Vec<String>>,
        ) -> Self {
            let orientation = orientation
                .unwrap_or_else(|| vec!["X".to_string(), "Y".to_string(), "Z".to_string()]);
            let mut raster = Raster {
                width,
                height,
                length,
                brightness: 1.0,
                data: vec![RGB::new(0, 0, 0); width * height * length],
                orientation,
                transform: Vec::new(),
            };
            raster.compute_transform();
            raster
        }

        fn compute_transform(&mut self) {
            self.transform.clear();
            for coord in &self.orientation {
                let axis = coord.chars().last().unwrap(); // Get the axis (X, Y, or Z)
                let sign = if coord.starts_with('-') { -1 } else { 1 };
                let axis_idx = match axis {
                    'X' => 0,
                    'Y' => 1,
                    'Z' => 2,
                    _ => panic!("Invalid axis: {}", axis),
                };
                self.transform.push((axis_idx, sign));
            }
        }

        fn transform_coords(&self, x: usize, y: usize, z: usize) -> (usize, usize, usize) {
            let coords = [x, y, z];
            let mut result = [0, 0, 0];

            for (i, (axis, sign)) in self.transform.iter().enumerate() {
                if *sign == 1 {
                    result[i] = coords[*axis];
                } else {
                    // For negative axes, subtract from the maximum value
                    let max_val = match axis {
                        0 => self.width - 1,
                        1 => self.height - 1,
                        2 => self.length - 1,
                        _ => unreachable!(),
                    };
                    result[i] = max_val - coords[*axis];
                }
            }

            (result[0], result[1], result[2])
        }

        fn set_pix(&mut self, x: usize, y: usize, z: usize, color: RGB) -> PyResult<()> {
            if x >= self.width {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "x: {} width: {}",
                    x, self.width
                )));
            }
            if y >= self.height {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "y: {} height: {}",
                    y, self.height
                )));
            }
            if z >= self.length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "z: {} length: {}",
                    z, self.length
                )));
            }

            // Transform coordinates
            let (tx, ty, tz) = self.transform_coords(x, y, z);
            // Calculate index in the data array
            let idx = ty * self.width + tx + tz * self.width * self.height;
            self.data[idx] = color;
            Ok(())
        }

        fn clear(&mut self) {
            self.data = vec![RGB::new(0, 0, 0); self.width * self.height * self.length];
        }

        // Getters for Python compatibility
        fn get_width(&self) -> usize {
            self.width
        }
        fn get_height(&self) -> usize {
            self.height
        }
        fn get_length(&self) -> usize {
            self.length
        }
        fn get_brightness(&self) -> f32 {
            self.brightness
        }
        fn get_data(&self) -> Vec<RGB> {
            self.data.clone()
        }
        fn get_orientation(&self) -> Vec<String> {
            self.orientation.clone()
        }

        // Setters for Python compatibility
        fn set_brightness(&mut self, brightness: f32) {
            self.brightness = brightness;
        }

        // Direct access to data for compatibility with existing code
        fn get_data_mut(&mut self) -> PyResult<Vec<RGB>> {
            Ok(self.data.clone())
        }

        // Get pixel at coordinates
        fn get_pix(&self, x: usize, y: usize, z: usize) -> PyResult<RGB> {
            if x >= self.width || y >= self.height || z >= self.length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Coordinates out of bounds",
                ));
            }
            let (tx, ty, tz) = self.transform_coords(x, y, z);
            let idx = ty * self.width + tx + tz * self.width * self.height;
            Ok(self.data[idx].clone())
        }

        // Set pixel without coordinate transformation (for direct access)
        fn set_pix_direct(&mut self, x: usize, y: usize, z: usize, color: RGB) -> PyResult<()> {
            if x >= self.width || y >= self.height || z >= self.length {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Coordinates out of bounds",
                ));
            }
            let idx = y * self.width + x + z * self.width * self.height;
            self.data[idx] = color;
            Ok(())
        }
    }

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

        fn get_ip(&self) -> String {
            // Extract IP from target_addr (format: "ip:port")
            self.target_addr.split(':').next().unwrap_or("").to_string()
        }

        fn get_port(&self) -> u16 {
            // Extract port from target_addr (format: "ip:port")
            self.target_addr
                .split(':')
                .nth(1)
                .unwrap_or("0")
                .parse()
                .unwrap_or(0)
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
            // Check if this is a Rust Raster by looking for a specific method
            if raster.hasattr("get_data_mut")? {
                // This is likely a Rust Raster, try to get its data directly
                let width: usize = raster.getattr("width")?.extract()?;
                let height: usize = raster.getattr("height")?.extract()?;
                let length: usize = raster.getattr("length")?.extract()?;
                let brightness: f32 = raster.getattr("brightness")?.extract()?;
                let data: Vec<RGB> = raster.call_method0("get_data_mut")?.extract()?;

                return self.send_dmx_rust_raster_data(
                    base_universe,
                    width,
                    height,
                    length,
                    brightness,
                    data,
                    channels_per_universe,
                    universes_per_layer,
                    channel_span,
                    z_indices,
                );
            }

            // Fall back to Python raster
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

        fn send_dmx_rust_raster_data(
            &self,
            base_universe: u16,
            width: usize,
            height: usize,
            length: usize,
            brightness: f32,
            data: Vec<RGB>,
            channels_per_universe: usize,
            universes_per_layer: u16,
            channel_span: usize,
            z_indices: Option<Vec<usize>>,
        ) -> PyResult<()> {
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

                if end > data.len() {
                    continue;
                }

                for i in start..end {
                    let rgb = &data[i];
                    data_bytes.push(saturate_u8(rgb.red as f32 * brightness));
                    data_bytes.push(saturate_u8(rgb.green as f32 * brightness));
                    data_bytes.push(saturate_u8(rgb.blue as f32 * brightness));
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

        fn send_dmx_scatter_gather(
            &self,
            world_raster: &Bound<'_, PyAny>,
            universe: u16,
            coords: Vec<(i32, i32, i32)>,
        ) -> PyResult<()> {
            let channels_per_universe = 510;
            // Get world raster dimensions and brightness
            let width: usize = world_raster.getattr("width")?.extract()?;
            let height: usize = world_raster.getattr("height")?.extract()?;
            let length: usize = world_raster.getattr("length")?.extract()?;
            let brightness: f32 = world_raster.getattr("brightness")?.extract()?;

            // Check if this is a Rust Raster (has get_data_mut method)
            let data_bytes = if world_raster.hasattr("get_data_mut")? {
                // Fast path: Rust Raster with native Vec<RGB>
                let data: Vec<RGB> = world_raster.call_method0("get_data_mut")?.extract()?;

                // Sample pixels from world_raster at the specified coordinates
                let mut bytes = Vec::new();
                for coord in coords.iter() {
                    let x = coord.0 as usize;
                    let y = coord.1 as usize;
                    let z = coord.2 as usize;

                    // Ensure coordinates are within bounds
                    if x < width && y < height && z < length {
                        // Calculate index into flat Vec<RGB> array (format: [z][y][x])
                        let idx = z * (width * height) + y * width + x;

                        if idx < data.len() {
                            let pixel = &data[idx];
                            bytes.push(saturate_u8(pixel.red as f32 * brightness));
                            bytes.push(saturate_u8(pixel.green as f32 * brightness));
                            bytes.push(saturate_u8(pixel.blue as f32 * brightness));
                        } else {
                            // Index out of bounds - send black
                            bytes.push(0);
                            bytes.push(0);
                            bytes.push(0);
                        }
                    } else {
                        // Coordinates out of bounds - send black
                        bytes.push(0);
                        bytes.push(0);
                        bytes.push(0);
                    }
                }
                bytes
            } else {
                // Fall back to Python Raster with NumPy array
                let data_attr = world_raster.getattr("data")?;

                // Sample pixels from world_raster at the specified coordinates
                let mut bytes = Vec::new();
                for coord in coords.iter() {
                    let x = coord.0 as usize;
                    let y = coord.1 as usize;
                    let z = coord.2 as usize;

                    // Ensure coordinates are within bounds
                    if x < width && y < height && z < length {
                        // Access NumPy array using indexing: data[z, y, x] returns [r, g, b]
                        match data_attr.call_method1("__getitem__", ((z, y, x),)) {
                            Ok(pixel_obj) => {
                                // Extract RGB values from the pixel array
                                match (
                                    pixel_obj.call_method1("__getitem__", (0,)),
                                    pixel_obj.call_method1("__getitem__", (1,)),
                                    pixel_obj.call_method1("__getitem__", (2,)),
                                ) {
                                    (Ok(r_obj), Ok(g_obj), Ok(b_obj)) => {
                                        let r: u8 = r_obj.extract().unwrap_or(0);
                                        let g: u8 = g_obj.extract().unwrap_or(0);
                                        let b: u8 = b_obj.extract().unwrap_or(0);

                                        bytes.push(saturate_u8(r as f32 * brightness));
                                        bytes.push(saturate_u8(g as f32 * brightness));
                                        bytes.push(saturate_u8(b as f32 * brightness));
                                    }
                                    _ => {
                                        // Failed to extract RGB - send black
                                        bytes.push(0);
                                        bytes.push(0);
                                        bytes.push(0);
                                    }
                                }
                            }
                            Err(_) => {
                                // Index out of bounds - send black
                                bytes.push(0);
                                bytes.push(0);
                                bytes.push(0);
                            }
                        }
                    } else {
                        // Coordinates out of bounds - send black
                        bytes.push(0);
                        bytes.push(0);
                        bytes.push(0);
                    }
                }
                bytes
            };

            // Send data in chunks of channels_per_universe bytes
            let mut current_universe = universe;
            let mut data_to_send = &data_bytes[..];
            while !data_to_send.is_empty() {
                let chunk_size = std::cmp::min(data_to_send.len(), channels_per_universe);
                let chunk = &data_to_send[..chunk_size];
                let dmx_packet = self.create_dmx_packet(current_universe, chunk);
                self.socket.send_to(&dmx_packet, &self.target_addr)?;

                data_to_send = &data_to_send[chunk_size..];
                current_universe += 1;
            }

            // Send a sync packet after all data for this curtain is sent
            let sync_packet = self.create_sync_packet();
            self.socket.send_to(&sync_packet, &self.target_addr)?;

            Ok(())
        }
    }
}
