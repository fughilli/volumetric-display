use pyo3::prelude::*;
use pyo3::types::PyList;
use std::net::UdpSocket;

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
}
