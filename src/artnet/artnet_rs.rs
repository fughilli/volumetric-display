use std::net::UdpSocket;

fn saturate_u8(value: f32) -> u8 {
    value.max(0.0).min(255.0) as u8
}

pub struct ArtNetSender {
    socket: UdpSocket,
    target_addr: String,
}

impl ArtNetSender {
    pub fn new(ip: &str, port: u16) -> Result<Self, std::io::Error> {
        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.set_broadcast(true)?;
        let target_addr = format!("{}:{}", ip, port);
        Ok(ArtNetSender {
            socket,
            target_addr,
        })
    }

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

    pub fn send_volume_data(
        &self,
        base_universe: u16,
        width: usize,
        height: usize,
        depth: usize,
        brightness: f32,
        data: &[(u8, u8, u8)], // RGB tuples
        channels_per_universe: usize,
        universes_per_layer: u16,
        channel_span: usize,
    ) -> Result<(), std::io::Error> {
        let z_indices: Vec<usize> = (0..depth).step_by(channel_span).collect();
        let mut data_bytes = Vec::with_capacity(width * height * 3);

        for (out_z, &z) in z_indices.iter().enumerate() {
            let mut universe = (out_z / channel_span) as u16 * universes_per_layer + base_universe;

            let start = z * width * height;
            let end = (z + 1) * width * height;

            if end > data.len() {
                continue;
            }

            for i in start..end {
                let (r, g, b) = data[i];
                data_bytes.push(saturate_u8(r as f32 * brightness));
                data_bytes.push(saturate_u8(g as f32 * brightness));
                data_bytes.push(saturate_u8(b as f32 * brightness));
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
