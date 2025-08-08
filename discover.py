import socket
import struct

import netifaces

# ArtNet Constants
ARTNET_PORT = 6454
ARTNET_POLL = (
    b"Art-Net\x00"
    + struct.pack("<H", 0x2000)
    + struct.pack("<H", 14)
    + struct.pack("B", 0x00)
    + struct.pack("B", 0x00)
)


def get_local_interfaces():
    """
    Retrieve local IP addresses from all network interfaces.
    """
    interfaces = []
    for iface in netifaces.interfaces():
        try:
            iface_info = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in iface_info:
                for link in iface_info[netifaces.AF_INET]:
                    local_ip = link.get("addr")
                    broadcast_ip = link.get("broadcast")
                    if local_ip and broadcast_ip:
                        interfaces.append((iface, local_ip, broadcast_ip))
        except (ValueError, KeyError):
            continue
    return interfaces


def parse_artnet_reply(data):
    """
    Parse an ArtNet Poll Reply packet to extract the UDP port.
    """
    try:
        if not data.startswith(b"Art-Net\x00"):
            return None

        # Check if it's an ArtPollReply (opcode 0x2100)
        opcode = struct.unpack_from("<H", data, 8)[0]
        if opcode != 0x2100:
            return None

        # UDP Port is at offset 14-15 in the packet
        udp_port = struct.unpack_from("<H", data, 14)[0]
        return udp_port

    except Exception as e:
        print(f"⚠️ Failed to parse reply: {e}")
        return None


def discover_artnet_on_interface(local_ip, broadcast_ip):
    """
    Discover ArtNet controllers on a specific interface.
    """
    print(f"\n🔍 Searching on Interface: {local_ip} → {broadcast_ip}")
    discovered_controllers = []

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((local_ip, ARTNET_PORT))
        sock.settimeout(1.0)

        try:
            # Send ArtNet Poll packet to the broadcast address
            sock.sendto(ARTNET_POLL, (broadcast_ip, ARTNET_PORT))
            print(f"📡 Poll packet sent on {broadcast_ip}. Listening for replies...")

            while True:
                try:
                    data, addr = sock.recvfrom(1024)
                    if data.startswith(b"Art-Net\x00"):
                        ip_address = addr[0]
                        udp_port = parse_artnet_reply(data) or ARTNET_PORT
                        if ip_address not in discovered_controllers:
                            print(f"🎯 Found controller at {ip_address} " f"(UDP Port: {udp_port})")
                            discovered_controllers.append((ip_address, udp_port))
                except socket.timeout:
                    break  # Stop listening after timeout

        except Exception as e:
            print(f"⚠️ Error on interface {local_ip}: {e}")

    return discovered_controllers


def main():
    interfaces = get_local_interfaces()
    if not interfaces:
        print("❌ No network interfaces detected.")
        return

    print("✅ Detected Network Interfaces:")
    for iface, local_ip, broadcast_ip in interfaces:
        print(f" - {iface}: Local IP: {local_ip}, Broadcast IP: {broadcast_ip}")

    all_discovered_controllers = []
    for iface, local_ip, broadcast_ip in interfaces:
        controllers = discover_artnet_on_interface(local_ip, broadcast_ip)
        all_discovered_controllers.extend(controllers)

    if all_discovered_controllers:
        print("\n✅ Discovered ArtNet Controllers:")
        for i, (controller, port) in enumerate(set(all_discovered_controllers), 1):
            print(f"{i}. {controller} (UDP Port: {port})")
    else:
        print("\n❌ No ArtNet controllers found on any interface.")


if __name__ == "__main__":
    main()
