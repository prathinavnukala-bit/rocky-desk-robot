"""
Finds Rocky's boards on the LAN when their DHCP address has changed. The
router has reassigned both ESP32 boards' IPs repeatedly (main body went
.12 -> .16 -> .5 -> .2 -> .7; camera .9 -> .2 -> .11 -> .17), and every time it
silently broke the whole pipeline until someone edited a hardcoded IP.
This is the code-side fix: on a failed connection, scan the subnet for the
board instead of assuming its last address.
"""
import socket
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def _port_open(ip, port, timeout=1.5):  # these boards' Wi-Fi latency spikes
    # past 0.5s at times (measured ~536ms pings), so a tighter timeout
    # missed a healthy camera
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def resolve_local(hostname, timeout=3.0):
    """Resolve an mDNS name (e.g. rocky-body.local) to an IP, or None."""
    old = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        return socket.gethostbyname(hostname)
    except OSError:
        return None
    finally:
        socket.setdefaulttimeout(old)


def find_open_hosts(prefix, port):
    """IPs in `prefix` (e.g. "192.168.0.") with `port` accepting connections."""
    ips = [f"{prefix}{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=64) as ex:
        opened = list(ex.map(lambda ip: _port_open(ip, port), ips))
    return [ip for ip, ok in zip(ips, opened) if ok]


def find_rocky_body(prefix, port=8266):
    """The main ESP32-S3's raw TCP control port. Nothing else on this LAN
    should be listening on it; if several hosts are, the first is used."""
    hosts = find_open_hosts(prefix, port)
    return hosts[0] if hosts else None


def find_rocky_camera(prefix):
    """Any host on port 80 whose front page says ROCKY CAMERA (the router
    and other devices also serve port 80, so the page is checked)."""
    for ip in find_open_hosts(prefix, 80):
        try:
            page = urllib.request.urlopen(f"http://{ip}/", timeout=2).read(400).decode(errors="ignore")
        except Exception:
            continue
        if "ROCKY CAMERA" in page:
            return ip
    return None
