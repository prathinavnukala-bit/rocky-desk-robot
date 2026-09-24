"""
Shared, persistent WiFi/TCP connection to Rocky's ESP32-S3 body -- replaces
the earlier USB-serial link entirely. Serial turned out to be fundamentally
unreliable for the audio-heavy parts of this protocol even after extensive
tuning (see body.py's history comments for the full debugging story); WiFi/
TCP has real flow control and much higher real throughput. It's also a
required step toward Rocky eventually running fully untethered -- no USB
cable at all, only power.

Exposes the same small interface the rest of this codebase (body.py,
listen.py) was already built against -- write/read/readline/timeout/
reset_input_buffer -- via SocketLink, so none of that already-debugged
protocol logic (retries, busy-tracking, etc.) needed to be rewritten, only
the transport underneath it.
"""
import socket
import time

import netfind

HOST = "192.168.0.7"  # Last known IP of Rocky's ESP32-S3 on the home
                      # network. The router reassigns it (fifth time now:
                      # .12 -> .16 -> .5 -> .2 -> .7), so this is only a
                      # starting guess -- _connect_with_retry() rescans the
                      # subnet and updates it automatically if it's stale.
PORT = 8266

_link = None


class SocketLink:
    """Pyserial-like interface (write/read/readline/timeout/
    reset_input_buffer/is_open) over a raw TCP socket."""

    def __init__(self, sock):
        self._sock = sock
        self._buf = b""
        self._timeout = None
        self.eof = False

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = value
        self._sock.settimeout(value)

    @property
    def is_open(self):
        return self._sock.fileno() != -1 and not self.eof

    def write(self, data):
        self._sock.sendall(data)

    def _fill(self, n=4096):
        try:
            chunk = self._sock.recv(n)
        except socket.timeout:
            return False
        except OSError:
            self.eof = True  # reset/broken connection, not just a quiet line
            return False
        if not chunk:
            # The peer closed the connection (e.g. the ESP32 rebooted).
            # Previously this looked identical to a read timeout, so a dead
            # link was treated as "still connected, just idle" forever --
            # callers would spin on instant empty reads and never reconnect.
            self.eof = True
            return False
        self._buf += chunk
        return True

    def read(self, size):
        while len(self._buf) < size:
            if not self._fill():
                break
        result, self._buf = self._buf[:size], self._buf[size:]
        return result

    def readBytes(self, buf_or_size):
        """Mirrors the small subset of pyserial's readBytes-like usage this
        codebase relies on: read(size) already covers it, kept as an alias
        for readability at call sites ported from the old serial code."""
        return self.read(buf_or_size)

    def readline(self):
        while b"\n" not in self._buf:
            if not self._fill():
                result, self._buf = self._buf, b""
                return result
        idx = self._buf.index(b"\n") + 1
        result, self._buf = self._buf[:idx], self._buf[idx:]
        return result

    def reset_input_buffer(self):
        self._buf = b""
        self._sock.settimeout(0)
        try:
            while True:
                chunk = self._sock.recv(4096)
                if not chunk:
                    break
        except (BlockingIOError, socket.timeout, OSError):
            pass
        finally:
            self._sock.settimeout(self._timeout)

    def close(self):
        try:
            self._sock.close()
        except OSError:
            pass


def get_connection():
    global _link
    if _link is None or not _link.is_open:
        _link = _connect_with_retry()
    return _link


def _enable_keepalive(sock):
    """Makes a dead connection detectable within seconds.

    Confirmed live: when the ESP32 reboots or drops off Wi-Fi it never sends
    a FIN, so this side kept a half-open connection that looks perfectly
    healthy -- reads just time out with nothing new, exactly like an idle
    room -- and brain.py waited on it forever while the freshly-booted
    board sat listening for a new client. Every "not responding to Hey
    Rocky" after a board reset/Wi-Fi flap looked like that. With keepalive,
    the rebooted board answers our probe with a reset (or the probes go
    unanswered), recv() then errors, SocketLink marks itself eof, and the
    listen loop reconnects. Total detection time is roughly idle + interval
    * count = 10 + 5*3 = 25s worst case, near-instant for a reboot."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, value in (("TCP_KEEPALIVE", 10),   # macOS name for idle seconds
                        ("TCP_KEEPIDLE", 10),     # Linux name
                        ("TCP_KEEPINTVL", 5),
                        ("TCP_KEEPCNT", 3)):
        opt = getattr(socket, name, None)
        if opt is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, opt, value)
            except OSError:
                pass


def _handshake(link):
    """Sends one harmless command (re-draws the neutral face) right after
    every new connection and consumes its "Ready.".

    Why: the firmware sets awaitingBrainResponse after it hears the wake
    word and only clears it when a command arrives from RockyBrain -- but
    nothing clears it when a NEW connection is accepted. If the previous
    connection dropped while that flag was set (a restart mid-turn, a
    desync reconnect), the new session's flag stays set and the board
    silently stops listening for "Hey Rocky" forever, because brain.py sits
    idle waiting for a wake event and never sends a command to clear it.
    Confirmed as the cause of "not responding to Hey Rocky" with a healthy
    network and a connected brain. Any command clears it, so send one."""
    try:
        link.timeout = 3
        link.write(b"face neutral\n")
        for _ in range(4):
            if link.readline().strip() == b"Ready.":
                break
    except OSError:
        pass


def _connect_with_retry(max_wait=8):
    """A successful TCP connect means the ESP32's control server is up and
    accepting -- unlike the old serial link, there's no ambiguity about
    whether the board has actually finished booting, so no readiness-probe
    dance is needed here, just retry the connect itself if the board hasn't
    started listening yet (e.g. mid-boot).

    If the first attempt fails, HOST may simply be stale (the router has
    reassigned this board's address many times already) -- scan the subnet
    once for the control port and switch to whatever answers, instead of
    failing until someone edits HOST by hand.

    max_wait was 25s until this was measured live: with the board genuinely
    offline (WiFi dropped, not just moved), every single call -- and nearly
    every voice command starts with one, via body.set_expression() -- blocked
    for a confirmed 27 seconds before failing. That reads as "Rocky is
    stuck," not "Rocky can't reach its body right now," and it repeats on
    every command until the board comes back. 8s is enough for the mDNS
    lookup + one subnet scan + a couple of quick retries; past that, the
    board just isn't there right now and waiting longer doesn't help --
    failing fast lets the caller (or the user) actually notice and act."""
    global HOST
    deadline = time.time() + max_wait
    last_err = None
    scanned = False
    while time.time() < deadline:
        try:
            sock = socket.create_connection((HOST, PORT), timeout=2)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            _enable_keepalive(sock)
            link = SocketLink(sock)
            _handshake(link)
            link.timeout = 10  # default read timeout -- callers override as needed
            return link
        except OSError as e:
            last_err = e
            if not scanned:
                scanned = True
                # The firmware advertises itself as rocky-body.local (mDNS),
                # which macOS resolves natively and instantly, whatever DHCP
                # address the router gave it. A subnet scan is only the
                # fallback if that name doesn't resolve.
                found = netfind.resolve_local("rocky-body.local") or \
                        netfind.find_rocky_body(HOST.rsplit(".", 1)[0] + ".", PORT)
                if found and found != HOST:
                    print(f"Rocky moved: {HOST} -> {found} (found by network scan)")
                    HOST = found
                    continue
                else:
                    # The board isn't answering anywhere on the subnet --
                    # genuinely offline, not just moved. No point burning
                    # the rest of the budget re-trying a host that just
                    # failed a full scan; fail now instead of limping
                    # through several more multi-second timeouts.
                    break
            time.sleep(0.5)
    raise ConnectionError(f"Could not connect to Rocky at {HOST}:{PORT}: {last_err}")


def close():
    global _link
    if _link is not None and _link.is_open:
        _link.close()
    _link = None


def reconnect():
    """Drops the current connection (if any) and establishes a fresh one.
    This is the actual fix for protocol desync (confirmed live: a response
    that was merely slow, not lost, made a caller retry by resending onto
    the same still-open connection -- the ESP32 ended up with two
    overlapping transfers on one stream, and every read after that pulled
    garbage bytes indefinitely, with nothing ever recovering on its own).
    Rather than try to detect and parse around a corrupted stream, start
    over with a known-clean TCP connection -- confirmed safe with the
    current firmware: main.cpp's loop() re-accepts a new client the moment
    the old one disconnects (`if (!controlClient || !controlClient.connected())`),
    so both ends always land back in a clean state. Callers should use this
    instead of a plain resend whenever a response times out or looks
    malformed."""
    close()
    return get_connection()


# Busy-tracking retained from the serial era -- no longer strictly needed now
# that TCP has real flow control/ordering, but harmless to keep as a
# defensive no-op unless something explicitly calls mark_busy_for again.
_busy_until = 0.0


def mark_busy_for(seconds):
    global _busy_until
    _busy_until = max(_busy_until, time.time() + seconds)


def wait_until_free():
    """Sleeps out any fire-and-forget command's estimated response time,
    THEN discards whatever landed in the socket buffer while we waited.

    That second step is the fix for a real, reproduced bug: a fire-and-
    forget command (set_expression's "face curious\n", say_caption) never
    reads its own "Ready." -- it just marks busy time and trusts the NEXT
    caller to sleep past it. That was fine as long as the next caller's own
    read loop tolerated an extra stray line. It is NOT fine for play_speech:
    its _send_and_confirm() stops at the FIRST "Ready." it sees, with no way
    to tell it apart from a DIFFERENT command's leftover one. Reproduced
    live: set_expression() right before a multi-sentence reply left its
    "Ready." sitting unread; the first sentence's play_speech() consumed
    THAT stray "Ready." and returned success immediately, while the
    firmware was still physically mid-playback of a command it hadn't even
    started -- the real "Speech playback done."/"Ready." for that sentence
    then arrived late and sat unread until the NEXT command's turn (in
    practice, the follow-up "listen"), which is exactly the "Timed out
    waiting for audio start." failures seen after nearly every real turn.
    Discarding stray bytes here, once, centrally, fixes it for every caller
    of wait_until_free() at once rather than each one having to remember."""
    global _busy_until
    remaining = _busy_until - time.time()
    _busy_until = 0.0
    if remaining > 0:
        time.sleep(remaining)
    if _link is not None and _link.is_open:
        _link.reset_input_buffer()
