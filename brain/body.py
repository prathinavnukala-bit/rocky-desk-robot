"""
Sends movement and OLED face/caption commands to Rocky's ESP32-S3 body over
the shared WiFi control link (esp_link.py) also used by listen.py for the
mic -- closes the perception-to-body gap: Rocky can see/hear/talk, and now
that also means moving and putting captions on its own face, not just the
Mac's speakers.

The firmware only exposes locomotion primitives (stand, rest, forward,
backward, left, right) -- no dedicated expressive gestures like a wave or
nod exist yet. Reactions here are built from those primitives.

This used to run over USB serial. That was replaced with WiFi/TCP because
serial turned out to be fundamentally unreliable for the audio-heavy parts
of this protocol -- see play_speech()'s history comment below for the full
story. WiFi is also a required step toward Rocky eventually running fully
untethered (no USB cable at all, only power).

The ESP32 processes each command synchronously (each gait blocks for ~2-4s of
servo delays before it reads the next line) -- commands sent back-to-back
just queue on the connection and run in order, so no artificial pacing is
needed between writes here.
"""
import time
import wave

import esp_link

VALID_COMMANDS = (
    "stand", "rest", "forward", "backward", "left", "right",
    "wave", "dance", "swim", "point", "pushup", "bow", "cute", "freaky",
    "worm", "shake", "shrug", "dead", "crab", "nod", "blink",
)


def _reconnect_quietly():
    """esp_link.reconnect() raises ConnectionError when the board is
    unreachable. Every caller here calls it from inside an error handler
    whose whole contract is "return False, never raise" -- so an offline
    board used to escape as an unhandled exception instead (confirmed live:
    brain.py crashed at startup on its boot-greeting wave, and a speech
    worker thread died the same way, whenever the board was down). The next
    real command reconnects on its own, so failing here is harmless."""
    try:
        return esp_link.reconnect()
    except Exception as e:
        print(f"(board unreachable, will retry on next command: {e})")
        return None


def _wait_for_ready(ser, timeout_s=15):
    """Blocks until the ESP32's "Ready." response for the command just sent,
    reading (and discarding) anything in between. Movement commands wait for
    this -- not fire-and-forget -- specifically so nothing is left unread in
    the buffer for the next caller (e.g. listen.py's capture_audio()) to trip
    over. Confirmed necessary: without this, a body command immediately
    followed by a mic capture call reproducibly broke the capture, because
    the gait's own multi-second-delayed "Ready." arrived after the capture
    had already started reading and got misread as part of its response."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "Ready.":
            return True
    return False


def send_commands(commands, _retry=True):
    """Sends a sequence of movement commands, waiting for each to actually
    finish (see _wait_for_ready) before sending the next or returning.

    The board's WiFi drops and reconnects in bursts (confirmed via
    rocky_brain.log: runs of a dozen+ back-to-back connect timeouts,
    resolving within a few seconds once the board rejoins). A command sent
    right in the middle of one of those windows used to just fail silently
    -- callers like the boot-greeting wave or the idle-transition "dead"
    never checked the return value, so the gesture was dropped with zero
    visibility, while speak() (which has its own retry) kept working and
    made it look like only the physical movement was broken. One
    reconnect-and-retry here, at the single choke point all commands go
    through, covers every caller instead of patching each call site."""
    bad = [c for c in commands if c not in VALID_COMMANDS]
    if bad:
        print(f"Unknown body command(s): {bad}")
        return False
    try:
        ser = esp_link.get_connection()
        esp_link.wait_until_free()  # in case a caption is still finishing
        for command in commands:
            ser.write(f"{command}\n".encode())
            if not _wait_for_ready(ser):
                # A timed-out "Ready." leaves the connection's state
                # ambiguous -- the response might still arrive late and
                # desync the next command's read. Reconnect rather than
                # limping along on a connection that might be corrupted
                # (see esp_link.reconnect's docstring for the confirmed
                # failure mode this prevents).
                print(f"Body command '{command}' timed out -- reconnecting.")
                _reconnect_quietly()
                if _retry:
                    print(f"Retrying body command(s) {commands} once after reconnect.")
                    return send_commands(commands, _retry=False)
                return False
        return True
    except Exception as e:
        print(f"Body command failed: {e}")
        _reconnect_quietly()
        if _retry:
            print(f"Retrying body command(s) {commands} once after reconnect.")
            return send_commands(commands, _retry=False)
        return False


def send_command(command):
    return send_commands([command])


def greet_gesture():
    """A little happy wiggle for when Rocky recognizes someone."""
    return send_command("wave")


def _volume_command(command, _retry=True):
    """Sends "volume" (query) or "volume:<0-100>" (set) and returns the
    percentage the board reports, or None on failure. A stale socket gets one
    reconnect-and-retry so the command isn't lost."""
    try:
        ser = esp_link.get_connection()
        esp_link.wait_until_free()
        ser.timeout = 5
        ser.write(f"{command}\n".encode())
        value = None
        deadline = time.time() + 6
        while time.time() < deadline:
            line = ser.readline().decode(errors="ignore").strip()
            if line.startswith("VOLUME "):
                try:
                    value = int(line.split()[1])
                except ValueError:
                    pass
            if line == "Ready." and value is not None:
                return value   # a bare "Ready." with no VOLUME line is a stale one from an earlier command
        _reconnect_quietly()
        return _volume_command(command, False) if (value is None and _retry) else value
    except Exception as e:
        print(f"Volume command failed: {e}")
        _reconnect_quietly()
        return _volume_command(command, False) if _retry else None


def get_volume():
    return _volume_command("volume")


def set_volume(pct):
    """Sets the speaker volume 0-100% on the board (persisted in its flash,
    so it survives reboots). Returns the value the board applied."""
    pct = max(0, min(100, int(pct)))
    return _volume_command(f"volume:{pct}")


VALID_EXPRESSIONS = ("neutral", "happy", "sad", "curious", "sleepy", "surprised")


EXPRESSION_BUSY_S = 0.15  # setExpression()/renderFace() on the firmware is a
# single I2C framebuffer redraw with no delay() of its own -- a few ms in
# practice, this is a generous margin, not a real wait.


def set_expression(expr):
    """Changes Rocky's OLED eye expression. The firmware (main.cpp) has
    fully implemented this for a while -- drawExpressionEyes() supports all
    six expressions below, and setExpression()/"face <expr>" has existed as
    a command since the OLED face was built -- but nothing on this side was
    ever calling it. setup() draws "neutral" once at boot and nothing ever
    changes currentExpression again, so Rocky's face was permanently stuck
    regardless of what was actually happening. This is the fix: an actual
    caller.

    Deliberately fire-and-forget (same busy-tracking pattern as
    say_caption(), not the blocking wait send_commands() uses) -- this now
    gets called at the start of every single turn, so blocking on a network
    round-trip here would add latency to the whole conversation's critical
    path for a change that's visually instant on the firmware. An unread
    "Ready." would normally risk desyncing the next command's read (see
    esp_link.reconnect's docstring for why that matters), but
    mark_busy_for() makes the next caller sleep out the response's arrival
    first, the same protection say_caption() already relies on."""
    return _set_expression(expr, _retry=True)


def _set_expression(expr, _retry):
    if expr not in VALID_EXPRESSIONS:
        print(f"Unknown expression: {expr}")
        return False
    try:
        ser = esp_link.get_connection()
        esp_link.wait_until_free()
        ser.write(f"face {expr}\n".encode())
        esp_link.mark_busy_for(EXPRESSION_BUSY_S)
        return True
    except Exception as e:
        print(f"Expression command failed: {e}")
        _reconnect_quietly()
        if _retry:
            # This is the fallback callers reach for specifically when a
            # pose command like "dead" already failed and retried once --
            # e.g. listen_and_respond()'s `if not body.send_command("dead"):
            # body.set_expression("sleepy")`. Without its own retry here,
            # a bad WiFi window could fail BOTH the pose and this fallback,
            # leaving the eyes stuck on whatever they last showed (visible
            # as "idle eyes not going to idle mode") with no further chance
            # to recover until the next turn happens to touch expression.
            return _set_expression(expr, _retry=False)
        return False


CHARS_PER_CHUNK = 10  # must match maxCharsPerChunk in the firmware
CHUNK_HOLD_S = 0.65   # must match holdMs in the firmware
CHUNK_OVERHEAD_S = 0.1  # margin for the I2C display refresh per chunk


def say_caption(text, duration_ms):
    """Sends spoken text as an OLED caption -- short phrase chunks that each
    appear, hold, then get replaced by the next (see firmware
    showCaptionChunks), not a scroll. duration_ms is still sent for protocol
    compatibility but isn't used for pacing -- chunks use a fixed hold time.
    Deliberately fire-and-forget (does not wait for the ESP32's "Ready.") so
    it runs in parallel with audio playback instead of blocking until all
    chunks finish. Marks the link busy for an estimate of the actual chunk
    sequence duration so the next command waits instead of racing the
    still-in-flight response (see esp_link.wait_until_free) -- this estimate
    must track the firmware's real per-chunk timing, not duration_ms, or the
    same race this was built to fix comes back."""
    try:
        ser = esp_link.get_connection()
        esp_link.wait_until_free()  # in case a previous caption is still finishing
        safe_text = text.replace("\n", " ").replace("\r", " ")
        ser.write(f"say:{duration_ms}:{safe_text}\n".encode())

        num_chunks = max(1, -(-len(safe_text) // CHARS_PER_CHUNK))  # ceil div
        esp_link.mark_busy_for(num_chunks * (CHUNK_HOLD_S + CHUNK_OVERHEAD_S) + 0.3)
        return True
    except Exception as e:
        print(f"Caption command failed: {e}")
        _reconnect_quietly()
        return False


SPK_SAMPLE_RATE = 22050  # must match SPK_SAMPLE_RATE in the ESP32 firmware


def _send_and_confirm(ser, timeout_s):
    """Reads lines until the ESP32's "Ready.", watching for a
    TRANSFER_INCOMPLETE report along the way (see firmware
    playStreamedSpeech -- it refuses to play anything unless every byte it
    was told to expect actually arrived, rather than risk playing glitchy
    audio from a partial/corrupted receive). Returns False if that happened,
    so the caller knows to retry rather than assuming silence meant success."""
    deadline = time.time() + timeout_s
    incomplete = False
    while time.time() < deadline:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("TRANSFER_INCOMPLETE"):
            incomplete = True
            print(f"play_speech: {line}")
        if line == "Ready.":
            return not incomplete
    return False


def play_speech(text, wav_path, max_attempts=3):
    """Sends a synthesized speech wav to the ESP32, which buffers the whole
    thing in PSRAM before playing it from its onboard speaker (firmware:
    playStreamedSpeech) -- this is what makes Rocky's voice come from its own
    body instead of the Mac's speakers, with the OLED caption advancing in
    step with actual playback progress. Blocks until the ESP32 confirms
    playback finished (same "Ready." pattern as send_commands) so the caller
    knows it's truly done, not just that the bytes were sent, and so nothing
    is left unread for the next command.

    History: this used to run over USB serial with the audio deliberately
    sent in tiny paced chunks (64 bytes/20ms), because serial silently
    dropped data at any faster pace -- confirmed by direct testing
    (a checksum-comparison debug command, "checktest") after the first
    "proven reliable" config turned out to only be reliable in isolation and
    failed 9/9 times under real load. That pacing made a single utterance
    take ~35s just to transfer. Serial has since been replaced entirely with
    a WiFi/TCP control link (esp_link.py), which has real flow control, so
    the whole payload is now sent in one write -- no artificial pacing
    needed. The integrity check and retry logic stay as a defensive safety
    net regardless of transport; they should rarely if ever trigger now."""
    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            print(
                f"play_speech: expected mono 16-bit audio, got "
                f"{wf.getnchannels()}ch/{wf.getsampwidth() * 8}bit -- may sound wrong."
            )
        if wf.getframerate() != SPK_SAMPLE_RATE:
            print(
                f"play_speech: wav is {wf.getframerate()}Hz, firmware expects "
                f"{SPK_SAMPLE_RATE}Hz -- pitch/speed will be off."
            )
        pcm_data = wf.readframes(wf.getnframes())

    safe_text = text.replace("\n", " ").replace("\r", " ")
    audio_s = len(pcm_data) / 2 / SPK_SAMPLE_RATE

    for attempt in range(1, max_attempts + 1):
        try:
            # Never resend onto a connection that might still be mid-
            # processing the previous attempt. Confirmed live: a response
            # that was merely slow (not lost -- e.g. under real network
            # congestion) triggered a retry-by-resend on the same still-open
            # connection, leaving the ESP32 with two overlapping transfers
            # on one stream -- every read after that pulled garbage bytes
            # indefinitely, with nothing recovering on its own until this
            # fix. Reconnecting guarantees a clean slate on both ends (see
            # esp_link.reconnect's docstring).
            ser = esp_link.reconnect() if attempt > 1 else esp_link.get_connection()
            esp_link.wait_until_free()  # in case a previous caption/command is still finishing

            # sendall()'s timeout is a limit on the WHOLE transfer, and it was
            # silently inheriting whatever ser.timeout the last caller left
            # (6s from the idle wake-word wait). A 200-400KB clip over a
            # Wi-Fi link that sometimes hiccups for seconds can exceed that,
            # so the send raised "timed out" halfway through -- leaving the
            # firmware holding a partial transfer -- and every retry failed
            # the same way (reproduced: 3 timeouts in a row, a 63s stall).
            # Give the send a generous, explicit limit, then a short one for
            # reading the confirmation.
            ser.timeout = 30
            ser.write(f"speak:{len(pcm_data)}:{safe_text}\n".encode())
            ser.write(pcm_data)
            ser.timeout = 5

            if _send_and_confirm(ser, timeout_s=audio_s + 15):
                return True
            print(
                f"play_speech: attempt {attempt}/{max_attempts} incomplete"
                + (", retrying..." if attempt < max_attempts else ", giving up.")
            )
        except Exception as e:
            print(f"Speech playback failed (attempt {attempt}/{max_attempts}): {e}")

    _reconnect_quietly()  # leave a clean connection behind even after giving up
    return False
