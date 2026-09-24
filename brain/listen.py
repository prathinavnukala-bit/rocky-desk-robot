import re
import wave

import esp_link

SAMPLE_RATE = 16000

# If the connection has desynced (confirmed real failure mode: a slow-not-
# lost response elsewhere triggered a resend that corrupted the protocol
# stream -- see esp_link.reconnect's docstring), reading it back here just
# produces a wall of garbage lines that never says START_AUDIO. Bail out and
# reconnect after this many unexpected non-empty lines rather than looping
# on it forever -- normal operation should essentially never see more than
# a handful of real device log lines in a row here.
MAX_UNEXPECTED_LINES = 50

# The board reports whether its voice-activity detector ever heard sustained
# speech in a recording. Without any, the audio is room noise, and Whisper
# invents words for noise ("You", "Thanks for watching") that Rocky would then
# answer -- and the follow-up listening chained that into a conversation with
# nobody. The audio is still read off the connection (it must be, to keep the
# stream aligned) but reported as "nothing captured".
_VAD_RESULT = re.compile(r"VAD_RESULT speech=(\d)")

def capture_audio(output_path="recording.wav"):
    ser = esp_link.get_connection()
    # Waits out any fire-and-forget command still finishing (e.g. a caption
    # mid-scroll) rather than racing its straggler response.
    esp_link.wait_until_free()

    print("Sending 'listen' command...")
    # The device's next line (DONE LISTENING / START_AUDIO) only arrives once
    # recording ends -- up to MAX_RECORD_SECONDS (25s, see main.cpp) later.
    # This used to inherit whatever timeout the previous call left (5-6s),
    # so a longer utterance was misread as "timed out waiting for audio
    # start" -- kept comfortably above the firmware's own cap.
    ser.timeout = 30
    ser.reset_input_buffer()
    ser.write(b"listen\n")

    unexpected = 0
    speech = True
    first_line = True
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        print(f"  [device] {line}")
        m = _VAD_RESULT.match(line)
        if m:
            speech = m.group(1) == "1"
        if line == "START_AUDIO":
            break
        if not line:
            if first_line:
                # Nothing at all came back, not even a stray leftover line --
                # a genuine 30s stall is implausible for this exchange (the
                # device always answers "LISTENING..." within milliseconds),
                # so this is almost certainly a dropped/desynced connection,
                # not real silence. Reconnect now instead of waiting for the
                # next wake cycle to notice -- confirmed live that leaving a
                # connection in this state cost one whole turn before the
                # passive is_open check caught it on its own.
                print("No response at all -- connection looks broken, reconnecting.")
                esp_link.reconnect()
            else:
                print("Timed out waiting for audio start.")
            return None
        first_line = False
        unexpected += 1
        if unexpected >= MAX_UNEXPECTED_LINES:
            print("Too many unexpected lines -- connection looks desynced, reconnecting.")
            esp_link.reconnect()
            return None

    try:
        byte_count_line = ser.readline().decode(errors="ignore").strip()
        byte_count = int(byte_count_line)
    except ValueError:
        print(f"Bad byte-count line ({byte_count_line!r}) -- connection looks desynced, reconnecting.")
        esp_link.reconnect()
        return None
    print(f"Expecting {byte_count} bytes of audio data...")

    audio_data = ser.read(byte_count)
    print(f"Received {len(audio_data)} bytes.")

    if not speech:
        print("No speech was detected in that recording -- ignoring it.")
        return None

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)

    print(f"Saved to {output_path}")
    return output_path

def wait_for_wake_and_capture(output_path="recording.wav", on_idle_tick=None):
    """Like capture_audio(), but doesn't send the 'listen' command -- the
    ESP32 now runs its own on-device "Hey Rocky" wake-word detector
    (wake_word.h/tickWakeWord() in the firmware) and pushes this exact same
    START_AUDIO/byte-count/PCM protocol unprompted the moment it hears the
    wake word, instead of waiting to be told to listen. So this just blocks
    reading lines until that shows up, for as long as it takes -- there's no
    fixed timeout, since the wake word could come at any time.

    on_idle_tick, if given, is called on every read that times out with
    nothing new -- used for the proactive "greet on sight" camera check
    (see brain.py's maybe_greet_on_sight), piggybacked onto this loop
    instead of a separate thread so it can never race with this same
    connection's reads/writes (see esp_link.py's docstring for why
    concurrent access to this link has caused real bugs before)."""
    ser = esp_link.get_connection()
    # Waits out any fire-and-forget command still finishing (e.g. a caption
    # mid-scroll) rather than racing its straggler response.
    esp_link.wait_until_free()

    print("Waiting for wake word...")
    # Shortened from 30s: purely client-side read pacing, doesn't change
    # firmware wake-word behavior at all -- just how often on_idle_tick runs.
    ser.timeout = 6

    unexpected = 0
    speech = True
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"  [device] {line}")
        m = _VAD_RESULT.match(line)
        if m:
            speech = m.group(1) == "1"
        if line == "START_AUDIO":
            break
        if not line:
            if ser.eof:
                # The board closed the connection (reboot, WiFi drop).
                # That used to look exactly like an idle timeout, so this
                # loop would spin forever on instant empty reads without
                # ever reconnecting.
                print("Connection to Rocky dropped -- reconnecting.")
                ser = esp_link.reconnect()
                ser.timeout = 6
                continue
            # Otherwise an empty line means this read timed out with
            # nothing new -- not a failure, just keep waiting.
            if on_idle_tick is not None:
                on_idle_tick()
            continue
        unexpected += 1
        if unexpected >= MAX_UNEXPECTED_LINES:
            print("Too many unexpected lines -- connection looks desynced, reconnecting.")
            esp_link.reconnect()
            return None

    try:
        byte_count_line = ser.readline().decode(errors="ignore").strip()
        byte_count = int(byte_count_line)
    except ValueError:
        print(f"Bad byte-count line ({byte_count_line!r}) -- connection looks desynced, reconnecting.")
        esp_link.reconnect()
        return None
    print(f"Expecting {byte_count} bytes of audio data...")

    audio_data = ser.read(byte_count)
    print(f"Received {len(audio_data)} bytes.")

    if not speech:
        print("No speech was detected in that recording -- ignoring it.")
        return None

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data)

    print(f"Saved to {output_path}")
    return output_path

if __name__ == "__main__":
    wait_for_wake_and_capture()
