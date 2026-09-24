import os
import subprocess
import sys
import tempfile
import threading
import wave

VOICE_DIR = os.path.join(os.path.dirname(__file__), "voices")
VOICE_MODEL = os.path.join(VOICE_DIR, "en_GB-alan-medium.onnx")
VOICE_CONFIG = VOICE_MODEL + ".json"

# Piper's raw output run through a light ffmpeg chain to sound less like a
# stock assistant and more like a synthesized, translated voice.
FILTER_PRESETS = {
    "subtle": "asetrate=22050*0.95,aresample=22050,atempo=1.0526,"
              "chorus=0.6:0.9:40:0.3:0.2:1.5",
    "pronounced": "asetrate=22050*0.90,aresample=22050,atempo=1.1111,"
                  "chorus=0.5:0.85:55:0.4:0.3:2,aecho=0.7:0.6:25:0.15",
    # Pitch shift alone, no chorus/echo -- isolates what the pitch itself sounds like.
    "pitch_moderate": "asetrate=22050*0.95,aresample=22050,atempo=1.0526",
    "pitch_deep": "asetrate=22050*0.90,aresample=22050,atempo=1.1111",
    "none": None,
}

VOICE_PRESET = "none"

# Phoneme length multiplier -- higher is slower, more deliberate speech.
# Piper's default is 1.0; slightly below it reads closer to natural human pace.
LENGTH_SCALE = 0.92
# Extra pause after each sentence, in seconds, for a more considered cadence.
SENTENCE_SILENCE = 0.25
# Piper's "expressiveness" knobs: noise_scale varies pitch/timbre from one
# utterance to the next, noise_w_scale varies phoneme timing/rhythm. None =
# the voice's own defaults.
NOISE_SCALE = None
NOISE_W_SCALE = None

# ---------------------------------------------------------------------------
# In-process synthesis. The old path launched a fresh `python -m piper`
# process for EVERY sentence, which re-loaded the voice model each time:
# measured ~1.0s per sentence. Keeping the voice loaded in this process
# measures 0.06-0.14s -- roughly 10x faster, and that time is on the critical
# path of every single reply. The subprocess path remains as a fallback.
# ---------------------------------------------------------------------------
_voice_cache = {}
_cache_lock = threading.Lock()
_synth_lock = threading.Lock()  # one synthesis at a time; each takes ~0.1s


def _get_voice(model_path):
    from piper import PiperVoice
    with _cache_lock:
        voice = _voice_cache.get(model_path)
        if voice is None:
            voice = PiperVoice.load(model_path, config_path=model_path + ".json")
            _voice_cache[model_path] = voice
        return voice


def warmup(model_path=None):
    """Load the voice and run one throwaway synthesis so the first real reply
    doesn't pay the model-load / first-inference cost."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        synthesize("Ready.", path, voice_model=model_path)
        os.remove(path)
        return True
    except Exception as e:
        print(f"TTS warm-up failed (will fall back to per-call loading): {e}")
        return False


def _synthesize_in_process(text, raw_path, model_path, length_scale, sentence_silence,
                           noise_scale, noise_w_scale):
    from piper import SynthesisConfig
    voice = _get_voice(model_path)
    kwargs = {"length_scale": length_scale}
    if noise_scale is not None:
        kwargs["noise_scale"] = noise_scale
    if noise_w_scale is not None:
        kwargs["noise_w_scale"] = noise_w_scale
    cfg = SynthesisConfig(**kwargs)
    with _synth_lock:
        with wave.open(raw_path, "wb") as wf:
            voice.synthesize_wav(text, wf, syn_config=cfg)
    if sentence_silence and sentence_silence > 0:
        with wave.open(raw_path, "rb") as r:
            params, frames = r.getparams(), r.readframes(r.getnframes())
        pad = b"\x00" * (int(params.framerate * sentence_silence) * params.sampwidth * params.nchannels)
        with wave.open(raw_path, "wb") as w:
            w.setparams(params)
            w.writeframes(frames + pad)


def _synthesize_subprocess(text, raw_path, model_path, length_scale, sentence_silence):
    subprocess.run(
        [
            sys.executable, "-m", "piper",
            "-m", model_path, "-c", model_path + ".json", "-f", raw_path,
            "--length-scale", str(length_scale),
            "--sentence-silence", str(sentence_silence),
        ],
        input=text.encode(),
        check=True,
        capture_output=True,
    )


def synthesize(text, out_path, preset=VOICE_PRESET, length_scale=LENGTH_SCALE,
                sentence_silence=SENTENCE_SILENCE, voice_model=None,
                noise_scale=None, noise_w_scale=None):
    """Render text to a WAV file at out_path using Piper + Rocky's voice processing."""
    model_path = voice_model or VOICE_MODEL
    if noise_scale is None:
        noise_scale = NOISE_SCALE
    if noise_w_scale is None:
        noise_w_scale = NOISE_W_SCALE
    raw_path = out_path + ".raw.wav"
    try:
        _synthesize_in_process(text, raw_path, model_path, length_scale,
                               sentence_silence, noise_scale, noise_w_scale)
    except Exception as e:
        print(f"In-process TTS failed ({e}); using the slower subprocess path.")
        _synthesize_subprocess(text, raw_path, model_path, length_scale, sentence_silence)

    filt = FILTER_PRESETS.get(preset)
    if filt:
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path, "-af", filt, out_path],
            check=True,
            capture_output=True,
        )
        os.remove(raw_path)
    else:
        os.replace(raw_path, out_path)

    return out_path


def speak(text, preset=VOICE_PRESET):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name
    try:
        synthesize(text, out_path, preset=preset)
        subprocess.run(["afplay", out_path])
    finally:
        if os.path.exists(out_path):
            os.remove(out_path)


if __name__ == "__main__":
    speak("Hello. I am Rocky. Good to meet you.")
