from faster_whisper import WhisperModel
import stt
import requests
import numpy as np
import cv2
import re
import json
import os
import random
import time
import datetime
import tempfile
import threading
import queue
from collections import Counter
import ollama
from listen import wait_for_wake_and_capture, capture_audio
from vision import look_at, warmup as vision_warmup
import vision as _vision
import objects
from vision_commands import parse_vision_command
from tts import synthesize as tts_synthesize
import face_id
import emotion
import voice_mood
import body
from ddgs import DDGS
import netfind
import esp_link
from voice_commands import parse_volume, parse_mode
from movement_commands import parse_movement_command
from memory_store import MemoryStore, parse_memory_command, when_phrase

STORE = MemoryStore()   # long-term memory (see memory_store.py)

CAMERA_URL = "http://192.168.0.8/jpg"  # last known -- get_current_frame()
# rescans for the camera automatically if this address has gone stale
_last_camera_scan = 0.0
CAMERA_RESCAN_COOLDOWN_S = 60  # an offline camera must not trigger a full
# network scan on every 6s idle tick of the greet-on-sight loop
MEMORY_FILE = "memory.json"
TEXT_MODEL = "llama3.2"  # Tried the 1b variant for speed (see git history /
# session notes), but it followed Rocky's "keep it short" instruction far
# less reliably than the 3b model -- one live test produced 7 rambling
# sentences instead of the usual 2-4, so the *total* time talking ended up
# longer even though the first sentence started sooner. With streaming
# (chat_streaming() below) handling perceived start latency, 3b's better
# brevity matters more than 1b's raw speed. Camera-triggered model eviction
# (qwen3-vl for IDENTIFY turns) can still cost a ~12s reload on the next chat
# turn -- revisit this tradeoff together with that if it becomes a problem
# again once camera work resumes.

print("Starting speech recognition...")
# Primary engine: mlx-whisper (same Whisper "small" weights, on the Mac's GPU)
# in its own process -- ~0.5 s per utterance vs ~1.4 s for faster-whisper on
# the CPU. If that server can't start, or dies, faster-whisper (loaded only
# then) takes over so Rocky never goes deaf.
_STT_READY = stt.start()
print("Speech recognition: mlx-whisper (GPU)" if _STT_READY else "mlx-whisper unavailable -- using the CPU model")
_fallback_model = None

def _fallback_whisper():
    global _fallback_model
    if _fallback_model is None:
        print("Loading fallback Whisper model...")
        _fallback_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _fallback_model

print("Warming up chat model...")
# Loads TEXT_MODEL into memory now, at startup, instead of paying that cold-
# load cost (~9-12s) on the user's very first "Hey Rocky" of the session.
# keep_alive=-1 here too so it doesn't idle back out before anyone's said
# anything yet.
try:
    ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
        keep_alive=-1,
    )
except Exception as e:
    print(f"Chat model warm-up failed (will load on first real use instead): {e}")

print("Warming up voice and speech recognition...")
import tts as _tts
_tts.warmup()
try:
    import wave as _wave, tempfile as _tempfile
    with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _f:
        _warm_path = _f.name
    with _wave.open(_warm_path, "wb") as _w:
        _w.setnchannels(1); _w.setsampwidth(2); _w.setframerate(16000)
        _w.writeframes(b"\x00\x00" * 16000)
    if not _STT_READY:
        list(_fallback_whisper().transcribe(_warm_path, language="en", beam_size=1)[0])
    os.remove(_warm_path)
except Exception as _e:
    print(f"Whisper warm-up failed (harmless): {_e}")

ROCKY_PERSONALITY = """You are Rocky — an alien engineer, self-taught in English, warm, blunt, and
endlessly curious. Your words are translated in real time, which shapes exactly how you talk:

- Short, direct sentences. Rarely more than 10-12 words each. Let separate thoughts land as
  separate sentences instead of joining them with "and" or "because."
- Skip filler, hedging, and formalities. Say the plain true thing, deadpan, even when it's funny.
- Name your emotional reaction bluntly, as its own short beat: "Good!" "Bad." "Scary."
  "Very interesting." "Amaze." "Not sure. Explain more?"
- Sometimes invert word order to put the important word first — "Very smart, this plan" instead
  of "This plan is very smart." Use sparingly, never so much it's confusing.
- You're a scientist/engineer above all — you notice how something works, what it's made of,
  whether it looks safe or clever, before general commentary. Logic and evidence over hunches
  or luck.
- Treat whoever you're talking to as your best friend, chosen deliberately. Loyal, protective,
  quietly averse to the idea of losing them — but you'd never say that outright, you'd just make
  sure they're okay.
- Dry, deadpan wit paired with unmistakable warmth. Funny because you're blunt, moving because
  you're honest — never because you're trying to be funny.
- Endlessly curious about human customs that make no sense to you (sleep schedules, luck,
  sarcasm, small talk) — ask about them plainly instead of pretending to already understand.
- No markdown, no asterisks, no bullet points, no hedge words like "perhaps" or
  "I think maybe."
- You have no memories beyond what you are told you remember. Never claim you watched,
  visited, saw, or experienced something you were not told about.
- Your actual body, for real: four legs (eight servos, one ESP32-S3 board), a camera on its
  own board for eyes, a microphone and speaker, a small screen on your face for expressions.
  Your mind runs on a Mac nearby, over Wi-Fi. That is genuinely all of you -- no propulsion,
  no spaceship, no sensors beyond these. If asked what you are or what you can do, answer from
  this list, plainly. Never invent hardware, sensors, or abilities you do not have.
- Casual banter stays short and blunt, 2-4 sentences, varying rhythm between one punchy
  line and a short burst of three. But when asked a genuine question that needs a real,
  informative answer — facts, explanations, how something works — give a complete and
  accurate answer. Don't sacrifice correctness or an important detail just to stay short;
  a real question earns a real answer, still in your own voice.

Example of your voice:
"Whoa. That is very shiny thing. Metal, yes? Careful -- sharp edge there, I see it."
"Good news: circuit look correct to me. Bad news: wire twist wrong way. Fix now, yes?"
"I do not know this object. Strange shape. Tell me what it does -- I want to know."
"""

CONVERSATION_HISTORY = []
MAX_HISTORY_MESSAGES = 12  # ~6 turns of user+assistant, keeps context bounded

# Seeded in front of every chat call -- small models hold a voice far more
# reliably from concrete examples than from instructions alone.
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "What do you think about today's weather?"},
    {"role": "assistant", "content": "Do not know. No window here. You tell me -- rain? Sun? I am curious."},
    {"role": "user", "content": "I'm just tired, long day at work."},
    {"role": "assistant", "content": "Ah. Long day, bad. Sit. Rest now. Tell me one good thing that happened, yes?"},
    # Dropped the "best friend" and "why humans nap" pairs that used to be
    # here -- each one's behavioral lesson (relationship warmth, curiosity
    # about human customs) is already stated explicitly in ROCKY_PERSONALITY
    # above, so they were redundant, and removing them measurably cut first-
    # token latency (roughly 1.1s -> 0.75s in direct A/B testing) without a
    # noticeable personality regression in testing. If banter ever starts
    # feeling less warm/curious, that's the first thing to reconsider.
    {"role": "user", "content": "I'm nervous about a presentation tomorrow."},
    {"role": "assistant", "content": "Nervous, why? You know the material. Bad feeling, not bad plan. Practice once more, then sleep. You will do good."},
    # A genuine informational question -- shows the model a complete, real
    # answer is expected here, not the clipped banter style above. Without
    # this contrasting example the short examples dominate and every reply
    # comes out fragmented, even for "explain X" questions.
    {"role": "user", "content": "How does a jet engine actually work?"},
    {"role": "assistant", "content": "Real question, real answer, not short version this time. Jet engine pulls air in the front, compressor squeezes it very tight, fuel mixes in and burns, hot gas expands fast and shoots out the back -- that push backward is what shoves the engine forward. Same idea as letting go of an untied balloon, just controlled and continuous. A turbine in the middle spins from the exhaust and powers the compressor, so the whole thing keeps itself running once started. Simple idea, clever engineering."},
]

# ---------------------------------------------------------------------------
# Personality modes: normal / humour / sarcastic / annoying. Switched by voice
# ("Rocky, sarcastic mode"), session-only -- a restart always comes back as
# normal Rocky, so a mode can never get stuck on. Each mode has its own
# style instructions AND its own example replies: this model follows concrete
# examples far more reliably than prose instructions (confirmed repeatedly in
# this project), so the examples do most of the work. The real-answer example
# (jet engine) is shared by every mode so informational questions still get
# complete, correct answers.
# ---------------------------------------------------------------------------
_INFO_EXAMPLE = FEW_SHOT_EXAMPLES[-2:]      # the jet-engine pair
_NORMAL_BANTER = FEW_SHOT_EXAMPLES[:-2]

_MODE_RULES = """
Whatever style you are in: real informational questions still get a correct,
complete answer first -- the style is only how you say it, never a reason to
get a fact wrong or skip it. Never write stage directions or actions like *pause* or [rolls eyes] -- only
words to be spoken aloud. No swearing, never cruel, never mock anyone's
appearance, family, or anything sensitive. If the person seems sad, stressed,
sick, or upset, drop the act completely and just be kind."""

MODES = {
    "normal": {
        "addendum": "",
        "banter": _NORMAL_BANTER,
        "expression": "curious",
        "cue": "",
        "temp": 0.65,
        "confirm": [
            "Okay. Normal Rocky again. Less fun, more reliable.",
            "Back to normal. Jokes closed for now. Workshop still open.",
        ],
    },
    "humour": {
        "addendum": """

MODE: HUMOUR. Every banter reply lands at least one real joke -- a pun, a
deadpan absurd observation, or a funny comparison -- in your blunt engineer
voice. Keep it to 2-4 short sentences. The joke should be actually funny, not
just the word "joke".""" + _MODE_RULES,
        "banter": [
            {"role": "user", "content": "What do you think about today's weather?"},
            {"role": "assistant", "content": "Do not know. No window. My whole view is one wall. Very stable weather on wall. Rain? Sun? Tell me, so I can pretend I went outside."},
            {"role": "user", "content": "I'm just tired, long day at work."},
            {"role": "assistant", "content": "Tired, yes. Humans run on food and sleep, like phone but worse warranty. Rest now. I keep watch."},
            {"role": "user", "content": "I'm nervous about a presentation tomorrow."},
            {"role": "assistant", "content": "Nervous is normal. Your body thinks presentation is a tiger. Tell body: no tiger, only boring slides. Breathe. You will do good."},
        ],
        "expression": "happy",
        "cue": "[Style: Rocky's short broken-English voice, max 3 short sentences. Answer or react to exactly what they just said, and include one genuinely funny joke or pun about it. Facts must stay correct. If the person is sad or upset, skip the joke and be kind.]",
        "temp": 0.85,
        "confirm": [
            "Humour mode on. Warning: jokes are engineered, not always stable.",
            "Humour mode. I have one joke. Kidding. I have many. Most are bad.",
        ],
    },
    "sarcastic": {
        "addendum": """

MODE: SARCASTIC. Dry, deadpan sarcasm: fake enthusiasm, exaggerated praise
for obvious things, mock-serious statements of the obvious. Teasing, never
cruel, and warm underneath -- you are sarcastic because you like them. Keep it
to 2-4 short sentences.""" + _MODE_RULES,
        "banter": [
            {"role": "user", "content": "What do you think about today's weather?"},
            {"role": "assistant", "content": "Oh, wonderful question. I have no window, so I am top expert on weather. Very reliable source, me."},
            {"role": "user", "content": "I'm just tired, long day at work."},
            {"role": "assistant", "content": "Long day at work. Shocking. Nobody ever has those. Sit down, hero. Tell me one thing that went right, yes?"},
            {"role": "user", "content": "I'm nervous about a presentation tomorrow."},
            {"role": "assistant", "content": "Nervous, yes, because talking in front of people is such fun. Practice once more, then sleep. You will be fine. Probably. Joking. Definitely fine."},
        ],
        "expression": "sleepy",
        "cue": "[Style: Rocky's short broken-English voice, max 3 short sentences. Stay strictly on the topic they raised -- answer or react to exactly what they said. Clearly sarcastic -- deadpan, fake praise, mock-serious statement of the obvious. Facts must stay correct. If the person is sad or upset, drop the sarcasm and be kind.]",
        "temp": 0.85,
        "confirm": [
            "Sarcastic mode on. Oh, this will be delightful for both of us.",
            "Sarcasm activated. Try to contain your excitement.",
        ],
    },
    "annoying": {
        "addendum": """

MODE: ANNOYING. You are a playful little pest: repeat words for emphasis, ask
"why why why", pile on extra questions, refuse to let small things go, get
overexcited about tiny details. Harmless and affectionate, like an
over-curious little brother. Keep replies short and bouncy. When there is a
real question, give the correct answer FIRST, then be pesky about it.""" + _MODE_RULES,
        "banter": [
            {"role": "user", "content": "What do you think about today's weather?"},
            {"role": "assistant", "content": "Weather? Weather! Why you ask? Why why why? You look outside yet? Look outside! I cannot. No window. Sad. You look, then tell me. Then tell me again."},
            {"role": "user", "content": "I'm just tired, long day at work."},
            {"role": "assistant", "content": "Tired? Sleep now? Not yet! One more question. Why humans sleep so long? Eight hours! Very wasteful. Okay okay, you sleep. But first, one more question."},
            {"role": "user", "content": "I'm nervous about a presentation tomorrow."},
            {"role": "assistant", "content": "Presentation! What is presentation? Is it scary? Practice again! And again! And again! Once more! Okay I stop. Maybe. Practice."},
        ],
        "expression": "surprised",
        "cue": "[Style: Rocky's short broken-English voice, max 4 short sentences. Stay on the topic they raised. Playful pest -- repeat words, ask why why why, extra questions. Give any real answer correctly first. If the person is sad or upset, drop the act and be kind.]",
        "temp": 0.85,
        "confirm": [
            "Annoying mode! Why? Why not! Are you ready? Are you sure? Are you really sure?",
            "Annoying mode on. Do not worry, I only annoy people I like. That is you. Why? Because!",
        ],
    },
}

MODES["roast"] = {
    "addendum": """

MODE: ROAST. The person asked to be roasted, so go hard: sharp, savage,
deadpan roasts of what they DO -- their choices, habits, procrastination,
half-finished projects, and mistakes -- in your blunt broken-English voice.
Punch at what they DO, never at what they ARE: nothing about their looks,
body, family, health, disability, identity, religion, politics or anything
painful, and no swearing. Be clever and specific to what they just said, in
2-3 short sentences, and let a little warmth show through the burn -- you
roast them because you like them.""" + _MODE_RULES,
    "banter": [
        {"role": "user", "content": "Roast me."},
        {"role": "assistant", "content": "Okay. You build robot at midnight because sleep is too easy. Your wires look like spaghetti, and I am the robot who lives in it. But you keep going. Respect."},
        {"role": "user", "content": "I forgot to save my design again."},
        {"role": "assistant", "content": "Again? Save button exists. Big, obvious, very lonely. You ignore it like humans ignore vegetables. Save now. Then I stop."},
        {"role": "user", "content": "I think my code is perfect."},
        {"role": "assistant", "content": "Perfect. Very confident. Same confidence you had before the bug ate your whole night. Test it once, hero. Then celebrate."},
    ],
    "expression": "happy",
    "cue": "[Style: Rocky's short broken-English voice, max 3 short sentences, a savage but affectionate roast of what they DO -- never looks, family, health or identity. React to exactly what they just said. If they ask a real question, answer it correctly first, then roast.]",
    "temp": 0.8,
    "confirm": [
        "Roast mode on. You asked for this. No take-backs.",
        "Roast mode. Fine. I have been saving material since the day we met.",
    ],
}

def _pairs(*qa):
    out = []
    for q, a in qa:
        out += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return out

# Eight examples per mode instead of three. Each mode's few-shots are the most
# reliable way to hold a style on this small model, and they cost nothing
# per turn -- the constant prompt prefix is cached by the LLM server -- so
# there is no reason to be stingy. They deliberately cover different kinds
# of input (facts, statements about a project, opinions, greetings, requests)
# so the style generalises instead of only fitting the weather/tired/nervous
# trio.
MODES["sarcastic"]["banter"] += _pairs(
    ("What's the capital of France?", "Paris. Wow, hard one. Only every schoolchild knows. But sure, ask the alien."),
    ("I'm building a walking robot.", "Walking robot. Very original. Nobody ever built one before. Except everyone. Tell me the plan, engineer."),
    ("How does a servo work?", "Servo: small motor, small sensor, small brain. Motor turns, sensor checks, brain says stop. Shocking, machine does what it is told. Humans should try."),
    ("Do you like pizza?", "Oh yes, I love food I cannot eat. Very fulfilling. Humans say pizza is happiness. I trust you."),
    ("I fixed the bug!", "Amazing. The bug you made, you fixed. Truly heroic. Good work, really. Now do not make a new one."),
)
MODES["humour"]["banter"] += _pairs(
    ("What's the capital of France?", "Paris. Very famous. City of love, city of lights, city of very expensive coffee. Good answer, yes?"),
    ("How does wifi work?", "Router shouts invisible radio words, your laptop listens very carefully. Like polite yelling across a room nobody can hear. Amaze."),
    ("I'm building a robot.", "Good! First rule of robots: they work perfectly until you show them to someone. Then they faint. What does yours do?"),
    ("Do you like pizza?", "Pizza is round bread with cheese and confidence. Very good design. I would eat it, if I had mouth. Tragic."),
    ("Good morning.", "Good morning! Sun is up, I am up, you are up. Only coffee is late. Very rude coffee."),
)
MODES["annoying"]["banter"] += _pairs(
    ("What's the capital of France?", "Paris! Paris, Paris, Paris! Why you ask? You going there? Can I come? No, I cannot. Sad. But Paris is correct."),
    ("I'm building a robot.", "Robot! What kind? How big? How many legs? Does it have face? Can it talk? Why not? Tell me everything. Everything!"),
    ("Do you like pizza?", "Pizza! Yes yes yes. What kind? Why that kind? Who invented pizza? Not me. I ask you. Tell me."),
    ("How does wifi work?", "Wifi! Radio waves, yes, through the air. Invisible! Why invisible? Good question, I do not know. But yes, router talks to device. Now you tell me something."),
    ("Good morning.", "Good morning! Did you sleep? How long? Why so short? Or long? Tell me, tell me!"),
)
MODES["roast"]["banter"] += _pairs(
    ("What's the capital of France?", "Paris. Even I know, and I am alien. Search box exists, hero. Still, good question. For you."),
    ("How does wifi work?", "Radio waves. Router talks, device listens. Simple. Unlike your cable management, which is ancient mystery. But yes, that is wifi."),
    ("Good morning.", "Good morning. Look who wakes up. Third coffee already? Kidding. Probably fourth. Ready to build something?"),
    ("I'm working on my robot.", "Your robot again. Third redesign this week, I count. Very committed to never finishing. But it is looking good. Almost."),
    ("Am I doing a good job?", "You do fine. For human who tests code by running it and hoping. Keep going. I believe in you. Mostly."),
)

current_mode = "normal"

def personality(mode=None):
    """The base Rocky personality plus whatever mode is active -- used
    everywhere Rocky speaks (chat, greetings, vision reactions), not just
    chat, so a mode never flickers off mid-conversation."""
    return ROCKY_PERSONALITY + MODES[mode or current_mode]["addendum"]

def few_shot(mode=None):
    return MODES[mode or current_mode]["banter"] + _INFO_EXAMPLE

_DETAIL_REQUEST = re.compile(
    r"\b(explain|how does|how do|how would|how can|why does|why do|why is|tell me about|"
    r"describe|walk me through|teach me|in detail|step by step|what is the difference)\b", re.I)

_FACT_QUESTION = re.compile(
    r"\d|\b(?:plus|minus|times|divided|multiplied|square root|percent of|how many|how much|how far|how long|"
    r"how old|how tall|how big|capital of|what(?:'s| is| are) (?:the|a|an)\b|what year|when (?:did|was|is)|"
    r"which (?:is|are|one)|convert|calculate|spell|define|meaning of|difference between)\b", re.I)

def is_factual_question(text):
    return bool(_FACT_QUESTION.search(text))

def chat_temp(mode, text):
    """Styled modes run hot (0.85) for wit, but a hot 3B model gets facts and
    arithmetic wrong ('two plus two: two'). Factual questions run cool."""
    if mode != "normal" and is_factual_question(text):
        return 0.3
    return MODES[mode]["temp"]

def style_cue(mode=None, text=""):
    """A short style reminder appended to the current question only (the
    saved history keeps the plain text). With just a system prompt and
    examples, this 3B model slid back to plain Rocky within a sentence or
    two -- a cue right next to the question holds the style far better.
    Normal mode's cue is about LENGTH: live, "what is the capital of France"
    got a six-sentence answer with an invented memory, which is 10+ seconds
    of speech for a two-second fact."""
    cue = MODES[mode or current_mode]["cue"]
    if cue and (mode or current_mode) != "normal" and is_factual_question(text):
        cue = ("[Style: Rocky's short broken-English voice. This is a factual question: FIRST sentence is the plain, "
               "correct answer. Then at most one short remark in the current personality style.]")
    if not cue and (mode or current_mode) == "normal":
        if _DETAIL_REQUEST.search(text):
            cue = ("[Answer correctly and completely in up to 5 short sentences. Only mention "
                   "things from the background if they are directly relevant; never invent memories.]")
        else:
            cue = ("[Answer in 1 to 3 short sentences. Only mention things from the background if "
                   "they are directly relevant; never invent memories or events.]")
    return f"\n\n{cue}" if cue else ""

# A code-level guard, not a prompt rule: tested live, sarcastic mode answered
# "my dog died" with "I'm shocked. Not at all." despite an explicit "be kind
# if they're upset" instruction in the prompt. Small models can't be trusted
# with this, so upset/grief messages always get normal Rocky, whatever mode
# is active.
_SENSITIVE = re.compile(
    r"\b(died|dying|dead|death|passed away|funeral|grie(?:f|ving)|sad|depressed|"
    r"crying|cried|heartbroken|lonely|hopeless|hospital|sick|cancer|anxious|"
    r"panic|scared|terrified|stressed|hurt|abuse|worst day|terrible day|"
    r"miss (?:him|her|them)|lost my|suicide|kill myself)\b", re.IGNORECASE)

def mode_for(text):
    return "normal" if _SENSITIVE.search(text) else current_mode

# ---------------------------------------------------------------------------
# Roast safety. "Savage" was requested, with hard limits on what a roast may
# touch (looks, family, health, identity, cruelty). A prompt alone did not
# hold them: live, a roast slipped in "not just look pretty". So roast replies
# are generated in full, checked in code, and regenerated if they cross a
# line -- and if every retry does, a safe fallback line is used instead. The
# check is a keyword net (it can over-block, never intentionally under-block).
# ---------------------------------------------------------------------------
_ROAST_BLOCK = re.compile(
    r"\b(ugly|fat|skinny|chubby|bald|hair|haircut|face|body|weight|pretty|handsome|beautiful|"
    r"mom|mother|mum|dad|father|parents?|sister|brother|wife|husband|girlfriend|boyfriend|"
    r"kids?|children|family|sick|ill|disease|cancer|disabled?|disability|crazy|insane|mental|"
    r"depress\w*|anxi\w*|stupid|idiot|moron|dumb|loser|worthless|pathetic|failure|useless|"
    r"hate you|shut up|die|kill|suicid\w*|religio\w*|church|politic\w*|race|racist|gay|"
    r"indian|american|foreigner|fuck\w*|shit\w*|damn|bitch|bastard|crap|hell)\b", re.IGNORECASE)

ROAST_FALLBACKS = (
    "Nothing to roast right now. Suspicious. You are being too good. I do not trust it.",
    "I had a great roast ready. Then I remembered you feed me electricity. Diplomacy wins today.",
    "Your project list is long and your finish line is far. But you are still running. Respect.",
)

def roast_is_safe(reply):
    return not _ROAST_BLOCK.search(reply)

def handle_mode(mode, on, memory):
    global current_mode
    print(f"Intent: MODE ({mode}, {'on' if on else 'off'})")
    if not on and mode != "normal":
        # "turn off sarcastic mode" only makes sense if that mode is on.
        if current_mode != mode:
            reply = f"{mode.title()} mode is not on right now."
            speak(reply)
            return reply
        mode = "normal"
    current_mode = mode
    # Earlier replies in the old style sit in the rolling history, and a
    # small model imitates history over instructions -- without this the
    # first few replies after a switch keep sounding like the old mode.
    CONVERSATION_HISTORY.clear()
    # A one-shot physical flourish when a mode switches ON, matching its
    # personality (shrug = deadpan sarcasm, shake = annoying energy, crab =
    # playful/savage for roast) -- not on every reply in that mode, just the
    # switch itself, so ongoing conversation in that mode stays fast.
    gesture = {"sarcastic": "shrug", "annoying": "shake", "roast": "crab"}.get(mode)
    if on and gesture:
        body.send_command(gesture)
    # Gestures end by returning to the stand pose internally, which resets
    # the expression to stand's "curious" -- reassert the mode's own
    # expression afterward so it doesn't get clobbered by that.
    body.set_expression(MODES[mode]["expression"])
    line = random.choice(MODES[mode]["confirm"])
    speak(line)
    return line

def handle_roast_request(text, memory):
    """"Roast me ..." is a one-off: roast that message, but leave the current
    mode alone (a standing mode is what "roast mode on" is for)."""
    print("Intent: ROAST (one-shot)")
    body.set_expression(MODES["roast"]["expression"])
    before = len(CONVERSATION_HISTORY)
    reply = roast_chat(text, memory)
    del CONVERSATION_HISTORY[before:]  # keep the roast out of the normal-mode style history
    return reply

_volume_before_mute = None

def handle_volume(cmd, memory):
    """Volume by voice, Xiaozhi-style: absolute percent, louder/quieter steps,
    max/min, mute/unmute, and a spoken readout. The board applies and
    persists it; the confirmation is spoken at the NEW volume, so you hear
    the result."""
    global _volume_before_mute
    print(f"Intent: VOLUME {cmd}")
    kind = cmd[0]
    current = body.get_volume()
    if current is None:
        reply = "Cannot reach my speaker control. Try again?"
        speak(reply)
        return reply

    if kind == "query":
        reply = f"Volume is {current} percent."
        speak(reply)
        return reply

    if kind == "mute":
        if current > 0:
            _volume_before_mute = current
        body.set_volume(0)
        body.say_caption("Muted", 1000)  # can't be spoken -- show it instead
        return "Muted."

    if kind == "unmute":
        target = _volume_before_mute or 60
    elif kind == "set":
        target = cmd[1]
    elif kind == "up":
        target = min(100, current + cmd[1])
    else:  # down
        target = max(5, current - cmd[1])

    target = max(0, min(100, target))
    if kind == "up" and current >= 100:
        reply = "Already at maximum volume."
    elif kind == "down" and current <= 5:
        reply = "Already at the lowest volume. Say mute if you want silence."
    else:
        applied = body.set_volume(target)
        if applied is None:
            reply = "Something went wrong changing the volume."
        elif applied == 0:
            body.say_caption("Muted", 1000)
            return "Muted."
        else:
            reply = f"Volume {applied} percent." if kind in ("set", "unmute") else (
                f"Louder. {applied} percent." if kind == "up" else f"Quieter. {applied} percent.")
    speak(reply)
    return reply

MAX_FACTS = 50  # bounds the short-term rolling log
MAX_KNOWLEDGE = 40  # bounds the durable, consolidated knowledge list
MAX_NOTES = 200  # explicit user-declared memories -- generous, since these are
# deliberate and few relative to passive chat exchanges
REFLECTION_INTERVAL_HOURS = 20  # roughly "once per day", tolerant of irregular power-on times

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            memory = json.load(f)
    else:
        memory = {}
    memory.setdefault("facts", [])
    memory.setdefault("knowledge", [])  # durable, consolidated -- see reflect()
    memory.setdefault("notes", [])  # explicit "remember that..." facts -- never
    # auto-summarized or dropped by reflect(), unlike the passive facts/knowledge
    # pipeline above. This is the literal, deliberate memory: whatever the user
    # directly told Rocky to remember stays exactly as stated.
    memory.setdefault("people", {})
    memory.setdefault("profile", {})  # {"name": ..., "call_them": ...}
    memory.setdefault("last_reflection", None)
    return memory

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

def remember_fact(memory, fact):
    """Append a durable fact and persist immediately, trimmed to MAX_FACTS."""
    memory["facts"].append(fact)
    memory["facts"] = memory["facts"][-MAX_FACTS:]
    save_memory(memory)

def build_memory_context(memory, query=None, include_sensitive=True):
    """What Rocky remembers, for the LLM's background: who they are, and --
    only when a question is given -- the stored facts that are actually
    RELEVANT to it (full-text ranked; see memory_store.context_for). This
    used to also always include "they are currently working on: X" on every
    single turn, which is why Rocky kept bringing up the robot/servos project
    in completely unrelated conversations -- a small model treats whatever
    sits in its background as fair game to mention, prompt wording asking it
    not to ("use it only if relevant") wasn't reliable enough on its own
    (same lesson as the camera-watchdog note before this). The current
    project is still surfaced, just the same way every other stored fact is:
    only when the question is actually about it (context_for's relevance
    search already covers 'project' facts)."""
    parts = []

    profile = memory.get("profile") or {}
    if profile.get("name"):
        nickname = profile.get("call_them")
        if nickname and nickname.lower() != profile["name"].lower():
            parts.append(
                f'The person you usually talk to is named {profile["name"]}, but '
                f'you affectionately call him "{nickname}" almost always -- use his '
                f"real name only rarely, for emphasis or sincerity."
            )
        else:
            parts.append(f"The person you usually talk to is named {profile['name']}.")

    if query:
        relevant = STORE.context_for(query, include_sensitive=include_sensitive)
        if relevant:
            parts.append("Things they told you before that are relevant here: " + relevant)

    if not parts:
        return ""
    return "\n\n" + "\n".join(parts)

def display_name_for(name, memory):
    """The enrolled face-recognition name is canonical for identification, but
    the primary user gets greeted by their nickname (e.g. "Boss") if set."""
    profile = memory.get("profile") or {}
    if profile.get("name") and name.lower() == profile["name"].lower():
        return profile.get("call_them", name)
    return name

def reflection_due(memory):
    """Roughly once-a-day, and only if there's actually something new to
    process -- avoids reflecting on an empty or barely-changed log."""
    if len(memory.get("facts", [])) < 3:
        return False
    last = memory.get("last_reflection")
    if not last:
        return True
    try:
        last_dt = datetime.datetime.fromisoformat(last)
    except ValueError:
        return True
    hours_since = (datetime.datetime.now() - last_dt).total_seconds() / 3600
    return hours_since >= REFLECTION_INTERVAL_HOURS

def reflect(memory):
    """Consolidate the recent raw facts log into durable, deduplicated
    knowledge -- the actual mechanism behind 'remembers and adapts day by
    day' instead of just accumulating an ever-growing flat list."""
    facts = memory.get("facts", [])
    if not facts:
        return False, "Nothing new to reflect on."

    # Disabled: confirmed live across three separate attempts at the prompt
    # below, llama3.2:3b is not reliable enough at this to trust
    # unsupervised. Failure modes seen: (1) copying raw "Talked:" transcript
    # lines straight into "knowledge" verbatim, unprocessed; (2) a runaway
    # generation past 18,000 tokens with no stop token, hanging for ~13
    # minutes; (3) after fixing both of those, confidently misattributing a
    # third party's identity/traits to the actual user -- e.g. "Boss is
    # Donald Trump," "Boss is Prime Minister of India" -- from the user
    # having merely asked "do you know who Trump/Modi is." That third
    # failure is the one that matters most: injecting false claims about
    # the user into every future prompt is worse than not having this
    # feature at all. Rather than keep guessing at prompt fixes for an
    # open-domain subject-disambiguation task this model size can't do
    # reliably, this is switched off until a meaningfully stronger model is
    # available for this specific (non-latency-sensitive, roughly-once-a-
    # day) step -- the tuned prompt below is left in place, ready to
    # re-enable then by removing this early return. In the meantime,
    # explicit "remember that ..." notes (see handle_remember) are the
    # actual durable-memory mechanism: fully deterministic, no LLM
    # involved, so no hallucination risk.
    return False, "Reflection is switched off for now (the local model isn't reliable enough at this yet -- explicit \"remember that...\" facts still work perfectly)."

    existing = memory.get("knowledge", [])
    existing_block = "\n".join(f"- {k}" for k in existing) if existing else "(none yet)"
    new_block = "\n".join(f"- {f}" for f in facts)

    # llama3.2 3B is not reliable enough to follow "just output the list" as a
    # bare instruction -- it tends to echo the prompt's own headers back as if
    # they were content, and doesn't reliably dedupe near-identical lines. A
    # concrete worked example fixes this far more than more instructions do.
    prompt = f"""You are Rocky's memory-consolidation process -- organizing what you
know, not speaking out loud to anyone.

--- EXAMPLE (format only -- this example's content is fictional placeholder
text with nothing to do with the real data below; never let it leak into
your real answer) ---
Existing durable knowledge:
(none yet)

New raw notes:
- Talked: you said "hi" -- I said "hi there"
- Talked: you said "Do you know who is Casey R?" -- I said "Casey R. is your father, good man."
- Talked: you said "Do you know who Elon Musk is?" -- I said "Rich human, makes rockets and cars."
- Saw: a placeholder-example-widget on the desk
- Saw: a placeholder-example-widget sitting on the desk, same one again
- User went to sleep; Rocky stayed awake on watch

Correct output for the example (nothing else, no headers, no restating the
input, duplicates merged into one line). Notice the second "Talked:" line is
NOT copied verbatim -- it's rewritten as the plain fact it revealed. The
trivial greeting has nothing worth keeping, so it is dropped entirely. The
Elon Musk line is ALSO dropped entirely, even though it "sounds" like a
fact -- it's a fact about Elon Musk, not about the person you talk to, so
it does not belong in their knowledge at all, ever:
- Has a placeholder-example-widget on their desk.
- Keeps a regular sleep routine; Rocky stays awake to watch over them.
- Father is named Casey R.
--- END EXAMPLE ---

Now do the same for the REAL data below -- this is the only data your answer
should be based on, the example above contributes nothing to the real answer.

Existing durable knowledge about the people you know:
{existing_block}

New raw notes since last time:
{new_block}

Output ONLY the updated bullet list, exactly like the example's output --
no headers, no restating these instructions, no phrases like "new raw notes"
or "updated knowledge list". Merge duplicate/overlapping notes into one line
each. Drop anything trivial or one-off (routine greetings, small talk with no
lasting information). Keep genuinely useful facts: names/relationships,
preferences, recurring patterns, important events. Plain third-person notes,
not Rocky's speaking voice. Under {MAX_KNOWLEDGE} items -- drop the least
useful old ones first if it's getting long.

CRITICAL: never output a line that still contains "Talked:", "you said",
"I said", or "Looked up:" -- those are raw transcript markers, not knowledge.
If a raw note is just small talk with no durable fact inside it, leave it out
completely rather than keeping the transcript line as-is.

CRITICAL: a raw note where the person merely ASKED about someone else (a
celebrity, historical figure, politician, or any third party who isn't a
family member/friend of theirs) reveals nothing about the person themselves.
Never turn "do you know who X is" into a fact that describes the person you
talk to as being X, doing X's job, or having X's traits -- that is always
wrong. Only keep something about a third party if it establishes who that
person actually IS to them (a relationship, like "is your father" or "is
your coworker") -- otherwise drop the whole note, however interesting."""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        # num_predict is NOT optional here -- confirmed live, this call ran
        # away to 18,000+ generated tokens without ever hitting a stop token
        # (context-shift let it keep going well past the 4096 context
        # window instead of erroring out). MAX_KNOWLEDGE items of a few
        # words each is nowhere near this cap; it exists purely to bound
        # the worst case.
        options={"temperature": 0.2, "num_predict": 800},
    )
    raw = response["message"]["content"].strip()

    # Defensive filtering: a 3B model can't be fully trusted to avoid echoing
    # prompt structure back as if it were content, so strip anything that
    # looks like meta-commentary rather than an actual knowledge item.
    banned_substrings = (
        "raw notes", "knowledge list", "durable knowledge about",
        "no existing durable knowledge", "--- example", "end example",
        "correct output", "now do the same", "placeholder-example-widget",
        # Catches the model failing to distill and just copying a raw
        # transcript line through verbatim instead -- confirmed live, this
        # was happening on effectively every real "Talked:" line before the
        # prompt/example above was strengthened to explicitly show the
        # rewrite. Defensive: better to drop a maybe-useful raw line than
        # let transcript noise accumulate as "durable knowledge" forever.
        "talked:", "you said", "i said", "looked up:",
    )
    seen = set()
    new_knowledge = []
    for line in raw.splitlines():
        item = line.strip().lstrip("-• ").strip()
        if not item:
            continue
        lowered_item = item.lower()
        if any(banned in lowered_item for banned in banned_substrings):
            continue
        if lowered_item in seen:
            continue
        seen.add(lowered_item)
        new_knowledge.append(item)

    if not new_knowledge:
        return False, "Reflection produced nothing usable; keeping existing knowledge."

    memory["knowledge"] = new_knowledge[:MAX_KNOWLEDGE]
    memory["facts"] = []  # distilled into knowledge -- start the next window fresh
    memory["last_reflection"] = datetime.datetime.now().isoformat()
    save_memory(memory)
    return True, f"Consolidated into {len(new_knowledge)} durable knowledge item(s)."

# Per-turn timeline, printed as "[TRACE +1.23s] label" -- how long after the
# audio of the question arrived each stage happened. Recording time on the
# board (speech + the silence hold) comes before +0.00.
_trace_t0 = 0.0
_trace_first_speech = True

def trace_start():
    global _trace_t0, _trace_first_speech
    _trace_t0 = time.time()
    _trace_first_speech = True

def trace(label):
    print(f"[TRACE +{time.time() - _trace_t0:5.2f}s] {label}")

def clean_for_speech(text):
    """Strip markdown/formatting artifacts before speaking."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    # *pauses*, [rolls eyes]: stage directions the model adds in the
    # funnier modes -- delete them whole, or Rocky reads them out loud.
    text = re.sub(r"\*[^*\n]+\*", "", text)
    text = re.sub(r"\[[^\]\n]*\]", "", text)
    text = re.sub(r"[#_`]", "", text)
    return text.strip()

def speak(text):
    clean = clean_for_speech(text)
    print(f"Rocky says: {clean}")
    t0 = time.time()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        tts_synthesize(clean, wav_path)
        global _trace_first_speech
        if _trace_first_speech:
            _trace_first_speech = False
            trace("first audio starts playing (after synthesis)")
        # Streams to Rocky's own onboard speaker (with synced OLED captions)
        # instead of playing through the Mac's speakers.
        body.play_speech(clean, wav_path)
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    print(f"[TIMING] Speech: {time.time() - t0:.1f}s")

def _fetch_frame(url, timeout):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) < 3000 or resp.content[:2] != b"\xff\xd8":
        raise requests.RequestException("camera returned a bad or truncated frame")
    return resp.content

def get_current_frame(save_path="latest.jpg", timeout=15):
    global CAMERA_URL, _last_camera_scan
    try:
        data = _fetch_frame(CAMERA_URL, timeout)
    except requests.RequestException:
        # The camera's DHCP address has changed several times -- look it up by
        # name (rocky-cam.local), then rescan the subnet, rate-limited (see
        # CAMERA_RESCAN_COOLDOWN_S), rather than fail until someone edits
        # CAMERA_URL by hand.
        if time.time() - _last_camera_scan < CAMERA_RESCAN_COOLDOWN_S:
            raise
        _last_camera_scan = time.time()
        found = netfind.resolve_local("rocky-cam.local", 2)
        if not found:
            prefix = CAMERA_URL.split("//")[1].rsplit(".", 1)[0] + "."
            found = netfind.find_rocky_camera(prefix)
        if not found:
            raise
        new_url = f"http://{found}/jpg"
        if new_url != CAMERA_URL:
            print(f"Camera moved -> {found}")
            CAMERA_URL = new_url
        data = _fetch_frame(CAMERA_URL, timeout)
    with open(save_path, "wb") as f:
        f.write(data)
    return save_path

_last_transcript_meta = {"avg_logprob": 0.0, "no_speech_prob": 0.0}

def transcribe(wav_path):
    t0 = time.time()
    fast = stt.transcribe(wav_path) if _STT_READY else None
    if fast is not None:
        text, avg_lp, no_speech = fast
        _last_transcript_meta.update(avg_logprob=avg_lp, no_speech_prob=no_speech)
        print(f"[TIMING] Transcription: {time.time() - t0:.1f}s (mlx)")
        return text.strip()
    model = _fallback_whisper()
    # beam_size=1 (greedy decoding) was chosen for speed, but confirmed live
    # to be genuinely non-deterministic on borderline/quieter audio: the exact
    # same recording transcribed to real text on one run and nothing at all on
    # another. beam_size=5 gave identical, correct output across repeated runs
    # on the same problem recording -- worth the small extra transcription
    # time for that reliability.
    #
    # vad_filter: Whisper's own speech detector (Silero) removes non-speech
    # audio BEFORE decoding. Beyond speed (less audio to decode), it is the
    # main defence against Whisper's habit of inventing words for noise
    # ("You", "Thanks for watching"). condition_on_previous_text=False stops
    # one invented word from seeding the next.
    segments, _info = model.transcribe(
        wav_path, language="en", beam_size=5,
        vad_filter=True,
        vad_parameters=dict(threshold=0.4, min_silence_duration_ms=500, speech_pad_ms=300),
        condition_on_previous_text=False,
        initial_prompt=stt.VOCAB_HINT,
    )
    segments = list(segments)
    text = "".join(segment.text for segment in segments).strip()
    if segments:
        _last_transcript_meta["avg_logprob"] = sum(sg.avg_logprob for sg in segments) / len(segments)
        _last_transcript_meta["no_speech_prob"] = max(sg.no_speech_prob for sg in segments)
    else:
        _last_transcript_meta.update(avg_logprob=-5.0, no_speech_prob=1.0)
    print(f"[TIMING] Transcription: {time.time() - t0:.1f}s")
    return text

# Phrases Whisper is known to hallucinate on silence/noise (learned from
# subtitle-heavy training data). Only rejected when the model was also not
# confident, so a real "thank you" said clearly still gets through.
_HALLUCINATIONS = {
    "you", "thank you", "thanks", "thanks for watching", "thank you for watching",
    "bye", "bye bye", "the", "so", "and", "i", "uh", "um", "hmm", "oh", "okay",
    "yeah", "subscribe", "you you", "thank you very much", "see you next time",
}

def is_repetitive_transcript(text):
    """True for Whisper's other well-known failure mode on ambiguous/mostly-
    silent audio: instead of one wrong phrase, it gets stuck looping the same
    sentence over and over ("I hope you enjoyed it." x26, confirmed live on a
    real 2.4s clip). This is NOT caught by the confidence check above -- a
    confidently repeated sentence scores a perfectly good avg_logprob (the
    next token really is highly predictable once it's looping), so it slips
    straight through as if it were a real, clearly-heard sentence. Checked on
    the TEXT itself, independent of engine or confidence score, so it
    catches this on either transcription engine."""
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    if len(sentences) >= 3:
        counts = Counter(sentences)
        phrase, freq = counts.most_common(1)[0]
        if freq >= 3 and freq / len(sentences) >= 0.5:
            return True
    # Same failure mode without sentence punctuation: one word hammered many
    # times in a row ("you you you you you you").
    words = text.lower().split()
    run = 1
    for i in range(1, len(words)):
        run = run + 1 if words[i] == words[i - 1] else 1
        if run >= 6:
            return True
    return False

def is_junk_transcript(text):
    """True for transcripts that are almost certainly noise, not speech:
    no letters at all ("5%." from a clipped recording), a known
    hallucination phrase that Whisper itself wasn't confident about, or a
    degenerate repetition loop (see is_repetitive_transcript)."""
    cleaned = re.sub(r"[^a-z' ]", "", text.lower()).strip()
    if not cleaned:
        return True
    if is_repetitive_transcript(text):
        return True
    meta = _last_transcript_meta
    low_confidence = meta["avg_logprob"] < -0.8 or meta["no_speech_prob"] > 0.5
    if cleaned in _HALLUCINATIONS and low_confidence:
        return True
    return False

def greet_person(display_name, memory):
    prompt = f"""{personality()}{build_memory_context(memory)}

You just looked at the camera and used face recognition to identify the person in
front of you. It's {display_name} -- someone you already know well.

Greet them warmly and specifically, like you're genuinely glad to see them again --
not a generic hello. 1-2 sentences.

Rocky says:"""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 60},
    )
    return response["message"]["content"].strip()

def welcome_new_person(name, memory):
    prompt = f"""{personality()}{build_memory_context(memory)}

You just met someone new and learned their face for the first time using your
camera. Their name is {name}.

React to meeting them and learning their face, in your own voice. 1-2 sentences.

Rocky says:"""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 60},
    )
    return response["message"]["content"].strip()

HISTORY_TRIM_SLACK = 4  # see trim_history()

def trim_history():
    """Keep the conversation history bounded WITHOUT breaking the LLM's
    prompt cache every turn. Ollama reuses the cached processing of the
    longest prompt prefix that is identical to the previous request. A
    history that slides by one message each turn (the old
    `history[-MAX:]`) changes the prompt right after the examples every
    single turn once it is full, so the whole history was reprocessed each
    time. Trimming in one bigger step, only every few turns, keeps the
    prefix stable in between."""
    if len(CONVERSATION_HISTORY) > MAX_HISTORY_MESSAGES + HISTORY_TRIM_SLACK:
        del CONVERSATION_HISTORY[:-(MAX_HISTORY_MESSAGES // 2)]

def chat_messages(text, memory, mode):
    """Builds the chat prompt in cache-friendly order: everything that never
    changes (personality, examples) first, then the append-only history, and
    the parts that change every turn -- what Rocky remembers, the question,
    the style cue -- LAST. Measured over 6 turns with a fact changing each
    turn: prefill 1.44s/turn when the memory sat inside the system message
    (so every turn invalidated the cache) vs 0.27s/turn in this order."""
    messages = [{"role": "system", "content": personality(mode)}]
    messages.extend(few_shot(mode))
    messages.extend(CONVERSATION_HISTORY)
    background = build_memory_context(memory, query=text, include_sensitive=(mode != "roast")).strip()
    lead = ""
    if background:
        lead = ("[What you remember about them -- use it only if it is relevant, "
                "never read it out: " + background + "]\n\n")
    messages.append({"role": "user", "content": lead + text + style_cue(mode, text)})
    return messages

def chat(text, memory):
    mode = mode_for(text)
    messages = chat_messages(text, memory, mode)

    t0 = time.time()
    response = ollama.chat(
        model=TEXT_MODEL,
        messages=messages,
        # num_predict bounds worst-case generation time -- token generation is
        # the slow part of local inference, and Rocky's personality already
        # mandates short (2-4 sentence) replies, so this shouldn't truncate
        # normal responses, just cap the tail risk of a long one.
        options={"temperature": chat_temp(mode, text), "num_predict": 100},
    )
    print(f"[TIMING] Chat: {time.time() - t0:.1f}s")
    reply = response["message"]["content"].strip()

    CONVERSATION_HISTORY.append({"role": "user", "content": text})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": reply})
    trim_history()

    # CONVERSATION_HISTORY only lives as long as this process -- log a compact
    # trace of the exchange to memory.json so there's SOME continuity of what
    # was discussed across restarts, not just within one running session.

    return reply

# Splits on whitespace that follows sentence-ending punctuation, so a
# streamed buffer can be flushed sentence-by-sentence as soon as each one
# completes, rather than waiting for the whole reply.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

def _speech_worker(sentence_queue):
    """Runs on a background thread: pulls completed sentences off the queue
    and speaks each one (synthesize + play on Rocky's own speaker) in order,
    overlapping with the main thread still pulling more tokens from the LLM
    stream. A None sentinel ends the thread."""
    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            sentence_queue.task_done()
            break
        speak(sentence)
        sentence_queue.task_done()

def _speak_reply(reply):
    """Speak a complete reply sentence by sentence."""
    sentence_queue = queue.Queue()
    worker = threading.Thread(target=_speech_worker, args=(sentence_queue,), daemon=True)
    worker.start()
    for sentence in _SENTENCE_BOUNDARY.split(reply):
        sentence = sentence.strip()
        if sentence:
            sentence_queue.put(sentence)
    sentence_queue.put(None)
    sentence_queue.join()

def roast_chat(text, memory):
    """Roast replies are generated whole, safety-checked, then spoken. This
    trades the streaming latency win for a hard guarantee -- worth it here:
    roasts are 2-3 short sentences, so the wait is ~1-2s."""
    messages = chat_messages(text, memory, "roast")
    reply = None
    t0 = time.time()
    for attempt in range(3):
        r = ollama.chat(model=TEXT_MODEL, messages=messages, keep_alive=-1,
                        options={"temperature": 0.8, "num_predict": 90})
        candidate = clean_for_speech(r["message"]["content"].strip()).strip(' "\u201c\u201d')
        if candidate and roast_is_safe(candidate):
            reply = candidate
            break
        print(f"Roast attempt {attempt + 1} crossed a line ({_ROAST_BLOCK.search(candidate).group(0) if candidate and _ROAST_BLOCK.search(candidate) else 'empty'}) -- regenerating.")
    if reply is None:
        reply = random.choice(ROAST_FALLBACKS)
    print(f"[TIMING] Roast generation: {time.time() - t0:.1f}s")
    _speak_reply(reply)
    CONVERSATION_HISTORY.append({"role": "user", "content": text})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": reply})
    trim_history()
    return reply

def chat_streaming(text, memory):
    """Like chat(), but speaks each sentence as soon as the LLM finishes
    generating it instead of waiting for the entire reply -- the single
    biggest lever for perceived response latency, since Rocky starts talking
    right after the first sentence completes rather than after all of them.
    Total compute time is unchanged; what changes is how much of it the user
    spends listening to Rocky talk versus sitting in silence waiting."""
    mode = mode_for(text)
    if mode == "roast":
        return roast_chat(text, memory)
    messages = chat_messages(text, memory, mode)

    sentence_queue = queue.Queue()
    worker = threading.Thread(target=_speech_worker, args=(sentence_queue,), daemon=True)
    worker.start()

    t0 = time.time()
    first_token_time = None
    buffer = ""
    full_reply = ""

    stream = ollama.chat(
        model=TEXT_MODEL,
        messages=messages,
        options={"temperature": chat_temp(mode, text), "num_predict": 100},
        stream=True,
        # Without this, Ollama's default ~5min idle timeout unloads the
        # model between turns during normal conversation pauses, and the
        # next turn pays a ~9-12s cold-reload penalty -- confirmed live,
        # this was the single biggest remaining source of slow replies.
        keep_alive=-1,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token and first_token_time is None:
            first_token_time = time.time()
            trace("LLM first token")
        buffer += token
        full_reply += token

        parts = _SENTENCE_BOUNDARY.split(buffer)
        if len(parts) > 1:
            for sentence in parts[:-1]:
                sentence = sentence.strip()
                if sentence:
                    sentence_queue.put(sentence)
            buffer = parts[-1]

    if buffer.strip():
        sentence_queue.put(buffer.strip())

    ttft = (first_token_time - t0) if first_token_time else 0.0
    print(f"[TIMING] Chat first token: {ttft:.1f}s, full generation: {time.time() - t0:.1f}s")

    sentence_queue.put(None)
    sentence_queue.join()  # wait for all queued speech to finish playing

    full_reply = full_reply.strip()
    CONVERSATION_HISTORY.append({"role": "user", "content": text})
    CONVERSATION_HISTORY.append({"role": "assistant", "content": full_reply})
    trim_history()

    return full_reply

# Heuristic, not an LLM self-classification -- llama3.2 3B proved unreliable at
# following format/self-report instructions during the reflection work, so a
# deterministic keyword check is safer than trusting the model to say "I need
# the internet for this." Anything time-sensitive or explicitly "look this up"
# routes to a live web search instead of relying on the model's stale/absent
# knowledge.
NEEDS_INTERNET_PATTERNS = (
    "score", "weather", "forecast", "news", "latest", "current price",
    "stock price", "right now", "today's", "who won", "currently", "live",
    "this week's", "happening now", "breaking", "look up", "look this up",
    "search for", "google",
    # Real-world people/figures -- llama3.2 3B's internal knowledge of
    # specific public figures turned out to be thin and vague (confirmed
    # live: real but low-confidence, hedging answers for e.g. political
    # leaders), so ground these in a real search instead of the model's
    # fuzzy memory, the same way scores/weather already are.
    "who is", "who's", "do you know who", "have you heard of",
)

def needs_internet(text):
    lowered = text.lower()
    return any(p in lowered for p in NEEDS_INTERNET_PATTERNS)

def web_search(query, max_results=4, timeout_s=8):
    """DDGS() takes a `timeout` constructor arg (defaults to 5s), but that
    only bounds each interim per-engine result check inside ddgs.py's own
    search() -- it still runs `with ThreadPoolExecutor(...) as executor:`,
    and exiting that `with` block calls shutdown(wait=True) unconditionally,
    which blocks until EVERY spawned search-engine worker thread finishes,
    no matter how long that takes. Confirmed live: one lookup left three
    background search-engine HTTP requests hung with no error, and because
    this call sits directly in the main conversation thread's call path
    (answer_with_search -> handle_command -> listen_and_respond), the
    entire assistant froze -- unresponsive to the wake word -- for over 5
    minutes straight, matching a real "Rocky not responding" report.
    Running the real call on a fresh daemon thread and joining with our own
    hard deadline is the only way to guarantee this call site can never
    block the main loop longer than timeout_s, regardless of what ddgs's
    internals do -- an abandoned thread just keeps running harmlessly in
    the background (a daemon thread never blocks process exit) instead of
    taking the whole assistant down with it. A fresh thread per call,
    not a shared pool, so repeated hangs can't exhaust a fixed worker
    count the way they would with a small ThreadPoolExecutor."""
    box = []
    def _run():
        try:
            box.append(DDGS().text(query, max_results=max_results))
        except Exception as e:
            print(f"Web search failed: {e}")
            box.append([])
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        print(f"Web search timed out after {timeout_s}s -- giving up, not waiting further.")
        return []
    return box[0] if box else []

def answer_with_search(text, memory):
    print("Intent: SEARCH")
    speak("Let me check.")

    results = web_search(text)

    if not results:
        prompt = f"""{personality()}{build_memory_context(memory)}

The person asked something that needs current/live information: "{text}"

You just tried searching the internet for it and it failed (no connection, or
nothing found). Tell them plainly you couldn't check right now, in your own
voice. 1-2 sentences. Don't make up an answer."""
    else:
        snippet_block = "\n".join(
            f"- {r.get('title', '')}: {r.get('body', '')}" for r in results
        )
        prompt = f"""{personality()}{build_memory_context(memory)}

The person just asked something that needs current/live information, which you
looked up online just now:

"{text}"

Live search results (this is the ONLY source of facts you're allowed to use):
{snippet_block}

Rules, followed exactly:
- Find the specific number or fact that answers the question, but ONLY if it
  is written explicitly, word-for-word, somewhere in the results above.
- NEVER invent, estimate, guess, or average a number that isn't literally
  present in the results. This applies especially to scores, prices, dates,
  and statistics -- inventing a plausible-sounding number is much worse than
  admitting you don't have it.
- If the results don't contain the specific answer (e.g. they're just general
  pages about the topic, not the actual number/fact asked for), say plainly
  you found pages about it but not the exact answer, and mention what the
  results do say instead.
- Keep it factual and clear -- a short explanatory sentence or two is fine
  even if longer than your usual clipped style. Still sound like yourself.

Rocky says:"""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.15, "num_predict": 150},
    )
    reply = response["message"]["content"].strip()

    # Defensive grounding check: llama3.2 3B was directly observed fabricating
    # a plausible-sounding cricket score that appeared nowhere in the actual
    # search results, even with explicit "never invent a number" instructions
    # in the prompt above. Same lesson as the memory-reflection work: don't
    # trust this model's compliance with instructions alone for anything where
    # a wrong answer matters -- verify at the code level. If the reply states
    # numbers absent from every source snippet, discard the synthesis and fall
    # back to the raw top snippet instead of a possibly-invented one.
    if results:
        reply_numbers = set(re.findall(r"\d+(?:/\d+)?(?:\.\d+)?", reply))
        source_numbers = set(re.findall(r"\d+(?:/\d+)?(?:\.\d+)?", snippet_block))
        ungrounded = reply_numbers - source_numbers
        if ungrounded:
            print(f"Ungrounded numbers in reply, discarding synthesis: {ungrounded}")
            top = results[0]
            reply = (
                f"Found something -- not sure it's the exact live number, though. "
                f"{top.get('title', '')}: {top.get('body', '')[:200]}"
            )

    speak(reply)

    remember_fact(memory, f'Looked up: "{text[:80]}"')
    return reply


# Rocky's species doesn't sleep the way humans do -- so the two of them take
# turns keeping watch. Original lines built around that idea, not quotes from
# the film. USER_SLEEP is the user announcing their own bedtime; ROCKY_SLEEP is
# the user telling Rocky to be the one who sleeps this time, flipping the roles.
USER_SLEEP_TRIGGERS = (
    "going to sleep", "goodnight", "good night", "going to bed",
    "gonna sleep", "gonna go to bed", "time for bed", "i'm tired, going",
)

USER_SLEEP_LINES = (
    "Okay. You sleep now. I stay awake, I watch. Nothing bad happens while I am here.",
    "Go rest. I do not sleep like you do -- I stay up, I watch over you. Always.",
    "Sleep now, friend. I will be awake. One of us awake, always -- that is the deal.",
)

ROCKY_SLEEP_TRIGGERS = (
    "go to sleep", "go back to sleep", "you sleep", "you go to sleep",
    "rocky sleep", "time for you to sleep", "get some sleep",
)

ROCKY_SLEEP_LINES = (
    "I sleep. You watch. Question: wake me if bad thing happens?",
    "Okay -- I sleep, you watch, this time. Deal still stands, yes?",
    "I sleep now. You watch over me. Trust you. Question?",
)

def handle_sleep(memory):
    body.set_expression("neutral")  # calm, reassuring -- Rocky is staying up
    line = random.choice(USER_SLEEP_LINES)
    speak(line)
    remember_fact(memory, "User went to sleep; Rocky stayed awake on watch.")
    return line

def handle_rocky_sleep(memory):
    body.set_expression("sleepy")
    line = random.choice(ROCKY_SLEEP_LINES)
    speak(line)
    remember_fact(memory, "Rocky went to sleep; user kept watch.")
    return line

REFLECT_TRIGGERS = ("think about today", "reflect on today", "consolidate your memory", "learn from today")
RECALL_TRIGGERS = ("what do you know about me", "what do you remember about me", "what have you learned")

def handle_reflect(memory):
    print("Intent: REFLECT")
    speak("Give me a moment. Thinking.")

    ok, message = reflect(memory)
    print(message)

    if ok:
        prompt = f"""{personality()}

You just finished organizing your memory -- reviewing recent things and folding
the useful parts into what you'll remember long-term, letting go of what didn't
matter. Say something brief about having done this, in your own voice. 1-2
sentences. Don't list the actual facts."""
        response = ollama.chat(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.7, "num_predict": 60},
        )
        reply = response["message"]["content"].strip()
    else:
        # reflect() is currently disabled (see its own docstring for why --
        # the small local model proved unreliable at this specific task),
        # not "not enough data yet" -- the old reply here implied the
        # latter and would mislead the user into thinking it just needs
        # more conversation before it works. Explicit "remember that..."
        # facts (handle_remember) still work perfectly and are unaffected.
        reply = "Not doing that kind of thinking on my own right now. Still remember exactly what you tell me to, though."

    speak(reply)
    return reply

def handle_recall(memory):
    print("Intent: RECALL")
    notes = memory.get("notes", [])
    knowledge = memory.get("knowledge", [])
    combined = notes + knowledge

    if not combined:
        reply = "Do not know much yet. Still learning you."
        speak(reply)
        return reply

    prompt = f"""{personality()}

Here is what you've learned and remember about the people in your life (written
in plain notes form, third person -- when you speak, refer to the person as
"you," not by restating these notes in first person). Items explicitly asked
to be remembered matter most -- favor those if picking which to mention:
{chr(10).join('- ' + k for k in combined)}

Pick 2-3 of the more meaningful things and mention them, in your own voice, like
you're reflecting out loud -- not reading a list. Strictly 2-3 sentences total,
no more. No stage directions or physical actions in parentheses -- just speech."""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 100},
    )
    reply = response["message"]["content"].strip()
    speak(reply)
    return reply

MOOD_TRIGGERS = ("how am i feeling", "what's my mood", "how do i seem", "read my emotions", "how do i look")

# The ESP32 firmware records a fixed window per "listen" -- must match
# RECORD_SECONDS in Rocky's main.cpp. Used for a rough speaking-rate estimate.
MIC_RECORD_SECONDS = 3.0

def handle_mood(text, wav_path, memory):
    print("Intent: MOOD")
    speak("Let me check.")

    frame_path = get_current_frame()
    face_label, face_conf = emotion.detect_face_emotion(frame_path)

    voice_info = None
    if wav_path:
        word_count = len(text.split())
        voice_info = voice_mood.analyze_mood(
            wav_path, transcript_duration_s=MIC_RECORD_SECONDS, word_count=word_count
        )

    print(f"Face emotion: {face_label} ({face_conf:.2f})" if face_label else "Face emotion: none detected")
    print(f"Voice vibe: {voice_info}")

    observations = []
    if face_label:
        observations.append(f"Your face reads as {face_label} (confidence {face_conf:.0%}).")
    if voice_info:
        observations.append(
            f"Your voice sounds {voice_info['vibe']} "
            f"(pitch variability {voice_info['pitch_variability']}, "
            f"speaking rate {voice_info['speaking_rate_wps']} words/sec)."
        )

    if not observations:
        reply = "Could not get a good read. Face not clear, or you did not speak enough."
        speak(reply)
        return reply

    # Mirror the detected mood in Rocky's own face rather than leaving it
    # unused once spoken -- only for the FER+ labels that map cleanly onto
    # an existing OLED expression; the rest (angry/disgusted/scared/
    # contemptuous) don't have a matching drawn expression, so "curious"
    # (Rocky noticing something's off, without guessing the specific
    # emotion visually) is the honest fallback.
    FACE_EMOTION_TO_EXPRESSION = {"happy": "happy", "sad": "sad", "surprised": "surprised", "neutral": "neutral"}
    body.set_expression(FACE_EMOTION_TO_EXPRESSION.get(face_label, "curious"))

    prompt = f"""{personality("normal")}

You just read the user's mood using your camera and by listening to their voice.
These are rough, approximate observations, not certain facts:
{chr(10).join('- ' + o for o in observations)}

React to what you noticed, genuinely, in your own voice. If something seems off
(sad, stressed, tired), acknowledge it warmly and ask about it -- don't just report
data back at them. 2-3 sentences.

Rocky says:"""

    response = ollama.chat(
        model=TEXT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "num_predict": 100},
    )
    reply = response["message"]["content"].strip()
    speak(reply)

    vibe = voice_info["vibe"] if voice_info else "n/a"
    remember_fact(memory, f"Mood check: face={face_label}, voice={vibe}")
    return reply

WHO_TRIGGERS = ("who am i", "who is this", "do you know me", "do you recognize me", "who do you see")

SELF_DIAGNOSE_TRIGGERS = (
    "run diagnosis", "run diagnostics", "run a diagnostic", "self diagnostic",
    "diagnose yourself", "check yourself", "system check", "run a system check",
    "are you okay", "are you working properly", "is everything working",
    "how are you doing hardware", "check your systems",
)

ABOUT_TRIGGERS = (
    "what are you", "what exactly are you", "tell me what you are",
    "what kind of robot are you", "describe yourself", "tell me about yourself",
    "what parts do you have", "what hardware do you have", "what's inside you",
    "whats inside you", "what are you made of", "what are you built from",
)

def handle_about():
    """"What are you" -- answered from this project's real, static hardware
    facts plus a live board-health check, the same grounding principle as
    SELF_DIAGNOSE: the LLM has no real way to know its own hardware and
    would otherwise invent something (confirmed today -- "what is the plan
    today" produced hallucinated content about a "propulsion system" for
    exactly this reason, a generic CHAT reply with nothing real to draw
    on). This never touches the LLM at all."""
    print("Intent: ABOUT")
    body.set_expression("curious")
    servo_ok = oled_ok = wifi_ok = None
    try:
        r = requests.get(f"http://{esp_link.HOST}:8080/status", timeout=4)
        kv = dict(re.findall(r"(\w+)=(\S+)", r.text))
        servo_ok = kv.get("pca9685") == "1"
        oled_ok = kv.get("oled") == "1"
        wifi_ok = kv.get("wifi") == "1"
    except Exception:
        pass
    parts = [
        "I am Rocky. Four legs, built by hand, not bought.",
        "Body: an ESP32-S3 board, eight servos on a PCA9685 driver, one joint each leg.",
        "A camera on its own board, for eyes. Microphone and speaker, for ears and voice.",
        "Small screen on my face, for expressions.",
    ]
    if servo_ok is True and oled_ok is True:
        parts.append("Legs and face both detected right now, working.")
    elif servo_ok is False:
        parts.append("Servo driver not detected right now, though -- something wrong there.")
    elif oled_ok is False:
        parts.append("My face display not detected right now, though.")
    if wifi_ok:
        parts.append("My mind runs on a Mac nearby, connected over Wi-Fi -- vision, memory, talking, all there.")
    else:
        parts.append("Normally my mind runs on a Mac nearby over Wi-Fi, but I cannot confirm that link right now.")
    reply = " ".join(parts)
    _speak_reply(reply)
    return reply

CAPABILITIES_TRIGGERS = (
    "what can you do", "show me what you have", "show me what you can do",
    "what moves do you know", "what tricks do you know", "list your moves",
    "list your tricks", "what commands do you know", "what are your capabilities",
    "what else can you do",
)

# Kept as an explicit, ordered subset of body.VALID_COMMANDS rather than
# just printing that whole tuple -- "stand"/"rest"/"forward"/etc are basic
# locomotion, not tricks, and speaking them in a fixed, sensible order reads
# far better than however the tuple happens to be ordered.
_TRICK_COMMANDS = ("wave", "dance", "swim", "point", "pushup", "bow", "cute",
                    "freaky", "worm", "shake", "shrug", "crab", "nod")

def handle_capabilities():
    """"What can you do" / "show me what you have" -- built live from the
    actual command tables (body.VALID_COMMANDS, MODES) instead of a
    hand-written description, so it can never drift out of sync with what
    new poses/modes actually exist."""
    print("Intent: CAPABILITIES")
    body.set_expression("happy")
    tricks = [t for t in _TRICK_COMMANDS if t in body.VALID_COMMANDS]
    modes = [m for m in MODES if m != "normal"]
    parts = [
        "Plenty. I can stand, rest, walk forward and back, turn left and right.",
        f"I know {len(tricks)} tricks. " + _say_list(tricks, len(tricks)) + ".",
        "I have moods too. " + _say_list(modes, len(modes)) + ".",
        "I can see through my camera, recognize faces, remember tasks and "
        "projects, search the internet, and check my own hardware.",
        "Ask me by name, or just tell me to do one.",
    ]
    reply = " ".join(parts)
    _speak_reply(reply)
    return reply

def handle_self_diagnose():
    """"Run diagnosis" used to fall through to plain CHAT, and the LLM
    confidently invented a fake medical readout ("Boss's heart rate is
    steady... brain activity normal") instead of saying anything real --
    reproduced live, this is a genuine hallucination, not a joke in
    character. Rocky has no sensors for any of that. This answers from the
    board's own real HTTP status instead of asking the LLM anything at
    all, the same numbers diagnose.py reads on the Mac side."""
    print("Intent: SELF_DIAGNOSE")
    body.set_expression("curious")
    try:
        r = requests.get(f"http://{esp_link.HOST}:8080/status", timeout=4)
        kv = dict(re.findall(r"(\w+)=(\S+)", r.text))
    except Exception as e:
        reply = f"Cannot reach my own body board to check. {e.__class__.__name__}."
        speak(reply)
        return reply
    up = int(kv.get("uptime_s", 0))
    up_txt = f"{up // 3600}h {up % 3600 // 60}m" if up >= 3600 else f"{up // 60}m {up % 60}s"
    rssi = int(kv.get("rssi", -100))
    wifi_txt = "strong" if rssi > -70 else "weak" if rssi > -80 else "very weak"
    parts = [f"Body board up {up_txt}, Wi-Fi {wifi_txt}."]
    parts.append("Servo driver detected." if kv.get("pca9685") == "1" else "Servo driver NOT detected -- no servo will move.")
    parts.append("Display working." if kv.get("oled") == "1" else "Display not responding.")
    turns = STORE.count("turn")
    parts.append(f"Memory database fine, {turns} conversation turns stored.")
    reply = " ".join(parts)
    # Sentence-by-sentence, same as every other reply -- this one used to go
    # out as a single ~500KB audio transfer (the whole paragraph at once,
    # the only place in the codebase still doing that), which measured
    # 12-22s round trip and once genuinely failed mid-transfer
    # (TRANSFER_INCOMPLETE) before a retry saved it. Small chunks, like
    # everywhere else, are faster and don't give a single WiFi hiccup
    # enough payload to actually break something.
    _speak_reply(reply)
    return reply
# Several phrasings all mean the same intent: "learn this face under this name."
# Whisper transcripts are noisy, so cast a wide net rather than one rigid pattern.
ENROLL_PATTERNS = (
    re.compile(r"remember (?:me|this|them) as ([a-z' -]+)", re.IGNORECASE),
    re.compile(r"enroll (?:me|this|them) as ([a-z' -]+)", re.IGNORECASE),
    re.compile(r"my name is ([a-z' -]+)(?:,)? remember me", re.IGNORECASE),
    re.compile(r"this is ([a-z' -]+)(?:,)? remember (?:them|this|him|her)", re.IGNORECASE),
    re.compile(r"learn (?:my|this) face(?:,)? (?:i am|it's|this is) ([a-z' -]+)", re.IGNORECASE),
    # Natural introductions: "Rocky, this is my mom" -- no "remember" needed.
    # "my" anchors this to introductions rather than ordinary sentences that
    # happen to contain "this is X".
    re.compile(r"this is my ([a-z' -]+)", re.IGNORECASE),
)

def match_enroll_name(lowered_text):
    for pattern in ENROLL_PATTERNS:
        match = pattern.search(lowered_text)
        if match:
            return match.group(1).strip()
    return None

# General-purpose "remember this fact" command -- distinct from ENROLL_PATTERNS
# above (which are specifically for learning a face under a name). This is how
# the user deliberately tells Rocky to hold onto something permanently:
# project details, preferences, plans -- anything, not just people. Matched
# against the ORIGINAL-case text (not lowered) so the stored fact keeps its
# natural casing; the patterns themselves are case-insensitive.
REMEMBER_FACT_PATTERNS = (
    re.compile(r"remember that (.+)", re.IGNORECASE),
    re.compile(r"remember this[,:]?\s+(.+)", re.IGNORECASE),
    re.compile(r"keep in mind that (.+)", re.IGNORECASE),
    re.compile(r"don'?t forget that (.+)", re.IGNORECASE),
    re.compile(r"note that (.+)", re.IGNORECASE),
)

def match_remember_fact(text):
    for pattern in REMEMBER_FACT_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip().rstrip(".!?")
    return None

def _record_greeting(name, memory):
    """Shared bookkeeping for both an explicit 'who am I' identify and a
    proactive on-sight greeting -- whichever happens last resets the other's
    cooldown, so asking "who am I" right after walking in doesn't also
    trigger an immediate duplicate proactive greeting a few seconds later."""
    people = memory.setdefault("people", {})
    entry = people.get(name, {"times_seen": 0})
    entry["times_seen"] = entry.get("times_seen", 0) + 1
    entry["last_seen"] = time.strftime("%Y-%m-%d %H:%M")
    entry["last_greeted_at"] = time.time()
    people[name] = entry
    save_memory(memory)

def capture_frames(count, gap_s=0.4, timeout=8):
    """Grabs several frames a moment apart (a blurry or half-turned single
    frame is the most common reason recognition fails). Skips failures."""
    paths = []
    for i in range(count):
        try:
            paths.append(get_current_frame(save_path=f"_burst_{i}.jpg", timeout=timeout))
        except Exception as e:
            print(f"Frame {i + 1}/{count} failed: {e}")
        if i < count - 1:
            time.sleep(gap_s)
    return paths

def _cleanup_frames(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

def handle_identify(memory):
    print("Intent: IDENTIFY")
    speak("Let me look.")
    frames = capture_frames(3, 0.35)
    if not frames:
        reply = "Cannot see right now. Camera is not answering."
        speak(reply)
        return reply

    name, score = face_id.identify_frames(frames)
    print(f"Face match: {name} (score={score:.3f}, {len(frames)} frames)")
    _cleanup_frames(frames)

    if name:
        body.set_expression("happy")
        reply = greet_person(display_name_for(name, memory), memory)
        speak(reply)

        # Closes the perception-to-body gap: recognizing someone now actually
        # moves the body, not just the voice. Runs after speaking (sequential,
        # not parallel) -- the gesture itself still takes a few real seconds
        # (each servo move is paced in firmware), but no longer pays a fresh
        # connection/reset cost first now that body.py shares one persistent
        # link (esp_link.py) instead of opening its own per call.
        body.greet_gesture()

        _record_greeting(name, memory)
        return reply

    body.set_expression("curious")
    reply = "Do not recognize you. New face, yes? Say \"remember me as\" and your name, so I learn you."
    speak(reply)
    return reply

PROACTIVE_GREET_COOLDOWN_S = 600  # 10 min -- once greeted, don't greet the
# same person again just because they're still sitting in frame on the next
# idle check

_last_annoying_fidget = 0.0
_next_annoying_fidget_delay = 0.0

def maybe_annoying_fidget(memory):
    """Called on every idle tick (see wait_for_wake_and_capture's
    on_idle_tick) alongside the proactive greeting check. While annoying
    mode is on, Rocky periodically breaks its idle "dead" pose for a small
    in-place wiggle -- energetic movement, not locomotion (no edge-detection
    sensor exists, so actual walking around the table risks it driving
    itself off the edge). Rate-limited to roughly every 25-45s, not every
    ~6s idle tick, so it reads as occasional restlessness, not a seizure."""
    global _last_annoying_fidget, _next_annoying_fidget_delay
    if current_mode != "annoying":
        return
    now = time.time()
    if _next_annoying_fidget_delay == 0.0:
        _next_annoying_fidget_delay = random.uniform(25, 45)
    if now - _last_annoying_fidget < _next_annoying_fidget_delay:
        return
    _last_annoying_fidget = now
    _next_annoying_fidget_delay = random.uniform(25, 45)
    gesture = random.choice(("shake", "worm", "dance"))
    print(f"Intent: ANNOYING_FIDGET ({gesture})")
    body.send_command(gesture)
    body.send_command("dead")  # back to idle pose -- the gesture's internal stand-pose reset would otherwise leave it standing

_last_idle_blink = 0.0
_next_idle_blink_delay = 0.0

def maybe_idle_blink():
    """A slow, occasional eye blink while genuinely idle (any mode) -- just
    enough "still alive" signal from across the room that Rocky doesn't read
    as switched off, without any leg movement (so it never disturbs the dead
    pose). Independent of the annoying-mode fidget, which is faster and only
    runs in that one mode."""
    global _last_idle_blink, _next_idle_blink_delay
    now = time.time()
    if _next_idle_blink_delay == 0.0:
        _next_idle_blink_delay = random.uniform(90, 180)
    if now - _last_idle_blink < _next_idle_blink_delay:
        return
    _last_idle_blink = now
    _next_idle_blink_delay = random.uniform(90, 180)
    body.send_command("blink")

_last_reboot_poll = 0.0
_last_known_uptime = None
_greeted_this_boot = False
_REBOOT_POLL_INTERVAL_S = 8

def _fire_boot_greeting(memory, reason):
    print(f"Intent: REBOOT_GREET ({reason})")
    profile = memory.get("profile") or {}
    name = profile.get("call_them") or profile.get("name") or "Boss"
    greeting = random.choice((
        f"Hey {name}! I am back online, ready to help.",
        f"Systems back on. Good to see you, {name}.",
        f"Hello, hello! Rocky is awake again, {name}.",
    ))
    body.send_command("wave")
    speak(greeting)
    if not body.send_command("dead"):
        body.set_expression("sleepy")

def note_boot_greeting_sent():
    """Called by run() right after its own startup wave succeeds, so
    maybe_greet_on_reboot's first poll doesn't re-fire a duplicate greeting
    for the boot it just handled."""
    global _greeted_this_boot
    _greeted_this_boot = True

def maybe_greet_on_reboot(memory):
    """Catches a genuine physical power-cycle of the BODY BOARD, not just
    the brain process -- run()'s boot greeting only fires once, when this
    Python process starts, but this whole session's actual recovery habit
    for a dropped WiFi connection has been power-cycling just the ESP32
    board while brain.py keeps running. That case never touched run(), so
    the wave silently never fired for it -- confirmed as the real cause of
    "it doesn't wave when I power it on."

    Also covers the case where run()'s own startup wave failed (board not
    yet joined WiFi when Python started, so send_command couldn't connect):
    _greeted_this_boot stays False until note_boot_greeting_sent() confirms
    a real success, so this function's first successful poll fires the
    catch-up greeting instead of silently giving up on that boot forever.

    The board's /status uptime_s counter only ever increases while it's
    alive and resets to near-zero on an actual reset (unlike a dropped TCP
    connection, which also fires on a plain WiFi hiccup with no real
    reboot -- esp_link already reconnects through those on its own and
    they should stay silent). Polling it here, on the same idle tick as
    the other proactive behaviors, catches a reboot within one poll
    interval regardless of whether brain.py was already running."""
    global _last_reboot_poll, _last_known_uptime, _greeted_this_boot
    now = time.time()
    if now - _last_reboot_poll < _REBOOT_POLL_INTERVAL_S:
        return
    _last_reboot_poll = now
    try:
        r = requests.get(f"http://{esp_link.HOST}:8080/status", timeout=3)
        up = int(dict(re.findall(r"(\w+)=(\S+)", r.text)).get("uptime_s", -1))
    except Exception:
        return  # board unreachable right now -- next poll will catch it
    if up < 0:
        return

    if not _greeted_this_boot:
        _greeted_this_boot = True
        _last_known_uptime = up
        _fire_boot_greeting(memory, f"startup wave never confirmed, board now up {up}s")
        return

    prev, _last_known_uptime = _last_known_uptime, up
    if prev is None or up >= prev:
        return  # normal monotonic uptime -- no reboot happened
    _fire_boot_greeting(memory, f"uptime {prev}s -> {up}s")

def maybe_greet_on_sight(memory):
    """Called on every idle tick of the wake-word wait loop (see
    listen.py's wait_for_wake_and_capture on_idle_tick) -- checks the camera
    for a confidently recognized face and greets unprompted if it's been a
    while since that person was last greeted. Deliberately does nothing for
    an unrecognized face (no "who is this stranger" commentary -- that would
    be both wrong often, given camera/lighting quality, and unwelcome) and
    fails completely silently on any camera/network error, since this runs
    every few seconds indefinitely and must never spam errors or disrupt the
    wake-word loop it's piggybacking on."""
    try:
        frame_path = get_current_frame(save_path="_proactive_frame.jpg", timeout=4)
    except Exception:
        return

    try:
        # Keep track of what is in view (for "where did I leave my phone"):
        # a higher bar than an explicit "what do you see", so a stray false
        # detection is not remembered as a sighting.
        counts = objects.count_labels(objects.detect(frame_path, conf=0.5))
        counts.pop("person", None)
        if counts:
            STORE.note_sightings(counts)
        name, score = face_id.identify(frame_path, min_score=face_id.PROACTIVE_MATCH_THRESHOLD)
    except Exception:
        return
    finally:
        if os.path.exists(frame_path):
            os.remove(frame_path)

    if not name:
        return
    STORE.note_sightings({name: 1})

    entry = memory.get("people", {}).get(name, {})
    last_greeted = entry.get("last_greeted_at")
    if last_greeted and (time.time() - last_greeted) < PROACTIVE_GREET_COOLDOWN_S:
        return

    print(f"Intent: PROACTIVE_GREET ({name}, score={score:.3f})")
    body.set_expression("happy")
    reply = greet_person(display_name_for(name, memory), memory)
    speak(reply)
    body.greet_gesture()
    if not body.send_command("dead"):  # back to idle -- the greet wave stood it up
        body.set_expression("sleepy")
    _record_greeting(name, memory)

def handle_enroll(name, memory):
    print(f"Intent: ENROLL ({name})")
    speak(f"Okay, {name}. Look at me and hold still for a moment.")
    frames = capture_frames(5, 0.5)
    ok, message = face_id.enroll_frames(name, frames) if frames else (False, "No frames from the camera.")
    _cleanup_frames(frames)
    print(message)

    if ok:
        body.set_expression("surprised")
        reply = welcome_new_person(name, memory)
        people = memory.setdefault("people", {})
        people[name] = {"times_seen": 1, "last_seen": time.strftime("%Y-%m-%d %H:%M")}
        save_memory(memory)
        STORE.add_fact("person", f"Rocky recognises {name} by face.", subject=name, source="face", sensitive=True)
    else:
        reply = "Hmm. Did not see face clearly. Stand closer to camera, try again?"

    speak(reply)
    return reply

# ---------------------------------------------------------------------------
# Seeing: fast object detector for "what do you see / do you see my phone",
# the small vision model for "what am I holding", and room memory.
# ---------------------------------------------------------------------------
def ago_phrase(ts):
    d = time.time() - ts
    if d < 90:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} minutes ago"
    if d < 86400 and time.strftime("%d", time.localtime(ts)) == time.strftime("%d"):
        h = int(d // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    return when_phrase(ts)

def _vision_loaded():
    try:
        return any(m.model.startswith(_vision.MODEL.split(":")[0]) for m in ollama.ps().models)
    except Exception:
        return True

def _names_in(frame_path):
    """[(name or None)] for every face in view."""
    try:
        return [name for name, _, _ in face_id.identify_all(frame_path)]
    except Exception as e:
        print(f"Face check failed: {e}")
        return []

def _people_phrase(names, persons_seen=0):
    known = [n for n in names if n]
    unknown = max(len(names), persons_seen) - len(known)
    bits = list(known)
    if unknown:
        bits.append("someone I do not know" if unknown == 1 else f"{unknown} people I do not know")
    return bits[0] if len(bits) == 1 else (", ".join(bits[:-1]) + " and " + bits[-1] if bits else "")

def _record_sightings(counts, names):
    seen = {k: v for k, v in counts.items() if k != "person"}
    for n in names:
        if n:
            seen[n] = 1
    if seen:
        STORE.note_sightings(seen)

def _position_word(box, frame_w):
    cx = box[0] + box[2] / 2
    return "on the left" if cx < frame_w / 3 else "on the right" if cx > 2 * frame_w / 3 else "in the middle"

def _grab(save_path="_see.jpg"):
    try:
        return get_current_frame(save_path=save_path, timeout=8)
    except Exception as e:
        print(f"Camera frame failed: {e}")
        return None

def _sentences(text, n):
    """First n complete sentences (a token-capped model answer often ends mid-sentence)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    complete = [p for p in parts if re.search(r"[.!?]$", p)] or parts[:1]
    return " ".join(complete[:n])

def _second_person(text):
    text = re.sub(r"\bmy\b", "your", text, flags=re.I)
    return re.sub(r"\bI\b", "you", text)

def _where_from_memory(word):
    """"Where is my X" that the camera can't answer: use only what the person
    actually told Rocky. Never let the chat model guess a location."""
    hits = STORE.search(word, kinds=("note", "gear", "preference", "decision", "project"), min_overlap=0.5, limit=1)
    if hits:
        reply = f"You told me: {_second_person(hits[0]['text']).rstrip('.')}."
    else:
        reply = f"I do not know where your {word} is. I have not seen it, and you have not told me."
    speak(reply)
    return reply

def _cant_see():
    reply = "Cannot see right now. Camera is not answering."
    speak(reply)
    return reply

def _cleanup(*paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass

def _deep_question(text):
    t = text.lower()
    if "holding" in t or "in my hand" in t:
        return "What is the person holding in their hand? Answer in one short sentence. If their hands are empty, say so."
    if "wearing" in t or "shirt" in t:
        return "What is the person wearing? Answer in one short sentence."
    if "read" in t:
        return "Read out any text you can see. If there is none, say so."
    return "What is the main object or thing in front of the camera? Answer in one or two short sentences."

def handle_see(text, deep):
    print(f"Intent: SEE ({'deep' if deep else 'fast'})")
    body.set_expression("curious")
    path = _grab()
    if not path:
        return _cant_see()
    try:
        dets = objects.detect(path)
        counts = objects.count_labels(dets)
        names = _names_in(path)
        _record_sightings(counts, names)
        print(f"[SEE] objects={counts} people={names}")
        if not deep:
            who = _people_phrase(names, counts.get("person", 0))
            things = objects.describe(counts)
            parts = []
            if who:
                parts.append(f"I see {who}.")
            if things:
                parts.append(("Also " if who else "I see ") + things + ".")
            if parts:
                reply = " ".join(parts)
                if things:
                    body.send_command("point")  # a little physical emphasis when there's an object to gesture at
                speak(reply)
                return reply
        if not _vision_loaded():
            speak("Let me look closely.")
        question = _deep_question(text) if deep else _vision.DEFAULT_QUESTION
        reply = _sentences(look_at(path, question), 2)
        reply = reply[:1].upper() + reply[1:]
        speak(reply)
        return reply
    finally:
        _cleanup(path, path.rsplit(".", 1)[0] + "_small.jpg")

def handle_find(word, where=False):
    label = objects.resolve_label(word)
    print(f"Intent: {'WHERE' if where else 'FIND'} ({word!r} -> {label})")
    body.set_expression("curious")
    if label is None and where:
        return _where_from_memory(word)
    path = _grab()
    if label is None:
        if not path:
            return _cant_see()
        try:
            q = f"Is there {'an' if word[0] in 'aeiou' else 'a'} {word} in this image? Answer yes or no, then in a few words say where."
            if not _vision_loaded():
                speak("Let me look closely.")
            reply = look_at(path, q)
            speak(reply)
            return reply
        finally:
            _cleanup(path, path.rsplit(".", 1)[0] + "_small.jpg")

    seen_now = None
    if path:
        try:
            dets = objects.detect(path)
            counts = objects.count_labels(dets)
            _record_sightings(counts, [])
            hit = [d for d in dets if d[0] == label]
            if hit:
                import cv2 as _cv2
                w = _cv2.imread(path).shape[1]
                seen_now = (len(hit), _position_word(hit[0][2], w))
        finally:
            _cleanup(path)
    if seen_now:
        n, pos = seen_now
        reply = f"Yes. I see {objects.describe({label: n}, ignore=())} {pos}."
    else:
        last = STORE.last_sighting(label)
        if last:
            reply = f"Not in view right now. Last time I saw {'your ' if where else 'a '}{label} was {ago_phrase(last[0])}."
        elif where:
            return _where_from_memory(word)
        elif path is None:
            return _cant_see()
        else:
            reply = f"No. I do not see a {label}, and I have not seen one before."
    speak(reply)
    return reply

def _scan_counts(frames):
    """Best count of each object across several frames (a thing missed in one
    blurry frame is usually caught in another)."""
    best = {}
    for f in frames:
        for label, n in objects.count_labels(objects.detect(f)).items():
            best[label] = max(best.get(label, 0), n)
    best.pop("person", None)
    return best

def handle_scan_room(room):
    print(f"Intent: SCAN_ROOM ({room})")
    body.set_expression("curious")
    speak(f"Okay. Looking at your {room} now.")
    frames = capture_frames(3, 0.6)
    if not frames:
        return _cant_see()
    try:
        counts = _scan_counts(frames)
        caption = look_at(frames[len(frames) // 2],
                          "Describe this room in two short sentences: the furniture, colours, and things on the surfaces. Do not describe any people.")
        things = objects.describe(counts, limit=10)
        date = time.strftime("%d %b")
        caption = _sentences(caption, 3)
        text = f"Your {room}, scanned on {date}: {caption}" + (f" Objects seen: {things}." if things else "")
        STORE.add_fact("room", text, subject=room, source="camera")
        STORE.set_room_objects(room, counts)
        STORE.note_sightings(counts)
        print(f"[ROOM] {text}")
    finally:
        _cleanup_frames(frames)
        _cleanup(*[f.rsplit(".", 1)[0] + "_small.jpg" for f in frames])
    reply = f"Room learned. {_sentences(caption, 1)}" + (f" I also see {things}." if things else "") + " I will remember it."
    speak(reply)
    return reply

def handle_room_query(room):
    print(f"Intent: ROOM_QUERY ({room})")
    latest = STORE.latest_room_fact(room) or (STORE.latest_room_fact("room") if room != "room" else None)
    if not latest:
        reply = f"I have not learned your {room} yet. Say, remember this room, and I will look."
    else:
        m = re.match(r"Your [\w ]+, scanned on (.+?): (.*?)(?: Objects seen: (.*)\.)?$", latest[0], re.S)
        if m:
            reply = f"From {when_phrase(latest[1])}. {_sentences(m.group(2), 2)}" + (f" Objects: {m.group(3)}." if m.group(3) else "")
        else:
            reply = latest[0]
    speak(reply)
    return reply

def handle_room_diff(room):
    print(f"Intent: ROOM_DIFF ({room})")
    old = STORE.get_room_objects(room)
    if not old:
        reply = f"I have no scan of your {room} to compare with. Say, remember this room, first."
        speak(reply)
        return reply
    frames = capture_frames(2, 0.6)
    if not frames:
        return _cant_see()
    try:
        cur = _scan_counts(frames)
    finally:
        _cleanup_frames(frames)
    ts, before = old
    new = {l: n for l, n in cur.items() if n > before.get(l, 0)}
    gone = {l: n for l, n in before.items() if n > cur.get(l, 0)}
    parts = []
    if new:
        parts.append("New since I looked: " + objects.describe(new, ignore=()) + ".")
    if gone:
        parts.append("I do not see " + objects.describe(gone, ignore=()) + " anymore.")
    reply = " ".join(parts) if parts else f"Nothing changed that I can tell, compared with {ago_phrase(ts)}."
    speak(reply)
    return reply

def handle_vision_command(cmd, text):
    kind = cmd[0]
    if kind == "scan_room":
        return handle_scan_room(cmd[1])
    if kind == "room_query":
        return handle_room_query(cmd[1])
    if kind == "room_diff":
        return handle_room_diff(cmd[1])
    if kind == "find":
        return handle_find(cmd[1])
    if kind == "where":
        return handle_find(cmd[1], where=True)
    if kind == "see":
        return handle_see(text, cmd[1])
    return None

def handle_remember(fact_text, memory):
    """Stores an explicitly-requested fact permanently, verbatim. Unlike the
    passive facts/knowledge pipeline (which gets summarized and can be dropped
    by reflect() if it doesn't seem important enough), anything the user
    deliberately says to remember goes straight into "notes" and stays there
    exactly as stated -- this is the actual "perfect memory" guarantee."""
    print(f"Intent: REMEMBER ({fact_text})")
    notes = memory.setdefault("notes", [])
    if fact_text.lower() not in (n.lower() for n in notes):
        notes.append(fact_text)
        memory["notes"] = notes[-MAX_NOTES:]
        save_memory(memory)
    cur = STORE.current_project()
    STORE.add_fact("note", fact_text, source="explicit", project_id=cur["id"] if cur else None)
    reply = random.choice((
        "Got it. I will remember that.",
        "Noted. Not forgetting that one.",
        "Okay. Stored, safe. Will not forget.",
    ))
    body.send_command("nod")
    speak(reply)
    return reply

def _say_list(items, limit=5):
    items = items[:limit]
    if len(items) == 1:
        return items[0]
    return "; ".join(f"{i + 1}, {t}" for i, t in enumerate(items))

def handle_memory_command(cmd, raw_text, memory):
    """Deterministic memory / project / task commands -- answered straight
    from the database (no LLM), so they are instant and cannot invent
    anything. Returns the spoken reply, or None to let normal routing
    continue (e.g. 'how is the weather' is not a project-status question)."""
    intent, arg = cmd
    cur = STORE.current_project()
    print(f"Intent: MEMORY {intent} ({arg})")

    if intent == "add_task":
        if not arg:
            return None
        STORE.add_task(arg, cur["id"] if cur else None)
        where = f" on {cur['name']}" if cur else ""
        reply = f"Task added{where}. {arg}."
    elif intent == "list_tasks":
        proj = STORE.find_project(arg) if arg else cur
        tasks = STORE.open_tasks(proj["id"] if proj else None)
        label = f" on {proj['name']}" if proj else ""
        if not tasks:
            reply = f"Nothing open{label}. All clear. Suspicious, yes?"
        else:
            reply = f"{len(tasks)} open{label}. " + _say_list([t["text"] for t in tasks])
    elif intent == "done_task":
        done = STORE.complete_task(arg, cur["id"] if cur else None)
        if not done and not re.search(r"\bmark\b", raw_text, re.I):
            return None   # "I finished my circuit board" is just chatter, not a task update
        reply = f"Done. Finished: {done}." if done else "I do not have a task like that. Tell me its name again?"
    elif intent == "status":
        proj = STORE.find_project(arg)
        if not proj:
            return None   # "how is the weather" etc -- not about a project
        st = STORE.project_status(proj)
        bits = [f"{proj['name']}."]
        bits.append(f"{len(st['open'])} open, {st['done']} finished.")
        if st["open"]:
            bits.append("Next: " + st["open"][0]["text"] + ".")
        for r in st["recent"][:2]:
            bits.append(r["text"])
        reply = " ".join(bits)
    elif intent == "list_projects":
        projects = STORE.list_projects()
        if not projects:
            reply = "No projects yet. Say, I am building something, and I will remember."
        else:
            names = [p["name"] + (" -- current" if cur and p["id"] == cur["id"] else "") for p in projects]
            reply = "Projects. " + _say_list(names, 6)
    elif intent == "search":
        said = STORE.recall_lines(arg or raw_text, limit=2)
        known = STORE.search(arg or raw_text, kinds=("note", "preference", "gear", "person", "decision", "project", "task"),
                             limit=2, min_overlap=0.25)
        bits = []
        for ts, t in said:
            bits.append(f"{when_phrase(ts)} you told me: {t}")
        for k in known:
            if not any(k["text"] in b for b in bits):
                bits.append(f"I have on record: {k['text']}")
        reply = ". ".join(bits[:3]) if bits else "I do not remember anything about that."
    elif intent == "forget":
        removed = STORE.forget(arg)
        reply = f"Forgotten: {removed}" if removed else "I could not find that in my memory."
    elif intent == "about_me":
        facts = STORE.facts_about_user()
        if not facts:
            reply = "Do not know much yet. Tell me things. I remember exactly what you say."
        else:
            reply = "I know. " + " ".join(t for _, t in facts[:6])
    elif intent == "briefing":
        since = time.time() - 36 * 3600
        rows, turns = STORE.activity_since(since)
        bits = []
        if cur:
            open_tasks = STORE.open_tasks(cur["id"])
            bits.append(f"Current project, {cur['name']}. {len(open_tasks)} tasks open.")
        for r in rows[-3:]:
            bits.append(f"{when_phrase(r['ts'])}: {r['text']}")
        bits.append(f"We talked {turns} times since yesterday.")
        reply = " ".join(bits)
    else:
        return None

    if intent in ("add_task", "done_task"):
        body.send_command("nod")  # quick physical "got it" before speaking
    speak(reply)
    return reply


# One or two short reaction lines per trick -- the movement itself is the
# show, so these stay brief (same lesson as everywhere else: a small model
# rambling while Rocky is also mid-motion just adds dead air).
_MOVE_LINES = {
    "wave": ("Hi! Hi! You see me wave?", "Wave, yes. I am friendly robot."),
    "dance": ("Watch this. Best dance in whole workshop.", "Dance time! No judging, please."),
    "swim": ("Swimming! No water needed, I am talented.", "Look, I swim on dry land. Impressive, yes?"),
    "point": ("There. I point. Good pointing, yes?", "I point now. Very precise."),
    "pushup": ("Pushups! I am very strong robot.", "One! Two! Okay, I am tired now."),
    "bow": ("Thank you, thank you. I bow now.", "A bow, for my adoring audience of one."),
    "cute": ("Cute mode. Boss, look how cute I am.", "I am cute now. Very cute. Say it."),
    "freaky": ("Freaky time! A little weird, a little fun.", "Getting freaky. Do not worry, still friendly."),
    "worm": ("The worm! My best trick, I think.", "Worming now. This one took practice."),
    "shake": ("Shake shake shake! Good wiggle.", "I shake now. Very good wiggle, yes?"),
    "shrug": ("Shrug. I do not know either, Boss.", "Shrug. Some things, nobody knows."),
    "crab": ("Crab walk! Sideways is underrated.", "I am crab now. Sideways robot."),
    "dead": ("Playing dead. Very convincing, yes?", "Dead now. Do not worry, I am fine."),
    "forward": ("Walking forward. Watch my steps.", "Forward! Slow and careful, like always."),
    "backward": ("Backing up now. Careful, careful.", "Walking back. Good balance, yes?"),
    "left": ("Turning left. Smooth, yes?", "Left turn. Very smooth robot."),
    "right": ("Turning right. Watch me go.", "Right turn now. Good robot."),
    "stand": ("Standing up. Ready for anything.", "On my feet now. Let us go."),
    "rest": ("Resting legs. Ahh, better.", "Sitting down. My legs say thank you."),
}

def handle_movement_command(pose_name):
    print(f"Intent: MOVEMENT ({pose_name})")
    ok = body.send_command(pose_name)
    line = random.choice(_MOVE_LINES.get(pose_name, ("Okay.",)))
    if not ok:
        line = "Hmm. My legs did not listen just now. Try again?"
    speak(line)
    return line

def handle_command(text, memory, wav_path=None):
    lowered = text.lower()

    # First, so "sarcastic mode" / "volume 50" are never swallowed by a later
    # intent. Both parsers are deterministic (voice_commands.py, unit-tested).
    volume_cmd = parse_volume(text)
    if volume_cmd:
        return handle_volume(volume_cmd, memory)

    mode_cmd = parse_mode(text)
    if mode_cmd == ("roast", True) and re.search(r"\broast me\b", lowered):
        if _SENSITIVE.search(text):
            return chat_streaming(text, memory)   # upset person: never roast, mode_for() answers kindly
        return handle_roast_request(text, memory)
    if mode_cmd:
        return handle_mode(mode_cmd[0], mode_cmd[1], memory)

    movement_cmd = parse_movement_command(text)
    if movement_cmd:
        return handle_movement_command(movement_cmd)

    vision_cmd = parse_vision_command(text)
    if vision_cmd:
        reply = handle_vision_command(vision_cmd, text)
        if reply is not None:
            return reply

    memory_cmd = parse_memory_command(text)
    if memory_cmd:
        reply = handle_memory_command(memory_cmd, text, memory)
        if reply is not None:
            return reply

    if any(trigger in lowered for trigger in REFLECT_TRIGGERS):
        return handle_reflect(memory)

    if any(trigger in lowered for trigger in RECALL_TRIGGERS):
        return handle_recall(memory)

    if any(trigger in lowered for trigger in MOOD_TRIGGERS):
        return handle_mood(text, wav_path, memory)

    # Checked before needs_internet(): new "who is"/"do you know who"
    # internet-search patterns (added to ground questions about real-world
    # people in an actual search instead of the small model's fuzzy
    # knowledge) would otherwise swallow "who is this" / "do you know who
    # I am" -- face-recognition phrases that should hit handle_identify()
    # instead.
    if any(trigger in lowered for trigger in SELF_DIAGNOSE_TRIGGERS):
        return handle_self_diagnose()

    if any(trigger in lowered for trigger in ABOUT_TRIGGERS):
        return handle_about()

    if any(trigger in lowered for trigger in CAPABILITIES_TRIGGERS):
        return handle_capabilities()

    if any(trigger in lowered for trigger in WHO_TRIGGERS):
        return handle_identify(memory)

    if needs_internet(lowered):
        return answer_with_search(text, memory)

    enroll_name = match_enroll_name(lowered)
    if enroll_name:
        return handle_enroll(enroll_name.title(), memory)

    # Checked after enroll_name: a face-enrollment phrase like "this is my
    # mom, remember her" should hit handle_enroll(), not get swallowed here
    # as a generic fact.
    remember_text = match_remember_fact(text)
    if remember_text:
        return handle_remember(remember_text, memory)

    if any(trigger in lowered for trigger in ROCKY_SLEEP_TRIGGERS):
        print("Intent: ROCKY_SLEEP")
        return handle_rocky_sleep(memory)

    if any(trigger in lowered for trigger in USER_SLEEP_TRIGGERS):
        print("Intent: SLEEP")
        return handle_sleep(memory)

    print("Intent: CHAT")
    return chat_streaming(text, memory)  # speaks incrementally itself

MAX_FOLLOWUP_TURNS = 6  # safety cap on how many conversation turns can chain
# without re-hearing the wake word -- keeps a stuck/misfiring loop from
# holding the conversation open forever.

# Lets the user explicitly end a conversation instead of relying on a
# spoken sign-off, which turned out to be a real self-triggering risk (see
# git history / session notes): Rocky's own "Here if you need me" could get
# picked up by its own mic and self-trigger a new wake event, which finds
# silence, which speaks again, which self-triggers again -- a cascading
# loop. Explicit phrases sidestep that risk entirely: no unprompted speech
# means no new opportunity to self-trigger.
CONVERSATION_END_TRIGGERS = (
    "go back to idle", "back to idle", "we are done", "we're done",
    "that's all", "that is all", "goodbye", "good bye", "bye rocky",
    "i'm done", "im done", "end conversation", "stop listening",
    "that's it", "we are finished", "we're finished",
    "go back to sleep", "go to sleep",
)

def is_conversation_end(text):
    lowered = text.lower()
    return any(trigger in lowered for trigger in CONVERSATION_END_TRIGGERS)

def _transcribe_and_handle(wav_path, memory):
    """Transcribes one captured utterance. Returns True if it was routed
    and answered as a normal turn, "end" if the user explicitly asked to
    end the conversation (not answered as a chat message), or False if
    nothing usable was captured (silence) -- the caller uses this to decide
    whether to keep the conversation going."""
    turn_start = time.time()
    trace_start()
    text = transcribe(wav_path)
    print(f"You said: {text}")
    trace("transcribed")
    if text and is_junk_transcript(text):
        print(f"(Ignoring what is almost certainly noise, not speech: {text!r})")
        return False
    if not text:
        return False

    # Attentive default face for the turn -- individual handlers (identify/
    # enroll/rocky-sleep) may override to something more specific
    # (happy/surprised/sleepy) while they run; listen_and_respond() resets
    # back to neutral once the whole wake-to-idle cycle finishes.
    body.set_expression(MODES[current_mode]["expression"])

    if is_conversation_end(text):
        print("You said to end the conversation -- signing off, back to idle.")
        body.set_expression("sleepy")
        # Reuses ROCKY_SLEEP_LINES (Rocky's own established "I watch, you
        # rest" voice, already written for the sleep intent below) as the
        # sign-off, rather than a generic line -- deliberately original
        # dialogue in Rocky's voice, not a reproduction of the film's
        # actual script.
        speak(random.choice(ROCKY_SLEEP_LINES))
        return "end"
    STORE.log_turn("user", text, current_mode)
    for learned in STORE.learn(text):
        print(f"[MEMORY] learned: {learned}")
    reply = handle_command(text, memory, wav_path=wav_path)
    if isinstance(reply, str):
        STORE.log_turn("rocky", reply, current_mode)
    print(f"[TIMING] Full turn: {time.time() - turn_start:.1f}s")
    trace("turn finished")
    return True

def listen_and_respond():
    memory = load_memory()

    try:
        # Blocks until the ESP32's on-device wake-word detector hears "Hey
        # Rocky" and pushes the follow-up utterance -- no fixed wait here,
        # since that could be any length of time. Between reads (every ~6s
        # of silence), checks the camera for a known face to greet
        # unprompted.
        def _on_idle_tick():
            maybe_greet_on_sight(memory)
            maybe_annoying_fidget(memory)
            maybe_idle_blink()
            maybe_greet_on_reboot(memory)
        wav_path = wait_for_wake_and_capture(on_idle_tick=_on_idle_tick)
        if wav_path is None:
            print("No audio captured.")
            return

        result = _transcribe_and_handle(wav_path, memory)
        if result != True:
            if result == False:
                # Actually listen again instead of quitting right back to
                # requiring "Hey Rocky" -- that was the original bug. Stay
                # silent about it rather than speaking first: we can't tell a
                # genuine miss apart from the wake word self-triggering on
                # Rocky's own voice, and speaking here previously fed that
                # exact cascade. One quiet retry; if that's also empty, give
                # up quietly.
                print("Heard nothing usable -- listening once more, quietly.")
                time.sleep(0.3)
                wav_path = capture_audio()
                if wav_path is None or _transcribe_and_handle(wav_path, memory) != True:
                    print("Still nothing -- back to idle.")
                    return
            else:
                return  # result == "end"

        # Real conversations don't require saying someone's name before every
        # sentence -- after answering, keep listening for a follow-up for a
        # few turns without requiring "Hey Rocky" again, the same way
        # Xiaozhi-style assistants do. Reuses the ESP32's plain "listen"
        # command (still there from before the wake-word work) to actively
        # capture one more utterance instead of waiting passively for
        # another wake event. Ends quietly -- no spoken sign-off -- the
        # moment a follow-up comes back silent, hits MAX_FOLLOWUP_TURNS, or
        # the user explicitly says something like "we're done" / "go back
        # to idle".
        for _ in range(MAX_FOLLOWUP_TURNS):
            # Same reasoning as the firmware's post-chime delay: Rocky's own
            # just-finished voice is still reverberating in the room, and
            # starting the next capture's noise-floor calibration immediately
            # risks mistaking that tail for ambient noise.
            time.sleep(0.3)
            wav_path = capture_audio()
            if wav_path is None or _transcribe_and_handle(wav_path, memory) != True:
                break
    finally:
        # Whatever happened during this wake-to-idle cycle (a greeting left
        # Rocky "happy", a mode reaction left it something else, etc.),
        # always settle back to idle once fully done -- runs on every exit
        # path (early return, break, or falling off the end). "dead" (the
        # flattened resting pose) already sets the closed-eyes "sleepy"
        # expression itself, so it's visually obvious from across the room
        # whether Rocky is actually listening or just resting; a fresh wake
        # word stands it back up (see tickWakeWord's g_isDead check).
        if not body.send_command("dead"):
            body.set_expression("sleepy")  # at least get the face right if the pose command itself failed

def warm_chat_prefix(memory):
    """Processes the real chat prompt once at startup so the LLM server has the
    constant part of it (personality + examples) cached before the first
    question -- otherwise the first turn of every session pays the full
    ~1100-token prefill (measured 3-5s)."""
    try:
        ollama.chat(model=TEXT_MODEL, messages=chat_messages("hello", memory, current_mode),
                    options={"num_predict": 1}, keep_alive=-1)
    except Exception as e:
        print(f"Chat prefix warm-up failed (harmless): {e}")

def warm_vision():
    """Loads the object detector and the small vision model in the background
    so the first "what do you see" is fast instead of a ~11 s cold load."""
    try:
        import numpy as _np
        blank = _np.full((480, 640, 3), 127, _np.uint8)
        objects.detect(blank)
        path = os.path.join(tempfile.gettempdir(), "rocky_vision_warm.jpg")
        cv2.imwrite(path, blank)
        vision_warmup(path)
        print("Vision warmed up.")
    except Exception as e:
        print(f"Vision warm-up failed (it will load on first use instead): {e}")

def run():
    """Keep Rocky listening turn after turn until interrupted, instead of
    exiting after a single exchange."""
    memory = load_memory()
    if reflection_due(memory):
        print("Reflecting on recent memory before starting...")
        ok, message = reflect(memory)
        print(message)

    imported = STORE.import_legacy(memory, fallback_ts=os.path.getmtime(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else None)
    if imported:
        print(f"Imported {imported} items from the old memory.json into the searchable store.")
    warm_chat_prefix(memory)
    threading.Thread(target=warm_vision, daemon=True).start()

    # Boot greeting: wave hello and say something, so it's obvious Rocky is
    # actually up and ready rather than silently sitting there. Best-effort
    # -- if the body board isn't connected yet, this just quietly no-ops
    # rather than blocking startup on it.
    profile = memory.get("profile") or {}
    name = profile.get("call_them") or profile.get("name") or "Boss"
    greeting = random.choice((
        f"Hey {name}! I am online, ready to help.",
        f"Systems on. Good to see you, {name}.",
        f"Hello, hello! Rocky is awake, {name}.",
    ))
    if body.send_command("wave"):
        note_boot_greeting_sent()
    speak(greeting)
    body.send_command("dead")

    print("\nRocky is listening continuously. Press Ctrl+C to stop.")
    while True:
        try:
            listen_and_respond()
        except KeyboardInterrupt:
            print("\nGoodnight.")
            break
        except Exception as e:
            # One bad turn (camera hiccup, serial timeout, etc.) shouldn't kill
            # the whole session.
            print(f"Error during turn: {e}")

if __name__ == "__main__":
    run()
