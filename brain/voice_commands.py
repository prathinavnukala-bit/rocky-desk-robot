"""
Parsers for spoken control phrases that must be recognised deterministically
(never by the LLM): volume, and personality-mode on/off. Kept free of any
hardware or model imports so they can be unit-tested in isolation
(see test_voice_commands.py).
"""
import re

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_number(text):
    """'seventy five' / 'seventy-five' / 'one hundred' / '50' -> int, else None."""
    text = text.lower().replace("-", " ").strip()
    if re.fullmatch(r"\d{1,3}", text):
        return int(text)
    total, seen = 0, False
    for w in text.split():
        if w == "and":
            continue
        if w in _UNITS:
            total += _UNITS[w]; seen = True
        elif w in _TENS:
            total += _TENS[w]; seen = True
        elif w == "hundred":
            total = max(total, 1) * 100; seen = True
        else:
            return None
    return total if seen else None


_NUM = r"(\d{1,3}|(?:(?:zero|one|two|three|four|five|six|seventeen|seventy|seven|eighteen|eighty|eight|nineteen|ninety|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|twenty|thirty|forty|fifty|sixty|hundred|and)\b[ -]?)+)"

# "volume 50", "set the volume to fifty percent", "volume at seventy five"
_SET_VOL = re.compile(r"\bvolume\b\s*(?:level\s*)?(?:to|at|of|=)?\s*(?:a\s+)?" + _NUM + r"\s*(?:%|percent|per cent)?", re.I)
# "50 percent volume", "fifty percent" only counts if 'volume' or 'sound' is nearby
_PCT_VOL = re.compile(r"(?:\ba\s+)?" + _NUM + r"\s*(?:%|percent|per cent)\s*(?:volume|sound)\b", re.I)
_SET_IT = re.compile(r"\b(?:set|turn|put|make)\s+(?:it|the sound|the speaker)\s+(?:to|at)\s*(?:a\s+)?" + _NUM + r"\s*(?:%|percent|per cent)", re.I)

_MAX = re.compile(r"\b(?:max(?:imum)?|full|highest)\s+volume\b|\bvolume\s+(?:to\s+|at\s+)?(?:max(?:imum)?|full|highest)\b", re.I)
_MIN = re.compile(r"\b(?:min(?:imum)?|lowest)\s+volume\b|\bvolume\s+(?:to\s+|at\s+)?(?:min(?:imum)?|lowest)\b", re.I)
_UNMUTE = re.compile(r"\bunmute\b|\bun-mute\b|\bsound on\b", re.I)
_MUTE = re.compile(r"\bmute\b|\bsilent mode\b", re.I)
_QUERY = re.compile(r"\b(?:what(?:'s| is)? (?:the |your )?(?:current )?volume(?!\s+of\b)|how loud (?:are you|is it)|(?:current )?volume level)\b", re.I)
_UP = re.compile(r"\bvolume up\b|\bturn (?:it |the volume |the sound )?up\b|\blouder\b|\b(?:increase|raise|boost) (?:the )?volume\b|\bspeak up\b|\bcan'?t hear you\b|\btoo (?:quiet|soft)\b", re.I)
_DOWN = re.compile(r"\bvolume down\b|\bturn (?:it |the volume |the sound )?down\b|\bquieter\b|\bsofter\b|\b(?:lower|decrease|reduce) (?:the )?volume\b|\bspeak (?:softer|lower|quieter)\b|\btoo (?:loud|noisy)\b", re.I)
_BIT = re.compile(r"\b(?:a (?:bit|little|touch)|slightly)\b", re.I)
_LOT = re.compile(r"\b(?:a lot|much|way|way more|way too)\b", re.I)

STEP_NORMAL, STEP_SMALL, STEP_BIG = 15, 7, 30


def parse_volume(text):
    """Returns None if this isn't a volume command, else a tuple:
       ("set", pct) | ("up", step) | ("down", step) | ("mute",) | ("unmute",) | ("query",)"""
    t = text.lower()

    if _UNMUTE.search(t):
        return ("unmute",)
    if _MUTE.search(t):
        return ("mute",)
    if _MAX.search(t):
        return ("set", 100)
    if _MIN.search(t):
        return ("set", 5)  # audible; use mute for silence

    for pat in (_SET_VOL, _PCT_VOL, _SET_IT):
        m = pat.search(t)
        if m:
            n = words_to_number(m.group(1))
            if n is not None and 0 <= n <= 100:
                return ("set", n)

    if _QUERY.search(t):
        return ("query",)

    step = STEP_SMALL if _BIT.search(t) else STEP_BIG if _LOT.search(t) else STEP_NORMAL
    if _UP.search(t):
        return ("up", step)
    if _DOWN.search(t):
        return ("down", step)
    return None


# ---------------------------------------------------------------------------
# Personality modes: explicit on/off phrasing.
# ---------------------------------------------------------------------------
_MODE_NAMES = {
    "sarcastic": r"sarcastic|sarcasm",
    "roast": r"roast|roasting",
    "humour": r"humou?r|funny|joke|jokes|joking",
    "annoying": r"annoying",
}

def parse_mode(text):
    """Returns None, or (mode, on_off) where mode is one of MODE names or
    'normal'. Handles: "sarcastic mode", "sarcastic mode on/off", "turn on
    roast mode", "turn off sarcasm", "stop being sarcastic", "roast me",
    "normal mode", "be serious"."""
    t = text.lower().strip()

    if re.search(r"\bnormal mode\b|\bback to normal\b|\bbe normal\b|\bserious mode\b|\bbe serious\b|\bstop (?:all )?(?:the )?(?:modes?|jokes|joking)\b", t):
        return ("normal", False)

    for mode, names in _MODE_NAMES.items():
        named = re.search(rf"\b(?:{names})\b", t)
        if not named:
            continue
        mode_phrase = re.search(rf"\b(?:{names})\s+mode\b|\bmode\b", t) or re.search(rf"\bbe\s+(?:{names})\b|\bstart\s+being\s+(?:{names})\b", t)
        off = re.search(rf"\b(?:turn|switch)\s+off\b|\b(?:mode|{names})\s+off\b|\bstop\s+(?:being\s+)?(?:{names})\b|\bstop\s+(?:the\s+)?(?:{names})\b|\bno more\s+(?:{names})\b|\b(?:disable|deactivate|end|exit|cancel|quit)\b.*\b(?:{names})\b", t)
        on = re.search(rf"\b(?:turn|switch)\s+on\b|\b(?:mode|{names})\s+on\b|\b(?:enable|activate|start)\b|\bbe\s+(?:{names})\b|\b(?:{names})\s+mode\b", t)
        if off:
            return (mode, False)
        if mode == "roast" and re.search(r"\broast me\b", t):
            return ("roast", True)
        if on or mode_phrase:
            return (mode, True)
    return None
