"""Deterministic parsing of camera-related voice commands (no LLM needed).

parse_vision_command(text) -> None or one of:
    ("scan_room", room_name)      "remember this room", "scan my workshop"
    ("room_query", room_name)     "what's in my room", "describe my room"
    ("room_diff", room_name)      "what changed in my room", "is anything different"
    ("find", object_words)        "do you see my phone", "is there a cup"
    ("where", object_words)       "where did I leave my phone", "when did you last see my keys"
    ("see", False)                "what do you see", "look around"   (fast: object detector)
    ("see", True)                 "what am I holding", "what is this"  (deep: vision model)
"""
import re

_ROOMS = r"(?:living room|bed ?room|room|workshop|lab|office|study|kitchen|garage|desk|workspace|hall|balcony)"
_ROOM = rf"(?:my|the|this|our)\s+(?:\w+\s+)?({_ROOMS})\b"
_END = r"\s*(?:please|for me|for later|now|properly|carefully|as\s+[\w ]+|in detail)?\s*[.!?]*\s*$"

_SCAN = re.compile(
    rf"\b(?:remember|memori[sz]e|learn|scan|map|study|save|look at)\s+(?:what\s+)?{_ROOM}(?:\s+looks?\s+like)?{_END}"
    rf"|\b(?:scan|look around)\b.*\b(?:and\s+)?(?:remember|memori[sz]e)\b.*\b{_ROOMS}\b", re.I)
_ROOM_Q = re.compile(
    rf"\bwhat(?:'s| is| do you (?:see|remember|know)| have you (?:seen|learn(?:ed|t)))\s+(?:in|inside|about|on)\s+{_ROOM}"
    rf"|\bwhat does\s+{_ROOM}\s+look like\b"
    rf"|\b(?:describe|tell me about|what do you remember about|what do you know about)\s+{_ROOM}", re.I)
_DIFF = re.compile(
    r"\bwhat(?:'s| has| is)?\s+(?:changed|different|new|moved)\b|\banything\s+(?:changed|different|new|moved|missing)\b"
    r"|\bwhat did i (?:move|change)\b|\bdid (?:i|anything) (?:move|change)\b|\bis anything (?:different|missing)\b", re.I)
_FIND = re.compile(
    r"\b(?:do you see|can you see|can you spot|do you spot|is there|are there|do you notice)\s+"
    r"(?:an?\s+|any\s+|my\s+|the\s+|some\s+)?(?P<x>[a-z][a-z ]{1,28}?)(?:\s+(?:here|there|in front of you|on the (?:desk|table)|in the room|around|nearby|anywhere))?\s*[?.!]*$", re.I)
_WHERE = re.compile(
    r"\bwhere(?:'s| is| are| did i (?:leave|put|keep|place)| have i (?:left|put|kept|placed))\s+(?:my|the)\s+(?P<x>[a-z][a-z ]{1,28}?)\s*[?.!]*$"
    r"|\bwhen did you (?:last )?see\s+(?:my|the)\s+(?P<y>[a-z][a-z ]{1,28}?)\s*[?.!]*$"
    r"|\bhave you (?:seen|noticed)\s+(?:my|the)\s+(?P<z>[a-z][a-z ]{1,28}?)\s*[?.!]*$", re.I)
_SEE = re.compile(
    r"\bwhat (?:do|can|did) you see\b|\bwhat(?:'s| is| are)\s+(?:in front of you|around you|on (?:the|my) (?:desk|table)|there to see)\b"
    r"|\blook around\b|\bwhat objects\b|\bwhat can you (?:spot|detect|make out)\b|\bwhat things (?:do you|can you) see\b", re.I)
_DEEP = re.compile(
    r"\bwhat(?:'s| is| are)\s+(?:this|that|these|those)\b(?!\s+(?:supposed|mean|about|for|called))|\bwhat am i (?:holding|wearing)\b"
    r"|\bwhat(?:'s| is) in my hand\b|\bdescribe (?:this|that|what you see)\b|\b(?:look|take a look|have a look) at\b"
    r"|\btell me what (?:this|that) is\b|\bwhat (?:colou?r|kind|type) (?:is|of)\b.*\b(?:this|that|my shirt|i am holding|i'm holding)\b"
    r"|\bcan you (?:read|identify) (?:this|that)\b|\bidentify (?:this|that)\b", re.I)

_NOT_OBJECT = re.compile(r"^(?:what|that|how|why|it|this|the point|a problem|a way|anything|something|someone|anyone|somebody|a reason|"
                         r"any way|an issue|the difference|a chance|a pattern|the problem|a bug)\b", re.I)


def parse_vision_command(text):
    t = " ".join(text.lower().strip().split())

    m = _SCAN.search(t)
    if m:
        room = re.search(rf"\b({_ROOMS})\b", t)
        return ("scan_room", room.group(1) if room else "room")
    m = _ROOM_Q.search(t)
    if m:
        room = re.search(rf"\b({_ROOMS})\b", t)
        return ("room_query", room.group(1) if room else "room")
    if _DIFF.search(t):
        room = re.search(rf"\b({_ROOMS})\b", t)
        # "what changed" alone could be about code or a document -- only claim it with a room word or a clear physical cue
        if room or re.search(r"\b(?:around|in here|on my desk|since (?:last|before|yesterday)|anything (?:different|missing|moved))\b", t):
            return ("room_diff", room.group(1) if room else "room")
    m = _WHERE.search(t)
    if m:
        return ("where", (m.group("x") or m.group("y") or m.group("z")).strip())
    m = _FIND.search(t)
    if m:
        x = m.group("x").strip()
        if not _NOT_OBJECT.match(x) and len(x.split()) <= 3:
            return ("find", x)
    if _SEE.search(t):
        return ("see", False)
    if _DEEP.search(t) and len(t.split()) <= 12:
        return ("see", True)
    return None
