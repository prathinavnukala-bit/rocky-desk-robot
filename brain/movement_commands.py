"""Deterministic parsing of voice commands that trigger a physical pose or
movement, so "do a dance" doesn't have to round-trip through the LLM.

parse_movement_command(text) -> None or a pose name from body.VALID_COMMANDS.
"""
import re

# Ordered so a more specific phrase is checked before a generic one that
# would otherwise swallow it (e.g. "wave" alone shouldn't fire before
# checking it's not part of "shake" or a sentence about something else).
_PATTERNS = (
    ("wave", r"\bwave\b(?!\s*(?:length|form))|\bsay hi with your (?:hand|paw)\b"),
    ("dance", r"\bdance\b|\bdo a dance\b|\bbust a move\b"),
    ("swim", r"\bswim\b"),
    ("point", r"\bpoint(?:\s+at)?\b"),
    ("pushup", r"\bpush[\s-]?up[s]?\b"),
    ("bow", r"\bbow\b(?!\s*tie)|\btake a bow\b"),
    ("cute", r"\bbe cute\b|\bdo (?:something|the) cute\b|\bcute pose\b"),
    ("freaky", r"\bfreaky\b|\bget freaky\b|\bweird(?:o)? (?:pose|dance)\b"),
    ("worm", r"\bdo the worm\b|\bworm\b"),
    ("shake", r"\bshake\b(?!\s+(?:my|your)?\s*hand)|\bwiggle\b"),
    ("shrug", r"\bshrug\b"),
    ("crab", r"\bcrab\b(?:\s*walk)?"),
    ("dead", r"\bplay dead\b|\bplay\s+possum\b"),
    ("forward", r"\bwalk forward\b|\bmove forward\b|\bgo forward\b|\bwalk ahead\b"),
    ("backward", r"\bwalk back(?:ward)?s?\b|\bmove back(?:ward)?s?\b|\bback up\b|\bgo back(?:ward)?s?\b(?!\s+to\s+(?:idle|sleep))"),
    ("left", r"\bturn left\b|\bgo left\b|\brotate left\b"),
    ("right", r"\bturn right\b|\bgo right\b|\brotate right\b"),
    # Bare "stand" (no "up") needs a negative lookbehind for "can't/cannot/
    # don't/won't stand" -- confirmed live: "Stand." alone used to fall
    # through to plain CHAT and got a made-up reply instead of the actual
    # movement, since only "stand up" was recognized.
    ("stand", r"\bstand up\b|\bget up\b|\bstand pose\b|\bstop laying down\b|\bstop lying down\b"
              r"|(?<!can't )(?<!cannot )(?<!dont )(?<!don't )(?<!wont )(?<!won't )\bstand\b(?! by| for| it)"),
    ("rest", r"\bsit down\b|\brest pose\b|\btake a rest\b|\brelax your legs\b"),
)

_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in _PATTERNS]


def parse_movement_command(text):
    t = text.lower().strip()
    for name, rx in _COMPILED:
        if rx.search(t):
            return name
    return None
