"""Unit tests for the deterministic voice-command parsers. Run: python3 test_voice_commands.py"""
import sys
from voice_commands import parse_volume, parse_mode, words_to_number

failures = []

def check(fn, text, expected):
    got = fn(text)
    if got != expected:
        failures.append(f"{fn.__name__}({text!r}) -> {got!r}, expected {expected!r}")

# ---- words_to_number
for text, n in [("50", 50), ("fifty", 50), ("seventy five", 75), ("seventy-five", 75),
                ("one hundred", 100), ("hundred", 100), ("ten", 10), ("zero", 0), ("twenty one", 21),
                ("banana", None), ("", None)]:
    got = words_to_number(text)
    if got != n:
        failures.append(f"words_to_number({text!r}) -> {got!r}, expected {n!r}")

# ---- volume: set
V = parse_volume
for text, exp in [
    ("set the volume to 50 percent", ("set", 50)),
    ("volume 70", ("set", 70)),
    ("volume to seventy five percent", ("set", 75)),
    ("Set the volume to a hundred percent", ("set", 100)),
    ("set it to a hundred percent", ("set", 100)),
    ("set volume at 30%", ("set", 30)),
    ("make the volume forty", ("set", 40)),
    ("Rocky set the volume to one hundred percent", ("set", 100)),
    ("volume level 20", ("set", 20)),
    ("50 percent volume", ("set", 50)),
    ("turn it to sixty percent", ("set", 60)),
    ("set it to 25 percent", ("set", 25)),
    ("maximum volume", ("set", 100)),
    ("max volume please", ("set", 100)),
    ("volume to max", ("set", 100)),
    ("full volume", ("set", 100)),
    ("minimum volume", ("set", 5)),
    ("volume to zero percent", ("set", 0)),
]:
    check(V, text, exp)

# ---- volume: relative, mute, query
for text, exp in [
    ("volume up", ("up", 15)),
    ("turn it up", ("up", 15)),
    ("louder", ("up", 15)),
    ("a bit louder", ("up", 7)),
    ("a little louder please", ("up", 7)),
    ("much louder", ("up", 30)),
    ("speak up", ("up", 15)),
    ("i can't hear you", ("up", 15)),
    ("volume down", ("down", 15)),
    ("turn it down", ("down", 15)),
    ("quieter", ("down", 15)),
    ("a bit quieter", ("down", 7)),
    ("way too loud", ("down", 30)),
    ("that's too loud", ("down", 15)),
    ("lower the volume", ("down", 15)),
    ("mute", ("mute",)),
    ("please mute yourself", ("mute",)),
    ("unmute", ("unmute",)),
    ("what's the volume", ("query",)),
    ("what is the volume", ("query",)),
    ("how loud are you", ("query",)),
]:
    check(V, text, exp)

# ---- volume: must NOT trigger on ordinary conversation
for text in [
    "what is the volume of a sphere", "how do magnets work", "tell me a joke",
    "what time is it", "turn on the light", "the book has three volumes",
    "he was loud at the party", "set a timer for ten minutes", "who is narendra modi",
    "my project is a robot", "what's up", "i turned it on yesterday",
]:
    check(V, text, None)

# ---- modes
M = parse_mode
for text, exp in [
    ("sarcastic mode", ("sarcastic", True)),
    ("sarcastic mode on", ("sarcastic", True)),
    ("turn on sarcastic mode", ("sarcastic", True)),
    ("be sarcastic", ("sarcastic", True)),
    ("enable sarcasm", ("sarcastic", True)),
    ("sarcastic mode off", ("sarcastic", False)),
    ("turn off sarcastic mode", ("sarcastic", False)),
    ("turn off sarcasm", ("sarcastic", False)),
    ("stop being sarcastic", ("sarcastic", False)),
    ("switch off the sarcasm", ("sarcastic", False)),
    ("roast mode", ("roast", True)),
    ("roast mode on", ("roast", True)),
    ("roast me", ("roast", True)),
    ("turn on roast mode", ("roast", True)),
    ("turn off roast mode", ("roast", False)),
    ("stop roasting me", ("roast", False)),
    ("roast mode off", ("roast", False)),
    ("humour mode", ("humour", True)),
    ("humor mode on", ("humour", True)),
    ("funny mode", ("humour", True)),
    ("turn off humour mode", ("humour", False)),
    ("annoying mode", ("annoying", True)),
    ("turn off annoying mode", ("annoying", False)),
    ("stop being annoying", ("annoying", False)),
    ("normal mode", ("normal", False)),
    ("back to normal", ("normal", False)),
    ("be serious", ("normal", False)),
]:
    check(M, text, exp)

for text in [
    "what is sarcasm", "the joke was funny", "tell me about roast chicken",
    "i like my humor dry", "why are you so annoying sometimes",
]:
    check(M, text, None)

if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print("  ", f)
    sys.exit(1)
print("all voice-command parser tests passed")
