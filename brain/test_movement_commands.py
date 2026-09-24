import sys
from movement_commands import parse_movement_command as p

cases = [
    ("Do a dance for me", "dance"),
    ("wave hello", "wave"),
    ("show me the worm", "worm"),
    ("play dead", "dead"),
    ("walk forward", "forward"),
    ("go forward a bit", "forward"),
    ("walk backwards", "backward"),
    ("turn left", "left"),
    ("turn right", "right"),
    ("stand up", "stand"),
    ("sit down", "rest"),
    ("shake it", "shake"),
    ("do a pushup", "pushup"),
    ("take a bow", "bow"),
    ("crab walk", "crab"),
    ("shrug", "shrug"),
    ("swim for me", "swim"),
    ("point at the door", "point"),
    ("be cute", "cute"),
    ("get freaky", "freaky"),
    # must NOT trigger
    ("go back to idle", None),
    ("go back to sleep", None),
    ("shake my hand", None),
    ("what's the wavelength of red light", None),
    ("what's two plus two", None),
    ("turn on sarcastic mode", None),
]
bad = [(t, e, p(t)) for t, e in cases if p(t) != e]
for t, e, g in bad:
    print("FAIL", repr(t), "expected", e, "got", g)
if bad:
    sys.exit(1)
print("all movement-command parser tests passed")
