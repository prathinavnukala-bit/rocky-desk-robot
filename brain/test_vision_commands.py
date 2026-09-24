import sys
from vision_commands import parse_vision_command as p

cases = [
 ("Remember this room", ("scan_room", "room")),
 ("remember my workshop", ("scan_room", "workshop")),
 ("Scan the room please", ("scan_room", "room")),
 ("learn my bedroom", ("scan_room", "bedroom")),
 ("remember what my room looks like", ("scan_room", "room")),
 ("What's in my room", ("room_query", "room")),
 ("what do you remember about my room", ("room_query", "room")),
 ("Describe my workshop", ("room_query", "workshop")),
 ("what does my room look like", ("room_query", "room")),
 ("what changed in my room", ("room_diff", "room")),
 ("is anything different", ("room_diff", "room")),
 ("Do you see my phone", ("find", "phone")),
 ("can you see a cup", ("find", "cup")),
 ("is there a laptop", ("find", "laptop")),
 ("Where did I leave my phone", ("where", "phone")),
 ("where is my keyboard", ("where", "keyboard")),
 ("when did you last see my keys", ("where", "keys")),
 ("What do you see", ("see", False)),
 ("look around", ("see", False)),
 ("What am I holding", ("see", True)),
 ("what is this", ("see", True)),
 ("what's in my hand", ("see", True)),
 ("look at this", ("see", True)),
 # must NOT be treated as camera commands
 ("Remember that my room is on the second floor", None),
 ("remember my soldering iron is in the top drawer", None),
 ("what changed in my code", None),
 ("do you see what I mean", None),
 ("is there a way to fix this", None),
 ("what is that supposed to mean", None),
 ("that looks good", None),
 ("what's the temperature in my room", None),
 ("Set the volume to fifty percent", None),
 ("what is a transistor", None),
 ("how is the weather", None),
 ("what is the capital of France", None),
]
bad = [(t, e, p(t)) for t, e in cases if p(t) != e]
for t, e, g in bad:
    print("FAIL", repr(t), "expected", e, "got", g)
if bad:
    sys.exit(1)
print("all vision-command parser tests passed")
