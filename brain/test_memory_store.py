"""Tests for memory_store.py. Run: python3 test_memory_store.py"""
import os, sys, tempfile, time
from memory_store import MemoryStore, extract_learnings, parse_memory_command, fts_query, when_phrase

fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

def kinds(text): return [x["kind"] for x in extract_learnings(text)]

# ---------- extraction: what SHOULD be learned
cases = [
    ("I'm building a walking robot called Rocky", ["project"]),
    ("we are working on the OLED face panel", ["project"]),
    ("I am currently designing a camera holder", ["project"]),
    ("start a new project called Hail Mary shell", ["project"]),
    ("I like dark chocolate", ["preference"]),
    ("I really love mechanical keyboards", ["preference"]),
    ("I prefer PETG over PLA", ["preference"]),
    ("my favorite color is orange", ["preference"]),
    ("I use a Prusa MK4 for printing", ["gear"]),
    ("I'm using an ESP32-S3 for the brain", ["gear"]),
    ("my sister is Priya", ["person"]),
    ("Rahul is my friend", ["person"]),
    ("we decided to use the ESP32-S3", ["decision"]),
    ("let's go with PETG for the shell", ["decision"]),
    ("my name is Prathinav", ["identity"]),
    ("call me Boss", ["identity"]),
]
for text, want in cases:
    got = kinds(text)
    check(all(w in got for w in want), f"extract({text!r}) -> {got}, wanted {want}")

# ---------- extraction: what must NOT be learned
for text in [
    "what is the capital of France", "do you like pizza?", "how does a servo work", "I like it",
    "I'm tired", "I am working on it", "I have a question", "I use it a lot", "hello rocky",
    "can you help me with my project?", "who is my friend?", "I love you", "I hate that",
    "tell me a joke", "my mom is great",       # relation without a name: nothing to store
]:
    got = kinds(text)
    check(got == [], f"extract({text!r}) should learn nothing, got {got}")

# ---------- privacy flags
for x in extract_learnings("my sister is Priya"):
    check(x["sensitive"] is True, "family facts must be flagged sensitive")

# ---------- commands
P = parse_memory_command
for text, want in [
    ("what are my tasks", "list_tasks"), ("show my to-do list", "list_tasks"), ("what's left on the robot", "list_tasks"),
    ("add a task to test the camera", "add_task"), ("remind me to order servos", "add_task"),
    ("mark the camera test as done", "done_task"), ("I finished the wiring", "done_task"), ("done with the OLED panel", "done_task"),
    ("what's the status of the robot", "status"), ("how is the shell going", "status"),
    ("list my projects", "list_projects"), ("what am I working on", "list_projects"),
    ("what did I say about servos", "search"), ("what do you remember about the camera", "search"),
    ("do you remember what board we picked", "search"),
    ("forget that I like chocolate", "forget"),
    ("brief me", "briefing"), ("catch me up", "briefing"),
    ("what do you know about me", "about_me"),
]:
    got = P(text)
    check(got is not None and got[0] == want, f"parse({text!r}) -> {got}, wanted {want}")
for text in ["what is the capital of France", "how does wifi work", "forget it", "forget that", "tell me a story",
             "how is the weather", "set the volume to 40", "I like pizza", "good morning"]:
    got = P(text)
    check(got is None, f"parse({text!r}) should be None, got {got}")

# ---------- fts_query
check(fts_query("the and of") is None, "stopwords only -> no query")
check(fts_query("servos") == '"servo"*', f"stemming: {fts_query('servos')}")

# ---------- store behaviour
tmp = tempfile.mkdtemp(); st = MemoryStore(os.path.join(tmp, "t.db"))
for u in ["I'm building a walking robot called Rocky", "we decided to use the ESP32-S3 for the brain",
          "I like dark chocolate", "my sister is Priya", "I use a Prusa MK4 for printing",
          "I prefer PETG over PLA for the shell", "my name is Prathinav"]:
    st.learn(u)
check(st.core_profile().startswith("The person you talk to is named Prathinav"), f"profile: {st.core_profile()}")
check("walking robot" in st.core_profile().lower(), f"profile should include current project: {st.core_profile()}")

ctx = st.context_for("Which board did we pick for the brain?")
check("ESP32-S3" in ctx, f"relevant decision should be retrieved: {ctx!r}")
check(st.context_for("what is the capital of France") == "", f"unrelated question must retrieve nothing: {st.context_for('what is the capital of France')!r}")
check(st.context_for("tell me about pizza") == "", "unrelated: pizza")
check("Priya" in st.context_for("how is my sister Priya doing"), "person retrieval")
check("Priya" not in st.context_for("how is my sister Priya doing", include_sensitive=False), "sensitive facts excluded when asked")
check("PETG" in st.context_for("what filament should I print the shell in"), f"pref retrieval: {st.context_for('what filament should I print the shell in')!r}")

# duplicates are not stored twice
before = st.count(); st.learn("I like dark chocolate"); check(st.count() == before, "duplicate fact stored twice")

# tasks
pid = st.current_project()["id"]
st.add_task("test the camera watchdog", pid); st.add_task("order more servos", pid); st.add_task("print the face panel", pid)
check(len(st.open_tasks(pid)) == 3, "3 open tasks")
done = st.complete_task("the camera test")
check(done and "camera" in done, f"fuzzy complete: {done}")
check(len(st.open_tasks(pid)) == 2, "2 open after completing one")
check(st.complete_task("launch the rocket") is None, "must not complete an unrelated task")
status = st.project_status(st.current_project())
check(status["done"] == 1 and len(status["open"]) == 2, f"status counts: {status['done']} {len(status['open'])}")

# forget: deactivates a fact, never touches verbatim turns
st.log_turn("user", "I like dark chocolate very much")
removed = st.forget("that I like dark chocolate")
check(removed and "chocolate" in removed, f"forget: {removed}")
check("chocolate" not in st.context_for("do I like chocolate"), "forgotten fact must not be retrieved")
check(st.forget("something I never said at all") is None, "forget of unknown thing -> None")

# verbatim recall
st.log_turn("user", "the servo on leg L2 keeps jittering when the battery is low")
st.log_turn("rocky", "Maybe power problem, yes?")
lines = st.recall_lines("servo jittering")
check(lines and "L2" in lines[0][1], f"verbatim recall: {lines}")
check(all("Maybe power" not in t for _, t in lines), "recall returns what the USER said, not Rocky's replies")

# project dedup: restating the same project in different words must land
# on the same row, not fork -- this was a real bug (see memory_store.py's
# upsert_project docstring): "4-legged robot" then "robot with servers"
# (a mis-transcription of "servos") created two separate project rows.
pid_a = st.upsert_project("4-legged robot")
pid_b = st.upsert_project("robot with servers")
check(pid_a == pid_b, f"restating a project in different words must merge, got {pid_a} vs {pid_b}")
pid_c = st.upsert_project("baking sourdough bread")
check(pid_c != pid_a, f"an unrelated project must NOT merge into the robot project, got {pid_c}")

# when phrases
check(when_phrase(time.time()) == "today", "when today")
check(when_phrase(time.time() - 86400) == "yesterday", "when yesterday")

if fails:
    print(f"{len(fails)} FAILURE(S):")
    for f in fails: print("  -", f)
    sys.exit(1)
print("all memory store tests passed")
