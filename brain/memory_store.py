"""
Rocky's long-term memory -- built to behave like a project assistant's, not a
chatbot's short-term buffer.

Design rules, each learned the hard way earlier in this project:
  * NOTHING here uses the small local LLM to decide what is true. The
    llama3.2 3B model, asked to summarise conversations into "knowledge",
    copied transcripts, ran away for 18,000 tokens, then confidently wrote
    "Boss is Donald Trump". Facts enter memory only through explicit
    commands or deterministic phrase rules (regexes), each stored with its
    provenance, so every stored fact is something the user actually said.
  * Everything said is also kept VERBATIM (turns table), full-text
    searchable. Recall is a database lookup, not a guess.
  * Only what is RELEVANT to the current question is handed to the model
    (BM25 full-text ranking), instead of dumping every note into every
    prompt -- which made Rocky bring up unrelated old notes and invent
    memories.

Storage: one SQLite file (WAL mode) with an FTS5 full-text index. Stdlib only.
"""
import datetime
import json
import os
import re
import sqlite3
import threading
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "rocky_memory.db")

# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
_STOP = set("""a an the and or but if then so of to in on at for from with by about as is are was were be been
being am do does did done have has had having i me my mine we our us you your yours he him his she her it its they
them their this that these those there here what which who whom whose when where why how can could should would will
shall may might must not no yes yeah okay ok please just really very too also again still yet more most some any all
each every much many few lot lots thing things stuff going gonna get got make made take took tell told say said talk
know think want need let lets like rocky hey hi hello thanks thank""".split())

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-']{1,}")


def _is_near_duplicate(a, b):
    """True when `a` is essentially a restatement of `b` -- used so a "did I
    tell you about X" query never recalls itself. Reproduced live: the turn
    log records the user's question a moment before recall_lines() searches
    it, so a plain word-overlap search matched the just-asked question
    against ITSELF and Rocky echoed it back as if it were something "they
    told me before" ("today you told me: Do you remember the project I was
    working on?"). Checked by meaning (shared content words), not exact
    text, since a real paraphrase like "the robot project" vs "about the
    robot project I was working on" is still obviously the same sentence."""
    wa, wb = set(content_words(a)), set(content_words(b))
    if not wa or not wb:
        return False
    return len(wa & wb) / max(len(wa), len(wb)) >= 0.6


def content_words(text):
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) >= 3]


def fts_query(text, max_terms=8):
    """OR-query over the meaningful words, with prefix matching so 'servos'
    finds 'servo'. Returns None when there is nothing meaningful to search."""
    words = []
    for w in content_words(text):
        w = re.sub(r"[^a-z0-9\-]", "", w)
        stem = w[:-1] if w.endswith("s") and len(w) > 4 else w
        if stem and stem not in words:
            words.append(stem)
    if not words:
        return None
    return " OR ".join(f'"{w}"*' for w in words[:max_terms])


def _clean_phrase(s, max_words=8):
    s = s.strip().strip(".,;:!?\"' ")
    s = re.split(r"\b(?:because|so that|which|but|and then|right now|at the moment|these days|today|tonight|"
                 r"with you|for you|again)\b", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"\s+", " ", s).strip(".,;:!?\"' ")
    return " ".join(s.split()[:max_words])


def when_phrase(ts, now=None):
    now = now or time.time()
    d = datetime.datetime.fromtimestamp(ts).date()
    today = datetime.datetime.fromtimestamp(now).date()
    delta = (today - d).days
    if delta <= 0:
        return "today"
    if delta == 1:
        return "yesterday"
    if delta < 7:
        return d.strftime("on %A")
    return d.strftime("on %B %d").replace(" 0", " ")


# --------------------------------------------------------------------------
# Deterministic extraction rules (what Rocky learns on its own)
# --------------------------------------------------------------------------
_NAME = r"(?P<name>[^.,;!?]{3,60})"
_NAME2 = r"(?P<name2>[^.,;!?]{3,60})"

_R_PROJECT_WORK = re.compile(
    r"\b(?:i(?:'m| am)|we(?:'re| are)|i(?:'ve| have) been|let(?:'s| us)|i(?:'m| am) going to|i want to|i(?:'ll| will))\s+"
    r"(?:currently\s+|now\s+|also\s+|still\s+)?(?:working on|building|making|designing|developing|creating|"
    r"coding|programming|assembling|prototyping)\s+(?:a |an |the |my |our |some )?" + _NAME, re.I)
_R_PROJECT_START = re.compile(
    r"\b(?:start|create|begin|open|set up)\s+(?:a\s+|the\s+)?(?:new\s+)?project\s+(?:called|named|for)?\s*" + _NAME
    + r"|\bnew project\s+(?:called|named|:)?\s*" + _NAME2, re.I)
_R_PROJECT_SWITCH = re.compile(
    r"\b(?:switch|change|go|move)\s+(?:over\s+)?to\s+(?:the\s+)?project\s+" + _NAME
    + r"|\bwork(?:ing)? on project\s+" + _NAME2, re.I)
_R_PREF = re.compile(
    r"^(?:i|i really|i actually|i just)\s+(?P<verb>like|love|prefer|hate|enjoy|dislike|can't stand|cannot stand|adore)\s+"
    r"(?P<obj>[^.,;!?]{2,60})", re.I)
_R_FAVORITE = re.compile(r"\bmy favou?rite (?P<what>[a-z ]{2,20}?) (?:is|are)\s+(?P<obj>[^.,;!?]{2,50})", re.I)
_R_GEAR = re.compile(r"^(?:i|we)(?:'m| am)?\s*(?:use|own|am using|'m using|using|got|bought)\s+(?:a |an |the |some |my )?(?P<obj>[^.,;!?]{3,60})", re.I)
_R_RELATION = re.compile(
    r"\b(?i:my)\s+(?P<rel>(?i:mom|mother|mum|dad|father|sister|brother|wife|husband|girlfriend|boyfriend|friend|best friend|son|"
    r"daughter|uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|roommate|teacher|colleague))(?i:'s name)?\s+"
    r"(?i:is|are|called|named)\s+(?P<name>[A-Z][A-Za-z]{1,19}(?:\s[A-Z][A-Za-z]{1,19})?)")
_R_RELATION2 = re.compile(
    r"\b(?P<name>[A-Z][a-z]{1,19})\s+is\s+my\s+(?P<rel>mom|mother|mum|dad|father|sister|brother|wife|husband|"
    r"girlfriend|boyfriend|friend|best friend|son|daughter|uncle|aunt|cousin|roommate|teacher|colleague)\b")
_R_DECISION = re.compile(
    r"\b(?:we|i)(?:'ve| have)?\s+(?:decided|agreed|chose|choose)\s+(?:to|on)\s+(?P<x>[^.;!?]{3,100})"
    r"|\b(?:let(?:'s| us))\s+(?:go with|use|switch to)\s+(?P<y>[^.;!?]{3,80})"
    r"|\bi(?:'m| am)\s+going with\s+(?P<z>[^.;!?]{3,80})", re.I)
_R_NAME = re.compile(r"\b(?:my name is|call me|i am called|people call me)\s+(?P<name>[A-Za-z]{2,20})", re.I)

_GEAR_REJECT = re.compile(r"^(?:to\b|it\b|that\b|this\b|them\b|you\b|him\b|her\b|the same\b|a lot\b|some time\b|"
                          r"an idea\b|a question\b|a problem\b|a bug\b|a feeling\b)", re.I)
_PREF_REJECT = re.compile(r"^(?:to\b|it\b|that\b|this\b|you\b|when\b|how\b|the way\b)", re.I)

FAMILY_RELATIONS = {"mom", "mother", "mum", "dad", "father", "sister", "brother", "wife", "husband", "girlfriend",
                    "boyfriend", "son", "daughter", "uncle", "aunt", "cousin", "grandma", "grandmother", "grandpa",
                    "grandfather"}


def extract_learnings(text):
    """Returns a list of dicts {kind, text, subject, sensitive, project_name?, set_current?} that can be
    inferred from ONE user utterance by the deterministic rules. Never invents anything -- every stored text
    is built from the user's own words."""
    out = []
    t = text.strip()
    if not t or t.endswith("?") or re.match(r"^(?:do|does|did|can|could|will|would|are|is|what|who|why|how|when|where)\b", t, re.I):
        return out

    m = _R_NAME.search(t)
    if m:
        out.append({"kind": "identity", "text": f"The user's name is {m.group('name').title()}.", "subject": "user",
                    "sensitive": False, "profile_name": m.group("name").title()})

    for rx in (_R_PROJECT_WORK, _R_PROJECT_START, _R_PROJECT_SWITCH):
        m = rx.search(t)
        if m:
            raw = next((g for g in m.groups() if g), "")
            name = _clean_phrase(raw, 6)
            if len(name) >= 3 and not re.match(r"^(?:it|that|this|something|stuff|things?|on it)\b", name, re.I):
                out.append({"kind": "project", "text": f"The user is working on: {name}.", "subject": "user",
                            "sensitive": False, "project_name": name, "set_current": True})
                break

    m = _R_PREF.search(t)
    if m and not _PREF_REJECT.match(m.group("obj")):
        verb, obj = m.group("verb").lower(), _clean_phrase(m.group("obj"), 8)
        if obj:
            out.append({"kind": "preference", "text": f"The user {verb.replace('cannot', 'can not')}s {obj}."
                        if verb in ("like", "love", "hate", "enjoy", "dislike", "adore", "prefer")
                        else f"The user {verb} {obj}.", "subject": "user", "sensitive": False})
    m = _R_FAVORITE.search(t)
    if m:
        out.append({"kind": "preference", "text": f"The user's favorite {m.group('what').strip()} is {_clean_phrase(m.group('obj'), 6)}.",
                    "subject": "user", "sensitive": False})

    m = _R_GEAR.search(t)
    if m and not _GEAR_REJECT.match(m.group("obj")):
        obj = _clean_phrase(m.group("obj"), 8)
        if obj and len(obj) >= 3:
            out.append({"kind": "gear", "text": f"The user uses {obj}.", "subject": "user", "sensitive": False})

    for rx, order in ((_R_RELATION, ("rel", "name")), (_R_RELATION2, ("rel", "name"))):
        m = rx.search(t)
        if m:
            rel, name = m.group("rel").lower(), m.group("name").strip().title()
            out.append({"kind": "person", "text": f"{name} is the user's {rel}.", "subject": name,
                        "sensitive": True, "relation": rel})
            break

    m = _R_DECISION.search(t)
    if m:
        x = _clean_phrase(next((g for g in m.groups() if g), ""), 14)
        if len(x) >= 3:
            out.append({"kind": "decision", "text": f"Decided: {x}.", "subject": "user", "sensitive": False,
                        "attach_project": True})
    return out


# --------------------------------------------------------------------------
# Voice commands about memory / projects / tasks (deterministic routing)
# --------------------------------------------------------------------------
_C = [
    ("list_tasks", re.compile(r"\b(?:what(?:'s| is| are)?|show|list|read)\s+(?:me\s+)?(?:my\s+|the\s+)?(?:open\s+|remaining\s+)?(?:tasks|to-?dos?|todo list|to do list)\b"
                              r"|\bwhat(?:'s| is) left(?: to do)?(?: on| for)?\s*(?P<proj>[^.?!]{0,40})"
                              r"|\bwhat do i (?:still )?(?:need|have) to do\b"
                              # "what's the plan today" etc -- a forward-looking question about
                              # the user's own day, which Rocky has no way to answer except from
                              # stored tasks (confirmed live: routed to free-form CHAT instead, the
                              # small model hallucinated unrelated engineering content rather than
                              # admitting it doesn't know -- same "ground it in real data" fix as
                              # SELF_DIAGNOSE below).
                              r"|\bwhat(?:'s| is) (?:the |my )?plan(?: for)? today\b"
                              r"|\bwhat do i have(?: on| for)? today\b"
                              r"|\bwhat(?:'s| is) (?:on|happening) today\b"
                              r"|\bwhat am i doing today\b", re.I)),
    ("add_task", re.compile(r"\b(?:add|create|make)\s+(?:a\s+)?(?:new\s+)?(?:task|to-?do|todo)\s*(?:to|for|called|that|:)?\s*(?P<x>[^.;!?]{3,120})"
                            r"|\bremind me to\s+(?P<y>[^.;!?]{3,120})", re.I)),
    ("done_task", re.compile(r"\bmark\s+(?P<x>[^.;!?]{3,80}?)\s+(?:as\s+)?(?:done|complete[d]?|finished)\b"
                             r"|\bi(?:'ve| have)?\s+(?:just\s+)?(?:finished|completed)\s+(?P<y>[^.;!?]{3,80})"
                             r"|\b(?:done|finished) with\s+(?P<z>[^.;!?]{3,80})", re.I)),
    ("status", re.compile(r"\b(?:what(?:'s| is) the status of|status of|update on)\s+(?:the\s+|my\s+)?(?:project\s+)?(?P<name>[^.?!]{3,50}?)\s*\??$"
                         r"|\bhow(?:'s| is)\s+(?:the\s+|my\s+)?(?:project\s+)?(?P<name2>[^.?!]{3,50}?)\s+(?:going|coming along|progressing)\b", re.I)),
    ("list_projects", re.compile(r"\b(?:list|show|what are)\s+(?:me\s+)?(?:my\s+|all my\s+)?projects\b|\bwhat am i (?:currently )?working on\b|\bwhat (?:project )?(?:am i|are we) (?:currently )?on\b", re.I)),
    ("search", re.compile(r"\bwhat did (?:i|we) (?:say|tell you|talk|discuss|decide)(?: about)?\s+(?P<q>[^.?!]{3,80})"
                          r"|\bwhat do you (?:remember|know) about\s+(?P<q2>(?!me\b)[^.?!]{3,80})"
                          r"|\bsearch (?:your |the )?memory for\s+(?P<q3>[^.?!]{3,80})"
                          r"|\bdo you remember\s+(?P<q4>[^.?!]{3,80})", re.I)),
    ("forget", re.compile(r"\bforget(?: about)?(?: that)?\s+(?P<x>(?!it\b|that\b|this\b|everything\b|about it\b)[^.;!?]{3,100})", re.I)),
    ("briefing", re.compile(r"\b(?:brief me|give me a briefing|catch me up|daily summary|daily briefing|what did (?:i|we) do (?:yesterday|today|last time))\b", re.I)),
    ("about_me", re.compile(r"\bwhat do you (?:know|remember) about me\b|\bwhat have you learn(?:ed|t)(?: about me)?\b", re.I)),
]


def parse_memory_command(text):
    """Returns None or (intent, argument-string-or-None). Order matters: the
    more specific phrasings come first."""
    t = text.strip()
    for intent, rx in _C:
        m = rx.search(t)
        if not m:
            continue
        gd = {k: v for k, v in m.groupdict().items() if v}
        arg = next(iter(gd.values()), None)
        if intent == "list_tasks":
            arg = (gd.get("proj") or "").strip() or None
        if arg:
            arg = arg.strip().strip(".,;:!?\"' ")
        return intent, arg
    return None


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------
PROJECT_MERGE_THRESHOLD = 0.5


class MemoryStore:
    def __init__(self, path=DB_PATH):
        self.path = path
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()
        self.session = time.strftime("%Y%m%d-%H%M%S")

    def _init_schema(self):
        with self._lock, self._db:
            self._db.executescript("""
            CREATE TABLE IF NOT EXISTS docs(
                id INTEGER PRIMARY KEY, ts REAL NOT NULL, kind TEXT NOT NULL, subject TEXT DEFAULT 'user',
                text TEXT NOT NULL, project_id INTEGER, source TEXT DEFAULT 'rule',
                active INTEGER DEFAULT 1, sensitive INTEGER DEFAULT 0, done INTEGER DEFAULT 0, done_ts REAL,
                session TEXT, mode TEXT);
            CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(text, content='docs', content_rowid='id');
            CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON docs BEGIN
                INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text); END;
            CREATE TRIGGER IF NOT EXISTS docs_ad AFTER DELETE ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.id, old.text); END;
            CREATE TRIGGER IF NOT EXISTS docs_au AFTER UPDATE OF text ON docs BEGIN
                INSERT INTO docs_fts(docs_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO docs_fts(rowid, text) VALUES (new.id, new.text); END;
            CREATE TABLE IF NOT EXISTS projects(
                id INTEGER PRIMARY KEY, name TEXT UNIQUE COLLATE NOCASE NOT NULL,
                status TEXT DEFAULT 'active', created REAL, updated REAL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE INDEX IF NOT EXISTS docs_kind ON docs(kind, active);
            CREATE TABLE IF NOT EXISTS sightings(
                obj TEXT PRIMARY KEY COLLATE NOCASE, ts REAL NOT NULL, count INTEGER DEFAULT 1, first_ts REAL NOT NULL);
            """)

    # ---- generic ----------------------------------------------------------
    def _meta_get(self, key, default=None):
        r = self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def _meta_set(self, key, value):
        with self._lock, self._db:
            self._db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))

    def count(self, kind=None):
        q, a = "SELECT COUNT(*) c FROM docs WHERE active=1", []
        if kind:
            q += " AND kind=?"; a.append(kind)
        return self._db.execute(q, a).fetchone()["c"]

    # ---- verbatim log -----------------------------------------------------
    def log_turn(self, role, text, mode=None):
        text = (text or "").strip()
        if not text:
            return None
        with self._lock, self._db:
            cur = self._db.execute("INSERT INTO docs(ts,kind,subject,text,source,session,mode) VALUES(?,?,?,?,?,?,?)",
                                   (time.time(), "turn", role, text, "log", self.session, mode))
            return cur.lastrowid

    # ---- facts ------------------------------------------------------------
    def add_fact(self, kind, text, subject="user", source="rule", project_id=None, sensitive=False, ts=None):
        norm = re.sub(r"\W+", " ", text.lower()).strip()
        with self._lock, self._db:
            for r in self._db.execute("SELECT id, text FROM docs WHERE kind=? AND active=1 AND subject=?", (kind, subject)):
                if re.sub(r"\W+", " ", r["text"].lower()).strip() == norm:
                    return r["id"]     # already known: don't store duplicates
            cur = self._db.execute(
                "INSERT INTO docs(ts,kind,subject,text,project_id,source,sensitive,session) VALUES(?,?,?,?,?,?,?,?)",
                (ts or time.time(), kind, subject, text, project_id, source, 1 if sensitive else 0, self.session))
            return cur.lastrowid

    def forget(self, query):
        """Deactivates the best matching stored fact (never deletes verbatim turns). Returns its text or None."""
        hits = self.search(query, kinds=("note", "preference", "gear", "person", "decision", "project", "task", "room", "identity"),
                           limit=1, min_overlap=0.5)
        if not hits:
            return None
        with self._lock, self._db:
            self._db.execute("UPDATE docs SET active=0 WHERE id=?", (hits[0]["id"],))
        return hits[0]["text"]

    # ---- projects ---------------------------------------------------------
    def upsert_project(self, name):
        """Finds the project this name most likely refers to, or creates a new
        one. Restating an existing project in different words -- "the 4-legged
        robot" today, "robot with servos" tomorrow -- must land on the SAME
        row, or a project silently forks into duplicates and Rocky starts
        contradicting itself about what "the project" even is (confirmed
        live: this happened for real, see git history / session notes).
        Plain substring containment used to be the only fuzzy check, which
        missed exactly this case (neither name is a substring of the other),
        so it now shares find_project()'s word-overlap scoring, just with a
        write-side threshold: this decides whether two mentions get MERGED
        (costly to get wrong -- a false merge blends two real projects),
        while find_project()'s lookup-side threshold decides whether a
        one-off query MATCHES (costly to get wrong the other way -- a missed
        match just falls through to "no project found")."""
        name = name.strip()
        with self._lock, self._db:
            r = self._db.execute("SELECT id FROM projects WHERE name=?", (name,)).fetchone()
            if r:
                self._db.execute("UPDATE projects SET updated=? WHERE id=?", (time.time(), r["id"]))
                return r["id"]
            words = set(content_words(name))
            best, best_score = None, 0.0
            for p in self._db.execute("SELECT id, name FROM projects WHERE status='active'"):
                pw = set(content_words(p["name"]))
                if not pw or not words:
                    continue
                # Overlap coefficient (intersection / smaller set), not Jaccard:
                # "robot" vs "robot with servos" should merge even though the
                # second name adds words the first doesn't have.
                score = len(words & pw) / min(len(words), len(pw))
                if score > best_score:
                    best, best_score = p, score
            if best is not None and best_score >= PROJECT_MERGE_THRESHOLD:
                self._db.execute("UPDATE projects SET updated=? WHERE id=?", (time.time(), best["id"]))
                return best["id"]
            cur = self._db.execute("INSERT INTO projects(name,created,updated) VALUES(?,?,?)", (name, time.time(), time.time()))
            return cur.lastrowid

    def set_current_project(self, name_or_id):
        pid = name_or_id if isinstance(name_or_id, int) else self.upsert_project(name_or_id)
        self._meta_set("current_project", str(pid))
        return pid

    def current_project(self):
        v = self._meta_get("current_project")
        if not v:
            return None
        r = self._db.execute("SELECT * FROM projects WHERE id=?", (int(v),)).fetchone()
        return dict(r) if r else None

    def find_project(self, name):
        name = (name or "").strip()
        if not name:
            return None
        r = self._db.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
        if r:
            return dict(r)
        words = set(content_words(name))
        best, best_score = None, 0
        for p in self._db.execute("SELECT * FROM projects"):
            pw = set(content_words(p["name"]))
            if not pw or not words:
                continue
            score = len(words & pw) / max(1, len(words | pw))
            if score > best_score:
                best, best_score = dict(p), score
        return best if best_score >= 0.34 else None

    def list_projects(self):
        return [dict(r) for r in self._db.execute("SELECT * FROM projects WHERE status='active' ORDER BY updated DESC")]

    # ---- tasks & project items -------------------------------------------
    def add_task(self, text, project_id=None):
        return self.add_fact("task", text.strip().rstrip("."), subject="user", source="explicit", project_id=project_id)

    def open_tasks(self, project_id=None):
        q = "SELECT * FROM docs WHERE kind='task' AND active=1 AND done=0"
        a = []
        if project_id:
            q += " AND project_id=?"; a.append(project_id)
        return [dict(r) for r in self._db.execute(q + " ORDER BY ts", a)]

    def complete_task(self, query, project_id=None):
        hits = self.search(query, kinds=("task",), limit=3, min_overlap=0.5, only_open=True)
        if project_id:
            hits = [h for h in hits if h["project_id"] == project_id] or hits
        if not hits:
            return None
        with self._lock, self._db:
            self._db.execute("UPDATE docs SET done=1, done_ts=? WHERE id=?", (time.time(), hits[0]["id"]))
        return hits[0]["text"]

    def project_status(self, project):
        pid = project["id"]
        tasks_open = self.open_tasks(pid)
        done = self._db.execute("SELECT COUNT(*) c FROM docs WHERE kind='task' AND done=1 AND project_id=?", (pid,)).fetchone()["c"]
        recent = [dict(r) for r in self._db.execute(
            "SELECT * FROM docs WHERE project_id=? AND kind IN ('decision','note','gear') AND active=1 ORDER BY ts DESC LIMIT 4", (pid,))]
        return {"project": project, "open": tasks_open, "done": done, "recent": recent}

    # ---- retrieval --------------------------------------------------------
    def search(self, query, kinds=None, limit=5, min_overlap=0.0, only_open=False, include_sensitive=True, recency=True):
        """BM25 full-text search over stored docs. `min_overlap` (0-1) requires that share of the query's
        meaningful words to actually appear in a hit -- used where a wrong match would be harmful (forget,
        mark-done)."""
        q = fts_query(query)
        if not q:
            return []
        sql = ("SELECT d.*, bm25(docs_fts) AS score FROM docs_fts JOIN docs d ON d.id = docs_fts.rowid "
               "WHERE docs_fts MATCH ? AND d.active=1")
        args = [q]
        if kinds:
            sql += " AND d.kind IN (%s)" % ",".join("?" * len(kinds)); args += list(kinds)
        if only_open:
            sql += " AND d.done=0"
        if not include_sensitive:
            sql += " AND d.sensitive=0"
        sql += " ORDER BY score LIMIT 40"
        try:
            rows = [dict(r) for r in self._db.execute(sql, args)]
        except sqlite3.OperationalError:
            return []
        qwords = set(w[:-1] if w.endswith("s") and len(w) > 4 else w for w in content_words(query))
        now = time.time()
        ranked = []
        for r in rows:
            twords = set(w[:-1] if w.endswith("s") and len(w) > 4 else w for w in content_words(r["text"]))
            overlap = len(qwords & twords) / max(1, len(qwords))
            if overlap < min_overlap:
                continue
            score = -r["score"] + 2.0 * overlap
            if recency:
                age_days = max(0.0, (now - r["ts"]) / 86400.0)
                score += 0.6 / (1.0 + age_days / 14.0)
            if r["kind"] in ("decision", "note", "task", "project"):
                score += 0.4
            ranked.append((score, r))
        ranked.sort(key=lambda x: -x[0])
        return [r for _, r in ranked[:limit]]

    def context_for(self, query, include_sensitive=True, max_items=4, max_chars=420):
        """Relevant memory for a question, as short lines for the LLM's background. Structured knowledge only
        (never raw transcript): a stored preference/decision/task/note/room fact that shares meaningful
        words with the question. Empty string when nothing is relevant."""
        hits = self.search(query, kinds=("note", "preference", "gear", "person", "decision", "project", "task", "room", "identity"),
                           limit=max_items, min_overlap=0.25, include_sensitive=include_sensitive)
        lines, used = [], 0
        for h in hits:
            line = h["text"]
            if h["kind"] == "task" and h.get("done"):
                line = "(finished) " + line
            if used + len(line) > max_chars:
                break
            lines.append(line); used += len(line)
        return " ".join(lines)

    def core_profile(self):
        """The small, rarely-changing always-on part: who the user is and what they are working on now."""
        parts = []
        name = self._meta_get("user_name")
        if name:
            parts.append(f"The person you talk to is named {name}.")
        cur = self.current_project()
        if cur:
            parts.append(f"They are currently working on: {cur['name']}.")
        return " ".join(parts)

    # ---- learning ---------------------------------------------------------
    def learn(self, text):
        """Runs the deterministic extractors on one user utterance and stores the results. Returns a list of
        human-readable strings describing what was learned (for logging), never speaks."""
        learned = []
        cur = self.current_project()
        for item in extract_learnings(text):
            pid = None
            if item["kind"] == "identity":
                self._meta_set("user_name", item["profile_name"])
            elif item["kind"] == "project":
                pid = self.upsert_project(item["project_name"])
                if item.get("set_current"):
                    self.set_current_project(pid)
                    cur = self.current_project()
            elif item.get("attach_project") and cur:
                pid = cur["id"]
            self.add_fact(item["kind"], item["text"], subject=item.get("subject", "user"), source="rule",
                          project_id=pid, sensitive=item.get("sensitive", False))
            learned.append(item["text"])
        return learned

    # ---- recall for spoken answers ---------------------------------------
    def recall_lines(self, query, limit=3):
        """Verbatim recall for 'what did I say about X': what the user said, with when."""
        q = fts_query(query)
        if not q:
            return []
        rows = self.search(query, kinds=("turn",), limit=limit * 3, min_overlap=0.34, recency=False)
        rows = [r for r in rows if r["subject"] == "user" and not _is_near_duplicate(r["text"], query)][:limit]
        return [(r["ts"], r["text"]) for r in rows]

    def facts_about_user(self, kinds=("preference", "gear", "project", "person", "identity", "note", "decision")):
        out = []
        for kind in kinds:
            for r in self._db.execute("SELECT text FROM docs WHERE kind=? AND active=1 ORDER BY ts DESC LIMIT 6", (kind,)):
                out.append((kind, r["text"]))
        return out

    def activity_since(self, since_ts):
        rows = [dict(r) for r in self._db.execute(
            "SELECT * FROM docs WHERE ts>=? AND active=1 AND kind IN ('task','decision','note','project','gear','room') ORDER BY ts", (since_ts,))]
        turns = self._db.execute("SELECT COUNT(*) c FROM docs WHERE ts>=? AND kind='turn' AND subject='user'", (since_ts,)).fetchone()["c"]
        return rows, turns

    # ---- legacy import ----------------------------------------------------
    def import_legacy(self, memory, fallback_ts=None):
        """One-time import of memory.json's explicit notes / people / profile into the store."""
        if self._meta_get("legacy_imported"):
            return 0
        n = 0
        for note in memory.get("notes", []):
            self.add_fact("note", note, source="explicit"); n += 1
        prof = memory.get("profile") or {}
        if prof.get("name"):
            self._meta_set("user_name", prof["name"])
        for name, info in (memory.get("people") or {}).items():
            self.add_fact("person", f"{name} is someone Rocky knows and can recognize.", subject=name, source="face", sensitive=False); n += 1
        n += self.import_talked(memory.get("facts", []), fallback_ts)
        self._meta_set("legacy_imported", "1")
        return n

    def import_talked(self, lines, base_ts=None):
        """Imports old rolling-log lines ('Talked: you said X -- I said Y') as searchable verbatim history.
        Junk transcripts (single hallucinated words like 'You') are skipped; timestamps are unknown, so
        they are spaced backwards from base_ts in their original order."""
        talked = re.compile(r'^Talked: you said "(?P<u>.*?)" -- I said "(?P<r>.*)"$')
        base = base_ts or time.time()
        n = 0
        for i, line in enumerate(lines):
            m = talked.match(line)
            if not m:
                continue
            user = m.group("u").strip()
            if len(re.sub(r"[^A-Za-z]", "", user)) < 4 or user.lower() in ("you", "thank you", "thanks for watching"):
                continue
            with self._lock, self._db:
                dup = self._db.execute("SELECT 1 FROM docs WHERE kind='turn' AND subject='user' AND text=?", (user,)).fetchone()
                if dup:
                    continue
                self._db.execute("INSERT INTO docs(ts,kind,subject,text,source,session) VALUES(?,?,?,?,?,?)",
                                 (base - (len(lines) - i) * 60, "turn", "user", user, "legacy", "legacy"))
            n += 1
        return n

    # ---- what the camera has seen ------------------------------------------
    def note_sightings(self, counts, ts=None):
        """counts: {label: how_many}. Remembers when each thing (object or person) was last in view."""
        ts = ts or time.time()
        with self._lock, self._db:
            for obj, n in counts.items():
                self._db.execute(
                    "INSERT INTO sightings(obj, ts, count, first_ts) VALUES(?,?,?,?) "
                    "ON CONFLICT(obj) DO UPDATE SET ts=excluded.ts, count=excluded.count", (obj, ts, int(n), ts))

    def last_sighting(self, obj):
        row = self._db.execute("SELECT ts, count FROM sightings WHERE obj=?", (obj,)).fetchone()
        return (row["ts"], row["count"]) if row else None

    def set_room_objects(self, room, counts, ts=None):
        self._meta_set(f"room_objects:{room}", json.dumps({"ts": ts or time.time(), "counts": counts}))

    def get_room_objects(self, room):
        raw = self._meta_get(f"room_objects:{room}")
        if not raw:
            return None
        d = json.loads(raw)
        return d["ts"], d["counts"]

    def latest_room_fact(self, room):
        row = self._db.execute(
            "SELECT text, ts FROM docs WHERE kind='room' AND active=1 AND subject=? ORDER BY ts DESC LIMIT 1", (room,)).fetchone()
        return (row["text"], row["ts"]) if row else None

    def close(self):
        with self._lock:
            self._db.close()
