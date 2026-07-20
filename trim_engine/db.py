"""
SQLite database layer — one project.db per video.

Full DDL from §6. Connection manager with auto-migration.
All downstream engines use these helpers to read/write analysis results.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

DDL = """
-- ═══════════════════════════════════════════════════════════════
-- Video & Job metadata
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS video (
    id              TEXT PRIMARY KEY,
    path            TEXT NOT NULL,
    duration_s      REAL NOT NULL,
    fps             REAL NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    codec           TEXT,
    is_vfr          INTEGER DEFAULT 0,
    content_class   TEXT DEFAULT 'standard',       -- standard|talking_head|screencast
    av_offset_ms    REAL DEFAULT 0.0,              -- measured container A/V start-time delta
    readiness_level INTEGER DEFAULT 0,             -- 0=uploaded, 1=speech, 2=visual, 3=semantic, 4=story
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS job_stages (
    stage       TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending|running|done|failed
    version     TEXT,
    duration_s  REAL,
    error       TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS coverage (
    analyzer    TEXT PRIMARY KEY,
    status      TEXT NOT NULL,                     -- available|unavailable|heuristic|low_confidence
    note        TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- Scenes
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS scenes (
    id              INTEGER PRIMARY KEY,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    shot_type       TEXT,
    camera_motion   TEXT,
    location        TEXT,
    emotion_label   TEXT,
    emotion_intensity REAL,
    caption         TEXT,
    is_broll        INTEGER DEFAULT 0,
    indoor          INTEGER,                         -- 1=indoor, 0=outdoor, NULL=unknown
    motion_score    REAL,
    importance      REAL,
    importance_why  TEXT,
    story_role      TEXT
);

CREATE INDEX IF NOT EXISTS idx_scenes_time ON scenes(start_time, end_time);

CREATE TABLE IF NOT EXISTS keyframes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id    INTEGER NOT NULL REFERENCES scenes(id),
    position    REAL NOT NULL,                   -- 0.1, 0.5, 0.9
    path        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_keyframes_scene ON keyframes(scene_id);

-- ═══════════════════════════════════════════════════════════════
-- Audio Intelligence
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS utterances (
    id            INTEGER PRIMARY KEY,
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL,
    text          TEXT NOT NULL,
    speaker_id    TEXT,
    dialogue_act  TEXT,                           -- question|answer|statement
    topic_id      INTEGER REFERENCES topics(id)
);

CREATE INDEX IF NOT EXISTS idx_utterances_time ON utterances(start_time, end_time);

CREATE TABLE IF NOT EXISTS words (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    utt_id          INTEGER NOT NULL REFERENCES utterances(id),
    idx             INTEGER NOT NULL,
    word            TEXT NOT NULL,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    prob            REAL,
    snap_tolerance  TEXT DEFAULT 'normal'           -- normal|wide (wide = decoder-only, no aligner)
);

CREATE INDEX IF NOT EXISTS idx_words_time ON words(start_time, end_time);
CREATE INDEX IF NOT EXISTS idx_words_utt ON words(utt_id);

CREATE TABLE IF NOT EXISTS silences (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    duration    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_silences_time ON silences(start_time, end_time);

CREATE TABLE IF NOT EXISTS beats (
    time_s        REAL PRIMARY KEY,
    is_downbeat   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS breaths (
    start_time    REAL NOT NULL,
    end_time      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fillers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    word        TEXT NOT NULL,
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    confidence  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_fillers_time ON fillers(start_time, end_time);

CREATE TABLE IF NOT EXISTS audio_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    type        TEXT NOT NULL,                    -- music|laughter|applause
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_audio_events_time ON audio_events(start_time, end_time);

CREATE TABLE IF NOT EXISTS beats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t           REAL NOT NULL,
    is_downbeat INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT NOT NULL,
    class       TEXT,                             -- intro|product|pricing|sponsor|joke|story|offtopic|other
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_topics_time ON topics(start_time, end_time);

CREATE TABLE IF NOT EXISTS retake_clusters (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    cluster_id  INTEGER NOT NULL,
    utt_id      INTEGER NOT NULL REFERENCES utterances(id),
    take_index  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_retakes_cluster ON retake_clusters(cluster_id);

-- ═══════════════════════════════════════════════════════════════
-- Knowledge Graph
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,                 -- person:A, object:laptop, location:office
    kind            TEXT NOT NULL,                    -- person|object|location|action|topic
    label           TEXT NOT NULL,
    description     TEXT,
    is_owner        INTEGER DEFAULT 0,
    owner_inferred  INTEGER DEFAULT 0                 -- 1 if owner was guessed by heuristic
);

CREATE TABLE IF NOT EXISTS relations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    src             TEXT NOT NULL,                    -- entity ID
    rel             TEXT NOT NULL,                    -- appears_in|contains|located_in|performs|expresses|speaks_about|holds|followed_by
    dst             TEXT NOT NULL,                    -- entity ID or scene_id reference
    scene_id        INTEGER REFERENCES scenes(id),
    t_start         REAL,
    t_end           REAL,
    confidence      REAL,
    source          TEXT,                             -- which analyzer produced this
    model_version   TEXT,                             -- version of the model that produced this
    evidence_ref    TEXT,                             -- reference to raw artifact (provenance)
    needs_verification INTEGER DEFAULT 0              -- single-source high-impact claims
);

CREATE INDEX IF NOT EXISTS idx_relations_src ON relations(src, rel);
CREATE INDEX IF NOT EXISTS idx_relations_dst ON relations(rel, dst);
CREATE INDEX IF NOT EXISTS idx_relations_scene ON relations(scene_id);

-- ═══════════════════════════════════════════════════════════════
-- Story
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS story_beats (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL,
    scene_ids   TEXT NOT NULL,                    -- JSON array of scene IDs
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS story_deps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    setup_scene     INTEGER NOT NULL,
    payoff_scene    INTEGER NOT NULL,
    why             TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- Edits & Cost Tracking
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS edits (
    version     INTEGER PRIMARY KEY,
    prompt      TEXT NOT NULL,
    intent_json TEXT NOT NULL,
    plan_json   TEXT NOT NULL,
    verdict_json TEXT,
    output_path TEXT,
    report_path TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS edit_sessions (
    id              TEXT PRIMARY KEY,
    video_id        TEXT NOT NULL,
    state           TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    version         INTEGER,
    budget_json     TEXT NOT NULL,
    session_json    TEXT NOT NULL,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT NOT NULL DEFAULT (datetime('now')),
    prompt_name     TEXT NOT NULL,
    in_tokens       INTEGER NOT NULL,
    out_tokens      INTEGER NOT NULL,
    cache_read      INTEGER DEFAULT 0,
    latency_ms      REAL NOT NULL,
    cost_usd        REAL NOT NULL
);

-- ═══════════════════════════════════════════════════════════════
-- Vector ID mappings (FAISS row-id ↔ DB id)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS scene_vectors (
    vector_id       INTEGER PRIMARY KEY,
    scene_id        INTEGER NOT NULL REFERENCES scenes(id),
    vector_type     TEXT NOT NULL,                   -- clip|text
    model_version   TEXT                             -- e.g. 'ViT-B-32/openai' or 'all-MiniLM-L6-v2'
);

CREATE TABLE IF NOT EXISTS utt_vectors (
    vector_id       INTEGER PRIMARY KEY,
    utt_id          INTEGER NOT NULL REFERENCES utterances(id),
    model_version   TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- Audio: Loudness Curve & Speaker Embeddings
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS loudness_curve (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    t           REAL NOT NULL,                       -- center of 200ms window
    rms_db      REAL NOT NULL,                       -- RMS energy in dB
    lufs_s      REAL                                 -- short-term LUFS (optional)
);

CREATE INDEX IF NOT EXISTS idx_loudness_time ON loudness_curve(t);

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  TEXT NOT NULL,
    embedding   BLOB NOT NULL,                       -- MFCC or d-vector embedding (numpy bytes)
    dim         INTEGER NOT NULL
);

-- ═══════════════════════════════════════════════════════════════
-- Cross-Modal Derived Events
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS derived_moments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,                       -- funny|applause|awkward
    start_time  REAL NOT NULL,
    end_time    REAL NOT NULL,
    formula     TEXT NOT NULL,                       -- e.g. 'laughter(audio) ∩ smile(visual)'
    confidence  REAL
);

CREATE INDEX IF NOT EXISTS idx_derived_moments_time ON derived_moments(start_time, end_time);

-- ═══════════════════════════════════════════════════════════════
-- Model Manifest (KB sealed against one manifest)
-- ═══════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS model_manifest (
    analyzer        TEXT PRIMARY KEY,                -- 'asr', 'clip', 'vlm', etc.
    model_name      TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    sealed_at       TEXT
);
"""


class ProjectDB:
    """
    SQLite connection manager for a single project.db.

    Usage:
        db = ProjectDB(project_dir / "project.db")
        db.initialize()
        with db.conn() as c:
            c.execute("INSERT INTO scenes ...")
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        import threading
        self._local = threading.local()

    @property
    def _connection(self) -> sqlite3.Connection | None:
        return getattr(self._local, "connection", None)

    @_connection.setter
    def _connection(self, conn: sqlite3.Connection | None) -> None:
        self._local.connection = conn

    def initialize(self) -> None:
        """Create the database and run all DDL migrations."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(DDL)
            
            # Migration guards — add columns that may be missing in older DBs
            _migrations = [
                ("scenes", "indoor", "INTEGER"),
            ]
            for table, col, col_type in _migrations:
                try:
                    c.execute(f"SELECT {col} FROM {table} LIMIT 1")
                except sqlite3.OperationalError:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
            
            c.execute("""
                CREATE TABLE IF NOT EXISTS breaths (
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    confidence REAL,
                    PRIMARY KEY (start_time, end_time)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cut_affinity (
                    t REAL PRIMARY KEY,
                    score REAL NOT NULL
                )
            """)

            from trim_engine.config import CFG
            try:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS engine_manifest (version TEXT PRIMARY KEY)"
                )
                CURRENT_VERSION = "0.2.0"
                row = c.execute("SELECT version FROM engine_manifest").fetchone()
                if row:
                    if row["version"] != CURRENT_VERSION:
                        import logging
                        logging.getLogger("trim").warning(
                            f"Database created by older engine version ({row['version']}). "
                            f"Some data might be incompatible with {CURRENT_VERSION}."
                        )
                else:
                    c.execute("INSERT INTO engine_manifest (version) VALUES (?)", (CURRENT_VERSION,))
            except Exception:
                pass

    @contextmanager
    def conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Provide a transactional connection (auto-commit on success)."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
                timeout=60.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None

    
    
    

    def set_video(
        self, video_id: str, path: str, duration_s: float,
        fps: float, width: int, height: int, codec: str = "", is_vfr: bool = False,
        content_class: str = "standard", av_offset_ms: float = 0.0,
    ) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO video
                   (id, path, duration_s, fps, width, height, codec, is_vfr, content_class, av_offset_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (video_id, path, duration_s, fps, width, height, codec, int(is_vfr),
                 content_class, av_offset_ms),
            )

    def get_video(self) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM video LIMIT 1").fetchone()
            return dict(row) if row else None

    
    
    

    def set_stage(self, stage: str, status: str, version: str | None = None, duration_s: float | None = None, error: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO job_stages (stage, status, version, duration_s, error, updated_at)
                   VALUES (?, ?, ?, ?, ?, datetime('now'))""",
                (stage, status, version, duration_s, error),
            )

    def get_stage(self, stage: str) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM job_stages WHERE stage = ?", (stage,)).fetchone()
            return dict(row) if row else None

    def get_all_stages(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM job_stages ORDER BY rowid").fetchall()]

    
    
    

    def set_coverage(self, analyzer: str, status: str, note: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO coverage (analyzer, status, note) VALUES (?, ?, ?)",
                (analyzer, status, note),
            )

    def get_coverage(self) -> dict[str, str]:
        with self.conn() as c:
            rows = c.execute("SELECT analyzer, status FROM coverage").fetchall()
            return {r["analyzer"]: r["status"] for r in rows}

    
    
    

    def insert_scene(self, scene_id: int, start: float, end: float, **kwargs: Any) -> None:
        cols = ["id", "start_time", "end_time"] + list(kwargs.keys())
        vals = [scene_id, start, end] + list(kwargs.values())
        placeholders = ", ".join(["?"] * len(vals))
        col_str = ", ".join(cols)
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO scenes ({col_str}) VALUES ({placeholders})", vals)

    def insert_silence(self, start: float, end: float) -> None:
        with self.conn() as c:
            c.execute("INSERT OR REPLACE INTO silences (start_time, end_time, duration) VALUES (?, ?, ?)", (start, end, end - start))

    def insert_breath(self, start: float, end: float, confidence: float = 1.0) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO breaths (start_time, end_time) VALUES (?, ?)", (start, end))
        
    def insert_cut_affinity(self, data: list[tuple[float, float]]) -> None:
        with self.conn() as c:
            c.executemany("INSERT OR REPLACE INTO cut_affinity (t, score) VALUES (?, ?)", data)

    def get_silences(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM silences ORDER BY start_time").fetchall()]

    def get_breaths(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM breaths ORDER BY start_time").fetchall()]
        
    def get_cut_affinity(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(row) for row in c.execute("SELECT time_s, score FROM cut_affinity ORDER BY time_s").fetchall()]
            
    def insert_beats(self, beats: list[tuple[float, int]]) -> None:
        with self.conn() as c:
            c.executemany(
                "INSERT INTO beats (time_s, is_downbeat) VALUES (?, ?) ON CONFLICT(time_s) DO UPDATE SET is_downbeat=excluded.is_downbeat",
                beats
            )
            
    def get_beats(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(row) for row in c.execute("SELECT time_s, is_downbeat FROM beats ORDER BY time_s").fetchall()]

    def get_scenes(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM scenes ORDER BY start_time").fetchall()]

    def get_scene(self, scene_id: int) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM scenes WHERE id = ?", (scene_id,)).fetchone()
            return dict(row) if row else None

    def update_scene(self, scene_id: int, **kwargs: Any) -> None:
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        with self.conn() as c:
            c.execute(f"UPDATE scenes SET {set_clause} WHERE id = ?", [*kwargs.values(), scene_id])

    
    
    

    def insert_keyframe(self, scene_id: int, position: float, path: str) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO keyframes (scene_id, position, path) VALUES (?, ?, ?)",
                (scene_id, position, path),
            )

    def get_keyframes(self, scene_id: int) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM keyframes WHERE scene_id = ? ORDER BY position", (scene_id,)
            ).fetchall()]

    def get_all_keyframes(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM keyframes ORDER BY position").fetchall()]

    
    
    

    def insert_utterance(self, utt_id: int, start: float, end: float, text: str,
                         speaker_id: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO utterances (id, start_time, end_time, text, speaker_id) VALUES (?, ?, ?, ?, ?)",
                (utt_id, start, end, text, speaker_id),
            )

    def get_utterances(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM utterances ORDER BY start_time").fetchall()]

    def update_utterance(self, utt_id: int, **kwargs: Any) -> None:
        if not kwargs:
            return
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        with self.conn() as c:
            c.execute(f"UPDATE utterances SET {set_clause} WHERE id = ?", [*kwargs.values(), utt_id])

    def insert_word(self, utt_id: int, idx: int, word: str, start: float, end: float, prob: float | None = None, snap_tolerance: str = "normal") -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO words (utt_id, idx, word, start_time, end_time, prob, snap_tolerance) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (utt_id, idx, word, start, end, prob, snap_tolerance),
            )

    def get_words(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM words ORDER BY start_time").fetchall()]

    def get_words_in_range(self, start: float, end: float) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM words WHERE end_time >= ? AND start_time <= ? ORDER BY start_time",
                (start, end),
            ).fetchall()]

    
    
    

    def insert_silence(self, start: float, end: float) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO silences (start_time, end_time, duration) VALUES (?, ?, ?)",
                (start, end, end - start),
            )

    def get_silences(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM silences ORDER BY start_time").fetchall()]

    def insert_filler(self, word: str, start: float, end: float, confidence: float) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO fillers (word, start_time, end_time, confidence) VALUES (?, ?, ?, ?)",
                (word, start, end, confidence),
            )

    def get_fillers(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM fillers ORDER BY start_time").fetchall()]

    def insert_audio_event(self, event_type: str, start: float, end: float, confidence: float | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO audio_events (type, start_time, end_time, confidence) VALUES (?, ?, ?, ?)",
                (event_type, start, end, confidence),
            )

    def get_audio_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        with self.conn() as c:
            if event_type:
                return [dict(r) for r in c.execute(
                    "SELECT * FROM audio_events WHERE type = ? ORDER BY start_time", (event_type,)
                ).fetchall()]
            return [dict(r) for r in c.execute("SELECT * FROM audio_events ORDER BY start_time").fetchall()]

    def insert_beat(self, t: float, is_downbeat: int = 0) -> None:
        with self.conn() as c:
            c.execute("INSERT OR IGNORE INTO beats (time_s, is_downbeat) VALUES (?, ?)", (t, is_downbeat))

    def get_beats(self) -> list[float]:
        with self.conn() as c:
            return [r["time_s"] for r in c.execute("SELECT time_s FROM beats ORDER BY time_s").fetchall()]

    
    
    

    def insert_topic(self, label: str, topic_class: str, start: float, end: float) -> int:
        with self.conn() as c:
            cursor = c.execute(
                "INSERT INTO topics (label, class, start_time, end_time) VALUES (?, ?, ?, ?)",
                (label, topic_class, start, end),
            )
            return cursor.lastrowid  

    def get_topics(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM topics ORDER BY start_time").fetchall()]

    
    
    

    def insert_retake(self, cluster_id: int, utt_id: int, take_index: int) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO retake_clusters (cluster_id, utt_id, take_index) VALUES (?, ?, ?)",
                (cluster_id, utt_id, take_index),
            )

    def get_retake_clusters(self) -> dict[int, list[dict[str, Any]]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM retake_clusters ORDER BY cluster_id, take_index").fetchall()
            clusters: dict[int, list[dict[str, Any]]] = {}
            for r in rows:
                d = dict(r)
                clusters.setdefault(d["cluster_id"], []).append(d)
            return clusters

    
    
    

    def insert_entity(self, entity_id: str, kind: str, label: str,
                      description: str | None = None, is_owner: bool = False,
                      owner_inferred: bool = False) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO entities
                   (id, kind, label, description, is_owner, owner_inferred)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (entity_id, kind, label, description, int(is_owner), int(owner_inferred)),
            )

    def get_entities(self, kind: str | None = None) -> list[dict[str, Any]]:
        with self.conn() as c:
            if kind:
                return [dict(r) for r in c.execute(
                    "SELECT * FROM entities WHERE kind = ?", (kind,)
                ).fetchall()]
            return [dict(r) for r in c.execute("SELECT * FROM entities").fetchall()]

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)).fetchone()
            return dict(row) if row else None

    def insert_relation(self, src: str, rel: str, dst: str, scene_id: int | None = None,
                        t_start: float | None = None, t_end: float | None = None,
                        confidence: float | None = None, source: str | None = None,
                        model_version: str | None = None, evidence_ref: str | None = None,
                        needs_verification: bool = False) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO relations
                   (src, rel, dst, scene_id, t_start, t_end, confidence, source, model_version, evidence_ref, needs_verification)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (src, rel, dst, scene_id, t_start, t_end, confidence, source,
                 model_version, evidence_ref, int(needs_verification)),
            )

    def update_relation_verification(self, rel_id: int, needs_verification: bool) -> None:
        with self.conn() as c:
            c.execute(
                "UPDATE relations SET needs_verification = ? WHERE id = ?",
                (int(needs_verification), rel_id),
            )

    def get_relations(self, src: str | None = None, rel: str | None = None,
                      dst: str | None = None) -> list[dict[str, Any]]:
        with self.conn() as c:
            clauses: list[str] = []
            params: list[Any] = []
            if src:
                clauses.append("src = ?")
                params.append(src)
            if rel:
                clauses.append("rel = ?")
                params.append(rel)
            if dst:
                clauses.append("dst = ?")
                params.append(dst)
            where = " AND ".join(clauses) if clauses else "1=1"
            return [dict(r) for r in c.execute(
                f"SELECT * FROM relations WHERE {where} ORDER BY t_start", params
            ).fetchall()]

    
    
    

    def insert_story_beat(self, role: str, scene_ids: list[int], summary: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO story_beats (role, scene_ids, summary) VALUES (?, ?, ?)",
                (role, json.dumps(scene_ids), summary),
            )

    def get_story_beats(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM story_beats ORDER BY id").fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["scene_ids"] = json.loads(d["scene_ids"])
                result.append(d)
            return result

    def insert_story_dep(self, setup_scene: int, payoff_scene: int, why: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO story_deps (setup_scene, payoff_scene, why) VALUES (?, ?, ?)",
                (setup_scene, payoff_scene, why),
            )

    def get_story_deps(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM story_deps").fetchall()]

    
    
    

    def next_edit_version(self) -> int:
        with self.conn() as c:
            row = c.execute("SELECT MAX(version) as v FROM edits").fetchone()
            return (row["v"] or 0) + 1

    def insert_edit(self, version: int, prompt: str, intent_json: str, plan_json: str,
                    verdict_json: str | None = None, output_path: str | None = None,
                    report_path: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO edits (version, prompt, intent_json, plan_json, verdict_json, output_path, report_path)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version, prompt, intent_json, plan_json, verdict_json, output_path, report_path),
            )

    def get_edit(self, version: int) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM edits WHERE version = ?", (version,)).fetchone()
            return dict(row) if row else None

    def get_all_edits(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM edits ORDER BY version").fetchall()]

    def insert_edit_session(self, session_id: str, video_id: str, state: str, prompt: str,
                            version: int | None, budget_json: str, session_json: str) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO edit_sessions (id, video_id, state, prompt, version, budget_json, session_json, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (session_id, video_id, state, prompt, version, budget_json, session_json),
            )

    def update_edit_session(self, session_id: str, state: str, budget_json: str, session_json: str, version: int | None = None) -> None:
        with self.conn() as c:
            if version is not None:
                c.execute(
                    """UPDATE edit_sessions SET state = ?, budget_json = ?, session_json = ?, version = ?, updated_at = datetime('now')
                       WHERE id = ?""",
                    (state, budget_json, session_json, version, session_id),
                )
            else:
                c.execute(
                    """UPDATE edit_sessions SET state = ?, budget_json = ?, session_json = ?, updated_at = datetime('now')
                       WHERE id = ?""",
                    (state, budget_json, session_json, session_id),
                )

    def get_edit_session(self, session_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM edit_sessions WHERE id = ?", (session_id,)).fetchone()
            return dict(row) if row else None

    def get_active_edit_sessions(self, video_id: str) -> list[dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM edit_sessions WHERE video_id = ? AND state NOT IN ('delivered', 'render_failed', 'resolved_noop')", (video_id,)).fetchall()
            return [dict(r) for r in rows]

    def delete_edit_session(self, session_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM edit_sessions WHERE id = ?", (session_id,))

    
    
    

    def log_llm_call(self, prompt_name: str, in_tokens: int, out_tokens: int,
                     cache_read: int, latency_ms: float, cost_usd: float) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO llm_calls (prompt_name, in_tokens, out_tokens, cache_read, latency_ms, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (prompt_name, in_tokens, out_tokens, cache_read, latency_ms, cost_usd),
            )

    def get_total_cost(self) -> float:
        with self.conn() as c:
            row = c.execute("SELECT COALESCE(SUM(cost_usd), 0) as total FROM llm_calls").fetchone()
            return row["total"]

    def get_llm_calls(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM llm_calls ORDER BY ts").fetchall()]

    
    
    

    def insert_scene_vector(self, vector_id: int, scene_id: int, vector_type: str,
                            model_version: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO scene_vectors
                   (vector_id, scene_id, vector_type, model_version)
                   VALUES (?, ?, ?, ?)""",
                (vector_id, scene_id, vector_type, model_version),
            )

    def insert_utt_vector(self, vector_id: int, utt_id: int,
                          model_version: str | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO utt_vectors
                   (vector_id, utt_id, model_version)
                   VALUES (?, ?, ?)""",
                (vector_id, utt_id, model_version),
            )

    def get_scene_id_for_vector(self, vector_id: int, vector_type: str) -> int | None:
        with self.conn() as c:
            row = c.execute(
                "SELECT scene_id FROM scene_vectors WHERE vector_id = ? AND vector_type = ?",
                (vector_id, vector_type),
            ).fetchone()
            return row["scene_id"] if row else None

    def get_utt_id_for_vector(self, vector_id: int) -> int | None:
        with self.conn() as c:
            row = c.execute("SELECT utt_id FROM utt_vectors WHERE vector_id = ?", (vector_id,)).fetchone()
            return row["utt_id"] if row else None

    
    
    

    def insert_loudness_sample(self, t: float, rms_db: float, lufs_s: float | None = None) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO loudness_curve (t, rms_db, lufs_s) VALUES (?, ?, ?)",
                (t, rms_db, lufs_s),
            )

    def get_loudness_curve(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM loudness_curve ORDER BY t").fetchall()]

    
    
    

    def insert_speaker_embedding(self, speaker_id: str, embedding: bytes, dim: int) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO speaker_embeddings (speaker_id, embedding, dim) VALUES (?, ?, ?)",
                (speaker_id, embedding, dim),
            )

    def get_speaker_embeddings(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM speaker_embeddings").fetchall()]

    
    
    

    def insert_derived_moment(self, kind: str, start: float, end: float,
                              formula: str, confidence: float | None = None) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT INTO derived_moments (kind, start_time, end_time, formula, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (kind, start, end, formula, confidence),
            )

    def get_derived_moments(self, kind: str | None = None) -> list[dict[str, Any]]:
        with self.conn() as c:
            if kind:
                return [dict(r) for r in c.execute(
                    "SELECT * FROM derived_moments WHERE kind = ? ORDER BY start_time", (kind,)
                ).fetchall()]
            return [dict(r) for r in c.execute(
                "SELECT * FROM derived_moments ORDER BY start_time"
            ).fetchall()]

    
    
    

    def set_model_manifest(self, analyzer: str, model_name: str, model_version: str) -> None:
        with self.conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO model_manifest
                   (analyzer, model_name, model_version, sealed_at)
                   VALUES (?, ?, ?, datetime('now'))""",
                (analyzer, model_name, model_version),
            )

    def get_model_manifest(self) -> dict[str, dict[str, str]]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM model_manifest").fetchall()
            return {r["analyzer"]: {"model_name": r["model_name"],
                                     "model_version": r["model_version"]}
                    for r in rows}

    
    
    

    def update_readiness_level(self, level: int) -> None:
        """Update the video's readiness level (only increases, never decreases)."""
        with self.conn() as c:
            c.execute(
                "UPDATE video SET readiness_level = MAX(readiness_level, ?) WHERE 1=1",
                (level,),
            )

    def get_readiness_level(self) -> int:
        video = self.get_video()
        return video.get("readiness_level", 0) if video else 0
