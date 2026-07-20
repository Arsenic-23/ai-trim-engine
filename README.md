<p align="center">
  <h1 align="center">AI Trim Engine</h1>
</p>

<p align="center">
  <strong>Natural-language video editing that feels like a professional human editor.</strong><br>
  <em>Prompt in → verified, frame-accurate, narratively-aware edit out.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FFmpeg-7.x-red?style=for-the-badge&logo=ffmpeg&logoColor=white" alt="FFmpeg">
  <img src="https://img.shields.io/badge/Claude-AWS_Bedrock-purple?style=for-the-badge&logo=anthropic&logoColor=white" alt="Anthropic Bedrock">
  <img src="https://img.shields.io/badge/FAISS-Vector_Search-green?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/CPU_Only-No_GPU_Required-orange?style=for-the-badge" alt="CPU Only">
</p>

---

## What it is

CRAON converts a raw video into a **queryable semantic knowledge base** — transcript words, silences, breaths, scenes, faces, speakers, objects, emotions, story beats, beat grids, and a 10 Hz *cut-affinity curve* — then treats every natural-language editing request as a **retrieval + reasoning problem** over that knowledge base:

```
craon edit <video_id> "Remove filler words and make it a fast-paced 30-second trailer" --yes
```

No hardcoded features. A prompt the engine has never seen before is compiled into a structured intent by Claude, grounded against the video's actual content, planned deterministically, validated by an adversarial critic, and rendered by FFmpeg with human-editor craft: cuts land on breaths and motion minima, J/L audio offsets bridge seams, tempo curves shape pacing, and platform presets reframe for 9:16.

## Why it's different

| Conventional auto-trimmers | CRAON |
|---|---|
| Fixed feature list (silence removal, etc.) | **Generalizes** — any prompt becomes retrieval + reasoning |
| LLM decides frame numbers (hallucination risk) | **LLM plans, never edits** — deterministic planner owns all frame math |
| Cuts wherever thresholds fire | **Cut-Point Engine** — cuts snap to breaths, blinks, motion minima via a precomputed affinity curve |
| Chronological output only | **True reordering** — hook-first, trailer structure, match-cut chains |
| Silent failure / fake success | **Honesty layer** — no-match is reported, degraded signals refuse gracefully, every claim links to evidence |
| One-shot | **Multi-agent verification** — an LLM critic re-validates every plan against the original intent, with typed retry routing |

## Core capabilities

- **Basic edits** — filler words (incl. multi-word phrases), silences/pauses, retakes (cross-take similarity clustering), dead air.
- **Scene-based** — intro/outro, B-roll, indoor/outdoor, "everything before I enter the frame" (temporal anchor resolution).
- **Person & object** — face detection → identity clustering → "remove every shot with Person B"; object-relation + CLIP visual search for "keep only shots with the product visible."
- **Emotion & action** — laughter/applause via audio-event classification, derived moments (`funny`, `awkward`, `applause`), emotion-intensity channel.
- **Speech & content** — topic segmentation ("remove pricing mentions"), per-utterance dialogue acts ("keep only questions"), off-topic detection, sponsor mentions.
- **Cinematic** — beat-aligned cutting (Demucs-separated music stem → beat grid, with onset fallback for non-music footage), match cuts (CLIP boundary-frame chaining), tempo-curve pacing, trailer/hook-first restructuring.
- **Intelligent understanding** — "make it shorter," "under 30 seconds," "more engaging," platform presets (TikTok / Reels / Shorts → 9:16 + duration caps).
- **Conversational revision** — `--revise v3 "put the intro back"` produces a delta timeline with region-only re-rendering.

## Architecture at a glance

```
 INGESTION (once per video, parallel DAG)          QUERY (per prompt, interactive)
 ┌────────────────────────────────────┐            ┌─────────────────────────────────┐
 │ normalize → scenes ─┬─ faces       │            │ Intent Compiler   (Claude)      │
 │            → audio ─┼─ cut_affinity│            │        ↓                        │
 │ audio_separation ───┴─ beat_grid   │  prompt →  │ Hybrid Retrieval  (7 channels)  │
 │            → vision → graph        │            │        ↓                        │
 │            → index  → story        │            │ Story Agent       (knapsack +   │
 │                                    │            │        ↓           reordering)  │
 │  SQLite KB + FAISS indexes +       │──────────→ │ Timeline Planner  (deterministic│
 │  BM25 + cut-affinity curve         │            │        ↓           frame math)  │
 └────────────────────────────────────┘            │ Critic Validator  (3 tiers)     │
                                                   │        ↓                        │
                                                   │ FFmpeg Renderer   (3-rung       │
                                                   │                    fallback +   │
                                                   │                    verification)│
                                                   └─────────────────────────────────┘
```

Full design rationale, data model, state machines, and failure taxonomy: **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Quickstart

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | `3.12.*` | pinned in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | latest | package manager |
| FFmpeg | `7.x` | on `PATH` (`ffmpeg -version`) |
| AWS credentials | — | Bedrock access with an Anthropic Claude model enabled |

Everything else — Whisper, Silero-VAD, SentenceTransformers, CLIP, FAISS, MediaPipe, Respiro-en, Demucs — runs **CPU-only** and is installed automatically. Bedrock is the only paid dependency.

### Install

```bash
# One-liner (macOS / Linux) — installs the global `craon` command
curl -fsSL https://raw.githubusercontent.com/Arsenic-23/ai-trim-engine/main/install.sh | bash
```

### Verify Bedrock connectivity

```bash
aws configure                # or export AWS_PROFILE / AWS_REGION
craon bedrock-smoke          # one structured call + one vision call, asserts cost metering
```

Model and region are configurable via `.env` / environment:

```bash
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-6   # default
AWS_REGION=us-east-1                              # default
```

## Usage

### 1 — Ingest (once per video)

```bash
craon ingest path/to/video.mp4
```

Runs the parallel ingestion DAG: normalization → scene/shot detection → ASR with word-level CTC alignment → VAD/silence/filler/breath detection → Demucs speech/music separation → beat grid → face tracking + identity clustering → vision tagging → knowledge graph → FAISS/BM25 indexes → story mapping → cut-affinity curve. Stages are content-hashed, resumable, and lease-protected — re-running skips completed work.

Prints a `video_id` (content hash) used by every subsequent command.

### 2 — Ask (verify retrieval before editing)

```bash
craon ask <video_id> "When do I mention pricing?"
```

### 3 — Edit

```bash
# Mechanical
craon edit <video_id> "Remove filler words and long pauses" --yes

# Semantic
craon edit <video_id> "Remove every shot where the whiteboard is visible" --yes

# Narrative restructure
craon edit <video_id> "Start with my strongest hook, then build to the reveal" --yes

# Platform + duration
craon edit <video_id> "Make this a 45-second TikTok" --yes
```

Without `--yes`, the engine shows predicted duration / removal ratio / clip count and a stream-copied preview before committing to the render.

### 4 — Revise conversationally

```bash
craon edit <video_id> --revise v1 "put the intro back and tighten the middle" --yes
```

### 5 — Inspect

```bash
craon status <video_id>       # stage status, analysis coverage, total LLM cost
```

Per-edit artifacts land in `projects/<video_id>/edits/vN/`:

| Artifact | Purpose |
|---|---|
| `output.mp4` | rendered edit |
| `timeline.json` | the exact deterministic render contract |
| `report.md` | evidence-linked explanation of every cut |
| `render_log.json` | per-clip strategy, verification results |
| `output.srt` / `output.vtt` | resynced captions |
| `preview.mp4` / `thumbnail.jpg` | fast preview + thumbnail |

### Regression suite

```bash
craon suite <video_id>        # 25-prompt assignment suite → sample_outputs/ with per-prompt
                              # intent/plan/verdict/report artifacts and an index README
```

### Interactive shell & HTTP API

```bash
craon                         # interactive shell: /ingest /edit /ask /status /suite
uvicorn trim_engine.api:app   # REST: /ingest /edit/{id} /status/{id} /edits/{id}/{v}/output
```

### Frontend Demo

The repository includes a web frontend demo for testing the AI Trim Engine visually.

```bash
cd frontend_demo
python3 -m http.server 8000
```
Then open `http://localhost:8000` in your browser. Ensure the FastAPI backend is also running in a separate terminal via `uvicorn trim_engine.api:app`.

## Engineering guarantees

- **Determinism** — identical `timeline.json` in, identical cut list out. All LLM outputs are schema-validated (Pydantic) with self-repair retries; frame math never touches an LLM.
- **Budget envelopes** — per-session caps on LLM calls, retries, wall-clock, and USD spend; oscillation (plan flapping) detection aborts pathological retry loops.
- **Render resilience** — three-rung fallback ladder (hw-codec fast path → software re-encode → MoviePy), each rung gated by post-render probes (container integrity, duration drift, A/V sync).
- **Honest failure** — no-match resolves to an explicit no-op, never a silent passthrough; degraded analyzers write coverage flags that gate what the engine will claim it can do; every failure carries a typed exception and a recovery hint.
- **Cost transparency** — every LLM call is metered (tokens, cache hits, latency, USD) into the project DB; `craon status` shows the running total.

## Project layout

```
trim_engine/
├── cli.py               # typer CLI (+ central error boundary)
├── api.py               # FastAPI surface
├── shell.py             # interactive REPL
├── config.py            # ALL tunables, single source of truth
├── schemas.py           # Pydantic contracts (EditIntent, Timeline, …)
├── db.py                # SQLite KB (25+ tables, migrations, cost meter)
├── llm.py               # Bedrock wrapper: structured output, retries, caching, cost
├── models/              # bundled CPU models (Respiro-en, MediaPipe landmarker)
├── ingest/              # ingestion DAG stages (see ARCHITECTURE.md §4)
└── query/               # prompt → edit pipeline (see ARCHITECTURE.md §5)
    └── renderer/        # strategy planning, FFmpeg execution, verification
prompts/                 # versioned system prompts (one file per agent)
tests/                   # unit + golden e2e suite (pytest; `-m "not llm"` for offline)
projects/<video_id>/     # per-video KB, indexes, edits
```

## Development

```bash
uv sync --extra dev
uv run pytest -m "not llm"     # fast offline tests
uv run pytest                  # full suite incl. golden e2e (needs ingested fixture + Bedrock)
uv run ruff check .
```

---

> **Compute profile:** ingestion is the heavy phase (~2–5 min for a 10-min video on a modern laptop CPU, one-time per video). Querying and editing are interactive. No GPU is used anywhere; AWS Bedrock (Claude) is the sole external service.
