# AI Trim Engine · System Architecture

> **One-line summary:** A video-intelligence platform that converts video into a queryable semantic knowledge base at ingestion time, compiles natural-language prompts into structured edit intents, plans edits through a multi-agent retrieve→reason→plan→verify loop, and renders deterministically via FFmpeg — so *any* trimming request is handled by composition, never by a hardcoded feature.

---

## Table of contents

1. [Design principles](#1-design-principles)
2. [System overview](#2-system-overview)
3. [Data model — the Video Knowledge Base](#3-data-model--the-video-knowledge-base)
4. [Ingestion plane](#4-ingestion-plane)
5. [Query plane](#5-query-plane)
6. [The Cut-Point Engine](#6-the-cut-point-engine)
7. [Rendering pipeline](#7-rendering-pipeline)
8. [LLM integration layer](#8-llm-integration-layer)
9. [Failure taxonomy & honesty layer](#9-failure-taxonomy--honesty-layer)
10. [Session state machine & budgets](#10-session-state-machine--budgets)
11. [Interfaces](#11-interfaces)
12. [Configuration & extensibility](#12-configuration--extensibility)
13. [Testing strategy](#13-testing-strategy)

---

## 1. Design principles

Every architectural decision traces back to one of these six principles:

| # | Principle | Consequence |
|---|-----------|-------------|
| **P1** | **The video disappears early.** Pixels → semantics once, at ingestion. | All editing happens against a knowledge base, never against raw media. Query latency is decoupled from video length. |
| **P2** | **The LLM plans; it never edits.** | Claude emits structured JSON (intents, orderings, verdicts). A deterministic planner owns every frame number. No hallucinated timecodes can reach the renderer. |
| **P3** | **Editing = Retrieve → Reason → Plan → Verify → Execute.** | Novel prompts are handled by composing retrieval channels and reasoning stages — generalization is architectural, not feature-by-feature. |
| **P4** | **Every plan is verified before and after rendering.** | A three-tier critic validates the plan against the original intent; post-render probes validate the artifact. Failed verification triggers typed retry routing, never silent delivery. |
| **P5** | **Cut like a human editor.** | Cut placement is driven by a precomputed cut-affinity curve (breaths, motion minima, word boundaries); seams are concealed with J/L offsets and crossfades; pacing follows tempo curves. |
| **P6** | **Fail honestly.** | No-match → explicit no-op. Degraded analyzer → coverage flag → capability-gated claims. Every error is typed and carries a recovery hint. Evidence links every removal to its source signal. |

---

## 2. System overview

The engine is split into two strictly separated planes. The **ingestion plane** runs once per video (heavy, parallel, resumable). The **query plane** runs per prompt (interactive, LLM-orchestrated, budgeted).

```
════════════════════════ INGESTION PLANE — parallel DAG, once per video ════════════════════════

  original.mp4
      │
      ▼
 ┌───────────┐    proxy.mp4 / audio.wav / frame_lut
 │ normalize │──────────────┬────────────────┬──────────────────┐
 └───────────┘              │                │                  │
      ▼                     ▼                ▼                  ▼
 ┌───────────┐        ┌───────────┐   ┌──────────────────┐ ┌──────────────┐
 │  scenes   │        │   audio   │   │ audio_separation │ │    (copy)    │
 │ PySceneDet│        │ Whisper + │   │     Demucs       │ └──────────────┘
 └─────┬─────┘        │ CTC align │   └────────┬─────────┘
       │              │ VAD/breath│            ▼
       ▼              └─────┬─────┘      ┌───────────┐
 ┌───────────┐              │            │ beat_grid │
 │   faces   │              ▼            │  librosa  │
 │SCRFD+Arc- │        ┌──────────────┐   └───────────┘
 │Face+Byte- │        │ cut_affinity │
 │Track+HAC  │        │ 10Hz curve   │
 └─────┬─────┘        └──────────────┘
       │
       ▼  (scenes ∧ audio ∧ faces)
 ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
 │  vision   │─────▶│   graph   │─────▶│   index   │─────▶│   story   │
 │Claude VLM │      │ entities+ │      │FAISS+BM25 │      │ beats +   │
 │ tagging   │      │ relations │      │ 3 spaces  │      │ deps      │
 └───────────┘      └───────────┘      └───────────┘      └───────────┘

                              ▼
        SQLite KB (25+ tables) + FAISS indexes + BM25 + cut-affinity curve

═════════════════════════ QUERY PLANE — per prompt, interactive ═════════════════════════

   User prompt ──┬── Video Memory (this video's KB)
                 ├── Editor Profile (learned preferences)
                 └── Platform Knowledge (TikTok/Reels/YouTube presets)
                 ▼
      ┌─────────────────────┐   fast-path pattern bank OR Claude structured compile,
      │  INTENT COMPILER    │   grounded with video summary + profile; anchors,
      │      (Agent 1)      │   ambiguities, out-of-scope resolution
      └──────────┬──────────┘
                 ▼
      ┌─────────────────────┐   7 fused channels: structured (silences/fillers/retakes/
      │  HYBRID RETRIEVAL   │   audio-events/objects/emotions/derived-moments/dialogue-
      │      (Agent 2)      │   acts/anchors) + BM25 + MiniLM FAISS + CLIP text→image +
      └──────────┬──────────┘   metadata; RRF fusion + cross-encoder re-rank
                 ▼
      ┌─────────────────────┐   importance-scored knapsack under duration budget,
      │    STORY AGENT      │   narrative templates (hook_first/trailer/highlight),
      │      (Agent 3)      │   CLIP match-cut chaining, LLM taste pass → ordered_segments
      └──────────┬──────────┘
                 ▼
      ┌─────────────────────┐   100% deterministic: removal merge → cut-affinity snapping
      │  TIMELINE PLANNER   │   → micro-gap merge → min-clip → beat snap → frame-LUT
      │     (no LLM)        │   quantization → reorder permutation → invariant repair
      └──────────┬──────────┘   → tempo map → J/L offsets → Timeline JSON
                 ▼
      ┌─────────────────────┐   Tier 0 structural asserts · Tier 1 semantic re-query
      │  CRITIC VALIDATOR   │   ("is the removed thing really gone?") · Tier 2 Claude
      │      (Agent 4)      │   judgment on the kept transcript in final order
      └──────────┬──────────┘   fail → typed retry route (retrieval/story/planner)
                 ▼
      ┌─────────────────────┐   strategy per clip (smart-copy/boundary/full re-encode),
      │  RENDER PIPELINE    │   3-rung fallback ladder, two-pass loudnorm, captions,
      │   (deterministic)   │   post-render probes (duration/AV-sync/container)
      └─────────────────────┘
```

### 2.1 Why two planes

Heavy ML inference (ASR, CLIP, face tracking, source separation) is amortized once at ~2–5 min per 10-min video, all CPU. After that, edits touch only the KB: intent compilation and retrieval complete in seconds, and render time scales with *output* length, not source length (stream-copy fast paths). This is what makes conversational iteration ("tighter", "put that back") viable.

---

## 3. Data model — the Video Knowledge Base

One SQLite database per video (`projects/<video_id>/project.db`, WAL mode), plus sidecar FAISS/BM25 indexes. `video_id` is a content hash — re-ingesting the same file is a no-op.

### 3.1 Table groups (`db.py`)

| Group | Tables | Written by | Consumed by |
|---|---|---|---|
| **Media** | `video`, `job_stages`, `coverage`, `model_manifest` | orchestrator | everything; coverage gates capability claims |
| **Visual** | `scenes` (incl. `indoor`, `shot_type`, `emotion_label`, `is_broll`, `story_role`), `keyframes` | scenes, vision | retrieval metadata channel, render strategy |
| **Speech** | `utterances` (incl. per-utterance `dialogue_act`), `words` (CTC-aligned), `silences`, `fillers`, `topics`, `retake_clusters`, `speaker_embeddings` | audio | structured retrieval, planner snapping, critic re-query, captions |
| **Audio events** | `audio_events` (laughter/applause/music), `beats` (Demucs-stem grid), `breaths` (Respiro-en), `loudness_curve` | audio, beat_grid | retrieval, beat snapping, cut affinity |
| **Graph** | `entities` (people/objects/locations), `relations` (`appears_in`, `contains`, `expresses`, `performs`), `derived_moments` (`funny`/`awkward`/`applause`/`off_topic`/`montage`) | faces, vision, graph | structured + anchor retrieval |
| **Story** | `story_beats` (hook/setup/climax/payoff…), `story_deps` (setup→payoff) | story | story agent knapsack + reordering |
| **Cut craft** | `cut_affinity` (10 Hz scored curve) | cut_affinity | planner snapping (§6) |
| **Vectors** | `scene_vectors`, `utt_vectors` (FAISS row maps) | index | vector retrieval |
| **Edit history** | `edits`, `edit_sessions` (checkpointed state), `llm_calls` (cost meter) | query plane | resume, revision lineage, `craon status` |

Schema migrations are additive (`ALTER TABLE ... ADD COLUMN` guards on open), so old projects upgrade in place.

### 3.2 The contracts (`schemas.py`)

Three Pydantic models form the load-bearing contracts between stages:

**`EditIntent`** — output of the Intent Compiler; the *only* thing the rest of the pipeline knows about the prompt:

```
EditIntent
├── operations: [Operation]            # action: remove|keep_only|compress|reorder|stylize|…
│   └── target: SegmentTarget          # modality[], NL query, optional graph_pattern,
│                                      # negation flag, temporal anchor
│                                      #   {type: before|after|between, subject_query}
├── constraints: EditConstraints       # target_duration_s + mode(max|exact|approx),
│                                      # platform, aspect_ratio, preserve_story
├── style: EditStyle                   # pacing, cut_style(beat|match), narrative_shape,
│                                      # ordering, pacing_curve: TempoMap
├── ambiguities: [Ambiguity]           # blocking → AWAITING_USER
└── out_of_scope_reason                # honest refusal text
```

**`RetrievalResult`** — grounded segments per operation, each `Segment` carrying `evidence[]` (source channel, detail, timestamp) so every eventual cut is explainable; plus `ordered_segments` — the story agent's non-chronological ordering (None = chronological).

**`Timeline`** — the renderer contract. Deterministic, zero AI knowledge: source path, fps, `OutputSpec` (aspect, target LUFS), `video_clips[]` / `audio_clips[]` with independent in/out points (J/L cuts) and fade parameters. Serialized to `edits/vN/timeline.json` — the canonical, replayable description of the edit.

---

## 4. Ingestion plane

### 4.1 Orchestration (`ingest/orchestrator.py`)

Stages form an explicit **DAG** (`STAGE_DAG`) of `StageNode`s declaring inputs, outputs, dependencies, versions, and timeouts. The orchestrator topologically sorts and executes independent stages **in parallel** (thread pool), with:

- **Resumability** — stage status persisted in `job_stages`; completed stages with matching versions are skipped on re-run.
- **Lease protection** — a stage stuck in `running` beyond its timeout is reclaimed (crash-safe).
- **Version-based invalidation** — bumping a `StageNode.version` re-runs that stage and its dependents only.
- **Readiness levels** — each completion raises a `readiness_level` (0–4) that the query plane consults: level 1 unlocks mechanical edits (transcript+silences), level 4 unlocks narrative restructuring (story map). Prompts requiring a higher level than available get an honest "needs deeper analysis" response rather than a bad edit.
- **Scene-aligned chunking** — videos >15 min are processed in ~3-minute chunks snapped to scene boundaries, bounding memory.

### 4.2 Stage inventory

| Stage | Depends on | Technology | Outputs |
|---|---|---|---|
| `normalize` | — | FFmpeg | 720p proxy, 16 kHz mono WAV, 48 kHz FLAC, probe metadata, **frame-PTS LUT** (exact frame timestamps for quantization), A/V drift check |
| `scenes` | normalize | PySceneDetect (Content+Adaptive detectors, union + flash-cut merge; FFmpeg `select` fallback; silence-snapped pseudo-shots for static video) | shots → HSV-histogram + speaker-continuity grouping into semantic scenes; 3 keyframes/scene |
| `audio` | normalize | faster-whisper `medium` (int8, word timestamps, low-logprob beam escalation) → **wav2vec2 CTC forced alignment** re-timing every word; Silero-VAD silences; n-gram filler detection; AST audio-event classifier (laughter/applause/music); **Respiro-en breath detection** (ZCR+mel-variance heuristic fallback, coverage-flagged); ECAPA speaker embeddings → agglomerative diarization (min 1 speaker); MiniLM retake clustering (2 s–120 s window); Claude topic segmentation with **per-utterance dialogue acts** | the entire speech group |
| `audio_separation` | normalize | **Demucs htdemucs** (CPU, cached) | `vocals.wav` (clean speech), `no_vocals.wav` (clean music) |
| `beat_grid` | audio_separation | librosa `beat_track` on the *music stem* | tempo-confident beat grid (no fake downbeats — phase accuracy is not claimed) |
| `faces` | scenes | insightface `FaceAnalysis` (SCRFD detect + ArcFace `get_feat` embeddings, landmark-aligned) → ByteTrack-style Kalman tracking → HAC identity clustering → `appears_in` relations; speaker–face binding; coverage-flagged fallbacks | face tracks, person entities |
| `cut_affinity` | audio | Farneback optical flow + breaths + word timings (§6) | 10 Hz `cut_affinity` curve |
| `vision` | scenes, audio, faces | Claude vision on keyframe batches (montage grids for high-motion scenes): captions, `shot_type`, location + **indoor bit**, emotion+intensity, actions, objects, `is_broll`, person descriptions with cross-batch identity context | scene tags, `contains`/`performs`/`expresses` relations |
| `graph` | audio, vision | deterministic cross-modal derivation: entity resolution, `derived_moments` — `applause` (audio event), `funniest` (laughter ∩ joy), `awkward` (long pause after question), `off_topic` (topic-centroid dissimilarity), `montage` (broll ∩ music); contradiction logging | knowledge graph |
| `index` | graph | CLIP/SigLIP image embeddings (frame + scene), MiniLM text embeddings (scene + utterance), BM25 over transcript+captions+labels — all persisted with row maps | 3 FAISS indexes + BM25 pickle |
| `story` | index | Claude importance scoring (batched) + story-role mapping + setup/payoff dependency extraction | `story_beats`, `story_deps` |

**Model philosophy:** heavy-but-free CPU models run once at ingest; Claude is reserved for judgment tasks (tagging, segmentation, scoring) where structured output + video grounding beat any local model. Every analyzer writes a `coverage` row (`available` / `heuristic` / `fallback` / `music_only_heuristic` / `unavailable`) that downstream stages treat as a capability gate.

---

## 5. Query plane

### 5.1 Intent Compiler (`query/intent.py`)

Five-phase pipeline, `P0–P4`:

- **P0 Normalize** — collapse whitespace, strip polite prefixes ("can you please…"), empty-prompt guard, language detection, deictic resolution ("undo the last one" → concrete prior prompt).
- **P1 Routing** — genuine content *questions* (question shape AND no edit verb) route to KB Q&A, not editing. A **fast-path pattern bank** (~18 regex/exact patterns for high-frequency mechanical intents) skips the LLM entirely for "remove filler words"-class prompts — a latency cache, not a capability boundary.
- **P2 LLM compile** — everything else goes to Claude (`prompts/intent_compiler.md`) with **grounding context**: the video summary (duration, people with owner flags, locations, topics, story beats, coverage) and the editor profile. The 299-line prompt teaches goal decomposition, multi-intent splitting, temporal anchors, platform vocabulary, and duration semantics, with few-shot examples. Degenerate output (no operations, no ambiguity, no scope reason) falls back to a deterministic heuristic parser — the pipeline never silently no-ops.
- **P3 Grounding** — referenced entities resolved against the KB (unknown person → non-blocking ambiguity with candidates); duration constraints clamped to physical bounds; confidence calibration.
- **P4 Policy** — readiness-level gating (story prompts need level 4), profile merge (standing pacing/platform preferences), operation-conflict detection.

### 5.2 Hybrid Retrieval (`query/retrieval.py`)

Each operation's `SegmentTarget` fans out across channels, fused by **weighted reciprocal-rank fusion** (structured 3×, metadata 2×, keyword/vector 1×) and re-ranked by a cross-encoder:

| Channel | Answers | Backed by |
|---|---|---|
| **Structured** | silences, fillers, retakes (keeps final take), person appearances, owner resolution, audio events (laughter/applause), object `contains` relations (fuzzy label match), emotion+intensity, derived moments, dialogue acts, off-topic | KB tables & graph |
| **Temporal anchors** | "before I enter the frame", "after the unboxing" | `first_appearance()` / first channel hit → `[0,t)` / `(t,end]` range ops |
| **Keyword** | exact/fuzzy transcript & caption mentions | BM25 |
| **Vector (text)** | paraphrased content queries | MiniLM FAISS (scene + utterance) |
| **Vector (visual)** | "coffee-making shots" with no transcript mention | CLIP text→image FAISS |
| **Metadata** | B-roll, story roles, indoor/outdoor | scene columns |
| **Compound** | "X while Y" | channel intersection |

Zero-hit queries trigger one **LLM query expansion** (synonyms/related phrasings) before conceding `no_match`. Scores in the confirmation band (0.35–0.55) are marked `needs_confirmation` and surface as preview warnings rather than silent cuts. Adjacent hits merge (≤1 s gaps) before planning.

### 5.3 Story Agent (`query/story_agent.py`)

Runs only when the stage plan demands narrative reasoning (compression, restructure, trailer/highlight shapes):

1. **Budgeted selection** — 0/1 knapsack over importance-scored scenes under the duration budget (×0.95 margin), followed by pairwise-swap local search; evidence-centered truncation when a single scene must shrink.
2. **Dependency enforcement** — `story_deps` guarantees setups travel with their payoffs (bypassed for trailer mode, where teasing without payoff is the point).
3. **Ordering** — narrative templates (`hook_first`, `trailer`, `highlight`), LLM-proposed orderings (invariant-checked, revert-on-violation), and greedy CLIP boundary-frame chaining for match cuts. The result is exported as `ordered_segments` — and **the planner honors it** (§5.4), so reordering is real end-to-end.
4. **Taste pass** — one bounded Claude call may propose ≤5 scene swaps; each is validated against invariants before acceptance.

### 5.4 Timeline Planner (`query/planner.py`) — 100 % deterministic

The only component allowed to produce frame numbers. Fixed-point rewrite loop:

```
removal ranges (from ops; keep_only/compress inverted)
  → clamp to source bounds, drop degenerates, merge overlaps
  → CUT-AFFINITY SNAPPING (§6; fallback: silence-mid > word-gap > word-edge)
  → iterate to fixpoint: micro-gap merge (300 ms) ∘ min-clip drop (700 ms) ∘ re-snap
  → beat snapping (±40 ms) if beat_cut — on-demand full-audio beat extraction with
    onset-detection fallback and honest "no rhythmic structure" downgrade
  → frame-LUT quantization (exact PTS from normalize; fps-grid fallback)
  → REORDER PERMUTATION from story ordered_segments (chronological math first,
    narrative order last — snapping/merging always operate in source-time domain)
  → invariant self-repair: clamp, drop zero-length, resolve overlaps, minimum-output floor
  → hard asserts: non-empty ∘ positive durations ∘ no source-time overlap ∘ in-bounds
  → TEMPO MAP (query/rhythm.py): per-clip target shot lengths from pacing curve
    (fast≈2 s, neutral≈5 s, cinematic≈8 s), over-length clips trimmed toward
    high-affinity points
  → J/L audio offsets at scene boundaries; crossfades on non-adjacent reordered joins
  → EditPlan (ops + repairs + rule logs) + Timeline (renderer contract)
```

Every transformation appends a `Repair` or rule-log entry — the final report can explain *why* each cut moved ("moved 0.24 s → breath + motion minimum, affinity 4").

### 5.5 Critic Validator (`query/critic.py`)

Three tiers, fresh context (the critic never sees the planner's reasoning, only its output):

- **Tier 0 — structural (<10 ms):** duration-target compliance per mode (max/exact/approx), minimum-output floor, excessive-removal guard, mid-word-cut detection against word timings, and **reorder verification** — a `hook_first`/`trailer` plan whose clip order is still chronological *fails* with route `story`.
- **Tier 1 — semantic re-query (~200 ms):** re-runs structured search for every removal target and flags matching content still present in kept ranges ("did the edit actually remove the thing?").
- **Tier 2 — LLM judgment:** Claude reads the kept transcript + scene captions **in final playback order** against the intent — leftover content, missing retentions, orphaned references, broken Q→A pairs, hook placement. Objective satisfaction only; explicitly not artistic taste.

Failures carry a typed **retry route** (`retrieval` / `story` / `planner`) consumed by the executor. Anti-flapping: timeline hashes are tracked across retries; a repeated plan aborts the loop (`RunawayError`) instead of oscillating.

### 5.6 Revision engine (`query/revision.py`)

`--revise vN "prompt"` loads the parent `timeline.json`, applies delta operations (restore/remove/swap/extend/shorten), clamps all clips to probed source bounds, and re-renders **only dirty regions** — clean segments are copied from the parent's segment cache.

### 5.7 Profile learning (`query/profile.py`)

Post-delivery, a lightweight classifier extracts durable preferences from the prompt ("always remove ums" → standing instruction) into `~/.trim_engine/profile.json`, injected into every future compile. The engine converges toward *your* editor.

---

## 6. The Cut-Point Engine

*Why CRAON's cuts feel human.* Professional editors don't cut where thresholds fire — they cut on breaths, at motion settle, never mid-word. CRAON precomputes this taste as data.

**Ingest (`ingest/cut_affinity.py`, `audio.py::_run_breath_detection`):**

- **Breaths** — [Respiro-en](https://github.com/ydqmkkx/Respiro-en) frame-wise inhale detection (bundled in `trim_engine/models/respiro_en/`); plain VAD absorbs breaths into speech regions, so this is a separate detector. Heuristic ZCR+mel-variance fallback is coverage-flagged.
- **Motion** — Farneback optical flow at 10 Hz on a 320×180 proxy → motion-energy curve; local minima = "the gesture finished."
- **Word timings** — CTC-aligned word boundaries mark forbidden zones.

These fuse into a single **10 Hz scalar curve** stored in `cut_affinity(t, score)`:

```
score(t) = +3·breath  +2·blink  +1·motion-minimum  −10·mid-word  −2·mid-gesture
```

**Query:** the planner's snap function (`_snap_to_word_boundary`) first maximizes affinity within the snap window (ties broken by minimal displacement); the classic silence/word-gap ladder remains as fallback when the curve is absent. One deterministic argmax replaces a stack of special cases — and the chosen score is written into the repair log, so reports read like an editor's notes.

---

## 7. Rendering pipeline

### 7.1 Strategy planning (`renderer/plan.py`)

Per clip: **smart-copy** (keyframe-aligned boundaries, no filters — stream copy, near-instant), **boundary re-encode**, or **full re-encode**. Strategy mix decides the top-level path: parallel per-segment renders + concat demuxer, or a single `filter_complex` graph.

### 7.2 Execution ladder (`renderer/execute.py`)

Each rung is **verified before acceptance** (post-render probes); a failed rung falls through:

1. **Fast path** — hardware codec (`h264_videotoolbox`, software fallback per segment), clamped audio fades (fade duration ≤ clip/2 — degenerate filter graphs are impossible), parallel segment extraction, concat, **two-pass loudnorm** (measured-linear second pass, −14 LUFS).
2. **Simple re-encode** — single-pass `libx264` + `aac`, minimal filter graph, maximal compatibility.
3. **MoviePy** — programmatic last resort.

All rungs fail → `RenderFailError` with recovery hint. **A wrong or unedited output is never silently delivered** — there is deliberately no "copy the source" rung.

Guards up front: source existence, non-empty timeline. Smart-copy segments are size/exit-code validated (mid-GOP cuts silently produce broken output otherwise) with automatic re-encode fallthrough.

### 7.3 Post-render verification (`renderer/verify.py`)

`ffprobe` container-integrity pass, duration drift vs. plan (tolerance-gated), first-packet A/V sync offset (≤100 ms). Results + per-clip strategies land in `render_log.json`.

### 7.4 Artifacts

`output.mp4` · `timeline.json` · `report.md` (evidence-linked removal table, duration delta, critic verdict) · `render_log.json` · resynced `output.srt`/`output.vtt` (word timestamps remapped through the cut list in playback order) · `thumbnail.jpg` · stream-copied `preview.mp4` (+ cut-inspection mode rendering ±1.5 s around each cut).

---

## 8. LLM integration layer

One wrapper (`llm.py`) serves every agent; prompts are versioned files in `prompts/` (one per agent: `intent_compiler`, `vision_tagger`, `topic_segmenter`, `importance_scorer`, `story_mapper`, `critic`, `query_expansion`, `profile_classifier`, `kb_answer`).

| Concern | Mechanism |
|---|---|
| Structured output | Pydantic schema validation; defensive JSON extraction (fences, prose-wrapped, balanced-brace scan, trailing-comma repair); on validation failure → **self-repair retry** with the error fed back (≤2) |
| Transient failures | typed classification (SDK exception types + status codes) → exponential backoff with jitter (≤3), separate from the validation budget; exhaustion → `LLMTransientError` with recovery hint |
| Truncation | `stop_reason == max_tokens` → double budget and retry (≤4× cap) without consuming a validation attempt |
| Prompt injection | user content fenced in `<user_input>` delimiters with an explicit no-override instruction |
| Cost | per-call token/cache/latency/USD metering → `llm_calls` table; session-level `BudgetEnvelope` (§10) |
| Caching | system prompts carry `cache_control: ephemeral` breakpoints; the video summary is frozen (no interpolated timestamps) to maximize cache hits |
| Vision | base64 keyframe batches with cross-batch "known people" context for identity continuity |

Claude appears at exactly six decision points — intent compile, vision tagging, topic segmentation, importance/story mapping, story-agent taste pass, critic Tier 2 — plus Q&A and query expansion. Everything between those points is deterministic Python.

---

## 9. Failure taxonomy & honesty layer

### 9.1 Typed exceptions (`query/exceptions.py`)

`QueryEngineError(message, recovery_hint)` subclassed into: `LLMTransientError`, `SemanticError`, `RetrievalGapError`, `InfeasibleError`, `PlannerBreachError`, `RenderFailError`, `StaleKBError`, `RunawayError`. The CLI's central error boundary (`cli.py::handle_engine_errors`) renders every one as a titled panel: what failed, why, and **what to do** — never a raw traceback.

### 9.2 Honesty mechanisms

| Mechanism | Behavior |
|---|---|
| **No-match → no-op** | all operations `no_match` → `RESOLVED_NOOP` with per-op explanation; never a passthrough render with a green checkmark |
| **Coverage gating** | every analyzer writes `coverage`; degraded signals (`heuristic`, `fallback`, `music_only_heuristic`) bound what the engine claims — beat-cut on non-music footage says so and downgrades to fast-cut pacing |
| **Readiness gating** | story-level prompts before story analysis completes → honest "needs ~40 s of reprocessing" |
| **Ambiguity surfacing** | blocking ambiguities halt at `AWAITING_USER` with candidates; sub-threshold retrieval marks `needs_confirmation` |
| **Evidence chain** | every removal in `report.md` links to its source signal (channel, detail, timestamp) |
| **Verdict integrity** | the suite runner skips rendering on failed verdicts; nothing failed ships as ✓ |
| **Graceful degradation ≠ silent degradation** | preview/report/profile failures degrade to a note post-render; nothing non-essential can fail a successful edit, and nothing essential can succeed silently |

---

## 10. Session state machine & budgets

### 10.1 States (`query/executor.py`)

```
CREATED → COMPILING → RETRIEVING → [REASONING] → PLANNING → VALIDATING
   → PREVIEW_READY → RENDERING → DELIVERED
                                                   ↘ critic fail: typed retry route
terminal: DELIVERED · RESOLVED_NOOP · AWAITING_USER · RENDER_FAILED
```

The **stage plan is compiled per intent class**: mechanical edits skip REASONING and VALIDATING's LLM tier; narrative edits run the full loop; Q&A and out-of-scope resolve immediately. Critic failures route backward by failure type; a requested stage absent from the plan falls back to PLANNING.

Every transition is **checkpointed** to `edit_sessions` — a crash resumes exactly where it stopped; new sessions supersede stale actives. Cache: prompt+video keyed session restore, gated on the cached timeline still being renderable (source exists, clips non-empty).

### 10.2 Budget envelope

Per session: ≤8 LLM calls · ≤4 retries · ≤300 s pre-render wall clock · ≤$0.50 — any breach raises `RunawayError` with a "break your request into smaller edits" hint. Plus: timeline-hash flap detection and a 50-transition state-machine backstop. **The pipeline cannot spin, spend, or oscillate unboundedly.**

---

## 11. Interfaces

| Surface | Entry | Notes |
|---|---|---|
| **CLI** | `craon ingest / edit / ask / status / suite / bedrock-smoke` | primary surface; central error boundary; `--yes` auto-approve; `--revise vN` |
| **Interactive shell** | `craon` (no args) | prompt-toolkit REPL with command/video-id completion; delegates to CLI functions (inherits error handling) |
| **HTTP API** | `trim_engine.api` (FastAPI) | `POST /ingest` (upload), `POST /edit/{video_id}`, `GET /status/{id}`, `GET /edits/{id}`, artifact downloads |
| **Suite** | `craon suite <video_id>` | 25-prompt regression matrix → `sample_outputs/` with per-prompt intent/plan/verdict/report + index |

---

## 12. Configuration & extensibility

**All tunables live in `config.py`** — frozen dataclasses per subsystem (`LLMConfig`, `AudioConfig`, `PlannerConfig`, `RendererConfig`, …), env-overridable where operational (`BEDROCK_MODEL_ID`, `AWS_REGION`), `.env` auto-loaded. No magic numbers elsewhere.

**Platform presets** (`PLATFORM_TEMPLATES`): TikTok / Reels / Shorts / YouTube — aspect, duration caps, hook windows, cut-rate hints. Adding a platform is one dataclass instance.

**Extension points, by design:**

| To add… | Touch |
|---|---|
| a new perception signal | one ingest `StageNode` + table + coverage flag |
| a new retrieval capability | one channel branch in `_structured_search` (auto-fused by RRF) |
| a new narrative shape | one template in the story agent (ordering is already honored downstream) |
| a new platform | one `PlatformTemplate` |
| a new agent | one prompt file + one `call_structured` call site |

The trim feature is app #1 on this substrate; highlight reels, trailers, and platform derivatives already reuse the same KB, retrieval, and renderer — that is the platform claim, made concrete.

---

## 13. Testing strategy

- **Unit (offline, `-m "not llm"`)** — planner invariants (clamping, overlap repair, reorder permutation), renderer filter construction (fade clamping), JSON extraction, intent fast paths and routing.
- **Golden e2e (`-m llm`)** — the assignment prompt matrix against an ingested fixture, asserting *outcome properties*: duration ≤ target, ground-truth spans absent from kept ranges, non-chronological order when the shape demands it, honest no-ops on no-match, `verdict.passed` on every delivered edit.
- **Runtime probes** — `bedrock-smoke` (structured + vision + cost-meter assertion), post-render verification on every real edit, `model_manifest` recording which perception models actually loaded.
