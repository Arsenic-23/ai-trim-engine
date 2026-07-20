"""
Pydantic models — the canonical data contracts for every stage.

All schemas use strict mode (additionalProperties: false, all fields required)
for structured-output compatibility with Claude Sonnet 4.6.

Rules from §3: no minLength/minimum in JSON schema (validated client-side),
no recursive schemas, strict additionalProperties: false everywhere.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field






class Action(str, Enum):
    REMOVE = "remove"
    KEEP_ONLY = "keep_only"
    COMPRESS = "compress"
    RESTRUCTURE = "restructure"
    STYLIZE = "stylize"


class Priority(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class DurationMode(str, Enum):
    MAX = "max"
    EXACT = "exact"
    APPROX = "approx"


class Pacing(str, Enum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    CINEMATIC = "cinematic"


class CutStyle(str, Enum):
    MATCH_CUT = "match_cut"
    BEAT_CUT = "beat_cut"
    JUMP_CUT_OK = "jump_cut_ok"


class NarrativeShape(str, Enum):
    HOOK_FIRST = "hook_first"
    TRAILER = "trailer"
    HIGHLIGHT = "highlight"
    STORY_ARC = "story_arc"


class RetryRoute(str, Enum):
    RETRIEVAL = "retrieval"
    STORY = "story"
    PLANNER = "planner"


class ShotType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSEUP = "closeup"
    POV = "pov"
    SCREEN_RECORDING = "screen_recording"


class CameraMotion(str, Enum):
    STATIC = "static"
    PAN = "pan"
    ZOOM = "zoom"
    HANDHELD = "handheld"
    TRACKING = "tracking"


class TopicClass(str, Enum):
    INTRO = "intro"
    PRODUCT = "product"
    PRICING = "pricing"
    SPONSOR = "sponsor"
    JOKE = "joke"
    STORY = "story"
    OFFTOPIC = "offtopic"
    OTHER = "other"


class DialogueAct(str, Enum):
    QUESTION = "question"
    ANSWER = "answer"
    STATEMENT = "statement"


class StoryRole(str, Enum):
    HOOK = "hook"
    INTRO = "intro"
    SETUP = "setup"
    DEVELOPMENT = "development"
    CLIMAX = "climax"
    REVEAL = "reveal"
    PAYOFF = "payoff"
    OUTRO = "outro"
    FILLER = "filler"


class EntityKind(str, Enum):
    PERSON = "person"
    OBJECT = "object"
    LOCATION = "location"
    ACTION = "action"
    TOPIC = "topic"






class VideoMeta(BaseModel):
    """Probe results for a video file."""
    path: str
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    is_vfr: bool = False
    content_class: str = "standard"  






class PersonTag(BaseModel):
    key: str = Field(description="Identifier like 'person_1'")
    description: str = Field(description="Appearance description: 'man, 30s, blue hoodie, glasses'")
    is_speaking: bool
    prominence: float = Field(description="0.0-1.0 screen presence")


class ObjectTag(BaseModel):
    label: str
    prominence: float = Field(description="0.0-1.0 visual prominence")


class LocationTag(BaseModel):
    label: str
    indoor: bool


class EmotionTag(BaseModel):
    label: str
    intensity: float = Field(description="0.0-1.0 intensity")


class BBoxHint(BaseModel):
    label: str = Field(description="object/person label")
    region: str = Field(description="left|center|right")
    size: str = Field(description="small|medium|large")


class SceneTags(BaseModel):
    """Output of one scene from the Claude vision tagger (§4.5)."""
    scene_id: int
    caption: str
    people: list[PersonTag]
    objects: list[ObjectTag]
    actions: list[str]
    location: LocationTag
    shot_type: str  
    camera_motion: str  
    emotion: EmotionTag
    is_broll: bool
    visible_text: list[str]
    bbox_hints: list[BBoxHint]


class VisionBatchResponse(BaseModel):
    """Structured output for a batch of scenes from Claude vision."""
    scenes: list[SceneTags]






class TopicSegment(BaseModel):
    """One topic segment from Claude topic segmentation."""
    utterance_ids: list[int]
    topic_label: str
    topic_class: str  
    dialogue_acts: dict[int, str] = Field(
        default_factory=dict,
        description="Map of utterance_id to dialogue act (e.g., question, assertion, exclamation, filler)"
    )

class TopicSegmentationResponse(BaseModel):
    """Structured output from the topic segmentation Claude call."""
    segments: list[TopicSegment]






class StoryBeat(BaseModel):
    scene_ids: list[int]
    role: str  
    summary: str


class StoryDependency(BaseModel):
    setup_scene: int
    payoff_scene: int
    why: str


class HookCandidate(BaseModel):
    scene_id: int
    hook_score: float
    why: str


class StoryMapResponse(BaseModel):
    """Structured output from the story mapper Claude call."""
    beats: list[StoryBeat]
    dependencies: list[StoryDependency]
    hook_candidates: list[HookCandidate]
    payoff_candidates: list[HookCandidate]


class SceneImportance(BaseModel):
    scene_id: int
    importance: float = Field(description="0.0-1.0 raw LLM score")
    justification: str


class ImportanceBatchResponse(BaseModel):
    """Structured output from the importance scorer Claude call."""
    scores: list[SceneImportance]






class TemporalAnchor(BaseModel):
    type: str = Field(description="'before', 'after', 'between', or 'absolute'")
    subject_query: str | None = Field(default=None, description="Subject query for relative anchors (before, after, between).")
    start_s: float | None = Field(default=None, description="Start timestamp in seconds for absolute anchors.")
    end_s: float | None = Field(default=None, description="End timestamp in seconds for absolute anchors.")

class SegmentTarget(BaseModel):
    """What to search for — resolved by the retrieval engine, not by the LLM."""
    modality: list[str] = Field(
        description="Which signals: visual, speech, audio, emotion, story, person, object, action, location"
    )
    query: str = Field(description="Natural-language segment query")
    graph_pattern: str | None = Field(
        default=None, description="Optional explicit graph traversal pattern"
    )
    negation: bool = Field(default=False, description="True if query is negated (everything NOT matching)")
    anchor: TemporalAnchor | None = Field(
        default=None,
        description="Use this to specify absolute timestamps (type='absolute', start_s, end_s) or relative anchors."
    )


class Operation(BaseModel):
    """A single edit operation within an intent."""
    action: str  
    target: SegmentTarget
    priority: str  
    confidence: float = Field(description="0.0-1.0 compiler confidence")


class EditConstraints(BaseModel):
    target_duration_s: float | None = None
    duration_mode: str | None = None  
    platform: str | None = None
    aspect_ratio: str | None = None
    preserve_story: bool = True
    preserve_speech_integrity: bool = True


class TempoPhase(str, Enum):
    SLOW = "slow"
    MEDIUM = "medium"
    FAST = "fast"
    FLAT_FAST = "flat_fast"

class TempoMap(BaseModel):
    start: TempoPhase | None = None
    build: TempoPhase | None = None
    climax: TempoPhase | None = None
    outro: TempoPhase | None = None

class EditStyle(BaseModel):
    pacing: str | None = None  
    pacing_curve: TempoMap | None = None
    cut_style: str | None = None  
    narrative_shape: str | None = None  
    ordering: str | None = None  


class Ambiguity(BaseModel):
    issue: str
    candidates: list[str]
    blocking: bool


class EditIntent(BaseModel):
    """
    The structured output of the Intent Compiler (§5.1).
    This is THE contract between intent compilation and the rest of the pipeline.
    """
    intent_id: str
    operations: list[Operation]
    constraints: EditConstraints
    style: EditStyle
    ambiguities: list[Ambiguity]
    profile_applied: list[str]
    out_of_scope_reason: str | None = None






class Evidence(BaseModel):
    """A single piece of evidence supporting a retrieval match."""
    source: str  
    detail: str
    t: float | None = None  


class Segment(BaseModel):
    """A candidate segment returned by the retrieval engine."""
    start: float
    end: float
    scene_ids: list[int]
    score: float
    evidence: list[Evidence]
    needs_confirmation: bool = False


class RetrievalResult(BaseModel):
    """Full result for one operation's retrieval."""
    operation_index: int
    segments: list[Segment]
    no_match: bool = False
    suggestions: list[str] | None = None  
    ordered_segments: list[Segment] | None = None  # §2.1: non-chronological ordering from story agent






class Repair(BaseModel):
    """A repair applied at a cut point."""
    type: str  
    detail: str


class PlanOperation(BaseModel):
    """A single operation in the edit plan."""
    op_id: str
    type: str  
    range_start: float
    range_end: float
    reason: str
    evidence_ref: str | None = None
    repairs: list[Repair]
    depends_on: list[str]


class EditPlan(BaseModel):
    """The full edit plan — output of the Timeline Planner (§5.4)."""
    plan_id: str
    operations: list[PlanOperation]
    predicted_output_duration_s: float
    removal_ratio: float
    clip_count: int
    rule_logs: list[dict] | None = None






class Transition(BaseModel):
    type: str = "cut"  # cut | crossfade | match_cut | punch_in
    ms: int = 0


class VideoClip(BaseModel):
    src_in: float
    src_out: float
    transition_out: Transition = Field(default_factory=lambda: Transition(type="cut"))
    reason: str | None = None
    evidence_ref: str | None = None


class AudioClip(BaseModel):
    src_in: float
    src_out: float
    fade_in_ms: int = 0
    fade_out_ms: int = 0


class OutputSpec(BaseModel):
    aspect: str = "16:9"
    target_lufs: float = -14.0


class Timeline(BaseModel):
    """The renderer contract — deterministic, zero AI knowledge."""
    version: int
    source: str  
    fps: float
    output: OutputSpec
    video_clips: list[VideoClip]
    audio_clips: list[AudioClip]
    captions: str = "auto_resync"
    provenance: list[PlanOperation] = Field(default_factory=list)






class StorySwap(BaseModel):
    """A proposed swap by the Story Agent."""
    remove_scene_id: int
    add_scene_id: int
    reason: str


class StoryAgentResponse(BaseModel):
    """Structured output from the Story Agent Claude call."""
    swaps: list[StorySwap]
    ordering: list[int] | None = None  
    reasoning: str






class CriticFailure(BaseModel):
    """A single failure found by the critic."""
    operation_index: int
    issue: str
    leftover_segments: list[str] | None = None
    route: str  


class CriticVerdict(BaseModel):
    """Structured output from the Critic (§5.5)."""
    passed: bool
    failures: list[CriticFailure]
    coherence_ok: bool
    notes: str | None = None






class RemovalRecord(BaseModel):
    """One removal in the edit report."""
    start: float
    end: float
    reason: str
    evidence_quote: str | None = None


class CutQualityMetrics(BaseModel):
    tempo_curve_adherence_pct: float = 100.0
    lufs_target_achieved: bool = True
    av_sync_offset_ms: float = 0.0
    cuts_on_breath_or_silence_pct: float = 100.0


class EditReport(BaseModel):
    """The full edit report — accompanies every output."""
    video_id: str
    version: int
    prompt: str
    duration_before_s: float
    duration_after_s: float
    reduction_pct: float
    removals: list[RemovalRecord]
    unsatisfied_ops: list[str]
    continuity_warnings: list[str]
    profile_preferences_applied: list[str]
    critic_verdict_summary: str
    cost_usd: float
    quality_metrics: CutQualityMetrics | None = None






class PreferenceEvidence(BaseModel):
    pref: str
    learned_from: str
    count: int


class EditorProfile(BaseModel):
    """Standing preferences for the editor."""
    version: int = 1
    pacing: str = "medium"
    platforms: list[str] = Field(default_factory=list)
    pause_tolerance_s: float = 1.2
    always: list[str] = Field(default_factory=list)
    never: list[str] = Field(default_factory=list)
    evidence: list[PreferenceEvidence] = Field(default_factory=list)






class StageStatus(BaseModel):
    stage: str
    status: str  
    duration_s: float | None = None
    error: str | None = None


class JobStatus(BaseModel):
    video_id: str
    stages: list[StageStatus]
    coverage: dict[str, str] = Field(default_factory=dict)






class PersonSummary(BaseModel):
    key: str
    description: str
    is_owner: bool
    speaking_time_s: float


class VideoSummary(BaseModel):
    """Compact summary of a video's knowledge base — injected into LLM prompts."""
    video_id: str
    duration_s: float
    scene_count: int
    people: list[PersonSummary]
    locations: list[str]
    topics: list[str]
    story_beats: list[str]
    coverage_flags: dict[str, str]






class LLMCallRecord(BaseModel):
    """Record of a single LLM API call for cost tracking."""
    timestamp: datetime
    prompt_name: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    latency_ms: float
    cost_usd: float
