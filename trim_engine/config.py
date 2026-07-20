"""
Central configuration — ALL tunables live here, nowhere else.

Every threshold, model name, path template, and platform preset is defined
once and imported by the engine that needs it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import platformdirs






def _load_dotenv() -> None:
    """Parse PROJECT_ROOT/.env into os.environ if the file exists.

    Only sets keys that are NOT already in the environment so that
    explicit exports always win.  Handles KEY="value" and KEY=value.
    """
    # Check standard install path first
    env_path = Path(__file__).resolve().parent.parent / ".env"
    
    # Check current working directory (important when installed globally via uv tool)
    if not env_path.exists():
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            env_path = cwd_env
            
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

_load_dotenv()


_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not Path(_ssl_cert).exists():
    del os.environ["SSL_CERT_FILE"]






PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
PROJECTS_DIR = Path(platformdirs.user_data_dir("ai-trim-engine")) / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_PATH = Path.home() / ".trim_engine" / "profile.json"






@dataclass(frozen=True)
class LLMConfig:
    model_id: str = field(default_factory=lambda: os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6"))
    aws_region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))
    max_tokens_default: int = 8192
    max_tokens_large: int = 16384
    stream_threshold: int = 16000  
    max_retries_sdk: int = 2
    max_retries_validation: int = 2
    
    effort_low: str = "low"
    effort_medium: str = "medium"
    effort_high: str = "high"






@dataclass(frozen=True)
class NormalizeConfig:
    proxy_height: int = 720
    proxy_fps: int = 30
    proxy_bitrate: str = "2M"
    proxy_codec_hw: str = "libx264"
    proxy_codec_sw: str = "libx264"  
    proxy_gop_s: int = 1  
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    audio_highpass_hz: int = 10  
    max_drift_ms: float = 50.0  
    av_offset_threshold_ms: float = 40.0  
    build_frame_lut: bool = True  






@dataclass(frozen=True)
class SceneConfig:
    detector_threshold: float = 27.0
    min_scene_duration_s: float = 0.5  
    max_scene_duration_s: float = 60.0  
    keyframes_per_scene: int = 3
    keyframe_positions: tuple[float, ...] = (0.10, 0.50, 0.90)
    keyframe_quality: int = 85  
    motion_fps: int = 1  
    pseudo_scene_interval_s: float = 20.0  
    min_scenes_threshold: int = 3  






@dataclass(frozen=True)
class AudioConfig:
    
    whisper_model_size: str = "medium"  
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"
    whisper_beam_escalation_threshold: float = -0.8  
    whisper_beam_escalation_size: int = 5
    language_confidence_threshold: float = 0.6

    
    ctc_alignment_model: str = "facebook/wav2vec2-base-960h"
    ctc_multilingual_model: str = "facebook/wav2vec2-large-xlsr-53"
    ctc_language_adaptive: bool = True  

    
    min_silence_duration_s: float = 0.3  
    vad_threshold: float = 0.5

    
    chunk_overlap_s: float = 8.0  

    
    filler_words_always: tuple[str, ...] = ("um", "uh", "hmm", "erm", "mmm")
    filler_words_contextual: tuple[str, ...] = ("like", "you know", "so", "actually")

    
    max_speakers: int = 4
    min_speakers: int = 1

    
    spectral_flatness_threshold: float = 0.15  

    
    loudness_window_ms: int = 200  

    
    retake_similarity_threshold: float = 0.87
    retake_min_gap_s: float = 5.0
    retake_max_gap_s: float = 120.0    # reject pairs farther than 2 min apart
    retake_duration_ratio_threshold: float = 0.5  






@dataclass(frozen=True)
class VisionConfig:
    batch_size: int = 6  
    keyframes_per_scene: int = 3  
    max_retries_per_batch: int = 1
    person_merge_cosine_threshold: float = 0.8
    person_merge_temporal_bonus: float = 0.1






@dataclass(frozen=True)
class EmbeddingConfig:
    clip_model_name: str = "ViT-B-16-SigLIP"
    clip_pretrained: str = "webli"
    clip_dim: int = 768
    text_model_name: str = "all-MiniLM-L6-v2"
    text_dim: int = 384






@dataclass(frozen=True)
class FaceConfig:
    """Face detection, recognition, tracking, and privacy settings."""
    
    detector_model: str = "buffalo_m"  
    detection_confidence: float = 0.5
    min_face_size: int = 30  

    
    recog_model: str = "w600k_r50"  
    embedding_dim: int = 512

    
    track_high_thresh: float = 0.6  
    track_low_thresh: float = 0.1   
    track_iou_threshold: float = 0.5
    track_max_age: int = 30   
    track_min_hits: int = 3   
    track_frame_sample_rate: int = 15  

    
    cluster_distance_threshold: float = 0.5  
    min_cluster_size: int = 2  

    
    cross_video_similarity: float = 0.65  

    
    store_embeddings: bool = True
    enable_blur: bool = False  

    
    mouth_motion_window_frames: int = 3  
    speaking_confidence_mouth_weight: float = 0.6
    speaking_confidence_presence_weight: float = 0.4


@dataclass(frozen=True)
class StoryConfig:
    importance_batch_size: int = 15
    importance_weights: tuple[float, float, float] = (0.70, 0.15, 0.15)  
    story_roles: tuple[str, ...] = (
        "hook", "intro", "setup", "development", "climax",
        "reveal", "payoff", "outro", "filler",
    )






@dataclass(frozen=True)
class RetrievalConfig:
    fusion_weights: dict[str, float] = field(default_factory=lambda: {
        "structured": 3.0,
        "keyword": 1.0,
        "vector": 1.0,
        "metadata": 2.0,
    })
    score_threshold: float = 0.35
    confirmation_band: tuple[float, float] = (0.35, 0.55)
    top_k_per_channel: int = 20
    nearest_neighbor_suggestions: int = 3






@dataclass(frozen=True)
class PlannerConfig:
    
    word_snap_search_ms: float = 400.0  
    word_gap_min_ms: float = 120.0  

    
    silence_search_ms: float = 700.0  
    silence_min_ms: float = 300.0  

    
    micro_gap_max_ms: float = 300.0  

    
    min_clip_ms: float = 700.0  

    
    crossfade_ms: float = 80.0  

    
    jl_cut_scene_boundary_ms: float = 500.0
    jl_cut_offset_max_ms: float = 80.0
    reorder_crossfade_ms: float = 120.0   # §2.2: crossfade for non-adjacent reordered joins

    
    beat_snap_ms: float = 40.0  

    
    duration_tolerance_approx: float = 0.05  
    max_removal_ratio_guard: float = 0.90  
    min_output_duration: float = 1.0  
    min_output_floor_ratio: float = 0.05  






@dataclass(frozen=True)
class CriticConfig:
    max_retries: int = 2
    duration_tolerance: float = 0.05  






@dataclass(frozen=True)
class StoryAgentConfig:
    removal_threshold_for_story: float = 0.30  
    knapsack_budget_margin: float = 0.95  
    max_scene_swaps: int = 5  
    match_cut_min_similarity: float = 0.72  






@dataclass(frozen=True)
class RendererConfig:
    codec_hw: str = "libx264"
    codec_sw: str = "libx264"
    concat_demuxer_threshold: int = 0
    loudnorm_i: float = -14.0
    loudnorm_tp: float = -1.5
    duration_tolerance: float = 0.02  
    av_sync_tolerance_ms: float = 100.0






@dataclass(frozen=True)
class PlatformTemplate:
    name: str
    max_duration_s: float | None = None
    min_duration_s: float | None = None
    hook_max_s: float | None = None
    aspect_ratio: str = "16:9"
    min_cut_rate: float | None = None  
    chapters: bool = False


PLATFORM_TEMPLATES: dict[str, PlatformTemplate] = {
    "tiktok": PlatformTemplate(
        name="tiktok", max_duration_s=60, hook_max_s=2.0,
        aspect_ratio="9:16", min_cut_rate=8.0,
    ),
    "reels": PlatformTemplate(
        name="reels", max_duration_s=90,
        aspect_ratio="9:16",
    ),
    "youtube": PlatformTemplate(
        name="youtube", min_duration_s=120,
        aspect_ratio="16:9", chapters=True,
    ),
    "youtube_shorts": PlatformTemplate(
        name="youtube_shorts", max_duration_s=60,
        aspect_ratio="9:16", hook_max_s=2.0,
    ),
}






@dataclass(frozen=True)
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    story: StoryConfig = field(default_factory=StoryConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    story_agent: StoryAgentConfig = field(default_factory=StoryAgentConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)


CFG = Config()
