"""
Revision Engine (§10) — conversational delta timelines, lineage tree, and region-only rendering.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from trim_engine.config import CFG
from trim_engine.schemas import Timeline, VideoClip, AudioClip, Transition

class RevisionNode:
    def __init__(self, version: int, prompt: str, parent_version: int | None = None):
        self.version = version
        self.prompt = prompt
        self.parent_version = parent_version
        self.children: list[RevisionNode] = []

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "prompt": self.prompt,
            "parent_version": self.parent_version,
            "children": [c.to_dict() for c in self.children]
        }

def compile_revision_timeline(
    parent_timeline: Timeline,
    prompt: str,
) -> tuple[Timeline, list[int]]:
    """
    Applies revision delta instructions (restore, also_remove, swap_order, adjust_duration)
    to a parent timeline. Returns the updated Timeline and the indices of modified (dirty) clips.
    """
    prompt_clean = prompt.strip().lower()
    
    video_clips = list(parent_timeline.video_clips)
    audio_clips = list(parent_timeline.audio_clips)
    
    dirty_indices = []

    import re

    
    if re.search(r"\b(remove|delete)\s+(the\s+)?first\s+clip\b", prompt_clean):
        if video_clips:
            video_clips.pop(0)
            audio_clips.pop(0)
            dirty_indices = list(range(len(video_clips))) 

    
    elif re.search(r"\b(restore|keep|put\s+back)\b.*\b(first\s+clip|intro)\b", prompt_clean):
        
        
        source_path = Path(parent_timeline.source)
        intro_end = 5.0  
        
        try:
            import subprocess, json as _json
            probe = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", str(source_path)],
                capture_output=True, text=True, timeout=10
            )
            total_dur = float(_json.loads(probe.stdout)["format"]["duration"])
            
            if parent_timeline.video_clips:
                intro_end = parent_timeline.video_clips[0].src_in
                if intro_end <= 0.1:
                    
                    intro_end = min(5.0, total_dur * 0.1)
        except Exception:
            pass
        intro_end = max(intro_end, 1.0)  
        v_clip = VideoClip(src_in=0.0, src_out=intro_end, transition_out=Transition(type="cut"))
        a_clip = AudioClip(src_in=0.0, src_out=intro_end, fade_in_ms=0, fade_out_ms=0)
        video_clips.insert(0, v_clip)
        audio_clips.insert(0, a_clip)
        dirty_indices = [0] + list(range(1, len(video_clips)))

    
    elif re.search(r"\b(remove|delete|cut)\b.*\b(pricing|cost|rates)\b", prompt_clean):
        
        
        
        from rich.console import Console as _Console
        _Console().print("    [yellow]⚠ Semantic content removal in revision mode requires a full re-edit. "
                         "Run `trim edit` with this prompt instead of `--revise`.[/yellow]")

    
    elif re.search(r"\b(swap|reorder)\b", prompt_clean):
        if len(video_clips) >= 2:
            video_clips[0], video_clips[1] = video_clips[1], video_clips[0]
            audio_clips[0], audio_clips[1] = audio_clips[1], audio_clips[0]
            dirty_indices = [0, 1]

    
    elif re.search(r"\b(shorter|shrink)\b", prompt_clean):
        if video_clips:
            orig = video_clips[0]
            video_clips[0] = VideoClip(src_in=orig.src_in, src_out=max(orig.src_in + 1.0, orig.src_out - 2.0))
            orig_a = audio_clips[0]
            audio_clips[0] = AudioClip(src_in=orig_a.src_in, src_out=max(orig_a.src_in + 1.0, orig_a.src_out - 2.0))
            dirty_indices = [0]

    
    elif m := re.search(r"\b(longer|extend)\b.*\b(\d+)\s*s", prompt_clean):
        if video_clips:
            add_s = float(m.group(2))
            orig = video_clips[0]
            video_clips[0] = VideoClip(src_in=orig.src_in, src_out=orig.src_out + add_s)
            orig_a = audio_clips[0]
            audio_clips[0] = AudioClip(src_in=orig_a.src_in, src_out=orig_a.src_out + add_s)
            dirty_indices = [0]
    elif re.search(r"\b(longer|extend)\b", prompt_clean):
        if video_clips:
            orig = video_clips[0]
            video_clips[0] = VideoClip(src_in=orig.src_in, src_out=orig.src_out + 2.0)
            orig_a = audio_clips[0]
            audio_clips[0] = AudioClip(src_in=orig_a.src_in, src_out=orig_a.src_out + 2.0)
            dirty_indices = [0]

    # Clamp all clips to source bounds so "extend"/"longer" deltas can never
    # request frames past the end of the video (which would fail the render).
    total_dur = None
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "json", str(parent_timeline.source)],
            capture_output=True, text=True, timeout=10,
        )
        total_dur = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        pass

    if total_dur:
        for lst in (video_clips, audio_clips):
            for idx, clip in enumerate(lst):
                s = max(0.0, min(clip.src_in, total_dur))
                e = max(s, min(clip.src_out, total_dur))
                if (s, e) != (clip.src_in, clip.src_out):
                    lst[idx] = type(clip)(**{**clip.model_dump(), "src_in": s, "src_out": e})

    # Drop any degenerate (zero/negative-length) clips left after clamping.
    kept = [
        (vc, ac) for vc, ac in zip(video_clips, audio_clips)
        if vc.src_out - vc.src_in > 0.01
    ]
    if kept:
        video_clips, audio_clips = [list(x) for x in zip(*kept)]

    new_timeline = Timeline(
        version=parent_timeline.version + 1,
        source=parent_timeline.source,
        fps=parent_timeline.fps,
        output=parent_timeline.output,
        video_clips=video_clips,
        audio_clips=audio_clips,
    )

    return new_timeline, dirty_indices

def render_dirty_regions_only(
    timeline: Timeline,
    parent_timeline: Timeline,
    dirty_indices: list[int],
    output_dir: Path,
) -> Path:
    """
    Region-only rendering optimization (§10.3).
    Only re-encodes dirty indices; copies pre-rendered segments for clean indices.
    """
    
    
    
    parent_segments_dir = output_dir.parent / f"v{parent_timeline.version}" / "segments"
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True, parents=True)

    source = Path(timeline.source)
    cfg = CFG.renderer
    segment_paths = []

    for i, (vc, ac) in enumerate(zip(timeline.video_clips, timeline.audio_clips)):
        seg_path = segments_dir / f"seg_{i:04d}.mp4"
        parent_seg = parent_segments_dir / f"seg_{i:04d}.mp4"

        if i not in dirty_indices and parent_seg.exists():
            
            import shutil
            shutil.copy(parent_seg, seg_path)
        else:
            
            vfilter = f"trim={vc.src_in}:{vc.src_out},setpts=PTS-STARTPTS"
            afilter = f"atrim={ac.src_in}:{ac.src_out},asetpts=PTS-STARTPTS"
            cmd = [
                "ffmpeg", "-y",
                "-i", str(source),
                "-vf", vfilter,
                "-af", afilter,
                "-c:v", cfg.codec_sw,
                "-crf", "23", "-preset", "fast",
                str(seg_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        segment_paths.append(seg_path)

    
    concat_list = output_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg}'\n")

    output_path = output_dir / "output.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
