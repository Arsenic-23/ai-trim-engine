"""
Render Execution Planning (§8.1) — selects Smart-copy, Boundary re-encode, or Full re-encode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from trim_engine.db import ProjectDB
from trim_engine.schemas import Timeline, VideoClip

def plan_render_strategy(
    timeline: Timeline,
    db: ProjectDB,
) -> list[dict[str, Any]]:
    """
    Selects rendering strategy per clip:
    - Smart-copy: clip starts and ends on keyframe boundaries and no filters apply.
    - Boundary re-encode: clip starts mid-GOP but interior is copyable.
    - Full re-encode: default fallback.
    """
    scenes = db.get_scenes()
    keyframes = db.get_all_keyframes()
    
    
    kf_times = {kf["position"] for kf in keyframes}
    for s in scenes:
        kf_times.add(s["start_time"])
        kf_times.add(s["end_time"])

    strategies = []
    
    for idx, clip in enumerate(timeline.video_clips):
        # Force full-reencode to guarantee frame-accurate cuts and eliminate GOP keyframe boundary freezing
        strategy = "full-reencode"
            
        strategies.append({
            "clip_index": idx,
            "src_in": clip.src_in,
            "src_out": clip.src_out,
            "strategy": strategy,
        })
        
    return strategies
