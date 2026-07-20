"""
Story & Importance Engine (§4.8) — story map + importance scores.

Depends on: everything
Two Claude calls:
(a) Story map — beats, dependencies, hook/payoff candidates
(b) Importance scores — batched 15 scenes/call, composite weighting
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.llm import call_structured
from trim_engine.schemas import StoryMapResponse, ImportanceBatchResponse

console = Console()


def _build_scene_summary(scenes: list[dict], utterances: list[dict]) -> str:
    """Build ordered scene list for the story mapper."""
    lines = []
    for scene in scenes:
        
        scene_utts = [
            u for u in utterances
            if u["end_time"] > scene["start_time"] and u["start_time"] < scene["end_time"]
        ]
        transcript_snippet = " ".join(u["text"] for u in scene_utts)[:200]

        lines.append(
            f"Scene {scene['id']} [{scene['start_time']:.1f}s–{scene['end_time']:.1f}s] "
            f"({scene['end_time'] - scene['start_time']:.1f}s): "
            f"caption=\"{scene.get('caption', 'N/A')}\" | "
            f"location={scene.get('location', 'unknown')} | "
            f"emotion={scene.get('emotion_label', 'neutral')} | "
            f"transcript=\"{transcript_snippet}\""
        )

    return "\n".join(lines)


def _run_story_map(scenes: list[dict], utterances: list[dict], db: ProjectDB) -> None:
    """Extract story structure via Claude."""
    console.print("    [dim]Extracting story map...[/dim]")

    scene_summary = _build_scene_summary(scenes, utterances)

    user_content = (
        f"Analyze the narrative structure of this video. "
        f"There are {len(scenes)} scenes:\n\n{scene_summary}\n\n"
        f"Identify story beats, dependencies between scenes, and candidates for "
        f"hooks (strong opening moments) and payoffs (satisfying conclusions)."
    )

    response = call_structured(
        prompt_name="story_mapper",
        user_content=user_content,
        schema=StoryMapResponse,
        effort="medium",
        db=db,
    )

    
    for beat in response.beats:
        db.insert_story_beat(beat.role, beat.scene_ids, beat.summary)
        
        for scene_id in beat.scene_ids:
            db.update_scene(scene_id, story_role=beat.role)

    
    for dep in response.dependencies:
        db.insert_story_dep(dep.setup_scene, dep.payoff_scene, dep.why)

    
    for hook in response.hook_candidates:
        scene = db.get_scene(hook.scene_id)
        if scene:
            existing_why = scene.get("importance_why") or ""
            db.update_scene(
                hook.scene_id,
                importance_why=f"{existing_why} [HOOK: {hook.why}]",
            )

    console.print(f"    Story beats: {len(response.beats)}")
    console.print(f"    Dependencies: {len(response.dependencies)}")
    console.print(f"    Hook candidates: {len(response.hook_candidates)}")


def _run_importance_scoring(scenes: list[dict], utterances: list[dict], db: ProjectDB) -> None:
    """Score scene importance via Claude (batched) + composite weighting."""
    console.print("    [dim]Scoring importance...[/dim]")

    batch_size = CFG.story.importance_batch_size
    weights = CFG.story.importance_weights  

    
    video = db.get_video()
    total_duration = video["duration_s"] if video else 1.0

    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i + batch_size]

        
        batch_lines = []
        for scene in batch:
            scene_utts = [
                u for u in utterances
                if u["end_time"] > scene["start_time"] and u["start_time"] < scene["end_time"]
            ]
            transcript = " ".join(u["text"] for u in scene_utts)[:150]
            batch_lines.append(
                f"Scene {scene['id']}: caption=\"{scene.get('caption', 'N/A')}\" | "
                f"story_role={scene.get('story_role', 'unknown')} | "
                f"emotion={scene.get('emotion_label', 'neutral')}({scene.get('emotion_intensity', 0):.1f}) | "
                f"transcript=\"{transcript}\""
            )

        user_content = (
            f"Rate the importance of each scene (0.0-1.0) based on: "
            f"information novelty, emotional intensity, narrative necessity, and delivery quality.\n\n"
            + "\n".join(batch_lines)
        )

        response = call_structured(
            prompt_name="importance_scorer",
            user_content=user_content,
            schema=ImportanceBatchResponse,
            effort="low",
            db=db,
        )

        
        for score in response.scores:
            scene = db.get_scene(score.scene_id)
            if not scene:
                continue

            llm_score = score.importance
            motion_score = scene.get("motion_score") or 0.0

            
            scene_dur = scene["end_time"] - scene["start_time"]
            scene_words = db.get_words_in_range(scene["start_time"], scene["end_time"])
            speech_density = min(len(scene_words) / max(scene_dur, 0.1) / 4.0, 1.0)  

            
            composite = (
                weights[0] * llm_score
                + weights[1] * motion_score
                + weights[2] * speech_density
            )

            db.update_scene(
                score.scene_id,
                importance=round(composite, 3),
                importance_why=score.justification,
            )

    console.print(f"    Importance scored: {len(scenes)} scenes")


def run_story_analysis(project_dir: Path, db: ProjectDB) -> None:
    """Run the full story analysis stage."""
    scenes = db.get_scenes()
    utterances = db.get_utterances()

    if not scenes:
        console.print("    [yellow]No scenes — skipping story analysis[/yellow]")
        db.set_coverage("story", "unavailable", note="no scenes")
        return

    
    _run_story_map(scenes, utterances, db)

    
    _run_importance_scoring(scenes, utterances, db)

    db.set_coverage("story", "available")
    db.set_model_manifest("story", CFG.llm.model_id, "claude-sonnet-4-6")
    console.print("    [dim]Story analysis complete[/dim]")
