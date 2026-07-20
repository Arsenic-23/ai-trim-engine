"""
Visual Intelligence Engine (INGESTION_ENGINE.md §4.5, §6.4, §8)

Production VLM pipeline with cost control ladder and contextual analysis.

- Cost ladder: 3-tier effort (low/medium/high) based on motion score + content class
- Montage grids: multi-keyframe scenes → single composited image to reduce API calls
- Prior-shot context: last scene's caption carried forward for continuity reasoning
- Batching: groups scenes into batches of size CFG.vision.batch_size
- Known-people roster: cross-batch person identity consistency
- Schema conformity: bbox_hints, objects, people, visible_text with model_version
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.llm import call_vision
from trim_engine.schemas import VisionBatchResponse, SceneTags

console = Console()


def _slug_id(value: str) -> str:
    """Create a stable graph-id fragment from model-provided labels."""
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "unknown"






def _determine_effort_tier(
    motion_score: float,
    duration: float,
    content_class: str,
    has_speech: bool,
) -> str:
    """
    Determine VLM effort tier based on scene characteristics.

    Tiers:
      low    — still or near-still shots (screencasts, static B-roll)
                Reduces to 1 keyframe, shorter prompt, effort='low'
      medium — standard talking-head / moderate motion
                Normal keyframes, standard prompt, effort='medium'
      high   — action sequences, montages, complex visual storytelling
                All keyframes, detailed prompt with motion-strip, effort='high'
    """
    if content_class == "screencast" and motion_score < 0.1:
        return "low"
    if motion_score < 0.05 and duration < 5.0:
        return "low"
    if motion_score > 0.5 or (duration > 30 and not has_speech):
        return "high"
    return "medium"


def _create_montage_grid(
    keyframe_paths: list[Path],
    max_cols: int = 3,
    cell_size: tuple[int, int] = (320, 240),
) -> Path | None:
    """
    Create a montage grid image from multiple keyframes.

    For scenes with many keyframes, combines them into a single panoramic
    image to reduce VLM API calls while preserving temporal information.
    """
    if len(keyframe_paths) <= 1:
        return None

    frames = []
    for p in keyframe_paths[:9]:  
        img = cv2.imread(str(p))
        if img is not None:
            img = cv2.resize(img, cell_size)
            frames.append(img)

    if len(frames) < 2:
        return None

    
    n_cols = min(max_cols, len(frames))
    n_rows = (len(frames) + n_cols - 1) // n_cols

    
    while len(frames) < n_rows * n_cols:
        frames.append(np.zeros((cell_size[1], cell_size[0], 3), dtype=np.uint8))

    rows = []
    for r in range(n_rows):
        row_imgs = frames[r * n_cols:(r + 1) * n_cols]
        rows.append(np.hstack(row_imgs))
    montage = np.vstack(rows)

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, p in enumerate(keyframe_paths[:9]):
        r, c = divmod(i, n_cols)
        x = c * cell_size[0] + 5
        y = r * cell_size[1] + 20
        cv2.putText(montage, f"#{i+1}", (x, y), font, 0.5, (255, 255, 255), 1)

    
    montage_path = keyframe_paths[0].parent / f"montage_{keyframe_paths[0].stem}.jpg"
    cv2.imwrite(str(montage_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return montage_path


def _build_batch_context(
    scenes: list[dict],
    utterances: list[dict],
    db: ProjectDB,
    prior_caption: str = "",
) -> list[dict]:
    """
    Build scene context for a batch — each scene gets:
    - Its keyframe paths (with montage grid for multi-keyframe scenes)
    - Overlapping transcript excerpt
    - Duration, motion score, and effort tier
    - Prior-shot caption for continuity reasoning
    """
    video = db.get_video()
    content_class = video.get("content_class", "standard") if video else "standard"

    batch_info = []

    for scene in scenes:
        keyframes = db.get_keyframes(scene["id"])
        kf_paths = [Path(kf["path"]) for kf in keyframes if Path(kf["path"]).exists()]

        
        scene_start = scene["start_time"]
        scene_end = scene["end_time"]
        scene_utts = [
            u for u in utterances
            if u["end_time"] > scene_start and u["start_time"] < scene_end
        ]
        transcript = " ".join(u["text"] for u in scene_utts)
        has_speech = bool(scene_utts)
        duration = scene_end - scene_start
        motion_score = scene.get("motion_score", 0.0)

        
        effort = _determine_effort_tier(motion_score, duration, content_class, has_speech)

        
        effective_kf_paths = kf_paths
        if effort == "low" and len(kf_paths) > 1:
            effective_kf_paths = [kf_paths[len(kf_paths) // 2]]  

        
        montage_path = None
        if (effort == "high" or motion_score > 0.15) and len(kf_paths) > 2:
            montage_path = _create_montage_grid(kf_paths)

        batch_info.append({
            "scene_id": scene["id"],
            "start": scene_start,
            "end": scene_end,
            "duration": duration,
            "motion_score": motion_score,
            "transcript": transcript[:500],
            "keyframe_paths": effective_kf_paths,
            "montage_path": montage_path,
            "effort": effort,
            "prior_caption": prior_caption,
        })

        
        if scene.get("caption"):
            prior_caption = scene["caption"]

    return batch_info


def _build_batch_text(batch_info: list[dict], content_class: str) -> str:
    """Build the text instruction for a vision batch, including effort tiers and prior-shot context."""
    lines = [
        f"Analyze the following scenes from a video. Content type: '{content_class}'.\n"
        "For each scene, keyframes and context are provided.\n"
    ]

    for info in batch_info:
        effort_note = f" [Analysis depth: {info['effort'].upper()}]" if info["effort"] != "medium" else ""
        prior_note = f"\nPrevious scene context: \"{info['prior_caption'][:200]}\"" if info["prior_caption"] else ""

        lines.append(
            f"\n--- SCENE {info['scene_id']} ---{effort_note}\n"
            f"Time: {info['start']:.1f}s – {info['end']:.1f}s (duration: {info['duration']:.1f}s)\n"
            f"Motion score: {info['motion_score']:.2f}\n"
            f"Transcript: \"{info['transcript']}\""
            f"{prior_note}\n"
            f"Keyframes follow:"
        )

        if info.get("montage_path"):
            lines.append("(A montage grid is also provided showing temporal progression.)")

    lines.append(
        "\n\nProvide your analysis for ALL scenes in the batch. "
        "Include coarse visual bounding box hints (bbox_hints) for any prominent objects or people "
        "(e.g., label='person:A', region='left', size='large'). "
        "Return a JSON object with a 'scenes' array containing one entry per scene."
    )

    return "\n".join(lines)


def _build_known_people_context(db: ProjectDB) -> str:
    """Build the known-people roster from previously tagged scenes."""
    entities = db.get_entities(kind="person")
    if not entities:
        return ""

    lines = []
    for e in entities:
        lines.append(f"- {e['id']}: {e.get('description', 'unknown appearance')}")

    return "\n".join(lines)


def _process_batch(
    batch_scenes: list[dict],
    all_utterances: list[dict],
    db: ProjectDB,
    known_people_context: str,
    content_class: str,
    prior_caption: str = "",
) -> tuple[list[SceneTags], str]:
    """Process a single batch of scenes through Claude vision with effort tiers."""
    batch_info = _build_batch_context(batch_scenes, all_utterances, db, prior_caption)

    
    all_image_paths: list[Path] = []
    for info in batch_info:
        all_image_paths.extend(info["keyframe_paths"])
        if info.get("montage_path") and info["montage_path"].exists():
            all_image_paths.append(info["montage_path"])

    if not all_image_paths:
        console.print(f"    [yellow]No keyframes for batch — skipping[/yellow]")
        return [], prior_caption

    batch_text = _build_batch_text(batch_info, content_class)

    
    batch_effort = max(
        (info["effort"] for info in batch_info),
        key=lambda e: {"low": 0, "medium": 1, "high": 2}[e],
    )

    try:
        response = call_vision(
            prompt_name="vision_tagger",
            image_paths=all_image_paths,
            user_text=batch_text,
            schema=VisionBatchResponse,
            effort=batch_effort,
            db=db,
            known_people_context=known_people_context,
        )
        
        last_caption = ""
        if response.scenes:
            last_caption = response.scenes[-1].caption or ""
        return response.scenes, last_caption

    except Exception as e:
        console.print(f"    [yellow]Batch failed: {e} — trying per-scene fallback[/yellow]")

        
        results: list[SceneTags] = []
        last_caption = prior_caption
        for scene_info in batch_info:
            try:
                single_text = _build_batch_text([scene_info], content_class)
                single_paths = list(scene_info["keyframe_paths"])
                if scene_info.get("montage_path") and scene_info["montage_path"].exists():
                    single_paths.append(scene_info["montage_path"])

                response = call_vision(
                    prompt_name="vision_tagger",
                    image_paths=single_paths,
                    user_text=single_text,
                    schema=VisionBatchResponse,
                    effort=scene_info["effort"],
                    db=db,
                    known_people_context=known_people_context,
                )
                results.extend(response.scenes)
                if response.scenes:
                    last_caption = response.scenes[-1].caption or ""
            except Exception as e2:
                console.print(f"    [red]Scene {scene_info['scene_id']} failed: {e2}[/red]")
                db.set_coverage(
                    f"vision_scene_{scene_info['scene_id']}",
                    "unavailable",
                    note=str(e2),
                )

        return results, last_caption


def _apply_tags_to_db(tags: SceneTags, db: ProjectDB) -> None:
    """Write a single scene's tags and bbox relations to the database."""
    scene = db.get_scene(tags.scene_id)
    t_start = scene["start_time"] if scene else None
    t_end = scene["end_time"] if scene else None

    db.update_scene(
        tags.scene_id,
        shot_type=tags.shot_type,
        camera_motion=tags.camera_motion,
        location=tags.location.label if tags.location else None,
        indoor=int(tags.location.indoor) if tags.location else None,
        emotion_label=tags.emotion.label if tags.emotion else None,
        emotion_intensity=tags.emotion.intensity if tags.emotion else None,
        caption=tags.caption,
        is_broll=int(tags.is_broll),
    )

    
    for obj in tags.objects:
        obj_id = f"object:{_slug_id(obj.label)}"
        db.insert_entity(
            obj_id,
            kind="object",
            label=obj.label,
            description=f"prominence={obj.prominence:.2f}",
        )
        db.insert_relation(
            src=f"scene:{tags.scene_id}",
            rel="contains",
            dst=obj_id,
            scene_id=tags.scene_id,
            t_start=t_start,
            t_end=t_end,
            confidence=obj.prominence,
            source="vision",
            model_version="2.0",
        )

    
    for action in tags.actions:
        action_id = f"action:{_slug_id(action)}"
        db.insert_entity(action_id, kind="action", label=action)
        db.insert_relation(
            src=f"scene:{tags.scene_id}",
            rel="performs",
            dst=action_id,
            scene_id=tags.scene_id,
            t_start=t_start,
            t_end=t_end,
            confidence=0.7,
            source="vision",
            model_version="2.0",
        )

    
    for text in tags.visible_text:
        text_id = f"object:text:{_slug_id(text)[:80]}"
        db.insert_entity(text_id, kind="object", label=text, description="visible_text")
        db.insert_relation(
            src=f"scene:{tags.scene_id}",
            rel="contains",
            dst=text_id,
            scene_id=tags.scene_id,
            t_start=t_start,
            t_end=t_end,
            confidence=0.8,
            source="vision",
            model_version="2.0",
            evidence_ref=text,
        )

    
    for hint in tags.bbox_hints:
        db.insert_relation(
            src=hint.label,
            rel="holds_bbox",
            dst=f"{hint.region}:{hint.size}",
            scene_id=tags.scene_id,
            confidence=0.8,
            source="vision",
            model_version="2.0",
        )


def run_vision_tagging(project_dir: Path, db: ProjectDB) -> None:
    """Run the full vision tagging stage."""
    scenes = db.get_scenes()
    utterances = db.get_utterances()

    if not scenes:
        console.print("    [yellow]No scenes found — skipping vision tagging[/yellow]")
        db.set_coverage("vision", "unavailable", note="no scenes")
        return

    
    video = db.get_video()
    content_class = video.get("content_class", "standard") if video else "standard"

    console.print(f"    Tagging {len(scenes)} scenes in batches of {CFG.vision.batch_size}... (content_class: {content_class})")

    
    batch_size = CFG.vision.batch_size
    all_tags: list[SceneTags] = []
    known_people_context = ""
    prior_caption = ""  

    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(scenes) + batch_size - 1) // batch_size

        console.print(f"    Batch {batch_num}/{total_batches} (scenes {batch[0]['id']}–{batch[-1]['id']})...")

        
        known_people_context = _build_known_people_context(db)

        tags, prior_caption = _process_batch(
            batch, utterances, db, known_people_context, content_class, prior_caption
        )
        all_tags.extend(tags)

        
        for tag in tags:
            _apply_tags_to_db(tag, db)

            
            for person in tag.people:
                entity_id = f"person_raw:{tag.scene_id}:{person.key}"
                db.insert_entity(
                    entity_id=entity_id,
                    kind="person",
                    label=person.key,
                    description=person.description,
                )

    tagged_count = len(all_tags)
    total_count = len(scenes)
    console.print(f"    Tagged: {tagged_count}/{total_count} scenes")

    if tagged_count == total_count:
        db.set_coverage("vision", "available")
    elif tagged_count > 0:
        db.set_coverage("vision", "available",
                       note=f"{total_count - tagged_count} scenes failed")
    else:
        db.set_coverage("vision", "unavailable", note="all batches failed")

    
    db.set_model_manifest("vision", CFG.llm.model_id, "claude-sonnet-4-6")

    gc.collect()
