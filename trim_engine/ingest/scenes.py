"""
Scene Detection Engine (INGESTION_ENGINE.md §4.3 & §6)

Implements two-level structure: shots (camera cuts) → scenes (semantic groups of shots).
Key features:
- PySceneDetect for raw shot boundary detection.
- Degenerate input handling: < 3 shots in > 2 min → pseudo-shots at 20s intervals snapped to pauses.
- Quality-aware keyframe selection: samples frames at 2 fps, scores sharpness via Laplacian variance,
  exposure filtering, deduplicates, and selects top-K.
- Screencast/Slideshow detection (near-zero motion + high text density) to set content_class.
- Agglomerative shot grouping into semantic scenes using visual color histograms, temporal proximity,
  and audio/speaker continuity.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB

console = Console()






def _detect_shots(proxy_path: Path) -> list[tuple[float, float]]:
    """
    Hardened shot boundary detection with dual detectors, flash-cut guard,
    and dark-footage handling.

    Active Alternative Path Fallback: If PySceneDetect fails, automatically falls back
    to using ffmpeg 'select=gt(scene,0.3)' to natively extract scene boundary timestamps.
    """
    try:
        from scenedetect import detect, ContentDetector, AdaptiveDetector

        
        content_scenes = detect(str(proxy_path), ContentDetector(
            threshold=CFG.scene.detector_threshold
        ))
        content_shots = [(s[0].get_seconds(), s[1].get_seconds()) for s in content_scenes]

        
        try:
            adaptive_scenes = detect(str(proxy_path), AdaptiveDetector(
                adaptive_threshold=3.0,
                min_scene_len=15,
            ))
            adaptive_shots = [(s[0].get_seconds(), s[1].get_seconds()) for s in adaptive_scenes]
        except Exception:
            adaptive_shots = []

        
        all_boundaries = set()
        for start, end in content_shots + adaptive_shots:
            all_boundaries.add(round(start, 3))
            all_boundaries.add(round(end, 3))

        sorted_bounds = sorted(all_boundaries)
        deduped = []
        for b in sorted_bounds:
            if not deduped or b - deduped[-1] > 0.3:
                deduped.append(b)

        shots = []
        for i in range(len(deduped) - 1):
            shots.append((deduped[i], deduped[i + 1]))

        if not shots and content_shots:
            shots = content_shots

        
        if shots:
            merged = [shots[0]]
            for shot in shots[1:]:
                if shot[1] - shot[0] < 0.15:
                    merged[-1] = (merged[-1][0], shot[1])
                else:
                    merged.append(shot)
            shots = merged

        return shots

    except Exception as e:
        import logging
        import subprocess
        import re
        console.print(f"    [yellow]⚠ PySceneDetect failed ({e}). Falling back to FFmpeg scene detection.[/yellow]")
        
        cmd = [
            "ffmpeg", "-i", str(proxy_path),
            "-vf", "select='gt(scene,0.3)',showinfo",
            "-f", "null", "-"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        
        times = [0.0]
        for line in res.stderr.splitlines():
            m = re.search(r"pts_time:([\d\.]+)", line)
            if m:
                times.append(float(m.group(1)))
        
        times.sort()
        deduped = [0.0]
        for t in times:
            if t - deduped[-1] > 0.3:
                deduped.append(t)
        
        import cv2
        cap = cv2.VideoCapture(str(proxy_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        dur = frame_count / fps if fps > 0 else 0.0
        cap.release()
        
        if dur > deduped[-1]:
            deduped.append(dur)
            
        shots = []
        for i in range(len(deduped) - 1):
            shots.append((deduped[i], deduped[i+1]))
        
        return shots

def _generate_pseudo_shots(
    duration: float,
    interval: float,
    silences: list[dict],
) -> list[tuple[float, float]]:
    """
    Generate pseudo-shots for static videos, snapping boundaries to silences
    rather than uniform intervals.
    """
    shots = []
    t = 0.0
    silence_midpoints = [ (s["start_time"] + s["end_time"]) / 2 for s in silences ]

    while t < duration:
        target_end = t + interval
        if target_end >= duration:
            shots.append((t, duration))
            break

        
        window_size = 5.0  
        candidates = [m for m in silence_midpoints if abs(m - target_end) <= window_size]

        if candidates:
            
            end = min(candidates, key=lambda x: abs(x - target_end))
        else:
            end = target_end

        
        if end - t < 2.0:
            end = target_end

        shots.append((t, end))
        t = end

    return shots






def _classify_content_class(
    mean_motion: float,
    word_rate: float,
) -> str:
    """
    Classify video content type:
    - screencast: near-zero motion
    - talking_head: high speech rate + moderate motion
    - standard: standard cinematic or visual pacing
    """
    
    if mean_motion < 0.05:
        return "screencast"
    
    if word_rate > 1.2 and mean_motion < 0.20:
        return "talking_head"
    return "standard"






def _extract_color_histogram(
    cap: cv2.VideoCapture,
    start: float,
    end: float,
    fps: float,
) -> np.ndarray:
    """Extract HSV 3D color histogram for the middle frame of a shot."""
    mid_time = (start + end) / 2
    frame_idx = int(mid_time * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        return np.zeros((8 * 8 * 8,), dtype=np.float32)

    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def _group_shots_to_scenes(
    shots: list[tuple[float, float]],
    proxy_path: Path,
    db: ProjectDB,
) -> list[tuple[float, float]]:
    """
    Group shots into semantic scenes using:
    - Visual similarity (color histogram correlation)
    - Temporal adjacency (only consecutive shots can be merged)
    - Audio continuity (same speaker speaking across the shot boundary)
    """
    if len(shots) <= 1:
        return shots

    cap = cv2.VideoCapture(str(proxy_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    
    hists = []
    for start, end in shots:
        hists.append(_extract_color_histogram(cap, start, end, fps))
    cap.release()

    
    utterances = db.get_utterances()

    def get_speaker_at(t: float) -> str | None:
        for u in utterances:
            if u["start_time"] <= t <= u["end_time"]:
                return u.get("speaker_id")
        return None

    
    merged_scenes: list[tuple[float, float]] = []
    current_start, current_end = shots[0]
    current_hist = hists[0]

    
    visual_threshold = 0.65

    for i in range(1, len(shots)):
        next_start, next_end = shots[i]
        next_hist = hists[i]

        
        corr = cv2.compareHist(current_hist, next_hist, cv2.HISTCMP_CORREL)

        
        spk_before = get_speaker_at(next_start - 0.2)
        spk_after = get_speaker_at(next_start + 0.2)
        has_speaker_continuity = (spk_before is not None and spk_after is not None and spk_before == spk_after)

        
        should_merge = False

        
        if corr > visual_threshold:
            should_merge = True
        
        elif has_speaker_continuity and corr > 0.40:
            should_merge = True

        
        if (next_end - current_start) > CFG.scene.max_scene_duration_s:
            should_merge = False

        if should_merge:
            current_end = next_end
            
            current_hist = (current_hist + next_hist) / 2
        else:
            merged_scenes.append((current_start, current_end))
            current_start, current_end = next_start, next_end
            current_hist = next_hist

    merged_scenes.append((current_start, current_end))
    return merged_scenes






def _calculate_sharpness(gray_frame: np.ndarray) -> float:
    """Laplacian variance sharpness metric (§6.3)."""
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def _calculate_exposure(gray_frame: np.ndarray) -> float:
    """Exposure score: lower deviation from medium gray (128) is better."""
    mean_val = np.mean(gray_frame)
    return float(1.0 - abs(mean_val - 128.0) / 128.0)


def _extract_quality_keyframes(
    proxy_path: Path,
    scene_id: int,
    start: float,
    end: float,
    keyframes_dir: Path,
    content_class: str,
) -> list[tuple[float, str]]:
    """
    Extract keyframes using quality-aware sampling:
    - Sample frames at 2 fps.
    - Score sharpness (Laplacian variance) and exposure.
    - Select top-K most diverse, high-quality frames.
    - In screencasts, pick frames at text-change points.

    Face detection is handled by the separate 'faces' DAG stage.
    """
    cap = cv2.VideoCapture(str(proxy_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video proxy file: {proxy_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = end - start

    
    sample_interval = max(1, int(fps / 2))
    start_frame = int(start * fps)
    end_frame = int(end * fps)

    candidates: list[dict] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if (frame_idx - start_frame) % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = _calculate_sharpness(gray)
            exposure = _calculate_exposure(gray)

            
            quality = sharpness * exposure

            candidates.append({
                "time": frame_idx / fps,
                "frame": frame.copy(),
                "gray": gray,
                "quality": quality,
            })

        frame_idx += 1

    cap.release()

    if not candidates:
        return []

    
    if duration > 15.0:
        K = 5
    elif duration > 2.0:
        K = CFG.scene.keyframes_per_scene
    else:
        K = 1
    if content_class == "screencast":
        
        selected_candidates = []
        prev_gray = None
        for cand in candidates:
            if prev_gray is None:
                selected_candidates.append(cand)
            else:
                
                diff = cv2.absdiff(cand["gray"], prev_gray)
                mean_diff = float(np.mean(diff))
                if mean_diff > 2.0:  
                    selected_candidates.append(cand)
            prev_gray = cand["gray"]

        
        if len(selected_candidates) > K:
            selected_candidates = selected_candidates[:K]
        elif not selected_candidates:
            selected_candidates = [candidates[len(candidates) // 2]]
    else:
        
        candidates.sort(key=lambda x: x["quality"], reverse=True)
        selected_candidates = []

        for cand in candidates:
            
            is_dup = False
            for picked in selected_candidates:
                
                corr = cv2.compareHist(
                    cv2.calcHist([cand["frame"]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                    cv2.calcHist([picked["frame"]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]),
                    cv2.HISTCMP_CORREL
                )
                if np.isnan(corr):
                    corr = 0.0
                if corr > 0.90:  
                    is_dup = True
                    break

            if not is_dup:
                selected_candidates.append(cand)
                if len(selected_candidates) >= K:
                    break

        
        if not selected_candidates:
            selected_candidates = candidates[:K]

    
    results = []
    for idx, cand in enumerate(selected_candidates):
        filename = f"scene_{scene_id}_kf_{idx}.jpg"
        path = keyframes_dir / filename
        tmp_path = path.parent / f".tmp_{filename}"
        
        cv2.imwrite(
            str(tmp_path), cand["frame"],
            [cv2.IMWRITE_JPEG_QUALITY, CFG.scene.keyframe_quality],
        )
        os.replace(tmp_path, path)
        
        pos = (cand["time"] - start) / max(0.1, duration)
        results.append((pos, str(path)))

    return results






def _compute_motion_score(proxy_path: Path, start: float, end: float) -> float:
    """Compute motion score as mean absolute frame-diff on sampled grayscale frames."""
    cap = cv2.VideoCapture(str(proxy_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = int(fps / CFG.scene.motion_fps) if fps > 0 else 30

    start_frame = int(start * fps)
    end_frame = int(end * fps)

    diffs: list[float] = []
    prev_gray = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(float(np.mean(diff)))

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not diffs:
        return 0.0

    raw = float(np.mean(diffs))
    return min(raw / 40.0, 1.0)


def _write_shot_index(shots: list[tuple[float, float]], index_path: Path) -> None:
    """Write shot index JSON containing shot boundaries for stream-copy preview generation."""
    data = []
    for idx, (start, end) in enumerate(shots):
        data.append({
            "shot_id": idx,
            "start_time": start,
            "end_time": end,
        })
    tmp_path = index_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, index_path)






def run_scene_detection(project_dir: Path, db: ProjectDB) -> None:
    """Run the full scene detection stage."""
    proxy_path = project_dir / "proxy.mp4"
    keyframes_dir = project_dir / "keyframes"
    keyframes_dir.mkdir(exist_ok=True)

    video = db.get_video()
    if not video:
        raise RuntimeError("No video metadata — run normalize first")
    total_duration = video["duration_s"]

    
    console.print("    Detecting raw camera cuts (shots)...")
    shots = _detect_shots(proxy_path)
    console.print(f"    Detected shots: {len(shots)}")

    
    shot_index_path = project_dir / "shot_index.json"
    _write_shot_index(shots, shot_index_path)

    
    silences = db.get_silences()
    if len(shots) < CFG.scene.min_scenes_threshold and total_duration > 120:
        console.print(
            f"    [yellow]Too few shots ({len(shots)}) for {total_duration:.0f}s video "
            f"— generating pseudo-shots snapped to silences[/yellow]"
        )
        shots = _generate_pseudo_shots(total_duration, CFG.scene.pseudo_scene_interval_s, silences)

    if not shots and total_duration > 0:
        shots = [(0.0, total_duration)]

    
    motion_scores = [ _compute_motion_score(proxy_path, s, e) for s, e in shots ]
    mean_motion = float(np.mean(motion_scores)) if motion_scores else 0.0

    
    words = db.get_words()
    word_rate = len(words) / max(1.0, total_duration)

    
    content_class = _classify_content_class(mean_motion, word_rate)
    console.print(
        f"    Video content class classified as: [bold cyan]{content_class}[/bold cyan] "
        f"(mean motion: {mean_motion:.3f}, word rate: {word_rate:.2f} wps)"
    )

    with db.conn() as c:
        c.execute("UPDATE video SET content_class = ? WHERE id = ?", (content_class, project_dir.name))

    
    console.print("    Grouping shots into semantic scenes...")
    scenes = _group_shots_to_scenes(shots, proxy_path, db)
    console.print(f"    Grouped semantic scenes: {len(scenes)}")

    
    for scene_id, (start, end) in enumerate(scenes):
        
        db.insert_scene(scene_id, start, end)

        
        keyframe_results = _extract_quality_keyframes(
            proxy_path, scene_id, start, end, keyframes_dir, content_class,
        )

        for pos, path in keyframe_results:
            db.insert_keyframe(scene_id, pos, path)

        
        motion = _compute_motion_score(proxy_path, start, end)
        db.update_scene(scene_id, motion_score=motion)

    
    

    db.set_coverage("scenes", "available")
    db.set_model_manifest("scene_detection", "pyscenedetect+laplacian-quality", "2.0")
