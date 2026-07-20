"""
Timeline Planner (§5.4) — 100% deterministic code, no LLM.

Converts keep/remove segments into EditPlan → timeline.json.
Runs rewrites iteratively to a fixed point.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.schemas import (
    AudioClip, EditIntent, EditPlan, OutputSpec, PlanOperation,
    Repair, RetrievalResult, Segment, Timeline, Transition, VideoClip,
)

console = Console()

PLANNER_PROFILES = {
    "default": {"micro_gap_max_ms": 300.0, "min_clip_ms": 1000.0},
    "fast-cut": {"micro_gap_max_ms": 100.0, "min_clip_ms": 500.0},
    "respect-pauses": {"micro_gap_max_ms": 800.0, "min_clip_ms": 1500.0},
}

def _compute_on_demand_beats(db: ProjectDB) -> list[float]:
    import librosa
    import numpy as np
    import glob
    
    # §4.2: Fallback on-demand beat/onset detection for talking-head videos
    console.print("    [dim]Computing on-demand beats...[/dim]")
    
    proj_dir = Path(db.db_path).parent
    mp4s = glob.glob(f"{proj_dir}/*.mp4")
    if not mp4s:
        return []
        
    try:
        y, sr = librosa.load(mp4s[0], sr=16000, mono=True)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # 1. Try beat tracking
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Compute confidence
        ac = librosa.autocorrelate(onset_env, max_size=onset_env.shape[0])
        ac_norm = ac / (ac[0] + 1e-10)
        bpm_range = [int(60.0 / bpm * sr / 512) for bpm in [60, 200] if bpm > 0]
        if len(bpm_range) >= 2:
            search_range = ac_norm[min(bpm_range):max(bpm_range)]
            confidence = float(np.max(search_range)) if len(search_range) > 0 else 0.0
        else:
            confidence = 0.5
            
        if confidence >= 0.3:
            console.print(f"      [dim]Found rhythmic structure (confidence: {confidence:.2f})[/dim]")
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        else:
            # 2. Fall back to onset detection (speech emphasis points)
            console.print(f"      [dim]Low rhythmic confidence ({confidence:.2f}), falling back to onset detection...[/dim]")
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, backtrack=True, units='frames', delta=0.2)
            beat_times = librosa.frames_to_time(onset_frames, sr=sr)
            
        # Cache to DB
        for bt in beat_times:
            db.insert_beat(float(bt), is_downbeat=0)
            
        return db.get_beats()
        
    except Exception as e:
        console.print(f"      [dim]On-demand beat tracking failed: {e}[/dim]")
        return []

def _get_removal_ranges(
    intent: EditIntent,
    retrieval_results: list[RetrievalResult],
    total_duration: float,
) -> list[tuple[float, float, str, str | None, str | None]]:
    removals = []
    
    # Track all kept segments across all keep_only and compress operations
    # so we can union them before inverting into removals.
    global_keeps = []
    has_positive_filter = False

    for i, op in enumerate(intent.operations):
        if i >= len(retrieval_results):
            continue
        result = retrieval_results[i]

        if op.action == "remove":
            for seg in result.segments:
                reason = f"{op.action}: {op.target.query}"
                evidence = seg.evidence[0].detail if seg.evidence else None
                source = seg.evidence[0].source if seg.evidence else None
                removals.append((seg.start, seg.end, reason, evidence, source))
        elif op.action in ("keep_only", "compress", "highlight"):
            has_positive_filter = True
            if result.segments:
                global_keeps.extend(result.segments)

    if has_positive_filter:
        if not global_keeps:
            # Positive filter requested but nothing found -> remove everything
            removals.append((0.0, total_duration, "keep_only/compress: no segments matched", None, None))
        else:
            # Union all kept segments
            global_keeps = sorted(global_keeps, key=lambda s: s.start)
            merged_keeps = []
            for seg in global_keeps:
                if not merged_keeps:
                    merged_keeps.append([seg.start, seg.end])
                else:
                    prev = merged_keeps[-1]
                    if seg.start <= prev[1] + 0.1:
                        prev[1] = max(prev[1], seg.end)
                    else:
                        merged_keeps.append([seg.start, seg.end])
            
            # Invert merged_keeps to get removals
            prev_end = 0.0
            for start, end in merged_keeps:
                if start > prev_end:
                    removals.append((prev_end, start, "positive filter: inverse of unioned kept segments", None, None))
                prev_end = max(prev_end, end)
            if prev_end < total_duration:
                removals.append((prev_end, total_duration, "positive filter: inverse of unioned kept segments", None, None))
    return removals

def _merge_overlapping_removals(
    removals: list[tuple[float, float, str, str | None, str | None]],
) -> list[tuple[float, float, str, str | None, str | None]]:
    if not removals:
        return removals
    sorted_removals = sorted(removals, key=lambda r: r[0])
    merged = [sorted_removals[0]]
    for start, end, reason, evidence, source in sorted_removals[1:]:
        prev_start, prev_end, prev_reason, prev_evidence, prev_source = merged[-1]
        if start <= prev_end + 0.01:
            merged[-1] = (prev_start, max(prev_end, end), f"{prev_reason}; {reason}", prev_evidence or evidence, prev_source or source)
        else:
            merged.append((start, end, reason, evidence, source))
    return merged

def _snap_to_word_boundary(
    t: float, db: ProjectDB, search_ms: float | None = None, prefer_silence: bool = True,
) -> tuple[float, list[Repair]]:
    cfg = CFG.planner
    search_ms = search_ms or cfg.word_snap_search_ms
    repairs = []
    search_s = search_ms / 1000.0
    start_search = max(0, t - search_s)
    end_search = t + search_s

    # Track A: Maximize cut affinity in the snap window
    try:
        affinity = db.get_cut_affinity()
        if affinity:
            # Filter to window
            window = [a for a in affinity if start_search <= a["t"] <= end_search]
            if window:
                # Find max score; on tie, minimize distance to t
                best = max(window, key=lambda a: (a["score"], -abs(a["t"] - t)))
                if best["score"] >= 0 and abs(best["t"] - t) > 0.05:
                    repairs.append(Repair(type="snap_to_affinity", detail=f"moved {t:.3f}→{best['t']:.3f} (affinity {best['score']})"))
                    return best["t"], repairs
    except Exception:
        pass # Fallback to heuristic if affinity table missing

    if prefer_silence:
        silences = db.get_silences()
        silence_search_s = cfg.silence_search_ms / 1000.0
        best_silence = None
        best_silence_dist = float("inf")
        for s in silences:
            if s["duration"] < cfg.silence_min_ms / 1000.0:
                continue
            mid = (s["start_time"] + s["end_time"]) / 2
            dist = abs(mid - t)
            if dist < silence_search_s and dist < best_silence_dist:
                if start_search <= mid <= end_search:
                    best_silence = mid
                    best_silence_dist = dist
                    
        if best_silence is not None and best_silence != t:
            
            words = db.get_words_in_range(start_search, end_search)
            cuts_word = False
            for w in words:
                if w["start_time"] + 0.015 < best_silence < w["end_time"] - 0.015:
                    cuts_word = True
                    break
            
            if not cuts_word:
                repairs.append(Repair(type="snap_to_silence", detail=f"moved {t:.3f}→{best_silence:.3f} (silence midpoint)"))
                return best_silence, repairs

    words = db.get_words_in_range(start_search, end_search)
    best_gap = None
    best_gap_dist = float("inf")
    for i in range(len(words) - 1):
        gap_start = words[i]["end_time"]
        gap_end = words[i + 1]["start_time"]
        gap_duration = (gap_end - gap_start) * 1000
        if gap_duration >= cfg.word_gap_min_ms:
            gap_mid = (gap_start + gap_end) / 2
            dist = abs(gap_mid - t)
            if dist < best_gap_dist:
                best_gap = gap_mid
                best_gap_dist = dist

    if best_gap is not None and best_gap != t:
        repairs.append(Repair(type="snap_to_word_boundary", detail=f"moved {t:.3f}→{best_gap:.3f} (word gap)"))
        return best_gap, repairs

    if words:
        best_edge = None
        best_edge_dist = float("inf")
        for w in words:
            for edge in [w["start_time"], w["end_time"]]:
                dist = abs(edge - t)
                if dist < best_edge_dist:
                    best_edge = edge
                    best_edge_dist = dist
        if best_edge is not None and best_edge != t:
            repairs.append(Repair(type="snap_to_word_edge", detail=f"moved {t:.3f}→{best_edge:.3f} (word edge)"))
            return best_edge, repairs

    return t, repairs

def _apply_micro_gap_merge(
    keeps: list[tuple[float, float]],
    max_gap_ms: float,
    rule_logs: list[dict] | None = None,
) -> list[tuple[float, float]]:
    if len(keeps) < 2:
        return keeps

    merged = [keeps[0]]
    for start, end in keeps[1:]:
        prev_start, prev_end = merged[-1]
        gap = (start - prev_end) * 1000
        if gap < max_gap_ms:
            merged[-1] = (prev_start, end)
            if rule_logs is not None:
                rule_logs.append({
                    "rule": "micro_gap_merge",
                    "before": f"gap {gap:.1f}ms",
                    "after": "merged",
                    "reason": f"gap was smaller than threshold {max_gap_ms}ms"
                })
        else:
            merged.append((start, end))
    return merged

def _apply_min_clip_rule(
    keeps: list[tuple[float, float]],
    min_clip_ms: float,
    is_beat_cut: bool = False,
    rule_logs: list[dict] | None = None,
) -> list[tuple[float, float]]:
    if is_beat_cut:
        return keeps

    filtered = []
    for s, e in keeps:
        dur = (e - s) * 1000
        if dur >= min_clip_ms:
            filtered.append((s, e))
        else:
            if rule_logs is not None:
                rule_logs.append({
                    "rule": "min_clip_drop",
                    "before": f"clip {s:.1f}s–{e:.1f}s (dur={dur:.0f}ms)",
                    "after": "dropped",
                    "reason": f"duration was below threshold {min_clip_ms}ms"
                })
    if not filtered and keeps:
        longest = max(keeps, key=lambda k: k[1] - k[0])
        filtered.append(longest)
        if rule_logs is not None:
            rule_logs.append({
                "rule": "min_clip_drop_fallback",
                "before": "empty timeline",
                "after": f"kept {longest[0]:.1f}s–{longest[1]:.1f}s",
                "reason": "prevented timeline from becoming empty"
            })
            
    return filtered

def _check_jl_cut(
    cut_time: float,
    scenes: list[dict],
    scene_boundary_ms: float,
    offset_max_ms: float,
) -> tuple[float, float, list[Repair]]:
    repairs = []
    video_cut = cut_time
    audio_cut = cut_time

    for scene in scenes:
        for boundary in [scene["start_time"], scene["end_time"]]:
            dist_ms = abs(cut_time - boundary) * 1000
            if dist_ms <= scene_boundary_ms:
                offset = min(dist_ms, offset_max_ms) / 1000.0
                if cut_time < boundary:
                    audio_cut = cut_time + offset
                else:
                    audio_cut = cut_time - offset
                repairs.append(Repair(type="jl_cut", detail=f"J/L cut: video={video_cut:.3f}, audio={audio_cut:.3f}"))
                return video_cut, audio_cut, repairs

    return video_cut, audio_cut, repairs

def _snap_to_beat(
    t: float,
    beats: list[float],
    tolerance_ms: float,
    db: ProjectDB | None = None,
) -> tuple[float, list[Repair]]:
    if not beats:
        return t, []
    nearest = min(beats, key=lambda b: abs(b - t))
    dist_ms = abs(nearest - t) * 1000
    if dist_ms <= tolerance_ms:
        if db:
            words = db.get_words_in_range(nearest - 0.1, nearest + 0.1)
            for w in words:
                if w["start_time"] + 0.015 < nearest < w["end_time"] - 0.015:
                    return t, []  
        return nearest, [Repair(type="beat_snap", detail=f"snapped {t:.3f}→{nearest:.3f} (beat alignment)")]
    return t, []

def _quantize_time(t: float, frame_lut: list[dict], fps: float) -> float:
    if not frame_lut:
        frame_time = 1.0 / fps
        return round(t / frame_time) * frame_time
    t_us = int(t * 1_000_000)
    closest = min(frame_lut, key=lambda x: abs(x["pts_us"] - t_us))
    return closest["pts_us"] / 1_000_000.0

def plan_timeline(
    intent: EditIntent,
    retrieval_results: list[RetrievalResult],
    db: ProjectDB,
    project_dir: Path | None = None,
) -> tuple[EditPlan, Timeline]:
    """
    Plan frame-accurate edits via convergent fixed-point rewrite loops.
    """
    try:
        if project_dir is None:
            project_dir = Path(db.db_path).parent if hasattr(db, 'db_path') else Path(".")

        profile_key = intent.style.pacing or "default"
        planner_cfg = PLANNER_PROFILES.get(profile_key, PLANNER_PROFILES["default"])

        cfg = CFG.planner
        video = db.get_video()
        if not video:
            raise RuntimeError("No video metadata found")

        total_duration = video["duration_s"]
        fps = video["fps"]
        scenes = db.get_scenes()
        is_beat_cut = intent.style.cut_style == "beat_cut"

        removals = _get_removal_ranges(intent, retrieval_results, total_duration)

        # Sanitize: clamp every removal to source bounds and drop degenerate
        # ranges — retrieval / LLM output can reference times past the video end.
        clamped = []
        for start, end, reason, evidence, source in removals:
            start = max(0.0, min(total_duration, start))
            end = max(0.0, min(total_duration, end))
            if end - start > 0.01:
                clamped.append((start, end, reason, evidence, source))
        removals = _merge_overlapping_removals(clamped)

        snapped_removals = []
        for start, end, reason, evidence, source in removals:
            if source in ("vad", "filler_detector"):
                snapped_start, repairs_s = _snap_to_word_boundary(start, db, prefer_silence=False)
                snapped_end, repairs_e = _snap_to_word_boundary(end, db, prefer_silence=False)
            else:
                snapped_start, repairs_s = _snap_to_word_boundary(start, db)
                snapped_end, repairs_e = _snap_to_word_boundary(end, db)
                
            if snapped_start >= snapped_end:
                snapped_start, snapped_end = start, end
                repairs_s, repairs_e = [], []
                
            snapped_removals.append((snapped_start, snapped_end, reason, evidence, repairs_s + repairs_e))

        keeps = []
        prev_end = 0.0
        for start, end, *_ in sorted(snapped_removals, key=lambda r: r[0]):
            if start > prev_end:
                keeps.append((prev_end, start))
            prev_end = max(prev_end, end)
        if prev_end < total_duration:
            keeps.append((prev_end, total_duration))

        rule_logs = []

        
        for iteration in range(5):
            prev_keeps = list(keeps)

            
            keeps = _apply_micro_gap_merge(keeps, planner_cfg["micro_gap_max_ms"], rule_logs)

            
            keeps = _apply_min_clip_rule(keeps, planner_cfg["min_clip_ms"], is_beat_cut, rule_logs)

            
            snapped_keeps = []
            for start, end in keeps:
                snapped_start, _ = _snap_to_word_boundary(start, db)
                snapped_end, _ = _snap_to_word_boundary(end, db)
                if snapped_start < snapped_end:
                    snapped_keeps.append((snapped_start, snapped_end))
            keeps = snapped_keeps

            if keeps == prev_keeps:
                break

        
        beats = []
        if is_beat_cut:
            beats = db.get_beats()
            if not beats:
                beats = _compute_on_demand_beats(db)
            if not beats:
                intent.style.cut_style = "fast-cut"
                is_beat_cut = False
                rule_logs.append("No rhythmic structure found — applied fast-paced cuts instead.")
                planner_cfg = PLANNER_PROFILES["fast-cut"]
                
        if beats:
            snapped_keeps = []
            for start, end in keeps:
                new_start, _ = _snap_to_beat(start, beats, cfg.beat_snap_ms, db)
                new_end, _ = _snap_to_beat(end, beats, cfg.beat_snap_ms, db)
                if new_start < new_end:
                    snapped_keeps.append((new_start, new_end))
            keeps = snapped_keeps

        if not keeps:
            min_dur = max(cfg.min_output_duration, total_duration * cfg.min_output_floor_ratio)
            keeps = [(0.0, min(min_dur, total_duration))]

        
        frame_lut = []
        lut_path = project_dir / "frame_lut.json"
        if lut_path.exists():
            try:
                with open(lut_path) as f:
                    frame_lut = json.load(f)
            except Exception:
                pass

        quantized_keeps = []
        for s, e in keeps:
            qs = _quantize_time(s, frame_lut, fps)
            qe = _quantize_time(e, frame_lut, fps)
            if qs < qe:
                quantized_keeps.append((qs, qe))
        keeps = quantized_keeps

        # Self-repair invariants before asserting: snapping/quantization can
        # nudge clips out of bounds or out of order. Clamp, sort, and resolve
        # overlaps deterministically instead of discarding the whole plan.
        repaired = []
        for s, e in sorted(keeps):
            s = max(0.0, min(s, total_duration))
            e = max(0.0, min(e, total_duration))
            if e - s <= 0.01:
                continue
            if repaired and s < repaired[-1][1]:
                s = repaired[-1][1]
                if e - s <= 0.01:
                    continue
            repaired.append((s, e))
        keeps = repaired

        if not keeps:
            min_dur = max(cfg.min_output_duration, total_duration * cfg.min_output_floor_ratio)
            keeps = [(0.0, min(min_dur, total_duration))]

        assert len(keeps) > 0, "No clips remaining in timeline"
        assert all(keeps[idx][0] < keeps[idx][1] for idx in range(len(keeps))), "Timeline clip duration <= 0"
        # §2.1: Relaxed — reordered timelines may have non-monotonic src_in.
        # We only require no two clips overlap in source time.
        assert all(0.0 <= s <= total_duration and 0.0 <= e <= total_duration for s, e in keeps), "Timeline clip out of source bounds"

        # §2.1: Apply reorder permutation from story agent's ordered_segments
        has_reorder = False
        if retrieval_results:
            for rr in retrieval_results:
                if rr.ordered_segments:
                    # Build a mapping: (src_start, src_end) of ordered segments → desired position
                    ordered_ranges = [(round(s.start, 2), round(s.end, 2)) for s in rr.ordered_segments]
                    keep_ranges = [(round(s, 2), round(e, 2)) for s, e in keeps]
                    # Map each ordered range to its containing keep range
                    reordered_keeps = []
                    used_keeps = set()
                    for o_start, o_end in ordered_ranges:
                        for k_idx, (k_start, k_end) in enumerate(keep_ranges):
                            if k_idx in used_keeps:
                                continue
                            # Check if the ordered segment overlaps this keep range
                            if k_start <= o_start + 0.5 and k_end >= o_end - 0.5:
                                reordered_keeps.append(keeps[k_idx])
                                used_keeps.add(k_idx)
                                break
                    # Append any keeps not covered by ordered_segments
                    for k_idx in range(len(keeps)):
                        if k_idx not in used_keeps:
                            reordered_keeps.append(keeps[k_idx])
                    if len(reordered_keeps) == len(keeps):
                        keeps = reordered_keeps
                        has_reorder = True
                    break  # only process first ordered result

        plan_ops = []
        for start, end, reason, evidence, repairs in snapped_removals:
            
            crossfade_dur = cfg.crossfade_ms
            repairs.append(Repair(type="audio_crossfade", detail=f"{crossfade_dur:.0f}ms crossfade"))
            repairs.append(Repair(type="room_tone_patch", detail="applied 50ms room-tone patch to cut boundary"))

            plan_ops.append(PlanOperation(
                op_id=f"op_{len(plan_ops)}",
                type="delete",
                range_start=start,
                range_end=end,
                reason=reason,
                evidence_ref=evidence,
                repairs=repairs,
                depends_on=[],
            ))

        kept_duration = sum(end - start for start, end in keeps)
        removal_ratio = (total_duration - kept_duration) / total_duration if total_duration > 0 else 0.0

        aspect = intent.constraints.aspect_ratio or "16:9"
        if intent.constraints.platform:
            from trim_engine.config import PLATFORM_TEMPLATES
            tmpl = PLATFORM_TEMPLATES.get(intent.constraints.platform)
            if tmpl:
                aspect = tmpl.aspect_ratio

        video_clips = []
        audio_clips = []

        for i, (start, end) in enumerate(keeps):
            v_start, a_start = start, start
            v_end, a_end = end, end

            # §2.2: Detect non-adjacent join (reordered clip)
            is_nonadj = False
            if i > 0:
                prev_end_src = keeps[i - 1][1]
                if has_reorder and abs(start - prev_end_src) > 1.0:
                    is_nonadj = True

            if i > 0 and scenes and not is_nonadj:
                _, a_start_new, _ = _check_jl_cut(start, scenes, cfg.jl_cut_scene_boundary_ms, cfg.jl_cut_offset_max_ms)
                if a_start_new != a_start:
                    a_start = a_start_new
                else:
                    # Track A: Micro J/L cuts everywhere
                    jl_micro_s = 0.120 # ~120ms
                    target_a_start = max(keeps[i-1][1], start - jl_micro_s)
                    if target_a_start < start:
                        # Check if this region contains any words
                        words_in_gap = db.get_words_in_range(target_a_start, start)
                        if not words_in_gap:
                            a_start = target_a_start

            transition_type = "match_cut" if intent.style.cut_style == "match_cut" and i < len(keeps) - 1 else "cut"
            
            video_clips.append(VideoClip(src_in=v_start, src_out=v_end, transition_out=Transition(type=transition_type)))

            # §2.2: Use reorder crossfade for non-adjacent joins
            if is_nonadj:
                fade_in = int(cfg.reorder_crossfade_ms)
                fade_out = int(cfg.reorder_crossfade_ms) if i < len(keeps) - 1 else 0
            else:
                fade_in = int(cfg.crossfade_ms) if i > 0 else 0
                fade_out = int(cfg.crossfade_ms) if i < len(keeps) - 1 else 0

            audio_clips.append(AudioClip(src_in=a_start, src_out=a_end, fade_in_ms=fade_in, fade_out_ms=fade_out))

        edit_plan = EditPlan(
            plan_id=str(uuid.uuid4())[:8],
            operations=plan_ops,
            predicted_output_duration_s=round(kept_duration, 2),
            removal_ratio=round(removal_ratio, 4),
            clip_count=len(keeps),
            rule_logs=rule_logs,
        )

        timeline = Timeline(
            version=1,
            source=str(video["path"]),
            fps=fps,
            output=OutputSpec(aspect=aspect, target_lufs=CFG.renderer.loudnorm_i),
            video_clips=video_clips,
            audio_clips=audio_clips,
            provenance=plan_ops,
        )

        # 5) Track B1: Tempo Curves / Rhythm shaping
        from trim_engine.query.rhythm import generate_tempo_map
        timeline = generate_tempo_map(timeline, intent)

        console.print(f"  [green]✓ Produced timeline with {len(timeline.video_clips)} clips[/green]")
        return edit_plan, timeline

    except Exception as e:
        console.print(f"    [red]⚠ Planner failed with {e}. Falling back to naive chronological plan.[/red]")
        video = db.get_video()
        total_duration = video["duration_s"] if video else 100.0
        fps = video["fps"] if video else 30.0
        aspect = intent.constraints.aspect_ratio or "16:9"
        
        edit_plan = EditPlan(
            plan_id="fallback_0",
            operations=[],
            predicted_output_duration_s=total_duration,
            removal_ratio=0.0,
            clip_count=1,
            rule_logs=[{"rule": "fallback", "reason": f"Planner crashed: {e}"}]
        )
        
        timeline = Timeline(
            version=1,
            source=str(video["path"]) if video else "source.mp4",
            fps=fps,
            output=OutputSpec(aspect=aspect, target_lufs=-14.0),
            video_clips=[VideoClip(src_in=0.0, src_out=total_duration, transition_out=Transition(type="cut"))],
            audio_clips=[AudioClip(src_in=0.0, src_out=total_duration, fade_in_ms=0, fade_out_ms=0)],
            provenance=[],
        )
        return edit_plan, timeline
