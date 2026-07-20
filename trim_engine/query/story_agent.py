"""
Story Agent (§5.3) — post-retrieval narrative restructuring and knapsack logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.llm import call_structured
from trim_engine.schemas import (
    EditIntent, RetrievalResult, Segment, StoryAgentResponse, Evidence,
)

console = Console()

class Tradeoff:
    def __init__(self, key: str, impact_s: float, description: str):
        self.key = key
        self.impact_s = impact_s
        self.description = description

    def to_dict(self) -> dict:
        return {"key": self.key, "impact_s": self.impact_s, "description": self.description}


def _template_chronological(segs: list[Segment], db: ProjectDB | None = None) -> list[Segment]:
    return sorted(segs, key=lambda s: s.start)

def _template_hook_first(segs: list[Segment], db: ProjectDB | None = None) -> list[Segment]:
    """§2.3: Real hook_first — use DB story_beats to find the actual hook scene, move it first."""
    hook_ids: set[int] = set()
    if db:
        beats = db.get_story_beats()
        for b in beats:
            if b["role"] == "hook":
                hook_ids.update(b["scene_ids"])
    # Fallback: check evidence strings
    if not hook_ids:
        for s in segs:
            if any("hook" in e.detail for e in s.evidence):
                hook_ids.update(s.scene_ids)
    # Also check metadata story_role on scenes
    if not hook_ids and db:
        scenes = db.get_scenes()
        for sc in scenes:
            if sc.get("story_role") == "hook":
                hook_ids.add(sc["id"])

    hooks = [s for s in segs if any(sid in hook_ids for sid in s.scene_ids)]
    rest = [s for s in segs if not any(sid in hook_ids for sid in s.scene_ids)]
    rest.sort(key=lambda s: s.start)  # rest stays chronological
    return hooks + rest

def _template_trailer(segs: list[Segment], db: ProjectDB | None = None) -> list[Segment]:
    """§2.3: Real trailer — hook → climax peek → development highlights → outro."""
    role_map: dict[str, list[Segment]] = {"hook": [], "climax": [], "payoff": [], "development": [], "outro": [], "other": []}
    beat_scene_roles: dict[int, str] = {}
    if db:
        beats = db.get_story_beats()
        for b in beats:
            for sid in b["scene_ids"]:
                beat_scene_roles[sid] = b["role"]
    # Also check scene metadata
    if db:
        scenes = db.get_scenes()
        for sc in scenes:
            if sc.get("story_role") and sc["id"] not in beat_scene_roles:
                beat_scene_roles[sc["id"]] = sc["story_role"]

    for s in segs:
        placed = False
        for sid in s.scene_ids:
            role = beat_scene_roles.get(sid)
            if role and role in role_map:
                role_map[role].append(s)
                placed = True
                break
        if not placed:
            role_map["other"].append(s)

    # Trailer order: hook → climax peek → development highlights (by score) → outro
    result = []
    result.extend(sorted(role_map["hook"], key=lambda s: s.start))
    result.extend(sorted(role_map["climax"], key=lambda s: s.start))
    dev_and_other = role_map["development"] + role_map["other"] + role_map["payoff"]
    result.extend(sorted(dev_and_other, key=lambda s: s.score, reverse=True))
    result.extend(sorted(role_map["outro"], key=lambda s: s.start))
    return result

def _template_highlight(segs: list[Segment], db: ProjectDB | None = None) -> list[Segment]:
    return sorted(segs, key=lambda s: s.score, reverse=True)

def _template_build_tension(segs: list[Segment], db: ProjectDB | None = None) -> list[Segment]:
    """Order by ascending intensity/score for tension build."""
    return sorted(segs, key=lambda s: s.score)

NARRATIVE_TEMPLATES: dict[str, callable] = {
    "chronological": _template_chronological,
    "hook_first": _template_hook_first,
    "trailer": _template_trailer,
    "highlight": _template_highlight,
    "build_tension": _template_build_tension,
}

def _needs_story_agent(intent: EditIntent) -> bool:
    if intent.style.cut_style in ("beat_cut", "match_cut"):
        return True

    if intent.style.narrative_shape in ("hook_first", "trailer", "highlight"):
        return True

    for op in intent.operations:
        if op.action in ("restructure", "compress"):
            return True

    if intent.constraints.target_duration_s is not None:
        for op in intent.operations:
            if op.action == "keep_only":
                return True

    return False

def _load_boundary_frame_embeddings(
    db: ProjectDB,
) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
    """
    Load per-scene (first_frame_emb, last_frame_emb) from the frame CLIP index.

    Returns None when the frame index or LUT is unavailable (match cuts then
    degrade gracefully to chronological order).
    """
    import pickle

    import numpy as np

    faiss_dir = db.db_path.parent / "faiss"
    index_path = faiss_dir / "frame_clip.index"
    lut_path = faiss_dir / "frame_lut_map.pkl"
    if not index_path.exists() or not lut_path.exists():
        return None

    try:
        import faiss

        index = faiss.read_index(str(index_path))
        with open(lut_path, "rb") as f:
            lut: list[dict] = pickle.load(f)
    except Exception as e:
        console.print(f"    [yellow]Match cut: frame index load failed ({e})[/yellow]")
        return None

    if index.ntotal == 0 or not lut:
        return None

    vectors = index.reconstruct_n(0, index.ntotal)

    
    
    boundaries: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for entry, vec in zip(lut, vectors):
        sid = entry["scene_id"]
        if sid not in boundaries:
            boundaries[sid] = (vec, vec)
        else:
            first, _ = boundaries[sid]
            boundaries[sid] = (first, vec)

    
    normalized: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for sid, (first, last) in boundaries.items():
        fn = np.linalg.norm(first)
        ln = np.linalg.norm(last)
        if fn == 0 or ln == 0:
            continue
        normalized[sid] = (first / fn, last / ln)
    return normalized or None


def _order_for_match_cuts(
    segments: list[Segment],
    db: ProjectDB,
    min_similarity: float | None = None,
) -> list[Segment]:
    """
    Order kept segments so consecutive joins maximize visual similarity
    between the outgoing segment's last frame and the incoming segment's
    first frame (greedy nearest-neighbor chain over CLIP embeddings).

    Joins whose similarity clears `min_similarity` are annotated with
    `match_cut` evidence, which the planner uses to label the transition
    and the report uses to explain the ordering.
    """
    if len(segments) < 2:
        return segments
    if min_similarity is None:
        min_similarity = CFG.story_agent.match_cut_min_similarity

    boundaries = _load_boundary_frame_embeddings(db)
    if boundaries is None:
        console.print(
            "    [yellow]Match cut: frame embeddings unavailable — keeping chronological order[/yellow]"
        )
        return segments

    def seg_boundary(seg: Segment) -> tuple[np.ndarray, np.ndarray] | None:
        """(first_frame_emb of first scene, last_frame_emb of last scene)."""
        with_embs = [sid for sid in seg.scene_ids if sid in boundaries]
        if not with_embs:
            return None
        first_sid = min(with_embs)
        last_sid = max(with_embs)
        return boundaries[first_sid][0], boundaries[last_sid][1]

    embeddable = [(seg, seg_boundary(seg)) for seg in segments]
    if sum(1 for _, b in embeddable if b is not None) < 3:
        return segments

    
    
    
    remaining = list(embeddable)
    remaining.sort(key=lambda x: x[0].start)
    chain = [remaining.pop(0)]
    match_similarities: list[float] = []

    while remaining:
        _, tail_b = chain[-1]
        if tail_b is None:
            
            nxt_idx = min(range(len(remaining)), key=lambda i: remaining[i][0].start)
            chain.append(remaining.pop(nxt_idx))
            match_similarities.append(0.0)
            continue

        tail_last = tail_b[1]
        best_idx, best_sim = None, -2.0
        for i, (_, cand_b) in enumerate(remaining):
            if cand_b is None:
                continue
            sim = float(np.dot(tail_last, cand_b[0]))
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx is None:
            nxt_idx = min(range(len(remaining)), key=lambda i: remaining[i][0].start)
            chain.append(remaining.pop(nxt_idx))
            match_similarities.append(0.0)
        else:
            chain.append(remaining.pop(best_idx))
            match_similarities.append(best_sim)

    
    matched = 0
    for (seg, _), sim in zip(chain[1:], match_similarities):
        if sim >= min_similarity:
            seg.evidence.append(
                Evidence(
                    source="match_cut",
                    detail=f"visual similarity {sim:.2f} with preceding clip",
                    t=seg.start,
                )
            )
            matched += 1

    console.print(
        f"    Match cut: ordered {len(chain)} segments, {matched} joins ≥ {min_similarity:.2f} similarity"
    )
    return [seg for seg, _ in chain]


def _knapsack_selection(
    segments: list[Segment],
    scenes: list[dict],
    target_duration: float | None,
    total_duration: float,
) -> tuple[list[Segment], list[Segment]]:
    """
    0/1 knapsack using local-search pairwise-swap optimization.
    """
    if not segments or not scenes:
        return segments, []

    scene_map = {s["id"]: s for s in scenes}
    scored_segments = []

    for seg in segments:
        duration = seg.end - seg.start
        base_importance = 0.0
        for sid in seg.scene_ids:
            if sid in scene_map:
                base_importance = max(base_importance, scene_map[sid].get("importance", 0.5))
        
        
        
        importance = (seg.score * 100.0) + base_importance

        scored_segments.append((seg, importance, duration))

    
    scored_segments.sort(key=lambda x: x[1] / max(x[2], 0.01), reverse=True)

    if target_duration is None:
        target_duration = float('inf')

    kept: list[Segment] = []
    dropped: list[Segment] = []
    
    
    high_score_segs = [s for s in scored_segments if s[0].score >= 0.5]
    low_score_segs = [s for s in scored_segments if s[0].score < 0.5]
    
    if high_score_segs and target_duration < sum(s[2] for s in high_score_segs):
        
        
        total_importance = sum(s[1] for s in high_score_segs)
        
        current_duration = 0.0
        for seg, importance, duration in high_score_segs:
            allowed = max((importance / total_importance) * target_duration, 3.0)
            if allowed > duration:
                allowed = duration
            
            ev_t = seg.evidence[0].t if seg.evidence and seg.evidence[0].t else seg.start
            new_start = max(seg.start, ev_t - (allowed / 2.0))
            new_end = min(seg.end, new_start + allowed)
            new_start = max(seg.start, new_end - allowed)
            
            truncated_seg = Segment(
                start=new_start, end=new_end, scene_ids=seg.scene_ids,
                score=seg.score, evidence=seg.evidence,
                needs_confirmation=seg.needs_confirmation
            )
            kept.append(truncated_seg)
            current_duration += (new_end - new_start)
            console.print(f"    [yellow]Knapsack: Fair-allocated {new_end - new_start:.1f}s to high-score segment to fit budget[/yellow]")
        
        for seg, _, _ in low_score_segs:
            dropped.append(seg)
    else:
        
        current_duration = 0.0
        for seg, importance, duration in scored_segments:
            if current_duration + duration <= target_duration:
                kept.append(seg)
                current_duration += duration
            elif current_duration < target_duration:
                allowed = max(target_duration - current_duration, 3.0)
                ev_t = seg.evidence[0].t if seg.evidence and seg.evidence[0].t else seg.start
                new_start = max(seg.start, ev_t - (allowed / 2.0))
                new_end = min(seg.end, new_start + allowed)
                new_start = max(seg.start, new_end - allowed)
                
                truncated_seg = Segment(
                    start=new_start, end=new_end, scene_ids=seg.scene_ids,
                    score=seg.score, evidence=seg.evidence,
                    needs_confirmation=seg.needs_confirmation
                )
                kept.append(truncated_seg)
                current_duration += (new_end - new_start)
                console.print(f"    [yellow]Knapsack: Truncated segment to {new_end - new_start:.1f}s centered at {ev_t:.1f}s to fit budget[/yellow]")
            else:
                dropped.append(seg)

    
    improved = True
    while improved:
        improved = False
        best_swap = None
        best_gain = 0.0

        for i, k_seg in enumerate(kept):
            k_dur = k_seg.end - k_seg.start
            k_imp = max((scene_map[sid].get("importance", 0.5) for sid in k_seg.scene_ids), default=k_seg.score)
            
            for j, d_seg in enumerate(dropped):
                d_dur = d_seg.end - d_seg.start
                d_imp = max((scene_map[sid].get("importance", 0.5) for sid in d_seg.scene_ids), default=d_seg.score)

                new_dur = current_duration - k_dur + d_dur
                if new_dur <= target_duration:
                    gain = d_imp - k_imp
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j, new_dur)

        if best_swap:
            i, j, new_dur = best_swap
            k_seg = kept.pop(i)
            d_seg = dropped.pop(j)
            kept.append(d_seg)
            dropped.append(k_seg)
            current_duration = new_dur
            improved = True

    
    if not kept and segments:
        
        best_seg = min(segments, key=lambda s: s.end - s.start)
        kept.append(best_seg)
        dropped = [s for s in segments if s != best_seg]
        console.print(f"    [yellow]Knapsack: target duration {target_duration:.1f}s impossible, fallback to shortest segment ({best_seg.end - best_seg.start:.1f}s)[/yellow]")

    kept.sort(key=lambda s: s.start)
    return kept, dropped

def _enforce_dependencies(
    kept: list[Segment],
    dropped: list[Segment],
    db: ProjectDB,
    target_duration: float | None,
) -> tuple[list[Segment], list[Segment], list[Tradeoff]]:
    """Enforce setup-payoff story dependencies and return structured tradeoffs."""
    tradeoffs: list[Tradeoff] = []
    deps = db.get_story_deps()
    if not deps:
        return kept, dropped, tradeoffs

    kept_scene_ids = {sid for s in kept for sid in s.scene_ids}

    for dep in deps:
        payoff = dep["payoff_scene"]
        setup = dep["setup_scene"]

        if payoff in kept_scene_ids and setup not in kept_scene_ids:
            for i, seg in enumerate(dropped):
                if setup in seg.scene_ids:
                    setup_dur = seg.end - seg.start
                    current_dur = sum(s.end - s.start for s in kept)
                    print(f"DEBUG ENFORCE: target={target_duration} current_dur={current_dur} setup_dur={setup_dur}")
                    if target_duration is not None and current_dur + setup_dur > target_duration:
                        allowed = max(target_duration - current_dur, 3.0)
                        center = seg.start + (setup_dur / 2.0)
                        truncated_seg = Segment(
                            start=max(seg.start, center - (allowed / 2.0)),
                            end=min(seg.end, center + (allowed / 2.0)),
                            scene_ids=seg.scene_ids,
                            score=seg.score,
                            evidence=seg.evidence
                        )
                        tradeoffs.append(Tradeoff(
                            key=f"setup_{setup}",
                            impact_s=allowed,
                            description=f"Requires setup scene {setup} for payoff {payoff} (truncated to {allowed:.1f}s to fit budget)"
                        ))
                        print(f"DEBUG ENFORCE: Truncated {setup} to {allowed}s")
                        kept.append(truncated_seg)
                    else:
                        print(f"DEBUG ENFORCE: Kept {setup} full duration {setup_dur}")
                        kept.append(seg)

                    dropped.pop(i)
                    kept_scene_ids.add(setup)
                    break

    kept.sort(key=lambda s: s.start)
    return kept, dropped, tradeoffs

def _call_story_reasoning(
    intent: EditIntent,
    kept: list[Segment],
    dropped: list[Segment],
    db: ProjectDB,
) -> list[Segment]:
    story_beats = db.get_story_beats()
    deps = db.get_story_deps()
    target_duration = intent.constraints.target_duration_s

    kept_summary = "\n".join(f"  KEEP: {s.start:.1f}s–{s.end:.1f}s (score={s.score:.2f}, scenes={s.scene_ids})" for s in kept)
    dropped_summary = "\n".join(f"  DROP: {s.start:.1f}s–{s.end:.1f}s (score={s.score:.2f}, scenes={s.scene_ids})" for s in dropped[:10])
    beat_summary = "\n".join(f"  {b['role']}: scenes {b['scene_ids']}" for b in story_beats)

    user_content = (
        f"EDIT INTENT:\n  Style: {intent.style.cut_style or 'standard'}\n  Ordering: {intent.style.ordering or 'chronological'}\n\n"
        f"KNAPSACK RESULT:\n{kept_summary}\n\n"
        f"DROPPED:\n{dropped_summary}\n\n"
        f"STORY BEATS:\n{beat_summary}\n\n"
        f"Review this selection. Propose up to 5 swaps if they improve the edit. Each swap must maintain total duration."
    )

    response = call_structured(
        prompt_name="planner",
        user_content=user_content,
        schema=StoryAgentResponse,
        effort="medium",
        db=db,
    )

    if response.swaps:
        console.print(f"    Story Agent: {len(response.swaps)} proposed swap(s)")
        pre_swap_kept = list(kept)
        pre_swap_dropped = list(dropped)

        for swap in response.swaps[:5]:
            remove_idx = next((i for i, seg in enumerate(kept) if swap.remove_scene_id in seg.scene_ids), None)
            add_seg = next((seg for seg in dropped if swap.add_scene_id in seg.scene_ids), None)

            if remove_idx is not None and add_seg is not None:
                removed = kept.pop(remove_idx)
                kept.append(add_seg)
                dropped.append(removed)

        
        current_dur = sum(s.end - s.start for s in kept)
        deps_violated = False
        kept_scene_ids = {sid for s in kept for sid in s.scene_ids}
        for dep in deps:
            if dep["payoff_scene"] in kept_scene_ids and dep["setup_scene"] not in kept_scene_ids:
                deps_violated = True

        if (target_duration is not None and current_dur > target_duration * 1.15) or deps_violated:
            console.print("    [yellow]Story swaps violated invariants — Reverting swaps[/yellow]")
            kept = pre_swap_kept
            dropped = pre_swap_dropped

        kept.sort(key=lambda s: s.start)

    
    if response.ordering and intent.style.ordering != "chronological":
        order_map = {sid: idx for idx, sid in enumerate(response.ordering)}
        kept.sort(key=lambda s: min((order_map.get(sid, 999) for sid in s.scene_ids), default=999))

    return kept

def maybe_run_story_agent(
    intent: EditIntent,
    retrieval_results: list[RetrievalResult],
    db: ProjectDB,
) -> list[RetrievalResult]:
    if not _needs_story_agent(intent):
        return retrieval_results

    console.print("  Story Agent: optimizing narrative structure...")
    video = db.get_video()
    total_duration = video["duration_s"] if video else 999.0
    scenes = db.get_scenes()
    target_duration = intent.constraints.target_duration_s

    positive_op_indices = [
        i for i, op in enumerate(intent.operations) 
        if op.action in ("keep_only", "compress", "highlight", "restructure") and i < len(retrieval_results)
    ]
    
    if not positive_op_indices:
        return retrieval_results

    all_segments = []
    has_compress = False
    
    for i in positive_op_indices:
        op = intent.operations[i]
        result = retrieval_results[i]
        if op.action == "compress":
            has_compress = True
        if result.segments:
            all_segments.extend(result.segments)
            
    if has_compress:
        from trim_engine.query.retrieval import Segment
        from trim_engine.schemas import Evidence
        existing_sids = {sid for s in all_segments for sid in s.scene_ids}
        for scene in scenes:
            if scene["id"] not in existing_sids:
                all_segments.append(Segment(
                    start=scene["start_time"],
                    end=scene["end_time"],
                    scene_ids=[scene["id"]],
                    score=0.1,  
                    evidence=[Evidence(source="metadata", detail=f"story_role={scene.get('story_role', 'none')}")]
                ))

    if not all_segments:
        return retrieval_results

    kept, dropped = _knapsack_selection(all_segments, scenes, target_duration, total_duration)

    style_key = intent.style.narrative_shape or intent.style.ordering or "chronological"

    if style_key not in ("trailer", "highlight"):
        kept, dropped, tradeoffs = _enforce_dependencies(kept, dropped, db, target_duration)
        if tradeoffs:
            console.print(f"    [yellow]Story Agent surfaced {len(tradeoffs)} tradeoff warnings[/yellow]")
    else:
        console.print(f"    [dim]Skipping dependency enforcement for '{style_key}' shape[/dim]")

    if style_key in NARRATIVE_TEMPLATES:
        kept = NARRATIVE_TEMPLATES[style_key](kept, db)

    readiness = db.get_readiness_level()
    if readiness >= 4:
        try:
            kept = _call_story_reasoning(intent, kept, dropped, db)
        except Exception as e:
            console.print(f"    [yellow]Claude taste call failed: {e}. Using solver output.[/yellow]")

    if intent.style.cut_style == "match_cut":
        kept = _order_for_match_cuts(kept, db)

    chrono = sorted(kept, key=lambda s: s.start)
    is_reordered = [s.start for s in kept] != [s.start for s in chrono]

    if is_reordered:
        console.print(f"    [dim]Story agent produced non-chronological ordering ({style_key})[/dim]")

    for i in positive_op_indices:
        result = retrieval_results[i]
        if is_reordered:
            result.ordered_segments = list(kept)
        result.segments = kept

    return retrieval_results
