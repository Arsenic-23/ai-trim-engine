"""
Critic / Validator (§5.5) — fresh-context plan validation.

Three Tiers of verification:
1. Tier 0 — Structural assertions & boundaries (always, <10 ms)
2. Tier 1 — Semantic re-query checking for leftover content (always, ~200 ms)
3. Tier 2 — LLM cognitive satisfaction & coherence judgment (per stage-plan, ~3 s)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.llm import build_video_summary, call_structured
from trim_engine.schemas import (
    CriticFailure, CriticVerdict, EditIntent, EditPlan, RetrievalResult,
)

console = Console()

RetryHandler = Callable[
    [str, list[CriticFailure], int],
    tuple[EditPlan, list[RetrievalResult]] | None,
]

def check_plan_flapping(
    current_plan: EditPlan,
    previous_plans: list[EditPlan],
) -> bool:
    """
    Compares current plan with historical plans.
    If the plan operations did not change across retries, short-circuits.
    """
    for prev in previous_plans:
        if len(prev.operations) != len(current_plan.operations):
            continue
        match = True
        for op1, op2 in zip(prev.operations, current_plan.operations):
            if abs(op1.range_start - op2.range_start) > 0.05 or abs(op1.range_end - op2.range_end) > 0.05:
                match = False
                break
        if match:
            return True
    return False

def _run_deterministic_checks(
    intent: EditIntent,
    edit_plan: EditPlan,
    db: ProjectDB,
    timeline: Timeline | None = None,
) -> list[CriticFailure]:
    failures: list[CriticFailure] = []
    video = db.get_video()
    if not video:
        return failures

    original_duration = video["duration_s"]
    predicted_duration = edit_plan.predicted_output_duration_s

    
    if intent.constraints.target_duration_s is not None:
        target = intent.constraints.target_duration_s
        mode = intent.constraints.duration_mode or "approx"

        if mode == "max" and predicted_duration > target:
            failures.append(CriticFailure(
                operation_index=0,
                issue=f"Output duration ({predicted_duration:.1f}s) exceeds maximum target limit ({target:.1f}s)",
                leftover_segments=None,
                route="story",
            ))
        elif mode == "exact" and abs(predicted_duration - target) > 2.0:
            failures.append(CriticFailure(
                operation_index=0,
                issue=f"Output duration ({predicted_duration:.1f}s) deviates from exact target limit ({target:.1f}s)",
                leftover_segments=None,
                route="story",
            ))
        elif mode == "approx" and predicted_duration > target * 1.15:
            failures.append(CriticFailure(
                operation_index=0,
                issue=f"Output duration ({predicted_duration:.1f}s) exceeds approximate target limit ({target:.1f}s) by more than 15%",
                leftover_segments=None,
                route="story",
            ))

    
    if edit_plan.predicted_output_duration_s < 1.0:
        failures.append(CriticFailure(
            operation_index=-1,
            issue=f"Output duration ({edit_plan.predicted_output_duration_s:.1f}s) is below the minimum output floor of 1.0s",
            leftover_segments=None,
            route="planner",
        ))

    
    is_compressive = any(op.action in ("compress", "keep_only") for op in intent.operations)
    if edit_plan.removal_ratio > 0.98 and not is_compressive:
        failures.append(CriticFailure(
            operation_index=0,
            issue=f"Excessive removal ratio ({edit_plan.removal_ratio:.1%}) detected without compressive intent.",
            leftover_segments=None,
            route="story",
        ))

    
    words = db.get_words()
    mid_word_cuts = []

    for op in edit_plan.operations:
        if op.type == "delete":
            for cut_point in (op.range_start, op.range_end):
                for w in words:
                    if w["start_time"] + 0.040 < cut_point < w["end_time"] - 0.040:
                        mid_word_cuts.append(
                            f"Cut at {cut_point:.3f}s cuts inside word '{w['word']}' ({w['start_time']:.2f}s-{w['end_time']:.2f}s)"
                        )

    if mid_word_cuts:
        failures.append(CriticFailure(
            operation_index=0,
            issue=f"Mid-word cut(s) detected: {'; '.join(mid_word_cuts[:3])}",
            leftover_segments=None,
            route="planner",
        ))

    # §2.4 Tier-0: Verify reordering actually happened for narrative shapes that require it
    narrative_shape = intent.style.narrative_shape if intent.style else None
    if narrative_shape in ("hook_first", "trailer") and edit_plan.clip_count > 1:
        if timeline and timeline.video_clips:
            src_ins = [clip.src_in for clip in timeline.video_clips]
            if src_ins == sorted(src_ins):
                # Clips are still in chronological order despite a non-chronological narrative shape
                failures.append(CriticFailure(
                    operation_index=0,
                    issue=f"Narrative shape '{narrative_shape}' requested but output clips are in chronological order — reordering was not applied",
                    leftover_segments=None,
                    route="story",
                ))

    return failures

def _run_semantic_requery_checks(
    intent: EditIntent,
    edit_plan: EditPlan,
    db: ProjectDB,
) -> list[CriticFailure]:
    failures: list[CriticFailure] = []
    video = db.get_video()
    total_duration = video["duration_s"] if video else 0.0

    removals = sorted(
        [(op.range_start, op.range_end) for op in edit_plan.operations if op.type == "delete"],
        key=lambda r: r[0],
    )

    keeps: list[tuple[float, float]] = []
    prev_end = 0.0
    for start, end in removals:
        if start > prev_end:
            keeps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < total_duration:
        keeps.append((prev_end, total_duration))

    for i, op in enumerate(intent.operations):
        if op.action == "remove":
            from trim_engine.query.retrieval import _structured_search
            hits = _structured_search(op.target.query, op.target.modality, db)
            
            leftovers = []
            for hit in hits:
                for k_start, k_end in keeps:
                    overlap = min(hit.end, k_end) - max(hit.start, k_start)
                    if overlap > 0.5:
                        leftovers.append(f"[{hit.start:.1f}s–{hit.end:.1f}s]")

            if leftovers:
                failures.append(CriticFailure(
                    operation_index=i,
                    issue=f"Leftover matching content found in kept ranges: {'; '.join(leftovers[:2])}",
                    leftover_segments=leftovers,
                    route="retrieval",
                ))

    return failures

def _build_critic_context(
    intent: EditIntent,
    edit_plan: EditPlan,
    db: ProjectDB,
) -> str:
    video_summary = build_video_summary(db)
    video = db.get_video()
    total_duration = video["duration_s"] if video else 0.0

    removals = sorted(
        [(op.range_start, op.range_end) for op in edit_plan.operations if op.type == "delete"],
        key=lambda r: r[0],
    )

    keeps: list[tuple[float, float]] = []
    prev_end = 0.0
    for start, end in removals:
        if start > prev_end:
            keeps.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < total_duration:
        keeps.append((prev_end, total_duration))

    words = db.get_words()
    scenes = db.get_scenes()
    kept_transcript_parts: list[str] = []

    for keep_start, keep_end in keeps:
        kept_words = [
            w for w in words
            if w["start_time"] >= keep_start and w["end_time"] <= keep_end
        ]
        kept_scenes = [
            s for s in scenes
            if min(s["end_time"], keep_end) - max(s["start_time"], keep_start) > 0.5
        ]
        
        for s in kept_scenes:
            caption = s.get("caption") or "Unknown visual scene"
            kept_transcript_parts.append(f"[{keep_start:.1f}s–{keep_end:.1f}s] [VISUAL SCENE: {caption}]")
            
        if kept_words:
            text = " ".join(w["word"] for w in kept_words)
            kept_transcript_parts.append(f"[{keep_start:.1f}s–{keep_end:.1f}s] {text}")

    kept_transcript = "\n".join(kept_transcript_parts)
    ops_summary = [f"Op {i}: {op.action} \"{op.target.query}\" ({op.priority})" for i, op in enumerate(intent.operations)]
    
    removal_summary = [
        f"  Removed {op.range_start:.1f}s–{op.range_end:.1f}s: {op.reason}"
        for op in edit_plan.operations if op.type == "delete"
    ]

    system_prompt = (
        f"You are a strict QA critic for an AI video editor.\n"
        f"Review if the EditPlan satisfies the EditIntent.\n"
        f"The video editor CAN reorder clips into non-chronological playback order.\n\n"
        f"Evaluate the plan strictly against the user's explicit intent:\n\n"
        f"1. Check if ANY explicitly requested removals are present in the output.\n"
        f"2. Check if ANY explicitly required retentions are missing from the output.\n"
        f"3. If the intent requests a narrative shape (hook_first, trailer), verify the output order reflects it.\n"
        f"4. Do NOT evaluate artistic merit or pacing, only objective intent satisfaction.\n\n"
        f"NOTE: Small fragments of removed scenes (e.g., < 3 seconds) might be kept to preserve adjacent speech boundaries. Do not flag these as failures.\n\n"
        f"If the plan fails, explain exactly why in the 'failures' list and set passed=false."
    )

    return (
        f"{system_prompt}\n\n"
        f"INTENT:\n"
        f"{'  '.join(ops_summary)}\n\n"
        f"EDIT PLAN:\n"
        f"  Predicted duration: {edit_plan.predicted_output_duration_s:.1f}s\n"
        f"  Removal ratio: {edit_plan.removal_ratio:.1%}\n"
        f"  Clips: {edit_plan.clip_count}\n"
        f"\nREMOVALS:\n{'  '.join(removal_summary)}\n\n"
        f"KEPT TRANSCRIPT (in final order):\n{kept_transcript}\n\n"
        f"VIDEO CONTEXT:\n{video_summary}"
    )

def validate_plan(
    intent: EditIntent,
    edit_plan: EditPlan,
    retrieval_results: list[RetrievalResult],
    db: ProjectDB,
    max_retries: int = 0,
    retry_handler: RetryHandler | None = None,
    timeline: Timeline | None = None,
) -> CriticVerdict:
    """
    Validate edit plan using structural (Tier 0), semantic (Tier 1), and LLM (Tier 2) checks.
    """
    def run_once(plan: EditPlan, retrieval: list[RetrievalResult]) -> CriticVerdict:
        console.print("  [dim]Running critic validation...[/dim]")

        
        failures = _run_deterministic_checks(intent, plan, db, timeline)

        
        semantic_failures = _run_semantic_requery_checks(intent, plan, db)
        failures.extend(semantic_failures)

        
        verdict = CriticVerdict(passed=True, failures=[], coherence_ok=True, notes="Initial pass")

        if db.get_readiness_level() >= 3:
            try:
                user_content = _build_critic_context(intent, plan, db)
                verdict = call_structured(
                    prompt_name="critic",
                    user_content=user_content,
                    schema=CriticVerdict,
                    effort="medium",
                    db=db,
                )
            except Exception as e:
                console.print(f"    [yellow]Claude LLM critic failed: {e}. Relying on deterministic checks.[/yellow]")

        
        if failures:
            verdict.failures.extend(failures)
            verdict.passed = False

        if verdict.passed:
            console.print("  [green]✓ Critic: PASS[/green]")
        else:
            console.print(f"  [yellow]⚠ Critic: FAIL ({len(verdict.failures)} issues)[/yellow]")

        return verdict

    verdict = run_once(edit_plan, retrieval_results)
    if verdict.passed or max_retries <= 0 or retry_handler is None:
        if not verdict.passed and verdict.failures:
            routes = ", ".join(sorted({f.route for f in verdict.failures}))
            suffix = f" Retry routes requested: {routes}."
            verdict.notes = f"{verdict.notes or ''}{suffix}".strip()
        return verdict

    seen_plan_hashes = {edit_plan.model_dump_json()}
    current_plan = edit_plan
    current_retrieval = retrieval_results

    for attempt in range(1, max_retries + 1):
        route = verdict.failures[0].route if verdict.failures else "planner"
        console.print(f"    [yellow]Critic retry {attempt}/{max_retries}: rerouting to {route}[/yellow]")

        updated = retry_handler(route, verdict.failures, attempt)
        if updated is None:
            verdict.notes = f"{verdict.notes or ''} Retry handler declined route '{route}'.".strip()
            break

        current_plan, current_retrieval = updated
        plan_hash = current_plan.model_dump_json()
        if plan_hash in seen_plan_hashes:
            verdict.notes = f"{verdict.notes or ''} Retry stopped: repeated edit plan.".strip()
            break
        seen_plan_hashes.add(plan_hash)

        verdict = run_once(current_plan, current_retrieval)
        if verdict.passed:
            verdict.notes = f"{verdict.notes or ''} Passed after critic retry {attempt}.".strip()
            break

    return verdict
