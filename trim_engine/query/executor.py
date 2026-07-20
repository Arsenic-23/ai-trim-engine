from __future__ import annotations

import json
import time
import uuid
import hashlib
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console

from trim_engine.db import ProjectDB
from trim_engine.schemas import EditIntent, RetrievalResult, EditPlan, Timeline, CriticVerdict
from trim_engine.query.intent import compile_intent
from trim_engine.query.retrieval import retrieve_segments
from trim_engine.query.story_agent import maybe_run_story_agent
from trim_engine.query.planner import plan_timeline
from trim_engine.query.critic import validate_plan
from trim_engine.query.renderer import render_timeline
from trim_engine.query.exceptions import (
    QueryEngineError, LLMTransientError, SemanticError, RetrievalGapError,
    InfeasibleError, PlannerBreachError, RenderFailError, StaleKBError, RunawayError
)

console = Console()

class State(str, Enum):
    CREATED = "created"
    COMPILING = "compiling"
    RETRIEVING = "retrieving"
    REASONING = "reasoning"
    PLANNING = "planning"
    VALIDATING = "validating"
    PREVIEW_READY = "preview_ready"
    RENDERING = "rendering"
    DELIVERED = "delivered"
    RENDER_FAILED = "render_failed"
    AWAITING_USER = "awaiting_user"
    RESOLVED_NOOP = "resolved_noop"

class QueryCache:
    def __init__(self, project_dir: Path):
        self.cache_dir = project_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "query_cache.json"
        self.data = {}
        if self.cache_file.exists():
            try:
                self.data = json.loads(self.cache_file.read_text())
            except Exception:
                pass

    def _key(self, video_id: str, prompt: str) -> str:
        return hashlib.sha256(f"{video_id}:{prompt}".encode()).hexdigest()

    def get(self, video_id: str, prompt: str) -> dict | None:
        key = self._key(video_id, prompt)
        return self.data.get(key)

    def set(self, video_id: str, prompt: str, value: dict) -> None:
        key = self._key(video_id, prompt)
        self.data[key] = value
        try:
            self.cache_file.write_text(json.dumps(self.data, indent=2))
        except Exception:
            pass


class BudgetEnvelope:
    def __init__(
        self,
        max_llm_calls: int = 24,
        max_retries_total: int = 10,
        max_wall_clock_s: float = 900.0,
        max_usd_cost: float = 1.00,
    ):
        self.max_llm_calls = max_llm_calls
        self.max_retries_total = max_retries_total
        self.max_wall_clock_s = max_wall_clock_s
        self.max_usd_cost = max_usd_cost
        self.llm_calls = 0
        self.retries = 0
        self.usd_cost = 0.0
        self.start_time = time.time()

    def record_llm_call(self, cost: float = 0.0) -> None:
        self.llm_calls += 1
        self.usd_cost += cost
        if self.llm_calls > self.max_llm_calls:
            raise RunawayError(f"Budget exceeded: LLM call limit ({self.max_llm_calls}) reached")
        if self.usd_cost > self.max_usd_cost:
            raise RunawayError(f"Budget exceeded: USD cost limit (${self.max_usd_cost:.2f}) reached")

    def record_retry(self) -> None:
        self.retries += 1
        if self.retries > self.max_retries_total:
            raise RunawayError(f"Budget exceeded: Retry limit ({self.max_retries_total}) reached")

    def check_wall_clock(self) -> None:
        elapsed = time.time() - self.start_time
        if elapsed > self.max_wall_clock_s:
            raise RunawayError(f"Budget exceeded: Pre-render wall-clock limit ({self.max_wall_clock_s}s) reached")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_llm_calls": self.max_llm_calls,
            "max_retries_total": self.max_retries_total,
            "max_wall_clock_s": self.max_wall_clock_s,
            "max_usd_cost": self.max_usd_cost,
            "llm_calls": self.llm_calls,
            "retries": self.retries,
            "usd_cost": round(self.usd_cost, 4),
            "elapsed_s": round(time.time() - self.start_time, 2),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], start_time: float | None = None) -> BudgetEnvelope:
        b = cls(
            max_llm_calls=d.get("max_llm_calls", 24),
            max_retries_total=d.get("max_retries_total", 10),
            max_wall_clock_s=d.get("max_wall_clock_s", 900.0),
        )
        b.llm_calls = d.get("llm_calls", 0)
        b.retries = d.get("retries", 0)
        if start_time is not None:
            b.start_time = start_time
        else:
            b.start_time = time.time() - d.get("elapsed_s", 0.0)
        return b

class EditSession:
    def __init__(
        self,
        session_id: str,
        video_id: str,
        prompt: str,
        state: State = State.CREATED,
        version: int | None = None,
        budget: BudgetEnvelope | None = None,
        intent: EditIntent | None = None,
        retrieval: list[RetrievalResult] | None = None,
        plan: EditPlan | None = None,
        timeline: Timeline | None = None,
        verdict: CriticVerdict | None = None,
        output_path: str | None = None,
        preview_path: str | None = None,
        report_path: str | None = None,
        error: str | None = None,
    ):
        self.session_id = session_id
        self.video_id = video_id
        self.prompt = prompt
        self.state = state
        self.version = version
        self.budget = budget or BudgetEnvelope()
        self.intent = intent
        self.retrieval = retrieval
        self.plan = plan
        self.timeline = timeline
        self.verdict = verdict
        self.output_path = output_path
        self.preview_path = preview_path
        self.report_path = report_path
        self.error = error
        self.previous_timelines: set[str] = set()

    def checkpoint(self, db: ProjectDB) -> None:
        session_json = json.dumps(self.to_dict())
        budget_json = json.dumps(self.budget.to_dict())
        db.insert_edit_session(
            session_id=self.session_id,
            video_id=self.video_id,
            state=self.state.value,
            prompt=self.prompt,
            version=self.version,
            budget_json=budget_json,
            session_json=session_json,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "video_id": self.video_id,
            "prompt": self.prompt,
            "state": self.state.value,
            "version": self.version,
            "intent": self.intent.model_dump() if self.intent else None,
            "retrieval": [r.model_dump() for r in self.retrieval] if self.retrieval else None,
            "plan": self.plan.model_dump() if self.plan else None,
            "timeline": self.timeline.model_dump() if self.timeline else None,
            "verdict": self.verdict.model_dump() if self.verdict else None,
            "output_path": self.output_path,
            "preview_path": self.preview_path,
            "report_path": self.report_path,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], budget_d: dict[str, Any] | None = None) -> EditSession:
        intent = EditIntent.model_validate(d["intent"]) if d.get("intent") else None
        retrieval = [RetrievalResult.model_validate(r) for r in d["retrieval"]] if d.get("retrieval") else None
        plan = EditPlan.model_validate(d["plan"]) if d.get("plan") else None
        timeline = Timeline.model_validate(d["timeline"]) if d.get("timeline") else None
        verdict = CriticVerdict.model_validate(d["verdict"]) if d.get("verdict") else None

        budget = BudgetEnvelope.from_dict(budget_d) if budget_d else None

        return cls(
            session_id=d["session_id"],
            video_id=d["video_id"],
            prompt=d["prompt"],
            state=State(d["state"]),
            version=d.get("version"),
            budget=budget,
            intent=intent,
            retrieval=retrieval,
            plan=plan,
            timeline=timeline,
            verdict=verdict,
            output_path=d.get("output_path"),
            preview_path=d.get("preview_path"),
            report_path=d.get("report_path"),
            error=d.get("error"),
        )

def compile_stage_plan(intent: EditIntent) -> list[State]:
    """
    Compiles intent class to custom list of states (Pipeline shapes per §2.2).
    """
    if intent.out_of_scope_reason:
        return [State.RESOLVED_NOOP]
    if any(amb.blocking for amb in intent.ambiguities):
        return [State.AWAITING_USER]

    
    if intent.style.narrative_shape == "q_and_a":
        return [State.RESOLVED_NOOP]

    
    is_mechanical = all(
        op.action in ("remove", "keep_only") and any(kw in op.target.query.lower() for kw in ("filler", "silence", "pause", "um", "uh"))
        for op in intent.operations
    )
    if is_mechanical:
        return [State.RETRIEVING, State.PLANNING, State.PREVIEW_READY, State.RENDERING, State.DELIVERED]

    
    is_narrative = any(
        op.action in ("compress", "summarize", "reorder", "highlight")
        or intent.style.narrative_shape in ("hook_first", "trailer", "highlight")
        for op in intent.operations
    )
    if is_narrative:
        return [State.RETRIEVING, State.REASONING, State.PLANNING, State.VALIDATING, State.PREVIEW_READY, State.RENDERING, State.DELIVERED]

    
    return [State.RETRIEVING, State.PLANNING, State.VALIDATING, State.PREVIEW_READY, State.RENDERING, State.DELIVERED]

def get_next_state(current_state: State, stage_plan: list[State]) -> State:
    if current_state == State.CREATED:
        return State.COMPILING
    if current_state == State.COMPILING:
        return stage_plan[0] if stage_plan else State.RESOLVED_NOOP
    if current_state in stage_plan:
        idx = stage_plan.index(current_state)
        if idx + 1 < len(stage_plan):
            return stage_plan[idx + 1]
        else:
            return State.DELIVERED
    return State.RENDER_FAILED

def execute_session_pipeline(
    session: EditSession,
    db: ProjectDB,
    project_dir: Path,
    auto_approve: bool = False,
) -> EditSession:
    """
    Executes the edit pipeline utilizing dynamic compiled stage plans and caching.
    """
    cache = QueryCache(project_dir)

    try:
        
        if session.state == State.CREATED:
            cached_val = cache.get(session.video_id, session.prompt)
            if cached_val:
                try:
                    restored = EditSession.from_dict(cached_val, session.budget.to_dict())
                except Exception:
                    restored = None
                # Only fast-restore when the cached session is actually renderable.
                if restored and restored.timeline and restored.timeline.video_clips and Path(restored.timeline.source).exists():
                    console.print("  [dim]Query Cache Hit! Fast-restoring session state...[/dim]")
                    session.intent = restored.intent
                    session.retrieval = restored.retrieval
                    session.plan = restored.plan
                    session.timeline = restored.timeline
                    session.verdict = restored.verdict
                    session.state = State.PREVIEW_READY
                    session.checkpoint(db)

        max_loop_iterations = 50  # hard backstop against state-machine stalls
        loop_iterations = 0
        while session.state not in (State.DELIVERED, State.RENDER_FAILED, State.RESOLVED_NOOP, State.AWAITING_USER):
            loop_iterations += 1
            if loop_iterations > max_loop_iterations:
                raise RunawayError(
                    f"Pipeline state machine exceeded {max_loop_iterations} transitions (stuck in '{session.state.value}').",
                    recovery_hint="Rerun the edit; if it recurs, try a simpler prompt.",
                )
            session.budget.check_wall_clock()

            stage_plan = compile_stage_plan(session.intent) if session.intent else []
            next_step = get_next_state(session.state, stage_plan)

            if session.state in (State.CREATED, State.COMPILING):
                session.state = State.COMPILING
                session.checkpoint(db)
                session.intent = compile_intent(session.prompt, db)
                session.budget.record_llm_call()

                if session.intent.out_of_scope_reason:
                    session.state = State.RESOLVED_NOOP
                    session.checkpoint(db)
                    break
                if any(amb.blocking for amb in session.intent.ambiguities):
                    session.state = State.AWAITING_USER
                    session.checkpoint(db)
                    break

                session.state = get_next_state(session.state, compile_stage_plan(session.intent))
                session.checkpoint(db)

            elif session.state == State.RETRIEVING:
                session.retrieval = retrieve_segments(session.intent, db, project_dir, retry_count=session.budget.retries)
                if all(res.no_match for res in session.retrieval):
                    session.state = State.RESOLVED_NOOP
                    session.checkpoint(db)
                    break

                session.state = next_step
                session.checkpoint(db)

            elif session.state == State.REASONING:
                session.retrieval = maybe_run_story_agent(session.intent, session.retrieval, db)
                session.state = next_step
                session.checkpoint(db)

            elif session.state == State.PLANNING:
                session.plan, session.timeline = plan_timeline(session.intent, session.retrieval or [], db, project_dir)
                session.state = next_step
                session.checkpoint(db)

            elif session.state == State.VALIDATING:
                session.verdict = validate_plan(session.intent, session.plan, session.retrieval or [], db, timeline=session.timeline)

                video_meta = db.get_video()
                duration_before = video_meta["duration_s"] if video_meta else 0.0
                duration_after = session.plan.predicted_output_duration_s
                removal_ratio = (duration_before - duration_after) / duration_before if duration_before > 0 else 0
                
                has_hard_ops = any(op.action in ("remove", "keep_only") for op in session.intent.operations)
                has_reorder = any(op.action == "reorder" for op in session.intent.operations)
                
                if removal_ratio < 0.01 and has_hard_ops and not has_reorder:
                    session.state = State.RESOLVED_NOOP
                    session.intent.out_of_scope_reason = "No material was removed (removal ratio < 1%) for a hard edit operation."
                    session.checkpoint(db)
                    console.print(f"\n[yellow]Out of scope: {session.intent.out_of_scope_reason}[/yellow]")
                    return session
                
                if session.verdict.passed:
                    session.state = State.PREVIEW_READY
                    
                    cache.set(session.video_id, session.prompt, session.to_dict())
                    session.checkpoint(db)
                else:
                    console.print(f"  [yellow]⚠ Critic: FAIL ({len(session.verdict.failures)} issues)[/yellow]")
                    for i, failure in enumerate(session.verdict.failures):
                        console.print(f"    {i+1}. {failure.issue} (route: {failure.route})")
                    session.budget.record_retry()
                    
                    timeline_str = session.timeline.model_dump_json()
                    timeline_hash = hashlib.sha256(timeline_str.encode()).hexdigest()
                    
                    if timeline_hash in session.previous_timelines:
                        raise RunawayError(
                            "Critic retry oscillation (flapping) detected. Failing plan.",
                            recovery_hint="Try breaking your query into smaller, targeted edits."
                        )
                        
                    session.previous_timelines.add(timeline_hash)

                    route = session.verdict.failures[0].route if session.verdict.failures else "planner"
                    console.print(f"    [yellow]Retry routing initiated: {route}[/yellow]")
                    
                    if route == "retrieval":
                        next_state = State.RETRIEVING
                    elif route == "story":
                        next_state = State.REASONING
                    else:
                        next_state = State.PLANNING
                        
                    if next_state not in stage_plan:
                        
                        
                        console.print(f"    [dim]Critic requested '{route}' but stage not in plan — falling back to PLANNING[/dim]")
                        next_state = State.PLANNING
                        if next_state not in stage_plan:
                            raise QueryEngineError(f"Critic retry routing failed: PLANNING stage also not in execution plan.")
                        
                    session.state = next_state
                        
                    session.checkpoint(db)

            elif session.state == State.PREVIEW_READY:
                from trim_engine.query.preview import generate_preview
                try:
                    session.preview_path = str(generate_preview(session.timeline, project_dir))
                except Exception as e:
                    # Preview is a convenience artifact — never block the render on it.
                    console.print(f"    [dim]Preview generation skipped: {e}[/dim]")
                    session.preview_path = None
                if auto_approve:
                    session.state = State.RENDERING
                    session.checkpoint(db)
                else:
                    session.checkpoint(db)
                    break

            elif session.state == State.RENDERING:
                session.version = db.next_edit_version()
                session.output_path = str(render_timeline(session.timeline, project_dir, session.version, db))
                from trim_engine.query.report import generate_report
                session.report_path = str(project_dir / "edits" / f"v{session.version}" / "report.md")
                try:
                    generate_report(
                        video_id=session.video_id,
                        version=session.version,
                        prompt=session.prompt,
                        intent=session.intent,
                        edit_plan=session.plan,
                        verdict=session.verdict,
                        db=db,
                    )
                except Exception as e:
                    # The video is already rendered — a report failure must not fail delivery.
                    console.print(f"    [dim]Report generation skipped: {e}[/dim]")

                try:
                    from trim_engine.query.profile import learn_from_prompt
                    learn_from_prompt(session.prompt, db)
                except Exception as e:
                    console.print(f"    [dim]Profile learning skipped: {e}[/dim]")
                session.state = State.DELIVERED
                session.checkpoint(db)

    except RunawayError as e:
        session.error = str(e)
        session.state = State.RENDER_FAILED
        session.checkpoint(db)
        console.print(f"    [red]Runaway limit reached: {e}[/red]")
        raise e
    except QueryEngineError as e:
        # Already classified — preserve the type and recovery hint.
        session.error = str(e)
        session.state = State.RENDER_FAILED
        session.checkpoint(db)
        raise
    except Exception as e:
        session.error = str(e)
        session.state = State.RENDER_FAILED
        session.checkpoint(db)

        msg = str(e).lower()
        if "ffprobe" in msg or "ffmpeg" in msg:
            raise RenderFailError(
                f"Rendering subsystem failed: {e}",
                recovery_hint="Verify FFmpeg 7.x is installed and on PATH (`ffmpeg -version`).",
            ) from e
        elif "sqlite3" in msg or "database" in msg:
            raise StaleKBError(
                f"VKB database error: {e}",
                recovery_hint="The project database may be corrupt or from an older version. Re-ingest the video.",
            ) from e
        else:
            raise QueryEngineError(f"Pipeline execution error: {e}") from e

    return session

def create_or_resume_session(video_id: str, prompt: str, db: ProjectDB) -> EditSession:
    """Supercede any existing active sessions, and return a clean session."""
    active = db.get_active_edit_sessions(video_id)
    for s_dict in active:
        console.print(f"  [dim]Superseding active session: {s_dict['id']}[/dim]")
        db.delete_edit_session(s_dict["id"])

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    session = EditSession(session_id=session_id, video_id=video_id, prompt=prompt)
    session.checkpoint(db)
    return session
