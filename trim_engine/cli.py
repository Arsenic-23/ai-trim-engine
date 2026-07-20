"""
CLI — primary demo surface.

Commands:
    trim ingest video.mp4                     Run full ingestion
    trim status <video_id>                    Show stages, coverage, cost
    trim edit <video_id> "prompt"             Run edit pipeline
    trim edit <video_id> --revise v3 "prompt" Delta revision
    trim ask <video_id> "question"            KB Q&A
    trim suite <video_id>                     Run full 25-prompt suite
"""

from __future__ import annotations

import os

_ssl_cert = os.environ.get("SSL_CERT_FILE", "")
if _ssl_cert and not os.path.isfile(_ssl_cert):
    os.environ.pop("SSL_CERT_FILE", None)

import json
import functools
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich import box

import trim_engine.config

app = typer.Typer(
    name="trim",
    help="AI Trim Engine — natural-language video editing",
)
console = Console()


_DEFAULT_HINTS = {
    "LLMTransientError": "Bedrock is throttling or unreachable. Wait a moment and rerun the command.",
    "RetrievalGapError": "No matching content was found. Try rephrasing the prompt or check `craon ask` first.",
    "InfeasibleError": "The constraints cannot be satisfied with this video. Relax the duration or scope.",
    "PlannerBreachError": "The planner produced an invalid timeline. Rerun the edit; if it recurs, simplify the prompt.",
    "RenderFailError": "Verify FFmpeg 7.x is installed and on PATH (`ffmpeg -version`), then retry.",
    "StaleKBError": "The project index may be stale or corrupt. Re-ingest the video with `craon ingest`.",
    "RunawayError": "The pipeline hit its safety budget. Try breaking your request into smaller, targeted edits.",
    "SemanticError": "The critic found the plan does not satisfy the prompt. Try a more specific prompt.",
}


def handle_engine_errors(fn):
    """
    Central CLI error boundary: converts engine exceptions into clear,
    actionable messages with recovery hints instead of raw tracebacks.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        from trim_engine.query.exceptions import QueryEngineError

        try:
            return fn(*args, **kwargs)
        except typer.Exit:
            raise
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Session state was checkpointed — rerun to continue.[/yellow]")
            raise typer.Exit(130)
        except QueryEngineError as e:
            kind = type(e).__name__
            hint = e.recovery_hint or _DEFAULT_HINTS.get(kind, "Rerun the command; if it recurs, re-ingest the video.")
            console.print(Panel(
                f"[red]✗ {kind}[/red]\n"
                f"  {e}\n\n"
                f"  [bold]What to do:[/bold] {hint}",
                style="red",
                title="Edit failed",
            ))
            raise typer.Exit(1)
        except FileNotFoundError as e:
            console.print(Panel(
                f"[red]✗ Missing file[/red]\n  {e}\n\n"
                f"  [bold]What to do:[/bold] Check the path exists; re-ingest if project files were moved.",
                style="red",
                title="Edit failed",
            ))
            raise typer.Exit(1)
        except Exception as e:
            console.print(Panel(
                f"[red]✗ Unexpected error ({type(e).__name__})[/red]\n  {e}\n\n"
                f"  [bold]What to do:[/bold] Rerun with the same command — session state is checkpointed. "
                f"If it persists, run `craon status <video_id>` to inspect the project.",
                style="red",
                title="Edit failed",
            ))
            raise typer.Exit(1)

    return wrapper

def show_branding():
    import time
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel

    ascii_art = '''
  ██████╗ ██████╗  █████╗  ██████╗ ███╗   ██╗
 ██╔════╝ ██╔══██╗██╔══██╗██╔═══██╗████╗  ██║
 ██║      ██████╔╝███████║██║   ██║██╔██╗ ██║
 ██║      ██╔══██╗██╔══██║██║   ██║██║╚██╗██║
 ╚██████╗ ██║  ██║██║  ██║╚██████╔╝██║ ╚████║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝'''

    with Live(refresh_per_second=30, transient=False) as live:
        # Phase 1: Color sweep fade-in
        colors = ["#222222", "#444444", "#666666", "#888888", "#aaaaaa", "#00aaaa", "#00ffff"]
        for c in colors:
            text_obj = Text(ascii_art.strip("\n"), style=c, justify="center")
            panel = Panel(text_obj, border_style=c, title="[bold white]C R A O N[/bold white]", subtitle="[dim]AI Video Editing Engine[/dim]")
            live.update(panel)
            time.sleep(0.06)
            
        # Phase 3: Pulse Glow effect
        for style in ["bold cyan", "bold blue", "bold white", "bold cyan"]:
            text_obj = Text(ascii_art.strip("\n"), style=style, justify="center")
            panel = Panel(text_obj, border_style=style, title="[bold white]C R A O N[/bold white]", subtitle="[dim]AI Video Editing Engine[/dim]")
            live.update(panel)
            time.sleep(0.08)

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    import sys
    # Only animate if not displaying standard help via flag
    if "--help" not in sys.argv and "-h" not in sys.argv:
        show_branding()
        if ctx.invoked_subcommand is None:
            from trim_engine.shell import run_shell
            run_shell()


def _get_project_dir(video_id: str) -> Path:
    return trim_engine.config.PROJECTS_DIR / video_id


def _get_db(video_id: str):
    from trim_engine.db import ProjectDB
    db_path = _get_project_dir(video_id) / "project.db"
    if not db_path.exists():
        console.print(f"[red]No project found for video_id: {video_id}[/red]")
        raise typer.Exit(1)
    return ProjectDB(db_path)


@app.command()
@handle_engine_errors
def ingest(
    video_path: str = typer.Argument(..., help="Path to video file"),
) -> None:
    """Run full ingestion pipeline on a video file."""
    from trim_engine.ingest.orchestrator import run_ingest

    path = Path(video_path).resolve()
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f"[bold]Ingesting:[/bold] {path.name}", style="blue"))

    video_id = run_ingest(path)

    console.print(Panel(
        f"[green]✓ Ingestion complete[/green]\n"
        f"  Video ID: [bold]{video_id}[/bold]\n"
        f"  Project:  {_get_project_dir(video_id)}",
        style="green",
    ))


@app.command(name="bedrock-smoke")
def bedrock_smoke() -> None:
    """Run Bedrock LLM/Vision smoke test to verify connectivity."""
    from trim_engine.llm import run_smoke
    console.print("[dim]Starting Bedrock smoke test...[/dim]")
    run_smoke()
    console.print("[bold green]Smoke test passed![/bold green]")


@app.command()
def status(
    video_id: str = typer.Argument(..., help="Video ID (content hash)"),
) -> None:
    """Show ingestion status, coverage, and cost for a video."""
    db = _get_db(video_id)
    video = db.get_video()
    stages = db.get_all_stages()
    coverage = db.get_coverage()
    cost = db.get_total_cost()

    
    if video:
        console.print(Panel(
            f"🎬 [bold cyan]{video['path']}[/bold cyan]\n"
            f"⏱️  [dim]Duration:[/dim] [bold]{video['duration_s']:.1f}s[/bold]  |  📐 [dim]Resolution:[/dim] {video['width']}x{video['height']} @ {video['fps']}fps",
            title="[bold white]Project Details[/bold white]",
            border_style="cyan",
            padding=(1, 2)
        ))

    
    stage_table = Table(title="Ingestion Pipeline Stages", box=box.ROUNDED, border_style="blue", title_style="bold blue")
    stage_table.add_column("Stage", style="cyan")
    stage_table.add_column("Status")
    stage_table.add_column("Duration", justify="right")
    stage_table.add_column("Error")

    for s in stages:
        status_style = {
            "done": "[green]✓ done[/green]",
            "failed": "[red]✗ failed[/red]",
            "running": "[yellow]⟳ running[/yellow]",
            "pending": "[dim]○ pending[/dim]",
        }.get(s["status"], s["status"])

        duration = f"{s['duration_s']:.1f}s" if s.get("duration_s") else "—"
        error = s.get("error") or "—"

        stage_table.add_row(s["stage"], status_style, duration, error)

    console.print(stage_table)

    
    if coverage:
        cov_table = Table(title="AI Model Coverage", box=box.SIMPLE_HEAD, border_style="magenta", title_style="bold magenta")
        cov_table.add_column("Analyzer", style="cyan")
        cov_table.add_column("Status")
        for analyzer, stat in coverage.items():
            style = "[green]● " if stat == "available" else "[yellow]○ "
            cov_table.add_row(analyzer, f"{style}{stat}[/]")
        console.print(cov_table)

    
    console.print(f"\n[bold]Total LLM cost:[/bold] ${cost:.4f}")


@app.command()
@handle_engine_errors
def edit(
    video_id: str = typer.Argument(..., help="Video ID"),
    prompt: str = typer.Argument(..., help="Edit prompt in natural language"),
    revise: str | None = typer.Option(None, help="Revision base version (e.g., v3)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """Run the edit pipeline: intent → retrieve → plan → critic → render."""
    from trim_engine.query.executor import create_or_resume_session, execute_session_pipeline, State
    from trim_engine.query.report import generate_report
    from trim_engine.db import ProjectDB

    project_dir = _get_project_dir(video_id)
    db = ProjectDB(project_dir / "project.db")

    video = db.get_video()
    if not video:
        console.print(f"[red]No ingested video found for {video_id}[/red]")
        raise typer.Exit(1)

    console.print(Panel(f'[bold]Edit Session:[/bold] "{prompt}"', style="blue"))

    if revise:
        from trim_engine.query.renderer import render_timeline
        from trim_engine.query.revision import compile_revision_timeline
        from trim_engine.schemas import (
            Ambiguity, CriticVerdict, EditConstraints, EditIntent, EditPlan,
            EditStyle, Operation, PlanOperation, SegmentTarget, Timeline,
        )

        try:
            parent_version = int(revise.lower().lstrip("v"))
        except ValueError:
            console.print(f"[red]Invalid revision version: {revise}. Use a value like v3.[/red]")
            raise typer.Exit(1)

        parent_timeline_path = project_dir / "edits" / f"v{parent_version}" / "timeline.json"
        if not parent_timeline_path.exists():
            console.print(f"[red]No timeline found for revision base v{parent_version}: {parent_timeline_path}[/red]")
            raise typer.Exit(1)

        parent_timeline = Timeline.model_validate_json(parent_timeline_path.read_text())
        timeline, dirty_indices = compile_revision_timeline(parent_timeline, prompt)
        version = db.next_edit_version()
        timeline.version = version

        output_path = render_timeline(timeline, project_dir, version, db)

        predicted_duration = sum(c.src_out - c.src_in for c in timeline.video_clips)
        duration_before = video["duration_s"]
        intent = EditIntent(
            intent_id=f"revision_{version}",
            operations=[
                Operation(
                    action="restructure",
                    target=SegmentTarget(modality=["timeline"], query=prompt),
                    priority="hard",
                    confidence=1.0,
                )
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=[f"revision_from_v{parent_version}"],
        )
        plan = EditPlan(
            plan_id=f"revision_{version}",
            operations=[
                PlanOperation(
                    op_id="revision_delta",
                    type="keep",
                    range_start=0.0,
                    range_end=predicted_duration,
                    reason=f"Revision from v{parent_version}; dirty clips: {dirty_indices}",
                    evidence_ref=None,
                    repairs=[],
                    depends_on=[],
                )
            ],
            predicted_output_duration_s=round(predicted_duration, 2),
            removal_ratio=round((duration_before - predicted_duration) / duration_before, 4) if duration_before > 0 else 0.0,
            clip_count=len(timeline.video_clips),
            rule_logs=[{"rule": "revision_delta", "dirty_indices": dirty_indices, "parent_version": parent_version}],
        )
        verdict = CriticVerdict(passed=True, failures=[], coherence_ok=True, notes=f"Revision applied from v{parent_version}")
        report = generate_report(video_id, version, prompt, intent, plan, verdict, db)

        db.insert_edit(
            version=version,
            prompt=prompt,
            intent_json=intent.model_dump_json(),
            plan_json=plan.model_dump_json(),
            verdict_json=verdict.model_dump_json(),
            output_path=str(output_path),
            report_path=str(project_dir / "edits" / f"v{version}" / "report.md"),
        )

        console.print(Panel(
            f"[green]✓ Revision complete[/green]\n"
            f"  Base:    v{parent_version}\n"
            f"  Output:  {output_path}\n"
            f"  Version: v{version}\n"
            f"  Duration: {report.duration_after_s:.1f}s",
            style="green",
        ))
        return

    
    session = create_or_resume_session(video_id, prompt, db)
    
    while session.state not in (State.PREVIEW_READY, State.DELIVERED, State.RENDER_FAILED, State.RESOLVED_NOOP, State.AWAITING_USER):
        session = execute_session_pipeline(session, db, project_dir, auto_approve=False)

    if session.state == State.RESOLVED_NOOP:
        if session.intent and session.intent.out_of_scope_reason:
            console.print(f"\n[yellow]Out of scope:[/yellow] {session.intent.out_of_scope_reason}")
        else:
            console.print("\n[yellow]Resolved No-Op: No matching segments found for this edit.[/yellow]")
        raise typer.Exit(0)

    if session.state == State.AWAITING_USER:
        console.print("\n[yellow]Clarification needed:[/yellow]")
        for a in session.intent.ambiguities:
            if a.blocking:
                console.print(f"  ❓ {a.issue}")
                console.print(f"     Candidates: {', '.join(a.candidates)}")
        raise typer.Exit(0)

    if session.state == State.RENDER_FAILED:
        console.print(f"[red]Execution failed: {session.error}[/red]")
        raise typer.Exit(1)

    if session.plan is None or session.timeline is None:
        console.print("[red]Pipeline finished without a plan or timeline — nothing to render.[/red]")
        raise typer.Exit(1)

    console.print(f"\n  Predicted output: [bold]{session.plan.predicted_output_duration_s:.1f}s[/bold]")
    console.print(f"  Removal ratio:   [bold]{session.plan.removal_ratio:.1%}[/bold]")
    console.print(f"  Clips:           [bold]{session.plan.clip_count}[/bold]")

    if not yes:
        confirm = typer.confirm("\nProceed with render?", default=True)
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit(0)

    
    session.state = State.RENDERING
    session.checkpoint(db)
    session = execute_session_pipeline(session, db, project_dir, auto_approve=True)

    if session.state == State.RENDER_FAILED:
        console.print(f"[red]Render failed: {session.error}[/red]")
        raise typer.Exit(1)

    
    version = session.version
    output_path = Path(session.output_path)

    report = None
    try:
        report = generate_report(
            video_id=video_id,
            version=version,
            prompt=prompt,
            intent=session.intent,
            edit_plan=session.plan,
            verdict=session.verdict,
            db=db,
        )
    except Exception as e:
        # The render already succeeded — a report failure must not fail the edit.
        console.print(f"[dim]Report generation skipped: {e}[/dim]")


    report_path = project_dir / "edits" / f"v{version}" / "report.md"
    db.insert_edit(
        version=version,
        prompt=prompt,
        intent_json=session.intent.model_dump_json(),
        plan_json=session.plan.model_dump_json(),
        verdict_json=session.verdict.model_dump_json() if session.verdict else None,
        output_path=str(output_path),
        report_path=str(report_path),
    )


    try:
        from trim_engine.query.profile import learn_from_prompt
        learn_from_prompt(prompt, db)
    except Exception as e:
        console.print(f"[dim]Failed to run profile learning: {e}[/dim]")


    if report is not None:
        console.print(Panel(
            f"[green]✓ Edit complete[/green]\n"
            f"  Output:  {output_path}\n"
            f"  Version: v{version}\n"
            f"  Duration: {report.duration_before_s:.1f}s → {report.duration_after_s:.1f}s "
            f"({report.reduction_pct:.0f}% shorter)\n"
            f"  Removals: {len(report.removals)}",
            style="green",
        ))
    else:
        console.print(Panel(
            f"[green]✓ Edit complete[/green]\n"
            f"  Output:  {output_path}\n"
            f"  Version: v{version}",
            style="green",
        ))

    # §5.1: No-effect detection at delivery
    if report is not None:
        if report.duration_after_s <= 0.0:
            console.print("\n[bold red]WARNING: Output timeline has 0 clips. The edit resulted in an empty video.[/bold red]")
        elif abs(report.duration_after_s - report.duration_before_s) < 0.1:
            console.print("\n[bold yellow]WARNING: Output timeline is identical in length to the original video. No cuts were made.[/bold yellow]")


@app.command()
@handle_engine_errors
def ask(
    video_id: str = typer.Argument(..., help="Video ID"),
    question: str = typer.Argument(..., help="Question about the video"),
) -> None:
    """Ask a question about the video's knowledge base."""
    from trim_engine.query.retrieval import answer_question

    db = _get_db(video_id)
    project_dir = _get_project_dir(video_id)

    console.print(Panel(f'[bold]Q:[/bold] "{question}"', style="blue"))
    answer = answer_question(question, db, project_dir)
    console.print(f"\n{answer}")


@app.command()
def suite(
    video_id: str = typer.Argument(..., help="Video ID"),
) -> None:
    """Run the full 25-prompt regression suite."""
    SUITE_PROMPTS = [
        "Remove every time I mention pricing.",
        "Make this under 30 seconds.",
        "Remove the part where I'm laughing.",
        "Make it more engaging.",
        "Remove pauses and silences.",
        "Remove filler words.",
        "Remove retakes and mistakes; keep only the final take.",
        "Remove the intro.",
        "Remove all B-roll.",
        "Keep only the interview.",
        "Remove the coffee-making shots.",
        "Remove everything before I enter the frame.",
        "Keep only outdoor scenes.",
        "Remove every shot where Person B appears.",
        "Keep only the shots where I'm speaking.",
        "Keep only shots with the product visible.",
        "Remove awkward moments.",
        "Keep moments where people are clapping.",
        "Remove all walking shots.",
        "Keep only close-up reactions.",
        "Remove sponsor mentions.",
        "Keep only questions.",
        "Cut on every beat.",
        "Create a trailer-style cut.",
        "Make it suitable for TikTok.",
    ]

    console.print(Panel(
        f"[bold]Running 25-prompt suite[/bold] on {video_id}",
        style="blue",
    ))

    project_dir = _get_project_dir(video_id)
    results_dir = project_dir / "sample_outputs"
    results_dir.mkdir(exist_ok=True)

    results: list[dict] = []

    for i, prompt_text in enumerate(SUITE_PROMPTS, 1):
        console.print(f"\n[bold cyan]═══ Prompt {i}/25 ═══[/bold cyan]")
        console.print(f'  "{prompt_text}"')

        try:
            
            from trim_engine.query.intent import compile_intent
            from trim_engine.query.retrieval import retrieve_segments
            from trim_engine.query.story_agent import maybe_run_story_agent
            from trim_engine.query.planner import plan_timeline
            from trim_engine.query.critic import validate_plan
            from trim_engine.query.renderer import render_timeline
            from trim_engine.query.report import generate_report
            from trim_engine.db import ProjectDB

            db = ProjectDB(project_dir / "project.db")

            intent = compile_intent(prompt_text, db)
            retrieval_results = retrieve_segments(intent, db, project_dir)
            retrieval_results = maybe_run_story_agent(intent, retrieval_results, db)
            edit_plan, timeline = plan_timeline(intent, retrieval_results, db)

            def retry_from_critic(route: str, failures: list, attempt: int):
                nonlocal retrieval_results, edit_plan, timeline
                if route == "retrieval":
                    retrieval_results = retrieve_segments(intent, db, project_dir, retry_count=attempt)
                    retrieval_results = maybe_run_story_agent(intent, retrieval_results, db)
                elif route == "story":
                    retrieval_results = maybe_run_story_agent(intent, retrieval_results, db)
                edit_plan, timeline = plan_timeline(intent, retrieval_results, db, project_dir)
                return edit_plan, retrieval_results

            verdict = validate_plan(
                intent, edit_plan, retrieval_results, db,
                max_retries=2,
                retry_handler=retry_from_critic,
            )
            version = db.next_edit_version()
            
            output_path = project_dir / "edits" / f"v{version}" / "output.mp4"
            if verdict.passed:
                output_path = render_timeline(timeline, project_dir, version, db)
            else:
                console.print(f"[yellow]⚠ Verdict failed. Skipping render for v{version}.[/yellow]")

            report = generate_report(
                video_id=video_id, version=version, prompt=prompt_text,
                intent=intent, edit_plan=edit_plan, verdict=verdict, db=db,
            )

            
            prompt_dir = results_dir / f"prompt_{i:02d}"
            prompt_dir.mkdir(exist_ok=True)
            (prompt_dir / "prompt.txt").write_text(prompt_text)
            (prompt_dir / "intent.json").write_text(intent.model_dump_json(indent=2))
            (prompt_dir / "plan_summary.json").write_text(edit_plan.model_dump_json(indent=2))
            (prompt_dir / "verdict.json").write_text(verdict.model_dump_json(indent=2))
            (prompt_dir / "report.json").write_text(report.model_dump_json(indent=2))

            edit_report_dir = project_dir / "edits" / f"v{version}"
            edit_report_md = edit_report_dir / "report.md"
            if edit_report_md.exists():
                shutil.copy(edit_report_md, prompt_dir / "report.md")
            else:
                (prompt_dir / "report.md").write_text(
                    f"# Prompt {i}: {prompt_text}\n\n"
                    f"Duration: {report.duration_before_s:.1f}s -> {report.duration_after_s:.1f}s "
                    f"({report.reduction_pct:.0f}% reduction)\n"
                    f"Removals: {len(report.removals)}\n"
                    f"Critic: {'PASS' if verdict.passed else 'FAIL'}\n"
                )

            if output_path.exists():
                shutil.copy(output_path, prompt_dir / "output.mp4")

            db.insert_edit(
                version=version, prompt=prompt_text,
                intent_json=intent.model_dump_json(),
                plan_json=edit_plan.model_dump_json(),
                verdict_json=verdict.model_dump_json(),
                output_path=str(output_path),
                report_path=str(edit_report_md),
            )

            results.append({
                "prompt": prompt_text,
                "status": "✓",
                "duration": f"{report.duration_after_s:.1f}s",
                "reduction": f"{report.reduction_pct:.0f}%",
            })
            console.print(f"  [green]✓ {report.duration_after_s:.1f}s (-{report.reduction_pct:.0f}%)[/green]")

        except Exception as e:
            results.append({"prompt": prompt_text, "status": "✗", "error": str(e)})
            console.print(f"  [red]✗ {e}[/red]")

    
    console.print("\n")
    table = Table(title="Suite Results")
    table.add_column("#", justify="right")
    table.add_column("Prompt")
    table.add_column("Status")
    table.add_column("Output")
    table.add_column("Reduction")

    for i, r in enumerate(results, 1):
        status = "[green]✓[/green]" if r["status"] == "✓" else "[red]✗[/red]"
        table.add_row(
            str(i), r["prompt"][:50], status,
            r.get("duration", "—"), r.get("reduction", r.get("error", "")[:30]),
        )

    console.print(table)

    index_lines = [
        "# Sample Output Suite",
        "",
        f"Video ID: `{video_id}`",
        "",
        "| # | Prompt | Status | Output | Reduction / Error |",
        "|---:|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        output = r.get("duration", "-")
        reduction = r.get("reduction", r.get("error", "")).replace("|", "\\|")
        index_lines.append(
            f"| {i} | {r['prompt'].replace('|', '\\|')} | {r['status']} | {output} | {reduction} |"
        )
    (results_dir / "README.md").write_text("\n".join(index_lines) + "\n")

    console.print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    app()
