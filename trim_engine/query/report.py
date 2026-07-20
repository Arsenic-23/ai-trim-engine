"""
Edit Report Generator (§5.7) — transparency and trust surface.

Outputs Markdown + JSON alongside the rendered video.
Every claim links to evidence — this is what makes edits trustworthy.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from trim_engine.db import ProjectDB
from trim_engine.schemas import (
    CriticVerdict, EditIntent, EditPlan, EditReport, RemovalRecord,
)

console = Console()


def generate_report(
    video_id: str,
    version: int,
    prompt: str,
    intent: EditIntent,
    edit_plan: EditPlan,
    verdict: CriticVerdict,
    db: ProjectDB,
) -> EditReport:
    """
    Generate the edit report — accompanies every output.

    Contents:
    - Per-removal list (time range, reason, evidence quote)
    - Duration before/after
    - Unsatisfied ops (with why + nearest matches)
    - Continuity warnings
    - Profile preferences applied
    - Critic verdict summary
    - Cost of the run
    """
    video = db.get_video()
    duration_before = video["duration_s"] if video else 0.0
    duration_after = edit_plan.predicted_output_duration_s
    reduction = ((duration_before - duration_after) / duration_before * 100) if duration_before > 0 else 0

    
    removals: list[RemovalRecord] = []
    for op in edit_plan.operations:
        if op.type == "delete":
            
            evidence_quote = None
            words = db.get_words_in_range(op.range_start, op.range_end)
            if words:
                evidence_quote = " ".join(w["word"] for w in words[:20])
                if len(words) > 20:
                    evidence_quote += "..."

            removals.append(RemovalRecord(
                start=op.range_start,
                end=op.range_end,
                reason=op.reason,
                evidence_quote=evidence_quote,
            ))

    
    unsatisfied: list[str] = []
    if verdict:
        for failure in verdict.failures:
            unsatisfied.append(f"Op {failure.operation_index}: {failure.issue}")

    
    continuity_warnings: list[str] = []
    if verdict:
        if not verdict.coherence_ok:
            continuity_warnings.append("Potential coherence issues detected in kept content")
        if verdict.notes:
            continuity_warnings.append(verdict.notes)

    
    cost = db.get_total_cost()

    report = EditReport(
        video_id=video_id,
        version=version,
        prompt=prompt,
        duration_before_s=duration_before,
        duration_after_s=duration_after,
        reduction_pct=round(reduction, 1),
        removals=removals,
        unsatisfied_ops=unsatisfied,
        continuity_warnings=continuity_warnings,
        profile_preferences_applied=intent.profile_applied,
        critic_verdict_summary="BYPASSED (Mechanical)" if not verdict else ("PASS" if verdict.passed else f"FAIL: {len(verdict.failures)} issues"),
        cost_usd=cost,
        quality_metrics=None, # Will be populated in Stage 1/2
    )

    
    _write_markdown_report(report, db, version)

    return report


def _write_markdown_report(report: EditReport, db: ProjectDB, version: int) -> None:
    """Write a human-readable markdown report."""
    video = db.get_video()
    if not video:
        return

    project_dir = Path(video["path"]).parent
    report_dir = project_dir / "edits" / f"v{version}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    lines = [
        f"# Edit Report — v{version}",
        f"",
        f"**Prompt:** \"{report.prompt}\"",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Duration (before) | {report.duration_before_s:.1f}s ({report.duration_before_s/60:.1f} min) |",
        f"| Duration (after) | {report.duration_after_s:.1f}s ({report.duration_after_s/60:.1f} min) |",
        f"| Reduction | {report.reduction_pct:.1f}% |",
        f"| Removals | {len(report.removals)} |",
        f"| Critic | {report.critic_verdict_summary} |",
        f"| LLM Cost | ${report.cost_usd:.4f} |",
        f"",
    ]
    
    if report.quality_metrics:
        lines.extend([
            f"## Craft Quality Receipts",
            f"",
            f"- **Tempo Adherence**: {report.quality_metrics.tempo_curve_adherence_pct:.1f}%",
            f"- **Audio Sync Offset**: {report.quality_metrics.av_sync_offset_ms:.1f}ms",
            f"- **Cuts on Breath/Silence**: {report.quality_metrics.cuts_on_breath_or_silence_pct:.1f}%",
            f"- **LUFS Target Achieved**: {'Yes' if report.quality_metrics.lufs_target_achieved else 'No'}",
            f""
        ])

    if report.removals:
        lines.extend([
            f"## ✂ Removals",
            f"",
        ])

        for i, r in enumerate(report.removals, 1):
            lines.append(f"**{i}.** `{r.start:.1f}s – {r.end:.1f}s` — {r.reason}")
            if r.evidence_quote:
                lines.append(f"   > «{r.evidence_quote}»")
            lines.append("")

    if report.unsatisfied_ops:
        lines.extend([
            f"## ⚠ Unsatisfied Operations",
            f"",
        ])
        for op in report.unsatisfied_ops:
            lines.append(f"- {op}")
        lines.append("")

    if report.continuity_warnings:
        lines.extend([
            f"## 🔗 Continuity Warnings",
            f"",
        ])
        for w in report.continuity_warnings:
            lines.append(f"- {w}")
        lines.append("")

    if report.profile_preferences_applied:
        lines.extend([
            f"## 👤 Profile Preferences Applied",
            f"",
        ])
        for p in report.profile_preferences_applied:
            lines.append(f"- {p}")
        lines.append("")

    report_path.write_text("\n".join(lines))

    
    json_path = report_dir / "report.json"
    json_path.write_text(report.model_dump_json(indent=2))

    console.print(f"  Report: {report_path}")
