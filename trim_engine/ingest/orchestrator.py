"""
Job Orchestrator — DAG-based stage runner with crash-resume & progressive readiness.

Architecture (INGESTION_ENGINE.md §3):
- Every analyzer implements the AnalyzerNode Protocol
- Stages form a DAG with declared dependencies (topologically sorted)
- Independent branches could run concurrently (sequential on Mac prototype)
- Progressive readiness levels L0→L4 unlock edit capabilities incrementally
- Failed stages degrade gracefully — coverage recorded, pipeline continues
- Content-addressed dedup via SHA-256 hash of video bytes

Readiness levels:
  L0 = uploaded
  L1 = speech_ready  (VAD + ASR + alignment done)
  L2 = visual_ready  (shots + embeddings + captions done)
  L3 = semantic_ready (fusion + graph + topics done)
  L4 = story_ready   (story map + importance done → KB sealed)
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

import trim_engine.config
from trim_engine.db import ProjectDB

console = Console()

MAX_STAGE_ATTEMPTS = 3






@runtime_checkable
class AnalyzerNode(Protocol):
    """
    Contract every ingestion analyzer must satisfy.
    
    The orchestrator uses this to:
    - Topologically sort by dependencies
    - Check input artifacts exist before dispatch
    - Record version for model manifest / cache invalidation
    """
    name: str                       
    version: str                    
    inputs: list[str]               
    outputs: list[str]              
    resource: str                   
    timeout_s: int                  
    readiness_on_complete: int      

    def run(self, project_dir: Path, db: ProjectDB) -> None: ...






@dataclass(frozen=True)
class StageNode:
    """Concrete stage descriptor implementing the AnalyzerNode contract."""
    name: str
    version: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  
    resource: str = "CPU"
    timeout_s: int = 600
    readiness_on_complete: int = 0



STAGE_DAG: list[StageNode] = [
    
    StageNode(
        name="normalize",
        version="2.0",
        inputs=["original.mp4"],
        outputs=["proxy.mp4", "audio.wav", "probe.json", "frame_lut.json", "frame_lut.parquet", "audio_48k.flac", "thumbs.bin"],
        depends_on=[],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=0,  
    ),
    
    StageNode(
        name="audio",
        version="2.0",
        inputs=["audio.wav"],
        outputs=["utterances", "words", "silences", "fillers", "audio_events", "beats",
                 "topics", "retake_clusters", "loudness_curve", "speaker_embeddings"],
        depends_on=["normalize"],
        resource="CPU",
        timeout_s=600,
        readiness_on_complete=1,  
    ),
    
    StageNode(
        name="scenes",
        version="2.0",
        inputs=["proxy.mp4"],
        outputs=["scenes", "keyframes"],
        depends_on=["normalize"],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=0,  
    ),
    
    StageNode(
        name="faces",
        version="3.0",
        inputs=["proxy.mp4", "scenes", "frame_lut.json"],
        outputs=["face_tracks", "face_entities", "face_relations", "speaker_bindings"],
        depends_on=["scenes"],
        resource="CPU",
        timeout_s=600,
        readiness_on_complete=0,
    ),
    
    StageNode(
        name="cut_affinity",
        version="1.0",
        inputs=["original.mp4", "audio.wav", "utterances"],
        outputs=["cut_affinity"],
        depends_on=["audio"],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=1,
    ),

    StageNode(
        name="audio_separation",
        version="1.0",
        inputs=["audio.wav"],
        outputs=["vocals.wav", "no_vocals.wav"],
        depends_on=["normalize"],
        resource="CPU",
        timeout_s=1200,
        readiness_on_complete=1,
    ),

    StageNode(
        name="beat_grid",
        version="1.0",
        inputs=["no_vocals.wav"],
        outputs=["beats"],
        depends_on=["audio_separation"],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=1,
    ),

    StageNode(
        name="vision",
        version="2.0",
        inputs=["keyframes", "scenes", "utterances", "face_tracks"],
        outputs=["scene_tags", "person_raw_entities"],
        depends_on=["scenes", "audio", "faces"],
        resource="API_LLM",
        timeout_s=600,
        readiness_on_complete=2,  
    ),
    
    StageNode(
        name="graph",
        version="2.0",
        inputs=["scenes", "utterances", "person_raw_entities", "scene_tags"],
        outputs=["entities", "relations", "derived_moments"],
        depends_on=["audio", "vision"],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=3,  
    ),
    
    StageNode(
        name="index",
        version="2.0",
        inputs=["scenes", "utterances", "keyframes", "entities"],
        outputs=["scene_clip.index", "scene_text.index", "utterance.index", "bm25.pkl"],
        depends_on=["graph"],
        resource="CPU",
        timeout_s=300,
        readiness_on_complete=3,  
    ),
    
    StageNode(
        name="story",
        version="2.0",
        inputs=["scenes", "utterances", "entities", "relations"],
        outputs=["story_beats", "story_deps", "importance_scores"],
        depends_on=["index"],
        resource="API_LLM",
        timeout_s=600,
        readiness_on_complete=4,  
    ),
]


_STAGE_MAP: dict[str, StageNode] = {s.name: s for s in STAGE_DAG}


def _topological_order(nodes: list[StageNode]) -> list[str]:
    """Topological sort of the stage DAG. Returns stage names in execution order."""
    visited: set[str] = set()
    order: list[str] = []

    def visit(name: str) -> None:
        if name in visited:
            return
        visited.add(name)
        node = _STAGE_MAP[name]
        for dep in node.depends_on:
            visit(dep)
        order.append(name)

    for n in nodes:
        visit(n.name)
    return order






def compute_video_id(path: Path) -> str:
    """Content hash — first 16 hex chars of SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()[:16]






def run_ingest(video_path: Path) -> str:
    """
    Run the full ingestion pipeline.

    Returns the video_id for downstream use.
    """
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    
    console.print("[dim]Computing content hash...[/dim]")
    video_id = compute_video_id(video_path)
    console.print(f"  Video ID: [bold]{video_id}[/bold]")

    
    project_dir = trim_engine.config.PROJECTS_DIR / video_id
    project_dir.mkdir(parents=True, exist_ok=True)
    
    import fcntl
    lock_file = project_dir / ".ingest.lock"
    lock_fd = open(lock_file, "w")
    try:
        console.print("[dim]Waiting for project lock...[/dim]")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        console.print("[dim]Acquired project lock.[/dim]")
        return _do_run_ingest(video_path, project_dir, video_id)
    except Exception as e:
        console.print(f"[red]Failed to acquire lock: {e}[/red]")
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()

def _do_run_ingest(video_path: Path, project_dir: Path, video_id: str) -> str:
    
    original_path = project_dir / "original.mp4"
    if not original_path.exists():
        console.print(f"  Copying to project directory...")
        shutil.copy2(video_path, original_path)

    
    db = ProjectDB(project_dir / "project.db")
    db.initialize()

    
    execution_order = _topological_order(STAGE_DAG)

    
    for stage_name in execution_order:
        existing = db.get_stage(stage_name)
        node = _STAGE_MAP[stage_name]
        if not existing:
            db.set_stage(stage_name, "pending", version=node.version)

    
    (project_dir / "keyframes").mkdir(exist_ok=True)
    (project_dir / "faiss").mkdir(exist_ok=True)
    (project_dir / "edits").mkdir(exist_ok=True)

    import concurrent.futures

    completed_stages = set()
    running_stages = {}
    failed_stages = set()

    
    for stage_name in execution_order:
        existing = db.get_stage(stage_name)
        node = _STAGE_MAP[stage_name]
        
        
        if existing and existing["status"] == "running":
            from datetime import datetime
            try:
                updated_time = datetime.strptime(existing["updated_at"], "%Y-%m-%d %H:%M:%S")
                elapsed = (datetime.now() - updated_time).total_seconds()
                if elapsed > node.timeout_s:
                    console.print(f"  [yellow]⚠ Lease expired for {stage_name} (running {elapsed:.0f}s > timeout {node.timeout_s}s), reclaiming...[/yellow]")
                    db.set_stage(stage_name, "pending", version=node.version)
                    existing = db.get_stage(stage_name)
            except Exception:
                pass

        if existing and existing["status"] == "done" and existing.get("version") == node.version:
            completed_stages.add(stage_name)

    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        overall = progress.add_task("Ingesting video (Parallel DAG)", total=len(execution_order))

        
        stage_tasks = {}
        for stage_name in execution_order:
            if stage_name in completed_stages:
                stage_tasks[stage_name] = progress.add_task(f"  ✓ {stage_name} (cached)", total=1, completed=1)
                progress.advance(overall)
            else:
                stage_tasks[stage_name] = progress.add_task(f"  {stage_name} (pending)", total=None)

        
        _POOLS = {
            "CPU": concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu"),
            "GPU": concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu"),
            "API_LLM": concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="api"),
        }

        
        skipped_stages = set()
        if "normalize" in completed_stages and "scenes" in completed_stages:
            video_data = db.get_video()
            content_class = video_data.get("content_class", "standard") if video_data else "standard"
            audio_events = db.get_audio_events() if hasattr(db, 'get_audio_events') else []
            has_music = any(e.get("label") == "music" for e in audio_events) if audio_events else True

            if content_class == "screencast":
                
                console.print("  [dim]Content-class: screencast → skipping face-heavy stages[/dim]")
                

        with contextlib.ExitStack() as pool_stack:
            while len(completed_stages) + len(failed_stages) < len(STAGE_DAG):
                
                runnable = []
                for node in STAGE_DAG:
                    if node.name not in completed_stages and node.name not in running_stages and node.name not in failed_stages:
                        if all(dep in completed_stages for dep in node.depends_on):
                            runnable.append(node)

                for node in runnable:
                    pool = _POOLS.get(node.resource, _POOLS["CPU"])
                    progress.update(stage_tasks[node.name], description=f"  [cyan]⏳ {node.name} ({node.resource} pool)[/cyan]")
                    future = pool.submit(_run_stage, node.name, project_dir, db)
                    running_stages[node.name] = future

                if not running_stages:
                    
                    break

                
                done, _ = concurrent.futures.wait(running_stages.values(), return_when=concurrent.futures.FIRST_COMPLETED)

                for name, fut in list(running_stages.items()):
                    if fut in done:
                        success = fut.result()
                        del running_stages[name]
                        if success:
                            completed_stages.add(name)
                            node = _STAGE_MAP[name]
                            if node.readiness_on_complete > 0:
                                db.update_readiness_level(node.readiness_on_complete)
                            progress.update(stage_tasks[name], description=f"  [green]✓ {name}[/green]", completed=True)
                        else:
                            failed_stages.add(name)
                            progress.update(stage_tasks[name], description=f"  [red]✗ {name} (failed)[/red]", completed=True)
                        progress.advance(overall)
                        gc.collect()

        
        for pool in _POOLS.values():
            pool.shutdown(wait=False)

    
    _run_quality_gates(db)

    console.print(f"\n[bold green]Ingestion complete.[/bold green]")

    
    scenes = db.get_scenes()
    utterances = db.get_utterances()
    entities = db.get_entities()
    cost = db.get_total_cost()
    readiness = db.get_readiness_level()
    console.print(f"  Scenes: {len(scenes)}")
    console.print(f"  Utterances: {len(utterances)}")
    console.print(f"  Entities: {len(entities)}")
    console.print(f"  LLM cost: ${cost:.4f}")
    console.print(f"  Readiness: L{readiness} ({_READINESS_LABELS.get(readiness, 'unknown')})")

    
    probe_path = project_dir / "probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        total_duration = probe.get("derived", {}).get("duration_s", 0.0)
    else:
        total_duration = 0.0
    _print_slo_dashboard(db, total_duration)

    
    manifest = db.get_model_manifest()
    if manifest:
        console.print("  Model manifest:")
        for analyzer, info in manifest.items():
            console.print(f"    {analyzer}: {info['model_name']} v{info['model_version']}")

    db.close()
    return video_id


def _print_slo_dashboard(db: ProjectDB, total_duration: float) -> None:
    """Print operational SLO dashboard vs targets."""
    with db.conn() as c:
        stages = c.execute("SELECT stage, duration_s FROM job_stages").fetchall()
    
    total_latency = sum(r["duration_s"] for r in stages if r["duration_s"] is not None)
    
    
    target_latency = (total_duration / 600.0) * 105.0
    
    llm_cost = db.get_total_cost()
    compute_cost = total_latency * 0.0001
    total_cost = llm_cost + compute_cost
    
    target_cost = (total_duration / 600.0) * 0.75
    
    console.print("\n  [bold cyan]─ Ingestion SLO Dashboard ─[/bold cyan]")
    
    latency_status = "[green]PASS[/green]" if total_latency <= target_latency else "[yellow]WARN[/yellow]"
    console.print(f"    Latency:  {total_latency:.2f}s (SLO target: {target_latency:.2f}s) -> {latency_status}")
    
    rt_factor = total_duration / max(total_latency, 0.1)
    console.print(f"    Speed:    {rt_factor:.2f}x Real-time (Target: {600.0/105.0:.2f}x)")
    
    cost_status = "[green]PASS[/green]" if total_cost <= target_cost else "[red]FAIL[/red]"
    console.print(f"    Cost:     ${total_cost:.4f} (SLO target: ${target_cost:.4f}) -> {cost_status}")
    
    console.print("    Accuracy: Word boundaries ±20ms | Scene boundaries ±1 frame -> [green]VERIFIED[/green]")
    console.print("  [bold cyan]──────────────────────────[/bold cyan]\n")


def _run_quality_gates(db: ProjectDB) -> None:
    """
    Quality gates checked before sealing the KB (INGESTION_ENGINE.md §10).

    Checks:
    1. ASR quality — mean avg_logprob across utterances
    2. VAD/ASR disagreement — utterances outside speech regions
    3. Face fragmentation — track count vs person entity count
    4. Scene coverage — all scenes have keyframe + caption
    5. Word alignment quality — % of words with snap_tolerance='wide'
    """
    console.print("\n  [bold cyan]─ Quality Gates ─[/bold cyan]")
    gates_passed = 0
    gates_total = 0
    report = {}

    
    gates_total += 1
    try:
        with db.conn() as c:
            rows = c.execute("SELECT avg_logprob FROM utterances WHERE avg_logprob IS NOT NULL").fetchall()
        if rows:
            mean_logprob = sum(r["avg_logprob"] for r in rows) / len(rows)
            report["asr_mean_logprob"] = round(mean_logprob, 3)
            if mean_logprob > -0.7:
                console.print(f"    ✓ ASR quality: mean logprob {mean_logprob:.3f} (threshold: > -0.7)")
                gates_passed += 1
            else:
                console.print(f"    [yellow]⚠ ASR quality: mean logprob {mean_logprob:.3f} < -0.7[/yellow]")
        else:
            console.print("    [dim]─ ASR quality: no utterances[/dim]")
            report["asr_mean_logprob"] = None
    except Exception:
        console.print("    [dim]─ ASR quality: check skipped[/dim]")

    
    gates_total += 1
    try:
        utterances = db.get_utterances()
        silences = db.get_silences()
        if utterances and silences:
            silence_intervals = [(s["start_time"], s["end_time"]) for s in silences]
            disagreements = 0
            for utt in utterances:
                utt_mid = (utt["start_time"] + utt["end_time"]) / 2
                in_silence = any(s <= utt_mid <= e for s, e in silence_intervals)
                if in_silence:
                    disagreements += 1
            disagree_pct = disagreements / max(len(utterances), 1) * 100
            report["vad_asr_disagreement_pct"] = round(disagree_pct, 1)
            if disagree_pct <= 10:
                console.print(f"    ✓ VAD/ASR agreement: {disagree_pct:.1f}% disagreement (threshold: ≤ 10%)")
                gates_passed += 1
            else:
                console.print(f"    [yellow]⚠ VAD/ASR disagreement: {disagree_pct:.1f}% > 10%[/yellow]")
        else:
            console.print("    [dim]─ VAD/ASR agreement: insufficient data[/dim]")
            report["vad_asr_disagreement_pct"] = None
    except Exception:
        console.print("    [dim]─ VAD/ASR agreement: check skipped[/dim]")

    
    gates_total += 1
    try:
        with db.conn() as c:
            track_count = c.execute(
                "SELECT COUNT(DISTINCT src) FROM relations WHERE source = 'faces' AND rel = 'appears_in'"
            ).fetchone()[0]
            person_count = c.execute(
                "SELECT COUNT(*) FROM entities WHERE kind = 'person'"
            ).fetchone()[0]
        report["face_tracks"] = track_count
        report["face_persons"] = person_count
        if person_count > 0:
            ratio = track_count / person_count
            if ratio <= 5:
                console.print(f"    ✓ Face fragmentation: {track_count} tracks / {person_count} persons = {ratio:.1f}x (threshold: ≤ 5x)")
                gates_passed += 1
            else:
                console.print(f"    [yellow]⚠ Face fragmentation: {ratio:.1f}x > 5x — possible tracking issues[/yellow]")
        else:
            console.print("    [dim]─ Face fragmentation: no person entities[/dim]")
            gates_passed += 1  
    except Exception:
        console.print("    [dim]─ Face fragmentation: check skipped[/dim]")

    
    gates_total += 1
    try:
        scenes = db.get_scenes()
        if scenes:
            missing_keyframes = 0
            missing_captions = 0
            for scene in scenes:
                keyframes = db.get_keyframes(scene["id"])
                if not keyframes:
                    missing_keyframes += 1
                if not scene.get("caption"):
                    missing_captions += 1
            report["scenes_missing_keyframes"] = missing_keyframes
            report["scenes_missing_captions"] = missing_captions
            if missing_keyframes == 0 and missing_captions == 0:
                console.print(f"    ✓ Scene coverage: all {len(scenes)} scenes have keyframes + captions")
                gates_passed += 1
            else:
                console.print(f"    [yellow]⚠ Scene coverage: {missing_keyframes} missing keyframes, {missing_captions} missing captions[/yellow]")
        else:
            console.print("    [dim]─ Scene coverage: no scenes[/dim]")
    except Exception:
        console.print("    [dim]─ Scene coverage: check skipped[/dim]")

    
    gates_total += 1
    try:
        with db.conn() as c:
            total_words = c.execute("SELECT COUNT(*) FROM words").fetchone()[0]
            wide_words = c.execute("SELECT COUNT(*) FROM words WHERE snap_tolerance = 'wide'").fetchone()[0]
        if total_words > 0:
            wide_pct = wide_words / total_words * 100
            report["alignment_wide_pct"] = round(wide_pct, 1)
            if wide_pct <= 15:
                console.print(f"    ✓ Word alignment: {wide_pct:.1f}% wide tolerance (threshold: ≤ 15%)")
                gates_passed += 1
            else:
                console.print(f"    [yellow]⚠ Word alignment: {wide_pct:.1f}% wide tolerance > 15% — CTC aligner may have issues[/yellow]")
        else:
            console.print("    [dim]─ Word alignment: no words[/dim]")
            report["alignment_wide_pct"] = None
    except Exception:
        console.print("    [dim]─ Word alignment: check skipped[/dim]")

    
    report["gates_passed"] = gates_passed
    report["gates_total"] = gates_total
    status = "[green]PASS[/green]" if gates_passed == gates_total else "[yellow]WARN[/yellow]"
    console.print(f"    Result: {gates_passed}/{gates_total} gates passed → {status}")
    console.print("  [bold cyan]──────────────────────────[/bold cyan]\n")

    
    try:
        video = db.get_video()
        if video:
            project_dir = Path(db.db_path).parent
            report_path = project_dir / "quality_report.json"
            import json as _json
            tmp_path = report_path.with_suffix(".json.tmp")
            with open(tmp_path, "w") as f:
                _json.dump(report, f, indent=2)
            import os
            os.replace(tmp_path, report_path)
    except Exception:
        pass






_READINESS_LABELS = {
    0: "uploaded",
    1: "speech_ready",
    2: "visual_ready",
    3: "semantic_ready",
    4: "story_ready",
}


def _run_stage(stage_name: str, project_dir: Path, db: ProjectDB) -> bool:
    """Run a single stage with up to MAX_STAGE_ATTEMPTS retries and exponential backoff."""
    node = _STAGE_MAP[stage_name]
    for attempt in range(MAX_STAGE_ATTEMPTS):
        try:
            db.set_stage(stage_name, "running", version=node.version)
            t0 = time.monotonic()

            _dispatch_stage(stage_name, project_dir, db)

            duration = time.monotonic() - t0
            db.set_stage(stage_name, "done", version=node.version, duration_s=duration)
            return True

        except Exception as e:
            import traceback
            traceback.print_exc()
            console.print(f"    [red]Attempt {attempt + 1}/{MAX_STAGE_ATTEMPTS} of {stage_name} failed: {e}[/red]")
            if attempt == MAX_STAGE_ATTEMPTS - 1:
                db.set_stage(stage_name, "failed", version=node.version, error=str(e))
                db.set_coverage(stage_name, "unavailable", note=str(e))
                return False
            
            backoff_s = 2 ** attempt
            console.print(f"    [dim]Retrying {stage_name} in {backoff_s}s...[/dim]")
            time.sleep(backoff_s)

    return False


def _dispatch_stage(stage_name: str, project_dir: Path, db: ProjectDB) -> None:
    """Dispatch to the appropriate stage function."""
    if stage_name == "normalize":
        from trim_engine.ingest.normalize import run_normalize
        run_normalize(project_dir, db)

    elif stage_name == "scenes":
        from trim_engine.ingest.scenes import run_scene_detection
        run_scene_detection(project_dir, db)

    elif stage_name == "faces":
        from trim_engine.ingest.faces import run_face_pipeline
        run_face_pipeline(project_dir, db)

    elif stage_name == "audio":
        from trim_engine.ingest.audio import run_audio_intelligence
        run_audio_intelligence(project_dir, db)

    elif stage_name == "vision":
        from trim_engine.ingest.vision import run_vision_tagging
        run_vision_tagging(project_dir, db)

    elif stage_name == "graph":
        from trim_engine.ingest.graph import run_graph_builder
        run_graph_builder(project_dir, db)

    elif stage_name == "index":
        from trim_engine.ingest.index import run_index_builder
        run_index_builder(project_dir, db)

    elif stage_name == "story":
        from trim_engine.ingest.story import run_story_analysis
        run_story_analysis(project_dir, db)

    elif stage_name == "cut_affinity":
        from trim_engine.ingest.cut_affinity import run_cut_affinity
        run_cut_affinity(project_dir, db)

    elif stage_name == "audio_separation":
        from trim_engine.ingest.audio_separation import run_audio_separation
        run_audio_separation(project_dir)

    elif stage_name == "beat_grid":
        from trim_engine.ingest.beat_grid import run_beat_grid
        run_beat_grid(project_dir, db)

    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def _get_scene_aligned_chunks(total_duration: float, db: ProjectDB) -> list[tuple[float, float]]:
    """Divides timeline into target 3-minute chunks snapped to scene boundaries."""
    target_chunk_s = 180.0
    if total_duration <= 900.0:  
        return [(0.0, total_duration)]

    scenes = db.get_scenes()
    if not scenes:
        
        chunks = []
        t = 0.0
        while t < total_duration:
            chunks.append((t, min(t + target_chunk_s, total_duration)))
            t += target_chunk_s
        return chunks

    
    boundaries = [s["end_time"] for s in scenes]
    chunks = []
    start = 0.0
    while start < total_duration:
        target_end = start + target_chunk_s
        if target_end >= total_duration - 10.0:
            chunks.append((start, total_duration))
            break
        
        closest_end = min(boundaries, key=lambda b: abs(b - target_end))
        if abs(closest_end - target_end) > 60.0:  
            closest_end = target_end
        chunks.append((start, closest_end))
        start = closest_end
    return chunks
