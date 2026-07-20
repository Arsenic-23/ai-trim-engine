"""
FastAPI wrapper — programmatic access to the trim engine.

Endpoints:
    POST /ingest          Upload video, run full ingestion
    GET  /status/{id}     Job/coverage status
    POST /edit/{id}       Submit edit prompt
    GET  /edits/{id}      List all edits for a video
    GET  /health          Health check
"""

from __future__ import annotations

import shutil
import uuid
import sys
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import trim_engine.config
from trim_engine.db import ProjectDB

ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class ActiveLogStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.active_log_path = None

    def set_active_log(self, path: Path):
        self.active_log_path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")

    def clear_active_log(self):
        self.active_log_path = None

    def write(self, data):
        self.original_stream.write(data)
        if self.active_log_path:
            try:
                clean_data = ansi_escape.sub('', data)
                with open(self.active_log_path, "a", encoding="utf-8") as f:
                    f.write(clean_data)
            except Exception:
                pass

    def flush(self):
        self.original_stream.flush()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

sys.stdout = ActiveLogStream(sys.stdout)
sys.stderr = ActiveLogStream(sys.stderr)

app = FastAPI(
    title="AI Trim Engine",
    description="Natural-language video editing API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class EditRequest(BaseModel):
    prompt: str
    revise_from: int | None = None  


class EditResponse(BaseModel):
    version: int
    output_path: str
    duration_before_s: float
    duration_after_s: float
    reduction_pct: float
    clip_count: int
    cost_usd: float
    report_path: str


class IngestResponse(BaseModel):
    video_id: str
    project_dir: str


class StatusResponse(BaseModel):
    video_id: str
    video_info: dict[str, Any] | None
    stages: list[dict[str, Any]]
    coverage: dict[str, str]
    total_cost_usd: float
    scene_count: int
    utterance_count: int
    entity_count: int




def _get_db(video_id: str) -> ProjectDB:
    project_dir = trim_engine.config.PROJECTS_DIR / video_id
    db_path = project_dir / "project.db"
    if not db_path.exists():
        raise HTTPException(status_code=404, detail=f"No project found for video_id: {video_id}")
    db = ProjectDB(db_path)
    return db




@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(video: UploadFile = File(...)):
    """Upload a video file and run the full ingestion pipeline."""
    
    temp_dir = trim_engine.config.PROJECTS_DIR / "_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex[:8]}_{video.filename}"

    try:
        with open(temp_path, "wb") as f:
            content = await video.read()
            f.write(content)

        from trim_engine.ingest.orchestrator import run_ingest
        video_id = run_ingest(temp_path)

        return IngestResponse(
            video_id=video_id,
            project_dir=str(trim_engine.config.PROJECTS_DIR / video_id),
        )
    finally:
        
        if temp_path.exists():
            temp_path.unlink()


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Get ingestion status and pipeline coverage."""
    db = _get_db(video_id)

    stages = []
    with db.conn() as c:
        rows = c.execute("SELECT stage, status, version, updated_at FROM job_stages").fetchall()
        for r in rows:
            stages.append(dict(r))

    video = db.get_video()
    scenes = db.get_scenes()
    utterances = db.get_utterances()
    entities = db.get_entities()

    return StatusResponse(
        video_id=video_id,
        video_info=video,
        stages=stages,
        total_cost_usd=db.get_total_cost(),
        coverage={
            "duration": f"{video['duration_s']:.1f}s" if video and video.get('duration_s') else "0s",
            "scenes": str(len(scenes)),
            "utterances": str(len(utterances)),
            "entities": str(len(entities)),
        },
        scene_count=len(scenes),
        utterance_count=len(utterances),
        entity_count=len(entities),
    )

import asyncio
import json

@app.post("/ingest/stream")
async def ingest_stream(file: UploadFile = File(...)):
    """Stream ingestion progress."""
    upload_id = uuid.uuid4().hex[:12]
    upload_dir = trim_engine.config.PROJECTS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{upload_id}_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async def event_generator():
        import threading
        from trim_engine.ingest.orchestrator import run_ingest, STAGE_DAG, compute_video_id
        
        video_id = compute_video_id(temp_path)
        yield f"data: {json.dumps({'event': 'hash', 'video_id': video_id})}\n\n"
        
        log_path = trim_engine.config.PROJECTS_DIR / video_id / "ingest.log"
        sys.stdout.set_active_log(log_path)
        sys.stderr.set_active_log(log_path)
        
        result = {"video_id": None, "error": None}
        def run():
            try:
                result["video_id"] = run_ingest(temp_path)
            except Exception as e:
                result["error"] = str(e)
                
        thread = threading.Thread(target=run)
        thread.start()
        
        try:
            import os
            for _ in range(50):
                if log_path.exists():
                    break
                await asyncio.sleep(0.1)
                
            db = None
            last_reported_stages = set()
            
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    while thread.is_alive() or os.path.getsize(log_path) > f.tell():
                        line = f.readline()
                        if line:
                            yield f"data: {json.dumps({'event': 'log', 'text': line.rstrip()})}\n\n"
                        else:
                            try:
                                if db is None:
                                    db_path = trim_engine.config.PROJECTS_DIR / video_id / "project.db"
                                    if db_path.exists():
                                        db = ProjectDB(db_path)
                                if db:
                                    with db.conn() as c:
                                        stages = c.execute("SELECT stage, status FROM job_stages WHERE status = 'done'").fetchall()
                                    for stage in stages:
                                        s_name = stage["stage"]
                                        if s_name not in last_reported_stages:
                                            last_reported_stages.add(s_name)
                                            yield f"data: {json.dumps({'event': 'stage_done', 'stage': s_name})}\n\n"
                            except Exception:
                                pass
                            await asyncio.sleep(0.1)
            
            thread.join()
            if result["error"]:
                yield f"data: {json.dumps({'event': 'error', 'error': result['error']})}\n\n"
            else:
                yield f"data: {json.dumps({'event': 'done', 'video_id': result['video_id']})}\n\n"
        finally:
            sys.stdout.clear_active_log()
            sys.stderr.clear_active_log()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/edit/{video_id}/stream")
async def edit_stream(video_id: str, prompt: str = Form(...)):
    """Stream edit pipeline progress."""
    async def event_generator():
        import threading
        from trim_engine.query.executor import create_or_resume_session, execute_session_pipeline
        
        db = _get_db(video_id)
        session = create_or_resume_session(video_id, prompt, db)
        session_id = session.session_id
        
        log_path = trim_engine.config.PROJECTS_DIR / video_id / "edit.log"
        sys.stdout.set_active_log(log_path)
        sys.stderr.set_active_log(log_path)
        
        result = {"error": None}
        def run():
            try:
                execute_session_pipeline(session, db, trim_engine.config.PROJECTS_DIR / video_id, auto_approve=True)
            except Exception as e:
                result["error"] = str(e)
                
        thread = threading.Thread(target=run)
        thread.start()
        
        try:
            import os
            for _ in range(50):
                if log_path.exists():
                    break
                await asyncio.sleep(0.1)
                
            last_state = None
            if log_path.exists():
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    while thread.is_alive() or os.path.getsize(log_path) > f.tell():
                        line = f.readline()
                        if line:
                            yield f"data: {json.dumps({'event': 'log', 'text': line.rstrip()})}\n\n"
                        else:
                            try:
                                with db.conn() as c:
                                    row = c.execute("SELECT state FROM edit_sessions WHERE id = ?", (session_id,)).fetchone()
                                if row:
                                    current_state = row["state"]
                                    if current_state != last_state:
                                        last_state = current_state
                                        yield f"data: {json.dumps({'event': 'state_change', 'state': current_state})}\n\n"
                            except Exception:
                                pass
                            await asyncio.sleep(0.1)
            
            thread.join()
            if result["error"]:
                yield f"data: {json.dumps({'event': 'error', 'error': result['error']})}\n\n"
            else:
                with db.conn() as c:
                    row = c.execute("SELECT state FROM edit_sessions WHERE id = ?", (session_id,)).fetchone()
                    if row and row["state"] in ("delivered", "resolved_noop"):
                        yield f"data: {json.dumps({'event': 'state_change', 'state': row['state']})}\n\n"
                        try:
                            latest_version = c.execute("SELECT MAX(version) as v FROM edit_sessions WHERE video_id = ?", (video_id,)).fetchone()["v"] or 1
                            report_path = trim_engine.config.PROJECTS_DIR / video_id / "edits" / f"v{latest_version}" / "report.json"
                            if report_path.exists():
                                with open(report_path) as f:
                                    report_data = json.load(f)
                            else:
                                v_info = db.get_video()
                                dur = v_info["duration_s"] if v_info and v_info.get("duration_s") else 0.0
                                report_data = {
                                    "duration_before_s": dur,
                                    "duration_after_s": dur,
                                    "reduction_pct": 0.0,
                                    "clip_count": 1,
                                    "cost_usd": db.get_total_cost(),
                                    "removals": []
                                }
                            yield f"data: {json.dumps({'event': 'report', 'report': report_data, 'version': latest_version})}\n\n"
                        except Exception as e:
                            yield f"data: {json.dumps({'event': 'error', 'error': 'Report fetch error: ' + str(e)})}\n\n"
                    elif row and row["state"] == "render_failed":
                        s_row = c.execute("SELECT session_json FROM edit_sessions WHERE id = ?", (session_id,)).fetchone()
                        try:
                            s_data = json.loads(s_row["session_json"])
                            yield f"data: {json.dumps({'event': 'error', 'error': s_data.get('error', 'Render failed')})}\n\n"
                        except:
                            yield f"data: {json.dumps({'event': 'error', 'error': 'Render failed'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'event': 'error', 'error': 'Unknown terminal state: ' + str(row['state'] if row else 'None')})}\n\n"
        finally:
            sys.stdout.clear_active_log()
            sys.stderr.clear_active_log()

    return StreamingResponse(event_generator(), media_type="text/event-stream")



@app.post("/edit/{video_id}", response_model=EditResponse)
async def edit(video_id: str, request: EditRequest):
    """Run the edit pipeline on an ingested video."""
    project_dir = trim_engine.config.PROJECTS_DIR / video_id
    db = _get_db(video_id)

    video = db.get_video()
    if not video:
        raise HTTPException(status_code=404, detail="Video not ingested")

    from trim_engine.query.intent import compile_intent
    from trim_engine.query.retrieval import retrieve_segments
    from trim_engine.query.story_agent import maybe_run_story_agent
    from trim_engine.query.planner import plan_timeline
    from trim_engine.query.critic import validate_plan
    from trim_engine.query.renderer import render_timeline
    from trim_engine.query.report import generate_report

    
    intent = compile_intent(request.prompt, db)

    if intent.out_of_scope_reason:
        raise HTTPException(status_code=422, detail=intent.out_of_scope_reason)

    if any(a.blocking for a in intent.ambiguities):
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Clarification needed",
                "ambiguities": [
                    {"issue": a.issue, "candidates": a.candidates}
                    for a in intent.ambiguities if a.blocking
                ],
            },
        )

    
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
    output_path = render_timeline(timeline, project_dir, version, db)

    
    report = generate_report(
        video_id=video_id, version=version, prompt=request.prompt,
        intent=intent, edit_plan=edit_plan, verdict=verdict, db=db,
    )

    
    db.insert_edit(
        version=version, prompt=request.prompt,
        intent_json=intent.model_dump_json(),
        plan_json=edit_plan.model_dump_json(),
        verdict_json=verdict.model_dump_json(),
        output_path=str(output_path),
    )

    
    try:
        from trim_engine.query.profile import learn_from_prompt
        learn_from_prompt(request.prompt, db)
    except Exception:
        pass

    report_path = project_dir / "edits" / f"v{version}" / "report.md"

    return EditResponse(
        version=version,
        output_path=str(output_path),
        duration_before_s=report.duration_before_s,
        duration_after_s=report.duration_after_s,
        reduction_pct=report.reduction_pct,
        clip_count=edit_plan.clip_count,
        cost_usd=report.cost_usd,
        report_path=str(report_path),
    )


@app.get("/edits/{video_id}")
async def list_edits(video_id: str):
    """List all edits for a video."""
    db = _get_db(video_id)
    edits = db.get_all_edits()
    return {"video_id": video_id, "edits": edits}


@app.get("/edits/{video_id}/{version}/output")
async def download_output(video_id: str, version: int):
    """Download the rendered output video."""
    output_path = trim_engine.config.PROJECTS_DIR / video_id / "edits" / f"v{version}" / "output.mp4"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(output_path, media_type="video/mp4", filename=f"edit_v{version}.mp4")


@app.get("/edits/{video_id}/{version}/report")
async def get_report(video_id: str, version: int):
    """Get the edit report as JSON."""
    report_path = trim_engine.config.PROJECTS_DIR / video_id / "edits" / f"v{version}" / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not found")

    import json
    with open(report_path) as f:
        return json.load(f)


@app.get("/edits/{video_id}/{version}/timeline")
async def get_timeline(video_id: str, version: int):
    """Get the timeline JSON for an edit."""
    timeline_path = trim_engine.config.PROJECTS_DIR / video_id / "edits" / f"v{version}" / "timeline.json"
    if not timeline_path.exists():
        raise HTTPException(status_code=404, detail="Timeline not found")

    import json
    with open(timeline_path) as f:
        return json.load(f)


@app.get("/kb/{video_id}/scenes")
async def get_scenes(video_id: str):
    """Get all scenes from the knowledge base."""
    db = _get_db(video_id)
    return {"scenes": db.get_scenes()}


@app.get("/kb/{video_id}/entities")
async def get_entities(video_id: str, kind: str | None = None):
    """Get entities from the knowledge base."""
    db = _get_db(video_id)
    return {"entities": db.get_entities(kind=kind)}


@app.get("/kb/{video_id}/transcript")
async def get_transcript(video_id: str):
    """Get the full transcript."""
    db = _get_db(video_id)
    return {"utterances": db.get_utterances()}


@app.get("/kb/{video_id}/topics")
async def get_topics(video_id: str):
    """Get topic segments."""
    db = _get_db(video_id)
    return {"topics": db.get_topics()}


@app.get("/kb/{video_id}/story")
async def get_story(video_id: str):
    """Get story beats and dependencies."""
    db = _get_db(video_id)
    return {
        "beats": db.get_story_beats(),
        "dependencies": db.get_story_deps(),
    }

@app.get("/video/{video_id}")
async def get_original_video(video_id: str):
    """Serve the original unedited video."""
    video_path = trim_engine.config.PROJECTS_DIR / video_id / "original.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(video_path, media_type="video/mp4")

# Mount static frontend at the end to avoid intercepting API routes
frontend_dir = Path(__file__).resolve().parent.parent / "frontend_demo"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

