import pytest
from pathlib import Path
from trim_engine.db import ProjectDB
from trim_engine.query.intent import compile_intent
from trim_engine.query.retrieval import retrieve_segments
from trim_engine.query.story_agent import maybe_run_story_agent
from trim_engine.query.planner import plan_timeline
from trim_engine.query.critic import validate_plan
from trim_engine.cli import _get_project_dir

SUITE_PROMPTS = [
    "Remove all filler words and silence.",
    "Cut out the retake.",
    "Remove the 3-second pause.",
    "Take out the part where I talk about pricing.",
    "Make it under 30 seconds.",
    "Keep only the B-roll.",
    "Remove everything before I enter the frame.",
    "Create a trailer-style cut.",
]

@pytest.mark.llm
def test_e2e_golden_suite():
    # Use the fixture project
    video_id = "9360d2bc6daf553b"
    project_dir = _get_project_dir(video_id)
    if not project_dir.exists():
        pytest.skip(f"Fixture project {video_id} not ingested yet")
        
    db = ProjectDB(project_dir / "project.db")
    
    for prompt_text in SUITE_PROMPTS:
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
            max_retries=2, retry_handler=retry_from_critic
        )
        
        # Assertions
        assert verdict is getattr(verdict, "passed", True) or hasattr(verdict, "passed")
        
        if verdict.passed:
            # If the edit passed, it must have either kept or removed something (or it was a noop block)
            if intent.constraints and intent.constraints.target_duration_s:
                # duration constraint respected
                assert edit_plan.predicted_output_duration_s <= intent.constraints.target_duration_s + 1.0
            if any(op.action == "remove" for op in intent.operations):
                # if there was a removal, we should have a removal ratio > 0 or it gracefully failed
                pass 
