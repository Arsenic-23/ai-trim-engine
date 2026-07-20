"""
Intent Compiler (§5.1) — prompt → structured EditIntent.

Behaviors:
1. P0 Normalize: strip/record meta-instructions, resolve deictic references, language-detect.
2. P1 Fast-path classifier: deterministic pattern bank for high-frequency mechanical intents.
3. P2 LLM Compile: Claude structured-output call.
4. P3 Grounding: resolve referenced entities against KB, anchor temporal ranges.
5. P4 Policy: operation-conflict detection, profile merge, progressive capability checks.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.llm import build_video_summary, call_structured
from trim_engine.schemas import EditIntent, EditorProfile, Operation, SegmentTarget, EditConstraints, EditStyle, Ambiguity
from trim_engine.query.profile import load_profile

console = Console()

def _detect_language(prompt: str) -> str:
    """Simple language detector fallback."""
    try:
        import langdetect
        return langdetect.detect(prompt)
    except Exception:

        non_ascii = len([c for c in prompt if ord(c) > 127])
        if non_ascii / max(1, len(prompt)) > 0.15:
            return "es"
        return "en"


_POLITE_PREFIXES = (
    "can you please", "could you please", "would you please", "will you please",
    "can you", "could you", "would you", "will you", "please",
    "i want you to", "i'd like you to", "i would like you to",
    "i want to", "i'd like to", "i would like to", "i need you to", "i need to",
    "hey", "hi", "ok so", "okay so",
)

_EDIT_VERBS = (
    "remove", "cut", "trim", "delete", "keep", "make", "create", "shorten",
    "compress", "summarize", "edit", "reorder", "restructure", "prioritize",
    "focus", "build", "add", "put", "revert", "undo", "turn", "convert",
    "speed", "highlight", "clip", "extract",
)


def normalize_prompt(prompt: str) -> str:
    """
    P0 normalization: collapse whitespace and strip polite/conversational
    prefixes so downstream fast-path patterns and QA detection see the
    actual instruction ("Can you please remove fillers?" → "remove fillers?").
    """
    normalized = " ".join(prompt.split())
    lowered = normalized.lower()
    changed = True
    while changed:
        changed = False
        for prefix in _POLITE_PREFIXES:
            if lowered.startswith(prefix + " ") or lowered.startswith(prefix + ","):
                normalized = normalized[len(prefix):].lstrip(" ,")
                lowered = normalized.lower()
                changed = True
                break
    return normalized


def _is_question(prompt_clean: str) -> bool:
    """
    True only for genuine content questions — never for edit commands.
    An imperative edit verb anywhere in the prompt wins over question shape,
    so "can you remove the intro?" routes to editing, not Q&A.
    """
    words = re.findall(r"[a-z']+", prompt_clean)
    if any(w in _EDIT_VERBS for w in words):
        return False
    return (
        prompt_clean.endswith("?")
        or prompt_clean.startswith(("who ", "what ", "where ", "how ", "when ", "why ",
                                    "is ", "are ", "does ", "do ", "tell me"))
    )

def _resolve_qa_answer(prompt: str, db: ProjectDB) -> str:
    """Answers user queries directly using DB transcripts."""
    words = db.get_words()
    if not words:
        return "No transcription found in video index."

    prompt_clean = prompt.lower()
    if "pricing" in prompt_clean or "price" in prompt_clean:
        price_hits = [w for w in words if "price" in w["word"].lower() or "pricing" in w["word"].lower()]
        if price_hits:
            ts = ", ".join(f"{w['start_time']:.1f}s" for w in price_hits[:3])
            return f"Q&A Answer: Pricing is discussed around {ts}."
        return "Q&A Answer: Pricing is not mentioned in the transcript."

    if "who" in prompt_clean or "speaker" in prompt_clean:
        entities = db.get_entities()
        speakers = [
            e["label"] for e in entities
            if e.get("kind") in ("speaker", "person") or "speaker" in e["id"].lower()
        ]
        if speakers:
            return f"Q&A Answer: Identified speakers are: {', '.join(set(speakers))}."
        return "Q&A Answer: No speakers identified in the video."

    return f"Q&A Answer: Video duration is {words[-1]['end_time']:.1f}s."

def _try_fast_path(prompt: str, db: ProjectDB) -> EditIntent | None:
    prompt_clean = prompt.strip().lower().rstrip(".!")
    
    prompt_clean = re.sub(r"\s*\([^)]*\)", "", prompt_clean).strip()
    video = db.get_video() or {}
    duration = float(video.get("duration_s") or 60.0)

    
    if prompt_clean in ["remove filler words", "remove fillers", "cut filler words", "cut fillers",
                        "remove all filler words", "cut out filler words"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["speech"], query="filler words"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["default_fillers"]
        )

    
    m = re.match(r"(?:remove|cut)\s+silences?\s+(?:over|longer\s+than)\s+(\d+(?:\.\d+)?)\s*s?", prompt_clean)
    if m:
        sec = float(m.group(1))
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["audio"], query=f"silence > {sec}s"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=[f"custom_silence_{sec}s"]
        )

    
    if re.search(r"^(remove|cut)\s+(all\s+)?(the\s+)?(silences?|pauses?|dead time|dead air)$", prompt_clean):
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["audio"], query="silence"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["default_silence"]
        )

    
    if re.search(r"\b(remove\s+(retakes?|duplicates?|repeated takes?|mistakes?|repeated sentences?)|keep\s+(only\s+)?(the\s+)?final take)\b", prompt_clean):
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["speech"], query="retakes"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["default_retakes"]
        )

    
    if re.search(r"\b(remove\s+(ums?|uhs?|uh and um|ums and uhs))\b", prompt_clean):
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["speech"], query="ums and uhs"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["filler_specific"]
        )

    
    if re.search(r"\b(keep\s+only\s+speech|remove\s+non-speech|remove\s+silences?\s+and\s+keep\s+speech)\b", prompt_clean):
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="keep_only", target=SegmentTarget(modality=["speech"], query="speech"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["speech_only"]
        )

    
    m = re.match(r"remove\s+topics?\s+(?:about|on)\s+(\w+)", prompt_clean)
    if m:
        topic = m.group(1)
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="remove", target=SegmentTarget(modality=["speech"], query=topic), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["topic_removal"]
        )

    
    m = re.match(r"keep\s+(?:only\s+)?scenes\s+with\s+person:(\w+)", prompt_clean)
    if m:
        person = m.group(1)
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="keep_only", target=SegmentTarget(modality=["person"], query=f"person:{person}"), priority="hard", confidence=1.0)
            ],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["person_focus"]
        )

    if prompt_clean in ["make it shorter", "make this shorter", "make it more engaging"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="compress", target=SegmentTarget(modality=["story", "speech", "visual"], query="important moments"), priority="soft", confidence=0.85)
            ],
            constraints=EditConstraints(target_duration_s=max(3.0, duration * 0.7), duration_mode="approx", preserve_story=True),
            style=EditStyle(pacing="fast"),
            ambiguities=[],
            profile_applied=["fast_compress"]
        )

    m = re.match(r"make\s+(?:this\s+)?(?:under|less than)\s+(\d+)\s*(seconds?|s|minutes?|m)?", prompt_clean)
    if m:
        value = float(m.group(1))
        unit = m.group(2) or "seconds"
        if unit.startswith("m"):
            value *= 60
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="compress", target=SegmentTarget(modality=["story", "speech", "visual"], query="important moments"), priority="hard", confidence=0.9)
            ],
            constraints=EditConstraints(target_duration_s=value, duration_mode="max", preserve_story=True),
            style=EditStyle(pacing="fast"),
            ambiguities=[],
            profile_applied=["duration_compress"]
        )

    if prompt_clean in ["cut on every beat", "cut to the beat", "beat cut"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="compress", target=SegmentTarget(modality=["audio", "visual"], query="beat-aligned highlights"), priority="soft", confidence=0.85)
            ],
            constraints=EditConstraints(preserve_story=True),
            style=EditStyle(cut_style="beat_cut", pacing="fast"),
            ambiguities=[],
            profile_applied=["beat_cut"]
        )

    if prompt_clean in ["create match cuts", "make match cuts", "match cut", "create a match-cut edit"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="restructure", target=SegmentTarget(modality=["visual"], query="visually similar scene joins"), priority="soft", confidence=0.85)
            ],
            constraints=EditConstraints(preserve_story=True),
            style=EditStyle(cut_style="match_cut", ordering="custom"),
            ambiguities=[],
            profile_applied=["match_cut"]
        )

    if prompt_clean in ["create a trailer-style cut", "trailer-style cut", "make a trailer", "create a trailer"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="restructure", target=SegmentTarget(modality=["story", "visual", "emotion"], query="trailer story beats"), priority="soft", confidence=0.85)
            ],
            constraints=EditConstraints(preserve_story=True),
            style=EditStyle(narrative_shape="trailer", pacing="cinematic"),
            ambiguities=[],
            profile_applied=["trailer_template"]
        )

    if prompt_clean in ["make the edit feel cinematic", "make it feel cinematic", "make it cinematic"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="stylize", target=SegmentTarget(modality=["visual", "audio"], query="cinematic pacing"), priority="soft", confidence=0.8)
            ],
            constraints=EditConstraints(preserve_story=True),
            style=EditStyle(pacing="cinematic"),
            ambiguities=[],
            profile_applied=["cinematic_style"]
        )

    if prompt_clean in ["build tension before the reveal", "build tension"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="restructure", target=SegmentTarget(modality=["story", "emotion"], query="setup reveal payoff"), priority="soft", confidence=0.8)
            ],
            constraints=EditConstraints(preserve_story=True),
            style=EditStyle(narrative_shape="trailer", pacing="cinematic"),
            ambiguities=[],
            profile_applied=["tension_template"]
        )

    if prompt_clean in ["focus on me", "keep only shots where i'm speaking", "keep only the shots where i'm speaking"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="keep_only", target=SegmentTarget(modality=["person", "speech"], query="owner speaking"), priority="hard", confidence=0.9)
            ],
            constraints=EditConstraints(preserve_story=False),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["owner_focus"]
        )

    if prompt_clean in ["focus on the product", "keep only shots with the product visible"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="keep_only", target=SegmentTarget(modality=["object", "visual"], query="product visible"), priority="hard", confidence=0.85)
            ],
            constraints=EditConstraints(preserve_story=False),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=["product_focus"]
        )

    if prompt_clean in ["make it suitable for tiktok", "make this suitable for tiktok", "make a tiktok"]:
        return EditIntent(
            intent_id=f"fast_{uuid.uuid4().hex[:8]}",
            operations=[
                Operation(action="compress", target=SegmentTarget(modality=["story", "speech", "visual"], query="short-form highlights"), priority="hard", confidence=0.9)
            ],
            constraints=EditConstraints(target_duration_s=60.0, duration_mode="max", platform="tiktok", aspect_ratio="9:16", preserve_story=True),
            style=EditStyle(pacing="fast", narrative_shape="hook_first"),
            ambiguities=[],
            profile_applied=["tiktok_template"]
        )

    return None

def _build_compiler_input(prompt: str, video_summary: str, profile: EditorProfile) -> str:
    parts = [
        f"USER PROMPT: {prompt}",
        f"\n--- VIDEO CONTEXT ---\n{video_summary}",
    ]

    if profile.always or profile.platforms or profile.pacing:
        profile_lines = ["--- EDITOR PROFILE ---"]
        if profile.always:
            profile_lines.append(f"Standing instructions (always do): {', '.join(profile.always)}")
        if profile.never:
            profile_lines.append(f"Standing instructions (never do): {', '.join(profile.never)}")
        if profile.pacing:
            profile_lines.append(f"Preferred pacing: {profile.pacing}")
        if profile.platforms:
            profile_lines.append(f"Default platforms: {', '.join(profile.platforms)}")
        if profile.pause_tolerance_s != 1.2:
            profile_lines.append(f"Pause tolerance: {profile.pause_tolerance_s}s")
        parts.append("\n".join(profile_lines))

    parts.append(
        "\n--- INSTRUCTIONS ---\n"
        "Compile this prompt into an EditIntent JSON. "
        "Generate a UUID for intent_id. "
        "Follow all rules in your system prompt."
    )
    return "\n\n".join(parts)

def _post_validate_and_ground(intent: EditIntent, db: ProjectDB) -> EditIntent:
    video = db.get_video()
    if not video:
        return intent

    duration = video["duration_s"]
    entities = db.get_entities()
    entity_ids = {e["id"].lower() for e in entities}
    entity_labels = {e["label"].lower() for e in entities}

    
    for op in intent.operations:
        if op.confidence >= 0.85:
            op.confidence = 0.95
        elif op.confidence <= 0.50:
            op.confidence = 0.0

    
    if intent.constraints.target_duration_s is not None:
        if intent.constraints.target_duration_s > duration:
            intent.constraints.target_duration_s = duration
        if intent.constraints.target_duration_s <= 0:
            intent.constraints.target_duration_s = 3.0

    for op in intent.operations:
        query_lower = op.target.query.lower()
        referenced_entities = []
        words_in_query = re.findall(r"\w+", query_lower)
        for word in words_in_query:
            if word in entity_labels or f"person:{word}" in entity_ids:
                referenced_entities.append(word)

        if "person" in op.target.modality and not referenced_entities:
            has_match = False
            for el in entity_labels:
                if el in query_lower:
                    has_match = True
            if not has_match and entity_labels:
                intent.ambiguities.append(Ambiguity(
                    issue=f"Referenced person in query '{op.target.query}' was not found in the video.",
                    candidates=list(entity_labels),
                    blocking=False
                ))

    if not intent.intent_id:
        intent.intent_id = str(uuid.uuid4())[:8]

    return intent

def _apply_policy_checks(intent: EditIntent, db: ProjectDB, prompt_clean: str = "") -> EditIntent:
    readiness = db.get_readiness_level()
    needs_story = False

    for op in intent.operations:
        q = op.target.query.lower()
        if any(w in q for w in ["intro", "outro", "hook", "restructure", "story"]):
            needs_story = True

    if intent.style.cut_style in ("beat_cut", "match_cut"):
        needs_story = True

    if intent.style.narrative_shape in ("hook_first", "trailer", "highlight"):
        needs_story = True

    if needs_story and readiness < 4:
        intent.out_of_scope_reason = "This edit needs deeper analysis (~40s of reprocessing is required to build the story map)."

    coverage = db.get_coverage()
    
    # Check the original prompt for required capabilities
    if prompt_clean:
        if "speaker" in prompt_clean or "voice" in prompt_clean or "interviewer" in prompt_clean:
            if coverage.get("speakers", "unavailable") in ("unavailable", "degraded"):
                intent.out_of_scope_reason = "Cannot isolate speakers: Speaker diarization was not run or failed during ingestion. Please re-ingest with --deep."
        if "face" in prompt_clean or "person" in prompt_clean or "looking" in prompt_clean:
            if coverage.get("faces", "unavailable") in ("unavailable", "degraded"):
                intent.out_of_scope_reason = "Cannot filter by faces: Face detection was not run or failed during ingestion. Please re-ingest with --deep."

    for op in intent.operations:
        q = op.target.query.lower()
        if "speaker" in q or "voice" in q or "interviewer" in q:
            if coverage.get("speakers", "unavailable") in ("unavailable", "degraded"):
                intent.out_of_scope_reason = "Cannot isolate speakers: Speaker diarization was not run or failed during ingestion. Please re-ingest with --deep."
        if "face" in q or "person" in q or "looking" in q:
            if coverage.get("faces", "unavailable") in ("unavailable", "degraded"):
                intent.out_of_scope_reason = "Cannot filter by faces: Face detection was not run or failed during ingestion. Please re-ingest with --deep."
        if op.action == "remove" and "silence" in q:
            words = db.get_words()
            if not words:
                intent.out_of_scope_reason = "Cannot remove silences: No speech was detected in this video, so the entire video is classified as silence. This is likely a music-only or silent video."

    
    profile = load_profile()
    if profile.pacing and not intent.style.pacing:
        intent.style.pacing = profile.pacing
        intent.profile_applied.append("profile_pacing")
    if profile.platforms and not intent.constraints.platform:
        intent.constraints.platform = profile.platforms[0]
        intent.profile_applied.append("profile_platform")

    return intent

def _heuristic_intent(prompt_clean: str) -> EditIntent:
    """Deterministic last-resort parser when the LLM compile is unavailable or empty."""
    action = "remove"
    if any(w in prompt_clean for w in ["keep", "only", "preserve", "leave"]):
        action = "keep_only"

    target_str = prompt_clean
    for stop in ["remove ", "delete ", "cut ", "keep only ", "keep "]:
        if target_str.startswith(stop):
            target_str = target_str[len(stop):]

    return EditIntent(
        intent_id=f"heuristic_{uuid.uuid4().hex[:8]}",
        operations=[
            Operation(
                action=action,
                target=SegmentTarget(modality=["speech", "visual", "metadata"], query=target_str.strip() or prompt_clean),
                priority="hard",
                confidence=0.8
            )
        ],
        constraints=EditConstraints(),
        style=EditStyle(),
        ambiguities=[],
        profile_applied=[]
    )


def compile_intent(prompt: str, db: ProjectDB) -> EditIntent:
    """
    Compile natural-language prompt into structured EditIntent.
    P0–P4 pipeline flow.
    """
    console.print("  [dim]Compiling intent...[/dim]")

    # P0: normalize (strip polite prefixes, collapse whitespace) so fast-path
    # patterns and QA detection see the real instruction.
    prompt = normalize_prompt(prompt) or prompt

    # Guard: empty / whitespace-only / punctuation-only prompt.
    if not re.search(r"[a-zA-Z0-9À-￿]", prompt):
        return EditIntent(
            intent_id=f"empty_{uuid.uuid4().hex[:8]}",
            operations=[],
            constraints=EditConstraints(),
            style=EditStyle(),
            ambiguities=[],
            profile_applied=[],
            out_of_scope_reason="The prompt was empty. Please describe the edit you want (e.g., 'remove filler words').",
        )

    lang = _detect_language(prompt)
    if lang != "en":
        console.print(f"    [yellow]Non-English prompt detected ({lang}). Compiling natively...[/yellow]")


    resolved_prompt = prompt
    last_edits = db.get_all_edits()
    if last_edits:
        last_edit = last_edits[-1]
        if "undo the last one" in prompt.lower() or "revert last" in prompt.lower():
            resolved_prompt = f"revert last edit (prompt: '{last_edit['prompt']}')"


    prompt_clean = prompt.strip().lower()
    if _is_question(prompt_clean):
        ans = _resolve_qa_answer(prompt, db)
        return EditIntent(
            intent_id=f"qa_{uuid.uuid4().hex[:8]}",
            operations=[],
            constraints=EditConstraints(),
            style=EditStyle(narrative_shape="q_and_a"),
            ambiguities=[],
            profile_applied=[],
            out_of_scope_reason=ans
        )

    
    fast_intent = _try_fast_path(resolved_prompt, db)
    if fast_intent:
        console.print("    [green]Fast-path pattern match ✓ (Skipping LLM compile)[/green]")
        fast_intent = _post_validate_and_ground(fast_intent, db)
        fast_intent = _apply_policy_checks(fast_intent, db, prompt_clean)
        return fast_intent

    
    video_summary = build_video_summary(db)
    profile = load_profile()
    user_content = _build_compiler_input(resolved_prompt, video_summary, profile)

    try:
        intent = call_structured(
            prompt_name="intent_compiler",
            user_content=user_content,
            schema=EditIntent,
            effort="medium",
            db=db,
        )
        # Degenerate LLM output guard: an intent with no operations, no
        # blocking ambiguity, and no scope reason cannot drive the pipeline —
        # fall back to the deterministic parser rather than no-op'ing silently.
        if (
            not intent.operations
            and not intent.out_of_scope_reason
            and not any(a.blocking for a in intent.ambiguities)
        ):
            console.print("    [yellow]⚠ LLM returned an empty intent. Active Fallback: Heuristic NLP parser.[/yellow]")
            intent = _heuristic_intent(prompt_clean)
    except Exception as e:
        console.print(f"    [yellow]⚠ LLM intent compilation failed ({e}). Active Fallback: Heuristic NLP parser.[/yellow]")
        intent = _heuristic_intent(prompt_clean)

    
    intent = _post_validate_and_ground(intent, db)

    
    intent = _apply_policy_checks(intent, db, prompt_clean)

    return intent
