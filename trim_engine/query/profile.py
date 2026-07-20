"""
Editor Profile — standing preferences that merge with each prompt.

Profile is a YAML/JSON file at `profile.json`.
Explicit prompt instructions always override profile preferences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console

from trim_engine.config import PROFILE_PATH
from trim_engine.db import ProjectDB
from trim_engine.schemas import EditorProfile, PreferenceEvidence

console = Console()


class PromptPreferenceFeatures(BaseModel):
    """Features extracted from a prompt to update the profile."""
    pacing: str | None = Field(default=None, description="fast | medium | slow | cinematic")
    platform: str | None = Field(default=None, description="tiktok | reels | youtube | youtube_shorts")
    always_rules: list[str] = Field(default_factory=list, description="Rules to always apply, e.g., 'remove fillers'")
    never_rules: list[str] = Field(default_factory=list, description="Rules to never apply, e.g., 'never cut during jokes'")


def load_profile() -> EditorProfile:
    """Load editor profile from YAML/JSON. Returns empty profile if not found."""
    if not PROFILE_PATH.exists():
        return EditorProfile()

    try:
        text = PROFILE_PATH.read_text()

        
        if PROFILE_PATH.suffix in (".yml", ".yaml"):
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ImportError:
                data = json.loads(text)
        else:
            data = json.loads(text)
            
        
        if "version" not in data:
            data["version"] = 1
            if "platforms" not in data:
                data["platforms"] = []
            if "evidence" not in data:
                data["evidence"] = []

        return EditorProfile.model_validate(data)

    except Exception:
        return EditorProfile(version=1)


def save_profile(profile: EditorProfile) -> None:
    """Save the updated profile back to disk."""
    try:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        PROFILE_PATH.write_text(profile.model_dump_json(indent=2))
    except Exception as e:
        console.print(f"[yellow]Could not save profile: {e}[/yellow]")


def learn_from_prompt(prompt: str, db: ProjectDB) -> None:
    """
    Analyze the prompt to find recurring preferences.
    Updates the profile evidence tracker, and updates preferences when threshold is met.
    """
    
    from trim_engine.llm import call_structured

    system_prompt = (
        "You are an Editor Profile Assistant. Analyze the user's prompt to identify if they "
        "are requesting a standing preference for pacing, target platforms, or permanent keep/remove rules."
    )

    try:
        
        
        
        
        features = call_structured(
            prompt_name="profile_classifier",
            user_content=f"PROMPT: {prompt}",
            schema=PromptPreferenceFeatures,
            effort="low",
            db=db,
        )
    except Exception as e:
        console.print(f"[dim]Preference learning skipped: {e}[/dim]")
        return

    profile = load_profile()
    updated = False

    
    def track_preference(pref_key: str, pref_val: str, learned_from: str) -> int:
        for item in profile.evidence:
            if item.pref == f"{pref_key}:{pref_val}":
                item.count += 1
                return item.count
        
        profile.evidence.append(PreferenceEvidence(
            pref=f"{pref_key}:{pref_val}",
            learned_from=learned_from,
            count=1
        ))
        return 1

    
    if features.pacing:
        count = track_preference("pacing", features.pacing, prompt)
        if count >= 3 and profile.pacing != features.pacing:
            profile.pacing = features.pacing
            console.print(f"  [green]👤 Profile: auto-configured default pacing to '{features.pacing}' (requested {count} times)[/green]")
            updated = True

    
    if features.platform:
        count = track_preference("platform", features.platform, prompt)
        if count >= 3 and features.platform not in profile.platforms:
            profile.platforms.append(features.platform)
            console.print(f"  [green]👤 Profile: added default platform target '{features.platform}' (requested {count} times)[/green]")
            updated = True

    
    for rule in features.always_rules:
        count = track_preference("always", rule, prompt)
        if count >= 3 and rule not in profile.always:
            profile.always.append(rule)
            console.print(f"  [green]👤 Profile: added permanent rule '{rule}' to default behavior (requested {count} times)[/green]")
            updated = True

    
    for rule in features.never_rules:
        count = track_preference("never", rule, prompt)
        if count >= 3 and rule not in profile.never:
            profile.never.append(rule)
            console.print(f"  [green]👤 Profile: added permanent exclusion '{rule}' (requested {count} times)[/green]")
            updated = True

    if updated or features.pacing or features.platform or features.always_rules or features.never_rules:
        save_profile(profile)
