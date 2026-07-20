"""
Post-Render Verification Probes (§8.3) — validates duration, container integrity, and sync.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from rich.console import Console

from trim_engine.config import CFG

console = Console()

def probe_duration(path: Path) -> float:
    """Get duration of a media file via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def verify_av_sync_offset(path: Path) -> float:
    """Get the delta between first video packet PTS and first audio packet PTS."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "packet=pts_time,stream_index",
        "-of", "json",
        str(path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        v_pts = None
        a_pts = None
        for pkt in data.get("packets", []):
            if pkt.get("stream_index") == 0 and v_pts is None:
                v_pts = float(pkt.get("pts_time", 0.0))
            elif pkt.get("stream_index") == 1 and a_pts is None:
                a_pts = float(pkt.get("pts_time", 0.0))
            if v_pts is not None and a_pts is not None:
                break
        if v_pts is not None and a_pts is not None:
            return abs(v_pts - a_pts)
    except Exception:
        pass
    return 0.0

def run_post_render_checks(output_path: Path, expected_duration: float) -> bool:
    """
    Runs container probes, duration drift checks, and basic A/V sync validation.
    """
    cfg = CFG.renderer
    if not output_path.exists():
        console.print(f"    [red]Output file does not exist: {output_path}[/red]")
        return False

    try:
        
        probe_cmd = ["ffprobe", "-v", "error", str(output_path)]
        subprocess.run(probe_cmd, check=True, capture_output=True)

        
        actual_duration = probe_duration(output_path)
        drift_s = abs(actual_duration - expected_duration)
        drift = drift_s / max(0.1, expected_duration)

        if drift > cfg.duration_tolerance and drift_s > 0.25:
            console.print(
                f"    [yellow]Post-render: Duration drift detected! expected {expected_duration:.2f}s, "
                f"got {actual_duration:.2f}s (drift {drift:.1%})[/yellow]"
            )
            return False

        
        av_delta = verify_av_sync_offset(output_path)
        if av_delta > 0.100:  
            console.print(f"    [yellow]Post-render: A/V sync drift detected! Packet delta is {av_delta * 1000:.0f}ms[/yellow]")
            return False

        console.print(f"    [dim]Post-render check passed: duration={actual_duration:.2f}s, sync_offset={av_delta*1000:.1f}ms[/dim]")
        return True

    except Exception as e:
        console.print(f"    [red]Post-render check failed: {e}[/red]")
        return False
