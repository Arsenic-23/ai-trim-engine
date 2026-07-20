"""
Demucs Audio Separation

Separates a video's audio into `vocals.wav` (speech) and `no_vocals.wav` (music/noise)
using Demucs. This provides a clean speech stem for better VAD and a clean music stem
for beat tracking.
"""

import sys
import subprocess
import time
from pathlib import Path
from rich.console import Console

console = Console()

def run_audio_separation(project_dir: Path, force: bool = False):
    """
    Run Demucs to separate the original audio into speech (vocals) and music (no_vocals).
    """
    start_time = time.time()
    
    # Check if we already have the separated audio
    out_dir = project_dir / "htdemucs" / "audio"
    vocals_path = out_dir / "vocals.wav"
    no_vocals_path = out_dir / "no_vocals.wav"
    
    if not force and vocals_path.exists() and no_vocals_path.exists():
        console.print(f"  ✓ Audio separation (cached) [{time.time() - start_time:.2f}s]")
        return
        
    audio_path = project_dir / "audio.wav"
    if not audio_path.exists():
        console.print("  [red]Audio separation failed: audio.wav not found.[/red]")
        return

    console.print("  Running Demucs audio separation... (this may take a while)")
    
    # We use htdemucs model and --two-stems=vocals to output just 2 files
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", "htdemucs",
        "--two-stems=vocals",
        "-o", str(project_dir),
        str(audio_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        console.print(f"  ✓ Audio separation complete [{time.time() - start_time:.2f}s]")
    except subprocess.CalledProcessError as e:
        console.print(f"  [red]Demucs failed:[/red] {e.stderr}")
