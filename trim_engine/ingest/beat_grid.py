"""
Beat Grid extraction

Uses librosa on the separated music stem to generate a 10Hz beat/downbeat grid.
"""

import time
from pathlib import Path
from rich.console import Console
from trim_engine.db import ProjectDB

console = Console()

def run_beat_grid(project_dir: Path, db: ProjectDB, force: bool = False):
    """
    Extract beats from the demucs-separated music stem and save to the DB.
    """
    start_time = time.time()
    
    if not force and db.get_beats():
        console.print(f"  ✓ Beat grid (cached) [{time.time() - start_time:.2f}s]")
        return
        
    no_vocals_path = project_dir / "htdemucs" / "audio" / "no_vocals.wav"
    if not no_vocals_path.exists():
        console.print("  [yellow]Beat grid skipped: no music stem found (run audio_separation first)[/yellow]")
        return

    import librosa
    
    console.print("  Extracting beat grid with librosa...")
    y, sr = librosa.load(str(no_vocals_path), sr=None, mono=True)
    
    # Extract beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    
    # Save to DB
    beats = [(float(t), 1) for t in beat_times]  # Simple beat tracking for now (treating all beats as downbeats)
    db.insert_beats(beats)
    
    console.print(f"  ✓ Extracted {len(beats)} beats [{time.time() - start_time:.2f}s]")
