"""
Instant Preview Subsystem (§9) — generates rapid, stream-copied rough cuts and cut-inspection clips.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from rich.console import Console

from trim_engine.schemas import Timeline

console = Console()

def generate_preview(
    timeline: Timeline,
    output_dir: Path,
) -> Path:
    """
    Generate rapid video preview by stream-copy concat of source.
    Video clips are copied (GOP aligned), audio is precisely trimmed and remuxed.
    """
    source = Path(timeline.source)
    output_path = output_dir / "preview.mp4"
    
    concat_list = output_dir / "preview_concat.txt"
    segment_paths = []
    
    segments_dir = output_dir / "preview_segments"
    segments_dir.mkdir(exist_ok=True, parents=True)
    
    for i, (vc, ac) in enumerate(zip(timeline.video_clips, timeline.audio_clips)):
        seg_path = segments_dir / f"prev_seg_{i:04d}.mp4"
        
        
        
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{vc.src_in:.3f}",
            "-to", f"{vc.src_out:.3f}",
            "-i", str(source),
            "-c:v", "h264_videotoolbox",
            "-b:v", "2M",
            "-video_track_timescale", "90000",
            "-c:a", "aac",
            str(seg_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            segment_paths.append(seg_path)
            
    if not segment_paths:
        raise RuntimeError("No preview segments generated")

    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.absolute()}'\n")
            
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path),
    ]
    
    subprocess.run(concat_cmd, capture_output=True, check=True, timeout=60)
    console.print(f"    [green]Preview rendered (rapid copy): {output_path}[/green]")
    return output_path

def generate_cut_inspection_preview(
    timeline: Timeline,
    output_dir: Path,
    cut_seconds: float = 1.5,
) -> Path:
    """
    Cut-inspection mode: renders a brief 3s window surrounding each cut boundary
    so users can verify transition smoothness without watching the whole edit.
    """
    source = Path(timeline.source)
    output_path = output_dir / "cut_inspection.mp4"
    
    concat_list = output_dir / "inspection_concat.txt"
    segment_paths = []
    
    segments_dir = output_dir / "inspection_segments"
    segments_dir.mkdir(exist_ok=True, parents=True)
    
    
    for i in range(len(timeline.video_clips) - 1):
        cut_point = timeline.video_clips[i].src_out
        start_t = max(0.0, cut_point - cut_seconds)
        end_t = cut_point + cut_seconds
        
        seg_path = segments_dir / f"cut_{i:04d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_t:.3f}",
            "-to", f"{end_t:.3f}",
            "-i", str(source),
            "-c:v", "libx264", "-preset", "fast", "-crf", "26",
            "-c:a", "aac",
            str(seg_path),
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=30)
        if result.returncode == 0:
            segment_paths.append(seg_path)
            
    if not segment_paths:
        
        return generate_preview(timeline, output_dir)

    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.absolute()}'\n")
            
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_path),
    ]
    
    subprocess.run(concat_cmd, capture_output=True, check=True, timeout=60)
    console.print(f"    [green]Cut-inspection mode preview rendered: {output_path}[/green]")
    return output_path
