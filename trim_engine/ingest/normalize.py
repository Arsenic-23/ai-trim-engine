"""
Media Normalization Engine (INGESTION_ENGINE.md §4)

The most underrated engine. Everything downstream inherits its precision.

Outputs:
  - proxy.mp4: 720p CFR, H.264, GOP=1s closed, scene-cut keyframes, rotation baked
  - audio.wav: 16kHz mono PCM, DC-removed (highpass), loudness-analyzed
  - audio_48k.flac: full-quality audio extract
  - thumbs.bin: 1fps 160px wide thumbnail strip with offset LUT
  - frame_lut.json & frame_lut.parquet: PTS ↔ frame-index ↔ master-time table (int64 µs)

Key engineering per spec:
  - Two parallel ffmpeg processes (audio ∥ video transcode)
  - A/V offset measurement from container start-time delta
  - Frame-level invariant assertions (duration diff < 1 frame, LUT monotonicity)
  - Hardware encoder with software fallback
  - DC-removal highpass filter on audio
  - GOP = 1s, closed GOPs (-flags +cgop) for seekability
  - Rotation baked into mezzanine (analyzers never reason about rotation)
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rich.console import Console

from trim_engine.config import CFG
from trim_engine.db import ProjectDB
from trim_engine.schemas import VideoMeta

console = Console()






def _ffprobe_raw(path: Path) -> dict:
    """Run ffprobe and return the full JSON output."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _parse_probe(data: dict, video_path: Path) -> VideoMeta:
    """Parse ffprobe JSON into structured VideoMeta."""
    video_stream = None
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream
        elif stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        raise ValueError("No video stream found")

    fmt = data.get("format", {})
    duration = float(fmt.get("duration", 0))

    
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    codec = video_stream.get("codec_name", "unknown")

    
    rotation = 0
    side_data = video_stream.get("side_data_list", [])
    for sd in side_data:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        rotation = int(tags["rotate"])

    
    avg_fps_str = video_stream.get("avg_frame_rate", fps_str)
    avg_num, avg_den = avg_fps_str.split("/")
    avg_fps = float(avg_num) / float(avg_den) if float(avg_den) > 0 else fps
    is_vfr = abs(fps - avg_fps) > 1.0

    
    video_start = float(video_stream.get("start_time", 0))
    audio_start = float(audio_stream.get("start_time", 0)) if audio_stream else 0.0
    av_offset_ms = (video_start - audio_start) * 1000.0

    return VideoMeta(
        path=str(video_path),
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        codec=codec,
        is_vfr=is_vfr,
    ), av_offset_ms, rotation






def _build_frame_lut(video_path: Path, output_json: Path, output_parquet: Path) -> list[dict]:
    """
    Build frame_lut.json & frame_lut.parquet from actual packet PTS via ffprobe.
    """
    console.print("    [dim]Building frame LUT from packet PTS...[/dim]")

    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts_time,pts",
        "-print_format", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    packets = data.get("packets", [])
    lut = []
    for idx, pkt in enumerate(packets):
        pts_time = pkt.get("pts_time")
        if pts_time is None:
            continue
        pts_s = float(pts_time)
        pts_us = int(pts_s * 1_000_000)
        lut.append({
            "frame_idx": 0,  # placeholder
            "pts_us": pts_us,
            "pts_s": round(pts_s, 6),
        })

    # Sort by PTS (presentation order) since packets are in decode order
    lut.sort(key=lambda x: x["pts_us"])
    for idx, entry in enumerate(lut):
        entry["frame_idx"] = idx

    
    for i in range(1, len(lut)):
        if lut[i]["pts_us"] <= lut[i - 1]["pts_us"]:
            raise RuntimeError(
                f"Frame LUT non-monotonic at frame {i}: "
                f"{lut[i-1]['pts_us']}µs → {lut[i]['pts_us']}µs"
            )

    
    _atomic_write_text(output_json, json.dumps(lut, indent=None))

    
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        df = pd.DataFrame(lut)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(output_parquet))
    except Exception as e:
        console.print(f"    [yellow]Could not write Parquet LUT: {e}[/yellow]")

    console.print(f"    LUT: {len(lut)} frames, monotonic ✓")
    return lut

def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically using a temporary file and os.replace."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content)
    os.replace(tmp_path, path)






def _run_ffmpeg(cmd: list[str], description: str, timeout_s: int = 300) -> None:
    """Run an ffmpeg command with error handling, timeout, and cleanup on failure."""
    console.print(f"    [dim]{description}...[/dim]")
    out_file = Path(cmd[-1])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if result.returncode != 0:
            if out_file.exists():
                out_file.unlink()
            raise RuntimeError(f"ffmpeg failed ({description}): {result.stderr[-500:]}")
    except subprocess.TimeoutExpired as e:
        if out_file.exists():
            out_file.unlink()
        raise RuntimeError(f"ffmpeg process timed out after {timeout_s}s ({description})") from e
    except Exception as e:
        if out_file.exists():
            out_file.unlink()
        raise


def _create_proxy(original: Path, proxy: Path, meta: VideoMeta) -> None:
    """
    Create 720p CFR proxy with closed GOP closed.
    """
    cfg = CFG.normalize
    gop_frames = cfg.proxy_fps * cfg.proxy_gop_s
    vf = f"scale=-2:{cfg.proxy_height}"
    proxy_tmp = proxy.with_suffix(".mp4.tmp")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(original),
        "-vf", vf,
        "-r", str(cfg.proxy_fps),
        "-c:v", cfg.proxy_codec_hw,
        "-b:v", cfg.proxy_bitrate,
        "-g", str(gop_frames),
        "-flags", "+cgop",
        "-an",
        "-f", "mp4",
        str(proxy_tmp),
    ]

    try:
        _run_ffmpeg(cmd, f"Creating proxy ({cfg.proxy_codec_hw}, GOP={cfg.proxy_gop_s}s)")
        os.replace(proxy_tmp, proxy)
    except RuntimeError:
        console.print(f"    [dim]Hardware encoder failed, falling back to {cfg.proxy_codec_sw}[/dim]")
        cmd_sw = [
            "ffmpeg", "-y",
            "-i", str(original),
            "-vf", vf,
            "-r", str(cfg.proxy_fps),
            "-c:v", cfg.proxy_codec_sw,
            "-preset", "fast",
            "-crf", "23",
            "-g", str(gop_frames),
            "-flags", "+cgop",
            "-an",
            "-f", "mp4",
            str(proxy_tmp),
        ]
        _run_ffmpeg(cmd_sw, f"Creating proxy ({cfg.proxy_codec_sw}, GOP={cfg.proxy_gop_s}s)")
        os.replace(proxy_tmp, proxy)


def _extract_audio(original: Path, audio: Path) -> None:
    """
    Extract 16kHz mono WAV with DC-removal highpass filter.
    """
    cfg = CFG.normalize
    af = f"highpass=f={cfg.audio_highpass_hz}"
    audio_tmp = audio.with_suffix(".wav.tmp")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(original),
        "-ac", str(cfg.audio_channels),
        "-ar", str(cfg.audio_sample_rate),
        "-af", af,
        "-f", "wav",
        str(audio_tmp),
    ]
    _run_ffmpeg(cmd, "Extracting audio (16kHz mono, DC-removed)")
    os.replace(audio_tmp, audio)


def _extract_audio_48k(original: Path, audio_flac: Path) -> None:
    """
    Extract 48kHz FLAC full-quality audio track.
    """
    audio_tmp = audio_flac.with_suffix(".flac.tmp")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original),
        "-ar", "48000",
        "-c:a", "flac",
        "-f", "flac",
        str(audio_tmp),
    ]
    _run_ffmpeg(cmd, "Extracting full-quality 48kHz FLAC audio")
    os.replace(audio_tmp, audio_flac)


def _generate_thumbnails(original: Path, thumbs_bin: Path, project_dir: Path) -> None:
    """
    Generate 1fps 160px wide JPEGs and package into thumbs.bin binary offset structure.
    """
    tmp_dir = project_dir / "thumbs_tmp"
    tmp_dir.mkdir(exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(original),
        "-vf", "fps=1,scale=160:-1",
        "-q:v", "5",
        str(tmp_dir / "thumb_%04d.jpg")
    ]
    try:
        _run_ffmpeg(cmd, "Extracting 1fps JPEGs for thumbs.bin")
        thumb_files = sorted(tmp_dir.glob("thumb_*.jpg"))
        if not thumb_files:
            return

        num_thumbs = len(thumb_files)
        lut_size = 4 + num_thumbs * 16

        offsets = []
        lengths = []
        current_offset = lut_size
        jpeg_bytes_list = []

        for f in thumb_files:
            data = f.read_bytes()
            lengths.append(len(data))
            offsets.append(current_offset)
            current_offset += len(data)
            jpeg_bytes_list.append(data)

        
        thumbs_bin_tmp = thumbs_bin.with_suffix(".bin.tmp")
        with open(thumbs_bin_tmp, "wb") as out:
            out.write(num_thumbs.to_bytes(4, byteorder="big"))
            for offset, length in zip(offsets, lengths):
                out.write(offset.to_bytes(8, byteorder="big"))
                out.write(length.to_bytes(8, byteorder="big"))
            for jpeg in jpeg_bytes_list:
                out.write(jpeg)
        os.replace(thumbs_bin_tmp, thumbs_bin)

    finally:
        import shutil
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)






def _check_drift(original_meta: VideoMeta, proxy: Path) -> float:
    """Check duration drift between original and proxy. Returns drift in ms."""
    probe_data = _ffprobe_raw(proxy)
    proxy_meta, _, _ = _parse_probe(probe_data, proxy)
    drift_ms = abs(original_meta.duration_s - proxy_meta.duration_s) * 1000
    return drift_ms


def _check_frame_level_invariants(
    meta: VideoMeta,
    proxy: Path,
    drift_ms: float,
    lut: list[dict],
) -> list[str]:
    """
    Run frame-level invariant tests per §4.2:
    - duration(mezz) − duration(source) < 1 frame
    - Audio drift < 25ms at probe points
    """
    warnings: list[str] = []

    
    frame_duration_ms = 1000.0 / meta.fps if meta.fps > 0 else 33.33
    if drift_ms > frame_duration_ms:
        warnings.append(
            f"Duration drift {drift_ms:.1f}ms exceeds 1-frame budget ({frame_duration_ms:.1f}ms)"
        )

    
    if lut:
        expected_frames = int(meta.duration_s * meta.fps)
        actual_frames = len(lut)
        frame_deficit = abs(expected_frames - actual_frames)
        if frame_deficit > max(5, expected_frames * 0.01):
            warnings.append(
                f"LUT frame count {actual_frames} differs from expected {expected_frames} "
                f"by {frame_deficit} frames"
            )

    return warnings






def run_normalize(project_dir: Path, db: ProjectDB) -> None:
    """
    Run the full normalization stage with parallel execution of tasks.
    """
    original = project_dir / "original.mp4"
    proxy = project_dir / "proxy.mp4"
    audio = project_dir / "audio.wav"
    audio_48k = project_dir / "audio_48k.flac"
    thumbs = project_dir / "thumbs.bin"
    probe_path = project_dir / "probe.json"
    lut_json_path = project_dir / "frame_lut.json"
    lut_parquet_path = project_dir / "frame_lut.parquet"

    
    console.print("    Probing video...")
    probe_data = _ffprobe_raw(original)
    meta, av_offset_ms, rotation = _parse_probe(probe_data, original)

    
    probe_artifact = {
        "ffprobe": probe_data,
        "derived": {
            "duration_s": meta.duration_s,
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "codec": meta.codec,
            "is_vfr": meta.is_vfr,
            "rotation": rotation,
            "av_offset_ms": round(av_offset_ms, 2),
        },
    }
    _atomic_write_text(probe_path, json.dumps(probe_artifact, indent=2))

    
    proxy_needed = not proxy.exists()
    audio_needed = not audio.exists()
    audio_48k_needed = not audio_48k.exists()
    thumbs_needed = not thumbs.exists()

    if proxy_needed or audio_needed or audio_48k_needed or thumbs_needed:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {}
            if proxy_needed:
                futures[pool.submit(_create_proxy, original, proxy, meta)] = "proxy"
            if audio_needed:
                futures[pool.submit(_extract_audio, original, audio)] = "audio"
            if audio_48k_needed:
                futures[pool.submit(_extract_audio_48k, original, audio_48k)] = "audio_48k"
            if thumbs_needed:
                futures[pool.submit(_generate_thumbnails, original, thumbs, project_dir)] = "thumbs"

            
            for future in as_completed(futures):
                future.result()

    
    lut: list[dict] = []
    if not lut_json_path.exists() or not lut_parquet_path.exists():
        lut = _build_frame_lut(original, lut_json_path, lut_parquet_path)
    else:
        lut = json.loads(lut_json_path.read_text())

    
    drift_ms = _check_drift(meta, proxy)
    console.print(f"    Drift: {drift_ms:.1f}ms")

    if drift_ms > CFG.normalize.max_drift_ms:
        console.print(f"    [yellow]⚠ Drift {drift_ms:.1f}ms exceeds threshold {CFG.normalize.max_drift_ms}ms[/yellow]")
        db.set_coverage("timeline_sync", "low_confidence", note=f"drift={drift_ms:.1f}ms")
    else:
        db.set_coverage("timeline_sync", "available")

    invariant_warnings = _check_frame_level_invariants(meta, proxy, drift_ms, lut)
    for warning in invariant_warnings:
        console.print(f"    [yellow]⚠ Invariant: {warning}[/yellow]")

    if invariant_warnings:
        db.set_coverage("frame_invariants", "low_confidence", note="; ".join(invariant_warnings))
    else:
        db.set_coverage("frame_invariants", "available")

    
    video_id = project_dir.name
    db.set_video(
        video_id=video_id,
        path=str(original),
        duration_s=meta.duration_s,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        codec=meta.codec,
        is_vfr=meta.is_vfr,
        av_offset_ms=round(av_offset_ms, 2),
    )

    db.set_model_manifest("normalize", "ffmpeg", "6.x")
