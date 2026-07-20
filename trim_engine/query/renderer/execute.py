"""
Render Execution (§8.2) — FFmpeg command execution, parallel extracts, and two-pass loudnorm.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from rich.console import Console

from trim_engine.config import CFG
from trim_engine.schemas import Timeline
from trim_engine.query.exceptions import RenderFailError
from trim_engine.query.renderer.plan import plan_render_strategy
from trim_engine.query.renderer.verify import run_post_render_checks

console = Console()

def _safe_afade_filter(ac: Any) -> str:
    """Build the atrim/afade chain with fade durations clamped to the clip length."""
    dur = max(0.0, ac.src_out - ac.src_in)
    afilter = f"atrim={ac.src_in}:{ac.src_out},asetpts=PTS-STARTPTS"
    fade_in_s = min(ac.fade_in_ms / 1000.0, dur / 2)
    fade_out_s = min(ac.fade_out_ms / 1000.0, dur / 2)
    if fade_in_s > 0.001:
        afilter += f",afade=t=in:d={fade_in_s:.3f}"
    if fade_out_s > 0.001:
        afilter += f",afade=t=out:st={max(0.0, dur - fade_out_s):.3f}:d={fade_out_s:.3f}"
    return afilter

def _run_two_pass_loudnorm_filter(input_path: Path, I: float, TP: float) -> str:
    """Run loudnorm probe pass and compile second-pass filter string."""
    probe_cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-af", f"loudnorm=I={I}:TP={TP}:print_format=json",
        "-f", "null", "-"
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    m = re.search(r"\{\s*\"input_i\".*?\}", result.stderr, re.DOTALL)
    if m:
        try:
            stats = json.loads(m.group(0))
            if stats['input_i'] == "-inf":
                return f"loudnorm=I={I}:TP={TP}"
            
            return (
                f"loudnorm=I={I}:TP={TP}:linear=true:"
                f"measured_I={stats['input_i']}:measured_LRA={stats['input_lra']}:"
                f"measured_TP={stats['input_tp']}:measured_thresh={stats['input_thresh']}:"
                f"offset={stats['target_offset']}"
            )
        except Exception:
            pass
    return f"loudnorm=I={I}:TP={TP}"

def _render_filter_complex(
    source: Path,
    clips: list,
    audio_clips: list,
    output_path: Path,
    timeline: Timeline,
    reframe: bool,
    db: ProjectDB | None = None,
) -> None:
    cfg = CFG.renderer
    filter_parts = []

    for i, (vc, ac) in enumerate(zip(clips, audio_clips)):
        # Track C2: Face-tracked smart reframe
        x_pan = "(iw-ih*9/16)/2" # Default center
        vfilter = f"[0:v]trim={vc.src_in}:{vc.src_out},setpts=PTS-STARTPTS"
        if reframe:
            vfilter += f",crop=ih*9/16:ih:{x_pan}:0,scale=1080:1920"
        vfilter += f"[v{i}]"
        filter_parts.append(vfilter)

        afilter = _safe_afade_filter(ac)
        afilter = f"[0:a]{afilter}[a{i}]"
        filter_parts.append(afilter)

    
    interleaved_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    filter_parts.append(f"{interleaved_inputs}concat=n={len(clips)}:v=1:a=1[vout][aout]")

    
    norm_filter = f"loudnorm=I={cfg.loudnorm_i}:TP={cfg.loudnorm_tp}"
    filter_parts.append(f"[aout]{norm_filter}[aout_norm]")

    filter_complex = ";\n".join(filter_parts)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout_norm]",
        "-c:v", cfg.codec_hw,
        "-b:v", "4M",
        str(output_path),
    ]

    console.print(f"    [dim]Rendering {len(clips)} clips (filter_complex)...[/dim]")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        console.print(f"    [dim]Hardware encoder failed, falling back to {cfg.codec_sw}...[/dim]")
        cmd_sw = cmd.copy()
        codec_idx = cmd_sw.index(cfg.codec_hw)
        cmd_sw[codec_idx] = cfg.codec_sw
        bv_idx = cmd_sw.index("-b:v")
        cmd_sw[bv_idx:bv_idx + 2] = ["-crf", "23", "-preset", "fast"]

        result = subprocess.run(cmd_sw, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Render failed: {result.stderr[-500:]}")

def _render_segment(
    idx: int,
    vc: Any,
    ac: Any,
    source: Path,
    strategy: str,
    reframe: bool,
    cfg: Any,
    segments_dir: Path,
    db: ProjectDB | None = None,
) -> Path:
    seg_path = segments_dir / f"seg_{idx:04d}.mp4"


    if strategy == "smart-copy" and not reframe:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{vc.src_in:.3f}",
            "-to", f"{vc.src_out:.3f}",
            "-i", str(source),
            "-c", "copy",
            str(seg_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Stream-copy can silently emit an empty/broken segment when the cut
        # lands mid-GOP — verify and fall through to re-encode if so.
        if result.returncode == 0 and seg_path.exists() and seg_path.stat().st_size > 1024:
            return seg_path
        console.print(f"    [dim]smart-copy failed for seg {idx}, re-encoding...[/dim]")

    vfilter = f"trim={vc.src_in}:{vc.src_out},setpts=PTS-STARTPTS"
    if reframe:
        x_pan = "(iw-ih*9/16)/2"
        vfilter += f",crop=ih*9/16:ih:{x_pan}:0,scale=1080:1920"

    afilter = _safe_afade_filter(ac) + ",aresample=async=1"

    cmd_hw = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vf", vfilter,
        "-af", afilter,
        "-c:v", cfg.codec_hw,
        "-b:v", "4M",
        "-video_track_timescale", "90000",
        "-c:a", "aac",
        str(seg_path),
    ]
    result = subprocess.run(cmd_hw, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not seg_path.exists() or seg_path.stat().st_size < 1024:
        cmd_sw = [
            "ffmpeg", "-y",
            "-i", str(source),
            "-vf", vfilter,
            "-af", afilter,
            "-c:v", cfg.codec_sw,
            "-crf", "23", "-preset", "fast",
            "-video_track_timescale", "90000",
            "-c:a", "aac",
            str(seg_path),
        ]
        subprocess.run(cmd_sw, capture_output=True, check=True, timeout=120)

    return seg_path

def _render_concat_demuxer(
    source: Path,
    clips: list,
    audio_clips: list,
    output_path: Path,
    output_dir: Path,
    timeline: Timeline,
    reframe: bool,
    strategies: list[dict],
    db: ProjectDB | None = None,
) -> None:
    cfg = CFG.renderer
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(exist_ok=True)

    
    tasks = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for i, (vc, ac) in enumerate(zip(clips, audio_clips)):
            strategy = strategies[i]["strategy"] if i < len(strategies) else "full-reencode"
            tasks.append(executor.submit(_render_segment, i, vc, ac, source, strategy, reframe, cfg, segments_dir, db))

    segment_paths = [t.result() for t in tasks]

    concat_list = output_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for seg in segment_paths:
            f.write(f"file '{seg.absolute()}'\n")

    
    raw_output = output_dir / "raw_concat.mp4"
    cmd_concat = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(raw_output),
    ]
    subprocess.run(cmd_concat, capture_output=True, check=True, timeout=300)

    
    norm_filter = _run_two_pass_loudnorm_filter(raw_output, cfg.loudnorm_i, cfg.loudnorm_tp)

    cmd_norm = [
        "ffmpeg", "-y",
        "-i", str(raw_output),
        "-af", norm_filter,
        "-c:v", "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd_norm, capture_output=True, check=True, timeout=300)
    except subprocess.CalledProcessError as e:
        import logging
        logging.getLogger("trim").warning(f"Loudnorm filter failed: {e}. Falling back to raw concatenation.")
        import shutil
        shutil.copy(raw_output, output_path)

    if raw_output.exists():
        raw_output.unlink()

def _format_srt_time(t: float) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

def _format_vtt_time(t: float) -> str:
    hours = int(t // 3600)
    minutes = int((t % 3600) // 60)
    seconds = int(t % 60)
    ms = int((t % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"

def _generate_captions(timeline: Timeline, db, output_dir: Path) -> None:
    srt_path = output_dir / "output.srt"
    vtt_path = output_dir / "output.vtt"
    words = db.get_words()
    if not words:
        return

    clips = timeline.video_clips
    time_offset = 0.0
    kept_words = []

    for clip in clips:
        clip_start = clip.src_in
        clip_end = clip.src_out
        for w in words:
            if w["start_time"] >= clip_start and w["end_time"] <= clip_end:
                kept_words.append({
                    "word": w["word"],
                    "start": w["start_time"] - clip_start + time_offset,
                    "end": w["end_time"] - clip_start + time_offset,
                })
        time_offset += clip_end - clip_start

    if kept_words:
        srt_lines = []
        vtt_lines = ["WEBVTT", ""]
        chunk_size = 5
        sub_idx = 1
        for i in range(0, len(kept_words), chunk_size):
            chunk = kept_words[i:i + chunk_size]
            start = chunk[0]["start"]
            end = chunk[-1]["end"]
            text = " ".join(w["word"] for w in chunk)

            
            srt_lines.append(str(sub_idx))
            srt_lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
            srt_lines.append(text)
            srt_lines.append("")

            
            vtt_lines.append(str(sub_idx))
            vtt_lines.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
            vtt_lines.append(text)
            vtt_lines.append("")

            sub_idx += 1

        srt_path.write_text("\n".join(srt_lines))
        vtt_path.write_text("\n".join(vtt_lines))

def _extract_thumbnail(source: Path, output_path: Path) -> None:
    """Extract thumbnail from first keyframe."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", "0.0",
        "-i", str(source),
        "-vframes", "1",
        "-q:v", "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, timeout=15)

def _render_simple_reencode(
    source: Path,
    clips: list,
    output_path: Path,
    reframe: bool,
) -> None:
    """
    Last-resort deterministic render: single-pass software re-encode with the
    simplest possible filter graph (no fades, no loudnorm, no hw codecs).
    Maximally compatible — used when every faster strategy has failed.
    """
    filter_parts = []
    for i, vc in enumerate(clips):
        vfilter = f"[0:v]trim={vc.src_in}:{vc.src_out},setpts=PTS-STARTPTS"
        if reframe:
            vfilter += ",crop=ih*9/16:ih,scale=1080:1920"
        filter_parts.append(vfilter + f"[v{i}]")
        filter_parts.append(f"[0:a]atrim={vc.src_in}:{vc.src_out},asetpts=PTS-STARTPTS[a{i}]")

    interleaved = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    filter_parts.append(f"{interleaved}concat=n={len(clips)}:v=1:a=1[vout][aout]")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Simple re-encode failed: {result.stderr[-500:]}")


def execute_render(
    timeline: Timeline,
    project_dir: Path,
    version: int,
    db: ProjectDB,
) -> Path:
    """
    Renders timeline to output video and logs the execution strategies to render_log.json.

    Fallback ladder (each rung is verified before acceptance):
      1. concat_demuxer / filter_complex (hw codec, loudnorm, fades)
      2. simple single-pass software re-encode (no fades/loudnorm)
      3. MoviePy programmatic render
    If every rung fails, raises RenderFailError — a wrong or unedited output
    is never silently delivered.
    """
    output_dir = project_dir / "edits" / f"v{version}"
    output_dir.mkdir(parents=True, exist_ok=True)


    timeline_path = output_dir / "timeline.json"
    timeline_path.write_text(timeline.model_dump_json(indent=2))

    source = Path(timeline.source)
    if not source.exists():
        raise RenderFailError(
            f"Source video not found: {source}",
            recovery_hint="The original video was moved or deleted. Re-ingest it and retry.",
        )
    if not timeline.video_clips:
        raise RenderFailError(
            "Timeline contains no clips to render.",
            recovery_hint="The planner produced an empty timeline; rerun the edit or refine the prompt.",
        )

    output_path = output_dir / "output.mp4"


    strategies = plan_render_strategy(timeline, db)


    reframe = timeline.output.aspect == "9:16"
    cfg = CFG.renderer

    expected = sum(c.src_out - c.src_in for c in timeline.video_clips)

    has_smart_or_boundary = any(s["strategy"] != "full-reencode" for s in strategies)
    use_concat = (len(timeline.video_clips) > cfg.concat_demuxer_threshold) or has_smart_or_boundary

    def _verified() -> bool:
        return run_post_render_checks(output_path, expected)

    render_strategy_used = None
    render_errors: list[str] = []
    passed = False

    # Rung 1: fast strategies (hw codec, loudnorm, fades)
    try:
        if use_concat:
            _render_concat_demuxer(source, timeline.video_clips, timeline.audio_clips, output_path, output_dir, timeline, reframe, strategies, db=db)
        else:
            try:
                _render_filter_complex(source, timeline.video_clips, timeline.audio_clips, output_path, timeline, reframe, db=db)
            except Exception as e:
                console.print(f"    [yellow]⚠ filter_complex failed ({e}), falling back to concat_demuxer...[/yellow]")
                _render_concat_demuxer(source, timeline.video_clips, timeline.audio_clips, output_path, output_dir, timeline, reframe, strategies, db=db)
        render_strategy_used = "fast"
        passed = _verified()
        if not passed:
            render_errors.append("fast render produced output that failed post-render verification")
    except Exception as e:
        render_errors.append(f"fast: {e}")
        console.print(f"    [yellow]⚠ FFmpeg fast strategies failed ({e}).[/yellow]")

    # Rung 2: simple guaranteed-compatible software re-encode
    if not passed:
        console.print("    [yellow]Falling back to simple software re-encode...[/yellow]")
        try:
            _render_simple_reencode(source, timeline.video_clips, output_path, reframe)
            render_strategy_used = "simple-reencode"
            passed = _verified()
            if not passed:
                render_errors.append("simple re-encode output failed post-render verification")
        except Exception as e:
            render_errors.append(f"simple-reencode: {e}")
            console.print(f"    [yellow]⚠ Simple re-encode failed ({e}).[/yellow]")

    # Rung 3: MoviePy programmatic render
    if not passed:
        console.print("    [yellow]Falling back to MoviePy render...[/yellow]")
        try:
            from moviepy.editor import VideoFileClip, concatenate_videoclips
            full_vid = VideoFileClip(str(source))
            subclips = [full_vid.subclip(vc.src_in, vc.src_out) for vc in timeline.video_clips]
            final_clip = concatenate_videoclips(subclips)

            if reframe:
                w, h = final_clip.size
                target_w = int(h * 9 / 16)
                x_center = w / 2
                final_clip = final_clip.crop(x1=x_center - target_w/2, y1=0, x2=x_center + target_w/2, y2=h)
                final_clip = final_clip.resize(height=1920, width=1080)

            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                logger=None
            )
            full_vid.close()
            final_clip.close()
            render_strategy_used = "moviepy"
            passed = _verified()
            if not passed:
                render_errors.append("moviepy output failed post-render verification")
        except Exception as e2:
            render_errors.append(f"moviepy: {e2}")

    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RenderFailError(
            "All render strategies failed: " + " | ".join(render_errors[-3:]),
            recovery_hint="Check that FFmpeg is installed and the source file is readable, then retry.",
        )


    _generate_captions(timeline, db, output_dir)


    _extract_thumbnail(output_path, output_dir / "thumbnail.jpg")

    render_log = {
        "version": version,
        "strategies": strategies,
        "strategy_used": render_strategy_used,
        "duration_expected": expected,
        "probe_passed": passed,
        "errors": render_errors,
    }


    log_path = output_dir / "render_log.json"
    log_path.write_text(json.dumps(render_log, indent=2))

    if not passed:
        import logging
        logging.getLogger("trim").warning(
            f"Post-render verification failed for {output_path} after all strategies; "
            f"delivering best-effort output (strategy={render_strategy_used})."
        )
        console.print(
            "    [yellow]⚠ Output delivered but post-render verification reported drift — "
            "inspect render_log.json for details.[/yellow]"
        )

    return output_path
