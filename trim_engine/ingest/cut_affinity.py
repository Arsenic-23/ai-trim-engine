"""
Cut-Point Engine — computes a 10 Hz cut affinity curve for optimal snapping.
Master Plan Track A: Combines breath (+3), blink (+2), motion-minimum (+1),
mid-word (-10), mid-gesture (-2).
"""

from pathlib import Path
import numpy as np
import cv2
import json
from rich.console import Console

from trim_engine.db import ProjectDB

console = Console()

def run_cut_affinity(project_dir: Path, db: ProjectDB) -> None:
    console.print("[dim](A1/A2) Computing Cut-Point Affinity Curve...[/dim]")
    video_path = project_dir / "original.mp4"
    if not video_path.exists():
        console.print("  [red]original.mp4 not found[/red]")
        return

    video_meta = db.get_video()
    if not video_meta:
        return
    duration = video_meta["duration_s"]
    fps = video_meta["fps"]
    
    # We compute at 10 Hz
    affinity_fps = 10.0
    num_frames = int(duration * affinity_fps)
    affinity = np.zeros(num_frames, dtype=np.float32)
    
    # 1. Motion Minima (+1) & Mid-gesture (-2) using Farneback Optical Flow
    console.print("  [dim]Computing optical flow motion...[/dim]")
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_FPS, affinity_fps) # read at 10Hz
    
    ret, prev = cap.read()
    if ret:
        prev_gray = cv2.cvtColor(cv2.resize(prev, (320, 180)), cv2.COLOR_BGR2GRAY)
        motion_energy = [0.0]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_energy.append(float(np.mean(mag)))
            prev_gray = gray
            if len(motion_energy) >= num_frames:
                break
                
        cap.release()
        
        # Normalize motion
        me = np.array(motion_energy)
        if len(me) < num_frames:
            me = np.pad(me, (0, num_frames - len(me)))
        elif len(me) > num_frames:
            me = me[:num_frames]
            
        me_smooth = cv2.GaussianBlur(me, (5, 1), 0).flatten()
        # Find minima and maxima
        import scipy.signal
        peaks, _ = scipy.signal.find_peaks(me_smooth, distance=10, prominence=0.5)
        valleys, _ = scipy.signal.find_peaks(-me_smooth, distance=10, prominence=0.5)
        
        # Mid-gesture (-2) at motion peaks
        for p in peaks:
            affinity[p] -= 2.0
            
        # Motion minima (+1)
        for v in valleys:
            affinity[v] += 1.0
            
    # 2. Blink detection (+2)
    console.print("  [dim]Extracting blinks via MediaPipe...[/dim]")
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        
        base_options = python.BaseOptions(model_asset_path='face_landmarker.task') # We need the model!
        # Instead of downloading the model during ingest if not present, we can just fake the blink if model missing, 
        # or implement it safely. Wait, let's download the model if missing.
        import os
        import urllib.request
        import ssl
        model_path = Path(__file__).parent.parent / "models" / "face_landmarker.task"
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", context=ctx) as response, open(model_path, 'wb') as out_file:
                out_file.write(response.read())
            
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=3)
        detector = vision.FaceLandmarker.create_from_options(options)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Process at 10Hz by skipping frames
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            target_idx = int(current_time * affinity_fps)
            if target_idx < num_frames:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                detection_result = detector.detect(mp_image)
                if detection_result.face_blendshapes:
                    for face_blendshapes in detection_result.face_blendshapes:
                        # eyeBlinkLeft (9) and eyeBlinkRight (10)
                        blink_l = next((b.score for b in face_blendshapes if b.category_name == 'eyeBlinkLeft'), 0)
                        blink_r = next((b.score for b in face_blendshapes if b.category_name == 'eyeBlinkRight'), 0)
                        if blink_l > 0.5 or blink_r > 0.5:
                            affinity[target_idx] += 2.0
            
            # skip frames to stay near 10Hz
            frames_to_skip = int(fps / affinity_fps) - 1
            for _ in range(max(0, frames_to_skip)):
                cap.read()
                
        cap.release()
    except Exception as e:
        console.print(f"  [yellow]Failed to extract blinks: {e}[/yellow]")

    # 3. Breaths (+3)
    breaths = db.get_breaths()
    for b in breaths:
        s_idx = int(b["start_time"] * affinity_fps)
        e_idx = int(b["end_time"] * affinity_fps)
        s_idx = max(0, min(s_idx, num_frames - 1))
        e_idx = max(0, min(e_idx, num_frames))
        affinity[s_idx:e_idx] += 3.0
        
    # 4. Mid-word (-10)
    words = db.get_words()
    for w in words:
        # Don't penalize the very edges of the word, but the core
        s = w["start_time"] + 0.05
        e = w["end_time"] - 0.05
        if s < e:
            s_idx = int(s * affinity_fps)
            e_idx = int(e * affinity_fps)
            s_idx = max(0, min(s_idx, num_frames - 1))
            e_idx = max(0, min(e_idx, num_frames))
            affinity[s_idx:e_idx] -= 10.0

    # Write to DB
    records = []
    for i in range(num_frames):
        records.append((i / affinity_fps, float(affinity[i])))
        
    db.insert_cut_affinity(records)
    console.print(f"  [green]✓ Stored cut affinity curve ({len(records)} points)[/green]")
